#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
ROOT=Path('/home/hyr/clip/biomedclip_encoder_generalization'); SRC=ROOT/'open_clip/src'; sys.path[:0]=[str(ROOT),str(SRC)]
from open_clip import create_model_and_transforms
from open_clip.grade_stage2_v2 import forward_visual_multilevel, parse_selected_layers, visual_num_layers
from train_stage2_v2 import DEFAULT_MODEL, DEFAULT_PRETRAINED, read_checkpoint_state_dict
from open_clip.grade_stage2_v2 import PredictiveLocalTopKStage2Refiner
class DS(Dataset):
 def __init__(self,csv,pre,img_key): self.df=pd.read_csv(csv); self.pre=pre; self.img_key=img_key
 def __len__(self): return len(self.df)
 def __getitem__(self,i):
  p=str(self.df.iloc[i][self.img_key]); return self.pre(Image.open(p).convert('RGB')),p

def collate(b): return torch.stack([x[0] for x in b]),[x[1] for x in b]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--stage1-checkpoint',required=True); ap.add_argument('--refiner-checkpoint',required=True); ap.add_argument('--model',default=DEFAULT_MODEL); ap.add_argument('--pretrained',default=DEFAULT_PRETRAINED); ap.add_argument('--selected-layers',default='3,6,9,12'); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--workers',type=int,default=2); ap.add_argument('--device',default='cuda'); ap.add_argument('--img-key',default='Image Index'); ap.add_argument('--alpha-override',type=float,default=None,help='Override refiner alpha metadata/forward if z_ref is used')
 args=ap.parse_args(); device='cuda:0' if args.device=='cuda' and torch.cuda.is_available() else args.device
 model,_,pre=create_model_and_transforms(args.model,pretrained=args.pretrained,device=device,output_dict=True); msg=model.load_state_dict(read_checkpoint_state_dict(args.stage1_checkpoint),strict=False); model.eval()
 for p in model.parameters(): p.requires_grad=False
 layers=parse_selected_layers(args.selected_layers,visual_num_layers(model),one_based=True)
 ck=torch.load(args.refiner_checkpoint,map_location='cpu',weights_only=False); sd=ck.get('state_dict',ck); anchors=sd['anchors']
 ref=PredictiveLocalTopKStage2Refiner(anchors,topk=ck.get('args',{}).get('topk',4),alpha=ck.get('args',{}).get('alpha',0.5)).to(device); ref.load_state_dict(sd,strict=True);
 if args.alpha_override is not None:
  ref.alpha.data.fill_(float(args.alpha_override))
 ref.eval()
 dl=DataLoader(DS(args.input,pre,args.img_key),batch_size=args.batch_size,num_workers=args.workers,collate_fn=collate)
 zs=[]; us=[]; gs=[]; logits=[]; paths=[]
 with torch.no_grad():
  for im,p in tqdm(dl,desc='extract-predictive-components',ncols=100):
   z,t=forward_visual_multilevel(model.visual,im.to(device),layers); out=ref(z,t)
   zs.append(torch.nn.functional.normalize(z.float(),dim=-1).cpu().numpy().astype('float32'))
   us.append(out['u'].float().cpu().numpy().astype('float32'))
   gs.append(out['gate'].float().cpu().numpy().astype('float32'))
   logits.append(out['disease_logits'].float().cpu().numpy().astype('float32'))
   paths+=p
 outd=Path(args.output_dir); outd.mkdir(parents=True,exist_ok=True)
 np.save(outd/'z_base.npy',np.concatenate(zs)); np.save(outd/'u.npy',np.concatenate(us)); np.save(outd/'gate.npy',np.concatenate(gs)); np.save(outd/'disease_logits.npy',np.concatenate(logits))
 pd.DataFrame({'image_path':paths}).to_csv(outd/'feature_image_paths.csv',index=False)
 (outd/'components.json').write_text(json.dumps({'input':args.input,'refiner_checkpoint':args.refiner_checkpoint,'num_samples':len(paths),'alpha_checkpoint':float(ref.alpha.detach().cpu()),'missing':len(msg.missing_keys),'unexpected':len(msg.unexpected_keys)},indent=2),encoding='utf-8')
if __name__=='__main__': main()
