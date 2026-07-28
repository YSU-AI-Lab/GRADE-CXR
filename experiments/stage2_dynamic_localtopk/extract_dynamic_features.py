#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
ROOT=Path('/home/hyr/clip/biomedclip_encoder_generalization'); SRC=ROOT/'open_clip/src'; sys.path[:0]=[str(ROOT),str(SRC)]
from open_clip import create_model_and_transforms
from open_clip.grade_stage2_v2 import forward_visual_multilevel, parse_selected_layers, visual_num_layers
from train_stage2_v2 import DEFAULT_MODEL, DEFAULT_PRETRAINED, read_checkpoint_state_dict
from experiments.stage2_dynamic_localtopk.stage2_dynamic_localtopk import DynamicLocalTopKRefiner
class DS(Dataset):
    def __init__(self,csv_path,preprocess,img_key): self.df=pd.read_csv(csv_path); self.preprocess=preprocess; self.img_key=img_key
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        p=str(self.df.iloc[i][self.img_key]); im=Image.open(p).convert('RGB')
        return self.preprocess(im), p
def collate(b): return torch.stack([x[0] for x in b]), [x[1] for x in b]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--stage1-checkpoint',required=True); ap.add_argument('--refiner-checkpoint',required=True); ap.add_argument('--model',default=DEFAULT_MODEL); ap.add_argument('--pretrained',default=DEFAULT_PRETRAINED); ap.add_argument('--selected-layers',default='3,6,9,12'); ap.add_argument('--batch-size',type=int,default=96); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--device',default='cuda'); ap.add_argument('--img-key',default='Image Index')
    args=ap.parse_args(); device='cuda:0' if args.device=='cuda' and torch.cuda.is_available() else args.device
    model,_,preprocess=create_model_and_transforms(args.model,pretrained=args.pretrained,device=device,output_dict=True)
    msg=model.load_state_dict(read_checkpoint_state_dict(args.stage1_checkpoint),strict=False); model.eval();
    for p in model.parameters(): p.requires_grad=False
    layers=parse_selected_layers(args.selected_layers,visual_num_layers(model),one_based=True)
    ck=torch.load(args.refiner_checkpoint,map_location='cpu'); sd=ck.get('state_dict',ck)
    nq=sd['query_mlp.3.bias'].numel()//512 if 'query_mlp.3.bias' in sd else 4
    ref=DynamicLocalTopKRefiner(embed_dim=512,num_layers=len(layers),num_queries=nq,num_heads=8).to(device); ref.load_state_dict(sd,strict=True); ref.eval()
    ds=DS(args.input,preprocess,args.img_key); dl=DataLoader(ds,batch_size=args.batch_size,num_workers=args.workers,collate_fn=collate)
    feats=[]; paths=[]
    with torch.no_grad():
        for im,pth in tqdm(dl,desc='extract-dynamic-refined',ncols=100):
            im=im.to(device); z,toks=forward_visual_multilevel(model.visual,im,layers); out=ref(z,toks); feats.append(out['z_ref'].cpu().numpy().astype('float32')); paths+=pth
    outd=Path(args.output_dir); outd.mkdir(parents=True,exist_ok=True); np.save(outd/'features_dynamic_refined.npy',np.concatenate(feats)); pd.DataFrame({'image_path':paths}).to_csv(outd/'feature_image_paths.csv',index=False)
    (outd/'features_dynamic_refined.json').write_text(json.dumps({'input':args.input,'refiner_checkpoint':args.refiner_checkpoint,'stage1_checkpoint':args.stage1_checkpoint,'num_samples':len(paths),'load_missing_keys':len(msg.missing_keys),'load_unexpected_keys':len(msg.unexpected_keys)},indent=2),encoding='utf-8')
if __name__=='__main__': main()
