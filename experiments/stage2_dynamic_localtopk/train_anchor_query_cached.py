#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
ROOT=Path('/home/hyr/clip/biomedclip_encoder_generalization'); SRC=ROOT/'open_clip/src'; sys.path[:0]=[str(ROOT),str(SRC)]
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.grade_stage2_v2 import CHEXPERT14
from train_stage2_v2 import DEFAULT_MODEL, DEFAULT_PRETRAINED, read_checkpoint_state_dict, set_seed
from open_clip.grade_stage2_v2 import DiseaseAnchorQueryStage2Refiner, build_localtopk_oracle_teacher, predictive_localtopk_stage2_losses
class DS(Dataset):
 def __init__(self,c):
  c=Path(c); self.z=np.load(c/'z_base.float16.npy',mmap_mode='r'); self.t=np.load(c/'multi_tokens.float16.npy',mmap_mode='r'); self.y=np.load(c/'labels14.float32.npy',mmap_mode='r'); self.meta=json.loads((c/'cache_meta.json').read_text())
 def __len__(self): return len(self.y)
 def __getitem__(self,i): return {'z_base':torch.from_numpy(np.asarray(self.z[i],dtype=np.float32)),'multi_tokens':torch.from_numpy(np.asarray(self.t[i],dtype=np.float32)),'labels14':torch.from_numpy(np.asarray(self.y[i],dtype=np.float32))}
def collate(b): return {k:torch.stack([x[k] for x in b]) for k in b[0]}
@torch.no_grad()
def anchors(args,device):
 model,_,_=create_model_and_transforms(args.model,pretrained=args.pretrained,device=device,output_dict=True); msg=model.load_state_dict(read_checkpoint_state_dict(args.stage1_checkpoint),strict=False); model.eval(); tok=get_tokenizer(args.model)
 a=F.normalize(model.encode_text(tok(['This is a chest X-ray image of '+x.lower()+'.' for x in CHEXPERT14]).to(device),normalize=True).float(),dim=-1).detach(); return a, {'missing_keys':list(msg.missing_keys),'unexpected_keys':list(msg.unexpected_keys)}
def save(p,model,args,ep,row,load):
 p.parent.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':model.state_dict(),'epoch':ep,'metrics':row,'args':vars(args),'stage1_load_info':load,'refiner_type':'disease_anchor_query'},p)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--cache-dir',required=True); ap.add_argument('--stage1-checkpoint',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--model',default=DEFAULT_MODEL); ap.add_argument('--pretrained',default=DEFAULT_PRETRAINED); ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--batch-size',type=int,default=512); ap.add_argument('--workers',type=int,default=2); ap.add_argument('--lr',type=float,default=5e-4); ap.add_argument('--weight-decay',type=float,default=0.01); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--device',default='cuda'); ap.add_argument('--topk',type=int,default=4); ap.add_argument('--alpha',type=float,default=0.5); ap.add_argument('--num-heads',type=int,default=8); ap.add_argument('--lambda-z',type=float,default=2.0); ap.add_argument('--lambda-bce',type=float,default=1.0); ap.add_argument('--lambda-pres',type=float,default=0.2)
 args=ap.parse_args(); set_seed(args.seed); device='cuda:0' if args.device=='cuda' and torch.cuda.is_available() else args.device; out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
 ds=DS(args.cache_dir); gen=torch.Generator().manual_seed(args.seed); dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,generator=gen,num_workers=args.workers,collate_fn=collate,drop_last=True,pin_memory=True)
 a,load=anchors(args,device); ref=DiseaseAnchorQueryStage2Refiner(a.cpu(),num_heads=args.num_heads,alpha=args.alpha).to(device); opt=torch.optim.AdamW(ref.parameters(),lr=args.lr,weight_decay=args.weight_decay); rows=[]
 (out/'config_resolved.json').write_text(json.dumps({**vars(args),'cache_meta':ds.meta},indent=2),encoding='utf-8')
 for ep in range(1,args.epochs+1):
  ref.train(); sums=[]
  for batch in tqdm(dl,desc=f'anchor-query-stage2 ep{ep}',ncols=100):
   z=F.normalize(batch['z_base'].to(device),dim=-1); t=batch['multi_tokens'].to(device); y=batch['labels14'].to(device)
   outp=ref(z,t); teach=build_localtopk_oracle_teacher(z,t,y,a,args.topk,args.alpha,0.07); ls=predictive_localtopk_stage2_losses(z,outp,y,teach,args.lambda_z,args.lambda_bce,args.lambda_pres)
   opt.zero_grad(set_to_none=True); ls['loss'].backward(); opt.step()
   with torch.no_grad(): sums.append({'loss':float(ls['loss'].cpu()),'loss_z':float(ls['loss_z'].cpu()),'loss_bce':float(ls['loss_router_bce'].cpu()),'loss_pres':float(ls['loss_pres'].cpu()),'gate_mean':float(outp['gate'].mean().cpu()),'alpha':float(ref.alpha.detach().cpu()),'z_ref_z_base_cos':float((outp['z_ref']*z).sum(-1).mean().cpu()),'z_ref_z_tgt_cos':float((outp['z_ref']*teach['z_tgt']).sum(-1).mean().cpu()),'valid_teacher_ratio':float(teach['valid_teacher'].float().mean().cpu())})
  row={'epoch':ep}; row.update({k:float(np.mean([x[k] for x in sums])) for k in sums[0]}); rows.append(row); save(out/'checkpoints'/f'epoch_{ep}.pt',ref,args,ep,row,load); save(out/'checkpoints'/'latest.pt',ref,args,ep,row,load)
  with open(out/'metrics.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
  print(row,flush=True)
if __name__=='__main__': main()
