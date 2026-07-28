#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,random,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,accuracy_score
LABEL_COLS=["Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices","No Finding"]
def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def load_labels(csv_path):
 df=pd.read_csv(csv_path); y=df[LABEL_COLS].replace(-1,np.nan).to_numpy(dtype=np.float32); m=~np.isnan(y); return np.nan_to_num(y,nan=0).astype(np.float32),m.astype(bool)
def macro_metrics(y,mask,prob,thr=None):
 aucs=[]; aps=[]; f1s=[]; accs=[]
 if thr is None: thr=np.full(y.shape[1],0.5,dtype=np.float32)
 for c in range(y.shape[1]):
  mm=mask[:,c]; yy=y[mm,c].astype(int); pp=prob[mm,c]
  if len(yy) and yy.min()!=yy.max(): aucs.append(float(roc_auc_score(yy,pp))); aps.append(float(average_precision_score(yy,pp)))
  if len(yy): pred=(pp>=thr[c]).astype(int); f1s.append(float(f1_score(yy,pred,zero_division=0))); accs.append(float(accuracy_score(yy,pred)))
 return {'AUC':float(np.mean(aucs)),'mAP':float(np.mean(aps)),'F1':float(np.mean(f1s)),'ACC':float(np.mean(accs))}
def best_thresholds(yv,mv,pv):
 th=np.full(yv.shape[1],0.5,dtype=np.float32); grid=np.linspace(0.01,0.99,99)
 for c in range(yv.shape[1]):
  m=mv[:,c]; yy=yv[m,c].astype(int); pp=pv[m,c]
  if len(yy)==0 or yy.min()==yy.max(): continue
  th[c]=float(grid[int(np.argmax([f1_score(yy,(pp>=t).astype(int),zero_division=0) for t in grid]))])
 return th
def fit_probe(xtr,ytr,mtr,xv,yv,mv,seed,epochs,lr,wd,batch,device):
 seed_all(seed); xtr=torch.tensor(xtr,dtype=torch.float32); ytr=torch.tensor(ytr,dtype=torch.float32); mtr=torch.tensor(mtr,dtype=torch.float32); xv_t=torch.tensor(xv,dtype=torch.float32,device=device)
 model=torch.nn.Linear(xtr.shape[1],ytr.shape[1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd); best=-1; state=None; n=xtr.shape[0]
 for ep in range(epochs):
  idx=torch.randperm(n); model.train()
  for st in range(0,n,batch):
   ids=idx[st:st+batch]; xb=xtr[ids].to(device); yb=ytr[ids].to(device); mb=mtr[ids].to(device)
   loss=(torch.nn.functional.binary_cross_entropy_with_logits(model(xb),yb,reduction='none')*mb).sum()/mb.sum().clamp_min(1)
   opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
  model.eval();
  with torch.no_grad(): pv=torch.sigmoid(model(xv_t)).cpu().numpy()
  score=macro_metrics(yv,mv,pv)['AUC']
  if score>best: best=score; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(state); model.eval(); return model
def l2n(a): return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-8)
def run_variant(name,paths,split_root,seeds,args):
 ytr,mtr=load_labels(Path(split_root)/'train.csv'); yv,mv=load_labels(Path(split_root)/'val.csv'); yt,mt=load_labels(Path(split_root)/'test.csv')
 xtr=l2n(np.load(paths[0])); xv=l2n(np.load(paths[1])); xt=l2n(np.load(paths[2])); rows=[]
 for seed in seeds:
  model=fit_probe(xtr,ytr,mtr,xv,yv,mv,seed,args.epochs,args.lr,args.weight_decay,args.batch,args.device)
  with torch.no_grad():
   pv=torch.sigmoid(model(torch.tensor(xv,dtype=torch.float32,device=args.device))).cpu().numpy(); pt=torch.sigmoid(model(torch.tensor(xt,dtype=torch.float32,device=args.device))).cpu().numpy()
  th=best_thresholds(yv,mv,pv); met=macro_metrics(yt,mt,pt,th); rows.append({'variant':name,'seed':seed,**met})
  print(rows[-1],flush=True)
 return rows
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--split-root',required=True); ap.add_argument('--feature-root',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--seeds',default='0,1,2,3,4'); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--batch',type=int,default=1024); ap.add_argument('--device',default='cuda')
 args=ap.parse_args(); Path(args.output_dir).mkdir(parents=True,exist_ok=True); seeds=[int(x) for x in args.seeds.split(',')]
 root=Path(args.feature_root); specs={'stage1_base':('base','features_stage2_v2_base.npy'),'predictive_stage2':('v2_ep4','features_stage2_v2_refined.npy')}
 rows=[]
 for name,(sub,file) in specs.items():
  paths=[root/sub/s/file for s in ['train','val','test']]
  if not all(p.exists() for p in paths): print('skip',name,paths); continue
  rows += run_variant(name,paths,args.split_root,seeds,args)
 raw=pd.DataFrame(rows); raw.to_csv(Path(args.output_dir)/'per_seed.csv',index=False)
 summ=raw.groupby('variant').agg({m:['mean','std'] for m in ['AUC','mAP','F1','ACC']}); summ.columns=['_'.join(c) for c in summ.columns]; summ=summ.reset_index(); summ.to_csv(Path(args.output_dir)/'summary_mean_std.csv',index=False)
 print('---SUMMARY---'); print(summ.to_string(index=False))
if __name__=='__main__': main()
