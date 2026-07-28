#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,accuracy_score
LABEL_COLS=["Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices","No Finding"]
def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def l2n(a): return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-8)
def load_labels(csv_path):
 df=pd.read_csv(csv_path); y=df[LABEL_COLS].replace(-1,np.nan).to_numpy(dtype=np.float32); m=~np.isnan(y); return np.nan_to_num(y,nan=0).astype(np.float32),m.astype(bool)
def materialize(comp_root,out_root,alpha):
 comp=Path(comp_root); out=Path(out_root)
 for split in ['train','val','test']:
  od=out/split; od.mkdir(parents=True,exist_ok=True)
  z=np.load(comp/split/'z_base.npy'); u=np.load(comp/split/'u.npy'); g=np.load(comp/split/'gate.npy')
  np.save(od/'features_predictive_stage2.npy',l2n(z+float(alpha)*g*u).astype('float32'))
 (out/'predictive_stage2_features.json').write_text(json.dumps({'components_root':str(comp),'alpha':float(alpha)},indent=2),encoding='utf-8')
def metrics(y,mask,prob,thr=None):
 if thr is None: thr=np.full(y.shape[1],0.5,dtype=np.float32)
 aucs=[]; aps=[]; f1s=[]; accs=[]
 for c in range(y.shape[1]):
  m=mask[:,c]; yy=y[m,c].astype(int); pp=prob[m,c]
  if len(yy) and yy.min()!=yy.max(): aucs.append(float(roc_auc_score(yy,pp))); aps.append(float(average_precision_score(yy,pp)))
  if len(yy): pred=(pp>=thr[c]).astype(int); f1s.append(float(f1_score(yy,pred,zero_division=0))); accs.append(float(accuracy_score(yy,pred)))
 return {'AUC':float(np.mean(aucs)),'mAP':float(np.mean(aps)),'F1':float(np.mean(f1s)),'ACC':float(np.mean(accs))}
def best_thr(y,m,p):
 th=np.full(y.shape[1],0.5,dtype=np.float32); grid=np.linspace(0.01,0.99,99)
 for c in range(y.shape[1]):
  mm=m[:,c]; yy=y[mm,c].astype(int); pp=p[mm,c]
  if len(yy)==0 or yy.min()==yy.max(): continue
  th[c]=float(grid[int(np.argmax([f1_score(yy,(pp>=t).astype(int),zero_division=0) for t in grid]))])
 return th
def fit(xtr,ytr,mtr,xv,yv,mv,seed,args):
 seed_all(seed); xtr=torch.tensor(xtr,dtype=torch.float32); ytr=torch.tensor(ytr,dtype=torch.float32); mtr=torch.tensor(mtr,dtype=torch.float32); xv_t=torch.tensor(xv,dtype=torch.float32,device=args.device)
 model=torch.nn.Linear(xtr.shape[1],ytr.shape[1]).to(args.device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); best=-1; state=None; n=xtr.shape[0]
 for _ in range(args.epochs):
  idx=torch.randperm(n); model.train()
  for st in range(0,n,args.batch):
   ids=idx[st:st+args.batch]; xb=xtr[ids].to(args.device); yb=ytr[ids].to(args.device); mb=mtr[ids].to(args.device)
   loss=(torch.nn.functional.binary_cross_entropy_with_logits(model(xb),yb,reduction='none')*mb).sum()/mb.sum().clamp_min(1)
   opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
  model.eval();
  with torch.no_grad(): pv=torch.sigmoid(model(xv_t)).cpu().numpy()
  sc=metrics(yv,mv,pv)['AUC']
  if sc>best: best=sc; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(state); return model
def eval_variant(name,paths,split_root,seeds,args):
 ytr,mtr=load_labels(Path(split_root)/'train.csv'); yv,mv=load_labels(Path(split_root)/'val.csv'); yt,mt=load_labels(Path(split_root)/'test.csv')
 xtr=l2n(np.load(paths[0])); xv=l2n(np.load(paths[1])); xt=l2n(np.load(paths[2])); rows=[]
 for seed in seeds:
  model=fit(xtr,ytr,mtr,xv,yv,mv,seed,args); model.eval()
  with torch.no_grad():
   pv=torch.sigmoid(model(torch.tensor(xv,dtype=torch.float32,device=args.device))).cpu().numpy(); pt=torch.sigmoid(model(torch.tensor(xt,dtype=torch.float32,device=args.device))).cpu().numpy()
  th=best_thr(yv,mv,pv); rows.append({'variant':name,'seed':seed,**metrics(yt,mt,pt,th)})
  print(rows[-1],flush=True)
 return rows
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--split-root',required=True); ap.add_argument('--stage1-feature-root',required=True); ap.add_argument('--components-root',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--alpha',type=float,default=1.25); ap.add_argument('--seeds',default='0,1,2,3,4'); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--batch',type=int,default=1024); ap.add_argument('--device',default='cuda')
 args=ap.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); seeds=[int(x) for x in args.seeds.split(',')]
 pred_root=out/'features_predictive_stage2'; materialize(args.components_root,pred_root,args.alpha)
 specs={
  'Stage1_base':[Path(args.stage1_feature_root)/s/'features_stage2_v2_base.npy' for s in ['train','val','test']],
  f'Predictive_Stage2_alpha{args.alpha:g}':[pred_root/s/'features_predictive_stage2.npy' for s in ['train','val','test']],
 }
 rows=[]
 for n,p in specs.items(): rows += eval_variant(n,p,args.split_root,seeds,args)
 raw=pd.DataFrame(rows); raw.to_csv(out/'per_seed.csv',index=False)
 summ=raw.groupby('variant').agg({m:['mean','std'] for m in ['AUC','mAP','F1','ACC']}); summ.columns=['_'.join(c) for c in summ.columns]; summ=summ.reset_index(); summ.to_csv(out/'summary_mean_std.csv',index=False)
 b=summ[summ.variant=='Stage1_base'].iloc[0]; r=summ[summ.variant!= 'Stage1_base'].iloc[0]
 delta={'alpha':args.alpha,**{f'{m}_delta_pp':float((r[f'{m}_mean']-b[f'{m}_mean'])*100) for m in ['AUC','mAP','F1','ACC']}}
 pd.DataFrame([delta]).to_csv(out/'delta_vs_base.csv',index=False)
 (out/'run_config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
 print('---SUMMARY---'); print(summ.to_string(index=False)); print('---DELTA---'); print(pd.DataFrame([delta]).to_string(index=False))
if __name__=='__main__': main()
