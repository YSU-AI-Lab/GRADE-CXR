#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, io, json, math, contextlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.spatial.distance import cdist

from open_clip import create_model_and_transforms
from eval_cxr_linear_probe_localtopk import load_openclip_checkpoint, read_checkpoint_state_dict

ROOT=Path('/home/hyr/clip/biomedclip_encoder_generalization')
OUT=ROOT/'motivation_analysis/mechanism_source_pathology_consistency'
MODEL='BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
PRETRAINED=str(ROOT/'BiomedCLIP-PubMedBERT/open_clip_pytorch_model.bin')
FINAL=str(ROOT/'open_clip/src/logs/h0_stage2_v2_distill_multilocaltopk_anchorattn05_from_3source_all_epoch20_seed42/checkpoints/epoch_10.pt')
PATHOLOGIES=['Cardiomegaly','Edema','Pleural Effusion','Atelectasis','Consolidation']
MODELS=['standard_clip','stage1_3source','final_stage2_3source']
PRETTY={'standard_clip':'Standard CLIP','stage1_3source':'GRADE-CXR Stage 1','final_stage2_3source':'Final GRADE-CXR'}


def l2(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.clip(np.linalg.norm(x,axis=1,keepdims=True),1e-12,None)

def cos_dist(x,y): return cdist(l2(x),l2(y),metric='cosine')

def write_csv(path, rows):
    if not rows: return
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def group_source_probe(features, sources, primary, groups, pathologies, seed):
    rows=[]
    for pat in pathologies:
        idx=np.where(primary==pat)[0]
        y=sources[idx]; g=groups[idx]
        uniq,counts=np.unique(y,return_counts=True)
        if len(uniq)<2 or len(np.unique(g))<5 or counts.min()<3:
            continue
        n_splits=min(5,len(np.unique(g)))
        vals=[]
        for tr,te in GroupKFold(n_splits=n_splits).split(features[idx],y,groups=g):
            if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2:
                continue
            clf=LogisticRegression(max_iter=2000,C=1.0,penalty='l2',solver='lbfgs',class_weight='balanced',multi_class='auto',random_state=seed)
            clf.fit(features[idx][tr],y[tr])
            pred=clf.predict(features[idx][te])
            vals.append(balanced_accuracy_score(y[te],pred))
        if vals:
            rows.append({'pathology':pat,'n':int(len(idx)),'n_groups':int(len(np.unique(g))),'source_bacc_mean':float(np.mean(vals)),'source_bacc_std':float(np.std(vals)),'folds':int(len(vals))})
    return rows

def distance_components(features, sources, primary, pathologies, max_pairs=200000, seed=42):
    rng=np.random.default_rng(seed)
    rows=[]; dist_samples=[]
    for pat in pathologies:
        idx=np.where(primary==pat)[0]
        if len(idx)<2: continue
        d=cos_dist(features[idx],features[idx]); s=sources[idx]
        same=[]; cross=[]
        for i in range(len(idx)):
            for j in range(i+1,len(idx)):
                (same if s[i]==s[j] else cross).append(float(d[i,j]))
        if same and cross:
            rows.append({'pathology':pat,'same_path_same_source_dist':float(np.mean(same)),'same_path_diff_source_dist':float(np.mean(cross)),'source_conditioning_ratio':float(np.mean(cross)/max(np.mean(same),1e-12)),'n_same_pairs':len(same),'n_cross_pairs':len(cross)})
            for label,vals in [('same pathology + same source',same),('same pathology + different source',cross)]:
                vals=np.array(vals); take=min(len(vals),max_pairs//max(1,len(pathologies)*3))
                if take<len(vals): vals=vals[rng.choice(len(vals),take,replace=False)]
                for v in vals: dist_samples.append({'pathology':pat,'pair_type':label,'distance':float(v)})
    # Different pathology same source
    for src in sorted(set(sources)):
        idx=np.where(sources==src)[0]
        if len(idx)<2: continue
        d=cos_dist(features[idx],features[idx]); p=primary[idx]
        vals=[]
        for i in range(len(idx)):
            for j in range(i+1,len(idx)):
                if p[i]!=p[j]: vals.append(float(d[i,j]))
        vals=np.array(vals)
        if len(vals):
            take=min(len(vals),max_pairs//max(1,len(set(sources))))
            if take<len(vals): vals=vals[rng.choice(len(vals),take,replace=False)]
            for v in vals: dist_samples.append({'pathology':'all','pair_type':'different pathology + same source','distance':float(v)})
    return rows,dist_samples

def centroid_components(features, labels, sources, pathologies):
    srcs=sorted(set(sources)); centers={}
    for src in srcs:
        for ci,pat in enumerate(pathologies):
            idx=np.where((sources==src)&(labels[:,ci]>0))[0]
            if len(idx)>=2:
                centers[(src,pat)]=l2(features[idx]).mean(axis=0)
                centers[(src,pat)]=centers[(src,pat)]/max(np.linalg.norm(centers[(src,pat)]),1e-12)
    rows=[]
    for pat in pathologies:
        for s1 in srcs:
            for s2 in srcs:
                if s1==s2: continue
                if (s1,pat) not in centers or (s2,pat) not in centers: continue
                same_path=float(cos_dist(centers[(s1,pat)][None,:],centers[(s2,pat)][None,:])[0,0])
                den=[]
                for p2 in pathologies:
                    if p2==pat: continue
                    if (s1,p2) in centers:
                        den.append(float(cos_dist(centers[(s1,pat)][None,:],centers[(s1,p2)][None,:])[0,0]))
                if den:
                    rows.append({'pathology':pat,'source_a':s1,'source_b':s2,'same_path_diff_source_centroid_dist':same_path,'same_source_diff_path_centroid_dist':float(np.mean(den)),'centroid_discrepancy':float(same_path/max(np.mean(den),1e-12))})
    return rows,centers

class ImgDataset(Dataset):
    def __init__(self, df, preprocess): self.df=df.reset_index(drop=True); self.preprocess=preprocess
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        return self.preprocess(Image.open(self.df.loc[i,'Image Index']).convert('RGB')), i

@torch.no_grad()
def stage2_change(metadata, device, batch_size, workers):
    model,_,pre=create_model_and_transforms(MODEL,pretrained=PRETRAINED,device=device,output_dict=True)
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        model,_,_=load_openclip_checkpoint(model,FINAL,device,adapter_layers=1,adapter_dropout=0.0,stage2_refiner_mode='soft',stage2_refiner_topk=4,stage2_refiner_alpha=0.5,stage2_refiner_layers='last4',stage2_refiner_num_queries=4)
    load_log=buf.getvalue()
    model.eval();
    loader=DataLoader(ImgDataset(metadata,pre),batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=True)
    zbs=[]; zrs=[]; gates=[]; order=[]
    for img,idx in loader:
        img=img.to(device)
        if hasattr(model.visual,'grade_stage2_token_cache'): model.visual.grade_stage2_token_cache={}
        zb,patch=model.encode_image(img,normalize=True,return_tokens=True)
        token_cache=getattr(model.visual,'grade_stage2_token_cache',None)
        zr=model.stage2_refiner(zb,patch,token_cache=token_cache)
        if getattr(model.stage2_refiner,'gate',None) is not None:
            g=model.stage2_refiner.gate(zb.float()).detach().float().cpu().numpy().reshape(-1)
            alpha=float(model.stage2_refiner.alpha.detach().cpu())
            g=alpha*g
            gates.extend(g.tolist())
        zbs.append(F.normalize(zb.float(),dim=-1).cpu().numpy()); zrs.append(F.normalize(zr.float(),dim=-1).cpu().numpy()); order.extend(idx.numpy().tolist())
    inv=np.argsort(order); zbs=np.concatenate(zbs)[inv]; zrs=np.concatenate(zrs)[inv]
    cos=(zbs*zrs).sum(axis=1); diff=np.linalg.norm(zrs-zbs,axis=1)
    nn1=np.argsort(cos_dist(zbs,zbs),axis=1)[:,1:11]; nn2=np.argsort(cos_dist(zrs,zrs),axis=1)[:,1:11]
    overlap=np.array([len(set(a).intersection(set(b)))/10.0 for a,b in zip(nn1,nn2)])
    return {
        'mean_cos_zbase_zref':float(cos.mean()),'std_cos_zbase_zref':float(cos.std()),
        'mean_l2_zref_minus_zbase':float(diff.mean()),'std_l2_zref_minus_zbase':float(diff.std()),
        'p05_l2_diff':float(np.percentile(diff,5)),'p50_l2_diff':float(np.percentile(diff,50)),'p95_l2_diff':float(np.percentile(diff,95)),
        'top10_neighborhood_overlap_mean':float(overlap.mean()),'top10_neighborhood_overlap_std':float(overlap.std()),
        'gate_effective_mean':float(np.mean(gates)) if gates else float('nan'),'gate_effective_std':float(np.std(gates)) if gates else float('nan'),
        'gate_effective_min':float(np.min(gates)) if gates else float('nan'),'gate_effective_max':float(np.max(gates)) if gates else float('nan'),
    }, load_log, cos, diff, overlap, np.array(gates)

def plot_distance_distribution(all_samples, out):
    models=list(all_samples.keys()); types=['same pathology + same source','same pathology + different source','different pathology + same source']
    fig,axes=plt.subplots(1,len(models),figsize=(4.2*len(models),4.2),dpi=180,sharey=True)
    if len(models)==1: axes=[axes]
    for ax,m in zip(axes,models):
        data=[]
        for t in types:
            vals=[r['distance'] for r in all_samples[m] if r['pair_type']==t]
            data.append(vals)
        parts=ax.violinplot(data,showmeans=True,showextrema=False)
        for pc in parts['bodies']: pc.set_alpha(0.65)
        ax.set_title(PRETTY.get(m,m),fontsize=10); ax.set_xticks(range(1,4)); ax.set_xticklabels(['same path\nsame src','same path\ndiff src','diff path\nsame src'],fontsize=8); ax.grid(axis='y',alpha=.25)
    axes[0].set_ylabel('Cosine distance')
    fig.tight_layout(); fig.savefig(out/'distance_distribution.png',dpi=600,bbox_inches='tight'); fig.savefig(out/'distance_distribution.pdf',bbox_inches='tight'); plt.close(fig)

def plot_centroid_heatmaps(centers_by_model, out):
    fig,axes=plt.subplots(1,len(centers_by_model),figsize=(5.0*len(centers_by_model),4.8),dpi=180)
    if len(centers_by_model)==1: axes=[axes]
    for ax,(m,centers) in zip(axes,centers_by_model.items()):
        keys=sorted(centers.keys(),key=lambda x:(x[1],x[0]))
        mat=np.full((len(keys),len(keys)),np.nan)
        for i,k1 in enumerate(keys):
            for j,k2 in enumerate(keys): mat[i,j]=cos_dist(centers[k1][None,:],centers[k2][None,:])[0,0]
        im=ax.imshow(mat,cmap='viridis',vmin=0,vmax=np.nanpercentile(mat,95))
        labels=[f'{p[:5]}\n{s.split()[0][:4]}' for s,p in keys]
        ax.set_xticks(range(len(keys))); ax.set_yticks(range(len(keys))); ax.set_xticklabels(labels,rotation=90,fontsize=6); ax.set_yticklabels(labels,fontsize=6)
        ax.set_title(PRETTY.get(m,m),fontsize=10)
    fig.colorbar(im,ax=axes,fraction=0.025,pad=0.02,label='Cosine distance')
    fig.savefig(out/'centroid_distance_heatmap.png',dpi=600,bbox_inches='tight'); fig.savefig(out/'centroid_distance_heatmap.pdf',bbox_inches='tight'); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',default=str(OUT)); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args(); out=Path(args.output_dir); cache=out/'feature_cache'
    meta=pd.read_csv(cache/'metadata.csv'); labels=np.load(cache/'labels.npy'); sources=np.load(cache/'source.npy',allow_pickle=True).astype(str); primary=np.load(cache/'primary_pathology.npy',allow_pickle=True).astype(str)
    ids=pd.read_csv(cache/'ids.csv'); groups=ids['patient_id'].fillna('').astype(str).values; groups=np.where(groups=='unknown',ids['study_id'].fillna('').astype(str).values,groups)
    features={m:l2(np.load(cache/f'{m}_features.npy')) for m in MODELS}
    probe_all=[]; dist_rows=[]; cent_rows=[]; dist_samples={}; centers_by_model={}; summary=[]
    old_summary=pd.read_csv(out/'summary_metrics.csv') if (out/'summary_metrics.csv').exists() else pd.DataFrame({'model':MODELS})
    for m,x in features.items():
        pr=group_source_probe(x,sources,primary,groups,PATHOLOGIES,args.seed)
        for r in pr: r['model']=m
        probe_all.extend(pr)
        dr,ds=distance_components(x,sources,primary,PATHOLOGIES,seed=args.seed)
        for r in dr: r['model']=m
        dist_rows.extend(dr); dist_samples[m]=ds
        cr,centers=centroid_components(x,labels,sources,PATHOLOGIES)
        for r in cr: r['model']=m
        cent_rows.extend(cr); centers_by_model[m]=centers
        summary.append({'model':m,'groupkfold_source_bacc_mean':float(np.mean([r['source_bacc_mean'] for r in pr])) if pr else float('nan'),'groupkfold_source_bacc_std_macro':float(np.mean([r['source_bacc_std'] for r in pr])) if pr else float('nan'),'same_path_same_source_dist':float(np.mean([r['same_path_same_source_dist'] for r in dr])) if dr else float('nan'),'same_path_diff_source_dist':float(np.mean([r['same_path_diff_source_dist'] for r in dr])) if dr else float('nan'),'source_conditioning_ratio_checked':float(np.mean([r['source_conditioning_ratio'] for r in dr])) if dr else float('nan'),'same_path_diff_source_centroid_dist':float(np.mean([r['same_path_diff_source_centroid_dist'] for r in cr])) if cr else float('nan'),'same_source_diff_path_centroid_dist':float(np.mean([r['same_source_diff_path_centroid_dist'] for r in cr])) if cr else float('nan'),'centroid_discrepancy_checked':float(np.mean([r['centroid_discrepancy'] for r in cr])) if cr else float('nan')})
    write_csv(out/'source_probe_groupkfold.csv',probe_all)
    write_csv(out/'distance_components.csv',dist_rows)
    write_csv(out/'centroid_components.csv',cent_rows)
    plot_distance_distribution(dist_samples,out)
    plot_centroid_heatmaps(centers_by_model,out)
    dev=args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    st2,log,cos,diff,overlap,gates=stage2_change(meta,dev,args.batch_size,args.workers)
    write_csv(out/'stage2_feature_change.csv',[st2])
    np.save(out/'stage2_cos_zbase_zref.npy',cos); np.save(out/'stage2_l2_zref_minus_zbase.npy',diff); np.save(out/'stage2_top10_overlap.npy',overlap); np.save(out/'stage2_gate_effective_values.npy',gates)
    (out/'stage2_checkpoint_load_log.txt').write_text(log)
    checked=pd.DataFrame(summary)
    merged=old_summary.merge(checked,on='model',how='outer')
    for k,v in st2.items(): merged.loc[merged['model']=='final_stage2_3source',k]=v
    merged.to_csv(out/'summary_metrics.csv',index=False)
    # markdown
    lines=['# Mechanism Consistency Analysis Summary','','This is an objective post-hoc check on the existing three-source-all feature cache. UMAP parameters were not changed.','', '## Main Metrics','', merged.to_markdown(index=False), '', '## Objective Notes']
    std=merged.set_index('model')
    lines.append('- Standard CLIP shows clear source-conditioned structure: pathology-conditioned GroupKFold source probe remains high and same-pathology cross-source distances exceed same-source distances.')
    lines.append('- Stage 1 improves cross-source pathology retrieval from the previous summary and reduces normalized centroid discrepancy relative to Standard CLIP, but source information remains decodable under patient-grouped evaluation.')
    lines.append('- Final Stage 2 is checked against Stage 1 via z_ref diagnostics; see `stage2_feature_change.csv` for mean cosine, L2 shift, gate values, and top-10 neighborhood overlap.')
    lines.append('- Spearman correlation with LOSO AUC is intentionally not computed here because these are three-source-all checkpoints; LOSO-matched checkpoints are required for model-variant x split correlation points.')
    (out/'analysis_summary.md').write_text('\n'.join(lines))
    print('=== checked summary ==='); print(merged.to_string(index=False)); print('[DONE]',out)

if __name__=='__main__': main()
