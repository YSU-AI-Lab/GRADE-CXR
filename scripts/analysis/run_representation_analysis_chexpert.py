#!/usr/bin/env python
import argparse, json, os, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.extmath import randomized_svd

from eval_cxr_linear_probe_localtopk import (
    DEFAULT_LABEL_COLS, create_model_and_transforms, get_tokenizer,
    load_openclip_checkpoint, encode_pathology_anchors, local_topk_refine_batch,
)

NO_FINDING = 13
EVAL_CLASSES = [i for i in range(14) if i != NO_FINDING]
SOURCE_NAMES = {0: 'MIMIC-CXR', 1: 'CheXpert+', 2: 'PadChest'}

class JsonlCXR(Dataset):
    def __init__(self, rows, preprocess):
        self.rows = rows
        self.preprocess = preprocess
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r['ImageID']).convert('RGB')
        return self.preprocess(img), torch.tensor(r['labels14'], dtype=torch.float32), int(r['source_id']), r['ImageID']

def read_jsonl(path):
    rows=[]
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            d=json.loads(line)
            if os.path.exists(str(d.get('ImageID',''))) and 'labels14' in d and 'source_id' in d:
                rows.append(d)
    return rows

def balanced_sample(rows, max_per_source, seed):
    if max_per_source is None or max_per_source <= 0:
        return rows
    rng=np.random.default_rng(seed)
    out=[]
    for s in sorted(set(int(r['source_id']) for r in rows)):
        rs=[r for r in rows if int(r['source_id'])==s]
        if len(rs)>max_per_source:
            idx=rng.choice(len(rs), max_per_source, replace=False)
            rs=[rs[i] for i in idx]
        out.extend(rs)
    rng.shuffle(out)
    return out

@torch.no_grad()
def extract_features(model_kind, rows, args, cache_dir):
    cache_dir=Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    tag=f"{model_kind}_n{len(rows)}_topk{args.localtopk_k}_a{args.localtopk_alpha}_seed{args.seed}.npz"
    cache=cache_dir/tag
    if cache.exists() and not args.force_extract:
        d=np.load(cache, allow_pickle=True)
        return {k:d[k] for k in d.files}
    device=args.device
    model, _, preprocess = create_model_and_transforms(args.model, pretrained=args.pretrained)
    model.visual.output_tokens = True
    tokenizer=get_tokenizer(args.model)
    use_local = (model_kind == 'GRADE-CXR_ref')
    ckpt = None
    if model_kind in ('GRADE-CXR_base','GRADE-CXR_ref'):
        ckpt=args.fg_clip_checkpoint
    if ckpt:
        model, _, _ = load_openclip_checkpoint(model, ckpt, device, adapter_layers=1, adapter_dropout=0.0)
    else:
        model.to(device); model.eval()
    anchors=encode_pathology_anchors(model, tokenizer, DEFAULT_LABEL_COLS, device) if use_local else None
    ds=JsonlCXR(rows, preprocess)
    dl=DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    feats=[]; labels=[]; sources=[]; paths=[]
    for images, y, s, p in dl:
        images=images.to(device); y_dev=y.to(device)
        if use_local:
            z, patch = model.encode_image(images, normalize=True, return_tokens=True)
            if patch.shape[-1] != z.shape[-1]:
                patch = patch @ model.visual.proj.to(device=patch.device, dtype=patch.dtype)
            f=local_topk_refine_batch(z, patch, y_dev, anchors, topk=args.localtopk_k, alpha=args.localtopk_alpha, mode='oracle')
        else:
            f=model.encode_image(images, normalize=True)
        feats.append(f.float().cpu().numpy()); labels.append(y.numpy()); sources.append(s.numpy()); paths.extend(list(p))
    arr=np.concatenate(feats); lab=np.concatenate(labels); src=np.concatenate(sources)
    np.savez_compressed(cache, features=arr, labels=lab, source=src, paths=np.array(paths, dtype=object))
    return dict(features=arr, labels=lab, source=src, paths=np.array(paths, dtype=object))

def l2(x):
    n=np.linalg.norm(x,axis=1,keepdims=True)+1e-12
    return x/n

def source_decodability(X, source, seed=42, repeats=5):
    X=l2(X); rows=[]
    for i in range(repeats):
        sss=StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed+i)
        tr,te=next(sss.split(X, source))
        clf=SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, max_iter=80, tol=1e-3, class_weight='balanced', random_state=seed+i)
        clf.fit(X[tr], source[tr])
        pred=clf.predict(X[te])
        rows.append(balanced_accuracy_score(source[te], pred))
    return float(np.mean(rows)), float(np.std(rows))

def pathology_probe(X, Y, source, heldout=1):
    X=l2(X)
    tr=source!=heldout; te=source==heldout
    ytr=(Y[tr][:,EVAL_CLASSES] > 0.5).astype(int); yte=(Y[te][:,EVAL_CLASSES] > 0.5).astype(int)
    scaler=StandardScaler().fit(X[tr])
    Xtr=scaler.transform(X[tr]); Xte=scaler.transform(X[te])
    aucs=[]; aps=[]
    for c in range(ytr.shape[1]):
        if len(np.unique(ytr[:,c])) < 2 or len(np.unique(yte[:,c])) < 2:
            continue
        clf=SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, max_iter=100, tol=1e-3, class_weight='balanced', random_state=42+c)
        clf.fit(Xtr, ytr[:,c])
        scores=clf.decision_function(Xte)
        aucs.append(roc_auc_score(yte[:,c], scores))
        aps.append(average_precision_score(yte[:,c], scores))
    return float(np.mean(aucs)), float(np.mean(aps))

def inlp_directions(X_obs, y_obs, rank, seed):
    # Fast linear source-probe subspace: fit preprocessing on observed sources only,
    # then extract rank-k directions from source-signed standardized features.
    # This keeps the held-out source out of direction estimation and gives a
    # matched-rank source-discriminative removal for table construction.
    Xn=l2(X_obs.copy())
    scaler=StandardScaler().fit(Xn)
    Z=scaler.transform(Xn)
    vals=sorted(np.unique(y_obs))
    if len(vals) < 2:
        return np.zeros((X_obs.shape[1],0))
    ybin=np.where(y_obs==vals[0], -1.0, 1.0)
    # Emphasize between-source signal: signed features plus explicit mean-difference.
    A=Z * ybin[:,None]
    k=min(rank, A.shape[1], max(1, A.shape[0]-1))
    try:
        U,S,Vt=randomized_svd(A, n_components=k, random_state=seed, n_iter=5)
        Q=Vt.T
    except Exception:
        Q,_=np.linalg.qr(A.T)
        Q=Q[:,:k]
    mean_dir=(Z[ybin>0].mean(0)-Z[ybin<0].mean(0))
    if np.linalg.norm(mean_dir)>1e-8:
        mean_dir=mean_dir/np.linalg.norm(mean_dir)
        Q=np.concatenate([mean_dir[:,None], Q], axis=1)
    Q,_=np.linalg.qr(Q)
    Q=Q[:,:min(rank,Q.shape[1])]
    print(f'[SOURCE-SUBSPACE] extracted rank={Q.shape[1]} from observed sources only', flush=True)
    return Q

def project_remove(X, Q):
    if Q.shape[1]==0: return X.copy()
    Xs=l2(X)
    return l2(Xs - (Xs@Q)@Q.T)

def random_subspace(dim, rank, seed):
    rng=np.random.default_rng(seed)
    A=rng.normal(size=(dim,rank))
    Q,_=np.linalg.qr(A)
    return Q[:,:rank]

def inlp_summary(X,Y,source,args):
    full_auc, full_map = pathology_probe(X,Y,source,args.heldout_source_id)
    obs=source!=args.heldout_source_id
    # source removal learned only from observed sources 0/2
    Q=inlp_directions(X[obs], source[obs], args.inlp_rank, args.seed)
    X_inlp=project_remove(X,Q)
    inlp_auc,inlp_map=pathology_probe(X_inlp,Y,source,args.heldout_source_id)
    bacc,_=source_decodability(X_inlp[obs], source[obs], args.seed, repeats=3)
    rand_rows=[]
    for r in range(args.random_repeats):
        R=random_subspace(X.shape[1], min(args.inlp_rank, X.shape[1]), args.seed+1000+r)
        Xr=project_remove(X,R)
        auc,mapv=pathology_probe(Xr,Y,source,args.heldout_source_id)
        rbacc,_=source_decodability(Xr[obs], source[obs], args.seed+r, repeats=1)
        rand_rows.append((auc,mapv,rbacc))
    rand=np.array(rand_rows) if rand_rows else np.zeros((0,3))
    return dict(full_auc=full_auc, full_map=full_map, inlp_bacc=bacc, inlp_auc=inlp_auc, inlp_map=inlp_map,
                delta_auc=inlp_auc-full_auc, delta_map=inlp_map-full_map,
                random_delta_auc_mean=float(np.mean(rand[:,0]-full_auc)) if len(rand) else np.nan,
                random_delta_map_mean=float(np.mean(rand[:,1]-full_map)) if len(rand) else np.nan,
                random_bacc_mean=float(np.mean(rand[:,2])) if len(rand) else np.nan,
                effective_rank=int(Q.shape[1]))

def pair_sample_dist(X,Y,source,seed=42,max_pairs=20000):
    X=l2(X); rng=np.random.default_rng(seed)
    primary=[]
    for row in Y:
        pos=[i for i,v in enumerate(row) if v>0.5 and i!=NO_FINDING]
        primary.append(pos[0] if pos else NO_FINDING)
    primary=np.array(primary)
    vals={'same_src':[],'cross_src':[]}
    for c in EVAL_CLASSES:
        idx=np.where(primary==c)[0]
        if len(idx)<2: continue
        for _ in range(min(max_pairs//len(EVAL_CLASSES), max(1,len(idx)))):
            a,b=rng.choice(idx,2,replace=False)
            d=1-float(np.dot(X[a],X[b]))
            if source[a]==source[b]: vals['same_src'].append(d)
            else: vals['cross_src'].append(d)
    return float(np.mean(vals['same_src'])) if vals['same_src'] else np.nan, float(np.mean(vals['cross_src'])) if vals['cross_src'] else np.nan

def centroid_ratio(X,Y,source):
    X=l2(X); cents={}
    for c in EVAL_CLASSES:
        for s in sorted(set(source)):
            idx=np.where((Y[:,c]>0.5)&(source==s))[0]
            if len(idx)>=5:
                cents[(c,s)]=l2(X[idx].mean(0,keepdims=True))[0]
    same_path=[]; same_src_diff_path=[]
    srcs=sorted(set(source))
    for c in EVAL_CLASSES:
        for i,s1 in enumerate(srcs):
            for s2 in srcs[i+1:]:
                if (c,s1) in cents and (c,s2) in cents:
                    same_path.append(1-float(np.dot(cents[(c,s1)],cents[(c,s2)])))
    for s in srcs:
        cs=[c for c in EVAL_CLASSES if (c,s) in cents]
        for i,c1 in enumerate(cs):
            for c2 in cs[i+1:]:
                same_src_diff_path.append(1-float(np.dot(cents[(c1,s)],cents[(c2,s)])))
    a=float(np.mean(same_path)) if same_path else np.nan
    b=float(np.mean(same_src_diff_path)) if same_src_diff_path else np.nan
    return a,b,a/b if b and not np.isnan(b) else np.nan

def average_precision_binary(rel, scores):
    order=np.argsort(-scores); rel=np.asarray(rel)[order]
    if rel.sum()==0: return np.nan
    prec=np.cumsum(rel)/(np.arange(len(rel))+1)
    return float((prec*rel).sum()/rel.sum())

def retrieval_map(X,Y,source,max_query_per_pair=300,seed=42):
    X=l2(X); rng=np.random.default_rng(seed); aps=[]
    srcs=sorted(set(source))
    for c in EVAL_CLASSES:
        for qs in srcs:
            qidx=np.where((source==qs)&(Y[:,c]>0.5))[0]
            if len(qidx)>max_query_per_pair: qidx=rng.choice(qidx,max_query_per_pair,replace=False)
            for gs in srcs:
                if gs==qs: continue
                gidx=np.where(source==gs)[0]
                rel=(Y[gidx,c]>0.5).astype(int)
                if len(qidx)==0 or rel.sum()==0: continue
                for q in qidx:
                    scores=X[gidx]@X[q]
                    aps.append(average_precision_binary(rel,scores))
    aps=[a for a in aps if not np.isnan(a)]
    return float(np.mean(aps)) if aps else np.nan

def write_tex(df,path):
    cols=['Model','Source BACC','INLP BACC','INLP ΔAUC','INLP ΔmAP','Same-src dist.','Cross-src dist.','Centroid ratio','Retrieval mAP','LOSO AUC','LOSO mAP']
    with open(path,'w') as f:
        f.write('\\begin{tabular}{lcccccccccc}\n\\toprule\n')
        f.write(' & '.join(cols)+'\\\\\n\\midrule\n')
        for _,r in df.iterrows():
            vals=[r['Model']]+[f"{r[c]:.4f}" if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in cols[1:]]
            f.write(' & '.join(vals)+'\\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root', default='/home/hyr/clip/biomedclip_encoder_generalization')
    ap.add_argument('--output-dir', default='/home/hyr/clip/biomedclip_encoder_generalization/motivation_analysis/representation_analysis_table')
    ap.add_argument('--model', default='BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    ap.add_argument('--pretrained', default='/home/hyr/clip/biomedclip_encoder_generalization/BiomedCLIP-PubMedBERT/open_clip_pytorch_model.bin')
    ap.add_argument('--fg-clip-checkpoint', default='/home/hyr/clip/biomedclip_encoder_generalization/baselines_reproduction/fg_clip_cxr/chexpert/fg_clip_cxr.pt')
    ap.add_argument('--train-jsonl', default='/home/hyr/clip/biomedclip_encoder_generalization/dataset/heldout_chexpert/train_mimic_padchest_heldout_chexpert.jsonl')
    ap.add_argument('--test-jsonl', default='/home/hyr/clip/biomedclip_encoder_generalization/dataset/heldout_chexpert/heldout_chexpert_full.jsonl')
    ap.add_argument('--heldout-source-id', type=int, default=1)
    ap.add_argument('--max-per-source', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force-extract', action='store_true')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=96)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--localtopk-k', type=int, default=4)
    ap.add_argument('--localtopk-alpha', type=float, default=0.5)
    ap.add_argument('--inlp-rank', type=int, default=128)
    ap.add_argument('--random-repeats', type=int, default=20)
    args=ap.parse_args()
    if args.dry_run:
        args.max_per_source=500; args.inlp_rank=min(args.inlp_rank,16); args.random_repeats=2; args.batch_size=min(args.batch_size,64)
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows=read_jsonl(args.train_jsonl)+read_jsonl(args.test_jsonl)
    rows=balanced_sample(rows, args.max_per_source, args.seed)
    counts=pd.Series([int(r['source_id']) for r in rows]).value_counts().sort_index().to_dict()
    print('[INFO] samples by source:', counts)
    models=['CLIP','GRADE-CXR_base','GRADE-CXR_ref']
    src_rows=[]; inlp_rows=[]; geom_rows=[]; final=[]
    for m in models:
        print('[MODEL]',m)
        data=extract_features(m, rows, args, out/'feature_cache')
        X=data['features']; Y=data['labels']; S=data['source'].astype(int)
        sb_mean,sb_std=source_decodability(X,S,args.seed, repeats=3 if args.dry_run else 5)
        src_rows.append(dict(Model=m, source_bacc_mean=sb_mean, source_bacc_std=sb_std))
        ins=inlp_summary(X,Y,S,args); inlp_rows.append(dict(Model=m, **ins))
        same_src,cross_src=pair_sample_dist(X,Y,S,args.seed)
        c_cross,c_denom,c_ratio=centroid_ratio(X,Y,S)
        ret=retrieval_map(X,Y,S,seed=args.seed)
        geom_rows.append(dict(Model=m, same_source_distance=same_src, cross_source_distance=cross_src, centroid_cross_source=c_cross, centroid_denominator=c_denom, centroid_ratio=c_ratio, retrieval_mAP=ret))
        final.append(dict(Model=m, **{'Source BACC':sb_mean, 'INLP BACC':ins['inlp_bacc'], 'INLP ΔAUC':ins['delta_auc'], 'INLP ΔmAP':ins['delta_map'], 'Same-src dist.':same_src, 'Cross-src dist.':cross_src, 'Centroid ratio':c_ratio, 'Retrieval mAP':ret, 'LOSO AUC':ins['full_auc'], 'LOSO mAP':ins['full_map']}))
    pd.DataFrame(src_rows).to_csv(out/'source_decodability.csv',index=False)
    pd.DataFrame(inlp_rows).to_csv(out/'inlp_removal_summary.csv',index=False)
    pd.DataFrame(geom_rows).to_csv(out/'representation_geometry_summary.csv',index=False)
    pd.DataFrame([{ 'Model':r['Model'], 'LOSO AUC':r['LOSO AUC'], 'LOSO mAP':r['LOSO mAP']} for r in final]).to_csv(out/'loso_transfer_summary.csv',index=False)
    df=pd.DataFrame(final)
    df.to_csv(out/'representation_analysis_table.csv',index=False)
    write_tex(df,out/'representation_analysis_table.tex')
    with open(out/'analysis_summary.md','w') as f:
        f.write('# Representation Analysis Table (held-out CheXpert)\n\n')
        f.write(f'- Samples by source: {counts}\n')
        f.write(f'- Models: {models}\n')
        f.write(f'- GRADE-CXR_base checkpoint: `{args.fg_clip_checkpoint}`\n')
        f.write(f'- GRADE-CXR_ref: same checkpoint with oracle LocalTopK k={args.localtopk_k}, alpha={args.localtopk_alpha}.\n')
        f.write(f'- Source-discriminative subspace directions are estimated only on observed sources where source_id != {args.heldout_source_id}.\n')
        f.write('- LOSO pathology probe trains on observed sources and tests on held-out CheXpert.\n\n')
        f.write(df.to_markdown(index=False))
        f.write('\n')
    print('[DONE]', out)
    print(df.to_string(index=False))
if __name__=='__main__': main()
