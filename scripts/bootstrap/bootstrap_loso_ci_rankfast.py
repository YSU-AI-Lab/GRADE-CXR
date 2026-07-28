#!/usr/bin/env python
import json, re, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
LABELS=["Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices","No Finding"]
ROOT=Path('/home/hyr/clip/biomedclip_encoder_generalization')
MANIFEST={
'mimic':{'grade_base':'linear_probe_results/stage1_hcpa_p_clip_vitb16_epoch10/heldout_mimic_epoch10_mimic_cxr/predictions.csv','grade_ref_localtop':'linear_probe_results/stage1_hcpa_p_clip_vitb16_epoch10_localtopk_oracle_k4_alpha0.5/heldout_mimic_epoch10_mimic_cxr/predictions.csv','fg_clip_localtop':'linear_probe_results/fgclip_mgll_localtopk_full_seed42/fg_clip/mimic_full_localtopk_oracle_k4_alpha0.5/predictions.csv','mgll_localtop':'linear_probe_results/fgclip_mgll_localtopk_full_seed42/mgll/mimic_full_localtopk_oracle_k4_alpha0.5/predictions.csv','biomedclip_localtop':'linear_probe_results/localtopk_compare_backbones_full_seed42/biomedclip/localtopk/mimic_p100/predictions.csv','clipceil_localtop':'linear_probe_results/localtopk_compare_backbones_full_seed42/clipceil/localtopk/mimic_p100/predictions.csv'},
'chexpert':{'grade_base':'linear_probe_results/stage1_hcpa_p_clip_vitb16_epoch10/heldout_chexpert_epoch10_chexpert_labeled_split/predictions.csv','grade_ref_localtop':'linear_probe_results/stage1_hcpa_p_clip_vitb16_epoch10_localtopk_oracle_k4_alpha0.5/heldout_chexpert_epoch10_chexpert_labeled_split/predictions.csv','fg_clip_localtop':'linear_probe_results/fgclip_mgll_localtopk_full_seed42/fg_clip/chexpert_full_localtopk_oracle_k4_alpha0.5/predictions.csv','mgll_localtop':'linear_probe_results/fgclip_mgll_localtopk_full_seed42/mgll/chexpert_full_localtopk_oracle_k4_alpha0.5/predictions.csv','biomedclip_localtop':'linear_probe_results/localtopk_compare_backbones_full_seed42/biomedclip/localtopk/chexpert_p100/predictions.csv','clipceil_localtop':'linear_probe_results/localtopk_compare_backbones_full_seed42/clipceil/localtopk/chexpert_p100/predictions.csv'},
'padchest':{'grade_base':'linear_probe_results/h0_stage1_hcpa_p_heldout_padchest_nooldcpa_epoch20_padchest_full/predictions.csv','grade_ref_localtop':'linear_probe_results/h0_stage1_hcpa_p_heldout_padchest_nooldcpa_epoch20_padchest_full_localtopk_k4_alpha0.5/predictions.csv','fg_clip_localtop':'linear_probe_results/fgclip_mgll_localtopk_full_seed42/fg_clip/padchest_full_localtopk_oracle_k4_alpha0.5/predictions.csv','mgll_localtop':'linear_probe_results/fgclip_mgll_localtopk_full_seed42/mgll/padchest_full_localtopk_oracle_k4_alpha0.5/predictions.csv','biomedclip_localtop':'linear_probe_results/localtopk_compare_backbones_full_seed42/biomedclip/localtopk/padchest_p100/predictions.csv','clipceil_localtop':'linear_probe_results/localtopk_compare_backbones_full_seed42/clipceil/localtopk/padchest_p100/predictions.csv'}}
BASELINES=['fg_clip_localtop','mgll_localtop','biomedclip_localtop','clipceil_localtop']
def gid(path,fold):
 s=str(path)
 if fold=='mimic':
  m=re.search(r'/p(\d+)/s(\d+)/',s)
  if m: return 'p'+m.group(1)
 if fold=='chexpert':
  m=re.search(r'(patient\d+)[/\\](study\d+)',s,re.I)
  if m: return m.group(1).lower()
 return Path(s).stem
def load(path,fold):
 df=pd.read_csv(ROOT/path).sort_values('filename').reset_index(drop=True)
 info=pd.DataFrame({'filename':df.filename.astype(str)}); info['group']=[gid(x,fold) for x in info.filename]
 y=np.stack([df['label_'+c].to_numpy(float) for c in LABELS],1); p=np.stack([df['prob_'+c].to_numpy(float) for c in LABELS],1)
 mask=np.isfinite(y); y=np.nan_to_num(y,nan=0.0).astype(np.int8)
 return info,y,p,mask
def weighted_auc_sorted(y_sorted,w_sorted):
 pos=(y_sorted==1); neg=~pos
 wp=w_sorted*pos; wn=w_sorted*neg
 P=wp.sum(); N=wn.sum()
 if P<=0 or N<=0: return np.nan
 cum_neg=np.cumsum(wn)-wn
 return float((wp*cum_neg).sum()/(P*N))
def weighted_ap_sorted_desc(y_sorted,w_sorted):
 pos=(y_sorted==1); wp=w_sorted*pos; P=wp.sum()
 if P<=0 or (w_sorted*(~pos)).sum()<=0: return np.nan
 tp=np.cumsum(wp); denom=np.cumsum(w_sorted)
 precision=np.divide(tp,denom,out=np.zeros_like(tp,dtype=float),where=denom>0)
 return float((precision*wp).sum()/P)
def metric_fast(prep,w=None):
 aucs=[]; aps=[]; n=0
 if w is None: w=np.ones(prep['n'],float)
 for cls in prep['classes']:
  valid=cls['valid']
  ww=w[valid]
  if ww.sum()<=0: continue
  asc=cls['asc']; desc=cls['desc']; y=cls['y']
  auc=weighted_auc_sorted(y[asc],ww[asc]); ap=weighted_ap_sorted_desc(y[desc],ww[desc])
  if not np.isnan(auc) and not np.isnan(ap): aucs.append(auc); aps.append(ap); n+=1
 return {'AUC':float(np.mean(aucs)) if aucs else np.nan,'mAP':float(np.mean(aps)) if aps else np.nan,'n_valid_classes':n}
def prep(y,p,mask):
 classes=[]
 for j in range(y.shape[1]):
  valid=mask[:,j]
  yy=y[valid,j]; pp=p[valid,j]
  asc=np.argsort(pp,kind='mergesort'); desc=asc[::-1].copy()
  classes.append({'valid':valid,'y':yy,'asc':asc,'desc':desc})
 return {'n':len(y),'classes':classes}
def raw_sklearn(y,p,mask):
 aucs=[]; aps=[]; n=0
 for j in range(y.shape[1]):
  m=mask[:,j]; yy=y[m,j]; pp=p[m,j]
  if len(yy)==0 or yy.min()==yy.max(): continue
  aucs.append(roc_auc_score(yy,pp)); aps.append(average_precision_score(yy,pp)); n+=1
 return {'AUC':float(np.mean(aucs)),'mAP':float(np.mean(aps)),'n_valid_classes':n}
def bootstrap(data,fold,a,b,rng,n):
 info,prepa,rawa=data[fold][a]; _,prepb,rawb=data[fold][b]
 groups=np.array(sorted(pd.unique(info.group))); code=pd.Categorical(info.group,categories=groups).codes
 diffs=np.empty((n,2),float)
 probs=np.full(len(groups),1/len(groups))
 for i in range(n):
  counts=rng.multinomial(len(groups),probs); w=counts[code].astype(float)
  aa=metric_fast(prepa,w); bb=metric_fast(prepb,w)
  diffs[i]=[100*(aa['AUC']-bb['AUC']),100*(aa['mAP']-bb['mAP'])]
 return rawa,rawb,diffs,len(groups),len(info)
def ci(x): return float(np.nanmean(x)),float(np.nanpercentile(x,2.5)),float(np.nanpercentile(x,97.5))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',default='experiments/loso_paired_bootstrap_ci_rankfast'); ap.add_argument('--n_boot',type=int,default=1000); ap.add_argument('--seed',type=int,default=20260722); args=ap.parse_args()
 out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
 data={}; audit=[]; rawrows=[]
 for fold,methods in MANIFEST.items():
  data[fold]={}; ref=None
  for m,path in methods.items():
   if not (ROOT/path).exists(): audit.append({'fold':fold,'method':m,'path':path,'status':'missing'}); continue
   info,y,p,mask=load(path,fold)
   if ref is None: ref=info.filename.tolist()
   elif ref!=info.filename.tolist(): raise SystemExit(f'Filename mismatch {fold} {m}')
   pr=prep(y,p,mask); rw=raw_sklearn(y,p,mask)
   data[fold][m]=(info,pr,rw); rawrows.append({'fold':fold,'method':m,**rw})
   audit.append({'fold':fold,'method':m,'path':path,'status':'loaded','n':len(info),'n_bootstrap_groups':info.group.nunique(),'grouping':'parsed_patient_except_padchest_image_fallback'})
 pd.DataFrame(audit).to_csv(out/'input_audit.csv',index=False); raw=pd.DataFrame(rawrows); raw.to_csv(out/'raw_metrics_from_predictions.csv',index=False)
 strongest=[]
 for fold in data:
  sub=raw[(raw.fold==fold)&(raw.method.isin(BASELINES))]
  strongest += [{'fold':fold,'metric':'AUC','baseline':sub.sort_values('AUC',ascending=False).iloc[0].method},{'fold':fold,'metric':'mAP','baseline':sub.sort_values('mAP',ascending=False).iloc[0].method}]
 strong=pd.DataFrame(strongest); strong.to_csv(out/'strongest_baseline_by_fold_metric.csv',index=False)
 rng=np.random.default_rng(args.seed); rows=[]; store={}
 def add(fold,a,b,label,only=None):
  ra,rb,d,ng,ni=bootstrap(data,fold,a,b,rng,args.n_boot); store[(fold,label)]=d
  for k,met in enumerate(['AUC','mAP']):
   if only and met!=only: continue
   mean,lo,hi=ci(d[:,k]); rows.append({'scope':'fold','fold':fold,'comparison':label,'metric':met,'raw_delta_pp':100*(ra[met]-rb[met]),'boot_mean_delta_pp':mean,'ci95_low_pp':lo,'ci95_high_pp':hi,'n_groups':ng,'n_images':ni,'valid_classes_a':ra['n_valid_classes'],'valid_classes_b':rb['n_valid_classes']})
 for fold in data:
  add(fold,'grade_ref_localtop','grade_base','GRADE-CXR_ref - GRADE-CXR_base')
  for met in ['AUC','mAP']:
   b=strong.query('fold==@fold and metric==@met').iloc[0].baseline
   add(fold,'grade_ref_localtop',b,f'GRADE-CXR_ref - strongest_baseline_for_{met} ({b})',met)
 for label in ['GRADE-CXR_ref - GRADE-CXR_base']:
  for met,k in [('AUC',0),('mAP',1)]:
   vals=np.vstack([store[(fold,label)][:,k] for fold in data]).mean(0)
   rawd=[100*(raw[(raw.fold==fold)&(raw.method=='grade_ref_localtop')].iloc[0][met]-raw[(raw.fold==fold)&(raw.method=='grade_base')].iloc[0][met]) for fold in data]
   mean,lo,hi=ci(vals); rows.append({'scope':'average','fold':'average','comparison':label,'metric':met,'raw_delta_pp':float(np.mean(rawd)),'boot_mean_delta_pp':mean,'ci95_low_pp':lo,'ci95_high_pp':hi})
 for met,k in [('AUC',0),('mAP',1)]:
  vals=[]; rawd=[]
  for fold in data:
   b=strong.query('fold==@fold and metric==@met').iloc[0].baseline; label=f'GRADE-CXR_ref - strongest_baseline_for_{met} ({b})'
   vals.append(store[(fold,label)][:,k]); rawd.append(100*(raw[(raw.fold==fold)&(raw.method=='grade_ref_localtop')].iloc[0][met]-raw[(raw.fold==fold)&(raw.method==b)].iloc[0][met]))
  vals=np.vstack(vals).mean(0); mean,lo,hi=ci(vals); rows.append({'scope':'average','fold':'average','comparison':f'GRADE-CXR_ref - strongest_baseline_for_{met}','metric':met,'raw_delta_pp':float(np.mean(rawd)),'boot_mean_delta_pp':mean,'ci95_low_pp':lo,'ci95_high_pp':hi})
 res=pd.DataFrame(rows); res.to_csv(out/'paired_bootstrap_ci.csv',index=False)
 (out/'paired_bootstrap_ci.json').write_text(json.dumps({'raw_metrics':raw.to_dict('records'),'strongest':strong.to_dict('records'),'ci':res.to_dict('records'),'note':'MIMIC/CheXpert use parsed patient IDs; PadChest patient ID unavailable in current prediction/split files, so image basename is used as fallback group.'},indent=2),encoding='utf-8')
 lines=['\\begin{tabular}{llccc}','\\toprule','Comparison & Metric & Raw $\\Delta$ & Bootstrap mean $\\Delta$ & 95\\% CI \\','\\midrule']
 for _,r in res[res.scope=='average'].iterrows(): lines.append(f"{r.comparison} & {r.metric} & {r.raw_delta_pp:.2f} & {r.boot_mean_delta_pp:.2f} & [{r.ci95_low_pp:.2f}, {r.ci95_high_pp:.2f}] \\")
 lines+=['\\bottomrule','\\end{tabular}']; (out/'paired_bootstrap_ci_table.tex').write_text('\n'.join(lines),encoding='utf-8')
 s=[]
 for _,r in res[res.scope=='average'].iterrows(): s.append(f"{r.comparison} changed the average {r.metric} by {r.raw_delta_pp:.2f} percentage points (paired bootstrap mean: {r.boot_mean_delta_pp:.2f}, 95% CI: {r.ci95_low_pp:.2f} to {r.ci95_high_pp:.2f}).")
 (out/'bootstrap_result_sentences.txt').write_text('\n'.join(s),encoding='utf-8')
 print('RAW'); print(raw.to_string(index=False)); print('\nSTRONGEST'); print(strong.to_string(index=False)); print('\nCI'); print(res.to_string(index=False)); print('\nSENTENCES'); print('\n'.join(s))
if __name__=='__main__': main()
