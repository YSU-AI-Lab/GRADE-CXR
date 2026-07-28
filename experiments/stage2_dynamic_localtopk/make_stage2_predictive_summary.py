#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd
base=Path('experiments/stage2_dynamic_localtopk')
out=base/'predictive_stage2_summary'
out.mkdir(parents=True,exist_ok=True)
parts=[]
for p,label in [
 (base/'alpha_sweep_predictive_epoch3/results/alpha_sweep_summary.csv','alpha_sweep_0p25_1p25'),
 (base/'alpha_sweep_predictive_epoch3_ext/results/alpha_sweep_ext_summary.csv','alpha_sweep_1p50_2p00'),
]:
 if p.exists():
  df=pd.read_csv(p); df['source_file']=str(p); df['experiment']=label; parts.append(df)
if parts:
 sweep=pd.concat(parts,ignore_index=True)
 sweep['display_variant']=sweep['variant'].replace({'stage1_base_v2extract':'Stage1 base','stage2_v2_ep4_refined':'Predictive Stage2'})
 sweep.to_csv(out/'alpha_sweep_all.csv',index=False)
else:
 sweep=pd.DataFrame()
ms=[]
for p,label,alpha in [
 (base/'multiseed_probe_predictive_alpha125/results/summary_mean_std.csv','predictive_alpha1p25',1.25),
 (base/'multiseed_probe_predictive_alpha2/results/summary_mean_std.csv','predictive_alpha2p00',2.0),
]:
 if p.exists():
  df=pd.read_csv(p); df['alpha']=alpha; df['experiment']=label; ms.append(df)
if ms:
 multi=pd.concat(ms,ignore_index=True)
 multi['display_variant']=multi['variant'].replace({'stage1_base':'Stage1 base','predictive_stage2':'Predictive Stage2'})
 multi.to_csv(out/'multiseed_mean_std_all.csv',index=False)
 # paired deltas vs stage1 within same alpha table
 rows=[]
 for alpha,g in multi.groupby('alpha'):
  b=g[g.variant=='stage1_base'].iloc[0]; r=g[g.variant=='predictive_stage2'].iloc[0]
  rows.append({
   'alpha':alpha,
   'AUC_delta_pp':(r.AUC_mean-b.AUC_mean)*100,
   'mAP_delta_pp':(r.mAP_mean-b.mAP_mean)*100,
   'F1_delta_pp':(r.F1_mean-b.F1_mean)*100,
   'ACC_delta_pp':(r.ACC_mean-b.ACC_mean)*100,
   'AUC_base_mean':b.AUC_mean,'AUC_stage2_mean':r.AUC_mean,
   'mAP_base_mean':b.mAP_mean,'mAP_stage2_mean':r.mAP_mean,
   'F1_base_mean':b.F1_mean,'F1_stage2_mean':r.F1_mean,
   'ACC_base_mean':b.ACC_mean,'ACC_stage2_mean':r.ACC_mean,
  })
 deltas=pd.DataFrame(rows); deltas.to_csv(out/'multiseed_deltas.csv',index=False)
else:
 multi=pd.DataFrame(); deltas=pd.DataFrame()
# Add representation shift estimates from previous component cache.
shift_rows=[]
try:
 import numpy as np
 root=base/'predictive_components_padchest_epoch3/test'
 z=np.load(root/'z_base.npy',mmap_mode='r'); u=np.load(root/'u.npy',mmap_mode='r'); gate=np.load(root/'gate.npy',mmap_mode='r')
 def l2n(x): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
 for a in [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]:
  vals=[]
  for s in range(0,len(z),4096):
   zz=np.asarray(z[s:s+4096],dtype='float32'); uu=np.asarray(u[s:s+4096],dtype='float32'); gg=np.asarray(gate[s:s+4096],dtype='float32')
   zr=l2n(zz+a*gg*uu); vals.append((zr*zz).sum(-1))
  vals=np.concatenate(vals)
  shift_rows.append({'alpha':a,'z_ref_z_base_cos_mean':float(vals.mean()),'p05':float(np.percentile(vals,5)),'p50':float(np.percentile(vals,50)),'p95':float(np.percentile(vals,95))})
 pd.DataFrame(shift_rows).to_csv(out/'zref_zbase_shift_by_alpha.csv',index=False)
except Exception as e:
 (out/'shift_error.txt').write_text(str(e))
md=[]
md.append('# Predictive Stage2 quick validation summary\n')
md.append('Held-out source: PadChest. Frozen Stage-1 checkpoint: `open_clip/src/logs/h0_stage1_hcpa_p_heldout_padchest_nooldcpa/checkpoints/epoch_20.pt`. Linear probe uses the same PadChest train/val/test split and matched probe seeds.\n')
md.append('## Key finding\n')
md.append('The original free-attention Stage2 variants degraded or barely matched Stage1, while the predictive localtopk Stage2 gives a stable improvement. This suggests the useful part of localtopk is disease-conditioned local patch pooling, but the oracle label selection must be replaced by an image-only disease router.\n')
if not deltas.empty:
 md.append('## Multi-seed matched probe\n')
 for _,r in deltas.sort_values('alpha').iterrows():
  md.append(f"- alpha={r.alpha:.2f}: AUC {r.AUC_base_mean:.5f} -> {r.AUC_stage2_mean:.5f} ({r.AUC_delta_pp:+.2f} pp); mAP {r.mAP_base_mean:.5f} -> {r.mAP_stage2_mean:.5f} ({r.mAP_delta_pp:+.2f} pp); F1 delta {r.F1_delta_pp:+.2f} pp; ACC delta {r.ACC_delta_pp:+.2f} pp.\n")
 md.append('\nRecommended paper-facing quick config: alpha=1.25, because it improves AUC/mAP/F1/ACC simultaneously and keeps the representation more conservative than alpha=2.0. Alpha=2.0 gives the best AUC/mAP but lowers ACC and shifts features more strongly.\n')
if shift_rows:
 md.append('\n## Representation preservation on test split\n')
 for r in shift_rows:
  if r['alpha'] in (1.25,2.0): md.append(f"- alpha={r['alpha']:.2f}: mean cosine(z_ref,z_base)={r['z_ref_z_base_cos_mean']:.3f}, p05={r['p05']:.3f}, p95={r['p95']:.3f}.\n")
md.append('\n## Artifact files\n')
md.append('- `alpha_sweep_all.csv`: single-seed alpha sweep.\n')
md.append('- `multiseed_mean_std_all.csv`: 5-seed mean/std for alpha=1.25 and alpha=2.0.\n')
md.append('- `multiseed_deltas.csv`: percentage-point improvements over Stage1 base.\n')
md.append('- `zref_zbase_shift_by_alpha.csv`: representation preservation by alpha.\n')
(out/'README.md').write_text(''.join(md),encoding='utf-8')
print('WROTE',out)
print((out/'README.md').read_text())
