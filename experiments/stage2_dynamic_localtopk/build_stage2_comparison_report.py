#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd, json
root=Path('experiments/stage2_dynamic_localtopk')
out=root/'predictive_stage2_summary'
out.mkdir(parents=True,exist_ok=True)
rows=[]
# Current stage2 single-seed cached eval.
p=Path('experiments/stage2_current_effect_eval/results/summary.csv')
if p.exists():
    df=pd.read_csv(p)
    name_map={
        'stage1_base_v2extract':'Stage1 base',
        'stage2_v2_ep4_refined':'Current Stage2 v2 epoch4',
        'stage2_simple_ep2_refined':'Simple cross-attn Stage2 epoch2',
    }
    for _,r in df.iterrows():
        rows.append({'setting':'single_seed_probe','variant':name_map.get(r.variant,r.variant),'AUC':r.AUC,'mAP':r.mAP,'F1':r.F1,'ACC':r.ACC,'notes':'current implementation cached eval'})
# Dynamic epoch1 single-seed, if exists. Its label in old script is misleading.
p=Path('experiments/stage2_dynamic_localtopk/eval_epoch1/results/summary.csv')
if p.exists():
    df=pd.read_csv(p)
    for _,r in df.iterrows():
        if r.variant=='stage2_v2_ep4_refined':
            rows.append({'setting':'single_seed_probe','variant':'Dynamic query localtopk imitation epoch1','AUC':r.AUC,'mAP':r.mAP,'F1':r.F1,'ACC':r.ACC,'notes':'old label stage2_v2_ep4_refined was a symlink to dynamic epoch1'})
# Predictive alpha sweep single-seed.
for p in [root/'alpha_sweep_predictive_epoch3/results/alpha_sweep_summary.csv',root/'alpha_sweep_predictive_epoch3_ext/results/alpha_sweep_ext_summary.csv']:
    if p.exists():
        df=pd.read_csv(p)
        for _,r in df.iterrows():
            if r.variant=='stage2_v2_ep4_refined':
                rows.append({'setting':'single_seed_probe','variant':f'Predictive localtopk Stage2 alpha={r.alpha:g}','AUC':r.AUC,'mAP':r.mAP,'F1':r.F1,'ACC':r.ACC,'notes':'offline alpha materialization from predictive epoch3 components'})
comp=pd.DataFrame(rows)
if not comp.empty:
    base=comp[(comp.setting=='single_seed_probe') & (comp.variant=='Stage1 base')].head(1)
    if not base.empty:
        b=base.iloc[0]
        for m in ['AUC','mAP','F1','ACC']:
            comp[f'delta_{m}_pp']= (comp[m]-b[m])*100
    comp.to_csv(out/'stage2_variant_comparison_single_seed.csv',index=False)
# Multi-seed deltas already computed; make a concise table.
p=out/'multiseed_mean_std_all.csv'
if p.exists():
    ms=pd.read_csv(p)
    ms.to_csv(out/'stage2_predictive_multiseed_table.csv',index=False)
# Add diagnosis markdown.
md=[]
md.append('# Stage2 diagnosis and selected implementation\n\n')
md.append('## What failed\n')
md.append('- The free cross-attention Stage2 variants changed the representation too much and did not improve PadChest linear probe. Current Stage2 v2 epoch4 had lower AUC/mAP than Stage1 base in cached evaluation. Simple cross-attention was worse.\n')
md.append('- The large gains from previous LocalTopK were not directly deployable because oracle LocalTopK used test labels to choose active diseases. Non-oracle prediction-based LocalTopK was much weaker, showing that disease selection is the hard part.\n\n')
md.append('## What worked\n')
md.append('- Predictive localtopk Stage2 replaces oracle disease selection with an image-only disease router from z_base. It uses frozen disease anchors to compute per-disease top-k local vectors, predicts disease weights from z_base, and adds a gated local mixture to z_base.\n')
md.append('- It preserves the paper constraint that inference only needs the input image and the frozen Stage1 backbone plus refiner. No labels, reports, source IDs, or teacher anchors are needed during inference.\n\n')
md.append('## Recommended quick configuration\n')
md.append('- Stage1 checkpoint: `open_clip/src/logs/h0_stage1_hcpa_p_heldout_padchest_nooldcpa/checkpoints/epoch_20.pt`\n')
md.append('- Cached training: `experiments/stage2_dynamic_localtopk/predictive_alpha05_ep3`\n')
md.append('- Recommended inference alpha: `1.25` for balanced AUC/mAP/F1/ACC and conservative representation shift.\n')
md.append('- Higher alpha `2.0` gives best AUC/mAP but reduces ACC and shifts farther from z_base.\n\n')
if (out/'multiseed_deltas.csv').exists():
    d=pd.read_csv(out/'multiseed_deltas.csv')
    md.append('## Verified matched 5-seed probe deltas\n')
    for _,r in d.sort_values('alpha').iterrows():
        md.append(f"- alpha={r.alpha:g}: ΔAUC {r.AUC_delta_pp:+.2f} pp, ΔmAP {r.mAP_delta_pp:+.2f} pp, ΔF1 {r.F1_delta_pp:+.2f} pp, ΔACC {r.ACC_delta_pp:+.2f} pp.\n")
md.append('\n## Files\n')
md.append('- `stage2_variant_comparison_single_seed.csv`: quick comparison against current Stage2 implementations.\n')
md.append('- `stage2_predictive_multiseed_table.csv`: matched 5-seed mean/std.\n')
md.append('- `README.md`: compact result summary.\n')
(out/'stage2_diagnosis.md').write_text(''.join(md),encoding='utf-8')
print((out/'stage2_diagnosis.md').read_text())
print('--- comparison ---')
if (out/'stage2_variant_comparison_single_seed.csv').exists(): print(pd.read_csv(out/'stage2_variant_comparison_single_seed.csv').to_string(index=False))
