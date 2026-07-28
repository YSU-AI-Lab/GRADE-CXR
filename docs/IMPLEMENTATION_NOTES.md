# Implementation Notes

## Stage 1

The main Stage-1 training entry is OpenCLIP-style:

```bash
python -m open_clip_train.main [args]
```

The default GRADE-CXR Stage-1 objective is:

```text
L_stage1 = L_disease + lambda_r * L_inst + lambda_cpa * L_USCPA
```

where:

- `L_disease`: image-to-disease-anchor contrastive alignment;
- `L_inst`: image-to-report/description instance-level contrastive alignment;
- `L_USCPA`: cross-source pathology-aware alignment term;
- `lambda_r = 1.0`;
- `lambda_cpa = 0.05`.

The optional BCE branch remains available for controlled ablations but is disabled by default and should not be considered part of the default Stage-1 method.

## Stage 2

Stage-2 code is separated from Stage-1. It loads a Stage-1 checkpoint, freezes the visual and text encoders, trains only the refinement module, and exports either base or refined features for the common frozen linear probe.

Two families are currently kept:

- anchor-sparse multi-level patch-to-global refinement in `open_clip/src/open_clip/grade_stage2_v2.py` and `scripts/stage2/train_stage2_v2.py`;
- image-only LocalTopK / HardTopK refiners in `experiments/stage2_dynamic_localtopk/`.

Before public release, choose one as the official Stage-2 method and move the other to an ablation/appendix folder.

## Linear Probe

The common frozen evaluation trains only a linear classifier on frozen image features. The usual configuration in the current project is:

- L2-normalized frozen features;
- linear classifier;
- `BCEWithLogitsLoss` with valid-label masking;
- no `pos_weight` by default;
- AdamW optimizer;
- same train/val/test split across methods.

