# GRADE-CXR

This repository is a cleaned code release for **GRADE-CXR: Cross-Source Pathology Learning for Transferable Chest X-Ray Representations**.

The release keeps the core training and evaluation code for:

- Stage 1 cross-source pathology-aware CXR vision-language pretraining;
- Stage 2 pathology-guided patch-to-global refinement;
- frozen-feature extraction and linear-probe evaluation;
- representation analysis, PadChest fine-grained label auditing, and paired bootstrap confidence intervals.

Large artifacts are intentionally excluded: raw datasets, cached features, logs, checkpoints, and result folders.

## Repository Layout

```text
open_clip/
  src/open_clip/          # modified OpenCLIP model code and GRADE Stage-2 modules
  src/open_clip_train/    # Stage-1 training, data loading, losses, and CLI arguments
configs/                 # curated Stage-2 and example experiment configs
scripts/stage2/          # Stage-2 training / feature extraction entry points
scripts/evaluation/      # frozen linear probe and low-label evaluation scripts
scripts/analysis/        # representation geometry / source-pathology diagnostics
scripts/bootstrap/       # patient-level paired bootstrap CI utility
scripts/padchest/        # PadChest 193-label audit and fine-grained transfer utilities
experiments/stage2_dynamic_localtopk/
                         # image-only LocalTopK / HardTopK Stage-2 prototype scripts
dataset_manifests/       # lightweight manifest/stat files only, not image data
```

## Stage 1 Default Loss

The released Stage-1 default follows the paper setting without BCE:

```text
L_stage1 = L_disease + lambda_r * L_inst + lambda_cpa * L_USCPA
```

Default values used by the cleaned configuration are:

```text
lambda_r = 1.0
lambda_cpa = 0.05
```

A point-wise / masked BCE branch is kept in the code for ablation compatibility, but it is **off by default** (`grade_bce_weight=0.0`) and should not be described as part of the main Stage-1 method unless explicitly enabled.

## Stage 2

The code includes both the multi-level anchor-sparse Stage-2 implementation and the later image-only LocalTopK / HardTopK refinement scripts used for fast diagnosis. The final paper-facing Stage-2 should be selected explicitly in the experiment config and documented in `docs/IMPLEMENTATION_NOTES.md`.

## What Is Not Included

The following are intentionally not copied into this release tree:

- raw image/report datasets;
- experiment logs and checkpoints;
- temporary ablation folders;
- MLLM replacement experiments;
- baseline reproduction scratch scripts unless needed for the main paper table;
- generated figures and cached `.npy/.npz` features.

