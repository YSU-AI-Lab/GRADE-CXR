# GRADE-CXR Code Inventory

This document records what has been selected for the first cleaned code-release tree under `/home/hyr/clip/GRADE`.

## Included Core Code

### 1. Modified OpenCLIP Core

- `open_clip/src/open_clip_train/main.py`: main Stage-1 training entry.
- `open_clip/src/open_clip_train/train.py`: Stage-1 loss composition, US-CPA/HCPA/optional BCE branches, training loop.
- `open_clip/src/open_clip_train/data.py`: JSONL dataset, disease/report fields, labels, masks, source IDs.
- `open_clip/src/open_clip_train/params.py`: CLI arguments for GRADE losses and data fields.
- `open_clip/src/open_clip/*.py`: model, tokenizer, factory, transforms, and GRADE Stage-2 modules.

### 2. Stage 2

- `scripts/stage2/train_stage2.py`
- `scripts/stage2/train_stage2_v2.py`
- `scripts/stage2/train_stage2_v2_cached.py`
- `scripts/stage2/cache_stage2_v2_tokens.py`
- `scripts/stage2/extract_stage2_features.py`
- `scripts/stage2/extract_stage2_v2_features.py`
- `experiments/stage2_dynamic_localtopk/*.py`: LocalTopK / HardTopK diagnostic/refinement scripts only, without result folders.

### 3. Frozen Evaluation

- `scripts/evaluation/eval_cxr_linear_probe.py`
- `scripts/evaluation/eval_cxr_linear_probe_frozen_encoder.py`
- `scripts/evaluation/eval_cxr_linear_probe_localtopk.py`
- `scripts/evaluation/cached_lowlabel_linear_probe_repeats.py`

### 4. Analysis Utilities

- `scripts/analysis/run_representation_analysis_chexpert.py`
- `scripts/analysis/plot_pathology_distance_scatter.py`
- `scripts/analysis/posthoc_mechanism_consistency_checks.py`
- `scripts/bootstrap/bootstrap_loso_ci_rankfast.py`
- `scripts/padchest/audit_padchest_193_labels.py`
- `scripts/padchest/run_padchest_finegrained_31_transfer.py`
- `scripts/padchest/plot_padchest_31_finegrained_distribution.py`

## Deliberately Excluded For Now

- `open_clip/src/logs/` and all checkpoints.
- Full raw datasets and cached features.
- MLLM/Qwen/LLaVA/LLaVA-Med experiments.
- Baseline reproduction scratch experiments: FG-CLIP, CXR-CLIP, DLILP, UniChest, and text-denoising controls.
- One-off plotting scripts for manuscript figures.
- Temporary `.bak_*` files and stale ablation scripts.

These can be moved into `baselines/`, `mllm/`, or `paper_figures/` later if you want a broader artifact release, but they should not be mixed into the first clean method repository.

## Stage 1 BCE Policy

BCE is not part of the default Stage-1 method. The code retains BCE options for ablation only:

- default: `grade_bce_weight=0.0`;
- if enabled, it creates/uses a classification branch in `open_clip_train/train.py`;
- paper-facing Stage 1 should be described without BCE unless an ablation explicitly turns it on.

