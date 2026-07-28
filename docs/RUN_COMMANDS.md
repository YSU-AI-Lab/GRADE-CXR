# Example Commands

These commands are templates. Replace manifest, checkpoint, and output paths before running.

## Environment

```bash
cd /path/to/GRADE/open_clip
pip install -e .
pip install -r requirements-training.txt
export PYTHONPATH=/path/to/GRADE/open_clip/src:$PYTHONPATH
```

## Stage 1: GRADE-CXR Pretraining

Stage 1 default does **not** use BCE. BCE remains available only as an optional ablation by setting `--grade-bce-weight > 0`.

```bash
python -m open_clip_train.main \
  --train-data /path/to/train_two_source.jsonl \
  --dataset-type jsonl \
  --model ViT-B-16 \
  --pretrained /path/to/initial_checkpoint.pt \
  --batch-size 32 \
  --epochs 20 \
  --grade-disease-weight 1.0 \
  --grade-description-weight 1.0 \
  --grade-lambda-rep 1.0 \
  --grade-lambda-hcpa 0.05 \
  --grade-bce-weight 0.0 \
  --logs /path/to/output_logs
```

## Stage 2: Refinement

```bash
python scripts/stage2/train_stage2_v2.py \
  --config configs/stage2_v2_anchor_sparse.yaml \
  --stage1-checkpoint /path/to/stage1_checkpoint.pt \
  --train-manifest /path/to/train_manifest.jsonl \
  --val-manifest /path/to/val_manifest.jsonl \
  --output-dir /path/to/stage2_output
```

## Feature Extraction

```bash
python scripts/stage2/extract_stage2_v2_features.py \
  --stage1-checkpoint /path/to/stage1_checkpoint.pt \
  --stage2-checkpoint /path/to/stage2_checkpoint.pt \
  --manifest /path/to/test_manifest.jsonl \
  --output /path/to/features.npz \
  --feature-kind refined
```

## Frozen Linear Probe

```bash
python scripts/evaluation/eval_cxr_linear_probe.py \
  --train-features /path/to/train_features.npz \
  --val-features /path/to/val_features.npz \
  --test-features /path/to/test_features.npz \
  --output-dir /path/to/probe_results
```

## Patient-Level Paired Bootstrap CI

```bash
python scripts/bootstrap/bootstrap_loso_ci_rankfast.py \
  --predictions-dir /path/to/predictions \
  --output-dir /path/to/bootstrap_ci \
  --n-bootstrap 1000 \
  --seed 42
```

