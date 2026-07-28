#!/usr/bin/env python3
"""Cached frozen-feature low-label repeated linear probe.

This script is intentionally independent from existing result folders. It first
extracts frozen features once per method/source split, then trains linear heads
on cached features for repeated seed/fraction settings. Metrics follow the
current linear probe scripts: macro AUC, mAP, macro F1, macro ACC with per-class
F1 thresholds selected on validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve, roc_auc_score

LABEL_COLS_14 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]

MIMIC_SPLIT_ROOT = "/home/hyr/clip/MedCLIP-SAMv2-main/data/loso_splits_two_text/chexpert14_csv_by_holdout/holdout_sid0_MIMIC-CXR"
CHEXPERT_SPLIT_ROOT = "/home/hyr/clip/MedCLIP-SAMv2-main/biomedclip_finetuning/open_clip/configs/source_split"
PADCHEST_SPLIT_ROOT = "/home/hyr/clip/MedCLIP-SAMv2-main/data/loso_splits_two_text/chexpert14_csv_by_holdout/holdout_sid2_PadChest_full_from_biomedclip_jsonl"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_label(x, uncertainty_mode="ignore"):
    if pd.isna(x):
        return np.nan
    x = float(x)
    if x == -1:
        if uncertainty_mode == "positive":
            return 1.0
        if uncertainty_mode == "negative":
            return 0.0
        return np.nan
    return 1.0 if x > 0 else 0.0


class CSVDataset(Dataset):
    def __init__(self, csv_path, img_key, label_cols, transform, uncertainty_mode="ignore"):
        self.csv_path = str(csv_path)
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=[img_key]).reset_index(drop=True)
        self.img_key = img_key
        self.label_cols = label_cols
        self.transform = transform
        self.uncertainty_mode = uncertainty_mode
        missing = [c for c in label_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing label columns in {csv_path}: {missing}")
        bad = (~self.df[img_key].map(lambda p: os.path.exists(str(p)))).sum()
        if bad:
            raise FileNotFoundError(f"{csv_path} has {bad} missing images")
        print(f"[DATA] {csv_path}: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row[self.img_key])
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor([normalize_label(row[c], self.uncertainty_mode) for c in self.label_cols], dtype=torch.float32)
        return x, y, path, idx


def target_paths(target: str) -> Tuple[str, str, str, str]:
    t = target.lower()
    if t in {"mimic", "mimic-cxr"}:
        return "mimic", f"{MIMIC_SPLIT_ROOT}/train.csv", f"{MIMIC_SPLIT_ROOT}/val.csv", f"{MIMIC_SPLIT_ROOT}/test.csv"
    if t in {"chexpert", "chexpert-plus", "chexpertplus"}:
        return "chexpert", f"{CHEXPERT_SPLIT_ROOT}/chexpert_train.csv", f"{CHEXPERT_SPLIT_ROOT}/chexpert_val.csv", f"{CHEXPERT_SPLIT_ROOT}/chexpert_test.csv"
    if t == "padchest":
        return "padchest", f"{PADCHEST_SPLIT_ROOT}/train.csv", f"{PADCHEST_SPLIT_ROOT}/val.csv", f"{PADCHEST_SPLIT_ROOT}/test.csv"
    raise ValueError(f"Unknown target: {target}")


def ckpt_for_openclip(method: str, target: str, work_root: str) -> Tuple[str, float, int]:
    seed = 42
    m = method.lower()
    t = target.lower()
    if m == "biomedclip":
        return f"{work_root}/BiomedCLIP-PubMedBERT/open_clip_pytorch_model.bin", 0.2, 1
    # GRADE-CXR Stage-1 checkpoints for LOSO-style frozen representation evaluation.
    # grade_stage2_localtopk reuses the same Stage-1 backbone and applies the existing
    # oracle LocalTopK patch-to-global refinement during feature extraction.
    if m == "grade_stage2_localtopk":
        m = "grade_stage1"
    table = {
        ("mgll", "mimic"): f"{work_root}/open_clip/src/logs/compare_mgll_heldout_mimic_seed{seed}/checkpoints/best.pt",
        ("mgll", "chexpert"): f"{work_root}/open_clip/src/logs/compare_mgll_heldout_chexpert_seed{seed}/checkpoints/best.pt",
        ("mgll", "padchest"): f"{work_root}/open_clip/src/logs/compare_mgll_heldout_padchest_seed{seed}/checkpoints/best.pt",
        ("clipceil", "mimic"): f"{work_root}/open_clip/src/logs/compare_clipceil_heldout_mimic_seed{seed}/checkpoints/best.pt",
        ("clipceil", "chexpert"): f"{work_root}/open_clip/src/logs/compare_clipceil_heldout_chexpert_seed{seed}/checkpoints/best.pt",
        ("clipceil", "padchest"): f"{work_root}/open_clip/src/logs/compare_clipceil_heldout_padchest_seed{seed}/checkpoints/best.pt",
        ("clip", "mimic"): f"{work_root}/open_clip/src/logs/h0_stage1_basic_dualclip_heldout_mimic_seed42/checkpoints/epoch_11.pt",
        ("clip", "chexpert"): f"{work_root}/open_clip/src/logs/h0_stage1_basic_dualclip_heldout_chexpert_seed42/checkpoints/epoch_12.pt",
        ("clip", "padchest"): f"{work_root}/open_clip/src/logs/h0_heldout_padchest_mimic_chexpert_lanr_disease_clipdesc05_seed42/epoch_20.pt",
        ("grade_stage1", "mimic"): f"{work_root}/open_clip/src/logs/h0_stage1_basic_dualclip_heldout_mimic_seed42/checkpoints/epoch_11.pt",
        ("grade_stage1", "chexpert"): f"{work_root}/open_clip/src/logs/h0_stage1_basic_dualclip_heldout_chexpert_seed42/checkpoints/epoch_12.pt",
        ("grade_stage1", "padchest"): f"{work_root}/open_clip/src/logs/h0_stage1_hcpa_p_heldout_padchest_nooldcpa/checkpoints/epoch_20.pt",
    }
    ckpt = table.get((m, t))
    if not ckpt:
        raise ValueError(f"No OpenCLIP checkpoint mapping for {method}/{target}")
    scale = 1.0 if m == "clipceil" else 0.2
    return ckpt, scale, 1


def build_openclip_extractor(method, target, args, device):
    sys.path.insert(0, f"{args.work_root}/open_clip/src")
    from open_clip import create_model_and_transforms, get_tokenizer
    from eval_cxr_linear_probe_localtopk import load_openclip_checkpoint, infer_image_dim, LinearProbeModel, encode_pathology_anchors
    ckpt, adapter_scale, adapter_layers = ckpt_for_openclip(method, target, args.work_root)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    model, _, preprocess = create_model_and_transforms(
        args.model_name, pretrained=args.biomedclip_pretrained, device=device, output_dict=True
    )
    model, use_stage2, use_stage2_refiner = load_openclip_checkpoint(
        model=model,
        ckpt_path=ckpt,
        device=device,
        adapter_layers=adapter_layers,
        adapter_dropout=0.0,
    )
    use_local_topk = method.lower() == "grade_stage2_localtopk"
    if use_local_topk and hasattr(getattr(model, "visual", None), "output_tokens"):
        model.visual.output_tokens = True
    local_topk_anchors = None
    if use_local_topk:
        tokenizer = get_tokenizer(args.model_name)
        local_topk_anchors = encode_pathology_anchors(model, tokenizer, LABEL_COLS_14, device)
        print(f"[INFO] LocalTopK enabled for {method}/{target}: anchors={tuple(local_topk_anchors.shape)} k={args.local_topk_k} alpha={args.local_topk_alpha}")
    dim = infer_image_dim(model, device, use_stage2_adapter=use_stage2 and not use_stage2_refiner, adapter_scale=adapter_scale)
    wrapper = LinearProbeModel(
        clip_model=model,
        image_dim=dim,
        num_classes=len(LABEL_COLS_14),
        use_stage2_adapter=use_stage2 and not use_stage2_refiner,
        adapter_scale=adapter_scale,
        use_stage2_refiner=use_stage2_refiner,
        use_local_topk=use_local_topk,
        local_topk_anchors=local_topk_anchors,
        local_topk=args.local_topk_k,
        local_topk_alpha=args.local_topk_alpha,
        local_topk_mode=args.local_topk_mode,
        local_topk_pred_topm=args.local_topk_pred_topm,
        local_topk_soft_tau=args.local_topk_soft_tau,
    ).to(device).eval()

    @torch.no_grad()
    def encode(images, labels=None):
        return wrapper.encode_features(images.to(device), labels=labels.to(device) if labels is not None else None).float()

    meta = {"encoder_type": "openclip", "checkpoint": ckpt, "feature_dim": dim, "adapter_scale": adapter_scale, "use_local_topk": use_local_topk, "local_topk_k": args.local_topk_k if use_local_topk else None, "local_topk_alpha": args.local_topk_alpha if use_local_topk else None, "local_topk_mode": args.local_topk_mode if use_local_topk else None}
    return encode, preprocess, dim, meta


def build_original_clip_extractor(args, device):
    from transformers import CLIPImageProcessor, CLIPModel
    model = CLIPModel.from_pretrained(args.clip_vitb16_dir, local_files_only=True).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(args.clip_vitb16_dir, local_files_only=True)
    transform = lambda img: proc(images=img, return_tensors="pt")["pixel_values"][0]
    with torch.no_grad():
        dummy = torch.zeros(1, 3, proc.size.get("shortest_edge", 224), proc.size.get("shortest_edge", 224), device=device)
        dim = int(model.get_image_features(pixel_values=dummy).shape[-1])

    @torch.no_grad()
    def encode(images, labels=None):
        feats = model.get_image_features(pixel_values=images.to(device))
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return encode, transform, dim, {"encoder_type": "hf_clip", "model_dir": args.clip_vitb16_dir, "feature_dim": dim}


def pil_to_tensor(image, image_size, mean, std):
    image = image.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(image).astype("float32") / 255.0
    arr = (arr - np.asarray(mean, dtype="float32")) / np.asarray(std, dtype="float32")
    return torch.from_numpy(arr).permute(2, 0, 1)


def build_rad_dino_extractor(args, device):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(args.rad_dino_dir, local_files_only=True)
    model = AutoModel.from_pretrained(args.rad_dino_dir, local_files_only=True).to(device).eval()
    size = getattr(proc, "size", {}) or {}
    image_size = int(size.get("height") or size.get("shortest_edge") or 518)
    mean = tuple(getattr(proc, "image_mean", [0.485, 0.456, 0.406]))
    std = tuple(getattr(proc, "image_std", [0.229, 0.224, 0.225]))
    transform = lambda img: pil_to_tensor(img, image_size, mean, std)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        out = model(pixel_values=dummy)
        feats = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        dim = int(feats.shape[-1])

    @torch.no_grad()
    def encode(images, labels=None):
        out = model(pixel_values=images.to(device))
        feats = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return encode, transform, dim, {"encoder_type": "rad_dino", "model_dir": args.rad_dino_dir, "feature_dim": dim}


def build_extractor(method, target, args, device):
    if method in {"clip", "clipceil", "mgll", "biomedclip", "grade_stage1", "grade_stage2_localtopk"}:
        return build_openclip_extractor(method, target, args, device)
    if method == "original_clip":
        return build_original_clip_extractor(args, device)
    if method == "rad_dino":
        return build_rad_dino_extractor(args, device)
    raise ValueError(f"Unknown method: {method}")


def extract_split(csv_path, out_npz, encode, transform, args, device):
    if out_npz.exists() and not args.force_extract:
        print(f"[CACHE] reuse {out_npz}")
        return
    ds = CSVDataset(csv_path, args.img_key, LABEL_COLS_14, transform, args.uncertainty_mode)
    loader = DataLoader(ds, batch_size=args.extract_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    feats, labels, paths, indices = [], [], [], []
    for images, y, p, idx in tqdm(loader, desc=f"extract {out_npz.name}", ncols=120):
        z = encode(images, y)
        feats.append(z.detach().cpu().numpy().astype("float32"))
        labels.append(y.numpy().astype("float32"))
        paths.extend(list(p))
        indices.extend(idx.numpy().tolist())
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        features=np.concatenate(feats, axis=0),
        labels=np.concatenate(labels, axis=0),
        paths=np.asarray(paths, dtype=object),
        row_indices=np.asarray(indices, dtype=np.int64),
    )
    print(f"[CACHE] wrote {out_npz}")


def ensure_feature_cache(method, target, train_csv, val_csv, test_csv, args, device):
    cache_dir = Path(args.cache_root) / method / target
    needed = [cache_dir / "train_full.npz", cache_dir / "val.npz", cache_dir / "test.npz"]
    if all(p.exists() for p in needed) and not args.force_extract:
        print(f"[CACHE] all present method={method} target={target}")
        return cache_dir
    encode, transform, dim, meta = build_extractor(method, target, args, device)
    extract_split(train_csv, cache_dir / "train_full.npz", encode, transform, args, device)
    extract_split(val_csv, cache_dir / "val.npz", encode, transform, args, device)
    extract_split(test_csv, cache_dir / "test.npz", encode, transform, args, device)
    with (cache_dir / "feature_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    return cache_dir


def masked_bce_loss(logits, labels):
    valid = ~torch.isnan(labels)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.nan_to_num(labels, nan=0.0), reduction="none")
    return loss[valid].mean()


def threshold_array(threshold, n):
    if np.isscalar(threshold):
        return np.full(n, float(threshold), dtype=np.float32)
    return np.asarray(threshold, dtype=np.float32)


def select_validation_f1_thresholds(y_true, y_prob, label_cols, default_threshold=0.5):
    thresholds = np.full(len(label_cols), default_threshold, dtype=np.float32)
    info = {}
    for i, name in enumerate(label_cols):
        valid = ~np.isnan(y_true[:, i])
        yt, yp = y_true[valid, i], y_prob[valid, i]
        if len(yt) == 0 or len(np.unique(yt)) < 2:
            info[name] = {"threshold": float(thresholds[i]), "validation_F1": None, "note": "fallback_default_threshold"}
            continue
        p, r, th = precision_recall_curve(yt, yp)
        f1 = (2 * p * r) / np.clip(p + r, 1e-12, None)
        candidates = np.concatenate([th, [1.0]])
        best = int(np.nanargmax(f1))
        thresholds[i] = float(candidates[min(best, len(candidates) - 1)])
        info[name] = {"threshold": float(thresholds[i]), "validation_F1": float(f1[best])}
    return thresholds, info


def compute_metrics(y_true, y_prob, threshold=0.5):
    thresholds = threshold_array(threshold, y_prob.shape[1])
    y_pred = (y_prob >= thresholds[None, :]).astype(np.float32)
    aucs, aps, f1s, accs, per_class = [], [], [], [], {}
    for i, name in enumerate(LABEL_COLS_14):
        valid = ~np.isnan(y_true[:, i])
        yt, yp, yd = y_true[valid, i], y_prob[valid, i], y_pred[valid, i]
        if len(yt) == 0:
            auc = ap = f1 = acc = np.nan
        else:
            auc = np.nan if len(np.unique(yt)) < 2 else roc_auc_score(yt, yp)
            ap = np.nan if len(np.unique(yt)) < 2 else average_precision_score(yt, yp)
            acc = accuracy_score(yt, yd)
            tp = float(np.sum((yt == 1) & (yd == 1)))
            fp = float(np.sum((yt == 0) & (yd == 1)))
            fn = float(np.sum((yt == 1) & (yd == 0)))
            f1 = 0.0 if (2 * tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn)
        per_class[name] = {
            "AUC": None if np.isnan(auc) else float(auc),
            "AP": None if np.isnan(ap) else float(ap),
            "F1": None if np.isnan(f1) else float(f1),
            "ACC": None if np.isnan(acc) else float(acc),
            "threshold": float(thresholds[i]),
            "num_valid": int(valid.sum()),
            "num_pos": int(np.sum(yt == 1)) if len(yt) else 0,
            "num_neg": int(np.sum(yt == 0)) if len(yt) else 0,
        }
        if not np.isnan(auc): aucs.append(auc)
        if not np.isnan(ap): aps.append(ap)
        if not np.isnan(f1): f1s.append(f1)
        if not np.isnan(acc): accs.append(acc)
    return {
        "macro_AUC": float(np.mean(aucs)) if aucs else None,
        "mAP": float(np.mean(aps)) if aps else None,
        "macro_F1": float(np.mean(f1s)) if f1s else None,
        "macro_ACC": float(np.mean(accs)) if accs else None,
        "per_class": per_class,
    }


class FeatureLinear(nn.Module):
    def __init__(self, dim, classes):
        super().__init__()
        self.classifier = nn.Linear(dim, classes)

    def forward(self, x):
        return self.classifier(x)


def predict_probs(model, x, batch_size, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i:i+batch_size]).float().to(device)
            outs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_probe_cached(cache_dir: Path, base_train_csv: str, method: str, target: str, frac: float, seed: int, args, device):
    tag = f"p{int(round(frac*100)):03d}"
    out_dir = Path(args.result_root) / method / target / tag / f"seed_{seed}"
    if (out_dir / "test_metrics.json").exists() and not args.force_probe:
        print(f"[DONE_SKIP] {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    train_npz = np.load(cache_dir / "train_full.npz", allow_pickle=True)
    val_npz = np.load(cache_dir / "val.npz", allow_pickle=True)
    test_npz = np.load(cache_dir / "test.npz", allow_pickle=True)
    x_train_full = train_npz["features"].astype("float32")
    y_train_full = train_npz["labels"].astype("float32")
    paths_train_full = train_npz["paths"]
    x_val, y_val = val_npz["features"].astype("float32"), val_npz["labels"].astype("float32")
    x_test, y_test = test_npz["features"].astype("float32"), test_npz["labels"].astype("float32")

    n_full = len(x_train_full)
    n = n_full if frac >= 0.999 else max(1, int(round(n_full * frac)))
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(n_full, size=n, replace=False))
    x_train, y_train = x_train_full[idx], y_train_full[idx]

    pd.DataFrame({"selected_order": np.arange(len(idx)), "row_index": idx, "image_path": paths_train_full[idx]}).to_csv(out_dir / "sample_ids.csv", index=False)

    set_seed(seed)
    model = FeatureLinear(x_train.shape[1], y_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    gen = torch.Generator(); gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=args.probe_batch_size, shuffle=True, generator=gen)
    best_auc = -1.0
    best_state = None
    history = []
    for ep in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = masked_bce_loss(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_prob = predict_probs(model, x_val, args.eval_batch_size, device)
        val_metrics = compute_metrics(y_val, val_prob, threshold=0.5)
        val_auc = val_metrics["macro_AUC"]
        history.append({"epoch": ep+1, "train_loss": float(np.mean(losses)), **{k: val_metrics[k] for k in ["macro_AUC", "mAP", "macro_F1", "macro_ACC"]}})
        if val_auc is not None and val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, out_dir / "best_linear_probe.pt")
    val_prob = predict_probs(model, x_val, args.eval_batch_size, device)
    thresholds, th_info = select_validation_f1_thresholds(y_val, val_prob, LABEL_COLS_14, default_threshold=args.threshold)
    test_prob = predict_probs(model, x_test, args.eval_batch_size, device)
    test_metrics = compute_metrics(y_test, test_prob, threshold=thresholds)
    test_metrics.update({
        "method": method, "target": target, "fraction": frac, "seed": seed,
        "num_train_full": int(n_full), "num_train_selected": int(n),
        "threshold_selection": "per-class validation threshold maximizing F1",
        "feature_cache": str(cache_dir),
    })
    with (out_dir / "test_metrics.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)
    with (out_dir / "validation_thresholds.json").open("w") as f:
        json.dump({"thresholds": {c: float(thresholds[i]) for i,c in enumerate(LABEL_COLS_14)}, "per_class": th_info}, f, indent=2)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    print(f"[RESULT] {method} {target} {tag} seed={seed} AUC={test_metrics['macro_AUC']} mAP={test_metrics['mAP']} out={out_dir}")


def summarize(result_root: str):
    root = Path(result_root)
    rows = []
    for p in sorted(root.glob("*/*/*/seed_*/test_metrics.json")):
        d = json.load(open(p))
        rows.append({
            "method": d.get("method", p.parts[-5]), "target": d.get("target", p.parts[-4]),
            "fraction": d.get("fraction"), "seed": d.get("seed"),
            "AUC": d.get("macro_AUC"), "mAP": d.get("mAP"),
            "F1": d.get("macro_F1"), "ACC": d.get("macro_ACC"), "metrics_path": str(p),
        })
    if not rows:
        print("[SUMMARY] no rows")
        return
    pd.DataFrame(rows).to_csv(root / "per_seed_results.csv", index=False)
    summary = []
    for (method, target, frac), g in pd.DataFrame(rows).groupby(["method", "target", "fraction"]):
        rec = {"method": method, "target": target, "fraction": frac, "n": len(g)}
        for metric in ["AUC", "mAP", "F1", "ACC"]:
            rec[f"{metric}_mean"] = float(g[metric].mean())
            rec[f"{metric}_std"] = float(g[metric].std(ddof=1)) if len(g) > 1 else 0.0
        summary.append(rec)
    pd.DataFrame(summary).sort_values(["method", "target", "fraction"]).to_csv(root / "summary_mean_std.csv", index=False)
    print(f"[SUMMARY] wrote {root/'per_seed_results.csv'} and {root/'summary_mean_std.csv'} rows={len(rows)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work-root", default="/home/hyr/clip/biomedclip_encoder_generalization")
    p.add_argument("--cache-root", default="/home/hyr/clip/biomedclip_encoder_generalization/linear_probe_feature_cache/lowlabel_repeats_backbones")
    p.add_argument("--result-root", default="/home/hyr/clip/biomedclip_encoder_generalization/linear_probe_results/lowlabel_repeats_cached_seed0_4_backbones")
    p.add_argument("--methods", default="clip,clipceil,mgll,rad_dino,biomedclip,original_clip")
    p.add_argument("--targets", default="mimic,chexpert,padchest")
    p.add_argument("--fractions", default="0.01,0.10,0.50,1.00")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--device", default="cuda")
    p.add_argument("--img-key", default="Image Index")
    p.add_argument("--uncertainty-mode", default="ignore", choices=["ignore", "positive", "negative"])
    p.add_argument("--model-name", default="BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    p.add_argument("--biomedclip-pretrained", default="/home/hyr/clip/biomedclip_encoder_generalization/BiomedCLIP-PubMedBERT/open_clip_pytorch_model.bin")
    p.add_argument("--clip-vitb16-dir", default="/home/hyr/clip/biomedclip_encoder_generalization/clip-vit-base-patch16")
    p.add_argument("--rad-dino-dir", default="/home/hyr/clip/checkpoints/rad-dino")
    p.add_argument("--extract-batch-size", type=int, default=128)
    p.add_argument("--probe-batch-size", type=int, default=4096)
    p.add_argument("--eval-batch-size", type=int, default=8192)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--local-topk-k", type=int, default=8)
    p.add_argument("--local-topk-alpha", type=float, default=0.2)
    p.add_argument("--local-topk-mode", default="oracle", choices=["oracle", "all", "pred_topm", "soft"])
    p.add_argument("--local-topk-pred-topm", type=int, default=5)
    p.add_argument("--local-topk-soft-tau", type=float, default=0.07)
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--force-probe", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--only-summarize", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.only_summarize:
        summarize(args.result_root)
        return
    device = args.device if torch.cuda.is_available() else "cpu"
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    fracs = [float(x) for x in re.split(r"[, ]+", args.fractions.strip()) if x]
    seeds = [int(x) for x in re.split(r"[, ]+", args.seeds.strip()) if x]
    Path(args.result_root).mkdir(parents=True, exist_ok=True)
    with open(Path(args.result_root)/"experiment_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    for target in targets:
        slug, train_csv, val_csv, test_csv = target_paths(target)
        for method in methods:
            print(f"\n=== method={method} target={slug} ===")
            cache_dir = Path(args.cache_root) / method / slug
            if not args.skip_extract:
                cache_dir = ensure_feature_cache(method, slug, train_csv, val_csv, test_csv, args, device)
            elif not cache_dir.exists():
                raise FileNotFoundError(f"cache missing: {cache_dir}")
            for seed in seeds:
                for frac in fracs:
                    train_probe_cached(cache_dir, train_csv, method, slug, frac, seed, args, device)
            summarize(args.result_root)
    summarize(args.result_root)


if __name__ == "__main__":
    main()
