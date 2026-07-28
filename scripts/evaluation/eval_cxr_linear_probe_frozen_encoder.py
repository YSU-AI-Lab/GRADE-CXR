#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve, roc_auc_score


DEFAULT_LABEL_COLS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pil_to_tensor(image, image_size, mean, std):
    image = image.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(image).astype("float32") / 255.0
    arr = (arr - np.asarray(mean, dtype="float32")) / np.asarray(std, dtype="float32")
    return torch.from_numpy(arr).permute(2, 0, 1)


class CXRDataset(Dataset):
    def __init__(self, csv_path, img_key, label_cols, uncertainty_mode, transform):
        self.df = pd.read_csv(csv_path).dropna(subset=[img_key]).reset_index(drop=True)
        self.img_key = img_key
        self.label_cols = label_cols
        self.uncertainty_mode = uncertainty_mode
        self.transform = transform
        missing = [c for c in label_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing label columns: {missing}")
        missing_images = (~self.df[img_key].map(lambda p: os.path.exists(str(p)))).sum()
        if missing_images:
            raise FileNotFoundError(f"{csv_path} contains {missing_images} missing image paths.")
        print(f"[INFO] Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self):
        return len(self.df)

    def _label(self, x):
        if pd.isna(x):
            return np.nan
        x = float(x)
        if x == -1:
            if self.uncertainty_mode == "positive":
                return 1.0
            if self.uncertainty_mode == "negative":
                return 0.0
            return np.nan
        return 1.0 if x > 0 else 0.0

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row[self.img_key])
        image = Image.open(path).convert("RGB")
        labels = torch.tensor([self._label(row[c]) for c in self.label_cols], dtype=torch.float32)
        return self.transform(image), labels, path


class FrozenProbe(nn.Module):
    def __init__(self, encoder, dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(dim, num_classes)
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, images):
        feats = self.encoder(images)
        return feats / feats.norm(dim=-1, keepdim=True)

    def forward(self, images):
        return self.classifier(self.encode(images))


class EVAXEncoder(nn.Module):
    def __init__(self, repo_dir, checkpoint):
        super().__init__()
        sys.path.insert(0, repo_dir)
        old_load = torch.load
        def load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return old_load(*args, **kwargs)
        torch.load = load_compat
        from eva_x import eva_x_base_patch16
        try:
            self.model = eva_x_base_patch16(pretrained=checkpoint)
        finally:
            torch.load = old_load
        self.model.eval()

    def forward(self, x):
        feats = self.model.forward_features(x)
        return self.model.forward_head(feats, pre_logits=True)


class TransformersAutoEncoder(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.model.eval()

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]


def build_encoder(args, device):
    if args.encoder == "eva_x_base":
        encoder = EVAXEncoder(args.repo_dir, args.checkpoint)
        image_size = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif args.encoder == "transformers_auto":
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
        encoder = TransformersAutoEncoder(args.model_dir)
        size = getattr(proc, "size", {}) or {}
        image_size = int(size.get("height") or size.get("shortest_edge") or 518)
        mean = tuple(getattr(proc, "image_mean", [0.485, 0.456, 0.406]))
        std = tuple(getattr(proc, "image_std", [0.229, 0.224, 0.225]))
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")

    encoder.to(device).eval()
    transform = lambda img: pil_to_tensor(img, image_size, mean, std)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        dim = int(encoder(dummy).shape[-1])
    print(f"[INFO] encoder={args.encoder} image_size={image_size} feature_dim={dim}")
    return encoder, transform, dim


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


def compute_metrics(y_true, y_prob, label_cols, threshold=0.5):
    thresholds = threshold_array(threshold, y_prob.shape[1])
    y_pred = (y_prob >= thresholds[None, :]).astype(np.float32)
    aucs, aps, f1s, accs, per_class = [], [], [], [], {}
    for i, name in enumerate(label_cols):
        valid = ~np.isnan(y_true[:, i])
        yt, yp, yd = y_true[valid, i], y_prob[valid, i], y_pred[valid, i]
        num_pos = int(np.sum(yt == 1)) if len(yt) else 0
        num_neg = int(np.sum(yt == 0)) if len(yt) else 0
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
            "num_pos": num_pos,
            "num_neg": num_neg,
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


@torch.no_grad()
def evaluate(model, loader, device, label_cols, threshold=0.5, desc="Evaluating"):
    model.eval()
    probs, labels_all, paths_all = [], [], []
    total_loss = 0.0
    batches = 0
    for images, labels, paths in tqdm(loader, desc=desc, ncols=120):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(masked_bce_loss(logits, labels).item())
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels_all.append(labels.cpu().numpy())
        paths_all.extend(paths)
        batches += 1
    y_prob = np.concatenate(probs, axis=0)
    y_true = np.concatenate(labels_all, axis=0)
    metrics = compute_metrics(y_true, y_prob, label_cols, threshold)
    metrics["loss"] = total_loss / max(batches, 1)
    return metrics, y_prob, y_true, paths_all


def train_one_epoch(model, loader, optimizer, device, epoch, epochs):
    model.train()
    total = 0.0
    batches = 0
    for images, labels, _ in tqdm(loader, desc=f"Training Epoch {epoch + 1}/{epochs}", ncols=120):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        loss = masked_bce_loss(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        batches += 1
    return total / max(batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=["eva_x_base", "transformers_auto"])
    parser.add_argument("--repo_dir", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model_dir", default="")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--img_key", default="Image Index")
    parser.add_argument("--label_cols", default=",".join(DEFAULT_LABEL_COLS))
    parser.add_argument("--uncertainty_mode", default="ignore", choices=["ignore", "positive", "negative"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    label_cols = [x.strip() for x in args.label_cols.split(",") if x.strip()]
    encoder, transform, dim = build_encoder(args, device)

    model = FrozenProbe(encoder, dim, len(label_cols)).to(device)
    train_ds = CXRDataset(args.train_csv, args.img_key, label_cols, args.uncertainty_mode, transform)
    val_ds = CXRDataset(args.val_csv, args.img_key, label_cols, args.uncertainty_mode, transform)
    test_ds = CXRDataset(args.test_csv, args.img_key, label_cols, args.uncertainty_mode, transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auc = -1.0
    best_path = os.path.join(args.output_dir, "best_linear_probe.pt")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        val_metrics, _, _, _ = evaluate(model, val_loader, device, label_cols, args.threshold, f"Validating Epoch {epoch + 1}/{args.epochs}")
        print(f"[Epoch {epoch + 1}/{args.epochs}] train_loss={train_loss:.6f} val_AUC={val_metrics['macro_AUC']} val_mAP={val_metrics['mAP']} val_F1={val_metrics['macro_F1']} val_ACC={val_metrics['macro_ACC']}")
        if val_metrics["macro_AUC"] is not None and val_metrics["macro_AUC"] > best_auc:
            best_auc = val_metrics["macro_AUC"]
            torch.save(model.classifier.state_dict(), best_path)
    if os.path.exists(best_path):
        model.classifier.load_state_dict(torch.load(best_path, map_location=device))

    val_metrics, val_prob, val_true, _ = evaluate(model, val_loader, device, label_cols, args.threshold, "Selecting validation F1 thresholds")
    thresholds, threshold_info = select_validation_f1_thresholds(val_true, val_prob, label_cols, args.threshold)
    with open(os.path.join(args.output_dir, "validation_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({
            "selection": "per-class threshold selected on validation set by maximizing F1",
            "thresholds": {name: float(thresholds[i]) for i, name in enumerate(label_cols)},
            "per_class": threshold_info,
            "validation_metrics_at_default_threshold": val_metrics,
        }, f, indent=4, ensure_ascii=False)

    test_metrics, y_prob, y_true, paths = evaluate(model, test_loader, device, label_cols, thresholds, "Testing")
    test_metrics["threshold_selection"] = "per-class validation threshold maximizing F1"
    test_metrics["feature_extractor"] = args.encoder
    print("\n========== Test Metrics ==========")
    print(f"macro_AUC: {test_metrics['macro_AUC']}")
    print(f"mAP:       {test_metrics['mAP']}")
    print(f"macro_F1:  {test_metrics['macro_F1']}")
    print(f"macro_ACC: {test_metrics['macro_ACC']}")
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4, ensure_ascii=False)
    pred = pd.DataFrame({"filename": paths})
    for i, name in enumerate(label_cols):
        pred[f"prob_{name}"] = y_prob[:, i]
        pred[f"label_{name}"] = y_true[:, i]
    pred.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)


if __name__ == "__main__":
    main()
