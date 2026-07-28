# Standalone linear probe evaluator for biomedclip_encoder_generalization.
# This copy supports both Stage-1 OpenCLIP checkpoints and Stage-2 CLS-adapter checkpoints.

import argparse
import json
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from tqdm import tqdm

from open_clip import create_model_and_transforms
from open_clip.grade_stage2_cls_adapter import attach_multilevel_cls_adapter


DEFAULT_LABEL_COLS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]


PADCHEST_SPLIT_DIR = "/home/hyr/clip/MedCLIP-SAMv2-main/data/loso_splits_two_text/chexpert14_csv_by_holdout/holdout_sid2_PadChest_full_from_biomedclip_jsonl"
DEFAULT_STAGE2_CKPT = "/home/hyr/clip/biomedclip_encoder_generalization/open_clip/src/logs/h0_stage1_standard_diseaseonly_locked_text_seed42/checkpoints/epoch_16.pt"
DEFAULT_PRETRAINED = "/home/hyr/clip/biomedclip_encoder_generalization/BiomedCLIP-PubMedBERT/open_clip_pytorch_model.bin"
RESULT_ROOT = "/home/hyr/clip/biomedclip_encoder_generalization/linear_probe_results"
DEFAULT_STAGE2_ADAPTER_SCALE = 0.2


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v
    return new_state_dict


def read_checkpoint_state_dict(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    return strip_module_prefix(state_dict)


def checkpoint_has_stage2_adapter(state_dict) -> bool:
    return any(k.startswith("stage2_adapter.") for k in state_dict.keys())


def load_openclip_checkpoint(model, ckpt_path: str, device: str, adapter_layers: int, adapter_dropout: float):
    state_dict = read_checkpoint_state_dict(ckpt_path)
    use_stage2 = checkpoint_has_stage2_adapter(state_dict)

    if use_stage2 and not hasattr(model, "stage2_adapter"):
        model = attach_multilevel_cls_adapter(
            model,
            num_layers=adapter_layers,
            dropout=adapter_dropout,
        )

    msg = model.load_state_dict(state_dict, strict=False)

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    print(f"[INFO] Stage2 adapter checkpoint: {use_stage2}")
    print(f"[INFO] Missing keys: {len(msg.missing_keys)}")
    print(f"[INFO] Unexpected keys: {len(msg.unexpected_keys)}")
    if len(msg.missing_keys) > 0:
        print("[WARN] First missing keys:", msg.missing_keys[:10])
    if len(msg.unexpected_keys) > 0:
        print("[WARN] First unexpected keys:", msg.unexpected_keys[:10])

    model.to(device)
    model.eval()
    return model, use_stage2


class CXRMultiLabelDataset(Dataset):
    def __init__(self, csv_path: str, preprocess, img_key: str, label_cols: List[str], uncertainty_mode: str = "ignore"):
        self.df = pd.read_csv(csv_path)
        self.preprocess = preprocess
        self.img_key = img_key
        self.label_cols = label_cols
        self.uncertainty_mode = uncertainty_mode

        if self.img_key not in self.df.columns:
            raise ValueError(f"Image key '{self.img_key}' not found. CSV columns are: {self.df.columns.tolist()}")
        missing = [c for c in self.label_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing label columns: {missing}\nCSV columns are: {self.df.columns.tolist()}")

        self.df = self.df.dropna(subset=[self.img_key]).reset_index(drop=True)
        missing_images = (~self.df[self.img_key].map(lambda p: os.path.exists(str(p)))).sum()
        if missing_images > 0:
            raise FileNotFoundError(f"{csv_path} contains {missing_images} missing image paths.")
        print(f"[INFO] Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self):
        return len(self.df)

    def _process_label(self, x):
        if pd.isna(x):
            return np.nan
        x = float(x)
        if x == -1:
            if self.uncertainty_mode == "positive":
                return 1.0
            if self.uncertainty_mode == "negative":
                return 0.0
            if self.uncertainty_mode == "ignore":
                return np.nan
            raise ValueError(f"Unknown uncertainty_mode: {self.uncertainty_mode}")
        return 1.0 if x > 0 else 0.0

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(row[self.img_key])
        image = Image.open(img_path).convert("RGB")
        image = self.preprocess(image)
        labels = torch.tensor([self._process_label(row[c]) for c in self.label_cols], dtype=torch.float32)
        return image, labels, img_path


class LinearProbeModel(nn.Module):
    def __init__(self, clip_model, image_dim: int, num_classes: int, use_stage2_adapter: bool, adapter_scale: float):
        super().__init__()
        self.clip_model = clip_model
        self.use_stage2_adapter = bool(use_stage2_adapter)
        self.adapter_scale = float(adapter_scale)
        self.classifier = nn.Linear(image_dim, num_classes)
        for p in self.clip_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_features(self, images):
        if self.use_stage2_adapter:
            z_base, cls_tokens = self.clip_model.encode_image(images, normalize=True, return_cls_tokens=True)
            z_adp = self.clip_model.stage2_adapter(cls_tokens.to(dtype=z_base.dtype))
            feats = torch.nn.functional.normalize(z_base + self.adapter_scale * z_adp, dim=-1)
        else:
            feats = self.clip_model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def forward(self, images):
        image_features = self.encode_features(images)
        return self.classifier(image_features)


@torch.no_grad()
def infer_image_dim(clip_model, device: str, use_stage2_adapter: bool, adapter_scale: float, image_size: int = 224):
    dummy = torch.randn(1, 3, image_size, image_size).to(device)
    if use_stage2_adapter:
        z_base, cls_tokens = clip_model.encode_image(dummy, normalize=True, return_cls_tokens=True)
        z_adp = clip_model.stage2_adapter(cls_tokens.to(dtype=z_base.dtype))
        feat = torch.nn.functional.normalize(z_base + adapter_scale * z_adp, dim=-1)
    else:
        feat = clip_model.encode_image(dummy)
    return feat.shape[-1]


def masked_bce_loss(logits, labels):
    valid = ~torch.isnan(labels)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    labels_filled = torch.nan_to_num(labels, nan=0.0)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels_filled, reduction="none")
    return loss[valid].mean()


def compute_metrics(y_true, y_prob, label_cols, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.float32)
    aucs, aps, accs = [], [], []
    per_class = {}
    for i, name in enumerate(label_cols):
        valid = ~np.isnan(y_true[:, i])
        yt = y_true[valid, i]
        yp = y_prob[valid, i]
        yd = y_pred[valid, i]
        if len(yt) == 0:
            auc = ap = acc = np.nan
            num_pos = num_neg = 0
        else:
            num_pos = int(np.sum(yt == 1))
            num_neg = int(np.sum(yt == 0))
            if len(np.unique(yt)) < 2:
                auc = ap = np.nan
            else:
                auc = roc_auc_score(yt, yp)
                ap = average_precision_score(yt, yp)
            acc = accuracy_score(yt, yd)
        per_class[name] = {
            "AUC": None if np.isnan(auc) else float(auc),
            "AP": None if np.isnan(ap) else float(ap),
            "ACC": None if np.isnan(acc) else float(acc),
            "num_valid": int(valid.sum()),
            "num_pos": num_pos,
            "num_neg": num_neg,
        }
        if not np.isnan(auc):
            aucs.append(auc)
        if not np.isnan(ap):
            aps.append(ap)
        if not np.isnan(acc):
            accs.append(acc)
    return {
        "macro_AUC": float(np.mean(aucs)) if aucs else None,
        "mAP": float(np.mean(aps)) if aps else None,
        "macro_ACC": float(np.mean(accs)) if accs else None,
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, label_cols, threshold=0.5, desc="Evaluating"):
    model.eval()
    all_probs, all_labels, all_paths = [], [], []
    total_loss = 0.0
    total_batches = 0
    for images, labels, paths in tqdm(dataloader, desc=desc, ncols=120):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = masked_bce_loss(logits, labels)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_paths.extend(paths)
        total_loss += float(loss.item())
        total_batches += 1
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(y_true, y_prob, label_cols, threshold=threshold)
    metrics["loss"] = total_loss / max(total_batches, 1)
    return metrics, y_prob, y_true, all_paths


def train_one_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_batches = 0
    for images, labels, _ in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{total_epochs}", ncols=120):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = masked_bce_loss(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def build_default_output_dir(checkpoint: str, dataset_name: str = "padchest_full") -> str:
    ckpt = os.path.abspath(checkpoint)
    ckpt_stem = os.path.splitext(os.path.basename(ckpt))[0]
    parent = os.path.basename(os.path.dirname(ckpt))
    if parent == "checkpoints":
        run_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
    else:
        run_name = os.path.splitext(os.path.basename(ckpt))[0]
    return os.path.join(RESULT_ROOT, f"{run_name}_{ckpt_stem}_{dataset_name}_linear_probe")


def save_predictions(output_dir, paths, y_prob, y_true, label_cols):
    pred_df = pd.DataFrame({"filename": paths})
    for i, name in enumerate(label_cols):
        pred_df[f"prob_{name}"] = y_prob[:, i]
        pred_df[f"label_{name}"] = y_true[:, i]
    pred_path = os.path.join(output_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"[INFO] Saved predictions to: {pred_path}")


def make_loader(csv_path, preprocess, args, label_cols, shuffle=False):
    dataset = CXRMultiLabelDataset(
        csv_path=csv_path,
        preprocess=preprocess,
        img_key=args.img_key,
        label_cols=label_cols,
        uncertainty_mode=args.uncertainty_mode,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default=f"{PADCHEST_SPLIT_DIR}/train.csv")
    parser.add_argument("--val_csv", type=str, default=f"{PADCHEST_SPLIT_DIR}/val.csv")
    parser.add_argument("--test_csv", type=str, default=f"{PADCHEST_SPLIT_DIR}/test.csv")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_STAGE2_CKPT)
    parser.add_argument("--model", type=str, default="BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--pretrained", type=str, default=DEFAULT_PRETRAINED)
    parser.add_argument("--img_key", type=str, default="Image Index")
    parser.add_argument("--label_cols", type=str, default=",".join(DEFAULT_LABEL_COLS))
    parser.add_argument("--uncertainty_mode", type=str, default="ignore", choices=["ignore", "positive", "negative"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--adapter-scale", type=float, default=DEFAULT_STAGE2_ADAPTER_SCALE, help="Stage-2 z_ref = z_base + scale * z_adp.")
    parser.add_argument("--adapter-layers", type=int, default=1, help="Needed to instantiate Stage-2 adapter before loading checkpoint.")
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = build_default_output_dir(args.checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    label_cols = [x.strip() for x in args.label_cols.split(",") if x.strip()]
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output dir: {args.output_dir}")
    print(f"[INFO] Label columns: {label_cols}")

    clip_model, _, preprocess_val = create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        device=device,
        output_dict=True,
    )
    clip_model, use_stage2 = load_openclip_checkpoint(
        model=clip_model,
        ckpt_path=args.checkpoint,
        device=device,
        adapter_layers=args.adapter_layers,
        adapter_dropout=args.adapter_dropout,
    )

    image_dim = infer_image_dim(
        clip_model=clip_model,
        device=device,
        use_stage2_adapter=use_stage2,
        adapter_scale=args.adapter_scale,
    )
    print(f"[INFO] Image feature dim: {image_dim}")
    print(f"[INFO] Linear probe feature source: {'Stage2 z_ref' if use_stage2 else 'base image feature'}")

    model = LinearProbeModel(
        clip_model=clip_model,
        image_dim=image_dim,
        num_classes=len(label_cols),
        use_stage2_adapter=use_stage2,
        adapter_scale=args.adapter_scale,
    ).to(device)

    train_loader = make_loader(args.train_csv, preprocess_val, args, label_cols, shuffle=True)
    val_loader = make_loader(args.val_csv, preprocess_val, args, label_cols, shuffle=False)
    test_loader = make_loader(args.test_csv, preprocess_val, args, label_cols, shuffle=False)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auc = -1.0
    best_path = os.path.join(args.output_dir, "best_linear_probe.pt")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        val_metrics, _, _, _ = evaluate(model, val_loader, device, label_cols, args.threshold, desc=f"Validating Epoch {epoch + 1}/{args.epochs}")
        val_auc = val_metrics["macro_AUC"]
        print(f"[Epoch {epoch + 1}/{args.epochs}] train_loss={train_loss:.6f} val_AUC={val_auc} val_mAP={val_metrics['mAP']} val_ACC={val_metrics['macro_ACC']}")
        if val_auc is not None and val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.classifier.state_dict(), best_path)
            print(f"[INFO] Saved best linear probe to: {best_path}")

    if os.path.exists(best_path):
        model.classifier.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[INFO] Loaded best linear probe from: {best_path}")

    test_metrics, y_prob, y_true, paths = evaluate(model, test_loader, device, label_cols, args.threshold, desc="Testing")
    print("\n========== Test Metrics ==========")
    print(f"macro_AUC: {test_metrics['macro_AUC']}")
    print(f"mAP:       {test_metrics['mAP']}")
    print(f"macro_ACC: {test_metrics['macro_ACC']}")
    print(f"loss:      {test_metrics['loss']}")
    print("\n========== Per-class Metrics ==========")
    for name, vals in test_metrics["per_class"].items():
        print(f"{name:28s} AUC={vals['AUC']} AP={vals['AP']} ACC={vals['ACC']} valid={vals['num_valid']} pos={vals['num_pos']} neg={vals['num_neg']}")

    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Saved metrics to: {metrics_path}")
    save_predictions(args.output_dir, paths, y_prob, y_true, label_cols)


if __name__ == "__main__":
    main()
