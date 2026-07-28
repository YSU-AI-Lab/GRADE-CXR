#!/usr/bin/env python3
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("/home/hyr/clip/biomedclip_encoder_generalization")
OUT_DIR = ROOT / "motivation_analysis/padchest_193_label_audit/finegrained_transfer_31labels"
AUDIT_DIR = ROOT / "motivation_analysis/padchest_193_label_audit"
LABEL_SET = AUDIT_DIR / "padchest_finegrained_31_label_set.csv"
PHYSICIAN_193 = Path("/home/hyr/clip/UniChest-main/A1_DATA/Physician_label193_all.csv")
PADCHEST_SPLIT = Path(
    "/home/hyr/clip/MedCLIP-SAMv2-main/data/loso_splits_two_text/"
    "chexpert14_csv_by_holdout/holdout_sid2_PadChest_full_from_biomedclip_jsonl"
)

OPENCLIP_MODEL = "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BIOMED_PRETRAINED = ROOT / "BiomedCLIP-PubMedBERT/open_clip_pytorch_model.bin"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2norm(x, eps=1e-12):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)


def basename(path):
    return os.path.basename(str(path))


def build_matched_csvs(label_names):
    split_dir = OUT_DIR / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    usecols = ["img_path"] + label_names
    labels = pd.read_csv(PHYSICIAN_193, usecols=usecols)
    labels["basename"] = labels["img_path"].map(basename)
    for c in label_names:
        labels[c] = pd.to_numeric(labels[c], errors="coerce").fillna(0).clip(0, 1).astype(int)
    out_paths = {}
    for split in ["train", "val", "test"]:
        src = pd.read_csv(PADCHEST_SPLIT / f"{split}.csv")
        src["basename"] = src["Image Index"].map(basename)
        merged = src[["Image Index", "basename"]].merge(labels[["basename"] + label_names], on="basename", how="inner")
        out = split_dir / f"{split}_31labels.csv"
        merged.to_csv(out, index=False)
        out_paths[split] = out
        print(f"[SPLIT] {split}: {len(merged)} matched samples -> {out}")
    return out_paths


class ImagePathDataset(Dataset):
    def __init__(self, csv_path, label_names, transform_kind, processor=None, preprocess=None):
        self.df = pd.read_csv(csv_path)
        self.label_names = label_names
        self.transform_kind = transform_kind
        self.processor = processor
        self.preprocess = preprocess
        miss = (~self.df["Image Index"].map(lambda p: os.path.exists(str(p)))).sum()
        if miss:
            raise FileNotFoundError(f"{csv_path} has {miss} missing images")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(str(row["Image Index"])).convert("RGB")
        if self.transform_kind == "hf":
            x = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        else:
            x = self.preprocess(img)
        y = torch.tensor(row[self.label_names].astype(float).values, dtype=torch.float32)
        return x, y, str(row["Image Index"])


class HFCLIPEncoder(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        from transformers import CLIPImageProcessor, CLIPModel
        self.processor = CLIPImageProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_dir, local_files_only=True)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x):
        z = self.model.get_image_features(pixel_values=x)
        return torch.nn.functional.normalize(z, dim=-1)


class HFAutoEncoder(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x):
        out = self.model(pixel_values=x)
        z = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        return torch.nn.functional.normalize(z, dim=-1)


class OpenCLIPEncoder(nn.Module):
    def __init__(self, checkpoint=None, stage2_mode=None):
        super().__init__()
        sys.path.insert(0, str(ROOT / "open_clip/src"))
        from open_clip import create_model_and_transforms
        from eval_cxr_linear_probe_localtopk import load_openclip_checkpoint

        model, _, preprocess = create_model_and_transforms(
            OPENCLIP_MODEL,
            pretrained=str(BIOMED_PRETRAINED),
        )
        self.preprocess = preprocess
        self.use_stage2_adapter = False
        self.use_stage2_refiner = False
        self.adapter_scale = 1.0
        if checkpoint:
            model, use_stage2, use_refiner = load_openclip_checkpoint(
                model,
                str(checkpoint),
                "cpu",
                adapter_layers=1,
                adapter_dropout=0.0,
                stage2_refiner_mode="soft",
                stage2_refiner_topk=4,
                stage2_refiner_alpha=0.5,
                stage2_refiner_layers="last4",
                stage2_refiner_num_queries=4,
            )
            self.use_stage2_refiner = bool(use_refiner)
            self.use_stage2_adapter = bool(use_stage2 and not use_refiner)
        self.model = model.eval()
        self.stage2_mode = stage2_mode

    @torch.no_grad()
    def forward(self, x):
        if self.use_stage2_refiner:
            z_base, patch_tokens = self.model.encode_image(x, normalize=True, return_tokens=True)
            if self.model.stage2_refiner.__class__.__name__ == "Stage2PatchGlobalRefiner":
                z = self.model.stage2_refiner(
                    z_base, patch_tokens, token_cache=getattr(self.model.visual, "grade_stage2_token_cache", None)
                )
            else:
                z = self.model.stage2_refiner(z_base, patch_tokens)
        elif self.use_stage2_adapter:
            z_base, cls_tokens = self.model.encode_image(x, normalize=True, return_cls_tokens=True)
            z_adp = self.model.stage2_adapter(cls_tokens.to(dtype=z_base.dtype))
            z = torch.nn.functional.normalize(z_base + self.adapter_scale * z_adp, dim=-1)
        else:
            z = self.model.encode_image(x)
            z = torch.nn.functional.normalize(z, dim=-1)
        return z


def build_encoder(method):
    configs = {
        "PubMedCLIP": ("hf_clip", ROOT / "pubmed-clip-vit-base-patch32", "Off-the-shelf PubMedCLIP ViT-B/32"),
        "BioMedCLIP": ("openclip", None, f"Off-the-shelf BioMedCLIP: {BIOMED_PRETRAINED}"),
        "RAD-DINO": ("hf_auto", Path("/home/hyr/clip/checkpoints/rad-dino"), "Off-the-shelf microsoft/rad-dino"),
        "CLIP": ("hf_clip", ROOT / "clip-vit-base-patch16", "Off-the-shelf OpenAI CLIP ViT-B/16"),
        "CLIPCEIL": ("openclip", ROOT / "open_clip/src/logs/compare_clipceil_heldout_padchest_seed42/checkpoints/best.pt", "CLIPCEIL held-out PadChest best.pt"),
        "MGLL": ("openclip", ROOT / "open_clip/src/logs/compare_mgll_heldout_padchest_seed42/checkpoints/best.pt", "MGLL held-out PadChest best.pt"),
        "GRADE-CXR_base": ("openclip", ROOT / "open_clip/src/logs/h0_heldout_padchest_mimic_chexpert_lanr_disease_clipdesc05_seed42/epoch_20.pt", "GRADE-CXR Stage-1 held-out PadChest epoch_20.pt"),
        "GRADE-CXR_ref": ("openclip", ROOT / "open_clip/src/logs/h0_stage2_v2_distill_multilocaltopk_anchorattn05_from_3source_all_epoch20_seed42/checkpoints/epoch_10.pt", "GRADE-CXR Stage-2 refiner epoch_10.pt"),
    }
    kind, path, desc = configs[method]
    if path is not None and not Path(path).exists():
        raise FileNotFoundError(f"Missing {method} weight/model path: {path}")
    if kind == "hf_clip":
        enc = HFCLIPEncoder(str(path))
        return enc, "hf", enc.processor, None, desc
    if kind == "hf_auto":
        enc = HFAutoEncoder(str(path))
        return enc, "hf", enc.processor, None, desc
    if kind == "openclip":
        enc = OpenCLIPEncoder(checkpoint=path)
        return enc, "openclip", None, enc.preprocess, desc
    raise ValueError(kind)


@torch.no_grad()
def extract_features(method, split_paths, label_names, device, batch_size=64, workers=4):
    cache_dir = OUT_DIR / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    cache_used = {}
    enc, transform_kind, processor, preprocess, desc = build_encoder(method)
    enc.to(device).eval()
    if method == "RAD-DINO":
        batch_size = min(batch_size, 24)
    for split, csv_path in split_paths.items():
        cache = cache_dir / f"{method.replace('/', '_')}_{split}.npz"
        if cache.exists():
            data = np.load(cache, allow_pickle=True)
            outputs[split] = {"features": data["features"], "labels": data["labels"], "paths": data["paths"].tolist()}
            cache_used[split] = True
            print(f"[CACHE] {method} {split}: {cache}")
            continue
        ds = ImagePathDataset(csv_path, label_names, transform_kind, processor=processor, preprocess=preprocess)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
        feats, labs, paths = [], [], []
        for x, y, p in tqdm(dl, desc=f"extract {method} {split}", ncols=110):
            x = x.to(device, non_blocking=True)
            z = enc(x).detach().cpu().float().numpy()
            feats.append(z)
            labs.append(y.numpy())
            paths.extend(list(p))
        feat = l2norm(np.concatenate(feats, axis=0).astype("float32"))
        lab = np.concatenate(labs, axis=0).astype("float32")
        np.savez_compressed(cache, features=feat, labels=lab, paths=np.array(paths, dtype=object))
        outputs[split] = {"features": feat, "labels": lab, "paths": paths}
        cache_used[split] = False
        print(f"[SAVE] {method} {split}: {cache} {feat.shape}")
    del enc
    torch.cuda.empty_cache()
    return outputs, cache_used, desc


class LinearHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)
    def forward(self, x):
        return self.fc(x)


def train_classifier(data, seed=42, epochs=50, lr=1e-3, wd=1e-4, batch_size=256, patience=8, device="cuda"):
    set_seed(seed)
    xtr = torch.tensor(data["train"]["features"], dtype=torch.float32)
    ytr = torch.tensor(data["train"]["labels"], dtype=torch.float32)
    xva = torch.tensor(data["val"]["features"], dtype=torch.float32)
    yva = torch.tensor(data["val"]["labels"], dtype=torch.float32)
    dim, nc = xtr.shape[1], ytr.shape[1]
    model = LinearHead(dim, nc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    tr_dl = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True)
    best_state, best_map, bad = None, -1.0, 0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        val_prob = predict(model, xva.numpy(), device=device)
        val_map = macro_metrics(yva.numpy(), val_prob)[1]
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_mAP": val_map})
        if val_map > best_map:
            best_map = val_map
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, x, device="cuda", batch_size=4096):
    model.eval()
    probs = []
    for i in range(0, len(x), batch_size):
        xb = torch.tensor(x[i:i+batch_size], dtype=torch.float32, device=device)
        probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs, axis=0)


def per_label_metrics(y, p, label_names):
    rows, skipped = [], []
    for i, name in enumerate(label_names):
        yt, yp = y[:, i], p[:, i]
        pos, neg = int((yt == 1).sum()), int((yt == 0).sum())
        auc = ap = np.nan
        if len(np.unique(yt)) >= 2:
            auc = roc_auc_score(yt, yp)
            ap = average_precision_score(yt, yp)
        else:
            skipped.append(name)
        rows.append({"label": name, "AUC": auc, "AP": ap, "num_pos": pos, "num_neg": neg})
    return pd.DataFrame(rows), skipped


def macro_metrics(y, p):
    vals = per_label_metrics(y, p, [str(i) for i in range(y.shape[1])])[0]
    return float(vals["AUC"].mean(skipna=True)), float(vals["AP"].mean(skipna=True))


def group_metrics(per_df, label_meta):
    m = per_df.merge(label_meta[["label", "finegrained_prevalence_group"]], on="label", how="left")
    rows = []
    for group in ["Frequent", "Intermediate", "Rare", "Overall"]:
        sub = m if group == "Overall" else m[m["finegrained_prevalence_group"] == group]
        rows.append({
            "group": group,
            "num_labels": int(len(sub)),
            "AUC": float(sub["AUC"].mean(skipna=True)),
            "mAP": float(sub["AP"].mean(skipna=True)),
        })
    return pd.DataFrame(rows)


def bootstrap_ci(y, p, label_names, label_meta, method, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    rows, skipped = [], []
    n = len(y)
    group_labels = {
        g: label_meta.index[label_meta["finegrained_prevalence_group"].eq(g)].tolist()
        for g in ["Frequent", "Intermediate", "Rare"]
    }
    for group, idxs in group_labels.items():
        vals_auc, vals_ap = [], []
        skipped_count = 0
        for b in range(n_boot):
            ii = rng.integers(0, n, size=n)
            aucs, aps = [], []
            for j in idxs:
                yt, yp = y[ii, j], p[ii, j]
                if len(np.unique(yt)) < 2:
                    skipped_count += 1
                    continue
                aucs.append(roc_auc_score(yt, yp))
                aps.append(average_precision_score(yt, yp))
            if aucs:
                vals_auc.append(float(np.mean(aucs)))
                vals_ap.append(float(np.mean(aps)))
        for metric, vals in [("AUC", vals_auc), ("mAP", vals_ap)]:
            rows.append({
                "method": method,
                "group": group,
                "metric": metric,
                "mean_bootstrap": float(np.mean(vals)) if vals else np.nan,
                "ci_low": float(np.percentile(vals, 2.5)) if vals else np.nan,
                "ci_high": float(np.percentile(vals, 97.5)) if vals else np.nan,
                "valid_bootstraps": int(len(vals)),
                "skipped_label_resamples": int(skipped_count),
            })
    return pd.DataFrame(rows)


def draw_bar(group_df, ci_df):
    import matplotlib.pyplot as plt
    methods = ["PubMedCLIP", "BioMedCLIP", "RAD-DINO", "CLIP", "CLIPCEIL", "MGLL", "GRADE-CXR_base", "GRADE-CXR_ref"]
    groups = ["Frequent", "Intermediate", "Rare"]
    colors = {
        "PubMedCLIP": "#C779C9",
        "BioMedCLIP": "#E2837E",
        "RAD-DINO": "#B07BD4",
        "CLIP": "#8BA7C9",
        "CLIPCEIL": "#7F9F59",
        "MGLL": "#E2B44E",
        "GRADE-CXR_base": "#5A8CC2",
        "GRADE-CXR_ref": "#C23B3B",
    }
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    x = np.arange(len(groups))
    width = 0.095
    offsets = (np.arange(len(methods)) - (len(methods)-1)/2) * width
    for k, method in enumerate(methods):
        vals, lo, hi = [], [], []
        for g in groups:
            row = group_df[(group_df.method == method) & (group_df.group == g)].iloc[0]
            vals.append(row["mAP"])
            ci = ci_df[(ci_df.method == method) & (ci_df.group == g) & (ci_df.metric == "mAP")].iloc[0]
            lo.append(max(0, row["mAP"] - ci["ci_low"]))
            hi.append(max(0, ci["ci_high"] - row["mAP"]))
        ax.bar(x + offsets[k], vals, width=width*0.92, color=colors[method], edgecolor="white", linewidth=0.3, label=method, zorder=3)
        ax.errorbar(x + offsets[k], vals, yerr=[lo, hi], fmt="none", ecolor="#333333", elinewidth=0.55, capsize=1.5, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["Frequent\n(>5%)", "Intermediate\n(1%-5%)", "Rare\n(<1%)"])
    ax.set_ylabel("Mean Average Precision")
    ax.set_title("Fine-grained PadChest finding transfer", pad=8)
    ax.grid(axis="y", color="#d8d8d8", linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(ncol=4, frameon=True, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "finegrained_31label_transfer_bar.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "finegrained_31label_transfer_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_meta = pd.read_csv(LABEL_SET)
    label_names = label_meta["label"].tolist()
    label_meta = label_meta.reset_index().rename(columns={"index": "label_index"})
    split_paths = build_matched_csvs(label_names)
    methods = ["PubMedCLIP", "BioMedCLIP", "RAD-DINO", "CLIP", "CLIPCEIL", "MGLL", "GRADE-CXR_base", "GRADE-CXR_ref"]
    per_all, group_all, overall_rows, ci_all = [], [], [], []
    log_lines = []
    log_lines.append("# Fine-grained PadChest Finding Transfer Log\n")
    log_lines.append("Dataset: PadChest physician-label193 matched subset, 31 fine-grained labels.\n")
    log_lines.append(f"Splits: " + ", ".join(f"{k}={len(pd.read_csv(v))}" for k, v in split_paths.items()) + "\n")
    log_lines.append("## Labels\n")
    for _, r in label_meta.iterrows():
        log_lines.append(f"- {r['label']}: positives={r['matched_positive_in_current_split']}, prevalence={r['prevalence_in_current_matched_split']:.4f}, group={r['finegrained_prevalence_group']}, type={r.get('selection_mapping_type','')}")
    log_lines.append("\n## Linear Classifier Hyperparameters\n- classifier: Linear(D,31)\n- loss: BCEWithLogitsLoss\n- optimizer: AdamW lr=1e-3 weight_decay=1e-4\n- epochs=50, early stopping patience=8 on validation mAP\n- batch_size=256\n")
    log_lines.append("\n## Groups\n")
    for g in ["Frequent", "Intermediate", "Rare"]:
        labs = label_meta[label_meta.finegrained_prevalence_group == g]["label"].tolist()
        log_lines.append(f"- {g} ({len(labs)}): {', '.join(labs)}")
    for method in methods:
        try:
            t0 = time.time()
            data, cache_used, source_desc = extract_features(method, split_paths, label_names, device=device, batch_size=64, workers=4)
            model, hist = train_classifier(data, device=device, seed=42)
            test_prob = predict(model, data["test"]["features"], device=device)
            test_y = data["test"]["labels"]
            per_df, skipped = per_label_metrics(test_y, test_prob, label_names)
            per_df["method"] = method
            gdf = group_metrics(per_df, label_meta)
            gdf["method"] = method
            ci = bootstrap_ci(test_y, test_prob, label_names, label_meta, method, n_boot=1000, seed=42)
            per_all.append(per_df)
            group_all.append(gdf)
            ci_all.append(ci)
            wide = {f"{r.group} AUC": r.AUC for _, r in gdf.iterrows()}
            wide.update({f"{r.group} mAP": r.mAP for _, r in gdf.iterrows()})
            overall_rows.append({"Method": method, **wide})
            log_lines.append(f"\n## {method}\n- source: {source_desc}\n- feature cache used: {cache_used}\n- feature_dim: {data['train']['features'].shape[1]}\n- train history epochs: {len(hist)}\n- skipped labels in point estimate: {skipped if skipped else 'None'}\n- elapsed_sec: {time.time()-t0:.1f}\n")
        except Exception as e:
            log_lines.append(f"\n## {method}\n- ERROR: {repr(e)}\n")
            print(f"[ERROR] {method}: {e}")
    per = pd.concat(per_all, ignore_index=True) if per_all else pd.DataFrame()
    grp = pd.concat(group_all, ignore_index=True) if group_all else pd.DataFrame()
    ci = pd.concat(ci_all, ignore_index=True) if ci_all else pd.DataFrame()
    overall = pd.DataFrame(overall_rows)
    # Requested table order.
    cols = ["Method", "Frequent AUC", "Frequent mAP", "Intermediate AUC", "Intermediate mAP", "Rare AUC", "Rare mAP", "Overall AUC", "Overall mAP"]
    overall = overall.reindex(columns=cols)
    overall.to_csv(OUT_DIR / "finegrained_31label_transfer_metrics.csv", index=False)
    per.merge(label_meta[["label", "finegrained_prevalence_group", "selection_mapping_type", "chexpert14_mapping"]], on="label", how="left").to_csv(OUT_DIR / "finegrained_31label_per_label_metrics.csv", index=False)
    grp.to_csv(OUT_DIR / "finegrained_31label_group_metrics.csv", index=False)
    ci.to_csv(OUT_DIR / "finegrained_31label_bootstrap_ci.csv", index=False)
    if not grp.empty and not ci.empty:
        draw_bar(grp, ci)
    log_lines.append("\n## Bootstrap\n- n_bootstrap=1000\n- resamples with a single class for a label are skipped for that label; counts are in finegrained_31label_bootstrap_ci.csv.\n")
    (OUT_DIR / "finegrained_31label_transfer_log.md").write_text("\n".join(log_lines))
    print(f"[DONE] {OUT_DIR}")


if __name__ == "__main__":
    main()
