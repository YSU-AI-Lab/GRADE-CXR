#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path("/home/hyr/clip/biomedclip_encoder_generalization")
SRC = ROOT / "open_clip/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open_clip import create_model_and_transforms, get_tokenizer  # noqa: E402
from open_clip.grade_stage2_v2 import (  # noqa: E402
    CHEXPERT14,
    Stage2Refiner,
    build_sparse_teacher,
    forward_visual_multilevel,
    parse_selected_layers,
    semantic_ref_loss,
    stage2_local_losses,
    visual_num_layers,
)

DEFAULT_MODEL = "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DEFAULT_PRETRAINED = str(ROOT / "BiomedCLIP-PubMedBERT/open_clip_pytorch_model.bin")
DEFAULT_OUT = str(ROOT / "open_clip/src/logs/h0_stage2_v2_anchor_sparse_loso")

LOSO_JSONL = {
    "mimic": str(ROOT / "dataset/heldout_mimic/train_chexpert_padchest_heldout_mimic.jsonl"),
    "chexpert": str(ROOT / "dataset/heldout_chexpert/train_mimic_padchest_heldout_chexpert.jsonl"),
    "padchest": str(ROOT / "dataset/heldout_padchest/train_mimic_chexpert_heldout_padchest.jsonl"),
}
LOSO_STAGE1 = {
    "mimic": str(ROOT / "open_clip/src/logs/h0_stage1_hcpa_p_clip_vitb16_heldout_mimic_seed42/checkpoints/epoch_10.pt"),
    "chexpert": str(ROOT / "open_clip/src/logs/h0_stage1_hcpa_p_clip_vitb16_heldout_chexpert_seed42/checkpoints/epoch_9.pt"),
    "padchest": str(ROOT / "open_clip/src/logs/h0_stage1_hcpa_p_heldout_padchest_nooldcpa/checkpoints/epoch_20.pt"),
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}


def read_checkpoint_state_dict(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return strip_state_dict(ckpt["state_dict"])
    if isinstance(ckpt, dict) and "model" in ckpt:
        return strip_state_dict(ckpt["model"])
    return strip_state_dict(ckpt)


class JsonlCXRDataset(Dataset):
    def __init__(self, path: str, limit: int = 0):
        self.rows = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                img = row.get("ImageID") or row.get("image_path") or row.get("Image Index") or ""
                if not img or not Path(img).exists():
                    skipped += 1
                    continue
                self.rows.append(row)
                if limit and len(self.rows) >= limit:
                    break
        if not self.rows:
            raise RuntimeError(f"No usable images found in {path}")
        print(f"[DATA] {path}: loaded={len(self.rows)}, skipped_missing={skipped}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def row_image(row: dict) -> str:
    return row.get("ImageID") or row.get("image_path") or row.get("Image Index") or ""


def collate_rows(rows: List[dict]) -> Dict:
    labels = []
    for row in rows:
        labels.append([float(v) for v in row.get("labels14", row.get("labels", [0] * 14))])
    return {
        "image_path": [row_image(r) for r in rows],
        "labels14": torch.tensor(labels, dtype=torch.float32),
        "disease_text": [str(r.get("Disease") or "chest x-ray finding.") for r in rows],
        "report_text": [str(r.get("Description") or r.get("report") or r.get("Disease") or "").strip() for r in rows],
    }


def load_stage1(args, device):
    model, _, preprocess = create_model_and_transforms(args.model, pretrained=args.pretrained, device=device, output_dict=True)
    msg = model.load_state_dict(read_checkpoint_state_dict(args.stage1_checkpoint), strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[LOAD] Stage-1: {args.stage1_checkpoint}")
    print(f"[LOAD] missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    return model, preprocess, {"missing_keys": list(msg.missing_keys), "unexpected_keys": list(msg.unexpected_keys)}


@torch.no_grad()
def encode_disease_anchors(model, tokenizer, device):
    texts = ["This is a chest X-ray image of " + name.lower() + "." for name in CHEXPERT14]
    return F.normalize(model.encode_text(tokenizer(texts).to(device), normalize=True).float(), dim=-1).detach()


def encode_reports(model, tokenizer, report_texts: List[str], device) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.tensor([bool(t.strip()) for t in report_texts], device=device, dtype=torch.bool)
    out = torch.zeros((len(report_texts), getattr(model, "embed_dim", 512)), device=device)
    if mask.any():
        texts = [report_texts[i] for i in torch.where(mask)[0].detach().cpu().tolist()]
        out[mask] = model.encode_text(tokenizer(texts).to(device), normalize=True).float()
    return out, mask


def batch_images(paths: List[str], preprocess, device):
    return torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device, non_blocking=True)


def save_checkpoint(path: Path, refiner: Stage2Refiner, args, epoch: int, stats: Dict, load_info: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage2_refiner_v2": refiner.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "stats": stats,
            "stage1_checkpoint": args.stage1_checkpoint,
            "load_info": load_info,
        },
        path,
    )


def smoke_random(args, device) -> Dict:
    set_seed(args.seed)
    refiner = Stage2Refiner(
        embed_dim=args.embed_dim,
        num_layers=len(args.selected_layers_resolved),
        num_queries=args.num_queries,
        num_heads=args.num_heads,
        residual_scale=args.residual_scale,
    ).to(device)
    z = F.normalize(torch.randn(3, args.embed_dim, device=device), dim=-1)
    tokens = torch.randn(3, len(args.selected_layers_resolved), 196, args.embed_dim, device=device)
    labels = torch.zeros(3, 14, device=device)
    labels[0, 1] = 1
    labels[1, [4, 9]] = 1
    labels[2, 13] = 1
    anchors = F.normalize(torch.randn(14, args.embed_dim, device=device), dim=-1)
    out = refiner(z, tokens)
    teacher = build_sparse_teacher(out["projected_tokens"], labels, anchors, args.sparse_topk, args.teacher_temperature)
    losses = stage2_local_losses(z, out, teacher, args.lambda_pres, args.lambda_local, args.lambda_attn)
    total = losses["loss_local_total"] + semantic_ref_loss(out["z_ref"], labels, anchors)["loss_ref_sem"]
    total.backward()
    pi_sum = out["pi_pred"].sum(dim=-1)
    tgt_sum = teacher["pi_tgt"][teacher["valid_teacher"]].sum(dim=-1)
    outside_nonzero = []
    for b in torch.where(teacher["valid_teacher"])[0]:
        outside_nonzero.append(int((teacher["pi_tgt"][b] > 0).sum().item()) <= int(args.sparse_topk) * int((labels[b, :13] > 0.5).sum().item()))
    return {
        "random_forward_backward": True,
        "random_pi_pred_sum_min": float(pi_sum.min().detach().cpu()),
        "random_pi_pred_sum_max": float(pi_sum.max().detach().cpu()),
        "random_pi_tgt_sum_min": float(tgt_sum.min().detach().cpu()) if tgt_sum.numel() else None,
        "random_pi_tgt_sum_max": float(tgt_sum.max().detach().cpu()) if tgt_sum.numel() else None,
        "random_topk_support_ok": bool(all(outside_nonzero)),
        "random_z_ref_norm_mean": float(out["z_ref"].norm(dim=-1).mean().detach().cpu()),
    }


def run_smoke(args, device, outdir: Path):
    model, preprocess, load_info = load_stage1(args, device)
    tokenizer = get_tokenizer(args.model)
    anchors = encode_disease_anchors(model, tokenizer, device)
    ds = JsonlCXRDataset(args.train_jsonl, limit=max(args.batch_size * 12, 32))
    loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=False, num_workers=0, collate_fn=collate_rows)
    refiner = Stage2Refiner(
        embed_dim=args.embed_dim,
        num_layers=len(args.selected_layers_resolved),
        num_queries=args.num_queries,
        num_heads=args.num_heads,
        residual_scale=args.residual_scale,
    ).to(device)
    opt = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    report = smoke_random(args, device)
    batch = None
    for candidate in loader:
        labels = candidate["labels14"]
        positive = torch.isfinite(labels) & (labels > 0.5)
        positive[:, 13] = False
        if positive.any():
            batch = candidate
            break
    if batch is None:
        batch = next(iter(loader))
    images = batch_images(batch["image_path"], preprocess, device)
    with torch.no_grad():
        z_base, multi_tokens = forward_visual_multilevel(model.visual, images, args.selected_layers_resolved)
        report_features, report_mask = encode_reports(model, tokenizer, batch["report_text"], device)
    outputs = refiner(z_base, multi_tokens)
    teacher = build_sparse_teacher(outputs["projected_tokens"], batch["labels14"].to(device), anchors, args.sparse_topk, args.teacher_temperature)
    sem = semantic_ref_loss(outputs["z_ref"], batch["labels14"].to(device), anchors, report_features, report_mask)
    loc = stage2_local_losses(z_base, outputs, teacher, args.lambda_pres, args.lambda_local, args.lambda_attn)
    loss = sem["loss_ref_sem"] + loc["loss_local_total"]
    opt.zero_grad()
    loss.backward()
    opt.step()
    backbone_has_grad = any(p.grad is not None and p.grad.detach().abs().sum().item() > 0 for p in model.parameters())
    optimizer_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    overlap_backbone = sum(1 for p in model.parameters() if id(p) in optimizer_param_ids)
    ckpt = outdir / "smoke_stage2_v2.pt"
    save_checkpoint(ckpt, refiner, args, 0, {"smoke_loss": float(loss.detach().cpu())}, load_info)
    reloaded = Stage2Refiner(
        embed_dim=args.embed_dim,
        num_layers=len(args.selected_layers_resolved),
        num_queries=args.num_queries,
        num_heads=args.num_heads,
        residual_scale=args.residual_scale,
    ).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)["stage2_refiner_v2"]
    reloaded.load_state_dict(state)
    report.update(
        {
            "real_forward_backward": True,
            "backbone_has_grad": bool(backbone_has_grad),
            "optimizer_backbone_param_count": int(overlap_backbone),
            "pi_pred_sum_min": float(outputs["pi_pred"].sum(dim=-1).min().detach().cpu()),
            "pi_pred_sum_max": float(outputs["pi_pred"].sum(dim=-1).max().detach().cpu()),
            "valid_teacher_count": int(teacher["valid_teacher"].sum().detach().cpu()),
            "pi_tgt_sum_min": float(teacher["pi_tgt"][teacher["valid_teacher"]].sum(dim=-1).min().detach().cpu()) if teacher["valid_teacher"].any() else None,
            "pi_tgt_sum_max": float(teacher["pi_tgt"][teacher["valid_teacher"]].sum(dim=-1).max().detach().cpu()) if teacher["valid_teacher"].any() else None,
            "z_ref_norm_mean": float(outputs["z_ref"].norm(dim=-1).mean().detach().cpu()),
            "checkpoint_reload_ok": True,
            "loss": float(loss.detach().cpu()),
            "loss_ref_sem": float(sem["loss_ref_sem"].detach().cpu()),
            "loss_pres": float(loc["loss_pres"].detach().cpu()),
            "loss_local": float(loc["loss_local"].detach().cpu()),
            "loss_attn": float(loc["loss_attn"].detach().cpu()),
            "selected_layers_zero_based": args.selected_layers_resolved,
            "selected_layers_public": args.selected_layers,
        }
    )
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def train(args, device, outdir: Path):
    model, preprocess, load_info = load_stage1(args, device)
    tokenizer = get_tokenizer(args.model)
    anchors = encode_disease_anchors(model, tokenizer, device)
    ds = JsonlCXRDataset(args.train_jsonl, limit=args.limit_samples)
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=gen,
        num_workers=args.workers,
        pin_memory=str(device).startswith("cuda"),
        collate_fn=collate_rows,
        drop_last=True,
    )
    refiner = Stage2Refiner(
        embed_dim=args.embed_dim,
        num_layers=len(args.selected_layers_resolved),
        num_queries=args.num_queries,
        num_heads=args.num_heads,
        residual_scale=args.residual_scale,
    ).to(device)
    opt = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rows = []
    for epoch in range(1, args.epochs + 1):
        refiner.train()
        sums = []
        for batch in tqdm(loader, desc=f"stage2-v2 epoch {epoch}", ncols=100):
            images = batch_images(batch["image_path"], preprocess, device)
            labels = batch["labels14"].to(device)
            with torch.no_grad():
                z_base, multi_tokens = forward_visual_multilevel(model.visual, images, args.selected_layers_resolved)
                report_features, report_mask = encode_reports(model, tokenizer, batch["report_text"], device)
            outputs = refiner(z_base, multi_tokens)
            teacher = build_sparse_teacher(outputs["projected_tokens"], labels, anchors, args.sparse_topk, args.teacher_temperature)
            sem = semantic_ref_loss(outputs["z_ref"], labels, anchors, report_features, report_mask)
            loc = stage2_local_losses(z_base, outputs, teacher, args.lambda_pres, args.lambda_local, args.lambda_attn)
            loss = sem["loss_ref_sem"] + loc["loss_local_total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sums.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "loss_ref_sem": float(sem["loss_ref_sem"].detach().cpu()),
                    "loss_soft_disease": float(sem["loss_soft_disease"].detach().cpu()),
                    "loss_masked_bce": float(sem["loss_masked_bce"].detach().cpu()),
                    "loss_report_sem": float(sem["loss_report_sem"].detach().cpu()),
                    "loss_pres": float(loc["loss_pres"].detach().cpu()),
                    "loss_local": float(loc["loss_local"].detach().cpu()),
                    "loss_attn": float(loc["loss_attn"].detach().cpu()),
                    "valid_teacher_ratio": float(teacher["valid_teacher"].float().mean().detach().cpu()),
                    "gate_mean": float(outputs["gate"].mean().detach().cpu()),
                    "z_ref_z_base_cos": float((outputs["z_ref"] * z_base).sum(dim=-1).mean().detach().cpu()),
                }
            )
        row = {"epoch": epoch}
        for key in sums[0]:
            row[key] = float(np.mean([x[key] for x in sums]))
        rows.append(row)
        save_checkpoint(outdir / "checkpoints" / f"epoch_{epoch}.pt", refiner, args, epoch, row, load_info)
        save_checkpoint(outdir / "checkpoints" / "latest.pt", refiner, args, epoch, row, load_info)
        with open(outdir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(row, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Train/smoke Stage-2 v2 anchor-guided sparse patch-to-global refiner.")
    p.add_argument("--heldout", choices=["mimic", "chexpert", "padchest"], default="padchest")
    p.add_argument("--train-jsonl", default="")
    p.add_argument("--stage1-checkpoint", default="")
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    p.add_argument("--selected-layers", default="3,6,9,12")
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--num-queries", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--sparse-topk", type=int, default=4)
    p.add_argument("--teacher-temperature", type=float, default=0.07)
    p.add_argument("--residual-scale", type=float, default=0.2)
    p.add_argument("--lambda-pres", type=float, default=0.5)
    p.add_argument("--lambda-local", type=float, default=1.0)
    p.add_argument("--lambda-attn", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--limit-samples", type=int, default=0)
    args = p.parse_args()
    if not args.train_jsonl:
        args.train_jsonl = LOSO_JSONL[args.heldout]
    if not args.stage1_checkpoint:
        args.stage1_checkpoint = LOSO_STAGE1[args.heldout]
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    if device == "cuda":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tmp_model, _, _ = create_model_and_transforms(args.model, pretrained=args.pretrained, device="cpu", output_dict=True)
    n_layers = visual_num_layers(tmp_model)
    args.selected_layers_resolved = parse_selected_layers(args.selected_layers, n_layers, one_based=True)
    del tmp_model
    outdir = Path(args.output_dir) / f"heldout_{args.heldout}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config_resolved.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    if args.smoke_test:
        run_smoke(args, device, outdir)
    else:
        train(args, device, outdir)


if __name__ == "__main__":
    main()
