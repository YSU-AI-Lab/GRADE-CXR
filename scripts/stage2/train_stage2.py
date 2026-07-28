#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
from open_clip.grade_stage2_simple import (  # noqa: E402
    CHEXPERT14,
    FixedAnchorTeacher,
    Stage2Refiner,
    forward_stage1_tokens,
    parse_selected_layers,
    semantic_ref_loss,
    stage2_losses,
    visual_num_layers,
)

DEFAULT_MODEL = "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DEFAULT_PRETRAINED = str(ROOT / "BiomedCLIP-PubMedBERT/open_clip_pytorch_model.bin")
DEFAULT_OUTPUT = str(ROOT / "open_clip/src/logs/h0_stage2_simple")
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


def strip_state_dict(sd):
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}


def read_state_dict(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return strip_state_dict(ckpt["state_dict"])
    if isinstance(ckpt, dict) and "model" in ckpt:
        return strip_state_dict(ckpt["model"])
    return strip_state_dict(ckpt)


class JsonlDataset(Dataset):
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
            raise RuntimeError(f"No usable image rows in {path}")
        print(f"[DATA] loaded={len(self.rows)} skipped_missing={skipped} path={path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def get_image_path(row):
    return row.get("ImageID") or row.get("image_path") or row.get("Image Index") or ""


def collate(rows: List[dict]):
    labels = [[float(v) for v in r.get("labels14", r.get("labels", [0] * 14))] for r in rows]
    return {
        "image_path": [get_image_path(r) for r in rows],
        "labels14": torch.tensor(labels, dtype=torch.float32),
        "report_text": [str(r.get("Description") or r.get("report") or r.get("Disease") or "").strip() for r in rows],
    }


def load_stage1(args, device):
    model, _, preprocess = create_model_and_transforms(args.model, pretrained=args.pretrained, device=device, output_dict=True)
    msg = model.load_state_dict(read_state_dict(args.stage1_checkpoint), strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[LOAD] stage1={args.stage1_checkpoint}")
    print(f"[LOAD] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    return model, preprocess, {"missing_keys": list(msg.missing_keys), "unexpected_keys": list(msg.unexpected_keys)}


@torch.no_grad()
def disease_anchors(model, tokenizer, device):
    texts = ["This is a chest X-ray image of " + x.lower() + "." for x in CHEXPERT14]
    return F.normalize(model.encode_text(tokenizer(texts).to(device), normalize=True).float(), dim=-1).detach()


@torch.no_grad()
def report_features(model, tokenizer, texts: List[str], device):
    mask = torch.tensor([bool(t.strip()) for t in texts], device=device, dtype=torch.bool)
    feat = torch.zeros((len(texts), getattr(model, "embed_dim", 512)), device=device)
    if mask.any():
        idx = torch.where(mask)[0].detach().cpu().tolist()
        feat[mask] = model.encode_text(tokenizer([texts[i] for i in idx]).to(device), normalize=True).float()
    return feat, mask


def load_images(paths: List[str], preprocess, device):
    return torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device, non_blocking=True)


@torch.no_grad()
def encode_refined_image(model, refiner: Stage2Refiner, image: torch.Tensor, selected_layers: List[int]) -> torch.Tensor:
    z_base, selected_tokens, _final_patch = forward_stage1_tokens(model.visual, image, selected_layers)
    return refiner(z_base, selected_tokens)["z_ref"]


def save_ckpt(path: Path, refiner: Stage2Refiner, args, epoch: int, row: Dict, load_info: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage2_refiner": refiner.state_dict(),
            "stage2_variant": "simple_single_query_fixed_teacher",
            "stage1_checkpoint": args.stage1_checkpoint,
            "args": vars(args),
            "epoch": epoch,
            "metrics": row,
            "load_info": load_info,
        },
        path,
    )


def run_one_batch(model, preprocess, tokenizer, refiner, teacher, anchors, batch, args, device):
    images = load_images(batch["image_path"], preprocess, device)
    labels = batch["labels14"].to(device)
    with torch.no_grad():
        z_base, selected_tokens, final_patch = forward_stage1_tokens(model.visual, images, args.selected_layers_resolved)
        rep_feat, rep_mask = report_features(model, tokenizer, batch["report_text"], device)
    out = refiner(z_base, selected_tokens)
    tgt = teacher(final_patch, labels, anchors)
    sem = semantic_ref_loss(out["z_ref"], labels, anchors, rep_feat, rep_mask)
    aux = stage2_losses(z_base, out, tgt, args.lambda_pres, args.lambda_local, args.lambda_attn)
    loss = sem["loss_ref_sem"] + aux["loss_stage2_aux"]
    return loss, out, tgt, sem, aux, z_base


def smoke_test(args, device, outdir: Path):
    model, preprocess, load_info = load_stage1(args, device)
    tokenizer = get_tokenizer(args.model)
    anchors = disease_anchors(model, tokenizer, device)
    ds = JsonlDataset(args.train_jsonl, limit=max(args.batch_size * 16, 64))
    loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=False, num_workers=0, collate_fn=collate)
    refiner = Stage2Refiner(args.embed_dim, len(args.selected_layers_resolved), args.num_heads).to(device)
    teacher = FixedAnchorTeacher(args.topk_per_pathology, args.teacher_temperature)
    opt = torch.optim.AdamW(refiner.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    positive_batch = None
    no_positive_batch = None
    for batch in loader:
        pos = torch.isfinite(batch["labels14"]) & (batch["labels14"] > 0.5)
        pos[:, 13] = False
        if pos.any() and positive_batch is None:
            positive_batch = batch
        if (not pos.any()) and no_positive_batch is None:
            no_positive_batch = batch
        if positive_batch is not None and no_positive_batch is not None:
            break
    batch = positive_batch or next(iter(loader))
    loss, out, tgt, sem, aux, z_base = run_one_batch(model, preprocess, tokenizer, refiner, teacher, anchors, batch, args, device)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    backbone_grad = any(p.grad is not None and p.grad.detach().abs().sum().item() > 0 for p in model.parameters())
    opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    opt_frozen_count = sum(1 for p in model.parameters() if id(p) in opt_param_ids)
    pos_support_ok = True
    if tgt["valid_teacher"].any():
        max_allowed = args.topk_per_pathology * torch.clamp((batch["labels14"][:, :13] > 0.5).sum(dim=1), min=1)
        nnz = (tgt["pi_tgt"] > 0).sum(dim=1).cpu()
        pos_support_ok = bool((nnz[tgt["valid_teacher"].cpu()] <= max_allowed[tgt["valid_teacher"].cpu()]).all())

    no_pos_nan_ok = True
    if no_positive_batch is not None:
        loss2, out2, tgt2, sem2, aux2, _ = run_one_batch(model, preprocess, tokenizer, refiner, teacher, anchors, no_positive_batch, args, device)
        no_pos_nan_ok = bool(torch.isfinite(loss2) and torch.isfinite(aux2["loss_local"]) and torch.isfinite(aux2["loss_attn"]))

    with torch.no_grad():
        image_only = load_images(batch["image_path"][:1], preprocess, device)
        z_img_only = encode_refined_image(model, refiner, image_only, args.selected_layers_resolved)

    ckpt = outdir / "smoke_stage2_simple.pt"
    save_ckpt(ckpt, refiner, args, 0, {"smoke_loss": float(loss.detach().cpu())}, load_info)
    reload_refiner = Stage2Refiner(args.embed_dim, len(args.selected_layers_resolved), args.num_heads).to(device)
    reload_refiner.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["stage2_refiner"])
    report = {
        "z_base_shape": list(z_base.shape),
        "u_shape": list(out["u"].shape),
        "u_tgt_shape": list(tgt["u_tgt"].shape),
        "z_ref_shape": list(out["z_ref"].shape),
        "backbone_and_text_have_grad": bool(backbone_grad),
        "optimizer_frozen_param_count": int(opt_frozen_count),
        "pi_tgt_sum_min": float(tgt["pi_tgt"][tgt["valid_teacher"]].sum(dim=-1).min().detach().cpu()) if tgt["valid_teacher"].any() else None,
        "pi_tgt_sum_max": float(tgt["pi_tgt"][tgt["valid_teacher"]].sum(dim=-1).max().detach().cpu()) if tgt["valid_teacher"].any() else None,
        "pi_pred_sum_min": float(out["pi_pred"].sum(dim=-1).min().detach().cpu()),
        "pi_pred_sum_max": float(out["pi_pred"].sum(dim=-1).max().detach().cpu()),
        "top4_outside_teacher_prob_zero": pos_support_ok,
        "no_positive_batch_nan_ok": no_pos_nan_ok,
        "z_ref_norm_mean": float(out["z_ref"].norm(dim=-1).mean().detach().cpu()),
        "forward_backward_step_ok": True,
        "checkpoint_reload_ok": True,
        "image_only_inference_shape": list(z_img_only.shape),
        "image_only_inference_norm": float(z_img_only.norm(dim=-1).mean().detach().cpu()),
        "loss": float(loss.detach().cpu()),
        "loss_ref_sem": float(sem["loss_ref_sem"].detach().cpu()),
        "loss_pres": float(aux["loss_pres"].detach().cpu()),
        "loss_local": float(aux["loss_local"].detach().cpu()),
        "loss_attn": float(aux["loss_attn"].detach().cpu()),
        "selected_layers_public": args.selected_layers,
        "selected_layers_zero_based": args.selected_layers_resolved,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def train(args, device, outdir: Path):
    model, preprocess, load_info = load_stage1(args, device)
    tokenizer = get_tokenizer(args.model)
    anchors = disease_anchors(model, tokenizer, device)
    ds = JsonlDataset(args.train_jsonl, limit=args.limit_samples)
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=gen, num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate)
    refiner = Stage2Refiner(args.embed_dim, len(args.selected_layers_resolved), args.num_heads).to(device)
    teacher = FixedAnchorTeacher(args.topk_per_pathology, args.teacher_temperature)
    opt = torch.optim.AdamW(refiner.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rows = []
    for epoch in range(1, args.epochs + 1):
        refiner.train()
        sums = []
        for batch in tqdm(loader, desc=f"stage2-simple epoch {epoch}", ncols=100):
            loss, out, tgt, sem, aux, z_base = run_one_batch(model, preprocess, tokenizer, refiner, teacher, anchors, batch, args, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sums.append({
                "loss": float(loss.detach().cpu()),
                "loss_ref_sem": float(sem["loss_ref_sem"].detach().cpu()),
                "loss_soft_disease": float(sem["loss_soft_disease"].detach().cpu()),
                "loss_masked_bce": float(sem["loss_masked_bce"].detach().cpu()),
                "loss_report_sem": float(sem["loss_report_sem"].detach().cpu()),
                "loss_pres": float(aux["loss_pres"].detach().cpu()),
                "loss_local": float(aux["loss_local"].detach().cpu()),
                "loss_attn": float(aux["loss_attn"].detach().cpu()),
                "valid_teacher_ratio": float(tgt["valid_teacher"].float().mean().detach().cpu()),
                "gate_mean": float(out["gate"].mean().detach().cpu()),
                "z_ref_z_base_cos": float((out["z_ref"] * z_base).sum(dim=-1).mean().detach().cpu()),
            })
        row = {"epoch": epoch}
        for key in sums[0]:
            row[key] = float(np.mean([x[key] for x in sums]))
        rows.append(row)
        save_ckpt(outdir / "checkpoints" / f"epoch_{epoch}.pt", refiner, args, epoch, row, load_info)
        save_ckpt(outdir / "checkpoints" / "latest.pt", refiner, args, epoch, row, load_info)
        with open(outdir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(row, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Simple single-query Stage-2 refiner.")
    p.add_argument("--heldout", choices=["mimic", "chexpert", "padchest"], default="padchest")
    p.add_argument("--train-jsonl", default="")
    p.add_argument("--stage1-checkpoint", default="")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    p.add_argument("--selected-layers", default="4,8,12")
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--topk-per-pathology", type=int, default=4)
    p.add_argument("--teacher-temperature", type=float, default=0.07)
    p.add_argument("--lambda-pres", type=float, default=0.1)
    p.add_argument("--lambda-local", type=float, default=1.0)
    p.add_argument("--lambda-attn", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
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
    device = "cuda:0" if args.device == "cuda" and torch.cuda.is_available() else args.device
    tmp, _, _ = create_model_and_transforms(args.model, pretrained=args.pretrained, device="cpu", output_dict=True)
    args.selected_layers_resolved = parse_selected_layers(args.selected_layers, visual_num_layers(tmp))
    del tmp
    outdir = Path(args.output_dir) / f"heldout_{args.heldout}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config_resolved.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    if args.smoke_test:
        smoke_test(args, device, outdir)
    else:
        train(args, device, outdir)


if __name__ == "__main__":
    main()
