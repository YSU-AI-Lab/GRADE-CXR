#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    semantic_ref_loss,
    stage2_local_losses,
)
from train_stage2_v2 import DEFAULT_MODEL, DEFAULT_PRETRAINED, read_checkpoint_state_dict, save_checkpoint, set_seed  # noqa: E402


class CachedStage2Dataset(Dataset):
    def __init__(self, cache_dir):
        cache = Path(cache_dir)
        self.z = np.load(cache / "z_base.float16.npy", mmap_mode="r")
        self.tokens = np.load(cache / "multi_tokens.float16.npy", mmap_mode="r")
        self.labels = np.load(cache / "labels14.float32.npy", mmap_mode="r")
        self.reports = json.loads((cache / "report_texts.json").read_text(encoding="utf-8"))
        self.meta = json.loads((cache / "cache_meta.json").read_text(encoding="utf-8"))
        if not (len(self.z) == len(self.tokens) == len(self.labels) == len(self.reports)):
            raise RuntimeError("Cached arrays have inconsistent lengths.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "z_base": torch.from_numpy(np.asarray(self.z[idx], dtype=np.float32)),
            "multi_tokens": torch.from_numpy(np.asarray(self.tokens[idx], dtype=np.float32)),
            "labels14": torch.from_numpy(np.asarray(self.labels[idx], dtype=np.float32)),
            "report_text": self.reports[idx],
        }


def collate(batch):
    return {
        "z_base": torch.stack([x["z_base"] for x in batch]),
        "multi_tokens": torch.stack([x["multi_tokens"] for x in batch]),
        "labels14": torch.stack([x["labels14"] for x in batch]),
        "report_text": [x["report_text"] for x in batch],
    }


@torch.no_grad()
def load_frozen_text(args, device):
    model, _, _ = create_model_and_transforms(args.model, pretrained=args.pretrained, device=device, output_dict=True)
    msg = model.load_state_dict(read_checkpoint_state_dict(args.stage1_checkpoint), strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = get_tokenizer(args.model)
    anchors = F.normalize(
        model.encode_text(tokenizer(["This is a chest X-ray image of " + x.lower() + "." for x in CHEXPERT14]).to(device), normalize=True).float(),
        dim=-1,
    ).detach()
    return model, tokenizer, anchors, {"missing_keys": list(msg.missing_keys), "unexpected_keys": list(msg.unexpected_keys)}


@torch.no_grad()
def encode_reports(model, tokenizer, texts, device):
    mask = torch.tensor([bool(str(t).strip()) for t in texts], device=device, dtype=torch.bool)
    out = torch.zeros((len(texts), getattr(model, "embed_dim", 512)), device=device)
    if mask.any():
        idx = torch.where(mask)[0].detach().cpu().tolist()
        out[mask] = model.encode_text(tokenizer([texts[i] for i in idx]).to(device), normalize=True).float()
    return out, mask


def main():
    p = argparse.ArgumentParser(description="Train Stage-2 v2 from cached z_base and multi-level patch tokens.")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--stage1-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--num-queries", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--sparse-topk", type=int, default=4)
    p.add_argument("--teacher-temperature", type=float, default=0.07)
    p.add_argument("--residual-scale", type=float, default=0.2)
    p.add_argument("--lambda-pres", type=float, default=0.5)
    p.add_argument("--lambda-local", type=float, default=1.0)
    p.add_argument("--lambda-attn", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda:0" if args.device == "cuda" and torch.cuda.is_available() else args.device
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    text_model, tokenizer, anchors, load_info = load_frozen_text(args, device)
    ds = CachedStage2Dataset(args.cache_dir)
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=gen, num_workers=args.workers, collate_fn=collate, drop_last=True)
    num_layers = int(ds.tokens.shape[1])
    refiner = Stage2Refiner(args.embed_dim, num_layers, args.num_queries, args.num_heads, args.residual_scale).to(device)
    opt = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rows = []
    resolved_args = vars(args).copy()
    resolved_args["cache_meta"] = ds.meta
    (out / "config_resolved.json").write_text(json.dumps(resolved_args, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        sums = []
        refiner.train()
        for batch in tqdm(loader, desc=f"cached-stage2-v2 epoch {epoch}", ncols=100):
            z_base = F.normalize(batch["z_base"].to(device), dim=-1)
            multi_tokens = batch["multi_tokens"].to(device)
            labels = batch["labels14"].to(device)
            with torch.no_grad():
                report_features, report_mask = encode_reports(text_model, tokenizer, batch["report_text"], device)
            outputs = refiner(z_base, multi_tokens)
            teacher = build_sparse_teacher(outputs["projected_tokens"], labels, anchors, args.sparse_topk, args.teacher_temperature)
            sem = semantic_ref_loss(outputs["z_ref"], labels, anchors, report_features, report_mask)
            loc = stage2_local_losses(z_base, outputs, teacher, args.lambda_pres, args.lambda_local, args.lambda_attn)
            loss = sem["loss_ref_sem"] + loc["loss_local_total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sums.append({
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
            })
        row = {"epoch": epoch}
        for key in sums[0]:
            row[key] = float(np.mean([x[key] for x in sums]))
        rows.append(row)
        save_checkpoint(out / "checkpoints" / f"epoch_{epoch}.pt", refiner, args, epoch, row, load_info)
        save_checkpoint(out / "checkpoints" / "latest.pt", refiner, args, epoch, row, load_info)
        with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(row, flush=True)
    print(f"[DONE] {out}")


if __name__ == "__main__":
    main()
