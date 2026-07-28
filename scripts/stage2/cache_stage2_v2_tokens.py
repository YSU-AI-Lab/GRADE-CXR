#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

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

from open_clip import create_model_and_transforms  # noqa: E402
from open_clip.grade_stage2_v2 import forward_visual_multilevel, parse_selected_layers, visual_num_layers  # noqa: E402
from train_stage2_v2 import DEFAULT_MODEL, DEFAULT_PRETRAINED, read_checkpoint_state_dict  # noqa: E402


def image_path(row):
    return row.get("ImageID") or row.get("image_path") or row.get("Image Index") or ""


def labels14(row):
    return [float(v) for v in row.get("labels14", row.get("labels", [0] * 14))]


def report_text(row):
    return str(row.get("Description") or row.get("report") or row.get("Disease") or "").strip()


class JsonlRows(Dataset):
    def __init__(self, path, limit_samples=0, seed=42, sample_random=True):
        rows = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                img = image_path(row)
                if not img or not Path(img).exists():
                    skipped += 1
                    continue
                rows.append(row)
        if limit_samples and len(rows) > limit_samples:
            if sample_random:
                rng = random.Random(seed)
                rows = rng.sample(rows, limit_samples)
                rows = sorted(rows, key=lambda r: image_path(r))
            else:
                rows = rows[:limit_samples]
        self.rows = rows
        self.skipped = skipped
        if not rows:
            raise RuntimeError(f"No usable images found in {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate(batch):
    return batch


def main():
    p = argparse.ArgumentParser(description="Cache frozen Stage-1 z_base and multi-level patch tokens for Stage-2 v2.")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--stage1-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    p.add_argument("--selected-layers", default="3,6,9,12")
    p.add_argument("--limit-samples", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-random-sample", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda:0" if args.device == "cuda" and torch.cuda.is_available() else args.device
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, _, preprocess = create_model_and_transforms(args.model, pretrained=args.pretrained, device=device, output_dict=True)
    msg = model.load_state_dict(read_checkpoint_state_dict(args.stage1_checkpoint), strict=False)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    selected = parse_selected_layers(args.selected_layers, visual_num_layers(model), one_based=True)

    ds = JsonlRows(args.train_jsonl, args.limit_samples, args.seed, sample_random=not args.no_random_sample)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)

    n = len(ds)
    z_mm = np.lib.format.open_memmap(out / "z_base.float16.npy", mode="w+", dtype=np.float16, shape=(n, 512))
    tok_mm = None
    labels = np.zeros((n, 14), dtype=np.float32)
    paths, reports = [], []
    cursor = 0
    with torch.no_grad():
        for rows in tqdm(loader, desc="cache-stage2-v2", ncols=100):
            images = torch.stack([preprocess(Image.open(image_path(r)).convert("RGB")) for r in rows]).to(device)
            z, tokens = forward_visual_multilevel(model.visual, images, selected)
            b, layers, patches, dim = tokens.shape
            if tok_mm is None:
                tok_mm = np.lib.format.open_memmap(
                    out / "multi_tokens.float16.npy",
                    mode="w+",
                    dtype=np.float16,
                    shape=(n, layers, patches, dim),
                )
            end = cursor + b
            z_mm[cursor:end] = F.normalize(z.float(), dim=-1).cpu().numpy().astype(np.float16)
            tok_mm[cursor:end] = tokens.cpu().numpy().astype(np.float16)
            labels[cursor:end] = np.asarray([labels14(r) for r in rows], dtype=np.float32)
            paths.extend([image_path(r) for r in rows])
            reports.extend([report_text(r) for r in rows])
            cursor = end
    z_mm.flush()
    tok_mm.flush()
    np.save(out / "labels14.float32.npy", labels)
    (out / "image_paths.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    (out / "report_texts.json").write_text(json.dumps(reports, ensure_ascii=False), encoding="utf-8")
    meta = {
        "train_jsonl": args.train_jsonl,
        "stage1_checkpoint": args.stage1_checkpoint,
        "model": args.model,
        "pretrained": args.pretrained,
        "selected_layers_public": args.selected_layers,
        "selected_layers_zero_based": selected,
        "num_samples": n,
        "z_shape": [n, 512],
        "token_shape": list(tok_mm.shape),
        "labels_shape": list(labels.shape),
        "skipped_missing": ds.skipped,
        "load_missing_keys": len(msg.missing_keys),
        "load_unexpected_keys": len(msg.unexpected_keys),
        "seed": args.seed,
        "sample_random": not args.no_random_sample,
    }
    (out / "cache_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
