#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
from open_clip.grade_stage2_simple import Stage2Refiner, forward_stage1_tokens, parse_selected_layers, visual_num_layers  # noqa: E402
from train_stage2 import DEFAULT_MODEL, DEFAULT_PRETRAINED, read_state_dict  # noqa: E402


class ImageDataset(Dataset):
    def __init__(self, path, preprocess, img_key):
        self.path = Path(path)
        self.preprocess = preprocess
        self.img_key = img_key
        if self.path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.df = pd.DataFrame(rows)
        else:
            self.df = pd.read_csv(self.path)
        if self.img_key not in self.df.columns:
            for key in ["ImageID", "image_path", "Image Index"]:
                if key in self.df.columns:
                    self.img_key = key
                    break
        if self.img_key not in self.df.columns:
            raise ValueError(f"No image column found in {path}")
        self.df = self.df.dropna(subset=[self.img_key]).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = str(self.df.iloc[idx][self.img_key])
        return self.preprocess(Image.open(p).convert("RGB")), p


def collate(batch):
    return torch.stack([x[0] for x in batch]), [x[1] for x in batch]


def main():
    p = argparse.ArgumentParser(description="Extract simple Stage-2 base/refined features.")
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stage1-checkpoint", required=True)
    p.add_argument("--stage2-checkpoint", default="")
    p.add_argument("--representation", choices=["base", "refined"], default="refined")
    p.add_argument("--img-key", default="Image Index")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    p.add_argument("--selected-layers", default="4,8,12")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = "cuda:0" if args.device == "cuda" and torch.cuda.is_available() else args.device

    model, _, preprocess = create_model_and_transforms(args.model, pretrained=args.pretrained, device=device, output_dict=True)
    msg = model.load_state_dict(read_state_dict(args.stage1_checkpoint), strict=False)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    selected = parse_selected_layers(args.selected_layers, visual_num_layers(model))
    refiner = None
    if args.representation == "refined":
        ckpt = torch.load(args.stage2_checkpoint, map_location=device, weights_only=False)
        cfg = ckpt.get("args", {})
        refiner = Stage2Refiner(
            embed_dim=int(cfg.get("embed_dim", 512)),
            num_layers=len(selected),
            num_heads=int(cfg.get("num_heads", 8)),
        ).to(device)
        refiner.load_state_dict(ckpt["stage2_refiner"])
        refiner.eval()

    ds = ImageDataset(args.input, preprocess, args.img_key)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)
    feats, paths = [], []
    with torch.no_grad():
        for images, batch_paths in tqdm(loader, desc=f"extract-{args.representation}", ncols=100):
            images = images.to(device, non_blocking=True)
            z_base, selected_tokens, _ = forward_stage1_tokens(model.visual, images, selected)
            z = refiner(z_base, selected_tokens)["z_ref"] if refiner is not None else z_base
            feats.append(F.normalize(z.float(), dim=-1).cpu().numpy())
            paths.extend(batch_paths)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"features_stage2_{args.representation}.npy", np.concatenate(feats, axis=0).astype(np.float32))
    pd.DataFrame({"image_path": paths}).to_csv(out / "feature_image_paths.csv", index=False)
    meta = {
        "input": args.input,
        "representation": args.representation,
        "stage1_checkpoint": args.stage1_checkpoint,
        "stage2_checkpoint": args.stage2_checkpoint,
        "selected_layers_public": args.selected_layers,
        "selected_layers_zero_based": selected,
        "num_samples": len(paths),
        "missing_keys": len(msg.missing_keys),
        "unexpected_keys": len(msg.unexpected_keys),
    }
    (out / f"features_stage2_{args.representation}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
