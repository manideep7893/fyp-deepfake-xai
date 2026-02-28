import os
import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from tqdm import tqdm

import timm
import torchvision.transforms as T


def get_device(req: str):
    if req:
        return req
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone_name = ckpt["backbone_name"]
    feat_dim = ckpt["feat_dim"]

    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    backbone.eval().to(device)

    head = nn.Sequential(
        nn.Linear(feat_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1)
    ).to(device)
    head.load_state_dict(ckpt["head_state"])
    head.eval()

    tfm = T.Compose([
        T.Resize((ckpt["img_size"], ckpt["img_size"])),
        T.ToTensor(),
        T.Normalize(mean=ckpt["mean"], std=ckpt["std"]),
    ])
    return backbone, head, tfm


def iter_images(frames_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    imgs = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in exts])
    return imgs


def aggregate(p_fake: np.ndarray):
    if len(p_fake) == 0:
        return {"mean_p_fake": None, "median_p_fake": None, "top10pct_mean_p_fake": None}
    mean = float(np.mean(p_fake))
    med = float(np.median(p_fake))
    k = max(1, int(np.ceil(0.10 * len(p_fake))))
    top10 = float(np.mean(np.sort(p_fake)[-k:]))
    return {"mean_p_fake": mean, "median_p_fake": med, "top10pct_mean_p_fake": top10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="Directory of face crops (jpg)")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--ckpt", required=True, help="Path to outputs/models_cnn/efficientnet_b0_linearprobe.pt")
    ap.add_argument("--device", default="", help="mps/cpu (default auto)")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = iter_images(frames_dir)
    if len(imgs) == 0:
        raise FileNotFoundError(f"No images found in: {frames_dir}")

    device = get_device(args.device)
    backbone, head, tfm = load_model(args.ckpt, device)

    rows = []
    with torch.no_grad():
        for p in tqdm(imgs, desc="Predicting (CNN)"):
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)

            feats = backbone(x)
            logits = head(feats)
            prob_fake = torch.sigmoid(logits).item()

            rows.append({
                "frame": p.name,
                "path": str(p),
                "p_fake": float(prob_fake),
            })

    df = pd.DataFrame(rows).sort_values("frame")
    df.to_csv(out_dir / "frame_predictions.csv", index=False)

    agg = aggregate(df["p_fake"].values)
    summary = {
        "model_id": f"efficientnet_b0_linearprobe::{args.ckpt}",
        "frames_dir": str(frames_dir),
        "num_frames": int(len(df)),
        "aggregation": agg
    }
    with open(out_dir / "video_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDONE ✅")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()