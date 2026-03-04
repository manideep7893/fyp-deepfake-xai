import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models_freq.dct_features import compute_dct_features


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="outputs/models_freq/m4b_dct_logreg.joblib")
    return ap.parse_args()


def summarize(probs):
    probs = np.asarray(probs, dtype=float)
    probs_sorted = np.sort(probs)
    k = max(1, int(0.1 * len(probs_sorted)))
    top10 = probs_sorted[-k:]
    return {
        "mean_p_fake": float(probs.mean()),
        "median_p_fake": float(np.median(probs)),
        "top10pct_mean_p_fake": float(top10.mean())
    }


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.ckpt)

    img_paths = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {frames_dir}")

    rows = []
    probs = []

    for p in tqdm(img_paths, desc="Predicting (M4b DCT)"):
        feat = compute_dct_features(str(p))
        prob = float(model.predict_proba([feat])[0, 1])
        probs.append(prob)
        rows.append({"frame": p.name, "path": str(p), "p_fake": prob})

    pd.DataFrame(rows).to_csv(out_dir / "frame_predictions.csv", index=False)

    summary = {
        "model_id": f"m4b_dct_logreg::{args.ckpt}",
        "frames_dir": str(frames_dir),
        "num_frames": len(img_paths),
        "aggregation": summarize(probs),
    }

    with open(out_dir / "video_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDONE ✅")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()