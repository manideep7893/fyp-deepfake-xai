import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models_freq.fft_features import compute_fft_image, radial_frequency_features

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="outputs/models_freq/m4_fft_logreg.joblib")
    ap.add_argument("--num_bins", type=int, default=30)
    return ap.parse_args()

def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.ckpt)
    scaler = bundle["scaler"]
    model = bundle["model"]

    img_paths = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not img_paths:
        raise FileNotFoundError(f"No images in {frames_dir}")

    rows = []
    for p in tqdm(img_paths, desc="Predicting (M4 FFT)"):
        try:
            fft_img = compute_fft_image(str(p))
            feats = radial_frequency_features(fft_img, num_bins=args.num_bins).reshape(1, -1)
            feats = scaler.transform(feats)
            p_fake = float(model.predict_proba(feats)[0, 1])
        except Exception:
            p_fake = float("nan")

        rows.append({"frame": p.name, "path": str(p), "p_fake": p_fake})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "frame_predictions.csv", index=False)

    valid = df["p_fake"].replace([np.inf, -np.inf], np.nan).dropna().values
    if len(valid) == 0:
        agg = {"mean_p_fake": float("nan"), "median_p_fake": float("nan"), "top10pct_mean_p_fake": float("nan")}
        n = 0
    else:
        n = int(len(valid))
        valid_sorted = np.sort(valid)
        top_k = max(1, int(np.ceil(0.10 * n)))
        agg = {
            "mean_p_fake": float(np.mean(valid)),
            "median_p_fake": float(np.median(valid)),
            "top10pct_mean_p_fake": float(np.mean(valid_sorted[-top_k:])),
        }

    summary = {
        "model_id": f"m4_fft_logreg::{args.ckpt}",
        "frames_dir": str(frames_dir),
        "num_frames": len(img_paths),
        "num_valid": n,
        "aggregation": agg,
    }
    with open(out_dir / "video_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDONE ✅")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()