import os, glob, json, argparse
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import AutoImageProcessor, AutoModelForImageClassification


def get_device(device: str):
    if device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def softmax_fake_prob(logits, id2label):
    """
    Returns P(fake) robustly using id2label when available.
    Works for cases like:
      id2label: {0:'Fake', 1:'Real'}
      id2label: {0:'REAL', 1:'FAKE'}
    """
    probs = torch.softmax(logits, dim=-1).detach().cpu()[0]

    # id2label keys sometimes come as strings from some configs
    id2label = {int(k): v for k, v in (id2label or {}).items()}

    labels = {i: str(id2label.get(i, i)).lower() for i in range(len(probs))}

    fake_keys = ["fake", "ai", "synthetic", "generated"]
    real_keys = ["real", "authentic"]

    fake_idx = None
    real_idx = None

    for i, name in labels.items():
        if any(k in name for k in fake_keys):
            fake_idx = i
        if any(k in name for k in real_keys):
            real_idx = i

    # Best case: explicit fake label exists
    if fake_idx is not None:
        return float(probs[fake_idx])

    # Next best: explicit real label exists
    if real_idx is not None:
        return float(1.0 - probs[real_idx])

    # Binary fallback (IMPORTANT):
    # Prefer class 0 as fake (common), but only as last resort.
    if len(probs) == 2:
        return float(probs[0])

    # Multi-class unknown fallback
    return float(probs.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="Folder with extracted frames (.jpg/.png)")
    ap.add_argument("--out_dir", required=True, help="Where to write predictions (csv + json)")
    ap.add_argument("--model_id", default="buildborderless/CommunityForensics-DeepfakeDet-ViT")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth frame")
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")) + glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not frames:
        raise FileNotFoundError(f"No frames found in: {args.frames_dir}")

    if args.stride > 1:
        frames = frames[::args.stride]
    if args.max_frames and args.max_frames > 0:
        frames = frames[:args.max_frames]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageClassification.from_pretrained(args.model_id)
    model.to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for fp in tqdm(frames, desc="Predicting frames"):
            img = Image.open(fp).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            p_fake = softmax_fake_prob(outputs.logits, model.config.id2label)

            rows.append({
                "frame_path": fp,
                "p_fake": p_fake
            })

    df = pd.DataFrame(rows)

    # Video-level aggregation (simple + effective):
    # mean, median, and "top-10% mean" to catch short fake segments
    mean_p = float(df["p_fake"].mean())
    med_p = float(df["p_fake"].median())
    topk = max(1, int(0.10 * len(df)))
    top10_mean = float(df["p_fake"].nlargest(topk).mean())

    summary = {
        "model_id": args.model_id,
        "frames_dir": args.frames_dir,
        "num_frames": int(len(df)),
        "aggregation": {
            "mean_p_fake": mean_p,
            "median_p_fake": med_p,
            "top10pct_mean_p_fake": top10_mean
        }
    }

    df.to_csv(out_dir / "frame_predictions.csv", index=False)
    with open(out_dir / "video_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDONE ✅")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()