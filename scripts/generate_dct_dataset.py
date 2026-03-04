import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.models_freq.dct_features import compute_dct_features

frames_root = Path("outputs/frames_faces")
rows = []

for video_dir in tqdm(list(frames_root.iterdir())):

    label = 1 if "fake" in video_dir.name else 0

    for img_path in video_dir.glob("*.jpg"):

        try:
            feats = compute_dct_features(str(img_path))

            row = {"label": label}

            for i, f in enumerate(feats):
                row[f"f{i}"] = f

            rows.append(row)

        except Exception:
            continue

df = pd.DataFrame(rows)

out = "outputs/eval/dct_frame_features.csv"
df.to_csv(out, index=False)

print("Saved:", out)
print("Samples:", len(df))