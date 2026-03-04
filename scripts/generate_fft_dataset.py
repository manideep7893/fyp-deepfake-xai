import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.models_freq.fft_features import compute_fft_image, radial_frequency_features

BASE_DIR = "outputs/frames_faces"
OUT_CSV = "outputs/eval/fft_frame_features_radial.csv"

rows = []

for label_dir in tqdm(os.listdir(BASE_DIR)):
    label_path = os.path.join(BASE_DIR, label_dir)
    if not os.path.isdir(label_path):
        continue

    label = 1 if "fake" in label_dir else 0

    for img_name in os.listdir(label_path):
        if not img_name.endswith(".jpg"):
            continue

        img_path = os.path.join(label_path, img_name)

        try:
            fft_img = compute_fft_image(img_path)
            feats = radial_frequency_features(fft_img, num_bins=30)
            rows.append([img_path, label] + feats.tolist())
        except:
            continue

columns = ["path", "label"] + [f"bin_{i}" for i in range(30)]
df = pd.DataFrame(rows, columns=columns)

os.makedirs("outputs/eval", exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print(f"Saved FFT radial dataset to {OUT_CSV}")
print("Total samples:", len(df))