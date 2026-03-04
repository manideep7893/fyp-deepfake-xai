import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.models_freq.dct_features import dct_feature_vector


def build_dataset(video_tags, frames_root):
    rows = []

    for tag in tqdm(video_tags, desc="Processing videos"):
        vid_dir = frames_root / tag

        if not vid_dir.exists():
            continue

        label = 1 if "fake" in tag else 0

        for img_path in sorted(vid_dir.glob("*.jpg")):
            feats = dct_feature_vector(str(img_path))

            rows.append({
                "tag": tag,
                "frame": img_path.name,
                "path": str(img_path),
                "gt_fake": label,
                **{f"f{i}": feats[i] for i in range(len(feats))}
            })

    return pd.DataFrame(rows)


def main():

    split_path = Path("outputs/eval/video_split_40_10.json")
    frames_root = Path("outputs/frames_faces")

    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    with open(split_path) as f:
        split = json.load(f)

    # -------- TRAIN DATASET --------
    print("\nBuilding TRAIN DCT dataset")

    df_train = build_dataset(split["train"], frames_root)

    train_csv = Path("outputs/eval/dct_train_frame_features.csv")
    train_csv.parent.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(train_csv, index=False)

    print("Saved:", train_csv)
    print("Train samples:", len(df_train))
    print("Train videos:", df_train["tag"].nunique())

    # -------- TEST DATASET --------
    print("\nBuilding TEST DCT dataset")

    df_test = build_dataset(split["test"], frames_root)

    test_csv = Path("outputs/eval/dct_test_frame_features.csv")

    df_test.to_csv(test_csv, index=False)

    print("Saved:", test_csv)
    print("Test samples:", len(df_test))
    print("Test videos:", df_test["tag"].nunique())


if __name__ == "__main__":
    main()