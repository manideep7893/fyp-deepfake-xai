# scripts/make_video_split.py
import json
from pathlib import Path

OUT = Path("outputs/eval")
OUT.mkdir(parents=True, exist_ok=True)

train_fake = [f"fake_{i:03d}" for i in range(1, 41)]
test_fake  = [f"fake_{i:03d}" for i in range(41, 51)]

train_real = [f"real_{i:03d}" for i in range(1, 41)]
test_real  = [f"real_{i:03d}" for i in range(41, 51)]

split = {
    "train": sorted(train_fake + train_real),
    "test":  sorted(test_fake + test_real),
}

out_path = OUT / "video_split_40_10.json"
out_path.write_text(json.dumps(split, indent=2))
print("Saved split:", out_path)
print("Train videos:", len(split["train"]))
print("Test videos:", len(split["test"]))