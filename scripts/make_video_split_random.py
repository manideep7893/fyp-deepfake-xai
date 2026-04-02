import json
import random
from pathlib import Path

OUT = Path("outputs/eval")
OUT.mkdir(parents=True, exist_ok=True)

# change this number later for split 2 and split 3
SEED = 999
random.seed(SEED)

fake_videos = [f"fake_{i:03d}" for i in range(1, 51)]
real_videos = [f"real_{i:03d}" for i in range(1, 51)]

random.shuffle(fake_videos)
random.shuffle(real_videos)

train_fake = fake_videos[:40]
test_fake = fake_videos[40:]

train_real = real_videos[:40]
test_real = real_videos[40:]

split = {
    "train": sorted(train_fake + train_real),
    "test": sorted(test_fake + test_real),
}

out_path = OUT / "video_split_random_3.json"
out_path.write_text(json.dumps(split, indent=2))

print("Saved split:", out_path)
print("Train videos:", len(split["train"]))
print("Test videos:", len(split["test"]))
print("Train sample tags:", split["train"][:5])
print("Test sample tags:", split["test"][:5])