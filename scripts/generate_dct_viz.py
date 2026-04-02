import cv2
import numpy as np
from pathlib import Path

IN_DIR = Path("outputs/frames_faces")
OUT_DIR = Path("outputs/xai/dct")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_TAGS = [
    "fake_019",
    "real_005",
    "fake_033",
    "real_042",
]

def get_mid_frame(video_tag):
    frames = sorted((IN_DIR / video_tag).glob("*.jpg"))
    if not frames:
        return None
    return frames[len(frames)//2]

for tag in VIDEO_TAGS:
    frame_path = get_mid_frame(tag)
    if frame_path is None:
        print(f"Skipping {tag}")
        continue

    img = cv2.imread(str(frame_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize to even dimensions for OpenCV DCT
    h, w = gray.shape
    if h % 2 == 1:
        h -= 1
    if w % 2 == 1:
        w -= 1
    gray = cv2.resize(gray, (w, h))

    dct = cv2.dct(np.float32(gray))
    dct_log = np.log(np.abs(dct) + 1)

    dct_norm = cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX)
    dct_uint8 = np.uint8(dct_norm)

    out_path = OUT_DIR / f"{tag}_dct.png"
    cv2.imwrite(str(out_path), dct_uint8)

    print(f"Saved: {out_path}")