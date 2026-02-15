import argparse
import cv2
import numpy as np
from pathlib import Path


def extract_faces_from_video(
    video_path: str,
    out_dir: Path,
    every_n_frames: int = 10,
    min_face_size: int = 120,
    min_sharpness: float = 60.0,
    min_std: float = 20.0,
    margin: float = 0.25,
    max_faces: int | None = None,
    debug: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    face = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    idx = 0
    saved = 0
    checked = 0
    rejected_small = 0
    rejected_quality = 0
    no_face = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        idx += 1
        if every_n_frames > 1 and (idx % every_n_frames != 0):
            continue

        checked += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(min_face_size, min_face_size),
        )

        if len(faces) == 0:
            no_face += 1
            continue

        # choose largest face
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

        if w < min_face_size or h < min_face_size:
            rejected_small += 1
            continue

        # expand crop
        H, W = frame.shape[:2]
        mx = int(w * margin)
        my = int(h * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(W, x + w + mx)
        y2 = min(H, y + h + my)

        crop = frame[y1:y2, x1:x2]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Quality filters (kills slide false positives)
        sharpness = cv2.Laplacian(crop_gray, cv2.CV_64F).var()
        contrast = float(crop_gray.std())

        if sharpness < min_sharpness or contrast < min_std:
            rejected_quality += 1
            continue

        out_path = out_dir / f"face_{saved:04d}.jpg"
        cv2.imwrite(str(out_path), crop)
        saved += 1

        if debug and saved % 25 == 0:
            print(f"[debug] saved={saved} checked={checked} frame_idx={idx}")

        if max_faces is not None and saved >= max_faces:
            break

    cap.release()

    return {
        "video": video_path,
        "checked_sampled_frames": checked,
        "saved_face_crops": saved,
        "no_face": no_face,
        "rejected_small": rejected_small,
        "rejected_low_quality": rejected_quality,
        "out_dir": str(out_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out_dir", required=True, help="Output folder for face crops")
    parser.add_argument("--every_n_frames", type=int, default=10)
    parser.add_argument("--min_face_size", type=int, default=120)
    parser.add_argument("--min_sharpness", type=float, default=60.0)
    parser.add_argument("--min_std", type=float, default=20.0)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--max_faces", type=int, default=0, help="0 = no limit")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    stats = extract_faces_from_video(
        video_path=args.video,
        out_dir=Path(args.out_dir),
        every_n_frames=args.every_n_frames,
        min_face_size=args.min_face_size,
        min_sharpness=args.min_sharpness,
        min_std=args.min_std,
        margin=args.margin,
        max_faces=None if args.max_faces == 0 else args.max_faces,
        debug=args.debug,
    )

    print("\n✅ Face extraction done")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()