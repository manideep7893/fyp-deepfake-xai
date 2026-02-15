import argparse
from pathlib import Path
import cv2
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="Folder containing .jpg/.png frames")
    ap.add_argument("--out_dir", required=True, help="Where to save cropped faces")
    ap.add_argument("--model", default="models/face_detection_yunet_2023mar.onnx", help="YuNet ONNX path")
    ap.add_argument("--min_face", type=int, default=120, help="Minimum face size in pixels (w/h)")
    ap.add_argument("--score_thr", type=float, default=0.9, help="Detection score threshold")
    ap.add_argument("--nms_thr", type=float, default=0.3, help="NMS threshold")
    ap.add_argument("--topk", type=int, default=50, help="TopK detections")
    ap.add_argument("--margin", type=float, default=0.25, help="Crop margin around face box")
    ap.add_argument("--max_faces", type=int, default=300, help="Max crops to save")
    return ap.parse_args()

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = str(Path(args.model))
    if not Path(model_path).exists():
        raise FileNotFoundError(f"YuNet model not found: {model_path}")

    img_paths = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not img_paths:
        raise FileNotFoundError(f"No images found in {frames_dir}")

    saved = 0
    processed = 0
    no_face = 0

    for p in img_paths:
        if saved >= args.max_faces:
            break

        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        processed += 1
        h, w = img_bgr.shape[:2]

        # YuNet requires input size set per image size
        detector = cv2.FaceDetectorYN.create(
            model_path, "", (w, h),
            args.score_thr, args.nms_thr, args.topk
        )
        _, faces = detector.detect(img_bgr)

        if faces is None or len(faces) == 0:
            no_face += 1
            continue

        # faces: [x, y, w, h, score, lmkx5..., lmky5...]
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)  # largest first

        # Save ONLY the largest face per frame (cleanest for your pipeline)
        f = faces[0]
        x, y, bw, bh, score = f[:5]

        if bw < args.min_face or bh < args.min_face:
            continue

        mx = int(bw * args.margin)
        my = int(bh * args.margin)

        x1 = clamp(int(x) - mx, 0, w - 1)
        y1 = clamp(int(y) - my, 0, h - 1)
        x2 = clamp(int(x + bw) + mx, 0, w)
        y2 = clamp(int(y + bh) + my, 0, h)

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        out_path = out_dir / f"face_{saved:05d}.jpg"
        cv2.imwrite(str(out_path), crop)
        saved += 1

    print("✅ Face extraction complete")
    print("frames_dir:", frames_dir)
    print("out_dir:", out_dir)
    print("processed_frames:", processed)
    print("saved_faces:", saved)
    print("no_face_frames:", no_face)

if __name__ == "__main__":
    main()