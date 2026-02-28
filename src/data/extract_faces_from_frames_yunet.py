import argparse
from pathlib import Path
import cv2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="Folder containing .jpg/.png frames")
    ap.add_argument("--out_dir", required=True, help="Where to save cropped faces")
    ap.add_argument("--model", default="models/face_detection_yunet_2023mar.onnx", help="YuNet ONNX path")

    ap.add_argument("--min_face", type=int, default=120, help="Minimum face size in pixels (w/h)")
    ap.add_argument("--score_thr", type=float, default=0.9, help="Detection score threshold")
    ap.add_argument("--nms_thr", type=float, default=0.3, help="NMS threshold")
    ap.add_argument("--topk", type=int, default=50, help="TopK detections")
    ap.add_argument("--margin", type=float, default=0.20, help="Crop margin around face box")
    ap.add_argument("--max_faces", type=int, default=300, help="Max crops to save")

    # New safety gates
    ap.add_argument("--min_area_ratio", type=float, default=0.01, help="Min box area / image area")
    ap.add_argument("--max_area_ratio", type=float, default=0.60, help="Max box area / image area")
    ap.add_argument("--min_ar", type=float, default=0.60, help="Min aspect ratio (w/h)")
    ap.add_argument("--max_ar", type=float, default=1.80, help="Max aspect ratio (w/h)")
    ap.add_argument("--recheck_crop", action="store_true", help="Run YuNet again on crop to verify face")
    return ap.parse_args()

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"YuNet model not found: {model_path}")

    img_paths = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not img_paths:
        raise FileNotFoundError(f"No images found in {frames_dir}")

    # Create detector once; setInputSize per image
    detector = cv2.FaceDetectorYN.create(
        str(model_path), "", (320, 320),  # dummy size; replaced each frame
        args.score_thr, args.nms_thr, args.topk
    )

    saved = 0
    processed = 0
    no_face = 0
    rejected = 0

    for p in img_paths:
        if saved >= args.max_faces:
            break

        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        processed += 1
        h, w = img_bgr.shape[:2]
        img_area = float(w * h)

        detector.setInputSize((w, h))
        _, faces = detector.detect(img_bgr)

        if faces is None or len(faces) == 0:
            no_face += 1
            continue

        # Pick highest-confidence detection (NOT largest)
        f = max(faces, key=lambda r: float(r[4]))
        x, y, bw, bh, score = map(float, f[:5])

        # Size gate
        if bw < args.min_face or bh < args.min_face:
            rejected += 1
            continue

        # Aspect ratio gate
        ar = bw / (bh + 1e-6)
        if ar < args.min_ar or ar > args.max_ar:
            rejected += 1
            continue

        # Area ratio gate
        box_area = bw * bh
        if box_area < args.min_area_ratio * img_area:
            rejected += 1
            continue
        if box_area > args.max_area_ratio * img_area:
            rejected += 1
            continue

        # Crop with margin
        mx = int(bw * args.margin)
        my = int(bh * args.margin)

        x1 = clamp(int(x) - mx, 0, w - 1)
        y1 = clamp(int(y) - my, 0, h - 1)
        x2 = clamp(int(x + bw) + mx, 0, w)
        y2 = clamp(int(y + bh) + my, 0, h)

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            rejected += 1
            continue

        # Optional: re-check that the crop still contains a face
        if args.recheck_crop:
            ch, cw = crop.shape[:2]
            if ch < args.min_face or cw < args.min_face:
                rejected += 1
                continue

            det2 = cv2.FaceDetectorYN.create(
                str(model_path), "", (cw, ch),
                args.score_thr, args.nms_thr, args.topk
            )
            _, faces2 = det2.detect(crop)
            if faces2 is None or len(faces2) == 0:
                rejected += 1
                continue

            best2 = max(faces2, key=lambda r: float(r[4]))
            if float(best2[4]) < args.score_thr:
                rejected += 1
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
    print("rejected_frames:", rejected)

if __name__ == "__main__":
    main()