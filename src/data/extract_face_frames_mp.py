import cv2
import mediapipe as mp
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--every_n_frames", type=int, default=5)
parser.add_argument("--min_face_px", type=int, default=80)
parser.add_argument("--conf", type=float, default=0.5)
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {args.video}")

mp_face_detection = mp.solutions.face_detection

idx = 0
saved = 0
checked = 0

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=args.conf) as face_det:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        idx += 1
        if idx % args.every_n_frames != 0:
            continue

        checked += 1
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = face_det.process(rgb)
        if not result.detections:
            continue

        # take largest detected face
        det = max(
            result.detections,
            key=lambda d: d.location_data.relative_bounding_box.width *
                          d.location_data.relative_bounding_box.height
        )

        box = det.location_data.relative_bounding_box
        x1 = max(0, int(box.xmin * w))
        y1 = max(0, int(box.ymin * h))
        x2 = min(w, int((box.xmin + box.width) * w))
        y2 = min(h, int((box.ymin + box.height) * h))

        bw = x2 - x1
        bh = y2 - y1
        if bw < args.min_face_px or bh < args.min_face_px:
            continue

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        out_path = out_dir / f"face_{saved:04d}.jpg"
        cv2.imwrite(str(out_path), face)
        saved += 1

cap.release()

print("Checked frames:", checked)
print("Saved faces:", saved)
print("Output dir:", out_dir)