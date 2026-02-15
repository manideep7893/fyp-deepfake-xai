import os
import cv2
import argparse

def extract_frames(video_path: str, out_dir: str, every_n: int = 1, max_frames: int | None = None,
                   resize: int | None = None, start_sec: float = 0.0, end_sec: float | None = None):

    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    print("DEBUG video_path:", video_path)
    print("DEBUG out_dir:", out_dir)
    print("DEBUG opened:", cap.isOpened())
    print("DEBUG fps:", cap.get(cv2.CAP_PROP_FPS))
    print("DEBUG frame_count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    end_frame = None
    if end_sec is not None:
        end_frame = int(end_sec * fps)

    frame_idx = start_frame
    saved = 0

    while True:
        if end_frame is not None and frame_idx >= end_frame:
            break
        if max_frames is not None and saved >= max_frames:
            break

        ret, frame = cap.read()

        if frame_idx < start_frame + 5:
            print("DEBUG read:", frame_idx, "ret=", ret, "frame_none=", frame is None)

        if not ret or frame is None:
            break

        if (frame_idx - start_frame) % every_n == 0:
            if resize is not None:
                frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)

            out_path = os.path.join(out_dir, f"frame_{saved:06d}.jpg")
            ok = cv2.imwrite(out_path, frame)

            if saved < 5:
                print("DEBUG write:", out_path, "ok=", ok)

            if not ok:
                raise RuntimeError(f"Failed to write frame: {out_path}")

            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Done. Saved {saved} frames to: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every_n", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--resize", type=int, default=None)
    ap.add_argument("--start_sec", type=float, default=0.0)
    ap.add_argument("--end_sec", type=float, default=None)
    args = ap.parse_args()

    extract_frames(
        video_path=args.video,
        out_dir=args.out,
        every_n=args.every_n,
        max_frames=args.max_frames,
        resize=args.resize,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--every_n", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--end_sec", type=float, default=None)

    args = parser.parse_args()

    stats = extract_frames(
        video_path=args.video,
        out_dir=args.out,
        every_n=args.every_n,
        max_frames=args.max_frames,
        resize=args.resize,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )

    print("DONE:", stats)