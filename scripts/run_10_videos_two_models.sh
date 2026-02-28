#!/usr/bin/env bash
set -e

EVERY_N=5
MODEL_A="prithivMLmods/Deep-Fake-Detector-Model"
MODEL_B="dima806/deepfake_vs_real_image_detection"
DEVICE="mps"

# Mac-compatible array assignment
FAKE_VIDS=($(find data/celebdf/Celeb-synthesis -type f -name "*.mp4" | head -n 5))
REAL_VIDS=($(find data/celebdf/Celeb-real -type f -name "*.mp4" | head -n 5))

echo "FAKE videos:"
printf "%s\n" "${FAKE_VIDS[@]}"
echo
echo "REAL videos:"
printf "%s\n" "${REAL_VIDS[@]}"
echo

run_one () {
  vid="$1"
  tag="$2"

  echo "=============================="
  echo "RUNNING $tag"
  echo "VIDEO: $vid"
  echo "=============================="

  mkdir -p "outputs/frames/$tag"
  mkdir -p "outputs/frames_faces/$tag"
  mkdir -p "outputs/preds/$tag"
  mkdir -p "outputs/preds_modelB/$tag"

  python src/data/video_to_frames.py \
    --video "$vid" \
    --out "outputs/frames/$tag" \
    --every_n "$EVERY_N"

  python src/data/extract_faces_from_frames_yunet.py \
    --frames_dir "outputs/frames/$tag" \
    --out_dir "outputs/frames_faces/$tag" \
    --min_face 120 \
    --score_thr 0.9 \
    --margin 0.25 \
    --max_faces 200

  FACE_COUNT=$(ls outputs/frames_faces/$tag | wc -l | tr -d ' ')

  if [ "$FACE_COUNT" -eq 0 ]; then
    echo "⚠️  No faces found for $tag. Skipping inference."
    return
  fi

  python src/models/hf_predict_frames.py \
    --frames_dir "outputs/frames_faces/$tag" \
    --out_dir "outputs/preds/$tag" \
    --model_id "$MODEL_A" \
    --device "$DEVICE"

  python src/models/hf_predict_frames.py \
    --frames_dir "outputs/frames_faces/$tag" \
    --out_dir "outputs/preds_modelB/$tag" \
    --model_id "$MODEL_B" \
    --device "$DEVICE"

  echo "✅ DONE $tag"
}

i=1
for v in "${FAKE_VIDS[@]}"; do
  tag=$(printf "fake_%03d" "$i")
  run_one "$v" "$tag"
  i=$((i+1))
done

i=1
for v in "${REAL_VIDS[@]}"; do
  tag=$(printf "real_%03d" "$i")
  run_one "$v" "$tag"
  i=$((i+1))
done

echo "✅ All 10 videos processed."
