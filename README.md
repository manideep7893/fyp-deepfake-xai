# DeepFake XAI – Explainable Deepfake Detection (Celeb-DF v2)

This project implements an **Explainable AI (XAI) pipeline for deepfake detection**, combining a pretrained Vision Transformer deepfake detector with:

- Grad-CAM visual explanations  
- Region-level attribution analysis  
- Faithfulness evaluation (Insertion / Deletion AUC)  
- Threshold calibration and ROC/PR analysis  

The system is evaluated on the **Celeb-DF (v2)** dataset and designed for **frame-level and face-level explainability**, suitable for academic research and final-year dissertation work.

---

## 🎯 Project Motivation

Deepfake detection models often act as black boxes.  
This project addresses two core research questions:

1. Can a pretrained Vision Transformer reliably discriminate real vs fake faces on Celeb-DF?
2. Are its decisions explainable and faithful to manipulated facial regions?

Rather than only reporting accuracy, this project evaluates:

- Discrimination (ROC-AUC, PR-AUC)
- Calibration and threshold stability
- Region-based attribution patterns
- Faithfulness via deletion/insertion curves

This transforms the project from simple model usage into **scientific model evaluation and explainability research**.

---

## 🧠 Model

- Base Model: `prithivMLmods/Deep-Fake-Detector-Model`
- Architecture: Vision Transformer (ViT-based)
- Inference: Frame-level, aggregated to video-level

Although the classifier is transformer-based, CNN principles remain central for:

- Face detection
- Spatial localisation
- Grad-CAM heatmap generation
- Faithfulness masking operations

---

## 🔍 Key Features

### 1️⃣ Video Preprocessing
- Frame extraction from videos
- Face detection using YuNet (ONNX)
- Face cropping for consistent model input

### 2️⃣ Deepfake Detection
- HuggingFace inference pipeline
- Frame-level probability prediction (`p_fake`)
- Video-level aggregation:
  - Mean probability
  - Median probability
  - Top 10% mean probability

### 3️⃣ Explainability (XAI)

- Grad-CAM heatmaps
- MediaPipe-based region attribution:
  - Mouth
  - Eye region
  - Face boundary
- Console-based explanation reports
- Saved heatmap overlays

### 4️⃣ Faithfulness Metrics

- Deletion AUC
- Insertion AUC
- Confidence drop analysis

### 5️⃣ Scientific Evaluation

- ROC Curve
- PR Curve
- Threshold calibration:
  - Default 0.5
  - Youden’s J
  - Best F1 threshold
- Frame-level confusion metrics

---

## 📊 Evaluation Outputs

Running evaluation generates:

```
outputs/eval/
├── frame_level_metrics.json
├── roc_curve.png
├── pr_curve.png
└── all_frames_combined.csv
```

Metrics include:

- ROC-AUC
- PR-AUC
- Accuracy
- Precision
- Recall (TPR)
- False Positive Rate
- Optimal threshold values

---

## 📂 Project Structure

```
deepfake-xai-celebdf/
├── src/
│   ├── data/
│   │   ├── video_to_frames.py
│   │   ├── extract_faces_from_frames_yunet.py
│   ├── models/
│   │   └── hf_predict_frames.py
│   ├── xai/
│   │   └── xai_console_report.py
│   ├── eval/
│   │   └── eval_thresholds.py
│   └── sanity.py
│
├── models/
│   └── face_detection_yunet_2023mar.onnx
│
├── data/
│   └── celebdf/
│
├── outputs/         (generated, not tracked)
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Extract Frames

```bash
python src/data/video_to_frames.py \
  --video path/to/video.mp4 \
  --out outputs/frames/sample \
  --every_n 5
```

---

### 3️⃣ Extract Faces

```bash
python src/data/extract_faces_from_frames_yunet.py \
  --frames_dir outputs/frames/sample \
  --out_dir outputs/frames_faces/sample \
  --min_face 120 \
  --score_thr 0.9 \
  --margin 0.25
```

---

### 4️⃣ Run Detection

```bash
python src/models/hf_predict_frames.py \
  --frames_dir outputs/frames_faces/sample \
  --out_dir outputs/preds/sample \
  --model_id prithivMLmods/Deep-Fake-Detector-Model \
  --device mps
```

---

### 5️⃣ Generate XAI Report

```bash
python src/xai/xai_console_report.py \
  --frame outputs/frames_faces/sample/face_00010.jpg \
  --model_id prithivMLmods/Deep-Fake-Detector-Model \
  --device mps \
  --target_class 0 \
  --out_dir outputs/xai/sample
```

---

### 6️⃣ Run Full Evaluation

```bash
python src/eval/eval_thresholds.py
```

This computes:

- ROC-AUC
- PR-AUC
- Optimal thresholds
- Frame-level metrics

---

## 📈 Research Contribution

This project goes beyond simple accuracy reporting by:

- Performing threshold calibration analysis
- Evaluating discrimination vs calibration trade-offs
- Analysing decision boundary stability
- Comparing attribution patterns across real and fake samples
- Measuring explanation faithfulness quantitatively

This transforms the system into an **explainable forensic analysis pipeline** rather than a binary classifier.

---

## ⚠️ Notes

- `outputs/` directory is not tracked in Git.
- Celeb-DF dataset must be downloaded separately.
- The repository contains code only, not dataset files.

---

## 📚 Dataset

- Celeb-DF v2  
- Real and synthesis videos  
- Frame-level face cropping applied before inference  

---

## 🏆 Academic Context

This repository supports a Final Year Project focused on:

> Explainable Deepfake Detection using Vision Transformers and Faithfulness Evaluation.

The emphasis is on **scientific evaluation, interpretability, and robustness analysis**, rather than model training alone.