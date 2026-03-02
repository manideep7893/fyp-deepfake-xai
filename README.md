# Reliability-Aware Explainable Deepfake Detection using Multi-Model Ensemble and Vision Transformers (Celeb-DF v2)

This project implements a **reliability-aware, explainable deepfake detection system**, combining:

- Vision Transformer-based deepfake detection
- Multi-model ensemble fusion (M1–M3)
- Scenario-based reliability analysis (S1–S4)
- Grad-CAM visual explanations
- Region-level attribution analysis
- Faithfulness evaluation (Insertion / Deletion AUC)
- Threshold calibration and ROC/PR analysis

The system is evaluated on **Celeb-DF (v2)** and designed for **video-level forensic analysis with explainability and uncertainty awareness**, suitable for academic research and final-year dissertation work.

---

## 🎯 Research Motivation

Deepfake detection models often behave as black boxes. High accuracy alone is insufficient for forensic trust and deployment in sensitive contexts.

This project addresses three core research questions:

1. Can a Vision Transformer reliably discriminate real vs fake faces on Celeb-DF?
2. Are its decisions explainable and faithful to manipulated facial regions?
3. Can ensemble agreement and alignment signals improve reliability and interpretability?

Rather than reporting accuracy alone, this system evaluates:

- Discrimination (ROC-AUC, PR-AUC)
- Threshold stability and calibration
- Model agreement and alignment
- Scenario-based reliability classification
- Attribution faithfulness via deletion/insertion curves

This transforms the project from a classifier into a **forensic, explainable deepfake analysis framework**.

---

## 🧠 System Overview

The pipeline consists of six stages:

1. **Video Preprocessing**
2. **Face Extraction (YuNet)**
3. **Frame-Level Deepfake Detection (M1–M3)**
4. **Reliability-Weighted Ensemble Fusion**
5. **Scenario Classification (S1–S4)**
6. **Explainability & Faithfulness Evaluation**

---

## 🔍 Core Components

### 1️⃣ Video Preprocessing

- Frame extraction from videos
- Face detection using YuNet (ONNX)
- Margin-aware cropping for consistent input

---

### 2️⃣ Deepfake Detection Models

- Base Model (M3): Vision Transformer (ViT-based)
- Additional models (M1, M2): auxiliary CNN-based predictors
- Frame-level fake probability (`p_fake`)
- Video-level aggregation:
  - Mean probability
  - Median probability
  - Top 10% mean

---

### 3️⃣ Reliability-Weighted Ensemble Fusion

A reliability head combines M1–M3 outputs using:

- Model confidence distance from decision boundary
- Cross-model agreement (A vs B)
- Alignment score
- Scenario-based interpretation

Instead of naive averaging, fusion is weighted by reliability
p_final = reliability * M3 + (1 - reliability) * mean(M1, M2, M3)

---

### 4️⃣ Scenario-Based Reliability Classification

Each video is assigned a scenario:

- **S1:** A/B consensus overrides (possible domain shift)
- **S2:** M3 dominant but weak consensus
- **S3:** Strong consensus (high confidence)
- **S4:** Disagreement / uncertain (needs review)

This introduces structured interpretability beyond probability scores.

---

### 5️⃣ Explainability (XAI)

- Grad-CAM heatmaps
- MediaPipe region attribution:
  - Mouth
  - Eye region
  - Face boundary
- Console explanation reports
- Saved heatmap overlays

---

### 6️⃣ Faithfulness Metrics

To evaluate explanation reliability:

- Deletion AUC
- Insertion AUC
- Confidence drop analysis

This ensures explanations are not visually appealing but scientifically grounded.

---

## 📊 Evaluation Outputs

Evaluation produces:
outputs/eval/
├── frame_level_metrics.json
├── roc_curve.png
├── pr_curve.png
├── video_level_metrics.csv
└── reliability_analysis.csv

Metrics include:

- ROC-AUC
- PR-AUC
- Accuracy
- F1-score
- Threshold calibration (Youden’s J, Best F1)
- Reliability-weighted fusion comparison
- Scenario distribution statistics

---

## 📂 Project Structure
deepfake-xai-celebdf/
├── src/
│   ├── data/
│   ├── models/
│   ├── ensemble/
│   │   └── reliability.py
│   ├── xai/
│   ├── eval/
│   └── utils/
│
├── scripts/
├── models/
├── data/
├── outputs/        (generated, not tracked)
├── README.md
└── requirements.txt

---

## 🚀 How to Run

### 
1️⃣ Install Dependencies

```bash
pip install -r requirements.txt

2️⃣ Extract Frames
python src/data/video_to_frames.py \
  --video path/to/video.mp4 \
  --out outputs/frames/sample \
  --every_n 5

3️⃣ Extract Faces
python src/data/extract_faces_from_frames_yunet.py \
  --frames_dir outputs/frames/sample \
  --out_dir outputs/frames_faces/sample \
  --min_face 120 \
  --score_thr 0.9 \
  --margin 0.25

4️⃣ Run Deepfake Detection
python src/models/hf_predict_frames.py \
  --frames_dir outputs/frames_faces/sample \
  --out_dir outputs/preds/sample \
  --model_id prithivMLmods/Deep-Fake-Detector-Model \
  --device mps

5️⃣ Generate XAI Report
python src/xai/xai_console_report.py \
  --frame outputs/frames_faces/sample/face_00010.jpg \
  --model_id prithivMLmods/Deep-Fake-Detector-Model \
  --device mps \
  --target_class 0 \
  --out_dir outputs/xai/sample

6️⃣ Run Reliability Evaluation
python scripts/run_reliability_eval.py
This evaluates:
	•	Model C baseline
	•	Naive ensemble
	•	Reliability-weighted fusion
	•	Scenario classification

📈 Research Contributions

This project contributes:
	•	Reliability-aware ensemble fusion for deepfake detection
	•	Scenario-based interpretability (S1–S4 classification)
	•	Grad-CAM + region-level attribution analysis
	•	Faithfulness evaluation using insertion/deletion metrics
	•	Video-level aggregation with calibration analysis
	•	Agreement-based uncertainty estimation

The system integrates discrimination, interpretability, and reliability into a unified forensic pipeline.

⚠️ Notes
	•	outputs/ is not tracked in Git.
	•	Celeb-DF dataset must be downloaded separately.
	•	No dataset files are included in this repository.

📚 Dataset
	•	Celeb-DF v2
	•	Real and synthetic videos
	•	Face-level preprocessing applied before inference

🏆 Academic Context

This repository supports a Final Year Project focused on:

Reliability-Aware Explainable Deepfake Detection using Vision Transformers and Ensemble Analysis.

The emphasis is on:
	•	Scientific evaluation
	•	Interpretability
	•	Reliability modelling
	•	Robustness analysis

Rather than model training alone.

  
