# DeepFake XAI – Explainable Deepfake Detection (Celeb-DF v2)

This project implements an **Explainable AI (XAI) pipeline for deepfake detection**, combining a pretrained deepfake detector with **Grad-CAM visual explanations**, **region-level attribution**, and **faithfulness metrics**.

The system is evaluated on the **Celeb-DF (v2)** dataset and is designed for **frame-level and face-level explainability**, suitable for academic projects and final-year dissertations.

---

## 🔍 Key Features

- Deepfake detection using **HuggingFace Vision Transformer models**
- **Frame extraction** and **face cropping** from videos
- **Grad-CAM heatmaps** for visual explanations
- Region-based attribution:
  - Mouth
  - Eyes
  - Face boundary
- **Faithfulness evaluation**:
  - Insertion AUC
  - Deletion AUC
- Console-based XAI reports + saved visual artefacts

---

## 📂 Project Structure
deepfake-xai-celebdf/
├── src/
│   ├── data/
│   │   ├── video_to_frames.py
│   │   ├── extract_face_frames.py
│   │   └── extract_faces_from_frames_yunet.py
│   ├── models/
│   │   └── hf_predict_frames.py
│   ├── xai/
│   │   └── xai_console_report.py
│   └── sanity.py
├── outputs/
│   ├── frames/
│   ├── frames_faces/
│   ├── preds_faces/
│   └── xai_faces/
├── data/
│   └── celebdf/
├── requirements/
├── README.md
└── .gitignore
