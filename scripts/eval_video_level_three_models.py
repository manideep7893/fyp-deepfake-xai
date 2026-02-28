import os, json, glob, subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

MODEL_A = "prithivMLmods/Deep-Fake-Detector-Model"
MODEL_B = "dima806/deepfake_vs_real_image_detection"

# --- NEW: CNN Model C checkpoint ---
MODEL_C_CKPT = "outputs/models_cnn/efficientnet_b0_linearprobe.pt"

EVERY_N = 5
MIN_FACE = 120
SCORE_THR = 0.9
MARGIN = 0.25
MAX_FACES = 200
DEVICE = "mps"

OUT_EVAL_DIR = Path("outputs/eval")
OUT_EVAL_DIR.mkdir(parents=True, exist_ok=True)

def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)

def get_mean_from_hf_summary(summary_path):
    with open(summary_path, "r") as f:
        j = json.load(f)
    return float(j["aggregation"]["mean_p_fake"]), int(j["num_frames"])

# --- NEW: CNN summary parser (same keys but separate function for clarity) ---
def get_mean_from_cnn_summary(summary_path):
    with open(summary_path, "r") as f:
        j = json.load(f)
    return float(j["aggregation"]["mean_p_fake"]), int(j["num_frames"])

def pick_videos(folder, n):
    vids = sorted(glob.glob(str(Path(folder) / "*.mp4")))
    return vids[:n]

def process_one(video_path, tag):
    frames_dir = Path("outputs/frames") / tag
    faces_dir  = Path("outputs/frames_faces") / tag
    predA_dir   = Path("outputs/preds") / tag
    predB_dir   = Path("outputs/preds_modelB") / tag
    predC_dir   = Path("outputs/preds_modelC") / tag  # NEW

    for d in [frames_dir, faces_dir, predA_dir, predB_dir, predC_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1) frames
    run(["python", "src/data/video_to_frames.py",
         "--video", video_path,
         "--out", str(frames_dir),
         "--every_n", str(EVERY_N)])

    # 2) faces
    run(["python", "src/data/extract_faces_from_frames_yunet.py",
         "--frames_dir", str(frames_dir),
         "--out_dir", str(faces_dir),
         "--min_face", str(MIN_FACE),
         "--score_thr", str(SCORE_THR),
         "--margin", str(MARGIN),
         "--max_faces", str(MAX_FACES)])

    face_files = list(faces_dir.glob("*.jpg"))
    if len(face_files) == 0:
        return {"tag": tag, "video": video_path, "note": "no_faces"}

    # 3) model A
    run(["python", "src/models/hf_predict_frames.py",
         "--frames_dir", str(faces_dir),
         "--out_dir", str(predA_dir),
         "--model_id", MODEL_A,
         "--device", DEVICE])

    # 4) model B
    run(["python", "src/models/hf_predict_frames.py",
         "--frames_dir", str(faces_dir),
         "--out_dir", str(predB_dir),
         "--model_id", MODEL_B,
         "--device", DEVICE])

    # 5) model C (CNN)
    # (It only needs faces_dir, ckpt, out_dir)
    run(["python", "src/models_cnn/predict_efficientnet_frames.py",
         "--frames_dir", str(faces_dir),
         "--out_dir", str(predC_dir),
         "--ckpt", MODEL_C_CKPT,
         "--device", DEVICE])

    # parse summaries
    A_mean, A_n = get_mean_from_hf_summary(predA_dir / "video_summary.json")
    B_mean, B_n = get_mean_from_hf_summary(predB_dir / "video_summary.json")
    C_mean, C_n = get_mean_from_cnn_summary(predC_dir / "video_summary.json")

    ens_AB  = (A_mean + B_mean) / 2.0
    ens_ABC = (A_mean + B_mean + C_mean) / 3.0

    return {
        "tag": tag,
        "video": video_path,
        "A_mean": A_mean, "A_frames": A_n,
        "B_mean": B_mean, "B_frames": B_n,
        "C_mean": C_mean, "C_frames": C_n,
        "ens_AB": ens_AB,
        "ens_ABC": ens_ABC,
        "note": ""
    }

def best_threshold(y, s):
    thresholds = np.unique(np.round(s, 6))
    best_j, best_t = -1e9, 0.5
    for t in thresholds:
        pred = (s >= t).astype(int)
        tp = ((pred==1) & (y==1)).sum()
        tn = ((pred==0) & (y==0)).sum()
        fp = ((pred==1) & (y==0)).sum()
        fn = ((pred==0) & (y==1)).sum()
        tpr = tp / (tp+fn) if (tp+fn) else 0
        fpr = fp / (fp+tn) if (fp+tn) else 0
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t

    best_f1, best_f1_t = -1, 0.5
    for t in thresholds:
        pred = (s >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_f1_t = float(f1), float(t)
    return float(best_t), float(best_f1_t), float(best_f1)

def report(name, y, scores):
    auc = roc_auc_score(y, scores)
    ap  = average_precision_score(y, scores)
    pred = (scores >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    f1  = f1_score(y, pred, zero_division=0)
    print(f"\n{name}")
    print(f"  ROC-AUC: {auc:.3f}")
    print(f"  PR-AUC : {ap:.3f}")
    print(f"  Acc@0.5: {acc:.3f}")
    print(f"  F1 @0.5: {f1:.3f}")

def main(n_real=10, n_fake=10):
    real = pick_videos("data/celebdf/Celeb-real", n_real)
    fake = pick_videos("data/celebdf/Celeb-synthesis", n_fake)

    rows = []
    for i, vp in enumerate(fake, 1):
        tag = f"fake_{i:03d}"
        r = process_one(vp, tag)
        r["gt_fake"] = 1
        rows.append(r)

    for i, vp in enumerate(real, 1):
        tag = f"real_{i:03d}"
        r = process_one(vp, tag)
        r["gt_fake"] = 0
        rows.append(r)

    df = pd.DataFrame(rows)
    out_csv = OUT_EVAL_DIR / f"video_level_{n_real}real_{n_fake}fake_three_models.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    eval_df = df[df["note"] == ""].copy()
    y = eval_df["gt_fake"].values

    print("\n=== Metrics (video-level, mean aggregation) ===")
    report("Model A", y, eval_df["A_mean"].values)
    report("Model B", y, eval_df["B_mean"].values)
    report("Model C (CNN)", y, eval_df["C_mean"].values)
    report("Ensemble AB", y, eval_df["ens_AB"].values)
    report("Ensemble ABC", y, eval_df["ens_ABC"].values)

    # Agreement / alignment: keep your original A vs B story (so it stays comparable)
    _, tA_bestf1, _ = best_threshold(y, eval_df["A_mean"].values)
    _, tB_bestf1, _ = best_threshold(y, eval_df["B_mean"].values)
    _, tC_bestf1, _ = best_threshold(y, eval_df["C_mean"].values)
    _, tE_bestf1, _ = best_threshold(y, eval_df["ens_ABC"].values)

    A_pred = (eval_df["A_mean"].values >= tA_bestf1).astype(int)
    B_pred = (eval_df["B_mean"].values >= tB_bestf1).astype(int)
    C_pred = (eval_df["C_mean"].values >= tC_bestf1).astype(int)
    E_pred = (eval_df["ens_ABC"].values >= tE_bestf1).astype(int)

    agree_AB = (A_pred == B_pred)
    alignment_AB = 1.0 - np.abs(eval_df["A_mean"].values - eval_df["B_mean"].values)
    E_correct = (E_pred == y)

    eval_df["A_pred"] = A_pred
    eval_df["B_pred"] = B_pred
    eval_df["C_pred"] = C_pred
    eval_df["agree_AB"] = agree_AB
    eval_df["alignment_AB"] = alignment_AB
    eval_df["ENS_ABC_pred"] = E_pred
    eval_df["ENS_ABC_correct"] = E_correct

    out_csv2 = OUT_EVAL_DIR / f"video_level_{n_real}real_{n_fake}fake_three_models_with_agreement.csv"
    eval_df.to_csv(out_csv2, index=False)
    print(f"Saved: {out_csv2}")

    def acc(mask):
        if mask.sum() == 0: return float("nan")
        return float((E_correct[mask]).mean())

    print("\n=== Agreement → Reliability (A vs B, evaluated by ENS_ABC correctness) ===")
    print(f"Agree_AB n={agree_AB.sum()}  | ENS_ABC Acc: {acc(agree_AB):.3f}")
    print(f"Disagree_AB n={(~agree_AB).sum()} | ENS_ABC Acc: {acc(~agree_AB):.3f}")

    low = alignment_AB <= 0.5
    mid = (alignment_AB > 0.5) & (alignment_AB <= 0.8)
    high = alignment_AB > 0.8
    print("\n=== Alignment buckets (A vs B) ===")
    print(f"low (<=0.5)  n={low.sum()}  | ENS_ABC Acc: {acc(low):.3f}")
    print(f"mid (0.5-0.8) n={mid.sum()} | ENS_ABC Acc: {acc(mid):.3f}")
    print(f"high (>0.8)  n={high.sum()} | ENS_ABC Acc: {acc(high):.3f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_real", type=int, default=10)
    ap.add_argument("--n_fake", type=int, default=10)
    args = ap.parse_args()
    main(args.n_real, args.n_fake)