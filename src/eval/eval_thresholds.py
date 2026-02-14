import os, glob, json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix
)

import matplotlib.pyplot as plt


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def compute_at_threshold(y_true, y_score, thr: float):
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = safe_div(tp + tn, tp + tn + fp + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)  # TPR
    fpr = safe_div(fp, fp + tn)
    f1 = safe_div(2 * prec * rec, (prec + rec))

    return {
        "threshold": float(thr),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "accuracy": acc,
        "precision": prec,
        "recall_tpr": rec,
        "fpr": fpr,
        "f1": f1,
    }


def main():
    preds_root = Path("outputs/preds")
    out_dir = Path("outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(preds_root.glob("*/frame_predictions.csv"))
    if not csvs:
        raise FileNotFoundError(f"No frame_predictions.csv found under {preds_root}")

    rows = []
    for csv_path in csvs:
        tag = csv_path.parent.name  # e.g., fake_001 / real_002

        if tag.lower().startswith("fake_"):
            y = 1
        elif tag.lower().startswith("real_"):
            y = 0
        else:
            # skip anything else
            continue

        df = pd.read_csv(csv_path)
        if "p_fake" not in df.columns:
            raise ValueError(f"{csv_path} missing column p_fake")

        # keep only required columns
        for _, r in df.iterrows():
            rows.append({
                "tag": tag,
                "frame_path": r["frame_path"],
                "y_true": y,
                "p_fake": float(r["p_fake"]),
            })

    all_df = pd.DataFrame(rows)
    y_true = all_df["y_true"].values.astype(int)
    y_score = all_df["p_fake"].values.astype(float)

    # --- Global metrics ---
    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))

    fpr, tpr, roc_thr = roc_curve(y_true, y_score)
    precision, recall, pr_thr = precision_recall_curve(y_true, y_score)

    # --- Optimal threshold: Youden’s J (TPR - FPR) ---
    j = tpr - fpr
    j_best_idx = int(np.argmax(j))
    thr_youden = float(roc_thr[j_best_idx])

    # --- Best F1 threshold (use PR thresholds; align sizes) ---
    # precision_recall_curve returns thr array of length (len(precision)-1)
    f1s = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    f1_best_idx = int(np.argmax(f1s))
    thr_f1 = float(pr_thr[f1_best_idx])

    # Evaluate at default 0.5 + at optimal thresholds
    metrics = {
        "num_frames_total": int(len(all_df)),
        "num_real_frames": int((all_df["y_true"] == 0).sum()),
        "num_fake_frames": int((all_df["y_true"] == 1).sum()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "thresholds": {
            "default_0.5": compute_at_threshold(y_true, y_score, 0.5),
            "youden_j": compute_at_threshold(y_true, y_score, thr_youden),
            "best_f1": compute_at_threshold(y_true, y_score, thr_f1),
            "youden_j_value": float(j[j_best_idx]),
            "youden_j_threshold": thr_youden,
            "best_f1_value": float(f1s[f1_best_idx]),
            "best_f1_threshold": thr_f1,
        }
    }

    # --- Save JSON ---
    (out_dir / "frame_level_metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- Plot ROC ---
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={roc_auc:.3f})")
    plt.grid(True)
    plt.savefig(out_dir / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Plot PR ---
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP={pr_auc:.3f})")
    plt.grid(True)
    plt.savefig(out_dir / "pr_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Save combined CSV (handy for later analysis) ---
    all_df.to_csv(out_dir / "all_frames_combined.csv", index=False)

    print("\n✅ DONE")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved:\n- {out_dir/'frame_level_metrics.json'}\n- {out_dir/'roc_curve.png'}\n- {out_dir/'pr_curve.png'}\n- {out_dir/'all_frames_combined.csv'}")


if __name__ == "__main__":
    main()