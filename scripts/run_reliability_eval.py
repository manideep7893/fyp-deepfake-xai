# scripts/run_reliability_eval.py

import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

from src.ensemble.reliability import compute_reliability_and_fusion

CSV_PATH = "outputs/eval/video_level_50real_50fake_three_models_with_agreement.csv"

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

def main():
    df = pd.read_csv(CSV_PATH)

    # Fix NaN note column if present
    if "note" in df.columns:
        df["note"] = df["note"].fillna("")

    # Only evaluate rows with gt + scores
    df = df[df["gt_fake"].notna()].copy()

    y = df["gt_fake"].astype(int).values

    # Baselines
    report("Model C only", y, df["C_mean"].astype(float).values)
    report("Naive ens_ABC", y, df["ens_ABC"].astype(float).values)

    # Reliability head
    out_p = []
    out_r = []
    out_s = []
    out_pred = []

    for _, row in df.iterrows():
        res = compute_reliability_and_fusion(
            A_mean=row.get("A_mean"),
            B_mean=row.get("B_mean"),
            C_mean=row.get("C_mean"),
            alignment_AB=row.get("alignment_AB"),
            agree_AB=row.get("agree_AB"),
        )
        out_p.append(res.p_final)
        out_r.append(res.reliability)
        out_s.append(res.scenario)
        out_pred.append(res.pred_final)

    df["p_final_reliable"] = out_p
    df["reliability"] = out_r
    df["scenario"] = out_s
    df["pred_final_reliable"] = out_pred

    report("Reliability-weighted fusion", y, df["p_final_reliable"].values)

    # Save for report tables
    out_path = "outputs/eval/video_level_50real_50fake_three_models_with_reliability.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Quick scenario counts
    print("\nScenario counts:")
    print(df["scenario"].value_counts())

if __name__ == "__main__":
    main()