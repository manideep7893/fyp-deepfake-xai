# scripts/run_reliability_eval.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

from src.ensemble.reliability import compute_reliability_and_fusion

EVAL_CSV = Path("outputs/eval/video_level_50real_50fake_three_models_with_agreement.csv")
PREDS_D_ROOT = Path("outputs/preds_modelD")  # M4 FFT outputs

OUT_CSV = Path("outputs/eval/video_level_50real_50fake_four_models_with_reliability.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def get_mean_from_summary(summary_path: Path) -> tuple[float, int]:
    with open(summary_path, "r") as f:
        j = json.load(f)
    return float(j["aggregation"]["mean_p_fake"]), int(j.get("num_frames", j.get("num_valid", 0)))


def report(name: str, y: np.ndarray, scores: np.ndarray) -> None:
    # handle degenerate cases safely
    if len(np.unique(y)) < 2:
        print(f"\n{name}\n  Not enough class variety to compute AUC.")
        return

    auc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)
    pred = (scores >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, zero_division=0)

    print(f"\n{name}")
    print(f"  ROC-AUC: {auc:.3f}")
    print(f"  PR-AUC : {ap:.3f}")
    print(f"  Acc@0.5: {acc:.3f}")
    print(f"  F1 @0.5: {f1:.3f}")


def safe_float(x, default=0.5) -> float:
    try:
        if x is None:
            return float(default)
        x = float(x)
        if np.isnan(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def main():
    if not EVAL_CSV.exists():
        raise FileNotFoundError(f"Eval CSV not found: {EVAL_CSV}")

    df = pd.read_csv(EVAL_CSV)

    required_cols = {"tag", "gt_fake", "A_mean", "B_mean", "C_mean"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Eval CSV missing columns: {missing}")

    # Make sure agree/alignment exist (older csv might not)
    if "agree_AB" not in df.columns:
        df["agree_AB"] = False
    if "alignment_AB" not in df.columns:
        df["alignment_AB"] = 0.5

    # Load D_mean for each tag from outputs/preds_modelD/<tag>/video_summary.json
    D_means = []
    D_frames = []
    D_notes = []

    for tag in df["tag"].astype(str).tolist():
        summary = PREDS_D_ROOT / tag / "video_summary.json"
        if not summary.exists():
            # keep NaN; later we’ll treat as 0.5 in fusion
            D_means.append(np.nan)
            D_frames.append(0)
            D_notes.append("missing_D_summary")
            continue

        try:
            m, n = get_mean_from_summary(summary)
            D_means.append(m)
            D_frames.append(n)
            D_notes.append("")
        except Exception as e:
            D_means.append(np.nan)
            D_frames.append(0)
            D_notes.append(f"bad_D_summary:{type(e).__name__}")

    df["D_mean"] = D_means
    df["D_frames"] = D_frames
    df["D_note"] = D_notes

    # Compute ABCD fusion + scenario + reliability
    p_final = []
    pred_final = []
    reliability = []
    scenario = []

    for _, r in df.iterrows():
        A = safe_float(r.get("A_mean"), 0.5)
        B = safe_float(r.get("B_mean"), 0.5)
        C = safe_float(r.get("C_mean"), 0.5)
        D = safe_float(r.get("D_mean"), 0.5)

        align = safe_float(r.get("alignment_AB"), 0.5)
        agree = bool(r.get("agree_AB")) if r.get("agree_AB") is not None else False

        res = compute_reliability_and_fusion(
            A_mean=A,
            B_mean=B,
            C_mean=C,
            D_mean=D,
            alignment_AB=align,
            agree_AB=agree,
        )

        p_final.append(res.p_final)
        pred_final.append(res.pred_final)
        reliability.append(res.reliability)
        scenario.append(res.scenario)

    df["p_final_ABCD"] = p_final
    df["pred_final_ABCD"] = pred_final
    df["reliability_ABCD"] = reliability
    df["scenario_ABCD"] = scenario

    # Metrics
    y = df["gt_fake"].astype(int).values

    report("Model C only", y, df["C_mean"].astype(float).values)

    if "ens_ABC" in df.columns:
        report("Naive ens_ABC", y, df["ens_ABC"].astype(float).values)

    # If your previous reliability fusion outputs were stored in CSV, report them too:
    if "p_final" in df.columns:
        report("Reliability-weighted fusion (ABC from CSV p_final)", y, df["p_final"].astype(float).values)

    report("Scenario-aware reliability fusion (ABCD)", y, df["p_final_ABCD"].astype(float).values)

    # Save
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    # Scenario counts
    print("\nScenario counts (ABCD):")
    print(df["scenario_ABCD"].value_counts())


if __name__ == "__main__":
    main()