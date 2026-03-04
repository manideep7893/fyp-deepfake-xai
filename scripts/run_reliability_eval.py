from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

from src.ensemble.reliability import compute_reliability_and_fusion

EVAL_CSV = Path("outputs/eval/video_level_50real_50fake_three_models_with_agreement.csv")

# DCT predictions
PREDS_E_ROOT = Path("outputs/preds_modelE")

OUT_CSV = Path("outputs/eval/video_level_50real_50fake_four_models_with_reliability.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def get_mean_from_summary(summary_path: Path) -> tuple[float, int]:
    with open(summary_path, "r") as f:
        j = json.load(f)

    return float(j["aggregation"]["mean_p_fake"]), int(j.get("num_frames", j.get("num_valid", 0)))


def report(name: str, y: np.ndarray, scores: np.ndarray) -> None:

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

    # Ensure alignment columns exist
    if "agree_AB" not in df.columns:
        df["agree_AB"] = False

    if "alignment_AB" not in df.columns:
        df["alignment_AB"] = 0.5

    # -------- Load DCT predictions --------

    E_means = []
    E_frames = []
    E_notes = []

    for tag in df["tag"].astype(str).tolist():

        summary = PREDS_E_ROOT / tag / "video_summary.json"

        if not summary.exists():

            E_means.append(np.nan)
            E_frames.append(0)
            E_notes.append("missing_E_summary")

            continue

        try:

            m, n = get_mean_from_summary(summary)

            E_means.append(m)
            E_frames.append(n)
            E_notes.append("")

        except Exception as e:

            E_means.append(np.nan)
            E_frames.append(0)
            E_notes.append(f"bad_E_summary:{type(e).__name__}")

    df["E_mean"] = E_means
    df["E_frames"] = E_frames
    df["E_note"] = E_notes

    # -------- Reliability Fusion --------

    p_final = []
    pred_final = []
    reliability = []
    scenario = []

    for _, r in df.iterrows():

        A = safe_float(r.get("A_mean"), 0.5)
        B = safe_float(r.get("B_mean"), 0.5)
        C = safe_float(r.get("C_mean"), 0.5)
        E = safe_float(r.get("E_mean"), 0.5)

        align = safe_float(r.get("alignment_AB"), 0.5)

        agree = bool(r.get("agree_AB")) if r.get("agree_AB") is not None else False

        res = compute_reliability_and_fusion(
            A_mean=A,
            B_mean=B,
            C_mean=C,
            E_mean=E,
            alignment_AB=align,
            agree_AB=agree,
        )

        p_final.append(res.p_final)
        pred_final.append(res.pred_final)
        reliability.append(res.reliability)
        scenario.append(res.scenario)

    df["p_final_ABCE"] = p_final
    df["pred_final_ABCE"] = pred_final
    df["reliability_ABCE"] = reliability
    df["scenario_ABCE"] = scenario

    # -------- Metrics --------

    y = df["gt_fake"].astype(int).values

    report("Model C only", y, df["C_mean"].astype(float).values)

    if "ens_ABC" in df.columns:
        report("Naive ens_ABC", y, df["ens_ABC"].astype(float).values)

    report("Scenario-aware reliability fusion (ABCE)", y, df["p_final_ABCE"].astype(float).values)

    # -------- Save --------

    df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved: {OUT_CSV}")

    print("\nScenario counts (ABCE):")

    print(df["scenario_ABCE"].value_counts())


if __name__ == "__main__":
    main()