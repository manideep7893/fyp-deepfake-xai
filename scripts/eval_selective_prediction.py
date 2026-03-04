import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

def safe_auc(y, s):
    # AUC requires both classes present
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, s)

def safe_ap(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return average_precision_score(y, s)

def metrics(y, p, score):
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(safe_auc(y, score)),
        "pr_auc": float(safe_ap(y, score)),
    }

def main(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)

    # --- normalize columns (handle your NaNs) ---
    # your 'note' is NaN for normal rows; treat as empty
    if "note" in df.columns:
        df["note"] = df["note"].fillna("")

    # pick the right columns based on your latest file
    # expected in four-model reliability CSV:
    # gt_fake, p_final or p_final_ABCD, reliability, scenario_ABCD
    # If you used different names, adjust here once.
    # I'll try common ones:
    score_col = None
    for c in ["p_final", "p_final_ABCD", "p_fused", "p_fused_ABCD"]:
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        # fallback: your script may store ENS prob in ens_ABC / ens_ABCD
        for c in ["ens_ABCD", "ens_ABC", "ens_ABC_reliability"]:
            if c in df.columns:
                score_col = c
                break
    if score_col is None:
        raise ValueError(f"Could not find fused probability column in CSV. Columns: {list(df.columns)}")

    rel_col = None
    for c in ["reliability", "reliability_ABCD", "reliability_score"]:
        if c in df.columns:
            rel_col = c
            break
    if rel_col is None:
        raise ValueError(f"Could not find reliability column in CSV. Columns: {list(df.columns)}")

    y = df["gt_fake"].values.astype(int)
    p = df[score_col].values.astype(float)
    r = df[rel_col].values.astype(float)

    thresholds = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows = []

    # overall
    rows.append({
        "bucket": "ALL",
        "thr": -1,
        "coverage": 1.0,
        **metrics(y, p, p)
    })

    # selective buckets
    for thr in thresholds:
        mask = r >= thr
        cov = float(mask.mean())
        if mask.sum() == 0:
            rows.append({"bucket": "VERIFIED", "thr": thr, "coverage": cov, "n": 0, "acc": np.nan, "f1": np.nan, "roc_auc": np.nan, "pr_auc": np.nan})
            continue
        rows.append({
            "bucket": "VERIFIED",
            "thr": thr,
            "coverage": cov,
            **metrics(y[mask], p[mask], p[mask])
        })

        mask2 = ~mask
        cov2 = float(mask2.mean())
        if mask2.sum() == 0:
            rows.append({"bucket": "INCONCLUSIVE", "thr": thr, "coverage": cov2, "n": 0, "acc": np.nan, "f1": np.nan, "roc_auc": np.nan, "pr_auc": np.nan})
        else:
            rows.append({
                "bucket": "INCONCLUSIVE",
                "thr": thr,
                "coverage": cov2,
                **metrics(y[mask2], p[mask2], p[mask2])
            })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(pd.DataFrame(rows))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/eval/video_level_50real_50fake_four_models_with_reliability.csv")
    ap.add_argument("--out", default="outputs/eval/selective_prediction_summary.csv")
    args = ap.parse_args()
    main(args.csv, args.out)