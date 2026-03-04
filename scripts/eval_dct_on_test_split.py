# scripts/eval_dct_on_test_split.py
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score


TEST_CSV = Path("outputs/eval/dct_test_frame_features.csv")  # make sure you generate this
MODEL_PATH = Path("outputs/models_freq/m4b_dct_logreg_split.joblib")

def main():
    bundle = load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_csv(TEST_CSV)
    y = df["gt_fake"].astype(int).values

    # keep only the same feature columns used in training
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in TEST CSV: {missing[:10]} ... ({len(missing)} total)")

    X = df[feature_cols].values.astype(np.float32)

    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)

    print("DCT Split Model (TEST)")
    print("ROC-AUC:", roc_auc_score(y, p))
    print("PR-AUC :", average_precision_score(y, p))
    print("Accuracy:", accuracy_score(y, pred))
    print("F1:", f1_score(y, pred))

if __name__ == "__main__":
    main()