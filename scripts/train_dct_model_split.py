# scripts/train_dct_model_split.py
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score


TRAIN_CSV = Path("outputs/eval/dct_train_frame_features.csv")
OUT_MODEL = Path("outputs/models_freq/m4b_dct_logreg_split.joblib")
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(TRAIN_CSV)

    # ---- label ----
    if "gt_fake" not in df.columns:
        raise ValueError("Expected column 'gt_fake' in CSV.")
    y = df["gt_fake"].astype(int).values

    # ---- feature matrix: keep ONLY numeric columns (drop strings like frame/path/tag) ----
    # This is the safest approach because even if you add more metadata columns later,
    # it won't break training.
    X_df = df.drop(columns=["gt_fake"], errors="ignore")
    X_df = X_df.select_dtypes(include=[np.number])

    if X_df.shape[1] == 0:
        raise ValueError("No numeric feature columns found after filtering. Check your CSV columns.")

    X = X_df.values.astype(np.float32)

    # ---- model ----
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="lbfgs"))
    ])

    model.fit(X, y)

    # quick sanity metrics on TRAIN (not final test; just to confirm training works)
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)

    print("DCT Split Model (TRAIN sanity)")
    print("ROC-AUC:", roc_auc_score(y, p))
    print("PR-AUC :", average_precision_score(y, p))
    print("Accuracy:", accuracy_score(y, pred))
    print("F1:", f1_score(y, pred))

    dump({"model": model, "feature_cols": list(X_df.columns)}, OUT_MODEL)
    print(f"Saved: {OUT_MODEL}")

if __name__ == "__main__":
    main()