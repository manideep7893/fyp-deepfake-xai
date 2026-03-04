import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score

IN_CSV = "outputs/eval/dct_frame_features.csv"
OUT_DIR = Path("outputs/models_freq")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MODEL = OUT_DIR / "m4b_dct_logreg.joblib"

df = pd.read_csv(IN_CSV)

X = df.drop(columns=["label"]).values
y = df["label"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        n_jobs=None
    ))
])

pipe.fit(X_train, y_train)

p = pipe.predict_proba(X_test)[:, 1]
pred = (p >= 0.5).astype(int)

print("DCT Model (scaled logistic regression)")
print("ROC-AUC:", roc_auc_score(y_test, p))
print("PR-AUC :", average_precision_score(y_test, p))
print("Accuracy:", accuracy_score(y_test, pred))
print("F1:", f1_score(y_test, pred))

joblib.dump(pipe, OUT_MODEL)
print("Saved:", OUT_MODEL)