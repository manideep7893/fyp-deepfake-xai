import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

CSV_PATH = "outputs/eval/fft_frame_features_radial.csv"
OUT_DIR = "outputs/models_freq"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

X = df.drop(columns=["path", "label"]).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression(max_iter=3000, C=1.0)
clf.fit(X_train, y_train)

probs = clf.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

auc = roc_auc_score(y_test, probs)
acc = accuracy_score(y_test, preds)

print("\nFrequency Model (M4) Results:")
print("ROC-AUC:", round(auc, 3))
print("Accuracy:", round(acc, 3))

joblib.dump({"scaler": scaler, "model": clf}, os.path.join(OUT_DIR, "m4_fft_logreg.joblib"))
print(f"Saved: {OUT_DIR}/m4_fft_logreg.joblib")