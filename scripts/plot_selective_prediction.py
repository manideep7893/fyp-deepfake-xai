import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/eval/selective_prediction_summary.csv")

# Verified only
v = df[df["bucket"]=="VERIFIED"].copy()
v = v[v["thr"] >= 0]

# Coverage vs Accuracy
plt.figure()
plt.plot(v["coverage"], v["acc"], marker="o")
plt.xlabel("Coverage (fraction verified)")
plt.ylabel("Accuracy on verified subset")
plt.title("Selective Prediction: Coverage vs Accuracy")
plt.grid(True)
plt.savefig("outputs/eval/coverage_vs_accuracy.png", dpi=200)

# Coverage vs F1
plt.figure()
plt.plot(v["coverage"], v["f1"], marker="o")
plt.xlabel("Coverage (fraction verified)")
plt.ylabel("F1 on verified subset")
plt.title("Selective Prediction: Coverage vs F1")
plt.grid(True)
plt.savefig("outputs/eval/coverage_vs_f1.png", dpi=200)

print("Saved plots to outputs/eval/")