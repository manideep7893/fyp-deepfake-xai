import pandas as pd

path = "outputs/eval/video_level_50real_50fake_three_models_with_agreement.csv"

df = pd.read_csv(path)

print("COLUMNS:")
print(df.columns.tolist())

print("\nHEAD:")
print(df.head()) 