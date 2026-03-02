import pandas as pd

path = "outputs/eval/video_level_50real_50fake_three_models_with_agreement.csv"

df = pd.read_csv(path)

# ✅ Fix NaNs safely