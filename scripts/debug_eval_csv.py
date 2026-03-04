import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from src.models_freq.fft_features import compute_fft_image, radial_frequency_features

image_path = "outputs/frames_faces/fake_001/face_00005.jpg"

fft_img = compute_fft_image(image_path)

# ---- Visualise FFT ----
plt.imshow(fft_img, cmap="gray")
plt.title("FFT Log Magnitude")
plt.colorbar()
plt.show()

# ---- Extract radial features ----
features = radial_frequency_features(fft_img)

print("Feature vector shape:", features.shape)
print("First 5 features:", features[:5])