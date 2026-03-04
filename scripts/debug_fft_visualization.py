import matplotlib.pyplot as plt
from src.models_freq.fft_features import compute_fft_image

image_path = "outputs/frames_faces/fake_001/face_00005.jpg"

fft_img = compute_fft_image(image_path)

plt.imshow(fft_img, cmap="gray")
plt.title("FFT Log Magnitude")
plt.colorbar()
plt.show()