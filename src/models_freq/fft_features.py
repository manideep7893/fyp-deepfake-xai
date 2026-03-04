import cv2
import numpy as np

def compute_fft_image(image_path, size=224):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (size, size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    log_magnitude = np.log1p(magnitude)

    return log_magnitude.astype(np.float32)


def radial_frequency_features(fft_img, num_bins=30):
    """
    Compute radial energy distribution features.
    """
    h, w = fft_img.shape
    center = (h // 2, w // 2)

    y, x = np.indices((h, w))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    r = r.astype(np.int32)
    max_radius = np.max(r)

    bin_size = max_radius // num_bins
    features = []

    for i in range(num_bins):
        mask = (r >= i * bin_size) & (r < (i + 1) * bin_size)
        if np.any(mask):
            features.append(np.mean(fft_img[mask]))
        else:
            features.append(0.0)

    return np.array(features, dtype=np.float32)