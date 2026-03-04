import cv2
import numpy as np


def compute_dct_features(image_path, size=224, block=16):
    """
    Compute DCT-based frequency features from an image.

    Steps:
    1. Load grayscale image
    2. Resize to fixed size
    3. Normalize + mean-center
    4. Compute 2D DCT
    5. Extract low-frequency block
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Resize
    img = cv2.resize(img, (size, size))

    # Normalize
    img = img.astype(np.float32) / 255.0

    # Mean-center (improves DCT stability)
    img = img - np.mean(img)

    # Compute DCT
    dct = cv2.dct(img)

    # Log magnitude stabilisation
    dct = np.log(np.abs(dct) + 1e-8)

    # Extract low-frequency block
    dct_block = dct[:block, :block]

    # Flatten to feature vector
    features = dct_block.flatten().astype(np.float32)

    return features