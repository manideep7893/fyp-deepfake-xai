import cv2
import numpy as np


def compute_dct_features(image_path, size=224, block=16, drop_dc=True):
    """
    Compute DCT-based frequency features from an image.

    Steps:
    1. Load grayscale image
    2. Resize to fixed size
    3. Normalize
    4. Compute 2D DCT
    5. Extract low-frequency block
    6. Flatten to feature vector
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0

    dct = cv2.dct(img)

    block = min(block, size)
    dct_block = dct[:block, :block].copy()

    if drop_dc:
        dct_block[0, 0] = 0.0

    features = dct_block.flatten().astype(np.float32)

    return features


def dct_feature_vector(image_path, size=224, block=16, drop_dc=True):
    """
    Compatibility wrapper used by dataset-generation scripts.
    """
    return compute_dct_features(
        image_path=image_path,
        size=size,
        block=block,
        drop_dc=drop_dc,
    )