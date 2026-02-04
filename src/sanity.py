import sys
import torch
import cv2
import numpy as np

print("Python exe:", sys.executable)
print("Torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("Device:", "mps" if torch.backends.mps.is_available() else "cpu")
print("OpenCV:", cv2.__version__)
print("Numpy:", np.__version__)