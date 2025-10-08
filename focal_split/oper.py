import cv2
import numpy as np

def calculate_laplacian(I1: np.ndarray, I2: np.ndarray) -> np.ndarray:

    avg_I = (I1 + I2) / 2.0
    laplacian_I = cv2.Laplacian(avg_I, cv2.CV_32F, ksize=3)
    return laplacian_I

def calculate_Is(I1_aligned: np.ndarray, I2_aligned: np.ndarray) -> np.ndarray:

    Is = I2_aligned - I1_aligned
    return Is