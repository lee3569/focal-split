import cv2
import numpy as np

def calculate_laplacian(I1: np.ndarray, I2: np.ndarray) -> np.ndarray:
    # Paper Eq. 12: Laplacian_I = ∇²((I1(Rx+t) + I2(x)) / 2)
    avg_I = (I1 + I2) / 2.0

    # ensure grayscale
    if len(avg_I.shape) == 3:
        avg_I = cv2.cvtColor(avg_I, cv2.COLOR_BGR2GRAY)

    # ensure float32 type
    avg_I = avg_I.astype(np.float32)

    laplacian_I = cv2.Laplacian(avg_I, cv2.CV_32F, ksize=3)
    return laplacian_I


def calculate_Is(I1_aligned: np.ndarray, I2_aligned: np.ndarray) -> np.ndarray:
    # Paper Eq. 12: Is = I1(Rx+t) - I2(x)
    Is = I2_aligned - I1_aligned
    return Is