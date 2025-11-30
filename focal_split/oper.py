import cv2
import numpy as np

def compute_laplacian_and_It(I1: np.ndarray, I2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Luo Eq. 12:
      I_avg = (I1 + I2)/2
      ∇²I = Lap * I_avg
      I_t = (I2 - I1)/2
    """
    I_avg = (I1 + I2) * 0.5
    I_avg = I_avg.astype(np.float32)

    lap_I = cv2.Laplacian(I_avg, cv2.CV_32F, ksize=3)
    It = (I2 - I1) * 0.5

    return lap_I, It
