import numpy as np
import cv2

def compute_laplacian_and_It(I_rhoPlus: np.ndarray,
                             I_rhoMinus: np.ndarray,
                             kernel_size: int = 21,
                             sigma: float = 1.0):
    """
    Junjie / paper-style:
      I_avg = 0.5*(I+ + I-), blurred
      I_t   = 0.5*(I+ - I-), blurred
      I_lap = Laplacian(I_avg)
    """
    I_rhoPlus = I_rhoPlus.astype(np.float32)
    I_rhoMinus = I_rhoMinus.astype(np.float32)

    I_avg = 0.5 * (I_rhoPlus + I_rhoMinus)
    I_t   = 0.5 * (I_rhoPlus - I_rhoMinus)

    # Gaussian smoothing (separable 느낌 내려고 그냥 cv2 GaussianBlur)
    k = (kernel_size, kernel_size)
    I_avg = cv2.GaussianBlur(I_avg, k, sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    I_t   = cv2.GaussianBlur(I_t,   k, sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)

    # Laplacian (ksize=3 고정이 무난)
    I_lap = cv2.Laplacian(I_avg, cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REFLECT)

    return I_lap, I_t
