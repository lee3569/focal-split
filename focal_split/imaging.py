import cv2
import numpy as np
from scipy.ndimage import uniform_filter

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() > 1.5: 
        img /= 255.0
    return img


def highpass_filter(img: np.ndarray, ksize: int = 31) -> np.ndarray:

    bias = uniform_filter(img, size=ksize, mode="reflect")
    I_clean = img - bias
    return I_clean

def aberration_correction(img: np.ndarray, K: int = 21) -> np.ndarray:
    """
    Paper Eq. 16: Remove non-uniform background lighting
    
    I_bck = I - BoxFilter(I)
    """
    box_filtered = cv2.boxFilter(img, -1, (K, K), normalize=True)
    I_bck = img - box_filtered
    return I_bck

def noise_attenuation(img: np.ndarray, sigma: float = 11.0) -> np.ndarray:
    """
    Paper Eq. 17: Gaussian smoothing to suppress sensor noise
    
    I_clean = GaussianBlur(I_bck, sigma)
    """
    I_clean = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return I_clean