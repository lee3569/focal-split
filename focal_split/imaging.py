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
