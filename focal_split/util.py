import cv2
import numpy as np

def align_images(I1: np.ndarray, I2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    h, w = I1.shape
    crop_size = 20
    I1_aligned = I1[crop_size:-crop_size, crop_size:-crop_size]
    I2_aligned = I2[crop_size:-crop_size, crop_size:-crop_size]
    return I1_aligned, I2_aligned