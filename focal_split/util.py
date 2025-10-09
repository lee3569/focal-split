import cv2
import numpy as np

def align_images(I1: np.ndarray, I2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    crop = 20
    return I1[crop:-crop, crop:-crop], I2[crop:-crop, crop:-crop]