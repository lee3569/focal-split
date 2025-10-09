import cv2
import numpy as np

def load_pinhole_image(path: str) -> np.ndarray:
    """Luo Paper Eq. 3: P(x;s)"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0

def add_noise(image: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Gaussian noise(Section 3.3 noise analysis)"""
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 1)