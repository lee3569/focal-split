import cv2
import numpy as np

def load_pinhole_image(path: str) -> np.ndarray:

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"no image: {path}")
    return img.astype(np.float32) / 255.0

def add_noise(image: np.ndarray, mean: float = 0.0, std: float = 0.01) -> np.ndarray:

    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image