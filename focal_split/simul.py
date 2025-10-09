import numpy as np
from scipy.signal import fftconvolve
def generate_defocused_image(pinhole_image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    # Luo Paper Eq. 2: I(x;s) = P(x;s) * k(x;s)
    defocused_image = fftconvolve(pinhole_image, psf, mode='same')
    return np.clip(defocused_image, 0, 1)