import numpy as npfrom scipy.signal import fftconvolve

def generate_defocused_image(pinhole_image: np.ndarry, pdf: np.ndarry) -> np.ndarry:

    defocused_image = fftconvolve(pinhole_image, psf, mode='same')
    return np.clip(defocused_image, 0, 1)