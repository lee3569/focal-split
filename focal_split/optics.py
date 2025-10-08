import numpy as np

def calculate_defocus_sigma(A: float, f: float, s: float, Z: float) -> float:
    if Z <= 0: Z = 1e-9
    s_focus = 1 / (1/f - 1/Z)
    sigma = A * abs(s - s_focus) / s_focus
    return sigma

def create_psf(A: float, f: float, s: float, Z: float, pixel_pitch: float, size: int = 31) -> np.ndarray:
    
    sigma_meters = calculate_defocus_sigma(A, f, s, Z)
    sigma_pixels = sigma_meters / pixel_pitch
    if sigma_pixels < 0.1:
        psf = np.zeros((size, size))
        psf[size//2, size//2] = 1.0
        return psf
    
    x = np.linspace(-size//2, size//2, size)
    xx, yy = np.meshgrid(x, x)

    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma_pixels**2))
    psf /= np.sum(psf)
    return psf

def get_optical_constants_ab(A: float, f: float, s: float) -> tuple[float, float]:
    a = -A**2
    b = -A**2 * (1/f - 1/s)
    return a, b

