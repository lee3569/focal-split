# optics.py
import numpy as np

def calculate_defocus_sigma(A: float, f: float, s: float, Z: float) -> float:
    """
    Luo paper Eq. 5: sigma = A|1/Z - sigma|(s + A)
    rho = 1/f (optical power)
    """
    if Z <= 0:
        Z = 1e-9
    rho = 1 / f  # ρ: optical power
    sigma = A * abs(1/Z - rho) * (s + A)
    return sigma

def create_psf(A: float, f: float, s: float, Z: float, pixel_pitch: float, size: int = 31) -> np.ndarray:
    """
    Luo paper Eq. 4: k(x;s) = (1/sigma²)exp(-||x||²/2sigma²)
    Gaussian PSF creation
    """
    sigma_meters = calculate_defocus_sigma(A, f, s, Z)
    sigma_pixels = sigma_meters / pixel_pitch
    
    if sigma_pixels < 0.1:  # when almost in focus
        psf = np.zeros((size, size))
        psf[size//2, size//2] = 1.0
        return psf
    
    x = np.linspace(-size//2, size//2, size)
    xx, yy = np.meshgrid(x, x)
    
    # Eq. 4: Gaussian PSF
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma_pixels**2))
    psf /= np.sum(psf)
    return psf

def get_optical_constants_ab(A: float, f: float, s: float) -> tuple[float, float]:
    """
    Luo Paper Eq. 11 below:
    a = -A², b = -A²(1/f - 1/s)
    """
    a = -A**2
    b = -A**2 * (1/f - 1/s)
    return a, b