import numpy as np
import constants as const

def calculate_defocus_sigma(A: float, f: float, s: float, Z: float) -> float:
    """
    Luo Paper Eq. 5:
        sigma = A| 1/Z - rho | (s + A)
    where rho = 1/f (optical power).
    """
    if Z <= 0:
        Z = 1e-9
    rho = 1.0 / f
    sigma = A * abs(1.0 / Z - rho) * (s + A)
    return sigma


def create_psf(A: float, f: float, s: float, Z: float,
               pixel_pitch: float, size: int) -> np.ndarray:
    """
    Luo Paper Eq. 4:
        k(x; s) = (1 / sigma^2) * exp( - ||x||^2 / (2 sigma^2) )

    """
    sigma_m = calculate_defocus_sigma(A, f, s, Z)
    sigma_pixels = sigma_m / pixel_pitch

    if sigma_pixels < 0.1:

        psf = np.zeros((size, size), np.float32)
        psf[size // 2, size // 2] = 1.0
        return psf

    x = np.linspace(-size / 2, size / 2, size)
    xx, yy = np.meshgrid(x, x)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_pixels ** 2))
    psf /= psf.sum()
    return psf.astype(np.float32)


def get_optical_constants_ab(A: float, f: float, s: float) -> tuple[float, float]:
    """
    Luo Paper Eq. 11 theoretical a, b:
        a_th = -A^2
        b_th = -A^2 (1/f - 1/s)

    """
    a_th = -A ** 2
    b_th = -A ** 2 * (1.0 / f - 1.0 / s)
    return a_th, b_th


def get_calibrated_ab() -> tuple[float, float]:

    return const.A_CALIB, const.B_CALIB
