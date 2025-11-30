import numpy as np
import constants as const

def calculate_depth_map(It: np.ndarray, lap_I: np.ndarray,
                        A: float | None = None,
                        B: float | None = None,
                        eps: float = 1e-10) -> np.ndarray:
    """
    Eq. 11:
        Z(x) = ∇²I / (A ∇²I + B I_t)

    """
    if A is None:
        A = const.A_CALIB
    if B is None:
        B = const.B_CALIB

    denom = A * lap_I + B * It
    depth_map = lap_I / (denom + eps)

    depth_map[depth_map < 0] = 0
    return depth_map.astype(np.float32)
