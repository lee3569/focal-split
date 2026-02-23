import numpy as np
import cv2
import constants as const

EPS = 1e-10

def calculate_depth_map(It: np.ndarray,
                        lap_I: np.ndarray,
                        window: int = 21,
                        A: float = None,
                        B: float = None) -> np.ndarray:
    """
    Junjie / paper-style aggregated estimator:
      V = lap
      W = A*lap + B*It
      Z = box(V*W) / (box(W^2) + eps)
    """
    if A is None: A = const.A_CALIB
    if B is None: B = const.B_CALIB

    lap_I = lap_I.astype(np.float32)
    It    = It.astype(np.float32)

    V = lap_I
    W = A * lap_I + B * It

    k = (window, window)
    VW = cv2.boxFilter(V * W, ddepth=-1, ksize=k, normalize=False, borderType=cv2.BORDER_REFLECT)
    W2 = cv2.boxFilter(W * W, ddepth=-1, ksize=k, normalize=False, borderType=cv2.BORDER_REFLECT)

    Z = VW / (W2 + EPS)
    return Z
