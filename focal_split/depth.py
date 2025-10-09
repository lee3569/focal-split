import numpy as np

def calculate_depth_map(Is: np.ndarray, laplacian_I: np.ndarray, a: float, b: float) -> np.ndarray:
    # Luo Paper Eq. 11: Z(x) = a / (b + Ix(x;s)/∇²I(x;s))

    
    ratio = Is / (laplacian_I)
    depth_map = a / (b + ratio)
    depth_map[depth_map < 0] = 0
    return depth_map
