import numpy as np

def calculate_depth_map(Is: np.ndarray, laplacian_I: np.ndarray, a: float, b: float) -> np.ndarray:

    epsilon = 1e-8
    
    ratio = Is / (laplacian_I + epsilon)
    
    depth_map = a / (b + ratio + epsilon)

    depth_map[depth_map < 0] = 0 
    
    return depth_map