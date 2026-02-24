# calibration_heatmap.py 수정
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import util

def removeLowFreqInfo(img: np.ndarray, ksize: int = 21):
    """교수님 코드"""
    kernel = np.ones((ksize, ksize), dtype=np.float32)
    kernel /= np.sum(kernel)
    bias = signal.fftconvolve(img, kernel, "same")
    return img - bias

def generate_heatmap_with_ab(A_test, B_test, max_samples=None, save_path="Heatmap.png"):
    # ...
    
    for idx, sample in enumerate(data):
        I1_rgb, I2_rgb, Z_true = util.dataset_sample_to_images_and_depth(sample)
        
        # 교수님 preprocessing
        I1 = cv2.cvtColor(I1_rgb, cv2.COLOR_BGR2GRAY)
        I2 = cv2.cvtColor(I2_rgb, cv2.COLOR_BGR2GRAY)
        
        I1 = removeLowFreqInfo(I1, 21)
        I2 = removeLowFreqInfo(I2, 21)
        
        I1 = cv2.GaussianBlur(I1, (11, 11), 0)
        I2 = cv2.GaussianBlur(I2, (11, 11), 0)
        
        # 교수님 Laplacian
        img_avg = (I1 + I2) / 2
        Laplacian_I = img_avg - cv2.GaussianBlur(img_avg, (11, 11), 0)
        I_s_t = (I1 - I2) / 2
        
        # Crop (교수님 스타일)
        texture_position = np.array([[100, 480], [150, 530]])
        Laplacian_I = Laplacian_I[texture_position[0,0]:texture_position[0,1], 
                                    texture_position[1,0]:texture_position[1,1]]
        I_s_t = I_s_t[texture_position[0,0]:texture_position[0,1], 
                      texture_position[1,0]:texture_position[1,1]]
        
        # 교수님 depth formula
        V = Laplacian_I
        W = A_test * Laplacian_I + B_test * I_s_t
        
        kernel = np.ones((21, 1))
        VW = signal.convolve2d(V * W, kernel, "same", "symm")
        VW = signal.convolve2d(VW, kernel.T, "same", "symm")
        W2 = signal.convolve2d(W**2, kernel, "same", "symm")
        W2 = signal.convolve2d(W2, kernel.T, "same", "symm")
        Z_pred = np.divide(VW, W2 + 1e-10)
        
        