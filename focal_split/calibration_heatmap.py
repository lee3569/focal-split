import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import util
import imaging
import oper

def removeLowFreqInfo(img: np.ndarray, ksize: int = 21) -> np.ndarray:
    """
    Professor's high-pass filter: subtract box-filtered version
    """
    kernel = np.ones((ksize, ksize), dtype=np.float32)
    kernel /= np.sum(kernel)
    bias = signal.fftconvolve(img, kernel, "same")
    return img - bias

def compute_confidence(I_t: np.ndarray) -> np.ndarray:
    """Confidence based on |I_t|^2"""
    conf = I_t ** 2
    conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
    return conf_norm

def filter_by_confidence(
    Z_array: np.ndarray,
    Z_confidence: np.ndarray,
    confidence_level: float = 0.95,
    working_range: tuple = (0.3, 1.5)
) -> np.ndarray:
    """Filter depth estimates by confidence"""
    Z_flat = Z_array.flatten()
    C_flat = Z_confidence.flatten()
    
    valid_mask = np.isfinite(Z_flat) & np.isfinite(C_flat)
    Z_flat = Z_flat[valid_mask]
    C_flat = C_flat[valid_mask]
    
    if len(C_flat) == 0:
        return np.array([])
    
    sorted_conf = np.sort(C_flat)
    conf_idx = int((1 - confidence_level) * len(sorted_conf))
    conf_threshold = sorted_conf[conf_idx] if conf_idx < len(sorted_conf) else sorted_conf[0]
    
    result = np.where(
        (C_flat > conf_threshold) & 
        (Z_flat >= working_range[0]) & 
        (Z_flat <= working_range[1]),
        Z_flat,
        np.nan
    )
    
    return result[~np.isnan(result)]

def generate_heatmap_with_ab(A_test, B_test, max_samples=50, save_path="heatmap.png"):
    """
    Generate FocalTrack-style heatmap: True Depth vs Estimated Depth
    
    Calibration dataset is already aligned, so we skip SIFT alignment.
    """
    print(f"\n{'='*60}")
    print(f"Testing A={A_test:.4f}, B={B_test:.4f}")
    print(f"{'='*60}")
    
    data = util.load_dataset()
    if max_samples:
        data = data[:max_samples]
    
    all_Z_true = []
    all_Z_pred = []
    
    for idx, sample in enumerate(data):
        try:
            I1_rgb, I2_rgb, Z_true = util.dataset_sample_to_images_and_depth(sample)
            
            # Professor's preprocessing
            I1 = cv2.cvtColor(I1_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
            I2 = cv2.cvtColor(I2_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            # High-pass filter (remove low frequency)
            I1 = removeLowFreqInfo(I1, 21)
            I2 = removeLowFreqInfo(I2, 21)
            
            # Denoise
            I1 = cv2.GaussianBlur(I1, (11, 11), 0)
            I2 = cv2.GaussianBlur(I2, (11, 11), 0)
            
            # Simple crop (remove edges)
            crop = 50
            I1_crop = I1[crop:-crop, crop:-crop]
            I2_crop = I2[crop:-crop, crop:-crop]
            
            # Professor's Laplacian (difference of Gaussians)
            img_avg = (I1_crop + I2_crop) / 2
            Laplacian_I = img_avg - cv2.GaussianBlur(img_avg, (11, 11), 0)
            I_s_t = (I1_crop - I2_crop) / 2
            
            # Professor's depth formula
            V = Laplacian_I
            W = A_test * Laplacian_I + B_test * I_s_t
            
            # Spatial aggregation (professor's method)
            kernelSize = 21
            kernel = np.ones((kernelSize, 1))
            VW = signal.convolve2d(V * W, kernel, "same", "symm")
            VW = signal.convolve2d(VW, kernel.T, "same", "symm")
            W2 = signal.convolve2d(W**2, kernel, "same", "symm")
            W2 = signal.convolve2d(W2, kernel.T, "same", "symm")
            Z_pred = np.divide(VW, W2 + 1e-10, where=(W2 != 0))
            
            # Confidence
            confidence = compute_confidence(I_s_t)
            
            # Filter by confidence (professor uses 0.95)
            Z_filtered = filter_by_confidence(
                Z_pred, 
                confidence, 
                confidence_level=0.95,
                working_range=(0.0, 2.5)
            )
            
            if len(Z_filtered) > 0:
                all_Z_true.extend([Z_true] * len(Z_filtered))
                all_Z_pred.extend(Z_filtered.tolist())
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1}/{len(data)} samples")
                
        except Exception as e:
            print(f"  [Error] Sample {idx}: {e}")
            continue
    
    print(f"\nTotal valid pixels: {len(all_Z_true)}")
    
    if len(all_Z_true) < 100:
        print("[Warning] Too few valid pixels for heatmap!")
        return
    
    # ===== Use professor's plotting code exactly =====
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    # Histogram (professor uses 40 bins)
    HEATMAP_RANGE = [[0.0, 2.5], [0.0, 2.5]]
    heatmap, xedges, yedges = np.histogram2d(
        all_Z_true,
        all_Z_pred,
        bins=40,
        range=HEATMAP_RANGE
    )
    heatmap = heatmap.T
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    # Linear scale (NO LogNorm like professor's code!)
    plot_heatmap = ax.imshow(heatmap, extent=extent, origin="lower")
    fig.colorbar(plot_heatmap, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set_xlabel("True Depth (m)")
    ax.set_ylabel("Estimated Depth (m)")
    ax.set_title(f"FocalTrack: A={A_test:.4f}, B={B_test:.4f}")
    ax.grid()
    
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
    print(f"Saved: {save_path}\n")

if __name__ == "__main__":
    # Test professor's values and nearby
    test_values = [
        (1.23, 0.19),  # Professor's exact values
        (1.20, 0.20),
        (1.15, 0.25),
        (1.10, 0.30),
        (1.00, 0.40),
    ]
    
    for A, B in test_values:
        save_path = f"heatmap_A{A:.2f}_B{B:.2f}.png"
        generate_heatmap_with_ab(A_test=A, B_test=B, max_samples=None, save_path=save_path)