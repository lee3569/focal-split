import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import util
import imaging
import oper

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
    """교수님의 filterResultByConfidence"""
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
            
            # Preprocessing (논문 방식)
            I1 = imaging.to_gray(I1_rgb)
            I2 = imaging.to_gray(I2_rgb)
            I1 = imaging.aberration_correction(I1, K=21)
            I2 = imaging.aberration_correction(I2, K=21)
            I1 = imaging.noise_attenuation(I1, sigma=11.0)
            I2 = imaging.noise_attenuation(I2, sigma=11.0)
            
            # Alignment
            I1_aligned, I2_aligned, H = util.align_images(I1, I2)
            if H is None:
                continue
            
            # Crop
            crop = 50
            I1_crop = I1_aligned[crop:-crop, crop:-crop]
            I2_crop = I2_aligned[crop:-crop, crop:-crop]
            
            # Derivatives (교수님 코드처럼 Gaussian smoothing)
            lap_I, It = oper.compute_laplacian_and_It(I1_crop, I2_crop)
            
            # FocalTrack depth estimation
            WINDOW = 21
            V = lap_I                    # alpha=1로 normalize
            W = A_test * lap_I + B_test * It
            
            num_blur = cv2.boxFilter(V * W, -1, (WINDOW, WINDOW))
            den_blur = cv2.boxFilter(W**2, -1, (WINDOW, WINDOW))
            Z_pred = np.divide(num_blur, den_blur + 1e-10)
            
            # Confidence
            confidence = compute_confidence(It)
            
            # Filter
            Z_filtered = filter_by_confidence(
                Z_pred, 
                confidence, 
                confidence_level=0.95,
                working_range=(0.3, 1.5)
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
        print("[Warning] Too few valid pixels!")
        return
    
    # Create heatmap (교수님 스타일)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    
    heatmap, xedges, yedges = np.histogram2d(
        all_Z_true,
        all_Z_pred,
        bins=50,
        range=[[0.3, 1.5], [0.3, 1.5]]
    )
    
    heatmap = heatmap.T
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    # Log scale colormap
    im = ax.imshow(
        heatmap, 
        extent=extent, 
        origin='lower',
        cmap='viridis',
        norm=LogNorm(vmin=max(1, heatmap[heatmap > 0].min()), vmax=heatmap.max()),
        aspect='auto'
    )
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pixel Count (log scale)', fontsize=12)
    
    # Ideal line
    ax.plot([0.3, 1.5], [0.3, 1.5], 'w--', linewidth=2, label='Ideal (y=x)')
    
    ax.set_xlabel('True Depth (m)', fontsize=14)
    ax.set_ylabel('Estimated Depth (m)', fontsize=14)
    ax.set_title(f'FocalTrack: A={A_test:.4f}, B={B_test:.4f}', fontsize=16)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Saved: {save_path}\n")

if __name__ == "__main__":
    # Test current values
    generate_heatmap_with_ab(A_test=1.08, B_test=0.82, max_samples=50)
    
    # Test other values
    # generate_heatmap_with_ab(A_test=1.20, B_test=0.80, max_samples=50, save_path="heatmap_A1.20_B0.80.png")