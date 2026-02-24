import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import util
import constants as const
from scipy import signal

def removeLowFreqInfo(img: np.ndarray, ksize: int = 21) -> np.ndarray:
    """Professor's high-pass filter"""
    kernel = np.ones((ksize, ksize), dtype=np.float32)
    kernel /= np.sum(kernel)
    bias = signal.fftconvolve(img, kernel, "same")
    return img - bias


def run_simulation_final_fix(max_samples=50, crop=50):
    """
    Process real-world dataset using professor's preprocessing method
    """
    data = util.load_dataset()
    if max_samples is not None:
        data = data[:max_samples]
    
    print(f"\n{'='*60}")
    print(f"Running Professor's Method on Real Data")
    print(f"Constants: A={const.A_CALIB:.4f}, B={const.B_CALIB:.4f}")
    print(f"{'='*60}\n")
    
    all_pixels_true = []
    all_pixels_pred = []
    
    WINDOW_SIZE = 21

    # Grid for visualization
    cols = 10
    rows = math.ceil(len(data) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(25, 3.0 * rows))
    axes_flat = axes.flatten() if len(data) > 1 else [axes]
    
    max_conf_found = 0.0

    for idx, sample in enumerate(data):
        try:
            # Get real data with ground truth depth
            I1_rgb, I2_rgb, Z_true = util.dataset_sample_to_images_and_depth(sample)
            
            # ===== Professor's preprocessing =====
            I1 = cv2.cvtColor(I1_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
            I2 = cv2.cvtColor(I2_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            # High-pass filter
            I1 = removeLowFreqInfo(I1, 21)
            I2 = removeLowFreqInfo(I2, 21)
            
            # Denoise
            I1 = cv2.GaussianBlur(I1, (11, 11), 0)
            I2 = cv2.GaussianBlur(I2, (11, 11), 0)
            
            # Alignment (SIFT for real data)
            I1_aligned, I2_aligned, H = util.align_images(I1, I2)
            
            if H is None:
                print(f"  [Warning] Sample {idx}: Alignment failed")
                if idx < len(axes_flat):
                    axes_flat[idx].axis('off')
                continue
            
            # Crop aligned images
            I1c = I1_aligned[crop:-crop, crop:-crop]
            I2c = I2_aligned[crop:-crop, crop:-crop]
            
            # ===== Professor's derivatives =====
            img_avg = (I1c + I2c) / 2
            Laplacian_I = img_avg - cv2.GaussianBlur(img_avg, (11, 11), 0)
            I_s_t = (I1c - I2c) / 2
            
            # ===== Professor's depth formula =====
            V = Laplacian_I
            W = const.A_CALIB * Laplacian_I + const.B_CALIB * I_s_t
            
            # Spatial aggregation (scipy convolution)
            kernel = np.ones((WINDOW_SIZE, 1))
            VW = signal.convolve2d(V * W, kernel, "same", "symm")
            VW = signal.convolve2d(VW, kernel.T, "same", "symm")
            W2 = signal.convolve2d(W**2, kernel, "same", "symm")
            W2 = signal.convolve2d(W2, kernel.T, "same", "symm")
            
            depth_map = np.divide(VW, W2 + 1e-10, where=(W2 != 0))
            
            # ===== Professor's confidence =====
            conf_map = I_s_t ** 2
            conf_map = (conf_map - conf_map.min()) / (conf_map.max() - conf_map.min() + 1e-8)
            
            # Track max confidence
            if conf_map.max() > max_conf_found:
                max_conf_found = conf_map.max()
            
            # Extract center patch (or full image)
            h, w = depth_map.shape
            margin = 30
            if h > 2*margin and w > 2*margin:
                patch_depth = depth_map[h//2-margin:h//2+margin, w//2-margin:w//2+margin]
                patch_conf = conf_map[h//2-margin:h//2+margin, w//2-margin:w//2+margin]
            else:
                patch_depth = depth_map
                patch_conf = conf_map
            
            flat_depth = patch_depth.flatten()
            flat_conf = patch_conf.flatten()
            
            # Filter: reasonable depth range + confidence threshold
            valid_mask = (
                (flat_depth > 0.0) & 
                (flat_depth < 2.5) & 
                (flat_conf > 0.80)  # Top 20%
            )
            
            valid_pixels = flat_depth[valid_mask]
            
            if len(valid_pixels) > 0:
                all_pixels_pred.extend(valid_pixels)
                all_pixels_true.extend([Z_true] * len(valid_pixels))
            
            # Visualization
            ax = axes_flat[idx]
            im = ax.imshow(depth_map, cmap="inferno", vmin=0.0, vmax=2.0)
            ax.set_title(f"#{idx}\nTrue: {Z_true:.3f}m", fontsize=9)
            ax.axis('off')
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1}/{len(data)} samples")

        except Exception as e:
            print(f"  [Error] Sample {idx}: {e}")
            if idx < len(axes_flat):
                axes_flat[idx].axis('off')
            continue
    
    # Hide unused subplots
    for i in range(idx + 1, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("all_depth_maps_grid.png", dpi=150)
    plt.close()
    print(f"\nSaved: all_depth_maps_grid.png")
    print(f"Max Confidence Found: {max_conf_found:.4f}")
    print(f"Total valid pixels: {len(all_pixels_true)}")

    # ===== Heatmap (professor's style) =====
    if len(all_pixels_true) > 100:
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        
        # Histogram
        HEATMAP_RANGE = [[0.0, 2.0], [0.0, 2.0]]
        heatmap, xedges, yedges = np.histogram2d(
            all_pixels_true,
            all_pixels_pred,
            bins=40,
            range=HEATMAP_RANGE
        )
        heatmap = heatmap.T
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        # Linear scale (professor's style)
        plot_heatmap = ax.imshow(heatmap, extent=extent, origin="lower")
        fig.colorbar(plot_heatmap, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_xlabel("True Depth (m)")
        ax.set_ylabel("Estimated Depth (m)")
        ax.set_title(f"Real Data Results\nA={const.A_CALIB:.2f}, B={const.B_CALIB:.2f}")
        ax.grid()
        
        fig.tight_layout()
        out_file = "final_heatmap_filtered.png"
        plt.savefig(out_file, dpi=150)
        plt.close()
        print(f"Saved: {out_file}\n")
    else:
        print("[Warning] Too few valid pixels for heatmap!\n")


if __name__ == "__main__":
    run_simulation_final_fix(max_samples=None)  # Use all samples