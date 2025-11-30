import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

import constants as const
import imaging
import util
import oper
import depth

HEATMAP_RANGE = [
    [0.6, 1.6],
    [0.6, 1.6],
]

def plot_heatmap(Z_pred: np.ndarray, Z_true: np.ndarray, path: str):
    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_subplot(1, 1, 1)

    heatmap, xedges, yedges = np.histogram2d(
        Z_true.flatten(), Z_pred.flatten(),
        bins=40, range=HEATMAP_RANGE
    )
    heatmap = heatmap.T
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    heatmap = heatmap / np.where(
        heatmap.max(axis=0) == 0, 1, np.nansum(heatmap, axis=0)
    )

    im = ax.imshow(heatmap, extent=extent, origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    lim0 = min(extent[0], extent[2])
    lim1 = max(extent[1], extent[3])
    ax.plot([lim0, lim1], [lim0, lim1], "w-", linewidth=1)

    ax.set_xlabel("True Depth (m)")
    ax.set_ylabel("Estimated Depth (m)")
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def run_on_dataset(max_samples: int | None = 50,
                   crop: int = util.CROP_DEFAULT,
                   highpass: bool = False):
    data = util.load_dataset()

    if max_samples is not None:
        data = data[:max_samples]
    
    num_samples = len(data)
    print(f"Processing {num_samples} samples (Highpass Filter: {highpass})...")

    cols = 10
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(25, 3.0 * rows))
    
    if num_samples == 1: axes_flat = [axes]
    else: axes_flat = axes.flatten()

    all_Z_true = []
    all_Z_pred = []

    first_I1_save = False 

    for idx, sample in enumerate(data):
        try:
            I1_rgb, I2_rgb, Ztrue = util.dataset_sample_to_images_and_depth(sample)
            
            # RGB -> Mono
            I1 = imaging.to_gray(I1_rgb)
            I2 = imaging.to_gray(I2_rgb)

            if not first_I1_save:
                cv2.imwrite("debug_input_image.png", (I1 * 255).astype(np.uint8))
                print(f"Debug: Saved first input image to debug_input_image.png (Mean val: {I1.mean():.4f})")
                first_I1_save = True

            I1c, I2c = util.align_images(I1, I2, crop=crop)
            
            if highpass:
                I1c = imaging.highpass_filter(I1c)
                I2c = imaging.highpass_filter(I2c)

            lap_I, It = oper.compute_laplacian_and_It(I1c, I2c)
            depth_map = depth.calculate_depth_map(It, lap_I)

            h, w = depth_map.shape
            patch = depth_map[h//2-20:h//2+20, w//2-20:w//2+20]
            Z_hat = float(np.median(patch))
            
            if 0.0 < Z_hat < 5.0:
                all_Z_true.append(Ztrue)
                all_Z_pred.append(Z_hat)

            ax = axes_flat[idx]
            im = ax.imshow(depth_map, cmap="inferno", vmin=0.0, vmax=3.0)
            ax.set_title(f"#{idx}\nTrue: {Ztrue:.2f}m", fontsize=9)
            ax.axis('off')
            
            print(f"[{idx+1}/{num_samples}] True={Ztrue:.4f}m | Pred={Z_hat:.4f}m")

        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            axes_flat[idx].axis('off')
            continue

    for i in range(idx + 1, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout()
    plt.savefig("all_depth_maps_grid.png")
    plt.close()

      if len(all_Z_true) > 0:
        all_Z_true = np.array(all_Z_true, dtype=np.float32)
        all_Z_pred = np.array(all_Z_pred, dtype=np.float32)

        A = np.vstack([all_Z_pred, np.ones(len(all_Z_pred))]).T
        m, c = np.linalg.lstsq(A, all_Z_true, rcond=None)[0]

        print(f"\nCalibration Result :")
        print(f"Formula: True_Depth = {m:.4f} * Predicted_Depth + {c:.4f}")


        plt.figure(figsize=(6, 6))
        plt.scatter(all_Z_true, all_Z_pred, alpha=0.5, label='Raw Prediction')
        plt.plot(all_Z_true, all_Z_true, 'r--', label='Ideal (Target)')
        
        plt.plot(all_Z_true, m * all_Z_pred + c, 'g-', linewidth=2, label='Calibrated Fit')
        
        plt.xlabel("True Distance (m)")
        plt.ylabel("Predicted Distance (m)")
        plt.title(f"Depth Estimation Accuracy\n(Correlation: {np.corrcoef(all_Z_true, all_Z_pred)[0,1]:.4f})")
        plt.legend()
        plt.grid(True)
        plt.savefig("calibration_graph.png")

if __name__ == "__main__":
    run_on_dataset(max_samples=50)
