import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import util
import imaging
import oper
import depth
import constants as const
from matplotlib.colors import LogNorm


def compute_confidence(I_gray: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(I_gray, cv2.CV_32F, ksize=3)
    conf = np.abs(lap)
    denom = conf.max() - conf.min()
    if denom == 0: denom = 1e-8
    conf = (conf - conf.min()) / denom
    return conf

def run_simulation_final_fix(max_samples=50, crop=util.CROP_DEFAULT):
    data = util.load_dataset()
    if max_samples is not None:
        data = data[:max_samples]
    
    print(f"Running Final Fix (Step 8 + Low Threshold)...")
    print(f"Constants: A={const.A_CALIB:.4f}, B={const.B_CALIB:.4f}")
    
    all_pixels_true = []
    all_pixels_pred = []
    
    WINDOW_SIZE = 21

    cols = 10
    rows = math.ceil(len(data) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(25, 3.0 * rows))
    axes_flat = axes.flatten() if len(data) > 1 else [axes]
    
    max_conf_found = 0.0

    for idx, sample in enumerate(data):
        try:
            I1_rgb, I2_rgb, Ztrue = util.dataset_sample_to_images_and_depth(sample)
            I1 = imaging.to_gray(I1_rgb)
            I2 = imaging.to_gray(I2_rgb)
            I1c, I2c = util.align_images(I1, I2, crop=crop)
            I1c = imaging.highpass_filter(I1c); I2c = imaging.highpass_filter(I2c)
            
            conf_map = compute_confidence(I1c)

            lap_I, It = oper.compute_laplacian_and_It(I1c, I2c)
            numerator = lap_I
            denominator = const.A_CALIB * lap_I + const.B_CALIB * It
            
            num_blur = cv2.boxFilter(numerator, -1, (WINDOW_SIZE, WINDOW_SIZE))
            den_blur = cv2.boxFilter(denominator, -1, (WINDOW_SIZE, WINDOW_SIZE))
            
            depth_map = np.divide(num_blur, den_blur + 1e-10)
            
            h, w = depth_map.shape
            margin = 30
            patch_depth = depth_map[h//2-margin:h//2+margin, w//2-margin:w//2+margin]
            patch_conf  = conf_map[h//2-margin:h//2+margin, w//2-margin:w//2+margin]
            
            flat_depth = patch_depth.flatten()
            flat_conf = patch_conf.flatten()
            
            if flat_conf.max() > max_conf_found:
                max_conf_found = flat_conf.max()
            

            valid_mask = (flat_depth > 0.0) & (flat_depth < 5.0) & (flat_conf > 0.05)
            
            valid_pixels = flat_depth[valid_mask]
            
            if len(valid_pixels) > 0:
                all_pixels_pred.extend(valid_pixels)
                all_pixels_true.extend([Ztrue] * len(valid_pixels))
            
            ax = axes_flat[idx]
            im = ax.imshow(depth_map, cmap="inferno", vmin=0.0, vmax=2.0)
            ax.set_title(f"#{idx}\nTrue: {Ztrue:.2f}m", fontsize=9)
            ax.axis('off')

        except Exception as e:
            if idx < len(axes_flat): axes_flat[idx].axis('off')
            continue
            
    for i in range(idx + 1, len(axes_flat)): axes_flat[i].axis('off')
    plt.tight_layout()
    plt.savefig("all_depth_maps_grid.png")
    plt.close()

    print(f"Max Confidence Found: {max_conf_found:.4f}")
#heatmap
    if len(all_pixels_true) > 0:
        plt.figure(figsize=(8, 7))
        
        counts, xedges, yedges, im = plt.hist2d(
            all_pixels_true,
            all_pixels_pred,
            bins=80,
            range=[[0, 0.7], [0, 0.7]],
            cmap='viridis',
            norm=LogNorm(vmin=1)
        )
        
        cbar = plt.colorbar(im)
        cbar.set_label('Pixel Count')
        
        plt.plot([0, 0.7], [0, 0.7], 'w--', linewidth=1.5, label='Ideal (y=x)')
        
        plt.xlabel("True Depth (m)")
        plt.ylabel("Estimated Depth (m)")
        plt.title(f"Final Heatmap (Step 8 + Filtered)\nA={const.A_CALIB:.2f}, B={const.B_CALIB:.2f}")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.2)
        
        out_file = "final_heatmap_filtered.png"
        plt.savefig(out_file)
        plt.close()
        print(f"[Success] 히트맵 저장 완료: {out_file}")
    else:
        print("[Error] 0.05로 낮췄는데도 데이터가 없습니다. 원본 이미지가 너무 어두운가 봅니다.")

if __name__ == "__main__":
    run_simulation_final_fix()