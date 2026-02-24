import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy import signal

# 사용자 모듈
import util
import constant as const

def removeLowFreqInfo(img: np.ndarray, ksize: int = 21) -> np.ndarray:
    """Professor's high-pass filter (aberration correction)"""
    kernel = np.ones((ksize, ksize), dtype=np.float32)
    kernel /= np.sum(kernel)
    bias = signal.fftconvolve(img, kernel, "same")
    return img - bias

def compute_confidence(It: np.ndarray) -> np.ndarray:
    """
    Professor's confidence for MASKING: |It|^2
    Returns raw It^2 values (not normalized)
    """
    conf = It ** 2
    return conf

def compute_confidence_display(It: np.ndarray) -> np.ndarray:
    """
    Professor's confidence VISUALIZATION (matching C++ code exactly)
    Range: 5e-7 to 1e-2, clamped, then inverted
    """
    I_s_t2 = It ** 2
    
    # Professor's exact normalization (from const)
    min_val = const.CONF_VIS_MIN  # 5e-7
    max_val = const.CONF_VIS_MAX  # 1e-2
    
    # Normalize
    I_s_t2_vis = (I_s_t2 - min_val) / (max_val - min_val) * 255.0
    
    # Clamp to [0, 255]
    I_s_t2_vis = np.clip(I_s_t2_vis, 0, 255).astype(np.uint8)
    
    # Invert (professor uses bitwise_not)
    I_s_t2_vis = 255 - I_s_t2_vis
    
    # Normalize to [0, 1] for matplotlib
    return I_s_t2_vis.astype(np.float32) / 255.0

def run_paper_visual_final(file_pattern="test*.png"):
    pairs = []
    for i in range(1, 7):
        f1 = f"test{i}.png"
        f2 = f"test{i}-1.png"
        if os.path.exists(f1) and os.path.exists(f2):
            pairs.append((f1, f2))
    
    num_pairs = len(pairs)
    if num_pairs == 0: 
        print("No test image pairs found!")
        return
    
    rows = num_pairs
    cols = 6
    
    fig, axes = plt.subplots(rows, cols, figsize=(22, 4 * rows), 
                             gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 0.05]})
    
    if rows == 1: axes = [axes]
    
    VMIN, VMAX = const.DEPTH_MIN, const.DEPTH_MAX
    
    # Create toggle directory
    toggle_dir = "toggle_comparison"
    os.makedirs(toggle_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing {num_pairs} image pairs")
    print(f"Using Professor's EXACT Implementation:")
    print(f"  A={const.A_CALIB:.4f}, B={const.B_CALIB:.4f}")
    print(f"  Depth Range: {const.DEPTH_MIN}m ~ {const.DEPTH_MAX}m")
    print(f"  Confidence Threshold: It^2 > 0.00001 (professor's C++ value)")
    print(f"  Confidence Display: 5e-7 ~ 1e-2 range, inverted")
    print(f"  Invalid pixels: WHITE")
    print(f"Toggle images will be saved in: {toggle_dir}/")
    print(f"{'='*60}\n")
    
    for idx, (f1_path, f2_path) in enumerate(pairs):
        print(f"[{idx+1}/{rows}] Processing {f1_path}...")
        
        # Load images
        I1_bgr = cv2.imread(f1_path)
        I2_bgr = cv2.imread(f2_path)
        I1_rgb = cv2.cvtColor(I1_bgr, cv2.COLOR_BGR2RGB)
        I2_rgb = cv2.cvtColor(I2_bgr, cv2.COLOR_BGR2RGB)
        
        # SIFT Alignment
        I1_gray = cv2.cvtColor(I1_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        I2_gray = cv2.cvtColor(I2_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        I1_aligned_gray, I2_aligned_gray, H = util.align_images(I1_gray, I2_gray, debug=(idx==0))
        
        # RGB warping + Consistent cropping
        if H is not None:
            h, w = I1_rgb.shape[:2]
            I2_aligned_rgb = cv2.warpPerspective(I2_rgb, H, (w, h))
            
            # Crop
            left_crop = 0
            right_crop = 65
            top_crop = 0
            bottom_crop = 50

            h, w = I1_rgb.shape[:2]

            I1_rgb_crop = I1_rgb[top_crop:h-bottom_crop, left_crop:w-right_crop]
            I2_rgb_crop = I2_rgb[top_crop:h-bottom_crop, left_crop:w-right_crop]
            I2_aligned_crop = I2_aligned_rgb[top_crop:h-bottom_crop, left_crop:w-right_crop]
            I1_gray_crop = I1_aligned_gray[top_crop:h-bottom_crop, left_crop:w-right_crop]
            I2_gray_crop = I2_aligned_gray[top_crop:h-bottom_crop, left_crop:w-right_crop]
            
            if idx == 0:
                print(f"  [Alignment] Original: {I1_gray.shape}, Cropped: {I1_gray_crop.shape}")
            
        else:
            # Fallback: simple crop
            print(f"  [WARNING] SIFT failed, using crop fallback")
            crop = 50
            I1_rgb_crop = I1_rgb[crop:-crop, crop:-crop]
            I2_rgb_crop = I2_rgb[crop:-crop, crop:-crop]
            I2_aligned_crop = I2_rgb[crop:-crop, crop:-crop]
            I1_gray_crop = I1_gray[crop:-crop, crop:-crop]
            I2_gray_crop = I2_gray[crop:-crop, crop:-crop]
        
        # Save toggle comparison images
        basename = os.path.splitext(os.path.basename(f1_path))[0]
        
        img1_path = os.path.join(toggle_dir, f"{basename}_1_img1.png")
        img2_path = os.path.join(toggle_dir, f"{basename}_2_aligned.png")
        
        cv2.imwrite(img1_path, cv2.cvtColor(I1_rgb_crop, cv2.COLOR_RGB2BGR))
        cv2.imwrite(img2_path, cv2.cvtColor(I2_aligned_crop, cv2.COLOR_RGB2BGR))
        
        print(f"  [Saved toggle] {basename}_1_img1.png & {basename}_2_aligned.png")
        
        # Professor's preprocessing
        I1_proc = removeLowFreqInfo(I1_gray_crop, const.HIGHPASS_SIZE)
        I2_proc = removeLowFreqInfo(I2_gray_crop, const.HIGHPASS_SIZE)
        
        I1_proc = cv2.GaussianBlur(I1_proc, (const.SMOOTH_SIZE, const.SMOOTH_SIZE), 0)
        I2_proc = cv2.GaussianBlur(I2_proc, (const.SMOOTH_SIZE, const.SMOOTH_SIZE), 0)
        
        # Professor's derivatives
        img_avg = (I1_proc + I2_proc) / 2
        Laplacian_I = img_avg - cv2.GaussianBlur(img_avg, (const.LAP_SIZE, const.LAP_SIZE), 0)
        I_s_t = (I1_proc - I2_proc) / 2
        
        # Professor's depth formula
        V = Laplacian_I
        W = const.A_CALIB * Laplacian_I + const.B_CALIB * I_s_t
        
        # Spatial aggregation
        kernel = np.ones((const.WINDOW_SIZE, 1))
        VW = signal.convolve2d(V * W, kernel, "same", "symm")
        VW = signal.convolve2d(VW, kernel.T, "same", "symm")
        W2 = signal.convolve2d(W**2, kernel, "same", "symm")
        W2 = signal.convolve2d(W2, kernel.T, "same", "symm")
        
        depth_map = np.divide(VW, W2 + 1e-10, where=(W2 != 0))
        
        # Confidence for MASKING (raw It^2)
        conf_mask = compute_confidence(I_s_t)
        
        # Confidence for DISPLAY (professor's exact visualization)
        conf_map_display = compute_confidence_display(I_s_t)
        
        # Professor's very low threshold (from C++ code)
        valid_mask = (
            (conf_mask > const.CONF_THRESHOLD) &  # It^2 > 0.00001
            (depth_map >= VMIN) & 
            (depth_map <= VMAX) &
            np.isfinite(depth_map)
        )
        
        # Clamp depth to range
        depth_clamped = np.clip(depth_map, VMIN, VMAX)
        
        # Normalize to 0-1 for colormap
        depth_norm = (depth_clamped - VMIN) / (VMAX - VMIN)
        
        # Apply colormap manually to set white for invalid
        vis_depth = plt.cm.jet(depth_norm)  # RGBA
        vis_depth = (vis_depth[:, :, :3] * 255).astype(np.uint8)  # RGB only
        
        # Set invalid pixels to WHITE
        vis_depth[~valid_mask] = [255, 255, 255]
        
        vis_depth_final = vis_depth
        
        # Visualization
        ax_row = axes[idx] if rows > 1 else axes[0]
        
        # Column 1: I1
        ax_row[0].imshow(I1_rgb_crop)
        ax_row[0].set_title("Image 1 (Far)", fontsize=11)
        ax_row[0].axis('off')
        
        # Column 2: I2
        ax_row[1].imshow(I2_rgb_crop)
        ax_row[1].set_title("Image 2 (Near)", fontsize=11)
        ax_row[1].axis('off')
        
        # Column 3: Aligned I2
        ax_row[2].imshow(I2_aligned_crop)
        ax_row[2].set_title("Aligned Image 2", fontsize=11, color='blue')
        ax_row[2].axis('off')
        
        # Column 4: Confidence Map (professor's exact visualization - already inverted!)
        ax_row[3].imshow(conf_map_display, cmap='gray', vmin=0, vmax=1)
        ax_row[3].set_title("Confidence Map", fontsize=11)
        ax_row[3].axis('off')
        
        # Column 5: Predicted Depth
        ax_row[4].imshow(vis_depth_final)
        ax_row[4].set_title("Predicted Depth\n(Calibrated)", fontsize=11, fontweight='bold')
        ax_row[4].axis('off')
        
        # Column 6: Colorbar
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=VMIN, vmax=VMAX)
        sm = cm.ScalarMappable(cmap='jet', norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, cax=ax_row[5])
        cbar.set_label('Distance (m)', fontsize=10)
        cbar.set_ticks([VMIN, (VMIN+VMAX)/2, VMAX])
        cbar.set_ticklabels([f'{VMIN}m\n(Near)', f'{(VMIN+VMAX)/2:.1f}m', f'{VMAX}m\n(Far)'])
    
    plt.tight_layout()
    output_filename = "paper_final_result_professor_exact.png"
    plt.savefig(output_filename, dpi=150)
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"COMPLETE! Professor's EXACT implementation:")
    print(f"  Main result: {output_filename}")
    print(f"  Toggle images: {toggle_dir}/")
    print(f"Settings:")
    print(f"  - A={const.A_CALIB}, B={const.B_CALIB}")
    print(f"  - Depth range: {const.DEPTH_MIN}~{const.DEPTH_MAX}m")
    print(f"  - Confidence: It^2 > 0.00001 (professor's C++ threshold)")
    print(f"  - Confidence display: 5e-7~1e-2 range, inverted (white bg)")
    print(f"  - Invalid pixels: WHITE")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_paper_visual_final()