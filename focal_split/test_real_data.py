import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# 사용자 모듈
import util
import imaging
import oper
import depth

def compute_confidence(I_gray: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(I_gray, cv2.CV_32F, ksize=3)
    conf = np.abs(lap)
    denom = conf.max() - conf.min()
    if denom == 0: denom = 1e-8
    conf = (conf - conf.min()) / denom
    return np.power(conf, 0.5)

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
    
    VMIN, VMAX = 0.0, 2.0 
    
    for idx, (f1_path, f2_path) in enumerate(pairs):
        print(f"[{idx+1}/{rows}] Processing {f1_path}...")
        
        # Load images
        I1_bgr = cv2.imread(f1_path)
        I2_bgr = cv2.imread(f2_path)
        I1_rgb = cv2.cvtColor(I1_bgr, cv2.COLOR_BGR2RGB)
        I2_rgb = cv2.cvtColor(I2_bgr, cv2.COLOR_BGR2RGB)
        
        # Grayscale conversion
        I1_gray = imaging.to_gray(I1_bgr)
        I2_gray = imaging.to_gray(I2_bgr)
        
        # SIFT Alignment (원본 크기로!)
        I1_aligned_gray, I2_aligned_gray, H = util.align_images(I1_gray, I2_gray, debug=(idx==0))
        
        # RGB warping + Consistent cropping
        if H is not None:
            h, w = I1_rgb.shape[:2]
            I2_aligned_rgb = cv2.warpPerspective(I2_rgb, H, (w, h))
            
            # Valid region 계산
            x1, y1, x2, y2 = util.get_valid_region_after_warp(I1_gray.shape, H, margin=30)
            
            # 세 이미지 모두 같은 영역으로 crop
            left_crop = 0
            right_crop = 65   # 오른쪽만 많이!
            top_crop = 0
            bottom_crop = 50

            h, w = I1_rgb.shape[:2]

            I1_rgb_crop = I1_rgb[top_crop:h-bottom_crop, left_crop:w-right_crop]
            I2_rgb_crop = I2_rgb[top_crop:h-bottom_crop, left_crop:w-right_crop]
            I2_aligned_crop = I2_aligned_rgb[top_crop:h-bottom_crop, left_crop:w-right_crop]
            I1_gray_crop = I1_aligned_gray[top_crop:h-bottom_crop, left_crop:w-right_crop]
            I2_gray_crop = I2_aligned_gray[top_crop:h-bottom_crop, left_crop:w-right_crop]


            
            if idx == 0:
                print(f"  [Valid region] Original: {I1_gray.shape}, Cropped: {I1_gray_crop.shape}")
                print(f"  [Crop coords] x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
        else:
            # Fallback: simple crop
            print(f"  [WARNING] SIFT failed, using crop fallback")
            crop = util.CROP_DEFAULT
            I1_rgb_crop = I1_rgb[crop:-crop, crop:-crop]
            I2_rgb_crop = I2_rgb[crop:-crop, crop:-crop]
            I2_aligned_crop = I2_rgb[crop:-crop, crop:-crop]
            I1_gray_crop = I1_gray[crop:-crop, crop:-crop]
            I2_gray_crop = I2_gray[crop:-crop, crop:-crop]
        
        # Depth estimation (cropped images 사용)
        I1_proc = imaging.highpass_filter(I1_gray_crop)
        I2_proc = imaging.highpass_filter(I2_gray_crop)
        
        lap_I, It = oper.compute_laplacian_and_It(I1_proc, I2_proc)
        depth_map = depth.calculate_depth_map(It, lap_I)
        
        conf_map = compute_confidence(I1_gray_crop)
        valid_mask = conf_map > 0.15
        
        vis_depth = depth_map.copy()
        vis_depth[~valid_mask] = np.nan
        
        kernel = np.ones((3,3), np.uint8)
        vis_depth_filled = np.nan_to_num(vis_depth)
        vis_depth_dilated = cv2.dilate(vis_depth_filled, kernel, iterations=1)
        mask_dilated = cv2.dilate(valid_mask.astype(np.uint8), kernel, iterations=1)
        vis_depth_dilated[mask_dilated == 0] = np.nan
        
        # Visualization
        ax_row = axes[idx] if rows > 1 else axes[0]
        
        # Column 1: I1 (cropped)
        ax_row[0].imshow(I1_rgb_crop)
        ax_row[0].set_title("Image 1 (Far)", fontsize=11)
        ax_row[0].axis('off')
        
        # Column 2: I2 (cropped to same region)
        ax_row[1].imshow(I2_rgb_crop)
        ax_row[1].set_title("Image 2 (Near)", fontsize=11)
        ax_row[1].axis('off')
        
        # Column 3: Aligned I2 (cropped)
        ax_row[2].imshow(I2_aligned_crop)
        ax_row[2].set_title("Aligned Image 2", fontsize=11, color='blue')
        ax_row[2].axis('off')
        
        # Column 4: Confidence Map
        ax_row[3].imshow(conf_map, cmap='gray')
        ax_row[3].set_title("Confidence Map", fontsize=11)
        ax_row[3].axis('off')
        
        # Column 5: Predicted Depth
        current_cmap = plt.cm.jet
        current_cmap.set_bad(color='black')
        
        im = ax_row[4].imshow(vis_depth_dilated, cmap=current_cmap, vmin=VMIN, vmax=VMAX)
        ax_row[4].set_title("Predicted Depth\n(with Mask)", fontsize=11, fontweight='bold')
        ax_row[4].axis('off')
        
        # Column 6: Colorbar
        cbar = plt.colorbar(im, cax=ax_row[5])
        cbar.set_label('Distance (m)', fontsize=10)
        cbar.set_ticks([VMIN, (VMIN+VMAX)/2, VMAX])
        cbar.set_ticklabels([f'{VMIN}m\n(Near)', f'{(VMIN+VMAX)/2}m', f'{VMAX}m\n(Far)'])
    
    plt.tight_layout()
    output_filename = "paper_final_result_consistent_crop.png"
    plt.savefig(output_filename, dpi=150)
    plt.close()
    print(f"\nSaved: {output_filename}")

if __name__ == "__main__":
    run_paper_visual_final()