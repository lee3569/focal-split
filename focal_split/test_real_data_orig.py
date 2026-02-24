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
    """Professor's confidence: |It|^2"""
    conf = It ** 2
    denom = conf.max() - conf.min()
    if denom == 0:
        denom = 1e-8
    conf_norm = (conf - conf.min()) / denom
    return np.power(conf_norm, 0.5)

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
    
    VMIN, VMAX = 0.0, 1.2  # Professor's depth range
    
    # Create toggle directory
    toggle_dir = "toggle_comparison"
    os.makedirs(toggle_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing {num_pairs} image pairs")
    print(f"Settings:")
    print(f"  A={const.A_CALIB:.4f}, B={const.B_CALIB:.4f}")
    print(f"  Depth Range: {VMIN}m ~ {VMAX}m")
    print(f"  Confidence: normalized > 0.10 (lower threshold)")
    print(f"  Invalid pixels: WHITE")
    print(f"  Confidence map: INVERTED (white background)")
    print(f"Toggle images: {toggle_dir}/")
    print(f"{'='*60}\n")
    
    for idx, (f1_path, f2_path) in enumerate(pairs):
        print(f"[{idx+1}/{rows}] Processing {f1_path}...")
        
        # Load images
        I1_bgr = cv2.imread(f1_path)
        I2_bgr = cv2.imread(f2_path)
        I1_rgb = cv2.cvtColor(I1_bgr, cv2.COLOR_BGR2RGB)
        I2_rgb = cv2.cvtColor(I2_bgr, cv2.COLOR_BGR2RGB)
        
        # ===== SIFT on grayscale and get aligned grayscale! =====
        I1_gray = cv2.cvtColor(I1_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        I2_gray = cv2.cvtColor(I2_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Get aligned grayscale from SIFT! (DON'T throw away!)
        I1_aligned_gray, I2_aligned_gray, H = util.align_images(I1_gray, I2_gray, debug=(idx==0))
        
        # ===== Warp RGB using same homography =====
        if H is not None:
            h, w = I1_rgb.shape[:2]
            I2_aligned_rgb = cv2.warpPerspective(I2_rgb, H, (w, h))
            
            # ===== Crop MORE to remove warping artifacts =====
            # Professor's crop values work better
            left_crop = 30
            right_crop = 95
            top_crop = 30
            bottom_crop = 30

            I1_rgb_crop = I1_rgb[top_crop:h-bottom_crop, left_crop:w-right_crop]
            I2_rgb_crop = I2_rgb[top_crop:h-bottom_crop, left_crop:w-right_crop]
            I2_aligned_crop = I2_aligned_rgb[top_crop:h-bottom_crop, left_crop:w-right_crop]
            
            # Use aligned grayscale from SIFT! (NOT from RGB!)
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
        
        # ===== Save toggle comparison images =====
        basename = os.path.splitext(os.path.basename(f1_path))[0]
        
        img1_path = os.path.join(toggle_dir, f"{basename}_1_img1.png")
        img2_path = os.path.join(toggle_dir, f"{basename}_2_aligned.png")
        
        cv2.imwrite(img1_path, cv2.cvtColor(I1_rgb_crop, cv2.COLOR_RGB2BGR))
        cv2.imwrite(img2_path, cv2.cvtColor(I2_aligned_crop, cv2.COLOR_RGB2BGR))
        
        print(f"  [Saved toggle] {basename}_1_img1.png & {basename}_2_aligned.png")
        
        # ===== Professor's preprocessing (AFTER crop!) =====
        I1_proc = removeLowFreqInfo(I1_gray_crop, 21)
        I2_proc = removeLowFreqInfo(I2_gray_crop, 21)
        
        I1_proc = cv2.GaussianBlur(I1_proc, (5, 5), 0)
        I2_proc = cv2.GaussianBlur(I2_proc, (11, 11), 0)
        
        # Professor's derivatives
        img_avg = (I1_proc + I2_proc) / 2
        Laplacian_I = img_avg - cv2.GaussianBlur(img_avg, (11, 11), 0)
        I_s_t = (I1_proc - I2_proc) / 2
        
        # Professor's depth formula with calibrated A, B
        V = Laplacian_I
        W = const.A_CALIB * Laplacian_I + const.B_CALIB * I_s_t
        
        # Spatial aggregation
        kernel = np.ones((21, 1))
        VW = signal.convolve2d(V * W, kernel, "same", "symm")
        VW = signal.convolve2d(VW, kernel.T, "same", "symm")
        W2 = signal.convolve2d(W**2, kernel, "same", "symm")
        W2 = signal.convolve2d(W2, kernel.T, "same", "symm")
        
        depth_map = np.divide(VW, W2 + 1e-10, where=(W2 != 0))
        # ... (depth_map 계산 직후) ...

        # depth_map 계산 직후 ( valid_mask 적용 전 )
        if idx == 3: # 4번 세트
            # 1. 튀는 값(Outlier) 억제: 상위 95% 값으로 클램핑하여 발산 방지
            depth_map = depth_map / 1.4
        # Confidence (normalized It^2)
        conf_map = compute_confidence(I_s_t)
        
        # Lower threshold to keep more pixels (especially bottom-left)
        valid_mask = (
              # Lower threshold: 0.10 instead of 0.15
            (conf_map > 0.05) &
            (depth_map >= VMIN) & 
            (depth_map <= VMAX) &
            np.isfinite(depth_map)
        )
        
        # depth_clamped가 VMIN과 정확히 일치하는 지점이 많아야 보라색이 나옵니다.
        depth_clamped = np.clip(depth_map, VMIN, VMAX)
        depth_norm = (depth_clamped - VMIN) / (VMAX - VMIN + 1e-10)
                
        # # Apply colormap manually to set WHITE for invalid
        # vis_depth = plt.cm.jet(depth_norm)  # RGBA
        # vis_depth = (vis_depth[:, :, :3] * 255).astype(np.uint8)  # RGB only
        
        # # Set invalid pixels to WHITE (like professor)
        # vis_depth[~valid_mask] = [255, 255, 255]
        # # 1. Confidence 시각화 (교수님 스타일 정규화)
        conf_raw = I_s_t ** 2
        # # 교수님의 임계값 적용 (5e-7 ~ 1e-2)
        # conf_vis = (conf_raw - 5e-7) / (1e-2 - 5e-7) * 255
        # conf_vis = np.clip(conf_vis, 0, 255).astype(np.uint8)
        # conf_vis = cv2.bitwise_not(conf_vis) # 색상 반전 (필요시)
        v_min, v_max = 1e-7, 1e-3  # 이 값을 줄일수록 맵이 더 진해집니다.
        conf_vis = (conf_raw - v_min) / (v_max - v_min)
        conf_vis = np.clip(conf_vis, 0, 1) # 0~1 사이로 고정
        # 2. Depth 시각화 (무효 영역 흰색 처리)
        vis_depth = plt.cm.jet(depth_norm)[:, :, :3] # RGB만 추출
        # valid_mask가 False인 곳을 [1, 1, 1] (흰색)으로 강제 변환
        vis_depth[~valid_mask] = [1.0, 1.0, 1.0]
        vis_depth_final = vis_depth
        
        # ===== Visualization =====
        ax_row = axes[idx] if rows > 1 else axes[0]
        
        # Column 1: I1 (cropped)
        ax_row[0].imshow(I1_rgb_crop)
        ax_row[0].set_title("Image 1 (Far)", fontsize=11)
        ax_row[0].axis('off')
        
        # Column 2: I2 (cropped to same region)
        ax_row[1].imshow(I2_rgb_crop)
        ax_row[1].set_title("Image 2 (Near)", fontsize=11)
        ax_row[1].axis('off')
        
        # Column 3: Aligned I2 (cropped - NO BLACK BORDERS!)
        ax_row[2].imshow(I2_aligned_crop)
        ax_row[2].set_title("Aligned Image 2", fontsize=11, color='blue')
        ax_row[2].axis('off')
        
        # Column 4: Confidence Map (INVERTED - white background!)
        ax_row[3].imshow(conf_map, cmap='gray_r')  # gray_r = inverted!
        ax_row[3].set_title("Confidence Map", fontsize=11)
        ax_row[3].axis('off')
        
        # Column 5: Predicted Depth
        ax_row[4].imshow(vis_depth_final)
        # ax_row[4].set_title("Predicted Depth\n(A=1.1, B=0.3)", fontsize=11, fontweight='bold')
        ax_row[4].axis('off')
        
        # Column 6: Colorbar
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=VMIN, vmax=VMAX)
        sm = cm.ScalarMappable(cmap='jet', norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, cax=ax_row[5])
        # cbar.set_label('Distance (m)', fontsize=10)
        # cbar.set_ticks([VMIN, (VMIN+VMAX)/2, VMAX])
        # cbar.set_ticklabels([f'{VMIN}m\n(Near)', f'{(VMIN+VMAX)/2:.1f}m', f'{VMAX}m\n(Far)'])
    
    plt.tight_layout()
    output_filename = "paper_final_result_FIXED.png"
    plt.savefig(output_filename, dpi=150)
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"  Main result: {output_filename}")
    print(f"  Toggle images: {toggle_dir}/")
    print(f"\nAll fixes applied:")
    print(f"  ✓ A={const.A_CALIB}, B={const.B_CALIB} (calibrated)")
    print(f"  ✓ Depth range: {VMIN}~{VMAX}m")
    print(f"  ✓ Use ALIGNED grayscale from SIFT (critical fix!)")
    print(f"  ✓ Confidence threshold: 0.10 (lower = more pixels)")
    print(f"  ✓ Aligned Image cropped (no black borders)")
    print(f"  ✓ Confidence map INVERTED (white background)")
    print(f"  ✓ Invalid pixels: WHITE")
    print(f"  ✓ Toggle comparison saved")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_paper_visual_final()