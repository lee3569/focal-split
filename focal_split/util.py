import cv2
import numpy as np
import pickle
from typing import Any, List, Tuple, Optional

import constants as const

# Global defaults
CROP_DEFAULT: int = 20


# Image alignment using SIFT + Homography
def align_images(
    I1: np.ndarray,
    I2: np.ndarray,
    debug: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Paper Section 4.1: Align I2 to I1 using SIFT + Homography
    
    Args:
        I1: Reference image (grayscale, any dtype)
        I2: Image to align (grayscale, any dtype)
        debug: Print matching statistics
        
    Returns:
        I1: Original I1 (unchanged)
        I2_aligned: Warped I2 matching I1's geometry
        H: Homography matrix (3x3) for applying to RGB images
    """
    
    # Convert to uint8 for SIFT (SIFT expects CV_8U, 0-255 range)
    if I1.dtype != np.uint8:
        if I1.max() <= 1.0:
            I1_uint8 = (I1 * 255).astype(np.uint8)
        else:
            I1_uint8 = np.clip(I1, 0, 255).astype(np.uint8)
    else:
        I1_uint8 = I1
    
    if I2.dtype != np.uint8:
        if I2.max() <= 1.0:
            I2_uint8 = (I2 * 255).astype(np.uint8)
        else:
            I2_uint8 = np.clip(I2, 0, 255).astype(np.uint8)
    else:
        I2_uint8 = I2
    
    # Check if images are valid
    if I1_uint8.size == 0 or I2_uint8.size == 0:
        print("[ERROR] Empty image provided to align_images")
        return I1, I2, None
    
    # Step 1: SIFT feature detection
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(I1_uint8, None)  # ← uint8 사용!
    kp2, des2 = sift.detectAndCompute(I2_uint8, None)  # ← uint8 사용!
    
    if des1 is None or des2 is None:
        print("[WARNING] SIFT failed - images might be too dark/uniform")
        print(f"  I1: {len(kp1) if kp1 else 0} keypoints, I2: {len(kp2) if kp2 else 0} keypoints")
        return I1, I2, None
    
    if debug:
        print(f"[SIFT] I1: {len(kp1)} keypoints, I2: {len(kp2)} keypoints")
    
    # Step 2: Feature matching with Lowe's ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des2, des1, k=2)
    
    # Lowe's ratio test (0.75 threshold from paper/best practice)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:  # Make sure we have 2 neighbors
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        print(f"[WARNING] Too few matches: {len(good_matches)} < 4")
        print(f"  Cannot compute homography - need at least 4 point pairs")
        return I1, I2, None
    
    # Step 3: Homography estimation with RANSAC
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
    
    if H is None:
        print("[WARNING] Homography estimation failed")
        return I1, I2, None
    
    inliers = int(mask.sum())
    if debug:
        print(f"[Homography] Total matches: {len(good_matches)}, Inliers: {inliers}")
        print(f"[Homography] Inlier ratio: {inliers/len(good_matches)*100:.1f}%")
    
    # Step 4: Warp I2 to align with I1
    h, w = I1.shape[:2]
    I2_aligned = cv2.warpPerspective(I2, H, (w, h), flags=cv2.INTER_LINEAR)
    
    return I1, I2_aligned, H


# Dataset loading
def load_dataset(path: Optional[str] = None) -> List[Any]:
    """
    Load Luo untethered snapshot dataset (.pkl)
    """
    if path is None:
        path = const.DATASET_PKL

    print(f"[util] Loading dataset: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Dataset must be list-like, got {type(data)}")

    print(f"[util] Loaded {len(data)} samples")
    return list(data)


# Sample unpacking
def dataset_sample_to_images_and_depth(
    sample: Any
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Unpack dataset sample into image pair and ground truth depth
    
    Returns:
        I_far: First image (far sensor)
        I_near: Second image (near sensor)  
        Z_true: Ground truth depth in meters
    """

    if isinstance(sample, (list, tuple)) and len(sample) >= 2:
        if isinstance(sample[0], dict) and isinstance(sample[1], dict):
            far = sample[0]
            near = sample[1]

            if "Img" not in far or "Loc" not in far:
                raise KeyError(
                    f"Expected keys 'Img', 'Loc' in far sample. Got {far.keys()}"
                )

            I_far = np.asarray(far["Img"], dtype=np.float32)
            I_near = np.asarray(near["Img"], dtype=np.float32)

            Z_raw = np.asarray(far["Loc"]).flatten()[0]
            Z_true = float(Z_raw) / 1_000_000.0  # µm → m

            return I_far, I_near, Z_true

    if isinstance(sample, dict):
        if "Img" in sample and "Loc" in sample:
            imgs = sample["Img"]
            if not isinstance(imgs, (list, tuple)) or len(imgs) < 2:
                raise ValueError("Img must contain at least 2 images")

            I_far = np.asarray(imgs[0], dtype=np.float32)
            I_near = np.asarray(imgs[1], dtype=np.float32)

            Z_true = float(np.asarray(sample["Loc"]).flatten()[0])
            return I_far, I_near, Z_true

    # Unsupported format
    raise TypeError(
        f"Unsupported dataset sample format: {type(sample)}"
    )

def get_valid_region_after_warp(
    img_shape: Tuple[int, int],
    H: np.ndarray,
    margin: int = 30
) -> Tuple[int, int, int, int]:
    """
    Find valid (non-black) region after warping with homography H
    
    Args:
        img_shape: (height, width) of original image
        H: Homography matrix (3x3)
        margin: Additional margin to remove from edges (pixels)
        
    Returns:
        (x1, y1, x2, y2): Valid region bounding box
    """
    h, w = img_shape[:2]
    
    # Define corners of original image
    corners = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ]).reshape(-1, 1, 2)
    
    # Find where corners end up after inverse warp
    # (We warp I2 to I1, so we need inverse to see what region of I1 is valid)
    H_inv = np.linalg.inv(H)
    warped_corners = cv2.perspectiveTransform(corners, H_inv)
    
    # Get bounding box of warped corners
    x_coords = warped_corners[:, 0, 0]
    y_coords = warped_corners[:, 0, 1]
    
    x1 = int(np.ceil(x_coords.min())) + margin
    y1 = int(np.ceil(y_coords.min())) + margin
    x2 = int(np.floor(x_coords.max())) - margin
    y2 = int(np.floor(y_coords.max())) - margin
    
    # Clamp to image bounds
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    # Safety check
    if x2 <= x1 or y2 <= y1:
        print(f"[WARNING] Invalid crop region computed, using default")
        crop = 30
        return crop, crop, w - crop, h - crop
    
    return x1, y1, x2, y2