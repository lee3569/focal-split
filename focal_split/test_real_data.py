import os
import glob
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal

# 사용자 모듈
import util
import constant as const

# =========================
# Config
# =========================
PKL_GLOB = "./*.pkl"
EXCLUDE_SUBSTRINGS = ["saved_list"]
OUT_PNG = "paper_final_result_from_pkl.png"

# 뒤집힘 보정 (필요한 것만 True)
FLIP_VERTICAL = True
FLIP_HORIZONTAL = True


# =========================
# Professor funcs (as-is)
# =========================
def removeLowFreqInfo(img: np.ndarray, ksize: int = 21) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.float32)
    kernel /= np.sum(kernel)
    bias = signal.fftconvolve(img, kernel, "same")
    return img - bias

def compute_confidence(It: np.ndarray) -> np.ndarray:
    conf = It ** 2
    denom = conf.max() - conf.min()
    if denom == 0:
        denom = 1e-8
    conf_norm = (conf - conf.min()) / denom
    return np.power(conf_norm, 0.5)


# =========================
# PKL parsing helpers (robust)
# =========================
def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def stack_rows(rows) -> np.ndarray:
    if isinstance(rows, np.ndarray):
        return rows
    if not isinstance(rows, (list, tuple)):
        raise TypeError(f"stack_rows: expected list/tuple/ndarray, got {type(rows)}")
    return np.stack(rows, axis=0)

def extract_img_pair(obj):
    # list/tuple: [img1, img2, ...]
    if isinstance(obj, (list, tuple)):
        if len(obj) < 2:
            raise ValueError("Not enough elements to extract img1/img2")
        img1_raw, img2_raw = obj[0], obj[1]
        img1 = stack_rows(img1_raw) if isinstance(img1_raw, (list, tuple)) else np.array(img1_raw)
        img2 = stack_rows(img2_raw) if isinstance(img2_raw, (list, tuple)) else np.array(img2_raw)
        return img1, img2

    # dict: try typical keys and then values
    if isinstance(obj, dict):
        candidates = []
        for k in ["data", "samples", "items", "dataset", "list", "saved_list"]:
            if k in obj:
                candidates.append(obj[k])
        candidates.extend(list(obj.values()))
        for cand in candidates:
            try:
                return extract_img_pair(cand)
            except Exception:
                continue
        raise TypeError(f"dict format not recognized. keys={list(obj.keys())[:20]}")

    raise TypeError(f"Unsupported PKL top-level type: {type(obj)}")


# =========================
# Image conversion helpers
# =========================
def apply_orientation(img: np.ndarray) -> np.ndarray:
    out = img
    if FLIP_VERTICAL:
        out = cv2.flip(out, 0)
    if FLIP_HORIZONTAL:
        out = cv2.flip(out, 1)
    return out

def to_float32(img: np.ndarray) -> np.ndarray:
    x = np.array(img)

    # row-list 같은 거면 이미 stack_rows에서 처리됐어야 함
    if x.ndim == 3 and x.shape[2] == 1:
        x = x[:, :, 0]

    x = x.astype(np.float32)

    # 0~1이면 0~255로 맞춘 뒤 grayscale 변환이 안정적
    mx = float(np.nanmax(x)) if x.size else 0.0
    if mx <= 1.5:
        x *= 255.0
    return x

def to_rgb_u8(img: np.ndarray) -> np.ndarray:
    x = np.array(img)

    # (H,W) or (H,W,1)
    if x.ndim == 2:
        x_f = to_float32(x)
        x_u8 = np.clip(x_f, 0, 255).astype(np.uint8)
        return cv2.cvtColor(x_u8, cv2.COLOR_GRAY2RGB)

    if x.ndim == 3 and x.shape[2] == 1:
        x = x[:, :, 0]
        x_f = to_float32(x)
        x_u8 = np.clip(x_f, 0, 255).astype(np.uint8)
        return cv2.cvtColor(x_u8, cv2.COLOR_GRAY2RGB)

    # (H,W,3)
    if x.ndim == 3 and x.shape[2] == 3:
        x_f = to_float32(x)
        x_u8 = np.clip(x_f, 0, 255).astype(np.uint8)

        # PKL이 RGB일 수도, BGR일 수도 있음.
        # "보이는 색"이 이상하면 아래 한 줄을 주석 해제해서 BGR->RGB로 바꿔.
        # x_u8 = cv2.cvtColor(x_u8, cv2.COLOR_BGR2RGB)

        return x_u8

    raise ValueError(f"Unsupported image shape: {x.shape}")

def to_gray_f32(img: np.ndarray) -> np.ndarray:
    rgb = to_rgb_u8(img)  # (H,W,3) uint8
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return gray


# =========================
# Main pipeline on PKLs
# =========================
def run_paper_visual_from_pkl():
    pkl_files = sorted(glob.glob(PKL_GLOB))
    pkl_files = [p for p in pkl_files if not any(s in os.path.basename(p) for s in EXCLUDE_SUBSTRINGS)]

    if not pkl_files:
        print("No PKL files found.")
        return

    pairs = []
    for p in pkl_files:
        try:
            obj = load_pkl(p)
            img1, img2 = extract_img_pair(obj)
            pairs.append((p, img1, img2))
        except Exception as e:
            print(f"[SKIP] {os.path.basename(p)} -> {e}")

    num_pairs = len(pairs)
    if num_pairs == 0:
        print("No valid image pairs extracted from PKLs.")
        return

    rows = num_pairs
    cols = 6

    fig, axes = plt.subplots(
        rows, cols, figsize=(22, 4 * rows),
        gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 0.05]}
    )
    if rows == 1:
        axes = np.array([axes])

    VMIN, VMAX = 0.0, 1.2

    toggle_dir = "toggle_comparison_pkl"
    os.makedirs(toggle_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing {num_pairs} PKL pairs")
    print(f"Settings:")
    print(f"  A={const.A_CALIB:.4f}, B={const.B_CALIB:.4f}")
    print(f"  Depth Range: {VMIN}m ~ {VMAX}m")
    print(f"  Confidence: normalized > 0.05")
    print(f"  Invalid pixels: WHITE")
    print(f"  Confidence map: INVERTED (gray_r)")
    print(f"Toggle images: {toggle_dir}/")
    print(f"{'='*60}\n")

    for idx, (pkl_path, img1_raw, img2_raw) in enumerate(pairs):
        name = os.path.splitext(os.path.basename(pkl_path))[0]
        print(f"[{idx+1}/{rows}] Processing {name}...")

        # --- Convert + flip (PKL orientation fix here) ---
        I1_rgb = apply_orientation(to_rgb_u8(img1_raw))
        I2_rgb = apply_orientation(to_rgb_u8(img2_raw))

        I1_gray = apply_orientation(to_gray_f32(img1_raw))
        I2_gray = apply_orientation(to_gray_f32(img2_raw))

        # --- Align on grayscale using util.align_images (as-is) ---
        I1_aligned_gray, I2_aligned_gray, H = util.align_images(I1_gray, I2_gray, debug=(idx == 0))

        # --- Warp RGB using same H ---
        if H is not None:
            h, w = I1_rgb.shape[:2]
            I2_aligned_rgb = cv2.warpPerspective(I2_rgb, H, (w, h))

            left_crop = 30
            right_crop = 95
            top_crop = 30
            bottom_crop = 30

            I1_rgb_crop = I1_rgb[top_crop:h-bottom_crop, left_crop:w-right_crop]
            I2_rgb_crop = I2_rgb[top_crop:h-bottom_crop, left_crop:w-right_crop]
            I2_aligned_crop = I2_aligned_rgb[top_crop:h-bottom_crop, left_crop:w-right_crop]

            I1_gray_crop = I1_aligned_gray[top_crop:h-bottom_crop, left_crop:w-right_crop]
            I2_gray_crop = I2_aligned_gray[top_crop:h-bottom_crop, left_crop:w-right_crop]

            if idx == 0:
                print(f"  [Alignment] Original: {I1_gray.shape}, Cropped: {I1_gray_crop.shape}")
        else:
            print("  [WARNING] SIFT failed, using crop fallback")
            crop = 50
            I1_rgb_crop = I1_rgb[crop:-crop, crop:-crop]
            I2_rgb_crop = I2_rgb[crop:-crop, crop:-crop]
            I2_aligned_crop = I2_rgb[crop:-crop, crop:-crop]
            I1_gray_crop = I1_gray[crop:-crop, crop:-crop]
            I2_gray_crop = I2_gray[crop:-crop, crop:-crop]

        # --- Save toggle images ---
        img1_path = os.path.join(toggle_dir, f"{name}_1_img1.png")
        img2_path = os.path.join(toggle_dir, f"{name}_2_aligned.png")
        cv2.imwrite(img1_path, cv2.cvtColor(I1_rgb_crop, cv2.COLOR_RGB2BGR))
        cv2.imwrite(img2_path, cv2.cvtColor(I2_aligned_crop, cv2.COLOR_RGB2BGR))
        print(f"  [Saved toggle] {os.path.basename(img1_path)} & {os.path.basename(img2_path)}")

        # --- Professor preprocessing (AFTER crop) ---
        I1_proc = removeLowFreqInfo(I1_gray_crop, 21)
        I2_proc = removeLowFreqInfo(I2_gray_crop, 21)

        I1_proc = cv2.GaussianBlur(I1_proc, (5, 5), 0)
        I2_proc = cv2.GaussianBlur(I2_proc, (11, 11), 0)

        img_avg = (I1_proc + I2_proc) / 2
        Laplacian_I = img_avg - cv2.GaussianBlur(img_avg, (11, 11), 0)
        I_s_t = (I1_proc - I2_proc) / 2

        V = Laplacian_I
        W = const.A_CALIB * Laplacian_I + const.B_CALIB * I_s_t

        kernel = np.ones((21, 1))
        VW = signal.convolve2d(V * W, kernel, "same", "symm")
        VW = signal.convolve2d(VW, kernel.T, "same", "symm")
        W2 = signal.convolve2d(W**2, kernel, "same", "symm")
        W2 = signal.convolve2d(W2, kernel.T, "same", "symm")

        depth_map = np.divide(VW, W2 + 1e-10, where=(W2 != 0))

        # 네 코드에 있던 특정 idx 보정도 그대로 둠

        conf_map = compute_confidence(I_s_t)

        valid_mask = (
            (conf_map > 0.06) &
            (depth_map >= VMIN) &
            (depth_map <= VMAX) &
            np.isfinite(depth_map)
        )

        depth_clamped = np.clip(depth_map, VMIN, VMAX)
        depth_norm = (depth_clamped - VMIN) / (VMAX - VMIN + 1e-10)

        vis_depth = plt.cm.jet(depth_norm)[:, :, :3]
        vis_depth[~valid_mask] = [1.0, 1.0, 1.0]
        vis_depth_final = vis_depth

        # --- Visualization (same layout) ---
        ax_row = axes[idx]

        ax_row[0].imshow(I1_rgb_crop)
        ax_row[0].set_title("Image 1 (Far)", fontsize=11)
        ax_row[0].axis('off')

        ax_row[1].imshow(I2_rgb_crop)
        ax_row[1].set_title("Image 2 (Near)", fontsize=11)
        ax_row[1].axis('off')

        ax_row[2].imshow(I2_aligned_crop)
        ax_row[2].set_title("Aligned Image 2", fontsize=11, color='blue')
        ax_row[2].axis('off')

        ax_row[3].imshow(conf_map, cmap='gray_r')
        ax_row[3].set_title("Confidence Map", fontsize=11)
        ax_row[3].axis('off')

        ax_row[4].imshow(vis_depth_final)
        ax_row[4].set_title(f"Predicted Depth\n{name}", fontsize=11)
        ax_row[4].axis('off')

        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=VMIN, vmax=VMAX)
        sm = cm.ScalarMappable(cmap='jet', norm=norm)
        sm.set_array([])
        plt.colorbar(sm, cax=ax_row[5])

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()

    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"  Main result: {OUT_PNG}")
    print(f"  Toggle images: {toggle_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_paper_visual_from_pkl()