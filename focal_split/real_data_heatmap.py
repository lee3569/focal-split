import os
import glob
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal

import util
import constant as const

# =========================
# Config
# =========================
PKL_GLOB = "./*.pkl"
EXCLUDE_SUBSTRINGS = ["saved_list"]

FLIP_VERTICAL = True
FLIP_HORIZONTAL = True

HEATMAP_RANGE = [[0.0, 2.5], [0.0, 2.5]]
HEATMAP_BINS = 40

# =========================
# Professor funcs
# =========================
def removeLowFreqInfo(img: np.ndarray, ksize: int = 21) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.float32)
    kernel /= np.sum(kernel)
    bias = signal.fftconvolve(img, kernel, "same")
    return img - bias

def compute_confidence(I_t: np.ndarray) -> np.ndarray:
    conf = I_t ** 2
    conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
    return conf_norm

def filter_by_confidence_pixelwise(
    Z_pred: np.ndarray,
    Z_true: np.ndarray,
    Z_conf: np.ndarray,
    confidence_level: float = 0.06,
    working_range: tuple = (0.0, 2.5),
):
    """
    픽셀 단위로 (true, pred) 페어를 뽑아서 리턴.
    confidence_level=0.95 => 하위 5% confidence 버림 (네 코드 로직 그대로).
    """
    Zp = Z_pred.reshape(-1)
    Zt = Z_true.reshape(-1)
    C  = Z_conf.reshape(-1)

    valid = np.isfinite(Zp) & np.isfinite(Zt) & np.isfinite(C)
    Zp, Zt, C = Zp[valid], Zt[valid], C[valid]
    if C.size == 0:
        return np.array([]), np.array([])

    sorted_conf = np.sort(C)
    conf_idx = int((1 - confidence_level) * len(sorted_conf))
    conf_threshold = sorted_conf[conf_idx] if conf_idx < len(sorted_conf) else sorted_conf[0]

    keep = (
        (C > conf_threshold) &
        (Zp >= working_range[0]) & (Zp <= working_range[1]) &
        (Zt >= working_range[0]) & (Zt <= working_range[1])
    )
    return Zt[keep], Zp[keep]

# =========================
# PKL parsing (robust)
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
    if isinstance(obj, (list, tuple)):
        if len(obj) < 2:
            raise ValueError("Not enough elements to extract img1/img2")
        img1_raw, img2_raw = obj[0], obj[1]
        img1 = stack_rows(img1_raw) if isinstance(img1_raw, (list, tuple)) else np.array(img1_raw)
        img2 = stack_rows(img2_raw) if isinstance(img2_raw, (list, tuple)) else np.array(img2_raw)
        return img1, img2

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

def extract_img_pair_and_depth(obj):
    """
    지원하는 케이스:
    1) (img1, img2)만 있는 PKL -> (img1,img2,None)
    2) (img1, img2, Z_true) -> Z_true가 scalar 또는 (H,W) depth map
    3) dict 내부에 위 구조가 숨어있음
    """
    if isinstance(obj, (list, tuple)):
        if len(obj) >= 3:
            img1, img2 = extract_img_pair(obj)
            Z_true = obj[2]
            return img1, img2, Z_true
        img1, img2 = extract_img_pair(obj)
        return img1, img2, None

    if isinstance(obj, dict):
        candidates = []
        for k in ["data", "samples", "items", "dataset", "list", "saved_list"]:
            if k in obj:
                candidates.append(obj[k])
        candidates.extend(list(obj.values()))
        for cand in candidates:
            try:
                return extract_img_pair_and_depth(cand)
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

def to_rgb_u8(img: np.ndarray) -> np.ndarray:
    x = np.array(img)
    if x.ndim == 2:
        x = x.astype(np.float32)
        if np.nanmax(x) <= 1.5:
            x *= 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
        return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

    if x.ndim == 3 and x.shape[2] == 1:
        return to_rgb_u8(x[:, :, 0])

    if x.ndim == 3 and x.shape[2] == 3:
        x = x.astype(np.float32)
        if np.nanmax(x) <= 1.5:
            x *= 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
        # PKL이 BGR이면 여기서 BGR->RGB 켜라
        # x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        return x

    raise ValueError(f"Unsupported image shape: {x.shape}")

def to_gray_f32(img: np.ndarray) -> np.ndarray:
    rgb = to_rgb_u8(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

def to_depth_f32(z, vmin=0.0, vmax=2.5) -> np.ndarray:
    if z is None:
        return None

    if np.isscalar(z):
        return float(z)

    arr = np.array(z)

    # (H,W,1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0].astype(np.float32)

    # (H,W) numeric
    if arr.ndim == 2:
        return arr.astype(np.float32)

    # (H,W,3) -> either replicated depth or color-mapped depth
    if arr.ndim == 3 and arr.shape[2] == 3:
        # float이면 0~1일 수도 있음
        if arr.dtype != np.uint8:
            a = arr.astype(np.float32)
            if np.nanmax(a) <= 1.5:
                a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
            else:
                a = np.clip(a, 0, 255).astype(np.uint8)
        else:
            a = arr

        # 1) 채널 복제된 depth면 채널0 쓰기
        if _looks_like_replicated_channels(a):
            return a[:, :, 0].astype(np.float32)

        # 2) jet 컬러맵 depth면 역변환 (근사)
        # 너 PKL이 BGR로 저장됐을 가능성 있으면 아래 한 줄 주석 해제해서 RGB로 변환
        # a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

        return decode_jet_rgb_to_depth(a, vmin=vmin, vmax=vmax)

    # 그 외는 depth라고 보기 힘듦
    return None
def _looks_like_replicated_channels(rgb: np.ndarray, tol: float = 1e-3) -> bool:
    a = rgb[:, :, 0].astype(np.float32)
    b = rgb[:, :, 1].astype(np.float32)
    c = rgb[:, :, 2].astype(np.float32)
    return (np.nanmax(np.abs(a - b)) < tol) and (np.nanmax(np.abs(a - c)) < tol)

def decode_jet_rgb_to_depth(z_rgb: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """
    z_rgb: (H,W,3) RGB or BGR. (너 PKL이 RGB인지 BGR인지 애매하면 둘 다 시도 가능)
    white([255,255,255])는 invalid로 NaN 처리.
    jet 역변환은 근사(가장 가까운 jet 색 찾기).
    """
    z = z_rgb.astype(np.uint8)

    # white -> invalid
    white = (z[:, :, 0] > 250) & (z[:, :, 1] > 250) & (z[:, :, 2] > 250)

    # build jet LUT (0..255)
    lut = (plt.cm.jet(np.arange(256))[:, :3] * 255).astype(np.uint8)  # (256,3)

    # flatten pixels
    pix = z.reshape(-1, 3).astype(np.int16)      # (N,3)
    lut16 = lut.astype(np.int16)                # (256,3)

    # nearest color index (vectorized, memory OK for ~420*515)
    # dist^2 = sum((pix - lut)^2)
    diff = pix[:, None, :] - lut16[None, :, :]  # (N,256,3)
    dist2 = np.sum(diff * diff, axis=2)         # (N,256)
    idx = np.argmin(dist2, axis=1).astype(np.float32)  # (N,)

    depth = vmin + (idx / 255.0) * (vmax - vmin)
    depth = depth.reshape(z.shape[0], z.shape[1]).astype(np.float32)
    depth[white] = np.nan
    return depth
# =========================
# Heatmap from PKLs
# =========================
def generate_heatmap_from_pkls(
    A_test: float,
    B_test: float,
    save_path: str = "heatmap.png",
    max_files = None,
    confidence_level: float = 0.95,
    working_range: tuple = (0.0, 2.5),
):
    pkl_files = sorted(glob.glob(PKL_GLOB))
    pkl_files = [p for p in pkl_files if not any(s in os.path.basename(p) for s in EXCLUDE_SUBSTRINGS)]
    if max_files:
        pkl_files = pkl_files[:max_files]

    if not pkl_files:
        print("No PKL files found.")
        return

    all_Z_true = []
    all_Z_pred = []

    print(f"\n{'='*60}")
    print(f"Heatmap | A={A_test:.4f}, B={B_test:.4f}")
    print(f"Files: {len(pkl_files)}")
    print(f"{'='*60}")

    for idx, pkl_path in enumerate(pkl_files):
        name = os.path.basename(pkl_path)
        try:
            obj = load_pkl(pkl_path)
            img1_raw, img2_raw, z_true_raw = extract_img_pair_and_depth(obj)

            # 이미지
            I1_rgb = apply_orientation(to_rgb_u8(img1_raw))
            I2_rgb = apply_orientation(to_rgb_u8(img2_raw))
            I1_gray = apply_orientation(to_gray_f32(img1_raw))
            I2_gray = apply_orientation(to_gray_f32(img2_raw))

            # (선택) 정합: calibration set이면 굳이 안 해도 되지만,
            # PKL들이 뒤섞여 있으면 일관성 위해 그대로 둠.
            I1_aligned_gray, I2_aligned_gray, H = util.align_images(I1_gray, I2_gray, debug=False)

            if H is not None:
                h, w = I1_rgb.shape[:2]
                I2_aligned_rgb = cv2.warpPerspective(I2_rgb, H, (w, h))
            else:
                I2_aligned_rgb = I2_rgb

            # crop (니 visual 코드 값)
            left_crop = 30
            right_crop = 95
            top_crop = 30
            bottom_crop = 30
            h, w = I1_gray.shape[:2]

            I1_gray_crop = I1_aligned_gray[top_crop:h-bottom_crop, left_crop:w-right_crop]
            I2_gray_crop = I2_aligned_gray[top_crop:h-bottom_crop, left_crop:w-right_crop]

            # depth true
            Z_true = to_depth_f32(z_true_raw, vmin=working_range[0], vmax=working_range[1])

            # Z_true가 depth map이면 같은 crop 적용 (그리고 H까지 따라가야 “진짜로” 맞음)
            # 여기서는 pragmatic하게:
            # - Z_true가 (H,W)이고 H가 있으면 warp도 해주고 crop.
            # - Z_true가 scalar면 그대로.
            Z_true_crop = None
            if isinstance(Z_true, np.ndarray):
                if H is not None:
                    Z_true_warp = cv2.warpPerspective(Z_true, H, (w, h))
                else:
                    Z_true_warp = Z_true
                Z_true_crop = Z_true_warp[top_crop:h-bottom_crop, left_crop:w-right_crop]

            # ===== professor pipeline =====
            I1 = removeLowFreqInfo(I1_gray_crop, 21)
            I2 = removeLowFreqInfo(I2_gray_crop, 21)
            I1 = cv2.GaussianBlur(I1, (11, 11), 0)
            I2 = cv2.GaussianBlur(I2, (11, 11), 0)

            img_avg = (I1 + I2) / 2
            Laplacian_I = img_avg - cv2.GaussianBlur(img_avg, (11, 11), 0)
            I_s_t = (I1 - I2) / 2

            V = Laplacian_I
            W = A_test * Laplacian_I + B_test * I_s_t

            kernelSize = 21
            kernel = np.ones((kernelSize, 1))
            VW = signal.convolve2d(V * W, kernel, "same", "symm")
            VW = signal.convolve2d(VW, kernel.T, "same", "symm")
            W2 = signal.convolve2d(W**2, kernel, "same", "symm")
            W2 = signal.convolve2d(W2, kernel.T, "same", "symm")

            Z_pred = np.divide(VW, W2 + 1e-10, where=(W2 != 0))
            confidence = compute_confidence(I_s_t)

            # ===== collect points =====
            if Z_true is None:
                # GT 없으면 heatmap 못 만듦. 스킵.
                print(f"  [SKIP] {name}: no Z_true in PKL")
                continue

            if np.isscalar(Z_true):
                # scalar GT: confidence 필터된 pred만 뽑아서 GT 반복
                Z_filtered = filter_by_confidence_pixelwise(
                    Z_pred,
                    np.full_like(Z_pred, float(Z_true), dtype=np.float32),
                    confidence,
                    confidence_level=confidence_level,
                    working_range=working_range
                )
                # 위 함수는 (Zt, Zp)를 리턴하니까 unpack
                Zt_keep, Zp_keep = Z_filtered
                if Zt_keep.size:
                    all_Z_true.extend(Zt_keep.tolist())
                    all_Z_pred.extend(Zp_keep.tolist())
            else:
                # depth map GT: 픽셀 단위로 매칭
                if Z_true_crop is None or Z_true_crop.shape != Z_pred.shape:
                    print(f"  [SKIP] {name}: Z_true shape mismatch. true={None if Z_true_crop is None else Z_true_crop.shape}, pred={Z_pred.shape}")
                    continue

                Zt_keep, Zp_keep = filter_by_confidence_pixelwise(
                    Z_pred,
                    Z_true_crop,
                    confidence,
                    confidence_level=confidence_level,
                    working_range=working_range
                )
                if Zt_keep.size:
                    all_Z_true.extend(Zt_keep.tolist())
                    all_Z_pred.extend(Zp_keep.tolist())

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1}/{len(pkl_files)}")

        except Exception as e:
            print(f"  [Error] {name}: {e}")
            continue

    print(f"\nTotal valid points: {len(all_Z_true)}")
    if len(all_Z_true) < 100:
        print("[Warning] Too few valid points for heatmap.")
        return

    # ===== plotting (your style) =====
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    heatmap, xedges, yedges = np.histogram2d(
        all_Z_true, all_Z_pred,
        bins=HEATMAP_BINS,
        range=HEATMAP_RANGE
    )
    heatmap = heatmap.T
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plot_heatmap = ax.imshow(heatmap, extent=extent, origin="lower")
    fig.colorbar(plot_heatmap, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("True Depth (m)")
    ax.set_ylabel("Estimated Depth (m)")
    ax.set_title(f"Heatmap (PKL) | A={A_test:.4f}, B={B_test:.4f}")
    ax.grid()

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    print(f"Saved: {save_path}\n")


if __name__ == "__main__":
    test_values = [
        (const.A_CALIB, const.B_CALIB),
        (1.20, 0.80),
        (1.15, 0.75),
        (1.10, 0.90),
        (1.00, 0.60),
    ]
    for A, B in test_values:
        save_path = f"heatmap_pkl_A{A:.2f}_B{B:.2f}.png"
        generate_heatmap_from_pkls(A_test=A, B_test=B, save_path=save_path, max_files=None)