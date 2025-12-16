import numpy as np
import pickle
from typing import Any, List, Tuple, Optional

import constants as const

# Global defaults
CROP_DEFAULT: int = 20


# Image alignment (simple crop sync)
def align_images(
    I1: np.ndarray,
    I2: np.ndarray,
    crop: int = CROP_DEFAULT
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop borders equally so I1, I2 stay pixel-aligned.
    """
    if crop <= 0:
        return I1, I2

    return (
        I1[crop:-crop, crop:-crop],
        I2[crop:-crop, crop:-crop],
    )


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
