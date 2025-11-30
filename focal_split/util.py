import cv2
import numpy as np
import pickle
from typing import Any, List, Tuple, Optional

import constants as const

CROP_DEFAULT: int = 20


def align_images(I1: np.ndarray,
                 I2: np.ndarray,
                 crop: int = CROP_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    """센터 맞춰서 네 귀퉁이 crop 잘라내기."""
    if crop <= 0:
        return I1, I2
    return (
        I1[crop:-crop, crop:-crop],
        I2[crop:-crop, crop:-crop],
    )


def load_dataset(path: Optional[str] = None) -> List[Any]:
    """Luo untethered snapshot dataset pkl"""
    if path is None:
        path = const.DATASET_PKL
    print(f"Loading dataset from: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def dataset_sample_to_images_and_depth(sample: Any) -> Tuple[np.ndarray, np.ndarray, float]:


    if isinstance(sample, (list, tuple)) and len(sample) >= 2 and isinstance(sample[0], dict):
        dict_far = sample[0]
        dict_near = sample[1]

        if 'Img' not in dict_far or 'Loc' not in dict_far:
            raise KeyError(f"필수 키('Img', 'Loc')가 없습니다. 현재 키: {list(dict_far.keys())}")

        I1_raw = np.asarray(dict_near['Img'])
        I2_raw = np.asarray(dict_far['Img'])


        Z_val = np.asarray(dict_far['Loc']).flatten()[0]
        Ztrue = float(Z_val) / 1_000_000.0

        return I1_raw, I2_raw, Ztrue

    elif isinstance(sample, dict):
        if 'Img' in sample and 'Loc' in sample:
            imgs = sample['Img']
            if len(imgs) >= 2:
                return np.asarray(imgs[0]), np.asarray(imgs[1]), float(np.asarray(sample['Loc']).flatten()[0])
    
    raise TypeError(f"지원하지 않는 데이터 구조입니다. Type: {type(sample)}")