from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
import torch

from .config import IMAGENET_MEAN, IMAGENET_STD, PAD_COLOR, TARGET_SIZE


@dataclass(frozen=True)
class PreprocessMeta:
    """Metadata required to map model-space outputs back to original image space."""

    orig_h: int
    orig_w: int
    resized_h: int
    resized_w: int
    scale: float
    x_offset: int
    y_offset: int
    target_size: int = TARGET_SIZE


def load_image(path: str) -> np.ndarray:
    """
    Load an image as RGB uint8 ndarray of shape (H, W, 3).
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8, copy=False)
    return rgb


def resize_with_padding(img: np.ndarray) -> Tuple[np.ndarray, PreprocessMeta]:
    """
    Aspect-safe resize to fit within TARGET_SIZE, then pad to (TARGET_SIZE, TARGET_SIZE).

    Returns:
      - padded_rgb: uint8 ndarray (TARGET_SIZE, TARGET_SIZE, 3)
      - meta: PreprocessMeta containing scale and offsets
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got shape={img.shape}")

    orig_h, orig_w = img.shape[:2]
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Invalid image size: {(orig_h, orig_w)}")

    # Scale longest side to TARGET_SIZE (never upscale above TARGET_SIZE if already smaller? We DO upscale
    # for consistent SOD model behavior; SOD models typically trained at fixed-ish sizes.)
    scale = float(TARGET_SIZE) / float(max(orig_h, orig_w))
    resized_w = max(1, int(round(orig_w * scale)))
    resized_h = max(1, int(round(orig_h * scale)))

    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    # Pad to square
    padded = np.full((TARGET_SIZE, TARGET_SIZE, 3), PAD_COLOR, dtype=np.uint8)
    x_offset = (TARGET_SIZE - resized_w) // 2
    y_offset = (TARGET_SIZE - resized_h) // 2
    padded[y_offset : y_offset + resized_h, x_offset : x_offset + resized_w] = resized

    meta = PreprocessMeta(
        orig_h=orig_h,
        orig_w=orig_w,
        resized_h=resized_h,
        resized_w=resized_w,
        scale=scale,
        x_offset=x_offset,
        y_offset=y_offset,
    )
    return padded, meta


def normalize(img: np.ndarray) -> torch.Tensor:
    """
    Normalize uint8 RGB image to float32 torch tensor: (1,3,TARGET_SIZE,TARGET_SIZE).
    """
    if img.shape[:2] != (TARGET_SIZE, TARGET_SIZE) or img.shape[2] != 3:
        raise ValueError(f"Expected shape ({TARGET_SIZE},{TARGET_SIZE},3), got {img.shape}")
    x = img.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # CHW
    t = torch.from_numpy(x).unsqueeze(0).contiguous()  # NCHW
    if t.dtype != torch.float32:
        t = t.float()
    return t


def meta_to_dict(meta: PreprocessMeta) -> Dict[str, int | float]:
    """Convenience helper if you want JSON-serializable metadata."""
    return {
        "orig_h": meta.orig_h,
        "orig_w": meta.orig_w,
        "resized_h": meta.resized_h,
        "resized_w": meta.resized_w,
        "scale": float(meta.scale),
        "x_offset": int(meta.x_offset),
        "y_offset": int(meta.y_offset),
        "target_size": int(meta.target_size),
    }

