from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import ALPHA_THRESHOLD, CROP_PADDING


@dataclass(frozen=True)
class CropMeta:
    x0: int
    y0: int
    x1: int
    y1: int


def inject_alpha(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    """
    Create a lossless RGBA PIL image from RGB uint8 and alpha float32 in [0,1].
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {rgb.shape}")
    if alpha.ndim != 2 or alpha.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"Alpha shape {alpha.shape} does not match RGB {rgb.shape[:2]}")

    a8 = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba = np.dstack([rgb, a8])
    return Image.fromarray(rgba, mode="RGBA")


def crop_to_content(alpha: np.ndarray, threshold: float = ALPHA_THRESHOLD, padding: int = CROP_PADDING) -> CropMeta:
    """
    Compute a padded bounding box around alpha > threshold.
    """
    h, w = alpha.shape[:2]
    ys, xs = np.where(alpha > float(threshold))
    if ys.size == 0 or xs.size == 0:
        raise RuntimeError("No foreground detected (alpha threshold).")

    x0 = max(0, int(xs.min()) - int(padding))
    y0 = max(0, int(ys.min()) - int(padding))
    x1 = min(w, int(xs.max()) + int(padding) + 1)
    y1 = min(h, int(ys.max()) + int(padding) + 1)
    return CropMeta(x0=x0, y0=y0, x1=x1, y1=y1)


def apply_crop(img: Image.Image, crop: CropMeta) -> Image.Image:
    return img.crop((crop.x0, crop.y0, crop.x1, crop.y1))


def save_rgba_png(img: Image.Image, out_path: str) -> None:
    """
    Save as lossless RGBA PNG.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    img.save(out_path, format="PNG", optimize=False)
