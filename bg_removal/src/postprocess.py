from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import ALPHA_THRESHOLD, EDGE_BLUR_RADIUS, ERODE_KERNEL_SIZE
from .preprocess import PreprocessMeta


def restore_mask_to_original(mask_1024: np.ndarray, meta: PreprocessMeta) -> np.ndarray:
    """
    Restore a model-space square mask back to original image resolution.

    Steps:
      1) remove padding using x/y offsets + resized sizes
      2) resize back to (orig_w, orig_h)
    """
    if mask_1024.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask_1024.shape}")
    if mask_1024.dtype != np.float32:
        mask_1024 = mask_1024.astype(np.float32, copy=False)

    x0, y0 = meta.x_offset, meta.y_offset
    x1, y1 = x0 + meta.resized_w, y0 + meta.resized_h
    cropped = mask_1024[y0:y1, x0:x1]
    if cropped.size == 0:
        raise ValueError("Mask crop is empty; check preprocessing meta.")

    restored = cv2.resize(cropped, (meta.orig_w, meta.orig_h), interpolation=cv2.INTER_LINEAR)
    restored = np.clip(restored, 0.0, 1.0).astype(np.float32, copy=False)
    return restored


def erode_mask(matte: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Morphological erosion on the matte (applied in original image space).

    Purpose: shrink foreground slightly to remove thin halo artifacts.
    """
    k = int(kernel_size)
    if k <= 0:
        return matte.astype(np.float32, copy=False)
    if k % 2 == 0:
        # keep behavior explicit + deterministic
        k += 1

    m = np.clip(matte.astype(np.float32, copy=False), 0.0, 1.0)
    m8 = (m * 255.0).astype(np.uint8)
    kernel = np.ones((k, k), np.uint8)
    eroded = cv2.erode(m8, kernel, iterations=1)
    return (eroded.astype(np.float32) / 255.0).astype(np.float32, copy=False)


def largest_connected_component(matte: np.ndarray, threshold: float = ALPHA_THRESHOLD) -> np.ndarray:
    """
    Keep the dominant connected component (product) and remove small dust blobs.
    """
    if matte.ndim != 2:
        raise ValueError(f"Expected 2D matte, got shape={matte.shape}")
    m = matte.astype(np.float32, copy=False)

    binary = (m > float(threshold)).astype(np.uint8)
    if int(binary.sum()) == 0:
        return np.zeros_like(m, dtype=np.float32)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return m

    # label 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_label = int(np.argmax(areas) + 1)
    keep = (labels == keep_label).astype(np.float32)
    return (m * keep).astype(np.float32, copy=False)


def edge_smooth_alpha(
    matte: np.ndarray,
    blur_radius: int = EDGE_BLUR_RADIUS,
    threshold: float = ALPHA_THRESHOLD,
) -> np.ndarray:
    """
    Smooth alpha only around edges:
      - detect edges on the matte
      - dilate edge region slightly
      - blend blurred matte into original matte only on edge band
    """
    if matte.ndim != 2:
        raise ValueError(f"Expected 2D matte, got shape={matte.shape}")
    if blur_radius <= 0:
        return matte.astype(np.float32, copy=False)

    m = np.clip(matte.astype(np.float32, copy=False), 0.0, 1.0)

    # Create a robust edge signal: run Canny on an 8-bit representation of the matte.
    m8 = (m * 255.0).astype(np.uint8)
    edges = cv2.Canny(m8, 60, 120)

    # Also add edges around the binary boundary to help with low-contrast mattes.
    binary = (m > float(threshold)).astype(np.uint8) * 255
    edges2 = cv2.Canny(binary, 60, 120)
    edges = cv2.bitwise_or(edges, edges2)

    # Expand the edge band
    k = 2 * blur_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    edge_band = cv2.dilate(edges, kernel, iterations=1)
    edge_band_f = (edge_band > 0).astype(np.float32)

    # Blur matte globally, then only blend on edge band
    sigma = max(0.5, float(blur_radius))
    blurred = cv2.GaussianBlur(m, (0, 0), sigmaX=sigma, sigmaY=sigma)

    out = m * (1.0 - edge_band_f) + blurred * edge_band_f
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def postprocess_matte(mask_1024: np.ndarray, meta: PreprocessMeta) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full post-processing:
      - restore to original resolution
      - keep largest component
      - edge smoothing

    Returns:
      - matte_orig: float32 (orig_h, orig_w) in [0,1]
      - matte_lcca: matte after LCCA (before edge smooth) for debugging if needed
    """
    matte_orig = restore_mask_to_original(mask_1024, meta)
    matte_lcca = largest_connected_component(matte_orig, threshold=ALPHA_THRESHOLD)
    matte_eroded = erode_mask(matte_lcca, kernel_size=ERODE_KERNEL_SIZE)
    matte_smooth = edge_smooth_alpha(matte_eroded, blur_radius=EDGE_BLUR_RADIUS, threshold=ALPHA_THRESHOLD)
    return matte_smooth, matte_lcca

