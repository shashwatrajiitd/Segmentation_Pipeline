from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .composite import apply_crop, crop_to_content, inject_alpha, save_rgba_png
from .config import ALPHA_THRESHOLD
from .inference import predict_matte
from .model import get_device, load_birefnet_hf, load_torchscript_matting_model
from .postprocess import postprocess_matte
from .preprocess import load_image, normalize, resize_with_padding


@dataclass(frozen=True)
class StageTimings:
    preprocess_s: float
    inference_s: float
    postprocess_s: float
    composite_s: float
    total_s: float


def process_image(
    image_path: str,
    out_path: str,
    model,
    device,
    *,
    fail_fast: bool = True,
) -> StageTimings:
    """
    Deterministic, linear pipeline:
      1) Load image
      2) Preprocess
      3) Inference
      4) Post-process
      5) Composite
      6) Save
    """
    t0 = time.perf_counter()

    # Preprocess
    t_pre0 = time.perf_counter()
    rgb = load_image(image_path)
    padded, meta = resize_with_padding(rgb)
    x = normalize(padded)
    t_pre1 = time.perf_counter()

    # Inference
    t_inf0 = time.perf_counter()
    matte_1024 = predict_matte(model, x, device)
    t_inf1 = time.perf_counter()

    # Post-process
    t_post0 = time.perf_counter()
    matte, _matte_lcca = postprocess_matte(matte_1024, meta)
    t_post1 = time.perf_counter()

    if float((matte > ALPHA_THRESHOLD).mean()) < 0.001:
        msg = f"No foreground detected for {os.path.basename(image_path)}"
        if fail_fast:
            raise RuntimeError(msg)
        matte[:] = 0.0

    # Composite
    t_comp0 = time.perf_counter()
    rgba = inject_alpha(rgb, matte)
    crop = crop_to_content(matte, threshold=ALPHA_THRESHOLD)
    rgba = apply_crop(rgba, crop)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_rgba_png(rgba, out_path)
    t_comp1 = time.perf_counter()

    t1 = time.perf_counter()
    return StageTimings(
        preprocess_s=t_pre1 - t_pre0,
        inference_s=t_inf1 - t_inf0,
        postprocess_s=t_post1 - t_post0,
        composite_s=t_comp1 - t_comp0,
        total_s=t1 - t0,
    )


def load_model_default(model_path: str):
    device = get_device()
    if isinstance(model_path, str) and model_path.startswith("hf:"):
        model = load_birefnet_hf(model_path[len("hf:") :], device=device)
    elif model_path in ("birefnet", "hf:birefnet"):
        model = load_birefnet_hf("ZhengPeng7/BiRefNet", device=device)
    else:
        model = load_torchscript_matting_model(model_path, device=device)
    return model, device

