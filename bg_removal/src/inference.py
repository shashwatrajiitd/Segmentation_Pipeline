from __future__ import annotations

import numpy as np
import torch

from .config import TARGET_SIZE
from .model import forward_model


def _extract_primary_output(y):
    """
    BiRefNet / segmentation models may return:
      - a single tensor
      - (tensor, ...) tuple/list (final stage is typically last)
      - dict / ModelOutput with tensor fields

    We conservatively pick the last tensor-like payload for tuple/list outputs.
    """
    if isinstance(y, torch.Tensor):
        return y
    if isinstance(y, (list, tuple)) and len(y) > 0:
        for item in reversed(y):
            if isinstance(item, torch.Tensor):
                return item
        return y[-1]
    if isinstance(y, dict):
        # Prefer common logits keys if present.
        for k in ("logits", "pred", "alpha", "mask"):
            v = y.get(k, None)
            if isinstance(v, torch.Tensor):
                return v
        for v in y.values():
            if isinstance(v, torch.Tensor):
                return v
        # fall back to first value
        return next(iter(y.values()))
    return y


def predict_matte(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Forward pass and convert logits -> probability matte.

    Output:
      - float32 numpy array in [0,1]
      - shape (TARGET_SIZE, TARGET_SIZE)
    """
    if x.dtype != torch.float32:
        x = x.float()
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"Expected input tensor (1,3,H,W), got {tuple(x.shape)}")

    x = x.to(device)
    y = forward_model(model, x)
    y = _extract_primary_output(y)
    if not isinstance(y, torch.Tensor):
        raise RuntimeError(f"Model output is not a tensor: {type(y)}")

    # Expect either (1,1,H,W) or (1,H,W) or (H,W)
    if y.ndim == 4:
        y = y[:, :1, :, :]
        y = y[0, 0]
    elif y.ndim == 3:
        y = y[0]
    elif y.ndim == 2:
        pass
    else:
        raise RuntimeError(f"Unexpected output tensor shape: {tuple(y.shape)}")

    if y.shape[-2:] != (TARGET_SIZE, TARGET_SIZE):
        # Some models output at different scales; we enforce target size here.
        y = torch.nn.functional.interpolate(
            y.unsqueeze(0).unsqueeze(0),
            size=(TARGET_SIZE, TARGET_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    y = y.float()
    # BiRefNet returns logits: convert to matte probability.
    p = torch.sigmoid(y)

    if torch.isnan(p).any():
        raise RuntimeError("NaNs detected in predicted matte.")

    matte = p.detach().to("cpu").numpy().astype(np.float32, copy=False)
    matte = np.clip(matte, 0.0, 1.0)
    if matte.shape != (TARGET_SIZE, TARGET_SIZE):
        raise RuntimeError(f"Unexpected matte shape after processing: {matte.shape}")
    return matte

