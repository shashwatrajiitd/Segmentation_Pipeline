from __future__ import annotations

import os
from typing import Any

import torch


def get_device() -> torch.device:
    """
    MPS-only per requirements.
    """
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "torch.backends.mps.is_available() is False. "
            "Install a PyTorch build with MPS support and run on Apple Silicon."
        )
    return torch.device("mps")


def load_torchscript_matting_model(model_path: str, device: torch.device | None = None) -> torch.nn.Module:
    """
    Load a TorchScript matting/segmentation model for local inference.

    Implementation note:
    - This loader expects a TorchScript module saved via torch.jit.save (extension can be .pth).
    - Pure state_dict checkpoints require the original model code; we intentionally avoid embedding
      third-party model source here to keep this repo clean and deterministic.
    """
    if device is None:
        device = get_device()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Always float32 (reject fp16).
    torch.set_default_dtype(torch.float32)

    try:
        # Register torchvision custom TorchScript ops (e.g. deform_conv2d) before loading.
        # If torchvision isn't imported, `torch.jit.load` can fail with:
        # "Unknown builtin op: torchvision::deform_conv2d".
        import torchvision  # noqa: F401

        # Load on CPU first, then explicitly cast to float32.
        # Some TorchScript archives contain float64 tensor attributes; moving float64 to MPS will fail.
        model = torch.jit.load(model_path, map_location="cpu")
    except Exception as e:  # noqa: BLE001 - surface a helpful error
        raise RuntimeError(
            "Failed to load model. This pipeline expects a TorchScript matting model "
            "saved with torch.jit.save(). If you have a state_dict .pth, you must export it "
            "to TorchScript first."
        ) from e

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Enforce float32 for all parameters/buffers/attributes that support casting.
    model = model.to(dtype=torch.float32)

    # Still call `.to(device)` for parameters/buffers.
    model.to(device)
    return model


# BiRefNet: eager HF loader (works on MPS; TorchScript CPU-trace introduces CPU constants).
def load_birefnet_hf(hf_repo: str = "ZhengPeng7/BiRefNet", device: torch.device | None = None) -> torch.nn.Module:
    """
    Load BiRefNet via Hugging Face transformers (trust_remote_code) for MPS inference.

    Notes:
    - We keep float32 only.
    - We disable meta-device init paths to avoid `.item()` on meta tensors.
    - Output is handled by inference.py (expects logits-like tensor in the final stage).
    """
    if device is None:
        device = get_device()

    try:
        from transformers import AutoModelForImageSegmentation
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("transformers is not installed. Run: pip install -r requirements.txt") from e

    torch.set_default_dtype(torch.float32)
    model = AutoModelForImageSegmentation.from_pretrained(
        hf_repo,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        _fast_init=False,
        device_map=None,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(dtype=torch.float32).to(device)
    return model


# Backwards-compatible name (older pipeline versions).
load_inspyrenet = load_torchscript_matting_model


def forward_model(model: torch.nn.Module, x: torch.Tensor) -> Any:
    """
    Run forward pass (kept separate so inference.py can remain simple).
    """
    with torch.no_grad():
        return model(x)

