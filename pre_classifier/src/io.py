from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

from .config import IMAGE_RESOLUTION


def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img


def _flatten_alpha_to_white(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        comp = Image.alpha_composite(bg, rgba)
        return comp.convert("RGB")
    return img.convert("RGB")


def normalize_image_rgb(img: Image.Image, resolution: int = IMAGE_RESOLUTION) -> Image.Image:
    """
    Enforce:
    - RGB
    - ~1K resolution (longest side == resolution)
    """
    img = _flatten_alpha_to_white(img)
    w, h = img.size
    if max(w, h) == resolution:
        return img

    scale = resolution / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def save_png(img: Image.Image, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(p), format="PNG", optimize=True)


def image_to_base64_png(img: Image.Image) -> str:
    img = _flatten_alpha_to_white(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def write_json(path: str, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def safe_product_id_from_relpath(relpath: str) -> str:
    """
    Make a stable, filesystem-safe product_id from a relative path.
    Example: "foo/bar/img 1.png" -> "foo__bar__img_1"
    """
    p = Path(relpath)
    stem = p.with_suffix("").as_posix()
    stem = stem.replace("/", "__")
    stem = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in stem)
    return stem


def pil_to_numpy_rgb(img: Image.Image) -> np.ndarray:
    img = _flatten_alpha_to_white(img)
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image array, got shape={arr.shape}")
    return arr

