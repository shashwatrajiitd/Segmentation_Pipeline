from __future__ import annotations

import base64
import io
import os
from typing import Optional

import requests
from PIL import Image

from .config import MAX_GEN_RETRIES, NANO_IMAGE_MODEL, NANO_PROMPT
from .io import image_to_base64_png


def _get_base_url() -> str:
    return os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")


def _get_timeout_s() -> float:
    try:
        return float(os.getenv("GEMINI_TIMEOUT_S", "30"))
    except ValueError:
        return 30.0


def _decode_image_part(part: dict) -> Optional[Image.Image]:
    """
    Best-effort decode of a Gemini image output part.
    """
    inline = part.get("inlineData") if isinstance(part, dict) else None
    if not isinstance(inline, dict):
        return None
    mime = inline.get("mimeType", "")
    data = inline.get("data")
    if not (isinstance(data, str) and data):
        return None
    if not mime.startswith("image/"):
        return None
    raw = base64.b64decode(data)
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img.convert("RGB")


def _gemini_generate_image_once(image: Image.Image) -> Image.Image:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    url = f"{_get_base_url()}/v1beta/models/{NANO_IMAGE_MODEL}:generateContent"
    b64 = image_to_base64_png(image)
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": NANO_PROMPT},
                    {"inlineData": {"mimeType": "image/png", "data": b64}},
                ],
            }
        ]
    }
    resp = requests.post(url, params={"key": api_key}, json=payload, timeout=_get_timeout_s())
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("No candidates returned from image generation model")
    content = (candidates[0] or {}).get("content") or {}
    parts = content.get("parts") or []

    # Find first image part.
    for p in parts:
        img = _decode_image_part(p)
        if img is not None:
            return img

    raise RuntimeError("No image payload found in model response")


def generate(image: Image.Image) -> Image.Image:
    """
    Calls image generation API. Retries up to MAX_GEN_RETRIES.
    """
    last_err: Optional[Exception] = None
    for _ in range(MAX_GEN_RETRIES + 1):
        try:
            return _gemini_generate_image_once(image)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Image generation failed after retries: {last_err}")

