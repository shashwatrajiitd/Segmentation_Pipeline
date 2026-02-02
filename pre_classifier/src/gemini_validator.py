from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests
from PIL import Image

from .config import GEMINI_GEN_MODEL, GEMINI_GEN_VALIDATE_PROMPT
from .io import image_to_base64_png


def _get_base_url() -> str:
    return os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")


def _get_timeout_s() -> float:
    try:
        return float(os.getenv("GEMINI_TIMEOUT_S", "12"))
    except ValueError:
        return 12.0


def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


def _gemini_generate_content(model: str, parts: list[dict]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    url = f"{_get_base_url()}/v1beta/models/{model}:generateContent"
    payload = {"contents": [{"role": "user", "parts": parts}]}
    resp = requests.post(url, params={"key": api_key}, json=payload, timeout=_get_timeout_s())
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    content = (candidates[0] or {}).get("content") or {}
    out_parts = content.get("parts") or []
    texts = [p.get("text", "") for p in out_parts if isinstance(p, dict) and "text" in p]
    return "\n".join([t for t in texts if t])


def validate_generation(orig: Image.Image, gen: Image.Image) -> bool:
    """
    Returns True iff Gemini replies match == "yes".
    """
    b64_orig = image_to_base64_png(orig)
    b64_gen = image_to_base64_png(gen)
    text = _gemini_generate_content(
        GEMINI_GEN_MODEL,
        parts=[
            {"text": GEMINI_GEN_VALIDATE_PROMPT},
            {"text": "ORIGINAL:"},
            {"inlineData": {"mimeType": "image/png", "data": b64_orig}},
            {"text": "GENERATED:"},
            {"inlineData": {"mimeType": "image/png", "data": b64_gen}},
        ],
    )
    obj = _extract_first_json(text) or {}
    return obj.get("match") == "yes"

