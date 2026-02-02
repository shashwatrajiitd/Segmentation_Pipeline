from __future__ import annotations

import json
import os
from typing import Any, Dict, Literal, Optional

import requests
from PIL import Image

from .config import GEMINI_ROUTE_MODEL, GEMINI_ROUTE_PROMPT
from .io import image_to_base64_png


def _get_base_url() -> str:
    return os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")


def _get_timeout_s() -> float:
    try:
        return float(os.getenv("GEMINI_TIMEOUT_S", "12"))
    except ValueError:
        return 12.0


def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Gemini sometimes wraps JSON in code fences; extract the first JSON object.
    """
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
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ]
    }
    resp = requests.post(url, params={"key": api_key}, json=payload, timeout=_get_timeout_s())
    resp.raise_for_status()
    data = resp.json()

    # Pull the first candidate text part, if present.
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    content = (candidates[0] or {}).get("content") or {}
    out_parts = content.get("parts") or []
    texts = [p.get("text", "") for p in out_parts if isinstance(p, dict) and "text" in p]
    return "\n".join([t for t in texts if t])


def validate_route(image: Image.Image) -> Literal["pass", "reroute"]:
    """
    Returns "pass" or "reroute".

    Timeout/network/parse failures MUST default to "reroute".
    """
    try:
        b64 = image_to_base64_png(image)
        text = _gemini_generate_content(
            GEMINI_ROUTE_MODEL,
            parts=[
                {"text": GEMINI_ROUTE_PROMPT},
                {"inlineData": {"mimeType": "image/png", "data": b64}},
            ],
        )
        obj = _extract_first_json(text)
        decision = (obj or {}).get("decision")
        if decision in ("pass", "reroute"):
            return decision
        return "reroute"
    except Exception:
        return "reroute"

