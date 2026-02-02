from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class _FakeCLIPRouter:
    def route(self, image: Image.Image):
        # Force Group B
        scores = {"simple": 0.1, "complex": 0.7, "hand": 0.1, "props": 0.1}
        return "B", scores


def _write_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 24), (10, 20, 30))
    img.save(str(path), format="PNG")


class _FakeResp:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_group_b_nano_generate_and_validator_match_yes(monkeypatch, tmp_path: Path):
    """
    End-to-end (mocked HTTP):
    - Force Group B route
    - Nano generation returns an image part (inlineData)
    - Validator returns {"match":"yes"}
    => preprocess.status == "ready", nano_used == True, nano_validation == "match"
    """
    from src import gemini_validator as gemini_validator_mod
    from src import nano_preprocess as nano_preprocess_mod
    from src import pipeline as pipeline_mod

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    pipeline_mod._CLIP_ROUTER = None
    monkeypatch.setattr(pipeline_mod, "_get_clip_router", lambda: _FakeCLIPRouter())

    gen_img = Image.new("RGB", (64, 48), (200, 0, 0))
    gen_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"inlineData": {"mimeType": "image/png", "data": _b64_png(gen_img)}},
                    ]
                }
            }
        ]
    }
    validate_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "```json\n{\n  \"match\": \"yes\",\n  \"issues\": []\n}\n```",
                        }
                    ]
                }
            }
        ]
    }

    def _fake_post(url, params=None, json=None, timeout=None):
        # nano_preprocess and gemini_validator both call /generateContent; distinguish by model name in URL
        if f"/models/{nano_preprocess_mod.NANO_IMAGE_MODEL}:generateContent" in url:
            return _FakeResp(gen_payload)
        if f"/models/{gemini_validator_mod.GEMINI_GEN_MODEL}:generateContent" in url:
            return _FakeResp(validate_payload)
        raise AssertionError(f"Unexpected URL: {url}")

    # Patch requests.post in both modules (they each import requests independently).
    monkeypatch.setattr(nano_preprocess_mod.requests, "post", _fake_post)
    monkeypatch.setattr(gemini_validator_mod.requests, "post", _fake_post)

    img_path = tmp_path / "in" / "p.png"
    _write_dummy_image(img_path)

    classification, preprocess, _artifacts = pipeline_mod.process_image_full(
        product_id="b_yes",
        image_path=str(img_path),
        output_root=str(tmp_path / "out"),
    )

    assert classification.route_initial == "B"
    assert classification.route_final == "B"
    assert preprocess.nano_used is True
    assert preprocess.nano_validation == "match"
    assert preprocess.status == "ready"


def test_group_b_validator_mismatch_flags(monkeypatch, tmp_path: Path):
    """
    If validator returns match == "no", pipeline should mark preprocess.status == "flagged".
    """
    from src import gemini_validator as gemini_validator_mod
    from src import nano_preprocess as nano_preprocess_mod
    from src import pipeline as pipeline_mod

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    pipeline_mod._CLIP_ROUTER = None
    monkeypatch.setattr(pipeline_mod, "_get_clip_router", lambda: _FakeCLIPRouter())

    gen_img = Image.new("RGB", (64, 48), (0, 200, 0))
    gen_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"inlineData": {"mimeType": "image/png", "data": _b64_png(gen_img)}},
                    ]
                }
            }
        ]
    }
    validate_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "{ \"match\": \"no\", \"issues\": [\"different product\"] }"},
                    ]
                }
            }
        ]
    }

    def _fake_post(url, params=None, json=None, timeout=None):
        if f"/models/{nano_preprocess_mod.NANO_IMAGE_MODEL}:generateContent" in url:
            return _FakeResp(gen_payload)
        if f"/models/{gemini_validator_mod.GEMINI_GEN_MODEL}:generateContent" in url:
            return _FakeResp(validate_payload)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(nano_preprocess_mod.requests, "post", _fake_post)
    monkeypatch.setattr(gemini_validator_mod.requests, "post", _fake_post)

    img_path = tmp_path / "in" / "p.png"
    _write_dummy_image(img_path)

    _classification, preprocess, _artifacts = pipeline_mod.process_image_full(
        product_id="b_no",
        image_path=str(img_path),
        output_root=str(tmp_path / "out"),
    )

    assert preprocess.nano_used is True
    assert preprocess.nano_validation == "mismatch"
    assert preprocess.status == "flagged"

