from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image


@dataclass
class _FakeCLIPRouter:
    route_initial: str

    def route(self, image: Image.Image):
        # scores shape matches what pipeline expects to serialize in metadata
        scores = {"simple": 0.9, "complex": 0.05, "hand": 0.03, "props": 0.02}
        return self.route_initial, scores


def _write_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 24), (200, 100, 50))
    img.save(str(path), format="PNG")


def test_classification_route_b_skips_gemini_and_runs_nano_path(monkeypatch, tmp_path: Path):
    """
    Classification-only intent:
    - If CLIP routes B, we should not call route validator.
    - Pipeline still goes through preprocess steps; mock Nano + validator to avoid network/model calls.
    """
    from src import pipeline as pipeline_mod

    # Avoid global singleton reuse across tests.
    pipeline_mod._CLIP_ROUTER = None
    monkeypatch.setattr(pipeline_mod, "_get_clip_router", lambda: _FakeCLIPRouter(route_initial="B"))

    called = {"validate_route": 0}

    def _validate_route(_img):
        called["validate_route"] += 1
        return "pass"

    # Important: pipeline imports these symbols directly, so patch on pipeline module.
    monkeypatch.setattr(pipeline_mod, "validate_route", _validate_route)
    monkeypatch.setattr(pipeline_mod, "nano_generate", lambda img: img)
    monkeypatch.setattr(pipeline_mod, "validate_generation", lambda _o, _g: True)

    img_path = tmp_path / "in" / "p.png"
    _write_dummy_image(img_path)

    classification, preprocess, _artifacts = pipeline_mod.process_image_full(
        product_id="p1",
        image_path=str(img_path),
        output_root=str(tmp_path / "out"),
    )

    assert classification.route_initial == "B"
    assert classification.route_final == "B"
    assert classification.gemini_decision == "skipped"
    assert called["validate_route"] == 0

    assert preprocess.status == "ready"
    assert preprocess.nano_used is True


def test_classification_route_a_gemini_pass_keeps_a(monkeypatch, tmp_path: Path):
    from src import pipeline as pipeline_mod

    pipeline_mod._CLIP_ROUTER = None
    monkeypatch.setattr(pipeline_mod, "_get_clip_router", lambda: _FakeCLIPRouter(route_initial="A"))
    monkeypatch.setattr(pipeline_mod, "validate_route", lambda _img: "pass")

    img_path = tmp_path / "in" / "p.png"
    _write_dummy_image(img_path)

    classification, preprocess, _artifacts = pipeline_mod.process_image_full(
        product_id="p2",
        image_path=str(img_path),
        output_root=str(tmp_path / "out"),
    )

    assert classification.route_initial == "A"
    assert classification.route_final == "A"
    assert classification.gemini_decision == "pass"
    assert preprocess.status == "ready"
    assert preprocess.nano_used is False


def test_classification_route_a_gemini_reroute_switches_to_b(monkeypatch, tmp_path: Path):
    from src import pipeline as pipeline_mod

    pipeline_mod._CLIP_ROUTER = None
    monkeypatch.setattr(pipeline_mod, "_get_clip_router", lambda: _FakeCLIPRouter(route_initial="A"))
    monkeypatch.setattr(pipeline_mod, "validate_route", lambda _img: "reroute")
    monkeypatch.setattr(pipeline_mod, "nano_generate", lambda img: img)
    monkeypatch.setattr(pipeline_mod, "validate_generation", lambda _o, _g: True)

    img_path = tmp_path / "in" / "p.png"
    _write_dummy_image(img_path)

    classification, preprocess, _artifacts = pipeline_mod.process_image_full(
        product_id="p3",
        image_path=str(img_path),
        output_root=str(tmp_path / "out"),
    )

    assert classification.route_initial == "A"
    assert classification.route_final == "B"
    assert classification.gemini_decision == "reroute"
    assert preprocess.status == "ready"
    assert preprocess.nano_used is True

