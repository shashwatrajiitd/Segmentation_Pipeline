from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from .clip_router import CLIPRouter
from .contracts import ClassificationResult, ImageMetadata, PreprocessResult
from .gemini_router import validate_route
from .gemini_validator import validate_generation
from .io import load_image, normalize_image_rgb, save_png, write_json
from .nano_preprocess import generate as nano_generate


@dataclass
class PipelineArtifacts:
    normalized_image_path: str
    metadata_json_path: str


_CLIP_ROUTER: Optional[CLIPRouter] = None


def _get_clip_router() -> CLIPRouter:
    global _CLIP_ROUTER
    if _CLIP_ROUTER is None:
        _CLIP_ROUTER = CLIPRouter()
    return _CLIP_ROUTER


def process_image_full(
    product_id: str,
    image_path: str,
    output_root: str = "output",
) -> Tuple[ClassificationResult, PreprocessResult, PipelineArtifacts]:
    """
    STRICT ORDER:
      1) Load image
      2) CLIPRouter.route -> route_initial
      3) If A: Gemini route validator (reroute -> B)
      4) If B: Nano generate + Gemini gen validator (fail -> flagged)
      5) Save normalized image
      6) Emit metadata JSON
    """
    output_root_p = Path(output_root)
    norm_dir = output_root_p / "normalized"
    meta_dir = output_root_p / "metadata"
    norm_path = str((norm_dir / f"{product_id}.png").resolve())
    meta_path = str((meta_dir / f"{product_id}.json").resolve())

    img0 = load_image(image_path)

    # 2) CLIP routing
    clip_router = _get_clip_router()
    route_initial, scores = clip_router.route(img0)

    route_final = route_initial
    gemini_decision = "skipped"

    # 3) If A: validate route with Gemini
    if route_initial == "A":
        decision = validate_route(img0)
        gemini_decision = decision
        if decision == "reroute":
            route_final = "B"

    classification = ClassificationResult(
        product_id=product_id,
        route_initial=route_initial,
        route_final=route_final,
        clip_scores=scores,
        gemini_decision=gemini_decision,
    )

    # 4) If B: normalize via Nano Banana + validate generation
    nano_used = False
    nano_validation = ""
    status: str = "ready"
    chosen: Image.Image = img0

    if route_final == "B":
        try:
            gen = nano_generate(img0)
            nano_used = True
            chosen = gen
            try:
                ok = validate_generation(img0, gen)
                nano_validation = "match" if ok else "mismatch"
                if not ok:
                    status = "flagged"
            except Exception as e:
                nano_validation = f"error: {type(e).__name__}"
                status = "flagged"
        except Exception as e:
            nano_used = False
            nano_validation = f"error: {type(e).__name__}"
            status = "flagged"
            chosen = img0

    # 5) Save normalized image (always)
    normalized = normalize_image_rgb(chosen)
    save_png(normalized, norm_path)

    preprocess = PreprocessResult(
        product_id=product_id,
        normalized_image_path=norm_path,
        nano_used=nano_used,
        nano_validation=nano_validation,
        status="ready" if status == "ready" else "flagged",
    )

    # 6) Emit metadata JSON (always)
    meta = ImageMetadata(
        classification=classification,
        preprocess=preprocess,
        source_path=str(Path(image_path).resolve()),
    )
    payload = meta.model_dump() if hasattr(meta, "model_dump") else meta.dict()
    write_json(meta_path, payload)

    return classification, preprocess, PipelineArtifacts(normalized_image_path=norm_path, metadata_json_path=meta_path)


def process_image(product_id: str, image_path: str, output_root: str = "output") -> PreprocessResult:
    """
    Public API per spec: returns PreprocessResult.
    """
    _, preprocess, _ = process_image_full(product_id, image_path, output_root=output_root)
    return preprocess

