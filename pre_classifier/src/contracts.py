from __future__ import annotations

from typing import Dict, Literal

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    product_id: str
    route_initial: Literal["A", "B"]
    route_final: Literal["A", "B"]
    clip_scores: Dict[str, float] = Field(default_factory=dict)
    gemini_decision: str = ""


class PreprocessResult(BaseModel):
    product_id: str
    normalized_image_path: str
    nano_used: bool
    nano_validation: str
    status: Literal["ready", "flagged"]


class ImageMetadata(BaseModel):
    """Convenience wrapper for metadata JSON emission."""

    classification: ClassificationResult
    preprocess: PreprocessResult
    source_path: str

