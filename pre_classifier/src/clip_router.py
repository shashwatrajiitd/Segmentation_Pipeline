from __future__ import annotations

from typing import Dict, Tuple

import open_clip
import torch
from PIL import Image

from .config import (
    CLIP_MODEL,
    CLIP_PRETRAIN,
    CLIP_PROMPTS,
    CLIP_THRESH_COMPLEX,
    CLIP_THRESH_SIMPLE,
)


def _best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CLIPRouter:
    def __init__(self):
        self.device = _best_device()
        model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAIN)
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        # Fixed prompt ordering for deterministic output dict.
        self._prompt_keys = list(CLIP_PROMPTS.keys())
        self._prompt_texts = [CLIP_PROMPTS[k] for k in self._prompt_keys]
        with torch.no_grad():
            tokens = self.tokenizer(self._prompt_texts)
            self._text_features = self.model.encode_text(tokens.to(self.device))
            self._text_features = self._text_features / self._text_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def route(self, image: Image.Image) -> Tuple[str, Dict[str, float]]:
        """
        Returns:
          - route_initial: "A" or "B"
          - scores: dict[prompt_key] -> softmax probability in [0, 1]
        """
        img_in = image.convert("RGB")
        img_tensor = self.preprocess(img_in).unsqueeze(0).to(self.device)

        image_features = self.model.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # CLIP-style logits -> softmax over the prompt set (probabilities sum to 1).
        logit_scale = self.model.logit_scale.exp()
        logits = (image_features @ self._text_features.T) * logit_scale
        probs = logits.softmax(dim=-1).squeeze(0)  # (num_prompts,)
        scores = {k: float(probs[i].item()) for i, k in enumerate(self._prompt_keys)}

        if scores["simple"] > CLIP_THRESH_SIMPLE and max(
            scores["complex"],
            scores["hand"],
            scores["props"],
        ) < CLIP_THRESH_COMPLEX:
            route = "A"
        else:
            route = "B"

        return route, scores

