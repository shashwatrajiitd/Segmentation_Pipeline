from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from tqdm import tqdm

from src.pipeline import load_model_default, process_image


def _iter_images(input_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(description="Local SOD background removal (MPS, float32, batch=1).")
    parser.add_argument("--input", required=True, type=str, help="Input directory containing images.")
    parser.add_argument("--output", required=True, type=str, help="Output directory for RGBA PNGs.")
    parser.add_argument(
        "--model",
        default="hf:ZhengPeng7/BiRefNet",
        type=str,
        help="Model spec. Use 'hf:ZhengPeng7/BiRefNet' (default) or a TorchScript file path.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    model, device = load_model_default(args.model)

    images = list(_iter_images(input_dir))
    if not images:
        print(f"No images found under {input_dir}")
        return 0

    total0 = time.perf_counter()
    for img_path in tqdm(images, desc="Processing", unit="img"):
        rel = img_path.relative_to(input_dir)
        out_path = (output_dir / rel).with_suffix(".png")
        timings = process_image(str(img_path), str(out_path), model, device, fail_fast=True)

        # Simple per-image timing log (kept minimal and deterministic).
        print(
            f"{img_path.name}: total={timings.total_s:.3f}s "
            f"(pre={timings.preprocess_s:.3f}s inf={timings.inference_s:.3f}s "
            f"post={timings.postprocess_s:.3f}s comp={timings.composite_s:.3f}s)"
        )

    total1 = time.perf_counter()
    print(f"Done. {len(images)} images in {total1-total0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

