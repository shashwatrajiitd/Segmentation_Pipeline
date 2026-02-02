from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.io import safe_product_id_from_relpath
from src.pipeline import process_image_full


def _iter_images(input_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Pre-classification + normalization (pre-BiRefNet).")
    parser.add_argument("--input", required=True, type=str, help="Input directory containing images.")
    parser.add_argument("--output", required=True, type=str, help="Output directory (will create normalized/ + metadata/).")
    parser.add_argument("--handoff-dir", type=str, required=True, help="Handoff root dir (creates images/ + manifest.jsonl).")
    parser.add_argument("--emit-ready-only", action="store_true", help="Only emit ready images to handoff.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    handoff_arg = Path(args.handoff_dir)

    # Allow passing either the handoff root or the images subdir.
    if handoff_arg.name == "images":
        handoff_root = handoff_arg.parent
        handoff_images_dir = handoff_arg
    else:
        handoff_root = handoff_arg
        handoff_images_dir = handoff_root / "images"

    handoff_images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = handoff_root / "manifest.jsonl"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    images = list(_iter_images(input_dir))
    if not images:
        print(f"No images found under {input_dir}")
        return 0

    stats = {
        "total": 0,
        "route_initial_A": 0,
        "route_initial_B": 0,
        "route_final_A": 0,
        "route_final_B": 0,
        "flagged": 0,
        "nano_used": 0,
    }

    t0 = time.perf_counter()
    with open(manifest_path, "a", encoding="utf-8") as manifest_fp:
        for img_path in tqdm(images, desc="Pre-classifying", unit="img"):
            rel = img_path.relative_to(input_dir).as_posix()
            product_id = safe_product_id_from_relpath(rel)

            classification, preprocess, _ = process_image_full(
                product_id=product_id,
                image_path=str(img_path),
                output_root=str(output_dir),
            )

            record = {
                "product_id": product_id,
                "source_image": str(img_path),
                "route": classification.route_final,
                "nano_used": preprocess.nano_used,
                "status": preprocess.status,
            }
            manifest_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest_fp.flush()

            stats["total"] += 1
            stats[f"route_initial_{classification.route_initial}"] += 1
            stats[f"route_final_{classification.route_final}"] += 1
            if preprocess.status == "flagged":
                stats["flagged"] += 1
            if preprocess.nano_used:
                stats["nano_used"] += 1

            # Validation rule: do NOT emit non-ready images to handoff.
            if preprocess.status != "ready":
                continue

            # Emit stable filename: <product_id>.png (retry-safe overwrite).
            if args.emit_ready_only:
                shutil.copy(
                    preprocess.normalized_image_path,
                    handoff_images_dir / f"{product_id}.png",
                )
            else:
                shutil.copy(
                    preprocess.normalized_image_path,
                    handoff_images_dir / f"{product_id}.png",
                )

    t1 = time.perf_counter()
    print(
        "Done.\n"
        f"- total: {stats['total']}\n"
        f"- route_initial: A={stats['route_initial_A']} B={stats['route_initial_B']}\n"
        f"- route_final:   A={stats['route_final_A']} B={stats['route_final_B']}\n"
        f"- nano_used: {stats['nano_used']}\n"
        f"- flagged:   {stats['flagged']} ({(stats['flagged'] / max(1, stats['total'])) * 100:.2f}%)\n"
        f"- elapsed_s: {t1 - t0:.2f}\n"
        f"- output: {output_dir.resolve()}\n"
        f"- handoff_images: {handoff_images_dir.resolve()}\n"
        f"- manifest: {manifest_path.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

