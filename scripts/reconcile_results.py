from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {path}") from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile bg_removal outputs against pre_classifier manifest.jsonl.")
    parser.add_argument("--manifest", required=True, type=str, help="Path to handoff/to_bg_removal/manifest.jsonl")
    parser.add_argument("--rgba-dir", required=True, type=str, help="Path to final_output/rgba directory")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rgba_dir = Path(args.rgba_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not rgba_dir.exists():
        raise FileNotFoundError(f"rgba dir not found: {rgba_dir}")

    for rec in _iter_jsonl(manifest_path):
        product_id = rec.get("product_id")
        status = rec.get("status")

        out_path = rgba_dir / f"{product_id}.png"
        if status != "ready":
            final_status = "not_ready"
            exists = False
        else:
            exists = out_path.exists()
            final_status = "segmented" if exists else "missing"

        print(
            json.dumps(
                {
                    **rec,
                    "expected_output": str(out_path),
                    "output_exists": exists,
                    "final_status": final_status,
                },
                ensure_ascii=False,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

