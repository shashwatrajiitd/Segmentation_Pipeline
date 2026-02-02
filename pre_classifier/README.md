# pre_classifier

Classification + pre-processing layer that runs **before** BiRefNet background removal.

## What it does

- Routes each image into **Group A** (simple, ready) or **Group B** (complex, needs normalization).
- Validates **Group A** with Gemini (can reroute to B).
- For **Group B**, generates a normalized image (white background) via a Nano Banana / image model.
- Validates the generation vs the original with Gemini.
- Writes artifacts:
  - `output/normalized/*.png`
  - `output/metadata/*.json`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

Create a `.env` (or export env vars) with at least:

- `GEMINI_API_KEY`: Gemini API key
- `GEMINI_BASE_URL` (optional): defaults to `https://generativelanguage.googleapis.com`
- `GEMINI_TIMEOUT_S` (optional): defaults to 12

## Run (batch)

```bash
python run.py --input images/ --output output/ --handoff-dir ../handoff/to_bg_removal --emit-ready-only
```

Outputs are always written to:

- `output/normalized/`
- `output/metadata/`

Additionally, ready images are emitted to:

- `../handoff/to_bg_removal/images/<product_id>.png`
- `../handoff/to_bg_removal/manifest.jsonl` (append-only audit log)

## Run background removal (no code changes)

```bash
cd ../bg_removal
python run.py \
  --input ../handoff/to_bg_removal/images \
  --output ../final_output/rgba
```

