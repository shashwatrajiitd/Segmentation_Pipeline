## Segmentation Pipeline (pre_classifier → bg_removal)

This repo wires `pre_classifier` output into `bg_removal` via a filesystem handoff directory.

### End-to-end (copy/paste)

```bash
# Step 1: Pre-classification (writes normalized images + metadata + handoff artifacts)
cd Segmentation_Pipeline/pre_classifier
python run.py \
  --input ../raw_images \
  --output ./output \
  --handoff-dir ../handoff/to_bg_removal \
  --emit-ready-only

# Step 2: Background removal (runs unchanged)
cd ../bg_removal
python run.py \
  --input ../handoff/to_bg_removal/images \
  --output ../final_output/rgba
```

### Handoff artifacts

- `handoff/to_bg_removal/images/<product_id>.png` (ready-only, stable names)
- `handoff/to_bg_removal/manifest.jsonl` (append-only audit log)

### Optional reconciliation

```bash
python scripts/reconcile_results.py \
  --manifest handoff/to_bg_removal/manifest.jsonl \
  --rgba-dir final_output/rgba
```

