"""
Centralized configuration constants for the background removal pipeline.

Ground rules:
- MPS + float32
- Batch size 1
"""

# NOTE: BiRefNet internally splits into patches; this size must be divisible by
# its patching grid. 1088 is the closest "1080-class" square that works reliably.
TARGET_SIZE = 1088
PAD_COLOR = 127
EDGE_BLUR_RADIUS = 1
ALPHA_THRESHOLD = 0.05
CROP_PADDING = 50

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Optional matte cleanup step (helps remove thin white halo artifacts after restoration + LCCA).
# Set to 0 to disable.
ERODE_KERNEL_SIZE = 3

