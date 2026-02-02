CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAIN = "openai"

CLIP_THRESH_SIMPLE = 0.65
CLIP_THRESH_COMPLEX = 0.45

GEMINI_ROUTE_MODEL = "gemini-2.5-flash"
GEMINI_GEN_MODEL = "gemini-2.5-flash"
NANO_IMAGE_MODEL = "gemini-3-pro-image-preview"

IMAGE_RESOLUTION = 1024

MAX_GEN_RETRIES = 1

# Versioned prompts (keep changes explicit + centralized).
CLIP_PROMPTS = {
    "simple": "a single product photographed on a plain white background",
    "complex": "a product with complex or colorful background",
    "hand": "a product being held by a person",
    "props": "multiple objects or props around a product",
}

GEMINI_ROUTE_PROMPT = """You are validating e-commerce product images.

Is this image a SINGLE product, clearly visible, with NO extra objects, NO hands,
NO props, and a SIMPLE plain background suitable for direct background removal?

Answer strictly in JSON:
{
  "decision": "pass" | "reroute",
  "reason": "short"
}
"""

NANO_PROMPT = """Extract ONLY the main product.
Remove all background, hands, props, and extra objects.
Recreate the product as-is on a clean, plain white background.
Do NOT change shape, proportions, branding, or details.
Studio lighting, neutral shadows. Keep the product as-it-is without any changes.
"""

GEMINI_GEN_VALIDATE_PROMPT = """Compare the two images.

Is the GENERATED image the SAME product as the ORIGINAL image?
Check shape, structure, components, and identity.
Ignore background differences.

Answer strictly in JSON:
{
  "match": "yes" | "no",
  "issues": []
}
"""

