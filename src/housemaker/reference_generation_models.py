"""Lightweight choices for generation-reference image providers."""

# ### Imports ###
from housemaker.plan_correction_models import (
    DEFAULT_PLAN_CORRECTION_MODEL,
    PLAN_CORRECTION_MODEL_OPTIONS,
)

# ### Model identifiers ###
REFERENCE_GENERATION_MODEL_SETTINGS_KEY = "generation/reference_generation_model"
REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION = "stable-diffusion-xl-img2img"
REFERENCE_GENERATION_MODEL_MESHY = "meshy-image-to-image"
REFERENCE_GENERATION_MODEL_OPTIONS = (
    *PLAN_CORRECTION_MODEL_OPTIONS,
    ("Meshy (Nano Banana)", REFERENCE_GENERATION_MODEL_MESHY),
    ("Stable Diffusion XL (img2img, Local)", REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION),
)
REFERENCE_GENERATION_MODELS = frozenset(
    model_id for _label, model_id in REFERENCE_GENERATION_MODEL_OPTIONS
)
DEFAULT_REFERENCE_GENERATION_MODEL = DEFAULT_PLAN_CORRECTION_MODEL

# ### Default SDXL guidance ###
DEFAULT_REFERENCE_NEGATIVE_PROMPT = (
    "side view, three-quarter view, tilted camera, perspective distortion, "
    "blurry, low resolution, jpeg artifacts, warped proportions, deformed geometry, "
    "cropped object, duplicate objects, extra furniture, room interior, wall, floor, "
    "clutter, mask outline, irregular cutout border, collage, multiple views, text, watermark"
)
