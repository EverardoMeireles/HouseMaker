# ### Model identifiers ###
PLAN_CORRECTION_MODEL_GPT_IMAGE_2 = "gpt-image-2"
PLAN_CORRECTION_MODEL_GPT_5_6_LUNA = "gpt-5.6-luna"
PLAN_CORRECTION_MODEL_GPT_5_6_TERRA = "gpt-5.6-terra"
PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1 = "Qwen-Image-2.1"
PLAN_CORRECTION_MODEL_OPTIONS = (
    (PLAN_CORRECTION_MODEL_GPT_IMAGE_2, PLAN_CORRECTION_MODEL_GPT_IMAGE_2),
    ("GPT-5.6 Luna", PLAN_CORRECTION_MODEL_GPT_5_6_LUNA),
    ("GPT-5.6 Terra", PLAN_CORRECTION_MODEL_GPT_5_6_TERRA),
    (
        PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
        PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
    ),
)
PLAN_CORRECTION_MODELS = frozenset(
    model_id for _label, model_id in PLAN_CORRECTION_MODEL_OPTIONS
)
PLAN_CORRECTION_MODEL_LABELS = {
    model_id: label for label, model_id in PLAN_CORRECTION_MODEL_OPTIONS
}
OPENAI_PLAN_CORRECTION_MODELS = frozenset(
    {
        PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
        PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
        PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
    }
)
DEFAULT_PLAN_CORRECTION_MODEL = PLAN_CORRECTION_MODEL_GPT_IMAGE_2


# ### Display helpers ###
def plan_correction_model_label(model_id: str) -> str:
    """Return the stable user-facing name for a correction model."""

    return PLAN_CORRECTION_MODEL_LABELS.get(str(model_id), str(model_id))
