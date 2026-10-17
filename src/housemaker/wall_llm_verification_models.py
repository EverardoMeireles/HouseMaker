# ### Constants ###
DEFAULT_WALL_VERIFICATION_MODEL = "gpt-5.6-luna"
WALL_VERIFICATION_MODEL_OPTIONS = (
    ("GPT-5.6 Luna", DEFAULT_WALL_VERIFICATION_MODEL),
    ("GPT-5.6 Terra", "gpt-5.6-terra"),
    ("Qwen3-VL 8B (Local)", "qwen3-vl-8b-local"),
)
WALL_VERIFICATION_MODELS = frozenset(
    model_id for _label, model_id in WALL_VERIFICATION_MODEL_OPTIONS
)
