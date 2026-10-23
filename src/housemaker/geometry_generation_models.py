"""Lightweight geometry model choices shared by the UI and Meshy client."""

# ### Model identifiers ###
GEOMETRY_MODEL_T2 = "meshy-t2"
GEOMETRY_MODEL_T2_5 = "meshy-t2.5"
GEOMETRY_MODEL_7_1 = "meshy-7.1"
GEOMETRY_MODEL_MESHY_7_1 = GEOMETRY_MODEL_7_1
GEOMETRY_MODEL_OPTIONS = (
    ("Meshy t2", GEOMETRY_MODEL_T2),
    ("Meshy t2.5", GEOMETRY_MODEL_T2_5),
    ("Meshy 7.1", GEOMETRY_MODEL_7_1),
)
DEFAULT_GEOMETRY_MODEL = GEOMETRY_MODEL_T2_5
GEOMETRY_RESOLUTION_OPTIONS = (
    ("Standard", "standard"),
    ("Ultra 2K", "2k"),
)
DEFAULT_GEOMETRY_RESOLUTION = "standard"
SMART_TOPOLOGY_MAX_POLYCOUNTS = {
    GEOMETRY_MODEL_T2: 15_000,
    GEOMETRY_MODEL_T2_5: 30_000,
}


# ### Validation helpers ###
def normalize_geometry_model(ai_model: str) -> str:
    """Accept explicit supported model IDs, never a changing provider alias."""

    if not isinstance(ai_model, str) or ai_model.strip() not in {
        model for _label, model in GEOMETRY_MODEL_OPTIONS
    }:
        raise ValueError(f"Unsupported Meshy geometry model: {ai_model!r}.")
    return ai_model.strip()


def normalize_geometry_resolution(geometry_resolution: str) -> str:
    """Limit the application to Standard and Ultra 2K geometry."""

    if not isinstance(geometry_resolution, str) or geometry_resolution.strip() not in {
        value for _label, value in GEOMETRY_RESOLUTION_OPTIONS
    }:
        raise ValueError("Meshy geometry resolution must be Standard or Ultra 2K.")
    return geometry_resolution.strip()


def maximum_geometry_target_polycount(ai_model: str) -> int:
    """Return the retained tris control's ceiling (unused by native Meshy 7.1)."""

    normalized = normalize_geometry_model(ai_model)
    return SMART_TOPOLOGY_MAX_POLYCOUNTS.get(normalized, 30_000)
