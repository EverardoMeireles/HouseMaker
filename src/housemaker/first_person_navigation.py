# ### Imports ###
import math


# ### Constants ###
FIRST_PERSON_NAVIGATION_MODE_GRAVITY = "gravity"
FIRST_PERSON_NAVIGATION_MODE_NOCLIP = "noclip"
FIRST_PERSON_NAVIGATION_MODES = frozenset(
    (
        FIRST_PERSON_NAVIGATION_MODE_GRAVITY,
        FIRST_PERSON_NAVIGATION_MODE_NOCLIP,
    )
)
FIRST_PERSON_NAVIGATION_MODE_OPTIONS = (
    ("Gravity", FIRST_PERSON_NAVIGATION_MODE_GRAVITY),
    ("Noclip", FIRST_PERSON_NAVIGATION_MODE_NOCLIP),
)
DEFAULT_FIRST_PERSON_NAVIGATION_MODE = FIRST_PERSON_NAVIGATION_MODE_GRAVITY


# ### Normalization ###
def normalize_first_person_navigation_mode(value: object) -> str:
    """Return one validated first-person navigation mode."""

    if not isinstance(value, str):
        raise ValueError("First-person navigation mode must be a string.")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in FIRST_PERSON_NAVIGATION_MODES:
        supported = ", ".join(sorted(FIRST_PERSON_NAVIGATION_MODES))
        raise ValueError(
            "Unknown first-person navigation mode "
            f"{value!r}; expected one of: {supported}."
        )
    return normalized


# ### Movement vectors ###
def build_first_person_forward_vector(
    yaw_degrees: float,
    pitch_degrees: float,
    mode: str,
) -> tuple[float, float, float]:
    """Return the mode-specific unit vector for forward movement."""

    yaw_radians = math.radians(float(yaw_degrees))
    if normalize_first_person_navigation_mode(mode) == (
        FIRST_PERSON_NAVIGATION_MODE_NOCLIP
    ):
        pitch_radians = math.radians(float(pitch_degrees))
        horizontal_scale = math.cos(pitch_radians)
        return (
            horizontal_scale * math.cos(yaw_radians),
            horizontal_scale * math.sin(yaw_radians),
            math.sin(pitch_radians),
        )
    return math.cos(yaw_radians), math.sin(yaw_radians), 0.0
