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
DEFAULT_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND = 2.5
MIN_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND = 0.1
MAX_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND = 20.0
FIRST_PERSON_MOVE_SPEED_STEP_METERS_PER_SECOND = 0.1


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


def normalize_first_person_speed_meters_per_second(value: object) -> float:
    """Return one finite first-person movement speed in supported bounds."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("First-person speed must be a number.")
    normalized = float(value)
    if not math.isfinite(normalized) or not (
        MIN_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND
        <= normalized
        <= MAX_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND
    ):
        raise ValueError(
            "First-person speed must be between "
            f"{MIN_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND} and "
            f"{MAX_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND} meters per "
            "second."
        )
    return round(normalized, 1)


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
