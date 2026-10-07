# ### Imports ###
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

# ### Constants ###
DEFAULT_DIRECTIONAL_LIGHT_COLOR = "#ffffff"
DEFAULT_DIRECTIONAL_LIGHT_INTENSITY = 1.0
DEFAULT_DIRECTIONAL_LIGHT_CAST_SHADOW = False
DEFAULT_DIRECTIONAL_LIGHT_TARGET_OFFSET = (0.0, 0.0, -1.0)
DIRECTIONAL_LIGHT_INTENSITY_WHEEL_STEP = 0.1
MAX_DIRECTIONAL_LIGHT_ID_LENGTH = 256
MAX_DIRECTIONAL_LIGHT_NAME_LENGTH = 256
_COLOR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")


# ### Data models ###
@dataclass(frozen=True, slots=True)
class DirectionalLightData:
    """One scene-global directional light authored in HouseMaker Z-up space."""

    light_id: str
    name: str
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    color: str = DEFAULT_DIRECTIONAL_LIGHT_COLOR
    intensity: float = DEFAULT_DIRECTIONAL_LIGHT_INTENSITY
    cast_shadow: bool = DEFAULT_DIRECTIONAL_LIGHT_CAST_SHADOW
    level_index: int | None = None

    def __post_init__(self) -> None:
        light_id = _normalize_required_text(
            self.light_id,
            "Directional-light ID",
            MAX_DIRECTIONAL_LIGHT_ID_LENGTH,
        )
        name = _normalize_required_text(
            self.name,
            "Directional-light name",
            MAX_DIRECTIONAL_LIGHT_NAME_LENGTH,
        )
        position = _normalize_world_point(
            self.position,
            "Directional-light position",
        )
        target = _normalize_world_point(
            self.target,
            "Directional-light target",
        )
        if math.dist(position, target) <= 1e-9:
            raise ValueError(
                "Directional-light position and target must be different."
            )
        color = str(self.color).strip().lower()
        if _COLOR_PATTERN.fullmatch(color) is None:
            raise ValueError("Directional-light color must use #RRGGBB format.")
        if isinstance(self.intensity, bool):
            raise TypeError("Directional-light intensity must be numeric.")
        intensity = float(self.intensity)
        if not math.isfinite(intensity) or intensity < 0.0:
            raise ValueError(
                "Directional-light intensity must be finite and non-negative."
            )
        if not isinstance(self.cast_shadow, bool):
            raise TypeError("Directional-light cast-shadow value must be boolean.")
        level_index = _normalize_optional_level_index(self.level_index)
        object.__setattr__(self, "light_id", light_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "level_index", level_index)

    def to_dict(self) -> dict[str, object]:
        """Serialize one light for a HouseMaker project file."""

        payload: dict[str, object] = {
            "light_id": self.light_id,
            "name": self.name,
            "position": list(self.position),
            "target": list(self.target),
            "color": self.color,
            "intensity": self.intensity,
            "cast_shadow": self.cast_shadow,
        }
        if self.level_index is not None:
            payload["level_index"] = self.level_index
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> DirectionalLightData:
        """Deserialize and validate one project light record."""

        if not isinstance(payload, dict):
            raise TypeError("A directional light must be an object.")
        return cls(
            light_id=str(payload["light_id"]),
            name=str(payload["name"]),
            position=_normalize_world_point(
                payload["position"],
                "Directional-light position",
            ),
            target=_normalize_world_point(
                payload["target"],
                "Directional-light target",
            ),
            color=str(
                payload.get("color", DEFAULT_DIRECTIONAL_LIGHT_COLOR)
            ),
            intensity=payload.get(
                "intensity",
                DEFAULT_DIRECTIONAL_LIGHT_INTENSITY,
            ),
            cast_shadow=payload.get(
                "cast_shadow",
                DEFAULT_DIRECTIONAL_LIGHT_CAST_SHADOW,
            ),
            level_index=payload.get("level_index"),
        )

    def translated(self, offset: Sequence[float]) -> DirectionalLightData:
        """Translate the light point and target without changing direction."""

        delta = _normalize_world_point(offset, "Directional-light translation")
        return replace(
            self,
            position=tuple(
                coordinate + delta[index]
                for index, coordinate in enumerate(self.position)
            ),
            target=tuple(
                coordinate + delta[index]
                for index, coordinate in enumerate(self.target)
            ),
        )

    def with_position(
        self,
        position: Sequence[float],
    ) -> DirectionalLightData:
        """Move the light point while preserving its target offset."""

        normalized_position = _normalize_world_point(
            position,
            "Directional-light position",
        )
        return self.translated(
            tuple(
                normalized_position[index] - coordinate
                for index, coordinate in enumerate(self.position)
            )
        )

    def with_intensity(self, intensity: object) -> DirectionalLightData:
        """Return a copy with a raw Three/R3F light intensity value."""

        return replace(self, intensity=intensity)

    def adjust_intensity(self, wheel_steps: object) -> DirectionalLightData:
        """Adjust intensity by standard wheel notches and clamp it at zero."""

        if isinstance(wheel_steps, bool):
            raise TypeError("Directional-light wheel steps must be numeric.")
        normalized_steps = float(wheel_steps)
        if not math.isfinite(normalized_steps):
            raise ValueError("Directional-light wheel steps must be finite.")
        adjusted = max(
            0.0,
            self.intensity
            + normalized_steps * DIRECTIONAL_LIGHT_INTENSITY_WHEEL_STEP,
        )
        return self.with_intensity(round(adjusted, 10))


# ### Collection helpers ###
def create_directional_light(
    existing_lights: Iterable[DirectionalLightData],
    *,
    position: Sequence[float],
    target: Sequence[float] | None = None,
    level_index: int | None = None,
) -> DirectionalLightData:
    """Create the next incrementally named point-placed light."""

    normalized_lights = _normalize_directional_lights(existing_lights)
    occupied_ids = {light.light_id for light in normalized_lights}
    light_number = 1
    while f"directional-light-{light_number}" in occupied_ids:
        light_number += 1
    normalized_position = _normalize_world_point(
        position,
        "Directional-light position",
    )
    normalized_target = (
        _normalize_world_point(target, "Directional-light target")
        if target is not None
        else tuple(
            normalized_position[index] + offset
            for index, offset in enumerate(
                DEFAULT_DIRECTIONAL_LIGHT_TARGET_OFFSET
            )
        )
    )
    return DirectionalLightData(
        light_id=f"directional-light-{light_number}",
        name=f"Directional light {light_number}",
        position=normalized_position,
        target=normalized_target,
        level_index=level_index,
    )


def directional_lights_to_dicts(
    lights: Iterable[DirectionalLightData],
) -> list[dict[str, object]]:
    """Serialize a validated directional-light collection for project JSON."""

    return [light.to_dict() for light in _normalize_directional_lights(lights)]


def directional_lights_from_payload(
    payload: object,
) -> tuple[DirectionalLightData, ...]:
    """Load valid unique lights while isolating malformed optional entries."""

    if not isinstance(payload, list | tuple):
        return ()
    lights: list[DirectionalLightData] = []
    occupied_ids: set[str] = set()
    for raw_light in payload:
        try:
            light = DirectionalLightData.from_dict(raw_light)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if light.light_id in occupied_ids:
            continue
        lights.append(light)
        occupied_ids.add(light.light_id)
    return tuple(lights)


def directional_lights_to_runtime_dicts(
    lights: Iterable[DirectionalLightData],
) -> list[dict[str, object]]:
    """Serialize lights in the glTF Y-up coordinates used by the R3F scene."""

    return [
        {
            "id": light.light_id,
            "name": light.name,
            "position": _world_point_to_gltf(light.position),
            "target": _world_point_to_gltf(light.target),
            "color": light.color,
            "intensity": light.intensity,
            "castShadow": light.cast_shadow,
        }
        for light in _normalize_directional_lights(lights)
    ]


# ### Validation helpers ###
def _normalize_directional_lights(
    lights: Iterable[DirectionalLightData],
) -> tuple[DirectionalLightData, ...]:
    if isinstance(lights, (str, bytes, bytearray)):
        raise TypeError("Directional lights must contain light records.")
    normalized = tuple(lights)
    if not all(isinstance(light, DirectionalLightData) for light in normalized):
        raise TypeError("Directional lights must contain DirectionalLightData values.")
    light_ids = tuple(light.light_id for light in normalized)
    if len(light_ids) != len(set(light_ids)):
        raise ValueError("Directional-light IDs must be unique.")
    return normalized


def _normalize_required_text(value: object, label: str, maximum: int) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} cannot be empty.")
    if len(normalized) > maximum:
        raise ValueError(f"{label} is too long.")
    return normalized


def _normalize_optional_level_index(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Directional-light level index must be an integer.")
    if value < 0:
        raise ValueError("Directional-light level index must be non-negative.")
    return value


def _normalize_world_point(
    value: object,
    label: str,
) -> tuple[float, float, float]:
    if (
        isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise ValueError(f"{label} must contain three coordinates.")
    coordinates = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in coordinates):
        raise ValueError(f"{label} coordinates must be finite.")
    return tuple(
        0.0 if math.isclose(component, 0.0, abs_tol=1e-12) else component
        for component in coordinates
    )


# ### Coordinate conversion helpers ###
def _world_point_to_gltf(
    point: tuple[float, float, float],
) -> list[float]:
    x_value, y_value, z_value = point
    return [
        0.0 if math.isclose(value, 0.0, abs_tol=1e-12) else float(value)
        for value in (x_value, z_value, -y_value)
    ]
