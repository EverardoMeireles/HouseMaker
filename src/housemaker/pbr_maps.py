# ### Imports ###
from __future__ import annotations

from collections.abc import Sequence


# ### Map constants ###
ATLAS_MAP_BASE_COLOR = "base_color"
PBR_MAP_NORMAL = "normal"
PBR_MAP_ROUGHNESS = "roughness"
PBR_MAP_METALLIC = "metallic"
PBR_MAP_TYPES = (
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
    PBR_MAP_METALLIC,
)
ATLAS_MAP_TYPES = (ATLAS_MAP_BASE_COLOR, *PBR_MAP_TYPES)
ATLAS_MAP_LABELS = {
    ATLAS_MAP_BASE_COLOR: "Base color",
    PBR_MAP_NORMAL: "Normal",
    PBR_MAP_ROUGHNESS: "Roughness",
    PBR_MAP_METALLIC: "Metallic",
}


# ### Validation helpers ###
def normalize_pbr_map_types(
    values: Sequence[str],
    *,
    label: str = "PBR maps",
) -> tuple[str, ...]:
    """Return supported map IDs once each in canonical display order."""

    if isinstance(values, str | bytes | bytearray):
        raise ValueError(f"{label} must contain a sequence.")
    try:
        normalized = {str(value).strip().lower() for value in values}
    except TypeError as error:
        raise ValueError(f"{label} must contain a sequence.") from error
    if "" in normalized:
        raise ValueError(f"{label} cannot contain an empty map ID.")
    unknown = normalized - set(PBR_MAP_TYPES)
    if unknown:
        raise ValueError(
            f"{label} contain unknown map IDs: " + ", ".join(sorted(unknown))
        )
    return tuple(map_type for map_type in PBR_MAP_TYPES if map_type in normalized)


# ### Public exports ###
__all__ = [
    "ATLAS_MAP_BASE_COLOR",
    "ATLAS_MAP_LABELS",
    "ATLAS_MAP_TYPES",
    "PBR_MAP_METALLIC",
    "PBR_MAP_NORMAL",
    "PBR_MAP_ROUGHNESS",
    "PBR_MAP_TYPES",
    "normalize_pbr_map_types",
]
