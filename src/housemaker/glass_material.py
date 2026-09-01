# ### Imports ###
from __future__ import annotations

import numpy as np
from trimesh.visual.material import PBRMaterial


# ### Public constants ###
HOUSEMAKER_GLASS_MATERIAL_NAME = "HouseMaker Glass"
HOUSEMAKER_GLASS_MATERIAL_PROFILE = "housemaker_glass_v1"
HOUSEMAKER_GLASS_BASE_COLOR_FACTOR = (205, 232, 242, 48)
HOUSEMAKER_GLASS_METALLIC_FACTOR = 1.0
HOUSEMAKER_GLASS_ROUGHNESS_FACTOR = 10.0 / 255.0
DEFAULT_HOUSEMAKER_GLASS_DOUBLE_SIDED = False


# ### Public material helpers ###
def build_housemaker_glass_material(
    double_sided: bool,
) -> PBRMaterial:
    """Build HouseMaker's atlas-independent prefab glass material."""

    if not isinstance(double_sided, bool):
        raise TypeError("Glass sidedness must be a boolean.")
    return PBRMaterial(
        name=HOUSEMAKER_GLASS_MATERIAL_NAME,
        baseColorFactor=np.asarray(
            HOUSEMAKER_GLASS_BASE_COLOR_FACTOR,
            dtype=np.uint8,
        ),
        metallicFactor=HOUSEMAKER_GLASS_METALLIC_FACTOR,
        roughnessFactor=HOUSEMAKER_GLASS_ROUGHNESS_FACTOR,
        alphaMode="BLEND",
        doubleSided=double_sided,
    )


def is_housemaker_glass_material(material: object) -> bool:
    """Return whether a material carries the stable prefab glass marker."""

    return str(getattr(material, "name", "")).strip() == (
        HOUSEMAKER_GLASS_MATERIAL_NAME
    )


def get_housemaker_glass_double_sided(
    material: object,
) -> bool | None:
    """Return prefab sidedness, or ``None`` for a non-glass material."""

    if not is_housemaker_glass_material(material):
        return None
    return bool(getattr(material, "doubleSided", False))


def get_housemaker_glass_runtime_key(material: object) -> str | None:
    """Return the cache key an external renderer can use for one singleton."""

    double_sided = get_housemaker_glass_double_sided(material)
    if double_sided is None:
        return None
    side_name = "double_sided" if double_sided else "single_sided"
    return f"{HOUSEMAKER_GLASS_MATERIAL_PROFILE}:{side_name}"
