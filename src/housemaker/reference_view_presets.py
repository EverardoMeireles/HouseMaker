"""Reusable reference-image views shared by settings and generation controls."""

# ### Imports ###
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from housemaker.app_settings import ApplicationSettingsStore


# ### Constants ###
REFERENCE_VIEW_PRESETS_SETTINGS_KEY = "generation/reference_view_presets"
MAX_REFERENCE_VIEW_NAME_LENGTH = 80
MAX_REFERENCE_VIEW_PROMPT_LENGTH = 4000
REMOVED_DEFAULT_REFERENCE_VIEW_IDS = frozenset({"side", "top"})


# ### Preset model ###
@dataclass(frozen=True)
class ReferenceViewPreset:
    preset_id: str
    name: str
    prompt: str

    def __post_init__(self) -> None:
        for field_name, maximum_length in (
            ("preset_id", None),
            ("name", MAX_REFERENCE_VIEW_NAME_LENGTH),
            ("prompt", MAX_REFERENCE_VIEW_PROMPT_LENGTH),
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Reference view {field_name} must not be empty.")
            value = value.strip()
            if maximum_length is not None and len(value) > maximum_length:
                raise ValueError(
                    f"Reference view {field_name} must be at most "
                    f"{maximum_length} characters."
                )
            object.__setattr__(self, field_name, value)
        if self.name.casefold() == "(write prompt)":
            raise ValueError(
                "The reference view name '(Write prompt)' is reserved for custom prompts."
            )


# ### Default views ###
def _default_prompt(view: str) -> str:
    return (
        f"Create a sharp, photorealistic {view} of the same object. "
        "Preserve its shape, proportions, materials, colors and distinctive details "
        "without redesigning it. Remove surrounding clutter and selection-mask "
        "artifacts. Center the complete object on a plain white background, filling "
        "90–95% of the frame along its longest dimension. Keep every component "
        "visible without cropping or stretching. Use even neutral lighting and "
        "clear, natural surface details. Show one object in one view, not a collage."
    )


DEFAULT_REFERENCE_VIEW_PRESETS = (
    ReferenceViewPreset(
        "elevated-three-quarter",
        "Slightly elevated three-quarter",
        _default_prompt(
            "slightly elevated front-left three-quarter view, showing the upper "
            "surface, front, side and visible supports"
        ),
    ),
    ReferenceViewPreset(
        "front", "Straight front", _default_prompt("straight-on front view")
    ),
    ReferenceViewPreset(
        "three-quarter",
        "Front-left three-quarter",
        _default_prompt(
            "front-left three-quarter view, with the camera 30–45 degrees to the "
            "left of the front, showing both the front and left side"
        ),
    ),
    ReferenceViewPreset(
        "low-three-quarter",
        "Slightly low three-quarter",
        _default_prompt(
            "slightly low front-left three-quarter view, revealing the underside "
            "while retaining a clear view of the front and side"
        ),
    ),
)


# ### Settings persistence ###
def read_reference_view_presets(
    settings: ApplicationSettingsStore | None,
) -> tuple[ReferenceViewPreset, ...]:
    if settings is None:
        return DEFAULT_REFERENCE_VIEW_PRESETS
    payload = settings.get(REFERENCE_VIEW_PRESETS_SETTINGS_KEY)
    if not isinstance(payload, list):
        return DEFAULT_REFERENCE_VIEW_PRESETS
    try:
        presets = []
        for item in payload:
            if not isinstance(item, dict):
                return DEFAULT_REFERENCE_VIEW_PRESETS
            presets.append(
                ReferenceViewPreset(
                    preset_id=item["preset_id"],
                    name=item["name"],
                    prompt=item["prompt"],
                )
            )
        normalized_presets = _unique_presets(presets)
        _validate_unique_names(normalized_presets)
        # Retire saved copies of the old built-ins without deleting custom views
        # or rewriting the settings file while another app instance may use it.
        retained_presets = tuple(
            preset
            for preset in normalized_presets
            if preset.preset_id not in REMOVED_DEFAULT_REFERENCE_VIEW_IDS
        )
        return _preferred_view_order(retained_presets)
    except (KeyError, TypeError, ValueError):
        return DEFAULT_REFERENCE_VIEW_PRESETS


def save_reference_view_presets(
    settings: ApplicationSettingsStore | None,
    presets: Iterable[ReferenceViewPreset],
) -> bool:
    if settings is None:
        return False
    try:
        requested_presets = tuple(presets)
        normalized_presets = _unique_presets(requested_presets)
        if len(normalized_presets) != len(requested_presets):
            return False
        _validate_unique_names(normalized_presets)
    except (TypeError, ValueError):
        return False
    return settings.set(
        REFERENCE_VIEW_PRESETS_SETTINGS_KEY,
        [
            {"preset_id": item.preset_id, "name": item.name, "prompt": item.prompt}
            for item in normalized_presets
        ],
    )


# ### Ordering helpers ###
def _preferred_view_order(
    presets: tuple[ReferenceViewPreset, ...],
) -> tuple[ReferenceViewPreset, ...]:
    """Move the elevated view first without restoring removed or edited presets."""

    elevated = tuple(
        preset for preset in presets if preset.preset_id == "elevated-three-quarter"
    )
    return elevated + tuple(
        preset for preset in presets if preset.preset_id != "elevated-three-quarter"
    )


# ### Validation helpers ###
def _unique_presets(
    presets: Iterable[ReferenceViewPreset],
) -> tuple[ReferenceViewPreset, ...]:
    result = []
    seen_ids: set[str] = set()
    for preset in presets:
        if not isinstance(preset, ReferenceViewPreset):
            raise TypeError("Reference views must be ReferenceViewPreset instances.")
        if preset.preset_id not in seen_ids:
            result.append(preset)
            seen_ids.add(preset.preset_id)
    return tuple(result)


def _validate_unique_names(presets: Iterable[ReferenceViewPreset]) -> None:
    seen_names: set[str] = set()
    for preset in presets:
        normalized_name = preset.name.casefold()
        if normalized_name in seen_names:
            raise ValueError("Reference view names must be unique.")
        seen_names.add(normalized_name)
