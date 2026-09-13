# ### Imports ###
from __future__ import annotations

from collections.abc import Mapping, Sequence

from housemaker.architectural_surface_edits import (
    SurfaceTopologyEditResult,
    parse_editable_surface_id,
)
from housemaker.models import LevelData
from housemaker.surface_geometry import build_fixed_surfaces


# ### Public orientation operations ###
def flip_surface_orientation(
    levels: Sequence[LevelData],
    surface_id: str,
) -> SurfaceTopologyEditResult:
    """Toggle one persistent manual orientation override by semantic surface ID."""

    level_sequence = _normalize_levels(levels)
    normalized_surface_id = _normalize_surface_id(surface_id)
    surface = next(
        (
            candidate
            for candidate in build_fixed_surfaces(level_sequence)
            if candidate.surface_id == normalized_surface_id
        ),
        None,
    )
    if surface is None:
        raise ValueError("The selected surface no longer exists.")

    level = next(
        (
            candidate
            for candidate in level_sequence
            if candidate.index == surface.level_index
        ),
        None,
    )
    if level is None:
        raise ValueError("The selected surface belongs to an unknown level.")

    flipped_surface_ids = set(level.flipped_surface_ids)
    if normalized_surface_id in flipped_surface_ids:
        flipped_surface_ids.remove(normalized_surface_id)
    else:
        flipped_surface_ids.add(normalized_surface_id)
    level.flipped_surface_ids = flipped_surface_ids

    return SurfaceTopologyEditResult(
        replacements={},
        selected_surface_ids=(normalized_surface_id,),
        requires_mesh_refresh=True,
        state_changed=True,
    )


def remap_flipped_surface_ids_with_lineage(
    levels: Sequence[LevelData],
    replacements: Mapping[str, Sequence[str]],
) -> None:
    """Carry an edited-face override to every replacement descendant."""

    levels_by_index = {
        level.index: level for level in _normalize_levels(levels)
    }
    replacement_graph = {
        _normalize_surface_id(raw_source_id): tuple(
            _normalize_surface_id(target_id) for target_id in raw_target_ids
        )
        for raw_source_id, raw_target_ids in replacements.items()
    }
    for level in levels_by_index.values():
        original_ids = set(level.flipped_surface_ids)
        remapped_ids = set(original_ids)
        for source_id in original_ids:
            parsed_source = parse_editable_surface_id(source_id)
            if (
                parsed_source is None
                or parsed_source[0] != level.index
                or source_id not in replacement_graph
            ):
                continue
            remapped_ids.remove(source_id)
            remapped_ids.update(
                target_id
                for target_id in _resolve_terminal_surface_ids(
                    source_id,
                    replacement_graph,
                    (),
                )
                if (
                    (parsed_target := parse_editable_surface_id(target_id))
                    is not None
                    and parsed_target[0] == level.index
                )
            )
        level.flipped_surface_ids = remapped_ids


def _resolve_terminal_surface_ids(
    surface_id: str,
    replacement_graph: Mapping[str, tuple[str, ...]],
    ancestors: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve chained lineage independent of mapping insertion order."""

    targets = replacement_graph.get(surface_id)
    if targets is None:
        return (surface_id,)
    if not targets:
        return ()
    if surface_id in ancestors:
        return (surface_id,)
    next_ancestors = (*ancestors, surface_id)
    return tuple(
        dict.fromkeys(
            terminal_id
            for target_id in targets
            for terminal_id in _resolve_terminal_surface_ids(
                target_id,
                replacement_graph,
                next_ancestors,
            )
        )
    )


# ### Validation helpers ###
def _normalize_levels(levels: Sequence[LevelData]) -> tuple[LevelData, ...]:
    try:
        normalized = tuple(levels)
    except TypeError as error:
        raise TypeError("Surface orientation edits require a level sequence.") from error
    if not normalized or not all(isinstance(level, LevelData) for level in normalized):
        raise TypeError("Surface orientation edit levels must contain LevelData values.")
    level_indices = tuple(level.index for level in normalized)
    if len(set(level_indices)) != len(level_indices):
        raise ValueError("Surface orientation edit level indices must be unique.")
    return normalized


def _normalize_surface_id(value: object) -> str:
    surface_id = str(value).strip().lower()
    if not surface_id or len(surface_id) > 512:
        raise ValueError("A surface orientation edit requires a valid surface ID.")
    return surface_id


__all__ = [
    "flip_surface_orientation",
    "remap_flipped_surface_ids_with_lineage",
]
