"""Deterministic room-owned object Atlases without changing source texel density."""

# ### Imports ###
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from types import MappingProxyType

from PIL import Image
from shapely import Polygon

from housemaker.atlas_export import MaterializedTextureAtlas
from housemaker.glb import (
    PlacedGeneratedModel,
    build_placed_generated_model_top_down_footprint,
)
from housemaker.pbr_maps import ATLAS_MAP_TYPES
from housemaker.room_zones import RoomZoneLayout, classify_object_room_ids
from housemaker.texture_atlas_state import (
    ATLAS_PACKING_MODE_FULL,
    ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
    ATLAS_RESOLUTIONS,
    ATLAS_SLOT_HALF_LEFT,
    ATLAS_SLOT_HALF_RIGHT,
    ATLAS_SLOT_QUADRANT_ORDER,
    ATLAS_TYPE_OBJECT,
    ATLAS_TYPE_SURFACE,
    TextureAtlasPlacement,
    TextureAtlasRecord,
    _allocate_quad,
    _QuadNode,
    _write_bytes_atomically,
    atlas_packing_slot_family,
    atlas_placement_pixel_regions,
)


# ### Immutable export plans ###
@dataclass(frozen=True)
class RoomAtlasExportPlan:
    atlases: tuple[MaterializedTextureAtlas, ...]
    source_room_ids: Mapping[str, tuple[str, ...]]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "atlases", tuple(self.atlases))
        object.__setattr__(
            self, "source_room_ids", MappingProxyType(dict(self.source_room_ids))
        )
        object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(frozen=True)
class _Source:
    atlas: MaterializedTextureAtlas
    placement: TextureAtlasPlacement


# ### Public room-based export preparation ###
def prepare_room_export_atlases(
    atlases: Sequence[MaterializedTextureAtlas],
    layout: RoomZoneLayout,
    placements: Sequence[PlacedGeneratedModel],
    *,
    output_directory: str | Path,
    placement_level_indices: Mapping[str, int] | None = None,
) -> RoomAtlasExportPlan:
    """Repack required object sources, keeping architectural Atlases unchanged.

    Sources seen in several rooms, crossing a boundary, or lacking trustworthy
    ownership share an Atlas instead of duplicating their images. Every copied
    region includes its existing guard pixels. Cached UV1 scene AO is preserved
    only when the Atlas layout is unchanged; old files are never deleted.
    """

    atlases = tuple(atlases)
    placements = tuple(placements)
    if not all(isinstance(item, MaterializedTextureAtlas) for item in atlases):
        raise TypeError("Room packing requires materialized texture Atlases.")
    if not isinstance(layout, RoomZoneLayout):
        raise TypeError("Room packing requires a RoomZoneLayout.")
    if not all(isinstance(item, PlacedGeneratedModel) for item in placements):
        raise TypeError("Room packing requires placed generated models.")
    source_rooms = _source_room_ids(layout, placements, placement_level_indices or {})
    sources: dict[str, _Source] = {}
    for item in atlases:
        if item.atlas.atlas_type == ATLAS_TYPE_SURFACE:
            continue
        for placement in item.atlas.placements:
            if placement.object_id in source_rooms:
                sources.setdefault(placement.object_id, _Source(item, placement))
    if not sources:
        return RoomAtlasExportPlan(atlases, source_rooms)

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    rooms_by_id = {room.room_id: room for room in layout.rooms}
    existing_by_id = {item.atlas.atlas_id: item for item in atlases}
    output: list[MaterializedTextureAtlas] = []
    library_fragments: dict[str, _Source] = {}
    # Keep unrequired library sources at their existing coordinates and paths.
    for item in atlases:
        if item.atlas.atlas_type == ATLAS_TYPE_SURFACE:
            output.append(item)
            continue
        retained = [p for p in item.atlas.placements if p.object_id not in sources]
        if len(retained) == len(item.atlas.placements):
            output.append(item)
        elif retained:
            try:
                record = replace(
                    item.atlas,
                    placements=retained,
                    room_id=None,
                    room_name="Shared objects",
                )
            except ValueError:
                # A surviving right half/bottom quadrant cannot stand alone.
                # Transport its exact pixel region to a valid shared slot.
                for placement in retained:
                    library_fragments.setdefault(
                        placement.object_id,
                        _Source(item, placement),
                    )
            else:
                output.append(replace(item, atlas=record))

    sources.update(library_fragments)
    for source_id in library_fragments:
        source_rooms.setdefault(source_id, ())
    grouped: dict[str | None, list[_Source]] = defaultdict(list)
    for source_id in sorted(sources):
        rooms = source_rooms[source_id]
        grouped[rooms[0] if len(rooms) == 1 else None].append(sources[source_id])

    warnings: list[str] = []
    occupied_names = {item.atlas.name.casefold() for item in output}
    for room_id in sorted(grouped, key=lambda value: (value is None, value or "")):
        room = rooms_by_id.get(room_id)
        room_name = (
            (room.name or room.room_id) if room is not None else "Shared objects"
        )
        pages = _pack_sources(grouped[room_id])
        for page_index, (resolution, packed) in enumerate(pages):
            digest = hashlib.sha256(
                json.dumps(
                    (room_id, sorted(p.object_id for p in packed), page_index),
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:24]
            atlas_id = f"room-objects-{digest}"
            name = room_name if len(pages) == 1 else f"{room_name} ({page_index + 1})"
            if f"[OBJECT] {name}".casefold() in occupied_names:
                name = f"{name} [{digest[:12]}]"
            record = TextureAtlasRecord(
                atlas_id=atlas_id,
                name=name,
                resolution=resolution,
                placements=packed,
                atlas_type=ATLAS_TYPE_OBJECT,
                room_id=room_id,
                room_name=room_name,
                image_path=f"{atlas_id}.png",
            )
            existing = existing_by_id.get(atlas_id)
            occupied_names.add(record.name.casefold())
            if existing is not None and _same_layout(existing.atlas, record):
                output.append(existing)
                continue
            page_sources = {p.object_id: sources[p.object_id] for p in packed}
            if any(
                source.atlas.surface_ao_image_path is not None
                for source in page_sources.values()
            ):
                warnings.append(
                    f"{record.name}: room Atlas layout changed; rebake scene AO."
                )
            output.append(_materialize_page(record, page_sources, destination))
    return RoomAtlasExportPlan(tuple(output), source_rooms, tuple(warnings))


# ### Source ownership and geometry-aware room classification ###
def _source_room_ids(layout, placements, level_indices) -> dict[str, tuple[str, ...]]:
    owners: dict[str, set[str]] = defaultdict(set)
    unknown: set[str] = set()
    for placement in placements:
        source_id = str(placement.atlas_source_object_id)
        footprint = build_placed_generated_model_top_down_footprint(placement)
        known_level = level_indices.get(placement.object_id)
        levels = (
            {int(known_level)}
            if known_level is not None
            else {
                room.level_index
                for room in layout.rooms
                if room.floor_z_meters - 1e-6
                <= placement.world_position[2]
                <= room.top_z_meters + 1e-6
            }
        )
        rooms = {
            room_id
            for level in sorted(levels)
            for room_id in classify_object_room_ids(
                layout,
                level,
                placement.world_position[:2],
                footprint,
            )
        }
        if len(rooms) == 1:
            room = next(room for room in layout.rooms if room.room_id in rooms)
            if not room.polygon.buffer(1e-7).covers(Polygon(footprint)):
                unknown.add(source_id)
        owners[source_id].update(rooms)
        if not rooms:
            unknown.add(source_id)
    # A source with one known and one unknown placement must still be shared.
    return {
        source_id: (() if source_id in unknown else tuple(sorted(room_ids)))
        for source_id, room_ids in sorted(owners.items())
    }


# ### Deterministic dyadic slot packing ###
def _source_units(sources: Sequence[_Source]) -> list[list[TextureAtlasPlacement]]:
    units: list[list[TextureAtlasPlacement]] = []
    partial: dict[tuple[str, int, int], list[TextureAtlasPlacement]] = defaultdict(list)
    for source in sorted(sources, key=lambda source: source.placement.object_id):
        placement = source.placement
        if placement.packing_mode == ATLAS_PACKING_MODE_FULL:
            units.append([placement])
        else:
            partial[
                (
                    atlas_packing_slot_family(placement.packing_mode),
                    placement.texture_resolution,
                    placement.size,
                )
            ].append(placement)
    for key in sorted(partial):
        members = partial[key]
        quarters = members[0].packing_mode == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER
        capacity = 4 if quarters else 2
        for start in range(0, len(members), capacity):
            units.append(
                [
                    replace(member, slot_quadrant=ATLAS_SLOT_QUADRANT_ORDER[index])
                    if quarters
                    else replace(
                        member,
                        slot_half=(
                            ATLAS_SLOT_HALF_LEFT
                            if index == 0
                            else ATLAS_SLOT_HALF_RIGHT
                        ),
                    )
                    for index, member in enumerate(members[start : start + capacity])
                ]
            )
    return sorted(
        units, key=lambda unit: (-unit[0].size, tuple(p.object_id for p in unit))
    )


def _try_pack(units, resolution):
    root = _QuadNode(0, 0, resolution)
    packed: list[TextureAtlasPlacement] = []
    remaining = []
    for unit in units:
        node = _allocate_quad(root, unit[0].size)
        if node is None:
            remaining.append(unit)
        else:
            packed.extend(replace(p, x=node.x, y=node.y) for p in unit)
    return packed, remaining


def _pack_sources(sources):
    units = _source_units(sources)
    pages = []
    while units:
        for resolution in sorted(ATLAS_RESOLUTIONS):
            packed, remaining = _try_pack(units, resolution)
            if not remaining or resolution == max(ATLAS_RESOLUTIONS):
                break
        if not packed:
            raise ValueError("An object texture exceeds the maximum Atlas size.")
        pages.append((resolution, packed))
        units = remaining
    return pages


# ### Exact pixel transport and cached-layout preservation ###
def _same_layout(first: TextureAtlasRecord, second: TextureAtlasRecord) -> bool:
    return (
        first.resolution == second.resolution
        and first.atlas_type == second.atlas_type
        and first.room_id == second.room_id
        and sorted(
            (p.to_dict() for p in first.placements), key=lambda p: p["object_id"]
        )
        == sorted(
            (p.to_dict() for p in second.placements), key=lambda p: p["object_id"]
        )
    )


def _materialize_page(record, sources, directory) -> MaterializedTextureAtlas:
    from housemaker.texture_atlas_workspace import (
        build_texture_atlas_map_image_relative_path,
    )

    paths = {}
    active_maps: set[str] = set()
    for source in sources.values():
        active_maps.update(source.atlas.active_map_types)
    for map_type in ATLAS_MAP_TYPES:
        canvas = Image.new("RGBA", (record.resolution, record.resolution))
        by_path: dict[Path, list[TextureAtlasPlacement]] = defaultdict(list)
        for placement in record.placements:
            by_path[sources[placement.object_id].atlas.map_paths[map_type]].append(
                placement
            )
        # Decode at most one original Atlas at a time. Many sparsely populated
        # authoring Atlases must not turn a small room page into gigabytes of RAM.
        for path, placements in by_path.items():
            with Image.open(path) as image:
                rgba = image.convert("RGBA")
                for placement in placements:
                    source = sources[placement.object_id]
                    if image.size != (source.atlas.atlas.resolution,) * 2:
                        raise ValueError(
                            "Room Atlas source image dimensions are inconsistent."
                        )
                    source_outer, _source_inner = atlas_placement_pixel_regions(
                        source.placement
                    )
                    target_outer, _target_inner = atlas_placement_pixel_regions(
                        placement
                    )
                    pixels = rgba.crop(source_outer)
                    if pixels.size != (
                        target_outer[2] - target_outer[0],
                        target_outer[3] - target_outer[1],
                    ):
                        raise ValueError(
                            "Room Atlas packing would change source texel density."
                        )
                    canvas.paste(pixels, target_outer[:2])
                rgba.close()
        relative_path = build_texture_atlas_map_image_relative_path(
            record.atlas_id, map_type
        )
        path = directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        output = BytesIO()
        canvas.save(output, format="PNG")
        _write_bytes_atomically(path, output.getvalue())
        paths[map_type] = path
    return MaterializedTextureAtlas(record, paths, frozenset(active_maps))
