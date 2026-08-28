# ### Imports ###
from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable

import cv2
import numpy as np


# ### Constants ###
ATLAS_RESOLUTIONS = frozenset({2048, 4096})
OBJECT_TEXTURE_RESOLUTIONS = frozenset({512, 1024, 2048})
ATLAS_BASE_CELL_SIZE = 512
MAX_ATLAS_COUNT = 1_000
MAX_ATLAS_NAME_LENGTH = 200
MAX_ATLAS_OBJECT_COUNT = 4_096
MAX_ATLAS_ID_LENGTH = 128
MAX_TEXTURE_PATH_LENGTH = 32_768
ATLAS_STATE_SCHEMA_VERSION = 5
LEGACY_ATLAS_STATE_SCHEMA_VERSION = 1
SYMMETRIC_HALF_ATLAS_STATE_SCHEMA_VERSION = 2
SYMMETRIC_QUARTER_ATLAS_STATE_SCHEMA_VERSION = 3
SYMMETRIC_PAIR_ATLAS_STATE_SCHEMA_VERSION = 4
ATLAS_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
ATLAS_PACKING_MODE_FULL = "full"
# Schemas v2-v4 persist these legacy modes and must retain their old geometry.
ATLAS_PACKING_MODE_SYMMETRIC_HALF = "symmetric_half"
ATLAS_PACKING_MODE_SYMMETRIC_QUARTER = "symmetric_quarter"
ATLAS_PACKING_MODE_SYMMETRIC_PAIR = "symmetric_pair"
# Schema v5 stores current half models in ordinary square slots.
ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR = "symmetric_square_pair"
ATLAS_PACKING_MODES = frozenset(
    {
        ATLAS_PACKING_MODE_FULL,
        ATLAS_PACKING_MODE_SYMMETRIC_HALF,
        ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
        ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
        ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
    }
)
ATLAS_SLOT_HALF_LEFT = "left"
ATLAS_SLOT_HALF_RIGHT = "right"
ATLAS_SLOT_HALVES = frozenset({ATLAS_SLOT_HALF_LEFT, ATLAS_SLOT_HALF_RIGHT})
ATLAS_SLOT_QUADRANT_TOP_LEFT = "top_left"
ATLAS_SLOT_QUADRANT_TOP_RIGHT = "top_right"
ATLAS_SLOT_QUADRANT_BOTTOM_LEFT = "bottom_left"
ATLAS_SLOT_QUADRANT_BOTTOM_RIGHT = "bottom_right"
ATLAS_SLOT_QUADRANT_ORDER = (
    ATLAS_SLOT_QUADRANT_TOP_LEFT,
    ATLAS_SLOT_QUADRANT_TOP_RIGHT,
    ATLAS_SLOT_QUADRANT_BOTTOM_LEFT,
    ATLAS_SLOT_QUADRANT_BOTTOM_RIGHT,
)
ATLAS_SLOT_QUADRANTS = frozenset(ATLAS_SLOT_QUADRANT_ORDER)
SYMMETRIC_PACKED_TEXTURE_RESOLUTIONS = frozenset({512, 1024})
ATLAS_LIMITED_RESOLUTION_PACKING_MODES = frozenset(
    {
        ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
        ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
        ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
    }
)
ATLAS_DOUBLE_SIZED_PACKING_MODES = frozenset(
    {
        ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
        ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
    }
)
ATLAS_HALF_SLOT_PACKING_MODES = frozenset(
    {
        ATLAS_PACKING_MODE_SYMMETRIC_HALF,
        ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
        ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
    }
)


# ### Public data models ###
@dataclass(frozen=True)
class TextureAtlasPlacement:
    """One texture source's square allocation inside an atlas.

    ``object_id`` remains the persisted field name for schema compatibility.
    """

    object_id: str
    texture_path: str
    texture_resolution: int
    x: int
    y: int
    size: int
    packing_mode: str = ATLAS_PACKING_MODE_FULL
    slot_half: str | None = None
    slot_quadrant: str | None = None

    def __post_init__(self) -> None:
        _validate_nonempty_text(self.object_id, "Texture source ID")
        object.__setattr__(
            self,
            "texture_path",
            _normalize_project_relative_path(self.texture_path),
        )
        _validate_texture_resolution(self.texture_resolution)
        if int(self.x) < 0 or int(self.y) < 0:
            raise ValueError("Atlas placement coordinates cannot be negative.")
        packing_mode = str(self.packing_mode).strip().lower()
        if packing_mode not in ATLAS_PACKING_MODES:
            raise ValueError("Unknown texture atlas packing mode.")
        slot_half = (
            None if self.slot_half is None else str(self.slot_half).strip().lower()
        )
        slot_quadrant = (
            None
            if self.slot_quadrant is None
            else str(self.slot_quadrant).strip().lower()
        )
        if packing_mode == ATLAS_PACKING_MODE_FULL:
            if slot_half is not None or slot_quadrant is not None:
                raise ValueError("Full texture placements cannot select a slot region.")
            expected_size = int(self.texture_resolution)
        elif packing_mode == ATLAS_PACKING_MODE_SYMMETRIC_HALF:
            if slot_half not in ATLAS_SLOT_HALVES or slot_quadrant is not None:
                raise ValueError(
                    "Symmetric half textures require one left or right slot."
                )
            expected_size = int(self.texture_resolution)
        elif packing_mode == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER:
            if (
                int(self.texture_resolution)
                not in SYMMETRIC_PACKED_TEXTURE_RESOLUTIONS
            ):
                raise ValueError(
                    "Symmetric quarter textures must use 512 or 1024 content."
                )
            if slot_half is not None or slot_quadrant not in ATLAS_SLOT_QUADRANTS:
                raise ValueError(
                    "Symmetric quarter textures require one Atlas quadrant."
                )
            expected_size = int(self.texture_resolution) * 2
        else:
            if (
                int(self.texture_resolution)
                not in SYMMETRIC_PACKED_TEXTURE_RESOLUTIONS
            ):
                raise ValueError(
                    "Symmetric pair textures must use 512 or 1024 content."
                )
            if slot_half not in ATLAS_SLOT_HALVES or slot_quadrant is not None:
                raise ValueError(
                    "Symmetric pair textures require one left or right slot."
                )
            expected_size = int(self.texture_resolution) * (
                2 if packing_mode == ATLAS_PACKING_MODE_SYMMETRIC_PAIR else 1
            )
        if int(self.size) != expected_size:
            raise ValueError(
                "Atlas placement size does not match its texture packing mode."
            )
        object.__setattr__(self, "packing_mode", packing_mode)
        object.__setattr__(self, "slot_half", slot_half)
        object.__setattr__(self, "slot_quadrant", slot_quadrant)

    @property
    def width(self) -> int:
        return int(self.size)

    @property
    def height(self) -> int:
        return int(self.size)

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "texture_path": self.texture_path,
            "texture_resolution": int(self.texture_resolution),
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.size),
            "height": int(self.size),
            "packing_mode": self.packing_mode,
            "slot_half": self.slot_half,
            "slot_quadrant": self.slot_quadrant,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "TextureAtlasPlacement":
        if not isinstance(payload, dict):
            raise ValueError("Texture atlas placement must contain an object.")
        width = int(payload["width"])
        height = int(payload["height"])
        if width != height:
            raise ValueError("Texture atlas placements must be square.")
        return cls(
            object_id=str(payload["object_id"]),
            texture_path=str(payload["texture_path"]),
            texture_resolution=int(payload["texture_resolution"]),
            x=int(payload["x"]),
            y=int(payload["y"]),
            size=width,
            packing_mode=str(
                payload.get("packing_mode", ATLAS_PACKING_MODE_FULL)
            ),
            slot_half=(
                None
                if payload.get("slot_half") is None
                else str(payload["slot_half"])
            ),
            slot_quadrant=(
                None
                if payload.get("slot_quadrant") is None
                else str(payload["slot_quadrant"])
            ),
        )


@dataclass
class TextureAtlasRecord:
    """One named atlas and its validated texture-source placements."""

    atlas_id: str
    name: str
    resolution: int
    placements: list[TextureAtlasPlacement] = field(default_factory=list)
    image_path: str | None = None

    def __post_init__(self) -> None:
        _validate_atlas_id(self.atlas_id)
        self.name = str(self.name).strip()
        _validate_atlas_name(self.name)
        _validate_atlas_resolution(self.resolution)
        if len(self.placements) > MAX_ATLAS_OBJECT_COUNT:
            raise ValueError("Texture atlas contains too many texture sources.")
        if not all(
            isinstance(placement, TextureAtlasPlacement)
            for placement in self.placements
        ):
            raise ValueError(
                "Texture atlas placements must be TextureAtlasPlacement values."
            )
        object_ids = [placement.object_id for placement in self.placements]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError(
                "A texture source can only occur once in one texture atlas."
            )
        _validate_placements_fit(self.resolution, self.placements)
        if self.image_path is not None:
            self.image_path = _normalize_project_relative_path(self.image_path)

    def placement_for_object(
        self,
        object_id: str,
    ) -> TextureAtlasPlacement | None:
        normalized_id = str(object_id)
        return next(
            (
                placement
                for placement in self.placements
                if placement.object_id == normalized_id
            ),
            None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "atlas_id": self.atlas_id,
            "name": self.name,
            "resolution": int(self.resolution),
            "image_path": self.image_path,
            "placements": [
                placement.to_dict() for placement in self.placements
            ],
        }

    @classmethod
    def from_dict(cls, payload: object) -> "TextureAtlasRecord":
        if not isinstance(payload, dict):
            raise ValueError("Texture atlas data must contain an object.")
        raw_placements = payload.get("placements", [])
        if not isinstance(raw_placements, list):
            raise ValueError("Texture atlas placements must contain a list.")
        return cls(
            atlas_id=str(payload["atlas_id"]),
            name=str(payload["name"]),
            resolution=int(payload["resolution"]),
            placements=[
                TextureAtlasPlacement.from_dict(raw_placement)
                for raw_placement in raw_placements
            ],
            image_path=(
                None
                if payload.get("image_path") is None
                else str(payload["image_path"])
            ),
        )


@dataclass
class TextureAtlasData:
    """Project-persisted collection of user-created texture atlases."""

    atlases: list[TextureAtlasRecord] = field(default_factory=list)
    selected_atlas_id: str | None = None

    def __post_init__(self) -> None:
        if len(self.atlases) > MAX_ATLAS_COUNT:
            raise ValueError("Project contains too many texture atlases.")
        if not all(isinstance(atlas, TextureAtlasRecord) for atlas in self.atlases):
            raise ValueError("Texture atlases must be TextureAtlasRecord values.")
        atlas_ids = [atlas.atlas_id for atlas in self.atlases]
        if len(atlas_ids) != len(set(atlas_ids)):
            raise ValueError("Texture atlas IDs must be unique.")
        atlas_names = [atlas.name.casefold() for atlas in self.atlases]
        if len(atlas_names) != len(set(atlas_names)):
            raise ValueError("Texture atlas names must be unique.")
        if (
            self.selected_atlas_id is not None
            and self.selected_atlas_id not in atlas_ids
        ):
            raise ValueError("Selected texture atlas ID does not exist.")

    def atlas_by_id(self, atlas_id: str) -> TextureAtlasRecord | None:
        normalized_id = str(atlas_id)
        return next(
            (atlas for atlas in self.atlases if atlas.atlas_id == normalized_id),
            None,
        )

    def clone(self) -> "TextureAtlasData":
        """Return a validated copy without sharing mutable placement lists."""

        return TextureAtlasData.from_dict(self.to_dict())

    def create_atlas(
        self,
        name: str,
        resolution: int,
        *,
        atlas_id: str | None = None,
    ) -> TextureAtlasRecord:
        if len(self.atlases) >= MAX_ATLAS_COUNT:
            raise ValueError("Project contains too many texture atlases.")
        normalized_id = str(atlas_id or uuid.uuid4())
        if self.atlas_by_id(normalized_id) is not None:
            raise ValueError(f"Texture atlas ID already exists: {normalized_id!r}.")
        normalized_name = str(name).strip()
        if any(
            atlas.name.casefold() == normalized_name.casefold()
            for atlas in self.atlases
        ):
            raise ValueError(f"Texture atlas name already exists: {normalized_name!r}.")
        atlas = TextureAtlasRecord(
            atlas_id=normalized_id,
            name=normalized_name,
            resolution=int(resolution),
        )
        self.atlases.append(atlas)
        if self.selected_atlas_id is None:
            self.selected_atlas_id = atlas.atlas_id
        return atlas

    def remove_atlas(self, atlas_id: str) -> bool:
        atlas = self.atlas_by_id(atlas_id)
        if atlas is None:
            return False
        self.atlases.remove(atlas)
        if self.selected_atlas_id == atlas.atlas_id:
            self.selected_atlas_id = (
                self.atlases[0].atlas_id if self.atlases else None
            )
        return True

    def select_atlas(self, atlas_id: str | None) -> None:
        if atlas_id is None:
            self.selected_atlas_id = None
            return
        atlas = self._require_atlas(atlas_id)
        self.selected_atlas_id = atlas.atlas_id

    def assign_object(
        self,
        atlas_id: str,
        object_id: str,
        texture_path: str | Path,
        texture_resolution: int,
        packing_mode: str = ATLAS_PACKING_MODE_FULL,
        *,
        allow_pairing: bool = True,
    ) -> TextureAtlasPlacement:
        """Add or update one source without moving unrelated placements."""

        atlas = self._require_atlas(atlas_id)
        normalized_id = str(object_id)
        normalized_path = _normalize_project_relative_path(texture_path)
        normalized_packing_mode = _normalize_packing_mode(packing_mode)
        _validate_nonempty_text(normalized_id, "Object ID")
        _validate_texture_resolution(texture_resolution)
        normalized_resolution = int(texture_resolution)
        existing = atlas.placement_for_object(normalized_id)
        if existing is None and len(atlas.placements) >= MAX_ATLAS_OBJECT_COUNT:
            raise ValueError("Texture atlas contains too many texture sources.")
        unaffected = [
            placement
            for placement in atlas.placements
            if placement.object_id != normalized_id
        ]
        preserves_existing_slot = (
            existing is not None
            and existing.texture_resolution == normalized_resolution
            and existing.packing_mode == normalized_packing_mode
        )
        if preserves_existing_slot:
            placement = TextureAtlasPlacement(
                object_id=normalized_id,
                texture_path=normalized_path,
                texture_resolution=normalized_resolution,
                x=existing.x,
                y=existing.y,
                size=existing.size,
                packing_mode=normalized_packing_mode,
                slot_half=existing.slot_half,
                slot_quadrant=existing.slot_quadrant,
            )
            _validate_placements_fit(
                atlas.resolution,
                [*unaffected, placement],
            )
        else:
            unaffected = _normalize_partial_slot_placements(unaffected)
            half_slot_target = (
                _find_compatible_unpaired_half_slot(
                    unaffected,
                    normalized_resolution,
                    normalized_packing_mode,
                )
                if normalized_packing_mode in ATLAS_HALF_SLOT_PACKING_MODES
                and bool(allow_pairing)
                else None
            )
            if half_slot_target is not None:
                placement = TextureAtlasPlacement(
                    object_id=normalized_id,
                    texture_path=normalized_path,
                    texture_resolution=normalized_resolution,
                    x=half_slot_target.x,
                    y=half_slot_target.y,
                    size=half_slot_target.size,
                    packing_mode=normalized_packing_mode,
                    slot_half=ATLAS_SLOT_HALF_RIGHT,
                )
                _validate_placements_fit(
                    atlas.resolution,
                    [*unaffected, placement],
                )
            elif (
                normalized_packing_mode
                == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER
                and bool(allow_pairing)
                and (
                    quarter_target := _find_compatible_quarter_group(
                        unaffected,
                        normalized_resolution,
                    )
                )
                is not None
            ):
                placement = TextureAtlasPlacement(
                    object_id=normalized_id,
                    texture_path=normalized_path,
                    texture_resolution=normalized_resolution,
                    x=quarter_target[0].x,
                    y=quarter_target[0].y,
                    size=quarter_target[0].size,
                    packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
                    slot_quadrant=_first_available_quadrant(quarter_target),
                )
                _validate_placements_fit(
                    atlas.resolution,
                    [*unaffected, placement],
                )
            else:
                placement = _find_available_placement(
                    atlas.resolution,
                    normalized_id,
                    normalized_path,
                    normalized_resolution,
                    unaffected,
                    preferred_position=(
                        None if existing is None else (existing.x, existing.y)
                    ),
                    packing_mode=normalized_packing_mode,
                    slot_half=(
                        ATLAS_SLOT_HALF_LEFT
                        if normalized_packing_mode
                        in ATLAS_HALF_SLOT_PACKING_MODES
                        else None
                    ),
                    slot_quadrant=(
                        ATLAS_SLOT_QUADRANT_TOP_LEFT
                        if normalized_packing_mode
                        == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER
                        else None
                    ),
                )
        if existing is None:
            atlas.placements.append(placement)
        elif preserves_existing_slot:
            atlas.placements = [
                placement if candidate.object_id == normalized_id else candidate
                for candidate in atlas.placements
            ]
        else:
            atlas.placements = [*unaffected, placement]
        atlas.image_path = None
        return placement

    def place_object_at(
        self,
        atlas_id: str,
        object_id: str,
        texture_path: str | Path,
        texture_resolution: int,
        x: int,
        y: int,
        packing_mode: str = ATLAS_PACKING_MODE_FULL,
        slot_half: str | None = None,
        slot_quadrant: str | None = None,
    ) -> TextureAtlasPlacement:
        """Add or move one source at an exact aligned atlas slot."""

        atlas = self._require_atlas(atlas_id)
        normalized_id = str(object_id)
        normalized_path = _normalize_project_relative_path(texture_path)
        normalized_resolution = int(texture_resolution)
        normalized_packing_mode = _normalize_packing_mode(packing_mode)
        normalized_x = _normalize_atlas_coordinate(x, "Atlas placement x")
        normalized_y = _normalize_atlas_coordinate(y, "Atlas placement y")
        _validate_nonempty_text(normalized_id, "Object ID")
        _validate_texture_resolution(normalized_resolution)
        if (
            atlas.placement_for_object(normalized_id) is None
            and len(atlas.placements) >= MAX_ATLAS_OBJECT_COUNT
        ):
            raise ValueError("Texture atlas contains too many texture sources.")

        existing = atlas.placement_for_object(normalized_id)
        normalized_slot_half = _normalize_slot_half(
            normalized_packing_mode,
            slot_half,
            allow_unspecified_half=True,
        )
        normalized_slot_quadrant = _normalize_slot_quadrant(
            normalized_packing_mode,
            slot_quadrant,
            allow_unspecified_quadrant=True,
        )
        if (
            existing is not None
            and existing.texture_resolution == normalized_resolution
            and existing.x == normalized_x
            and existing.y == normalized_y
            and existing.packing_mode == normalized_packing_mode
            and (
                normalized_slot_half is None
                or existing.slot_half == normalized_slot_half
            )
            and (
                normalized_slot_quadrant is None
                or existing.slot_quadrant == normalized_slot_quadrant
            )
        ):
            placement = TextureAtlasPlacement(
                object_id=normalized_id,
                texture_path=normalized_path,
                texture_resolution=normalized_resolution,
                x=normalized_x,
                y=normalized_y,
                size=existing.size,
                packing_mode=normalized_packing_mode,
                slot_half=existing.slot_half,
                slot_quadrant=existing.slot_quadrant,
            )
            placements = [
                placement if candidate.object_id == normalized_id else candidate
                for candidate in atlas.placements
            ]
            _validate_placements_fit(atlas.resolution, placements)
            atlas.placements = placements
            atlas.image_path = None
            return placement
        placements = _normalize_partial_slot_placements(
            [
                candidate
                for candidate in atlas.placements
                if candidate.object_id != normalized_id
            ]
        )
        if normalized_packing_mode in ATLAS_HALF_SLOT_PACKING_MODES:
            compatible_target = next(
                (
                    candidate
                    for candidate in placements
                    if candidate.packing_mode == normalized_packing_mode
                    and candidate.slot_half == ATLAS_SLOT_HALF_LEFT
                    and candidate.texture_resolution == normalized_resolution
                    and candidate.x == normalized_x
                    and candidate.y == normalized_y
                ),
                None,
            )
            if normalized_slot_half is None:
                normalized_slot_half = (
                    ATLAS_SLOT_HALF_RIGHT
                    if compatible_target is not None
                    else ATLAS_SLOT_HALF_LEFT
                )
            elif (
                normalized_slot_half == ATLAS_SLOT_HALF_RIGHT
                and compatible_target is None
            ):
                normalized_slot_half = ATLAS_SLOT_HALF_LEFT

        if normalized_packing_mode == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER:
            compatible_group = _quarter_group_at(
                placements,
                normalized_resolution,
                normalized_x,
                normalized_y,
            )
            if compatible_group is None:
                normalized_slot_quadrant = ATLAS_SLOT_QUADRANT_TOP_LEFT
            elif normalized_slot_quadrant is None:
                normalized_slot_quadrant = _first_available_quadrant(
                    compatible_group
                )
            elif normalized_slot_quadrant in {
                member.slot_quadrant for member in compatible_group
            }:
                normalized_slot_quadrant = _first_available_quadrant(
                    compatible_group
                )

        placement = TextureAtlasPlacement(
            object_id=normalized_id,
            texture_path=normalized_path,
            texture_resolution=normalized_resolution,
            x=normalized_x,
            y=normalized_y,
            size=_logical_slot_size(
                normalized_packing_mode,
                normalized_resolution,
            ),
            packing_mode=normalized_packing_mode,
            slot_half=normalized_slot_half,
            slot_quadrant=normalized_slot_quadrant,
        )
        placements.append(placement)
        _validate_placements_fit(atlas.resolution, placements)
        atlas.placements = placements
        atlas.image_path = None
        return placement

    def unassign_object(self, atlas_id: str, object_id: str) -> bool:
        atlas = self._require_atlas(atlas_id)
        normalized_id = str(object_id)
        placements = [
            placement
            for placement in atlas.placements
            if placement.object_id != normalized_id
        ]
        if len(placements) == len(atlas.placements):
            return False
        atlas.placements = _normalize_partial_slot_placements(placements)
        atlas.image_path = None
        return True

    def repack_atlas(self, atlas_id: str) -> list[TextureAtlasPlacement]:
        atlas = self._require_atlas(atlas_id)
        atlas.placements = _pack_placements(
            atlas.resolution,
            atlas.placements,
        )
        atlas.image_path = None
        return list(atlas.placements)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ATLAS_STATE_SCHEMA_VERSION,
            "selected_atlas_id": self.selected_atlas_id,
            "atlases": [atlas.to_dict() for atlas in self.atlases],
        }

    @classmethod
    def from_dict(cls, payload: object) -> "TextureAtlasData":
        if not isinstance(payload, dict):
            raise ValueError("Texture atlas state must contain an object.")
        schema_version = int(
            payload.get("schema_version", ATLAS_STATE_SCHEMA_VERSION)
        )
        if schema_version not in {
            LEGACY_ATLAS_STATE_SCHEMA_VERSION,
            SYMMETRIC_HALF_ATLAS_STATE_SCHEMA_VERSION,
            SYMMETRIC_QUARTER_ATLAS_STATE_SCHEMA_VERSION,
            SYMMETRIC_PAIR_ATLAS_STATE_SCHEMA_VERSION,
            ATLAS_STATE_SCHEMA_VERSION,
        }:
            raise ValueError(
                f"Unsupported texture atlas schema version: {schema_version}."
            )
        raw_atlases = payload.get("atlases", [])
        if not isinstance(raw_atlases, list):
            raise ValueError("Texture atlases must contain a list.")
        atlases = [
            TextureAtlasRecord.from_dict(raw_atlas)
            for raw_atlas in raw_atlases
        ]
        if schema_version < ATLAS_STATE_SCHEMA_VERSION and any(
            placement.packing_mode
            == ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR
            for atlas in atlases
            for placement in atlas.placements
        ):
            raise ValueError(
                "Symmetric square-pair placements require texture atlas "
                "schema version 5."
            )
        return cls(
            atlases=atlases,
            selected_atlas_id=(
                None
                if payload.get("selected_atlas_id") is None
                else str(payload["selected_atlas_id"])
            ),
        )

    def _require_atlas(self, atlas_id: str) -> TextureAtlasRecord:
        atlas = self.atlas_by_id(atlas_id)
        if atlas is None:
            raise ValueError(f"Unknown texture atlas ID: {str(atlas_id)!r}.")
        return atlas


# ### Atlas output ###
TextureSourceLoader = Callable[[TextureAtlasPlacement], np.ndarray]


def write_texture_atlas_png(
    atlas: TextureAtlasRecord,
    output_path: str | Path,
    *,
    asset_root: str | Path | None = None,
    source_loader: TextureSourceLoader | None = None,
    project_relative_image_path: str | Path | None = None,
) -> Path:
    """Composite the atlas and atomically write an RGBA PNG.

    Persisted placement paths are relative to ``asset_root`` and never resolved
    against the process working directory. A custom loader may be supplied when
    texture bytes are already available in memory. Absolute output paths need a
    separate safe ``project_relative_image_path`` to persist on the record.
    """

    if not isinstance(atlas, TextureAtlasRecord):
        raise TypeError("Atlas must be a TextureAtlasRecord value.")
    if source_loader is None:
        if asset_root is None:
            raise ValueError(
                "Texture atlas PNG output requires an asset root or source loader."
            )
        normalized_asset_root = Path(asset_root).resolve()
        def loader(placement: TextureAtlasPlacement) -> np.ndarray:
            return _load_texture_from_path(normalized_asset_root, placement)
    else:
        loader = source_loader
    canvas = np.zeros((atlas.resolution, atlas.resolution, 4), dtype=np.uint8)
    for placements in _group_placements_by_slot(atlas.placements):
        first = placements[0]
        y_end = first.y + first.size
        x_end = first.x + first.size
        if first.packing_mode == ATLAS_PACKING_MODE_FULL:
            source = _normalize_texture_pixels(loader(first), first)
            canvas[first.y:y_end, first.x:x_end] = source
            continue

        canvas[first.y:y_end, first.x:x_end] = (0, 0, 0, 255)
        if first.packing_mode == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER:
            for placement in placements:
                source = _normalize_texture_pixels(loader(placement), placement)
                content_size = placement.texture_resolution
                offset_x, offset_y = _quadrant_pixel_offset(
                    placement.slot_quadrant,
                    content_size,
                )
                destination_x = placement.x + offset_x
                destination_y = placement.y + offset_y
                canvas[
                    destination_y : destination_y + content_size,
                    destination_x : destination_x + content_size,
                ] = source[:content_size, :content_size]
            continue

        content_width = first.size // 2
        for placement in placements:
            source = _normalize_texture_pixels(loader(placement), placement)
            destination_x = placement.x + (
                content_width
                if placement.slot_half == ATLAS_SLOT_HALF_RIGHT
                else 0
            )
            canvas[
                placement.y:y_end,
                destination_x : destination_x + content_width,
            ] = source[:, :content_width]

    success, encoded = cv2.imencode(".png", canvas)
    if not success:
        raise ValueError("Texture atlas PNG could not be encoded.")
    destination = Path(output_path)
    _write_bytes_atomically(destination, encoded.tobytes())
    if project_relative_image_path is not None:
        atlas.image_path = _normalize_project_relative_path(
            project_relative_image_path
        )
    elif destination.is_absolute():
        atlas.image_path = None
    else:
        atlas.image_path = _normalize_project_relative_path(destination)
    return destination


def write_texture_atlas_metadata(
    atlas: TextureAtlasRecord,
    output_path: str | Path,
) -> Path:
    """Atomically write stable JSON placement metadata for external tools."""

    destination = Path(output_path)
    payload = json.dumps(atlas.to_dict(), indent=2, sort_keys=True).encode("utf-8")
    _write_bytes_atomically(destination, payload)
    return destination


# ### Quadtree packing ###
@dataclass
class _QuadNode:
    x: int
    y: int
    size: int
    occupied: bool = False
    children: tuple["_QuadNode", "_QuadNode", "_QuadNode", "_QuadNode"] | None = None


def _pack_placements(
    atlas_resolution: int,
    placements: list[TextureAtlasPlacement],
) -> list[TextureAtlasPlacement]:
    """Pack full textures and established symmetric groups as logical units."""

    _validate_atlas_resolution(atlas_resolution)
    _validate_placements_fit(atlas_resolution, placements)
    placements = _normalize_partial_slot_placements(placements)
    logical_units: list[list[TextureAtlasPlacement]] = []
    unpaired_half_slots_by_mode_and_resolution: dict[
        tuple[str, int],
        list[TextureAtlasPlacement],
    ] = {}
    ungrouped_quarters_by_resolution: dict[
        int,
        list[TextureAtlasPlacement],
    ] = {}
    for slot_group in _group_placements_by_slot(placements):
        packing_mode = slot_group[0].packing_mode
        if packing_mode == ATLAS_PACKING_MODE_FULL:
            logical_units.append(slot_group)
            continue
        if packing_mode in ATLAS_HALF_SLOT_PACKING_MODES:
            if len(slot_group) == 2:
                logical_units.append(slot_group)
            else:
                placement = slot_group[0]
                unpaired_half_slots_by_mode_and_resolution.setdefault(
                    (packing_mode, placement.texture_resolution),
                    [],
                ).append(placement)
            continue
        if len(slot_group) > 1:
            logical_units.append(slot_group)
            continue
        placement = slot_group[0]
        ungrouped_quarters_by_resolution.setdefault(
            placement.texture_resolution,
            [],
        ).append(placement)

    for mode_and_resolution in sorted(
        unpaired_half_slots_by_mode_and_resolution,
        key=lambda item: (item[0], -item[1]),
    ):
        unpaired = sorted(
            unpaired_half_slots_by_mode_and_resolution[mode_and_resolution],
            key=lambda placement: placement.object_id,
        )
        for index in range(0, len(unpaired), 2):
            members = unpaired[index : index + 2]
            logical_units.append(
                [
                    replace(
                        member,
                        slot_half=(
                            ATLAS_SLOT_HALF_LEFT
                            if member_index == 0
                            else ATLAS_SLOT_HALF_RIGHT
                        ),
                    )
                    for member_index, member in enumerate(members)
                ]
            )

    for resolution in sorted(ungrouped_quarters_by_resolution, reverse=True):
        ungrouped = sorted(
            ungrouped_quarters_by_resolution[resolution],
            key=lambda placement: placement.object_id,
        )
        for index in range(0, len(ungrouped), 4):
            members = ungrouped[index : index + 4]
            logical_units.append(
                [
                    replace(
                        member,
                        slot_quadrant=ATLAS_SLOT_QUADRANT_ORDER[
                            member_index
                        ],
                    )
                    for member_index, member in enumerate(members)
                ]
            )

    root = _QuadNode(0, 0, int(atlas_resolution))
    packed: list[TextureAtlasPlacement] = []
    for logical_unit in sorted(
        logical_units,
        key=lambda unit: (
            -unit[0].size,
            tuple(sorted(member.object_id for member in unit)),
        ),
    ):
        requested_size = logical_unit[0].size
        node = _allocate_quad(root, requested_size)
        if node is None:
            object_labels = ", ".join(
                repr(member.object_id) for member in logical_unit
            )
            raise ValueError(
                f"Texture atlas {atlas_resolution}x{atlas_resolution} has no "
                f"space for texture source(s) {object_labels} at "
                f"{requested_size}x{requested_size}."
            )
        packed.extend(
            replace(
                member,
                x=node.x,
                y=node.y,
                size=requested_size,
            )
            for member in logical_unit
        )
    _validate_placements_fit(atlas_resolution, packed)
    return packed


def _allocate_quad(node: _QuadNode, requested_size: int) -> _QuadNode | None:
    if node.occupied or node.size < requested_size:
        return None
    if node.children is not None:
        for child in node.children:
            allocated = _allocate_quad(child, requested_size)
            if allocated is not None:
                return allocated
        return None
    if node.size == requested_size:
        node.occupied = True
        return node

    half_size = node.size // 2
    node.children = (
        _QuadNode(node.x, node.y, half_size),
        _QuadNode(node.x + half_size, node.y, half_size),
        _QuadNode(node.x, node.y + half_size, half_size),
        _QuadNode(node.x + half_size, node.y + half_size, half_size),
    )
    return _allocate_quad(node, requested_size)


def _find_available_placement(
    atlas_resolution: int,
    object_id: str,
    texture_path: str,
    texture_resolution: int,
    unaffected: list[TextureAtlasPlacement],
    *,
    preferred_position: tuple[int, int] | None,
    packing_mode: str = ATLAS_PACKING_MODE_FULL,
    slot_half: str | None = None,
    slot_quadrant: str | None = None,
) -> TextureAtlasPlacement:
    """Find one aligned slot while treating every other placement as fixed."""

    logical_size = _logical_slot_size(packing_mode, texture_resolution)
    positions = [] if preferred_position is None else [preferred_position]
    for position in _iter_quad_slot_positions(
        0,
        0,
        int(atlas_resolution),
        logical_size,
    ):
        if position not in positions:
            positions.append(position)
    for position in _iter_base_grid_positions(
        int(atlas_resolution),
        logical_size,
    ):
        if position not in positions:
            positions.append(position)
    for x, y in positions:
        try:
            candidate = TextureAtlasPlacement(
                object_id=object_id,
                texture_path=texture_path,
                texture_resolution=texture_resolution,
                x=x,
                y=y,
                size=logical_size,
                packing_mode=packing_mode,
                slot_half=slot_half,
                slot_quadrant=slot_quadrant,
            )
            _validate_placements_fit(
                atlas_resolution,
                [*unaffected, candidate],
            )
        except ValueError:
            continue
        return candidate
    raise ValueError(
        f"Texture atlas {atlas_resolution}x{atlas_resolution} has no space "
        f"for texture source {object_id!r} at "
        f"{logical_size}x{logical_size}."
    )


def _iter_quad_slot_positions(
    x: int,
    y: int,
    node_size: int,
    requested_size: int,
) -> Iterator[tuple[int, int]]:
    if node_size == requested_size:
        yield (x, y)
        return
    half_size = node_size // 2
    for child_x, child_y in (
        (x, y),
        (x + half_size, y),
        (x, y + half_size),
        (x + half_size, y + half_size),
    ):
        yield from _iter_quad_slot_positions(
            child_x,
            child_y,
            half_size,
            requested_size,
        )


def _iter_base_grid_positions(
    atlas_resolution: int,
    requested_size: int,
) -> Iterator[tuple[int, int]]:
    """Yield every in-bounds origin on the shared 512-pixel Atlas grid."""

    maximum_origin = int(atlas_resolution) - int(requested_size)
    for y in range(0, maximum_origin + 1, ATLAS_BASE_CELL_SIZE):
        for x in range(0, maximum_origin + 1, ATLAS_BASE_CELL_SIZE):
            yield (x, y)


# ### Validation helpers ###
def _normalize_packing_mode(value: object) -> str:
    packing_mode = str(value).strip().lower()
    if packing_mode not in ATLAS_PACKING_MODES:
        raise ValueError("Unknown texture atlas packing mode.")
    return packing_mode


def _normalize_slot_half(
    packing_mode: str,
    value: object,
    *,
    allow_unspecified_half: bool,
) -> str | None:
    if packing_mode not in ATLAS_HALF_SLOT_PACKING_MODES:
        if value is not None:
            raise ValueError(
                "Only symmetric half or pair textures can select a slot half."
            )
        return None
    if value is None and allow_unspecified_half:
        return None
    slot_half = str(value).strip().lower()
    if slot_half not in ATLAS_SLOT_HALVES:
        raise ValueError(
            "Symmetric half or pair textures require a left or right slot."
        )
    return slot_half


def _normalize_slot_quadrant(
    packing_mode: str,
    value: object,
    *,
    allow_unspecified_quadrant: bool,
) -> str | None:
    if packing_mode != ATLAS_PACKING_MODE_SYMMETRIC_QUARTER:
        if value is not None:
            raise ValueError(
                "Only symmetric quarter textures can select a quadrant."
            )
        return None
    if value is None and allow_unspecified_quadrant:
        return None
    slot_quadrant = str(value).strip().lower()
    if slot_quadrant not in ATLAS_SLOT_QUADRANTS:
        raise ValueError(
            "Symmetric quarter textures require one Atlas quadrant."
        )
    return slot_quadrant


def _logical_slot_size(packing_mode: str, texture_resolution: int) -> int:
    if (
        packing_mode in ATLAS_LIMITED_RESOLUTION_PACKING_MODES
        and int(texture_resolution) not in SYMMETRIC_PACKED_TEXTURE_RESOLUTIONS
    ):
        raise ValueError(
            "High-density symmetric textures must use 512 or 1024 content."
        )
    multiplier = 2 if packing_mode in ATLAS_DOUBLE_SIZED_PACKING_MODES else 1
    return int(texture_resolution) * multiplier


def _group_placements_by_slot(
    placements: list[TextureAtlasPlacement],
) -> list[list[TextureAtlasPlacement]]:
    groups: dict[tuple[int, int, int], list[TextureAtlasPlacement]] = {}
    for placement in placements:
        groups.setdefault(
            (placement.x, placement.y, placement.size),
            [],
        ).append(placement)
    return list(groups.values())


def _normalize_partial_slot_placements(
    placements: list[TextureAtlasPlacement],
) -> list[TextureAtlasPlacement]:
    """Compact symmetric slot members into their canonical region order."""

    normalized: list[TextureAtlasPlacement] = []
    for slot_group in _group_placements_by_slot(placements):
        packing_mode = slot_group[0].packing_mode
        if packing_mode in ATLAS_HALF_SLOT_PACKING_MODES:
            ordered = sorted(
                slot_group,
                key=lambda placement: (
                    0
                    if placement.slot_half == ATLAS_SLOT_HALF_LEFT
                    else 1,
                    placement.object_id,
                ),
            )
            normalized.extend(
                replace(
                    placement,
                    slot_half=(
                        ATLAS_SLOT_HALF_LEFT
                        if index == 0
                        else ATLAS_SLOT_HALF_RIGHT
                    ),
                )
                for index, placement in enumerate(ordered)
            )
            continue
        if packing_mode == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER:
            quadrant_index = {
                quadrant: index
                for index, quadrant in enumerate(ATLAS_SLOT_QUADRANT_ORDER)
            }
            ordered = sorted(
                slot_group,
                key=lambda placement: (
                    quadrant_index.get(placement.slot_quadrant, 4),
                    placement.object_id,
                ),
            )
            normalized.extend(
                replace(
                    placement,
                    slot_quadrant=ATLAS_SLOT_QUADRANT_ORDER[index],
                )
                for index, placement in enumerate(ordered)
            )
            continue
        normalized.extend(slot_group)
    return normalized


def _find_compatible_unpaired_half_slot(
    placements: list[TextureAtlasPlacement],
    texture_resolution: int,
    packing_mode: str,
) -> TextureAtlasPlacement | None:
    for slot_group in _group_placements_by_slot(placements):
        if len(slot_group) != 1:
            continue
        placement = slot_group[0]
        if (
            placement.packing_mode == packing_mode
            and placement.slot_half == ATLAS_SLOT_HALF_LEFT
            and placement.texture_resolution == int(texture_resolution)
        ):
            return placement
    return None


def _find_compatible_quarter_group(
    placements: list[TextureAtlasPlacement],
    texture_resolution: int,
) -> list[TextureAtlasPlacement] | None:
    candidates = [
        slot_group
        for slot_group in _group_placements_by_slot(placements)
        if len(slot_group) < len(ATLAS_SLOT_QUADRANT_ORDER)
        and slot_group[0].packing_mode
        == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER
        and slot_group[0].texture_resolution == int(texture_resolution)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda group: (
            group[0].y,
            group[0].x,
            tuple(member.object_id for member in group),
        ),
    )


def _quarter_group_at(
    placements: list[TextureAtlasPlacement],
    texture_resolution: int,
    x: int,
    y: int,
) -> list[TextureAtlasPlacement] | None:
    expected_size = int(texture_resolution) * 2
    for slot_group in _group_placements_by_slot(placements):
        first = slot_group[0]
        if (
            len(slot_group) < len(ATLAS_SLOT_QUADRANT_ORDER)
            and first.packing_mode == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER
            and first.texture_resolution == int(texture_resolution)
            and first.size == expected_size
            and first.x == int(x)
            and first.y == int(y)
        ):
            return slot_group
    return None


def _first_available_quadrant(
    placements: list[TextureAtlasPlacement],
) -> str:
    occupied = {placement.slot_quadrant for placement in placements}
    for quadrant in ATLAS_SLOT_QUADRANT_ORDER:
        if quadrant not in occupied:
            return quadrant
    raise ValueError("The symmetric quarter Atlas slot is already full.")


def _quadrant_pixel_offset(
    slot_quadrant: str | None,
    content_size: int,
) -> tuple[int, int]:
    if slot_quadrant not in ATLAS_SLOT_QUADRANTS:
        raise ValueError("The symmetric quarter Atlas quadrant is invalid.")
    offset_x = (
        int(content_size)
        if slot_quadrant
        in {
            ATLAS_SLOT_QUADRANT_TOP_RIGHT,
            ATLAS_SLOT_QUADRANT_BOTTOM_RIGHT,
        }
        else 0
    )
    offset_y = (
        int(content_size)
        if slot_quadrant
        in {
            ATLAS_SLOT_QUADRANT_BOTTOM_LEFT,
            ATLAS_SLOT_QUADRANT_BOTTOM_RIGHT,
        }
        else 0
    )
    return offset_x, offset_y


def _validate_atlas_resolution(resolution: object) -> None:
    if isinstance(resolution, bool) or int(resolution) not in ATLAS_RESOLUTIONS:
        raise ValueError("Texture atlas resolution must be 2048 or 4096.")


def _validate_texture_resolution(resolution: object) -> None:
    if (
        isinstance(resolution, bool)
        or int(resolution) not in OBJECT_TEXTURE_RESOLUTIONS
    ):
        raise ValueError("Object texture resolution must be 512, 1024, or 2048.")


def _validate_atlas_name(name: object) -> None:
    _validate_nonempty_text(name, "Texture atlas name")
    if len(str(name)) > MAX_ATLAS_NAME_LENGTH:
        raise ValueError("Texture atlas name is too long.")


def _validate_atlas_id(atlas_id: object) -> None:
    raw_id = str(atlas_id)
    if (
        len(raw_id) > MAX_ATLAS_ID_LENGTH
        or ATLAS_ID_PATTERN.fullmatch(raw_id) is None
    ):
        raise ValueError(
            "Texture atlas ID must be a filename-safe identifier containing "
            "only letters, numbers, dots, underscores, or hyphens."
        )


def _validate_nonempty_text(value: object, label: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{label} cannot be empty.")


def _normalize_atlas_coordinate(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be an integer.") from error
    if normalized != value:
        raise ValueError(f"{label} must be an integer.")
    return normalized


def _normalize_project_relative_path(path: object) -> str:
    _validate_nonempty_text(path, "Texture path")
    raw_path = str(path).strip()
    if len(raw_path) > MAX_TEXTURE_PATH_LENGTH:
        raise ValueError("Texture path is too long.")
    windows_path = PureWindowsPath(raw_path)
    normalized_path = PurePosixPath(raw_path.replace("\\", "/"))
    if (
        raw_path == "."
        or windows_path.is_absolute()
        or normalized_path.is_absolute()
        or ".." in normalized_path.parts
    ):
        raise ValueError(
            "Texture atlas asset paths must be safe project-relative paths."
        )
    return str(normalized_path)


def _validate_placements_fit(
    atlas_resolution: int,
    placements: list[TextureAtlasPlacement],
) -> None:
    for placement in placements:
        if (
            placement.x % ATLAS_BASE_CELL_SIZE
            or placement.y % ATLAS_BASE_CELL_SIZE
        ):
            raise ValueError(
                "Texture atlas placement must align to the 512-pixel grid."
            )
        if (
            placement.x + placement.size > atlas_resolution
            or placement.y + placement.size > atlas_resolution
        ):
            raise ValueError("Texture atlas placement exceeds atlas bounds.")
    slot_groups = _group_placements_by_slot(placements)
    for slot_group in slot_groups:
        first = slot_group[0]
        if any(
            placement.packing_mode != first.packing_mode
            or placement.texture_resolution != first.texture_resolution
            or placement.size != first.size
            for placement in slot_group[1:]
        ):
            raise ValueError("Atlas placements overlap with incompatible slots.")
        if first.packing_mode == ATLAS_PACKING_MODE_FULL:
            if len(slot_group) != 1:
                raise ValueError("Full texture atlas placements cannot overlap.")
            continue
        if first.packing_mode in ATLAS_HALF_SLOT_PACKING_MODES:
            if (
                len(slot_group) == 1
                and first.slot_half == ATLAS_SLOT_HALF_RIGHT
            ):
                raise ValueError(
                    "A right-half atlas placement requires a matching left half."
                )
            if len(slot_group) > 2:
                raise ValueError(
                    "A texture atlas slot can contain at most two halves."
                )
            expected_halves = set(
                (ATLAS_SLOT_HALF_LEFT, ATLAS_SLOT_HALF_RIGHT)[: len(slot_group)]
            )
            if {placement.slot_half for placement in slot_group} != expected_halves:
                raise ValueError(
                    "Symmetric Atlas pairs must occupy left before right."
                )
            continue
        if len(slot_group) > len(ATLAS_SLOT_QUADRANT_ORDER):
            raise ValueError(
                "A symmetric quarter Atlas slot can contain at most four sources."
            )
        expected_quadrants = set(
            ATLAS_SLOT_QUADRANT_ORDER[: len(slot_group)]
        )
        if {
            placement.slot_quadrant for placement in slot_group
        } != expected_quadrants:
            raise ValueError(
                "Symmetric quarter Atlas members must occupy quadrants row-major."
            )

    for index, first_group in enumerate(slot_groups):
        first = first_group[0]
        for second_group in slot_groups[index + 1 :]:
            second = second_group[0]
            if not (
                first.x + first.size <= second.x
                or second.x + second.size <= first.x
                or first.y + first.size <= second.y
                or second.y + second.size <= first.y
            ):
                raise ValueError("Texture atlas logical slots overlap.")


# ### Image helpers ###
def _load_texture_from_path(
    asset_root: Path,
    placement: TextureAtlasPlacement,
) -> np.ndarray:
    source_path = (asset_root / placement.texture_path).resolve()
    if not source_path.is_relative_to(asset_root):
        raise ValueError("Texture path escapes the project asset directory.")
    try:
        encoded = np.frombuffer(source_path.read_bytes(), dtype=np.uint8)
    except OSError as error:
        raise ValueError(
            f"Unable to read texture image: {placement.texture_path}"
        ) from error
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Texture image is invalid: {placement.texture_path}")
    return image


def _normalize_texture_pixels(
    source: np.ndarray,
    placement: TextureAtlasPlacement,
) -> np.ndarray:
    image = np.asarray(source)
    expected_shape = (placement.size, placement.size)
    if image.dtype != np.uint8:
        raise ValueError("Texture pixels must use uint8 values.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Texture pixels must contain grayscale, BGR, or BGRA data.")
    if image.shape[:2] != expected_shape:
        raise ValueError(
            f"Texture source {placement.object_id!r} must be "
            f"{placement.size}x{placement.size}; received "
            f"{image.shape[1]}x{image.shape[0]}."
        )
    return np.ascontiguousarray(image)


# ### Atomic output helpers ###
def _write_bytes_atomically(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        file_descriptor, raw_temporary_path = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        temporary_path = Path(raw_temporary_path)
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except OSError as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ValueError(f"Unable to write texture atlas file: {destination}") from error
