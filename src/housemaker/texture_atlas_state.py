# ### Imports ###
from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable

import cv2
import numpy as np


# ### Constants ###
ATLAS_RESOLUTIONS = frozenset({2048, 4096})
OBJECT_TEXTURE_RESOLUTIONS = frozenset({512, 1024, 2048})
MAX_ATLAS_COUNT = 1_000
MAX_ATLAS_NAME_LENGTH = 200
MAX_ATLAS_OBJECT_COUNT = 4_096
MAX_ATLAS_ID_LENGTH = 128
MAX_TEXTURE_PATH_LENGTH = 32_768
ATLAS_STATE_SCHEMA_VERSION = 1
ATLAS_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


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

    def __post_init__(self) -> None:
        _validate_nonempty_text(self.object_id, "Texture source ID")
        object.__setattr__(
            self,
            "texture_path",
            _normalize_project_relative_path(self.texture_path),
        )
        _validate_texture_resolution(self.texture_resolution)
        if int(self.size) != int(self.texture_resolution):
            raise ValueError("Atlas placement size must match texture resolution.")
        if int(self.x) < 0 or int(self.y) < 0:
            raise ValueError("Atlas placement coordinates cannot be negative.")

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
    ) -> TextureAtlasPlacement:
        """Add or update one source without moving unrelated placements."""

        atlas = self._require_atlas(atlas_id)
        normalized_id = str(object_id)
        normalized_path = _normalize_project_relative_path(texture_path)
        _validate_nonempty_text(normalized_id, "Object ID")
        _validate_texture_resolution(texture_resolution)
        existing = atlas.placement_for_object(normalized_id)
        if existing is None and len(atlas.placements) >= MAX_ATLAS_OBJECT_COUNT:
            raise ValueError("Texture atlas contains too many texture sources.")
        unaffected = [
            placement
            for placement in atlas.placements
            if placement.object_id != normalized_id
        ]
        placement = _find_available_placement(
            atlas.resolution,
            normalized_id,
            normalized_path,
            int(texture_resolution),
            unaffected,
            preferred_position=(
                None if existing is None else (existing.x, existing.y)
            ),
        )
        if existing is None:
            atlas.placements.append(placement)
        else:
            atlas.placements = [
                placement if candidate.object_id == normalized_id else candidate
                for candidate in atlas.placements
            ]
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
    ) -> TextureAtlasPlacement:
        """Add or move one source at an exact aligned atlas slot."""

        atlas = self._require_atlas(atlas_id)
        normalized_id = str(object_id)
        normalized_path = _normalize_project_relative_path(texture_path)
        normalized_resolution = int(texture_resolution)
        normalized_x = _normalize_atlas_coordinate(x, "Atlas placement x")
        normalized_y = _normalize_atlas_coordinate(y, "Atlas placement y")
        _validate_nonempty_text(normalized_id, "Object ID")
        _validate_texture_resolution(normalized_resolution)
        if (
            atlas.placement_for_object(normalized_id) is None
            and len(atlas.placements) >= MAX_ATLAS_OBJECT_COUNT
        ):
            raise ValueError("Texture atlas contains too many texture sources.")

        placement = TextureAtlasPlacement(
            object_id=normalized_id,
            texture_path=normalized_path,
            texture_resolution=normalized_resolution,
            x=normalized_x,
            y=normalized_y,
            size=normalized_resolution,
        )
        placements = [
            candidate
            for candidate in atlas.placements
            if candidate.object_id != normalized_id
        ]
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
        atlas.placements = placements
        atlas.image_path = None
        return True

    def repack_atlas(self, atlas_id: str) -> list[TextureAtlasPlacement]:
        atlas = self._require_atlas(atlas_id)
        atlas.placements = _pack_assignments(
            atlas.resolution,
            [
                (
                    placement.object_id,
                    placement.texture_path,
                    placement.texture_resolution,
                )
                for placement in atlas.placements
            ],
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
        if schema_version != ATLAS_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported texture atlas schema version: {schema_version}."
            )
        raw_atlases = payload.get("atlases", [])
        if not isinstance(raw_atlases, list):
            raise ValueError("Texture atlases must contain a list.")
        return cls(
            atlases=[
                TextureAtlasRecord.from_dict(raw_atlas)
                for raw_atlas in raw_atlases
            ],
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
        loader = lambda placement: _load_texture_from_path(
            normalized_asset_root,
            placement,
        )
    else:
        loader = source_loader
    canvas = np.zeros((atlas.resolution, atlas.resolution, 4), dtype=np.uint8)
    for placement in atlas.placements:
        source = _normalize_texture_pixels(loader(placement), placement)
        y_end = placement.y + placement.size
        x_end = placement.x + placement.size
        canvas[placement.y:y_end, placement.x:x_end] = source

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


def _pack_assignments(
    atlas_resolution: int,
    assignments: list[tuple[str, str, int]],
) -> list[TextureAtlasPlacement]:
    _validate_atlas_resolution(atlas_resolution)
    if len(assignments) > MAX_ATLAS_OBJECT_COUNT:
        raise ValueError("Texture atlas contains too many texture sources.")
    normalized: list[tuple[str, str, int]] = []
    seen_ids: set[str] = set()
    for object_id, texture_path, texture_resolution in assignments:
        normalized_id = str(object_id)
        normalized_path = _normalize_project_relative_path(texture_path)
        normalized_resolution = int(texture_resolution)
        _validate_nonempty_text(normalized_id, "Object ID")
        _validate_texture_resolution(normalized_resolution)
        if normalized_id in seen_ids:
            raise ValueError(
                "A texture source can only occur once in one texture atlas."
            )
        seen_ids.add(normalized_id)
        normalized.append(
            (normalized_id, normalized_path, normalized_resolution)
        )

    root = _QuadNode(0, 0, int(atlas_resolution))
    placements: list[TextureAtlasPlacement] = []
    for object_id, texture_path, texture_resolution in sorted(
        normalized,
        key=lambda assignment: (-assignment[2], assignment[0]),
    ):
        node = _allocate_quad(root, texture_resolution)
        if node is None:
            raise ValueError(
                f"Texture atlas {atlas_resolution}x{atlas_resolution} has no "
                f"space for texture source {object_id!r} at "
                f"{texture_resolution}x{texture_resolution}."
            )
        placements.append(
            TextureAtlasPlacement(
                object_id=object_id,
                texture_path=texture_path,
                texture_resolution=texture_resolution,
                x=node.x,
                y=node.y,
                size=node.size,
            )
        )
    return placements


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
) -> TextureAtlasPlacement:
    """Find one aligned slot while treating every other placement as fixed."""

    positions = [] if preferred_position is None else [preferred_position]
    positions.extend(
        position
        for position in _iter_quad_slot_positions(
            0,
            0,
            int(atlas_resolution),
            int(texture_resolution),
        )
        if position != preferred_position
    )
    for x, y in positions:
        try:
            candidate = TextureAtlasPlacement(
                object_id=object_id,
                texture_path=texture_path,
                texture_resolution=texture_resolution,
                x=x,
                y=y,
                size=texture_resolution,
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
        f"{texture_resolution}x{texture_resolution}."
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


# ### Validation helpers ###
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
        if placement.x % placement.size or placement.y % placement.size:
            raise ValueError(
                "Texture atlas placement must align to its texture-size grid."
            )
        if (
            placement.x + placement.size > atlas_resolution
            or placement.y + placement.size > atlas_resolution
        ):
            raise ValueError("Texture atlas placement exceeds atlas bounds.")
    for index, first in enumerate(placements):
        for second in placements[index + 1 :]:
            if not (
                first.x + first.size <= second.x
                or second.x + second.size <= first.x
                or first.y + first.size <= second.y
                or second.y + second.size <= first.y
            ):
                raise ValueError("Texture atlas placements overlap.")


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
