# ### Imports ###
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from housemaker.door_state import (
    DOOR_SIDE_DUPLICATION_SIDES,
    DOOR_SIDE_DUPLICATION_UV_MODE,
)
from housemaker.glb import (
    HALF_MESH_EXTRAS_KEY,
    INSTANCE_SOURCE_ID_METADATA_KEY,
    INSTANCE_SOURCE_NAME_METADATA_KEY,
    PlacedGeneratedModel,
    build_placed_generated_model_gltf_transform,
)
from housemaker.tour_state import TourData, tours_to_runtime_dicts

# ### Constants ###
RUNTIME_SCENE_FORMAT = "housemaker-r3f-scene"
RUNTIME_SCENE_VERSION = 8
GLB_MAGIC = b"glTF"
GLB_VERSION = 2
GLB_JSON_CHUNK_TYPE = b"JSON"
GLB_BINARY_CHUNK_TYPE = b"BIN\0"
GLB_HEADER_BYTE_COUNT = 12
GLB_CHUNK_HEADER_BYTE_COUNT = 8
INSTANCE_SOURCE_SCENE_NAME = "HouseMaker Instance Sources"


# ### Runtime door reconstruction models ###
@dataclass(frozen=True, slots=True)
class DoorBodyReconstruction:
    """Runtime instructions for recreating one omitted door-body depth half."""

    placement_object_id: str
    body_object_id: str
    kept_side: str
    mirror_point: tuple[float, float, float]
    mirror_normal: tuple[float, float, float]

    def __post_init__(self) -> None:
        placement_object_id = str(self.placement_object_id).strip()
        body_object_id = str(self.body_object_id).strip()
        kept_side = str(self.kept_side).strip().lower()
        if not placement_object_id or not body_object_id:
            raise ValueError("Door reconstruction object IDs cannot be empty.")
        if kept_side not in DOOR_SIDE_DUPLICATION_SIDES:
            raise ValueError("Door reconstruction must keep Front or Back.")
        point = _normalize_runtime_vector(self.mirror_point, "point")
        normal = _normalize_runtime_vector(self.mirror_normal, "normal")
        if float(np.linalg.norm(normal)) <= 0.0:
            raise ValueError("Door reconstruction mirror normals cannot be zero.")
        object.__setattr__(self, "placement_object_id", placement_object_id)
        object.__setattr__(self, "body_object_id", body_object_id)
        object.__setattr__(self, "kept_side", kept_side)
        object.__setattr__(self, "mirror_point", point)
        object.__setattr__(self, "mirror_normal", normal)

    def to_runtime_dict(self) -> dict[str, object]:
        return {
            "placementObjectId": self.placement_object_id,
            "bodyObjectId": self.body_object_id,
            "sideDuplication": {
                "keptSide": self.kept_side,
                "mirrorPlane": {
                    "point": list(self.mirror_point),
                    "normal": list(self.mirror_normal),
                },
                "uvMode": DOOR_SIDE_DUPLICATION_UV_MODE,
                "applyAfter": HALF_MESH_EXTRAS_KEY,
            },
        }


# ### Public helpers ###


def write_runtime_scene_manifest(
    glb_path: str | Path,
    *,
    source_placements: Mapping[str, PlacedGeneratedModel],
    instance_placements: Sequence[PlacedGeneratedModel],
    tours: Sequence[TourData] = (),
    door_body_reconstructions: Sequence[DoorBodyReconstruction] = (),
) -> Path:
    """Write the R3F runtime companion JSON next to an exported GLB."""

    normalized_glb_path = Path(glb_path)
    try:
        original_glb_bytes = normalized_glb_path.read_bytes()
    except OSError as error:
        raise OSError(
            f"Unable to read the exported GLB: {normalized_glb_path}"
        ) from error
    normalized_sources = _normalize_source_placements(source_placements)
    normalized_instances = _normalize_instance_placements(
        instance_placements,
        normalized_sources,
    )
    source_groups = _build_instance_source_groups(
        normalized_sources,
        normalized_instances,
    )
    glb_bytes = _move_instance_sources_to_library_scene(
        original_glb_bytes,
        source_groups,
    )
    if glb_bytes != original_glb_bytes:
        _write_bytes_atomically(normalized_glb_path, glb_bytes)
    manifest = build_runtime_scene_manifest(
        glb_name=normalized_glb_path.name,
        glb_bytes=glb_bytes,
        source_placements=normalized_sources,
        instance_placements=normalized_instances,
        tours=tours,
        door_body_reconstructions=door_body_reconstructions,
    )
    manifest_path = normalized_glb_path.with_suffix(".json")
    _write_json_atomically(manifest_path, manifest)
    return manifest_path


def build_runtime_scene_manifest(
    *,
    glb_name: str,
    glb_bytes: bytes,
    source_placements: Mapping[str, PlacedGeneratedModel],
    instance_placements: Sequence[PlacedGeneratedModel],
    tours: Sequence[TourData] = (),
    door_body_reconstructions: Sequence[DoorBodyReconstruction] = (),
) -> dict[str, object]:
    """Build a versioned runtime manifest for instances and camera tours."""

    if not isinstance(glb_bytes, bytes) or not glb_bytes:
        raise ValueError("A runtime scene manifest requires exported GLB bytes.")
    normalized_sources = _normalize_source_placements(source_placements)
    normalized_instances = _normalize_instance_placements(
        instance_placements,
        normalized_sources,
    )
    normalized_door_reconstructions = _normalize_door_body_reconstructions(
        door_body_reconstructions
    )

    groups: list[dict[str, object]] = []
    for _source_id, source_model_placement, group_placements in (
        _build_instance_source_groups(normalized_sources, normalized_instances)
    ):
        group_instances = [
            {
                "id": placement.object_id,
                "worldMatrix": _matrix_to_column_major(
                    build_placed_generated_model_gltf_transform(placement)
                ),
            }
            for placement in sorted(
                group_placements,
                key=lambda value: value.object_id,
            )
        ]
        group: dict[str, object] = {
            "sourceNodeName": _get_instance_source_node_name(
                source_model_placement
            ),
            "instances": group_instances,
        }
        if _get_instance_source_symmetry(source_model_placement) is not None:
            group[HALF_MESH_EXTRAS_KEY] = True
        groups.append(group)

    return {
        "format": RUNTIME_SCENE_FORMAT,
        "version": RUNTIME_SCENE_VERSION,
        "coordinateSystem": "gltf-y-up",
        "matrixLayout": "column-major",
        "asset": {
            "glb": str(glb_name),
            "sha256": hashlib.sha256(glb_bytes).hexdigest(),
        },
        "instanceGroups": groups,
        "doorBodyReconstructions": [
            reconstruction.to_runtime_dict()
            for reconstruction in normalized_door_reconstructions
        ],
        "tours": tours_to_runtime_dicts(tours),
    }


# ### Validation helpers ###
def _normalize_runtime_vector(
    raw_values: object,
    field_name: str,
) -> tuple[float, float, float]:
    if (
        isinstance(raw_values, (str, bytes, bytearray))
        or not isinstance(raw_values, Sequence)
        or len(raw_values) != 3
    ):
        raise ValueError(
            f"Door reconstruction mirror {field_name} must contain XYZ values."
        )
    values = tuple(float(value) for value in raw_values)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"Door reconstruction mirror {field_name} must be finite."
        )
    return tuple(0.0 if abs(value) <= 1e-12 else value for value in values)


def _normalize_door_body_reconstructions(
    values: Sequence[DoorBodyReconstruction],
) -> tuple[DoorBodyReconstruction, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values,
        Sequence,
    ):
        raise TypeError("Door body reconstructions must contain a sequence.")
    normalized = tuple(values)
    if not all(isinstance(value, DoorBodyReconstruction) for value in normalized):
        raise TypeError(
            "Door body reconstructions require DoorBodyReconstruction values."
        )
    keys = [
        (value.placement_object_id, value.body_object_id)
        for value in normalized
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Door body reconstruction targets must be unique.")
    return tuple(
        sorted(
            normalized,
            key=lambda value: (
                value.placement_object_id,
                value.body_object_id,
            ),
        )
    )


def _normalize_source_placements(
    source_placements: Mapping[str, PlacedGeneratedModel],
) -> dict[str, PlacedGeneratedModel]:
    if not isinstance(source_placements, Mapping):
        raise TypeError("Instance sources must contain an object-placement mapping.")
    normalized: dict[str, PlacedGeneratedModel] = {}
    for raw_source_id, placement in source_placements.items():
        source_id = str(raw_source_id).strip()
        if not source_id or not isinstance(placement, PlacedGeneratedModel):
            raise TypeError("Instance sources contain an invalid placed object.")
        if source_id in normalized:
            raise ValueError("Instance source IDs must be globally unique.")
        if placement.object_id != source_id:
            raise ValueError("Instance source keys must match their placed object IDs.")
        if placement.source_object_id != source_id:
            raise ValueError("Instance sources must identify themselves as their source.")
        normalized[source_id] = placement
    return normalized


def _normalize_instance_placements(
    instance_placements: Sequence[PlacedGeneratedModel],
    source_placements: Mapping[str, PlacedGeneratedModel],
) -> tuple[PlacedGeneratedModel, ...]:
    if isinstance(instance_placements, (str, bytes, bytearray)) or not isinstance(
        instance_placements,
        Sequence,
    ):
        raise TypeError("Runtime instances must contain placed objects.")
    normalized = tuple(instance_placements)
    if not all(isinstance(value, PlacedGeneratedModel) for value in normalized):
        raise TypeError("Runtime instances must contain placed objects.")
    source_ids: set[str] = set()
    for instance in normalized:
        source_id = instance.source_object_id
        if source_id is None:
            raise ValueError("Runtime instances require a generated source object.")
        source_ids.add(source_id)
    occupied_ids = set(source_placements) | source_ids
    for instance in normalized:
        if instance.object_id in occupied_ids:
            raise ValueError(
                "Runtime source and instance IDs must be globally unique."
            )
        occupied_ids.add(instance.object_id)
    return tuple(sorted(normalized, key=lambda value: value.object_id))


# ### Instance-source helpers ###
def _build_instance_source_groups(
    source_placements: Mapping[str, PlacedGeneratedModel],
    instance_placements: Sequence[PlacedGeneratedModel],
) -> tuple[
    tuple[
        str,
        PlacedGeneratedModel,
        tuple[PlacedGeneratedModel, ...],
    ],
    ...,
]:
    """Resolve one stable prototype and all runtime transforms per source."""

    groups: list[
        tuple[str, PlacedGeneratedModel, tuple[PlacedGeneratedModel, ...]]
    ] = []
    occupied_node_names: set[str] = set()
    source_ids = sorted(
        {
            str(placement.source_object_id)
            for placement in instance_placements
            if placement.source_object_id is not None
        }
    )
    for source_id in source_ids:
        runtime_instances = tuple(
            placement
            for placement in instance_placements
            if placement.source_object_id == source_id
        )
        source = source_placements.get(source_id)
        source_model_placement = source or min(
            runtime_instances,
            key=lambda value: value.object_id,
        )
        expected_name = _get_instance_source_node_name(source_model_placement)
        if any(
            _get_instance_source_node_name(placement) != expected_name
            for placement in runtime_instances
        ):
            raise ValueError(
                f"Runtime instances for {source_id!r} do not share one object name."
            )
        expected_symmetry = _get_instance_source_symmetry(source_model_placement)
        if any(
            _get_instance_source_symmetry(placement) != expected_symmetry
            for placement in runtime_instances
        ):
            raise ValueError(
                f"Runtime instances for {source_id!r} do not share one "
                "half-mesh mirror plane."
            )
        if expected_name in occupied_node_names:
            raise ValueError(
                "Instanced objects must have unique names before export."
            )
        occupied_node_names.add(expected_name)
        group_placements = (
            runtime_instances if source is None else (*runtime_instances, source)
        )
        groups.append(
            (
                source_id,
                source_model_placement,
                tuple(group_placements),
            )
        )
    return tuple(groups)


def _get_instance_source_node_name(
    placement: PlacedGeneratedModel,
) -> str:
    """Return the single detached glTF root used by one instance group."""

    assert placement.object_name is not None
    return placement.object_name


def _get_instance_source_symmetry(
    placement: PlacedGeneratedModel,
) -> tuple[str, float] | None:
    """Return the authored source-space mirror definition, when present."""

    orientation = placement.symmetric_preview_orientation
    if orientation is None:
        return None
    plane_coordinate = placement.symmetric_preview_plane_coordinate
    assert plane_coordinate is not None
    return orientation, float(plane_coordinate)


# ### Detached glTF prototype helpers ###
def _move_instance_sources_to_library_scene(
    glb_bytes: bytes,
    source_groups: Sequence[
        tuple[
            str,
            PlacedGeneratedModel,
            tuple[PlacedGeneratedModel, ...],
        ]
    ],
) -> bytes:
    """Move Atlas-processed source roots out of the default rendered scene."""

    if not source_groups:
        return glb_bytes
    document, binary_payload = _parse_glb(glb_bytes, "The exported GLB is invalid.")
    source_root_indices: list[int] = []
    for source_id, placement, _group_placements in source_groups:
        root_name = _get_instance_source_node_name(placement)
        source_root_index = _find_integrated_instance_source_root(
            document,
            source_id=source_id,
            root_name=root_name,
        )
        if source_root_index is None:
            raise ValueError(
                f"The exported GLB is missing the prepared instance source "
                f"{root_name!r}. Rebuild the export and try again."
            )
        _detach_instance_source_root(document, source_root_index)
        source_root_indices.append(source_root_index)
    scenes = _get_gltf_collection(document, "scenes", create=True)
    if any(
        isinstance(scene, Mapping)
        and scene.get("name") == INSTANCE_SOURCE_SCENE_NAME
        for scene in scenes
    ):
        raise ValueError("The exported GLB already has an instance-source scene.")
    scenes.append(
        {
            "name": INSTANCE_SOURCE_SCENE_NAME,
            "nodes": source_root_indices,
        }
    )
    return _serialize_glb(document, binary_payload)


def _find_integrated_instance_source_root(
    document: Mapping[str, object],
    *,
    source_id: str,
    root_name: str,
) -> int | None:
    """Find one Atlas-processed source root already present in the GLB."""

    nodes = _get_gltf_collection(document, "nodes")
    matches: list[int] = []
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping) or node.get("name") != root_name:
            continue
        extras = node.get("extras")
        if (
            isinstance(extras, Mapping)
            and extras.get(INSTANCE_SOURCE_ID_METADATA_KEY) == source_id
            and extras.get(INSTANCE_SOURCE_NAME_METADATA_KEY) == root_name
        ):
            matches.append(index)
    if len(matches) > 1:
        raise ValueError(
            f"The exported GLB contains duplicate roots for {root_name!r}."
        )
    return matches[0] if matches else None


def _detach_instance_source_root(
    document: Mapping[str, object],
    root_index: int,
) -> None:
    """Remove one integrated source root from every renderable glTF scene."""

    nodes = _get_gltf_collection(document, "nodes")
    _require_gltf_index(root_index, len(nodes))
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise TypeError("The exported GLB contains an invalid node.")
        children = node.get("children")
        if not isinstance(children, list):
            continue
        if root_index in children:
            raise ValueError(
                "An integrated instance source must be a scene root, not a child "
                f"of node {index}."
            )
    scenes = _get_gltf_collection(document, "scenes")
    found = False
    for scene in scenes:
        if not isinstance(scene, dict):
            raise TypeError("The exported GLB contains an invalid scene.")
        roots = scene.get("nodes")
        if not isinstance(roots, list):
            continue
        if root_index not in roots:
            continue
        scene["nodes"] = [index for index in roots if index != root_index]
        found = True
    if not found:
        raise ValueError(
            "An integrated instance source is not linked from an exported scene."
        )


def _get_gltf_collection(
    document: Mapping[str, object],
    name: str,
    *,
    create: bool = False,
) -> list[object]:
    """Return one validated mutable top-level glTF collection."""

    value = document.get(name)
    if value is None:
        if not create:
            return []
        if not isinstance(document, dict):
            raise TypeError("A writable glTF document is required.")
        created: list[object] = []
        document[name] = created
        return created
    if not isinstance(value, list):
        raise TypeError(f"The glTF {name} collection is invalid.")
    return value


def _require_gltf_index(value: object, collection_size: int) -> int:
    """Return one validated glTF collection index."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("The exported GLB contains an invalid glTF index.")
    if value >= collection_size:
        raise ValueError("The exported GLB contains an out-of-range glTF index.")
    return value


# ### GLB serialization helpers ###
def _parse_glb(payload: bytes, failure_message: str) -> tuple[dict[str, object], bytes]:
    try:
        if (
            len(payload) < GLB_HEADER_BYTE_COUNT + GLB_CHUNK_HEADER_BYTE_COUNT
            or payload[:4] != GLB_MAGIC
            or int.from_bytes(payload[4:8], "little") != GLB_VERSION
            or int.from_bytes(payload[8:12], "little") != len(payload)
        ):
            raise ValueError("The GLB header is invalid.")
        offset = GLB_HEADER_BYTE_COUNT
        chunks: list[tuple[bytes, bytes]] = []
        while offset < len(payload):
            if offset + GLB_CHUNK_HEADER_BYTE_COUNT > len(payload):
                raise ValueError("The GLB chunk header is invalid.")
            chunk_length = int.from_bytes(payload[offset : offset + 4], "little")
            chunk_type = payload[offset + 4 : offset + 8]
            chunk_start = offset + GLB_CHUNK_HEADER_BYTE_COUNT
            chunk_end = chunk_start + chunk_length
            if chunk_end > len(payload):
                raise ValueError("The GLB chunk is truncated.")
            chunks.append((chunk_type, payload[chunk_start:chunk_end]))
            offset = chunk_end
        if not chunks or chunks[0][0] != GLB_JSON_CHUNK_TYPE:
            raise ValueError("The first GLB chunk is not JSON.")
        if any(
            chunk_type not in {GLB_JSON_CHUNK_TYPE, GLB_BINARY_CHUNK_TYPE}
            for chunk_type, _chunk_payload in chunks
        ):
            raise ValueError("The GLB contains an unsupported chunk.")
        if len(chunks) > 2 or sum(
            chunk_type == GLB_BINARY_CHUNK_TYPE
            for chunk_type, _chunk_payload in chunks
        ) > 1:
            raise ValueError("The GLB chunk layout is invalid.")
        raw_json = chunks[0][1].rstrip(b" \t\r\n\0")
        document = json.loads(raw_json.decode("utf-8"))
        if not isinstance(document, dict):
            raise TypeError("The GLB JSON root is invalid.")
        binary_payload = next(
            (
                chunk_payload
                for chunk_type, chunk_payload in chunks[1:]
                if chunk_type == GLB_BINARY_CHUNK_TYPE
            ),
            b"",
        )
    except (TypeError, UnicodeError, ValueError) as error:
        raise ValueError(failure_message) from error
    return document, binary_payload


def _serialize_glb(document: Mapping[str, object], binary_payload: bytes) -> bytes:
    try:
        encoded_document = json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, UnicodeError, ValueError) as error:
        raise ValueError("The exported instance-source GLB is invalid.") from error
    encoded_document += b" " * (-len(encoded_document) % 4)
    chunks = [
        len(encoded_document).to_bytes(4, "little"),
        GLB_JSON_CHUNK_TYPE,
        encoded_document,
    ]
    if binary_payload:
        padded_binary = binary_payload + b"\0" * (-len(binary_payload) % 4)
        chunks.extend(
            (
                len(padded_binary).to_bytes(4, "little"),
                GLB_BINARY_CHUNK_TYPE,
                padded_binary,
            )
        )
    body = b"".join(chunks)
    return b"".join(
        (
            GLB_MAGIC,
            GLB_VERSION.to_bytes(4, "little"),
            (GLB_HEADER_BYTE_COUNT + len(body)).to_bytes(4, "little"),
            body,
        )
    )


# ### Serialization helpers ###
def _matrix_to_column_major(matrix: np.ndarray) -> list[float]:
    normalized = np.asarray(matrix, dtype=float)
    if normalized.shape != (4, 4) or not np.all(np.isfinite(normalized)):
        raise ValueError("Runtime instance transforms must be finite 4 by 4 matrices.")
    values = normalized.T.reshape(-1)
    return [
        0.0 if math.isclose(float(value), 0.0, abs_tol=1e-12) else float(value)
        for value in values
    ]


def _write_json_atomically(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _write_bytes_atomically(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

