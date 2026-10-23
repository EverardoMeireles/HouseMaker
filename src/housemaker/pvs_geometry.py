"""Partition glTF drawables without re-encoding their vertices or materials.

The grid and bounds use final physical glTF Y-up coordinates. Only index
accessors and static child nodes are added: UVs, normals, AO, PBR images and
runtime reconstruction extras retain their original binary/JSON definitions.
"""

# ### Imports ###
from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

# ### Constants ###
DEFAULT_PVS_CLUSTER_CELL_SIZE_METERS = 4.0
PVS_CLUSTER_EXTRAS_KEY = "housemakerPvsCluster"
PVS_GEOMETRY_ALGORITHM_VERSION = 1
_GLB_HEADER = struct.Struct("<4sII")
_GLB_CHUNK_HEADER = struct.Struct("<II")
_GLB_JSON_CHUNK = 0x4E4F534A
_GLB_BIN_CHUNK = 0x004E4942
_INDEX_COMPONENT_DTYPES = {5121: np.dtype("u1"), 5123: np.dtype("<u2"), 5125: np.dtype("<u4")}
_INSTANCE_KEYS = frozenset(("housemaker_instance_source_id", "housemaker_instance_source_name"))
_DYNAMIC_KEYS = frozenset((
    "halfMesh", "housemaker_door_body", "housemakerDoorBody",
    "housemaker_door_component", "housemakerDoorComponent",
))


# ### Public immutable cluster descriptions ###
@dataclass(frozen=True)
class PvsGeometryCluster:
    cluster_id: str
    node_names: tuple[str, ...]
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    source_node: str
    cell: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class _NodeFrame:
    index: int
    path: tuple[str, ...]
    ancestors: tuple[int, ...]
    world: np.ndarray


@dataclass(frozen=True)
class _Primitive:
    definition: dict
    vertices: np.ndarray
    faces: np.ndarray


# ### Public byte-level geometry partitioning ###
def partition_glb_for_pvs(
    glb_bytes: bytes,
    *,
    world_transform: object = None,
    cell_size: float = DEFAULT_PVS_CLUSTER_CELL_SIZE_METERS,
) -> tuple[bytes, tuple[PvsGeometryCluster, ...]]:
    """Split static triangles by centroid, retaining complete original triangles.

    ``world_transform`` affects physical bounds/grid membership, never the
    source transforms or vertex streams. Ordinary object subtrees keep their
    exact names and receive one whole-object descriptor. Instance prototypes,
    half meshes, doors, animated/skinned and unsupported geometry stay intact
    and always visible. Running again with the same grid is a byte-level no-op.
    """

    size = _normalize_cell_size(cell_size)
    physical_transform = _normalize_transform(world_transform)
    document, binary, chunks = _decode_glb(glb_bytes)
    frames = _node_frames(document, physical_transform)
    nodes = document.get("nodes", [])
    meshes = document.get("meshes", [])
    names = Counter(_node_name(node, index) for index, node in enumerate(nodes))
    object_members: dict[int, list[tuple[_NodeFrame, tuple[_Primitive, ...] | None]]] = defaultdict(list)
    excluded_object_roots: set[int] = set()
    static_nodes: list[tuple[_NodeFrame, tuple[_Primitive, ...]]] = []
    clusters: list[PvsGeometryCluster] = []
    animated_nodes = _animated_node_indices(document)
    for frame in frames:
        node = nodes[frame.index]
        mesh_index = node.get("mesh")
        if not _valid_index(mesh_index, meshes):
            continue
        mesh = meshes[mesh_index]
        if not isinstance(mesh, dict):
            continue
        object_root = _object_root(document, frame)
        excluded = _is_protected(document, frame, animated_nodes)
        primitives = _read_mesh_primitives(document, binary, mesh)
        if object_root is not None:
            object_members[object_root].append((frame, primitives))
            if excluded or primitives is None:
                excluded_object_roots.add(object_root)
            continue
        if excluded or primitives is None:
            continue
        existing = _cluster_extras(node)
        if existing is not None:
            if (
                existing.get("cellSize") != size
                or existing.get("worldTransform") != physical_transform.tolist()
            ):
                raise ValueError("Repartition the original GLB when the PVS grid or scale changes.")
            points = _primitive_world_points(primitives, frame.world)
            clusters.append(_cluster(
                str(existing["id"]), (_node_name(node, frame.index),), points,
                str(existing["sourceNode"]), tuple(existing["cell"]),
            ))
            continue
        static_nodes.append((frame, primitives))

    # Runtime ownership must cover the complete subtree, including empty
    # transform groups. An unlisted descendant deliberately protects its parent
    # from culling, so ambiguous names must leave the entire object visible.
    frame_by_index = {frame.index: frame for frame in frames}
    for root, members in sorted(object_members.items()):
        root_frame = frame_by_index[root]
        root_name = _node_name(nodes[root], root)
        subtree = tuple(
            frame for frame in frames
            if frame.index == root or root in frame.ancestors
        )
        if (
            root in excluded_object_roots
            or any(
                not isinstance(nodes[frame.index].get("name"), str)
                or not nodes[frame.index]["name"]
                or names[nodes[frame.index]["name"]] != 1
                or _is_protected(document, frame, animated_nodes)
                for frame in subtree
            )
        ):
            continue
        points = np.vstack([
            _primitive_world_points(primitives, frame.world)
            for frame, primitives in members if primitives is not None
        ])
        clusters.append(_cluster(
            _cluster_id(root_frame.path, None),
            tuple(nodes[frame.index]["name"] for frame in subtree),
            points, root_name,
        ))

    binary_output = bytearray(binary)
    modified = False
    occupied_names = set(names)
    for frame, primitives in static_nodes:
        partitions = _partition_primitives(primitives, frame.world, size)
        if partitions is None:
            continue
        node = nodes[frame.index]
        source_name = _node_name(node, frame.index)
        source_mesh = meshes[node["mesh"]]
        new_children: list[int] = []
        pending_clusters: list[PvsGeometryCluster] = []
        # Resolve names before changing any JSON so an ambiguous user-authored
        # name fails safely instead of silently replacing an existing node.
        for cell in sorted(partitions):
            cluster_id = _cluster_id(frame.path, cell)
            if cluster_id in occupied_names:
                raise ValueError(f"A PVS cluster node name is already in use: {cluster_id}.")
        for cell, pieces in sorted(partitions.items()):
            cluster_id = _cluster_id(frame.path, cell)
            new_primitives = []
            points = []
            for primitive, face_indices in pieces:
                selected_faces = primitive.faces[face_indices]
                accessor = _append_index_accessor(document, binary_output, selected_faces)
                definition = copy.deepcopy(primitive.definition)
                definition["indices"] = accessor
                new_primitives.append(definition)
                points.append(_transform_points(primitive.vertices[selected_faces.reshape(-1)], frame.world))
            mesh = copy.deepcopy(source_mesh)
            mesh["name"] = cluster_id
            mesh["primitives"] = new_primitives
            mesh_index = len(meshes)
            meshes.append(mesh)
            child_index = len(nodes)
            nodes.append({
                "name": cluster_id,
                "mesh": mesh_index,
                "extras": {PVS_CLUSTER_EXTRAS_KEY: {
                    "id": cluster_id, "sourceNode": source_name, "cell": list(cell),
                    "cellSize": size, "worldTransform": physical_transform.tolist(),
                    "algorithmVersion": PVS_GEOMETRY_ALGORITHM_VERSION,
                }},
            })
            new_children.append(child_index)
            occupied_names.add(cluster_id)
            pending_clusters.append(_cluster(
                cluster_id, (cluster_id,), np.vstack(points), source_name, cell,
            ))
        if new_children:
            node.pop("mesh")
            node["children"] = [*node.get("children", []), *new_children]
            clusters.extend(pending_clusters)
            modified = True
    if modified:
        document["buffers"][0]["byteLength"] = len(binary_output)
        glb_bytes = _encode_glb(document, bytes(binary_output), chunks)
    return glb_bytes, tuple(sorted(clusters, key=lambda item: item.cluster_id))


def collect_pvs_occluder_triangles(
    glb_bytes: bytes,
    *,
    world_transform: object = None,
) -> np.ndarray:
    """Read opaque structural triangles without decoding images/material maps.

    Furniture, source libraries, doors, half meshes, animations, transparency
    and unsupported streams never become occluders. Missing evidence therefore
    weakens culling instead of incorrectly hiding a potentially visible area.
    Preserve single-sided winding; explicitly double-sided materials contribute
    reversed copies so the one-sided bake can occlude from either direction.
    """

    document, binary, _chunks = _decode_glb(glb_bytes)
    frames = _node_frames(document, _normalize_transform(world_transform))
    nodes, meshes = document.get("nodes", []), document.get("meshes", [])
    animated_nodes = _animated_node_indices(document)
    triangles = []
    for frame in frames:
        node = nodes[frame.index]
        mesh_index = node.get("mesh")
        if (
            not _valid_index(mesh_index, meshes)
            or _object_root(document, frame) is not None
            or _is_object_atlas(document, frame)
            or _is_protected(document, frame, animated_nodes)
        ):
            continue
        primitives = _read_mesh_primitives(document, binary, meshes[mesh_index])
        if primitives is None:
            continue
        # Three.js flips its front-face convention for mirrored transforms.
        # Transforming positions alone reverses triangle winding instead, so
        # restore the rendered front side before creating one-sided patches.
        determinant = float(np.linalg.det(frame.world[:3, :3]))
        if not math.isfinite(determinant) or determinant == 0.0:
            continue
        for primitive in primitives:
            if not _primitive_is_opaque(document, primitive.definition):
                continue
            points = _transform_points(primitive.vertices, frame.world)
            primitive_triangles = points[primitive.faces]
            if determinant < 0.0:
                primitive_triangles = primitive_triangles[:, (0, 2, 1)]
            triangles.append(primitive_triangles)
            material_index = primitive.definition.get("material")
            materials = document.get("materials", [])
            if (
                _valid_index(material_index, materials)
                and materials[material_index].get("doubleSided") is True
            ):
                triangles.append(primitive_triangles[:, (0, 2, 1)])
    return np.concatenate(triangles) if triangles else np.empty((0, 3, 3), dtype=np.float64)


# ### GLB container helpers ###
def _decode_glb(payload: bytes) -> tuple[dict, bytes, tuple[tuple[int, bytes], ...]]:
    if not isinstance(payload, bytes) or len(payload) < _GLB_HEADER.size:
        raise ValueError("PVS geometry requires a GLB byte payload.")
    magic, version, length = _GLB_HEADER.unpack_from(payload)
    if magic != b"glTF" or version != 2 or length != len(payload):
        raise ValueError("PVS geometry requires a complete glTF 2 GLB.")
    chunks = []
    offset = _GLB_HEADER.size
    while offset < len(payload):
        if offset + _GLB_CHUNK_HEADER.size > len(payload):
            raise ValueError("A GLB chunk header is incomplete.")
        size, kind = _GLB_CHUNK_HEADER.unpack_from(payload, offset)
        offset += _GLB_CHUNK_HEADER.size
        if size % 4 or offset + size > len(payload):
            raise ValueError("A GLB chunk has invalid bounds or alignment.")
        chunks.append((kind, payload[offset:offset + size]))
        offset += size
    if not chunks or chunks[0][0] != _GLB_JSON_CHUNK:
        raise ValueError("The GLB JSON chunk is missing.")
    try:
        document = json.loads(chunks[0][1].decode("utf-8").rstrip(" \t\r\n\0"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("The GLB JSON chunk is invalid.") from error
    if not isinstance(document, dict):
        raise TypeError("The GLB JSON chunk must contain an object.")
    binary = next((data for kind, data in chunks if kind == _GLB_BIN_CHUNK), b"")
    return document, binary, tuple(chunks)


def _encode_glb(document: dict, binary: bytes, chunks: tuple[tuple[int, bytes], ...]) -> bytes:
    json_data = json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    json_data += b" " * (-len(json_data) % 4)
    binary += b"\0" * (-len(binary) % 4)
    output = []
    for kind, data in chunks:
        data = json_data if kind == _GLB_JSON_CHUNK else binary if kind == _GLB_BIN_CHUNK else data
        output.append(_GLB_CHUNK_HEADER.pack(len(data), kind) + data)
    body = b"".join(output)
    return _GLB_HEADER.pack(b"glTF", 2, _GLB_HEADER.size + len(body)) + body


# ### Original vertex/accessor decoding and appended indices ###
def _read_accessor(document: dict, binary: bytes, index: object, *, positions: bool) -> np.ndarray | None:
    accessors, views = document.get("accessors", []), document.get("bufferViews", [])
    buffers = document.get("buffers", [])
    if not _valid_index(index, accessors) or not buffers or buffers[0].get("uri") is not None:
        return None
    accessor = accessors[index]
    view_index = accessor.get("bufferView")
    if accessor.get("sparse") is not None or not _valid_index(view_index, views):
        return None
    view = views[view_index]
    if view.get("buffer", 0) != 0 or view.get("extensions"):
        return None
    if positions:
        if accessor.get("componentType") != 5126 or accessor.get("type") != "VEC3" or accessor.get("normalized", False):
            return None
        dtype, width = np.dtype("<f4"), 3
    else:
        dtype, width = _INDEX_COMPONENT_DTYPES.get(accessor.get("componentType")), 1
        if dtype is None or accessor.get("type") != "SCALAR":
            return None
    count = accessor.get("count")
    offset, length = view.get("byteOffset", 0), view.get("byteLength")
    local_offset = accessor.get("byteOffset", 0)
    stride = view.get("byteStride", dtype.itemsize * width)
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in (count, offset, length, local_offset, stride))
        or count <= 0 or min(offset, length, local_offset) < 0
        or stride < dtype.itemsize * width
        or local_offset + (count - 1) * stride + dtype.itemsize * width > length
        or offset + length > min(len(binary), buffers[0].get("byteLength", 0))
    ):
        return None
    values = np.ndarray(
        (count, width), dtype=dtype, buffer=binary,
        offset=offset + local_offset, strides=(stride, dtype.itemsize),
    )
    return values if positions else values[:, 0]


def _read_mesh_primitives(document: dict, binary: bytes, mesh: dict) -> tuple[_Primitive, ...] | None:
    primitives = mesh.get("primitives")
    if not isinstance(primitives, list) or not primitives or mesh.get("weights") is not None:
        return None
    result = []
    for primitive in primitives:
        if (
            not isinstance(primitive, dict) or primitive.get("mode", 4) != 4
            or primitive.get("targets") or primitive.get("extensions", {}).get("KHR_draco_mesh_compression") is not None
            or not _attributes_are_supported(document, primitive.get("attributes", {}))
        ):
            return None
        vertices = _read_accessor(document, binary, primitive.get("attributes", {}).get("POSITION"), positions=True)
        if vertices is None or not np.all(np.isfinite(vertices)):
            return None
        indices = (
            _read_accessor(document, binary, primitive["indices"], positions=False)
            if "indices" in primitive else np.arange(len(vertices), dtype=np.uint32)
        )
        if indices is None or len(indices) % 3 or np.max(indices) >= len(vertices):
            return None
        result.append(_Primitive(primitive, vertices, np.asarray(indices, dtype=np.int64).reshape(-1, 3)))
    return tuple(result)


def _attributes_are_supported(document: dict, attributes: object) -> bool:
    if not isinstance(attributes, Mapping):
        return False
    accessors, views = document.get("accessors", []), document.get("bufferViews", [])
    for index in attributes.values():
        if not _valid_index(index, accessors):
            return False
        accessor = accessors[index]
        view_index = accessor.get("bufferView")
        if (
            "sparse" in accessor or not _valid_index(view_index, views)
            or views[view_index].get("extensions")
        ):
            return False
    return True


def _append_index_accessor(document: dict, binary: bytearray, faces: np.ndarray) -> int:
    binary.extend(b"\0" * (-len(binary) % 4))
    indices = np.ascontiguousarray(faces.reshape(-1), dtype="<u4")
    offset = len(binary)
    binary.extend(indices.tobytes())
    views = document.setdefault("bufferViews", [])
    accessors = document.setdefault("accessors", [])
    views.append({"buffer": 0, "byteOffset": offset, "byteLength": indices.nbytes, "target": 34963})
    accessor_index = len(accessors)
    accessors.append({
        "bufferView": len(views) - 1, "componentType": 5125, "count": len(indices),
        "type": "SCALAR", "min": [int(indices.min())], "max": [int(indices.max())],
    })
    return accessor_index


# ### Canonical hierarchy transforms and protected runtime nodes ###
def _node_frames(document: dict, world: np.ndarray) -> tuple[_NodeFrame, ...]:
    nodes = document.get("nodes", [])
    scenes = document.get("scenes", [])
    scene_index = document.get("scene", 0)
    if scenes and _valid_index(scene_index, scenes):
        roots = scenes[scene_index].get("nodes", [])
    else:
        children = {child for node in nodes for child in node.get("children", [])}
        roots = [index for index in range(len(nodes)) if index not in children]
    frames = []
    visited = set()

    def visit_siblings(indices, parent, ancestors, path):
        sibling_names = Counter(_node_name(nodes[index], index) for index in indices if _valid_index(index, nodes))
        seen_names = Counter()
        for index in indices:
            if not _valid_index(index, nodes) or index in visited:
                raise ValueError("The GLB hierarchy contains an invalid or repeated node.")
            visited.add(index)
            node = nodes[index]
            name = _node_name(node, index)
            ordinal = seen_names[name]
            seen_names[name] += 1
            segment = name if sibling_names[name] == 1 else f"{name}#{ordinal}"
            node_path = (*path, segment)
            transform = parent @ _node_transform(node)
            frames.append(_NodeFrame(index, node_path, ancestors, transform))
            visit_siblings(node.get("children", []), transform, (*ancestors, index), node_path)

    visit_siblings(roots, world, (), ())
    return tuple(frames)


def _node_transform(node: dict) -> np.ndarray:
    if "matrix" in node:
        values = np.asarray(node["matrix"], dtype=float)
        if values.shape != (16,):
            raise ValueError("A glTF node matrix requires sixteen values.")
        return _normalize_transform(values.reshape(4, 4, order="F"))
    translation = np.asarray(node.get("translation", (0.0, 0.0, 0.0)), dtype=float)
    scale = np.asarray(node.get("scale", (1.0, 1.0, 1.0)), dtype=float)
    quaternion = np.asarray(node.get("rotation", (0.0, 0.0, 0.0, 1.0)), dtype=float)
    if (
        translation.shape != (3,) or scale.shape != (3,) or quaternion.shape != (4,)
        or not all(np.all(np.isfinite(value)) for value in (translation, scale, quaternion))
    ):
        raise ValueError("A glTF node has invalid TRS coordinates.")
    length = float(np.linalg.norm(quaternion))
    if length <= 0:
        raise ValueError("A glTF node has an invalid rotation quaternion.")
    x, y, z, w = quaternion / length
    rotation = np.asarray((
        (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    ))
    transform = np.eye(4)
    transform[:3, :3] = rotation @ np.diag(scale)
    transform[:3, 3] = translation
    return transform


def _metadata(document: dict, index: int, *, include_mesh: bool = True) -> tuple[Mapping, ...]:
    node, meshes = document["nodes"][index], document.get("meshes", [])
    sources = [node.get("extras")]
    mesh_index = node.get("mesh")
    if include_mesh and _valid_index(mesh_index, meshes):
        mesh = meshes[mesh_index]
        sources.append(mesh.get("extras"))
        sources.extend(primitive.get("extras") for primitive in mesh.get("primitives", []) if isinstance(primitive, dict))
    return tuple(value for value in sources if isinstance(value, Mapping))


def _object_root(document: dict, frame: _NodeFrame) -> int | None:
    for index in (*frame.ancestors, frame.index):
        if any(
            data.get("housemaker_object_id")
            for data in _metadata(document, index)
        ):
            return index
    return None


def _is_object_atlas(document: dict, frame: _NodeFrame) -> bool:
    return any(
        data.get("housemaker_atlas_type") == "object"
        for index in (*frame.ancestors, frame.index)
        for data in _metadata(document, index)
    )


def _is_protected(document: dict, frame: _NodeFrame, animated_nodes: frozenset[int]) -> bool:
    for index in (*frame.ancestors, frame.index):
        node = document["nodes"][index]
        if (
            index in animated_nodes or "skin" in node or "weights" in node
            or "EXT_mesh_gpu_instancing" in node.get("extensions", {})
            or _node_name(node, index).startswith("[HALF]")
        ):
            return True
        if any(
            any(key in data for key in _INSTANCE_KEYS | _DYNAMIC_KEYS)
            or data.get("housemaker_dynamic") is True
            for data in _metadata(document, index)
        ):
            return True
    return False


def _animated_node_indices(document: dict) -> frozenset[int]:
    return frozenset(
        target
        for animation in document.get("animations", [])
        for channel in animation.get("channels", [])
        if isinstance(target := channel.get("target", {}).get("node"), int)
    )


def _primitive_is_opaque(document: dict, primitive: dict) -> bool:
    index, materials = primitive.get("material"), document.get("materials", [])
    if index is None:
        return True
    if not _valid_index(index, materials):
        return False
    material = materials[index]
    if material.get("alphaMode", "OPAQUE") != "OPAQUE" or str(material.get("name", "")).casefold().startswith("housemaker glass"):
        return False
    transmission = material.get("extensions", {}).get("KHR_materials_transmission", {})
    return not bool(transmission.get("transmissionFactor", 0.0) or transmission.get("transmissionTexture"))


# ### Grid, bounds and validation helpers ###
def _partition_primitives(primitives, world, size):
    partitions = defaultdict(list)
    for primitive in primitives:
        points = _transform_points(primitive.vertices, world)
        cells = np.floor(points[primitive.faces].mean(axis=1) / size)
        if not np.all(np.isfinite(cells)) or np.any(np.abs(cells) >= np.iinfo(np.int64).max):
            return None
        cells = cells.astype(np.int64)
        unique, inverse = np.unique(cells, axis=0, return_inverse=True)
        for ordinal, cell in enumerate(unique):
            partitions[tuple(int(value) for value in cell)].append((primitive, np.flatnonzero(inverse == ordinal)))
    return partitions


def _primitive_world_points(primitives, world):
    return np.vstack([
        _transform_points(primitive.vertices[primitive.faces.reshape(-1)], world)
        for primitive in primitives
    ])


def _transform_points(points, world):
    return np.asarray(points, dtype=np.float64) @ world[:3, :3].T + world[:3, 3]


def _cluster(cluster_id, names, points, source_node, cell=None):
    if not len(points) or not np.all(np.isfinite(points)):
        raise ValueError("A PVS cluster has no finite geometry bounds.")
    return PvsGeometryCluster(
        cluster_id, names, tuple(float(value) for value in points.min(axis=0)),
        tuple(float(value) for value in points.max(axis=0)), source_node, cell,
    )


def _cluster_id(path, cell):
    payload = json.dumps((path, cell), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "pvs_cluster_" + hashlib.sha256(payload).hexdigest()[:24]


def _cluster_extras(node):
    extras = node.get("extras")
    value = extras.get(PVS_CLUSTER_EXTRAS_KEY) if isinstance(extras, Mapping) else None
    return value if isinstance(value, Mapping) and value.get("id") else None


def _node_name(node, index):
    name = node.get("name")
    return name if isinstance(name, str) and name else f"node_{index}"


def _valid_index(index, values):
    return isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(values)


def _normalize_transform(value):
    if value is None:
        return np.eye(4, dtype=np.float64)
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)) or not np.allclose(transform[3], (0, 0, 0, 1)):
        raise ValueError("PVS world transforms must be finite affine 4×4 matrices.")
    return transform.copy()


def _normalize_cell_size(value):
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("PVS cluster cell size must be a number.")
    try:
        size = float(value)
    except OverflowError as error:
        raise ValueError("PVS cluster cell size must be finite and positive.") from error
    if not math.isfinite(size) or size <= 0:
        raise ValueError("PVS cluster cell size must be finite and positive.")
    return size
