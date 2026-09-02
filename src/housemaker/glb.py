# ### Imports ###
from __future__ import annotations

import copy
import math
import os
from io import BytesIO
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import permutations
from pathlib import Path

import numpy as np
import shapely
import trimesh
from PIL import Image
from shapely import Point, Polygon
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
)
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.doorway_geometry import (
    build_doorway_cross_section_outline,
)
from housemaker.floor_geometry import build_level_floor_mesh
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
)
from housemaker.models import (
    DEFAULT_DOORWAY_ARCH_AMOUNT,
    DEFAULT_DOORWAY_BOTTOM_HEIGHT_METERS,
    DEFAULT_DOORWAY_SHAPE,
    DEFAULT_LEVEL_HEIGHT_METERS,
    GROUND_LEVEL_INDEX,
    STAIR_STYLE_FLOATING,
    STAIR_STYLE_FLOATING_WITH_RISER,
    STAIR_STYLE_SUPPORTED,
    Edge,
    LevelData,
    PIXEL_TO_METER,
    RoomData,
    StairData,
    StairSectionData,
    Vertex,
    VertexData,
)
from housemaker.object_texture_variants import ObjectTextureVariants
from housemaker.texture_mapping import paint_wall_texture_crop
from housemaker.uv_layout import (
    RoomWall,
    UvLayout,
    UvWallPlacement,
    build_room_walls,
    build_uv_wall_layout,
    get_rotated_uv_corners,
)

# ### Constants ###
DEFAULT_WALL_HEIGHT_METERS = DEFAULT_LEVEL_HEIGHT_METERS
Z_UP_TO_GLTF_Y_UP_TRANSFORM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)
GLTF_Y_UP_TO_Z_UP_TRANSFORM = np.linalg.inv(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
ROOM_TEXTURE_BACKGROUND_COLOR = QColor("#f7f8fb")
ROOM_TEXTURE_WALL_FILL_COLOR = QColor("#dce0e8")
ROOM_TEXTURE_TEXT_COLOR = QColor("#20242a")
ROOM_TEXTURE_INDICATOR_BACKGROUND_COLOR = QColor(10, 12, 16, 180)
ROOM_TEXTURE_INDICATOR_TEXT_COLOR = QColor("#f5f7fa")
ROOM_TEXTURE_MIN_FONT_SIZE = 8
ROOM_TEXTURE_MAX_FONT_SIZE = 32
SURFACE_FACE_MATCH_DECIMALS = 7
SURFACE_FACE_COVERAGE_EPSILON = 1e-9
FALLBACK_QT_PLATFORM = "offscreen"
WALL_OPENING_EPSILON = 1e-6
WINDOW_COPLANAR_DEPTH_PIXELS = 1e-4
WINDOW_CUT_DEPTH_METERS = 0.5
WALL_REVEAL_PARALLEL_COSINE = math.cos(math.radians(10.0))
MAX_IMPORTED_GENERATED_MODEL_FACES = 1_000_000
DEFAULT_STAIR_RISER_HEIGHT_METERS = 0.175
DEFAULT_FLOATING_STAIR_TREAD_THICKNESS_METERS = 0.08
STAIR_GEOMETRY_EPSILON = 1e-6
STAIR_CURVE_SAMPLE_SPACING_METERS = 0.08
MAX_EXACT_STAIR_GUIDE_ORDER_COUNT = 12
MAX_TOPOLOGY_STAIR_GUIDE_ORDER_COUNT = 8
TEXTURE_PREVIEW_PLANE_SIZE_METERS = 2.0
SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION = {
    "vertical": 0,
    "horizontal": 2,
}
NAMED_MESH_ROLE_SURFACE = "surface"
NAMED_MESH_ROLE_OPENING_REVEAL = "opening_reveal"
NAMED_MESH_ROLE_STAIR = "stair"

# ### Module state ###
_fallback_qt_application: QGuiApplication | None = None

# ### Data models ###
@dataclass
class GeneratedModel:
    mesh: trimesh.Trimesh
    scene: trimesh.Scene
    glb_bytes: bytes
    preview_textured_walls: list["PreviewTexturedWall"] = field(default_factory=list)
    preview_textured_surfaces: list["PreviewTexturedSurface"] = field(
        default_factory=list
    )
    preview_untextured_mesh: trimesh.Trimesh | None = None
    object_texture_variants: ObjectTextureVariants | None = None
    preview_symmetric_objects: list["PreviewSymmetricObject"] = field(
        default_factory=list
    )
    preview_placed_objects: list["PreviewPlacedObject"] = field(
        default_factory=list
    )
    preview_base_mesh: trimesh.Trimesh | None = None


@dataclass(frozen=True)
class PlacedGeneratedModel:
    """One generated model anchored at a world-space floor position."""

    object_id: str
    model: GeneratedModel
    world_position: tuple[float, float, float]
    symmetric_preview_orientation: str | None = None
    symmetric_preview_plane_coordinate: float | None = None
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str):
            raise TypeError("Placed generated-object IDs must be strings.")
        normalized_object_id = self.object_id.strip()
        if not normalized_object_id:
            raise ValueError("Placed generated-object IDs cannot be empty.")
        if not isinstance(self.model, GeneratedModel):
            raise TypeError("Placed generated objects require a GeneratedModel.")
        normalized_position = _normalize_placed_world_position(
            self.world_position
        )
        orientation, plane_coordinate = _normalize_symmetric_preview(
            self.symmetric_preview_orientation,
            self.symmetric_preview_plane_coordinate,
        )
        normalized_rotation = _normalize_placed_rotation(
            self.rotation_degrees
        )
        object.__setattr__(self, "object_id", normalized_object_id)
        object.__setattr__(self, "world_position", normalized_position)
        object.__setattr__(self, "rotation_degrees", normalized_rotation)
        object.__setattr__(self, "symmetric_preview_orientation", orientation)
        object.__setattr__(
            self,
            "symmetric_preview_plane_coordinate",
            plane_coordinate,
        )


@dataclass(frozen=True)
class PreviewPlacedObject:
    """Local preview meshes and the rigid transform for one Canvas object."""

    object_id: str
    meshes: tuple[trimesh.Trimesh, ...]
    placement_transform: np.ndarray
    world_position: tuple[float, float, float]
    rotation_degrees: tuple[float, float, float]
    symmetric_preview_orientation: str | None = None
    symmetric_preview_plane_coordinate: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id.strip():
            raise ValueError("Placed preview object IDs cannot be empty.")
        if not isinstance(self.meshes, tuple) or not self.meshes:
            raise ValueError("Placed previews require local object meshes.")
        if not all(isinstance(mesh, trimesh.Trimesh) for mesh in self.meshes):
            raise TypeError("Placed preview meshes must be triangle meshes.")
        transform = _get_valid_source_transform(self.placement_transform).copy()
        transform.setflags(write=False)
        orientation, plane_coordinate = _normalize_symmetric_preview(
            self.symmetric_preview_orientation,
            self.symmetric_preview_plane_coordinate,
        )
        object.__setattr__(self, "object_id", self.object_id.strip())
        object.__setattr__(self, "placement_transform", transform)
        object.__setattr__(
            self,
            "world_position",
            _normalize_placed_world_position(self.world_position),
        )
        object.__setattr__(
            self,
            "rotation_degrees",
            _normalize_placed_rotation(self.rotation_degrees),
        )
        object.__setattr__(self, "symmetric_preview_orientation", orientation)
        object.__setattr__(
            self,
            "symmetric_preview_plane_coordinate",
            plane_coordinate,
        )


@dataclass(frozen=True)
class PreviewSymmetricObject:
    """Placed retained meshes and their viewer-only mirrored counterparts."""

    object_id: str
    meshes: tuple[trimesh.Trimesh, ...]
    orientation: str
    plane_coordinate: float
    mirrored_meshes: tuple[trimesh.Trimesh, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id.strip():
            raise ValueError("Symmetric preview object IDs cannot be empty.")
        if not isinstance(self.meshes, tuple) or not self.meshes:
            raise ValueError("Symmetric previews require retained object meshes.")
        if not all(isinstance(mesh, trimesh.Trimesh) for mesh in self.meshes):
            raise TypeError("Symmetric preview meshes must be triangle meshes.")
        if not isinstance(self.mirrored_meshes, tuple) or not all(
            isinstance(mesh, trimesh.Trimesh)
            for mesh in self.mirrored_meshes
        ):
            raise TypeError(
                "Mirrored symmetric preview meshes must be triangle meshes."
            )
        if self.mirrored_meshes and len(self.mirrored_meshes) != len(
            self.meshes
        ):
            raise ValueError(
                "Symmetric retained and mirrored previews must have equal parts."
            )
        orientation, plane_coordinate = _normalize_symmetric_preview(
            self.orientation,
            self.plane_coordinate,
        )
        assert orientation is not None and plane_coordinate is not None
        object.__setattr__(self, "object_id", self.object_id.strip())
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "plane_coordinate", plane_coordinate)


@dataclass
class NamedMesh:
    name: str
    mesh: trimesh.Trimesh
    source_transform: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=float)
    )
    export_role: str = NAMED_MESH_ROLE_SURFACE


@dataclass(frozen=True)
class PreviewTexturedWall:
    level_index: int
    room_index: int
    wall_key: str
    start_point: tuple[float, float, float]
    end_point: tuple[float, float, float]
    height_meters: float
    texture_rgba: np.ndarray


@dataclass(frozen=True)
class PreviewTexturedSurface:
    """One semantic surface carrying its exported texture and planar UVs."""

    surface_id: str
    surface_type: str
    mesh: trimesh.Trimesh
    level_index: int | None = None
    room_index: int | None = None
    wall_key: str | None = None
    double_sided: bool = False


@dataclass(frozen=True)
class WallOpening:
    """A validated wall cut in image and wall-local coordinates."""

    center_x: float
    center_y: float
    width_direction_x: float
    width_direction_y: float
    depth_direction_x: float
    depth_direction_y: float
    half_width_pixels: float
    half_depth_pixels: float
    height_meters: float
    bottom_height_meters: float = 0.0
    target_wall_key: str | None = None
    target_surface_id: str | None = None
    parallel_walls_only: bool = False
    penetrates_parallel_walls: bool = False
    profile_points: tuple[tuple[float, float], ...] = ()

    @property
    def top_height_meters(self) -> float:
        return self.bottom_height_meters + self.height_meters


@dataclass(frozen=True)
class WallPiece:
    """One visible polygonal wall patch in normalized wall coordinates."""

    points: tuple[tuple[float, float], ...]

    @property
    def start_ratio(self) -> float:
        return min(point[0] for point in self.points)

    @property
    def end_ratio(self) -> float:
        return max(point[0] for point in self.points)

    @property
    def bottom_height_meters(self) -> float:
        return min(point[1] for point in self.points)

    @property
    def top_height_meters(self) -> float:
        return max(point[1] for point in self.points)


@dataclass(frozen=True)
class WallSource:
    """One rendered wall line that can bound an opening tunnel."""

    key: str
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    height_meters: float
    wall_key: str | None = None
    surface_id: str | None = None
    exterior_direction: tuple[float, float] | None = None


@dataclass(frozen=True)
class WallOpeningContact:
    """The part of a parallel wall that lies inside one doorway footprint."""

    source_key: str
    low_width_point: tuple[float, float]
    high_width_point: tuple[float, float]
    low_width_position: float
    high_width_position: float
    depth_position: float
    opening_bottom_height_meters: float
    opening_top_height_meters: float
    wall_key: str | None = None
    surface_id: str | None = None


@dataclass(frozen=True)
class DoorwayRevealPair:
    """Two parallel wall contacts that form a valid opening tunnel."""

    first_contact: WallOpeningContact
    second_contact: WallOpeningContact
    low_width_position: float
    high_width_position: float


@dataclass(frozen=True)
class DoorwayReveal:
    """One doorway opening paired with its stable reveal owner."""

    opening: WallOpening
    reveal_pair: DoorwayRevealPair
    owner_surface_id: str


@dataclass(frozen=True)
class WindowReveal:
    """A window opening paired with the nearest outward wall contact."""

    opening: WallOpening
    reveal_pair: DoorwayRevealPair
    owner_surface_id: str


@dataclass(frozen=True)
class PngTexture:
    png_bytes: bytes
    format: str = "PNG"

    def save(self, file_object, format: str | None = None) -> None:
        file_object.write(self.png_bytes)

    def copy(self) -> "PngTexture":
        return PngTexture(png_bytes=bytes(self.png_bytes), format=self.format)

    def __array__(self, dtype=None):
        texture_array = np.frombuffer(self.png_bytes, dtype=np.uint8)
        if dtype is None:
            return texture_array

        return texture_array.astype(dtype)


# ### Public helpers ###
def build_texture_preview_plane_model(
    texture_rgba: np.ndarray,
) -> GeneratedModel:
    """Build one viewer-only upright square carrying an RGBA texture."""

    rgba = np.asarray(texture_rgba)
    if (
        rgba.dtype != np.uint8
        or rgba.ndim != 3
        or rgba.shape[2] not in {3, 4}
        or rgba.shape[0] <= 0
        or rgba.shape[1] <= 0
    ):
        raise ValueError(
            "A texture preview requires a non-empty RGB or RGBA uint8 image."
        )
    if rgba.shape[2] == 3:
        alpha = np.full(rgba.shape[:2] + (1,), 255, dtype=np.uint8)
        rgba = np.concatenate((rgba, alpha), axis=2)
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)

    plane_size = TEXTURE_PREVIEW_PLANE_SIZE_METERS
    vertices = np.asarray(
        (
            (-plane_size / 2.0, 0.0, 0.0),
            (plane_size / 2.0, 0.0, 0.0),
            (plane_size / 2.0, 0.0, plane_size),
            (-plane_size / 2.0, 0.0, plane_size),
        ),
        dtype=float,
    )
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    uv_coordinates = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=TextureVisuals(
            uv=uv_coordinates,
            material=PBRMaterial(
                name="Atlas surface texture preview",
                baseColorFactor=[255, 255, 255, 255],
                baseColorTexture=Image.fromarray(rgba),
                metallicFactor=0.0,
                roughnessFactor=0.8,
                doubleSided=True,
            ),
        ),
        process=False,
    )
    scene = trimesh.Scene(mesh)
    return GeneratedModel(
        mesh=mesh,
        scene=scene,
        glb_bytes=b"",
    )


def convert_to_glb(
    level_source: VertexData | Sequence[LevelData],
    wall_height_meters: float = DEFAULT_WALL_HEIGHT_METERS,
    blueprint_size_pixels: tuple[float, float] | None = None,
    surface_materials: Mapping[str, object] | None = None,
    surface_texture_world_size_meters: float = 2.0,
    stairs: Sequence[StairData] = (),
    export_untextured_surfaces: bool = True,
) -> GeneratedModel:
    """Build an export-ready house model and serialize its GLB payload.

    ``export_untextured_surfaces=False`` removes only semantic walls, floors,
    and ceilings without a Surface texture from the serialized scene. Preview
    geometry and non-surface meshes such as stairs remain unchanged.
    """

    if not isinstance(export_untextured_surfaces, bool):
        raise TypeError("The untextured-surface export option must be boolean.")

    return _build_blueprint_model(
        level_source=level_source,
        wall_height_meters=wall_height_meters,
        blueprint_size_pixels=blueprint_size_pixels,
        surface_materials=(surface_materials or {}),
        surface_texture_world_size_meters=(
            surface_texture_world_size_meters
        ),
        stairs=stairs,
        serialize_glb=True,
        export_untextured_surfaces=export_untextured_surfaces,
    )


def convert_to_preview_model(
    level_source: VertexData | Sequence[LevelData],
    wall_height_meters: float = DEFAULT_WALL_HEIGHT_METERS,
    blueprint_size_pixels: tuple[float, float] | None = None,
    surface_materials: Mapping[str, object] | None = None,
    surface_texture_world_size_meters: float = 2.0,
    stairs: Sequence[StairData] = (),
) -> GeneratedModel:
    """Build the complete interactive house model without GLB serialization."""

    return _build_blueprint_model(
        level_source=level_source,
        wall_height_meters=wall_height_meters,
        blueprint_size_pixels=blueprint_size_pixels,
        surface_materials=surface_materials,
        surface_texture_world_size_meters=(
            surface_texture_world_size_meters
        ),
        stairs=stairs,
        serialize_glb=False,
        export_untextured_surfaces=True,
    )


# ### Blueprint model construction ###
def _build_blueprint_model(
    *,
    level_source: VertexData | Sequence[LevelData],
    wall_height_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
    surface_materials: Mapping[str, object] | None,
    surface_texture_world_size_meters: float,
    stairs: Sequence[StairData],
    serialize_glb: bool,
    export_untextured_surfaces: bool,
) -> GeneratedModel:
    if isinstance(level_source, VertexData):
        if stairs:
            raise ValueError("Stairs require level data with endpoint levels.")
        preview_textured_walls: list[PreviewTexturedWall] = []
        wall_meshes = _build_level_meshes(
            vertex_data=level_source,
            wall_height_meters=wall_height_meters,
            base_z_meters=0.0,
            blueprint_size_pixels=blueprint_size_pixels,
        )
        named_meshes = _build_named_meshes_for_single_level(wall_meshes)
    else:
        named_meshes = _build_multi_level_meshes(
            level_source,
            blueprint_size_pixels=blueprint_size_pixels,
            stairs=stairs,
        )
        preview_textured_walls = _build_preview_textured_walls(
            level_source,
            blueprint_size_pixels=blueprint_size_pixels,
        )

    if not named_meshes:
        raise ValueError("The current blueprint data does not contain usable edges.")

    combined_mesh = _combine_mesh_geometry(
        [
            _build_transformed_named_mesh_copy(named_mesh)
            for named_mesh in named_meshes
        ]
    )
    scene = _build_export_scene(named_meshes)
    glb_bytes = scene.export(file_type="glb") if serialize_glb else b""
    model = GeneratedModel(
        mesh=combined_mesh,
        scene=scene,
        glb_bytes=glb_bytes,
        preview_textured_walls=preview_textured_walls,
    )
    if not surface_materials and export_untextured_surfaces:
        return model
    return _apply_surface_materials(
        model=model,
        named_meshes=named_meshes,
        level_source=level_source,
        surface_materials=surface_materials,
        surface_texture_world_size_meters=(
            surface_texture_world_size_meters
        ),
        serialize_glb=serialize_glb,
        export_untextured_surfaces=export_untextured_surfaces,
    )


# ### Placed generated-model composition ###
def compose_placed_generated_models(
    base_model: GeneratedModel,
    placements: Sequence[PlacedGeneratedModel],
) -> GeneratedModel:
    """Add generated-object scenes to a house without changing their assets.

    Each target is the world-space destination of the generated model's
    Z-up bottom center. Source scene hierarchy, node transforms, materials,
    and embedded textures are copied into the returned export scene.
    """

    return _compose_placed_generated_models(
        base_model,
        placements,
        serialize_glb=True,
    )


def compose_placed_generated_models_preview(
    base_model: GeneratedModel,
    placements: Sequence[PlacedGeneratedModel],
) -> GeneratedModel:
    """Compose the complete interactive scene without GLB serialization."""

    return _compose_placed_generated_models(
        base_model,
        placements,
        serialize_glb=False,
    )


def _compose_placed_generated_models(
    base_model: GeneratedModel,
    placements: Sequence[PlacedGeneratedModel],
    *,
    serialize_glb: bool,
) -> GeneratedModel:
    """Compose placed objects while preserving export and preview parity."""

    if not isinstance(base_model, GeneratedModel):
        raise TypeError("Placed models require a GeneratedModel house base.")
    if isinstance(placements, (str, bytes, bytearray)) or not isinstance(
        placements,
        Sequence,
    ):
        raise TypeError("Placed models must contain a sequence of placements.")
    normalized_placements = tuple(placements)
    if not all(
        isinstance(placement, PlacedGeneratedModel)
        for placement in normalized_placements
    ):
        raise TypeError(
            "Placed models must contain PlacedGeneratedModel values."
        )
    if not normalized_placements:
        return base_model
    if not isinstance(base_model.scene, trimesh.Scene):
        raise TypeError("The house base must contain a trimesh scene.")

    output_scene = copy.deepcopy(base_model.scene)
    occupied_geometry_names = set(output_scene.geometry)
    occupied_node_names = set(output_scene.graph.nodes)
    placed_meshes: list[trimesh.Trimesh] = []
    placed_untextured_meshes: list[trimesh.Trimesh] = []
    preview_textured_surfaces = list(base_model.preview_textured_surfaces)
    preview_symmetric_objects = list(base_model.preview_symmetric_objects)
    preview_placed_objects = list(base_model.preview_placed_objects)
    preview_base_mesh = base_model.preview_base_mesh
    if preview_base_mesh is None:
        preview_base_mesh = (
            base_model.preview_untextured_mesh
            if base_model.preview_textured_surfaces
            and base_model.preview_untextured_mesh is not None
            else base_model.mesh
        )

    for placement_index, placement in enumerate(
        normalized_placements,
        start=1,
    ):
        placement_transform = _build_placed_model_transform(placement)
        _append_placed_model_scene(
            output_scene=output_scene,
            placement=placement,
            placement_index=placement_index,
            placement_transform=placement_transform,
            occupied_geometry_names=occupied_geometry_names,
            occupied_node_names=occupied_node_names,
        )

        placed_mesh = placement.model.mesh.copy()
        placed_mesh.apply_transform(placement_transform)
        placed_meshes.append(placed_mesh)
        textured_surfaces, untextured_meshes, world_meshes, local_meshes = (
            _build_placed_model_preview_parts(
                placement,
                placement_index,
                placement_transform,
            )
        )
        preview_textured_surfaces.extend(textured_surfaces)
        placed_untextured_meshes.extend(untextured_meshes)
        preview_placed_objects.append(
            PreviewPlacedObject(
                object_id=placement.object_id,
                meshes=local_meshes,
                placement_transform=placement_transform,
                world_position=placement.world_position,
                rotation_degrees=placement.rotation_degrees,
                symmetric_preview_orientation=(
                    placement.symmetric_preview_orientation
                ),
                symmetric_preview_plane_coordinate=(
                    placement.symmetric_preview_plane_coordinate
                ),
            )
        )
        if placement.symmetric_preview_orientation is not None:
            axis = SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION[
                placement.symmetric_preview_orientation
            ]
            assert placement.symmetric_preview_plane_coordinate is not None
            plane_point = np.zeros(4, dtype=float)
            plane_point[axis] = (
                placement.symmetric_preview_plane_coordinate
            )
            plane_point[3] = 1.0
            world_plane_point = placement_transform @ plane_point
            preview_symmetric_objects.append(
                PreviewSymmetricObject(
                    object_id=placement.object_id,
                    meshes=world_meshes,
                    orientation=placement.symmetric_preview_orientation,
                    plane_coordinate=float(world_plane_point[axis]),
                    mirrored_meshes=(
                        _build_symmetric_preview_world_meshes(
                            local_meshes,
                            placement.symmetric_preview_orientation,
                            placement.symmetric_preview_plane_coordinate,
                            placement_transform,
                        )
                    ),
                )
            )

    combined_mesh = _combine_mesh_geometry(
        [base_model.mesh, *placed_meshes]
    )
    preview_untextured_mesh = None
    if preview_textured_surfaces:
        base_preview_mesh = base_model.mesh
        if (
            base_model.preview_textured_surfaces
            and base_model.preview_untextured_mesh is not None
        ):
            base_preview_mesh = base_model.preview_untextured_mesh
        preview_untextured_mesh = _combine_mesh_geometry(
            [base_preview_mesh, *placed_untextured_meshes]
        )

    glb_bytes = b""
    if serialize_glb:
        exported_glb = output_scene.export(file_type="glb")
        if not isinstance(exported_glb, (bytes, bytearray, memoryview)):
            raise ValueError(
                "The placed generated-object scene could not be exported."
            )
        glb_bytes = bytes(exported_glb)
    return GeneratedModel(
        mesh=combined_mesh,
        scene=output_scene,
        glb_bytes=glb_bytes,
        preview_textured_walls=list(base_model.preview_textured_walls),
        preview_textured_surfaces=preview_textured_surfaces,
        preview_untextured_mesh=preview_untextured_mesh,
        preview_symmetric_objects=preview_symmetric_objects,
        preview_placed_objects=preview_placed_objects,
        preview_base_mesh=preview_base_mesh,
    )


def _normalize_placed_world_position(
    raw_position: object,
) -> tuple[float, float, float]:
    if isinstance(raw_position, (str, bytes, bytearray)) or not isinstance(
        raw_position,
        Sequence,
    ):
        raise TypeError("Placed generated-object positions must be XYZ sequences.")
    if len(raw_position) != 3:
        raise ValueError(
            "Placed generated-object positions must contain three coordinates."
        )

    coordinates: list[float] = []
    for raw_coordinate in raw_position:
        if isinstance(raw_coordinate, bool):
            raise TypeError(
                "Placed generated-object coordinates must be numbers."
            )
        try:
            coordinate = float(raw_coordinate)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError(
                "Placed generated-object coordinates must be numbers."
            ) from error
        if not math.isfinite(coordinate):
            raise ValueError(
                "Placed generated-object coordinates must be finite."
            )
        coordinates.append(coordinate)
    return coordinates[0], coordinates[1], coordinates[2]


def _normalize_symmetric_preview(
    raw_orientation: object,
    raw_plane_coordinate: object,
) -> tuple[str | None, float | None]:
    if raw_orientation is None:
        if raw_plane_coordinate is not None:
            raise ValueError(
                "A symmetric preview plane requires an orientation."
            )
        return None, None
    if raw_plane_coordinate is None:
        raise ValueError(
            "A symmetric preview orientation requires a plane coordinate."
        )
    if not isinstance(raw_orientation, str):
        raise TypeError("Symmetric preview orientations must be strings.")
    orientation = raw_orientation.strip().lower()
    if orientation not in SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION:
        raise ValueError(
            "Symmetric preview orientation must be vertical or horizontal."
        )
    if isinstance(raw_plane_coordinate, bool):
        raise TypeError("Symmetric preview planes must be numbers.")
    try:
        plane_coordinate = float(raw_plane_coordinate)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("Symmetric preview planes must be numbers.") from error
    if not math.isfinite(plane_coordinate):
        raise ValueError("Symmetric preview planes must be finite.")
    return orientation, plane_coordinate


def _normalize_placed_rotation(
    raw_rotation: object,
) -> tuple[float, float, float]:
    if isinstance(raw_rotation, (str, bytes, bytearray)) or not isinstance(
        raw_rotation,
        Sequence,
    ):
        raise TypeError("Placed generated-object rotations must be XYZ sequences.")
    if len(raw_rotation) != 3:
        raise ValueError(
            "Placed generated-object rotations must contain three angles."
        )
    angles: list[float] = []
    for raw_angle in raw_rotation:
        if isinstance(raw_angle, bool) or not isinstance(
            raw_angle,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("Placed generated-object angles must be numbers.")
        angle = float(raw_angle)
        if not math.isfinite(angle):
            raise ValueError("Placed generated-object angles must be finite.")
        angles.append(angle)
    return angles[0], angles[1], angles[2]


def _build_placed_model_transform(
    placement: PlacedGeneratedModel,
) -> np.ndarray:
    mesh = placement.model.mesh
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Placed generated objects must contain a triangle mesh.")
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not len(vertices)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(faces)
    ):
        raise ValueError("Placed generated objects must contain non-empty meshes.")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Placed generated-object vertices must be finite.")
    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    bottom_center = np.array(
        [
            (minimum[0] + maximum[0]) / 2.0,
            (minimum[1] + maximum[1]) / 2.0,
            minimum[2],
        ],
        dtype=float,
    )
    rotation_radians = np.radians(
        np.asarray(placement.rotation_degrees, dtype=float)
    )
    rotation = trimesh.transformations.euler_matrix(
        *rotation_radians,
        axes="sxyz",
    )
    move_pivot_to_origin = np.eye(4, dtype=float)
    move_pivot_to_origin[:3, 3] = -bottom_center
    move_to_world = np.eye(4, dtype=float)
    move_to_world[:3, 3] = np.asarray(
        placement.world_position,
        dtype=float,
    )
    return move_to_world @ rotation @ move_pivot_to_origin


def _append_placed_model_scene(
    *,
    output_scene: trimesh.Scene,
    placement: PlacedGeneratedModel,
    placement_index: int,
    placement_transform: np.ndarray,
    occupied_geometry_names: set[object],
    occupied_node_names: set[object],
) -> None:
    source_scene = placement.model.scene
    if not isinstance(source_scene, trimesh.Scene):
        raise TypeError("Placed generated objects must contain trimesh scenes.")
    if not source_scene.geometry or not source_scene.graph.nodes_geometry:
        raise ValueError("Placed generated objects must contain scene geometry.")

    prefix = (
        f"placed_{placement_index}_{_slugify_name(placement.object_id)}"
    )
    root_name = _reserve_unique_scene_name(
        f"{prefix}_root",
        occupied_node_names,
    )
    geometry_names = {
        source_name: _reserve_unique_scene_name(
            f"{prefix}_geometry_{_slugify_scene_name(source_name)}",
            occupied_geometry_names,
        )
        for source_name in source_scene.geometry
    }
    for source_name, geometry in source_scene.geometry.items():
        copied_geometry = copy.deepcopy(geometry)
        copied_geometry.metadata = copy.deepcopy(
            dict(getattr(copied_geometry, "metadata", {}) or {})
        )
        copied_geometry.metadata["housemaker_object_id"] = placement.object_id
        output_scene.geometry[geometry_names[source_name]] = copied_geometry

    source_base_frame = source_scene.graph.base_frame
    node_names: dict[object, object] = {source_base_frame: root_name}
    for source_node_name in source_scene.graph.nodes:
        if source_node_name == source_base_frame:
            continue
        node_names[source_node_name] = _reserve_unique_scene_name(
            f"{prefix}_node_{_slugify_scene_name(source_node_name)}",
            occupied_node_names,
        )

    source_base_data = source_scene.graph.transforms.node_data.get(
        source_base_frame,
        {},
    )
    root_metadata = copy.deepcopy(source_base_data.get("metadata") or {})
    root_metadata["housemaker_object_id"] = placement.object_id
    root_kwargs: dict[str, object] = {
        "matrix": _source_to_gltf_y_up_transform(placement_transform),
        "metadata": root_metadata,
    }
    source_base_geometry = source_base_data.get("geometry")
    if source_base_geometry in geometry_names:
        root_kwargs["geometry"] = geometry_names[source_base_geometry]
    output_scene.graph.update(
        frame_to=root_name,
        frame_from=output_scene.graph.base_frame,
        **root_kwargs,
    )

    for source_from, source_to, raw_attributes in (
        source_scene.graph.to_edgelist()
    ):
        attributes = dict(raw_attributes)
        edge_kwargs: dict[str, object] = {
            "matrix": _get_valid_source_transform(attributes.get("matrix")),
        }
        source_geometry_name = attributes.get("geometry")
        if source_geometry_name is not None:
            if source_geometry_name not in geometry_names:
                raise ValueError(
                    "A placed generated-object node references missing geometry."
                )
            edge_kwargs["geometry"] = geometry_names[source_geometry_name]
        if attributes.get("metadata") is not None:
            edge_kwargs["metadata"] = copy.deepcopy(
                attributes["metadata"]
            )
        output_scene.graph.update(
            frame_to=node_names[source_to],
            frame_from=node_names[source_from],
            **edge_kwargs,
        )


def _build_placed_model_preview_parts(
    placement: PlacedGeneratedModel,
    placement_index: int,
    placement_transform: np.ndarray,
) -> tuple[
    list[PreviewTexturedSurface],
    list[trimesh.Trimesh],
    tuple[trimesh.Trimesh, ...],
    tuple[trimesh.Trimesh, ...],
]:
    source_scene = placement.model.scene
    textured_surfaces: list[PreviewTexturedSurface] = []
    untextured_meshes: list[trimesh.Trimesh] = []
    world_meshes: list[trimesh.Trimesh] = []
    local_meshes: list[trimesh.Trimesh] = []
    for node_index, source_node_name in enumerate(
        sorted(source_scene.graph.nodes_geometry, key=str),
        start=1,
    ):
        node_transform, source_geometry_name = source_scene.graph.get(
            source_node_name
        )
        source_geometry = source_scene.geometry.get(source_geometry_name)
        if not isinstance(source_geometry, trimesh.Trimesh):
            raise ValueError(
                "Placed generated-object scenes must contain triangle meshes."
            )
        local_mesh = copy.deepcopy(source_geometry)
        local_mesh.apply_transform(
            GLTF_Y_UP_TO_Z_UP_TRANSFORM
            @ _get_valid_source_transform(node_transform)
        )
        local_meshes.append(local_mesh)
        world_mesh = local_mesh.copy()
        world_mesh.apply_transform(placement_transform)
        world_meshes.append(world_mesh)
        if _mesh_supports_embedded_texture_preview(world_mesh):
            textured_surfaces.append(
                PreviewTexturedSurface(
                    surface_id=(
                        f"placed:{placement.object_id}:"
                        f"{placement_index}:{node_index}"
                    ),
                    surface_type="generated_object",
                    mesh=world_mesh,
                )
            )
        else:
            untextured_meshes.append(world_mesh)
    return (
        textured_surfaces,
        untextured_meshes,
        tuple(world_meshes),
        tuple(local_meshes),
    )


def _build_symmetric_preview_world_meshes(
    local_meshes: tuple[trimesh.Trimesh, ...],
    orientation: str,
    plane_coordinate: float,
    placement_transform: np.ndarray,
) -> tuple[trimesh.Trimesh, ...]:
    """Mirror local retained parts, then move the preview with its object."""

    axis = SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION[orientation]
    mirrored_meshes: list[trimesh.Trimesh] = []
    for local_mesh in local_meshes:
        mirrored_mesh = copy.deepcopy(local_mesh)
        vertices = np.asarray(mirrored_mesh.vertices, dtype=float).copy()
        vertices[:, axis] = float(plane_coordinate) * 2.0 - vertices[:, axis]
        mirrored_mesh.vertices = vertices
        mirrored_mesh.faces = np.asarray(
            mirrored_mesh.faces,
            dtype=np.int64,
        )[:, (0, 2, 1)]
        mirrored_mesh.apply_transform(placement_transform)
        mirrored_meshes.append(mirrored_mesh)
    return tuple(mirrored_meshes)


def _mesh_supports_embedded_texture_preview(mesh: trimesh.Trimesh) -> bool:
    visual = getattr(mesh, "visual", None)
    if getattr(visual, "kind", None) != "texture":
        return False
    material = getattr(visual, "material", None)
    texture = getattr(material, "baseColorTexture", None)
    if texture is None:
        texture = getattr(material, "image", None)
    if texture is None:
        return False
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    texture_coordinates = np.asarray(getattr(visual, "uv", ()))
    return bool(
        vertices.ndim == 2
        and vertices.shape[1:] == (3,)
        and faces.ndim == 2
        and faces.shape[1:] == (3,)
        and len(faces)
        and texture_coordinates.shape == (len(vertices), 2)
        and np.all(np.isfinite(texture_coordinates))
    )


def _reserve_unique_scene_name(
    preferred_name: str,
    occupied_names: set[object],
) -> str:
    candidate = preferred_name
    suffix = 2
    while candidate in occupied_names:
        candidate = f"{preferred_name}_{suffix}"
        suffix += 1
    occupied_names.add(candidate)
    return candidate


def _slugify_scene_name(raw_name: object) -> str:
    return _slugify_name(str(raw_name))


# ### Stair geometry helpers ###
def build_stair_meshes(
    levels: Sequence[LevelData],
    stairs: Sequence[StairData],
) -> list[NamedMesh]:
    """Build one world-space mesh per stairway.

    Stair sections intentionally remain in each owning level's local image
    coordinate system. Resolving them here through ``level_image_to_world_xy``
    means level scale and offsets automatically carry every route section.
    """

    if not stairs:
        return []

    level_lookup = {level.index: level for level in levels}
    base_z_by_level_index = build_level_base_z_lookup(levels)
    named_meshes: list[NamedMesh] = []
    for stair_index, stair in enumerate(stairs, start=1):
        mesh = _build_stair_mesh(
            stair=stair,
            level_lookup=level_lookup,
            base_z_by_level_index=base_z_by_level_index,
        )
        named_meshes.append(
            NamedMesh(
                name=_get_stair_object_name(stair_index, stair),
                mesh=mesh,
                export_role=NAMED_MESH_ROLE_STAIR,
            )
        )

    return named_meshes


def _build_stair_mesh(
    stair: StairData,
    level_lookup: Mapping[int, LevelData],
    base_z_by_level_index: Mapping[int, float],
) -> trimesh.Trimesh:
    route_sections = _resolve_stair_route_sections(stair, level_lookup)
    start_z_meters = _get_stair_endpoint_base_z(
        base_z_by_level_index,
        stair.start_level_index,
        "start",
    )
    end_z_meters = _get_stair_endpoint_base_z(
        base_z_by_level_index,
        stair.end_level_index,
        "end",
    )
    height_difference_meters = end_z_meters - start_z_meters
    if abs(height_difference_meters) <= STAIR_GEOMETRY_EPSILON:
        raise ValueError("Stair endpoints must be at different elevations.")

    if start_z_meters < end_z_meters:
        lower_elevation_meters = start_z_meters
        upper_elevation_meters = end_z_meters
    else:
        route_sections.reverse()
        lower_elevation_meters = end_z_meters
        upper_elevation_meters = start_z_meters

    # Curve guides are ordered cross-sections, not separate stair flights.
    # Sample one continuous spline through every section before deriving the
    # treads.  Using the full route here is important: treating the control
    # pairs as isolated linear prisms makes later guides look like detached
    # stair systems whenever the route changes direction more than once.
    route_sections = _build_smoothed_stair_route_sections(route_sections)
    cumulative_distances = _build_stair_route_distances(route_sections)
    total_run_meters = cumulative_distances[-1]
    total_rise_meters = upper_elevation_meters - lower_elevation_meters
    step_count = max(
        1,
        math.ceil(
            total_rise_meters / DEFAULT_STAIR_RISER_HEIGHT_METERS
        ),
    )
    riser_height_meters = total_rise_meters / step_count
    tread_thickness_meters = min(
        DEFAULT_FLOATING_STAIR_TREAD_THICKNESS_METERS,
        riser_height_meters,
    )

    meshes: list[trimesh.Trimesh] = []
    for step_index in range(step_count):
        step_start_distance = total_run_meters * step_index / step_count
        step_end_distance = total_run_meters * (step_index + 1) / step_count
        step_top_z_meters = lower_elevation_meters + (
            riser_height_meters * (step_index + 1)
        )
        if stair.style in {
            STAIR_STYLE_FLOATING,
            STAIR_STYLE_FLOATING_WITH_RISER,
        }:
            step_bottom_z_meters = step_top_z_meters - tread_thickness_meters
        elif stair.style == STAIR_STYLE_SUPPORTED:
            step_bottom_z_meters = min(
                lower_elevation_meters,
                step_top_z_meters - tread_thickness_meters,
            )
        else:
            raise ValueError(f"Unsupported stair style: {stair.style!r}.")

        step_distances = _split_stair_step_at_route_sections(
            step_start_distance,
            step_end_distance,
            cumulative_distances,
        )
        step_segment_count = len(step_distances) - 1
        for segment_index, (segment_start, segment_end) in enumerate(
            zip(
                step_distances,
                step_distances[1:],
            )
        ):
            step_start_a_xy, step_start_b_xy = _sample_stair_route(
                route_sections,
                cumulative_distances,
                segment_start,
            )
            step_end_a_xy, step_end_b_xy = _sample_stair_route(
                route_sections,
                cumulative_distances,
                segment_end,
            )
            meshes.append(
                _build_stair_step_prism(
                    start_a_xy=step_start_a_xy,
                    start_b_xy=step_start_b_xy,
                    end_a_xy=step_end_a_xy,
                    end_b_xy=step_end_b_xy,
                    bottom_z_meters=step_bottom_z_meters,
                    top_z_meters=step_top_z_meters,
                    include_start_cap=segment_index == 0,
                    include_end_cap=(
                        segment_index == step_segment_count - 1
                    ),
                )
            )

        if (
            stair.style == STAIR_STYLE_FLOATING_WITH_RISER
            and step_index > 0
        ):
            previous_step_top_z_meters = lower_elevation_meters + (
                riser_height_meters * step_index
            )
            if (
                step_bottom_z_meters
                > previous_step_top_z_meters + STAIR_GEOMETRY_EPSILON
            ):
                meshes.append(
                    _build_stair_riser_prism(
                        route_sections=route_sections,
                        cumulative_distances=cumulative_distances,
                        step_start_distance=step_start_distance,
                        first_step_segment_end_distance=step_distances[1],
                        bottom_z_meters=previous_step_top_z_meters,
                        top_z_meters=step_bottom_z_meters,
                        maximum_depth_meters=tread_thickness_meters,
                    )
                )

    return _combine_mesh_geometry(meshes)


def _resolve_stair_route_sections(
    stair: StairData,
    level_lookup: Mapping[int, LevelData],
) -> list[tuple[np.ndarray, np.ndarray]]:
    resolved_sections: list[tuple[np.ndarray, np.ndarray]] = []
    final_section_index = len(stair.sections) - 1
    for section_index, section in enumerate(stair.sections):
        section_name = _get_stair_section_name(
            section_index,
            final_section_index,
        )
        level = _get_stair_endpoint_level(
            level_lookup,
            section.level_index,
            section_name,
        )
        section_a_xy, section_b_xy = _resolve_stair_section_points(
            section,
            level,
        )
        _validate_world_stair_segment(
            section_a_xy,
            section_b_xy,
            section_name,
        )
        resolved_sections.append((section_a_xy, section_b_xy))

    ordered_sections = _order_stair_route_sections(resolved_sections)
    return _normalize_stair_rail_correspondence(ordered_sections)


def _order_stair_route_sections(
    route_sections: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Order guides spatially while keeping both stair endpoints fixed."""

    if len(route_sections) < 2:
        raise ValueError("A stair route requires two endpoint sections.")
    guide_sections = list(route_sections[1:-1])
    if len(guide_sections) <= 1:
        return list(route_sections)

    guide_order: tuple[int, ...] | None = None
    if len(guide_sections) <= MAX_TOPOLOGY_STAIR_GUIDE_ORDER_COUNT:
        guide_order = _find_shortest_topology_safe_stair_guide_order(
            route_sections[0],
            guide_sections,
            route_sections[-1],
        )
    if (
        guide_order is None
        and len(guide_sections) <= MAX_EXACT_STAIR_GUIDE_ORDER_COUNT
    ):
        guide_order = _find_shortest_stair_guide_order(
            route_sections[0],
            guide_sections,
            route_sections[-1],
        )
    elif guide_order is None:
        guide_order = _find_greedy_stair_guide_order(
            route_sections[0],
            guide_sections,
        )
    return [
        route_sections[0],
        *(guide_sections[index] for index in guide_order),
        route_sections[-1],
    ]


def _find_shortest_topology_safe_stair_guide_order(
    start_section: tuple[np.ndarray, np.ndarray],
    guide_sections: Sequence[tuple[np.ndarray, np.ndarray]],
    end_section: tuple[np.ndarray, np.ndarray],
) -> tuple[int, ...] | None:
    """Find the shortest guide permutation whose piecewise rails do not cross."""

    best_path: tuple[float, tuple[int, ...]] | None = None
    for guide_order in permutations(range(len(guide_sections))):
        route_sections = [
            start_section,
            *(guide_sections[index] for index in guide_order),
            end_section,
        ]
        normalized_sections = _normalize_stair_rail_correspondence(
            route_sections
        )
        if not _is_stair_route_topology_safe(normalized_sections):
            continue
        path_distance = _get_stair_route_center_distance(route_sections)
        if _is_better_stair_guide_path(
            path_distance,
            guide_order,
            best_path,
        ):
            best_path = (path_distance, guide_order)
    return None if best_path is None else best_path[1]


def _get_stair_route_center_distance(
    route_sections: Sequence[tuple[np.ndarray, np.ndarray]],
) -> float:
    centers = [
        _get_stair_section_center(section) for section in route_sections
    ]
    return sum(
        float(np.linalg.norm(end_center - start_center))
        for start_center, end_center in zip(centers, centers[1:])
    )


def _find_shortest_stair_guide_order(
    start_section: tuple[np.ndarray, np.ndarray],
    guide_sections: Sequence[tuple[np.ndarray, np.ndarray]],
    end_section: tuple[np.ndarray, np.ndarray],
) -> tuple[int, ...]:
    """Solve the fixed-endpoint guide path exactly with dynamic programming."""

    start_center = _get_stair_section_center(start_section)
    guide_centers = [
        _get_stair_section_center(section) for section in guide_sections
    ]
    end_center = _get_stair_section_center(end_section)
    guide_count = len(guide_sections)
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
    for guide_index, guide_center in enumerate(guide_centers):
        states[(1 << guide_index, guide_index)] = (
            float(np.linalg.norm(guide_center - start_center)),
            (guide_index,),
        )

    all_guides_mask = (1 << guide_count) - 1
    for visited_mask in range(1, all_guides_mask + 1):
        for final_guide_index in range(guide_count):
            state = states.get((visited_mask, final_guide_index))
            if state is None:
                continue
            current_distance, current_order = state
            for next_guide_index in range(guide_count):
                next_guide_bit = 1 << next_guide_index
                if visited_mask & next_guide_bit:
                    continue
                next_mask = visited_mask | next_guide_bit
                candidate_distance = current_distance + float(
                    np.linalg.norm(
                        guide_centers[next_guide_index]
                        - guide_centers[final_guide_index]
                    )
                )
                candidate_order = (*current_order, next_guide_index)
                state_key = (next_mask, next_guide_index)
                existing_state = states.get(state_key)
                if _is_better_stair_guide_path(
                    candidate_distance,
                    candidate_order,
                    existing_state,
                ):
                    states[state_key] = (
                        candidate_distance,
                        candidate_order,
                    )

    best_path: tuple[float, tuple[int, ...]] | None = None
    for final_guide_index in range(guide_count):
        path_distance, path_order = states[
            (all_guides_mask, final_guide_index)
        ]
        total_distance = path_distance + float(
            np.linalg.norm(end_center - guide_centers[final_guide_index])
        )
        if _is_better_stair_guide_path(
            total_distance,
            path_order,
            best_path,
        ):
            best_path = (total_distance, path_order)

    if best_path is None:
        raise ValueError("Unable to order the stair curve guides.")
    return best_path[1]


def _is_better_stair_guide_path(
    candidate_distance: float,
    candidate_order: tuple[int, ...],
    current_path: tuple[float, tuple[int, ...]] | None,
) -> bool:
    if current_path is None:
        return True
    current_distance, current_order = current_path
    if candidate_distance < current_distance - STAIR_GEOMETRY_EPSILON:
        return True
    return (
        abs(candidate_distance - current_distance)
        <= STAIR_GEOMETRY_EPSILON
        and candidate_order < current_order
    )


def _find_greedy_stair_guide_order(
    start_section: tuple[np.ndarray, np.ndarray],
    guide_sections: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[int, ...]:
    """Order unusually large guide sets with a deterministic nearest walk."""

    guide_centers = [
        _get_stair_section_center(section) for section in guide_sections
    ]
    current_center = _get_stair_section_center(start_section)
    remaining_indices = set(range(len(guide_sections)))
    guide_order: list[int] = []
    while remaining_indices:
        next_guide_index = min(
            remaining_indices,
            key=lambda guide_index: (
                float(
                    np.linalg.norm(
                        guide_centers[guide_index] - current_center
                    )
                ),
                guide_index,
            ),
        )
        guide_order.append(next_guide_index)
        remaining_indices.remove(next_guide_index)
        current_center = guide_centers[next_guide_index]
    return tuple(guide_order)


def _get_stair_section_center(
    section: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    return (section[0] + section[1]) / 2.0


def _resolve_stair_section_points(
    section: StairSectionData,
    level: LevelData,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        _resolve_stair_control_point(
            level,
            section.a_x,
            section.a_y,
            section.a_vertex_id,
        ),
        _resolve_stair_control_point(
            level,
            section.b_x,
            section.b_y,
            section.b_vertex_id,
        ),
    )


def _build_stair_route_distances(
    route_sections: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    cumulative_distances = [0.0]
    for previous_section, current_section in zip(
        route_sections,
        route_sections[1:],
    ):
        previous_center = (previous_section[0] + previous_section[1]) / 2.0
        current_center = (current_section[0] + current_section[1]) / 2.0
        segment_length = float(np.linalg.norm(current_center - previous_center))
        if segment_length <= STAIR_GEOMETRY_EPSILON:
            raise ValueError(
                "Consecutive stair section centers must be separated "
                "horizontally."
            )
        cumulative_distances.append(
            cumulative_distances[-1] + segment_length
        )
    return cumulative_distances


def _build_smoothed_stair_route_sections(
    control_sections: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sample a continuous route through every ordered stair cross-section.

    A straight stair deliberately stays as one segment.  For a curved stair,
    each rail is interpolated with a distance-parameterized cubic Hermite
    spline.  The original guide cross-sections are included verbatim at the
    interval boundaries, so every placed guide remains an exact part of the
    final route while the pieces between guides follow one continuous curve.
    """

    if len(control_sections) < 2:
        raise ValueError("A stair route requires two endpoint sections.")
    if len(control_sections) == 2:
        return [
            (section_a_xy.copy(), section_b_xy.copy())
            for section_a_xy, section_b_xy in control_sections
        ]

    control_distances = _build_stair_route_distances(control_sections)
    sampled_sections: list[tuple[np.ndarray, np.ndarray]] = []
    for segment_index, (segment_start, segment_end) in enumerate(
        zip(control_sections, control_sections[1:])
    ):
        segment_length = (
            control_distances[segment_index + 1]
            - control_distances[segment_index]
        )
        sample_count = max(
            1,
            math.ceil(segment_length / STAIR_CURVE_SAMPLE_SPACING_METERS),
        )
        for sample_index in range(sample_count):
            ratio = sample_index / sample_count
            if sample_index == 0:
                # Preserve every user-supplied guide exactly.  This also
                # avoids accumulating tiny floating-point offsets at joins.
                sampled_sections.append(
                    (segment_start[0].copy(), segment_start[1].copy())
                )
                continue
            sampled_sections.append(
                _sample_smoothed_stair_route_section(
                    control_sections,
                    control_distances,
                    segment_index,
                    ratio,
                )
            )

    final_a_xy, final_b_xy = control_sections[-1]
    sampled_sections.append((final_a_xy.copy(), final_b_xy.copy()))
    if not _is_stair_route_topology_safe(sampled_sections):
        # Cubic centerline tangents can overshoot around two very close,
        # sharply turning guides.  The guides already approximate the curve,
        # so first fall back to their faithful piecewise-linear route.
        linear_sections = [
            (section_a_xy.copy(), section_b_xy.copy())
            for section_a_xy, section_b_xy in control_sections
        ]
        if _is_stair_route_topology_safe(linear_sections):
            return linear_sections
        raise ValueError(
            "Stair curve guides produce crossing rails. Move the guides "
            "farther apart, reduce the stair width, or remove the sharpest "
            "turn."
        )
    return sampled_sections


def _is_stair_route_topology_safe(
    route_sections: Sequence[tuple[np.ndarray, np.ndarray]],
) -> bool:
    """Return whether a sampled route keeps two ordered, uncrossed rails."""

    rail_a = [section_a_xy for section_a_xy, _ in route_sections]
    rail_b = [section_b_xy for _, section_b_xy in route_sections]
    if _does_stair_rail_self_intersect(rail_a):
        return False
    if _does_stair_rail_self_intersect(rail_b):
        return False
    if _do_stair_rails_intersect(rail_a, rail_b):
        return False
    return _do_stair_sections_keep_handedness(route_sections)


def _does_stair_rail_self_intersect(
    rail_points: Sequence[np.ndarray],
) -> bool:
    for first_index in range(len(rail_points) - 1):
        for second_index in range(first_index + 2, len(rail_points) - 1):
            if _do_stair_line_segments_intersect(
                rail_points[first_index],
                rail_points[first_index + 1],
                rail_points[second_index],
                rail_points[second_index + 1],
            ):
                return True
    return False


def _do_stair_rails_intersect(
    rail_a: Sequence[np.ndarray],
    rail_b: Sequence[np.ndarray],
) -> bool:
    for rail_a_start, rail_a_end in zip(rail_a, rail_a[1:]):
        for rail_b_start, rail_b_end in zip(rail_b, rail_b[1:]):
            if _do_stair_line_segments_intersect(
                rail_a_start,
                rail_a_end,
                rail_b_start,
                rail_b_end,
            ):
                return True
    return False


def _do_stair_sections_keep_handedness(
    route_sections: Sequence[tuple[np.ndarray, np.ndarray]],
) -> bool:
    centers = [
        (section_a_xy + section_b_xy) / 2.0
        for section_a_xy, section_b_xy in route_sections
    ]
    segment_directions: list[np.ndarray] = []
    for start_center_xy, end_center_xy in zip(centers, centers[1:]):
        route_delta = end_center_xy - start_center_xy
        route_delta_length = float(np.linalg.norm(route_delta))
        if route_delta_length <= STAIR_GEOMETRY_EPSILON:
            return False
        segment_directions.append(route_delta / route_delta_length)
    route_directions = _build_stair_section_route_directions(
        segment_directions
    )

    reference_handedness = 0.0
    for (section_a_xy, section_b_xy), route_direction in zip(
        route_sections,
        route_directions,
    ):
        handedness = _get_stair_2d_cross_product(
            route_direction,
            section_b_xy - section_a_xy,
        )
        if abs(handedness) <= STAIR_GEOMETRY_EPSILON:
            return False
        if reference_handedness == 0.0:
            reference_handedness = handedness
        elif handedness * reference_handedness < 0.0:
            return False
    return True


def _do_stair_line_segments_intersect(
    first_start_xy: np.ndarray,
    first_end_xy: np.ndarray,
    second_start_xy: np.ndarray,
    second_end_xy: np.ndarray,
) -> bool:
    if not _do_stair_segment_bounds_overlap(
        first_start_xy,
        first_end_xy,
        second_start_xy,
        second_end_xy,
    ):
        return False

    first_start_side = _get_stair_2d_cross_product(
        first_end_xy - first_start_xy,
        second_start_xy - first_start_xy,
    )
    first_end_side = _get_stair_2d_cross_product(
        first_end_xy - first_start_xy,
        second_end_xy - first_start_xy,
    )
    second_start_side = _get_stair_2d_cross_product(
        second_end_xy - second_start_xy,
        first_start_xy - second_start_xy,
    )
    second_end_side = _get_stair_2d_cross_product(
        second_end_xy - second_start_xy,
        first_end_xy - second_start_xy,
    )
    if (
        first_start_side * first_end_side
        < -(STAIR_GEOMETRY_EPSILON**2)
        and second_start_side * second_end_side
        < -(STAIR_GEOMETRY_EPSILON**2)
    ):
        return True
    return (
        abs(first_start_side) <= STAIR_GEOMETRY_EPSILON
        and _is_point_on_stair_segment(
            second_start_xy,
            first_start_xy,
            first_end_xy,
        )
    ) or (
        abs(first_end_side) <= STAIR_GEOMETRY_EPSILON
        and _is_point_on_stair_segment(
            second_end_xy,
            first_start_xy,
            first_end_xy,
        )
    ) or (
        abs(second_start_side) <= STAIR_GEOMETRY_EPSILON
        and _is_point_on_stair_segment(
            first_start_xy,
            second_start_xy,
            second_end_xy,
        )
    ) or (
        abs(second_end_side) <= STAIR_GEOMETRY_EPSILON
        and _is_point_on_stair_segment(
            first_end_xy,
            second_start_xy,
            second_end_xy,
        )
    )


def _do_stair_segment_bounds_overlap(
    first_start_xy: np.ndarray,
    first_end_xy: np.ndarray,
    second_start_xy: np.ndarray,
    second_end_xy: np.ndarray,
) -> bool:
    return not (
        max(first_start_xy[0], first_end_xy[0])
        < min(second_start_xy[0], second_end_xy[0])
        - STAIR_GEOMETRY_EPSILON
        or max(second_start_xy[0], second_end_xy[0])
        < min(first_start_xy[0], first_end_xy[0])
        - STAIR_GEOMETRY_EPSILON
        or max(first_start_xy[1], first_end_xy[1])
        < min(second_start_xy[1], second_end_xy[1])
        - STAIR_GEOMETRY_EPSILON
        or max(second_start_xy[1], second_end_xy[1])
        < min(first_start_xy[1], first_end_xy[1])
        - STAIR_GEOMETRY_EPSILON
    )


def _is_point_on_stair_segment(
    point_xy: np.ndarray,
    segment_start_xy: np.ndarray,
    segment_end_xy: np.ndarray,
) -> bool:
    return bool(
        np.all(
            point_xy
            >= np.minimum(segment_start_xy, segment_end_xy)
            - STAIR_GEOMETRY_EPSILON
        )
        and np.all(
            point_xy
            <= np.maximum(segment_start_xy, segment_end_xy)
            + STAIR_GEOMETRY_EPSILON
        )
    )


def _sample_smoothed_stair_route_section(
    control_sections: Sequence[tuple[np.ndarray, np.ndarray]],
    control_distances: Sequence[float],
    segment_index: int,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one spline-interpolated cross-section inside a route interval."""

    interval_length = (
        control_distances[segment_index + 1]
        - control_distances[segment_index]
    )
    start_a_xy, start_b_xy = control_sections[segment_index]
    end_a_xy, end_b_xy = control_sections[segment_index + 1]
    start_center_xy = (start_a_xy + start_b_xy) / 2.0
    end_center_xy = (end_a_xy + end_b_xy) / 2.0
    center_xy = _interpolate_stair_cubic_hermite(
        start_center_xy,
        end_center_xy,
        _get_stair_route_center_tangent(
            control_sections,
            control_distances,
            segment_index,
        ),
        _get_stair_route_center_tangent(
            control_sections,
            control_distances,
            segment_index + 1,
        ),
        interval_length,
        ratio,
    )
    width_xy = _interpolate_stair_route_width(
        start_b_xy - start_a_xy,
        end_b_xy - end_a_xy,
        ratio,
    )
    return center_xy - (width_xy / 2.0), center_xy + (width_xy / 2.0)


def _get_stair_route_center_tangent(
    control_sections: Sequence[tuple[np.ndarray, np.ndarray]],
    control_distances: Sequence[float],
    section_index: int,
) -> np.ndarray:
    """Return a distance-aware tangent for the stair route centerline."""

    control_centers = [
        (section_a_xy + section_b_xy) / 2.0
        for section_a_xy, section_b_xy in control_sections
    ]
    final_section_index = len(control_sections) - 1
    if section_index == 0:
        return (
            control_centers[1] - control_centers[0]
        ) / (control_distances[1] - control_distances[0])
    if section_index == final_section_index:
        return (
            control_centers[-1] - control_centers[-2]
        ) / (control_distances[-1] - control_distances[-2])
    return (
        control_centers[section_index + 1]
        - control_centers[section_index - 1]
    ) / (
        control_distances[section_index + 1]
        - control_distances[section_index - 1]
    )


def _interpolate_stair_route_width(
    start_width_xy: np.ndarray,
    end_width_xy: np.ndarray,
    ratio: float,
) -> np.ndarray:
    """Interpolate a stair width without allowing its two rails to cross."""

    start_width = float(np.linalg.norm(start_width_xy))
    end_width = float(np.linalg.norm(end_width_xy))
    if (
        start_width <= STAIR_GEOMETRY_EPSILON
        or end_width <= STAIR_GEOMETRY_EPSILON
    ):
        raise ValueError("Stair route sections must have a positive width.")

    start_angle = math.atan2(start_width_xy[1], start_width_xy[0])
    end_angle = math.atan2(end_width_xy[1], end_width_xy[0])
    angle_delta = math.atan2(
        math.sin(end_angle - start_angle),
        math.cos(end_angle - start_angle),
    )
    interpolated_angle = start_angle + (angle_delta * float(ratio))
    interpolated_width = start_width + (
        (end_width - start_width) * float(ratio)
    )
    return float(interpolated_width) * np.asarray(
        (math.cos(interpolated_angle), math.sin(interpolated_angle)),
        dtype=float,
    )


def _interpolate_stair_cubic_hermite(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    start_tangent: np.ndarray,
    end_tangent: np.ndarray,
    interval_length: float,
    ratio: float,
) -> np.ndarray:
    """Interpolate a route rail while preserving its two endpoint sections."""

    ratio_squared = ratio * ratio
    ratio_cubed = ratio_squared * ratio
    return (
        ((2.0 * ratio_cubed) - (3.0 * ratio_squared) + 1.0) * start_xy
        + ((ratio_cubed) - (2.0 * ratio_squared) + ratio)
        * interval_length
        * start_tangent
        + ((-2.0 * ratio_cubed) + (3.0 * ratio_squared)) * end_xy
        + ((ratio_cubed) - ratio_squared) * interval_length * end_tangent
    )


def _split_stair_step_at_route_sections(
    step_start_distance: float,
    step_end_distance: float,
    cumulative_distances: Sequence[float],
) -> list[float]:
    internal_sections = [
        distance
        for distance in cumulative_distances[1:-1]
        if (
            distance > step_start_distance + STAIR_GEOMETRY_EPSILON
            and distance < step_end_distance - STAIR_GEOMETRY_EPSILON
        )
    ]
    return [step_start_distance, *internal_sections, step_end_distance]


def _sample_stair_route(
    route_sections: Sequence[tuple[np.ndarray, np.ndarray]],
    cumulative_distances: Sequence[float],
    distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    segment_index = int(
        np.searchsorted(cumulative_distances, distance, side="right") - 1
    )
    segment_index = min(max(segment_index, 0), len(route_sections) - 2)
    segment_start_distance = cumulative_distances[segment_index]
    segment_end_distance = cumulative_distances[segment_index + 1]
    segment_ratio = (
        (distance - segment_start_distance)
        / (segment_end_distance - segment_start_distance)
    )
    start_a_xy, start_b_xy = route_sections[segment_index]
    end_a_xy, end_b_xy = route_sections[segment_index + 1]
    return (
        _interpolate_stair_point(start_a_xy, end_a_xy, segment_ratio),
        _interpolate_stair_point(start_b_xy, end_b_xy, segment_ratio),
    )


def _get_stair_section_name(
    section_index: int,
    final_section_index: int,
) -> str:
    if section_index == 0:
        return "start"
    if section_index == final_section_index:
        return "end"
    return f"intermediate section {section_index}"


def _build_stair_step_prism(
    start_a_xy: np.ndarray,
    start_b_xy: np.ndarray,
    end_a_xy: np.ndarray,
    end_b_xy: np.ndarray,
    bottom_z_meters: float,
    top_z_meters: float,
    *,
    include_start_cap: bool = True,
    include_end_cap: bool = True,
) -> trimesh.Trimesh:
    step_height_meters = top_z_meters - bottom_z_meters
    if step_height_meters <= STAIR_GEOMETRY_EPSILON:
        raise ValueError("Stair step height must be greater than zero.")

    footprint = np.asarray(
        [start_a_xy, end_a_xy, end_b_xy, start_b_xy],
        dtype=float,
    )
    signed_area = _get_polygon_signed_area(footprint)
    if abs(signed_area) <= STAIR_GEOMETRY_EPSILON:
        raise ValueError("Stair control points produce a zero-area step.")
    if signed_area < 0.0:
        footprint = footprint[::-1]

    vertices = np.vstack(
        (
            np.column_stack(
                (footprint, np.full(4, bottom_z_meters, dtype=float))
            ),
            np.column_stack(
                (footprint, np.full(4, top_z_meters, dtype=float))
            ),
        )
    )
    faces: list[list[int]] = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
    ]
    # A guide can split one horizontal tread into several route pieces.  The
    # pieces share their guide cross-section, so retaining both closed-prism
    # caps would render a false vertical wall and make that guide look like an
    # isolated stair section.  Keep caps only at the true outer tread edges.
    if include_end_cap:
        faces.extend(
            (
                [1, 2, 6],
                [1, 6, 5],
            )
        )
    if include_start_cap:
        faces.extend(
            (
                [3, 0, 4],
                [3, 4, 7],
            )
        )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _build_stair_riser_prism(
    route_sections: Sequence[tuple[np.ndarray, np.ndarray]],
    cumulative_distances: Sequence[float],
    step_start_distance: float,
    first_step_segment_end_distance: float,
    bottom_z_meters: float,
    top_z_meters: float,
    maximum_depth_meters: float,
) -> trimesh.Trimesh:
    """Build the thin vertical panel closing one floating tread's gap.

    A curved step can be divided around one or more route sections. The riser
    is intentionally built once at the shared edge with the preceding tread,
    so bends do not create duplicate risers. Its shallow footprint only closes
    the space from that tread's top to this tread's underside.
    """

    first_segment_length = (
        first_step_segment_end_distance - step_start_distance
    )
    riser_depth_meters = min(
        maximum_depth_meters,
        first_segment_length / 2.0,
    )
    if riser_depth_meters <= STAIR_GEOMETRY_EPSILON:
        raise ValueError("Stair riser depth must be greater than zero.")

    riser_end_distance = step_start_distance + riser_depth_meters
    riser_start_a_xy, riser_start_b_xy = _sample_stair_route(
        route_sections,
        cumulative_distances,
        step_start_distance,
    )
    riser_end_a_xy, riser_end_b_xy = _sample_stair_route(
        route_sections,
        cumulative_distances,
        riser_end_distance,
    )
    return _build_stair_step_prism(
        start_a_xy=riser_start_a_xy,
        start_b_xy=riser_start_b_xy,
        end_a_xy=riser_end_a_xy,
        end_b_xy=riser_end_b_xy,
        bottom_z_meters=bottom_z_meters,
        top_z_meters=top_z_meters,
    )


def _resolve_stair_control_point(
    level: LevelData,
    fallback_x: float,
    fallback_y: float,
    vertex_id: int | None,
) -> np.ndarray:
    source_x = fallback_x
    source_y = fallback_y
    if vertex_id is not None:
        bound_vertex = level.vertex_data.get_vertex(vertex_id)
        if bound_vertex is not None:
            source_x = bound_vertex.x
            source_y = bound_vertex.y

    point = np.asarray(
        level_image_to_world_xy(level, source_x, source_y),
        dtype=float,
    )
    if not np.all(np.isfinite(point)):
        raise ValueError("Stair control points must have finite world coordinates.")
    return point


def _normalize_stair_rail_correspondence(
    route_sections: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Give every cross-section one route-wide left/right orientation.

    Point-pair click order is intentionally irrelevant.  Each ``B - A``
    vector is oriented to the same side of the ordered centerline.  A local
    shortest-distance choice is insufficient because it exchanges the two
    rails after sharp turns, which makes later curve guides appear jumbled.
    """

    if len(route_sections) < 2:
        raise ValueError("A stair route requires two endpoint sections.")

    centers = [
        (section_a_xy + section_b_xy) / 2.0
        for section_a_xy, section_b_xy in route_sections
    ]
    segment_directions: list[np.ndarray] = []
    for start_center_xy, end_center_xy in zip(centers, centers[1:]):
        center_delta = end_center_xy - start_center_xy
        center_distance = float(np.linalg.norm(center_delta))
        if center_distance <= STAIR_GEOMETRY_EPSILON:
            raise ValueError(
                "Consecutive stair section centers must be separated "
                "horizontally."
            )
        segment_directions.append(center_delta / center_distance)

    route_directions = _build_stair_section_route_directions(
        segment_directions
    )
    normalized_sections: list[tuple[np.ndarray, np.ndarray]] = []
    for (section_a_xy, section_b_xy), route_direction in zip(
        route_sections,
        route_directions,
    ):
        width_direction = section_b_xy - section_a_xy
        handedness = _get_stair_2d_cross_product(
            route_direction,
            width_direction,
        )
        if handedness < 0.0:
            normalized_sections.append((section_b_xy, section_a_xy))
        else:
            normalized_sections.append((section_a_xy, section_b_xy))

    return normalized_sections


def _build_stair_section_route_directions(
    segment_directions: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Build stable forward directions at all ordered stair sections."""

    route_directions = [segment_directions[0]]
    for incoming_direction, outgoing_direction in zip(
        segment_directions,
        segment_directions[1:],
    ):
        route_direction = incoming_direction + outgoing_direction
        route_direction_length = float(np.linalg.norm(route_direction))
        if route_direction_length <= STAIR_GEOMETRY_EPSILON:
            route_directions.append(outgoing_direction)
        else:
            route_directions.append(
                route_direction / route_direction_length
            )
    route_directions.append(segment_directions[-1])
    return route_directions


def _get_stair_2d_cross_product(
    first_xy: np.ndarray,
    second_xy: np.ndarray,
) -> float:
    return float(
        (first_xy[0] * second_xy[1])
        - (first_xy[1] * second_xy[0])
    )


def _validate_world_stair_segment(
    first_xy: np.ndarray,
    second_xy: np.ndarray,
    segment_name: str,
) -> None:
    if float(np.linalg.norm(second_xy - first_xy)) <= STAIR_GEOMETRY_EPSILON:
        raise ValueError(
            f"Stair {segment_name} points must be separated in world space."
        )


def _interpolate_stair_point(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    ratio: float,
) -> np.ndarray:
    return start_xy + ((end_xy - start_xy) * float(ratio))


def _get_polygon_signed_area(points: np.ndarray) -> float:
    shifted_points = np.roll(points, -1, axis=0)
    return float(
        0.5
        * np.sum(
            (points[:, 0] * shifted_points[:, 1])
            - (points[:, 1] * shifted_points[:, 0])
        )
    )


def _get_stair_endpoint_level(
    level_lookup: Mapping[int, LevelData],
    level_index: int,
    endpoint_name: str,
) -> LevelData:
    level = level_lookup.get(level_index)
    if level is None:
        raise ValueError(
            f"Stair {endpoint_name} level {level_index} does not exist."
        )
    return level


def _get_stair_endpoint_base_z(
    base_z_by_level_index: Mapping[int, float],
    level_index: int,
    endpoint_name: str,
) -> float:
    base_z_meters = base_z_by_level_index.get(level_index)
    if base_z_meters is None:
        raise ValueError(
            f"Stair {endpoint_name} level {level_index} has no elevation."
        )
    if not math.isfinite(base_z_meters):
        raise ValueError(
            f"Stair {endpoint_name} level {level_index} has an invalid "
            "elevation."
        )
    return float(base_z_meters)


def _get_stair_object_name(stair_index: int, stair: StairData) -> str:
    return f"stair_{stair_index}_{stair.style}"


def _apply_surface_materials(
    model: GeneratedModel,
    named_meshes: Sequence[NamedMesh],
    level_source: VertexData | Sequence[LevelData],
    surface_materials: Mapping[str, object],
    surface_texture_world_size_meters: float,
    serialize_glb: bool,
    export_untextured_surfaces: bool,
) -> GeneratedModel:
    """Replace assigned faces and optionally omit unassigned export surfaces."""

    if not isinstance(surface_materials, Mapping):
        raise TypeError("Surface materials must be provided as a mapping.")
    if isinstance(level_source, VertexData):
        raise ValueError(
            "Surface materials require level data with stable surface IDs."
        )

    from housemaker.surface_geometry import build_fixed_surfaces
    from housemaker.surface_materials import (
        build_world_planar_textured_mesh,
        normalize_texture_world_size,
        resolve_surface_materials,
    )

    levels = list(level_source)
    fixed_surfaces = build_fixed_surfaces(levels)
    base_surfaces = list(fixed_surfaces)
    known_surface_ids = {surface.surface_id for surface in fixed_surfaces}
    live_sources = {
        str(surface_id): source
        for surface_id, source in surface_materials.items()
        if str(surface_id) in known_surface_ids
    }
    if not live_sources and export_untextured_surfaces:
        return model
    texture_world_size = normalize_texture_world_size(
        surface_texture_world_size_meters
    )
    resolved_materials = resolve_surface_materials(live_sources)
    (
        textured_named_meshes,
        preview_textured_surfaces,
    ) = _build_surface_named_meshes(
        fixed_surfaces=fixed_surfaces,
        resolved_materials=resolved_materials,
        texture_world_size_meters=texture_world_size,
        build_textured_mesh=build_world_planar_textured_mesh,
    )
    if not textured_named_meshes and export_untextured_surfaces:
        return model
    if not textured_named_meshes:
        export_named_meshes = _build_assigned_surface_export_named_meshes(
            named_meshes,
            base_surfaces,
            textured_named_meshes,
        )
        export_scene = _build_export_scene(export_named_meshes)
        return replace(
            model,
            scene=export_scene,
            glb_bytes=(
                _serialize_export_scene(export_scene) if serialize_glb else b""
            ),
        )
    replacement_surface_ids = set(resolved_materials).intersection(
        surface.surface_id for surface in base_surfaces
    )
    replacement_surfaces = [
        surface
        for surface in base_surfaces
        if surface.surface_id in replacement_surface_ids
    ]
    level_by_index = {level.index: level for level in levels}
    partitioned_floor_levels = {
        surface.level_index
        for surface in replacement_surfaces
        if surface.surface_type == "floor"
        and level_by_index.get(surface.level_index) is not None
        and level_by_index[surface.level_index].floor_contour_vertex_ids
    }
    untextured_partition_surfaces = [
        surface
        for surface in base_surfaces
        if surface.surface_type == "floor"
        and surface.level_index in partitioned_floor_levels
        and surface.surface_id not in replacement_surface_ids
    ]
    removal_surfaces = [
        *replacement_surfaces,
        *untextured_partition_surfaces,
    ]
    replacement_face_keys = _build_oriented_surface_face_keys(
        removal_surfaces
    )
    replacement_plane_coverage = _build_surface_plane_coverage(
        removal_surfaces
    )
    retained_named_meshes = _remove_named_mesh_surface_faces(
        named_meshes,
        replacement_face_keys,
        replacement_plane_coverage,
    )
    retained_named_meshes.extend(
        NamedMesh(
            name=f"{_get_surface_object_name(surface.surface_id)}_untextured",
            mesh=surface.mesh.copy(),
        )
        for surface in untextured_partition_surfaces
    )
    preview_base_mesh = _combine_mesh_geometry(
        [
            _build_transformed_named_mesh_copy(named_mesh)
            for named_mesh in retained_named_meshes
        ]
    )
    combined_mesh = _combine_mesh_geometry(
        [
            preview_base_mesh,
            *[named_mesh.mesh for named_mesh in textured_named_meshes],
        ]
    )
    export_named_meshes = [
        *retained_named_meshes,
        *textured_named_meshes,
    ]
    if not export_untextured_surfaces:
        export_named_meshes = _build_assigned_surface_export_named_meshes(
            named_meshes,
            base_surfaces,
            textured_named_meshes,
        )
    scene = _build_export_scene(export_named_meshes)
    return GeneratedModel(
        mesh=combined_mesh,
        scene=scene,
        glb_bytes=(
            scene.export(file_type="glb") if serialize_glb else b""
        ),
        preview_textured_walls=model.preview_textured_walls,
        preview_textured_surfaces=preview_textured_surfaces,
        preview_untextured_mesh=preview_base_mesh,
    )


# ### Surface material helpers ###
def _build_surface_named_meshes(
    fixed_surfaces: Sequence[object],
    resolved_materials: Mapping[str, object],
    texture_world_size_meters: float,
    build_textured_mesh: Callable[..., trimesh.Trimesh],
) -> tuple[list[NamedMesh], list[PreviewTexturedSurface]]:
    named_meshes: list[NamedMesh] = []
    preview_surfaces: list[PreviewTexturedSurface] = []
    for surface in fixed_surfaces:
        surface_id = str(getattr(surface, "surface_id"))
        surface_type = str(getattr(surface, "surface_type"))
        material = resolved_materials.get(surface_id)
        if material is None:
            continue

        double_sided = (
            surface_type == "wall"
            and getattr(surface, "room_index", None) is None
        )

        object_name = _get_surface_object_name(surface_id)
        mesh = build_textured_mesh(
            getattr(surface, "mesh").copy(),
            surface_type,
            material,
            texture_world_size_meters=texture_world_size_meters,
            material_name=f"Surface {surface_id}",
            double_sided=double_sided,
        )
        mesh.metadata = copy.deepcopy(
            dict(getattr(mesh, "metadata", {}) or {})
        )
        mesh.metadata["housemaker_surface_id"] = surface_id
        preview_surfaces.append(
            PreviewTexturedSurface(
                surface_id=surface_id,
                surface_type=surface_type,
                mesh=mesh.copy(),
                level_index=getattr(surface, "level_index", None),
                room_index=getattr(surface, "room_index", None),
                wall_key=getattr(surface, "wall_key", None),
                double_sided=double_sided,
            )
        )
        named_meshes.append(
            NamedMesh(
                name=object_name,
                mesh=mesh,
            )
        )
    return named_meshes, preview_surfaces


def _build_assigned_surface_export_named_meshes(
    named_meshes: Sequence[NamedMesh],
    fixed_surfaces: Sequence[object],
    textured_named_meshes: Sequence[NamedMesh],
) -> list[NamedMesh]:
    """Keep assigned surfaces plus non-surface geometry in the export scene."""

    retained = _remove_named_mesh_surface_faces(
        named_meshes,
        _build_oriented_surface_face_keys(fixed_surfaces),
        _build_surface_plane_coverage(fixed_surfaces),
    )
    non_surface_meshes = [
        named_mesh
        for named_mesh in retained
        if named_mesh.export_role != NAMED_MESH_ROLE_SURFACE
    ]
    return [*non_surface_meshes, *textured_named_meshes]


def _build_oriented_surface_face_keys(
    surfaces: Iterable[object],
) -> set[tuple[tuple[float, float, float], ...]]:
    """Return winding-sensitive keys for semantic faces that replace base faces."""

    return {
        _build_oriented_triangle_key(triangle)
        for surface in surfaces
        for triangle, _normal in _iter_surface_replacement_triangles(surface)
    }


def _remove_named_mesh_surface_faces(
    named_meshes: Sequence[NamedMesh],
    replacement_face_keys: set[tuple[tuple[float, float, float], ...]],
    replacement_plane_coverage: Mapping[tuple[float, ...], object],
) -> list[NamedMesh]:
    """Remove exact oriented semantic faces while preserving source materials."""

    if not replacement_face_keys and not replacement_plane_coverage:
        return list(named_meshes)
    retained: list[NamedMesh] = []
    for named_mesh in named_meshes:
        world_mesh = _build_transformed_named_mesh_copy(named_mesh)
        keep_faces = np.asarray(
            [
                not _triangle_is_replaced(
                    triangle,
                    normal,
                    replacement_face_keys,
                    replacement_plane_coverage,
                )
                for triangle, normal in zip(
                    world_mesh.triangles,
                    world_mesh.face_normals,
                )
            ],
            dtype=bool,
        )
        if not np.any(keep_faces):
            continue
        if np.all(keep_faces):
            retained.append(named_mesh)
            continue
        filtered_mesh = named_mesh.mesh.copy()
        filtered_mesh.update_faces(keep_faces)
        filtered_mesh.remove_unreferenced_vertices()
        retained.append(
            NamedMesh(
                name=named_mesh.name,
                mesh=filtered_mesh,
                source_transform=named_mesh.source_transform.copy(),
                export_role=named_mesh.export_role,
            )
        )
    return retained


def _build_surface_plane_coverage(
    surfaces: Iterable[object],
) -> dict[tuple[float, ...], object]:
    polygons_by_plane: dict[tuple[float, ...], list[Polygon]] = {}
    for surface in surfaces:
        for triangle, normal in _iter_surface_replacement_triangles(surface):
            plane_key = _build_oriented_plane_key(triangle, normal)
            polygon = _project_triangle_to_plane(triangle, normal)
            if polygon.area > 0.0:
                polygons_by_plane.setdefault(plane_key, []).append(polygon)
    return {
        plane_key: shapely.union_all(polygons).buffer(
            SURFACE_FACE_COVERAGE_EPSILON
        )
        for plane_key, polygons in polygons_by_plane.items()
    }


def _iter_surface_replacement_triangles(
    surface: object,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    mesh = getattr(surface, "mesh")
    for triangle, normal in zip(mesh.triangles, mesh.face_normals):
        yield triangle, normal
        if (
            getattr(surface, "surface_type", None) == "wall"
            and getattr(surface, "room_index", None) is None
        ):
            yield triangle[::-1], -np.asarray(normal, dtype=float)


def _triangle_is_replaced(
    triangle: np.ndarray,
    normal: np.ndarray,
    replacement_face_keys: set[tuple[tuple[float, float, float], ...]],
    replacement_plane_coverage: Mapping[tuple[float, ...], object],
) -> bool:
    if _build_oriented_triangle_key(triangle) in replacement_face_keys:
        return True
    plane_key = _build_oriented_plane_key(triangle, normal)
    coverage = replacement_plane_coverage.get(plane_key)
    if coverage is None:
        return False
    projected = _project_triangle_to_plane(triangle, normal)
    return bool(coverage.covers(projected))


def _build_oriented_plane_key(
    triangle: np.ndarray,
    normal: np.ndarray,
) -> tuple[float, ...]:
    normalized = np.array(normal, dtype=float, copy=True)
    length = float(np.linalg.norm(normalized))
    if length <= 1e-12:
        return ()
    normalized /= length
    plane_offset = float(
        np.dot(normalized, np.asarray(triangle, dtype=float)[0])
    )
    return tuple(
        float(value)
        for value in np.round(
            np.append(normalized, plane_offset),
            SURFACE_FACE_MATCH_DECIMALS,
        )
    )


def _project_triangle_to_plane(
    triangle: np.ndarray,
    normal: np.ndarray,
) -> Polygon:
    drop_axis = int(np.argmax(np.abs(np.asarray(normal, dtype=float))))
    projected = np.delete(np.asarray(triangle, dtype=float), drop_axis, axis=1)
    return Polygon(projected)


def _build_oriented_triangle_key(
    triangle: np.ndarray,
) -> tuple[tuple[float, float, float], ...]:
    points = tuple(
        tuple(float(value) for value in point)
        for point in np.round(
            np.asarray(triangle, dtype=float),
            SURFACE_FACE_MATCH_DECIMALS,
        )
    )
    if len(points) != 3 or any(len(point) != 3 for point in points):
        raise ValueError("Surface replacement faces must be triangles.")
    return min(
        points,
        (points[1], points[2], points[0]),
        (points[2], points[0], points[1]),
    )


def _get_surface_object_name(surface_id: str) -> str:
    normalized = "".join(
        character if character.isalnum() else "_"
        for character in surface_id.lower()
    ).strip("_")
    return f"surface_{normalized or 'unnamed'}"


def export_glb_file(model: GeneratedModel, path: str | Path) -> Path:
    export_path = Path(path)
    export_path.write_bytes(model.glb_bytes)
    return export_path


def import_generated_glb(glb_bytes: bytes) -> GeneratedModel:
    """Load a provider GLB and build a Z-up preview without altering its export."""

    payload = bytes(glb_bytes)
    if not payload:
        raise ValueError("The generated GLB is empty.")
    try:
        loaded = trimesh.load(
            BytesIO(payload),
            file_type="glb",
            force="scene",
            process=False,
        )
    except Exception as error:
        raise ValueError("The generated GLB could not be loaded.") from error
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene(loaded)
    elif isinstance(loaded, trimesh.Scene):
        scene = loaded
    else:
        raise ValueError("The generated GLB contains no mesh scene.")

    try:
        preview_mesh = scene.to_geometry()
    except Exception as error:
        raise ValueError("The generated GLB geometry could not be combined.") from error
    if not isinstance(preview_mesh, trimesh.Trimesh):
        raise ValueError("The generated GLB contains no triangle mesh.")
    if len(preview_mesh.vertices) == 0 or len(preview_mesh.faces) == 0:
        raise ValueError("The generated GLB contains empty geometry.")
    if len(preview_mesh.faces) > MAX_IMPORTED_GENERATED_MODEL_FACES:
        raise ValueError("The generated GLB contains too many faces.")
    if not np.all(np.isfinite(preview_mesh.vertices)):
        raise ValueError("The generated GLB contains invalid vertex coordinates.")

    preview_mesh = preview_mesh.copy()
    preview_mesh.apply_transform(GLTF_Y_UP_TO_Z_UP_TRANSFORM)
    return GeneratedModel(
        mesh=preview_mesh,
        scene=scene,
        glb_bytes=payload,
    )


# ### Internal helpers ###
def _build_export_scene(named_meshes: list[NamedMesh]) -> trimesh.Scene:
    scene = trimesh.Scene()
    for named_mesh in named_meshes:
        scene.add_geometry(
            _to_gltf_y_up_mesh(named_mesh.mesh),
            geom_name=named_mesh.name,
            node_name=named_mesh.name,
            transform=_source_to_gltf_y_up_transform(
                named_mesh.source_transform
            ),
        )
    return scene


def _serialize_export_scene(scene: trimesh.Scene) -> bytes:
    """Serialize a scene, including the valid empty-scene GLB edge case."""

    if scene.geometry:
        payload = scene.export(file_type="glb")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError("The house scene could not be exported as GLB.")
        return bytes(payload)

    json_chunk = b'{"asset":{"version":"2.0"},"scene":0,"scenes":[{}]}'
    json_chunk += b" " * (-len(json_chunk) % 4)
    total_length = 12 + 8 + len(json_chunk)
    return b"".join(
        (
            b"glTF",
            (2).to_bytes(4, "little"),
            total_length.to_bytes(4, "little"),
            len(json_chunk).to_bytes(4, "little"),
            b"JSON",
            json_chunk,
        )
    )


def _to_gltf_y_up_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    export_mesh = mesh.copy()
    export_mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    return export_mesh


def _build_transformed_named_mesh_copy(named_mesh: NamedMesh) -> trimesh.Trimesh:
    transformed_mesh = named_mesh.mesh.copy()
    transformed_mesh.apply_transform(
        _get_valid_source_transform(named_mesh.source_transform)
    )
    return transformed_mesh


def _source_to_gltf_y_up_transform(source_transform: np.ndarray) -> np.ndarray:
    valid_source_transform = _get_valid_source_transform(source_transform)
    return (
        Z_UP_TO_GLTF_Y_UP_TRANSFORM
        @ valid_source_transform
        @ GLTF_Y_UP_TO_Z_UP_TRANSFORM
    )


def _get_valid_source_transform(source_transform: np.ndarray) -> np.ndarray:
    try:
        transform = np.asarray(source_transform, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Mesh source transform must be a 4 by 4 matrix.") from error

    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Mesh source transform must be a finite 4 by 4 matrix.")

    return transform


def _build_named_meshes_for_single_level(
    wall_meshes: list[trimesh.Trimesh],
) -> list[NamedMesh]:
    if not wall_meshes:
        return []

    return [
        NamedMesh(
            name="level_2_ground",
            mesh=_combine_mesh_geometry(wall_meshes),
        )
    ]


def _build_multi_level_meshes(
    levels: Sequence[LevelData],
    blueprint_size_pixels: tuple[float, float] | None,
    stairs: Sequence[StairData] = (),
) -> list[NamedMesh]:
    if not levels:
        raise ValueError("No levels are available for GLB conversion.")

    sorted_levels = sorted(levels, key=lambda level: level.index)
    level_lookup = {level.index: level for level in sorted_levels}
    named_meshes: list[NamedMesh] = []

    for level in sorted_levels:
        if not level.include_in_export:
            continue

        if not math.isfinite(level.height_meters) or level.height_meters <= 0.0:
            raise ValueError(f"Level {level.index} height must be greater than zero.")
        if (
            not math.isfinite(level.floor_thickness_meters)
            or level.floor_thickness_meters <= 0.0
        ):
            raise ValueError(
                f"Level {level.index} floor thickness must be greater than zero."
            )
        level_named_meshes = _build_named_meshes_for_level(
            level=level,
            level_lookup=level_lookup,
            blueprint_size_pixels=blueprint_size_pixels,
        )
        if not level_named_meshes:
            continue

        named_meshes.extend(level_named_meshes)

    named_meshes.extend(
        build_stair_meshes(
            sorted_levels,
            _filter_stairs_for_export(stairs, level_lookup),
        )
    )
    return named_meshes


def _filter_stairs_for_export(
    stairs: Sequence[StairData],
    level_lookup: Mapping[int, LevelData],
) -> list[StairData]:
    """Keep stairs only when every referenced route level is exported.

    An unknown route level is deliberately retained so ``build_stair_meshes``
    can report the invalid project data instead of silently omitting it.
    """

    exportable_stairs: list[StairData] = []
    for stair in stairs:
        route_levels = [
            level_lookup.get(section.level_index)
            for section in stair.sections
        ]
        if any(level is None for level in route_levels):
            exportable_stairs.append(stair)
            continue
        if all(
            level.include_in_export
            for level in route_levels
            if level is not None
        ):
            exportable_stairs.append(stair)
    return exportable_stairs


def _build_named_meshes_for_level(
    level: LevelData,
    level_lookup: dict[int, LevelData],
    blueprint_size_pixels: tuple[float, float] | None,
) -> list[NamedMesh]:
    base_z_meters = _get_level_base_z(level_lookup, level.index)
    level_blueprint_size = level.image_size_pixels or blueprint_size_pixels
    level_source_transform = _build_level_source_transform(
        level,
        level_blueprint_size,
    )
    room_vertex_sets = _get_room_vertex_sets(level.rooms)
    named_meshes: list[NamedMesh] = []
    floor_mesh = build_level_floor_mesh(
        level=level,
        floor_surface_z_meters=base_z_meters,
        blueprint_size_pixels=level_blueprint_size,
        point_to_world_xy=_point_to_world_xy,
    )
    if floor_mesh is not None:
        named_meshes.append(
            NamedMesh(
                name=_get_level_floor_object_name(level),
                mesh=floor_mesh,
            )
        )

    regular_wall_meshes = _build_level_meshes(
        vertex_data=level.vertex_data,
        wall_height_meters=level.height_meters,
        base_z_meters=base_z_meters,
        blueprint_size_pixels=level_blueprint_size,
        doorways=level.doorways,
        window_openings=_build_window_openings(level),
        ignored_vertex_ids=_get_room_center_vertex_ids(level.rooms),
        ignored_room_vertex_sets=room_vertex_sets,
    )

    if regular_wall_meshes:
        named_meshes.append(
            NamedMesh(
                name=_get_level_object_name(level),
                mesh=_combine_mesh_geometry(regular_wall_meshes),
            )
        )

    named_meshes.extend(
        _build_room_named_meshes(
            level=level,
            base_z_meters=base_z_meters,
            blueprint_size_pixels=level_blueprint_size,
        )
    )

    doorway_reveal_mesh = _build_level_doorway_reveal_mesh(
        level=level,
        base_z_meters=base_z_meters,
        blueprint_size_pixels=level_blueprint_size,
        room_vertex_sets=room_vertex_sets,
    )
    if doorway_reveal_mesh is not None:
        named_meshes.append(
            NamedMesh(
                name=_get_level_doorway_reveal_object_name(level),
                mesh=doorway_reveal_mesh,
                export_role=NAMED_MESH_ROLE_OPENING_REVEAL,
            )
        )

    window_reveal_mesh = _build_level_window_reveal_mesh(
        level=level,
        base_z_meters=base_z_meters,
        blueprint_size_pixels=level_blueprint_size,
    )
    if window_reveal_mesh is not None:
        named_meshes.append(
            NamedMesh(
                name=_get_level_window_reveal_object_name(level),
                mesh=window_reveal_mesh,
                export_role=NAMED_MESH_ROLE_OPENING_REVEAL,
            )
        )

    for named_mesh in named_meshes:
        named_mesh.source_transform = level_source_transform.copy()

    return named_meshes


def _build_room_named_meshes(
    level: LevelData,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
) -> list[NamedMesh]:
    named_meshes: list[NamedMesh] = []
    for room_index, room in enumerate(level.rooms):
        if room.height_meters <= 0.0:
            raise ValueError(
                f"Room {room.name or room_index + 1} height must be greater "
                "than zero."
            )

        room_mesh = _build_room_mesh(
            level=level,
            room=room,
            room_index=room_index,
            base_z_meters=base_z_meters,
            blueprint_size_pixels=blueprint_size_pixels,
        )
        if room_mesh is None:
            continue

        named_meshes.append(
            NamedMesh(
                name=_get_room_object_name(level, room, room_index),
                mesh=room_mesh,
            )
        )

    return named_meshes


def _build_level_meshes(
    vertex_data: VertexData,
    wall_height_meters: float,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
    doorways: Sequence[object] = (),
    window_openings: Sequence[WallOpening] = (),
    ignored_vertex_ids: set[int] | None = None,
    ignored_room_vertex_sets: list[set[int]] | None = None,
) -> list[trimesh.Trimesh]:
    if wall_height_meters <= 0.0:
        raise ValueError("Height level must be greater than zero.")

    ignored_ids = ignored_vertex_ids or set()
    room_vertex_sets = ignored_room_vertex_sets or []
    vertex_lookup = {vertex.id: vertex for vertex in vertex_data.vertices}
    doorway_openings = [
        *_build_wall_openings(doorways),
        *window_openings,
    ]
    return [
        wall_mesh
        for edge in vertex_data.edges
        if (
            edge.start_vertex_id not in ignored_ids
            and edge.end_vertex_id not in ignored_ids
            and not _is_edge_inside_any_room(edge, room_vertex_sets)
        )
        if (
            wall_mesh := _build_wall_mesh(
                edge=edge,
                vertex_lookup=vertex_lookup,
                wall_height_meters=wall_height_meters,
                base_z_meters=base_z_meters,
                blueprint_size_pixels=blueprint_size_pixels,
                doorway_openings=doorway_openings,
            )
        )
        is not None
    ]


def _build_room_mesh(
    level: LevelData,
    room: RoomData,
    room_index: int,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
) -> trimesh.Trimesh | None:
    room_walls = build_room_walls(room, level.vertex_data)
    if not room_walls:
        return None

    layout = build_uv_wall_layout(
        room=room,
        vertex_data=level.vertex_data,
        wall_height_meters=room.height_meters,
    )
    placements_by_key = _group_wall_placements_by_key(layout.placements)
    doorway_openings = _build_level_wall_openings(level)
    material = _build_room_material(level, room, room_index, layout)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    uv_coordinates: list[tuple[float, float]] = []

    for wall in room_walls:
        wall_placements = placements_by_key.get(wall.key, [])
        if not wall_placements:
            for wall_piece in _build_visible_wall_pieces(
                start_point=wall.start_point,
                end_point=wall.end_point,
                wall_height_meters=room.height_meters,
                doorway_openings=doorway_openings,
                wall_key=wall.key,
            ):
                wall_vertices = _build_wall_piece_vertices_from_points(
                    start_point=wall.start_point,
                    end_point=wall.end_point,
                    wall_piece=wall_piece,
                    base_z_meters=base_z_meters,
                    blueprint_size_pixels=blueprint_size_pixels,
                )
                if wall_vertices is None:
                    continue

                vertex_offset = len(vertices)
                vertices.extend(wall_vertices)
                faces.extend(
                    _build_wall_faces(
                        vertex_offset,
                        len(wall_vertices),
                        double_sided=False,
                    )
                )
                uv_coordinates.extend(
                    _build_hidden_wall_uv_coordinates(len(wall_vertices))
                )
            continue

        for placement in wall_placements:
            segment_start_point, segment_end_point = _get_wall_segment_points(
                wall=wall,
                placement=placement,
            )
            for wall_piece in _build_visible_wall_pieces(
                start_point=segment_start_point,
                end_point=segment_end_point,
                wall_height_meters=room.height_meters,
                doorway_openings=doorway_openings,
                wall_key=wall.key,
            ):
                wall_vertices = _build_wall_piece_vertices_from_points(
                    start_point=segment_start_point,
                    end_point=segment_end_point,
                    wall_piece=wall_piece,
                    base_z_meters=base_z_meters,
                    blueprint_size_pixels=blueprint_size_pixels,
                )
                if wall_vertices is None:
                    continue

                vertex_offset = len(vertices)
                vertices.extend(wall_vertices)
                faces.extend(
                    _build_wall_faces(
                        vertex_offset,
                        len(wall_vertices),
                        double_sided=False,
                    )
                )
                uv_coordinates.extend(
                    _build_wall_piece_uv_coordinates(
                        room=room,
                        placement=placement,
                        wall_piece=wall_piece,
                        wall_height_meters=room.height_meters,
                    )
                )

    if not vertices or not faces:
        return None

    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        visual=TextureVisuals(
            uv=np.asarray(uv_coordinates, dtype=float),
            material=material,
        ),
        process=False,
    )


def _group_wall_placements_by_key(
    placements: Sequence[UvWallPlacement],
) -> dict[str, list[UvWallPlacement]]:
    placements_by_key: dict[str, list[UvWallPlacement]] = {}
    for placement in placements:
        placements_by_key.setdefault(placement.wall.key, []).append(placement)

    return placements_by_key


# ### Material helpers ###
def _build_room_material(
    level: LevelData,
    room: RoomData,
    room_index: int,
    layout: UvLayout,
) -> PBRMaterial:
    return PBRMaterial(
        name=_get_room_material_name(level, room, room_index),
        baseColorFactor=[255, 255, 255, 255],
        baseColorTexture=_build_room_texture(room, layout),
        metallicFactor=0.0,
        roughnessFactor=0.65,
        doubleSided=False,
    )


def _build_room_texture(room: RoomData, layout: UvLayout) -> PngTexture:
    image = _build_room_texture_image(room, layout)
    return PngTexture(png_bytes=_qimage_to_png_bytes(image))


def _build_room_texture_image(room: RoomData, layout: UvLayout) -> QImage:
    _ensure_qt_application()
    texture_width = max(1, int(room.uv_map_width))
    texture_height = max(1, int(room.uv_map_height))
    image = QImage(
        texture_width,
        texture_height,
        QImage.Format.Format_RGBA8888,
    )
    image.fill(ROOM_TEXTURE_BACKGROUND_COLOR)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for placement in layout.placements:
        _paint_room_texture_wall(painter, room, placement)

    if layout.hidden_wall_count > 0:
        _paint_room_texture_hidden_indicator(
            painter=painter,
            texture_width=texture_width,
            texture_height=texture_height,
            hidden_wall_count=layout.hidden_wall_count,
        )

    painter.end()
    return image


def _ensure_qt_application() -> None:
    global _fallback_qt_application

    if QGuiApplication.instance() is not None:
        return

    os.environ.setdefault("QT_QPA_PLATFORM", FALLBACK_QT_PLATFORM)
    _fallback_qt_application = QGuiApplication([])


# ### Texture helpers ###
def _paint_room_texture_wall(
    painter: QPainter,
    room: RoomData,
    placement: UvWallPlacement,
) -> None:
    uv_x, uv_y, uv_width, uv_height = placement.uv_rect
    wall_width, wall_height = placement.natural_size
    texture_rect = QRectF(
        -wall_width / 2.0,
        -wall_height / 2.0,
        wall_width,
        wall_height,
    ).adjusted(0.5, 0.5, -0.5, -0.5)

    painter.save()
    painter.translate(uv_x + uv_width / 2.0, uv_y + uv_height / 2.0)
    painter.rotate(placement.rotation_degrees)
    texture_data = room.wall_textures.get(placement.wall.key)
    did_paint_texture = (
        texture_data is not None
        and paint_wall_texture_crop(
            painter,
            texture_data,
            texture_rect,
            placement.source_start_ratio,
            placement.source_end_ratio,
        )
    )

    if not did_paint_texture:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(ROOM_TEXTURE_WALL_FILL_COLOR)
        painter.drawRect(texture_rect)
        painter.setPen(QPen(ROOM_TEXTURE_TEXT_COLOR))
        painter.setFont(
            QFont("Segoe UI", _get_room_texture_label_font_size(texture_rect))
        )
        painter.drawText(
            texture_rect,
            int(Qt.AlignmentFlag.AlignCenter),
            f"{placement.wall.projection_direction}\n{placement.rotation_degrees} deg",
        )
    painter.restore()


def _paint_room_texture_hidden_indicator(
    painter: QPainter,
    texture_width: int,
    texture_height: int,
    hidden_wall_count: int,
) -> None:
    indicator_width = min(190.0, max(1.0, texture_width - 20.0))
    indicator_rect = QRectF(
        10.0,
        max(0.0, texture_height - 34.0),
        indicator_width,
        24.0,
    )
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(ROOM_TEXTURE_INDICATOR_BACKGROUND_COLOR)
    painter.drawRoundedRect(indicator_rect, 6.0, 6.0)
    painter.setPen(QPen(ROOM_TEXTURE_INDICATOR_TEXT_COLOR))
    painter.setFont(QFont("Segoe UI", 9))
    painter.drawText(
        indicator_rect,
        int(Qt.AlignmentFlag.AlignCenter),
        f"{hidden_wall_count} walls are not shown",
    )


def _get_room_texture_label_font_size(texture_rect: QRectF) -> int:
    raw_size = int(min(texture_rect.width(), texture_rect.height()) / 4.0)
    return min(
        ROOM_TEXTURE_MAX_FONT_SIZE,
        max(ROOM_TEXTURE_MIN_FONT_SIZE, raw_size),
    )


def _qimage_to_png_bytes(image: QImage) -> bytes:
    byte_array = QByteArray()
    buffer = QBuffer(byte_array)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(byte_array)


def _qimage_to_gl_rgba_array(image: QImage) -> np.ndarray:
    converted_image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    image_width = converted_image.width()
    image_height = converted_image.height()
    bytes_per_line = converted_image.bytesPerLine()
    image_buffer = converted_image.bits()
    image_array = np.frombuffer(
        image_buffer,
        dtype=np.uint8,
        count=bytes_per_line * image_height,
    )
    image_array = image_array.reshape((image_height, bytes_per_line))
    image_array = image_array[:, : image_width * 4].reshape(
        (image_height, image_width, 4)
    )
    return np.flip(np.swapaxes(image_array, 0, 1), axis=1).copy()


def _build_wall_preview_texture(
    room: RoomData,
    placement: UvWallPlacement,
) -> np.ndarray:
    _ensure_qt_application()
    wall_width, wall_height = placement.natural_size
    texture_width = max(1, int(math.ceil(wall_width)))
    texture_height = max(1, int(math.ceil(wall_height)))
    image = QImage(
        texture_width,
        texture_height,
        QImage.Format.Format_RGBA8888,
    )
    image.fill(ROOM_TEXTURE_WALL_FILL_COLOR)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    texture_rect = QRectF(
        0.5,
        0.5,
        max(1.0, texture_width - 1.0),
        max(1.0, texture_height - 1.0),
    )
    texture_data = room.wall_textures.get(placement.wall.key)
    did_paint_texture = (
        texture_data is not None
        and paint_wall_texture_crop(
            painter,
            texture_data,
            texture_rect,
            placement.source_start_ratio,
            placement.source_end_ratio,
        )
    )
    if not did_paint_texture:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(ROOM_TEXTURE_WALL_FILL_COLOR)
        painter.drawRect(texture_rect)
        painter.setPen(QPen(ROOM_TEXTURE_TEXT_COLOR))
        painter.setFont(
            QFont("Segoe UI", _get_room_texture_label_font_size(texture_rect))
        )
        painter.drawText(
            texture_rect,
            int(Qt.AlignmentFlag.AlignCenter),
            f"{placement.wall.projection_direction}\n{placement.rotation_degrees} deg",
        )
    painter.end()
    return _qimage_to_gl_rgba_array(image)


def _crop_wall_preview_texture(
    texture_rgba: np.ndarray,
    wall_piece: WallPiece,
    wall_height_meters: float,
) -> np.ndarray:
    """Crop a wall preview texture to the same geometry left after a cut."""
    if (
        texture_rgba.ndim != 3
        or texture_rgba.shape[0] <= 0
        or texture_rgba.shape[1] <= 0
        or wall_height_meters <= WALL_OPENING_EPSILON
    ):
        return texture_rgba

    width_pixels, height_pixels = texture_rgba.shape[:2]
    bottom_ratio = min(
        max(wall_piece.bottom_height_meters / wall_height_meters, 0.0),
        1.0,
    )
    top_ratio = min(
        max(wall_piece.top_height_meters / wall_height_meters, 0.0),
        1.0,
    )
    start_x, end_x = _get_texture_crop_bounds(
        wall_piece.start_ratio,
        wall_piece.end_ratio,
        width_pixels,
    )
    start_y, end_y = _get_texture_crop_bounds(
        bottom_ratio,
        top_ratio,
        height_pixels,
    )
    return texture_rgba[start_x:end_x, start_y:end_y].copy()


def _mask_wall_preview_texture(
    texture_rgba: np.ndarray,
    wall_pieces: Sequence[WallPiece],
    wall_height_meters: float,
) -> np.ndarray:
    """Mask one full wall image to the exact triangulated visible region."""

    if (
        texture_rgba.ndim != 3
        or texture_rgba.shape[2] != 4
        or texture_rgba.shape[0] <= 0
        or texture_rgba.shape[1] <= 0
        or wall_height_meters <= WALL_OPENING_EPSILON
    ):
        return texture_rgba
    texture_width, texture_height = texture_rgba.shape[:2]
    mask_image = QImage(
        texture_width,
        texture_height,
        QImage.Format.Format_RGBA8888,
    )
    mask_image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(mask_image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(Qt.GlobalColor.white)
    visible_path = QPainterPath()
    visible_path.setFillRule(Qt.FillRule.WindingFill)
    for wall_piece in wall_pieces:
        path = QPainterPath()
        for point_index, (wall_ratio, height_meters) in enumerate(
            wall_piece.points
        ):
            point_x = min(max(wall_ratio, 0.0), 1.0) * texture_width
            point_y = (
                1.0
                - min(max(height_meters / wall_height_meters, 0.0), 1.0)
            ) * texture_height
            if point_index == 0:
                path.moveTo(point_x, point_y)
            else:
                path.lineTo(point_x, point_y)
        path.closeSubpath()
        visible_path.addPath(path)
    painter.fillPath(visible_path.simplified(), Qt.GlobalColor.white)
    painter.end()
    mask_alpha = _qimage_to_gl_rgba_array(mask_image)[:, :, 3]
    masked_texture = texture_rgba.copy()
    masked_texture[:, :, 3] = (
        masked_texture[:, :, 3].astype(np.uint16)
        * mask_alpha.astype(np.uint16)
        // 255
    ).astype(np.uint8)
    return masked_texture


def _get_texture_crop_bounds(
    start_ratio: float,
    end_ratio: float,
    size_pixels: int,
) -> tuple[int, int]:
    safe_start_ratio = min(max(start_ratio, 0.0), 1.0)
    safe_end_ratio = min(max(end_ratio, safe_start_ratio), 1.0)
    start_index = min(
        max(0, int(math.floor(safe_start_ratio * size_pixels))),
        size_pixels - 1,
    )
    end_index = min(
        size_pixels,
        max(start_index + 1, int(math.ceil(safe_end_ratio * size_pixels))),
    )
    return start_index, end_index


# ### Wall geometry helpers ###
def _get_wall_opening_profile_points(
    opening: WallOpening,
) -> tuple[tuple[float, float], ...]:
    """Return one closed relative-height profile for every opening type."""

    if opening.profile_points:
        return opening.profile_points
    half_width_meters = opening.half_width_pixels * PIXEL_TO_METER
    return (
        (-half_width_meters, 0.0),
        (half_width_meters, 0.0),
        (half_width_meters, opening.height_meters),
        (-half_width_meters, opening.height_meters),
        (-half_width_meters, 0.0),
    )


def _build_wall_openings(doorways: Sequence[object]) -> list[WallOpening]:
    """Convert usable doorway data into clipping footprints once per mesh."""
    wall_openings: list[WallOpening] = []
    for doorway in doorways:
        try:
            center_x = float(getattr(doorway, "center_x"))
            center_y = float(getattr(doorway, "center_y"))
            width_meters = float(getattr(doorway, "width_meters"))
            height_meters = float(getattr(doorway, "height_meters"))
            depth_meters = float(getattr(doorway, "depth_meters"))
            rotation_degrees = float(getattr(doorway, "rotation_degrees"))
            bottom_height_meters = float(
                getattr(
                    doorway,
                    "bottom_height_meters",
                    DEFAULT_DOORWAY_BOTTOM_HEIGHT_METERS,
                )
            )
            profile_points = build_doorway_cross_section_outline(
                width_meters,
                height_meters,
                getattr(doorway, "shape", DEFAULT_DOORWAY_SHAPE),
                getattr(
                    doorway,
                    "arch_amount",
                    DEFAULT_DOORWAY_ARCH_AMOUNT,
                ),
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            continue

        doorway_values = (
            center_x,
            center_y,
            width_meters,
            height_meters,
            depth_meters,
            rotation_degrees,
            bottom_height_meters,
        )
        if not all(math.isfinite(value) for value in doorway_values):
            continue
        if (
            width_meters <= WALL_OPENING_EPSILON
            or height_meters <= WALL_OPENING_EPSILON
            or depth_meters <= WALL_OPENING_EPSILON
            or bottom_height_meters < 0.0
        ):
            continue

        rotation_radians = math.radians(rotation_degrees)
        depth_direction_x = math.cos(rotation_radians)
        depth_direction_y = math.sin(rotation_radians)
        wall_openings.append(
            WallOpening(
                center_x=center_x,
                center_y=center_y,
                width_direction_x=-depth_direction_y,
                width_direction_y=depth_direction_x,
                depth_direction_x=depth_direction_x,
                depth_direction_y=depth_direction_y,
                half_width_pixels=width_meters / PIXEL_TO_METER / 2.0,
                half_depth_pixels=depth_meters / PIXEL_TO_METER / 2.0,
                height_meters=height_meters,
                bottom_height_meters=bottom_height_meters,
                profile_points=profile_points,
            )
        )

    return wall_openings


def _build_level_wall_openings(level: LevelData) -> list[WallOpening]:
    """Build doorway and stable wall-attached window cuts for one level."""

    return [
        *_build_wall_openings(level.doorways),
        *_build_window_openings(level),
    ]


def _build_window_openings(level: LevelData) -> list[WallOpening]:
    openings: list[WallOpening] = []
    targets = _build_window_wall_target_lookup(level)
    for window in getattr(level, "windows", ()):
        wall_surface_id = str(getattr(window, "wall_surface_id", "")).strip()
        target = targets.get(wall_surface_id)
        if target is None:
            continue
        try:
            start_ratio = float(getattr(window, "start_ratio"))
            end_ratio = float(getattr(window, "end_ratio"))
            bottom_ratio = float(getattr(window, "bottom_ratio"))
            top_ratio = float(getattr(window, "top_ratio"))
        except (AttributeError, TypeError, ValueError, OverflowError):
            continue
        ratios = (start_ratio, end_ratio, bottom_ratio, top_ratio)
        if not all(math.isfinite(value) for value in ratios):
            continue
        if not (
            0.0 <= start_ratio < end_ratio <= 1.0
            and 0.0 <= bottom_ratio < top_ratio <= 1.0
        ):
            continue

        segment_delta_x = target.end_point[0] - target.start_point[0]
        segment_delta_y = target.end_point[1] - target.start_point[1]
        segment_length = math.hypot(segment_delta_x, segment_delta_y)
        if segment_length <= WALL_OPENING_EPSILON:
            continue
        width_direction_x = segment_delta_x / segment_length
        width_direction_y = segment_delta_y / segment_length
        center_ratio = (start_ratio + end_ratio) / 2.0
        center = _interpolate_2d_point(
            target.start_point,
            target.end_point,
            center_ratio,
        )
        (
            opening_center,
            depth_direction,
            half_depth_pixels,
        ) = _build_window_depth_footprint(level, target, center)
        opening_height = (top_ratio - bottom_ratio) * target.height_meters
        if opening_height <= WALL_OPENING_EPSILON:
            continue
        openings.append(
            WallOpening(
                center_x=opening_center[0],
                center_y=opening_center[1],
                width_direction_x=width_direction_x,
                width_direction_y=width_direction_y,
                depth_direction_x=depth_direction[0],
                depth_direction_y=depth_direction[1],
                half_width_pixels=(
                    segment_length * (end_ratio - start_ratio) / 2.0
                ),
                half_depth_pixels=half_depth_pixels,
                height_meters=opening_height,
                bottom_height_meters=bottom_ratio * target.height_meters,
                target_wall_key=target.key,
                target_surface_id=wall_surface_id,
                parallel_walls_only=True,
                penetrates_parallel_walls=True,
            )
        )
    return openings


def _build_window_wall_target_lookup(
    level: LevelData,
) -> dict[str, WallSource]:
    targets: dict[str, WallSource] = {}
    for room in level.rooms:
        for wall in build_room_walls(room, level.vertex_data):
            exterior_direction = _get_wall_right_normal(
                wall.start_point,
                wall.end_point,
            )
            surface_id = (
                f"level:{level.index}/room:{room.center_vertex_id}/"
                f"wall:{wall.key}"
            )
            targets[surface_id] = WallSource(
                key=wall.key,
                start_point=wall.start_point,
                end_point=wall.end_point,
                height_meters=room.height_meters,
                wall_key=wall.key,
                surface_id=surface_id,
                exterior_direction=exterior_direction,
            )

    room_vertex_sets = _get_room_vertex_sets(level.rooms)
    ignored_vertex_ids = _get_room_center_vertex_ids(level.rooms)
    vertex_lookup = {vertex.id: vertex for vertex in level.vertex_data.vertices}
    level_contour = _build_level_image_contour_polygon(level)
    for edge in level.vertex_data.edges:
        if (
            edge.start_vertex_id in ignored_vertex_ids
            or edge.end_vertex_id in ignored_vertex_ids
            or _is_edge_inside_any_room(edge, room_vertex_sets)
        ):
            continue
        start_vertex = vertex_lookup.get(edge.start_vertex_id)
        end_vertex = vertex_lookup.get(edge.end_vertex_id)
        if start_vertex is None or end_vertex is None:
            continue
        wall_key = (
            f"{min(edge.start_vertex_id, edge.end_vertex_id)}:"
            f"{max(edge.start_vertex_id, edge.end_vertex_id)}"
        )
        surface_id = f"level:{level.index}/wall:{wall_key}"
        targets[surface_id] = WallSource(
            key=wall_key,
            start_point=(start_vertex.x, start_vertex.y),
            end_point=(end_vertex.x, end_vertex.y),
            height_meters=level.height_meters,
            wall_key=wall_key,
            surface_id=surface_id,
            exterior_direction=_get_wall_exterior_from_contour(
                (start_vertex.x, start_vertex.y),
                (end_vertex.x, end_vertex.y),
                level_contour,
            ),
        )
    return targets


def _build_window_depth_footprint(
    level: LevelData,
    target: WallSource,
    center: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Extend a window 50 cm behind its selected semantic wall face."""

    depth_pixels = WINDOW_CUT_DEPTH_METERS / (
        PIXEL_TO_METER * _get_valid_level_scale(level)
    )
    exterior_direction = (
        target.exterior_direction
        if target.exterior_direction is not None
        else _get_wall_right_normal(
            target.start_point,
            target.end_point,
        )
    )
    if exterior_direction is None:
        return center, (1.0, 0.0), WINDOW_COPLANAR_DEPTH_PIXELS

    interior_margin = WINDOW_COPLANAR_DEPTH_PIXELS
    center_offset = (depth_pixels - interior_margin) / 2.0
    half_depth = (depth_pixels + interior_margin) / 2.0
    return (
        (
            center[0] + exterior_direction[0] * center_offset,
            center[1] + exterior_direction[1] * center_offset,
        ),
        exterior_direction,
        half_depth,
    )


def _get_wall_right_normal(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
) -> tuple[float, float] | None:
    delta_x = end_point[0] - start_point[0]
    delta_y = end_point[1] - start_point[1]
    length = math.hypot(delta_x, delta_y)
    if length <= WALL_OPENING_EPSILON:
        return None
    return delta_y / length, -delta_x / length


def _build_level_image_contour_polygon(level: LevelData) -> Polygon | None:
    points: list[tuple[float, float]] = []
    for vertex_id in level.floor_contour_vertex_ids:
        vertex = level.vertex_data.get_vertex(vertex_id)
        if vertex is not None:
            points.append((vertex.x, vertex.y))
    if len(points) < 3:
        return None
    polygon = Polygon(points)
    if not polygon.is_valid:
        repaired = polygon.buffer(0)
        if not isinstance(repaired, Polygon):
            return None
        polygon = repaired
    if polygon.area <= WALL_OPENING_EPSILON:
        return None
    return polygon


def _get_wall_exterior_from_contour(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    contour: Polygon | None,
) -> tuple[float, float] | None:
    if contour is None or contour.is_empty:
        return None
    right_normal = _get_wall_right_normal(start_point, end_point)
    if right_normal is None:
        return None
    left_normal = (-right_normal[0], -right_normal[1])
    midpoint = (
        (start_point[0] + end_point[0]) / 2.0,
        (start_point[1] + end_point[1]) / 2.0,
    )
    wall_length = _get_2d_point_distance(start_point, end_point)
    for probe_ratio in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        probe_distance = max(
            WALL_OPENING_EPSILON * 10.0,
            wall_length * probe_ratio,
        )
        right_is_inside = contour.contains(
            Point(
                midpoint[0] + right_normal[0] * probe_distance,
                midpoint[1] + right_normal[1] * probe_distance,
            )
        )
        left_is_inside = contour.contains(
            Point(
                midpoint[0] + left_normal[0] * probe_distance,
                midpoint[1] + left_normal[1] * probe_distance,
            )
        )
        if right_is_inside != left_is_inside:
            return left_normal if right_is_inside else right_normal
    return None


def _build_visible_wall_pieces(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    wall_height_meters: float,
    doorway_openings: Sequence[WallOpening],
    wall_key: str | None = None,
) -> list[WallPiece]:
    """Triangulate the visible wall around exact polygonal opening profiles."""
    if (
        not math.isfinite(wall_height_meters)
        or wall_height_meters <= WALL_OPENING_EPSILON
        or _get_2d_point_distance(start_point, end_point) <= WALL_OPENING_EPSILON
    ):
        return []

    applicable_openings: list[WallOpening] = []
    for doorway_opening in doorway_openings:
        if (
            doorway_opening.target_wall_key is not None
            and doorway_opening.target_wall_key != wall_key
            and not doorway_opening.penetrates_parallel_walls
        ):
            continue
        if (
            doorway_opening.parallel_walls_only
            and not _wall_segment_is_parallel_to_opening(
                start_point,
                end_point,
                doorway_opening,
            )
        ):
            continue
        applicable_openings.append(doorway_opening)

    full_wall_points = (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, wall_height_meters),
        (0.0, wall_height_meters),
    )
    if not applicable_openings:
        return [WallPiece(points=full_wall_points)]
    if all(
        _wall_opening_profile_is_rectangular(opening)
        for opening in applicable_openings
    ):
        return _build_visible_rectangular_wall_pieces(
            start_point,
            end_point,
            wall_height_meters,
            applicable_openings,
        )

    opening_polygons: list[Polygon] = []
    for doorway_opening in applicable_openings:
        opening_polygon = _build_opening_polygon_on_wall(
            start_point,
            end_point,
            wall_height_meters,
            doorway_opening,
        )
        if opening_polygon is not None:
            opening_polygons.append(opening_polygon)

    if not opening_polygons:
        return [WallPiece(points=full_wall_points)]

    wall_polygon = Polygon(full_wall_points)
    visible_geometry = wall_polygon.difference(shapely.union_all(opening_polygons))
    if visible_geometry.is_empty:
        return []
    return _triangulate_visible_wall_geometry(visible_geometry)


def _wall_opening_profile_is_rectangular(opening: WallOpening) -> bool:
    profile_points = _get_wall_opening_profile_points(opening)
    if len(profile_points) != 5 or profile_points[0] != profile_points[-1]:
        return False
    width_values = {round(point[0], 12) for point in profile_points[:-1]}
    height_values = {round(point[1], 12) for point in profile_points[:-1]}
    return len(width_values) == 2 and len(height_values) == 2


def _build_visible_rectangular_wall_pieces(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    wall_height_meters: float,
    openings: Sequence[WallOpening],
) -> list[WallPiece]:
    """Preserve the compact exact mesh used by rectangular cuts."""

    opening_intervals: list[tuple[float, float, float, float]] = []
    for opening in openings:
        interval = _clip_wall_segment_to_opening(
            start_point=start_point,
            end_point=end_point,
            doorway_opening=opening,
        )
        if interval is None:
            continue
        opening_bottom = min(
            max(0.0, opening.bottom_height_meters),
            wall_height_meters,
        )
        opening_top = min(
            max(opening_bottom, opening.top_height_meters),
            wall_height_meters,
        )
        if opening_top - opening_bottom <= WALL_OPENING_EPSILON:
            continue
        opening_intervals.append(
            (interval[0], interval[1], opening_bottom, opening_top)
        )
    if not opening_intervals:
        return [
            _build_rectangular_wall_piece(0.0, 1.0, 0.0, wall_height_meters)
        ]

    breakpoints = _get_opening_interval_breakpoints(opening_intervals)
    wall_pieces: list[WallPiece] = []
    for interval_start, interval_end in zip(breakpoints, breakpoints[1:]):
        if interval_end - interval_start <= WALL_OPENING_EPSILON:
            continue
        interval_midpoint = (interval_start + interval_end) / 2.0
        covered_vertical_intervals = _merge_wall_opening_vertical_intervals(
            [
                (opening_bottom, opening_top)
                for opening_start, opening_end, opening_bottom, opening_top
                in opening_intervals
                if opening_start - WALL_OPENING_EPSILON
                <= interval_midpoint
                <= opening_end + WALL_OPENING_EPSILON
            ]
        )
        visible_bottom = 0.0
        for opening_bottom, opening_top in covered_vertical_intervals:
            if opening_bottom - visible_bottom > WALL_OPENING_EPSILON:
                _append_or_merge_wall_piece(
                    wall_pieces,
                    _build_rectangular_wall_piece(
                        interval_start,
                        interval_end,
                        visible_bottom,
                        opening_bottom,
                    ),
                )
            visible_bottom = max(visible_bottom, opening_top)
        if wall_height_meters - visible_bottom > WALL_OPENING_EPSILON:
            _append_or_merge_wall_piece(
                wall_pieces,
                _build_rectangular_wall_piece(
                    interval_start,
                    interval_end,
                    visible_bottom,
                    wall_height_meters,
                ),
            )
    return wall_pieces


def _build_rectangular_wall_piece(
    start_ratio: float,
    end_ratio: float,
    bottom_height_meters: float,
    top_height_meters: float,
) -> WallPiece:
    return WallPiece(
        points=(
            (start_ratio, bottom_height_meters),
            (end_ratio, bottom_height_meters),
            (end_ratio, top_height_meters),
            (start_ratio, top_height_meters),
        )
    )


def _build_opening_polygon_on_wall(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    wall_height_meters: float,
    opening: WallOpening,
) -> Polygon | None:
    """Project an extruded opening profile into wall ratio/height space."""

    outer_interval = _clip_wall_segment_to_opening(
        start_point=start_point,
        end_point=end_point,
        doorway_opening=opening,
    )
    if outer_interval is None:
        return None

    segment_delta_x = end_point[0] - start_point[0]
    segment_delta_y = end_point[1] - start_point[1]
    relative_start_x = start_point[0] - opening.center_x
    relative_start_y = start_point[1] - opening.center_y
    start_width_position = (
        relative_start_x * opening.width_direction_x
        + relative_start_y * opening.width_direction_y
    )
    width_delta = (
        segment_delta_x * opening.width_direction_x
        + segment_delta_y * opening.width_direction_y
    )
    profile_points = _get_wall_opening_profile_points(opening)
    if abs(width_delta) <= WALL_OPENING_EPSILON:
        return _build_constant_width_opening_polygon(
            profile_points,
            start_width_position,
            opening,
            outer_interval,
            wall_height_meters,
        )

    projected_profile = Polygon(
        [
            (
                (
                    width_meters / PIXEL_TO_METER
                    - start_width_position
                )
                / width_delta,
                opening.bottom_height_meters + height_meters,
            )
            for width_meters, height_meters in profile_points
        ]
    )
    if not projected_profile.is_valid:
        projected_profile = projected_profile.buffer(0)
    if projected_profile.is_empty:
        return None
    interval_start, interval_end = outer_interval
    clip_polygon = shapely.box(
        interval_start,
        0.0,
        interval_end,
        wall_height_meters,
    )
    clipped_profile = projected_profile.intersection(clip_polygon)
    if not isinstance(clipped_profile, Polygon):
        polygon_parts = [
            part
            for part in shapely.get_parts(clipped_profile)
            if isinstance(part, Polygon)
        ]
        if not polygon_parts:
            return None
        clipped_profile = max(polygon_parts, key=lambda part: part.area)
    if clipped_profile.area <= WALL_OPENING_EPSILON:
        return None
    return clipped_profile


def _build_constant_width_opening_polygon(
    profile_points: Sequence[tuple[float, float]],
    width_position_pixels: float,
    opening: WallOpening,
    outer_interval: tuple[float, float],
    wall_height_meters: float,
) -> Polygon | None:
    """Handle the rare wall whose line is perpendicular to opening width."""

    profile_polygon = Polygon(
        [
            (
                width_meters / PIXEL_TO_METER,
                opening.bottom_height_meters + height_meters,
            )
            for width_meters, height_meters in profile_points
        ]
    )
    section = profile_polygon.intersection(
        shapely.LineString(
            (
                (width_position_pixels, 0.0),
                (width_position_pixels, wall_height_meters),
            )
        )
    )
    if section.is_empty:
        return None
    _minimum_width, minimum_height, _maximum_width, maximum_height = section.bounds
    if maximum_height - minimum_height <= WALL_OPENING_EPSILON:
        return None
    return shapely.box(
        outer_interval[0],
        minimum_height,
        outer_interval[1],
        maximum_height,
    )


def _triangulate_visible_wall_geometry(visible_geometry: object) -> list[WallPiece]:
    """Return deterministic triangles that honor every visible boundary."""

    wall_pieces: list[WallPiece] = []
    for polygon_part in shapely.get_parts(visible_geometry):
        if not isinstance(polygon_part, Polygon) or polygon_part.is_empty:
            continue
        triangles = shapely.constrained_delaunay_triangles(polygon_part)
        for triangle in shapely.get_parts(triangles):
            if not isinstance(triangle, Polygon) or triangle.is_empty:
                continue
            if (
                triangle.area <= WALL_OPENING_EPSILON
                or not polygon_part.buffer(WALL_OPENING_EPSILON).covers(triangle)
            ):
                continue
            points = tuple(
                (float(point_x), float(point_y))
                for point_x, point_y in list(triangle.exterior.coords)[:-1]
            )
            if len(points) == 3:
                wall_pieces.append(WallPiece(points=points))
    wall_pieces.sort(
        key=lambda piece: (
            round(piece.start_ratio, 12),
            round(piece.bottom_height_meters, 12),
            round(piece.end_ratio, 12),
            round(piece.top_height_meters, 12),
        )
    )
    return wall_pieces


def _wall_segment_is_parallel_to_opening(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    opening: WallOpening,
) -> bool:
    delta_x = end_point[0] - start_point[0]
    delta_y = end_point[1] - start_point[1]
    length = math.hypot(delta_x, delta_y)
    if length <= WALL_OPENING_EPSILON:
        return False
    alignment = abs(
        delta_x / length * opening.width_direction_x
        + delta_y / length * opening.width_direction_y
    )
    return alignment >= WALL_REVEAL_PARALLEL_COSINE


def _merge_wall_opening_vertical_intervals(
    intervals: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for interval_start, interval_end in sorted(intervals):
        if (
            merged
            and interval_start <= merged[-1][1] + WALL_OPENING_EPSILON
        ):
            merged[-1] = (
                merged[-1][0],
                max(merged[-1][1], interval_end),
            )
            continue
        merged.append((interval_start, interval_end))
    return merged


def _clip_wall_segment_to_opening(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    doorway_opening: WallOpening,
    *,
    half_width_pixels: float | None = None,
) -> tuple[float, float] | None:
    """Return the normalized segment interval inside an oriented doorway box."""
    segment_delta_x = end_point[0] - start_point[0]
    segment_delta_y = end_point[1] - start_point[1]
    relative_start_x = start_point[0] - doorway_opening.center_x
    relative_start_y = start_point[1] - doorway_opening.center_y
    interval_start = 0.0
    interval_end = 1.0
    effective_half_width = (
        doorway_opening.half_width_pixels
        if half_width_pixels is None
        else float(half_width_pixels)
    )
    if effective_half_width <= WALL_OPENING_EPSILON:
        return None

    for axis_x, axis_y, half_extent in (
        (
            doorway_opening.width_direction_x,
            doorway_opening.width_direction_y,
            effective_half_width,
        ),
        (
            doorway_opening.depth_direction_x,
            doorway_opening.depth_direction_y,
            doorway_opening.half_depth_pixels,
        ),
    ):
        start_projection = (
            relative_start_x * axis_x + relative_start_y * axis_y
        )
        delta_projection = segment_delta_x * axis_x + segment_delta_y * axis_y
        if abs(delta_projection) <= WALL_OPENING_EPSILON:
            if abs(start_projection) > half_extent + WALL_OPENING_EPSILON:
                return None
            continue

        first_crossing = (-half_extent - start_projection) / delta_projection
        second_crossing = (half_extent - start_projection) / delta_projection
        interval_start = max(
            interval_start,
            min(first_crossing, second_crossing),
        )
        interval_end = min(
            interval_end,
            max(first_crossing, second_crossing),
        )
        if interval_end - interval_start <= WALL_OPENING_EPSILON:
            return None

    return interval_start, interval_end


# ### Wall opening reveal helpers ###
def _build_level_doorway_reveal_mesh(
    level: LevelData,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
    room_vertex_sets: Sequence[set[int]],
) -> trimesh.Trimesh | None:
    """Build untextured jamb and soffit faces for valid doorway tunnels."""
    doorway_reveals = _build_level_doorway_reveals(
        level=level,
        room_vertex_sets=room_vertex_sets,
    )
    if not doorway_reveals:
        return None

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for doorway_reveal in doorway_reveals:
        _append_doorway_reveal_geometry(
            vertices=vertices,
            faces=faces,
            opening=doorway_reveal.opening,
            reveal_pair=doorway_reveal.reveal_pair,
            base_z_meters=base_z_meters,
            blueprint_size_pixels=blueprint_size_pixels,
        )

    if not vertices or not faces:
        return None

    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def _build_level_window_reveal_mesh(
    level: LevelData,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
) -> trimesh.Trimesh | None:
    """Build untextured jamb, sill, and head faces for window tunnels."""

    reveals = _build_level_window_reveals(level)
    if not reveals:
        return None
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for reveal in reveals:
        _append_window_reveal_geometry(
            vertices=vertices,
            faces=faces,
            opening=reveal.opening,
            reveal_pair=reveal.reveal_pair,
            base_z_meters=base_z_meters,
            blueprint_size_pixels=blueprint_size_pixels,
        )
    if not vertices or not faces:
        return None
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def _build_level_wall_sources(
    level: LevelData,
    room_vertex_sets: Sequence[set[int]],
) -> list[WallSource]:
    """Mirror the level's rendered wall lines without UV subdivision copies."""
    wall_sources_by_key: dict[str, WallSource] = {}
    ignored_vertex_ids = _get_room_center_vertex_ids(level.rooms)
    room_vertex_sets_list = list(room_vertex_sets)
    vertex_lookup = {vertex.id: vertex for vertex in level.vertex_data.vertices}

    for edge in level.vertex_data.edges:
        if (
            edge.start_vertex_id in ignored_vertex_ids
            or edge.end_vertex_id in ignored_vertex_ids
            or _is_edge_inside_any_room(edge, room_vertex_sets_list)
        ):
            continue

        start_vertex = vertex_lookup.get(edge.start_vertex_id)
        end_vertex = vertex_lookup.get(edge.end_vertex_id)
        if start_vertex is None or end_vertex is None:
            continue
        wall_key = (
            f"{min(edge.start_vertex_id, edge.end_vertex_id)}:"
            f"{max(edge.start_vertex_id, edge.end_vertex_id)}"
        )
        surface_id = f"level:{level.index}/wall:{wall_key}"

        _add_level_wall_source(
            wall_sources_by_key,
            WallSource(
                key=f"edge:{wall_key}",
                start_point=(start_vertex.x, start_vertex.y),
                end_point=(end_vertex.x, end_vertex.y),
                height_meters=level.height_meters,
                wall_key=wall_key,
                surface_id=surface_id,
            ),
        )

    for room in level.rooms:
        if (
            not math.isfinite(room.height_meters)
            or room.height_meters <= WALL_OPENING_EPSILON
        ):
            continue

        for wall in build_room_walls(room, level.vertex_data):
            surface_id = (
                f"level:{level.index}/room:{room.center_vertex_id}/"
                f"wall:{wall.key}"
            )
            _add_level_wall_source(
                wall_sources_by_key,
                WallSource(
                    key=f"room:{room.center_vertex_id}:{wall.key}",
                    start_point=wall.start_point,
                    end_point=wall.end_point,
                    height_meters=room.height_meters,
                    wall_key=wall.key,
                    surface_id=surface_id,
                ),
            )

    return list(wall_sources_by_key.values())


def _build_level_doorway_reveals(
    level: LevelData,
    room_vertex_sets: Sequence[set[int]] | None = None,
    *,
    include_single_wall_fallback: bool = False,
) -> list[DoorwayReveal]:
    """Resolve actual wall contacts and one stable owner per doorway tunnel."""

    doorway_openings = _build_wall_openings(level.doorways)
    if not doorway_openings:
        return []
    wall_sources = _build_level_wall_sources(
        level=level,
        room_vertex_sets=(
            _get_room_vertex_sets(level.rooms)
            if room_vertex_sets is None
            else room_vertex_sets
        ),
    )
    reveals: list[DoorwayReveal] = []
    for opening in doorway_openings:
        contacts = _build_doorway_reveal_contacts(wall_sources, opening)
        reveal_pair = _get_doorway_reveal_pair(contacts)
        if reveal_pair is None and include_single_wall_fallback:
            reveal_pair = _build_single_wall_doorway_reveal_pair(
                contacts,
                opening,
            )
        owner_surface_ids = sorted(
            {
                contact.surface_id
                for contact in contacts
                if contact.surface_id is not None
            }
        )
        if reveal_pair is None or not owner_surface_ids:
            continue
        reveals.append(
            DoorwayReveal(
                opening=opening,
                reveal_pair=reveal_pair,
                owner_surface_id=owner_surface_ids[0],
            )
        )
    return reveals


def _build_single_wall_doorway_reveal_pair(
    contacts: Sequence[WallOpeningContact],
    opening: WallOpening,
) -> DoorwayRevealPair | None:
    """Give selectable zero-thickness walls the configured doorway depth."""

    if not contacts:
        return None
    contact = min(
        contacts,
        key=lambda candidate: (
            abs(candidate.depth_position),
            candidate.source_key,
        ),
    )
    low_width = max(
        contact.low_width_position,
        -opening.half_width_pixels,
    )
    high_width = min(
        contact.high_width_position,
        opening.half_width_pixels,
    )
    if high_width - low_width <= WALL_OPENING_EPSILON:
        return None

    def build_contact(depth_position: float, suffix: str) -> WallOpeningContact:
        return WallOpeningContact(
            source_key=f"{contact.source_key}:{suffix}",
            low_width_point=_wall_opening_local_to_image(
                opening,
                low_width,
                depth_position,
            ),
            high_width_point=_wall_opening_local_to_image(
                opening,
                high_width,
                depth_position,
            ),
            low_width_position=low_width,
            high_width_position=high_width,
            depth_position=depth_position,
            opening_bottom_height_meters=(
                contact.opening_bottom_height_meters
            ),
            opening_top_height_meters=contact.opening_top_height_meters,
            wall_key=contact.wall_key,
            surface_id=contact.surface_id,
        )

    return DoorwayRevealPair(
        first_contact=build_contact(-opening.half_depth_pixels, "negative"),
        second_contact=build_contact(opening.half_depth_pixels, "positive"),
        low_width_position=low_width,
        high_width_position=high_width,
    )


def _add_level_wall_source(
    wall_sources_by_key: dict[str, WallSource],
    wall_source: WallSource,
) -> None:
    if (
        _get_2d_point_distance(
            wall_source.start_point,
            wall_source.end_point,
        )
        <= WALL_OPENING_EPSILON
    ):
        return

    existing_source = wall_sources_by_key.get(wall_source.key)
    if existing_source is None:
        wall_sources_by_key[wall_source.key] = wall_source
        return

    if wall_source.height_meters < existing_source.height_meters:
        wall_sources_by_key[wall_source.key] = wall_source


def _build_doorway_reveal_contacts(
    wall_sources: Sequence[WallSource],
    doorway_opening: WallOpening,
) -> list[WallOpeningContact]:
    contacts: list[WallOpeningContact] = []
    for wall_source in wall_sources:
        if not _is_wall_source_parallel_to_doorway_width(
            wall_source,
            doorway_opening,
        ):
            continue

        doorway_interval = _clip_wall_segment_to_opening(
            start_point=wall_source.start_point,
            end_point=wall_source.end_point,
            doorway_opening=doorway_opening,
        )
        if doorway_interval is None:
            continue

        interval_start, interval_end = doorway_interval
        contact_start = _interpolate_2d_point(
            wall_source.start_point,
            wall_source.end_point,
            interval_start,
        )
        contact_end = _interpolate_2d_point(
            wall_source.start_point,
            wall_source.end_point,
            interval_end,
        )
        start_width_position = _get_doorway_width_position(
            contact_start,
            doorway_opening,
        )
        end_width_position = _get_doorway_width_position(
            contact_end,
            doorway_opening,
        )
        if abs(end_width_position - start_width_position) <= WALL_OPENING_EPSILON:
            continue

        if start_width_position <= end_width_position:
            low_width_point = contact_start
            high_width_point = contact_end
            low_width_position = start_width_position
            high_width_position = end_width_position
        else:
            low_width_point = contact_end
            high_width_point = contact_start
            low_width_position = end_width_position
            high_width_position = start_width_position

        contact_midpoint = _interpolate_2d_point(
            low_width_point,
            high_width_point,
            0.5,
        )
        opening_bottom = min(
            max(0.0, doorway_opening.bottom_height_meters),
            wall_source.height_meters,
        )
        opening_top = min(
            max(opening_bottom, doorway_opening.top_height_meters),
            wall_source.height_meters,
        )
        if opening_top - opening_bottom <= WALL_OPENING_EPSILON:
            continue
        contacts.append(
            WallOpeningContact(
                source_key=wall_source.key,
                low_width_point=low_width_point,
                high_width_point=high_width_point,
                low_width_position=low_width_position,
                high_width_position=high_width_position,
                depth_position=_get_doorway_depth_position(
                    contact_midpoint,
                    doorway_opening,
                ),
                opening_bottom_height_meters=opening_bottom,
                opening_top_height_meters=opening_top,
                wall_key=wall_source.wall_key,
                surface_id=wall_source.surface_id,
            )
        )

    return contacts


def _is_wall_source_parallel_to_doorway_width(
    wall_source: WallSource,
    doorway_opening: WallOpening,
) -> bool:
    wall_delta_x = wall_source.end_point[0] - wall_source.start_point[0]
    wall_delta_y = wall_source.end_point[1] - wall_source.start_point[1]
    wall_length = math.hypot(wall_delta_x, wall_delta_y)
    if wall_length <= WALL_OPENING_EPSILON:
        return False

    width_alignment = abs(
        (
            wall_delta_x / wall_length * doorway_opening.width_direction_x
        )
        + (
            wall_delta_y / wall_length * doorway_opening.width_direction_y
        )
    )
    return width_alignment >= WALL_REVEAL_PARALLEL_COSINE


def _get_doorway_width_position(
    point: tuple[float, float],
    doorway_opening: WallOpening,
) -> float:
    return (
        (point[0] - doorway_opening.center_x)
        * doorway_opening.width_direction_x
        + (point[1] - doorway_opening.center_y)
        * doorway_opening.width_direction_y
    )


def _get_doorway_depth_position(
    point: tuple[float, float],
    doorway_opening: WallOpening,
) -> float:
    return (
        (point[0] - doorway_opening.center_x)
        * doorway_opening.depth_direction_x
        + (point[1] - doorway_opening.center_y)
        * doorway_opening.depth_direction_y
    )


def _get_doorway_reveal_pair(
    doorway_contacts: Sequence[WallOpeningContact],
) -> DoorwayRevealPair | None:
    negative_contacts = [
        contact
        for contact in doorway_contacts
        if contact.depth_position < -WALL_OPENING_EPSILON
    ]
    positive_contacts = [
        contact
        for contact in doorway_contacts
        if contact.depth_position > WALL_OPENING_EPSILON
    ]
    best_pair: DoorwayRevealPair | None = None
    best_score: tuple[float, float] | None = None

    for first_contact in negative_contacts:
        for second_contact in positive_contacts:
            if first_contact.source_key == second_contact.source_key:
                continue

            low_width_position = max(
                first_contact.low_width_position,
                second_contact.low_width_position,
            )
            high_width_position = min(
                first_contact.high_width_position,
                second_contact.high_width_position,
            )
            width_overlap = high_width_position - low_width_position
            if width_overlap <= WALL_OPENING_EPSILON:
                continue

            depth_separation = (
                second_contact.depth_position - first_contact.depth_position
            )
            score = (depth_separation, width_overlap)
            if best_score is not None and score <= best_score:
                continue

            best_score = score
            best_pair = DoorwayRevealPair(
                first_contact=first_contact,
                second_contact=second_contact,
                low_width_position=low_width_position,
                high_width_position=high_width_position,
            )

    return best_pair


def _build_level_window_reveals(level: LevelData) -> list[WindowReveal]:
    """Pair each selected wall with its nearest cut outward wall."""

    openings = _build_window_openings(level)
    if not openings:
        return []
    wall_sources = _build_level_wall_sources(
        level,
        _get_room_vertex_sets(level.rooms),
    )
    reveals: list[WindowReveal] = []
    for opening in openings:
        if (
            opening.target_wall_key is None
            or opening.target_surface_id is None
        ):
            continue
        contacts = _build_doorway_reveal_contacts(
            wall_sources,
            opening,
        )
        reveal_pair = _get_window_reveal_pair(contacts, opening)
        if reveal_pair is None:
            continue
        reveals.append(
            WindowReveal(
                opening=opening,
                reveal_pair=reveal_pair,
                owner_surface_id=opening.target_surface_id,
            )
        )
    return reveals


def _get_window_reveal_pair(
    contacts: Sequence[WallOpeningContact],
    opening: WallOpening,
) -> DoorwayRevealPair | None:
    """Choose the selected wall and nearest distinct outward contact."""

    target_contacts = [
        contact
        for contact in contacts
        if contact.surface_id == opening.target_surface_id
    ]
    if not target_contacts:
        return None
    target_contact = min(
        target_contacts,
        key=lambda contact: (
            contact.depth_position,
            contact.source_key,
        ),
    )
    outward_contacts = _merge_contiguous_opening_contacts(
        [
            contact
            for contact in contacts
            if contact.source_key != target_contact.source_key
            and contact.depth_position
            > target_contact.depth_position + WALL_OPENING_EPSILON
        ],
        opening,
    )

    best_pair: DoorwayRevealPair | None = None
    best_score: tuple[float, float, str] | None = None
    for outward_contact in outward_contacts:
        low_width_position = max(
            target_contact.low_width_position,
            outward_contact.low_width_position,
        )
        high_width_position = min(
            target_contact.high_width_position,
            outward_contact.high_width_position,
        )
        width_overlap = high_width_position - low_width_position
        opening_bottom = max(
            target_contact.opening_bottom_height_meters,
            outward_contact.opening_bottom_height_meters,
        )
        opening_top = min(
            target_contact.opening_top_height_meters,
            outward_contact.opening_top_height_meters,
        )
        if (
            width_overlap <= WALL_OPENING_EPSILON
            or opening_top - opening_bottom <= WALL_OPENING_EPSILON
        ):
            continue
        depth_separation = (
            outward_contact.depth_position - target_contact.depth_position
        )
        score = (
            depth_separation,
            -width_overlap,
            outward_contact.source_key,
        )
        if best_score is not None and score >= best_score:
            continue
        best_score = score
        best_pair = DoorwayRevealPair(
            first_contact=target_contact,
            second_contact=outward_contact,
            low_width_position=low_width_position,
            high_width_position=high_width_position,
        )
    return best_pair


def _merge_contiguous_opening_contacts(
    contacts: Sequence[WallOpeningContact],
    opening: WallOpening,
) -> list[WallOpeningContact]:
    """Join collinear outer-wall segments before choosing a reveal pair."""

    merged: list[WallOpeningContact] = []
    ordered_contacts = sorted(
        contacts,
        key=lambda contact: (
            contact.depth_position,
            contact.low_width_position,
            contact.high_width_position,
            contact.source_key,
        ),
    )
    for contact in ordered_contacts:
        if not merged:
            merged.append(contact)
            continue
        previous = merged[-1]
        vertical_bottom = max(
            previous.opening_bottom_height_meters,
            contact.opening_bottom_height_meters,
        )
        vertical_top = min(
            previous.opening_top_height_meters,
            contact.opening_top_height_meters,
        )
        same_depth = abs(
            previous.depth_position - contact.depth_position
        ) <= WINDOW_COPLANAR_DEPTH_PIXELS
        contiguous_width = (
            contact.low_width_position
            <= previous.high_width_position + WALL_OPENING_EPSILON
        )
        if (
            not same_depth
            or not contiguous_width
            or vertical_top - vertical_bottom <= WALL_OPENING_EPSILON
        ):
            merged.append(contact)
            continue

        low_width = min(
            previous.low_width_position,
            contact.low_width_position,
        )
        high_width = max(
            previous.high_width_position,
            contact.high_width_position,
        )
        depth_position = (
            previous.depth_position + contact.depth_position
        ) / 2.0
        merged[-1] = WallOpeningContact(
            source_key=f"{previous.source_key}+{contact.source_key}",
            low_width_point=_wall_opening_local_to_image(
                opening,
                low_width,
                depth_position,
            ),
            high_width_point=_wall_opening_local_to_image(
                opening,
                high_width,
                depth_position,
            ),
            low_width_position=low_width,
            high_width_position=high_width,
            depth_position=depth_position,
            opening_bottom_height_meters=vertical_bottom,
            opening_top_height_meters=vertical_top,
            wall_key=(
                previous.wall_key
                if previous.wall_key == contact.wall_key
                else None
            ),
            surface_id=(
                previous.surface_id
                if previous.surface_id == contact.surface_id
                else None
            ),
        )
    return merged


def _wall_opening_local_to_image(
    opening: WallOpening,
    width_position: float,
    depth_position: float,
) -> tuple[float, float]:
    return (
        opening.center_x
        + opening.width_direction_x * width_position
        + opening.depth_direction_x * depth_position,
        opening.center_y
        + opening.width_direction_y * width_position
        + opening.depth_direction_y * depth_position,
    )


def _append_doorway_reveal_geometry(
    vertices: list[list[float]],
    faces: list[list[int]],
    opening: WallOpening,
    reveal_pair: DoorwayRevealPair,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
) -> None:
    _append_wall_opening_reveal_geometry(
        vertices=vertices,
        faces=faces,
        opening=opening,
        reveal_pair=reveal_pair,
        base_z_meters=base_z_meters,
        blueprint_size_pixels=blueprint_size_pixels,
        include_sill=(
            opening.bottom_height_meters > WALL_OPENING_EPSILON
        ),
    )


def _append_window_reveal_geometry(
    vertices: list[list[float]],
    faces: list[list[int]],
    opening: WallOpening,
    reveal_pair: DoorwayRevealPair,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
) -> None:
    _append_wall_opening_reveal_geometry(
        vertices=vertices,
        faces=faces,
        opening=opening,
        reveal_pair=reveal_pair,
        base_z_meters=base_z_meters,
        blueprint_size_pixels=blueprint_size_pixels,
        include_sill=True,
    )


def _append_wall_opening_reveal_geometry(
    vertices: list[list[float]],
    faces: list[list[int]],
    opening: WallOpening,
    reveal_pair: DoorwayRevealPair,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
    *,
    include_sill: bool,
) -> None:
    for image_quad in _build_wall_opening_reveal_quads(
        reveal_pair,
        opening,
        include_sill=include_sill,
    ):
        world_quad = tuple(
            (
                float(world_xy[0]),
                float(world_xy[1]),
                base_z_meters + local_point[2],
            )
            for local_point in image_quad
            for world_xy in (
                _point_to_world_xy(
                    (local_point[0], local_point[1]),
                    blueprint_size_pixels,
                ),
            )
        )
        _append_double_sided_quad(
            vertices=vertices,
            faces=faces,
            corners=world_quad,
        )


def _build_wall_opening_reveal_quads(
    reveal_pair: DoorwayRevealPair,
    opening: WallOpening,
    *,
    include_sill: bool,
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    contact_bottom_height = max(
        reveal_pair.first_contact.opening_bottom_height_meters,
        reveal_pair.second_contact.opening_bottom_height_meters,
    )
    contact_top_height = min(
        reveal_pair.first_contact.opening_top_height_meters,
        reveal_pair.second_contact.opening_top_height_meters,
    )
    if (
        contact_top_height - contact_bottom_height
        <= WALL_OPENING_EPSILON
    ):
        return ()

    clipped_profiles = _clip_opening_profiles_to_reveal(
        opening,
        reveal_pair,
        contact_bottom_height,
        contact_top_height,
    )
    if not clipped_profiles:
        return ()
    quads: list[tuple[tuple[float, float, float], ...]] = []
    opening_floor_height = max(
        contact_bottom_height,
        opening.bottom_height_meters,
    )
    for profile_points in clipped_profiles:
        for first_profile_point, second_profile_point in zip(
            profile_points,
            profile_points[1:],
        ):
            first_width, first_height = first_profile_point
            second_width, second_height = second_profile_point
            if (
                not include_sill
                and abs(first_height - opening_floor_height)
                <= WALL_OPENING_EPSILON
                and abs(second_height - opening_floor_height)
                <= WALL_OPENING_EPSILON
            ):
                continue
            first_contact_start = _interpolate_wall_opening_contact(
                reveal_pair.first_contact,
                first_width,
            )
            second_contact_start = _interpolate_wall_opening_contact(
                reveal_pair.second_contact,
                first_width,
            )
            second_contact_end = _interpolate_wall_opening_contact(
                reveal_pair.second_contact,
                second_width,
            )
            first_contact_end = _interpolate_wall_opening_contact(
                reveal_pair.first_contact,
                second_width,
            )
            image_points = (
                first_contact_start,
                second_contact_start,
                second_contact_end,
                first_contact_end,
            )
            if not all(
                math.isfinite(coordinate)
                for point in image_points
                for coordinate in point
            ):
                continue
            quads.append(
                (
                    (*first_contact_start, first_height),
                    (*first_contact_end, second_height),
                    (*second_contact_end, second_height),
                    (*second_contact_start, first_height),
                )
            )
    return tuple(quads)


def _clip_opening_profiles_to_reveal(
    opening: WallOpening,
    reveal_pair: DoorwayRevealPair,
    contact_bottom_height: float,
    contact_top_height: float,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Clip one smooth opening profile to the two real wall contacts."""

    profile_polygon = Polygon(
        [
            (
                width_meters / PIXEL_TO_METER,
                opening.bottom_height_meters + height_meters,
            )
            for width_meters, height_meters in (
                _get_wall_opening_profile_points(opening)
            )
        ]
    )
    clip_polygon = shapely.box(
        reveal_pair.low_width_position,
        contact_bottom_height,
        reveal_pair.high_width_position,
        contact_top_height,
    )
    clipped_profile = shapely.orient_polygons(
        profile_polygon.intersection(clip_polygon),
        exterior_cw=False,
    )
    profiles: list[tuple[tuple[float, float], ...]] = []
    for polygon_part in shapely.get_parts(clipped_profile):
        if not isinstance(polygon_part, Polygon) or polygon_part.is_empty:
            continue
        points = tuple(
            (float(point_x), float(point_y))
            for point_x, point_y in polygon_part.exterior.coords
        )
        if len(points) >= 4:
            profiles.append(points)
    return tuple(profiles)


def _interpolate_wall_opening_contact(
    contact: WallOpeningContact,
    width_position: float,
) -> tuple[float, float]:
    width_span = contact.high_width_position - contact.low_width_position
    if width_span <= WALL_OPENING_EPSILON:
        return contact.low_width_point

    return _interpolate_2d_point(
        contact.low_width_point,
        contact.high_width_point,
        (width_position - contact.low_width_position) / width_span,
    )


def _append_double_sided_quad(
    vertices: list[list[float]],
    faces: list[list[int]],
    corners: Sequence[tuple[float, float, float]],
) -> None:
    if len(corners) != 4:
        return

    vertex_offset = len(vertices)
    vertices.extend([list(corner) for corner in corners])
    faces.extend(
        (
            [vertex_offset + 0, vertex_offset + 1, vertex_offset + 2],
            [vertex_offset + 0, vertex_offset + 2, vertex_offset + 3],
            [vertex_offset + 2, vertex_offset + 1, vertex_offset + 0],
            [vertex_offset + 3, vertex_offset + 2, vertex_offset + 0],
        )
    )


def _get_opening_interval_breakpoints(
    doorway_intervals: Sequence[Sequence[float]],
) -> list[float]:
    raw_breakpoints = [0.0, 1.0]
    for interval in doorway_intervals:
        interval_start, interval_end = interval[:2]
        raw_breakpoints.extend(
            (
                min(max(interval_start, 0.0), 1.0),
                min(max(interval_end, 0.0), 1.0),
            )
        )

    breakpoints: list[float] = []
    for breakpoint in sorted(raw_breakpoints):
        if (
            not breakpoints
            or breakpoint - breakpoints[-1] > WALL_OPENING_EPSILON
        ):
            breakpoints.append(breakpoint)

    return breakpoints


def _append_or_merge_wall_piece(
    wall_pieces: list[WallPiece],
    wall_piece: WallPiece,
) -> None:
    if (
        wall_pieces
        and abs(wall_pieces[-1].end_ratio - wall_piece.start_ratio)
        <= WALL_OPENING_EPSILON
        and math.isclose(
            wall_pieces[-1].bottom_height_meters,
            wall_piece.bottom_height_meters,
            abs_tol=WALL_OPENING_EPSILON,
        )
        and math.isclose(
            wall_pieces[-1].top_height_meters,
            wall_piece.top_height_meters,
            abs_tol=WALL_OPENING_EPSILON,
        )
    ):
        previous_piece = wall_pieces[-1]
        wall_pieces[-1] = _build_rectangular_wall_piece(
            previous_piece.start_ratio,
            wall_piece.end_ratio,
            previous_piece.bottom_height_meters,
            previous_piece.top_height_meters,
        )
        return

    wall_pieces.append(wall_piece)


def _build_wall_piece_vertices_from_points(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    wall_piece: WallPiece,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
) -> list[list[float]] | None:
    wall_vertices: list[list[float]] = []
    for wall_ratio, height_meters in wall_piece.points:
        image_point = _interpolate_2d_point(
            start_point,
            end_point,
            wall_ratio,
        )
        world_xy = _point_to_world_xy(image_point, blueprint_size_pixels)
        wall_vertices.append(
            [
                float(world_xy[0]),
                float(world_xy[1]),
                base_z_meters + height_meters,
            ]
        )
    if len(wall_vertices) < 3:
        return None
    return wall_vertices


def _build_wall_vertices_from_points(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    wall_height_meters: float,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
) -> list[list[float]] | None:
    start_xy = _point_to_world_xy(start_point, blueprint_size_pixels)
    end_xy = _point_to_world_xy(end_point, blueprint_size_pixels)
    if float(np.linalg.norm(end_xy - start_xy)) < 1e-6:
        return None

    bottom_z = base_z_meters
    top_z = base_z_meters + wall_height_meters
    return [
        [float(start_xy[0]), float(start_xy[1]), bottom_z],
        [float(end_xy[0]), float(end_xy[1]), bottom_z],
        [float(end_xy[0]), float(end_xy[1]), top_z],
        [float(start_xy[0]), float(start_xy[1]), top_z],
    ]


def _get_wall_segment_points(
    wall: RoomWall,
    placement: UvWallPlacement,
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        _interpolate_2d_point(
            wall.start_point,
            wall.end_point,
            placement.source_start_ratio,
        ),
        _interpolate_2d_point(
            wall.start_point,
            wall.end_point,
            placement.source_end_ratio,
        ),
    )


def _interpolate_2d_point(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    ratio: float,
) -> tuple[float, float]:
    safe_ratio = min(max(0.0, float(ratio)), 1.0)
    return (
        start_point[0] + (end_point[0] - start_point[0]) * safe_ratio,
        start_point[1] + (end_point[1] - start_point[1]) * safe_ratio,
    )


def _get_2d_point_distance(
    first_point: tuple[float, float],
    second_point: tuple[float, float],
) -> float:
    return math.hypot(
        second_point[0] - first_point[0],
        second_point[1] - first_point[1],
    )


def _build_wall_faces(
    vertex_offset: int,
    vertex_count: int = 4,
    *,
    double_sided: bool = True,
) -> list[list[int]]:
    faces = [
        [vertex_offset, vertex_offset + point_index, vertex_offset + point_index + 1]
        for point_index in range(1, vertex_count - 1)
    ]
    if double_sided:
        faces.extend(list(reversed(face)) for face in tuple(faces))
    return faces


def _build_wall_uv_coordinates(
    room: RoomData,
    placement: UvWallPlacement,
) -> list[tuple[float, float]]:
    texture_width = max(1.0, float(room.uv_map_width))
    texture_height = max(1.0, float(room.uv_map_height))
    top_left, top_right, bottom_right, bottom_left = get_rotated_uv_corners(placement)
    return [
        _normalize_uv_point(bottom_left, texture_width, texture_height),
        _normalize_uv_point(bottom_right, texture_width, texture_height),
        _normalize_uv_point(top_right, texture_width, texture_height),
        _normalize_uv_point(top_left, texture_width, texture_height),
    ]


def _build_wall_piece_uv_coordinates(
    room: RoomData,
    placement: UvWallPlacement,
    wall_piece: WallPiece,
    wall_height_meters: float,
) -> list[tuple[float, float]]:
    if wall_height_meters <= WALL_OPENING_EPSILON:
        return _build_hidden_wall_uv_coordinates(len(wall_piece.points))

    full_uv_coordinates = _build_wall_uv_coordinates(room, placement)
    return [
        _interpolate_wall_uv_coordinate(
            full_uv_coordinates,
            wall_ratio,
            min(max(height_meters / wall_height_meters, 0.0), 1.0),
        )
        for wall_ratio, height_meters in wall_piece.points
    ]


def _interpolate_wall_uv_coordinate(
    uv_coordinates: Sequence[tuple[float, float]],
    horizontal_ratio: float,
    vertical_ratio: float,
) -> tuple[float, float]:
    bottom_left, bottom_right, top_right, top_left = uv_coordinates
    safe_horizontal_ratio = min(max(horizontal_ratio, 0.0), 1.0)
    safe_vertical_ratio = min(max(vertical_ratio, 0.0), 1.0)
    bottom_point = _interpolate_2d_point(
        bottom_left,
        bottom_right,
        safe_horizontal_ratio,
    )
    top_point = _interpolate_2d_point(
        top_left,
        top_right,
        safe_horizontal_ratio,
    )
    return _interpolate_2d_point(
        bottom_point,
        top_point,
        safe_vertical_ratio,
    )


def _normalize_uv_point(
    uv_point: tuple[float, float],
    texture_width: float,
    texture_height: float,
) -> tuple[float, float]:
    return (
        uv_point[0] / texture_width,
        1.0 - (uv_point[1] / texture_height),
    )


def _build_hidden_wall_uv_coordinates(
    vertex_count: int = 4,
) -> list[tuple[float, float]]:
    return [(0.0, 0.0)] * max(0, vertex_count)


# ### Preview helpers ###
def _build_preview_textured_walls(
    levels: Sequence[LevelData],
    blueprint_size_pixels: tuple[float, float] | None,
) -> list[PreviewTexturedWall]:
    sorted_levels = sorted(levels, key=lambda level: level.index)
    level_lookup = {level.index: level for level in sorted_levels}
    preview_walls: list[PreviewTexturedWall] = []

    for level in sorted_levels:
        if not level.include_in_export:
            continue

        level_blueprint_size = level.image_size_pixels or blueprint_size_pixels
        preview_walls.extend(
            _build_level_preview_textured_walls(
                level=level,
                base_z_meters=_get_level_base_z(level_lookup, level.index),
                blueprint_size_pixels=level_blueprint_size,
                source_transform=_build_level_source_transform(
                    level,
                    level_blueprint_size,
                ),
            )
        )

    return preview_walls


def _build_level_preview_textured_walls(
    level: LevelData,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
    source_transform: np.ndarray,
) -> list[PreviewTexturedWall]:
    preview_walls: list[PreviewTexturedWall] = []
    doorway_openings = _build_level_wall_openings(level)
    for room_index, room in enumerate(level.rooms):
        layout = build_uv_wall_layout(
            room=room,
            vertex_data=level.vertex_data,
            wall_height_meters=room.height_meters,
        )
        if not layout.placements:
            continue

        for placement in layout.placements:
            segment_start_point, segment_end_point = _get_wall_segment_points(
                wall=placement.wall,
                placement=placement,
            )
            preview_texture = _build_wall_preview_texture(room, placement)
            wall_pieces = _build_visible_wall_pieces(
                start_point=segment_start_point,
                end_point=segment_end_point,
                wall_height_meters=room.height_meters,
                doorway_openings=doorway_openings,
                wall_key=placement.wall.key,
            )
            if not wall_pieces:
                continue
            if all(len(wall_piece.points) == 4 for wall_piece in wall_pieces):
                for wall_piece in wall_pieces:
                    preview_start_point = _interpolate_2d_point(
                        segment_start_point,
                        segment_end_point,
                        wall_piece.start_ratio,
                    )
                    preview_end_point = _interpolate_2d_point(
                        segment_start_point,
                        segment_end_point,
                        wall_piece.end_ratio,
                    )
                    piece_start_xy = _point_to_world_xy(
                        preview_start_point,
                        blueprint_size_pixels,
                    )
                    piece_end_xy = _point_to_world_xy(
                        preview_end_point,
                        blueprint_size_pixels,
                    )
                    preview_walls.append(
                        PreviewTexturedWall(
                            level_index=level.index,
                            room_index=room_index,
                            wall_key=placement.wall.key,
                            start_point=_transform_source_point(
                                (
                                    float(piece_start_xy[0]),
                                    float(piece_start_xy[1]),
                                    base_z_meters
                                    + wall_piece.bottom_height_meters,
                                ),
                                source_transform,
                            ),
                            end_point=_transform_source_point(
                                (
                                    float(piece_end_xy[0]),
                                    float(piece_end_xy[1]),
                                    base_z_meters
                                    + wall_piece.bottom_height_meters,
                                ),
                                source_transform,
                            ),
                            height_meters=(
                                wall_piece.top_height_meters
                                - wall_piece.bottom_height_meters
                            ),
                            texture_rgba=_crop_wall_preview_texture(
                                preview_texture,
                                wall_piece,
                                room.height_meters,
                            ),
                        )
                    )
                continue
            start_xy = _point_to_world_xy(
                segment_start_point,
                blueprint_size_pixels,
            )
            end_xy = _point_to_world_xy(
                segment_end_point,
                blueprint_size_pixels,
            )
            preview_walls.append(
                PreviewTexturedWall(
                    level_index=level.index,
                    room_index=room_index,
                    wall_key=placement.wall.key,
                    start_point=_transform_source_point(
                        (float(start_xy[0]), float(start_xy[1]), base_z_meters),
                        source_transform,
                    ),
                    end_point=_transform_source_point(
                        (float(end_xy[0]), float(end_xy[1]), base_z_meters),
                        source_transform,
                    ),
                    height_meters=room.height_meters,
                    texture_rgba=_mask_wall_preview_texture(
                        preview_texture,
                        wall_pieces,
                        room.height_meters,
                    ),
                )
            )

    return preview_walls


# ### Room helpers ###
def _get_room_vertex_sets(rooms: list[RoomData]) -> list[set[int]]:
    return [set(room.vertex_ids) for room in rooms]


def _is_edge_inside_any_room(
    edge: Edge,
    room_vertex_sets: list[set[int]],
) -> bool:
    return any(
        edge.start_vertex_id in room_vertex_set
        and edge.end_vertex_id in room_vertex_set
        for room_vertex_set in room_vertex_sets
    )


def _get_room_center_vertex_ids(rooms: list[RoomData]) -> set[int]:
    return {room.center_vertex_id for room in rooms}


# ### Level transform helpers ###
def _get_valid_level_scale(level: LevelData) -> float:
    raw_scale = level.scale
    if isinstance(raw_scale, bool):
        raise ValueError(
            f"Level {level.index} scale must be a finite number greater than zero."
        )

    try:
        scale = float(raw_scale)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Level {level.index} scale must be a finite number greater than zero."
        ) from error

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"Level {level.index} scale must be a finite number greater than zero."
        )

    return scale


def _get_valid_level_offset(level: LevelData, axis: str) -> float:
    raw_offset = getattr(level, f"offset_{axis}_meters")
    if isinstance(raw_offset, bool):
        raise ValueError(
            f"Level {level.index} {axis} offset must be a finite number."
        )

    try:
        offset = float(raw_offset)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Level {level.index} {axis} offset must be a finite number."
        ) from error

    if not math.isfinite(offset):
        raise ValueError(
            f"Level {level.index} {axis} offset must be a finite number."
        )

    return offset


def _build_level_source_transform(
    level: LevelData,
    blueprint_size_pixels: tuple[float, float] | None,
) -> np.ndarray:
    scale = _get_valid_level_scale(level)
    offset_x_meters = _get_valid_level_offset(level, "x")
    offset_y_meters = _get_valid_level_offset(level, "y")
    transform = np.eye(4, dtype=float)
    if scale != 1.0 and level.vertex_data.vertices:
        level_points = np.asarray(
            [
                _vertex_to_world_xy(vertex, blueprint_size_pixels)
                for vertex in level.vertex_data.vertices
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(level_points)):
            raise ValueError(
                f"Level {level.index} vertices must have finite positions."
            )

        minimum_point = np.min(level_points, axis=0)
        maximum_point = np.max(level_points, axis=0)
        pivot_point = (minimum_point + maximum_point) / 2.0
        transform[0, 0] = scale
        transform[1, 1] = scale
        transform[0, 3] = float(pivot_point[0] * (1.0 - scale))
        transform[1, 3] = float(pivot_point[1] * (1.0 - scale))

    transform[0, 3] += offset_x_meters
    transform[1, 3] += offset_y_meters
    return transform


def _transform_source_point(
    point: tuple[float, float, float],
    source_transform: np.ndarray,
) -> tuple[float, float, float]:
    source_point = np.array([*point, 1.0], dtype=float)
    transformed_point = _get_valid_source_transform(source_transform) @ source_point
    return (
        float(transformed_point[0]),
        float(transformed_point[1]),
        float(transformed_point[2]),
    )


# ### Level helpers ###
def _get_level_base_z(
    level_lookup: dict[int, LevelData],
    level_index: int,
) -> float:
    if level_index >= GROUND_LEVEL_INDEX:
        return sum(
            level_lookup[index].height_meters
            for index in range(GROUND_LEVEL_INDEX, level_index)
            if index in level_lookup
        )

    return -sum(
        level_lookup[index].height_meters
        for index in range(level_index, GROUND_LEVEL_INDEX)
        if index in level_lookup
    )


def _get_level_object_name(level: LevelData) -> str:
    return level.display_name.lower().replace(" ", "_")


def _get_level_floor_object_name(level: LevelData) -> str:
    return f"{_get_level_object_name(level)}_floor"


def _get_level_doorway_reveal_object_name(level: LevelData) -> str:
    return f"{_get_level_object_name(level)}_doorway_reveals"


def _get_level_window_reveal_object_name(level: LevelData) -> str:
    return f"{_get_level_object_name(level)}_window_reveals"


def _get_room_object_name(
    level: LevelData,
    room: RoomData,
    room_index: int,
) -> str:
    return f"{_get_level_object_name(level)}_{_slugify_name(room.name)}_{room_index + 1}"


def _get_room_material_name(
    level: LevelData,
    room: RoomData,
    room_index: int,
) -> str:
    return f"{level.display_name} {room.name or 'Room'} {room_index + 1}"


# ### Plain wall helpers ###
def _build_wall_mesh(
    edge: Edge,
    vertex_lookup: dict[int, Vertex],
    wall_height_meters: float,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
    doorway_openings: Sequence[WallOpening] = (),
) -> trimesh.Trimesh | None:
    start_vertex = vertex_lookup.get(edge.start_vertex_id)
    end_vertex = vertex_lookup.get(edge.end_vertex_id)
    if start_vertex is None or end_vertex is None:
        return None

    start_point = (start_vertex.x, start_vertex.y)
    end_point = (end_vertex.x, end_vertex.y)
    wall_key = (
        f"{min(edge.start_vertex_id, edge.end_vertex_id)}:"
        f"{max(edge.start_vertex_id, edge.end_vertex_id)}"
    )
    wall_pieces = _build_visible_wall_pieces(
        start_point=start_point,
        end_point=end_point,
        wall_height_meters=wall_height_meters,
        doorway_openings=doorway_openings,
        wall_key=wall_key,
    )
    if not wall_pieces:
        return None

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for wall_piece in wall_pieces:
        wall_vertices = _build_wall_piece_vertices_from_points(
            start_point=start_point,
            end_point=end_point,
            wall_piece=wall_piece,
            base_z_meters=base_z_meters,
            blueprint_size_pixels=blueprint_size_pixels,
        )
        if wall_vertices is None:
            continue

        vertex_offset = len(vertices)
        vertices.extend(wall_vertices)
        faces.extend(_build_wall_faces(vertex_offset, len(wall_vertices)))

    if not vertices or not faces:
        return None

    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


# ### Coordinate helpers ###
def _vertex_to_world_xy(
    vertex: Vertex,
    blueprint_size_pixels: tuple[float, float] | None,
) -> np.ndarray:
    return _point_to_world_xy((vertex.x, vertex.y), blueprint_size_pixels)


def _point_to_world_xy(
    point: tuple[float, float],
    blueprint_size_pixels: tuple[float, float] | None,
) -> np.ndarray:
    point_x, point_y = point
    if blueprint_size_pixels is None:
        centered_x = point_x
        centered_y = point_y
    else:
        blueprint_width, blueprint_height = blueprint_size_pixels
        centered_x = point_x - blueprint_width / 2.0
        centered_y = point_y - blueprint_height / 2.0

    return np.array(
        [
            centered_x * PIXEL_TO_METER,
            -centered_y * PIXEL_TO_METER,
        ],
        dtype=float,
    )


# ### Mesh helpers ###
def _combine_mesh_geometry(meshes: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    for mesh in meshes:
        mesh_vertices = np.asarray(mesh.vertices, dtype=float)
        mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
        if mesh_vertices.size == 0 or mesh_faces.size == 0:
            continue

        vertices.append(mesh_vertices)
        faces.append(mesh_faces + vertex_offset)
        vertex_offset += len(mesh_vertices)

    if not vertices or not faces:
        return trimesh.Trimesh(process=False)

    return trimesh.Trimesh(
        vertices=np.vstack(vertices),
        faces=np.vstack(faces),
        process=False,
    )


# ### Text helpers ###
def _slugify_name(name: str) -> str:
    normalized_name = "".join(
        character.lower() if character.isalnum() else "_"
        for character in name.strip()
    ).strip("_")
    return normalized_name or "room"
