# ### Imports ###
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field

# ### Constants ###
DEFAULT_LEVEL_HEIGHT_METERS = 3.0
DEFAULT_LEVEL_SCALE = 1.0
MIN_LEVEL_SCALE = 0.01
MAX_LEVEL_SCALE = 20.0
DEFAULT_CANVAS_LEVEL_SCALE = 1.0
MIN_CANVAS_LEVEL_SCALE = 0.01
MAX_CANVAS_LEVEL_SCALE = 20.0
DEFAULT_CANVAS_OFFSET_PIXELS = 0.0
MIN_CANVAS_OFFSET_PIXELS = -100000.0
MAX_CANVAS_OFFSET_PIXELS = 100000.0
DEFAULT_LEVEL_OFFSET_METERS = 0.0
MIN_LEVEL_OFFSET_METERS = -10000.0
MAX_LEVEL_OFFSET_METERS = 10000.0
DEFAULT_FLOOR_THICKNESS_METERS = 0.3
MIN_FLOOR_THICKNESS_METERS = 0.01
MAX_FLOOR_THICKNESS_METERS = 10.0
DEFAULT_DOORWAY_WIDTH_METERS = 0.90
DEFAULT_DOORWAY_HEIGHT_METERS = 2.10
DEFAULT_DOORWAY_DEPTH_METERS = 0.20
DEFAULT_DOORWAY_BOTTOM_HEIGHT_METERS = 0.0
DOORWAY_SHAPE_RECTANGULAR = "rectangular"
DOORWAY_SHAPE_ARCH = "arch"
DOORWAY_SHAPES = frozenset(
    {
        DOORWAY_SHAPE_RECTANGULAR,
        DOORWAY_SHAPE_ARCH,
    }
)
DEFAULT_DOORWAY_SHAPE = DOORWAY_SHAPE_RECTANGULAR
DEFAULT_DOORWAY_ARCH_AMOUNT = 1.0
MIN_DOORWAY_ARCH_AMOUNT = 0.0
MAX_DOORWAY_ARCH_AMOUNT = 1.0
MIN_DOORWAY_WIDTH_METERS = 0.10
MAX_DOORWAY_WIDTH_METERS = 20.0
MIN_DOORWAY_HEIGHT_METERS = 0.10
MAX_DOORWAY_HEIGHT_METERS = 20.0
MIN_DOORWAY_DEPTH_METERS = 0.01
MAX_DOORWAY_DEPTH_METERS = 10.0
MIN_DOORWAY_BOTTOM_HEIGHT_METERS = 0.0
MAX_DOORWAY_BOTTOM_HEIGHT_METERS = 20.0
MAX_WINDOW_ID_LENGTH = 128
MAX_WINDOW_SURFACE_ID_LENGTH = 512
MIN_WINDOW_RATIO_SPAN = 1e-6
MAX_OPEN_SPACE_ID_LENGTH = 128
MIN_OPEN_SPACE_SPAN_PIXELS = 1e-6
DEFAULT_ROOM_HEIGHT_METERS = 3.0
DEFAULT_INCLUDE_IN_EXPORT = True
DEFAULT_UV_MAP_WIDTH = 1024
DEFAULT_UV_MAP_HEIGHT = 1024
DEFAULT_WALL_UV_SCALE = 1.0
DEFAULT_WALL_UV_ROTATION_DEGREES = 0
STAIR_STYLE_SUPPORTED = "supported"
STAIR_STYLE_FLOATING = "floating"
STAIR_STYLE_FLOATING_WITH_RISER = "floating_with_riser"
STAIR_STYLES = frozenset(
    {
        STAIR_STYLE_SUPPORTED,
        STAIR_STYLE_FLOATING,
        STAIR_STYLE_FLOATING_WITH_RISER,
    }
)
DEFAULT_STAIR_STYLE = STAIR_STYLE_SUPPORTED
LEGACY_STAIR_WIDTH_PIXELS = 50.0
PIXEL_TO_METER = 0.02
GROUND_LEVEL_INDEX = 2
MIN_LEVEL_INDEX = 0
MAX_LEVEL_INDEX = 7
SNAP_ANGLE_DEGREES = 10.0
VERTEX_HIT_RADIUS_SCREEN = 12.0
_WINDOW_WALL_SURFACE_ID_PATTERN = re.compile(
    r"^level:(?:0|[1-9]\d*)/"
    r"(?:room:(?:0|[1-9]\d*)/)?"
    r"wall:[1-9]\d*:[1-9]\d*$"
)
EDITABLE_SURFACE_FRAME_WALL_RATIO = "wall_ratio"
EDITABLE_SURFACE_FRAME_LEVEL_IMAGE = "level_image"
EDITABLE_SURFACE_FRAME_KINDS = frozenset(
    {
        EDITABLE_SURFACE_FRAME_WALL_RATIO,
        EDITABLE_SURFACE_FRAME_LEVEL_IMAGE,
    }
)
EDITABLE_SURFACE_TYPES = frozenset({"wall", "floor", "ceiling"})
_EDITABLE_SURFACE_UUID_PATTERN = re.compile(r"^[0-9a-f]{32}$")

# ### Data models ###
@dataclass(frozen=True)
class Vertex:
    id: int
    x: float
    y: float


@dataclass(frozen=True)
class Edge:
    start_vertex_id: int
    end_vertex_id: int


@dataclass
class RoomData:
    name: str
    vertex_ids: tuple[int, ...]
    center_vertex_id: int
    color_rgb: tuple[int, int, int]
    height_meters: float = DEFAULT_ROOM_HEIGHT_METERS
    uv_map_width: int = DEFAULT_UV_MAP_WIDTH
    uv_map_height: int = DEFAULT_UV_MAP_HEIGHT
    wall_uv_scales: dict[str, float] = field(default_factory=dict)
    wall_uv_rotations: dict[str, int] = field(default_factory=dict)
    wall_uv_positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    wall_subdivisions: dict[str, int] = field(default_factory=dict)
    wall_subdivision_positions: dict[
        str,
        tuple[tuple[float, float], ...],
    ] = field(default_factory=dict)
    wall_subdivision_source_ranges: dict[
        str,
        tuple[tuple[float, float], ...],
    ] = field(default_factory=dict)
    wall_textures: dict[str, "WallTextureData"] = field(default_factory=dict)


@dataclass
class WallTextureData:
    image_path: str
    source_x: float
    source_y: float
    source_width: float
    source_height: float


@dataclass(frozen=True)
class DoorwayPreset:
    width_meters: float
    height_meters: float
    shape: str = DEFAULT_DOORWAY_SHAPE
    arch_amount: float = DEFAULT_DOORWAY_ARCH_AMOUNT

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", normalize_doorway_shape(self.shape))
        object.__setattr__(
            self,
            "arch_amount",
            normalize_doorway_arch_amount(self.arch_amount),
        )


@dataclass
class DoorwayData:
    center_x: float
    center_y: float
    width_meters: float
    height_meters: float
    depth_meters: float = DEFAULT_DOORWAY_DEPTH_METERS
    rotation_degrees: float = 0.0
    shape: str = DEFAULT_DOORWAY_SHAPE
    arch_amount: float = DEFAULT_DOORWAY_ARCH_AMOUNT
    bottom_height_meters: float = DEFAULT_DOORWAY_BOTTOM_HEIGHT_METERS

    def __post_init__(self) -> None:
        self.shape = normalize_doorway_shape(self.shape)
        self.arch_amount = normalize_doorway_arch_amount(self.arch_amount)
        self.bottom_height_meters = normalize_doorway_bottom_height_meters(
            self.bottom_height_meters
        )


@dataclass(frozen=True)
class WindowData:
    """One stable rectangular opening attached to a semantic wall."""

    window_id: str
    wall_surface_id: str
    start_ratio: float
    end_ratio: float
    bottom_ratio: float
    top_ratio: float

    def __post_init__(self) -> None:
        window_id = _normalize_window_text(
            self.window_id,
            "Window ID",
            MAX_WINDOW_ID_LENGTH,
        )
        wall_surface_id = _normalize_window_text(
            self.wall_surface_id,
            "Window wall surface ID",
            MAX_WINDOW_SURFACE_ID_LENGTH,
        )
        if not _WINDOW_WALL_SURFACE_ID_PATTERN.fullmatch(wall_surface_id):
            raise ValueError("A window must reference a stable wall surface ID.")

        start_ratio = _normalize_window_ratio(
            self.start_ratio,
            "start",
        )
        end_ratio = _normalize_window_ratio(self.end_ratio, "end")
        bottom_ratio = _normalize_window_ratio(
            self.bottom_ratio,
            "bottom",
        )
        top_ratio = _normalize_window_ratio(self.top_ratio, "top")
        if end_ratio - start_ratio <= MIN_WINDOW_RATIO_SPAN:
            raise ValueError("A window must have a positive wall-local width.")
        if top_ratio - bottom_ratio <= MIN_WINDOW_RATIO_SPAN:
            raise ValueError("A window must have a positive wall-local height.")

        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "wall_surface_id", wall_surface_id)
        object.__setattr__(self, "start_ratio", start_ratio)
        object.__setattr__(self, "end_ratio", end_ratio)
        object.__setattr__(self, "bottom_ratio", bottom_ratio)
        object.__setattr__(self, "top_ratio", top_ratio)

    def to_dict(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "wall_surface_id": self.wall_surface_id,
            "start_ratio": self.start_ratio,
            "end_ratio": self.end_ratio,
            "bottom_ratio": self.bottom_ratio,
            "top_ratio": self.top_ratio,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "WindowData":
        if not isinstance(payload, dict):
            raise ValueError("Window JSON must contain an object.")
        return cls(
            window_id=payload.get("window_id", ""),
            wall_surface_id=payload.get("wall_surface_id", ""),
            start_ratio=payload.get("start_ratio"),
            end_ratio=payload.get("end_ratio"),
            bottom_ratio=payload.get("bottom_ratio"),
            top_ratio=payload.get("top_ratio"),
        )


@dataclass(frozen=True)
class EditableSurfaceVertexData:
    """One persistent vertex in an editable source-surface frame."""

    vertex_id: str
    u: float
    v: float
    normal_offset_meters: float
    is_user_placed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vertex_id",
            _normalize_editable_surface_uuid(self.vertex_id, "vertex ID"),
        )
        object.__setattr__(self, "u", _normalize_finite_float(self.u, "vertex U"))
        object.__setattr__(self, "v", _normalize_finite_float(self.v, "vertex V"))
        object.__setattr__(
            self,
            "normal_offset_meters",
            _normalize_finite_float(
                self.normal_offset_meters,
                "vertex normal offset",
            ),
        )
        if not isinstance(self.is_user_placed, bool):
            raise TypeError("Editable vertex user-placement state must be boolean.")


@dataclass(frozen=True)
class EditableSurfaceFaceData:
    """One stable selectable polygon backed by ordered persistent vertices."""

    face_id: str
    vertex_ids: tuple[str, ...]
    surface_type: str
    is_directly_drawn: bool = False

    def __post_init__(self) -> None:
        face_id = _normalize_editable_surface_uuid(self.face_id, "face ID")
        try:
            vertex_ids = tuple(
                _normalize_editable_surface_uuid(vertex_id, "face vertex ID")
                for vertex_id in self.vertex_ids
            )
        except TypeError as error:
            raise ValueError(
                "Editable surface face vertex IDs must contain a sequence."
            ) from error
        if len(vertex_ids) < 3:
            raise ValueError("An editable surface face requires at least 3 vertices.")
        if len(set(vertex_ids)) != len(vertex_ids):
            raise ValueError("Editable surface face vertices must be unique.")
        surface_type = str(self.surface_type).strip().lower()
        if surface_type not in EDITABLE_SURFACE_TYPES:
            raise ValueError(
                f"Unknown editable surface type: {self.surface_type!r}."
            )
        if not isinstance(self.is_directly_drawn, bool):
            raise TypeError("Editable face drawing provenance must be boolean.")
        object.__setattr__(self, "face_id", face_id)
        object.__setattr__(self, "vertex_ids", vertex_ids)
        object.__setattr__(self, "surface_type", surface_type)


@dataclass(frozen=True)
class EditableSurfaceEdgeData:
    """One persistent user-authored edge between surface vertices."""

    start_vertex_id: str
    end_vertex_id: str

    def __post_init__(self) -> None:
        start_vertex_id = _normalize_editable_surface_uuid(
            self.start_vertex_id,
            "edge start vertex ID",
        )
        end_vertex_id = _normalize_editable_surface_uuid(
            self.end_vertex_id,
            "edge end vertex ID",
        )
        if start_vertex_id == end_vertex_id:
            raise ValueError("An editable surface edge requires two vertices.")
        object.__setattr__(self, "start_vertex_id", start_vertex_id)
        object.__setattr__(self, "end_vertex_id", end_vertex_id)


@dataclass(frozen=True)
class EditableSurfaceMeshData:
    """Persistent draft or replacement topology for one semantic surface."""

    source_surface_id: str
    frame_kind: str
    frame_normal_sign: int
    vertices: tuple[EditableSurfaceVertexData, ...]
    faces: tuple[EditableSurfaceFaceData, ...]
    edges: tuple[EditableSurfaceEdgeData, ...] = ()
    replaces_source_surface: bool = True

    def __post_init__(self) -> None:
        source_surface_id = str(self.source_surface_id).strip().lower()
        if not source_surface_id or len(source_surface_id) > 512:
            raise ValueError("Editable source surface ID is empty or too long.")
        frame_kind = str(self.frame_kind).strip().lower()
        if frame_kind not in EDITABLE_SURFACE_FRAME_KINDS:
            raise ValueError(f"Unknown editable surface frame: {self.frame_kind!r}.")
        if isinstance(self.frame_normal_sign, bool):
            raise TypeError("Editable surface frame normal sign must be -1 or 1.")
        try:
            frame_normal_sign = int(self.frame_normal_sign)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Editable surface frame normal sign must be -1 or 1."
            ) from error
        if frame_normal_sign not in {-1, 1}:
            raise ValueError("Editable surface frame normal sign must be -1 or 1.")
        try:
            vertices = tuple(self.vertices)
            faces = tuple(self.faces)
            edges = tuple(self.edges)
        except TypeError as error:
            raise ValueError(
                "Editable surface vertices, faces, and edges must contain sequences."
            ) from error
        if not isinstance(self.replaces_source_surface, bool):
            raise TypeError(
                "Editable surface replacement state must be a boolean."
            )
        if not vertices or not all(
            isinstance(vertex, EditableSurfaceVertexData) for vertex in vertices
        ):
            raise ValueError(
                "Editable surfaces require persistent vertex records."
            )
        if not all(isinstance(face, EditableSurfaceFaceData) for face in faces):
            raise ValueError("Editable surfaces require persistent face records.")
        if not faces and not self.replaces_source_surface:
            raise ValueError(
                "Editable surface drafts require persistent face records."
            )
        if not all(isinstance(edge, EditableSurfaceEdgeData) for edge in edges):
            raise ValueError("Editable surface edges must contain edge records.")
        if not faces and edges:
            raise ValueError(
                "Empty editable surface replacements cannot retain edges."
            )
        vertex_ids = tuple(vertex.vertex_id for vertex in vertices)
        face_ids = tuple(face.face_id for face in faces)
        if len(set(vertex_ids)) != len(vertex_ids):
            raise ValueError("Editable surface vertex IDs must be unique.")
        if len(set(face_ids)) != len(face_ids):
            raise ValueError("Editable surface face IDs must be unique.")
        known_vertex_ids = set(vertex_ids)
        if any(
            vertex_id not in known_vertex_ids
            for face in faces
            for vertex_id in face.vertex_ids
        ):
            raise ValueError(
                "Editable surface faces must reference existing vertices."
            )
        if any(
            vertex_id not in known_vertex_ids
            for edge in edges
            for vertex_id in (edge.start_vertex_id, edge.end_vertex_id)
        ):
            raise ValueError(
                "Editable surface edges must reference existing vertices."
            )
        edge_keys = [
            frozenset((edge.start_vertex_id, edge.end_vertex_id)) for edge in edges
        ]
        if len(set(edge_keys)) != len(edge_keys):
            raise ValueError("Editable surface edges must be unique.")
        object.__setattr__(self, "source_surface_id", source_surface_id)
        object.__setattr__(self, "frame_kind", frame_kind)
        object.__setattr__(self, "frame_normal_sign", frame_normal_sign)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "edges", edges)


@dataclass(frozen=True, init=False)
class StairSectionData:
    """One locally stored two-point cross-section along a stair route."""

    level_index: int
    a_x: float
    a_y: float
    b_x: float
    b_y: float
    a_vertex_id: int | None = None
    b_vertex_id: int | None = None

    def __init__(
        self,
        level_index: object,
        a_x: object,
        a_y: object,
        b_x: object,
        b_y: object,
        a_vertex_id: object = None,
        b_vertex_id: object = None,
    ) -> None:
        _validate_stair_level_index(level_index, "section")
        normalized_coordinates = {
            name: _normalize_stair_coordinate(value, f"section {name}")
            for name, value in (
                ("a x", a_x),
                ("a y", a_y),
                ("b x", b_x),
                ("b y", b_y),
            )
        }
        _validate_stair_segment_width(
            normalized_coordinates["a x"],
            normalized_coordinates["a y"],
            normalized_coordinates["b x"],
            normalized_coordinates["b y"],
            "section",
        )

        object.__setattr__(self, "level_index", level_index)
        object.__setattr__(self, "a_x", normalized_coordinates["a x"])
        object.__setattr__(self, "a_y", normalized_coordinates["a y"])
        object.__setattr__(self, "b_x", normalized_coordinates["b x"])
        object.__setattr__(self, "b_y", normalized_coordinates["b y"])
        object.__setattr__(
            self,
            "a_vertex_id",
            _normalize_optional_stair_vertex_id(a_vertex_id, "section a"),
        )
        object.__setattr__(
            self,
            "b_vertex_id",
            _normalize_optional_stair_vertex_id(b_vertex_id, "section b"),
        )

    @property
    def points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((self.a_x, self.a_y), (self.b_x, self.b_y))

    @property
    def center_x(self) -> float:
        return (self.a_x + self.b_x) / 2.0

    @property
    def center_y(self) -> float:
        return (self.a_y + self.b_y) / 2.0

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "level_index": int(self.level_index),
            "a_x": float(self.a_x),
            "a_y": float(self.a_y),
            "b_x": float(self.b_x),
            "b_y": float(self.b_y),
            "a_vertex_id": self.a_vertex_id,
            "b_vertex_id": self.b_vertex_id,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "StairSectionData":
        if not isinstance(payload, dict):
            raise ValueError("Stair section JSON must contain an object.")
        return cls(
            level_index=payload.get("level_index"),
            a_x=payload.get("a_x"),
            a_y=payload.get("a_y"),
            b_x=payload.get("b_x"),
            b_y=payload.get("b_y"),
            a_vertex_id=payload.get("a_vertex_id"),
            b_vertex_id=payload.get("b_vertex_id"),
        )


@dataclass(frozen=True, init=False)
class StairData:
    """A stairway joining ordered local plan segments on different levels.

    The ``a`` and ``b`` points define the full stair width at each elevation.
    Coordinates remain valid when no wall vertex is involved.  A nullable
    vertex id may additionally bind a point to a level vertex so moving that
    vertex can move the stair point without rewriting the stored fallback
    coordinates.

    Intermediate sections bend the stair route in their stored order. The
    custom initializer accepts the old single-point ``start_x`` / ``end_x``
    representation and projects without intermediate sections, keeping both
    earlier stair formats usable.
    """

    start_level_index: int
    start_a_x: float
    start_a_y: float
    start_b_x: float
    start_b_y: float
    end_level_index: int
    end_a_x: float
    end_a_y: float
    end_b_x: float
    end_b_y: float
    style: str = DEFAULT_STAIR_STYLE
    start_a_vertex_id: int | None = None
    start_b_vertex_id: int | None = None
    end_a_vertex_id: int | None = None
    end_b_vertex_id: int | None = None
    intermediate_sections: tuple[StairSectionData, ...] = ()

    def __init__(
        self,
        start_level_index: object,
        start_x: object = None,
        start_y: object = None,
        end_level_index: object = None,
        end_x: object = None,
        end_y: object = None,
        style: object = DEFAULT_STAIR_STYLE,
        *,
        start_a_x: object = None,
        start_a_y: object = None,
        start_b_x: object = None,
        start_b_y: object = None,
        end_a_x: object = None,
        end_a_y: object = None,
        end_b_x: object = None,
        end_b_y: object = None,
        start_a_vertex_id: object = None,
        start_b_vertex_id: object = None,
        end_a_vertex_id: object = None,
        end_b_vertex_id: object = None,
        intermediate_sections: object = (),
    ) -> None:
        _validate_stair_level_index(start_level_index, "start")
        _validate_stair_level_index(end_level_index, "end")
        if start_level_index == end_level_index:
            raise ValueError("Stair endpoints must belong to different levels.")

        canonical_coordinates = (
            start_a_x,
            start_a_y,
            start_b_x,
            start_b_y,
            end_a_x,
            end_a_y,
            end_b_x,
            end_b_y,
        )
        if any(value is not None for value in canonical_coordinates):
            if not all(value is not None for value in canonical_coordinates):
                raise ValueError(
                    "A stair requires two complete points on each level."
                )
            if any(
                value is not None
                for value in (start_x, start_y, end_x, end_y)
            ):
                raise ValueError(
                    "Do not mix legacy stair center points with four-point "
                    "coordinates."
                )
        else:
            legacy_coordinates = (start_x, start_y, end_x, end_y)
            if not all(value is not None for value in legacy_coordinates):
                raise ValueError(
                    "A stair requires two complete points on each level."
                )
            (
                start_a_x,
                start_a_y,
                start_b_x,
                start_b_y,
                end_a_x,
                end_a_y,
                end_b_x,
                end_b_y,
            ) = _migrate_legacy_stair_centerline(
                start_x,
                start_y,
                end_x,
                end_y,
            )

        normalized_coordinates = {
            name: _normalize_stair_coordinate(value, name.replace("_", " "))
            for name, value in (
                ("start_a_x", start_a_x),
                ("start_a_y", start_a_y),
                ("start_b_x", start_b_x),
                ("start_b_y", start_b_y),
                ("end_a_x", end_a_x),
                ("end_a_y", end_a_y),
                ("end_b_x", end_b_x),
                ("end_b_y", end_b_y),
            )
        }
        _validate_stair_segment_width(
            normalized_coordinates["start_a_x"],
            normalized_coordinates["start_a_y"],
            normalized_coordinates["start_b_x"],
            normalized_coordinates["start_b_y"],
            "start",
        )
        _validate_stair_segment_width(
            normalized_coordinates["end_a_x"],
            normalized_coordinates["end_a_y"],
            normalized_coordinates["end_b_x"],
            normalized_coordinates["end_b_y"],
            "end",
        )

        object.__setattr__(self, "start_level_index", start_level_index)
        object.__setattr__(self, "end_level_index", end_level_index)
        for name, value in normalized_coordinates.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "style", normalize_stair_style(style))
        object.__setattr__(
            self,
            "start_a_vertex_id",
            _normalize_optional_stair_vertex_id(
                start_a_vertex_id,
                "start a",
            ),
        )
        object.__setattr__(
            self,
            "start_b_vertex_id",
            _normalize_optional_stair_vertex_id(
                start_b_vertex_id,
                "start b",
            ),
        )
        object.__setattr__(
            self,
            "end_a_vertex_id",
            _normalize_optional_stair_vertex_id(end_a_vertex_id, "end a"),
        )
        object.__setattr__(
            self,
            "end_b_vertex_id",
            _normalize_optional_stair_vertex_id(end_b_vertex_id, "end b"),
        )
        object.__setattr__(
            self,
            "intermediate_sections",
            _normalize_stair_intermediate_sections(intermediate_sections),
        )

    @property
    def start_points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (self.start_a_x, self.start_a_y),
            (self.start_b_x, self.start_b_y),
        )

    @property
    def end_points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (self.end_a_x, self.end_a_y),
            (self.end_b_x, self.end_b_y),
        )

    @property
    def start_section(self) -> StairSectionData:
        return StairSectionData(
            level_index=self.start_level_index,
            a_x=self.start_a_x,
            a_y=self.start_a_y,
            b_x=self.start_b_x,
            b_y=self.start_b_y,
            a_vertex_id=self.start_a_vertex_id,
            b_vertex_id=self.start_b_vertex_id,
        )

    @property
    def end_section(self) -> StairSectionData:
        return StairSectionData(
            level_index=self.end_level_index,
            a_x=self.end_a_x,
            a_y=self.end_a_y,
            b_x=self.end_b_x,
            b_y=self.end_b_y,
            a_vertex_id=self.end_a_vertex_id,
            b_vertex_id=self.end_b_vertex_id,
        )

    @property
    def sections(self) -> tuple[StairSectionData, ...]:
        """Return the complete stair route from start through end."""

        return (
            self.start_section,
            *self.intermediate_sections,
            self.end_section,
        )

    @property
    def start_x(self) -> float:
        """Return the legacy start center for transitional callers."""

        return (self.start_a_x + self.start_b_x) / 2.0

    @property
    def start_y(self) -> float:
        """Return the legacy start center for transitional callers."""

        return (self.start_a_y + self.start_b_y) / 2.0

    @property
    def end_x(self) -> float:
        """Return the legacy end center for transitional callers."""

        return (self.end_a_x + self.end_b_x) / 2.0

    @property
    def end_y(self) -> float:
        """Return the legacy end center for transitional callers."""

        return (self.end_a_y + self.end_b_y) / 2.0

    @property
    def is_floating(self) -> bool:
        """Return whether this stair uses unsupported floating treads."""

        return self.style in {
            STAIR_STYLE_FLOATING,
            STAIR_STYLE_FLOATING_WITH_RISER,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "start_level_index": int(self.start_level_index),
            "start_a_x": float(self.start_a_x),
            "start_a_y": float(self.start_a_y),
            "start_b_x": float(self.start_b_x),
            "start_b_y": float(self.start_b_y),
            "end_level_index": int(self.end_level_index),
            "end_a_x": float(self.end_a_x),
            "end_a_y": float(self.end_a_y),
            "end_b_x": float(self.end_b_x),
            "end_b_y": float(self.end_b_y),
            "style": self.style,
            "start_a_vertex_id": self.start_a_vertex_id,
            "start_b_vertex_id": self.start_b_vertex_id,
            "end_a_vertex_id": self.end_a_vertex_id,
            "end_b_vertex_id": self.end_b_vertex_id,
            "intermediate_sections": [
                section.to_dict() for section in self.intermediate_sections
            ],
        }

    @classmethod
    def from_dict(cls, payload: object) -> "StairData":
        if not isinstance(payload, dict):
            raise ValueError("Stair JSON must contain an object.")

        common_values = {
            "start_level_index": payload.get("start_level_index"),
            "end_level_index": payload.get("end_level_index"),
            "style": payload.get("style", DEFAULT_STAIR_STYLE),
            "start_a_vertex_id": payload.get("start_a_vertex_id"),
            "start_b_vertex_id": payload.get("start_b_vertex_id"),
            "end_a_vertex_id": payload.get("end_a_vertex_id"),
            "end_b_vertex_id": payload.get("end_b_vertex_id"),
            "intermediate_sections": payload.get("intermediate_sections", ()),
        }
        canonical_names = (
            "start_a_x",
            "start_a_y",
            "start_b_x",
            "start_b_y",
            "end_a_x",
            "end_a_y",
            "end_b_x",
            "end_b_y",
        )
        if any(name in payload for name in canonical_names):
            legacy_names = ("start_x", "start_y", "end_x", "end_y")
            if any(name in payload for name in legacy_names):
                raise ValueError(
                    "Do not mix legacy stair center points with four-point "
                    "coordinates."
                )
            return cls(
                **common_values,
                **{name: payload.get(name) for name in canonical_names},
            )

        return cls(
            **common_values,
            start_x=payload.get("start_x"),
            start_y=payload.get("start_y"),
            end_x=payload.get("end_x"),
            end_y=payload.get("end_y"),
        )


@dataclass
class VertexData:
    vertices: list[Vertex] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    _next_vertex_id: int = 1

    def reset(self) -> None:
        self.vertices.clear()
        self.edges.clear()
        self._next_vertex_id = 1

    def clone(self) -> "VertexData":
        return copy.deepcopy(self)

    def copy_from(self, other: "VertexData") -> None:
        snapshot = other.clone()
        self.vertices = snapshot.vertices
        self.edges = snapshot.edges
        self._next_vertex_id = snapshot._next_vertex_id

    def add_vertex(self, x: float, y: float) -> Vertex:
        vertex = Vertex(id=self._next_vertex_id, x=float(x), y=float(y))
        self.vertices.append(vertex)
        self._next_vertex_id += 1
        return vertex

    def get_vertex(self, vertex_id: int) -> Vertex | None:
        for vertex in self.vertices:
            if vertex.id == vertex_id:
                return vertex
        return None

    def move_vertex(self, vertex_id: int, x: float, y: float) -> Vertex | None:
        for index, vertex in enumerate(self.vertices):
            if vertex.id != vertex_id:
                continue

            moved_vertex = Vertex(id=vertex.id, x=float(x), y=float(y))
            self.vertices[index] = moved_vertex
            return moved_vertex

        return None

    def delete_vertex(self, vertex_id: int) -> bool:
        vertex_to_delete = self.get_vertex(vertex_id)
        if vertex_to_delete is None:
            return False

        self.vertices = [
            vertex
            for vertex in self.vertices
            if vertex.id != vertex_id
        ]
        self.edges = [
            edge
            for edge in self.edges
            if edge.start_vertex_id != vertex_id and edge.end_vertex_id != vertex_id
        ]
        return True

    def add_edge(self, start_vertex_id: int, end_vertex_id: int) -> Edge | None:
        if start_vertex_id == end_vertex_id:
            return None
        if self.has_edge(start_vertex_id, end_vertex_id):
            return None

        edge = Edge(start_vertex_id=start_vertex_id, end_vertex_id=end_vertex_id)
        self.edges.append(edge)
        return edge

    def remove_edge(self, start_vertex_id: int, end_vertex_id: int) -> bool:
        expected_key = self._edge_key(start_vertex_id, end_vertex_id)
        original_edge_count = len(self.edges)
        self.edges = [
            edge
            for edge in self.edges
            if self._edge_key(edge.start_vertex_id, edge.end_vertex_id)
            != expected_key
        ]
        return len(self.edges) != original_edge_count

    def split_edge(
        self,
        start_vertex_id: int,
        end_vertex_id: int,
        middle_vertex_id: int,
    ) -> bool:
        if middle_vertex_id in (start_vertex_id, end_vertex_id):
            return False
        if self.get_vertex(start_vertex_id) is None:
            return False
        if self.get_vertex(end_vertex_id) is None:
            return False
        if self.get_vertex(middle_vertex_id) is None:
            return False
        if not self.remove_edge(start_vertex_id, end_vertex_id):
            return False

        self.add_edge(start_vertex_id, middle_vertex_id)
        self.add_edge(middle_vertex_id, end_vertex_id)
        return True

    def has_edge(self, start_vertex_id: int, end_vertex_id: int) -> bool:
        expected_key = self._edge_key(start_vertex_id, end_vertex_id)
        return any(
            self._edge_key(edge.start_vertex_id, edge.end_vertex_id) == expected_key
            for edge in self.edges
        )

    def to_dict(self) -> dict[str, list[dict[str, float | int]]]:
        return {
            "next_vertex_id": self._next_vertex_id,
            "vertices": [
                {"id": vertex.id, "x": vertex.x, "y": vertex.y}
                for vertex in self.vertices
            ],
            "edges": [
                {
                    "start_vertex_id": edge.start_vertex_id,
                    "end_vertex_id": edge.end_vertex_id,
                }
                for edge in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "VertexData":
        vertex_data = cls()
        vertex_data.vertices = [
            Vertex(
                id=int(vertex["id"]),
                x=float(vertex["x"]),
                y=float(vertex["y"]),
            )
            for vertex in payload.get("vertices", [])
        ]
        vertex_data.edges = [
            Edge(
                start_vertex_id=int(edge["start_vertex_id"]),
                end_vertex_id=int(edge["end_vertex_id"]),
            )
            for edge in payload.get("edges", [])
        ]

        default_next_vertex_id = max(
            [vertex.id for vertex in vertex_data.vertices],
            default=0,
        ) + 1
        vertex_data._next_vertex_id = int(
            payload.get("next_vertex_id", default_next_vertex_id)
        )
        return vertex_data

    @staticmethod
    def _edge_key(start_vertex_id: int, end_vertex_id: int) -> tuple[int, int]:
        return tuple(sorted((start_vertex_id, end_vertex_id)))


# ### Open-space models ###
@dataclass(frozen=True)
class OpenSpaceData:
    """One stable rectangular opening stored in level image coordinates."""

    open_space_id: str
    minimum_x: float
    minimum_y: float
    maximum_x: float
    maximum_y: float

    def __post_init__(self) -> None:
        open_space_id = _normalize_window_text(
            self.open_space_id,
            "Open-space ID",
            MAX_OPEN_SPACE_ID_LENGTH,
        )
        minimum_x = _normalize_open_space_coordinate(self.minimum_x, "minimum X")
        minimum_y = _normalize_open_space_coordinate(self.minimum_y, "minimum Y")
        maximum_x = _normalize_open_space_coordinate(self.maximum_x, "maximum X")
        maximum_y = _normalize_open_space_coordinate(self.maximum_y, "maximum Y")
        if maximum_x - minimum_x <= MIN_OPEN_SPACE_SPAN_PIXELS:
            raise ValueError("An open space must have a positive width.")
        if maximum_y - minimum_y <= MIN_OPEN_SPACE_SPAN_PIXELS:
            raise ValueError("An open space must have a positive height.")

        object.__setattr__(self, "open_space_id", open_space_id)
        object.__setattr__(self, "minimum_x", minimum_x)
        object.__setattr__(self, "minimum_y", minimum_y)
        object.__setattr__(self, "maximum_x", maximum_x)
        object.__setattr__(self, "maximum_y", maximum_y)

    def to_dict(self) -> dict[str, object]:
        return {
            "open_space_id": self.open_space_id,
            "minimum_x": self.minimum_x,
            "minimum_y": self.minimum_y,
            "maximum_x": self.maximum_x,
            "maximum_y": self.maximum_y,
        }

    @classmethod
    def from_dict(cls, payload: object) -> OpenSpaceData:
        if not isinstance(payload, dict):
            raise TypeError("Open-space JSON must contain an object.")
        return cls(
            open_space_id=payload.get("open_space_id", ""),
            minimum_x=payload.get("minimum_x"),
            minimum_y=payload.get("minimum_y"),
            maximum_x=payload.get("maximum_x"),
            maximum_y=payload.get("maximum_y"),
        )


# ### Level models ###
@dataclass
class LevelData:
    index: int
    name: str
    height_meters: float = DEFAULT_LEVEL_HEIGHT_METERS
    scale: float = DEFAULT_LEVEL_SCALE
    canvas_level_scale: float = DEFAULT_CANVAS_LEVEL_SCALE
    canvas_offset_x_pixels: float = DEFAULT_CANVAS_OFFSET_PIXELS
    canvas_offset_y_pixels: float = DEFAULT_CANVAS_OFFSET_PIXELS
    offset_x_meters: float = DEFAULT_LEVEL_OFFSET_METERS
    offset_y_meters: float = DEFAULT_LEVEL_OFFSET_METERS
    vertex_data: VertexData = field(default_factory=VertexData)
    rooms: list[RoomData] = field(default_factory=list)
    doorways: list[DoorwayData] = field(default_factory=list)
    image_path: str | None = None
    image_size_pixels: tuple[float, float] | None = None
    include_in_export: bool = DEFAULT_INCLUDE_IN_EXPORT
    floor_thickness_meters: float = DEFAULT_FLOOR_THICKNESS_METERS
    windows: list[WindowData] = field(default_factory=list)
    open_spaces: list[OpenSpaceData] = field(default_factory=list)
    editable_surfaces: list[EditableSurfaceMeshData] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return f"L{self.index} {self.name}"


# ### Doorway validation helpers ###
def normalize_doorway_shape(value: object) -> str:
    """Return one canonical doorway shape or raise a validation error."""

    if not isinstance(value, str):
        raise ValueError("Doorway shape must be a string.")

    shape = value.strip().lower()
    if shape not in DOORWAY_SHAPES:
        supported_shapes = ", ".join(sorted(DOORWAY_SHAPES))
        raise ValueError(
            f"Doorway shape must be one of: {supported_shapes}."
        )
    return shape


def normalize_doorway_arch_amount(value: object) -> float:
    """Return a finite normalized doorway arch amount."""

    if isinstance(value, bool):
        raise ValueError("Doorway arch amount must be a number.")
    try:
        arch_amount = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Doorway arch amount must be a number.") from error
    if not math.isfinite(arch_amount):
        raise ValueError("Doorway arch amount must be finite.")
    if not MIN_DOORWAY_ARCH_AMOUNT <= arch_amount <= MAX_DOORWAY_ARCH_AMOUNT:
        raise ValueError(
            "Doorway arch amount must be between "
            f"{MIN_DOORWAY_ARCH_AMOUNT:g} and {MAX_DOORWAY_ARCH_AMOUNT:g}."
        )
    return arch_amount


def normalize_doorway_bottom_height_meters(value: object) -> float:
    """Return a finite non-negative doorway bottom height."""

    if isinstance(value, bool):
        raise ValueError("Doorway bottom height must be a number.")
    try:
        bottom_height = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Doorway bottom height must be a number.") from error
    if not math.isfinite(bottom_height):
        raise ValueError("Doorway bottom height must be finite.")
    if not (
        MIN_DOORWAY_BOTTOM_HEIGHT_METERS
        <= bottom_height
        <= MAX_DOORWAY_BOTTOM_HEIGHT_METERS
    ):
        raise ValueError(
            "Doorway bottom height must be between "
            f"{MIN_DOORWAY_BOTTOM_HEIGHT_METERS:g} and "
            f"{MAX_DOORWAY_BOTTOM_HEIGHT_METERS:g} meters."
        )
    return bottom_height


# ### Editable surface validation helpers ###
def _normalize_editable_surface_uuid(value: object, field_name: str) -> str:
    normalized = str(value).strip().lower()
    if _EDITABLE_SURFACE_UUID_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            f"Editable surface {field_name} must be a 32-character UUID."
        )
    return normalized


def _normalize_finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"Editable surface {field_name} must be a number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Editable surface {field_name} must be a number."
        ) from error
    if not math.isfinite(normalized):
        raise ValueError(f"Editable surface {field_name} must be finite.")
    return normalized


# ### Window validation helpers ###
def _normalize_window_text(
    value: object,
    field_name: str,
    maximum_length: int,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")
    if len(normalized) > maximum_length:
        raise ValueError(f"{field_name} is too long.")
    if any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{field_name} cannot contain control characters.")
    return normalized


def _normalize_window_ratio(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Window {field_name} ratio must be a number.")
    try:
        ratio = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Window {field_name} ratio must be a number."
        ) from error
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError(
            f"Window {field_name} ratio must be between zero and one."
        )
    return ratio


# ### Open-space validation helpers ###
def _normalize_open_space_coordinate(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"Open-space {field_name} must be a number.")
    try:
        coordinate = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Open-space {field_name} must be a number."
        ) from error
    if not math.isfinite(coordinate):
        raise ValueError(f"Open-space {field_name} must be finite.")
    return coordinate


# ### Geometry helpers ###
def snap_point(
    base_vertex: Vertex,
    x: float,
    y: float,
    snap_step_degrees: float = SNAP_ANGLE_DEGREES,
) -> tuple[float, float]:
    delta_x = x - base_vertex.x
    delta_y = y - base_vertex.y
    distance = math.hypot(delta_x, delta_y)
    if distance == 0.0:
        return x, y

    angle_degrees = math.degrees(math.atan2(delta_y, delta_x))
    snapped_angle_degrees = round(angle_degrees / snap_step_degrees) * snap_step_degrees
    snapped_angle_radians = math.radians(snapped_angle_degrees)

    snapped_x = base_vertex.x + distance * math.cos(snapped_angle_radians)
    snapped_y = base_vertex.y + distance * math.sin(snapped_angle_radians)
    return snapped_x, snapped_y


# ### Stair validation helpers ###
def normalize_stair_style(value: object) -> str:
    """Return a supported stair style or raise a clear validation error."""

    if not isinstance(value, str):
        raise ValueError("Stair style must be a string.")

    style = value.strip().lower()
    if style not in STAIR_STYLES:
        supported_styles = ", ".join(sorted(STAIR_STYLES))
        raise ValueError(
            f"Stair style must be one of: {supported_styles}."
        )
    return style


def _validate_stair_level_index(level_index: object, endpoint_name: str) -> None:
    if isinstance(level_index, bool) or not isinstance(level_index, int):
        raise ValueError(
            f"Stair {endpoint_name} level index must be an integer."
        )
    if not MIN_LEVEL_INDEX <= level_index <= MAX_LEVEL_INDEX:
        raise ValueError(
            f"Stair {endpoint_name} level index must be between "
            f"{MIN_LEVEL_INDEX} and {MAX_LEVEL_INDEX}."
        )


def _normalize_stair_coordinate(value: object, coordinate_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Stair {coordinate_name} coordinate must be a number.")
    try:
        coordinate = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Stair {coordinate_name} coordinate must be a number."
        ) from error
    if not math.isfinite(coordinate):
        raise ValueError(
            f"Stair {coordinate_name} coordinate must be finite."
        )
    return coordinate


def _normalize_optional_stair_vertex_id(
    value: object,
    point_name: str,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"Stair {point_name} vertex id must be a positive integer or null."
        )
    return value


def _validate_stair_segment_width(
    first_x: float,
    first_y: float,
    second_x: float,
    second_y: float,
    segment_name: str,
) -> None:
    if math.hypot(second_x - first_x, second_y - first_y) <= 1e-9:
        raise ValueError(
            f"Stair {segment_name} points must be separated."
        )


def _normalize_stair_intermediate_sections(
    value: object,
) -> tuple[StairSectionData, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError("Stair intermediate sections must contain a list.")

    normalized_sections: list[StairSectionData] = []
    for section in value:
        if isinstance(section, StairSectionData):
            normalized_sections.append(section)
            continue
        normalized_sections.append(StairSectionData.from_dict(section))
    return tuple(normalized_sections)


def _migrate_legacy_stair_centerline(
    start_x: object,
    start_y: object,
    end_x: object,
    end_y: object,
) -> tuple[float, float, float, float, float, float, float, float]:
    normalized_start_x = _normalize_stair_coordinate(start_x, "start x")
    normalized_start_y = _normalize_stair_coordinate(start_y, "start y")
    normalized_end_x = _normalize_stair_coordinate(end_x, "end x")
    normalized_end_y = _normalize_stair_coordinate(end_y, "end y")
    run_x = normalized_end_x - normalized_start_x
    run_y = normalized_end_y - normalized_start_y
    run_length = math.hypot(run_x, run_y)
    if run_length <= 1e-9:
        lateral_x = LEGACY_STAIR_WIDTH_PIXELS / 2.0
        lateral_y = 0.0
    else:
        half_width = LEGACY_STAIR_WIDTH_PIXELS / 2.0
        lateral_x = (-run_y / run_length) * half_width
        lateral_y = (run_x / run_length) * half_width

    return (
        normalized_start_x + lateral_x,
        normalized_start_y + lateral_y,
        normalized_start_x - lateral_x,
        normalized_start_y - lateral_y,
        normalized_end_x + lateral_x,
        normalized_end_y + lateral_y,
        normalized_end_x - lateral_x,
        normalized_end_y - lateral_y,
    )


# ### Level helpers ###
def create_default_doorway_presets() -> list[DoorwayPreset]:
    return [
        DoorwayPreset(
            width_meters=DEFAULT_DOORWAY_WIDTH_METERS,
            height_meters=DEFAULT_DOORWAY_HEIGHT_METERS,
        )
    ]


def create_default_levels() -> list[LevelData]:
    return [
        LevelData(index=level_index, name=_get_level_name(level_index))
        for level_index in range(MIN_LEVEL_INDEX, MAX_LEVEL_INDEX + 1)
    ]


def _get_level_name(level_index: int) -> str:
    if level_index == 0:
        return "Underground 2"
    if level_index == 1:
        return "Underground 1"
    if level_index == GROUND_LEVEL_INDEX:
        return "Ground"

    story_number = level_index - GROUND_LEVEL_INDEX
    return f"Story {story_number}"
