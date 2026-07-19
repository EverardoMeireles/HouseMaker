# ### Imports ###
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

# ### Constants ###
DEFAULT_LEVEL_HEIGHT_METERS = 3.0
DEFAULT_FLOOR_THICKNESS_METERS = 0.3
MIN_FLOOR_THICKNESS_METERS = 0.01
MAX_FLOOR_THICKNESS_METERS = 10.0
DEFAULT_ROOM_HEIGHT_METERS = 3.0
DEFAULT_IMAGE_SCALE = 1.0
DEFAULT_IMAGE_OFFSET = 0.0
DEFAULT_INCLUDE_IN_EXPORT = True
DEFAULT_UV_MAP_WIDTH = 1024
DEFAULT_UV_MAP_HEIGHT = 1024
DEFAULT_WALL_UV_SCALE = 1.0
DEFAULT_WALL_UV_ROTATION_DEGREES = 0
PIXEL_TO_METER = 0.02
GROUND_LEVEL_INDEX = 2
MIN_LEVEL_INDEX = 0
MAX_LEVEL_INDEX = 7
SNAP_ANGLE_DEGREES = 10.0
VERTEX_HIT_RADIUS_SCREEN = 12.0

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


# ### Level models ###
@dataclass
class LevelData:
    index: int
    name: str
    height_meters: float = DEFAULT_LEVEL_HEIGHT_METERS
    vertex_data: VertexData = field(default_factory=VertexData)
    rooms: list[RoomData] = field(default_factory=list)
    image_path: str | None = None
    image_size_pixels: tuple[float, float] | None = None
    image_scale: float = DEFAULT_IMAGE_SCALE
    image_offset_x: float = DEFAULT_IMAGE_OFFSET
    image_offset_y: float = DEFAULT_IMAGE_OFFSET
    include_in_export: bool = DEFAULT_INCLUDE_IN_EXPORT
    floor_thickness_meters: float = DEFAULT_FLOOR_THICKNESS_METERS
    floor_contour_vertex_ids: tuple[int, ...] = ()

    @property
    def display_name(self) -> str:
        return f"L{self.index} {self.name}"


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


# ### Level helpers ###
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
