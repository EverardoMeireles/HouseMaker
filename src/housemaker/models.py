# ### Imports ###
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

# ### Constants ###
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

    def has_edge(self, start_vertex_id: int, end_vertex_id: int) -> bool:
        expected_key = self._edge_key(start_vertex_id, end_vertex_id)
        return any(
            self._edge_key(edge.start_vertex_id, edge.end_vertex_id) == expected_key
            for edge in self.edges
        )

    def to_dict(self) -> dict[str, list[dict[str, float | int]]]:
        return {
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

    @staticmethod
    def _edge_key(start_vertex_id: int, end_vertex_id: int) -> tuple[int, int]:
        return tuple(sorted((start_vertex_id, end_vertex_id)))


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
