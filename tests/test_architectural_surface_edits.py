# ### Imports ###
from __future__ import annotations

from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from shapely import Polygon
from shapely.ops import unary_union

from housemaker.architectural_surface_edits import (
    EDITABLE_SURFACE_ID_PATTERN,
    SURFACE_VERTEX_SNAP_KIND_EDGE,
    SurfaceDrawingVertexTarget,
    build_editable_surface_id,
    build_surface_drawing_overlay,
    delete_directly_drawn_surface_faces,
    extrude_surface_faces,
    insert_surface_vertex,
    place_surface_vertex,
    resolve_surface_vertex_preview,
    snap_surface_vertex_world_point,
)
from housemaker.models import (
    EditableSurfaceEdgeData,
    EditableSurfaceFaceData,
    EditableSurfaceVertexData,
    LevelData,
    VertexData,
)
from housemaker.project_io import load_project, save_project
from housemaker.surface_geometry import (
    FixedSurface,
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_FLOOR,
    build_fixed_surfaces,
)


# ### Fixture helpers ###
def _build_square_level() -> LevelData:
    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
    ):
        vertex_data.add_edge(start_id, end_id)
    return LevelData(index=2, name="Ground", vertex_data=vertex_data)


def _get_wall(level: LevelData, wall_key: str = "1:2") -> FixedSurface:
    return next(
        surface
        for surface in build_fixed_surfaces([level])
        if surface.surface_id == f"level:2/wall:{wall_key}"
    )


def _insert_into_first_wall_triangle(
    level: LevelData,
) -> tuple[str, tuple[str, ...]]:
    wall = _get_wall(level)
    insertion_point = tuple(
        float(component)
        for component in np.asarray(wall.mesh.triangles[0], dtype=float).mean(axis=0)
    )
    result = insert_surface_vertex([level], wall.surface_id, insertion_point)
    return wall.surface_id, result.selected_surface_ids


def _draw_closed_wall_face(level: LevelData) -> tuple[FixedSurface, str]:
    wall = _get_wall(level)
    active_vertex_id = None
    result = None
    for point in (
        (0.4, 0.0, 0.8),
        (1.6, 0.0, 0.8),
        (1.6, 0.0, 2.5),
        (0.4, 0.0, 2.5),
        (0.4, 0.0, 0.8),
    ):
        result = place_surface_vertex(
            [level],
            wall.surface_id,
            point,
            active_vertex_id,
        )
        active_vertex_id = result.active_vertex_id
    if result is None or len(result.created_surface_ids) != 1:
        raise AssertionError("The wall fixture did not create one closed face.")
    return wall, result.created_surface_ids[0]


def _draw_full_wall_face(level: LevelData) -> tuple[FixedSurface, str]:
    wall = _get_wall(level)
    bounds = np.asarray(wall.mesh.bounds, dtype=float)
    minimum, maximum = bounds
    active_vertex_id = None
    result = None
    for point in (
        (minimum[0], minimum[1], minimum[2]),
        (maximum[0], minimum[1], minimum[2]),
        (maximum[0], minimum[1], maximum[2]),
        (minimum[0], minimum[1], maximum[2]),
        (minimum[0], minimum[1], minimum[2]),
    ):
        result = place_surface_vertex(
            [level],
            wall.surface_id,
            point,
            active_vertex_id,
        )
        active_vertex_id = result.active_vertex_id
    if result is None or len(result.created_surface_ids) != 1:
        raise AssertionError("The wall fixture did not create one full face.")
    return wall, result.created_surface_ids[0]


def _draw_face_inside_generated_floor_side(
    level: LevelData,
    baseline_face_ids: set[str],
) -> str:
    editable_mesh = level.editable_surfaces[0]
    floor_side = next(
        face
        for face in editable_mesh.faces
        if face.face_id not in baseline_face_ids
        and face.surface_type == SURFACE_TYPE_FLOOR
    )
    floor_side_id = build_editable_surface_id(
        level.index,
        floor_side.face_id,
        floor_side.surface_type,
    )
    floor_surface = next(
        surface
        for surface in build_fixed_surfaces([level])
        if surface.surface_id == floor_side_id
    )
    bounds = np.asarray(floor_surface.mesh.bounds, dtype=float)
    minimum = bounds[0]
    span = bounds[1] - minimum
    first_x, second_x = minimum[0] + span[0] * np.asarray((0.25, 0.75))
    first_y, second_y = minimum[1] + span[1] * np.asarray((0.25, 0.75))
    height = float(np.median(floor_surface.mesh.vertices[:, 2]))
    active_vertex_id = None
    result = None
    for point in (
        (first_x, first_y, height),
        (second_x, first_y, height),
        (second_x, second_y, height),
        (first_x, second_y, height),
        (first_x, first_y, height),
    ):
        result = place_surface_vertex(
            [level],
            floor_side_id,
            point,
            active_vertex_id,
        )
        active_vertex_id = result.active_vertex_id
    if result is None or len(result.created_surface_ids) != 1:
        raise AssertionError("The generated side did not produce one inner face.")
    return result.created_surface_ids[0]


def _start_chain_inside_generated_floor_side(
    level: LevelData,
    baseline_face_ids: set[str],
) -> tuple[str, str]:
    editable_mesh = level.editable_surfaces[0]
    floor_side = next(
        face
        for face in editable_mesh.faces
        if face.face_id not in baseline_face_ids
        and face.surface_type == SURFACE_TYPE_FLOOR
    )
    floor_side_id = build_editable_surface_id(
        level.index,
        floor_side.face_id,
        floor_side.surface_type,
    )
    floor_surface = next(
        surface
        for surface in build_fixed_surfaces([level])
        if surface.surface_id == floor_side_id
    )
    bounds = np.asarray(floor_surface.mesh.bounds, dtype=float)
    minimum = bounds[0]
    span = bounds[1] - minimum
    height = float(np.median(floor_surface.mesh.vertices[:, 2]))
    first_result = place_surface_vertex(
        [level],
        floor_side_id,
        (
            minimum[0] + span[0] * 0.3,
            minimum[1] + span[1] * 0.35,
            height,
        ),
    )
    second_result = place_surface_vertex(
        [level],
        floor_side_id,
        (
            minimum[0] + span[0] * 0.7,
            minimum[1] + span[1] * 0.65,
            height,
        ),
        first_result.active_vertex_id,
    )
    if (
        first_result.active_vertex_id is None
        or second_result.active_vertex_id is None
        or second_result.created_edge_vertex_ids is None
    ):
        raise AssertionError("The generated side did not retain one open chain.")
    return first_result.active_vertex_id, second_result.active_vertex_id


def _draw_cap_adjacent_face_on_generated_floor_side(
    level: LevelData,
    cap_id: str,
    baseline_face_ids: set[str],
) -> str:
    cap_match = EDITABLE_SURFACE_ID_PATTERN.fullmatch(cap_id)
    if cap_match is None:
        raise AssertionError("The cap fixture did not expose an editable face ID.")
    cap_face_id = cap_match.group("face_id")
    editable_mesh = level.editable_surfaces[0]
    cap_face = next(
        face for face in editable_mesh.faces if face.face_id == cap_face_id
    )
    cap_vertex_ids = set(cap_face.vertex_ids)
    floor_side = next(
        face
        for face in editable_mesh.faces
        if face.face_id not in baseline_face_ids
        and face.surface_type == SURFACE_TYPE_FLOOR
        and len(face.vertex_ids) == 4
        and len(set(face.vertex_ids) & cap_vertex_ids) == 2
    )
    side_vertex_ids = floor_side.vertex_ids
    cap_pair_index = next(
        index
        for index in range(len(side_vertex_ids))
        if side_vertex_ids[index] in cap_vertex_ids
        and side_vertex_ids[(index + 1) % len(side_vertex_ids)] in cap_vertex_ids
    )
    first_cap_id = side_vertex_ids[cap_pair_index]
    second_cap_id = side_vertex_ids[(cap_pair_index + 1) % len(side_vertex_ids)]
    first_base_id = side_vertex_ids[(cap_pair_index - 1) % len(side_vertex_ids)]
    second_base_id = side_vertex_ids[(cap_pair_index + 2) % len(side_vertex_ids)]
    used_ids = {
        vertex.vertex_id for vertex in editable_mesh.vertices
    } | {face.face_id for face in editable_mesh.faces}
    next_id_value = 1

    def allocate_id() -> str:
        nonlocal next_id_value
        while True:
            candidate = f"{next_id_value:032x}"
            next_id_value += 1
            if candidate not in used_ids:
                used_ids.add(candidate)
                return candidate

    vertex_by_id = {
        vertex.vertex_id: vertex for vertex in editable_mesh.vertices
    }

    def midpoint_vertex(first_id: str, second_id: str) -> EditableSurfaceVertexData:
        first = vertex_by_id[first_id]
        second = vertex_by_id[second_id]
        return EditableSurfaceVertexData(
            vertex_id=allocate_id(),
            u=(first.u + second.u) * 0.5,
            v=(first.v + second.v) * 0.5,
            normal_offset_meters=(
                first.normal_offset_meters + second.normal_offset_meters
            )
            * 0.5,
            is_user_placed=True,
        )

    first_middle = midpoint_vertex(first_cap_id, first_base_id)
    second_middle = midpoint_vertex(second_cap_id, second_base_id)
    child_face = EditableSurfaceFaceData(
        face_id=allocate_id(),
        vertex_ids=(
            first_cap_id,
            second_cap_id,
            second_middle.vertex_id,
            first_middle.vertex_id,
        ),
        surface_type=floor_side.surface_type,
        is_directly_drawn=True,
    )
    remainder_face = EditableSurfaceFaceData(
        face_id=allocate_id(),
        vertex_ids=(
            second_base_id,
            first_base_id,
            first_middle.vertex_id,
            second_middle.vertex_id,
        ),
        surface_type=floor_side.surface_type,
    )
    level.editable_surfaces = [
        replace(
            editable_mesh,
            vertices=(
                *editable_mesh.vertices,
                first_middle,
                second_middle,
            ),
            faces=tuple(
                replacement
                for face in editable_mesh.faces
                for replacement in (
                    (child_face, remainder_face)
                    if face.face_id == floor_side.face_id
                    else (face,)
                )
            ),
            edges=(
                *editable_mesh.edges,
                EditableSurfaceEdgeData(
                    start_vertex_id=second_cap_id,
                    end_vertex_id=second_middle.vertex_id,
                ),
                EditableSurfaceEdgeData(
                    start_vertex_id=second_middle.vertex_id,
                    end_vertex_id=first_middle.vertex_id,
                ),
                EditableSurfaceEdgeData(
                    start_vertex_id=first_middle.vertex_id,
                    end_vertex_id=first_cap_id,
                ),
            ),
        )
    ]
    return build_editable_surface_id(
        level.index,
        child_face.face_id,
        child_face.surface_type,
    )


def _mesh_boundary_length(mesh: trimesh.Trimesh) -> float:
    edges = np.sort(np.asarray(mesh.edges, dtype=np.int64), axis=1)
    unique_edges, edge_counts = np.unique(
        edges,
        axis=0,
        return_counts=True,
    )
    boundary_edges = unique_edges[edge_counts == 1]
    vertices = np.asarray(mesh.vertices, dtype=float)
    return float(
        np.linalg.norm(
            vertices[boundary_edges[:, 1]] - vertices[boundary_edges[:, 0]],
            axis=1,
        ).sum()
    )


def _build_snap_surface() -> FixedSurface:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
        ),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id="level:2/edit-face:11111111111111111111111111111111:floor",
        surface_type="floor",
        level_index=2,
        room_index=None,
        mesh=mesh,
        area_square_meters=4.0,
        source_surface_id="level:2/floor",
    )


# ### Insertion and snapping tests ###
class SurfaceVertexInsertionTests(unittest.TestCase):
    def test_first_horizontal_vertex_ignores_a_degenerate_source_sliver(
        self,
    ) -> None:
        level = _build_square_level()
        source = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_type == SURFACE_TYPE_FLOOR
            and surface.room_index is None
        )
        vertices = np.asarray(source.mesh.vertices, dtype=float)
        sliver_vertices = np.asarray(
            (
                (0.5, -0.5, vertices[0, 2]),
                (0.50001, -0.5, vertices[0, 2]),
                (0.5, -0.50001, vertices[0, 2]),
            ),
            dtype=float,
        )
        poisoned_mesh = trimesh.Trimesh(
            vertices=np.vstack((vertices, sliver_vertices)),
            faces=np.vstack(
                (
                    np.asarray(source.mesh.faces, dtype=np.int64),
                    np.asarray(
                        (
                            (
                                len(vertices),
                                len(vertices) + 1,
                                len(vertices) + 2,
                            ),
                        )
                    ),
                )
            ),
            process=False,
        )
        poisoned_source = replace(source, mesh=poisoned_mesh)
        insertion_point = tuple(
            float(component)
            for component in np.asarray(source.mesh.triangles[0], dtype=float).mean(
                axis=0
            )
        )

        with patch(
            "housemaker.architectural_surface_edits.build_base_fixed_surfaces",
            return_value=[poisoned_source],
        ):
            result = place_surface_vertex(
                [level],
                source.surface_id,
                insertion_point,
            )
            overlay = build_surface_drawing_overlay([level])

        self.assertIsNotNone(result.active_vertex_id)
        self.assertIn(
            result.active_vertex_id,
            {vertex.vertex_id for vertex in overlay.vertices},
        )
        editable_mesh = level.editable_surfaces[0]
        self.assertEqual(len(editable_mesh.faces), len(source.mesh.faces))
        self.assertEqual(len(editable_mesh.vertices), len(vertices) + 1)

    def test_first_insert_snapshots_root_and_exposes_stable_child_faces(self) -> None:
        level = _build_square_level()
        wall = _get_wall(level)
        original_face_count = len(wall.mesh.faces)
        insertion_point = tuple(
            float(component)
            for component in np.asarray(wall.mesh.triangles[0], dtype=float).mean(
                axis=0
            )
        )

        result = insert_surface_vertex(
            [level],
            wall.surface_id,
            insertion_point,
        )

        self.assertEqual(len(level.editable_surfaces), 1)
        editable_mesh = level.editable_surfaces[0]
        self.assertEqual(editable_mesh.source_surface_id, wall.surface_id)
        self.assertEqual(len(editable_mesh.faces), original_face_count + 2)
        self.assertEqual(
            set(result.replacements[wall.surface_id]),
            {
                surface.surface_id
                for surface in build_fixed_surfaces([level])
                if surface.source_surface_id == wall.surface_id
            },
        )
        self.assertEqual(len(result.selected_surface_ids), 3)
        self.assertNotIn(
            wall.surface_id,
            {surface.surface_id for surface in build_fixed_surfaces([level])},
        )
        self.assertTrue(
            all(
                EDITABLE_SURFACE_ID_PATTERN.fullmatch(surface_id)
                for surface_id in result.replacements[wall.surface_id]
            )
        )

    def test_snap_helper_accepts_close_angle_but_rejects_over_one_centimeter(
        self,
    ) -> None:
        surface = _build_snap_surface()

        close_point = snap_surface_vertex_world_point(
            [surface],
            surface.surface_id,
            (0.8, 0.807, 0.02),
        )
        distant_point = snap_surface_vertex_world_point(
            [surface],
            surface.surface_id,
            (0.8, 0.822, 0.02),
        )

        self.assertAlmostEqual(close_point[0], close_point[1])
        self.assertAlmostEqual(close_point[2], 0.0)
        np.testing.assert_allclose(
            distant_point,
            (0.8, 0.822, 0.0),
            atol=1e-9,
        )

    def test_snap_helper_keeps_raw_projection_when_snap_leaves_polygon(self) -> None:
        surface = _build_snap_surface()

        resolved = snap_surface_vertex_world_point(
            [surface],
            surface.surface_id,
            (0.8, 0.03, 0.02),
        )

        np.testing.assert_allclose(resolved, (0.8, 0.03, 0.0), atol=1e-9)


# ### Continuous drawing tests ###
class SurfaceVertexDrawingTests(unittest.TestCase):
    def test_open_chain_persists_without_replacing_the_source_surface(self) -> None:
        level = _build_square_level()
        wall = _get_wall(level)

        first = place_surface_vertex(
            [level],
            wall.surface_id,
            (0.4, 0.0, 0.8),
        )
        second = place_surface_vertex(
            [level],
            wall.surface_id,
            (1.6, 0.0, 0.8),
            first.active_vertex_id,
        )

        self.assertFalse(first.requires_mesh_refresh)
        self.assertFalse(second.requires_mesh_refresh)
        self.assertTrue(second.state_changed)
        self.assertEqual(second.replacements, {})
        self.assertFalse(level.editable_surfaces[0].replaces_source_surface)
        self.assertEqual(
            [
                surface.surface_id
                for surface in build_fixed_surfaces([level])
                if surface.surface_id == wall.surface_id
            ],
            [wall.surface_id],
        )
        overlay = build_surface_drawing_overlay([level])
        self.assertEqual(len(overlay.edges), 1)
        overlay_vertex_ids = {item.vertex_id for item in overlay.vertices}
        self.assertIn(first.active_vertex_id, overlay_vertex_ids)
        self.assertIn(second.active_vertex_id, overlay_vertex_ids)
        editable_vertices = {
            vertex.vertex_id: vertex
            for vertex in level.editable_surfaces[0].vertices
        }
        self.assertTrue(editable_vertices[first.active_vertex_id].is_user_placed)
        self.assertTrue(editable_vertices[second.active_vertex_id].is_user_placed)

    def test_open_edge_to_edge_path_subdivides_the_wall_immediately(self) -> None:
        level = _build_square_level()
        wall = _get_wall(level)

        first = place_surface_vertex(
            [level],
            wall.surface_id,
            (0.0, 0.0, 1.5),
        )
        second = place_surface_vertex(
            [level],
            wall.surface_id,
            (2.0, 0.0, 1.5),
            first.active_vertex_id,
        )

        self.assertFalse(first.requires_mesh_refresh)
        self.assertTrue(second.requires_mesh_refresh)
        self.assertEqual(second.created_surface_ids, ())
        self.assertEqual(
            second.selected_surface_ids,
            second.replacements[wall.surface_id],
        )
        self.assertEqual(second.active_vertex_id, second.created_edge_vertex_ids[1])
        editable_mesh = level.editable_surfaces[0]
        self.assertTrue(editable_mesh.replaces_source_surface)
        self.assertEqual(len(editable_mesh.vertices), 6)
        self.assertEqual(len(editable_mesh.faces), 2)
        self.assertEqual(len(editable_mesh.edges), 1)
        self.assertTrue(
            all(not face.is_directly_drawn for face in editable_mesh.faces)
        )
        endpoint_ids = set(second.created_edge_vertex_ids)
        self.assertTrue(
            all(endpoint_ids.issubset(face.vertex_ids) for face in editable_mesh.faces)
        )

        child_surfaces = tuple(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
        )
        self.assertEqual(len(child_surfaces), 2)
        self.assertEqual(
            sorted(round(surface.area_square_meters, 6) for surface in child_surfaces),
            [2.4, 3.6],
        )
        self.assertAlmostEqual(
            sum(surface.area_square_meters for surface in child_surfaces),
            wall.area_square_meters,
        )
        overlay = build_surface_drawing_overlay([level])
        self.assertEqual(len(overlay.edges), 1)
        marked_vertices = {
            vertex.vertex_id: vertex
            for vertex in overlay.vertices
            if vertex.show_marker
        }
        self.assertEqual(set(marked_vertices), endpoint_ids)
        self.assertTrue(
            all(
                not vertex.direct_face_surface_ids
                for vertex in marked_vertices.values()
            )
        )

    def test_open_edge_to_edge_path_subdivides_floor_and_ceiling(self) -> None:
        for surface_type in (SURFACE_TYPE_FLOOR, SURFACE_TYPE_CEILING):
            with self.subTest(surface_type=surface_type):
                level = _build_square_level()
                surface = next(
                    candidate
                    for candidate in build_fixed_surfaces([level])
                    if candidate.surface_type == surface_type
                )
                height = float(surface.mesh.bounds[0, 2])

                first = place_surface_vertex(
                    [level],
                    surface.surface_id,
                    (0.0, -1.0, height),
                )
                second = place_surface_vertex(
                    [level],
                    surface.surface_id,
                    (2.0, -1.0, height),
                    first.active_vertex_id,
                )

                self.assertTrue(second.requires_mesh_refresh)
                self.assertEqual(second.created_surface_ids, ())
                child_surfaces = tuple(
                    candidate
                    for candidate in build_fixed_surfaces([level])
                    if candidate.source_surface_id == surface.surface_id
                )
                self.assertEqual(len(child_surfaces), 2)
                self.assertTrue(
                    all(
                        not child.is_directly_drawn
                        for child in child_surfaces
                    )
                )
                self.assertAlmostEqual(
                    sum(child.area_square_meters for child in child_surfaces),
                    surface.area_square_meters,
                )
                source_normal = np.mean(surface.mesh.face_normals, axis=0)
                self.assertTrue(
                    all(
                        float(
                            np.dot(
                                np.mean(child.mesh.face_normals, axis=0),
                                source_normal,
                            )
                        )
                        > 0.99
                        for child in child_surfaces
                    )
                )

    def test_boundary_to_interior_path_remains_an_unfinished_draft(self) -> None:
        level = _build_square_level()
        wall = _get_wall(level)

        first = place_surface_vertex(
            [level],
            wall.surface_id,
            (0.0, 0.0, 1.5),
        )
        second = place_surface_vertex(
            [level],
            wall.surface_id,
            (1.0, 0.0, 1.5),
            first.active_vertex_id,
        )

        self.assertFalse(second.requires_mesh_refresh)
        self.assertEqual(second.replacements, {})
        self.assertFalse(level.editable_surfaces[0].replaces_source_surface)

    def test_path_along_one_boundary_edge_remains_an_unfinished_draft(self) -> None:
        level = _build_square_level()
        wall = _get_wall(level)

        first = place_surface_vertex(
            [level],
            wall.surface_id,
            (0.0, 0.0, 1.0),
        )
        second = place_surface_vertex(
            [level],
            wall.surface_id,
            (0.0, 0.0, 2.0),
            first.active_vertex_id,
        )

        self.assertFalse(second.requires_mesh_refresh)
        self.assertEqual(second.replacements, {})
        self.assertFalse(level.editable_surfaces[0].replaces_source_surface)

    def test_polyline_commits_when_its_last_vertex_reaches_another_edge(self) -> None:
        level = _build_square_level()
        wall = _get_wall(level)

        first = place_surface_vertex(
            [level],
            wall.surface_id,
            (0.0, 0.0, 1.0),
        )
        middle = place_surface_vertex(
            [level],
            wall.surface_id,
            (1.0, 0.0, 1.5),
            first.active_vertex_id,
        )
        last = place_surface_vertex(
            [level],
            wall.surface_id,
            (2.0, 0.0, 1.0),
            middle.active_vertex_id,
        )

        self.assertFalse(first.requires_mesh_refresh)
        self.assertFalse(middle.requires_mesh_refresh)
        self.assertTrue(last.requires_mesh_refresh)
        self.assertEqual(len(last.selected_surface_ids), 2)
        self.assertEqual(len(level.editable_surfaces[0].faces), 2)
        self.assertEqual(len(level.editable_surfaces[0].edges), 2)

    def test_later_edge_cut_replaces_only_the_affected_editable_face(self) -> None:
        level = _build_square_level()
        wall = _get_wall(level)
        first = place_surface_vertex(
            [level],
            wall.surface_id,
            (0.0, 0.0, 1.5),
        )
        initial_cut = place_surface_vertex(
            [level],
            wall.surface_id,
            (2.0, 0.0, 1.5),
            first.active_vertex_id,
        )
        initial_surfaces = tuple(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
        )
        lower_surface = min(
            initial_surfaces,
            key=lambda surface: float(surface.mesh.centroid[2]),
        )
        upper_surface = max(
            initial_surfaces,
            key=lambda surface: float(surface.mesh.centroid[2]),
        )

        lower_first = place_surface_vertex(
            [level],
            lower_surface.surface_id,
            (1.0, 0.0, 0.3),
        )
        lower_cut = place_surface_vertex(
            [level],
            lower_surface.surface_id,
            (1.0, 0.0, 1.5),
            lower_first.active_vertex_id,
        )

        self.assertTrue(initial_cut.requires_mesh_refresh)
        self.assertTrue(lower_cut.requires_mesh_refresh)
        self.assertEqual(
            set(lower_cut.replacements),
            {lower_surface.surface_id},
        )
        self.assertEqual(len(lower_cut.selected_surface_ids), 2)
        next_surface_ids = {
            surface.surface_id
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
        }
        self.assertIn(upper_surface.surface_id, next_surface_ids)
        self.assertNotIn(lower_surface.surface_id, next_surface_ids)
        self.assertTrue(set(lower_cut.selected_surface_ids) <= next_surface_ids)

    def test_two_new_vertices_on_an_authored_edge_can_close_a_face(self) -> None:
        level = _build_square_level()
        wall = _get_wall(level)
        bridge_start = place_surface_vertex(
            [level],
            wall.surface_id,
            (0.0, 0.0, 1.8),
        )
        bridge_end = place_surface_vertex(
            [level],
            wall.surface_id,
            (2.0, 0.0, 1.8),
            bridge_start.active_vertex_id,
        )
        upper_surface = max(
            (
                surface
                for surface in build_fixed_surfaces([level])
                if surface.source_surface_id == wall.surface_id
            ),
            key=lambda surface: float(surface.mesh.centroid[2]),
        )

        first_edge_vertex = place_surface_vertex(
            [level],
            upper_surface.surface_id,
            (0.4, 0.0, 1.8),
        )
        peak = place_surface_vertex(
            [level],
            upper_surface.surface_id,
            (1.0, 0.0, 2.6),
            first_edge_vertex.active_vertex_id,
        )
        closing_preview = resolve_surface_vertex_preview(
            [level],
            upper_surface.surface_id,
            (1.6, 0.0, 1.8),
            peak.active_vertex_id,
        )

        self.assertEqual(
            set(closing_preview.snapped_edge_vertex_ids or ()),
            {
                first_edge_vertex.active_vertex_id,
                bridge_end.active_vertex_id,
            },
        )
        closed = place_surface_vertex(
            [level],
            upper_surface.surface_id,
            (1.6, 0.0, 1.8),
            peak.active_vertex_id,
        )

        self.assertTrue(closed.requires_mesh_refresh)
        self.assertEqual(len(closed.created_surface_ids), 1)
        self.assertEqual(closed.selected_surface_ids, closed.created_surface_ids)
        rebuilt = tuple(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
        )
        self.assertAlmostEqual(
            sum(surface.area_square_meters for surface in rebuilt),
            wall.area_square_meters,
        )
        extruded = extrude_surface_faces(
            [level],
            closed.created_surface_ids,
            0.2,
        )
        self.assertTrue(extruded.requires_mesh_refresh)

    def test_open_cut_nodes_generated_intersections_for_later_branches(
        self,
    ) -> None:
        level = _build_square_level()
        wall = _get_wall(level)
        _insert_into_first_wall_triangle(level)
        cut_height = 1.5
        editable_surfaces = tuple(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
        )
        start_surface = next(
            surface
            for surface in editable_surfaces
            if surface.mesh.bounds[0, 0] <= 0.0 <= surface.mesh.bounds[1, 0]
            and surface.mesh.bounds[0, 2]
            <= cut_height
            <= surface.mesh.bounds[1, 2]
        )
        cut_start = place_surface_vertex(
            [level],
            start_surface.surface_id,
            (0.0, 0.0, cut_height),
        )
        cut = place_surface_vertex(
            [level],
            start_surface.surface_id,
            (2.0, 0.0, cut_height),
            cut_start.active_vertex_id,
        )

        self.assertTrue(cut.requires_mesh_refresh)
        assert cut.created_edge_vertex_ids is not None
        cut_edges = level.editable_surfaces[0].edges
        self.assertEqual(
            cut_edges[0].start_vertex_id,
            cut.created_edge_vertex_ids[0],
        )
        self.assertEqual(
            cut_edges[-1].end_vertex_id,
            cut.created_edge_vertex_ids[1],
        )
        self.assertTrue(
            all(
                first.end_vertex_id == second.start_vertex_id
                for first, second in zip(cut_edges, cut_edges[1:])
            )
        )
        cut_endpoint_ids = set(cut.created_edge_vertex_ids or ())
        intersection = next(
            vertex
            for vertex in build_surface_drawing_overlay([level]).vertices
            if vertex.vertex_id not in cut_endpoint_ids
            and abs(vertex.world_point[2] - cut_height) <= 1e-8
            and 0.0 < vertex.world_point[0] < 2.0
        )
        incident_edge_count = sum(
            intersection.vertex_id
            in (edge.start_vertex_id, edge.end_vertex_id)
            for edge in level.editable_surfaces[0].edges
        )
        self.assertEqual(incident_edge_count, 2)

        intersection_surface = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
            and any(
                np.linalg.norm(vertex - intersection.world_point) <= 1e-8
                for vertex in np.asarray(surface.mesh.vertices, dtype=float)
            )
        )
        selected = place_surface_vertex(
            [level],
            intersection_surface.surface_id,
            intersection.world_point,
        )
        branch = place_surface_vertex(
            [level],
            intersection_surface.surface_id,
            (
                intersection.world_point[0],
                intersection.world_point[1],
                float(wall.mesh.bounds[0, 2]),
            ),
            selected.active_vertex_id,
        )

        self.assertTrue(branch.requires_mesh_refresh)
        rebuilt = tuple(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
        )
        self.assertAlmostEqual(
            sum(surface.area_square_meters for surface in rebuilt),
            wall.area_square_meters,
        )

    def test_closing_chain_hides_edges_and_marks_only_manual_vertices(self) -> None:
        level = _build_square_level()
        wall = _get_wall(level)
        points = (
            (0.4, 0.0, 0.8),
            (1.6, 0.0, 0.8),
            (1.6, 0.0, 2.5),
            (0.4, 0.0, 2.5),
        )
        placed_vertex_ids: list[str] = []
        active_vertex_id = None
        for point in points:
            placement = place_surface_vertex(
                [level],
                wall.surface_id,
                point,
                active_vertex_id,
            )
            assert placement.active_vertex_id is not None
            placed_vertex_ids.append(placement.active_vertex_id)
            active_vertex_id = placement.active_vertex_id

        open_overlay = build_surface_drawing_overlay([level])
        self.assertEqual(len(open_overlay.edges), 3)

        closed = place_surface_vertex(
            [level],
            wall.surface_id,
            points[0],
            active_vertex_id,
        )
        closed_overlay = build_surface_drawing_overlay([level])

        self.assertEqual(len(closed.created_surface_ids), 1)
        self.assertEqual(closed_overlay.edges, ())
        marked_vertices = {
            vertex.vertex_id: vertex
            for vertex in closed_overlay.vertices
            if vertex.show_marker
        }
        self.assertEqual(set(marked_vertices), set(placed_vertex_ids))
        self.assertTrue(
            all(
                vertex.direct_face_surface_ids == closed.created_surface_ids
                for vertex in marked_vertices.values()
            )
        )
        editable_vertices = level.editable_surfaces[0].vertices
        self.assertEqual(
            {
                vertex.vertex_id
                for vertex in editable_vertices
                if vertex.is_user_placed
            },
            set(placed_vertex_ids),
        )

    def test_subdividing_direct_face_keeps_earlier_manual_vertex_markers(
        self,
    ) -> None:
        level = _build_square_level()
        _wall, outer_surface_id = _draw_closed_wall_face(level)
        outer_vertex_ids = {
            vertex.vertex_id
            for vertex in level.editable_surfaces[0].vertices
            if vertex.is_user_placed
        }

        active_vertex_id = None
        inner_result = None
        for point in (
            (0.7, 0.0, 1.1),
            (1.3, 0.0, 1.1),
            (1.3, 0.0, 2.2),
            (0.7, 0.0, 2.2),
            (0.7, 0.0, 1.1),
        ):
            inner_result = place_surface_vertex(
                [level],
                outer_surface_id,
                point,
                active_vertex_id,
            )
            active_vertex_id = inner_result.active_vertex_id
        assert inner_result is not None

        self.assertEqual(len(inner_result.created_surface_ids), 1)
        overlay = build_surface_drawing_overlay([level])
        marked_vertices = {
            vertex.vertex_id: vertex
            for vertex in overlay.vertices
            if vertex.show_marker
        }
        user_vertex_ids = {
            vertex.vertex_id
            for vertex in level.editable_surfaces[0].vertices
            if vertex.is_user_placed
        }
        self.assertEqual(set(marked_vertices), user_vertex_ids)
        self.assertTrue(outer_vertex_ids < user_vertex_ids)
        self.assertTrue(
            all(
                marked_vertices[vertex_id].direct_face_surface_ids == ()
                for vertex_id in outer_vertex_ids
            )
        )
        self.assertEqual(
            {
                surface_id
                for marker in marked_vertices.values()
                for surface_id in marker.direct_face_surface_ids
            },
            set(inner_result.created_surface_ids),
        )

    def test_clicking_existing_vertex_selects_it_without_duplicate_state(
        self,
    ) -> None:
        level = _build_square_level()
        wall = _get_wall(level)
        first = place_surface_vertex(
            [level],
            wall.surface_id,
            (0.4, 0.0, 0.8),
        )
        vertex_count = len(level.editable_surfaces[0].vertices)

        selected = place_surface_vertex(
            [level],
            wall.surface_id,
            (0.4, 0.0, 0.8),
        )

        self.assertEqual(selected.active_vertex_id, first.active_vertex_id)
        self.assertFalse(selected.state_changed)
        self.assertFalse(selected.requires_mesh_refresh)
        self.assertEqual(len(level.editable_surfaces[0].vertices), vertex_count)

    def test_preview_snaps_to_boundary_and_authored_edges_but_not_hidden_diagonal(
        self,
    ) -> None:
        level = _build_square_level()
        wall = _get_wall(level)
        first = place_surface_vertex(
            [level],
            wall.surface_id,
            (0.4, 0.0, 0.8),
        )
        place_surface_vertex(
            [level],
            wall.surface_id,
            (1.6, 0.0, 0.8),
            first.active_vertex_id,
        )

        boundary = resolve_surface_vertex_preview(
            [level],
            wall.surface_id,
            (0.005, 0.0, 1.4),
        )
        authored = resolve_surface_vertex_preview(
            [level],
            wall.surface_id,
            (1.0, 0.0, 0.805),
        )
        hidden_diagonal = resolve_surface_vertex_preview(
            [level],
            wall.surface_id,
            (1.0, 0.0, 1.82),
        )

        self.assertEqual(boundary.snap_kind, SURFACE_VERTEX_SNAP_KIND_EDGE)
        self.assertEqual(authored.snap_kind, SURFACE_VERTEX_SNAP_KIND_EDGE)
        self.assertNotEqual(hidden_diagonal.snap_kind, SURFACE_VERTEX_SNAP_KIND_EDGE)

    def test_closed_loop_crosses_source_diagonal_without_overlap_and_can_extrude(
        self,
    ) -> None:
        level = _build_square_level()
        wall = _get_wall(level)
        active_vertex_id = None
        result = None
        for point in (
            (0.4, 0.0, 0.8),
            (1.6, 0.0, 0.8),
            (1.6, 0.0, 2.5),
            (0.4, 0.0, 2.5),
            (0.4, 0.0, 0.8),
        ):
            result = place_surface_vertex(
                [level],
                wall.surface_id,
                point,
                active_vertex_id,
            )
            active_vertex_id = result.active_vertex_id
        assert result is not None

        self.assertTrue(result.requires_mesh_refresh)
        self.assertTrue(level.editable_surfaces[0].replaces_source_surface)
        self.assertEqual(len(result.created_surface_ids), 1)
        self.assertEqual(result.selected_surface_ids, result.created_surface_ids)
        editable_faces = level.editable_surfaces[0].faces
        directly_drawn_faces = tuple(
            face for face in editable_faces if face.is_directly_drawn
        )
        self.assertEqual(len(directly_drawn_faces), 1)
        self.assertEqual(
            build_editable_surface_id(
                level.index,
                directly_drawn_faces[0].face_id,
                directly_drawn_faces[0].surface_type,
            ),
            result.created_surface_ids[0],
        )
        self.assertTrue(
            all(
                not face.is_directly_drawn
                for face in editable_faces
                if face is not directly_drawn_faces[0]
            )
        )
        rebuilt = [
            surface
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
        ]
        self.assertEqual(
            {
                surface.surface_id
                for surface in rebuilt
                if surface.is_directly_drawn
            },
            set(result.created_surface_ids),
        )
        self.assertAlmostEqual(
            sum(surface.area_square_meters for surface in rebuilt),
            wall.area_square_meters,
        )
        triangle_polygons = [
            Polygon([(point[0], point[2]) for point in triangle])
            for surface in rebuilt
            for triangle in np.asarray(surface.mesh.triangles, dtype=float)
        ]
        self.assertAlmostEqual(
            unary_union(triangle_polygons).area,
            sum(polygon.area for polygon in triangle_polygons),
        )
        enclosed = next(
            surface
            for surface in rebuilt
            if surface.surface_id == result.created_surface_ids[0]
        )
        self.assertAlmostEqual(enclosed.area_square_meters, 1.2 * 1.7)

        extrusion = extrude_surface_faces(
            [level],
            result.created_surface_ids,
            0.2,
        )
        self.assertTrue(extrusion.requires_mesh_refresh)
        self.assertEqual(extrusion.selected_surface_ids, result.created_surface_ids)

    def test_directly_drawn_face_can_be_deleted_without_removing_remainder(
        self,
    ) -> None:
        level = _build_square_level()
        wall, direct_surface_id = _draw_closed_wall_face(level)
        editable_mesh = level.editable_surfaces[0]
        previous_face_count = len(editable_mesh.faces)

        result = delete_directly_drawn_surface_faces(
            [level],
            (direct_surface_id,),
        )

        rebuilt = tuple(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
        )
        self.assertEqual(result.replacements, {direct_surface_id: ()})
        self.assertEqual(result.selected_surface_ids, ())
        self.assertTrue(result.requires_mesh_refresh)
        self.assertTrue(result.state_changed)
        self.assertEqual(
            len(level.editable_surfaces[0].faces),
            previous_face_count - 1,
        )
        self.assertNotIn(
            direct_surface_id,
            {surface.surface_id for surface in rebuilt},
        )
        self.assertTrue(rebuilt)
        self.assertTrue(all(not surface.is_directly_drawn for surface in rebuilt))
        self.assertEqual(level.editable_surfaces[0].edges, ())

    def test_last_direct_face_deletion_persists_as_an_empty_replacement(
        self,
    ) -> None:
        level = _build_square_level()
        wall, direct_surface_id = _draw_full_wall_face(level)

        delete_directly_drawn_surface_faces([level], (direct_surface_id,))

        tombstone = level.editable_surfaces[0]
        self.assertTrue(tombstone.replaces_source_surface)
        self.assertEqual(tombstone.faces, ())
        self.assertEqual(tombstone.edges, ())
        self.assertNotIn(
            wall.surface_id,
            {surface.surface_id for surface in build_fixed_surfaces([level])},
        )
        self.assertEqual(build_surface_drawing_overlay([level]).vertices, ())

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "deleted-wall.housemaker"
            save_project(project_path, level.index, [level])
            loaded = load_project(project_path)

        loaded_level = next(
            item for item in loaded.levels if item.index == level.index
        )
        self.assertEqual(loaded_level.editable_surfaces, level.editable_surfaces)
        self.assertNotIn(
            wall.surface_id,
            {
                surface.surface_id
                for surface in build_fixed_surfaces([loaded_level])
            },
        )

    def test_automatically_created_surface_face_cannot_be_deleted(self) -> None:
        level = _build_square_level()
        wall, _direct_surface_id = _draw_closed_wall_face(level)
        automatic_surface_id = next(
            surface.surface_id
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
            and not surface.is_directly_drawn
        )
        previous_meshes = tuple(level.editable_surfaces)

        with self.assertRaisesRegex(ValueError, "created by Add vertex"):
            delete_directly_drawn_surface_faces(
                [level],
                (automatic_surface_id,),
            )

        self.assertEqual(tuple(level.editable_surfaces), previous_meshes)

    def test_deleting_extruded_direct_face_retains_its_side_ring(self) -> None:
        level = _build_square_level()
        _wall, direct_surface_id = _draw_closed_wall_face(level)
        before_extrusion_ids = {
            face.face_id for face in level.editable_surfaces[0].faces
        }
        extrude_surface_faces([level], (direct_surface_id,), 0.2)
        extruded_mesh = level.editable_surfaces[0]
        generated_side_ids = {
            face.face_id
            for face in extruded_mesh.faces
            if face.face_id not in before_extrusion_ids
        }
        self.assertTrue(generated_side_ids)

        delete_directly_drawn_surface_faces([level], (direct_surface_id,))

        retained_ids = {
            face.face_id for face in level.editable_surfaces[0].faces
        }
        self.assertTrue(generated_side_ids.issubset(retained_ids))

    def test_floor_and_ceiling_loops_create_extrudable_faces(self) -> None:
        for surface_type in (SURFACE_TYPE_FLOOR, SURFACE_TYPE_CEILING):
            with self.subTest(surface_type=surface_type):
                level = _build_square_level()
                source = next(
                    surface
                    for surface in build_fixed_surfaces([level])
                    if surface.surface_type == surface_type
                    and surface.room_index is None
                )
                first, second, third = np.asarray(
                    source.mesh.triangles[0],
                    dtype=float,
                )
                points = (
                    (0.6 * first) + (0.2 * second) + (0.2 * third),
                    (0.2 * first) + (0.6 * second) + (0.2 * third),
                    (0.2 * first) + (0.2 * second) + (0.6 * third),
                )
                active_vertex_id = None
                result = None
                for point in (*points, points[0]):
                    result = place_surface_vertex(
                        [level],
                        source.surface_id,
                        point,
                        active_vertex_id,
                    )
                    active_vertex_id = result.active_vertex_id
                assert result is not None

                self.assertTrue(result.requires_mesh_refresh)
                self.assertEqual(len(result.created_surface_ids), 1)
                created_surface_id = result.created_surface_ids[0]
                self.assertTrue(created_surface_id.endswith(f":{surface_type}"))
                rebuilt = tuple(
                    surface
                    for surface in build_fixed_surfaces([level])
                    if surface.source_surface_id == source.surface_id
                )
                self.assertAlmostEqual(
                    sum(surface.area_square_meters for surface in rebuilt),
                    source.area_square_meters,
                )
                cap_before = next(
                    surface
                    for surface in rebuilt
                    if surface.surface_id == created_surface_id
                )
                centroid_before = np.asarray(cap_before.mesh.centroid, dtype=float)
                normal_before = np.asarray(
                    cap_before.mesh.face_normals[0],
                    dtype=float,
                )

                extrusion = extrude_surface_faces(
                    [level],
                    (created_surface_id,),
                    0.2,
                )

                self.assertTrue(extrusion.requires_mesh_refresh)
                cap_after = next(
                    surface
                    for surface in build_fixed_surfaces([level])
                    if surface.surface_id == created_surface_id
                )
                np.testing.assert_allclose(
                    np.asarray(cap_after.mesh.centroid, dtype=float),
                    centroid_before + normal_before * 0.2,
                    atol=1e-7,
                    rtol=0.0,
                )

    def test_diagonal_intersections_do_not_create_duplicate_remainder_vertices(
        self,
    ) -> None:
        level = _build_square_level()
        wall = _get_wall(level)
        active_vertex_id = None
        result = None
        for point in (
            (0.2, 0.0, 0.6),
            (1.6, 0.0, 0.7),
            (1.5, 0.0, 2.6),
            (0.3, 0.0, 2.4),
            (0.2, 0.0, 0.6),
        ):
            result = place_surface_vertex(
                [level],
                wall.surface_id,
                point,
                active_vertex_id,
            )
            active_vertex_id = result.active_vertex_id
        assert result is not None

        rebuilt = [
            surface
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
        ]
        self.assertEqual(len(result.created_surface_ids), 1)
        self.assertAlmostEqual(
            sum(surface.area_square_meters for surface in rebuilt),
            wall.area_square_meters,
        )


# ### Extrusion tests ###
class SurfaceFaceExtrusionTests(unittest.TestCase):
    def test_connected_face_extrusion_keeps_caps_and_adds_boundary_sides(self) -> None:
        level = _build_square_level()
        source_id, _selected_children = _insert_into_first_wall_triangle(level)
        source_edit = level.editable_surfaces[0]
        cap_ids = tuple(
            f"level:2/edit-face:{face.face_id}:{face.surface_type}"
            for face in source_edit.faces
        )
        surfaces_before = {
            surface.surface_id: surface for surface in build_fixed_surfaces([level])
        }

        result = extrude_surface_faces([level], cap_ids, 0.25)

        self.assertEqual(result.replacements, {})
        self.assertEqual(result.selected_surface_ids, cap_ids)
        surfaces_after = {
            surface.surface_id: surface for surface in build_fixed_surfaces([level])
        }
        self.assertEqual(len(level.editable_surfaces[0].faces), len(cap_ids) + 4)
        self.assertEqual(
            len(
                [
                    surface
                    for surface in surfaces_after.values()
                    if surface.source_surface_id == source_id
                ]
            ),
            len(cap_ids) + 4,
        )
        for cap_id in cap_ids:
            self.assertIn(cap_id, surfaces_after)
            before = np.asarray(surfaces_before[cap_id].mesh.centroid, dtype=float)
            after = np.asarray(surfaces_after[cap_id].mesh.centroid, dtype=float)
            self.assertAlmostEqual(float(np.linalg.norm(after - before)), 0.25)

    def test_connected_multi_face_extrusion_reverses_as_one_region(self) -> None:
        level = _build_square_level()
        _source_id, _selected_children = _insert_into_first_wall_triangle(level)
        baseline_mesh = level.editable_surfaces[0]
        cap_ids = tuple(
            build_editable_surface_id(
                level.index,
                face.face_id,
                face.surface_type,
            )
            for face in baseline_mesh.faces
        )

        extrude_surface_faces([level], cap_ids, 0.25)
        result = extrude_surface_faces([level], cap_ids, -0.25)

        self.assertEqual(level.editable_surfaces[0], baseline_mesh)
        self.assertEqual(result.selected_surface_ids, cap_ids)
        self.assertEqual(len(result.replacements), 4)

    def test_equal_opposite_extrusion_restores_closed_face_topology(self) -> None:
        level = _build_square_level()
        wall, cap_id = _draw_closed_wall_face(level)
        baseline_mesh = level.editable_surfaces[0]
        baseline_face_ids = {face.face_id for face in baseline_mesh.faces}
        baseline_surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
        }
        baseline_area = sum(
            surface.area_square_meters for surface in baseline_surfaces.values()
        )

        extrude_surface_faces([level], (cap_id,), 0.2)
        extruded_mesh = level.editable_surfaces[0]
        created_side_ids = tuple(
            sorted(
                build_editable_surface_id(
                    level.index,
                    face.face_id,
                    face.surface_type,
                )
                for face in extruded_mesh.faces
                if face.face_id not in baseline_face_ids
            )
        )
        self.assertEqual(len(created_side_ids), 4)

        result = extrude_surface_faces([level], (cap_id,), -0.2)

        self.assertEqual(level.editable_surfaces[0], baseline_mesh)
        self.assertEqual(result.selected_surface_ids, (cap_id,))
        self.assertEqual(
            result.replacements,
            {surface_id: () for surface_id in created_side_ids},
        )
        restored_surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
        }
        self.assertEqual(set(restored_surfaces), set(baseline_surfaces))
        self.assertAlmostEqual(
            sum(
                surface.area_square_meters
                for surface in restored_surfaces.values()
            ),
            baseline_area,
        )

    def test_partial_opposite_extrusion_contracts_existing_side_ring(self) -> None:
        level = _build_square_level()
        wall, cap_id = _draw_closed_wall_face(level)
        baseline_mesh = level.editable_surfaces[0]
        baseline_face_ids = {face.face_id for face in baseline_mesh.faces}
        baseline_surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
        }
        baseline_cap = baseline_surfaces[cap_id]
        baseline_area = sum(
            surface.area_square_meters for surface in baseline_surfaces.values()
        )
        boundary_length = _mesh_boundary_length(baseline_cap.mesh)
        cap_normal = np.asarray(baseline_cap.mesh.face_normals[0], dtype=float)

        extrude_surface_faces([level], (cap_id,), 0.25)
        extruded_mesh = level.editable_surfaces[0]
        extruded_vertex_ids = tuple(
            vertex.vertex_id for vertex in extruded_mesh.vertices
        )
        extruded_face_ids = tuple(face.face_id for face in extruded_mesh.faces)
        extruded_edges = extruded_mesh.edges
        created_side_ids = {
            build_editable_surface_id(
                level.index,
                face.face_id,
                face.surface_type,
            )
            for face in extruded_mesh.faces
            if face.face_id not in baseline_face_ids
        }
        self.assertEqual(len(created_side_ids), 4)

        result = extrude_surface_faces([level], (cap_id,), -0.1)

        contracted_mesh = level.editable_surfaces[0]
        self.assertEqual(
            tuple(vertex.vertex_id for vertex in contracted_mesh.vertices),
            extruded_vertex_ids,
        )
        self.assertEqual(
            tuple(face.face_id for face in contracted_mesh.faces),
            extruded_face_ids,
        )
        self.assertEqual(contracted_mesh.edges, extruded_edges)
        self.assertEqual(result.replacements, {})
        self.assertEqual(result.selected_surface_ids, (cap_id,))
        contracted_surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
        }
        self.assertTrue(created_side_ids.issubset(contracted_surfaces))
        np.testing.assert_allclose(
            np.asarray(contracted_surfaces[cap_id].mesh.centroid, dtype=float),
            np.asarray(baseline_cap.mesh.centroid, dtype=float)
            + cap_normal * 0.15,
            atol=1e-7,
            rtol=0.0,
        )
        self.assertAlmostEqual(
            sum(
                surface.area_square_meters
                for surface in contracted_surfaces.values()
            ),
            baseline_area + boundary_length * 0.15,
        )

    def test_opposite_extrusion_crosses_the_base_with_only_one_side_ring(
        self,
    ) -> None:
        level = _build_square_level()
        wall, cap_id = _draw_closed_wall_face(level)
        baseline_mesh = level.editable_surfaces[0]
        baseline_face_ids = {face.face_id for face in baseline_mesh.faces}
        baseline_surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
        }
        baseline_cap = baseline_surfaces[cap_id]
        baseline_area = sum(
            surface.area_square_meters for surface in baseline_surfaces.values()
        )
        boundary_length = _mesh_boundary_length(baseline_cap.mesh)
        cap_normal = np.asarray(baseline_cap.mesh.face_normals[0], dtype=float)

        extrude_surface_faces([level], (cap_id,), 0.2)
        extruded_mesh = level.editable_surfaces[0]
        first_ring_faces = tuple(
            face
            for face in extruded_mesh.faces
            if face.face_id not in baseline_face_ids
        )
        first_ring_face_ids = {face.face_id for face in first_ring_faces}
        first_ring_surface_ids = {
            build_editable_surface_id(
                level.index,
                face.face_id,
                face.surface_type,
            )
            for face in first_ring_faces
        }

        result = extrude_surface_faces([level], (cap_id,), -0.5)

        crossed_mesh = level.editable_surfaces[0]
        self.assertEqual(
            len(crossed_mesh.vertices),
            len(extruded_mesh.vertices),
        )
        self.assertEqual(
            len(crossed_mesh.faces),
            len(extruded_mesh.faces),
        )
        crossed_ring_face_ids = {
            face.face_id
            for face in crossed_mesh.faces
            if face.face_id not in baseline_face_ids
        }
        self.assertEqual(len(crossed_ring_face_ids), 4)
        self.assertTrue(first_ring_face_ids.isdisjoint(crossed_ring_face_ids))
        self.assertEqual(
            result.replacements,
            {surface_id: () for surface_id in first_ring_surface_ids},
        )
        crossed_surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == wall.surface_id
        }
        np.testing.assert_allclose(
            np.asarray(crossed_surfaces[cap_id].mesh.centroid, dtype=float),
            np.asarray(baseline_cap.mesh.centroid, dtype=float)
            - cap_normal * 0.3,
            atol=1e-7,
            rtol=0.0,
        )
        self.assertAlmostEqual(
            sum(surface.area_square_meters for surface in crossed_surfaces.values()),
            baseline_area + boundary_length * 0.3,
        )

    def test_same_direction_extrusion_resizes_one_ring_with_stable_ids(
        self,
    ) -> None:
        level = _build_square_level()
        _wall, cap_id = _draw_closed_wall_face(level)
        baseline_mesh = level.editable_surfaces[0]
        baseline_face_ids = {face.face_id for face in baseline_mesh.faces}
        baseline_cap = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_id == cap_id
        )
        cap_normal = np.asarray(baseline_cap.mesh.face_normals[0], dtype=float)

        extrude_surface_faces([level], (cap_id,), 0.2)
        first_layer_mesh = level.editable_surfaces[0]
        first_vertex_ids = tuple(
            vertex.vertex_id for vertex in first_layer_mesh.vertices
        )
        first_face_ids = tuple(face.face_id for face in first_layer_mesh.faces)
        first_ring_faces = tuple(
            face
            for face in first_layer_mesh.faces
            if face.face_id not in baseline_face_ids
        )

        result = extrude_surface_faces([level], (cap_id,), 0.1)

        second_layer_mesh = level.editable_surfaces[0]
        self.assertEqual(
            tuple(vertex.vertex_id for vertex in second_layer_mesh.vertices),
            first_vertex_ids,
        )
        self.assertEqual(
            tuple(face.face_id for face in second_layer_mesh.faces),
            first_face_ids,
        )
        self.assertEqual(len(first_ring_faces), 4)
        self.assertEqual(result.replacements, {})
        self.assertEqual(result.selected_surface_ids, (cap_id,))
        self.assertTrue(
            next(
                face
                for face in second_layer_mesh.faces
                if build_editable_surface_id(
                    level.index,
                    face.face_id,
                    face.surface_type,
                )
                == cap_id
            ).is_directly_drawn
        )
        self.assertTrue(
            all(not face.is_directly_drawn for face in first_ring_faces)
        )
        resized_cap = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_id == cap_id
        )
        np.testing.assert_allclose(
            np.asarray(resized_cap.mesh.centroid, dtype=float),
            np.asarray(baseline_cap.mesh.centroid, dtype=float)
            + cap_normal * 0.3,
            atol=1e-7,
            rtol=0.0,
        )

    def test_subdivided_side_ring_resizes_collapses_and_crosses_cleanly(
        self,
    ) -> None:
        cases = (
            ("extend", 0.1, 0.4, "resize"),
            ("contract", -0.1, 0.2, "resize"),
            ("collapse", -0.3, 0.0, "collapse"),
            ("cross", -0.5, -0.2, "cross"),
        )
        for case_name, delta_meters, expected_depth, outcome in cases:
            with self.subTest(case=case_name):
                level = _build_square_level()
                wall, cap_id = _draw_closed_wall_face(level)
                baseline_mesh = level.editable_surfaces[0]
                baseline_face_ids = {
                    face.face_id for face in baseline_mesh.faces
                }
                baseline_surfaces = {
                    surface.surface_id: surface
                    for surface in build_fixed_surfaces([level])
                    if surface.source_surface_id == wall.surface_id
                }
                baseline_cap = baseline_surfaces[cap_id]
                baseline_area = sum(
                    surface.area_square_meters
                    for surface in baseline_surfaces.values()
                )
                boundary_length = _mesh_boundary_length(baseline_cap.mesh)
                cap_normal = np.asarray(
                    baseline_cap.mesh.face_normals[0],
                    dtype=float,
                )

                extrude_surface_faces([level], (cap_id,), 0.3)
                inner_surface_id = _draw_face_inside_generated_floor_side(
                    level,
                    baseline_face_ids,
                )
                subdivided_mesh = level.editable_surfaces[0]
                subdivided_vertex_ids = tuple(
                    vertex.vertex_id for vertex in subdivided_mesh.vertices
                )
                subdivided_face_ids = tuple(
                    face.face_id for face in subdivided_mesh.faces
                )
                side_descendants = tuple(
                    face
                    for face in subdivided_mesh.faces
                    if face.face_id not in baseline_face_ids
                )
                side_surface_ids = {
                    build_editable_surface_id(
                        level.index,
                        face.face_id,
                        face.surface_type,
                    )
                    for face in side_descendants
                }
                inner_face = next(
                    face
                    for face in side_descendants
                    if build_editable_surface_id(
                        level.index,
                        face.face_id,
                        face.surface_type,
                    )
                    == inner_surface_id
                )
                inner_vertex_ids = set(inner_face.vertex_ids)

                result = extrude_surface_faces(
                    [level],
                    (cap_id,),
                    delta_meters,
                )
                result_mesh = level.editable_surfaces[0]
                result_surfaces = {
                    surface.surface_id: surface
                    for surface in build_fixed_surfaces([level])
                    if surface.source_surface_id == wall.surface_id
                }

                if outcome == "resize":
                    self.assertEqual(
                        tuple(
                            vertex.vertex_id for vertex in result_mesh.vertices
                        ),
                        subdivided_vertex_ids,
                    )
                    self.assertEqual(
                        tuple(face.face_id for face in result_mesh.faces),
                        subdivided_face_ids,
                    )
                    self.assertIn(inner_surface_id, result_surfaces)
                    self.assertEqual(result.replacements, {})
                    floor_triangles = tuple(
                        Polygon(triangle[:, :2])
                        for surface in result_surfaces.values()
                        if surface.surface_type == SURFACE_TYPE_FLOOR
                        for triangle in np.asarray(
                            surface.mesh.triangles,
                            dtype=float,
                        )
                    )
                    self.assertTrue(floor_triangles)
                    self.assertAlmostEqual(
                        sum(polygon.area for polygon in floor_triangles),
                        unary_union(floor_triangles).area,
                    )
                elif outcome == "collapse":
                    self.assertEqual(result_mesh, baseline_mesh)
                    self.assertTrue(
                        inner_vertex_ids.isdisjoint(
                            vertex.vertex_id for vertex in result_mesh.vertices
                        )
                    )
                    self.assertEqual(
                        result.replacements,
                        {surface_id: () for surface_id in sorted(side_surface_ids)},
                    )
                else:
                    fresh_ring = tuple(
                        face
                        for face in result_mesh.faces
                        if face.face_id not in baseline_face_ids
                    )
                    self.assertEqual(len(fresh_ring), 4)
                    self.assertTrue(
                        side_surface_ids.isdisjoint(
                            build_editable_surface_id(
                                level.index,
                                face.face_id,
                                face.surface_type,
                            )
                            for face in fresh_ring
                        )
                    )
                    self.assertTrue(
                        inner_vertex_ids.isdisjoint(
                            vertex.vertex_id for vertex in result_mesh.vertices
                        )
                    )
                    self.assertEqual(
                        len(result_mesh.vertices),
                        len(baseline_mesh.vertices) + 4,
                    )
                    self.assertEqual(
                        len(result_mesh.faces),
                        len(baseline_mesh.faces) + 4,
                    )

                expected_cap_center = (
                    np.asarray(baseline_cap.mesh.centroid, dtype=float)
                    + cap_normal * expected_depth
                )
                np.testing.assert_allclose(
                    np.asarray(result_surfaces[cap_id].mesh.centroid, dtype=float),
                    expected_cap_center,
                    atol=1e-7,
                    rtol=0.0,
                )
                self.assertAlmostEqual(
                    sum(
                        surface.area_square_meters
                        for surface in result_surfaces.values()
                    ),
                    baseline_area + boundary_length * abs(expected_depth),
                )

    def test_unfinished_side_chain_scales_with_the_existing_extrusion_ring(
        self,
    ) -> None:
        level = _build_square_level()
        _wall, cap_id = _draw_closed_wall_face(level)
        baseline_mesh = level.editable_surfaces[0]
        baseline_face_ids = {face.face_id for face in baseline_mesh.faces}
        baseline_cap = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_id == cap_id
        )
        cap_normal = np.asarray(baseline_cap.mesh.face_normals[0], dtype=float)
        base_origin = np.asarray(baseline_cap.mesh.centroid, dtype=float)

        extrude_surface_faces([level], (cap_id,), 0.3)
        chain_vertex_ids = _start_chain_inside_generated_floor_side(
            level,
            baseline_face_ids,
        )
        chain_mesh = level.editable_surfaces[0]
        vertex_ids_before = tuple(
            vertex.vertex_id for vertex in chain_mesh.vertices
        )
        face_ids_before = tuple(face.face_id for face in chain_mesh.faces)
        edges_before = chain_mesh.edges
        points_before = {
            vertex.vertex_id: np.asarray(vertex.world_point, dtype=float)
            for vertex in build_surface_drawing_overlay([level]).vertices
            if vertex.vertex_id in chain_vertex_ids
        }

        result = extrude_surface_faces([level], (cap_id,), 0.1)

        resized_mesh = level.editable_surfaces[0]
        self.assertEqual(
            tuple(vertex.vertex_id for vertex in resized_mesh.vertices),
            vertex_ids_before,
        )
        self.assertEqual(
            tuple(face.face_id for face in resized_mesh.faces),
            face_ids_before,
        )
        self.assertEqual(resized_mesh.edges, edges_before)
        self.assertEqual(result.replacements, {})
        points_after = {
            vertex.vertex_id: np.asarray(vertex.world_point, dtype=float)
            for vertex in build_surface_drawing_overlay([level]).vertices
            if vertex.vertex_id in chain_vertex_ids
        }
        self.assertEqual(set(points_after), set(chain_vertex_ids))
        for vertex_id in chain_vertex_ids:
            old_point = points_before[vertex_id]
            new_point = points_after[vertex_id]
            old_depth = float(np.dot(old_point - base_origin, cap_normal))
            new_depth = float(np.dot(new_point - base_origin, cap_normal))
            self.assertAlmostEqual(new_depth, old_depth * (0.4 / 0.3))
            displacement = new_point - old_point
            np.testing.assert_allclose(
                displacement,
                cap_normal * float(np.dot(displacement, cap_normal)),
                atol=1e-7,
                rtol=0.0,
            )

    def test_unfinished_side_chain_is_removed_on_collapse_and_crossing(
        self,
    ) -> None:
        for case_name, delta_meters in (("collapse", -0.3), ("cross", -0.5)):
            with self.subTest(case=case_name):
                level = _build_square_level()
                _wall, cap_id = _draw_closed_wall_face(level)
                baseline_mesh = level.editable_surfaces[0]
                baseline_face_ids = {
                    face.face_id for face in baseline_mesh.faces
                }
                extrude_surface_faces([level], (cap_id,), 0.3)
                chain_vertex_ids = set(
                    _start_chain_inside_generated_floor_side(
                        level,
                        baseline_face_ids,
                    )
                )

                extrude_surface_faces([level], (cap_id,), delta_meters)

                result_mesh = level.editable_surfaces[0]
                self.assertTrue(
                    chain_vertex_ids.isdisjoint(
                        vertex.vertex_id for vertex in result_mesh.vertices
                    )
                )
                self.assertTrue(
                    all(
                        edge.start_vertex_id not in chain_vertex_ids
                        and edge.end_vertex_id not in chain_vertex_ids
                        for edge in result_mesh.edges
                    )
                )
                if case_name == "collapse":
                    self.assertEqual(result_mesh, baseline_mesh)
                else:
                    self.assertEqual(
                        len(result_mesh.vertices),
                        len(baseline_mesh.vertices) + 4,
                    )
                    self.assertEqual(
                        len(result_mesh.faces),
                        len(baseline_mesh.faces) + 4,
                    )

    def test_cap_adjacent_quad_subdivision_reuses_the_existing_side_ring(
        self,
    ) -> None:
        level = _build_square_level()
        _wall, cap_id = _draw_closed_wall_face(level)
        baseline_mesh = level.editable_surfaces[0]
        baseline_face_ids = {face.face_id for face in baseline_mesh.faces}
        baseline_cap = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_id == cap_id
        )
        cap_normal = np.asarray(baseline_cap.mesh.face_normals[0], dtype=float)

        extrude_surface_faces([level], (cap_id,), 0.3)
        child_surface_id = _draw_cap_adjacent_face_on_generated_floor_side(
            level,
            cap_id,
            baseline_face_ids,
        )
        subdivided_mesh = level.editable_surfaces[0]
        cap_match = EDITABLE_SURFACE_ID_PATTERN.fullmatch(cap_id)
        child_match = EDITABLE_SURFACE_ID_PATTERN.fullmatch(child_surface_id)
        assert cap_match is not None
        assert child_match is not None
        cap_face = next(
            face
            for face in subdivided_mesh.faces
            if face.face_id == cap_match.group("face_id")
        )
        child_face = next(
            face
            for face in subdivided_mesh.faces
            if face.face_id == child_match.group("face_id")
        )
        self.assertEqual(len(child_face.vertex_ids), 4)
        self.assertEqual(
            len(set(child_face.vertex_ids) & set(cap_face.vertex_ids)),
            2,
        )
        vertex_ids_before = tuple(
            vertex.vertex_id for vertex in subdivided_mesh.vertices
        )
        face_ids_before = tuple(face.face_id for face in subdivided_mesh.faces)

        result = extrude_surface_faces([level], (cap_id,), 0.1)

        resized_mesh = level.editable_surfaces[0]
        self.assertEqual(
            tuple(vertex.vertex_id for vertex in resized_mesh.vertices),
            vertex_ids_before,
        )
        self.assertEqual(
            tuple(face.face_id for face in resized_mesh.faces),
            face_ids_before,
        )
        self.assertEqual(result.replacements, {})
        self.assertIn(
            child_surface_id,
            {
                surface.surface_id
                for surface in build_fixed_surfaces([level])
            },
        )
        resized_cap = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_id == cap_id
        )
        np.testing.assert_allclose(
            np.asarray(resized_cap.mesh.centroid, dtype=float),
            np.asarray(baseline_cap.mesh.centroid, dtype=float)
            + cap_normal * 0.4,
            atol=1e-7,
            rtol=0.0,
        )

    def test_extrusion_moves_manual_markers_instead_of_duplicating_them(
        self,
    ) -> None:
        level = _build_square_level()
        _wall, cap_id = _draw_closed_wall_face(level)

        def marked_vertices() -> dict[str, SurfaceDrawingVertexTarget]:
            return {
                vertex.vertex_id: vertex
                for vertex in build_surface_drawing_overlay([level]).vertices
                if vertex.show_marker
            }

        baseline_markers = marked_vertices()
        self.assertEqual(len(baseline_markers), 4)

        extrude_surface_faces([level], (cap_id,), 0.2)
        extruded_markers = marked_vertices()

        self.assertEqual(len(extruded_markers), 4)
        self.assertTrue(set(baseline_markers).isdisjoint(extruded_markers))
        self.assertTrue(
            all(
                marker.direct_face_surface_ids == (cap_id,)
                for marker in extruded_markers.values()
            )
        )

        extrude_surface_faces([level], (cap_id,), 0.1)
        resized_markers = marked_vertices()
        self.assertEqual(set(resized_markers), set(extruded_markers))

        extrude_surface_faces([level], (cap_id,), -0.3)
        self.assertEqual(marked_vertices(), baseline_markers)

        extrude_surface_faces([level], (cap_id,), -0.2)
        crossed_markers = marked_vertices()
        self.assertEqual(len(crossed_markers), 4)
        self.assertTrue(set(baseline_markers).isdisjoint(crossed_markers))
        self.assertTrue(
            all(
                marker.direct_face_surface_ids == (cap_id,)
                for marker in crossed_markers.values()
            )
        )

    def test_opposite_extrusion_is_reversible_after_project_reload(self) -> None:
        level = _build_square_level()
        _wall, cap_id = _draw_closed_wall_face(level)
        baseline_mesh = level.editable_surfaces[0]
        extrude_surface_faces([level], (cap_id,), -0.2)

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "extruded.housemaker"
            save_project(project_path, level.index, [level])
            loaded = load_project(project_path)

        loaded_level = next(
            item for item in loaded.levels if item.index == level.index
        )
        result = extrude_surface_faces([loaded_level], (cap_id,), 0.2)

        self.assertEqual(loaded_level.editable_surfaces[0], baseline_mesh)
        self.assertEqual(result.selected_surface_ids, (cap_id,))
        self.assertEqual(len(result.replacements), 4)
        self.assertTrue(all(not children for children in result.replacements.values()))

    def test_wall_frame_ratios_follow_later_wall_and_height_changes(self) -> None:
        level = _build_square_level()
        source_id, _selected_children = _insert_into_first_wall_triangle(level)
        edited_ids = {
            surface.surface_id
            for surface in build_fixed_surfaces([level])
            if surface.source_surface_id == source_id
        }

        level.vertex_data.move_vertex(2, 150.0, 0.0)
        level.height_meters = 4.0
        rebuilt = build_fixed_surfaces([level])

        self.assertEqual(
            edited_ids,
            {
                surface.surface_id
                for surface in rebuilt
                if surface.source_surface_id == source_id
            },
        )
        edited_wall_bounds = np.asarray(
            [
                surface.mesh.bounds
                for surface in rebuilt
                if surface.source_surface_id == source_id
            ]
        )
        self.assertAlmostEqual(float(edited_wall_bounds[:, 1, 0].max()), 3.0)
        self.assertAlmostEqual(
            float(edited_wall_bounds[:, 1, 2].max()),
            level.floor_thickness_meters + level.height_meters,
        )


# ### Persistence tests ###
class SurfaceTopologyPersistenceTests(unittest.TestCase):
    def test_editable_surface_graph_round_trips_with_project(self) -> None:
        level = _build_square_level()
        _insert_into_first_wall_triangle(level)

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "edited.housemaker"
            save_project(project_path, level.index, [level])

            loaded = load_project(project_path)

        loaded_level = next(item for item in loaded.levels if item.index == level.index)
        self.assertEqual(loaded_level.editable_surfaces, level.editable_surfaces)
        self.assertEqual(
            {surface.surface_id for surface in build_fixed_surfaces([loaded_level])},
            {surface.surface_id for surface in build_fixed_surfaces([level])},
        )

    def test_manual_vertex_and_direct_face_provenance_round_trips(self) -> None:
        level = _build_square_level()
        _wall, directly_drawn_surface_id = _draw_closed_wall_face(level)
        original_mesh = level.editable_surfaces[0]
        original_user_vertex_ids = {
            vertex.vertex_id
            for vertex in original_mesh.vertices
            if vertex.is_user_placed
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "provenance.housemaker"
            save_project(project_path, level.index, [level])
            loaded = load_project(project_path)

        loaded_level = next(item for item in loaded.levels if item.index == level.index)
        loaded_mesh = loaded_level.editable_surfaces[0]
        self.assertEqual(
            {
                vertex.vertex_id
                for vertex in loaded_mesh.vertices
                if vertex.is_user_placed
            },
            original_user_vertex_ids,
        )
        self.assertEqual(
            {
                build_editable_surface_id(
                    loaded_level.index,
                    face.face_id,
                    face.surface_type,
                )
                for face in loaded_mesh.faces
                if face.is_directly_drawn
            },
            {directly_drawn_surface_id},
        )
        self.assertEqual(
            {
                surface.surface_id
                for surface in build_fixed_surfaces([loaded_level])
                if surface.is_directly_drawn
            },
            {directly_drawn_surface_id},
        )

    def test_open_vertex_chain_round_trips_without_retiring_source(self) -> None:
        level = _build_square_level()
        wall = _get_wall(level)
        first = place_surface_vertex(
            [level],
            wall.surface_id,
            (0.4, 0.0, 0.8),
        )
        place_surface_vertex(
            [level],
            wall.surface_id,
            (1.6, 0.0, 0.8),
            first.active_vertex_id,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "open-chain.housemaker"
            save_project(project_path, level.index, [level])
            loaded = load_project(project_path)

        loaded_level = next(item for item in loaded.levels if item.index == level.index)
        self.assertEqual(loaded_level.editable_surfaces, level.editable_surfaces)
        self.assertFalse(loaded_level.editable_surfaces[0].replaces_source_surface)
        self.assertEqual(len(loaded_level.editable_surfaces[0].edges), 1)
        self.assertIn(
            wall.surface_id,
            {surface.surface_id for surface in build_fixed_surfaces([loaded_level])},
        )

    def test_loading_tolerates_missing_or_malformed_optional_edit_records(
        self,
    ) -> None:
        level = _build_square_level()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "legacy.housemaker"
            save_project(project_path, level.index, [level])
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["levels"][0].pop("editable_surfaces")
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded_without_field = load_project(project_path)

            payload["levels"][0]["editable_surfaces"] = [
                None,
                {"source_surface_id": "wrong"},
            ]
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            loaded_malformed = load_project(project_path)

        self.assertEqual(
            next(
                item
                for item in loaded_without_field.levels
                if item.index == level.index
            ).editable_surfaces,
            [],
        )
        self.assertEqual(
            next(
                item for item in loaded_malformed.levels if item.index == level.index
            ).editable_surfaces,
            [],
        )


if __name__ == "__main__":
    unittest.main()
