# ### Imports ###
from __future__ import annotations

from io import BytesIO
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from housemaker.glb import convert_to_glb
from housemaker.models import (
    DoorwayData,
    LevelData,
    RoomData,
    VertexData,
    WindowData,
)
from housemaker.project_io import load_project, save_project
from housemaker.surface_geometry import (
    WallWindowPlacement,
    add_wall_window,
    build_fixed_surfaces,
    build_wall_window_placement,
    get_wall_window_world_corners,
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
    center = vertex_data.add_vertex(50.0, 50.0)
    room = RoomData(
        name="Living room",
        vertex_ids=boundary_ids,
        center_vertex_id=center.id,
        color_rgb=(140, 180, 220),
    )
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[room],
        floor_contour_vertex_ids=boundary_ids,
    )


def _get_window_wall(level: LevelData):
    return next(
        surface
        for surface in build_fixed_surfaces([level])
        if surface.surface_id == "level:2/room:5/wall:1:2"
    )


def _build_adjacent_room_level() -> tuple[LevelData, str]:
    vertex_data = VertexData()
    bottom_left = vertex_data.add_vertex(0.0, 0.0)
    shared_bottom = vertex_data.add_vertex(100.0, 0.0)
    shared_top = vertex_data.add_vertex(100.0, 100.0)
    top_left = vertex_data.add_vertex(0.0, 100.0)
    bottom_right = vertex_data.add_vertex(200.0, 0.0)
    top_right = vertex_data.add_vertex(200.0, 100.0)
    for start_id, end_id in (
        (bottom_left.id, shared_bottom.id),
        (shared_bottom.id, shared_top.id),
        (shared_top.id, top_left.id),
        (top_left.id, bottom_left.id),
        (shared_bottom.id, bottom_right.id),
        (bottom_right.id, top_right.id),
        (top_right.id, shared_top.id),
    ):
        vertex_data.add_edge(start_id, end_id)
    first_center = vertex_data.add_vertex(50.0, 50.0)
    second_center = vertex_data.add_vertex(150.0, 50.0)
    rooms = [
        RoomData(
            name="Left",
            vertex_ids=(1, 2, 3, 4),
            center_vertex_id=first_center.id,
            color_rgb=(150, 170, 210),
        ),
        RoomData(
            name="Right",
            vertex_ids=(2, 5, 6, 3),
            center_vertex_id=second_center.id,
            color_rgb=(210, 170, 150),
        ),
    ]
    return (
        LevelData(
            index=2,
            name="Ground",
            vertex_data=vertex_data,
            rooms=rooms,
        ),
        "2:3",
    )


def _build_window_depth_level(
) -> tuple[LevelData, tuple[str, str, str, str]]:
    level = _build_square_level()
    level.scale = 2.0
    parallel_wall_ids: list[str] = []
    for image_y in (-12.5, -12.75, 12.5):
        start = level.vertex_data.add_vertex(0.0, image_y)
        end = level.vertex_data.add_vertex(100.0, image_y)
        level.vertex_data.add_edge(start.id, end.id)
        parallel_wall_ids.append(
            f"level:2/wall:{min(start.id, end.id)}:{max(start.id, end.id)}"
        )
    perpendicular_start = level.vertex_data.add_vertex(50.0, -5.0)
    perpendicular_end = level.vertex_data.add_vertex(50.0, -20.0)
    level.vertex_data.add_edge(perpendicular_start.id, perpendicular_end.id)
    parallel_wall_ids.append(
        "level:2/wall:"
        f"{min(perpendicular_start.id, perpendicular_end.id)}:"
        f"{max(perpendicular_start.id, perpendicular_end.id)}"
    )
    return level, tuple(parallel_wall_ids)


def _build_plain_window_depth_level(
) -> tuple[LevelData, tuple[str, str, str]]:
    vertex_data = VertexData()
    wall_ids: list[str] = []
    for image_y in (0.0, -12.5, 12.5):
        start = vertex_data.add_vertex(0.0, image_y)
        end = vertex_data.add_vertex(100.0, image_y)
        vertex_data.add_edge(start.id, end.id)
        wall_ids.append(
            f"level:2/wall:{min(start.id, end.id)}:{max(start.id, end.id)}"
        )
    return (
        LevelData(
            index=2,
            name="Plain walls",
            vertex_data=vertex_data,
            scale=2.0,
        ),
        tuple(wall_ids),
    )


def _get_wall_midpoint(surface) -> tuple[float, float, float]:
    assert surface.wall_start_world is not None
    assert surface.wall_end_world is not None
    start = np.asarray(surface.wall_start_world, dtype=float)
    end = np.asarray(surface.wall_end_world, dtype=float)
    midpoint = (start + end) / 2.0
    midpoint[2] = 1.5
    return tuple(float(value) for value in midpoint)


def _get_wall_ratio_point(
    surface,
    horizontal_ratio: float,
    height_meters: float,
) -> np.ndarray:
    assert surface.wall_start_world is not None
    assert surface.wall_end_world is not None
    start = np.asarray(surface.wall_start_world, dtype=float)
    end = np.asarray(surface.wall_end_world, dtype=float)
    point = start + (end - start) * horizontal_ratio
    point[2] = height_meters
    return point


def _mesh_covers_point_on_plane(
    mesh: trimesh.Trimesh,
    point: tuple[float, float, float],
    fixed_axis: int,
) -> bool:
    return bool(
        _count_mesh_triangles_covering_point_on_plane(
            mesh,
            point,
            fixed_axis,
        )
    )


def _count_mesh_triangles_covering_point_on_plane(
    mesh: trimesh.Trimesh,
    point: tuple[float, float, float],
    fixed_axis: int,
) -> int:
    point_array = np.asarray(point, dtype=float)
    plane_axes = [axis for axis in range(3) if axis != fixed_axis]
    covering_triangle_count = 0
    for triangle in np.asarray(mesh.triangles, dtype=float):
        if not np.allclose(
            triangle[:, fixed_axis],
            point_array[fixed_axis],
            atol=1e-6,
        ):
            continue
        if _point_is_in_triangle_2d(
            point_array[plane_axes],
            triangle[:, plane_axes],
        ):
            covering_triangle_count += 1
    return covering_triangle_count


def _point_is_in_triangle_2d(
    point: np.ndarray,
    triangle: np.ndarray,
) -> bool:
    first, second, third = triangle
    denominator = (
        (second[1] - third[1]) * (first[0] - third[0])
        + (third[0] - second[0]) * (first[1] - third[1])
    )
    if abs(float(denominator)) <= 1e-9:
        return False
    first_weight = (
        (second[1] - third[1]) * (point[0] - third[0])
        + (third[0] - second[0]) * (point[1] - third[1])
    ) / denominator
    second_weight = (
        (third[1] - first[1]) * (point[0] - third[0])
        + (first[0] - third[0]) * (point[1] - third[1])
    ) / denominator
    third_weight = 1.0 - first_weight - second_weight
    return min(first_weight, second_weight, third_weight) >= -1e-6


def _load_glb_world_mesh(glb_bytes: bytes) -> trimesh.Trimesh:
    loaded = trimesh.load(BytesIO(glb_bytes), file_type="glb")
    if isinstance(loaded, trimesh.Scene):
        return loaded.to_geometry()
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise AssertionError("GLB did not load as a mesh or scene.")


def _solid_png() -> bytes:
    output = BytesIO()
    Image.new("RGBA", (8, 8), (160, 110, 80, 255)).save(
        output,
        format="PNG",
    )
    return output.getvalue()


# ### Window model and persistence tests ###
class WallWindowStateTests(unittest.TestCase):
    def test_window_data_validates_normalized_wall_local_bounds(self) -> None:
        window = WindowData(
            window_id="window-1",
            wall_surface_id="level:2/room:5/wall:1:2",
            start_ratio=0.2,
            end_ratio=0.6,
            bottom_ratio=0.25,
            top_ratio=0.75,
        )

        self.assertEqual(
            WindowData.from_dict(window.to_dict()),
            window,
        )
        with self.assertRaises(ValueError):
            WindowData(
                window_id="bad",
                wall_surface_id="level:2/floor",
                start_ratio=0.2,
                end_ratio=0.6,
                bottom_ratio=0.25,
                top_ratio=0.75,
            )
        with self.assertRaises(ValueError):
            WindowData(
                window_id="bad",
                wall_surface_id="level:2/room:5/wall:1:2",
                start_ratio=0.6,
                end_ratio=0.2,
                bottom_ratio=0.25,
                top_ratio=0.75,
            )

    def test_project_round_trip_persists_windows_and_skips_invalid_data(
        self,
    ) -> None:
        level = _build_square_level()
        level.windows = [
            WindowData(
                window_id="window-1",
                wall_surface_id="level:2/room:5/wall:1:2",
                start_ratio=0.2,
                end_ratio=0.6,
                bottom_ratio=0.25,
                top_ratio=0.75,
            )
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "windows.json"
            save_project(project_path, 2, [level])

            loaded = load_project(project_path)
            loaded_level = next(item for item in loaded.levels if item.index == 2)
            self.assertEqual(loaded_level.windows, level.windows)

            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["levels"][0]["windows"].extend(
                (
                    payload["levels"][0]["windows"][0],
                    {
                        "window_id": "wrong-level",
                        "wall_surface_id": "level:3/wall:1:2",
                        "start_ratio": 0.2,
                        "end_ratio": 0.6,
                        "bottom_ratio": 0.25,
                        "top_ratio": 0.75,
                    },
                )
            )
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            sanitized = load_project(project_path)
            sanitized_level = next(
                item for item in sanitized.levels if item.index == 2
            )
            self.assertEqual(sanitized_level.windows, level.windows)


# ### Window geometry and export tests ###
class WallWindowGeometryTests(unittest.TestCase):
    def test_dragged_rectangle_commits_a_true_wall_hole(self) -> None:
        level = _build_square_level()
        wall = _get_window_wall(level)
        placement = build_wall_window_placement(
            wall,
            (0.5, 0.0, 0.75),
            (1.5, 0.0, 2.25),
        )

        window = add_wall_window(
            [level],
            placement,
            window_id="window-1",
        )
        rebuilt_wall = _get_window_wall(level)
        model = convert_to_glb([level])

        self.assertEqual(window.wall_surface_id, wall.surface_id)
        self.assertAlmostEqual(rebuilt_wall.area_square_meters, 4.5)
        self.assertFalse(
            _mesh_covers_point_on_plane(
                rebuilt_wall.mesh,
                (1.0, 0.0, 1.5),
                fixed_axis=1,
            )
        )
        self.assertFalse(
            any("window_reveal" in name for name in model.scene.geometry)
        )
        self.assertFalse(
            _mesh_covers_point_on_plane(
                model.mesh,
                (1.0, 0.0, 1.5),
                fixed_axis=1,
            )
        )
        self.assertTrue(
            _mesh_covers_point_on_plane(
                model.mesh,
                (0.25, 0.0, 1.5),
                fixed_axis=1,
            )
        )

        exported = _load_glb_world_mesh(model.glb_bytes)
        self.assertFalse(
            _mesh_covers_point_on_plane(
                exported,
                (1.0, 1.5, 0.0),
                fixed_axis=2,
            )
        )

    def test_window_ratios_follow_level_transform_and_wall_resize(self) -> None:
        level = _build_square_level()
        placement = WallWindowPlacement(
            wall_surface_id="level:2/room:5/wall:1:2",
            start_ratio=0.25,
            end_ratio=0.75,
            bottom_ratio=0.25,
            top_ratio=0.75,
        )
        level.scale = 2.0
        level.offset_x_meters = 3.0
        level.offset_y_meters = -4.0
        level.rooms[0].height_meters = 4.0
        wall = _get_window_wall(level)

        corners = get_wall_window_world_corners(wall, placement)

        np.testing.assert_allclose(corners[0], (3.0, -3.0, 1.0))
        np.testing.assert_allclose(corners[2], (5.0, -3.0, 3.0))

    def test_window_penetrates_50cm_outward_but_no_farther_or_inward(
        self,
    ) -> None:
        level, parallel_wall_ids = _build_window_depth_level()
        wall = _get_window_wall(level)
        wall_start = np.asarray(wall.wall_start_world, dtype=float)
        wall_end = np.asarray(wall.wall_end_world, dtype=float)
        first = wall_start + (wall_end - wall_start) * 0.25
        second = wall_start + (wall_end - wall_start) * 0.75
        first[2] = 0.75
        second[2] = 2.25
        placement = build_wall_window_placement(wall, first, second)

        add_wall_window([level], placement, window_id="deep-window")
        surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces([level])
        }
        target = surfaces[wall.surface_id]
        outer_at_50cm = surfaces[parallel_wall_ids[0]]
        outer_at_51cm = surfaces[parallel_wall_ids[1]]
        interior_at_50cm = surfaces[parallel_wall_ids[2]]
        perpendicular_wall = surfaces[parallel_wall_ids[3]]

        for cut_surface in (target, outer_at_50cm):
            self.assertFalse(
                _mesh_covers_point_on_plane(
                    cut_surface.mesh,
                    _get_wall_midpoint(cut_surface),
                    fixed_axis=1,
                )
            )
        for untouched_surface in (outer_at_51cm, interior_at_50cm):
            self.assertTrue(
                _mesh_covers_point_on_plane(
                    untouched_surface.mesh,
                    _get_wall_midpoint(untouched_surface),
                    fixed_axis=1,
                )
            )
        self.assertTrue(
            _mesh_covers_point_on_plane(
                perpendicular_wall.mesh,
                _get_wall_midpoint(perpendicular_wall),
                fixed_axis=0,
            )
        )

    def test_deep_window_builds_four_sided_reveal_tunnel(self) -> None:
        level, parallel_wall_ids = _build_window_depth_level()
        level.offset_x_meters = 3.0
        level.offset_y_meters = -4.0
        wall = _get_window_wall(level)
        first = _get_wall_ratio_point(wall, 0.25, 0.75)
        second = _get_wall_ratio_point(wall, 0.75, 2.25)
        add_wall_window(
            [level],
            build_wall_window_placement(wall, first, second),
            window_id="sealed-window",
        )
        surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces([level])
        }
        target = surfaces[wall.surface_id]
        outer = surfaces[parallel_wall_ids[0]]
        target_low = _get_wall_ratio_point(target, 0.25, 1.5)
        target_high = _get_wall_ratio_point(target, 0.75, 1.5)
        outer_low = _get_wall_ratio_point(outer, 0.25, 1.5)
        outer_high = _get_wall_ratio_point(outer, 0.75, 1.5)
        low_jamb = (target_low + outer_low) / 2.0
        high_jamb = (target_high + outer_high) / 2.0
        sill = (low_jamb + high_jamb) / 2.0
        sill[2] = 0.75
        head = sill.copy()
        head[2] = 2.25

        model = convert_to_glb([level])
        reveal_name = next(
            name
            for name in model.scene.geometry
            if name.endswith("_window_reveals")
        )
        self.assertEqual(len(model.scene.geometry[reveal_name].faces), 16)
        self.assertAlmostEqual(target.area_square_meters, 12.5)
        for mesh in (target.mesh, model.mesh):
            for point, fixed_axis in (
                (low_jamb, 0),
                (high_jamb, 0),
                (sill, 2),
                (head, 2),
            ):
                self.assertEqual(
                    _count_mesh_triangles_covering_point_on_plane(
                        mesh,
                        tuple(point),
                        fixed_axis=fixed_axis,
                    ),
                    4,
                )

        exported = _load_glb_world_mesh(model.glb_bytes)
        for point, fixed_axis in (
            (low_jamb, 0),
            (high_jamb, 0),
            (sill, 1),
            (head, 1),
        ):
            exported_point = (point[0], point[2], -point[1])
            self.assertEqual(
                _count_mesh_triangles_covering_point_on_plane(
                    exported,
                    tuple(exported_point),
                    fixed_axis=fixed_axis,
                ),
                4,
            )

    def test_window_reveal_stops_at_nearest_outward_wall(self) -> None:
        level, parallel_wall_ids = _build_window_depth_level()
        close_start = level.vertex_data.add_vertex(0.0, -5.0)
        close_end = level.vertex_data.add_vertex(100.0, -5.0)
        level.vertex_data.add_edge(close_start.id, close_end.id)
        close_surface_id = (
            f"level:2/wall:{min(close_start.id, close_end.id)}:"
            f"{max(close_start.id, close_end.id)}"
        )
        wall = _get_window_wall(level)
        add_wall_window(
            [level],
            build_wall_window_placement(
                wall,
                _get_wall_ratio_point(wall, 0.25, 0.75),
                _get_wall_ratio_point(wall, 0.75, 2.25),
            ),
            window_id="nearest-wall-window",
        )
        surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces([level])
        }
        target = surfaces[wall.surface_id]
        close_outer = surfaces[close_surface_id]
        farther_outer = surfaces[parallel_wall_ids[0]]
        target_center = _get_wall_ratio_point(target, 0.5, 0.75)
        close_center = _get_wall_ratio_point(close_outer, 0.5, 0.75)
        farther_center = _get_wall_ratio_point(farther_outer, 0.5, 0.75)
        sealed_point = (target_center + close_center) / 2.0
        unsealed_deeper_point = (close_center + farther_center) / 2.0
        model = convert_to_glb([level])

        self.assertTrue(
            _mesh_covers_point_on_plane(
                model.mesh,
                tuple(sealed_point),
                fixed_axis=2,
            )
        )
        self.assertFalse(
            _mesh_covers_point_on_plane(
                model.mesh,
                tuple(unsealed_deeper_point),
                fixed_axis=2,
            )
        )

    def test_window_reveal_merges_segmented_outer_wall_contacts(self) -> None:
        level = _build_square_level()
        level.scale = 2.0
        outer_vertices = [
            level.vertex_data.add_vertex(image_x, -10.0)
            for image_x in (0.0, 50.0, 100.0)
        ]
        level.vertex_data.add_edge(outer_vertices[0].id, outer_vertices[1].id)
        level.vertex_data.add_edge(outer_vertices[1].id, outer_vertices[2].id)
        wall = _get_window_wall(level)
        add_wall_window(
            [level],
            build_wall_window_placement(
                wall,
                _get_wall_ratio_point(wall, 0.25, 0.75),
                _get_wall_ratio_point(wall, 0.75, 2.25),
            ),
            window_id="segmented-outer-window",
        )
        rebuilt_target = _get_window_wall(level)
        target_y = _get_wall_midpoint(rebuilt_target)[1]
        outer_y = next(
            surface.wall_start_world[1]
            for surface in build_fixed_surfaces([level])
            if surface.room_index is None
            and surface.surface_type == "wall"
            and surface.wall_start_world is not None
        )
        reveal_y = (target_y + outer_y) / 2.0
        model = convert_to_glb([level])

        reveal_name = next(
            name
            for name in model.scene.geometry
            if name.endswith("_window_reveals")
        )
        self.assertEqual(len(model.scene.geometry[reveal_name].faces), 16)
        for horizontal_ratio in (0.3, 0.7):
            sill_point = _get_wall_ratio_point(
                rebuilt_target,
                horizontal_ratio,
                0.75,
            )
            sill_point[1] = reveal_y
            self.assertTrue(
                _mesh_covers_point_on_plane(
                    model.mesh,
                    tuple(sill_point),
                    fixed_axis=2,
                )
            )

    def test_window_reveal_skips_wall_below_window_vertical_span(self) -> None:
        level = _build_square_level()
        short_boundary_ids = tuple(
            level.vertex_data.add_vertex(*point).id
            for point in (
                (0.0, -10.0),
                (100.0, -10.0),
                (100.0, -20.0),
                (0.0, -20.0),
            )
        )
        for start_id, end_id in zip(
            short_boundary_ids,
            (*short_boundary_ids[1:], short_boundary_ids[0]),
        ):
            level.vertex_data.add_edge(start_id, end_id)
        short_center = level.vertex_data.add_vertex(50.0, -15.0)
        level.rooms.append(
            RoomData(
                name="Low exterior strip",
                vertex_ids=short_boundary_ids,
                center_vertex_id=short_center.id,
                height_meters=0.5,
                color_rgb=(170, 170, 170),
            )
        )
        tall_start = level.vertex_data.add_vertex(0.0, -20.0)
        tall_end = level.vertex_data.add_vertex(100.0, -20.0)
        level.vertex_data.add_edge(tall_start.id, tall_end.id)
        wall = _get_window_wall(level)
        add_wall_window(
            [level],
            build_wall_window_placement(
                wall,
                _get_wall_ratio_point(wall, 0.25, 0.75),
                _get_wall_ratio_point(wall, 0.75, 2.25),
            ),
            window_id="vertically-clipped-window",
        )
        target = _get_window_wall(level)
        target_center = _get_wall_ratio_point(target, 0.5, 0.75)
        tall_surface_id = (
            f"level:2/wall:{min(tall_start.id, tall_end.id)}:"
            f"{max(tall_start.id, tall_end.id)}"
        )
        tall_surface = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_id == tall_surface_id
        )
        tall_center = _get_wall_ratio_point(tall_surface, 0.5, 0.75)
        deeper_sill_point = target_center * 0.25 + tall_center * 0.75

        model = convert_to_glb([level])

        self.assertTrue(
            _mesh_covers_point_on_plane(
                model.mesh,
                tuple(deeper_sill_point),
                fixed_axis=2,
            )
        )

    def test_plain_wall_without_contour_uses_bounded_right_side_depth(
        self,
    ) -> None:
        level, wall_ids = _build_plain_window_depth_level()
        original_surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces([level])
        }
        target = original_surfaces[wall_ids[0]]
        wall_start = np.asarray(target.wall_start_world, dtype=float)
        wall_end = np.asarray(target.wall_end_world, dtype=float)
        first = wall_start + (wall_end - wall_start) * 0.25
        second = wall_start + (wall_end - wall_start) * 0.75
        first[2] = 0.75
        second[2] = 2.25

        add_wall_window(
            [level],
            build_wall_window_placement(target, first, second),
            window_id="plain-depth-window",
        )
        surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces([level])
        }

        for cut_wall_id in wall_ids[:2]:
            cut_surface = surfaces[cut_wall_id]
            self.assertFalse(
                _mesh_covers_point_on_plane(
                    cut_surface.mesh,
                    _get_wall_midpoint(cut_surface),
                    fixed_axis=1,
                )
            )
        inward_surface = surfaces[wall_ids[2]]
        self.assertTrue(
            _mesh_covers_point_on_plane(
                inward_surface.mesh,
                _get_wall_midpoint(inward_surface),
                fixed_axis=1,
            )
        )

    def test_deep_window_cut_survives_surface_textures_and_glb_export(
        self,
    ) -> None:
        level, parallel_wall_ids = _build_window_depth_level()
        wall = _get_window_wall(level)
        wall_start = np.asarray(wall.wall_start_world, dtype=float)
        wall_end = np.asarray(wall.wall_end_world, dtype=float)
        first = wall_start + (wall_end - wall_start) * 0.25
        second = wall_start + (wall_end - wall_start) * 0.75
        first[2] = 0.75
        second[2] = 2.25
        add_wall_window(
            [level],
            build_wall_window_placement(wall, first, second),
            window_id="textured-deep-window",
        )
        surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces([level])
        }
        target_surface = surfaces[wall.surface_id]
        near_surface = surfaces[parallel_wall_ids[0]]
        far_surface = surfaces[parallel_wall_ids[1]]

        model = convert_to_glb(
            [level],
            surface_materials={
                target_surface.surface_id: _solid_png(),
                near_surface.surface_id: _solid_png(),
                far_surface.surface_id: _solid_png(),
            },
            export_untextured_surfaces=False,
        )
        self.assertTrue(
            all(
                getattr(getattr(mesh.visual, "material", None), "name", None)
                not in {None, "DefaultMaterial"}
                for mesh in model.scene.geometry.values()
            )
        )
        textured_by_id = {
            surface.surface_id: surface.mesh
            for surface in model.preview_textured_surfaces
        }
        near_point = _get_wall_midpoint(near_surface)
        far_point = _get_wall_midpoint(far_surface)
        target_low = _get_wall_ratio_point(target_surface, 0.25, 1.5)
        target_high = _get_wall_ratio_point(target_surface, 0.75, 1.5)
        near_low = _get_wall_ratio_point(near_surface, 0.25, 1.5)
        near_high = _get_wall_ratio_point(near_surface, 0.75, 1.5)
        reveal_points = (
            ((target_low + near_low) / 2.0, 0),
            ((target_high + near_high) / 2.0, 0),
            (
                np.asarray(
                    (
                        near_point[0],
                        (near_point[1] + _get_wall_midpoint(target_surface)[1])
                        / 2.0,
                        0.75,
                    )
                ),
                2,
            ),
            (
                np.asarray(
                    (
                        near_point[0],
                        (near_point[1] + _get_wall_midpoint(target_surface)[1])
                        / 2.0,
                        2.25,
                    )
                ),
                2,
            ),
        )

        self.assertFalse(
            _mesh_covers_point_on_plane(
                textured_by_id[near_surface.surface_id],
                near_point,
                fixed_axis=1,
            )
        )
        self.assertTrue(
            _mesh_covers_point_on_plane(
                textured_by_id[far_surface.surface_id],
                far_point,
                fixed_axis=1,
            )
        )
        assert model.preview_untextured_mesh is not None
        for reveal_point, fixed_axis in reveal_points:
            self.assertEqual(
                _count_mesh_triangles_covering_point_on_plane(
                    textured_by_id[target_surface.surface_id],
                    tuple(reveal_point),
                    fixed_axis=fixed_axis,
                ),
                4,
            )
            self.assertEqual(
                _count_mesh_triangles_covering_point_on_plane(
                    model.mesh,
                    tuple(reveal_point),
                    fixed_axis=fixed_axis,
                ),
                4,
            )
            self.assertEqual(
                _count_mesh_triangles_covering_point_on_plane(
                    model.preview_untextured_mesh,
                    tuple(reveal_point),
                    fixed_axis=fixed_axis,
                ),
                0,
            )
        exported = _load_glb_world_mesh(model.glb_bytes)
        near_exported_point = (near_point[0], near_point[2], -near_point[1])
        far_exported_point = (far_point[0], far_point[2], -far_point[1])
        self.assertFalse(
            _mesh_covers_point_on_plane(
                exported,
                near_exported_point,
                fixed_axis=2,
            )
        )
        self.assertTrue(
            _mesh_covers_point_on_plane(
                exported,
                far_exported_point,
                fixed_axis=2,
            )
        )
        for reveal_point, fixed_axis in reveal_points:
            exported_point = (
                reveal_point[0],
                reveal_point[2],
                -reveal_point[1],
            )
            exported_fixed_axis = 1 if fixed_axis == 2 else fixed_axis
            self.assertEqual(
                _count_mesh_triangles_covering_point_on_plane(
                    exported,
                    tuple(exported_point),
                    fixed_axis=exported_fixed_axis,
                ),
                4,
            )

    def test_textured_wall_preview_and_export_preserve_window_hole(self) -> None:
        level = _build_square_level()
        wall = _get_window_wall(level)
        placement = build_wall_window_placement(
            wall,
            (0.5, 0.0, 0.75),
            (1.5, 0.0, 2.25),
        )
        add_wall_window([level], placement, window_id="window-1")

        model = convert_to_glb(
            [level],
            surface_materials={wall.surface_id: _solid_png()},
        )
        textured = next(
            surface.mesh
            for surface in model.preview_textured_surfaces
            if surface.surface_id == wall.surface_id
        )

        self.assertFalse(
            _mesh_covers_point_on_plane(
                textured,
                (1.0, 0.0, 1.5),
                fixed_axis=1,
            )
        )
        self.assertFalse(
            _mesh_covers_point_on_plane(
                model.mesh,
                (1.0, 0.0, 1.5),
                fixed_axis=1,
            )
        )

    def test_overlapping_or_out_of_bounds_window_is_rejected_atomically(
        self,
    ) -> None:
        level = _build_square_level()
        first = build_wall_window_placement(
            _get_window_wall(level),
            (0.4, 0.0, 0.6),
            (1.2, 0.0, 1.8),
        )
        add_wall_window([level], first, window_id="window-1")
        existing_windows = list(level.windows)
        wall = _get_window_wall(level)

        with self.assertRaises(ValueError):
            build_wall_window_placement(
                wall,
                (0.8, 0.0, 1.0),
                (1.6, 0.0, 2.2),
            )
        with self.assertRaises(ValueError):
            build_wall_window_placement(
                wall,
                (-0.2, 0.0, 0.5),
                (0.3, 0.0, 1.5),
            )
        self.assertEqual(level.windows, existing_windows)

    def test_window_cuts_both_coplanar_sides_of_a_shared_wall(self) -> None:
        level, shared_wall_key = _build_adjacent_room_level()
        shared_walls = [
            surface
            for surface in build_fixed_surfaces([level])
            if surface.wall_key == shared_wall_key
        ]
        self.assertEqual(len(shared_walls), 2)
        target = shared_walls[0]
        assert target.wall_start_world is not None
        assert target.wall_end_world is not None
        start = np.asarray(target.wall_start_world, dtype=float)
        end = np.asarray(target.wall_end_world, dtype=float)
        first = start + (end - start) * 0.25
        second = start + (end - start) * 0.75
        first[2] = 0.75
        second[2] = 2.25

        placement = build_wall_window_placement(target, first, second)
        add_wall_window([level], placement, window_id="shared-window")
        rebuilt_shared_walls = [
            surface
            for surface in build_fixed_surfaces([level])
            if surface.wall_key == shared_wall_key
        ]

        self.assertEqual(len(rebuilt_shared_walls), 2)
        for wall in rebuilt_shared_walls:
            self.assertAlmostEqual(wall.area_square_meters, 4.5)
            self.assertFalse(
                _mesh_covers_point_on_plane(
                    wall.mesh,
                    (2.0, -1.0, 1.5),
                    fixed_axis=0,
                )
            )

    def test_window_cannot_overlap_an_existing_doorway(self) -> None:
        level = _build_square_level()
        level.doorways = [
            DoorwayData(
                center_x=50.0,
                center_y=0.0,
                width_meters=0.8,
                height_meters=2.0,
                depth_meters=0.2,
                rotation_degrees=90.0,
            )
        ]
        wall = _get_window_wall(level)

        with self.assertRaises(ValueError):
            build_wall_window_placement(
                wall,
                (0.8, 0.0, 1.0),
                (1.2, 0.0, 2.5),
            )
        self.assertEqual(level.windows, [])


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
