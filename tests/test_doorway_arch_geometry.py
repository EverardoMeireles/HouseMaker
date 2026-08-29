# ### Imports ###
from __future__ import annotations

import unittest
from io import BytesIO

import numpy as np
import trimesh
from PIL import Image
from shapely import Point, Polygon

from housemaker.doorway_geometry import (
    DOORWAY_ARCH_SEGMENT_COUNT,
    build_doorway_cross_section_outline,
)
from housemaker.glb import (
    _build_level_doorway_reveal_mesh,
    _build_level_doorway_reveals,
    _build_visible_wall_pieces,
    _build_wall_opening_reveal_quads,
    _build_wall_openings,
    _mask_wall_preview_texture,
    convert_to_glb,
)
from housemaker.models import (
    DOORWAY_SHAPE_ARCH,
    DOORWAY_SHAPE_RECTANGULAR,
    DoorwayData,
    LevelData,
    RoomData,
    VertexData,
)
from housemaker.surface_geometry import build_fixed_surfaces


# ### Fixture helpers ###
def _add_wall(
    vertex_data: VertexData,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    start_vertex = vertex_data.add_vertex(*start)
    end_vertex = vertex_data.add_vertex(*end)
    vertex_data.add_edge(start_vertex.id, end_vertex.id)


def _build_parallel_arch_level(arch_amount: float = 1.0) -> LevelData:
    vertex_data = VertexData()
    _add_wall(vertex_data, (10.0, 45.0), (90.0, 45.0))
    _add_wall(vertex_data, (10.0, 55.0), (90.0, 55.0))
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        doorways=[
            DoorwayData(
                center_x=50.0,
                center_y=50.0,
                width_meters=0.9,
                height_meters=2.1,
                depth_meters=0.2,
                rotation_degrees=90.0,
                shape=DOORWAY_SHAPE_ARCH,
                arch_amount=arch_amount,
            )
        ],
    )


def _build_room_arch_level(arch_amount: float = 1.0) -> LevelData:
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
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[
            RoomData(
                name="Living room",
                vertex_ids=boundary_ids,
                center_vertex_id=center.id,
                color_rgb=(140, 180, 220),
            )
        ],
        doorways=[
            DoorwayData(
                center_x=50.0,
                center_y=0.0,
                width_meters=0.8,
                height_meters=2.0,
                depth_meters=0.2,
                rotation_degrees=90.0,
                shape=DOORWAY_SHAPE_ARCH,
                arch_amount=arch_amount,
            )
        ],
        floor_contour_vertex_ids=boundary_ids,
    )


def _wall_pieces_cover_point(
    pieces: list[object],
    wall_ratio: float,
    height_meters: float,
) -> bool:
    point = Point(wall_ratio, height_meters)
    return any(Polygon(piece.points).buffer(1e-9).covers(point) for piece in pieces)


def _profile_area(outline: tuple[tuple[float, float], ...]) -> float:
    return float(Polygon(outline).area)


def _profile_reveal_length(
    outline: tuple[tuple[float, float], ...],
) -> float:
    return sum(
        float(np.linalg.norm(np.subtract(second, first)))
        for first, second in zip(outline[1:], outline[2:])
    )


def _solid_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (2, 2), (120, 160, 210, 255)).save(
        buffer,
        format="PNG",
    )
    return buffer.getvalue()


# ### Profile tests ###
class DoorwayArchProfileTests(unittest.TestCase):
    def test_rectangular_and_zero_arch_profiles_are_exact_rectangles(self) -> None:
        expected = (
            (-0.45, 0.0),
            (0.45, 0.0),
            (0.45, 2.1),
            (-0.45, 2.1),
            (-0.45, 0.0),
        )

        self.assertEqual(
            build_doorway_cross_section_outline(
                0.9,
                2.1,
                DOORWAY_SHAPE_RECTANGULAR,
            ),
            expected,
        )
        self.assertEqual(
            build_doorway_cross_section_outline(
                0.9,
                2.1,
                DOORWAY_SHAPE_ARCH,
                0.0,
            ),
            expected,
        )

    def test_arch_uses_smooth_sloped_segments_without_steps(self) -> None:
        outline = build_doorway_cross_section_outline(
            0.9,
            2.1,
            DOORWAY_SHAPE_ARCH,
            1.0,
        )
        arc_points = outline[2:-1]

        self.assertEqual(len(arc_points), DOORWAY_ARCH_SEGMENT_COUNT + 1)
        self.assertEqual(outline[0], outline[-1])
        self.assertAlmostEqual(max(height for _width, height in outline), 2.1)
        self.assertAlmostEqual(arc_points[0][1], 1.65)
        self.assertAlmostEqual(arc_points[-1][1], 1.65)
        self.assertTrue(
            all(
                not math_is_close(first[0], second[0])
                and not math_is_close(first[1], second[1])
                for first, second in zip(arc_points, arc_points[1:])
            )
        )

    def test_arch_amount_controls_rise_without_changing_apex(self) -> None:
        shallow = build_doorway_cross_section_outline(
            0.9,
            2.1,
            DOORWAY_SHAPE_ARCH,
            0.25,
        )

        self.assertAlmostEqual(shallow[2][1], 2.1 - 0.45 * 0.25)
        self.assertAlmostEqual(max(height for _width, height in shallow), 2.1)
        self.assertLess(_profile_area(shallow), 0.9 * 2.1)

    def test_short_wide_arch_has_no_duplicate_or_zero_length_edges(self) -> None:
        outline = build_doorway_cross_section_outline(
            2.0,
            0.4,
            DOORWAY_SHAPE_ARCH,
            1.0,
        )

        self.assertEqual(outline[0], outline[-1])
        self.assertTrue(
            all(first != second for first, second in zip(outline, outline[1:]))
        )
        self.assertAlmostEqual(max(height for _width, height in outline), 0.4)


# ### Wall and reveal tests ###
class DoorwayArchWallGeometryTests(unittest.TestCase):
    def test_arch_profile_cuts_both_walls_with_a_sloped_boundary(self) -> None:
        level = _build_parallel_arch_level()
        opening = _build_wall_openings(level.doorways)[0]

        for wall_y in (45.0, 55.0):
            pieces = _build_visible_wall_pieces(
                start_point=(10.0, wall_y),
                end_point=(90.0, wall_y),
                wall_height_meters=3.0,
                doorway_openings=(opening,),
            )
            with self.subTest(wall_y=wall_y):
                self.assertFalse(_wall_pieces_cover_point(pieces, 0.5, 2.0))
                self.assertTrue(_wall_pieces_cover_point(pieces, 0.25, 2.0))
                self.assertTrue(_wall_pieces_cover_point(pieces, 0.5, 2.2))
                self.assertGreater(len(pieces), DOORWAY_ARCH_SEGMENT_COUNT)
                self.assertTrue(all(len(piece.points) == 3 for piece in pieces))

    def test_shallower_amount_changes_the_real_wall_cut(self) -> None:
        full_level = _build_parallel_arch_level(1.0)
        shallow_level = _build_parallel_arch_level(0.2)
        full_pieces = _build_visible_wall_pieces(
            (10.0, 45.0),
            (90.0, 45.0),
            3.0,
            _build_wall_openings(full_level.doorways),
        )
        shallow_pieces = _build_visible_wall_pieces(
            (10.0, 45.0),
            (90.0, 45.0),
            3.0,
            _build_wall_openings(shallow_level.doorways),
        )

        self.assertTrue(_wall_pieces_cover_point(full_pieces, 0.3125, 2.05))
        self.assertFalse(_wall_pieces_cover_point(shallow_pieces, 0.3125, 2.05))

    def test_reveal_uses_one_sloped_quad_per_profile_edge(self) -> None:
        level = _build_parallel_arch_level()
        reveal_mesh = _build_level_doorway_reveal_mesh(
            level,
            base_z_meters=0.0,
            blueprint_size_pixels=None,
            room_vertex_sets=(),
        )

        self.assertIsNotNone(reveal_mesh)
        assert reveal_mesh is not None
        self.assertGreaterEqual(
            len(np.unique(np.round(reveal_mesh.vertices[:, 2], 8))),
            DOORWAY_ARCH_SEGMENT_COUNT // 2,
        )
        self.assertFalse(
            np.any(
                np.all(
                    np.isclose(
                        reveal_mesh.vertices[reveal_mesh.faces][:, :, 2],
                        1.65,
                        atol=1e-8,
                    ),
                    axis=1,
                )
            )
        )

    def test_reveal_quad_normals_face_into_the_opening(self) -> None:
        level = _build_parallel_arch_level()
        opening = _build_wall_openings(level.doorways)[0]
        reveal = _build_level_doorway_reveals(level, room_vertex_sets=())[0]
        quads = _build_wall_opening_reveal_quads(
            reveal.reveal_pair,
            opening,
            include_sill=False,
        )
        jamb_checks = 0
        for quad in quads:
            local_points = np.asarray(
                [
                    (
                        (point[0] - opening.center_x) * opening.width_direction_x
                        + (point[1] - opening.center_y) * opening.width_direction_y,
                        (point[0] - opening.center_x) * opening.depth_direction_x
                        + (point[1] - opening.center_y) * opening.depth_direction_y,
                        point[2],
                    )
                    for point in quad
                ],
                dtype=float,
            )
            mean_width = float(np.mean(local_points[:, 0]))
            if not np.allclose(local_points[:, 0], mean_width, atol=1e-6):
                continue
            normal = np.cross(
                local_points[1] - local_points[0],
                local_points[2] - local_points[0],
            )
            self.assertLess(normal[0] * mean_width, 0.0)
            jamb_checks += 1
        self.assertEqual(jamb_checks, 2)

    def test_fixed_surface_uses_same_smooth_cut_and_reveal(self) -> None:
        level = _build_room_arch_level(0.65)
        doorway = level.doorways[0]
        outline = build_doorway_cross_section_outline(
            doorway.width_meters,
            doorway.height_meters,
            doorway.shape,
            doorway.arch_amount,
        )
        wall = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_id == "level:2/room:5/wall:1:2"
        )
        expected_reveal_area = (
            doorway.depth_meters * _profile_reveal_length(outline)
        )

        self.assertAlmostEqual(
            wall.area_square_meters,
            6.0 - _profile_area(outline) + expected_reveal_area,
            places=5,
        )
        self.assertGreater(len(wall.mesh.faces), DOORWAY_ARCH_SEGMENT_COUNT)
        np.testing.assert_allclose(
            wall.mesh.bounds[:, 1],
            np.asarray((-0.1, 0.1)),
            atol=1e-6,
        )

    def test_preview_alpha_is_smooth_and_has_no_triangle_seams(self) -> None:
        level = _build_parallel_arch_level()
        wall_pieces = _build_visible_wall_pieces(
            (10.0, 45.0),
            (90.0, 45.0),
            3.0,
            _build_wall_openings(level.doorways),
        )
        texture = np.full((256, 128, 4), 255, dtype=np.uint8)
        masked = _mask_wall_preview_texture(texture, wall_pieces, 3.0)

        self.assertTrue(np.all(masked[8:48, 8:120, 3] == 255))
        self.assertEqual(int(masked[128, 42, 3]), 0)
        self.assertEqual(int(masked[128, 105, 3]), 255)
        self.assertTrue(np.any((masked[:, :, 3] > 0) & (masked[:, :, 3] < 255)))

    def test_arch_wall_and_reveals_share_surface_texture_geometry(self) -> None:
        level = _build_room_arch_level()
        surface_id = "level:2/room:5/wall:1:2"
        fixed_surface = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_id == surface_id
        )
        model = convert_to_glb(
            [level],
            surface_materials={surface_id: _solid_png()},
        )

        textured_surface = next(
            surface
            for surface in model.preview_textured_surfaces
            if surface.surface_id == surface_id
        )
        self.assertAlmostEqual(textured_surface.mesh.area, fixed_surface.mesh.area)
        self.assertGreater(len(model.glb_bytes), 0)


# ### Numeric helpers ###
def math_is_close(first: float, second: float) -> bool:
    return bool(np.isclose(first, second, atol=1e-12))


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
