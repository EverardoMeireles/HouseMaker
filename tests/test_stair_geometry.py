# ### Environment setup ###
from __future__ import annotations

import math
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
from housemaker.glb import (
    DEFAULT_STAIR_RISER_HEIGHT_METERS,
    _build_smoothed_stair_route_sections,
    _normalize_stair_rail_correspondence,
    _order_stair_route_sections,
    build_stair_meshes,
    convert_to_glb,
)
from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.models import (
    STAIR_STYLE_FLOATING,
    STAIR_STYLE_FLOATING_WITH_RISER,
    STAIR_STYLE_SUPPORTED,
    LevelData,
    StairData,
    StairSectionData,
    VertexData,
)


# ### Fixture helpers ###
def _build_level(
    index: int,
    name: str,
    *,
    scale: float = 1.0,
    offset_x_meters: float = 0.0,
    offset_y_meters: float = 0.0,
) -> LevelData:
    vertex_data = VertexData()
    vertex_data.add_vertex(0.0, 0.0)
    vertex_data.add_vertex(200.0, 100.0)
    return LevelData(
        index=index,
        name=name,
        scale=scale,
        offset_x_meters=offset_x_meters,
        offset_y_meters=offset_y_meters,
        vertex_data=vertex_data,
    )


def _get_mesh_by_name(
    levels: list[LevelData],
    stair: StairData,
) -> object:
    named_meshes = build_stair_meshes(levels, [stair])
    return named_meshes[0].mesh


def _assert_mesh_contains_xy_points(
    test_case: unittest.TestCase,
    mesh: object,
    expected_points: list[tuple[float, float]],
) -> None:
    vertices = np.asarray(getattr(mesh, "vertices"), dtype=float)
    for expected_point in expected_points:
        expected_xy = np.asarray(expected_point, dtype=float)
        test_case.assertTrue(
            np.any(np.all(np.isclose(vertices[:, :2], expected_xy), axis=1)),
            msg=f"Expected stair mesh to contain XY point {expected_point!r}.",
        )


def _build_curved_stair(style: str) -> StairData:
    return StairData(
        start_level_index=2,
        start_a_x=0.0,
        start_a_y=0.0,
        start_b_x=0.0,
        start_b_y=50.0,
        end_level_index=3,
        end_a_x=225.0,
        end_a_y=175.0,
        end_b_x=175.0,
        end_b_y=175.0,
        style=style,
        intermediate_sections=(
            StairSectionData(
                level_index=2,
                a_x=100.0,
                a_y=0.0,
                b_x=100.0,
                b_y=50.0,
            ),
            StairSectionData(
                level_index=3,
                a_x=225.0,
                a_y=75.0,
                b_x=175.0,
                b_y=75.0,
            ),
        ),
    )


def _build_alternating_sharp_curve_sections(
) -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    list[np.ndarray],
]:
    """Return four curve guides with deliberately alternating click order."""

    segment_angles_degrees = (0.0, 70.0, -70.0, 70.0, -70.0)
    segment_directions = [
        np.asarray(
            (
                math.cos(math.radians(angle_degrees)),
                math.sin(math.radians(angle_degrees)),
            ),
            dtype=float,
        )
        for angle_degrees in segment_angles_degrees
    ]
    centers = [np.asarray((0.0, 0.0), dtype=float)]
    for segment_direction in segment_directions:
        centers.append(centers[-1] + (segment_direction * 1.5))

    route_directions = [segment_directions[0]]
    for incoming_direction, outgoing_direction in zip(
        segment_directions,
        segment_directions[1:],
    ):
        route_direction = incoming_direction + outgoing_direction
        route_directions.append(
            route_direction / np.linalg.norm(route_direction)
        )
    route_directions.append(segment_directions[-1])

    sections: list[tuple[np.ndarray, np.ndarray]] = []
    for section_index, (center_xy, route_direction) in enumerate(
        zip(centers, route_directions)
    ):
        width_xy = 1.2 * np.asarray(
            (-route_direction[1], route_direction[0]),
            dtype=float,
        )
        section_a_xy = center_xy - (width_xy / 2.0)
        section_b_xy = center_xy + (width_xy / 2.0)
        if section_index % 2:
            section_a_xy, section_b_xy = section_b_xy, section_a_xy
        sections.append((section_a_xy, section_b_xy))

    return sections, route_directions


def _does_test_polyline_self_intersect(
    points: list[np.ndarray],
) -> bool:
    def cross_product(first_xy: np.ndarray, second_xy: np.ndarray) -> float:
        return float(
            (first_xy[0] * second_xy[1])
            - (first_xy[1] * second_xy[0])
        )

    for first_index in range(len(points) - 1):
        first_start = points[first_index]
        first_end = points[first_index + 1]
        for second_index in range(first_index + 2, len(points) - 1):
            second_start = points[second_index]
            second_end = points[second_index + 1]
            first_start_side = cross_product(
                first_end - first_start,
                second_start - first_start,
            )
            first_end_side = cross_product(
                first_end - first_start,
                second_end - first_start,
            )
            second_start_side = cross_product(
                second_end - second_start,
                first_start - second_start,
            )
            second_end_side = cross_product(
                second_end - second_start,
                first_end - second_start,
            )
            if first_start_side * first_end_side < 0.0 and (
                second_start_side * second_end_side < 0.0
            ):
                return True
    return False


# ### Stair geometry tests ###
class StairGeometryTests(unittest.TestCase):
    def test_supported_and_floating_stairs_have_distinct_geometry(self) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        supported_stair = StairData(
            start_level_index=2,
            start_a_x=0.0,
            start_a_y=0.0,
            start_b_x=0.0,
            start_b_y=50.0,
            end_level_index=3,
            end_a_x=200.0,
            end_a_y=0.0,
            end_b_x=200.0,
            end_b_y=50.0,
            style=STAIR_STYLE_SUPPORTED,
        )
        floating_stair = StairData(
            start_level_index=2,
            start_a_x=0.0,
            start_a_y=100.0,
            start_b_x=0.0,
            start_b_y=150.0,
            end_level_index=3,
            end_a_x=200.0,
            end_a_y=100.0,
            end_b_x=200.0,
            end_b_y=150.0,
            style=STAIR_STYLE_FLOATING,
        )
        floating_with_riser_stair = StairData(
            start_level_index=2,
            start_a_x=0.0,
            start_a_y=200.0,
            start_b_x=0.0,
            start_b_y=250.0,
            end_level_index=3,
            end_a_x=200.0,
            end_a_y=200.0,
            end_b_x=200.0,
            end_b_y=250.0,
            style=STAIR_STYLE_FLOATING_WITH_RISER,
        )

        named_meshes = build_stair_meshes(
            [ground_level, upper_level],
            [supported_stair, floating_stair, floating_with_riser_stair],
        )
        meshes = {named_mesh.name: named_mesh.mesh for named_mesh in named_meshes}
        supported_mesh = meshes["stair_1_supported"]
        floating_mesh = meshes["stair_2_floating"]
        floating_with_riser_mesh = meshes["stair_3_floating_with_riser"]

        self.assertAlmostEqual(float(supported_mesh.bounds[0, 2]), 0.0)
        self.assertGreater(float(floating_mesh.bounds[0, 2]), 0.0)
        self.assertAlmostEqual(float(supported_mesh.bounds[1, 2]), 3.0)
        self.assertAlmostEqual(float(floating_mesh.bounds[1, 2]), 3.0)
        self.assertGreater(float(supported_mesh.volume), float(floating_mesh.volume))
        self.assertAlmostEqual(
            float(floating_with_riser_mesh.bounds[1, 2]),
            3.0,
        )
        self.assertGreater(
            float(floating_with_riser_mesh.bounds[0, 2]),
            0.0,
        )
        self.assertGreater(
            float(floating_with_riser_mesh.volume),
            float(floating_mesh.volume),
        )
        self.assertLess(
            float(floating_with_riser_mesh.volume),
            float(supported_mesh.volume),
        )
        self.assertGreater(
            len(floating_with_riser_mesh.faces),
            len(floating_mesh.faces),
        )

        model = convert_to_glb(
            [ground_level, upper_level],
            stairs=[supported_stair, floating_stair, floating_with_riser_stair],
        )
        self.assertIn("stair_1_supported", model.scene.geometry)
        self.assertIn("stair_2_floating", model.scene.geometry)
        self.assertIn("stair_3_floating_with_riser", model.scene.geometry)

    def test_each_intermediate_section_shapes_all_stair_styles(self) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")

        for style in (
            STAIR_STYLE_SUPPORTED,
            STAIR_STYLE_FLOATING,
            STAIR_STYLE_FLOATING_WITH_RISER,
        ):
            with self.subTest(style=style):
                stair = _build_curved_stair(style)
                mesh = _get_mesh_by_name([ground_level, upper_level], stair)
                intermediate_world_points = [
                    level_image_to_world_xy(
                        [ground_level, upper_level][section.level_index - 2],
                        x,
                        y,
                    )
                    for section in stair.intermediate_sections
                    for x, y in section.points
                ]

                _assert_mesh_contains_xy_points(
                    self,
                    mesh,
                    intermediate_world_points,
                )
                self.assertGreater(len(mesh.faces), 0)
                self.assertTrue(np.all(np.isfinite(mesh.vertices)))

    def test_multiple_guides_share_one_ordered_sampled_route(self) -> None:
        """Every later guide must extend the existing curve, not start a flight."""

        control_sections = [
            (
                np.asarray((0.0, 0.0)),
                np.asarray((0.0, 1.0)),
            ),
            (
                np.asarray((2.0, 1.5)),
                np.asarray((2.0, 2.5)),
            ),
            (
                np.asarray((4.5, -0.5)),
                np.asarray((4.5, 0.5)),
            ),
            (
                np.asarray((7.0, 2.0)),
                np.asarray((7.0, 3.0)),
            ),
        ]

        route_sections = _build_smoothed_stair_route_sections(
            control_sections
        )
        matched_indices: list[int] = []
        next_index = 0
        for control_a_xy, control_b_xy in control_sections:
            for route_index in range(next_index, len(route_sections)):
                route_a_xy, route_b_xy = route_sections[route_index]
                if np.allclose(route_a_xy, control_a_xy) and np.allclose(
                    route_b_xy,
                    control_b_xy,
                ):
                    matched_indices.append(route_index)
                    next_index = route_index + 1
                    break
            else:
                self.fail("A curved-stair guide was omitted from its route.")

        self.assertEqual(matched_indices, sorted(matched_indices))
        self.assertGreater(len(route_sections), len(control_sections))

    def test_alternating_sharp_guides_keep_one_non_crossing_rail_order(
        self,
    ) -> None:
        """Sharp guide bends must not exchange A/B according to click order."""

        clicked_sections, expected_route_directions = (
            _build_alternating_sharp_curve_sections()
        )
        normalized_sections = _normalize_stair_rail_correspondence(
            clicked_sections
        )

        for (section_a_xy, section_b_xy), route_direction in zip(
            normalized_sections,
            expected_route_directions,
        ):
            width_xy = section_b_xy - section_a_xy
            handedness = (
                (route_direction[0] * width_xy[1])
                - (route_direction[1] * width_xy[0])
            )
            self.assertGreater(handedness, 0.0)

        sampled_sections = _build_smoothed_stair_route_sections(
            normalized_sections
        )
        sampled_centers = [
            (section_a_xy + section_b_xy) / 2.0
            for section_a_xy, section_b_xy in sampled_sections
        ]
        for section_index, (section_a_xy, section_b_xy) in enumerate(
            sampled_sections
        ):
            if section_index == 0:
                route_delta = sampled_centers[1] - sampled_centers[0]
            elif section_index == len(sampled_sections) - 1:
                route_delta = sampled_centers[-1] - sampled_centers[-2]
            else:
                route_delta = (
                    sampled_centers[section_index + 1]
                    - sampled_centers[section_index - 1]
                )
            width_xy = section_b_xy - section_a_xy
            handedness = (
                (route_delta[0] * width_xy[1])
                - (route_delta[1] * width_xy[0])
            )
            self.assertGreater(handedness, 0.0)
            self.assertGreaterEqual(
                float(np.linalg.norm(width_xy)),
                1.2 - 1e-9,
            )
        self.assertFalse(
            _does_test_polyline_self_intersect(
                [section[0] for section in sampled_sections]
            )
        )
        self.assertFalse(
            _does_test_polyline_self_intersect(
                [section[1] for section in sampled_sections]
            )
        )

    def test_guides_are_spatially_ordered_between_fixed_endpoints(self) -> None:
        """Guide click order must not decide the final stair route order."""

        def section_at(center_x: float) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.asarray((center_x, -0.5), dtype=float),
                np.asarray((center_x, 0.5), dtype=float),
            )

        clicked_sections = [
            section_at(0.0),
            section_at(8.0),
            section_at(2.0),
            section_at(5.0),
            section_at(10.0),
        ]

        ordered_sections = _order_stair_route_sections(clicked_sections)
        ordered_center_x_values = [
            float(((section_a_xy + section_b_xy) / 2.0)[0])
            for section_a_xy, section_b_xy in ordered_sections
        ]

        self.assertEqual(
            ordered_center_x_values,
            [0.0, 2.0, 5.0, 8.0, 10.0],
        )

    def test_sharp_close_guides_are_rejected_before_a_rail_self_intersects(
        self,
    ) -> None:
        """An impossible control corridor must report its crossing rails."""

        control_sections = [
            (
                np.asarray((0.0, -0.5), dtype=float),
                np.asarray((0.0, 0.5), dtype=float),
            ),
            (
                np.asarray((2.0, -0.5), dtype=float),
                np.asarray((2.0, 0.5), dtype=float),
            ),
            (
                np.asarray((2.6, 0.1), dtype=float),
                np.asarray((1.6, 0.1), dtype=float),
            ),
            (
                np.asarray((2.6, 3.0), dtype=float),
                np.asarray((1.6, 3.0), dtype=float),
            ),
        ]

        with self.assertRaisesRegex(ValueError, "crossing rails"):
            _build_smoothed_stair_route_sections(control_sections)

    def test_curve_guide_does_not_create_an_internal_tread_cap(self) -> None:
        """A guide splits geometry, not the visible stair into separate parts."""

        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        stair = StairData(
            start_level_index=2,
            start_a_x=0.0,
            start_a_y=0.0,
            start_b_x=0.0,
            start_b_y=50.0,
            end_level_index=3,
            end_a_x=300.0,
            end_a_y=0.0,
            end_b_x=300.0,
            end_b_y=50.0,
            style=STAIR_STYLE_FLOATING,
            intermediate_sections=(
                StairSectionData(
                    level_index=2,
                    a_x=105.0,
                    a_y=0.0,
                    b_x=105.0,
                    b_y=50.0,
                ),
            ),
        )

        mesh = _get_mesh_by_name([ground_level, upper_level], stair)
        guide_world_x = level_image_to_world_xy(
            ground_level,
            105.0,
            0.0,
        )[0]
        vertices = np.asarray(mesh.vertices, dtype=float)
        internal_cap_face_count = sum(
            np.all(
                np.isclose(
                    vertices[np.asarray(face, dtype=np.int64), 0],
                    guide_world_x,
                )
            )
            for face in np.asarray(mesh.faces, dtype=np.int64)
        )

        self.assertEqual(internal_cap_face_count, 0)
        _assert_mesh_contains_xy_points(
            self,
            mesh,
            [
                level_image_to_world_xy(ground_level, 105.0, 0.0),
                level_image_to_world_xy(ground_level, 105.0, 50.0),
            ],
        )

    def test_curved_floating_riser_adds_one_panel_for_each_later_tread(
        self,
    ) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        floating_stair = _build_curved_stair(STAIR_STYLE_FLOATING)
        riser_stair = _build_curved_stair(STAIR_STYLE_FLOATING_WITH_RISER)

        floating_mesh = _get_mesh_by_name(
            [ground_level, upper_level],
            floating_stair,
        )
        riser_mesh = _get_mesh_by_name(
            [ground_level, upper_level],
            riser_stair,
        )
        step_count = math.ceil(3.0 / DEFAULT_STAIR_RISER_HEIGHT_METERS)

        self.assertEqual(
            len(riser_mesh.faces),
            len(floating_mesh.faces) + (step_count - 1) * 12,
        )
        self.assertGreater(float(riser_mesh.volume), float(floating_mesh.volume))

    def test_intermediate_section_uses_owner_level_transform_and_vertex_binding(
        self,
    ) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story", scale=1.25)
        bound_vertex = upper_level.vertex_data.add_vertex(225.0, 75.0)
        stair = StairData(
            start_level_index=2,
            start_a_x=0.0,
            start_a_y=0.0,
            start_b_x=0.0,
            start_b_y=50.0,
            end_level_index=3,
            end_a_x=225.0,
            end_a_y=175.0,
            end_b_x=175.0,
            end_b_y=175.0,
            intermediate_sections=(
                StairSectionData(
                    level_index=3,
                    a_x=225.0,
                    a_y=75.0,
                    b_x=175.0,
                    b_y=75.0,
                    a_vertex_id=bound_vertex.id,
                ),
            ),
        )
        upper_level.vertex_data.move_vertex(bound_vertex.id, 250.0, 90.0)

        mesh = _get_mesh_by_name([ground_level, upper_level], stair)
        moved_world = level_image_to_world_xy(upper_level, 250.0, 90.0)

        _assert_mesh_contains_xy_points(self, mesh, [moved_world])

    def test_missing_intermediate_section_level_is_rejected(self) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        stair = StairData(
            start_level_index=2,
            start_a_x=0.0,
            start_a_y=0.0,
            start_b_x=0.0,
            start_b_y=50.0,
            end_level_index=3,
            end_a_x=200.0,
            end_a_y=0.0,
            end_b_x=200.0,
            end_b_y=50.0,
            intermediate_sections=(
                StairSectionData(
                    level_index=4,
                    a_x=100.0,
                    a_y=0.0,
                    b_x=100.0,
                    b_y=50.0,
                ),
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Stair intermediate section 1 level 4 does not exist",
        ):
            build_stair_meshes([ground_level, upper_level], [stair])

    def test_endpoints_follow_each_level_current_transform(self) -> None:
        ground_level = _build_level(
            2,
            "Ground",
            scale=1.5,
            offset_x_meters=1.0,
            offset_y_meters=-0.5,
        )
        upper_level = _build_level(
            3,
            "Story",
            scale=0.75,
            offset_x_meters=-1.25,
            offset_y_meters=0.75,
        )
        stair = StairData(
            start_level_index=2,
            start_a_x=20.0,
            start_a_y=20.0,
            start_b_x=20.0,
            start_b_y=60.0,
            end_level_index=3,
            end_a_x=180.0,
            end_a_y=60.0,
            end_b_x=180.0,
            end_b_y=100.0,
            style=STAIR_STYLE_FLOATING,
        )

        mesh = _get_mesh_by_name([ground_level, upper_level], stair)
        initial_points = [
            level_image_to_world_xy(ground_level, x, y)
            for x, y in stair.start_points
        ] + [
            level_image_to_world_xy(upper_level, x, y)
            for x, y in stair.end_points
        ]
        _assert_mesh_contains_xy_points(self, mesh, initial_points)

        ground_level.offset_x_meters += 2.0
        upper_level.scale = 1.25
        updated_mesh = _get_mesh_by_name([ground_level, upper_level], stair)
        updated_points = [
            level_image_to_world_xy(ground_level, x, y)
            for x, y in stair.start_points
        ] + [
            level_image_to_world_xy(upper_level, x, y)
            for x, y in stair.end_points
        ]
        _assert_mesh_contains_xy_points(self, updated_mesh, updated_points)
        self.assertFalse(np.allclose(initial_points, updated_points))

    def test_descending_supported_stair_has_valid_terminal_support(self) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        stair = StairData(
            start_level_index=3,
            start_a_x=0.0,
            start_a_y=0.0,
            start_b_x=0.0,
            start_b_y=50.0,
            end_level_index=2,
            end_a_x=200.0,
            end_a_y=0.0,
            end_b_x=200.0,
            end_b_y=50.0,
            style=STAIR_STYLE_SUPPORTED,
        )

        mesh = _get_mesh_by_name([ground_level, upper_level], stair)

        self.assertGreater(len(mesh.faces), 0)
        self.assertTrue(np.all(np.isfinite(mesh.vertices)))
        self.assertTrue(math.isfinite(float(mesh.volume)))
        self.assertAlmostEqual(float(mesh.bounds[0, 2]), 0.0)
        self.assertAlmostEqual(float(mesh.bounds[1, 2]), 3.0)

    def test_missing_endpoint_level_is_rejected(self) -> None:
        ground_level = _build_level(2, "Ground")
        stair = StairData(
            start_level_index=2,
            start_a_x=0.0,
            start_a_y=0.0,
            start_b_x=0.0,
            start_b_y=50.0,
            end_level_index=3,
            end_a_x=200.0,
            end_a_y=0.0,
            end_b_x=200.0,
            end_b_y=50.0,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Stair end level 3 does not exist",
        ):
            build_stair_meshes([ground_level], [stair])

    def test_stair_is_not_exported_when_either_endpoint_level_is_excluded(
        self,
    ) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        first_vertex, second_vertex = ground_level.vertex_data.vertices
        ground_level.vertex_data.add_edge(first_vertex.id, second_vertex.id)
        upper_level.include_in_export = False
        stair = StairData(
            start_level_index=2,
            start_a_x=0.0,
            start_a_y=0.0,
            start_b_x=0.0,
            start_b_y=50.0,
            end_level_index=3,
            end_a_x=200.0,
            end_a_y=0.0,
            end_b_x=200.0,
            end_b_y=50.0,
            style=STAIR_STYLE_SUPPORTED,
        )

        model = convert_to_glb(
            [ground_level, upper_level],
            stairs=[stair],
        )

        self.assertNotIn("stair_1_supported", model.scene.geometry)
        self.assertTrue(model.scene.geometry)

    def test_stair_is_not_exported_when_intermediate_owner_level_is_excluded(
        self,
    ) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        route_owner_level = _build_level(4, "Route owner")
        first_vertex, second_vertex = ground_level.vertex_data.vertices
        ground_level.vertex_data.add_edge(first_vertex.id, second_vertex.id)
        route_owner_level.include_in_export = False
        stair = StairData(
            start_level_index=2,
            start_a_x=0.0,
            start_a_y=0.0,
            start_b_x=0.0,
            start_b_y=50.0,
            end_level_index=3,
            end_a_x=200.0,
            end_a_y=100.0,
            end_b_x=200.0,
            end_b_y=150.0,
            intermediate_sections=(
                StairSectionData(
                    level_index=4,
                    a_x=100.0,
                    a_y=0.0,
                    b_x=100.0,
                    b_y=50.0,
                ),
            ),
        )

        model = convert_to_glb(
            [ground_level, upper_level, route_owner_level],
            stairs=[stair],
        )

        self.assertNotIn("stair_1_supported", model.scene.geometry)
        self.assertTrue(model.scene.geometry)

    def test_reversed_upper_click_order_produces_the_same_stair_shape(self) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        common_values = {
            "start_level_index": 2,
            "start_a_x": 0.0,
            "start_a_y": 0.0,
            "start_b_x": 0.0,
            "start_b_y": 50.0,
            "end_level_index": 3,
            "style": STAIR_STYLE_FLOATING,
        }
        direct_stair = StairData(
            **common_values,
            end_a_x=200.0,
            end_a_y=0.0,
            end_b_x=200.0,
            end_b_y=50.0,
        )
        reversed_stair = StairData(
            **common_values,
            end_a_x=200.0,
            end_a_y=50.0,
            end_b_x=200.0,
            end_b_y=0.0,
        )

        direct_mesh = _get_mesh_by_name(
            [ground_level, upper_level],
            direct_stair,
        )
        reversed_mesh = _get_mesh_by_name(
            [ground_level, upper_level],
            reversed_stair,
        )

        self.assertTrue(np.allclose(direct_mesh.bounds, reversed_mesh.bounds))
        self.assertAlmostEqual(float(direct_mesh.volume), float(reversed_mesh.volume))

    def test_bound_control_point_follows_vertex_and_falls_back_if_deleted(
        self,
    ) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        bound_vertex = ground_level.vertex_data.add_vertex(10.0, 15.0)
        stair = StairData(
            start_level_index=2,
            start_a_x=10.0,
            start_a_y=15.0,
            start_b_x=10.0,
            start_b_y=65.0,
            end_level_index=3,
            end_a_x=200.0,
            end_a_y=15.0,
            end_b_x=200.0,
            end_b_y=65.0,
            start_a_vertex_id=bound_vertex.id,
        )

        ground_level.vertex_data.move_vertex(bound_vertex.id, 40.0, 20.0)
        moved_mesh = _get_mesh_by_name([ground_level, upper_level], stair)
        moved_world = level_image_to_world_xy(ground_level, 40.0, 20.0)
        _assert_mesh_contains_xy_points(self, moved_mesh, [moved_world])

        ground_level.vertex_data.delete_vertex(bound_vertex.id)
        fallback_mesh = _get_mesh_by_name([ground_level, upper_level], stair)
        fallback_world = level_image_to_world_xy(ground_level, 10.0, 15.0)
        _assert_mesh_contains_xy_points(self, fallback_mesh, [fallback_world])

    def test_zero_width_or_effectively_zero_run_is_rejected(self) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        shared_vertex = ground_level.vertex_data.vertices[0]
        zero_width_stair = StairData(
            start_level_index=2,
            start_a_x=0.0,
            start_a_y=0.0,
            start_b_x=0.0,
            start_b_y=50.0,
            end_level_index=3,
            end_a_x=200.0,
            end_a_y=0.0,
            end_b_x=200.0,
            end_b_y=50.0,
            start_a_vertex_id=shared_vertex.id,
            start_b_vertex_id=shared_vertex.id,
        )
        zero_run_stair = StairData(
            start_level_index=2,
            start_a_x=0.0,
            start_a_y=0.0,
            start_b_x=0.0,
            start_b_y=50.0,
            end_level_index=3,
            end_a_x=0.0,
            end_a_y=0.0,
            end_b_x=0.0,
            end_b_y=50.0,
        )

        with self.assertRaisesRegex(ValueError, "start points must be separated"):
            build_stair_meshes([ground_level, upper_level], [zero_width_stair])
        with self.assertRaisesRegex(ValueError, "centers must be separated"):
            build_stair_meshes([ground_level, upper_level], [zero_run_stair])

    def test_vertex_data_conversion_rejects_stairs_without_levels(self) -> None:
        stair = StairData(
            start_level_index=2,
            start_a_x=0.0,
            start_a_y=0.0,
            start_b_x=0.0,
            start_b_y=50.0,
            end_level_index=3,
            end_a_x=200.0,
            end_a_y=0.0,
            end_b_x=200.0,
            end_b_y=50.0,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Stairs require level data",
        ):
            convert_to_glb(VertexData(), stairs=[stair])
