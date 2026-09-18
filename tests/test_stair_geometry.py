# ### Environment setup ###
from __future__ import annotations

import json
import math
import os
import unittest
from dataclasses import replace
from io import BytesIO

import numpy as np
import trimesh
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
from housemaker.glb import (
    DEFAULT_STAIR_RISER_HEIGHT_METERS,
    STAIR_PART_RISERS,
    STAIR_PART_STRINGERS,
    STAIR_PART_SUPPORT,
    STAIR_PART_TREADS,
    _build_rounded_stair_tread_prism,
    _build_smoothed_stair_route_sections,
    _build_starting_step_profile_distances,
    _build_starting_step_route_sections,
    _normalize_stair_rail_correspondence,
    _order_stair_route_sections,
    build_canvas_stair_part_targets,
    build_stair_meshes,
    convert_to_glb,
)
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
)
from housemaker.models import (
    STAIR_NOSING_FRONT,
    STAIR_NOSING_LEFT,
    STAIR_NOSING_RIGHT,
    STAIR_STARTING_STEP_BULLNOSE,
    STAIR_STARTING_STEP_CURTAIL,
    STAIR_STARTING_STEP_NONE,
    STAIR_STRINGER_BOTH,
    STAIR_STRINGER_LEFT,
    STAIR_STRINGER_NONE,
    STAIR_STYLE_FLOATING,
    STAIR_STYLE_FLOATING_WITH_RISER,
    STAIR_STYLE_SUPPORTED,
    STAIR_TREAD_EDGE_ROUNDED,
    STAIR_TREAD_EDGE_STRAIGHT,
    STAIR_TYPE_FLOATING,
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


def _solid_png(color: tuple[int, int, int, int]) -> bytes:
    payload = BytesIO()
    Image.new("RGBA", (4, 4), color).save(payload, format="PNG")
    return payload.getvalue()


def _read_glb_document(payload: bytes) -> dict[str, object]:
    json_length = int.from_bytes(payload[12:16], "little")
    return json.loads(payload[20 : 20 + json_length].decode("utf-8"))


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


def _build_alternating_sharp_curve_sections() -> tuple[
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
        route_directions.append(route_direction / np.linalg.norm(route_direction))
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
        return float((first_xy[0] * second_xy[1]) - (first_xy[1] * second_xy[0]))

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
    def test_parameterized_stair_builds_one_joined_mesh_per_semantic_part(
        self,
    ) -> None:
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
            stringer_placement=STAIR_STRINGER_BOTH,
        )

        targets = build_canvas_stair_part_targets(
            [ground_level, upper_level],
            [stair],
        )
        targets_by_kind = {target.part_kind: target for target in targets}

        self.assertEqual(
            set(targets_by_kind),
            {
                STAIR_PART_TREADS,
                STAIR_PART_SUPPORT,
                STAIR_PART_RISERS,
                STAIR_PART_STRINGERS,
            },
        )
        for part_kind, target in targets_by_kind.items():
            expected_surface_type = (
                "floor" if part_kind == STAIR_PART_TREADS else "wall"
            )
            self.assertEqual(
                target.semantic_id,
                f"stair:{stair.stair_id}/part:{part_kind}:{expected_surface_type}",
            )
            self.assertEqual(target.level_indices, (2, 3))
            self.assertGreater(len(target.mesh.faces), 12)

        exported_parts = build_stair_meshes(
            [ground_level, upper_level],
            [stair],
        )
        self.assertEqual(len(exported_parts), 4)
        self.assertFalse(
            any(
                "EXT_mesh_gpu_instancing" in str(mesh.mesh.metadata)
                for mesh in exported_parts
            )
        )
        document = _read_glb_document(
            convert_to_glb(
                [ground_level, upper_level],
                stairs=[stair],
            ).glb_bytes
        )
        self.assertNotIn("EXT_mesh_gpu_instancing", document.get("extensionsUsed", ()))
        self.assertEqual(len(document["meshes"]), 4)
        self.assertTrue(
            all(len(mesh["primitives"]) == 1 for mesh in document["meshes"])
        )
        support_bounds = targets_by_kind[STAIR_PART_SUPPORT].mesh.bounds
        riser_mesh = targets_by_kind[STAIR_PART_RISERS].mesh
        stringer_bounds = targets_by_kind[STAIR_PART_STRINGERS].mesh.bounds
        self.assertGreater(support_bounds[0, 1], stringer_bounds[0, 1])
        self.assertLess(support_bounds[1, 1], stringer_bounds[1, 1])
        self.assertTrue(np.all(riser_mesh.face_normals[:, 0] < -0.99))

        def _face_coordinates(mesh: trimesh.Trimesh) -> set[tuple[object, ...]]:
            return {
                tuple(
                    sorted(
                        tuple(np.round(vertex, 6))
                        for vertex in mesh.vertices[face]
                    )
                )
                for face in mesh.faces
            }

        # Riser triangles are moved out of the support primitive, not drawn
        # on top of it where z-fighting could confuse selection.
        self.assertFalse(
            _face_coordinates(riser_mesh)
            & _face_coordinates(targets_by_kind[STAIR_PART_SUPPORT].mesh)
        )
        self.assertFalse(
            _face_coordinates(riser_mesh)
            & _face_coordinates(targets_by_kind[STAIR_PART_TREADS].mesh)
        )

    def test_supported_risers_contain_only_exposed_step_bands(self) -> None:
        ground_level = _build_level(2, "Ground")
        ground_level.height_meters = 0.52
        ground_level.floor_thickness_meters = 0.02
        upper_level = _build_level(3, "Story")
        upper_level.floor_thickness_meters = 0.02
        base_z_by_level = build_level_base_z_lookup(
            [ground_level, upper_level]
        )
        lower_floor_top = base_z_by_level[ground_level.index]
        total_rise = base_z_by_level[upper_level.index] - lower_floor_top

        for edge_profile, nosing_placements in (
            (STAIR_TREAD_EDGE_STRAIGHT, (STAIR_NOSING_FRONT,)),
            (
                STAIR_TREAD_EDGE_ROUNDED,
                (STAIR_NOSING_FRONT, STAIR_NOSING_LEFT, STAIR_NOSING_RIGHT),
            ),
        ):
            with self.subTest(profile=edge_profile):
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
                    tread_thickness_meters=0.05,
                    tread_edge_profile=edge_profile,
                    tread_overhang_meters=0.04,
                    nosing_placements=nosing_placements,
                    stringer_placement=STAIR_STRINGER_NONE,
                )
                step_count, step_rise = stair.calculate_step_layout(total_rise)
                self.assertGreater(step_count, 2)
                riser_mesh = next(
                    target.mesh
                    for target in build_canvas_stair_part_targets(
                        [ground_level, upper_level], [stair]
                    )
                    if target.part_kind == STAIR_PART_RISERS
                )
                face_vertices = riser_mesh.vertices[riser_mesh.faces]
                start_x = level_image_to_world_xy(ground_level, 0.0, 0.0)[0]
                end_x = level_image_to_world_xy(upper_level, 200.0, 0.0)[0]
                for step_index in range(step_count):
                    # The front support cap is on the nominal start of each
                    # step, independent of rounded/overhanging tread edges.
                    front_x = start_x + (end_x - start_x) * step_index / step_count
                    front_mask = np.all(
                        np.isclose(face_vertices[:, :, 0], front_x, atol=1e-6),
                        axis=1,
                    )
                    front_faces = face_vertices[front_mask]
                    self.assertGreater(
                        len(front_faces),
                        0,
                        f"The exposed front of step {step_index} was omitted",
                    )
                    expected_bottom = (
                        lower_floor_top
                        if step_index == 0
                        else lower_floor_top + step_rise * step_index
                    )
                    expected_top = (
                        lower_floor_top
                        + step_rise * (step_index + 1)
                        - stair.tread_thickness_meters
                    )
                    self.assertAlmostEqual(
                        float(np.min(front_faces[:, :, 2])), expected_bottom
                    )
                    self.assertAlmostEqual(
                        float(np.max(front_faces[:, :, 2])), expected_top
                    )
                    self.assertTrue(
                        np.all(riser_mesh.face_normals[front_mask, 0] < -0.99)
                    )

    def test_supported_risers_omit_zero_height_support_bands(self) -> None:
        ground_level = _build_level(2, "Ground")
        ground_level.height_meters = 0.52
        ground_level.floor_thickness_meters = 0.02
        upper_level = _build_level(3, "Story")
        upper_level.floor_thickness_meters = 0.02
        base_z_by_level = build_level_base_z_lookup(
            [ground_level, upper_level]
        )
        lower_floor_top = base_z_by_level[ground_level.index]
        total_rise = base_z_by_level[upper_level.index] - lower_floor_top
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
            tread_thickness_meters=0.05,
            tread_overhang_meters=0.04,
            nosing_placements=(STAIR_NOSING_FRONT,),
            stringer_placement=STAIR_STRINGER_NONE,
        )
        step_count, step_rise = stair.calculate_step_layout(total_rise)
        self.assertGreater(step_count, 1)
        stair = replace(stair, tread_thickness_meters=step_rise)
        targets = build_canvas_stair_part_targets(
            [ground_level, upper_level], [stair]
        )
        # Every support band is zero-height, and the overhanging first nose
        # belongs to Treads. There is no phantom riser pick or export target.
        self.assertFalse(
            any(target.part_kind == STAIR_PART_RISERS for target in targets)
        )
        tread_mesh = next(
            target.mesh
            for target in targets
            if target.part_kind == STAIR_PART_TREADS
        )
        first_step_top = lower_floor_top + step_rise
        self.assertGreater(len(tread_mesh.faces), 0)
        self.assertAlmostEqual(float(tread_mesh.bounds[0, 2]), lower_floor_top)
        self.assertGreaterEqual(float(tread_mesh.bounds[1, 2]), first_step_top)
        self.assertTrue(np.all(tread_mesh.area_faces > 1e-10))
        self.assertGreater(len(convert_to_glb(
            [ground_level, upper_level], stairs=[stair]
        ).glb_bytes), 0)

    def test_curved_supported_riser_bands_keep_valid_front_faces(self) -> None:
        ground_level = _build_level(2, "Ground")
        ground_level.height_meters = 0.52
        ground_level.floor_thickness_meters = 0.02
        upper_level = _build_level(3, "Story")
        upper_level.floor_thickness_meters = 0.02
        base_z_by_level = build_level_base_z_lookup(
            [ground_level, upper_level]
        )
        lower_floor_top = base_z_by_level[ground_level.index]
        total_rise = base_z_by_level[upper_level.index] - lower_floor_top
        stair = replace(
            _build_curved_stair(STAIR_STYLE_SUPPORTED),
            tread_thickness_meters=0.05,
            legacy_part_layout=False,
        )
        step_count, step_rise = stair.calculate_step_layout(total_rise)
        riser_mesh = next(
            target.mesh
            for target in build_canvas_stair_part_targets(
                [ground_level, upper_level], [stair]
            )
            if target.part_kind == STAIR_PART_RISERS
        )
        face_vertices = riser_mesh.vertices[riser_mesh.faces]
        face_bottoms = np.min(face_vertices[:, :, 2], axis=1)
        face_tops = np.max(face_vertices[:, :, 2], axis=1)
        for step_index in range(step_count):
            exposed_bottom = (
                lower_floor_top
                if step_index == 0
                else lower_floor_top + step_rise * step_index
            )
            exposed_top = (
                lower_floor_top
                + step_rise * (step_index + 1)
                - stair.tread_thickness_meters
            )
            exposed_faces = np.isclose(face_bottoms, exposed_bottom) & np.isclose(
                face_tops, exposed_top
            )
            self.assertEqual(
                int(np.count_nonzero(exposed_faces)),
                2,
                f"Step {step_index} lost its exposed riser front",
            )
            self.assertTrue(np.all(riser_mesh.area_faces[exposed_faces] > 1e-9))
            self.assertTrue(
                np.all(np.abs(riser_mesh.face_normals[exposed_faces, 2]) < 1e-9)
            )
            if step_index > 0:
                hidden_lower_faces = np.isclose(
                    face_bottoms, lower_floor_top
                ) & np.isclose(face_tops, exposed_top)
                self.assertFalse(np.any(hidden_lower_faces))

    def test_rounded_tread_has_consistent_outward_winding(self) -> None:
        tread_mesh = _build_rounded_stair_tread_prism(
            start_a_xy=np.asarray((0.0, 0.0)),
            start_b_xy=np.asarray((0.0, 1.0)),
            end_a_xy=np.asarray((1.0, 0.0)),
            end_b_xy=np.asarray((1.0, 1.0)),
            bottom_z_meters=0.0,
            top_z_meters=0.2,
            include_end_cap=True,
        )
        tread_mesh.merge_vertices()

        self.assertTrue(tread_mesh.is_watertight)
        self.assertTrue(tread_mesh.is_winding_consistent)
        self.assertTrue(tread_mesh.is_volume)
        self.assertGreater(float(tread_mesh.volume), 0.0)

    def test_rounded_tread_edge_radius_controls_curve_and_uses_available_depth(
        self,
    ) -> None:
        common_values = {
            "start_a_xy": np.asarray((0.0, 0.0)),
            "start_b_xy": np.asarray((0.0, 1.0)),
            "end_a_xy": np.asarray((1.0, 0.0)),
            "end_b_xy": np.asarray((1.0, 1.0)),
            "bottom_z_meters": 0.0,
            "top_z_meters": 0.08,
            "include_end_cap": True,
        }
        shallow_curve = _build_rounded_stair_tread_prism(
            **common_values,
            edge_radius_meters=0.01,
        )
        default_curve = _build_rounded_stair_tread_prism(
            **common_values,
            edge_radius_meters=0.04,
        )
        deep_curve = _build_rounded_stair_tread_prism(
            **common_values,
            edge_radius_meters=0.08,
        )
        capped_curve = _build_rounded_stair_tread_prism(
            **common_values,
            edge_radius_meters=100.0,
        )

        def leading_top_x(mesh: trimesh.Trimesh) -> float:
            top_vertices = mesh.vertices[
                np.isclose(mesh.vertices[:, 2], common_values["top_z_meters"])
            ]
            return float(np.min(top_vertices[:, 0]))

        self.assertAlmostEqual(leading_top_x(shallow_curve), 0.01)
        self.assertAlmostEqual(leading_top_x(default_curve), 0.04)
        self.assertAlmostEqual(leading_top_x(deep_curve), 0.08)
        self.assertAlmostEqual(leading_top_x(capped_curve), 0.99)
        capped_curve.merge_vertices()
        self.assertTrue(capped_curve.is_watertight)
        self.assertTrue(capped_curve.is_winding_consistent)
        self.assertGreater(float(capped_curve.volume), 0.0)

    def test_rounded_edge_radius_updates_every_tread_along_a_curved_route(
        self,
    ) -> None:
        levels = [_build_level(2, "Ground"), _build_level(3, "Story")]
        common_stair = replace(
            _build_curved_stair(STAIR_STYLE_FLOATING),
            stair_type=STAIR_TYPE_FLOATING,
            tread_edge_profile=STAIR_TREAD_EDGE_ROUNDED,
            stringer_placement=STAIR_STRINGER_NONE,
            legacy_part_layout=False,
        )
        shallow_stair = replace(
            common_stair,
            tread_edge_radius_meters=0.04,
        )
        deep_stair = replace(
            common_stair,
            tread_edge_radius_meters=0.08,
        )

        def tread_mesh(stair: StairData) -> trimesh.Trimesh:
            return next(
                target.mesh.copy()
                for target in build_canvas_stair_part_targets(levels, [stair])
                if target.part_kind == STAIR_PART_TREADS
            )

        shallow_mesh = tread_mesh(shallow_stair)
        deep_mesh = tread_mesh(deep_stair)
        total_rise_meters = (
            levels[0].height_meters + levels[1].floor_thickness_meters
        )
        step_count, actual_rise_meters = common_stair.calculate_step_layout(
            total_rise_meters
        )
        lower_elevation_meters = levels[0].floor_thickness_meters

        def top_xy_signature(
            mesh: trimesh.Trimesh,
            top_z: float,
        ) -> set[tuple[float, float]]:
            return {
                (round(float(vertex[0]), 7), round(float(vertex[1]), 7))
                for vertex in mesh.vertices[np.isclose(mesh.vertices[:, 2], top_z)]
            }

        for step_index in range(step_count):
            top_z = lower_elevation_meters + (
                actual_rise_meters * (step_index + 1)
            )
            self.assertNotEqual(
                top_xy_signature(shallow_mesh, top_z),
                top_xy_signature(deep_mesh, top_z),
            )

        for mesh in (shallow_mesh, deep_mesh):
            mesh.merge_vertices()
            self.assertTrue(mesh.is_watertight)
            self.assertTrue(mesh.is_winding_consistent)
            self.assertTrue(mesh.is_volume)
            self.assertGreater(float(mesh.volume), 0.0)

    def test_straight_tread_geometry_ignores_edge_radius(self) -> None:
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
            tread_edge_profile=STAIR_TREAD_EDGE_STRAIGHT,
            tread_edge_radius_meters=0.0,
            stringer_placement=STAIR_STRINGER_LEFT,
        )

        zero_radius_mesh = next(
            target.mesh
            for target in build_canvas_stair_part_targets(
                [ground_level, upper_level],
                [stair],
            )
            if target.part_kind == STAIR_PART_TREADS
        )
        maximum_radius_mesh = next(
            target.mesh
            for target in build_canvas_stair_part_targets(
                [ground_level, upper_level],
                [replace(stair, tread_edge_radius_meters=1.0)],
            )
            if target.part_kind == STAIR_PART_TREADS
        )

        np.testing.assert_allclose(
            maximum_radius_mesh.vertices,
            zero_radius_mesh.vertices,
        )
        np.testing.assert_array_equal(maximum_radius_mesh.faces, zero_radius_mesh.faces)

    def test_straight_tread_nosing_expands_only_selected_edges(self) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        common_values = {
            "start_level_index": 2,
            "start_a_x": 0.0,
            "start_a_y": 0.0,
            "start_b_x": 0.0,
            "start_b_y": 50.0,
            "end_level_index": 3,
            "end_a_x": 200.0,
            "end_a_y": 0.0,
            "end_b_x": 200.0,
            "end_b_y": 50.0,
            "tread_overhang_meters": 0.04,
            "tread_edge_profile": STAIR_TREAD_EDGE_STRAIGHT,
            "stringer_placement": STAIR_STRINGER_BOTH,
        }

        def build_parts(
            nosing_placements: tuple[str, ...],
        ) -> dict[str, object]:
            stair = StairData(
                **common_values,
                nosing_placements=nosing_placements,
            )
            return {
                target.part_kind: target.mesh
                for target in build_canvas_stair_part_targets(
                    [ground_level, upper_level],
                    [stair],
                )
            }

        base_parts = build_parts(())
        front_parts = build_parts((STAIR_NOSING_FRONT,))
        left_parts = build_parts((STAIR_NOSING_LEFT,))
        right_parts = build_parts((STAIR_NOSING_RIGHT,))
        all_parts = build_parts(
            (STAIR_NOSING_LEFT, STAIR_NOSING_RIGHT, STAIR_NOSING_FRONT)
        )

        expected_tread_bounds = {
            "base": ((0.0, -1.0), (4.0, 0.0)),
            "front": ((-0.04, -1.0), (4.0, 0.0)),
            "left": ((0.0, -1.0), (4.0, 0.04)),
            "right": ((0.0, -1.04), (4.0, 0.0)),
            "all": ((-0.04, -1.04), (4.0, 0.04)),
        }
        for name, parts in (
            ("base", base_parts),
            ("front", front_parts),
            ("left", left_parts),
            ("right", right_parts),
            ("all", all_parts),
        ):
            bounds = np.asarray(parts[STAIR_PART_TREADS].bounds, dtype=float)
            np.testing.assert_allclose(
                bounds[:, :2],
                expected_tread_bounds[name],
            )
            for unchanged_part in (STAIR_PART_SUPPORT, STAIR_PART_STRINGERS):
                np.testing.assert_allclose(
                    parts[unchanged_part].vertices,
                    base_parts[unchanged_part].vertices,
                )

    def test_left_nosing_keeps_physical_side_when_route_and_rails_reverse(
        self,
    ) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        common_values = {
            "stair_type": STAIR_TYPE_FLOATING,
            "stringer_placement": STAIR_STRINGER_LEFT,
            "tread_overhang_meters": 0.04,
            "nosing_placements": (STAIR_NOSING_LEFT,),
        }
        ascending = StairData(
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
            **common_values,
        )
        descending_with_reversed_rails = StairData(
            start_level_index=3,
            start_a_x=200.0,
            start_a_y=50.0,
            start_b_x=200.0,
            start_b_y=0.0,
            end_level_index=2,
            end_a_x=0.0,
            end_a_y=50.0,
            end_b_x=0.0,
            end_b_y=0.0,
            **common_values,
        )

        def tread_vertices(stair: StairData) -> np.ndarray:
            target = next(
                part
                for part in build_canvas_stair_part_targets(
                    [ground_level, upper_level],
                    [stair],
                )
                if part.part_kind == STAIR_PART_TREADS
            )
            return np.unique(np.round(target.mesh.vertices, decimals=9), axis=0)

        ascending_vertices = tread_vertices(ascending)
        descending_vertices = tread_vertices(descending_with_reversed_rails)
        np.testing.assert_allclose(descending_vertices, ascending_vertices)
        self.assertAlmostEqual(float(np.min(ascending_vertices[:, 1])), -1.0)
        self.assertAlmostEqual(float(np.max(ascending_vertices[:, 1])), 0.04)

    def test_rounded_treads_remain_valid_with_nosing_on_every_edge(self) -> None:
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
            stair_type=STAIR_TYPE_FLOATING,
            tread_overhang_meters=0.04,
            nosing_placements=(
                STAIR_NOSING_LEFT,
                STAIR_NOSING_RIGHT,
                STAIR_NOSING_FRONT,
            ),
            tread_edge_profile=STAIR_TREAD_EDGE_ROUNDED,
            stringer_placement=STAIR_STRINGER_LEFT,
        )

        tread_mesh = next(
            part.mesh.copy()
            for part in build_canvas_stair_part_targets(
                [ground_level, upper_level],
                [stair],
            )
            if part.part_kind == STAIR_PART_TREADS
        )
        np.testing.assert_allclose(
            tread_mesh.bounds[:, :2],
            ((-0.04, -1.04), (4.0, 0.04)),
        )
        tread_mesh.merge_vertices()
        self.assertTrue(tread_mesh.is_watertight)
        self.assertTrue(tread_mesh.is_winding_consistent)
        self.assertTrue(tread_mesh.is_volume)
        self.assertGreater(float(tread_mesh.volume), 0.0)

    def test_rounded_tread_edge_follows_enabled_lateral_nosing_sides(self) -> None:
        levels = [_build_level(2, "Ground"), _build_level(3, "Story")]
        common_stair = StairData(
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
            stair_type=STAIR_TYPE_FLOATING,
            stringer_placement=STAIR_STRINGER_NONE,
            tread_overhang_meters=0.04,
            tread_edge_radius_meters=0.03,
            legacy_part_layout=False,
        )
        total_rise_meters = levels[0].height_meters + levels[1].floor_thickness_meters
        _, riser_height_meters = common_stair.calculate_step_layout(total_rise_meters)
        first_top_z = levels[0].floor_thickness_meters + riser_height_meters
        first_bottom_z = first_top_z - common_stair.tread_thickness_meters
        first_mid_z = (first_top_z + first_bottom_z) / 2.0

        def tread_mesh(
            nosing_placements: tuple[str, ...],
            edge_profile: str,
        ) -> trimesh.Trimesh:
            stair = replace(
                common_stair,
                nosing_placements=nosing_placements,
                tread_edge_profile=edge_profile,
            )
            return next(
                target.mesh.copy()
                for target in build_canvas_stair_part_targets(levels, [stair])
                if target.part_kind == STAIR_PART_TREADS
            )

        def lateral_bounds_at_height(
            mesh: trimesh.Trimesh,
            height_meters: float,
        ) -> tuple[float, float]:
            y_coordinates = mesh.vertices[
                np.isclose(mesh.vertices[:, 2], height_meters), 1
            ]
            self.assertGreater(len(y_coordinates), 0)
            return float(np.min(y_coordinates)), float(np.max(y_coordinates))

        for placements in (
            (),
            (STAIR_NOSING_LEFT,),
            (STAIR_NOSING_RIGHT,),
            (STAIR_NOSING_LEFT, STAIR_NOSING_RIGHT),
            (STAIR_NOSING_LEFT, STAIR_NOSING_RIGHT, STAIR_NOSING_FRONT),
        ):
            with self.subTest(placements=placements):
                straight = tread_mesh(placements, STAIR_TREAD_EDGE_STRAIGHT)
                rounded = tread_mesh(placements, STAIR_TREAD_EDGE_ROUNDED)
                straight_top = lateral_bounds_at_height(straight, first_top_z)
                rounded_top = lateral_bounds_at_height(rounded, first_top_z)
                rounded_bottom = lateral_bounds_at_height(rounded, first_bottom_z)
                rounded_mid = lateral_bounds_at_height(rounded, first_mid_z)

                if STAIR_NOSING_RIGHT in placements:
                    self.assertGreater(
                        rounded_top[0], straight_top[0] + 0.005
                    )
                    self.assertGreater(
                        rounded_bottom[0], straight_top[0] + 0.005
                    )
                else:
                    self.assertAlmostEqual(rounded_top[0], straight_top[0])
                    self.assertAlmostEqual(rounded_bottom[0], straight_top[0])
                if STAIR_NOSING_LEFT in placements:
                    self.assertLess(
                        rounded_top[1], straight_top[1] - 0.005
                    )
                    self.assertLess(
                        rounded_bottom[1], straight_top[1] - 0.005
                    )
                else:
                    self.assertAlmostEqual(rounded_top[1], straight_top[1])
                    self.assertAlmostEqual(rounded_bottom[1], straight_top[1])
                np.testing.assert_allclose(rounded_mid, straight_top, atol=1e-7)

                rounded.merge_vertices()
                self.assertTrue(rounded.is_watertight)
                self.assertTrue(rounded.is_winding_consistent)
                self.assertTrue(rounded.is_volume)
                self.assertGreater(float(rounded.volume), 0.0)

    def test_rounded_lateral_tread_edges_remain_closed_across_route_guides(
        self,
    ) -> None:
        levels = [_build_level(2, "Ground"), _build_level(3, "Story")]
        common_stair = StairData(
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
            stair_type=STAIR_TYPE_FLOATING,
            stringer_placement=STAIR_STRINGER_NONE,
            tread_overhang_meters=0.04,
            nosing_placements=(STAIR_NOSING_LEFT, STAIR_NOSING_RIGHT),
            tread_edge_radius_meters=0.03,
            legacy_part_layout=False,
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

        def tread_mesh(edge_profile: str) -> trimesh.Trimesh:
            stair = replace(common_stair, tread_edge_profile=edge_profile)
            return next(
                target.mesh.copy()
                for target in build_canvas_stair_part_targets(levels, [stair])
                if target.part_kind == STAIR_PART_TREADS
            )

        guide_world_x = level_image_to_world_xy(levels[0], 105.0, 0.0)[0]
        straight = tread_mesh(STAIR_TREAD_EDGE_STRAIGHT)
        rounded = tread_mesh(STAIR_TREAD_EDGE_ROUNDED)
        guide_vertices = straight.vertices[
            np.isclose(straight.vertices[:, 0], guide_world_x)
        ]
        self.assertGreater(len(guide_vertices), 0)
        top_z = float(np.max(guide_vertices[:, 2]))
        bottom_z = float(np.min(guide_vertices[:, 2]))
        middle_z = (top_z + bottom_z) / 2.0

        def guide_lateral_bounds(
            mesh: trimesh.Trimesh,
            height_meters: float,
        ) -> tuple[float, float]:
            matching_vertices = mesh.vertices[
                np.isclose(mesh.vertices[:, 0], guide_world_x)
                & np.isclose(mesh.vertices[:, 2], height_meters)
            ]
            self.assertGreater(len(matching_vertices), 0)
            return (
                float(np.min(matching_vertices[:, 1])),
                float(np.max(matching_vertices[:, 1])),
            )

        straight_outer_edges = guide_lateral_bounds(straight, top_z)
        rounded_top = guide_lateral_bounds(rounded, top_z)
        rounded_bottom = guide_lateral_bounds(rounded, bottom_z)
        rounded_middle = guide_lateral_bounds(rounded, middle_z)
        for side_bounds in (rounded_top, rounded_bottom):
            self.assertGreater(side_bounds[0], straight_outer_edges[0] + 0.005)
            self.assertLess(side_bounds[1], straight_outer_edges[1] - 0.005)
        np.testing.assert_allclose(rounded_middle, straight_outer_edges, atol=1e-7)

        guide_cap_face_count = sum(
            np.all(
                np.isclose(
                    rounded.vertices[np.asarray(face, dtype=np.int64), 0],
                    guide_world_x,
                )
            )
            for face in rounded.faces
        )
        self.assertEqual(guide_cap_face_count, 0)

        rounded.merge_vertices()
        self.assertTrue(rounded.is_watertight)
        self.assertTrue(rounded.is_winding_consistent)
        self.assertTrue(rounded.is_volume)
        self.assertGreater(float(rounded.volume), 0.0)

    def test_rounded_lateral_edges_cover_starting_step_profile_segments(
        self,
    ) -> None:
        levels = [_build_level(2, "Ground"), _build_level(3, "Story")]
        common_stair = StairData(
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
            stair_type=STAIR_TYPE_FLOATING,
            stringer_placement=STAIR_STRINGER_NONE,
            tread_overhang_meters=0.04,
            nosing_placements=(STAIR_NOSING_LEFT, STAIR_NOSING_RIGHT),
            tread_edge_radius_meters=0.02,
            starting_step_edge_radius_meters=0.12,
            starting_step_edge_points=3,
            legacy_part_layout=False,
        )
        total_rise_meters = levels[0].height_meters + levels[1].floor_thickness_meters
        _, riser_height_meters = common_stair.calculate_step_layout(total_rise_meters)
        first_top_z = levels[0].floor_thickness_meters + riser_height_meters
        first_bottom_z = first_top_z - common_stair.tread_thickness_meters
        first_mid_z = (first_top_z + first_bottom_z) / 2.0

        def first_step_lateral_bounds(
            mesh: trimesh.Trimesh,
            height_meters: float,
        ) -> tuple[float, float]:
            y_coordinates = mesh.vertices[
                np.isclose(mesh.vertices[:, 2], height_meters), 1
            ]
            self.assertGreater(len(y_coordinates), 0)
            return float(np.min(y_coordinates)), float(np.max(y_coordinates))

        for starting_step in (
            STAIR_STARTING_STEP_BULLNOSE,
            STAIR_STARTING_STEP_CURTAIL,
        ):
            with self.subTest(starting_step=starting_step):
                def tread_mesh(edge_profile: str) -> trimesh.Trimesh:
                    stair = replace(
                        common_stair,
                        starting_step=starting_step,
                        tread_edge_profile=edge_profile,
                    )
                    return next(
                        target.mesh.copy()
                        for target in build_canvas_stair_part_targets(levels, [stair])
                        if target.part_kind == STAIR_PART_TREADS
                    )

                straight = tread_mesh(STAIR_TREAD_EDGE_STRAIGHT)
                rounded = tread_mesh(STAIR_TREAD_EDGE_ROUNDED)
                straight_outer = first_step_lateral_bounds(straight, first_top_z)
                rounded_top = first_step_lateral_bounds(rounded, first_top_z)
                rounded_mid = first_step_lateral_bounds(rounded, first_mid_z)
                self.assertGreater(rounded_top[0], straight_outer[0] + 0.005)
                self.assertLess(rounded_top[1], straight_outer[1] - 0.005)
                np.testing.assert_allclose(rounded_mid, straight_outer, atol=1e-7)

                rounded.merge_vertices()
                self.assertTrue(rounded.is_watertight)
                self.assertTrue(rounded.is_winding_consistent)
                self.assertTrue(rounded.is_volume)
                self.assertGreater(float(rounded.volume), 0.0)

    def test_starting_step_profiles_change_only_the_physical_lowest_tread(
        self,
    ) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        common_values = {
            "start_level_index": 2,
            "start_a_x": 0.0,
            "start_a_y": 0.0,
            "start_b_x": 0.0,
            "start_b_y": 50.0,
            "end_level_index": 3,
            "end_a_x": 200.0,
            "end_a_y": 0.0,
            "end_b_x": 200.0,
            "end_b_y": 50.0,
            "nosing_placements": (),
            "tread_edge_profile": STAIR_TREAD_EDGE_STRAIGHT,
            "stringer_placement": STAIR_STRINGER_BOTH,
        }

        def build_parts(starting_step: str) -> dict[str, trimesh.Trimesh]:
            stair = StairData(
                **common_values,
                starting_step=starting_step,
            )
            return {
                target.part_kind: target.mesh
                for target in build_canvas_stair_part_targets(
                    [ground_level, upper_level],
                    [stair],
                )
            }

        base_parts = build_parts(STAIR_STARTING_STEP_NONE)
        bullnose_parts = build_parts(STAIR_STARTING_STEP_BULLNOSE)
        curtail_parts = build_parts(STAIR_STARTING_STEP_CURTAIL)
        base_treads = base_parts[STAIR_PART_TREADS]
        bullnose_treads = bullnose_parts[STAIR_PART_TREADS]
        curtail_treads = curtail_parts[STAIR_PART_TREADS]

        self.assertLess(bullnose_treads.bounds[0, 1], base_treads.bounds[0, 1])
        self.assertGreater(bullnose_treads.bounds[1, 1], base_treads.bounds[1, 1])
        self.assertLess(curtail_treads.bounds[0, 1], base_treads.bounds[0, 1])
        self.assertGreater(curtail_treads.bounds[1, 1], base_treads.bounds[1, 1])
        self.assertLess(curtail_treads.bounds[0, 0], base_treads.bounds[0, 0])

        first_tread_top = float(np.min(np.unique(base_treads.vertices[:, 2])[1:]))

        def vertices_above_first_tread(mesh: trimesh.Trimesh) -> np.ndarray:
            later_vertices = mesh.vertices[
                mesh.vertices[:, 2] > first_tread_top + 1e-6
            ]
            return np.unique(np.round(later_vertices, decimals=9), axis=0)

        base_later_vertices = vertices_above_first_tread(base_treads)
        np.testing.assert_allclose(
            vertices_above_first_tread(bullnose_treads),
            base_later_vertices,
        )
        np.testing.assert_allclose(
            vertices_above_first_tread(curtail_treads),
            base_later_vertices,
        )
        for unchanged_part in (STAIR_PART_SUPPORT, STAIR_PART_STRINGERS):
            np.testing.assert_allclose(
                bullnose_parts[unchanged_part].vertices,
                base_parts[unchanged_part].vertices,
            )
            np.testing.assert_allclose(
                curtail_parts[unchanged_part].vertices,
                base_parts[unchanged_part].vertices,
            )

    def test_starting_step_edge_radius_controls_only_its_plan_lobes(
        self,
    ) -> None:
        levels = [_build_level(2, "Ground"), _build_level(3, "Story")]
        common_values = {
            "start_level_index": 2,
            "start_a_x": 0.0,
            "start_a_y": 0.0,
            "start_b_x": 0.0,
            "start_b_y": 50.0,
            "end_level_index": 3,
            "end_a_x": 200.0,
            "end_a_y": 0.0,
            "end_b_x": 200.0,
            "end_b_y": 50.0,
            "stair_type": STAIR_TYPE_FLOATING,
            "stringer_placement": STAIR_STRINGER_NONE,
            "tread_edge_profile": STAIR_TREAD_EDGE_STRAIGHT,
        }

        def tread_mesh(starting_step: str, radius_meters: float) -> trimesh.Trimesh:
            stair = StairData(
                **common_values,
                starting_step=starting_step,
                starting_step_edge_radius_meters=radius_meters,
            )
            return next(
                target.mesh.copy()
                for target in build_canvas_stair_part_targets(levels, [stair])
                if target.part_kind == STAIR_PART_TREADS
            )

        for starting_step in (
            STAIR_STARTING_STEP_BULLNOSE,
            STAIR_STARTING_STEP_CURTAIL,
        ):
            with self.subTest(starting_step=starting_step):
                flat_mesh = tread_mesh(starting_step, 0.0)
                default_radius_mesh = tread_mesh(starting_step, 0.05)
                small_radius_mesh = tread_mesh(starting_step, 0.08)
                large_radius_mesh = tread_mesh(starting_step, 0.20)
                self.assertLess(
                    float(flat_mesh.bounds[1, 1]),
                    float(default_radius_mesh.bounds[1, 1]),
                )
                self.assertLess(
                    float(default_radius_mesh.bounds[1, 1]),
                    float(small_radius_mesh.bounds[1, 1]),
                )
                self.assertLess(
                    float(small_radius_mesh.bounds[1, 1]),
                    float(large_radius_mesh.bounds[1, 1]),
                )
                for mesh in (
                    flat_mesh,
                    default_radius_mesh,
                    small_radius_mesh,
                    large_radius_mesh,
                ):
                    mesh.merge_vertices()
                    self.assertTrue(mesh.is_watertight)
                    self.assertTrue(mesh.is_winding_consistent)
                    self.assertTrue(mesh.is_volume)
                    self.assertGreater(float(mesh.volume), 0.0)

    def test_starting_step_points_refine_both_profile_halves_symmetrically(
        self,
    ) -> None:
        baseline_distances = _build_starting_step_profile_distances(
            0.0,
            1.0,
            0.0,
            1,
        )
        refined_distances = _build_starting_step_profile_distances(
            0.0,
            1.0,
            0.0,
            2,
        )

        np.testing.assert_allclose(baseline_distances, (0.0, 0.5, 1.0))
        first_side_ratio = (1.0 - math.cos(math.pi / 4.0)) / 2.0
        np.testing.assert_allclose(
            refined_distances,
            (
                0.0,
                first_side_ratio,
                0.5,
                1.0 - first_side_ratio,
                1.0,
            ),
        )
        np.testing.assert_allclose(
            refined_distances,
            1.0 - refined_distances[::-1],
        )

    def test_starting_step_points_stay_symmetric_around_edge_protection(
        self,
    ) -> None:
        protected_distances = _build_starting_step_profile_distances(
            0.0,
            1.0,
            0.3,
            4,
        )

        np.testing.assert_allclose(
            protected_distances,
            (
                0.0,
                0.3,
                (1.0 - math.cos(3.0 * math.pi / 8.0)) / 2.0,
                0.5,
                (1.0 - math.cos(5.0 * math.pi / 8.0)) / 2.0,
                0.7,
                1.0,
            ),
        )
        np.testing.assert_allclose(
            protected_distances,
            1.0 - protected_distances[::-1],
        )

    def test_starting_step_points_distribute_radius_across_side_points(
        self,
    ) -> None:
        route_sections = (
            (np.asarray((0.0, 0.0)), np.asarray((0.0, 1.0))),
            (np.asarray((1.0, 0.0)), np.asarray((1.0, 1.0))),
        )
        common_arguments = (
            route_sections,
            (0.0, 1.0),
            0.0,
            1.0,
            STAIR_STARTING_STEP_BULLNOSE,
            0.2,
        )
        baseline_distances, baseline_sections = (
            _build_starting_step_route_sections(
                *common_arguments,
                1,
                0.0,
                (),
                STAIR_TREAD_EDGE_STRAIGHT,
                0.0,
            )
        )
        refined_distances, refined_sections = (
            _build_starting_step_route_sections(
                *common_arguments,
                2,
                0.0,
                (),
                STAIR_TREAD_EDGE_STRAIGHT,
                0.0,
            )
        )
        _, smaller_radius_sections = _build_starting_step_route_sections(
            *common_arguments[:-1],
            0.1,
            2,
            0.0,
            (),
            STAIR_TREAD_EDGE_STRAIGHT,
            0.0,
        )

        np.testing.assert_allclose(baseline_distances, (0.0, 0.5, 1.0))
        side_angle = math.pi / 4.0
        side_distance = (1.0 - math.cos(side_angle)) / 2.0
        side_extension = 0.2 * math.sin(side_angle)
        np.testing.assert_allclose(refined_distances[1], side_distance)
        np.testing.assert_allclose(
            refined_sections[1][0],
            (side_distance, -side_extension),
        )
        np.testing.assert_allclose(
            refined_sections[-2][0],
            (1.0 - side_distance, -side_extension),
        )
        np.testing.assert_allclose(refined_sections[2][0], (0.5, -0.2))
        np.testing.assert_allclose(
            smaller_radius_sections[1][0],
            (side_distance, -(side_extension / 2.0)),
        )

        baseline_chord_point = (
            baseline_sections[0][0]
            + (
                (baseline_sections[1][0] - baseline_sections[0][0])
                * (side_distance / 0.5)
            )
        )
        self.assertLess(
            float(refined_sections[1][0][1]),
            float(baseline_chord_point[1]),
        )

    def test_starting_step_points_survive_rounded_tread_protection(
        self,
    ) -> None:
        route_sections = (
            (np.asarray((0.0, 0.0)), np.asarray((0.0, 1.0))),
            (np.asarray((0.21, 0.0)), np.asarray((0.21, 1.0))),
        )
        expected_protected_side_y = {
            STAIR_STARTING_STEP_BULLNOSE: (
                -0.019428571428571437,
                -0.037336899664593956,
            ),
            STAIR_STARTING_STEP_CURTAIL: (
                -0.014945054945054949,
                -0.0354797663299981,
            ),
        }

        for starting_step, expected_y in expected_protected_side_y.items():
            with self.subTest(starting_step=starting_step):
                results = [
                    _build_starting_step_route_sections(
                        route_sections,
                        (0.0, 0.21),
                        0.0,
                        0.21,
                        starting_step,
                        0.05,
                        points,
                        0.0,
                        (),
                        STAIR_TREAD_EDGE_ROUNDED,
                        0.04,
                    )
                    for points in (1, 2)
                ]
                baseline_distances, baseline_sections = results[0]
                refined_distances, refined_sections = results[1]

                # Rounded-edge protection inserts the same structural cuts
                # after filtering both point sets. Their lateral coordinates
                # must still follow the requested control polyline.
                np.testing.assert_allclose(
                    refined_distances,
                    baseline_distances,
                )
                np.testing.assert_allclose(
                    baseline_sections[1][0][1],
                    expected_y[0],
                )
                np.testing.assert_allclose(
                    refined_sections[1][0][1],
                    expected_y[1],
                )
                self.assertLess(
                    float(refined_sections[1][0][1]),
                    float(baseline_sections[1][0][1]),
                )
                np.testing.assert_allclose(
                    refined_sections[1][0][1],
                    refined_sections[-2][0][1],
                )

        levels = [_build_level(2, "Ground"), _build_level(3, "Story")]
        common_values = {
            "start_level_index": 2,
            "start_a_x": 0.0,
            "start_a_y": 0.0,
            "start_b_x": 0.0,
            "start_b_y": 50.0,
            "end_level_index": 3,
            "end_a_x": 200.0,
            "end_a_y": 0.0,
            "end_b_x": 200.0,
            "end_b_y": 50.0,
            "stair_type": STAIR_TYPE_FLOATING,
            "stringer_placement": STAIR_STRINGER_NONE,
            "tread_edge_profile": STAIR_TREAD_EDGE_ROUNDED,
            "tread_edge_radius_meters": 0.04,
            "starting_step_edge_radius_meters": 0.05,
        }
        for starting_step in expected_protected_side_y:
            with self.subTest(starting_step=starting_step, mesh=True):
                meshes = [
                    next(
                        target.mesh.copy()
                        for target in build_canvas_stair_part_targets(
                            levels,
                            (
                                StairData(
                                    **common_values,
                                    starting_step=starting_step,
                                    starting_step_edge_points=points,
                                ),
                            ),
                        )
                        if target.part_kind == STAIR_PART_TREADS
                    )
                    for points in (1, 2)
                ]
                self.assertGreater(
                    float(meshes[1].volume),
                    float(meshes[0].volume),
                )
                for mesh in meshes:
                    mesh.merge_vertices()
                    self.assertTrue(mesh.is_watertight)
                    self.assertTrue(mesh.is_winding_consistent)
                    self.assertTrue(mesh.is_volume)
                    self.assertGreater(float(mesh.volume), 0.0)

    def test_starting_step_points_add_symmetric_mesh_detail(
        self,
    ) -> None:
        levels = [_build_level(2, "Ground"), _build_level(3, "Story")]
        common_values = {
            "start_level_index": 2,
            "start_a_x": 0.0,
            "start_a_y": 0.0,
            "start_b_x": 0.0,
            "start_b_y": 50.0,
            "end_level_index": 3,
            "end_a_x": 200.0,
            "end_a_y": 0.0,
            "end_b_x": 200.0,
            "end_b_y": 50.0,
            "stair_type": STAIR_TYPE_FLOATING,
            "stringer_placement": STAIR_STRINGER_NONE,
            "tread_edge_profile": STAIR_TREAD_EDGE_STRAIGHT,
            "starting_step_edge_radius_meters": 0.15,
        }

        def tread_mesh(starting_step: str, points: int) -> trimesh.Trimesh:
            stair = StairData(
                **common_values,
                starting_step=starting_step,
                starting_step_edge_points=points,
            )
            return next(
                target.mesh.copy()
                for target in build_canvas_stair_part_targets(levels, [stair])
                if target.part_kind == STAIR_PART_TREADS
            )

        for starting_step in (
            STAIR_STARTING_STEP_BULLNOSE,
            STAIR_STARTING_STEP_CURTAIL,
        ):
            with self.subTest(starting_step=starting_step):
                baseline_mesh = tread_mesh(starting_step, 1)
                refined_mesh = tread_mesh(starting_step, 3)

                np.testing.assert_allclose(refined_mesh.bounds, baseline_mesh.bounds)
                self.assertGreater(len(refined_mesh.faces), len(baseline_mesh.faces))
                self.assertGreater(
                    float(refined_mesh.volume),
                    float(baseline_mesh.volume),
                )
                for mesh in (baseline_mesh, refined_mesh):
                    mesh.merge_vertices()
                    self.assertTrue(mesh.is_watertight)
                    self.assertTrue(mesh.is_winding_consistent)
                    self.assertTrue(mesh.is_volume)
                    self.assertGreater(float(mesh.volume), 0.0)

    def test_starting_step_shape_controls_are_ignored_when_starting_step_is_none(
        self,
    ) -> None:
        levels = [_build_level(2, "Ground"), _build_level(3, "Story")]
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
            starting_step=STAIR_STARTING_STEP_NONE,
            starting_step_edge_radius_meters=0.0,
            stringer_placement=STAIR_STRINGER_NONE,
        )
        zero_radius_mesh = next(
            target.mesh
            for target in build_canvas_stair_part_targets(levels, [stair])
            if target.part_kind == STAIR_PART_TREADS
        )
        maximum_radius_mesh = next(
            target.mesh
            for target in build_canvas_stair_part_targets(
                levels,
                [
                    replace(
                        stair,
                        starting_step_edge_radius_meters=2.0,
                        starting_step_edge_points=16,
                    )
                ],
            )
            if target.part_kind == STAIR_PART_TREADS
        )

        np.testing.assert_allclose(
            maximum_radius_mesh.vertices,
            zero_radius_mesh.vertices,
        )
        np.testing.assert_array_equal(
            maximum_radius_mesh.faces,
            zero_radius_mesh.faces,
        )

    def test_large_tread_radius_preserves_starting_step_plan_lobes(
        self,
    ) -> None:
        levels = [_build_level(2, "Ground"), _build_level(3, "Story")]
        common_values = {
            "start_level_index": 2,
            "start_a_x": 0.0,
            "start_a_y": 0.0,
            "start_b_x": 0.0,
            "start_b_y": 50.0,
            "end_level_index": 3,
            "end_a_x": 200.0,
            "end_a_y": 0.0,
            "end_b_x": 200.0,
            "end_b_y": 50.0,
            "stair_type": STAIR_TYPE_FLOATING,
            "stringer_placement": STAIR_STRINGER_NONE,
            "tread_edge_profile": STAIR_TREAD_EDGE_ROUNDED,
            "tread_edge_radius_meters": 1.0,
            "starting_step_edge_radius_meters": 0.20,
        }

        def tread_mesh(starting_step: str) -> trimesh.Trimesh:
            stair = StairData(**common_values, starting_step=starting_step)
            return next(
                target.mesh.copy()
                for target in build_canvas_stair_part_targets(levels, [stair])
                if target.part_kind == STAIR_PART_TREADS
            )

        ordinary_mesh = tread_mesh(STAIR_STARTING_STEP_NONE)
        for starting_step in (
            STAIR_STARTING_STEP_BULLNOSE,
            STAIR_STARTING_STEP_CURTAIL,
        ):
            with self.subTest(starting_step=starting_step):
                profiled_mesh = tread_mesh(starting_step)
                self.assertLess(
                    float(profiled_mesh.bounds[0, 1]),
                    float(ordinary_mesh.bounds[0, 1]),
                )
                self.assertGreater(
                    float(profiled_mesh.bounds[1, 1]),
                    float(ordinary_mesh.bounds[1, 1]),
                )
                profiled_mesh.merge_vertices()
                self.assertTrue(profiled_mesh.is_watertight)
                self.assertTrue(profiled_mesh.is_winding_consistent)
                self.assertTrue(profiled_mesh.is_volume)
                self.assertGreater(float(profiled_mesh.volume), 0.0)

    def test_starting_step_profiles_are_smooth_closed_and_rounded_at_front(
        self,
    ) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        common_values = {
            "start_level_index": 2,
            "start_a_x": 0.0,
            "start_a_y": 0.0,
            "start_b_x": 0.0,
            "start_b_y": 50.0,
            "end_level_index": 3,
            "end_a_x": 200.0,
            "end_a_y": 0.0,
            "end_b_x": 200.0,
            "end_b_y": 50.0,
            "stair_type": STAIR_TYPE_FLOATING,
            "tread_edge_profile": STAIR_TREAD_EDGE_ROUNDED,
            "tread_edge_radius_meters": 0.02,
            "stringer_placement": STAIR_STRINGER_NONE,
        }

        for starting_step in (
            STAIR_STARTING_STEP_BULLNOSE,
            STAIR_STARTING_STEP_CURTAIL,
        ):
            with self.subTest(starting_step=starting_step):
                stair = StairData(
                    **common_values,
                    starting_step=starting_step,
                    starting_step_edge_points=8,
                )
                tread_mesh = next(
                    target.mesh.copy()
                    for target in build_canvas_stair_part_targets(
                        [ground_level, upper_level],
                        [stair],
                    )
                    if target.part_kind == STAIR_PART_TREADS
                )
                lowest_tread_top = float(np.min(tread_mesh.vertices[:, 2])) + (
                    stair.tread_thickness_meters
                )
                lowest_tread_vertices = tread_mesh.vertices[
                    tread_mesh.vertices[:, 2] <= lowest_tread_top + 1e-6
                ]
                self.assertGreater(
                    len(np.unique(np.round(lowest_tread_vertices[:, :2], 7), axis=0)),
                    16,
                )
                self.assertTrue(
                    np.any(
                        (
                            lowest_tread_vertices[:, 2]
                            > np.min(lowest_tread_vertices[:, 2])
                        )
                        & (lowest_tread_vertices[:, 2] < lowest_tread_top)
                    )
                )
                lowest_tread_bottom = float(np.min(lowest_tread_vertices[:, 2]))
                rounded_center_z = (
                    lowest_tread_bottom + lowest_tread_top
                ) / 2.0
                leading_center_x = float(
                    np.min(
                        lowest_tread_vertices[
                            np.isclose(
                                lowest_tread_vertices[:, 2],
                                rounded_center_z,
                            ),
                            0,
                        ]
                    )
                )
                leading_top_x = float(
                    np.min(
                        lowest_tread_vertices[
                            np.isclose(
                                lowest_tread_vertices[:, 2],
                                lowest_tread_top,
                            ),
                            0,
                        ]
                    )
                )
                self.assertAlmostEqual(
                    leading_top_x - leading_center_x,
                    common_values["tread_edge_radius_meters"],
                )
                tread_mesh.merge_vertices()
                self.assertTrue(tread_mesh.is_watertight)
                self.assertTrue(tread_mesh.is_winding_consistent)
                self.assertTrue(tread_mesh.is_volume)
                self.assertGreater(float(tread_mesh.volume), 0.0)

    def test_starting_step_uses_same_physical_low_end_when_route_is_reversed(
        self,
    ) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        common_values = {
            "stair_type": STAIR_TYPE_FLOATING,
            "stringer_placement": STAIR_STRINGER_NONE,
            "starting_step": STAIR_STARTING_STEP_CURTAIL,
        }
        ascending = StairData(
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
            **common_values,
        )
        descending = StairData(
            start_level_index=3,
            start_a_x=200.0,
            start_a_y=50.0,
            start_b_x=200.0,
            start_b_y=0.0,
            end_level_index=2,
            end_a_x=0.0,
            end_a_y=50.0,
            end_b_x=0.0,
            end_b_y=0.0,
            **common_values,
        )

        def tread_vertices(stair: StairData) -> np.ndarray:
            mesh = next(
                target.mesh
                for target in build_canvas_stair_part_targets(
                    [ground_level, upper_level],
                    [stair],
                )
                if target.part_kind == STAIR_PART_TREADS
            )
            return np.unique(np.round(mesh.vertices, decimals=9), axis=0)

        np.testing.assert_allclose(
            tread_vertices(descending),
            tread_vertices(ascending),
        )

    def test_starting_step_profiles_follow_a_curved_route_without_open_seams(
        self,
    ) -> None:
        levels = [_build_level(2, "Ground"), _build_level(3, "Story")]
        curved_stair = _build_curved_stair(STAIR_STYLE_FLOATING)

        for starting_step in (
            STAIR_STARTING_STEP_BULLNOSE,
            STAIR_STARTING_STEP_CURTAIL,
        ):
            with self.subTest(starting_step=starting_step):
                stair = replace(curved_stair, starting_step=starting_step)
                tread_mesh = next(
                    target.mesh.copy()
                    for target in build_canvas_stair_part_targets(levels, [stair])
                    if target.part_kind == STAIR_PART_TREADS
                )
                self.assertTrue(np.all(np.isfinite(tread_mesh.vertices)))
                tread_mesh.merge_vertices()
                self.assertTrue(tread_mesh.is_watertight)
                self.assertTrue(tread_mesh.is_winding_consistent)
                self.assertTrue(tread_mesh.is_volume)
                self.assertGreater(float(tread_mesh.volume), 0.0)

    def test_descending_route_preserves_physical_left_stringer_side(self) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        common_values = {
            "stair_type": STAIR_TYPE_FLOATING,
            "stringer_placement": STAIR_STRINGER_LEFT,
        }
        ascending = StairData(
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
            **common_values,
        )
        descending = StairData(
            start_level_index=3,
            start_a_x=200.0,
            start_a_y=0.0,
            start_b_x=200.0,
            start_b_y=50.0,
            end_level_index=2,
            end_a_x=0.0,
            end_a_y=0.0,
            end_b_x=0.0,
            end_b_y=50.0,
            **common_values,
        )

        def stringer_vertices(stair: StairData) -> np.ndarray:
            target = next(
                part
                for part in build_canvas_stair_part_targets(
                    [ground_level, upper_level],
                    [stair],
                )
                if part.part_kind == STAIR_PART_STRINGERS
            )
            return np.unique(np.round(target.mesh.vertices, decimals=9), axis=0)

        np.testing.assert_allclose(
            stringer_vertices(descending),
            stringer_vertices(ascending),
        )

    def test_parameterized_tread_thickness_cannot_exceed_actual_rise(
        self,
    ) -> None:
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
            tread_thickness_meters=0.2,
            stringer_placement=STAIR_STRINGER_BOTH,
        )

        with self.assertRaisesRegex(
            ValueError,
            "cannot exceed its calculated actual step rise",
        ):
            build_stair_meshes([ground_level, upper_level], [stair])

    def test_legacy_stair_keeps_ceil_based_step_count(self) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        ground_level.height_meters = 2.7
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
            style=STAIR_STYLE_FLOATING,
        )

        mesh = _get_mesh_by_name([ground_level, upper_level], stair)

        self.assertEqual(len(mesh.faces), 18 * 12)

    def test_stringer_placement_and_rounded_edge_change_only_requested_parts(
        self,
    ) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        common_values = {
            "start_level_index": 2,
            "start_a_x": 0.0,
            "start_a_y": 0.0,
            "start_b_x": 0.0,
            "start_b_y": 50.0,
            "end_level_index": 3,
            "end_a_x": 200.0,
            "end_a_y": 0.0,
            "end_b_x": 200.0,
            "end_b_y": 50.0,
            "stair_type": STAIR_TYPE_FLOATING,
            "stringer_placement": STAIR_STRINGER_LEFT,
        }
        straight = StairData(
            **common_values,
            tread_edge_profile=STAIR_TREAD_EDGE_STRAIGHT,
        )
        rounded = StairData(
            **common_values,
            tread_edge_profile=STAIR_TREAD_EDGE_ROUNDED,
        )

        straight_parts = {
            target.part_kind: target.mesh
            for target in build_canvas_stair_part_targets(
                [ground_level, upper_level],
                [straight],
            )
        }
        rounded_parts = {
            target.part_kind: target.mesh
            for target in build_canvas_stair_part_targets(
                [ground_level, upper_level],
                [rounded],
            )
        }

        self.assertIn(STAIR_PART_STRINGERS, straight_parts)
        self.assertNotIn(STAIR_PART_SUPPORT, straight_parts)
        self.assertGreater(
            len(rounded_parts[STAIR_PART_TREADS].faces),
            len(straight_parts[STAIR_PART_TREADS].faces),
        )

    def test_explicit_no_stringer_keeps_modern_parts_without_stringer_mesh(
        self,
    ) -> None:
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
            stringer_placement=STAIR_STRINGER_NONE,
        )

        parts = build_canvas_stair_part_targets(
            [ground_level, upper_level],
            [stair],
        )
        part_kinds = {part.part_kind for part in parts}

        self.assertFalse(stair.uses_legacy_part_layout)
        self.assertIn(STAIR_PART_TREADS, part_kinds)
        self.assertIn(STAIR_PART_SUPPORT, part_kinds)
        self.assertNotIn(STAIR_PART_STRINGERS, part_kinds)

    def test_textured_stair_part_replaces_only_that_export_primitive(self) -> None:
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
            stringer_placement=STAIR_STRINGER_BOTH,
        )
        parts = build_canvas_stair_part_targets(
            [ground_level, upper_level],
            [stair],
        )
        tread_part = next(
            part for part in parts if part.part_kind == STAIR_PART_TREADS
        )

        model = convert_to_glb(
            [ground_level, upper_level],
            stairs=[stair],
            surface_materials={
                tread_part.semantic_id: _solid_png((180, 120, 60, 255))
            },
            export_untextured_surfaces=False,
        )

        self.assertEqual(len(model.scene.geometry), len(parts))
        matching_geometry = [
            mesh
            for mesh in model.scene.geometry.values()
            if mesh.metadata.get("housemaker_surface_id")
            == tread_part.semantic_id
        ]
        self.assertEqual(len(matching_geometry), 1)
        self.assertEqual(matching_geometry[0].visual.kind, "texture")

    def test_textured_supported_risers_leave_support_untextured(self) -> None:
        levels = [_build_level(2, "Ground"), _build_level(3, "Story")]
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
            stringer_placement=STAIR_STRINGER_BOTH,
        )
        parts_by_kind = {
            part.part_kind: part
            for part in build_canvas_stair_part_targets(levels, (stair,))
        }
        riser_id = parts_by_kind[STAIR_PART_RISERS].semantic_id
        support_id = parts_by_kind[STAIR_PART_SUPPORT].semantic_id
        model = convert_to_glb(
            levels,
            stairs=(stair,),
            surface_materials={riser_id: _solid_png((180, 120, 60, 255))},
            export_untextured_surfaces=False,
        )
        geometry_by_id = {
            mesh.metadata.get("housemaker_surface_id"): mesh
            for mesh in model.scene.geometry.values()
        }
        self.assertEqual(geometry_by_id[riser_id].visual.kind, "texture")
        self.assertNotEqual(geometry_by_id[support_id].visual.kind, "texture")
        all_preview_ranges = model.mesh.metadata.get(
            "_housemaker_preview_stair_face_ranges", ()
        )
        self.assertEqual(
            sum(end - start for start, end, stair_id in all_preview_ranges
                if stair_id == stair.stair_id),
            sum(len(part.mesh.faces) for part in parts_by_kind.values()),
        )
        self.assertIsNotNone(model.preview_untextured_mesh)
        assert model.preview_untextured_mesh is not None
        untextured_ranges = model.preview_untextured_mesh.metadata.get(
            "_housemaker_preview_stair_face_ranges", ()
        )
        self.assertEqual(
            sum(end - start for start, end, stair_id in untextured_ranges
                if stair_id == stair.stair_id),
            sum(
                len(part.mesh.faces)
                for kind, part in parts_by_kind.items()
                if kind != STAIR_PART_RISERS
            ),
        )

    def test_one_step_rounded_nose_exports_as_tread_without_riser(self) -> None:
        levels = [_build_level(2, "Ground"), _build_level(3, "Story")]
        for level in levels:
            level.height_meters = 0.08
            level.floor_thickness_meters = 0.02
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
            tread_thickness_meters=0.10,
            stringer_placement=STAIR_STRINGER_NONE,
            tread_edge_profile=STAIR_TREAD_EDGE_ROUNDED,
            nosing_placements=(
                STAIR_NOSING_FRONT,
                STAIR_NOSING_LEFT,
                STAIR_NOSING_RIGHT,
            ),
        )
        parts = build_canvas_stair_part_targets(levels, (stair,))
        self.assertEqual(
            {part.part_kind for part in parts},
            {STAIR_PART_TREADS},
        )
        tread_id = next(
            part.semantic_id
            for part in parts
            if part.part_kind == STAIR_PART_TREADS
        )
        model = convert_to_glb(
            levels,
            stairs=(stair,),
            surface_materials={tread_id: _solid_png((180, 120, 60, 255))},
            export_untextured_surfaces=False,
        )
        self.assertEqual(len(model.scene.geometry), 1)
        self.assertEqual(len(_read_glb_document(model.glb_bytes)["meshes"]), 1)
        matching = [
            mesh
            for mesh in model.scene.geometry.values()
            if mesh.metadata.get("housemaker_surface_id") == tread_id
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].visual.kind, "texture")

    def test_textured_stair_is_not_reintroduced_when_a_route_level_is_excluded(
        self,
    ) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        ground_level.vertex_data.add_edge(
            ground_level.vertex_data.vertices[0].id,
            ground_level.vertex_data.vertices[1].id,
        )
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
            stringer_placement=STAIR_STRINGER_BOTH,
        )
        tread_part = next(
            part
            for part in build_canvas_stair_part_targets(
                [ground_level, upper_level],
                [stair],
            )
            if part.part_kind == STAIR_PART_TREADS
        )
        upper_level.include_in_export = False

        model = convert_to_glb(
            [ground_level, upper_level],
            stairs=[stair],
            surface_materials={
                tread_part.semantic_id: _solid_png((180, 120, 60, 255))
            },
        )

        self.assertEqual(model.preview_stair_parts, [])
        self.assertFalse(
            any(
                mesh.metadata.get("housemaker_stair_id") == stair.stair_id
                for mesh in model.scene.geometry.values()
            )
        )

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
        lower_floor_top = ground_level.floor_thickness_meters
        upper_floor_top = (
            lower_floor_top
            + ground_level.height_meters
            + upper_level.floor_thickness_meters
        )

        self.assertAlmostEqual(
            float(supported_mesh.bounds[0, 2]),
            lower_floor_top,
        )
        self.assertGreater(
            float(floating_mesh.bounds[0, 2]),
            lower_floor_top,
        )
        self.assertAlmostEqual(
            float(supported_mesh.bounds[1, 2]),
            upper_floor_top,
        )
        self.assertAlmostEqual(
            float(floating_mesh.bounds[1, 2]),
            upper_floor_top,
        )
        self.assertGreater(float(supported_mesh.volume), float(floating_mesh.volume))
        self.assertAlmostEqual(
            float(floating_with_riser_mesh.bounds[1, 2]),
            upper_floor_top,
        )
        self.assertGreater(
            float(floating_with_riser_mesh.bounds[0, 2]),
            lower_floor_top,
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

        surfaces_omitted_model = convert_to_glb(
            [ground_level, upper_level],
            stairs=[supported_stair, floating_stair, floating_with_riser_stair],
            export_untextured_surfaces=False,
        )
        self.assertEqual(
            set(surfaces_omitted_model.scene.geometry),
            {
                "stair_1_supported",
                "stair_2_floating",
                "stair_3_floating_with_riser",
            },
        )

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

        route_sections = _build_smoothed_stair_route_sections(control_sections)
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
        normalized_sections = _normalize_stair_rail_correspondence(clicked_sections)

        for (section_a_xy, section_b_xy), route_direction in zip(
            normalized_sections,
            expected_route_directions,
        ):
            width_xy = section_b_xy - section_a_xy
            handedness = (route_direction[0] * width_xy[1]) - (
                route_direction[1] * width_xy[0]
            )
            self.assertGreater(handedness, 0.0)

        sampled_sections = _build_smoothed_stair_route_sections(normalized_sections)
        sampled_centers = [
            (section_a_xy + section_b_xy) / 2.0
            for section_a_xy, section_b_xy in sampled_sections
        ]
        for section_index, (section_a_xy, section_b_xy) in enumerate(sampled_sections):
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
            handedness = (route_delta[0] * width_xy[1]) - (route_delta[1] * width_xy[0])
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
        floor_to_floor_rise = (
            ground_level.height_meters + upper_level.floor_thickness_meters
        )
        step_count = math.ceil(floor_to_floor_rise / DEFAULT_STAIR_RISER_HEIGHT_METERS)

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
            level_image_to_world_xy(ground_level, x, y) for x, y in stair.start_points
        ] + [level_image_to_world_xy(upper_level, x, y) for x, y in stair.end_points]
        _assert_mesh_contains_xy_points(self, mesh, initial_points)

        ground_level.offset_x_meters += 2.0
        upper_level.scale = 1.25
        updated_mesh = _get_mesh_by_name([ground_level, upper_level], stair)
        updated_points = [
            level_image_to_world_xy(ground_level, x, y) for x, y in stair.start_points
        ] + [level_image_to_world_xy(upper_level, x, y) for x, y in stair.end_points]
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
        lower_floor_top = ground_level.floor_thickness_meters
        upper_floor_top = (
            lower_floor_top
            + ground_level.height_meters
            + upper_level.floor_thickness_meters
        )
        self.assertAlmostEqual(
            float(mesh.bounds[0, 2]),
            lower_floor_top,
        )
        self.assertAlmostEqual(
            float(mesh.bounds[1, 2]),
            upper_floor_top,
        )

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
