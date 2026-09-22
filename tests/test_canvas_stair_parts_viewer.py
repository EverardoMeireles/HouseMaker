# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PySide6.QtCore import QEvent, QPointF, QRect, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from housemaker.glb import (
    STAIR_PART_RISERS,
    STAIR_PART_STRINGERS,
    STAIR_PART_SUPPORT,
    STAIR_PART_SURFACE_TYPE,
    STAIR_PART_TREADS,
    GeneratedModel,
    PreviewPlacedObject,
    PreviewStairPart,
    _combine_preview_mesh_geometry,
    build_canvas_stair_part_targets,
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
    STAIR_STRINGER_NONE,
    STAIR_TREAD_EDGE_ROUNDED,
    STAIR_TREAD_EDGE_STRAIGHT,
    LevelData,
    StairData,
    VertexData,
)
from housemaker.surface_geometry import SURFACE_TYPE_WALL, FixedSurface
from housemaker.viewer import (
    ATLAS_SURFACE_HIGHLIGHT_COLOR,
    CANVAS_SELECTION_TARGET_STAIR_PART,
    CANVAS_STAIR_PREVIEW_MAX_OPACITY,
    GlbViewerWidget,
    _CanvasRectangleSelectionResult,
    _get_nearest_preview_stair_part_ray_hit,
    _rasterize_canvas_target_selection,
    _remove_semantic_stair_faces,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_stair_part(
    part_kind: str,
    *,
    center: tuple[float, float, float],
    stair_index: int = 0,
    stair_id: str = "stair-uuid",
    level_indices: tuple[int, ...] = (0, 1),
    two_components: bool = False,
) -> PreviewStairPart:
    mesh = trimesh.creation.box(extents=(1.2, 0.8, 0.24))
    mesh.apply_translation(center)
    if two_components:
        second = mesh.copy()
        second.apply_translation((0.0, 1.0, 0.25))
        mesh = trimesh.util.concatenate((mesh, second))
    return PreviewStairPart(
        stair_id=stair_id,
        stair_index=stair_index,
        semantic_id=(
            f"stair:{stair_id}/part:{part_kind}:"
            f"{STAIR_PART_SURFACE_TYPE[part_kind]}"
        ),
        part_kind=part_kind,
        surface_type=STAIR_PART_SURFACE_TYPE[part_kind],
        mesh=mesh,
        level_indices=level_indices,
    )


def _build_model(*parts: PreviewStairPart) -> GeneratedModel:
    meshes = tuple(part.mesh.copy() for part in parts)
    mesh = (
        trimesh.util.concatenate(meshes)
        if meshes
        else trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    )
    return GeneratedModel(
        mesh=mesh,
        scene=trimesh.Scene(mesh),
        glb_bytes=b"",
        preview_stair_parts=list(parts),
    )


def _projected_quad(depth: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            (
                (0.0, 0.0, depth, 1.0),
                (20.0, 0.0, depth, 1.0),
                (20.0, 20.0, depth, 1.0),
                (0.0, 20.0, depth, 1.0),
            ),
            dtype=float,
        ),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
    )


def _build_wall() -> FixedSurface:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            (
                (-2.0, 2.0, 0.0),
                (2.0, 2.0, 0.0),
                (2.0, 2.0, 2.0),
                (-2.0, 2.0, 2.0),
            ),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id="level:0/wall:0:1",
        surface_type=SURFACE_TYPE_WALL,
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=8.0,
        wall_key="0:1",
        wall_start_world=(-2.0, 2.0, 0.0),
        wall_end_world=(2.0, 2.0, 0.0),
        wall_height_meters=2.0,
    )


# ### Viewer stair-part tests ###
class CanvasStairPartViewerTests(unittest.TestCase):
    def test_one_step_supported_nose_selects_tread_without_support(
        self,
    ) -> None:
        vertices = VertexData()
        vertices.add_vertex(0.0, 0.0)
        vertices.add_vertex(200.0, 100.0)
        levels = [
            LevelData(
                index=index,
                name=str(index),
                height_meters=0.08,
                floor_thickness_meters=0.02,
                vertex_data=vertices,
            )
            for index in (2, 3)
        ]
        for profile, placements, starting_step in (
            (
                STAIR_TREAD_EDGE_STRAIGHT,
                (STAIR_NOSING_FRONT,),
                STAIR_STARTING_STEP_NONE,
            ),
            (
                STAIR_TREAD_EDGE_ROUNDED,
                (STAIR_NOSING_FRONT,),
                STAIR_STARTING_STEP_NONE,
            ),
            (
                STAIR_TREAD_EDGE_ROUNDED,
                (STAIR_NOSING_FRONT, STAIR_NOSING_LEFT, STAIR_NOSING_RIGHT),
                STAIR_STARTING_STEP_NONE,
            ),
            (
                STAIR_TREAD_EDGE_STRAIGHT,
                (STAIR_NOSING_FRONT,),
                STAIR_STARTING_STEP_BULLNOSE,
            ),
            (
                STAIR_TREAD_EDGE_STRAIGHT,
                (STAIR_NOSING_FRONT,),
                STAIR_STARTING_STEP_CURTAIL,
            ),
        ):
            with self.subTest(
                profile=profile,
                placements=placements,
                starting_step=starting_step,
            ):
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
                    tread_edge_profile=profile,
                    nosing_placements=placements,
                    starting_step=starting_step,
                    starting_step_edge_radius_meters=0.12,
                    starting_step_edge_points=3,
                )
                self.assertEqual(stair.calculate_step_layout(0.10)[0], 1)
                targets = tuple(build_canvas_stair_part_targets(levels, (stair,)))
                self.assertEqual(
                    {part.part_kind for part in targets},
                    {STAIR_PART_TREADS},
                )
                hit = _get_nearest_preview_stair_part_ray_hit(
                    targets, (-1.0, -0.5, 0.07), (1.0, 0.0, 0.0)
                )
                self.assertIsNotNone(hit)
                assert hit is not None
                self.assertEqual(hit[0].part_kind, STAIR_PART_TREADS)

    def test_one_step_plain_front_without_overhang_remains_riser(self) -> None:
        vertices = VertexData()
        vertices.add_vertex(0.0, 0.0)
        vertices.add_vertex(200.0, 100.0)
        levels = [
            LevelData(
                index=index,
                name=str(index),
                height_meters=0.08,
                floor_thickness_meters=0.02,
                vertex_data=vertices,
            )
            for index in (2, 3)
        ]
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
            tread_overhang_meters=0.0,
            nosing_placements=(),
            stringer_placement=STAIR_STRINGER_NONE,
            tread_edge_profile=STAIR_TREAD_EDGE_STRAIGHT,
        )
        targets = tuple(build_canvas_stair_part_targets(levels, (stair,)))
        self.assertEqual(
            {part.part_kind for part in targets},
            {STAIR_PART_TREADS, STAIR_PART_RISERS},
        )
        hit = _get_nearest_preview_stair_part_ray_hit(
            targets, (-1.0, -0.5, 0.07), (1.0, 0.0, 0.0)
        )
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit[0].part_kind, STAIR_PART_RISERS)

    def test_supported_riser_support_and_stringer_have_distinct_ray_targets(
        self,
    ) -> None:
        vertices = VertexData()
        vertices.add_vertex(0.0, 0.0)
        vertices.add_vertex(200.0, 100.0)
        levels = [
            LevelData(index=index, name=str(index), vertex_data=vertices)
            for index in (2, 3)
        ]
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
        targets = tuple(build_canvas_stair_part_targets(levels, (stair,)))
        for origin, direction, expected_part in (
            # The lower exposed support band is a riser; the leading tread
            # overhang is part of the tread, not the riser behind it.
            ((-1.0, -0.5, 0.34), (1.0, 0.0, 0.0), STAIR_PART_RISERS),
            ((-1.0, -0.5, 0.42), (1.0, 0.0, 0.0), STAIR_PART_TREADS),
            ((1.0, -1.5, 0.5), (0.0, 1.0, 0.0), STAIR_PART_SUPPORT),
            ((1.0, -1.5, 1.0), (0.0, 1.0, 0.0), STAIR_PART_STRINGERS),
        ):
            with self.subTest(part=expected_part):
                hit = _get_nearest_preview_stair_part_ray_hit(
                    targets, origin, direction
                )
                self.assertIsNotNone(hit)
                assert hit is not None
                self.assertEqual(hit[0].part_kind, expected_part)

        self.viewer.set_canvas_scene_levels(((2, "Ground"), (3, "Upper")))
        self.viewer.set_canvas_stair_part_targets(targets)
        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=(
                np.asarray((-1.0, -0.5, 0.42), dtype=float),
                np.asarray((1.0, 0.0, 0.0), dtype=float),
            ),
        ):
            self.viewer._handle_window_wall_pick_requested(QPointF())
        tread_id = next(
            target.semantic_id
            for target in targets
            if target.part_kind == STAIR_PART_TREADS
        )
        self.assertEqual(self.viewer.get_selected_canvas_stair_part_ids(), (tread_id,))

        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=(
                np.asarray((-1.0, -0.5, 0.34), dtype=float),
                np.asarray((1.0, 0.0, 0.0), dtype=float),
            ),
        ):
            self.viewer._handle_window_wall_pick_requested(QPointF())
        riser_id = next(
            target.semantic_id
            for target in targets
            if target.part_kind == STAIR_PART_RISERS
        )
        self.assertEqual(self.viewer.get_selected_canvas_stair_part_ids(), (riser_id,))

        rounded = StairData(
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
            tread_edge_profile=STAIR_TREAD_EDGE_ROUNDED,
            nosing_placements=(
                STAIR_NOSING_FRONT,
                STAIR_NOSING_LEFT,
                STAIR_NOSING_RIGHT,
            ),
        )
        rounded_targets = tuple(
            build_canvas_stair_part_targets(levels, (rounded,))
        )
        rounded_hit = _get_nearest_preview_stair_part_ray_hit(
            rounded_targets, (-1.0, -0.5, 0.42), (1.0, 0.0, 0.0)
        )
        self.assertIsNotNone(rounded_hit)
        assert rounded_hit is not None
        self.assertEqual(rounded_hit[0].part_kind, STAIR_PART_TREADS)

    def test_supported_starting_step_curves_keep_nose_with_treads(self) -> None:
        vertices = VertexData()
        vertices.add_vertex(0.0, 0.0)
        vertices.add_vertex(200.0, 100.0)
        levels = [
            LevelData(index=index, name=str(index), vertex_data=vertices)
            for index in (2, 3)
        ]
        for starting_step in (
            STAIR_STARTING_STEP_BULLNOSE,
            STAIR_STARTING_STEP_CURTAIL,
        ):
            with self.subTest(starting_step=starting_step):
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
                    nosing_placements=(STAIR_NOSING_FRONT,),
                    starting_step=starting_step,
                    starting_step_edge_radius_meters=0.12,
                    starting_step_edge_points=3,
                    stringer_placement=STAIR_STRINGER_NONE,
                )
                targets = tuple(build_canvas_stair_part_targets(levels, (stair,)))
                self.assertTrue(
                    {STAIR_PART_TREADS, STAIR_PART_RISERS}
                    <= {part.part_kind for part in targets}
                )
                for z, expected_part in (
                    (0.34, STAIR_PART_RISERS),
                    (0.42, STAIR_PART_TREADS),
                ):
                    hit = _get_nearest_preview_stair_part_ray_hit(
                        targets, (-1.0, -0.5, z), (1.0, 0.0, 0.0)
                    )
                    self.assertIsNotNone(hit)
                    assert hit is not None
                    self.assertEqual(hit[0].part_kind, expected_part)

    def test_hidden_lower_support_front_is_not_a_riser_pick_target(self) -> None:
        vertices = VertexData()
        vertices.add_vertex(0.0, 0.0)
        vertices.add_vertex(200.0, 100.0)
        ground = LevelData(
            index=2,
            name="Ground",
            height_meters=0.52,
            floor_thickness_meters=0.02,
            vertex_data=vertices,
        )
        upper = LevelData(
            index=3,
            name="Upper",
            floor_thickness_meters=0.02,
            vertex_data=vertices,
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
            tread_thickness_meters=0.05,
            tread_overhang_meters=0.04,
            nosing_placements=(STAIR_NOSING_FRONT,),
            stringer_placement=STAIR_STRINGER_NONE,
        )
        base_z_by_level = build_level_base_z_lookup((ground, upper))
        floor_top = base_z_by_level[ground.index]
        step_count, step_rise = stair.calculate_step_layout(
            base_z_by_level[upper.index] - floor_top
        )
        self.assertGreater(step_count, 1)
        start_x = level_image_to_world_xy(ground, 0.0, 0.0)[0]
        end_x = level_image_to_world_xy(upper, 200.0, 0.0)[0]
        front_x = start_x + (end_x - start_x) / step_count
        riser_target = next(
            target
            for target in build_canvas_stair_part_targets((ground, upper), (stair,))
            if target.part_kind == STAIR_PART_RISERS
        )
        direction = (1.0, 0.0, 0.0)
        hidden_origin = (front_x - 0.01, -0.5, floor_top + step_rise * 0.5)
        self.assertIsNone(
            _get_nearest_preview_stair_part_ray_hit(
                (riser_target,), hidden_origin, direction
            )
        )
        visible_origin = (front_x - 0.01, -0.5, floor_top + step_rise * 1.25)
        visible_hit = _get_nearest_preview_stair_part_ray_hit(
            (riser_target,), visible_origin, direction
        )
        self.assertIsNotNone(visible_hit)
        assert visible_hit is not None
        self.assertEqual(visible_hit[0].part_kind, STAIR_PART_RISERS)

    def setUp(self) -> None:
        self.viewer = GlbViewerWidget(window_editing_enabled=True)
        self.viewer.set_canvas_scene_levels(((0, "Ground"), (1, "Upper")))

    def tearDown(self) -> None:
        self.viewer.clear_canvas_stair_preview()
        self.viewer.close()
        self.viewer.deleteLater()
        _qt_application.processEvents()

    def _install(self, *parts: PreviewStairPart) -> None:
        self.viewer.set_model(_build_model(*parts))
        self.viewer.set_canvas_stair_part_targets(parts)

    def test_semantic_parts_select_as_whole_stable_targets(self) -> None:
        treads = _build_stair_part(
            STAIR_PART_TREADS,
            center=(0.0, 0.0, 0.5),
            two_components=True,
        )
        risers = _build_stair_part(
            STAIR_PART_RISERS,
            center=(2.0, 0.0, 0.5),
        )
        self._install(treads, risers)
        emitted: list[object] = []
        self.viewer.canvas_stair_part_selection_changed.connect(emitted.append)

        self.assertTrue(
            self.viewer.select_canvas_stair_part_target(treads.semantic_id)
        )
        self.assertTrue(
            self.viewer.select_canvas_stair_part_target(
                risers.semantic_id,
                additive=True,
            )
        )

        self.assertEqual(
            self.viewer.get_selected_canvas_stair_part_ids(),
            (treads.semantic_id, risers.semantic_id),
        )
        self.assertEqual(
            emitted,
            [
                (treads.semantic_id,),
                (treads.semantic_id, risers.semantic_id),
            ],
        )
        resolved = self.viewer.get_canvas_stair_part_target(treads.semantic_id)
        self.assertIs(resolved, treads)
        assert resolved is not None
        self.assertEqual(resolved.stair_index, 0)
        self.assertEqual(resolved.part_kind, STAIR_PART_TREADS)
        self.assertEqual(len(self.viewer._canvas_stair_part_selection_items), 2)

    def test_delete_selected_stair_part_requests_its_whole_stair(self) -> None:
        treads = _build_stair_part(
            STAIR_PART_TREADS,
            center=(0.0, 0.0, 0.5),
        )
        risers = _build_stair_part(
            STAIR_PART_RISERS,
            center=(2.0, 0.0, 0.5),
        )
        self._install(treads, risers)
        requested: list[object] = []
        self.viewer.canvas_stair_deletion_requested.connect(requested.append)
        self.viewer.set_selected_canvas_stair_part_ids(
            (treads.semantic_id, risers.semantic_id)
        )

        self.viewer._handle_view_delete_requested()

        self.assertEqual(requested, [(treads.stair_id,)])
        self.assertEqual(self.viewer.get_selected_canvas_stair_part_ids(), ())

    def test_clicking_any_component_selects_one_part_not_one_step(self) -> None:
        treads = _build_stair_part(
            STAIR_PART_TREADS,
            center=(0.0, 0.0, 0.5),
            two_components=True,
        )
        self._install(treads)

        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=(
                np.asarray((0.0, -2.0, 0.5), dtype=float),
                np.asarray((0.0, 1.0, 0.0), dtype=float),
            ),
        ):
            self.viewer._handle_window_wall_pick_requested(QPointF())

        self.assertEqual(
            self.viewer.get_selected_canvas_stair_part_ids(),
            (treads.semantic_id,),
        )

    def test_surface_and_object_selections_clear_stair_parts(self) -> None:
        part = _build_stair_part(
            STAIR_PART_TREADS,
            center=(0.0, 0.0, 0.5),
        )
        object_mesh = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
        placed_object = PreviewPlacedObject(
            object_id="chair",
            meshes=(object_mesh,),
            placement_transform=np.eye(4, dtype=float),
            world_position=(0.0, 0.0, 0.0),
            rotation_degrees=(0.0, 0.0, 0.0),
        )
        model = _build_model(part)
        model.preview_placed_objects = [placed_object]
        model.preview_base_mesh = part.mesh.copy()
        self.viewer.set_model(model)
        self.viewer.set_canvas_stair_part_targets((part,))
        wall = _build_wall()
        self.viewer.set_wall_targets((wall,))

        self.viewer.set_selected_canvas_stair_part_ids((part.semantic_id,))
        self.viewer.set_selected_placed_object_ids(("chair",))
        self.assertEqual(self.viewer.get_selected_canvas_stair_part_ids(), ())

        self.viewer.set_selected_canvas_stair_part_ids((part.semantic_id,))
        self.viewer.set_selected_canvas_surface_ids((wall.surface_id,))
        self.assertEqual(self.viewer.get_selected_canvas_stair_part_ids(), ())

    def test_staged_preview_pulses_separately_and_escape_discards_it(self) -> None:
        original = _build_stair_part(
            STAIR_PART_TREADS,
            center=(0.0, 0.0, 0.5),
        )
        staged = _build_stair_part(
            STAIR_PART_TREADS,
            center=(3.0, 0.0, 0.5),
        )
        self._install(original)
        self.viewer.select_canvas_stair_part_target(original.semantic_id)
        original_vertices = np.asarray(self.viewer.model.mesh.vertices).copy()
        cancelled: list[bool] = []
        self.viewer.canvas_stair_preview_cancelled.connect(
            lambda: cancelled.append(True)
        )

        self.assertTrue(self.viewer.set_canvas_stair_preview(0, (staged,)))
        self.assertEqual(len(self.viewer._canvas_stair_preview_groups), 1)
        group = self.viewer._canvas_stair_preview_groups[0]
        self.assertEqual(
            float(group.mesh_item.opts["color"][3]),
            CANVAS_STAIR_PREVIEW_MAX_OPACITY,
        )
        self.assertFalse(self.viewer.set_canvas_stair_preview(0, (staged,)))
        self.assertIs(
            self.viewer._canvas_stair_preview_groups[0].mesh_item,
            group.mesh_item,
        )
        first_edge_opacity = float(group.mesh_item.opts["edgeColor"][3])
        with patch.object(group.mesh_item, "setMeshData") as mesh_data_update:
            self.viewer._advance_canvas_stair_preview_fade()
            mesh_data_update.assert_not_called()

        self.assertNotEqual(
            float(group.mesh_item.opts["edgeColor"][3]),
            first_edge_opacity,
        )
        np.testing.assert_allclose(
            self.viewer.model.mesh.vertices,
            original_vertices,
        )
        self.viewer.view.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            )
        )

        self.assertEqual(cancelled, [True])
        self.assertIsNone(self.viewer._canvas_stair_preview_stair_index)
        self.assertEqual(self.viewer._canvas_stair_preview_parts, ())
        self.assertEqual(self.viewer._canvas_stair_preview_groups, [])
        np.testing.assert_allclose(
            self.viewer.model.mesh.vertices,
            original_vertices,
        )

    def test_preview_hides_only_its_original_stair_and_restores_it(self) -> None:
        selected = _build_stair_part(
            STAIR_PART_TREADS, center=(0.0, 0.0, 0.5)
        )
        other = _build_stair_part(
            STAIR_PART_RISERS,
            center=(3.0, 0.0, 0.5),
            stair_index=1,
            stair_id="other-stair",
        )
        staged = _build_stair_part(
            STAIR_PART_TREADS, center=(6.0, 0.0, 0.5)
        )
        self._install(selected, other)
        self.assertTrue(self.viewer.get_hide_stair_mesh_when_previewing())
        self.viewer.select_canvas_stair_part_target(selected.semantic_id)
        self.viewer.set_highlighted_canvas_stair_part_ids(
            (selected.semantic_id, other.semantic_id)
        )
        self.assertEqual(len(self.viewer._canvas_stair_part_selection_items), 1)
        self.assertEqual(len(self.viewer._atlas_stair_part_highlight_items), 2)

        self.viewer.set_canvas_stair_preview(0, (staged,))
        display_mesh = self.viewer._get_display_mesh()
        assert display_mesh is not None
        self.assertEqual(len(display_mesh.faces), len(other.mesh.faces))
        self.assertEqual(len(self.viewer._canvas_stair_preview_groups), 1)
        self.assertEqual(
            self.viewer.get_selected_canvas_stair_part_ids(),
            (selected.semantic_id,),
        )
        self.assertEqual(len(self.viewer._canvas_stair_part_selection_items), 0)
        self.assertEqual(len(self.viewer._atlas_stair_part_highlight_items), 1)

        revised = _build_stair_part(
            STAIR_PART_TREADS, center=(7.0, 0.0, 0.5)
        )
        with patch.object(
            self.viewer, "_repopulate_canvas_scene_preserving_camera"
        ) as rebuild:
            self.viewer.set_canvas_stair_preview(0, (revised,))
            rebuild.assert_not_called()

        self.viewer.clear_canvas_stair_preview()
        self.assertEqual(len(self.viewer._canvas_stair_part_selection_items), 1)
        self.assertEqual(len(self.viewer._atlas_stair_part_highlight_items), 2)
        self.viewer.set_canvas_stair_preview(0, (revised,))

        self.viewer.select_canvas_stair_part_target(None)
        restored = self.viewer._get_display_mesh()
        assert restored is not None
        self.assertEqual(
            len(restored.faces), len(selected.mesh.faces) + len(other.mesh.faces)
        )
        self.assertEqual(self.viewer._canvas_stair_preview_groups, [])

    def test_preview_hidden_stair_is_excluded_from_ray_and_box_picking(
        self,
    ) -> None:
        original = _build_stair_part(
            STAIR_PART_TREADS, center=(0.0, 0.0, 0.5)
        )
        other = _build_stair_part(
            STAIR_PART_RISERS,
            center=(3.0, 0.0, 0.5),
            stair_index=1,
            stair_id="other-stair",
        )
        staged = _build_stair_part(
            STAIR_PART_TREADS, center=(6.0, 0.0, 0.5)
        )
        self._install(original, other)
        self.viewer.set_canvas_stair_preview(0, (staged,))

        with (
            patch.object(
                self.viewer.view,
                "build_camera_ray",
                return_value=(
                    np.asarray((0.0, -2.0, 0.5), dtype=float),
                    np.asarray((0.0, 1.0, 0.0), dtype=float),
                ),
            ),
            patch(
                "housemaker.viewer._get_nearest_preview_stair_part_ray_hit",
                return_value=None,
            ) as stair_ray_hit,
        ):
            self.viewer._handle_window_wall_pick_requested(QPointF())
        ray_candidates = stair_ray_hit.call_args.args[0]
        self.assertEqual(
            tuple(part.semantic_id for part in ray_candidates),
            (other.semantic_id,),
        )

        self.viewer.set_canvas_stair_preview(0, (staged,))

        with patch(
            "housemaker.viewer._capture_face_selection_raster_input",
            return_value=(
                (
                    (np.empty((0, 4), dtype=float), np.empty((0, 3), dtype=np.int64))
                    for _ in range(2)
                ),
                (0, 0, 50, 50),
            ),
        ):
            captured = self.viewer._capture_canvas_rectangle_selection_targets(
                QPointF(0.0, 0.0), QPointF(50.0, 50.0)
            )
        assert captured is not None
        projected_targets, _rectangle = captured
        box_stair_ids = tuple(
            target_id
            for target_type, target_id, *_geometry in projected_targets
            if target_type == CANVAS_SELECTION_TARGET_STAIR_PART
        )
        self.assertEqual(box_stair_ids, (other.semantic_id,))

    def test_preview_setting_toggles_original_stair_mid_edit(self) -> None:
        original = _build_stair_part(
            STAIR_PART_TREADS, center=(0.0, 0.0, 0.5)
        )
        staged = _build_stair_part(
            STAIR_PART_TREADS, center=(3.0, 0.0, 0.5)
        )
        self._install(original)
        self.viewer.select_canvas_stair_part_target(original.semantic_id)
        self.viewer.set_highlighted_canvas_stair_part_ids(
            (original.semantic_id,)
        )
        self.viewer.set_canvas_stair_preview(0, (staged,))
        hidden = self.viewer._get_display_mesh()
        assert hidden is not None
        self.assertEqual(len(hidden.faces), 0)
        self.assertEqual(len(self.viewer._canvas_stair_part_selection_items), 0)
        self.assertEqual(len(self.viewer._atlas_stair_part_highlight_items), 0)

        self.viewer.set_hide_stair_mesh_when_previewing(False)
        restored = self.viewer._get_display_mesh()
        assert restored is not None
        self.assertEqual(len(restored.faces), len(original.mesh.faces))
        self.assertEqual(len(self.viewer._canvas_stair_preview_groups), 1)
        self.assertEqual(len(self.viewer._canvas_stair_part_selection_items), 1)
        self.assertEqual(len(self.viewer._atlas_stair_part_highlight_items), 1)

        self.viewer.set_hide_stair_mesh_when_previewing(True)
        hidden_again = self.viewer._get_display_mesh()
        assert hidden_again is not None
        self.assertEqual(len(hidden_again.faces), 0)
        self.assertEqual(len(self.viewer._canvas_stair_part_selection_items), 0)
        self.assertEqual(len(self.viewer._atlas_stair_part_highlight_items), 0)
        self.viewer.clear_canvas_stair_preview()
        restored_again = self.viewer._get_display_mesh()
        assert restored_again is not None
        self.assertEqual(len(restored_again.faces), len(original.mesh.faces))
        self.assertEqual(len(self.viewer._canvas_stair_part_selection_items), 1)
        self.assertEqual(len(self.viewer._atlas_stair_part_highlight_items), 1)

    def test_stair_face_quota_preserves_coincident_unrelated_surface(self) -> None:
        stair = _build_stair_part(
            STAIR_PART_TREADS, center=(0.0, 0.0, 0.5)
        )
        unrelated = stair.mesh.copy()
        combined = trimesh.util.concatenate((stair.mesh, unrelated))

        filtered = _remove_semantic_stair_faces(combined, (stair,))

        self.assertEqual(len(filtered.faces), len(unrelated.faces))

    def test_preview_face_ranges_preserve_earlier_and_later_coincident_faces(
        self,
    ) -> None:
        duplicate = trimesh.Trimesh(
            vertices=np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float),
            faces=np.asarray(((0, 1, 2),), dtype=np.int64),
            process=False,
        )
        unrelated = trimesh.Trimesh(
            vertices=np.asarray(
                (
                    (0, 0, 0), (1, 0, 0), (0, 1, 0),
                    (2, 0, 0), (3, 0, 0), (2, 1, 0),
                ),
                dtype=float,
            ),
            faces=np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64),
            process=False,
        )
        stair = _build_stair_part(STAIR_PART_RISERS, center=(0, 0, 0))
        stair = PreviewStairPart(
            stair_id=stair.stair_id,
            stair_index=stair.stair_index,
            semantic_id=stair.semantic_id,
            part_kind=stair.part_kind,
            surface_type=stair.surface_type,
            mesh=duplicate.copy(),
            level_indices=stair.level_indices,
        )
        stair.mesh.metadata["housemaker_stair_id"] = stair.stair_id
        base = _combine_preview_mesh_geometry((unrelated, stair.mesh))
        placed = duplicate.copy()
        scene_mesh = _combine_preview_mesh_geometry((base, placed))

        filtered = _remove_semantic_stair_faces(scene_mesh, (stair,))

        self.assertEqual(len(filtered.faces), 3)
        np.testing.assert_allclose(filtered.triangles[0], unrelated.triangles[0])
        np.testing.assert_allclose(filtered.triangles[1], unrelated.triangles[1])
        np.testing.assert_allclose(filtered.triangles[2], placed.triangles[0])

        # With every authored stair face assigned to its separate textured
        # item, the untextured preview may contain only a coincident wall.
        untextured_only = _combine_preview_mesh_geometry((unrelated,))
        self.assertEqual(
            untextured_only.metadata["_housemaker_preview_stair_face_ranges"],
            (),
        )
        untouched = _remove_semantic_stair_faces(untextured_only, (stair,))
        np.testing.assert_allclose(untouched.triangles, unrelated.triangles)

    def test_preview_start_and_clear_update_base_mesh_without_repopulating(
        self,
    ) -> None:
        original = _build_stair_part(STAIR_PART_TREADS, center=(0, 0, 0.5))
        staged = _build_stair_part(STAIR_PART_TREADS, center=(2, 0, 0.5))
        self._install(original)
        texture_item = Mock()
        self.viewer.textured_surface_items.append(texture_item)
        self.viewer._textured_surface_item_ids[texture_item] = original.semantic_id
        self.assertIsNotNone(self.viewer.mesh_item)
        assert self.viewer.mesh_item is not None

        with patch.object(self.viewer, "_populate_scene") as repopulate:
            self.assertTrue(self.viewer.set_canvas_stair_preview(0, (staged,)))
            repopulate.assert_not_called()
            self.assertFalse(self.viewer.mesh_item.visible())
            texture_item.setVisible.assert_called_with(False)
            self.viewer.set_textures_enabled(False)
            texture_item.setVisible.assert_called_with(False)
            self.assertTrue(self.viewer.clear_canvas_stair_preview())
            repopulate.assert_not_called()
            self.assertTrue(self.viewer.mesh_item.visible())
            texture_item.setVisible.assert_called_with(False)
            self.viewer.set_textures_enabled(True)
            texture_item.setVisible.assert_called_with(True)

    def test_transient_undo_and_deselect_cancel_a_staged_preview(self) -> None:
        part = _build_stair_part(
            STAIR_PART_TREADS,
            center=(0.0, 0.0, 0.5),
        )
        self._install(part)
        self.viewer.select_canvas_stair_part_target(part.semantic_id)
        cancelled: list[bool] = []
        self.viewer.canvas_stair_preview_cancelled.connect(
            lambda: cancelled.append(True)
        )

        self.viewer.set_canvas_stair_preview(0, (part,))
        self.assertTrue(
            self.viewer.cancel_uncommitted_canvas_interaction_for_undo()
        )
        self.viewer.set_canvas_stair_preview(0, (part,))
        self.viewer.select_canvas_stair_part_target(None)

        self.assertEqual(cancelled, [True, True])
        self.assertFalse(
            self.viewer.cancel_uncommitted_canvas_interaction_for_undo()
        )

    def test_box_result_selects_stair_parts_through_semantic_ids(self) -> None:
        treads = _build_stair_part(
            STAIR_PART_TREADS,
            center=(0.0, 0.0, 0.5),
        )
        risers = _build_stair_part(
            STAIR_PART_RISERS,
            center=(2.0, 0.0, 0.5),
        )
        self._install(treads, risers)

        self.viewer._apply_canvas_rectangle_selection_targets(
            _CanvasRectangleSelectionResult(
                request_revision=(
                    self.viewer._canvas_rectangle_selection_request_revision
                ),
                geometry_revision=self.viewer._canvas_selection_geometry_revision,
                surface_ids=(treads.semantic_id, risers.semantic_id),
                object_ids=(),
                additive=False,
            )
        )

        self.assertEqual(
            self.viewer.get_selected_canvas_stair_part_ids(),
            (treads.semantic_id, risers.semantic_id),
        )

    def test_hidden_level_excludes_target_selection_and_preview(self) -> None:
        lower = _build_stair_part(
            STAIR_PART_TREADS,
            center=(0.0, 0.0, 0.5),
            level_indices=(0,),
        )
        self._install(lower)
        self.viewer.select_canvas_stair_part_target(lower.semantic_id)

        self.viewer.set_visible_canvas_level_indices((1,))

        self.assertEqual(self.viewer.get_selected_canvas_stair_part_ids(), ())
        self.assertFalse(
            self.viewer.select_canvas_stair_part_target(lower.semantic_id)
        )
        display_mesh = self.viewer._get_display_mesh()
        self.assertIsNotNone(display_mesh)
        assert display_mesh is not None
        self.assertEqual(len(display_mesh.faces), 0)
        captured = self.viewer._capture_canvas_rectangle_selection_targets(
            QPointF(0.0, 0.0),
            QPointF(50.0, 50.0),
        )
        if captured is not None:
            targets, _rectangle = captured
            self.assertNotIn(
                (CANVAS_SELECTION_TARGET_STAIR_PART, lower.semantic_id),
                tuple(
                    (target_type, target_id)
                    for target_type, target_id, *_geometry in targets
                ),
            )
        self.viewer.set_canvas_stair_preview(0, (lower,))
        self.assertEqual(self.viewer._canvas_stair_preview_groups, [])

    def test_green_atlas_highlight_does_not_change_selection(self) -> None:
        part = _build_stair_part(
            STAIR_PART_RISERS,
            center=(0.0, 0.0, 0.5),
        )
        self._install(part)

        self.assertTrue(
            self.viewer.set_highlighted_canvas_stair_part_ids(
                (part.semantic_id, "missing")
            )
        )

        self.assertEqual(
            self.viewer.get_highlighted_canvas_stair_part_ids(),
            (part.semantic_id,),
        )
        self.assertEqual(self.viewer.get_selected_canvas_stair_part_ids(), ())
        self.assertEqual(len(self.viewer._atlas_stair_part_highlight_items), 1)
        self.assertEqual(
            tuple(self.viewer._atlas_stair_part_highlight_items[0].color),
            ATLAS_SURFACE_HIGHLIGHT_COLOR,
        )


# ### Stair target-raster tests ###
class CanvasStairPartRasterTests(unittest.TestCase):
    def test_stair_target_uses_semantic_selection_channel(self) -> None:
        semantic_id = "stair:stable-id/part:treads:floor"

        surface_ids, object_ids = _rasterize_canvas_target_selection(
            (
                (
                    CANVAS_SELECTION_TARGET_STAIR_PART,
                    semantic_id,
                    *_projected_quad(-0.2),
                ),
            ),
            QRect(0, 0, 21, 21),
        )

        self.assertEqual(surface_ids, (semantic_id,))
        self.assertEqual(object_ids, ())


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
