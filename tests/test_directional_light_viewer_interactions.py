# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from housemaker.directional_light_state import DirectionalLightData
from housemaker.glb import GeneratedModel, PreviewPlacedObject
from housemaker.viewer import (
    DIRECTIONAL_LIGHT_ARROW_LENGTH_SCALE,
    TRANSFORM_GIZMO_ROTATE,
    TRANSFORM_GIZMO_TRANSLATE,
    GlbViewerWidget,
    _build_tour_direction_arrow_positions,
    _DepthTestedOverlayLineItem,
    _get_directional_light_marker_size_pixels,
    _TransformGizmoHandle,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _light(
    *,
    light_id: str = "directional-light-1",
    intensity: float = 1.0,
    position: tuple[float, float, float] = (0.0, 0.0, 1.0),
    target: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> DirectionalLightData:
    return DirectionalLightData(
        light_id=light_id,
        name=light_id.replace("-", " ").title(),
        position=position,
        target=target,
        intensity=intensity,
    )


def _box_model() -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    return GeneratedModel(
        mesh=mesh,
        scene=trimesh.Scene(mesh),
        glb_bytes=b"",
    )


def _box_model_with_placed_object() -> GeneratedModel:
    base_mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    local_mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    placement_transform = np.eye(4, dtype=float)
    placement_transform[:3, 3] = (5.0, 0.0, 0.0)
    placed_object = PreviewPlacedObject(
        object_id="placed-object-1",
        meshes=(local_mesh,),
        placement_transform=placement_transform,
        world_position=(5.0, 0.0, 0.0),
        rotation_degrees=(0.0, 0.0, 0.0),
    )
    return GeneratedModel(
        mesh=base_mesh,
        scene=trimesh.Scene(base_mesh),
        glb_bytes=b"",
        preview_base_mesh=base_mesh,
        preview_placed_objects=[placed_object],
    )


def _process_events() -> None:
    _qt_application.processEvents()


# ### Viewer interaction tests ###
class DirectionalLightViewerInteractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.viewer = GlbViewerWidget(window_editing_enabled=True)
        self.viewer.resize(900, 650)
        self.viewer.show()
        _process_events()

    def tearDown(self) -> None:
        self.viewer.close()
        self.viewer.deleteLater()
        _process_events()

    def test_selected_light_owns_transform_gizmos_wheel_and_status(self) -> None:
        self.viewer.set_directional_lights((_light(intensity=1.4),))

        self.assertTrue(
            self.viewer.set_selected_directional_light_id("directional-light-1")
        )

        self.assertEqual(len(self.viewer._directional_light_gizmo_items), 9)
        self.assertTrue(self.viewer.view._directional_light_wheel_steps_enabled)
        self.assertIn(
            "intensity 1.4",
            self.viewer.object_transform_status_label.text(),
        )
        self.assertIn(
            "RGB ring",
            self.viewer.object_transform_status_label.text(),
        )

        emitted: list[tuple[str, int]] = []
        self.viewer.directional_light_intensity_step_requested.connect(
            lambda light_id, steps: emitted.append((light_id, steps))
        )
        self.viewer._handle_directional_light_wheel_steps_requested(2)
        self.assertEqual(emitted, [("directional-light-1", 2)])

    def test_marker_size_scales_gently_with_intensity(self) -> None:
        self.assertEqual(
            _get_directional_light_marker_size_pixels(0.0, selected=False),
            15.0,
        )
        self.assertEqual(
            _get_directional_light_marker_size_pixels(1.0, selected=False),
            18.0,
        )
        self.assertEqual(
            _get_directional_light_marker_size_pixels(4.0, selected=False),
            21.0,
        )
        self.assertEqual(
            _get_directional_light_marker_size_pixels(1.0, selected=True),
            22.0,
        )

        self.viewer.set_directional_lights((_light(intensity=4.0),))
        marker = self.viewer._directional_light_overlay_items[0]
        self.assertEqual(marker.size, 21.0)

    def test_directional_arrow_is_longer_without_changing_tour_default(self) -> None:
        default_positions = _build_tour_direction_arrow_positions(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 5.0),
        )
        directional_positions = _build_tour_direction_arrow_positions(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 5.0),
            length_scale=DIRECTIONAL_LIGHT_ARROW_LENGTH_SCALE,
        )

        assert default_positions is not None
        assert directional_positions is not None
        default_length = float(
            np.linalg.norm(default_positions[1] - default_positions[0])
        )
        directional_length = float(
            np.linalg.norm(directional_positions[1] - directional_positions[0])
        )
        self.assertAlmostEqual(
            directional_length,
            default_length * DIRECTIONAL_LIGHT_ARROW_LENGTH_SCALE,
        )

    def test_selected_light_projection_grid_is_depth_tested_and_removable(
        self,
    ) -> None:
        self.viewer.set_model(_box_model())
        self.viewer.set_directional_lights(
            (_light(position=(0.0, 0.0, 3.0)),)
        )

        self.viewer.set_selected_directional_light_id("directional-light-1")

        grid_item = self.viewer._directional_light_projection_grid_item
        self.assertIsInstance(grid_item, _DepthTestedOverlayLineItem)
        assert grid_item is not None
        self.assertGreater(len(grid_item.pos), 0)

        self.viewer.set_selected_directional_light_id(None)
        self.assertIsNone(self.viewer._directional_light_projection_grid_item)
        self.assertNotIn(grid_item, self.viewer.view.items)

    def test_projection_grid_reuses_translation_and_intensity_results(self) -> None:
        self.viewer.set_model(_box_model())
        initial = _light(position=(0.0, 0.0, 3.0))
        translated = _light(
            intensity=2.0,
            position=(2.0, 0.0, 3.0),
            target=(2.0, 0.0, 0.0),
        )
        rotated = _light(
            intensity=2.0,
            position=(2.0, 0.0, 3.0),
            target=(3.0, 0.0, 3.0),
        )
        positions = np.asarray(
            ((-1.0, 0.0, 1.003), (1.0, 0.0, 1.003)),
            dtype=np.float32,
        )
        with patch(
            "housemaker.viewer.build_directional_light_projection_grid",
            return_value=positions,
        ) as build_grid:
            self.viewer.set_directional_lights((initial,))
            self.viewer.set_selected_directional_light_id("directional-light-1")
            self.assertEqual(build_grid.call_count, 1)

            self.viewer.set_directional_lights((initial.with_intensity(2.0),))
            self.viewer.set_directional_lights((translated,))
            self.assertEqual(build_grid.call_count, 1)

            self.viewer.set_directional_lights((rotated,))
            self.assertEqual(build_grid.call_count, 2)

            self.viewer._canvas_selection_geometry_revision += 1
            self.viewer._refresh_directional_light_overlay_items()
            self.assertEqual(build_grid.call_count, 3)

    def test_empty_projection_grid_result_is_cached(self) -> None:
        self.viewer.set_model(_box_model())
        with patch(
            "housemaker.viewer.build_directional_light_projection_grid",
            return_value=np.empty((0, 3), dtype=np.float32),
        ) as build_grid:
            self.viewer.set_directional_lights(
                (_light(position=(0.0, 0.0, 3.0)),)
            )
            self.viewer.set_selected_directional_light_id("directional-light-1")
            self.viewer._refresh_directional_light_overlay_items()

        self.assertEqual(build_grid.call_count, 1)
        self.assertIsNone(self.viewer._directional_light_projection_grid_item)

    def test_projection_collision_mesh_tracks_placed_object_transform(self) -> None:
        self.viewer.set_model(_box_model_with_placed_object())

        initial_mesh = (
            self.viewer._get_directional_light_projection_collision_mesh()
        )
        assert initial_mesh is not None
        self.assertAlmostEqual(float(initial_mesh.bounds[1, 0]), 5.5)

        group = self.viewer._placed_object_render_groups["placed-object-1"]
        group.current_transform[0, 3] = 7.0
        moved_mesh = self.viewer._get_directional_light_projection_collision_mesh()

        assert moved_mesh is not None
        self.assertIsNot(moved_mesh, initial_mesh)
        self.assertAlmostEqual(float(moved_mesh.bounds[1, 0]), 7.5)

    def test_rotation_ring_can_be_picked_away_from_translation_axes(self) -> None:
        self.viewer.set_directional_lights((_light(),))
        self.viewer.set_selected_directional_light_id("directional-light-1")
        self.viewer._directional_light_gizmo_size = 1.0
        diagonal = 0.72 / np.sqrt(2.0)

        handle = self.viewer._pick_directional_light_gizmo_handle(
            np.asarray((5.0, diagonal, 1.0 + diagonal), dtype=float),
            np.asarray((-1.0, 0.0, 0.0), dtype=float),
        )

        self.assertEqual(
            handle,
            _TransformGizmoHandle(TRANSFORM_GIZMO_ROTATE, 0),
        )

    def test_light_drag_translates_position_and_target_by_the_same_delta(self) -> None:
        self.viewer.set_directional_lights((_light(),))
        self.viewer.set_selected_directional_light_id("directional-light-1")
        preview_events: list[tuple[str, object, object]] = []
        final_events: list[tuple[str, object, object]] = []
        self.viewer.directional_light_transform_preview_changed.connect(
            lambda *payload: preview_events.append(payload)
        )
        self.viewer.directional_light_transform_changed.connect(
            lambda *payload: final_events.append(payload)
        )
        handle = _TransformGizmoHandle(TRANSFORM_GIZMO_TRANSLATE, 0)

        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=(
                np.asarray((0.0, 0.0, 5.0), dtype=float),
                np.asarray((0.0, 0.0, -1.0), dtype=float),
            ),
        ) as camera_ray:
            self.assertTrue(
                self.viewer._begin_directional_light_gizmo_drag(
                    handle,
                    QPointF(),
                )
            )
            camera_ray.return_value = (
                np.asarray((2.0, 0.0, 5.0), dtype=float),
                np.asarray((0.0, 0.0, -1.0), dtype=float),
            )
            self.assertTrue(
                self.viewer._update_directional_light_gizmo_drag(QPointF(20.0, 0.0))
            )
            self.assertTrue(
                self.viewer._finish_directional_light_gizmo_drag(QPointF(20.0, 0.0))
            )

        self.assertEqual(
            preview_events[-1],
            ("directional-light-1", (2.0, 0.0, 1.0), (2.0, 0.0, 0.0)),
        )
        self.assertEqual(final_events[-1], preview_events[-1])
        np.testing.assert_allclose(
            np.asarray(final_events[-1][2]) - np.asarray(final_events[-1][1]),
            (0.0, 0.0, -1.0),
        )

    def test_light_rotation_changes_only_target_and_preserves_length(self) -> None:
        self.viewer.set_directional_lights((_light(),))
        self.viewer.set_selected_directional_light_id("directional-light-1")
        preview_events: list[tuple[str, object, object]] = []
        final_events: list[tuple[str, object, object]] = []
        self.viewer.directional_light_transform_preview_changed.connect(
            lambda *payload: preview_events.append(payload)
        )
        self.viewer.directional_light_transform_changed.connect(
            lambda *payload: final_events.append(payload)
        )
        handle = _TransformGizmoHandle(TRANSFORM_GIZMO_ROTATE, 0)

        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=(
                np.asarray((5.0, 1.0, 1.0), dtype=float),
                np.asarray((-1.0, 0.0, 0.0), dtype=float),
            ),
        ) as camera_ray:
            self.assertTrue(
                self.viewer._begin_directional_light_gizmo_drag(
                    handle,
                    QPointF(),
                )
            )
            camera_ray.return_value = (
                np.asarray((5.0, 0.0, 2.0), dtype=float),
                np.asarray((-1.0, 0.0, 0.0), dtype=float),
            )
            self.assertTrue(
                self.viewer._update_directional_light_gizmo_drag(
                    QPointF(20.0, 0.0)
                )
            )
            self.assertTrue(
                self.viewer._finish_directional_light_gizmo_drag(
                    QPointF(20.0, 0.0)
                )
            )

        final_light_id, final_position, final_target = final_events[-1]
        self.assertEqual(final_light_id, "directional-light-1")
        np.testing.assert_allclose(final_position, (0.0, 0.0, 1.0), atol=1e-9)
        np.testing.assert_allclose(final_target, (0.0, 1.0, 1.0), atol=1e-9)
        self.assertAlmostEqual(
            float(
                np.linalg.norm(
                    np.asarray(final_target) - np.asarray(final_position)
                )
            ),
            1.0,
        )
        self.assertEqual(final_events[-1], preview_events[-1])

    def test_selection_change_rolls_back_an_interrupted_light_drag(self) -> None:
        self.viewer.set_directional_lights(
            (
                _light(),
                _light(light_id="directional-light-2"),
            )
        )
        self.viewer.set_selected_directional_light_id("directional-light-1")
        preview_events: list[tuple[str, object, object]] = []
        self.viewer.directional_light_transform_preview_changed.connect(
            lambda *payload: preview_events.append(payload)
        )

        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=(
                np.asarray((0.0, 0.0, 5.0), dtype=float),
                np.asarray((0.0, 0.0, -1.0), dtype=float),
            ),
        ) as camera_ray:
            self.assertTrue(
                self.viewer._begin_directional_light_gizmo_drag(
                    _TransformGizmoHandle(TRANSFORM_GIZMO_TRANSLATE, 0),
                    QPointF(),
                )
            )
            camera_ray.return_value = (
                np.asarray((2.0, 0.0, 5.0), dtype=float),
                np.asarray((0.0, 0.0, -1.0), dtype=float),
            )
            self.assertTrue(
                self.viewer._update_directional_light_gizmo_drag(QPointF(20.0, 0.0))
            )

        self.assertTrue(
            self.viewer.set_selected_directional_light_id("directional-light-2")
        )
        self.assertEqual(
            preview_events[-1],
            ("directional-light-1", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
        )
        self.assertIsNone(self.viewer._directional_light_drag)

    def test_selection_change_rolls_back_an_interrupted_light_rotation(self) -> None:
        self.viewer.set_directional_lights(
            (
                _light(),
                _light(light_id="directional-light-2"),
            )
        )
        self.viewer.set_selected_directional_light_id("directional-light-1")
        preview_events: list[tuple[str, object, object]] = []
        self.viewer.directional_light_transform_preview_changed.connect(
            lambda *payload: preview_events.append(payload)
        )

        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=(
                np.asarray((5.0, 1.0, 1.0), dtype=float),
                np.asarray((-1.0, 0.0, 0.0), dtype=float),
            ),
        ) as camera_ray:
            self.assertTrue(
                self.viewer._begin_directional_light_gizmo_drag(
                    _TransformGizmoHandle(TRANSFORM_GIZMO_ROTATE, 0),
                    QPointF(),
                )
            )
            camera_ray.return_value = (
                np.asarray((5.0, 0.0, 2.0), dtype=float),
                np.asarray((-1.0, 0.0, 0.0), dtype=float),
            )
            self.assertTrue(
                self.viewer._update_directional_light_gizmo_drag(
                    QPointF(20.0, 0.0)
                )
            )

        self.assertTrue(
            self.viewer.set_selected_directional_light_id("directional-light-2")
        )
        self.assertEqual(
            preview_events[-1],
            ("directional-light-1", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
        )
        self.assertIsNone(self.viewer._directional_light_drag)

    def test_scene_point_placement_uses_fallback_plane_and_is_one_shot(self) -> None:
        placed: list[object] = []
        self.viewer.directional_light_placed.connect(placed.append)
        self.assertTrue(
            self.viewer.begin_directional_light_placement(fallback_plane_z=2.5)
        )

        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=(
                np.asarray((1.0, 2.0, 8.0), dtype=float),
                np.asarray((0.0, 0.0, -1.0), dtype=float),
            ),
        ):
            self.viewer._handle_tour_pointer_pressed(QPointF(100.0, 100.0))
            self.viewer._handle_tour_pointer_released(QPointF(100.0, 100.0))

        self.assertEqual(placed, [(1.0, 2.0, 2.5)])
        self.assertFalse(self.viewer.is_directional_light_placement_active)
        self.assertIsNone(self.viewer._directional_light_placement_hover_item)

    def test_marker_pick_precedes_wall_and_object_selection(self) -> None:
        self.viewer.set_directional_lights((_light(),))

        with (
            patch.object(
                self.viewer,
                "_pick_directional_light_id",
                return_value="directional-light-1",
            ),
            patch.object(self.viewer.view, "build_camera_ray") as camera_ray,
        ):
            self.viewer._handle_window_wall_pick_requested(QPointF())

        camera_ray.assert_not_called()
        self.assertEqual(
            self.viewer.get_selected_directional_light_id(),
            "directional-light-1",
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
