# ### Environment setup ###
from __future__ import annotations

import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QScrollArea, QWidget

from housemaker.glb import convert_to_glb
from housemaker.models import (
    DEFAULT_FLOOR_THICKNESS_METERS,
    DEFAULT_LEVEL_OFFSET_METERS,
    DEFAULT_LEVEL_SCALE,
    MAX_FLOOR_THICKNESS_METERS,
    MIN_FLOOR_THICKNESS_METERS,
    LevelData,
    RoomData,
    VertexData,
    create_default_levels,
)
from housemaker.project_io import load_project, save_project

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)
_qt_widgets: list[object] = []


# ### Fixture helpers ###
def _add_closed_wall_loop(
    vertex_data: VertexData,
    points: list[tuple[float, float]],
) -> tuple[int, ...]:
    vertices = [vertex_data.add_vertex(*point) for point in points]
    for start_vertex, end_vertex in zip(
        vertices,
        vertices[1:] + vertices[:1],
    ):
        vertex_data.add_edge(start_vertex.id, end_vertex.id)

    return tuple(vertex.id for vertex in vertices)


def _build_square_level(
    index: int = 2,
    name: str = "Ground",
    *,
    with_room: bool = False,
) -> LevelData:
    vertex_data = VertexData()
    outer_vertex_ids = _add_closed_wall_loop(
        vertex_data,
        [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
    )
    rooms: list[RoomData] = []
    if with_room:
        room_center = vertex_data.add_vertex(50.0, 50.0)
        rooms.append(
            RoomData(
                name="Room",
                vertex_ids=outer_vertex_ids,
                center_vertex_id=room_center.id,
                color_rgb=(140, 180, 220),
            )
        )

    return LevelData(
        index=index,
        name=name,
        vertex_data=vertex_data,
        rooms=rooms,
    )


def _add_inner_room(level: LevelData) -> None:
    room_vertex_ids = _add_closed_wall_loop(
        level.vertex_data,
        [(25.0, 25.0), (75.0, 25.0), (75.0, 75.0), (25.0, 75.0)],
    )
    room_center = level.vertex_data.add_vertex(50.0, 50.0)
    level.rooms.append(
        RoomData(
            name="Room",
            vertex_ids=room_vertex_ids,
            center_vertex_id=room_center.id,
            color_rgb=(140, 180, 220),
        )
    )


def _get_scene_world_mesh(
    scene: trimesh.Scene,
    object_name: str,
) -> trimesh.Trimesh:
    transform, geometry_name = scene.graph.get(object_name)
    mesh = scene.geometry[geometry_name].copy()
    mesh.apply_transform(transform)
    return mesh


def _send_wheel_event(widget: QWidget, angle_delta_y: int) -> None:
    local_position = QPointF(widget.rect().center())
    global_position = QPointF(widget.mapToGlobal(widget.rect().center()))
    event = QWheelEvent(
        local_position,
        global_position,
        QPoint(),
        QPoint(0, angle_delta_y),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(widget, event)


# ### Level transform tests ###
class LevelTransformTests(unittest.TestCase):
    def test_level_scale_applies_to_every_retained_level_object_and_glb_node(
        self,
    ) -> None:
        level = _build_square_level()
        _add_inner_room(level)
        level.scale = 2.0

        model = convert_to_glb([level])
        preview_points = np.asarray(
            [
                point
                for preview_wall in model.preview_textured_walls
                for point in (
                    preview_wall.start_point,
                    preview_wall.end_point,
                )
            ]
        )
        np.testing.assert_allclose(
            preview_points[:, :2].min(axis=0),
            np.asarray([0.0, -2.0]),
        )
        np.testing.assert_allclose(
            preview_points[:, :2].max(axis=0),
            np.asarray([2.0, 0.0]),
        )
        self.assertTrue(
            all(
                preview_wall.height_meters == 3.0
                for preview_wall in model.preview_textured_walls
            )
        )

        expected_world_bounds = {
            "l2_ground_floor": np.asarray(
                [[-1.0, 0.0, -1.0], [3.0, 0.3, 3.0]]
            ),
            "l2_ground": np.asarray(
                [[-1.0, 0.3, -1.0], [3.0, 3.3, 3.0]]
            ),
            "l2_ground_room_1": np.asarray(
                [[0.0, 0.3, 0.0], [2.0, 3.3, 2.0]]
            ),
        }
        self._assert_scene_node_transforms(
            model.scene,
            expected_world_bounds,
            expected_translation=np.asarray([-1.0, 0.0, -1.0]),
        )

        loaded_scene = trimesh.load(BytesIO(model.glb_bytes), file_type="glb")
        self.assertIsInstance(loaded_scene, trimesh.Scene)
        self._assert_scene_node_transforms(
            loaded_scene,
            expected_world_bounds,
            expected_translation=np.asarray([-1.0, 0.0, -1.0]),
            atol=1e-6,
        )

    def test_level_offsets_apply_to_preview_and_every_retained_exported_object(
        self,
    ) -> None:
        level = _build_square_level()
        _add_inner_room(level)
        level.scale = 2.0
        level.offset_x_meters = 1.25
        level.offset_y_meters = -0.75

        model = convert_to_glb([level])
        preview_points = np.asarray(
            [
                point
                for preview_wall in model.preview_textured_walls
                for point in (
                    preview_wall.start_point,
                    preview_wall.end_point,
                )
            ]
        )
        np.testing.assert_allclose(
            preview_points[:, :2].min(axis=0),
            np.asarray([1.25, -2.75]),
        )
        np.testing.assert_allclose(
            preview_points[:, :2].max(axis=0),
            np.asarray([3.25, -0.75]),
        )

        expected_world_bounds = {
            "l2_ground_floor": np.asarray(
                [[0.25, 0.0, -0.25], [4.25, 0.3, 3.75]]
            ),
            "l2_ground": np.asarray(
                [[0.25, 0.3, -0.25], [4.25, 3.3, 3.75]]
            ),
            "l2_ground_room_1": np.asarray(
                [[1.25, 0.3, 0.75], [3.25, 3.3, 2.75]]
            ),
        }
        expected_translation = np.asarray([0.25, 0.0, -0.25])
        self._assert_scene_node_transforms(
            model.scene,
            expected_world_bounds,
            expected_translation,
        )

        loaded_scene = trimesh.load(BytesIO(model.glb_bytes), file_type="glb")
        self.assertIsInstance(loaded_scene, trimesh.Scene)
        self._assert_scene_node_transforms(
            loaded_scene,
            expected_world_bounds,
            expected_translation,
            atol=1e-6,
        )

    def test_invalid_level_scale_is_rejected(self) -> None:
        level = _build_square_level()
        for invalid_scale in (0.0, -0.1, float("nan"), float("inf"), True):
            with self.subTest(invalid_scale=invalid_scale):
                level.scale = invalid_scale
                with self.assertRaisesRegex(ValueError, "scale"):
                    convert_to_glb([level])

    def test_invalid_level_offsets_are_rejected(self) -> None:
        level = _build_square_level()
        for attribute_name in ("offset_x_meters", "offset_y_meters"):
            for invalid_offset in (
                float("nan"),
                float("inf"),
                float("-inf"),
                True,
                "not-a-number",
            ):
                with self.subTest(
                    attribute_name=attribute_name,
                    invalid_offset=invalid_offset,
                ):
                    level.scale = DEFAULT_LEVEL_SCALE
                    level.offset_x_meters = DEFAULT_LEVEL_OFFSET_METERS
                    level.offset_y_meters = DEFAULT_LEVEL_OFFSET_METERS
                    setattr(level, attribute_name, invalid_offset)
                    with self.assertRaisesRegex(ValueError, "offset"):
                        convert_to_glb([level])

    def _assert_scene_node_transforms(
        self,
        scene: trimesh.Scene,
        expected_world_bounds: dict[str, np.ndarray],
        expected_translation: np.ndarray,
        *,
        atol: float = 0.0,
    ) -> None:
        for object_name, expected_bounds in expected_world_bounds.items():
            transform, _ = scene.graph.get(object_name)
            np.testing.assert_allclose(
                np.diag(transform)[:3],
                np.asarray([2.0, 1.0, 2.0]),
                atol=atol,
            )
            np.testing.assert_allclose(
                transform[:3, 3],
                expected_translation,
                atol=atol,
            )
            np.testing.assert_allclose(
                _get_scene_world_mesh(scene, object_name).bounds,
                expected_bounds,
                atol=atol,
            )


# ### Project compatibility tests ###
class LevelProjectCompatibilityTests(unittest.TestCase):
    def test_floor_thickness_persists_and_legacy_contour_is_ignored(
        self,
    ) -> None:
        levels = create_default_levels()
        levels[2].scale = 1.75
        levels[2].offset_x_meters = 1.25
        levels[2].offset_y_meters = -0.75
        levels[2].floor_thickness_meters = 0.37

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "level-project.json"
            save_project(project_path, 2, levels)

            payload = json.loads(project_path.read_text(encoding="utf-8"))
            saved_level = payload["levels"][2]
            self.assertAlmostEqual(saved_level["floor_thickness_meters"], 0.37)
            self.assertNotIn("floor_contour_vertex_ids", saved_level)

            saved_level["floor_contour_vertex_ids"] = [1, 2, 3, 4]
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            loaded_level = load_project(project_path).levels[2]

            self.assertAlmostEqual(loaded_level.scale, 1.75)
            self.assertAlmostEqual(loaded_level.offset_x_meters, 1.25)
            self.assertAlmostEqual(loaded_level.offset_y_meters, -0.75)
            self.assertAlmostEqual(loaded_level.floor_thickness_meters, 0.37)
            self.assertFalse(hasattr(loaded_level, "floor_contour_vertex_ids"))

            saved_level.pop("floor_thickness_meters")
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            legacy_level = load_project(project_path).levels[2]
            self.assertAlmostEqual(
                legacy_level.floor_thickness_meters,
                DEFAULT_FLOOR_THICKNESS_METERS,
            )


# ### Level controls tests ###
class LevelControlsTests(unittest.TestCase):
    def test_floor_thickness_control_updates_level_and_active_preview(
        self,
    ) -> None:
        from housemaker.main import BlueprintWorkspace

        workspace = BlueprintWorkspace()
        _qt_widgets.append(workspace)
        workspace.current_level.vertex_data = VertexData()
        _add_closed_wall_loop(
            workspace.current_level.vertex_data,
            [
                (0.0, 0.0),
                (100.0, 0.0),
                (100.0, 100.0),
                (0.0, 100.0),
            ],
        )
        workspace._sync_canvas_to_current_level()
        workspace.resize(1400, 850)
        workspace.show()
        workspace.workspace_tabs.setCurrentWidget(
            workspace.canvas_viewer_workspace
        )
        workspace.canvas_viewer_tabs.setCurrentIndex(
            workspace.canvas_3d_view_tab_index
        )
        _qt_application.processEvents()
        self.assertAlmostEqual(
            workspace.floor_thickness_spinbox.value(),
            DEFAULT_FLOOR_THICKNESS_METERS,
        )
        self.assertAlmostEqual(
            workspace.floor_thickness_spinbox.minimum(),
            MIN_FLOOR_THICKNESS_METERS,
        )
        self.assertAlmostEqual(
            workspace.floor_thickness_spinbox.maximum(),
            MAX_FLOOR_THICKNESS_METERS,
        )
        original_other_level_thickness = (
            workspace.levels[1].floor_thickness_meters
        )
        revision_before = workspace._viewer_preview_revision

        with patch.object(
            workspace,
            "_build_viewer_preview_model",
            wraps=workspace._build_viewer_preview_model,
        ) as build_preview:
            workspace.floor_thickness_spinbox.setValue(0.65)

            self.assertTrue(
                workspace._canvas_surface_mesh_update_timer.isActive()
            )
            self.assertTrue(workspace._pending_canvas_surface_mesh_update)
            self.assertEqual(
                workspace._viewer_preview_revision,
                revision_before,
            )
            self.assertAlmostEqual(
                workspace._build_viewer_preview_levels()[
                    workspace.current_level_index
                ].floor_thickness_meters,
                DEFAULT_FLOOR_THICKNESS_METERS,
            )
            workspace._run_scheduled_viewer_preview_refresh()
            build_preview.assert_not_called()

            workspace._commit_pending_canvas_surface_mesh_update()

            self.assertFalse(
                workspace._canvas_surface_mesh_update_timer.isActive()
            )
            self.assertFalse(workspace._pending_canvas_surface_mesh_update)
            self.assertAlmostEqual(
                workspace._build_viewer_preview_levels()[
                    workspace.current_level_index
                ].floor_thickness_meters,
                0.65,
            )
            workspace._run_scheduled_viewer_preview_refresh()

        self.assertAlmostEqual(
            workspace.current_level.floor_thickness_meters,
            0.65,
        )
        self.assertAlmostEqual(
            workspace.levels[1].floor_thickness_meters,
            original_other_level_thickness,
        )
        self.assertGreater(workspace._viewer_preview_revision, revision_before)
        build_preview.assert_called_once_with(None)
        self.assertIsNotNone(workspace._viewer_preview_model)
        assert workspace._viewer_preview_model is not None
        floor_mesh = workspace._viewer_preview_model.scene.geometry[
            "l2_ground_floor"
        ]
        np.testing.assert_allclose(
            floor_mesh.bounds[:, 1],
            np.asarray((0.0, 0.65)),
            atol=1e-9,
        )

        workspace.shutdown()
        workspace.close()

    def test_level_scale_control_updates_only_the_selected_level(self) -> None:
        from housemaker.main import (
            LEVEL_SCALE_SLIDER_FACTOR,
            BlueprintWorkspace,
        )

        workspace = BlueprintWorkspace()
        _qt_widgets.append(workspace)
        self.assertAlmostEqual(
            workspace.level_scale_slider.value()
            / LEVEL_SCALE_SLIDER_FACTOR,
            1.0,
        )
        workspace.level_scale_slider.setValue(
            round(1.75 * LEVEL_SCALE_SLIDER_FACTOR)
        )
        _qt_application.processEvents()

        self.assertAlmostEqual(workspace.current_level.scale, 1.0)
        workspace._commit_pending_level_transform_update()
        self.assertAlmostEqual(workspace.current_level.scale, 1.75)
        self.assertAlmostEqual(workspace.levels[1].scale, DEFAULT_LEVEL_SCALE)

    def test_level_offset_controls_update_only_the_selected_level(self) -> None:
        from housemaker.main import (
            LEVEL_OFFSET_SLIDER_FACTOR,
            BlueprintWorkspace,
        )

        workspace = BlueprintWorkspace()
        _qt_widgets.append(workspace)
        self.assertAlmostEqual(
            workspace.level_x_offset_slider.value()
            / LEVEL_OFFSET_SLIDER_FACTOR,
            DEFAULT_LEVEL_OFFSET_METERS,
        )
        self.assertAlmostEqual(
            workspace.level_y_offset_slider.value()
            / LEVEL_OFFSET_SLIDER_FACTOR,
            DEFAULT_LEVEL_OFFSET_METERS,
        )

        workspace.level_x_offset_slider.setValue(
            round(1.25 * LEVEL_OFFSET_SLIDER_FACTOR)
        )
        workspace.level_y_offset_slider.setValue(
            round(-0.75 * LEVEL_OFFSET_SLIDER_FACTOR)
        )
        _qt_application.processEvents()

        self.assertAlmostEqual(workspace.current_level.offset_x_meters, 0.0)
        self.assertAlmostEqual(workspace.current_level.offset_y_meters, 0.0)
        workspace._commit_pending_level_transform_update()
        self.assertAlmostEqual(workspace.current_level.offset_x_meters, 1.25)
        self.assertAlmostEqual(workspace.current_level.offset_y_meters, -0.75)
        self.assertAlmostEqual(
            workspace.levels[1].offset_x_meters,
            DEFAULT_LEVEL_OFFSET_METERS,
        )
        self.assertAlmostEqual(
            workspace.levels[1].offset_y_meters,
            DEFAULT_LEVEL_OFFSET_METERS,
        )

        workspace.levels_list.setCurrentRow(
            workspace._level_list_row_for_position(1)
        )
        _qt_application.processEvents()
        workspace.level_x_offset_slider.setValue(
            round(-2.0 * LEVEL_OFFSET_SLIDER_FACTOR)
        )
        workspace.level_y_offset_slider.setValue(
            round(3.5 * LEVEL_OFFSET_SLIDER_FACTOR)
        )
        _qt_application.processEvents()

        workspace._commit_pending_level_transform_update()
        self.assertAlmostEqual(workspace.levels[1].offset_x_meters, -2.0)
        self.assertAlmostEqual(workspace.levels[1].offset_y_meters, 3.5)
        self.assertAlmostEqual(workspace.levels[2].offset_x_meters, 1.25)
        self.assertAlmostEqual(workspace.levels[2].offset_y_meters, -0.75)

    def test_levels_list_orders_top_floors_before_underground_levels(
        self,
    ) -> None:
        from housemaker.main import BlueprintWorkspace

        workspace = BlueprintWorkspace()
        _qt_widgets.append(workspace)

        ordered_level_positions = [
            workspace.levels_list.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(workspace.levels_list.count())
        ]
        ordered_level_indices = [
            workspace.levels[position].index
            for position in ordered_level_positions
        ]

        self.assertEqual(
            ordered_level_indices,
            sorted(ordered_level_indices, reverse=True),
        )
        self.assertEqual(
            workspace.levels_list.currentRow(),
            workspace._level_list_row_for_position(
                workspace.current_level_index
            ),
        )

        underground_position = next(
            position
            for position, level in enumerate(workspace.levels)
            if level.index == 1
        )
        workspace.levels_list.setCurrentRow(
            workspace._level_list_row_for_position(underground_position)
        )
        _qt_application.processEvents()

        self.assertIs(
            workspace.current_level,
            workspace.levels[underground_position],
        )

    def test_ground_level_row_uses_semantic_highlight(self) -> None:
        from housemaker.main import (
            BlueprintWorkspace,
            _build_ground_level_background_color,
        )

        workspace = BlueprintWorkspace()
        _qt_widgets.append(workspace)
        ground_position = next(
            position
            for position, level in enumerate(workspace.levels)
            if level.index == 2
        )
        workspace.levels[ground_position].name = "Renamed ground floor"
        workspace._refresh_levels_list()

        ground_item = workspace.levels_list.item(
            workspace._level_list_row_for_position(ground_position)
        )
        self.assertEqual(
            ground_item.background().color(),
            _build_ground_level_background_color(
                workspace.levels_list.palette()
            ),
        )
        self.assertNotEqual(
            ground_item.background().color().rgb(),
            workspace.levels_list.palette().base().color().rgb(),
        )
        for row in range(workspace.levels_list.count()):
            item = workspace.levels_list.item(row)
            if item is ground_item:
                continue
            self.assertEqual(
                item.background().style(),
                Qt.BrushStyle.NoBrush,
            )

    def test_transform_groups_are_below_floor_thickness(
        self,
    ) -> None:
        from housemaker.main import BlueprintWorkspace

        workspace = BlueprintWorkspace()
        _qt_widgets.append(workspace)
        workspace.resize(1600, 900)
        workspace.show()
        _qt_application.processEvents()

        controls = (
            workspace.height_level_spinbox,
            workspace.floor_thickness_spinbox,
            workspace.level_transform_group,
            workspace.canvas_transform_group,
        )
        top_positions = [
            control.mapTo(workspace, QPoint()).y()
            for control in controls
        ]
        self.assertEqual(top_positions, sorted(top_positions))

        workspace.close()

    def test_include_controls_are_inside_the_levels_section(self) -> None:
        from housemaker.main import BlueprintWorkspace

        workspace = BlueprintWorkspace()
        _qt_widgets.append(workspace)
        workspace.resize(1600, 900)
        workspace.show()
        _qt_application.processEvents()

        controls = (
            workspace.load_image_button,
            workspace.blueprint_name_label,
            workspace.levels_list,
            workspace.include_yes_radio,
        )
        top_positions = [
            control.mapTo(workspace, QPoint()).y()
            for control in controls
        ]

        self.assertEqual(top_positions, sorted(top_positions))
        self.assertIs(
            workspace.include_yes_radio.parentWidget(),
            workspace.include_no_radio.parentWidget(),
        )
        self.assertLess(
            workspace.levels_group.mapTo(workspace, QPoint()).y(),
            workspace.save_button.mapTo(workspace, QPoint()).y(),
        )

        workspace.close()

    def test_wheel_over_general_inputs_scrolls_without_changing_values(
        self,
    ) -> None:
        from housemaker.main import BlueprintWorkspace

        workspace = BlueprintWorkspace()
        _qt_widgets.append(workspace)
        workspace.resize(1600, 900)
        workspace.show()
        _qt_application.processEvents()

        generals_tab = workspace.side_tabs.widget(0)
        self.assertIsInstance(generals_tab, QScrollArea)
        scroll_bar = generals_tab.verticalScrollBar()
        self.assertGreater(scroll_bar.maximum(), 0)

        controls = (
            ("height", workspace.height_level_spinbox),
            ("floor thickness", workspace.floor_thickness_spinbox),
            ("level scale", workspace.level_scale_slider),
            ("level X offset", workspace.level_x_offset_slider),
            ("level Y offset", workspace.level_y_offset_slider),
            ("Canvas level scale", workspace.canvas_level_scale_slider),
            ("Canvas X offset", workspace.canvas_x_offset_slider),
            ("Canvas Y offset", workspace.canvas_y_offset_slider),
        )
        for control_name, control in controls:
            with self.subTest(control=control_name):
                scroll_bar.setValue(0)
                _qt_application.processEvents()
                original_value = control.value()

                _send_wheel_event(control, angle_delta_y=-120)
                _qt_application.processEvents()

                self.assertEqual(control.value(), original_value)
                self.assertGreater(scroll_bar.value(), 0)

        scroll_bar.setValue(0)
        height_value = workspace.height_level_spinbox.value()
        _send_wheel_event(workspace.height_level_spinbox.lineEdit(), -120)
        _qt_application.processEvents()
        self.assertAlmostEqual(
            workspace.height_level_spinbox.value(),
            height_value,
        )
        self.assertGreater(scroll_bar.value(), 0)

        workspace.close()


if __name__ == "__main__":
    unittest.main()
