# ### Environment setup ###
from __future__ import annotations

from io import BytesIO
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.glb import convert_to_glb
from housemaker.models import (
    DEFAULT_DOORWAY_DEPTH_METERS,
    PIXEL_TO_METER,
    DoorwayData,
    DoorwayPreset,
    LevelData,
    VertexData,
    create_default_doorway_presets,
    create_default_levels,
)
from housemaker.project_io import load_project, save_project

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _add_wall(
    vertex_data: VertexData,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
) -> None:
    start_vertex = vertex_data.add_vertex(*start_point)
    end_vertex = vertex_data.add_vertex(*end_point)
    vertex_data.add_edge(start_vertex.id, end_vertex.id)


def _add_square_walls(vertex_data: VertexData) -> tuple[int, ...]:
    vertices = [
        vertex_data.add_vertex(*point)
        for point in ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
    ]
    for start_vertex, end_vertex in zip(vertices, vertices[1:] + vertices[:1]):
        vertex_data.add_edge(start_vertex.id, end_vertex.id)
    return tuple(vertex.id for vertex in vertices)


def _build_canvas(
    vertex_data: VertexData,
    doorways: list[DoorwayData] | None = None,
) -> BlueprintCanvas:
    canvas = BlueprintCanvas()
    canvas.resize(640, 520)
    canvas.set_level_data(
        vertex_data=vertex_data,
        rooms=[],
        image_path=None,
        doorways=doorways,
    )
    canvas.blueprint_image = QImage(100, 100, QImage.Format.Format_RGB32)
    canvas.blueprint_image.fill(Qt.GlobalColor.white)
    canvas.show()
    _qt_application.processEvents()
    return canvas


def _image_position(canvas: BlueprintCanvas, x: float, y: float) -> QPoint:
    return canvas._image_to_widget(x, y).toPoint()


def _send_wheel_event(
    canvas: BlueprintCanvas,
    image_position: QPoint,
    angle_delta_y: int,
) -> None:
    local_position = QPointF(image_position)
    global_position = QPointF(canvas.mapToGlobal(image_position))
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
    QApplication.sendEvent(canvas, event)


def _send_drag_move(canvas: BlueprintCanvas, image_position: QPoint) -> None:
    local_position = QPointF(image_position)
    global_position = QPointF(canvas.mapToGlobal(image_position))
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        local_position,
        global_position,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(canvas, event)


def _build_intersected_wall_level() -> LevelData:
    vertex_data = VertexData()
    _add_wall(vertex_data, (10.0, 50.0), (90.0, 50.0))
    _add_wall(vertex_data, (50.0, 10.0), (50.0, 90.0))
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
                rotation_degrees=0.0,
            )
        ],
    )


def _build_parallel_wall_doorway_level() -> LevelData:
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
            )
        ],
    )


def _get_depth_border_image_position(
    doorway: DoorwayData,
    depth_border_sign: float = 1.0,
) -> tuple[float, float]:
    rotation_radians = math.radians(doorway.rotation_degrees)
    half_depth_pixels = doorway.depth_meters / PIXEL_TO_METER / 2.0
    return (
        doorway.center_x
        + depth_border_sign * math.cos(rotation_radians) * half_depth_pixels,
        doorway.center_y
        + depth_border_sign * math.sin(rotation_radians) * half_depth_pixels,
    )


def _mesh_covers_point_on_plane(
    mesh: trimesh.Trimesh,
    point: tuple[float, float, float],
    fixed_axis: int,
) -> bool:
    point_array = np.asarray(point, dtype=float)
    plane_axes = [axis for axis in range(3) if axis != fixed_axis]
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
            return True

    return False


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
    loaded_model = trimesh.load(BytesIO(glb_bytes), file_type="glb")
    if isinstance(loaded_model, trimesh.Scene):
        return loaded_model.dump(concatenate=True)

    if isinstance(loaded_model, trimesh.Trimesh):
        return loaded_model

    raise AssertionError("GLB did not load as a mesh or scene.")


# ### Tests ###
class DoorwayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[object] = []

    def tearDown(self) -> None:
        for widget in reversed(self.widgets):
            close = getattr(widget, "close", None)
            if callable(close):
                close()
        _qt_application.processEvents()

    def _track_widget(self, widget: object) -> object:
        self.widgets.append(widget)
        return widget

    def test_project_round_trip_persists_doorway_presets_and_level_doorways(self) -> None:
        levels = create_default_levels()
        doorway_presets = [
            DoorwayPreset(width_meters=0.8, height_meters=2.0),
            DoorwayPreset(width_meters=1.4, height_meters=2.4),
        ]
        levels[2].doorways = [
            DoorwayData(
                center_x=64.0,
                center_y=32.0,
                width_meters=0.8,
                height_meters=2.0,
                depth_meters=0.36,
                rotation_degrees=90.0,
            )
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "doorway-project.json"
            save_project(
                project_path,
                current_level_index=2,
                levels=levels,
                doorway_presets=doorway_presets,
            )

            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["doorway_presets"],
                [
                    {"width_meters": 0.8, "height_meters": 2.0},
                    {"width_meters": 1.4, "height_meters": 2.4},
                ],
            )
            self.assertEqual(payload["levels"][2]["doorways"][0]["depth_meters"], 0.36)

            loaded_project = load_project(project_path)
            self.assertEqual(loaded_project.doorway_presets, doorway_presets)
            self.assertEqual(loaded_project.levels[2].doorways, levels[2].doorways)

            payload.pop("doorway_presets")
            payload["levels"][2].pop("doorways")
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            legacy_project = load_project(project_path)
            self.assertEqual(
                legacy_project.doorway_presets,
                create_default_doorway_presets(),
            )
            self.assertEqual(legacy_project.levels[2].doorways, [])

    def test_canvas_auto_aligns_doorway_to_wall_and_wheel_zooms(self) -> None:
        vertex_data = VertexData()
        _add_wall(vertex_data, (10.0, 50.0), (90.0, 50.0))
        canvas = self._track_widget(_build_canvas(vertex_data))
        self.assertIsInstance(canvas, BlueprintCanvas)
        original_vertices = tuple(vertex_data.vertices)
        original_edges = tuple(vertex_data.edges)
        emitted_changes: list[None] = []
        canvas.doorways_changed.connect(lambda: emitted_changes.append(None))

        canvas.start_doorway_placement(
            DoorwayPreset(width_meters=0.9, height_meters=2.1)
        )
        placement_position = _image_position(canvas, 56.0, 67.0)
        placement_image_point = canvas._widget_to_image(QPointF(placement_position))
        self.assertIsNotNone(placement_image_point)
        QTest.mouseMove(canvas, placement_position)
        _qt_application.processEvents()
        self.assertIsNotNone(canvas.pending_doorway)
        pending_rotation_degrees = canvas.pending_doorway.rotation_degrees
        self.assertAlmostEqual(
            abs(math.cos(math.radians(pending_rotation_degrees))),
            0.0,
        )
        self.assertAlmostEqual(
            abs(math.sin(math.radians(pending_rotation_degrees))),
            1.0,
        )
        self.assertAlmostEqual(
            canvas.pending_doorway.center_x,
            placement_image_point.x(),
        )
        self.assertAlmostEqual(canvas.pending_doorway.center_y, 50.0)

        original_zoom_scale = canvas.zoom_scale
        _send_wheel_event(canvas, placement_position, 120)
        _qt_application.processEvents()
        self.assertGreater(canvas.zoom_scale, original_zoom_scale)
        self.assertIsNotNone(canvas.pending_doorway)
        self.assertAlmostEqual(
            canvas.pending_doorway.rotation_degrees,
            pending_rotation_degrees,
        )

        placement_position = _image_position(canvas, 56.0, 67.0)
        QTest.mouseMove(canvas, placement_position)
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=placement_position)
        _qt_application.processEvents()

        self.assertEqual(len(canvas.doorways), 1)
        doorway = canvas.doorways[0]
        self.assertAlmostEqual(doorway.center_y, 50.0)
        self.assertAlmostEqual(doorway.width_meters, 0.9)
        self.assertAlmostEqual(doorway.height_meters, 2.1)
        self.assertAlmostEqual(doorway.depth_meters, DEFAULT_DOORWAY_DEPTH_METERS)
        self.assertAlmostEqual(
            abs(math.cos(math.radians(doorway.rotation_degrees))),
            0.0,
        )
        self.assertAlmostEqual(
            abs(math.sin(math.radians(doorway.rotation_degrees))),
            1.0,
        )
        self.assertEqual(emitted_changes, [None])
        self.assertEqual(tuple(vertex_data.vertices), original_vertices)
        self.assertEqual(tuple(vertex_data.edges), original_edges)

    def test_canvas_resizes_only_depth_borders_with_directional_cursors(self) -> None:
        vertex_data = VertexData()
        doorway = DoorwayData(
            center_x=50.0,
            center_y=50.0,
            width_meters=0.9,
            height_meters=2.1,
            depth_meters=0.2,
            rotation_degrees=0.0,
        )
        canvas = self._track_widget(_build_canvas(vertex_data, [doorway]))
        self.assertIsInstance(canvas, BlueprintCanvas)
        center_position = _image_position(canvas, doorway.center_x, doorway.center_y)
        QTest.mouseMove(canvas, center_position)
        _qt_application.processEvents()
        self.assertEqual(canvas.cursor().shape(), Qt.CursorShape.ArrowCursor)

        cursor_cases = (
            (0.0, Qt.CursorShape.SizeHorCursor),
            (90.0, Qt.CursorShape.SizeVerCursor),
            (45.0, Qt.CursorShape.SizeFDiagCursor),
            (135.0, Qt.CursorShape.SizeBDiagCursor),
        )
        for rotation_degrees, expected_cursor in cursor_cases:
            with self.subTest(rotation_degrees=rotation_degrees):
                canvas.doorways[0].rotation_degrees = rotation_degrees
                edge_position = _image_position(
                    canvas,
                    *_get_depth_border_image_position(canvas.doorways[0]),
                )
                QTest.mouseMove(canvas, _image_position(canvas, 15.0, 15.0))
                QTest.mouseMove(canvas, edge_position)
                _qt_application.processEvents()
                self.assertEqual(canvas.cursor().shape(), expected_cursor)

        canvas.doorways[0].rotation_degrees = 0.0
        emitted_changes: list[None] = []
        canvas.doorways_changed.connect(lambda: emitted_changes.append(None))
        initial_depth_meters = canvas.doorways[0].depth_meters

        QTest.mousePress(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=center_position,
        )
        _send_drag_move(canvas, _image_position(canvas, 70.0, 50.0))
        QTest.mouseRelease(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(canvas, 70.0, 50.0),
        )
        self.assertAlmostEqual(canvas.doorways[0].depth_meters, initial_depth_meters)
        self.assertEqual(emitted_changes, [])

        depth_border_position = _image_position(
            canvas,
            *_get_depth_border_image_position(canvas.doorways[0]),
        )
        depth_border_image_point = canvas._widget_to_image(
            QPointF(depth_border_position)
        )
        self.assertIsNotNone(depth_border_image_point)
        QTest.mousePress(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=depth_border_position,
        )
        resized_position = _image_position(canvas, 70.0, 50.0)
        resized_image_point = canvas._widget_to_image(QPointF(resized_position))
        self.assertIsNotNone(resized_image_point)
        _send_drag_move(canvas, resized_position)
        QTest.mouseRelease(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=resized_position,
        )
        _qt_application.processEvents()

        expected_depth_meters = (
            initial_depth_meters
            + 2.0
            * (resized_image_point.x() - depth_border_image_point.x())
            * PIXEL_TO_METER
        )
        self.assertAlmostEqual(
            canvas.doorways[0].depth_meters,
            expected_depth_meters,
        )
        self.assertEqual(emitted_changes, [None])

        canvas.undo_last_step()
        self.assertAlmostEqual(
            canvas.doorways[0].depth_meters,
            DEFAULT_DOORWAY_DEPTH_METERS,
        )

    def test_canvas_selects_doorway_center_and_delete_removes_it(self) -> None:
        doorway = DoorwayData(
            center_x=50.0,
            center_y=50.0,
            width_meters=0.9,
            height_meters=2.1,
            depth_meters=0.2,
            rotation_degrees=0.0,
        )
        canvas = self._track_widget(_build_canvas(VertexData(), [doorway]))
        self.assertIsInstance(canvas, BlueprintCanvas)
        emitted_changes: list[None] = []
        canvas.doorways_changed.connect(lambda: emitted_changes.append(None))

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(canvas, doorway.center_x, doorway.center_y),
        )
        self.assertEqual(canvas.selected_doorway_index, 0)

        QTest.keyClick(canvas, Qt.Key.Key_Delete)
        _qt_application.processEvents()
        self.assertEqual(canvas.doorways, [])
        self.assertIsNone(canvas.selected_doorway_index)
        self.assertEqual(emitted_changes, [None])

    def test_canvas_deselects_a_doorway_before_selecting_a_vertex(self) -> None:
        doorway = DoorwayData(
            center_x=50.0,
            center_y=50.0,
            width_meters=0.9,
            height_meters=2.1,
            depth_meters=0.2,
            rotation_degrees=0.0,
        )
        canvas = self._track_widget(_build_canvas(VertexData(), [doorway]))
        self.assertIsInstance(canvas, BlueprintCanvas)

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(canvas, doorway.center_x, doorway.center_y),
        )
        self.assertEqual(canvas.selected_doorway_index, 0)

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(canvas, 10.0, 10.0),
        )
        self.assertIsNone(canvas.selected_doorway_index)

        QTest.keyClick(canvas, Qt.Key.Key_Delete)
        _qt_application.processEvents()
        self.assertEqual(len(canvas.doorways), 1)

    def test_general_tab_adds_selects_and_starts_placing_doorway_preset(self) -> None:
        from housemaker.main import BlueprintWorkspace

        workspace = self._track_widget(BlueprintWorkspace())
        self.assertIsInstance(workspace, BlueprintWorkspace)
        workspace.resize(1500, 900)
        workspace.show()
        _qt_application.processEvents()

        initial_preset_count = len(workspace.doorway_presets)
        workspace.doorway_width_spinbox.setValue(1.35)
        workspace.doorway_height_spinbox.setValue(2.45)
        QTest.mouseClick(
            workspace.add_doorway_preset_button,
            Qt.MouseButton.LeftButton,
        )
        _qt_application.processEvents()

        self.assertEqual(len(workspace.doorway_presets), initial_preset_count + 1)
        self.assertEqual(
            workspace.doorway_preset_list.currentRow(),
            initial_preset_count,
        )
        self.assertEqual(
            workspace.doorway_presets[-1],
            DoorwayPreset(width_meters=1.35, height_meters=2.45),
        )

        workspace.doorway_preset_list.setCurrentRow(initial_preset_count)
        QTest.mouseClick(
            workspace.place_doorway_button,
            Qt.MouseButton.LeftButton,
        )
        _qt_application.processEvents()

        self.assertIs(
            workspace.workspace_tabs.currentWidget(),
            workspace.canvas_viewer_workspace,
        )
        self.assertEqual(
            workspace.canvas.pending_doorway_preset,
            workspace.doorway_presets[-1],
        )

    def test_glb_doorway_cuts_holes_in_each_intersected_wall(self) -> None:
        level = _build_intersected_wall_level()
        no_doorway_level = _build_intersected_wall_level()
        no_doorway_level.doorways = []

        model = convert_to_glb([level])
        model_without_doorway = convert_to_glb([no_doorway_level])

        self.assertTrue(
            set(model_without_doorway.scene.geometry).issubset(
                model.scene.geometry
            )
        )
        self.assertFalse(
            any(
                geometry_name.endswith("_doorway_reveals")
                for geometry_name in model.scene.geometry
            )
        )
        self.assertFalse(
            _mesh_covers_point_on_plane(
                model.mesh,
                (1.0, -1.0, 1.0),
                fixed_axis=1,
            )
        )
        self.assertTrue(
            _mesh_covers_point_on_plane(
                model.mesh,
                (0.5, -1.0, 1.0),
                fixed_axis=1,
            )
        )
        self.assertTrue(
            _mesh_covers_point_on_plane(
                model.mesh,
                (1.0, -1.0, 2.5),
                fixed_axis=1,
            )
        )
        self.assertFalse(
            _mesh_covers_point_on_plane(
                model.mesh,
                (1.0, -1.0, 1.0),
                fixed_axis=0,
            )
        )
        self.assertTrue(
            _mesh_covers_point_on_plane(
                model.mesh,
                (1.0, -0.3, 1.0),
                fixed_axis=0,
            )
        )
        self.assertTrue(
            _mesh_covers_point_on_plane(
                model.mesh,
                (1.0, -1.0, 2.5),
                fixed_axis=0,
            )
        )

        exported_mesh = _load_glb_world_mesh(model.glb_bytes)
        self.assertFalse(
            _mesh_covers_point_on_plane(
                exported_mesh,
                (1.0, 1.0, 1.0),
                fixed_axis=2,
            )
        )
        self.assertTrue(
            _mesh_covers_point_on_plane(
                exported_mesh,
                (0.5, 1.0, 1.0),
                fixed_axis=2,
            )
        )
        self.assertFalse(
            _mesh_covers_point_on_plane(
                exported_mesh,
                (1.0, 1.0, 1.0),
                fixed_axis=0,
            )
        )
        self.assertTrue(
            _mesh_covers_point_on_plane(
                exported_mesh,
                (1.0, 1.0, 0.3),
                fixed_axis=0,
            )
        )

    def test_glb_seals_parallel_wall_doorway_with_jambs_and_soffit(self) -> None:
        model = convert_to_glb([_build_parallel_wall_doorway_level()])

        # The openings in the front and back wall planes remain empty.
        for wall_plane_y in (-0.9, -1.1):
            with self.subTest(wall_plane_y=wall_plane_y):
                self.assertFalse(
                    _mesh_covers_point_on_plane(
                        model.mesh,
                        (1.0, wall_plane_y, 1.0),
                        fixed_axis=1,
                    )
                )

        # The two jambs bridge both wall contacts through doorway depth.
        for jamb_x in (0.55, 1.45):
            with self.subTest(jamb_x=jamb_x):
                self.assertTrue(
                    _mesh_covers_point_on_plane(
                        model.mesh,
                        (jamb_x, -1.0, 1.0),
                        fixed_axis=0,
                    )
                )

        # The top reveal seals the passage without filling its vertical opening.
        self.assertTrue(
            _mesh_covers_point_on_plane(
                model.mesh,
                (1.0, -1.0, 2.1),
                fixed_axis=2,
            )
        )
        self.assertFalse(
            _mesh_covers_point_on_plane(
                model.mesh,
                (1.0, -1.0, 1.0),
                fixed_axis=1,
            )
        )

        exported_mesh = _load_glb_world_mesh(model.glb_bytes)
        for wall_plane_z in (0.9, 1.1):
            with self.subTest(exported_wall_plane_z=wall_plane_z):
                self.assertFalse(
                    _mesh_covers_point_on_plane(
                        exported_mesh,
                        (1.0, 1.0, wall_plane_z),
                        fixed_axis=2,
                    )
                )

        for jamb_x in (0.55, 1.45):
            with self.subTest(exported_jamb_x=jamb_x):
                self.assertTrue(
                    _mesh_covers_point_on_plane(
                        exported_mesh,
                        (jamb_x, 1.0, 1.0),
                        fixed_axis=0,
                    )
                )

        self.assertTrue(
            _mesh_covers_point_on_plane(
                exported_mesh,
                (1.0, 2.1, 1.0),
                fixed_axis=1,
            )
        )
        self.assertFalse(
            _mesh_covers_point_on_plane(
                exported_mesh,
                (1.0, 1.0, 1.0),
                fixed_axis=2,
            )
        )

    def test_viewer_uses_open_wall_geometry_without_doorway_overlay(self) -> None:
        from housemaker.viewer import GlbViewerWidget
        import pyqtgraph.opengl as gl

        viewer = self._track_widget(GlbViewerWidget())
        self.assertIsInstance(viewer, GlbViewerWidget)
        model = convert_to_glb([_build_intersected_wall_level()])

        viewer.set_model(model)
        _qt_application.processEvents()

        self.assertFalse(getattr(model, "preview_doorways", []))
        self.assertFalse(getattr(viewer, "doorway_mesh_items", []))
        self.assertFalse(getattr(viewer, "doorway_outline_items", []))
        self.assertFalse(getattr(viewer, "doorway_label_items", []))
        self.assertFalse(
            any(isinstance(item, gl.GLTextItem) for item in viewer.view.items)
        )

        viewer.clear_model()
        _qt_application.processEvents()
        self.assertIsNone(viewer.model)


if __name__ == "__main__":
    unittest.main()
