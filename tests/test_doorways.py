# ### Environment setup ###
from __future__ import annotations

from io import BytesIO
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, sentinel

import numpy as np
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.glb import convert_to_glb
from housemaker.models import (
    DEFAULT_DOORWAY_DEPTH_METERS,
    DEFAULT_DOORWAY_SHAPE,
    DOORWAY_SHAPE_ARCH,
    DOORWAY_SHAPE_RECTANGULAR,
    PIXEL_TO_METER,
    DoorwayData,
    DoorwayPreset,
    LevelData,
    VertexData,
    create_default_doorway_presets,
    create_default_levels,
    normalize_doorway_shape,
)
from housemaker.project_io import load_project, save_project
from housemaker.settings_widget import (
    DOORWAY_MESH_UPDATE_DELAY_SECONDS_SETTING_KEY,
)

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


def _get_width_border_image_position(
    doorway: DoorwayData,
    width_border_sign: float = 1.0,
) -> tuple[float, float]:
    rotation_radians = math.radians(doorway.rotation_degrees)
    width_direction_x = -math.sin(rotation_radians)
    width_direction_y = math.cos(rotation_radians)
    half_width_pixels = doorway.width_meters / PIXEL_TO_METER / 2.0
    return (
        doorway.center_x
        + width_border_sign * width_direction_x * half_width_pixels,
        doorway.center_y
        + width_border_sign * width_direction_y * half_width_pixels,
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

    def test_doorway_shape_is_normalized_and_strictly_validated(self) -> None:
        doorway = DoorwayData(
            center_x=10.0,
            center_y=20.0,
            width_meters=0.9,
            height_meters=2.1,
            shape=" ARCH ",
        )

        self.assertEqual(doorway.shape, DOORWAY_SHAPE_ARCH)
        self.assertEqual(
            normalize_doorway_shape("RECTANGULAR"),
            DOORWAY_SHAPE_RECTANGULAR,
        )
        for invalid_shape in (None, True, "", "rounded", 1):
            with self.subTest(invalid_shape=invalid_shape):
                with self.assertRaisesRegex(ValueError, "Doorway shape"):
                    DoorwayData(
                        center_x=10.0,
                        center_y=20.0,
                        width_meters=0.9,
                        height_meters=2.1,
                        shape=invalid_shape,  # type: ignore[arg-type]
                    )

        preset = DoorwayPreset(
            width_meters=1.15,
            height_meters=2.35,
            shape=" ARCH ",
            arch_amount=0.425,
        )
        self.assertEqual(preset.shape, DOORWAY_SHAPE_ARCH)
        self.assertEqual(preset.arch_amount, 0.425)

        with self.assertRaisesRegex(ValueError, "Doorway arch amount"):
            DoorwayPreset(
                width_meters=1.15,
                height_meters=2.35,
                shape=DOORWAY_SHAPE_ARCH,
                arch_amount=1.1,
            )

    def test_project_round_trip_persists_doorway_presets_and_level_doorways(self) -> None:
        levels = create_default_levels()
        doorway_presets = [
            DoorwayPreset(width_meters=0.8, height_meters=2.0),
            DoorwayPreset(
                width_meters=1.4,
                height_meters=2.4,
                shape=DOORWAY_SHAPE_ARCH,
                arch_amount=0.375,
            ),
        ]
        levels[2].doorways = [
            DoorwayData(
                center_x=64.0,
                center_y=32.0,
                width_meters=0.8,
                height_meters=2.0,
                depth_meters=0.36,
                rotation_degrees=90.0,
                shape=DOORWAY_SHAPE_ARCH,
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
                    {
                        "width_meters": 0.8,
                        "height_meters": 2.0,
                        "shape": DEFAULT_DOORWAY_SHAPE,
                        "arch_amount": doorway_presets[0].arch_amount,
                    },
                    {
                        "width_meters": 1.4,
                        "height_meters": 2.4,
                        "shape": DOORWAY_SHAPE_ARCH,
                        "arch_amount": 0.375,
                    },
                ],
            )
            self.assertEqual(payload["levels"][2]["doorways"][0]["depth_meters"], 0.36)
            self.assertEqual(
                payload["levels"][2]["doorways"][0]["shape"],
                DOORWAY_SHAPE_ARCH,
            )

            loaded_project = load_project(project_path)
            self.assertEqual(loaded_project.doorway_presets, doorway_presets)
            self.assertEqual(loaded_project.levels[2].doorways, levels[2].doorways)

            for raw_preset in payload["doorway_presets"]:
                raw_preset.pop("shape")
                raw_preset.pop("arch_amount")
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            legacy_preset_project = load_project(project_path)
            self.assertEqual(
                legacy_preset_project.doorway_presets,
                [
                    DoorwayPreset(width_meters=0.8, height_meters=2.0),
                    DoorwayPreset(width_meters=1.4, height_meters=2.4),
                ],
            )

            payload.pop("doorway_presets")
            payload["levels"][2].pop("doorways")
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            legacy_project = load_project(project_path)
            self.assertEqual(
                legacy_project.doorway_presets,
                create_default_doorway_presets(),
            )
            self.assertEqual(legacy_project.levels[2].doorways, [])

    def test_legacy_and_malformed_doorway_shapes_load_as_rectangular(self) -> None:
        levels = create_default_levels()
        levels[2].doorways = [
            DoorwayData(
                center_x=64.0,
                center_y=32.0,
                width_meters=0.8,
                height_meters=2.0,
                shape=DOORWAY_SHAPE_ARCH,
            )
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "doorway-shapes.json"
            save_project(
                project_path,
                current_level_index=2,
                levels=levels,
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            raw_doorway = payload["levels"][2]["doorways"][0]

            raw_doorway.pop("shape")
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                load_project(project_path).levels[2].doorways[0].shape,
                DEFAULT_DOORWAY_SHAPE,
            )

            for malformed_shape in (None, True, "", "rounded", 1):
                with self.subTest(malformed_shape=malformed_shape):
                    raw_doorway["shape"] = malformed_shape
                    project_path.write_text(
                        json.dumps(payload),
                        encoding="utf-8",
                    )
                    self.assertEqual(
                        load_project(project_path).levels[2].doorways[0].shape,
                        DOORWAY_SHAPE_RECTANGULAR,
                    )

    def test_canvas_reports_doorway_selection_changes_once(self) -> None:
        doorway = DoorwayData(
            center_x=50.0,
            center_y=50.0,
            width_meters=0.9,
            height_meters=2.1,
        )
        canvas = self._track_widget(_build_canvas(VertexData(), [doorway]))
        selection_changes: list[int] = []
        canvas.selected_doorway_changed.connect(selection_changes.append)
        doorway_position = _image_position(canvas, doorway.center_x, doorway.center_y)

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=doorway_position,
        )
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=doorway_position,
        )
        self.assertEqual(selection_changes, [0])

        empty_position = _image_position(canvas, 10.0, 10.0)
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=empty_position,
        )
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=empty_position,
        )
        self.assertEqual(selection_changes, [0, -1])

    def test_canvas_changes_selected_doorway_shape_as_one_preview_edit(
        self,
    ) -> None:
        doorway = DoorwayData(
            center_x=50.0,
            center_y=50.0,
            width_meters=0.9,
            height_meters=2.1,
        )
        canvas = self._track_widget(_build_canvas(VertexData(), [doorway]))
        preview_changes: list[None] = []
        committed_changes: list[None] = []
        selection_changes: list[int] = []
        canvas.doorway_dimension_preview_changed.connect(
            lambda: preview_changes.append(None)
        )
        canvas.doorways_changed.connect(lambda: committed_changes.append(None))
        canvas.selected_doorway_changed.connect(selection_changes.append)
        original_plan_corners = canvas._get_doorway_corners(doorway)

        self.assertFalse(canvas.set_selected_doorway_shape(DOORWAY_SHAPE_ARCH))
        self.assertEqual(canvas.undo_stack, [])

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(canvas, doorway.center_x, doorway.center_y),
        )
        self.assertTrue(canvas.set_selected_doorway_shape(" ARCH "))
        self.assertEqual(canvas.doorways[0].shape, DOORWAY_SHAPE_ARCH)
        self.assertEqual(len(canvas.undo_stack), 1)
        self.assertEqual(preview_changes, [None])
        self.assertEqual(committed_changes, [])
        self.assertEqual(selection_changes, [0])
        self.assertEqual(
            canvas._get_doorway_corners(canvas.doorways[0]),
            original_plan_corners,
        )
        self.assertTrue(
            canvas._get_doorway_label_text(canvas.doorways[0]).startswith(
                "Arch 100%\n"
            )
        )

        self.assertFalse(canvas.set_selected_doorway_shape(DOORWAY_SHAPE_ARCH))
        self.assertEqual(len(canvas.undo_stack), 1)
        self.assertEqual(preview_changes, [None])

        canvas.undo_last_step()

        self.assertEqual(canvas.doorways[0].shape, DOORWAY_SHAPE_RECTANGULAR)
        self.assertEqual(committed_changes, [None])
        self.assertEqual(selection_changes, [0, -1])
        self.assertTrue(
            canvas._get_doorway_label_text(canvas.doorways[0]).startswith(
                "Doorway\n"
            )
        )

    def test_arch_shape_survives_wheel_height_and_width_drag_edits(self) -> None:
        doorway = DoorwayData(
            center_x=50.0,
            center_y=50.0,
            width_meters=0.9,
            height_meters=2.1,
            depth_meters=0.2,
            rotation_degrees=0.0,
            shape=DOORWAY_SHAPE_ARCH,
            arch_amount=0.4,
        )
        canvas = self._track_widget(_build_canvas(VertexData(), [doorway]))
        doorway_position = _image_position(canvas, doorway.center_x, doorway.center_y)
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=doorway_position,
        )

        _send_wheel_event(canvas, doorway_position, 120)
        self.assertEqual(canvas.doorways[0].shape, DOORWAY_SHAPE_ARCH)
        self.assertEqual(canvas.doorways[0].arch_amount, 0.4)

        width_border_position = _image_position(
            canvas,
            *_get_width_border_image_position(canvas.doorways[0]),
        )
        resized_position = _image_position(canvas, doorway.center_x, 85.0)
        QTest.mousePress(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=width_border_position,
        )
        _send_drag_move(canvas, resized_position)
        QTest.mouseRelease(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=resized_position,
        )

        self.assertEqual(canvas.doorways[0].shape, DOORWAY_SHAPE_ARCH)
        self.assertEqual(canvas.doorways[0].arch_amount, 0.4)
        self.assertGreater(canvas.doorways[0].width_meters, doorway.width_meters)

    def test_canvas_auto_aligns_doorway_to_wall_and_wheel_zooms(self) -> None:
        vertex_data = VertexData()
        _add_wall(vertex_data, (10.0, 50.0), (90.0, 50.0))
        canvas = self._track_widget(_build_canvas(vertex_data))
        self.assertIsInstance(canvas, BlueprintCanvas)
        original_vertices = tuple(vertex_data.vertices)
        original_edges = tuple(vertex_data.edges)
        committed_changes: list[None] = []
        preview_changes: list[None] = []
        canvas.doorways_changed.connect(lambda: committed_changes.append(None))
        canvas.doorway_dimension_preview_changed.connect(
            lambda: preview_changes.append(None)
        )

        canvas.start_doorway_placement(
            DoorwayPreset(
                width_meters=0.9,
                height_meters=2.1,
                shape=DOORWAY_SHAPE_ARCH,
                arch_amount=0.44,
            )
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
        self.assertEqual(doorway.shape, DOORWAY_SHAPE_ARCH)
        self.assertAlmostEqual(doorway.arch_amount, 0.44)
        self.assertAlmostEqual(
            abs(math.cos(math.radians(doorway.rotation_degrees))),
            0.0,
        )
        self.assertAlmostEqual(
            abs(math.sin(math.radians(doorway.rotation_degrees))),
            1.0,
        )
        self.assertEqual(committed_changes, [None])
        self.assertEqual(preview_changes, [])
        self.assertEqual(tuple(vertex_data.vertices), original_vertices)
        self.assertEqual(tuple(vertex_data.edges), original_edges)

    def test_canvas_resizes_depth_borders_with_directional_cursors(self) -> None:
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
        committed_changes: list[None] = []
        preview_changes: list[None] = []
        drag_events: list[str] = []
        canvas.doorways_changed.connect(lambda: committed_changes.append(None))
        canvas.doorway_dimension_preview_changed.connect(
            lambda: preview_changes.append(None)
        )
        canvas.doorway_dimension_drag_started.connect(
            lambda: drag_events.append("started")
        )
        canvas.doorway_dimension_drag_finished.connect(
            lambda: drag_events.append("finished")
        )
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
        self.assertEqual(committed_changes, [])
        self.assertEqual(preview_changes, [])
        self.assertEqual(drag_events, [])

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
        self.assertEqual(drag_events, ["started"])
        resized_position = _image_position(canvas, 70.0, 50.0)
        resized_image_point = canvas._widget_to_image(QPointF(resized_position))
        self.assertIsNotNone(resized_image_point)
        _send_drag_move(canvas, resized_position)
        self.assertEqual(committed_changes, [])
        self.assertEqual(preview_changes, [None])
        self.assertEqual(drag_events, ["started"])
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
        self.assertEqual(committed_changes, [])
        self.assertEqual(preview_changes, [None])
        self.assertEqual(drag_events, ["started", "finished"])

        canvas.undo_last_step()
        self.assertAlmostEqual(
            canvas.doorways[0].depth_meters,
            DEFAULT_DOORWAY_DEPTH_METERS,
        )
        self.assertEqual(committed_changes, [None])
        self.assertEqual(preview_changes, [None])

    def test_canvas_resizes_width_border_live_and_undoes_as_one_step(self) -> None:
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

        cursor_cases = (
            (0.0, Qt.CursorShape.SizeVerCursor),
            (90.0, Qt.CursorShape.SizeHorCursor),
            (45.0, Qt.CursorShape.SizeBDiagCursor),
            (135.0, Qt.CursorShape.SizeFDiagCursor),
        )
        for rotation_degrees, expected_cursor in cursor_cases:
            with self.subTest(rotation_degrees=rotation_degrees):
                canvas.doorways[0].rotation_degrees = rotation_degrees
                border_position = _image_position(
                    canvas,
                    *_get_width_border_image_position(canvas.doorways[0]),
                )
                QTest.mouseMove(canvas, _image_position(canvas, 15.0, 15.0))
                QTest.mouseMove(canvas, border_position)
                _qt_application.processEvents()
                self.assertEqual(canvas.cursor().shape(), expected_cursor)

        canvas.doorways[0].rotation_degrees = 0.0
        initial_doorway = DoorwayData(**vars(canvas.doorways[0]))
        initial_undo_count = len(canvas.undo_stack)
        room_changes: list[None] = []
        committed_changes: list[None] = []
        preview_changes: list[None] = []
        drag_events: list[str] = []
        canvas.rooms_changed.connect(lambda: room_changes.append(None))
        canvas.doorways_changed.connect(lambda: committed_changes.append(None))
        canvas.doorway_dimension_preview_changed.connect(
            lambda: preview_changes.append(None)
        )
        canvas.doorway_dimension_drag_started.connect(
            lambda: drag_events.append("started")
        )
        canvas.doorway_dimension_drag_finished.connect(
            lambda: drag_events.append("finished")
        )
        width_border_position = _image_position(
            canvas,
            *_get_width_border_image_position(canvas.doorways[0]),
        )
        width_border_image_point = canvas._widget_to_image(
            QPointF(width_border_position)
        )
        self.assertIsNotNone(width_border_image_point)

        QTest.mousePress(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=width_border_position,
        )
        self.assertEqual(drag_events, ["started"])
        orthogonal_position = _image_position(
            canvas,
            70.0,
            width_border_image_point.y(),
        )
        _send_drag_move(canvas, orthogonal_position)
        QTest.mouseRelease(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=orthogonal_position,
        )
        self.assertEqual(committed_changes, [])
        self.assertEqual(preview_changes, [])
        self.assertEqual(drag_events, ["started", "finished"])
        self.assertEqual(len(canvas.undo_stack), initial_undo_count)

        QTest.mousePress(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=width_border_position,
        )
        self.assertEqual(
            drag_events,
            ["started", "finished", "started"],
        )
        resized_position = _image_position(canvas, 50.0, 90.0)
        resized_image_point = canvas._widget_to_image(QPointF(resized_position))
        self.assertIsNotNone(resized_image_point)
        _send_drag_move(canvas, resized_position)

        expected_width_meters = (
            initial_doorway.width_meters
            + 2.0
            * (resized_image_point.y() - width_border_image_point.y())
            * PIXEL_TO_METER
        )
        resized_doorway = canvas.doorways[0]
        self.assertAlmostEqual(resized_doorway.width_meters, expected_width_meters)
        self.assertAlmostEqual(resized_doorway.center_x, initial_doorway.center_x)
        self.assertAlmostEqual(resized_doorway.center_y, initial_doorway.center_y)
        self.assertAlmostEqual(
            resized_doorway.height_meters,
            initial_doorway.height_meters,
        )
        self.assertAlmostEqual(
            resized_doorway.depth_meters,
            initial_doorway.depth_meters,
        )
        self.assertEqual(committed_changes, [])
        self.assertEqual(preview_changes, [None])
        self.assertEqual(
            drag_events,
            ["started", "finished", "started"],
        )
        self.assertEqual(len(canvas.undo_stack), initial_undo_count + 1)

        QTest.mouseRelease(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=resized_position,
        )
        _qt_application.processEvents()
        self.assertEqual(committed_changes, [])
        self.assertEqual(preview_changes, [None])
        self.assertEqual(
            drag_events,
            ["started", "finished", "started", "finished"],
        )
        self.assertEqual(len(canvas.undo_stack), initial_undo_count + 1)

        canvas.undo_last_step()
        self.assertAlmostEqual(
            canvas.doorways[0].width_meters,
            initial_doorway.width_meters,
        )
        self.assertEqual(committed_changes, [None])
        self.assertEqual(preview_changes, [None])
        self.assertEqual(room_changes, [])

    def test_doorway_drag_preserves_wheel_updated_height(self) -> None:
        doorway = DoorwayData(
            center_x=50.0,
            center_y=50.0,
            width_meters=0.9,
            height_meters=2.1,
            depth_meters=0.2,
            rotation_degrees=0.0,
        )
        canvas = self._track_widget(_build_canvas(VertexData(), [doorway]))
        width_border_position = _image_position(
            canvas,
            *_get_width_border_image_position(doorway),
        )
        first_resized_position = _image_position(
            canvas,
            doorway.center_x,
            90.0,
        )
        second_resized_position = _image_position(
            canvas,
            doorway.center_x,
            80.0,
        )

        QTest.mousePress(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=width_border_position,
        )
        _send_drag_move(canvas, first_resized_position)
        first_resized_width_meters = canvas.doorways[0].width_meters

        _send_wheel_event(canvas, first_resized_position, 120)
        wheel_updated_height_meters = canvas.doorways[0].height_meters
        self.assertAlmostEqual(
            wheel_updated_height_meters,
            doorway.height_meters + 0.05,
        )

        _send_drag_move(canvas, second_resized_position)
        self.assertNotEqual(
            canvas.doorways[0].width_meters,
            first_resized_width_meters,
        )
        self.assertAlmostEqual(
            canvas.doorways[0].height_meters,
            wheel_updated_height_meters,
        )

        QTest.mouseRelease(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=second_resized_position,
        )

    def test_vertex_undo_still_emits_rooms_changed(self) -> None:
        vertex_data = VertexData()
        vertex = vertex_data.add_vertex(20.0, 30.0)
        canvas = self._track_widget(_build_canvas(vertex_data))
        room_changes: list[None] = []
        canvas.rooms_changed.connect(lambda: room_changes.append(None))
        canvas._push_undo_state()
        canvas.vertex_data.move_vertex(vertex.id, 40.0, 50.0)

        canvas.undo_last_step()

        self.assertEqual(room_changes, [None])
        restored_vertex = canvas.vertex_data.get_vertex(vertex.id)
        self.assertIsNotNone(restored_vertex)
        self.assertAlmostEqual(restored_vertex.x, 20.0)
        self.assertAlmostEqual(restored_vertex.y, 30.0)

    def test_loading_level_finishes_active_doorway_dimension_drag(self) -> None:
        doorway = DoorwayData(
            center_x=50.0,
            center_y=50.0,
            width_meters=0.9,
            height_meters=2.1,
            depth_meters=0.2,
            rotation_degrees=0.0,
        )
        canvas = self._track_widget(_build_canvas(VertexData(), [doorway]))
        drag_events: list[str] = []
        canvas.doorway_dimension_drag_started.connect(
            lambda: drag_events.append("started")
        )
        canvas.doorway_dimension_drag_finished.connect(
            lambda: drag_events.append("finished")
        )
        width_border_position = _image_position(
            canvas,
            *_get_width_border_image_position(doorway),
        )

        QTest.mousePress(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=width_border_position,
        )
        self.assertEqual(drag_events, ["started"])

        canvas.set_level_data(
            vertex_data=VertexData(),
            rooms=[],
            image_path=None,
            doorways=[],
        )

        self.assertEqual(drag_events, ["started", "finished"])
        self.assertIsNone(canvas.pressed_doorway_index)

    def test_selected_doorway_wheel_changes_height_without_zooming(self) -> None:
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
        doorway_position = _image_position(canvas, doorway.center_x, doorway.center_y)
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=doorway_position,
        )
        self.assertEqual(canvas.selected_doorway_index, 0)

        initial_zoom_scale = canvas.zoom_scale
        initial_height_meters = canvas.doorways[0].height_meters
        initial_undo_count = len(canvas.undo_stack)
        committed_changes: list[None] = []
        preview_changes: list[None] = []
        canvas.doorways_changed.connect(lambda: committed_changes.append(None))
        canvas.doorway_dimension_preview_changed.connect(
            lambda: preview_changes.append(None)
        )

        _send_wheel_event(canvas, doorway_position, 120)
        self.assertAlmostEqual(
            canvas.doorways[0].height_meters,
            initial_height_meters + 0.05,
        )
        self.assertAlmostEqual(canvas.zoom_scale, initial_zoom_scale)
        self.assertEqual(committed_changes, [])
        self.assertEqual(preview_changes, [None])
        self.assertEqual(len(canvas.undo_stack), initial_undo_count + 1)

        _send_wheel_event(canvas, doorway_position, -120)
        self.assertAlmostEqual(
            canvas.doorways[0].height_meters,
            initial_height_meters,
        )
        self.assertAlmostEqual(canvas.zoom_scale, initial_zoom_scale)
        self.assertEqual(committed_changes, [])
        self.assertEqual(preview_changes, [None, None])
        self.assertEqual(len(canvas.undo_stack), initial_undo_count + 2)

        canvas.undo_last_step()
        self.assertAlmostEqual(
            canvas.doorways[0].height_meters,
            initial_height_meters + 0.05,
        )
        self.assertEqual(committed_changes, [None])
        canvas.undo_last_step()
        self.assertAlmostEqual(
            canvas.doorways[0].height_meters,
            initial_height_meters,
        )
        self.assertEqual(committed_changes, [None, None])
        self.assertEqual(preview_changes, [None, None])

        canvas.selected_doorway_index = None
        unselected_zoom_scale = canvas.zoom_scale
        _send_wheel_event(canvas, _image_position(canvas, 15.0, 15.0), 120)
        self.assertGreater(canvas.zoom_scale, unselected_zoom_scale)
        self.assertAlmostEqual(
            canvas.doorways[0].height_meters,
            initial_height_meters,
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

    def test_general_tab_places_doorway_without_dimension_editor(self) -> None:
        from housemaker.main import BlueprintWorkspace

        workspace = self._track_widget(BlueprintWorkspace())
        self.assertIsInstance(workspace, BlueprintWorkspace)
        workspace.resize(1500, 900)
        workspace.show()
        _qt_application.processEvents()

        self.assertFalse(hasattr(workspace, "doorway_width_spinbox"))
        self.assertFalse(hasattr(workspace, "doorway_height_spinbox"))
        self.assertFalse(hasattr(workspace, "add_doorway_preset_button"))
        self.assertGreater(len(workspace.doorway_presets), 0)
        workspace.doorway_preset_list.setCurrentRow(0)
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
            workspace.doorway_presets[0],
        )

    def test_general_tab_saves_selected_doorway_as_new_template(self) -> None:
        from housemaker.main import BlueprintWorkspace

        workspace = self._track_widget(BlueprintWorkspace())
        self.assertIsInstance(workspace, BlueprintWorkspace)
        workspace.resize(1500, 900)
        workspace.show()
        _qt_application.processEvents()

        self.assertEqual(
            workspace.save_doorway_template_button.text(),
            "Save doorway template",
        )
        self.assertFalse(workspace.save_doorway_template_button.isEnabled())

        doorway = DoorwayData(
            center_x=50.0,
            center_y=50.0,
            width_meters=1.35,
            height_meters=2.45,
            depth_meters=0.3,
            rotation_degrees=90.0,
            shape=DOORWAY_SHAPE_ARCH,
            arch_amount=0.625,
        )
        workspace.current_level.doorways.append(doorway)
        workspace.canvas._set_selected_doorway_index(0)
        self.assertTrue(workspace.save_doorway_template_button.isEnabled())

        original_preset_count = len(workspace.doorway_presets)
        QTest.mouseClick(
            workspace.save_doorway_template_button,
            Qt.MouseButton.LeftButton,
        )
        _qt_application.processEvents()

        self.assertEqual(
            len(workspace.doorway_presets),
            original_preset_count + 1,
        )
        self.assertEqual(
            workspace.doorway_presets[-1],
            DoorwayPreset(
                width_meters=1.35,
                height_meters=2.45,
                shape=DOORWAY_SHAPE_ARCH,
                arch_amount=0.625,
            ),
        )
        self.assertEqual(
            workspace.doorway_preset_list.currentRow(),
            original_preset_count,
        )
        self.assertEqual(
            workspace.doorway_preset_list.currentItem().text(),
            "1.35 m × 2.45 m — Arch 62.5%",
        )
        self.assertTrue(workspace.remove_doorway_preset_button.isEnabled())
        self.assertTrue(workspace.save_doorway_template_button.isEnabled())

        workspace.canvas._set_selected_doorway_index(None)
        self.assertFalse(workspace.save_doorway_template_button.isEnabled())

    def _build_workspace_with_committed_doorway(
        self,
        settings_directory: str,
        *,
        delay_seconds: float = 0.7,
    ) -> tuple[object, QPoint]:
        from housemaker.main import BlueprintWorkspace

        application_settings = ApplicationSettingsStore(
            Path(settings_directory) / "settings.json"
        )
        application_settings.set(
            DOORWAY_MESH_UPDATE_DELAY_SECONDS_SETTING_KEY,
            delay_seconds,
        )
        workspace = self._track_widget(
            BlueprintWorkspace(application_settings=application_settings)
        )
        self.assertIsInstance(workspace, BlueprintWorkspace)
        workspace.resize(1500, 900)
        workspace.show()
        _qt_application.processEvents()

        doorway = DoorwayData(
            center_x=50.0,
            center_y=50.0,
            width_meters=0.9,
            height_meters=2.1,
            depth_meters=0.2,
            rotation_degrees=0.0,
        )
        workspace.current_level.doorways[:] = [doorway]
        workspace._reset_viewer_doorway_snapshots()
        workspace.canvas.blueprint_image = QImage(
            100,
            100,
            QImage.Format.Format_RGB32,
        )
        workspace.canvas.blueprint_image.fill(Qt.GlobalColor.white)
        workspace.canvas.selected_doorway_index = 0
        doorway_position = _image_position(
            workspace.canvas,
            doorway.center_x,
            doorway.center_y,
        )
        return workspace, doorway_position

    def test_workspace_debounces_doorway_mesh_until_timer_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace, doorway_position = (
                self._build_workspace_with_committed_doorway(
                    temporary_directory,
                )
            )

            initial_revision = workspace._viewer_preview_revision
            with patch.object(
                workspace,
                "_schedule_viewer_preview_refresh",
                wraps=workspace._schedule_viewer_preview_refresh,
            ) as schedule_refresh:
                _send_wheel_event(workspace.canvas, doorway_position, 120)

                schedule_refresh.assert_not_called()
                self.assertEqual(
                    workspace._viewer_preview_revision,
                    initial_revision,
                )
                self.assertTrue(
                    workspace._doorway_mesh_update_timer.isSingleShot()
                )
                self.assertTrue(workspace._doorway_mesh_update_timer.isActive())
                self.assertEqual(
                    workspace._doorway_mesh_update_timer.interval(),
                    700,
                )
                self.assertEqual(
                    workspace._pending_doorway_mesh_level_index,
                    workspace.current_level.index,
                )
                self.assertIsNotNone(
                    workspace.viewer._doorway_preview_outline_positions
                )

                workspace._doorway_mesh_update_timer.timeout.emit()
                workspace._doorway_mesh_update_timer.timeout.emit()

                schedule_refresh.assert_called_once_with(
                    preserve_camera=True
                )

            self.assertFalse(workspace._doorway_mesh_update_timer.isActive())
            self.assertIsNone(workspace._pending_doorway_mesh_level_index)
            self.assertEqual(
                workspace._viewer_preview_revision,
                initial_revision + 1,
            )
            self.assertEqual(
                workspace._doorway_outline_commit_revision,
                workspace._viewer_preview_revision,
            )
            self.assertEqual(
                workspace._viewer_doorways_by_level_index[
                    workspace.current_level.index
                ],
                tuple(workspace.current_level.doorways),
            )

    def test_workspace_starts_doorway_delay_only_after_drag_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace, _doorway_position = (
                self._build_workspace_with_committed_doorway(
                    temporary_directory,
                    delay_seconds=0.1,
                )
            )
            doorway = workspace.canvas.doorways[0]
            width_border_position = _image_position(
                workspace.canvas,
                *_get_width_border_image_position(doorway),
            )
            resized_position = _image_position(
                workspace.canvas,
                doorway.center_x,
                80.0,
            )
            initial_revision = workspace._viewer_preview_revision

            with patch.object(
                workspace,
                "_schedule_viewer_preview_refresh",
                wraps=workspace._schedule_viewer_preview_refresh,
            ) as schedule_refresh:
                QTest.mousePress(
                    workspace.canvas,
                    Qt.MouseButton.LeftButton,
                    pos=width_border_position,
                )
                _send_drag_move(workspace.canvas, resized_position)

                self.assertTrue(
                    workspace._is_doorway_dimension_drag_active
                )
                self.assertFalse(
                    workspace._doorway_mesh_update_timer.isActive()
                )
                self.assertIsNotNone(
                    workspace.viewer._doorway_preview_outline_positions
                )

                QTest.qWait(150)

                schedule_refresh.assert_not_called()
                self.assertEqual(
                    workspace._viewer_preview_revision,
                    initial_revision,
                )
                self.assertFalse(
                    workspace._doorway_mesh_update_timer.isActive()
                )

                QTest.mouseRelease(
                    workspace.canvas,
                    Qt.MouseButton.LeftButton,
                    pos=resized_position,
                )

                self.assertFalse(
                    workspace._is_doorway_dimension_drag_active
                )
                self.assertTrue(
                    workspace._doorway_mesh_update_timer.isActive()
                )
                self.assertEqual(
                    workspace._doorway_mesh_update_timer.interval(),
                    100,
                )
                schedule_refresh.assert_not_called()

    def test_stationary_drag_does_not_replace_another_doorway_outline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace, doorway_a_position = (
                self._build_workspace_with_committed_doorway(
                    temporary_directory,
                )
            )
            doorway_b = DoorwayData(
                center_x=20.0,
                center_y=20.0,
                width_meters=0.2,
                height_meters=2.1,
                depth_meters=0.2,
                rotation_degrees=0.0,
            )
            workspace.current_level.doorways.append(doorway_b)
            workspace._reset_viewer_doorway_snapshots()
            committed_doorways = (
                workspace._viewer_doorways_by_level_index[
                    workspace.current_level.index
                ]
            )

            workspace.canvas.selected_doorway_index = 0
            _send_wheel_event(
                workspace.canvas,
                doorway_a_position,
                120,
            )
            doorway_a_outline = (
                workspace.viewer._doorway_preview_outline_positions
            )
            self.assertIsNotNone(doorway_a_outline)
            self.assertTrue(workspace._doorway_mesh_update_timer.isActive())

            doorway_b_border_position = _image_position(
                workspace.canvas,
                *_get_width_border_image_position(doorway_b),
            )
            QTest.mousePress(
                workspace.canvas,
                Qt.MouseButton.LeftButton,
                pos=doorway_b_border_position,
            )

            self.assertEqual(workspace.canvas.selected_doorway_index, 1)
            self.assertTrue(workspace._is_doorway_dimension_drag_active)
            self.assertFalse(workspace._doorway_mesh_update_timer.isActive())
            np.testing.assert_array_equal(
                workspace.viewer._doorway_preview_outline_positions,
                doorway_a_outline,
            )

            QTest.mouseRelease(
                workspace.canvas,
                Qt.MouseButton.LeftButton,
                pos=doorway_b_border_position,
            )

            self.assertFalse(workspace._is_doorway_dimension_drag_active)
            self.assertTrue(workspace._doorway_mesh_update_timer.isActive())
            self.assertEqual(
                workspace._doorway_mesh_update_timer.interval(),
                700,
            )
            self.assertEqual(
                workspace._pending_doorway_mesh_level_index,
                workspace.current_level.index,
            )
            np.testing.assert_array_equal(
                workspace.viewer._doorway_preview_outline_positions,
                doorway_a_outline,
            )
            self.assertEqual(
                workspace._viewer_doorways_by_level_index[
                    workspace.current_level.index
                ],
                committed_doorways,
            )

    def test_canvas_mode_reset_finishes_changed_doorway_drag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace, _doorway_position = (
                self._build_workspace_with_committed_doorway(
                    temporary_directory,
                )
            )
            doorway = workspace.canvas.doorways[0]
            committed_doorways = (
                workspace._viewer_doorways_by_level_index[
                    workspace.current_level.index
                ]
            )
            initial_revision = workspace._viewer_preview_revision
            initial_width_meters = doorway.width_meters
            width_border_position = _image_position(
                workspace.canvas,
                *_get_width_border_image_position(doorway),
            )
            resized_position = _image_position(
                workspace.canvas,
                doorway.center_x,
                80.0,
            )

            QTest.mousePress(
                workspace.canvas,
                Qt.MouseButton.LeftButton,
                pos=width_border_position,
            )
            _send_drag_move(workspace.canvas, resized_position)
            pending_outline = (
                workspace.viewer._doorway_preview_outline_positions
            )

            self.assertIsNotNone(pending_outline)
            self.assertNotEqual(
                workspace.canvas.doorways[0].width_meters,
                initial_width_meters,
            )
            self.assertTrue(workspace._is_doorway_dimension_drag_active)
            self.assertFalse(workspace._doorway_mesh_update_timer.isActive())
            self.assertEqual(
                workspace._pending_doorway_mesh_level_index,
                workspace.current_level.index,
            )

            workspace.canvas.start_floor_contour_designation()

            self.assertIsNone(workspace.canvas.selected_doorway_index)
            self.assertFalse(workspace._is_doorway_dimension_drag_active)
            self.assertTrue(workspace._doorway_mesh_update_timer.isActive())
            self.assertEqual(
                workspace._doorway_mesh_update_timer.interval(),
                700,
            )
            self.assertEqual(
                workspace._pending_doorway_mesh_level_index,
                workspace.current_level.index,
            )
            np.testing.assert_array_equal(
                workspace.viewer._doorway_preview_outline_positions,
                pending_outline,
            )
            self.assertEqual(
                workspace._viewer_doorways_by_level_index[
                    workspace.current_level.index
                ],
                committed_doorways,
            )
            self.assertEqual(
                workspace._viewer_preview_revision,
                initial_revision,
            )

    def test_returning_to_committed_doorway_cancels_pending_update(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace, doorway_position = (
                self._build_workspace_with_committed_doorway(
                    temporary_directory,
                )
            )
            level_index = workspace.current_level.index
            committed_doorways = workspace._viewer_doorways_by_level_index[
                level_index
            ]
            initial_revision = workspace._viewer_preview_revision

            with patch.object(
                workspace,
                "_schedule_viewer_preview_refresh",
                wraps=workspace._schedule_viewer_preview_refresh,
            ) as schedule_refresh:
                _send_wheel_event(workspace.canvas, doorway_position, 120)
                self.assertTrue(workspace._doorway_mesh_update_timer.isActive())

                workspace.current_level.doorways[:] = [
                    DoorwayData(**vars(committed_doorways[0]))
                ]
                workspace.canvas.doorway_dimension_preview_changed.emit()

                schedule_refresh.assert_not_called()

            self.assertFalse(workspace._doorway_mesh_update_timer.isActive())
            self.assertIsNone(workspace._pending_doorway_mesh_level_index)
            self.assertIsNone(workspace._doorway_outline_commit_revision)
            self.assertIsNone(
                workspace.viewer._doorway_preview_outline_positions
            )
            self.assertEqual(
                workspace._viewer_preview_revision,
                initial_revision,
            )
            self.assertEqual(
                workspace._viewer_doorways_by_level_index[level_index],
                committed_doorways,
            )

    def test_structural_doorway_change_cancels_pending_and_commits_now(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace, doorway_position = (
                self._build_workspace_with_committed_doorway(
                    temporary_directory,
                )
            )
            _send_wheel_event(workspace.canvas, doorway_position, 120)
            self.assertTrue(workspace._doorway_mesh_update_timer.isActive())
            initial_revision = workspace._viewer_preview_revision

            with patch.object(
                workspace,
                "_schedule_viewer_preview_refresh",
                wraps=workspace._schedule_viewer_preview_refresh,
            ) as schedule_refresh:
                workspace.canvas.doorways.clear()
                workspace.canvas.selected_doorway_index = None
                workspace.canvas.doorways_changed.emit()

                schedule_refresh.assert_called_once_with()

            self.assertFalse(workspace._doorway_mesh_update_timer.isActive())
            self.assertIsNone(workspace._pending_doorway_mesh_level_index)
            self.assertIsNone(workspace._doorway_outline_commit_revision)
            self.assertIsNone(
                workspace.viewer._doorway_preview_outline_positions
            )
            self.assertEqual(
                workspace._viewer_preview_revision,
                initial_revision + 1,
            )
            self.assertEqual(
                workspace._viewer_doorways_by_level_index[
                    workspace.current_level.index
                ],
                (),
            )

    def test_unrelated_preview_build_uses_committed_doorways_while_pending(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace, doorway_position = (
                self._build_workspace_with_committed_doorway(
                    temporary_directory,
                )
            )
            level_index = workspace.current_level.index
            committed_doorways = workspace._viewer_doorways_by_level_index[
                level_index
            ]
            _send_wheel_event(workspace.canvas, doorway_position, 120)
            live_height = workspace.current_level.doorways[0].height_meters
            self.assertNotEqual(
                live_height,
                committed_doorways[0].height_meters,
            )

            with patch(
                "housemaker.main.convert_to_preview_model",
                return_value=sentinel.preview_model,
            ) as convert_preview:
                result = workspace._build_viewer_preview_model(None)

            self.assertIs(result, sentinel.preview_model)
            preview_levels = convert_preview.call_args.args[0]
            preview_level = next(
                level for level in preview_levels if level.index == level_index
            )
            self.assertIsNot(preview_level, workspace.current_level)
            self.assertIsNot(
                preview_level.doorways,
                workspace.current_level.doorways,
            )
            self.assertEqual(
                tuple(preview_level.doorways),
                committed_doorways,
            )
            self.assertNotEqual(
                preview_level.doorways[0].height_meters,
                live_height,
            )
            self.assertTrue(workspace._doorway_mesh_update_timer.isActive())

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
