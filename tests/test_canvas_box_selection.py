# ### Environment setup ###
from __future__ import annotations

import os
import threading
import time
import unittest
from collections.abc import Callable
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PySide6.QtCore import QEvent, QPointF, QRect, Qt
from PySide6.QtGui import QKeyEvent, QMatrix4x4
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.glb import GeneratedModel, PreviewPlacedObject
from housemaker.surface_geometry import (
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    FixedSurface,
)
from housemaker.viewer import (
    CANVAS_SELECTION_TARGET_OBJECT,
    CANVAS_SELECTION_TARGET_OCCLUDER,
    CANVAS_SELECTION_TARGET_SURFACE,
    GlbViewerWidget,
    _CanvasRectangleSelectionResult,
    _capture_face_selection_raster_input,
    _rasterize_canvas_target_selection,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Test doubles ###
class _MouseButtonEvent:
    """Small deterministic pointer event for direct event-flow tests."""

    def __init__(
        self,
        *,
        button: Qt.MouseButton,
        position: QPointF,
    ) -> None:
        self._button = button
        self._position = QPointF(position)
        self.was_accepted = False

    def button(self) -> Qt.MouseButton:
        return self._button

    def position(self) -> QPointF:
        return QPointF(self._position)

    def accept(self) -> None:
        self.was_accepted = True


# ### Fixture helpers ###
def _translation_transform(
    x: float,
    y: float,
    z: float,
) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = (x, y, z)
    return transform


def _build_placed_object(
    object_id: str,
    *,
    world_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PreviewPlacedObject:
    mesh = trimesh.creation.box(extents=(0.8, 0.8, 1.0))
    mesh.apply_translation((0.0, 0.0, 0.5))
    return PreviewPlacedObject(
        object_id=object_id,
        meshes=(mesh,),
        placement_transform=_translation_transform(*world_position),
        world_position=world_position,
        rotation_degrees=(0.0, 0.0, 0.0),
    )


def _build_preview_model(
    *placed_objects: PreviewPlacedObject,
) -> GeneratedModel:
    base_mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    base_mesh.apply_translation((100.0, 100.0, 100.0))
    return GeneratedModel(
        mesh=base_mesh,
        scene=trimesh.Scene(base_mesh),
        glb_bytes=b"",
        preview_placed_objects=list(placed_objects),
        preview_base_mesh=base_mesh.copy(),
    )


def _build_wall(
    surface_id: str,
    *,
    y: float = 2.0,
) -> FixedSurface:
    vertices = np.asarray(
        (
            (-2.0, y, 0.0),
            (2.0, y, 0.0),
            (2.0, y, 2.0),
            (-2.0, y, 2.0),
        ),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id=surface_id,
        surface_type=SURFACE_TYPE_WALL,
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=8.0,
        wall_key="1:2",
        wall_start_world=(-2.0, y, 0.0),
        wall_end_world=(2.0, y, 0.0),
        wall_height_meters=2.0,
    )


def _build_horizontal_surface(
    surface_type: str,
    *,
    surface_id: str,
    z: float,
) -> FixedSurface:
    vertices = np.asarray(
        (
            (-2.0, -2.0, z),
            (2.0, -2.0, z),
            (2.0, 2.0, z),
            (-2.0, 2.0, z),
        ),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id=surface_id,
        surface_type=surface_type,
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=16.0,
    )


def _forward_ray() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((0.0, -5.0, 0.5), dtype=float),
        np.asarray((0.0, 1.0, 0.0), dtype=float),
    )


def _projected_quad(
    left: float,
    top: float,
    right: float,
    bottom: float,
    depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        (
            (left, top, depth, 1.0),
            (right, top, depth, 1.0),
            (right, bottom, depth, 1.0),
            (left, bottom, depth, 1.0),
        ),
        dtype=float,
    )
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    return vertices, faces


def _wait_until(
    predicate: Callable[[], bool],
    timeout_milliseconds: int = 2_000,
) -> bool:
    deadline = time.monotonic() + timeout_milliseconds / 1_000.0
    while time.monotonic() < deadline:
        _qt_application.processEvents()
        if predicate():
            return True
        QTest.qWait(10)
    _qt_application.processEvents()
    return bool(predicate())


# ### Canvas target-raster tests ###
class CanvasTargetRasterTests(unittest.TestCase):
    def test_camera_plane_crossing_surface_remains_box_selectable(self) -> None:
        class PerspectiveView:
            def width(self) -> int:
                return 100

            def height(self) -> int:
                return 100

            def projectionMatrix(self, *_args: object) -> QMatrix4x4:
                projection = QMatrix4x4()
                projection.perspective(90.0, 1.0, 1.0, 10.0)
                return projection

            def viewMatrix(self) -> QMatrix4x4:
                return QMatrix4x4()

        vertices = np.asarray(
            (
                (0.0, 0.0, 1.0),
                (-1.0, -1.0, -2.0),
                (1.0, -1.0, -2.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        captured = _capture_face_selection_raster_input(
            PerspectiveView(),  # type: ignore[arg-type]
            ((vertices, faces),),
            QRect(0, 0, 100, 100),
        )

        self.assertIsNotNone(captured)
        assert captured is not None
        projected_geometry, rectangle = captured
        surface_ids, object_ids = _rasterize_canvas_target_selection(
            (
                (
                    CANVAS_SELECTION_TARGET_SURFACE,
                    "floor",
                    *projected_geometry[0],
                ),
            ),
            QRect(*rectangle),
        )

        self.assertEqual(surface_ids, ("floor",))
        self.assertEqual(object_ids, ())

    def test_depth_raster_returns_every_visible_target_but_not_occluded_ones(
        self,
    ) -> None:
        wall = _projected_quad(0.0, 0.0, 100.0, 80.0, 0.4)
        occluded = _projected_quad(12.0, 12.0, 38.0, 58.0, 0.6)
        chair = _projected_quad(10.0, 10.0, 40.0, 60.0, -0.3)
        table = _projected_quad(60.0, 10.0, 90.0, 60.0, -0.2)

        surface_ids, object_ids = _rasterize_canvas_target_selection(
            (
                (CANVAS_SELECTION_TARGET_SURFACE, "wall", *wall),
                (CANVAS_SELECTION_TARGET_OBJECT, "hidden", *occluded),
                (CANVAS_SELECTION_TARGET_OBJECT, "chair", *chair),
                (CANVAS_SELECTION_TARGET_OBJECT, "table", *table),
            ),
            QRect(0, 0, 101, 81),
        )

        self.assertEqual(surface_ids, ("wall",))
        self.assertEqual(object_ids, ("chair", "table"))

    def test_front_depth_only_geometry_suppresses_a_covered_target(
        self,
    ) -> None:
        floor = _projected_quad(10.0, 10.0, 90.0, 70.0, 0.4)
        stair = _projected_quad(0.0, 0.0, 100.0, 80.0, -0.2)

        surface_ids, object_ids = _rasterize_canvas_target_selection(
            (
                (CANVAS_SELECTION_TARGET_SURFACE, "floor", *floor),
                (CANVAS_SELECTION_TARGET_OCCLUDER, "scene", *stair),
            ),
            QRect(0, 0, 101, 81),
        )

        self.assertEqual(surface_ids, ())
        self.assertEqual(object_ids, ())

    def test_coplanar_depth_only_geometry_keeps_target_with_other_diagonal(
        self,
    ) -> None:
        floor = _projected_quad(10.0, 10.0, 90.0, 70.0, 0.2)
        coplanar_vertices = floor[0].copy()
        coplanar_faces = np.asarray(
            ((0, 1, 3), (1, 2, 3)),
            dtype=np.int64,
        )

        surface_ids, object_ids = _rasterize_canvas_target_selection(
            (
                (CANVAS_SELECTION_TARGET_SURFACE, "floor", *floor),
                (
                    CANVAS_SELECTION_TARGET_OCCLUDER,
                    "scene",
                    coplanar_vertices,
                    coplanar_faces,
                ),
            ),
            QRect(0, 0, 101, 81),
        )

        self.assertEqual(surface_ids, ("floor",))
        self.assertEqual(object_ids, ())


# ### Canvas box-selection tests ###
class CanvasBoxSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[GlbViewerWidget] = []

    def tearDown(self) -> None:
        for widget in self.widgets:
            widget.exit_first_person_mode()
            widget.close()
            widget.deleteLater()
        _qt_application.processEvents()

    def _build_viewer(
        self,
        *placed_objects: PreviewPlacedObject,
        surfaces: tuple[FixedSurface, ...] = (),
    ) -> GlbViewerWidget:
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.resize(320, 240)
        viewer.set_model(_build_preview_model(*placed_objects))
        viewer.set_wall_targets(surfaces)
        self.widgets.append(viewer)
        return viewer

    def _result(
        self,
        viewer: GlbViewerWidget,
        *,
        surface_ids: tuple[str, ...] = (),
        object_ids: tuple[str, ...] = (),
        additive: bool = False,
    ) -> _CanvasRectangleSelectionResult:
        return _CanvasRectangleSelectionResult(
            request_revision=(
                viewer._canvas_rectangle_selection_request_revision
            ),
            geometry_revision=viewer._canvas_selection_geometry_revision,
            surface_ids=surface_ids,
            object_ids=object_ids,
            additive=additive,
        )

    def test_short_plain_gesture_delegates_to_existing_exact_click_picker(
        self,
    ) -> None:
        viewer = self._build_viewer(_build_placed_object("chair"))
        start = QPointF(30.0, 30.0)
        end = QPointF(33.0, 30.0)

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_forward_ray(),
        ):
            viewer._handle_placed_object_pointer_pressed(start)
            self.assertTrue(viewer.view.is_primary_pointer_drag_reserved)
            viewer._handle_canvas_gizmo_pointer_released(end)

        self.assertEqual(viewer.get_selected_placed_object_ids(), ("chair",))
        self.assertEqual(viewer.get_selected_placed_object_id(), "chair")
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)

    def test_right_click_cancel_cannot_select_on_later_left_release(self) -> None:
        viewer = self._build_viewer(_build_placed_object("chair"))
        position = QPointF(30.0, 30.0)

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_forward_ray(),
        ):
            viewer.view.mousePressEvent(
                _MouseButtonEvent(
                    button=Qt.MouseButton.LeftButton,
                    position=position,
                )
            )
            self.assertTrue(viewer.view.is_primary_pointer_drag_reserved)
            viewer.view.mousePressEvent(
                _MouseButtonEvent(
                    button=Qt.MouseButton.RightButton,
                    position=position,
                )
            )
            viewer.view.mouseReleaseEvent(
                _MouseButtonEvent(
                    button=Qt.MouseButton.LeftButton,
                    position=position,
                )
            )

        self.assertEqual(viewer.get_selected_placed_object_ids(), ())
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)

    def test_escape_cancel_cannot_select_on_later_left_release(self) -> None:
        viewer = self._build_viewer(_build_placed_object("chair"))
        position = QPointF(30.0, 30.0)

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_forward_ray(),
        ):
            viewer.view.mousePressEvent(
                _MouseButtonEvent(
                    button=Qt.MouseButton.LeftButton,
                    position=position,
                )
            )
            self.assertTrue(viewer.view.is_primary_pointer_drag_reserved)
            viewer.view.keyPressEvent(
                QKeyEvent(
                    QEvent.Type.KeyPress,
                    Qt.Key.Key_Escape,
                    Qt.KeyboardModifier.NoModifier,
                )
            )
            viewer.view.mouseReleaseEvent(
                _MouseButtonEvent(
                    button=Qt.MouseButton.LeftButton,
                    position=position,
                )
            )

        self.assertEqual(viewer.get_selected_placed_object_ids(), ())
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)

    def test_rectangle_cleanup_does_not_release_another_pointer_owner(
        self,
    ) -> None:
        viewer = self._build_viewer(_build_placed_object("chair"))
        viewer.view.reserve_primary_pointer_drag()

        self.assertFalse(viewer._cancel_canvas_rectangle_selection())

        self.assertTrue(viewer.view.is_primary_pointer_drag_reserved)
        viewer.view.cancel_primary_pointer_drag()

    def test_real_drag_shows_rubber_band_and_starts_one_box_request(self) -> None:
        viewer = self._build_viewer(_build_placed_object("chair"))
        start = QPointF(12.0, 18.0)
        current = QPointF(100.0, 90.0)

        with (
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=_forward_ray(),
            ),
            patch.object(
                viewer,
                "_start_canvas_rectangle_selection",
                return_value=True,
            ) as start_selection,
            patch(
                "housemaker.viewer.QApplication.keyboardModifiers",
                return_value=Qt.KeyboardModifier.ShiftModifier,
            ),
        ):
            viewer._handle_placed_object_pointer_pressed(start)
            viewer._handle_canvas_gizmo_pointer_moved(current)

            rubber_band = viewer._canvas_rectangle_selection_rubber_band
            self.assertIsNotNone(rubber_band)
            assert rubber_band is not None
            self.assertFalse(rubber_band.isHidden())
            self.assertEqual(
                rubber_band.geometry().normalized(),
                rubber_band.geometry(),
            )
            self.assertGreater(rubber_band.width(), 4)
            self.assertGreater(rubber_band.height(), 4)

            viewer._handle_canvas_gizmo_pointer_released(current)

        start_selection.assert_called_once()
        self.assertTrue(start_selection.call_args.kwargs["additive"])
        self.assertTrue(rubber_band.isHidden())
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)

    def test_capture_includes_only_targets_allowed_by_visibility_mode(
        self,
    ) -> None:
        wall = _build_wall("level:0/wall:1:2")
        floor = _build_horizontal_surface(
            SURFACE_TYPE_FLOOR,
            surface_id="level:0/floor",
            z=0.0,
        )
        ceiling = _build_horizontal_surface(
            SURFACE_TYPE_CEILING,
            surface_id="level:0/ceiling",
            z=3.0,
        )
        viewer = self._build_viewer(
            _build_placed_object("chair"),
            surfaces=(wall, floor, ceiling),
        )

        def identity_capture(
            _view: object,
            geometry: object,
            rectangle: QRect,
        ) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], tuple[int, ...]]:
            projected = tuple(
                (
                    np.column_stack(
                        (
                            np.asarray(vertices, dtype=float),
                            np.ones(len(vertices), dtype=float),
                        )
                    ),
                    np.asarray(faces, dtype=np.int64),
                )
                for vertices, faces in geometry  # type: ignore[union-attr]
            )
            return projected, (
                rectangle.x(),
                rectangle.y(),
                rectangle.width(),
                rectangle.height(),
            )

        with patch(
            "housemaker.viewer._capture_face_selection_raster_input",
            side_effect=identity_capture,
        ):
            viewer.set_canvas_ceiling_hidden(True)
            captured = viewer._capture_canvas_rectangle_selection_targets(
                QPointF(0.0, 0.0),
                QPointF(100.0, 100.0),
            )
            self.assertIsNotNone(captured)
            assert captured is not None
            targets, _rectangle = captured
            self.assertEqual(
                {(target_type, target_id) for target_type, target_id, *_ in targets},
                {
                    (CANVAS_SELECTION_TARGET_OBJECT, "chair"),
                    (CANVAS_SELECTION_TARGET_SURFACE, wall.surface_id),
                    (CANVAS_SELECTION_TARGET_SURFACE, floor.surface_id),
                    (CANVAS_SELECTION_TARGET_OCCLUDER, "scene"),
                },
            )

            viewer.set_canvas_surface_focus_type(SURFACE_TYPE_FLOOR)
            focused = viewer._capture_canvas_rectangle_selection_targets(
                QPointF(0.0, 0.0),
                QPointF(100.0, 100.0),
            )

        self.assertIsNotNone(focused)
        assert focused is not None
        focused_targets, _rectangle = focused
        self.assertEqual(
            tuple(
                (target_type, target_id)
                for target_type, target_id, *_ in focused_targets
            ),
            (
                (CANVAS_SELECTION_TARGET_SURFACE, floor.surface_id),
                (CANVAS_SELECTION_TARGET_OCCLUDER, "scene"),
            ),
        )

    def test_box_raster_runs_off_thread_and_delivers_on_the_gui_thread(
        self,
    ) -> None:
        viewer = self._build_viewer(_build_placed_object("chair"))
        projected_quad = _projected_quad(0.0, 0.0, 20.0, 20.0, 0.0)
        for array in projected_quad:
            array.setflags(write=False)
        captured_targets = (
            (
                (
                    CANVAS_SELECTION_TARGET_OBJECT,
                    "chair",
                    *projected_quad,
                ),
            ),
            (0, 0, 21, 21),
        )
        main_thread_id = threading.get_ident()
        started = threading.Event()
        release = threading.Event()
        delivery_threads: list[int] = []
        observations: dict[str, object] = {}
        viewer.placed_object_selection_set_changed.connect(
            lambda _ids: delivery_threads.append(threading.get_ident())
        )

        def blocking_raster(
            projected_targets: object,
            _rectangle: QRect,
            *,
            cancel_event: threading.Event | None = None,
        ) -> tuple[tuple[str, ...], tuple[str, ...]]:
            observations["thread_id"] = threading.get_ident()
            observations["daemon"] = threading.current_thread().daemon
            observations["read_only"] = all(
                not array.flags.writeable
                for _target_type, _target_id, vertices, faces
                in projected_targets  # type: ignore[union-attr]
                for array in (vertices, faces)
            )
            observations["cancel_event"] = cancel_event is not None
            started.set()
            release.wait(5.0)
            return (), ("chair",)

        try:
            with (
                patch.object(
                    viewer,
                    "_capture_canvas_rectangle_selection_targets",
                    return_value=captured_targets,
                ),
                patch(
                    "housemaker.viewer._rasterize_canvas_target_selection",
                    side_effect=blocking_raster,
                ),
            ):
                self.assertTrue(
                    viewer._start_canvas_rectangle_selection(
                        QPointF(0.0, 0.0),
                        QPointF(20.0, 20.0),
                    )
                )
                self.assertTrue(started.wait(2.0))
                release.set()
                self.assertTrue(
                    _wait_until(
                        lambda: viewer.get_selected_placed_object_ids()
                        == ("chair",)
                    )
                )
        finally:
            release.set()

        self.assertNotEqual(observations["thread_id"], main_thread_id)
        self.assertTrue(observations["daemon"])
        self.assertTrue(observations["read_only"])
        self.assertTrue(observations["cancel_event"])
        self.assertEqual(delivery_threads, [main_thread_id])

    def test_visible_objects_win_the_box_category_and_keep_active_compatibility(
        self,
    ) -> None:
        wall = _build_wall("level:0/wall:1:2")
        viewer = self._build_viewer(
            _build_placed_object("chair"),
            _build_placed_object("table", world_position=(2.0, 0.0, 0.0)),
            surfaces=(wall,),
        )
        viewer.select_canvas_surface_target(wall.surface_id)

        viewer._apply_canvas_rectangle_selection_result(
            self._result(
                viewer,
                surface_ids=(wall.surface_id,),
                object_ids=("chair", "table"),
            )
        )

        self.assertEqual(
            viewer.get_selected_placed_object_ids(),
            ("chair", "table"),
        )
        self.assertEqual(viewer.get_selected_placed_object_id(), "table")
        self.assertEqual(viewer.get_selected_canvas_surface_ids(), ())

    def test_surface_results_replace_objects_when_no_object_pixel_is_visible(
        self,
    ) -> None:
        first = _build_wall("level:0/wall:1:2")
        second = _build_wall("level:0/wall:3:4", y=3.0)
        viewer = self._build_viewer(
            _build_placed_object("chair"),
            surfaces=(first, second),
        )
        viewer.select_placed_object("chair")

        viewer._apply_canvas_rectangle_selection_result(
            self._result(
                viewer,
                surface_ids=(first.surface_id, second.surface_id),
            )
        )

        self.assertEqual(viewer.get_selected_placed_object_ids(), ())
        self.assertIsNone(viewer.get_selected_placed_object_id())
        self.assertEqual(
            viewer.get_selected_canvas_surface_ids(),
            (first.surface_id, second.surface_id),
        )

    def test_shift_box_adds_only_within_the_result_category(self) -> None:
        wall = _build_wall("level:0/wall:1:2")
        viewer = self._build_viewer(
            _build_placed_object("chair"),
            _build_placed_object("table", world_position=(2.0, 0.0, 0.0)),
            surfaces=(wall,),
        )
        viewer.select_placed_object("chair")

        viewer._apply_canvas_rectangle_selection_result(
            self._result(
                viewer,
                object_ids=("table",),
                additive=True,
            )
        )

        self.assertEqual(
            viewer.get_selected_placed_object_ids(),
            ("chair", "table"),
        )
        viewer._apply_canvas_rectangle_selection_result(
            self._result(
                viewer,
                surface_ids=(wall.surface_id,),
                additive=True,
            )
        )

        self.assertEqual(viewer.get_selected_placed_object_ids(), ())
        self.assertEqual(
            viewer.get_selected_canvas_surface_ids(),
            (wall.surface_id,),
        )

    def test_stale_async_box_results_do_not_replace_current_selection(self) -> None:
        viewer = self._build_viewer(_build_placed_object("chair"))
        stale_request = self._result(viewer, object_ids=("chair",))
        viewer._canvas_rectangle_selection_request_revision += 1

        viewer._apply_canvas_rectangle_selection_result(stale_request)

        self.assertEqual(viewer.get_selected_placed_object_ids(), ())

        stale_geometry = self._result(viewer, object_ids=("chair",))
        viewer._canvas_selection_geometry_revision += 1
        viewer._apply_canvas_rectangle_selection_result(stale_geometry)

        self.assertEqual(viewer.get_selected_placed_object_ids(), ())

    def test_newer_explicit_selection_invalidates_pending_box_result(self) -> None:
        viewer = self._build_viewer(
            _build_placed_object("chair"),
            _build_placed_object("table", world_position=(2.0, 0.0, 0.0)),
        )
        pending_result = self._result(viewer, object_ids=("chair",))

        viewer.select_placed_object("table")
        viewer._apply_canvas_rectangle_selection_result(pending_result)

        self.assertEqual(
            viewer.get_selected_placed_object_ids(),
            ("table",),
            "A completed old raster must not overwrite a newer selection.",
        )


if __name__ == "__main__":
    unittest.main()
