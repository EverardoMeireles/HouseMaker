# ### Environment setup ###
from __future__ import annotations

from collections.abc import Callable
import os
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QFocusEvent, QMatrix4x4, QVector4D
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QWidget

from housemaker.glb import GeneratedModel
from housemaker.viewer import (
    GlbViewerWidget,
    SelectableGLViewWidget,
    _FaceRectangleSelectionResult,
    _WireframeOverlayMeshItem,
    _get_nearest_triangle_ray_face_index,
    _project_vertices_to_view,
    _rasterize_face_selection,
    _select_face_indices_in_view_rectangle,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])


# ### Fixture helpers ###
def _triangle_model(vertices: np.ndarray) -> GeneratedModel:
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
    )
    scene = trimesh.Scene(mesh)
    return GeneratedModel(
        mesh=mesh,
        scene=scene,
        glb_bytes=bytes(scene.export(file_type="glb")),
    )


def _reference_visible_face_indices(
    projected_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    width: int,
    height: int,
) -> set[int]:
    """Brute-force exact pixel-center visibility for small test scenes."""

    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    face_buffer = np.full((height, width), -1, dtype=np.int32)
    for pixel_y in range(height):
        sample_y = pixel_y + 0.5
        for pixel_x in range(width):
            sample_x = pixel_x + 0.5
            for face_index, face in enumerate(faces):
                triangle = projected_vertices[face]
                first, second, third = triangle[:, :2]
                denominator = (
                    (first[0] - third[0])
                    * (second[1] - third[1])
                    - (second[0] - third[0])
                    * (first[1] - third[1])
                )
                if abs(float(denominator)) <= 1e-12:
                    continue
                first_weight = (
                    (second[1] - third[1]) * (sample_x - third[0])
                    + (third[0] - second[0]) * (sample_y - third[1])
                ) / denominator
                second_weight = (
                    (third[1] - first[1]) * (sample_x - third[0])
                    + (first[0] - third[0]) * (sample_y - third[1])
                ) / denominator
                third_weight = 1.0 - first_weight - second_weight
                if min(first_weight, second_weight, third_weight) < -1e-7:
                    continue
                depth = (
                    first_weight * triangle[0, 2]
                    + second_weight * triangle[1, 2]
                    + third_weight * triangle[2, 2]
                )
                if not -1.0 - 1e-6 <= depth <= 1.0 + 1e-6:
                    continue
                if depth >= depth_buffer[pixel_y, pixel_x]:
                    continue
                depth_buffer[pixel_y, pixel_x] = depth
                face_buffer[pixel_y, pixel_x] = face_index
    return {
        int(face_index)
        for face_index in np.unique(face_buffer)
        if face_index >= 0
    }


def _wait_until(
    predicate: Callable[[], bool],
    timeout_milliseconds: int = 2_000,
) -> bool:
    """Process Qt events until a small asynchronous-test condition succeeds."""

    deadline = time.monotonic() + timeout_milliseconds / 1_000.0
    while time.monotonic() < deadline:
        _qt_application.processEvents()
        if predicate():
            return True
        QTest.qWait(10)
    _qt_application.processEvents()
    return bool(predicate())


def _project_vertices_with_qt_map(
    vertices: np.ndarray,
    view_projection: QMatrix4x4,
    viewport_width: int,
    viewport_height: int,
) -> np.ndarray:
    """Reference the former per-vertex Qt projection implementation."""

    projected = np.full((len(vertices), 4), np.nan, dtype=float)
    for vertex_index, vertex in enumerate(np.asarray(vertices, dtype=float)):
        clip = view_projection.map(
            QVector4D(
                float(vertex[0]),
                float(vertex[1]),
                float(vertex[2]),
                1.0,
            )
        )
        clip_w = float(clip.w())
        if not np.isfinite(clip_w) or clip_w <= 1e-10:
            continue
        normalized = np.asarray(
            (float(clip.x()), float(clip.y()), float(clip.z())),
            dtype=float,
        ) / clip_w
        if not np.all(np.isfinite(normalized)):
            continue
        projected[vertex_index] = (
            (normalized[0] + 1.0) * 0.5 * viewport_width,
            (1.0 - normalized[1]) * 0.5 * viewport_height,
            normalized[2],
            1.0,
        )
    return projected


# ### Geometry helper tests ###
class ObjectFaceSelectionGeometryTests(unittest.TestCase):
    def test_vectorized_projection_matches_qmatrix_map(self) -> None:
        view_projection = QMatrix4x4()
        view_projection.perspective(57.0, 1.7, 0.1, 100.0)
        view_projection.translate(1.2, -0.7, -4.0)
        view_projection.rotate(33.0, 1.0, 2.0, 3.0)
        random = np.random.default_rng(20_260_831)
        vertices = random.uniform(-3.0, 3.0, (256, 3))
        vertices = np.vstack(
            (
                vertices,
                np.asarray(
                    (
                        (0.0, 0.0, 4.0),
                        (0.0, 0.0, 40.0),
                    ),
                    dtype=float,
                ),
            )
        )
        viewport_width = 1_913
        viewport_height = 1_077

        expected = _project_vertices_with_qt_map(
            vertices,
            view_projection,
            viewport_width,
            viewport_height,
        )
        actual = _project_vertices_to_view(
            vertices,
            view_projection,
            viewport_width,
            viewport_height,
        )

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=2e-6,
            atol=2e-4,
            equal_nan=True,
        )

    def test_large_projection_batch_avoids_per_vertex_qt_calls(self) -> None:
        class MatrixSpy:
            def __init__(self) -> None:
                self.data_calls = 0
                self.map_calls = 0
                self._data = QMatrix4x4().data()

            def data(self) -> tuple[float, ...]:
                self.data_calls += 1
                return self._data

            def map(self, _vertex: object) -> object:
                self.map_calls += 1
                raise AssertionError("Projection must not map vertices through Qt.")

        vertex_count = 250_000
        vertices = np.zeros((vertex_count, 3), dtype=np.float32)
        matrix = MatrixSpy()

        projected = _project_vertices_to_view(
            vertices,
            matrix,
            1_920,
            1_080,
        )

        self.assertEqual(projected.shape, (vertex_count, 4))
        self.assertEqual(matrix.data_calls, 1)
        self.assertEqual(matrix.map_calls, 0)
        np.testing.assert_array_equal(
            projected[[0, -1]],
            np.asarray(
                (
                    (960.0, 540.0, 0.0, 1.0),
                    (960.0, 540.0, 0.0, 1.0),
                )
            ),
        )

    def test_rectangle_selection_projects_through_qmatrix_api(self) -> None:
        class IdentityView:
            def width(self) -> int:
                return 100

            def height(self) -> int:
                return 100

            def projectionMatrix(self, *_args: object) -> QMatrix4x4:
                return QMatrix4x4()

            def viewMatrix(self) -> QMatrix4x4:
                return QMatrix4x4()

        vertices = np.asarray(
            ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.0, 0.5, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)

        selected = _select_face_indices_in_view_rectangle(
            IdentityView(),  # type: ignore[arg-type]
            [(vertices, faces)],
            QRect(20, 20, 60, 60),
            xray=False,
        )

        self.assertEqual(selected, {0})

    def test_ray_returns_the_nearest_logical_face_without_uvs(self) -> None:
        vertices = np.asarray(
            (
                (-1.0, -1.0, 0.0),
                (1.0, -1.0, 0.0),
                (0.0, 1.0, 0.0),
                (-1.0, -1.0, -1.0),
                (1.0, -1.0, -1.0),
                (0.0, 1.0, -1.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)

        hit = _get_nearest_triangle_ray_face_index(
            vertices,
            faces,
            (0.0, 0.0, 2.0),
            (0.0, 0.0, -1.0),
        )

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit[0], 0)
        self.assertAlmostEqual(hit[1], 2.0)

    def test_rectangle_depth_buffer_excludes_an_occluded_face(self) -> None:
        projected_vertices = np.asarray(
            (
                (0.0, 10.0, 0.5, 1.0),
                (10.0, 10.0, 0.5, 1.0),
                (5.0, 0.0, 0.5, 1.0),
                (0.0, 10.0, -0.5, 1.0),
                (10.0, 10.0, -0.5, 1.0),
                (5.0, 0.0, -0.5, 1.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
        rectangle = QRect(0, 0, 11, 11)

        visible = _rasterize_face_selection(
            [(projected_vertices, faces)],
            rectangle,
            xray=False,
        )
        xray = _rasterize_face_selection(
            [(projected_vertices, faces)],
            rectangle,
            xray=True,
        )

        self.assertEqual(visible, {1})
        self.assertEqual(xray, {0, 1})

    def test_slanted_overlap_uses_per_pixel_depth(self) -> None:
        projected_vertices = np.asarray(
            (
                (0.0, 10.0, -0.9, 1.0),
                (10.0, 10.0, 0.9, 1.0),
                (5.0, 0.0, 0.9, 1.0),
                (0.0, 10.0, 0.0, 1.0),
                (10.0, 10.0, 0.0, 1.0),
                (5.0, 0.0, 0.0, 1.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)

        visible = _rasterize_face_selection(
            [(projected_vertices, faces)],
            QRect(0, 0, 11, 11),
            xray=False,
        )

        self.assertEqual(visible, {0, 1})

    def test_fully_occluded_small_face_is_not_selected(self) -> None:
        projected_vertices = np.asarray(
            (
                (0.0, 10.0, -0.5, 1.0),
                (10.0, 10.0, -0.5, 1.0),
                (5.0, 0.0, -0.5, 1.0),
                (3.0, 7.0, 0.0, 1.0),
                (7.0, 7.0, 0.0, 1.0),
                (5.0, 3.0, 0.0, 1.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)

        visible = _rasterize_face_selection(
            [(projected_vertices, faces)],
            QRect(0, 0, 11, 11),
            xray=False,
        )

        self.assertEqual(visible, {0})

    def test_bounded_raster_matches_pixel_center_reference(self) -> None:
        random = np.random.default_rng(20_260_830)
        width = 24
        height = 20
        face_count = 8

        for scene_index in range(16):
            coordinates = np.column_stack(
                (
                    random.uniform(-2.0, width + 2.0, face_count * 3),
                    random.uniform(-2.0, height + 2.0, face_count * 3),
                    random.uniform(-0.8, 0.8, face_count * 3),
                    np.ones(face_count * 3),
                )
            )
            faces = np.arange(face_count * 3, dtype=np.int64).reshape(
                (-1, 3)
            )

            with self.subTest(scene_index=scene_index):
                expected = _reference_visible_face_indices(
                    coordinates,
                    faces,
                    width=width,
                    height=height,
                )
                actual = _rasterize_face_selection(
                    [(coordinates, faces)],
                    QRect(0, 0, width, height),
                    xray=False,
                )

                self.assertEqual(actual, expected)

    def test_last_pixel_center_remains_inside_raster_bounds(self) -> None:
        projected_vertices = np.asarray(
            (
                (23.2, 11.0, 0.0, 1.0),
                (23.8, 11.0, 0.0, 1.0),
                (23.5, 10.0, 0.0, 1.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)

        visible = _rasterize_face_selection(
            [(projected_vertices, faces)],
            QRect(0, 0, 24, 20),
            xray=False,
        )

        self.assertEqual(visible, {0})


# ### Widget interaction tests ###
class ObjectFaceSelectionWidgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[object] = []

    def tearDown(self) -> None:
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        _qt_application.processEvents()

    def _build_face_edit_viewer(self) -> GlbViewerWidget:
        vertices = np.asarray(
            (
                (-1.0, -1.0, 0.0),
                (0.0, -1.0, 0.0),
                (-0.5, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (1.0, -1.0, 0.0),
                (0.5, 1.0, 0.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
        viewer = GlbViewerWidget(face_editing_enabled=True)
        self.widgets.append(viewer)
        viewer.set_model(_triangle_model(vertices))
        viewer.set_face_edit_geometry(vertices, faces)
        return viewer

    def test_ctrl_click_is_opt_in_and_ordinary_click_is_untouched(self) -> None:
        view = SelectableGLViewWidget()
        self.widgets.append(view)
        view.resize(200, 160)
        view.show()
        pressed = QSignalSpy(view.face_selection_pointer_pressed)
        released = QSignalSpy(view.face_selection_pointer_released)

        QTest.mouseClick(
            view,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            QPoint(80, 60),
        )
        self.assertEqual(pressed.count(), 0)
        self.assertEqual(released.count(), 0)

        view.set_face_selection_gestures_enabled(True)
        QTest.mouseClick(
            view,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            QPoint(80, 60),
        )
        self.assertEqual(pressed.count(), 1)
        self.assertEqual(released.count(), 1)

    def test_plain_click_is_inert_in_face_editor_and_keeps_navigation_live(
        self,
    ) -> None:
        view = SelectableGLViewWidget()
        self.widgets.append(view)
        view.resize(200, 160)
        view.set_face_selection_gestures_enabled(True)
        view.show()
        clicked_items = QSignalSpy(view.items_clicked)
        viewport_clicked = QSignalSpy(view.viewport_clicked)

        with patch.object(
            view,
            "_get_clicked_items",
            side_effect=AssertionError(
                "Plain face-editor clicks must not run OpenGL item picking."
            ),
        ):
            QTest.mouseClick(
                view,
                Qt.MouseButton.LeftButton,
                pos=QPoint(80, 60),
            )

        self.assertEqual(clicked_items.count(), 0)
        self.assertEqual(viewport_clicked.count(), 0)
        self.assertFalse(view.is_face_selection_gesture_active)
        self.assertFalse(view.is_middle_navigation_active)
        self.assertIsNot(QWidget.mouseGrabber(), view)

        QTest.mousePress(
            view,
            Qt.MouseButton.MiddleButton,
            pos=QPoint(80, 60),
        )

        self.assertTrue(view.is_middle_navigation_active)
        self.assertIs(QWidget.mouseGrabber(), view)

        QTest.mouseRelease(
            view,
            Qt.MouseButton.MiddleButton,
            pos=QPoint(80, 60),
        )

        self.assertFalse(view.is_middle_navigation_active)
        self.assertIsNot(QWidget.mouseGrabber(), view)

    def test_object_viewer_never_uses_item_pick_for_plain_clicks(self) -> None:
        viewer = self._build_face_edit_viewer()
        viewer.resize(320, 240)
        viewer.show()
        viewer.set_face_editing_enabled(False)

        with patch.object(
            viewer.view,
            "_get_clicked_items",
            side_effect=AssertionError(
                "The Object viewer must not run legacy OpenGL item picking."
            ),
        ):
            QTest.mouseClick(
                viewer.view,
                Qt.MouseButton.LeftButton,
                pos=QPoint(80, 60),
            )

        self.assertFalse(viewer.view.is_face_selection_gesture_active)
        self.assertIsNot(QWidget.mouseGrabber(), viewer.view)

    def test_escape_cancel_suppresses_the_pending_left_release(self) -> None:
        view = SelectableGLViewWidget()
        self.widgets.append(view)
        view.resize(200, 160)
        view.set_face_selection_gestures_enabled(True)
        view.show()
        cancelled = QSignalSpy(
            view.face_selection_pointer_cancel_requested
        )
        viewport_clicked = QSignalSpy(view.viewport_clicked)

        QTest.mousePress(
            view,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            QPoint(80, 60),
        )
        QTest.keyClick(view, Qt.Key.Key_Escape)
        QTest.mouseRelease(
            view,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            QPoint(80, 60),
        )

        self.assertEqual(cancelled.count(), 1)
        self.assertEqual(viewport_clicked.count(), 0)

    def test_focus_loss_releases_middle_navigation_and_mouse_grab(self) -> None:
        view = SelectableGLViewWidget()
        self.widgets.append(view)
        view.resize(200, 160)
        view.show()

        QTest.mousePress(
            view,
            Qt.MouseButton.MiddleButton,
            pos=QPoint(80, 60),
        )

        self.assertTrue(view.is_middle_navigation_active)
        self.assertIs(QWidget.mouseGrabber(), view)

        view.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))

        self.assertFalse(view.is_middle_navigation_active)
        self.assertIsNot(QWidget.mouseGrabber(), view)
        QTest.mouseRelease(
            view,
            Qt.MouseButton.MiddleButton,
            pos=QPoint(80, 60),
        )

    def test_model_replacement_cancels_an_interrupted_ctrl_gesture(self) -> None:
        viewer = self._build_face_edit_viewer()
        viewer.resize(320, 240)
        viewer.show()
        _qt_application.processEvents()

        QTest.mousePress(
            viewer.view,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            QPoint(80, 60),
        )

        self.assertTrue(viewer.view.is_face_selection_gesture_active)

        replacement_vertices = np.asarray(
            (
                (-2.0, -1.0, 0.0),
                (0.0, -1.0, 0.0),
                (-1.0, 1.0, 0.0),
            ),
            dtype=float,
        )
        viewer.set_model(_triangle_model(replacement_vertices))

        self.assertFalse(viewer.view.is_face_selection_gesture_active)
        self.assertIsNot(QWidget.mouseGrabber(), viewer.view)

        QTest.mouseRelease(
            viewer.view,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            QPoint(80, 60),
        )

    def test_released_first_person_mode_can_select_without_exiting(self) -> None:
        view = SelectableGLViewWidget()
        self.widgets.append(view)
        view.resize(200, 160)
        view.set_face_selection_gestures_enabled(True)
        view.show()
        selected = QSignalSpy(view.face_selection_pointer_pressed)
        view.enter_first_person_mode()

        QTest.mouseClick(
            view,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            QPoint(80, 60),
        )
        self.assertEqual(selected.count(), 0)

        QTest.mouseClick(
            view,
            Qt.MouseButton.RightButton,
            pos=QPoint(80, 60),
        )
        self.assertTrue(view.is_first_person_active)
        self.assertFalse(view.is_first_person_pointer_captured)
        QTest.mouseClick(
            view,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            QPoint(80, 60),
        )

        self.assertEqual(selected.count(), 1)
        self.assertTrue(view.is_first_person_active)

    def test_rectangle_selection_raster_runs_on_a_daemon_thread(self) -> None:
        viewer = self._build_face_edit_viewer()
        viewer.resize(320, 240)
        viewer.show()
        _qt_application.processEvents()
        main_thread_id = threading.get_ident()
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        observations: dict[str, object] = {}
        delivery_threads: list[int] = []
        viewer.face_selection_changed.connect(
            lambda _indices: delivery_threads.append(threading.get_ident())
        )
        changed = QSignalSpy(viewer.face_selection_changed)

        def blocking_raster(
            projected_geometry: object,
            _rectangle: object,
            *,
            xray: bool,
            cancel_event: threading.Event | None = None,
        ) -> set[int]:
            arrays = tuple(
                array
                for pair in projected_geometry  # type: ignore[union-attr]
                for array in pair
            )
            observations["thread_id"] = threading.get_ident()
            observations["daemon"] = threading.current_thread().daemon
            observations["read_only"] = all(
                not array.flags.writeable for array in arrays
            )
            observations["xray"] = xray
            observations["cancel_event"] = cancel_event is not None
            started.set()
            release.wait(5.0)
            finished.set()
            return {1}

        try:
            with patch(
                "housemaker.viewer._rasterize_face_selection",
                side_effect=blocking_raster,
            ):
                before_start = time.monotonic()
                was_started = (
                    viewer._start_editable_face_rectangle_selection(
                        QPointF(10.0, 10.0),
                        QPointF(180.0, 160.0),
                    )
                )
                start_duration = time.monotonic() - before_start

                self.assertTrue(was_started)
                self.assertLess(start_duration, 1.0)
                self.assertTrue(started.wait(2.0))
                release.set()
                self.assertTrue(_wait_until(lambda: changed.count() == 1))
        finally:
            release.set()

        self.assertTrue(finished.wait(2.0))
        self.assertNotEqual(observations["thread_id"], main_thread_id)
        self.assertTrue(observations["daemon"])
        self.assertTrue(observations["read_only"])
        self.assertFalse(observations["xray"])
        self.assertTrue(observations["cancel_event"])
        self.assertEqual(delivery_threads, [main_thread_id])
        self.assertEqual(viewer.get_selected_face_indices(), (1,))

    def test_stale_rectangle_results_are_rejected_by_both_revisions(
        self,
    ) -> None:
        viewer = self._build_face_edit_viewer()
        changed = QSignalSpy(viewer.face_selection_changed)
        old_request_revision = (
            viewer._face_rectangle_selection_request_revision
        )
        current_geometry_revision = viewer._face_selection_geometry_revision

        viewer._invalidate_face_rectangle_selection_requests()
        viewer._face_rectangle_selection_completed.emit(
            _FaceRectangleSelectionResult(
                request_revision=old_request_revision,
                geometry_revision=current_geometry_revision,
                face_indices=frozenset((0,)),
            )
        )
        QTest.qWait(20)

        self.assertEqual(changed.count(), 0)
        self.assertEqual(viewer.get_selected_face_indices(), ())

        old_geometry_revision = viewer._face_selection_geometry_revision
        assert viewer._face_edit_vertices is not None
        assert viewer._face_edit_faces is not None
        viewer.set_face_edit_geometry(
            viewer._face_edit_vertices,
            viewer._face_edit_faces,
        )
        viewer._face_rectangle_selection_completed.emit(
            _FaceRectangleSelectionResult(
                request_revision=(
                    viewer._face_rectangle_selection_request_revision
                ),
                geometry_revision=old_geometry_revision,
                face_indices=frozenset((1,)),
            )
        )
        QTest.qWait(20)

        self.assertEqual(changed.count(), 0)
        self.assertEqual(viewer.get_selected_face_indices(), ())

    def test_close_cancels_pending_rectangle_result_delivery(self) -> None:
        viewer = self._build_face_edit_viewer()
        viewer.resize(320, 240)
        viewer.show()
        _qt_application.processEvents()
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        changed = QSignalSpy(viewer.face_selection_changed)

        def blocking_raster(
            _projected_geometry: object,
            _rectangle: object,
            *,
            xray: bool,
            cancel_event: threading.Event | None = None,
        ) -> set[int]:
            del xray
            del cancel_event
            started.set()
            release.wait(5.0)
            finished.set()
            return {0}

        try:
            with patch(
                "housemaker.viewer._rasterize_face_selection",
                side_effect=blocking_raster,
            ):
                self.assertTrue(
                    viewer._start_editable_face_rectangle_selection(
                        QPointF(10.0, 10.0),
                        QPointF(180.0, 160.0),
                    )
                )
                self.assertTrue(started.wait(2.0))
                viewer.close()
                release.set()
                self.assertTrue(finished.wait(2.0))
                QTest.qWait(50)
        finally:
            release.set()

        self.assertEqual(changed.count(), 0)
        self.assertEqual(viewer.get_selected_face_indices(), ())

    def test_mirrored_preview_hit_maps_to_the_retained_face(self) -> None:
        vertices = np.asarray(
            (
                (1.0, -1.0, -1.0),
                (1.0, 1.0, -1.0),
                (1.0, 0.0, 1.0),
            ),
            dtype=float,
        )
        viewer = GlbViewerWidget(face_editing_enabled=True)
        self.widgets.append(viewer)
        viewer.set_model(_triangle_model(vertices))
        viewer.set_face_edit_geometry(
            vertices,
            np.asarray(((0, 1, 2),), dtype=np.int64),
        )
        viewer.set_symmetric_division_preview("vertical", 0.0)
        viewer.view.build_camera_ray = lambda _position: (
            np.asarray((-3.0, 0.0, 0.0), dtype=float),
            np.asarray((1.0, 0.0, 0.0), dtype=float),
        )

        self.assertEqual(viewer._pick_editable_face(QPointF()), 0)

        viewer.set_selected_face_indices((0,))
        self.assertIsInstance(
            viewer._face_selection_item,
            _WireframeOverlayMeshItem,
        )
        self.assertIsInstance(
            viewer._mirrored_face_selection_item,
            _WireframeOverlayMeshItem,
        )


if __name__ == "__main__":
    unittest.main()
