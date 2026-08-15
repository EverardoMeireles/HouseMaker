# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QGuiApplication, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QSplitter,
    QTabWidget,
    QWidget,
)

from housemaker.external_viewer_host import ExternalFullscreenViewerHost
from housemaker.viewer import GlbViewerWidget


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Test cases ###
class ExternalFullscreenViewerHostTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hosts: list[ExternalFullscreenViewerHost] = []
        self.widgets: list[QWidget] = []

    def tearDown(self) -> None:
        for host in self.hosts:
            host.dispose()
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        _qt_application.processEvents()

    def test_splitter_viewer_returns_to_its_original_index(self) -> None:
        container = self._build_widget()
        splitter = QSplitter(container)
        left = QWidget(splitter)
        viewer = QWidget(splitter)
        right = QWidget(splitter)
        splitter.addWidget(left)
        splitter.addWidget(viewer)
        splitter.addWidget(right)
        host = self._build_host()

        host.show_on_screen(viewer, _primary_screen())

        self.assertTrue(host.is_active)
        self.assertIs(host.viewer, viewer)
        self.assertIs(viewer.parentWidget(), host.window)
        self.assertEqual(splitter.count(), 3)
        self.assertIsNot(splitter.widget(1), viewer)

        host.restore()

        self.assertFalse(host.is_active)
        self.assertIsNone(host.viewer)
        self.assertIs(splitter.widget(1), viewer)
        self.assertIs(viewer.parentWidget(), splitter)

    def test_layout_viewer_returns_to_its_original_index(self) -> None:
        container = self._build_widget()
        layout = QHBoxLayout(container)
        first = QWidget(container)
        viewer = QWidget(container)
        last = QWidget(container)
        layout.addWidget(first)
        layout.addWidget(viewer)
        layout.addWidget(last)
        host = self._build_host()

        host.show_on_screen(viewer, _primary_screen())

        self.assertEqual(layout.indexOf(viewer), -1)
        self.assertEqual(layout.count(), 3)

        host.restore()

        self.assertEqual(layout.indexOf(viewer), 1)
        self.assertIs(viewer.parentWidget(), container)

    def test_closing_external_window_restores_viewer_and_emits_request(self) -> None:
        container = self._build_widget()
        splitter = QSplitter(container)
        viewer = QWidget(splitter)
        splitter.addWidget(QWidget(splitter))
        splitter.addWidget(viewer)
        host = self._build_host()
        close_requests: list[bool] = []
        restored_viewers: list[QWidget] = []
        host.close_requested.connect(lambda: close_requests.append(True))
        host.viewer_restored.connect(restored_viewers.append)

        host.show_on_screen(viewer, _primary_screen())
        host.window.close()
        _qt_application.processEvents()

        self.assertEqual(close_requests, [True])
        self.assertEqual(restored_viewers, [viewer])
        self.assertFalse(host.is_active)
        self.assertIs(splitter.widget(1), viewer)

    def test_reselecting_a_screen_keeps_the_existing_detached_viewer(self) -> None:
        container = self._build_widget()
        layout = QHBoxLayout(container)
        viewer = QWidget(container)
        layout.addWidget(viewer)
        host = self._build_host()
        screen = _primary_screen()

        host.show_on_screen(viewer, screen)
        host.show_on_screen(viewer, screen)

        self.assertTrue(host.is_active)
        self.assertIs(host.viewer, viewer)
        self.assertIs(host.screen, screen)
        self.assertTrue(
            bool(host.window.windowState() & Qt.WindowState.WindowFullScreen)
        )

    def test_hidden_tab_viewer_is_shown_on_the_external_display(self) -> None:
        container = self._build_widget()
        layout = QHBoxLayout(container)
        tabs = QTabWidget(container)
        canvas_view = QWidget(tabs)
        viewer = QWidget(tabs)
        tabs.addTab(canvas_view, "2D view")
        tabs.addTab(viewer, "3D view")
        layout.addWidget(tabs)
        container.resize(800, 600)
        container.show()
        _qt_application.processEvents()
        host = self._build_host()

        self.assertFalse(viewer.isVisible())
        self.assertTrue(viewer.isHidden())

        host.show_on_screen(viewer, _primary_screen())
        _qt_application.processEvents()

        self.assertTrue(host.is_active)
        self.assertTrue(viewer.isVisible())
        self.assertFalse(viewer.isHidden())

        host.restore()
        _qt_application.processEvents()

        self.assertIs(tabs.widget(1), viewer)
        self.assertFalse(viewer.isVisible())
        self.assertTrue(viewer.isHidden())

    def test_invalid_screen_is_rejected_without_detaching_the_viewer(self) -> None:
        container = self._build_widget()
        layout = QHBoxLayout(container)
        viewer = QWidget(container)
        layout.addWidget(viewer)
        host = self._build_host()

        with self.assertRaises(TypeError):
            host.show_on_screen(viewer, None)  # type: ignore[arg-type]

        self.assertFalse(host.is_active)
        self.assertIs(viewer.parentWidget(), container)
        self.assertEqual(layout.indexOf(viewer), 0)

    def _build_host(self) -> ExternalFullscreenViewerHost:
        host = ExternalFullscreenViewerHost()
        self.hosts.append(host)
        return host

    def _build_widget(self) -> QWidget:
        widget = QWidget()
        self.widgets.append(widget)
        return widget


# ### Test helpers ###
def _primary_screen():
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        raise RuntimeError("A QScreen is required for the external viewer tests.")
    return screen


# ### Detached viewer input tests ###
class ExternalFullscreenViewerInputTests(unittest.TestCase):
    """Exercise navigation events after a GL viewer has been reparented."""

    def setUp(self) -> None:
        self.hosts: list[ExternalFullscreenViewerHost] = []
        self.widgets: list[QWidget] = []
        self.viewers: list[GlbViewerWidget] = []

    def tearDown(self) -> None:
        for host in self.hosts:
            host.dispose()
        for viewer in self.viewers:
            viewer.close()
            viewer.deleteLater()
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        _qt_application.processEvents()

    def test_detached_viewer_middle_drag_orbits_with_qtest_events(self) -> None:
        viewer, host = self._build_detached_viewer()
        orbit = Mock()
        pan = Mock()
        viewer.view.orbit = orbit
        viewer.view.pan = pan
        start = QPoint(100, 100)
        end = QPoint(130, 125)

        QTest.mousePress(
            viewer.view,
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.NoModifier,
            start,
        )
        QTest.mouseMove(viewer.view, end)
        QTest.mouseRelease(
            viewer.view,
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.NoModifier,
            end,
        )
        _qt_application.processEvents()

        self.assertTrue(host.is_active)
        orbit.assert_called_once_with(-30.0, 25.0)
        self.assertFalse(pan.called)

    def test_detached_viewer_shift_middle_drag_pans_with_modifier_event(self) -> None:
        viewer, host = self._build_detached_viewer()
        orbit = Mock()
        pan = Mock()
        viewer.view.orbit = orbit
        viewer.view.pan = pan
        start = QPoint(100, 100)
        end = QPoint(130, 125)

        QTest.mousePress(
            viewer.view,
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.ShiftModifier,
            start,
        )
        _send_mouse_move(
            viewer.view,
            end,
            buttons=Qt.MouseButton.MiddleButton,
            modifiers=Qt.KeyboardModifier.ShiftModifier,
        )
        QTest.mouseRelease(
            viewer.view,
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.ShiftModifier,
            end,
        )
        _qt_application.processEvents()

        self.assertTrue(host.is_active)
        pan.assert_called_once_with(30.0, 25.0, 0.0, relative="view")
        self.assertFalse(orbit.called)

    def test_detached_viewer_wheel_zooms_with_qtest_window_event(self) -> None:
        viewer, host = self._build_detached_viewer()
        window_handle = host.window.windowHandle()
        self.assertIsNotNone(window_handle)
        assert window_handle is not None
        original_distance = float(viewer.view.opts["distance"])
        view_position = viewer.view.mapTo(host.window, QPoint(100, 100))

        QTest.wheelEvent(
            window_handle,
            view_position,
            QPoint(0, 120),
        )
        _qt_application.processEvents()

        self.assertTrue(host.is_active)
        self.assertLess(float(viewer.view.opts["distance"]), original_distance)

    def _build_detached_viewer(
        self,
    ) -> tuple[GlbViewerWidget, ExternalFullscreenViewerHost]:
        container = QWidget()
        layout = QHBoxLayout(container)
        viewer = GlbViewerWidget()
        layout.addWidget(viewer)
        container.resize(800, 600)
        container.show()
        _qt_application.processEvents()

        host = ExternalFullscreenViewerHost()
        host.show_on_screen(viewer, _primary_screen())
        _qt_application.processEvents()

        self.widgets.append(container)
        self.viewers.append(viewer)
        self.hosts.append(host)
        self.assertTrue(viewer.view.isVisible())
        return viewer, host


# ### Detached viewer input helpers ###
def _send_mouse_move(
    widget: QWidget,
    position: QPoint,
    *,
    buttons: Qt.MouseButton,
    modifiers: Qt.KeyboardModifier,
) -> None:
    """Deliver a move event retaining drag buttons and keyboard modifiers.

    QTest.mouseMove intentionally has no buttons/modifiers parameters, so use
    Qt's standard mouse event dispatch for the modifier-bearing drag segment.
    """

    local_position = QPointF(position)
    global_position = QPointF(widget.mapToGlobal(position))
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        local_position,
        global_position,
        Qt.MouseButton.NoButton,
        buttons,
        modifiers,
    )
    QApplication.sendEvent(widget, event)
