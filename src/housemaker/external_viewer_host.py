# ### Imports ###
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QCloseEvent, QScreen
from PySide6.QtWidgets import QLayout, QSplitter, QVBoxLayout, QWidget


# ### Placement models ###
@dataclass
class _ViewerPlacement:
    """The original container location of a temporarily detached viewer."""

    parent: QWidget | None
    placeholder: QWidget | None
    layout: QLayout | None = None
    splitter: QSplitter | None = None
    index: int = -1
    was_visible: bool = False


# ### Fullscreen window ###
class _ExternalFullscreenViewerWindow(QWidget):
    """Borderless top-level window that asks its host to restore the viewer."""

    close_requested = Signal()

    def __init__(self) -> None:
        super().__init__(None)
        self.setObjectName("external-fullscreen-viewer-window")
        self.setWindowTitle("HouseMaker 3D Viewer")

        self.viewer_layout = QVBoxLayout(self)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer_layout.setSpacing(0)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self.close_requested.emit()
        event.accept()


# ### External viewer host ###
class ExternalFullscreenViewerHost(QObject):
    """Move one viewer into a selected screen and safely restore its placement.

    The host uses a lightweight placeholder while the supplied widget is shown
    externally. That keeps a split layout stable and returns the viewer to its
    exact prior position when the external window closes.
    """

    close_requested = Signal()
    viewer_attached = Signal(object, object)
    viewer_restored = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._window = _ExternalFullscreenViewerWindow()
        self._window.close_requested.connect(self._handle_window_close_request)
        self._viewer: QWidget | None = None
        self._placement: _ViewerPlacement | None = None
        self._screen: QScreen | None = None
        self._is_restoring = False

    @property
    def window(self) -> QWidget:
        """The external top-level window, primarily for integration and tests."""

        return self._window

    @property
    def viewer(self) -> QWidget | None:
        """The currently detached viewer, if any."""

        return self._viewer

    @property
    def screen(self) -> QScreen | None:
        """The screen currently hosting the detached viewer, if any."""

        return self._screen

    @property
    def is_active(self) -> bool:
        """Whether a viewer is currently hosted in the external window."""

        return self._viewer is not None and self._placement is not None

    def show_on_screen(self, viewer: QWidget, screen: QScreen) -> None:
        """Detach *viewer* and display it fullscreen on *screen*.

        Calling this again for the active viewer only moves the external window
        to the requested screen. A different active viewer is restored before
        the new one is detached.
        """

        _validate_viewer_and_screen(viewer, screen)

        if viewer is self._viewer and self.is_active:
            self._screen = screen
            self._show_window_on_screen(screen)
            _show_and_repaint_detached_viewer(viewer)
            return

        if self.is_active:
            self.restore()

        placement = _replace_viewer_with_placeholder(viewer)
        self._viewer = viewer
        self._placement = placement
        self._screen = screen

        try:
            self._window.viewer_layout.addWidget(viewer)
            self._show_window_on_screen(screen)
            _show_and_repaint_detached_viewer(viewer)
        except Exception:
            self.restore()
            raise

        self.viewer_attached.emit(viewer, screen)

    def restore(self) -> None:
        """Return the detached viewer to its original parent and layout slot."""

        if not self.is_active or self._is_restoring:
            return

        assert self._viewer is not None
        assert self._placement is not None
        viewer = self._viewer
        placement = self._placement
        self._is_restoring = True
        try:
            self._window.viewer_layout.removeWidget(viewer)
            _restore_viewer_placement(viewer, placement)
            self._window.hide()
            self._viewer = None
            self._placement = None
            self._screen = None
        finally:
            self._is_restoring = False

        self.viewer_restored.emit(viewer)

    def dispose(self) -> None:
        """Restore any viewer and release the reusable external window."""

        self.restore()
        self._window.hide()
        self._window.deleteLater()

    def _show_window_on_screen(self, screen: QScreen) -> None:
        self._window.setScreen(screen)
        self._window.setGeometry(screen.geometry())
        self._window.showFullScreen()
        self._window.raise_()
        self._window.activateWindow()

    def _handle_window_close_request(self) -> None:
        was_active = self.is_active
        self.restore()
        if was_active:
            self.close_requested.emit()


# ### Placement helpers ###
def _replace_viewer_with_placeholder(viewer: QWidget) -> _ViewerPlacement:
    parent = viewer.parentWidget()
    placeholder = _create_viewer_placeholder(viewer)
    was_visible = viewer.isVisible()

    if isinstance(parent, QSplitter):
        index = parent.indexOf(viewer)
        if index >= 0:
            parent.replaceWidget(index, placeholder)
            viewer.setParent(None)
            return _ViewerPlacement(
                parent=parent,
                placeholder=placeholder,
                splitter=parent,
                index=index,
                was_visible=was_visible,
            )

    layout = _find_containing_layout(viewer, parent)
    if layout is not None:
        index = layout.indexOf(viewer)
        if index >= 0:
            replaced_item = layout.replaceWidget(viewer, placeholder)
            del replaced_item
            viewer.setParent(None)
            return _ViewerPlacement(
                parent=parent,
                placeholder=placeholder,
                layout=layout,
                index=index,
                was_visible=was_visible,
            )

    return _ViewerPlacement(parent=parent, placeholder=None, was_visible=was_visible)


def _restore_viewer_placement(viewer: QWidget, placement: _ViewerPlacement) -> None:
    placeholder = placement.placeholder
    if placement.splitter is not None and placeholder is not None:
        splitter_index = placement.splitter.indexOf(placeholder)
        if splitter_index >= 0:
            placement.splitter.replaceWidget(splitter_index, viewer)
            _dispose_placeholder(placeholder)
            _restore_visibility(viewer, placement.was_visible)
            return

        placement.splitter.insertWidget(placement.index, viewer)
        _dispose_placeholder(placeholder)
        _restore_visibility(viewer, placement.was_visible)
        return

    if placement.layout is not None and placeholder is not None:
        placeholder_index = placement.layout.indexOf(placeholder)
        if placeholder_index >= 0:
            replaced_item = placement.layout.replaceWidget(placeholder, viewer)
            del replaced_item
        else:
            placement.layout.addWidget(viewer)
        _dispose_placeholder(placeholder)
        _restore_visibility(viewer, placement.was_visible)
        return

    viewer.setParent(placement.parent)
    _restore_visibility(viewer, placement.was_visible)


def _create_viewer_placeholder(viewer: QWidget) -> QWidget:
    placeholder = QWidget()
    placeholder.setObjectName("external-viewer-placeholder")
    placeholder.setMinimumSize(viewer.minimumSize())
    placeholder.setMaximumSize(viewer.maximumSize())
    placeholder.setSizePolicy(viewer.sizePolicy())
    return placeholder


def _find_containing_layout(
    viewer: QWidget,
    parent: QWidget | None,
) -> QLayout | None:
    if parent is None:
        return None
    layout = parent.layout()
    if layout is not None and layout.indexOf(viewer) >= 0:
        return layout
    return None


def _dispose_placeholder(placeholder: QWidget) -> None:
    placeholder.hide()
    placeholder.setParent(None)
    placeholder.deleteLater()


def _restore_visibility(viewer: QWidget, was_visible: bool) -> None:
    viewer.setVisible(was_visible)


def _show_and_repaint_detached_viewer(viewer: QWidget) -> None:
    """Make a detached viewer visible after its new top-level window is shown.

    Qt preserves a child's explicit hidden state when it is reparented.  That
    matters for the Canvas 3D view: its local tab is intentionally hidden
    before being moved to an external display.  Showing the fullscreen window
    alone would therefore leave its child blank.  ``update`` also asks Qt to
    repaint OpenGL-backed viewers after their top-level surface changes.
    """

    viewer.show()
    _focus_detached_viewer_navigation(viewer)
    viewer.update()


def _focus_detached_viewer_navigation(viewer: QWidget) -> None:
    """Give a hosted viewport its preferred input target, when provided."""

    focus_navigation = getattr(viewer, "focus_navigation", None)
    if callable(focus_navigation):
        focus_navigation()


def _validate_viewer_and_screen(viewer: QWidget, screen: QScreen) -> None:
    if not isinstance(viewer, QWidget):
        raise TypeError("viewer must be a QWidget.")
    if not isinstance(screen, QScreen):
        raise TypeError("screen must be a QScreen.")
