"""Logical-pixel, decorated-window work-area fitting without native windows."""

# ### Imports ###
from __future__ import annotations

import pytest
from PySide6.QtCore import QMargins, QPoint, QRect, QSize

from housemaker.window_geometry import fit_normal_window_to_screen


# ### Screen and window doubles ###
class _Screen:
    def __init__(self, work_area: QRect) -> None:
        self.work_area = work_area

    def availableGeometry(self) -> QRect:
        return QRect(self.work_area)


class _Window:
    """Mimic QWidget's client resize and frame-relative move conventions."""

    def __init__(
        self,
        size: QSize | None = None,
        margins: QMargins | None = None,
        *,
        native: bool = True,
        maximized: bool = False,
        full_screen: bool = False,
        minimum: QSize | None = None,
    ) -> None:
        self.client_size = QSize(460, 520) if size is None else QSize(size)
        self.margins = QMargins(12, 52, 12, 12) if margins is None else QMargins(margins)
        self.position = QPoint(20, 40)
        self.native = native
        self.maximized = maximized
        self.full_screen = full_screen
        self.minimum = QSize(1, 1) if minimum is None else QSize(minimum)
        self.requests = []

    def windowHandle(self):
        return self if self.native else None

    def frameMargins(self):
        return QMargins(self.margins)

    def isMaximized(self):
        return self.maximized

    def isFullScreen(self):
        return self.full_screen

    def width(self):
        return self.client_size.width()

    def height(self):
        return self.client_size.height()

    def frameGeometry(self):
        return QRect(
            self.position,
            QSize(
                self.width() + self.margins.left() + self.margins.right(),
                self.height() + self.margins.top() + self.margins.bottom(),
            ),
        )

    def geometry(self):
        return QRect(
            self.position + QPoint(self.margins.left(), self.margins.top()),
            self.client_size,
        )

    def resize(self, width, height):
        self.requests.append(("resize", width, height))
        self.client_size = QSize(width, height).expandedTo(self.minimum)

    def move(self, point):
        self.requests.append(("move", point.x(), point.y()))
        self.position = QPoint(point)


# ### Work-area contracts ###
@pytest.mark.parametrize("native", [True, False])
def test_centering_includes_title_bar_and_borders_on_negative_monitor(native):
    screen = _Screen(QRect(-1920, -2160, 1920, 1080))
    window = _Window(native=native)

    fit_normal_window_to_screen(window, screen)

    assert window.client_size == QSize(460, 520)
    assert window.position == QPoint(-1202, -1912)
    assert screen.work_area.contains(window.frameGeometry())


def test_small_work_area_clamps_client_by_the_actual_frame_margins():
    screen = _Screen(QRect(-1280, -1024, 420, 400))
    window = _Window()

    fit_normal_window_to_screen(window, screen)

    assert window.client_size == QSize(396, 336)
    assert window.frameGeometry() == screen.work_area


def test_right_top_alignment_clamps_offset_when_decorations_fill_height():
    screen = _Screen(QRect(-1000, -2000, 700, 600))
    window = _Window()

    fit_normal_window_to_screen(window, screen, edge_offset=32, align_right=True)

    assert window.position == QPoint(-816, -1984)
    assert screen.work_area.contains(window.frameGeometry())


def test_dpi_frame_margin_change_is_refitted_without_manual_device_scaling():
    screen = _Screen(QRect(-1000, -2000, 460, 520))
    window = _Window(margins=QMargins(8, 31, 8, 8))
    fit_normal_window_to_screen(window, screen)
    assert window.client_size == QSize(444, 481)

    # Windows finishes moving the native window onto the higher-DPI screen.
    window.margins = QMargins(12, 52, 12, 12)
    fit_normal_window_to_screen(window, screen)

    assert window.client_size == QSize(436, 456)
    assert window.frameGeometry() == screen.work_area


def test_already_fitted_window_does_not_repeat_resize_or_move_requests():
    screen = _Screen(QRect(-500, -800, 420, 400))
    window = _Window()
    fit_normal_window_to_screen(window, screen)
    window.requests.clear()

    fit_normal_window_to_screen(window, screen)

    assert not window.requests


@pytest.mark.parametrize("state", ["maximized", "full_screen"])
def test_native_maximized_and_full_screen_geometry_is_not_changed(state):
    window = _Window(**{state: True})

    fit_normal_window_to_screen(window, _Screen(QRect(0, 0, 400, 300)))

    assert not window.requests


def test_explicit_content_minimum_is_not_removed_to_force_a_fit():
    window = _Window(minimum=QSize(500, 600))
    screen = _Screen(QRect(-500, -800, 420, 400))

    fit_normal_window_to_screen(window, screen)

    assert window.minimum == QSize(500, 600)
    assert window.client_size == QSize(500, 600)
    assert window.position == screen.work_area.topLeft()


def test_invalid_work_area_is_rejected():
    with pytest.raises(ValueError, match="work area"):
        fit_normal_window_to_screen(_Window(), _Screen(QRect()))
