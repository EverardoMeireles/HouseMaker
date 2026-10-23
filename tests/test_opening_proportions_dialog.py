# ### Imports and Qt setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent, QTransform, QWheelEvent
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QComboBox

from housemaker.opening_proportions_dialog import OpeningProportionsDialog
from housemaker.opening_proportions_view import APEX_COLOR, OpeningProportionsView

# ### Qt fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def dialog():
    value = OpeningProportionsDialog()
    value.set_target("Window 1", False, 1.2, 1.5)
    yield value
    value.close()
    value.deleteLater()


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "opening.png"
    assert cv2.imwrite(str(path), np.full((200, 300, 3), 190, dtype=np.uint8))
    return path


@pytest.fixture
def video_path(tmp_path):
    path = tmp_path / "opening.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (120, 80)
    )
    if not writer.isOpened():
        pytest.skip("The test video encoder is unavailable.")
    for index in range(5):
        writer.write(np.full((80, 120, 3), 40 * index, dtype=np.uint8))
    writer.release()
    return path


# ### Target and payload tests ###
def test_dialog_has_no_model_or_global_scale_controls(dialog):
    assert not dialog.findChildren(QComboBox)
    assert dialog.windowFlags() & Qt.WindowType.WindowMaximizeButtonHint
    assert not dialog.apply_button.isEnabled()


def test_loading_image_emits_normalized_measurement(dialog, image_path):
    spy = QSignalSpy(dialog.outline_changed)
    dialog.load_source(image_path)
    assert spy.count() == 1
    payload = spy.at(0)[0]
    assert payload["frame_size"] == (300, 200)
    assert len(payload["outline"]) == 4
    assert payload["arch_apex"] is None
    assert payload["frame_index"] == 0
    assert payload["is_doorway"] is False
    assert not dialog.seek_slider.isEnabled()
    for point in payload["outline"]:
        assert all(0 <= coordinate <= 1 for coordinate in point)


def test_apply_is_explicit_and_requires_valid_result(dialog, image_path):
    spy = QSignalSpy(dialog.apply_requested)
    dialog.load_source(image_path)
    dialog.apply_button.click()
    assert spy.count() == 0
    dialog.set_result_text("Height: 1.60 m")
    dialog.apply_button.click()
    assert spy.count() == 1
    assert spy.at(0)[0] == dialog.current_measurement()


def test_error_disables_apply(dialog, image_path):
    dialog.load_source(image_path)
    dialog.set_result_text("Height: 1.60 m")
    assert dialog.apply_button.isEnabled()
    dialog.show_error("The outline is crossed.")
    assert not dialog.apply_button.isEnabled()
    assert dialog.result_label.text() == "The outline is crossed."


def test_doorway_has_fifth_apex_above_springline(dialog, image_path):
    dialog.set_target("Arched doorway", True, 1.0, 2.4, 0.4)
    dialog.load_source(image_path)
    payload = dialog.current_measurement()
    left, right = payload["outline"][:2]
    apex = payload["arch_apex"]
    assert apex is not None
    assert apex[0] == pytest.approx((left[0] + right[0]) * 0.5)
    assert apex[1] < left[1]
    assert payload["is_doorway"] is True


def test_rectangular_doorway_apex_starts_on_top_edge(dialog, image_path):
    dialog.set_target("Doorway", True, 1.0, 2.2)
    dialog.load_source(image_path)
    measurement = dialog.current_measurement()
    assert measurement["arch_apex"][1] == measurement["outline"][0][1]


def test_retargeting_doorway_as_window_removes_apex(dialog, image_path):
    dialog.set_target("Doorway", True, 1.0, 2.2)
    dialog.load_source(image_path)
    dialog.set_target("Window", False, 1.1, 1.4)
    assert dialog.current_measurement()["arch_apex"] is None


def test_failed_image_load_keeps_old_frame(dialog, image_path, tmp_path):
    dialog.load_source(image_path)
    original = dialog.current_measurement()
    with pytest.raises((ValueError, OSError)):
        dialog.load_source(tmp_path / "missing.png")
    assert dialog.current_measurement() == original


def test_empty_image_reports_a_readable_error(dialog, tmp_path):
    path = tmp_path / "empty.png"
    path.touch()
    with pytest.raises(ValueError, match="could not be opened"):
        dialog.load_source(path)


# ### Video navigation and reader lifetime ###
def test_video_load_seeks_requested_frame_and_clamps(dialog, video_path):
    dialog.load_source(video_path, 1000)
    assert dialog.current_measurement()["frame_index"] == 4
    assert dialog.seek_slider.maximum() == 4
    assert dialog.seek_slider.isEnabled()


def test_arrow_keys_step_two_video_frames(dialog, video_path):
    dialog.load_source(video_path, 2)
    dialog.show()
    _application.processEvents()
    QTest.keyClick(dialog, Qt.Key.Key_Right)
    assert dialog.current_measurement()["frame_index"] == 4
    QTest.keyClick(dialog, Qt.Key.Key_Left)
    assert dialog.current_measurement()["frame_index"] == 2


def test_arrow_keys_step_two_frames_with_focused_seekbar(dialog, video_path):
    dialog.load_source(video_path)
    dialog.show()
    dialog.seek_slider.setFocus()
    _application.processEvents()
    QTest.keyClick(dialog.seek_slider, Qt.Key.Key_Right)
    assert dialog.current_measurement()["frame_index"] == 2
    QTest.keyClick(dialog.seek_slider, Qt.Key.Key_Left)
    assert dialog.current_measurement()["frame_index"] == 0


def test_direct_key_handler_steps_two_frames(dialog, video_path):
    dialog.load_source(video_path)
    event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier
    )
    dialog.keyPressEvent(event)
    assert dialog.current_measurement()["frame_index"] == 2


def test_video_frames_keep_independent_outline_drafts(dialog, video_path):
    dialog.load_source(video_path)
    dialog.video_view._move_handle(0, (0.1, 0.15))
    expected = dialog.current_measurement()["outline"]
    dialog.step_frame(1)
    assert dialog.current_measurement()["outline"] != expected
    dialog.step_frame(-1)
    assert dialog.current_measurement()["outline"] == expected


def test_close_releases_video_reader(dialog, video_path):
    dialog.load_source(video_path)
    source = dialog._source
    assert source is not None
    dialog.close()
    assert dialog._source is None
    assert not source._capture.isOpened()


def test_reloading_video_closes_previous_reader(dialog, video_path, image_path):
    dialog.load_source(video_path)
    source = dialog._source
    dialog.load_source(image_path)
    assert not source._capture.isOpened()
    assert dialog._source is None


# ### Outline interaction tests ###
def test_corner_drag_updates_source_coordinates(dialog, image_path):
    dialog.load_source(image_path)
    dialog.show()
    _application.processEvents()
    view = dialog.video_view
    spy = QSignalSpy(dialog.outline_changed)
    origin = view.image_to_widget(view.outline[0])
    target = origin + QPointF(20, 10)
    expected = view.widget_to_image(target)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=origin.toPoint())
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        target,
        target,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(view, move)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=target.toPoint())
    assert view.outline[0] == pytest.approx(expected)
    assert spy.count() >= 1


def test_edit_after_preview_disables_apply_until_recalculated(dialog, image_path):
    dialog.load_source(image_path)
    dialog.set_result_text("Height: 1.60 m")
    dialog.video_view._move_handle(1, (0.8, 0.2))
    assert not dialog.apply_button.isEnabled()


def test_apex_drag_changes_arch_without_moving_corners(dialog, image_path):
    dialog.set_target("Doorway", True, 1.0, 2.0)
    dialog.load_source(image_path)
    original = dialog.video_view.outline
    dialog.video_view._move_handle(4, (0.5, 0.1))
    assert dialog.video_view.outline == original
    assert dialog.video_view.arch_apex == (0.5, 0.1)


def test_top_corner_drag_carries_apex_along_with_springline(dialog, image_path):
    dialog.set_target("Doorway", True, 1.0, 2.0, 0.2)
    dialog.load_source(image_path)
    original_apex = dialog.video_view.arch_apex
    left = dialog.video_view.outline[0]
    dialog.video_view._move_handle(0, (left[0] + 0.1, left[1] + 0.1))
    assert dialog.video_view.arch_apex == pytest.approx(
        (original_apex[0] + 0.05, original_apex[1] + 0.05)
    )


def test_zoom_keeps_source_point_under_mouse(dialog, image_path):
    dialog.load_source(image_path)
    dialog.resize(900, 700)
    dialog.show()
    _application.processEvents()
    view = dialog.video_view
    anchor = view.image_to_widget((0.6, 0.4))
    before = view.widget_to_image(anchor)
    event = QWheelEvent(
        anchor,
        view.mapToGlobal(anchor.toPoint()),
        QPointF().toPoint(),
        QPointF(0, 120).toPoint(),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(view, event)
    assert view._zoom > 1
    assert view.widget_to_image(anchor) == pytest.approx(before)


def test_mapping_roundtrip_after_zoom_and_pan(dialog, image_path):
    dialog.load_source(image_path)
    view = dialog.video_view
    view.resize(800, 600)
    view._zoom = 3.5
    view._pan = QPointF(55, -27)
    point = (0.23, 0.64)
    assert view.widget_to_image(view.image_to_widget(point)) == pytest.approx(point)


def test_view_rejects_invalid_geometry_points():
    view = OpeningProportionsView()
    with pytest.raises(ValueError, match="four corners"):
        view.set_outline(((0, 0), (1, 1)))
    with pytest.raises(ValueError, match="finite"):
        view.set_outline(((float("nan"), 0), (1, 0), (1, 1), (0, 1)))
    view.deleteLater()


# ### Reusable target state and preview intent ###
def test_window_and_doorway_share_title_instructions_and_layout(dialog, image_path):
    dialog.load_source(image_path)
    title = dialog.windowTitle()
    instructions = dialog.instructions_label.text()
    controls = dialog.layout().count()
    dialog.set_target("Doorway", True, 1.0, 2.0)
    assert dialog.windowTitle() == title == "Opening proportions"
    assert dialog.instructions_label.text() == instructions
    assert dialog.layout().count() == controls
    assert APEX_COLOR.name() == "#ff80bf"


def test_defaults_are_not_marked_as_manual_edits(dialog, video_path):
    spy = QSignalSpy(dialog.outline_changed)
    dialog.load_source(video_path)
    assert not spy.at(spy.count() - 1)[0]["outline_edited"]
    dialog.step_frame(1)
    assert not spy.at(spy.count() - 1)[0]["outline_edited"]


def test_corner_and_apex_moves_are_marked_as_manual_edits(dialog, image_path):
    dialog.set_target("Doorway", True, 1.0, 2.0)
    dialog.load_source(image_path)
    assert not dialog.current_measurement()["outline_edited"]
    dialog.video_view._move_handle(4, (0.5, 0.1))
    assert dialog.current_measurement()["outline_edited"]
    dialog.set_target("Window", False, 1.0, 1.2)
    assert not dialog.current_measurement()["outline_edited"]
    dialog.video_view._move_handle(0, (0.2, 0.2))
    assert dialog.current_measurement()["outline_edited"]


def test_handle_move_without_change_is_not_a_manual_edit(dialog, image_path):
    dialog.load_source(image_path)
    dialog.video_view._move_handle(0, dialog.video_view.outline[0])
    assert not dialog.current_measurement()["outline_edited"]


def test_retarget_keeps_video_reader_current_frame_and_annotations(dialog, video_path):
    window_key = ("window", 0, "one")
    door_key = ("doorway", 0, "two")
    dialog.set_target("Window", False, 1.0, 1.2, target_key=window_key)
    dialog.load_source(video_path, 2)
    source = dialog._source
    dialog.video_view._move_handle(0, (0.15, 0.25))
    window_outline = dialog.video_view.outline
    dialog.set_target("Doorway", True, 0.9, 2.1, target_key=door_key)
    assert dialog._source is source
    assert dialog.current_measurement()["frame_index"] == 2
    assert not dialog.current_measurement()["outline_edited"]
    assert dialog.current_measurement()["arch_apex"] is not None
    dialog.video_view._move_handle(4, (0.5, 0.08))
    door_apex = dialog.video_view.arch_apex
    dialog.set_target("Window", False, 1.0, 1.2, target_key=window_key)
    assert dialog.video_view.outline == window_outline
    assert dialog.current_measurement()["outline_edited"]
    assert dialog.current_measurement()["arch_apex"] is None
    dialog.set_target("Doorway", True, 0.9, 2.1, target_key=door_key)
    assert dialog.video_view.arch_apex == door_apex
    assert dialog.current_measurement()["outline_edited"]
    assert dialog._source is source


def test_manual_edit_flag_is_individual_to_frame(dialog, video_path):
    dialog.load_source(video_path)
    dialog.video_view._move_handle(0, (0.15, 0.25))
    assert dialog.current_measurement()["outline_edited"]
    dialog.step_frame(1)
    assert not dialog.current_measurement()["outline_edited"]
    dialog.step_frame(-1)
    assert dialog.current_measurement()["outline_edited"]


def test_target_annotations_are_individual_to_loaded_source(
    dialog, video_path, image_path
):
    dialog.set_target("Window", False, 1.0, 1.2, target_key=("window", 0, "one"))
    dialog.load_source(video_path)
    dialog.video_view._move_handle(0, (0.15, 0.25))
    outline = dialog.video_view.outline
    dialog.load_source(image_path)
    assert not dialog.current_measurement()["outline_edited"]
    dialog.load_source(video_path)
    assert dialog.video_view.outline == outline
    assert dialog.current_measurement()["outline_edited"]


def test_status_messages_do_not_change_status_area_or_video_size(dialog, image_path):
    dialog.load_source(image_path)
    dialog.show()
    _application.processEvents()
    initial_height = dialog.result_label.height()
    video_size = dialog.video_view.size()
    dialog.show_error("A long validation message. " * 20)
    _application.processEvents()
    assert dialog.result_label.height() == initial_height
    assert dialog.video_view.size() == video_size
    assert dialog.result_label.toolTip() == dialog.result_label.text()
    dialog.set_result_text("Ready to apply.")
    _application.processEvents()
    assert dialog.result_label.height() == initial_height
    assert dialog.video_view.size() == video_size


# ### Elliptical arch preview tests ###
def test_arch_preview_is_ellipse_not_quadratic_bezier(dialog, image_path):
    dialog.set_target("Doorway", True, 1.0, 2.0)
    dialog.load_source(image_path)
    view = dialog.video_view
    view.set_outline(((0.2, 0.4), (0.8, 0.4), (0.8, 0.9), (0.2, 0.9)), (0.5, 0.1))
    points = view._arch_curve_points()
    assert points[0] == pytest.approx((0.2, 0.4))
    assert points[-1] == pytest.approx((0.8, 0.4))
    assert points[16] == pytest.approx((0.5, 0.1))
    for x, y in points:
        assert ((x - 0.5) / 0.3) ** 2 + ((y - 0.4) / 0.3) ** 2 == pytest.approx(1.0)


def test_arch_preview_follows_oblique_outline_homography(dialog, image_path):
    dialog.set_target("Doorway", True, 1.0, 2.0)
    dialog.load_source(image_path)
    view = dialog.video_view
    outline = ((0.25, 0.36), (0.72, 0.42), (0.81, 0.89), (0.18, 0.82))
    transform = QTransform()
    assert QTransform.quadToQuad(
        [QPointF(0, 0), QPointF(1, 0), QPointF(1, 1), QPointF(0, 1)],
        [QPointF(*point) for point in outline],
        transform,
    )
    apex = transform.map(QPointF(0.5, -0.3))
    view.set_outline(outline, (apex.x(), apex.y()))
    inverse, _valid = transform.inverted()
    points = view._arch_curve_points()
    assert points[0] == pytest.approx(outline[0])
    assert points[-1] == pytest.approx(outline[1])
    assert points[16] == pytest.approx((apex.x(), apex.y()))
    for point in points:
        canonical = inverse.map(QPointF(*point))
        assert ((canonical.x() - 0.5) / 0.5) ** 2 + (
            canonical.y() / 0.3
        ) ** 2 == pytest.approx(1.0)


def test_arch_preview_survives_temporarily_flat_outline(dialog, image_path):
    dialog.set_target("Doorway", True, 1.0, 2.0)
    dialog.load_source(image_path)
    view = dialog.video_view
    view.set_outline(((0.2, 0.4), (0.8, 0.4), (0.8, 0.4), (0.2, 0.4)), (0.5, 0.1))
    assert view._arch_curve_points() == ((0.2, 0.4), (0.5, 0.1), (0.8, 0.4))
