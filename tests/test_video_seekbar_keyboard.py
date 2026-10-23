# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QSlider

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.ceiling_height_dialog import CeilingHeightAnalysisDialog
from housemaker.generation_jobs import GenerationJobManager
from housemaker.generation_workspace import GenerationWorkspace
from housemaker.merged_generation_workspace import MergedGenerationWorkspace
from housemaker.surface_texture_workspace import SurfaceTextureGenerationWorkspace

# ### Application and video fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    assert writer.isOpened()
    try:
        for frame_index in range(11):
            writer.write(np.full((48, 64, 3), frame_index * 20, np.uint8))
    finally:
        writer.release()


@pytest.fixture(scope="module", params=("objects", "surfaces", "merged", "ceiling"))
def seek_slider(request, tmp_path_factory) -> Iterator[QSlider]:
    root = tmp_path_factory.mktemp(f"video-seek-{request.param}")
    path = root / "frames.avi"
    _write_video(path)
    settings = ApplicationSettingsStore(root / "settings.json")
    workspaces = []
    if request.param == "ceiling":
        workspace = CeilingHeightAnalysisDialog(application_settings=settings)
        workspace.set_video_path(path)
        slider = workspace.seek_slider
        workspaces.append(workspace)
    elif request.param == "surfaces":
        workspace = SurfaceTextureGenerationWorkspace(
            asset_directory=root / "surfaces", application_settings=settings
        )
        workspace.load_video(str(path))
        slider = workspace.seekbar
        workspaces.append(workspace)
    else:
        jobs = GenerationJobManager()
        objects = GenerationWorkspace(
            asset_directory=root / "objects", job_manager=jobs
        )
        objects.load_video(str(path))
        workspace = objects
        workspaces.append(objects)
        if request.param == "merged":
            surfaces = SurfaceTextureGenerationWorkspace(
                asset_directory=root / "surfaces",
                application_settings=settings,
                job_manager=jobs,
                shared_controls=objects.get_shared_controls(),
            )
            surfaces.load_video(str(path))
            workspace = MergedGenerationWorkspace(
                surfaces, objects, jobs, application_settings=settings
            )
            workspace.sync_shared_controls()
            workspaces.extend((surfaces, workspace))
            assert workspace.seekbar is objects.seekbar is surfaces.seekbar
        slider = workspace.seekbar
    try:
        assert slider.isEnabled()
        yield slider
    finally:
        for workspace in reversed(workspaces):
            if isinstance(workspace, CeilingHeightAnalysisDialog):
                workspace.dispose()
            else:
                workspace.shutdown()
            workspace.close()
        _application.processEvents()


# ### Native seek-slider keyboard behavior ###
@pytest.mark.parametrize(
    ("key", "expected"),
    (
        (Qt.Key.Key_Right, 7),
        (Qt.Key.Key_Left, 3),
        (Qt.Key.Key_Up, 7),
        (Qt.Key.Key_Down, 3),
    ),
)
def test_video_arrows_seek_two_frames(seek_slider: QSlider, key, expected: int) -> None:
    seek_slider.setValue(5)
    QTest.keyClick(seek_slider, key)
    assert seek_slider.value() == expected


def test_held_video_arrow_keeps_two_frame_step(seek_slider: QSlider) -> None:
    seek_slider.setValue(3)
    for expected in (5, 7, 9):
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Right,
            Qt.KeyboardModifier.NoModifier,
            "",
            True,
            1,
        )
        QApplication.sendEvent(seek_slider, event)
        assert seek_slider.value() == expected


def test_video_arrows_clamp_to_first_and_last_frames(seek_slider: QSlider) -> None:
    seek_slider.setValue(9)
    QTest.keyClick(seek_slider, Qt.Key.Key_Right)
    assert seek_slider.value() == 10
    QTest.keyClick(seek_slider, Qt.Key.Key_Right)
    assert seek_slider.value() == 10
    seek_slider.setValue(1)
    QTest.keyClick(seek_slider, Qt.Key.Key_Left)
    assert seek_slider.value() == 0


def test_video_mouse_and_programmatic_seeking_keep_single_frame_precision(
    seek_slider: QSlider,
) -> None:
    seek_slider.setValue(3)
    assert seek_slider.value() == 3
    seek_slider.setValue(4)
    assert seek_slider.value() == 4


# ### Unrelated input behavior ###
def test_ordinary_slider_and_text_arrows_are_unchanged() -> None:
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 10)
    slider.setValue(5)
    QTest.keyClick(slider, Qt.Key.Key_Right)
    assert slider.value() == 6
    text = QLineEdit("example")
    text.setCursorPosition(3)
    QTest.keyClick(text, Qt.Key.Key_Left)
    assert text.cursorPosition() == 2
