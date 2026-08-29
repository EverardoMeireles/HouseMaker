# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.glb import GeneratedModel
from housemaker.level_coordinates import build_doorway_world_outline_positions
from housemaker.main import BlueprintWorkspace
from housemaker.models import (
    DOORWAY_SHAPE_ARCH,
    DOORWAY_SHAPE_RECTANGULAR,
    DoorwayData,
)
from housemaker.settings_widget import (
    DOORWAY_MESH_UPDATE_DELAY_SECONDS_SETTING_KEY,
)


# ### Constants ###
TEST_DOORWAY_DELAY_SECONDS = 10.0
TEST_BLUEPRINT_SIZE_PIXELS = 100


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_doorway(
    center_x: float,
    shape: str,
) -> DoorwayData:
    return DoorwayData(
        center_x=center_x,
        center_y=50.0,
        width_meters=0.4,
        height_meters=2.1,
        depth_meters=0.2,
        rotation_degrees=0.0,
        shape=shape,
    )


def _preview_model() -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    return GeneratedModel(
        mesh=mesh,
        scene=trimesh.Scene({"doorway-preview": mesh.copy()}),
        glb_bytes=b"",
    )


# ### Main-window doorway arch integration tests ###
class DoorwayArchMainIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        application_settings = ApplicationSettingsStore(
            Path(self._temporary_directory.name) / "settings.json"
        )
        application_settings.set(
            DOORWAY_MESH_UPDATE_DELAY_SECONDS_SETTING_KEY,
            TEST_DOORWAY_DELAY_SECONDS,
        )
        self.workspace = BlueprintWorkspace(
            application_settings=application_settings
        )
        self.workspace.resize(1500, 900)
        self.workspace.show()
        _qt_application.processEvents()

        blueprint_image = QImage(
            TEST_BLUEPRINT_SIZE_PIXELS,
            TEST_BLUEPRINT_SIZE_PIXELS,
            QImage.Format.Format_RGB32,
        )
        blueprint_image.fill(Qt.GlobalColor.white)
        self.workspace.canvas.blueprint_image = blueprint_image
        self.workspace.current_level.image_size_pixels = (
            float(TEST_BLUEPRINT_SIZE_PIXELS),
            float(TEST_BLUEPRINT_SIZE_PIXELS),
        )

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _install_doorways(self, doorways: list[DoorwayData]) -> None:
        self.workspace.current_level.doorways[:] = doorways
        self.workspace._reset_viewer_doorway_snapshots()
        self.workspace.canvas.update()
        _qt_application.processEvents()

    def _click_doorway(self, doorway: DoorwayData) -> None:
        position = self.workspace.canvas._image_to_widget(
            doorway.center_x,
            doorway.center_y,
        ).toPoint()
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=position,
        )
        _qt_application.processEvents()

    def test_selection_synchronizes_arch_checkbox_without_editing(self) -> None:
        rectangular = _build_doorway(25.0, DOORWAY_SHAPE_RECTANGULAR)
        arch = _build_doorway(75.0, DOORWAY_SHAPE_ARCH)
        self._install_doorways([rectangular, arch])
        preview_edits: list[None] = []
        self.workspace.canvas.doorway_dimension_preview_changed.connect(
            lambda: preview_edits.append(None)
        )

        self.assertFalse(self.workspace.selected_doorway_arch_checkbox.isEnabled())
        self.assertFalse(self.workspace.selected_doorway_arch_checkbox.isChecked())

        self._click_doorway(arch)

        self.assertEqual(self.workspace.canvas.selected_doorway_index, 1)
        self.assertTrue(self.workspace.selected_doorway_arch_checkbox.isEnabled())
        self.assertTrue(self.workspace.selected_doorway_arch_checkbox.isChecked())

        self._click_doorway(rectangular)

        self.assertEqual(self.workspace.canvas.selected_doorway_index, 0)
        self.assertTrue(self.workspace.selected_doorway_arch_checkbox.isEnabled())
        self.assertFalse(self.workspace.selected_doorway_arch_checkbox.isChecked())
        self.assertEqual(preview_edits, [])
        self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())

        self.workspace.canvas._set_selected_doorway_index(None)

        self.assertFalse(self.workspace.selected_doorway_arch_checkbox.isEnabled())
        self.assertFalse(self.workspace.selected_doorway_arch_checkbox.isChecked())

    def test_toggle_previews_arch_then_commits_geometry_after_delay(self) -> None:
        doorway = _build_doorway(50.0, DOORWAY_SHAPE_RECTANGULAR)
        self._install_doorways([doorway])
        self._click_doorway(doorway)
        self.workspace.canvas_viewer_tabs.setCurrentIndex(
            self.workspace.canvas_3d_view_tab_index
        )
        _qt_application.processEvents()
        initial_revision = self.workspace._viewer_preview_revision
        level_index = self.workspace.current_level.index
        self.assertTrue(self.workspace.selected_doorway_arch_checkbox.isEnabled())
        self.assertFalse(self.workspace.selected_doorway_arch_checkbox.isChecked())

        with (
            patch.object(
                self.workspace,
                "_build_viewer_preview_model",
                return_value=_preview_model(),
            ) as build_preview,
            patch(
                "housemaker.main.build_fixed_surfaces",
                return_value=(),
            ) as build_surfaces,
        ):
            self.workspace.selected_doorway_arch_checkbox.click()
            _qt_application.processEvents()

            self.assertEqual(
                self.workspace.current_level.doorways[0].shape,
                DOORWAY_SHAPE_ARCH,
            )
            self.assertEqual(
                self.workspace._viewer_doorways_by_level_index[level_index][0].shape,
                DOORWAY_SHAPE_RECTANGULAR,
            )
            self.assertEqual(
                self.workspace._build_viewer_preview_levels()[
                    self.workspace.current_level_index
                ].doorways[0].shape,
                DOORWAY_SHAPE_RECTANGULAR,
            )
            self.assertEqual(
                self.workspace._viewer_preview_revision,
                initial_revision,
            )
            self.assertTrue(self.workspace._doorway_mesh_update_timer.isActive())
            self.assertEqual(
                self.workspace._doorway_mesh_update_timer.interval(),
                round(TEST_DOORWAY_DELAY_SECONDS * 1000.0),
            )
            build_preview.assert_not_called()
            build_surfaces.assert_not_called()

            expected_outline = np.asarray(
                build_doorway_world_outline_positions(
                    self.workspace.levels,
                    self.workspace.current_level,
                    self.workspace.current_level.doorways[0],
                ),
                dtype=float,
            )
            np.testing.assert_allclose(
                self.workspace.viewer._doorway_preview_outline_positions,
                expected_outline,
            )

            self.workspace._doorway_mesh_update_timer.timeout.emit()
            _qt_application.processEvents()

        self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())
        self.assertEqual(
            self.workspace._viewer_doorways_by_level_index[level_index][0].shape,
            DOORWAY_SHAPE_ARCH,
        )
        self.assertEqual(
            self.workspace._viewer_preview_revision,
            initial_revision + 1,
        )
        build_preview.assert_called_once_with(None)
        build_surfaces.assert_called_once()
        rendered_levels = build_surfaces.call_args.args[0]
        self.assertEqual(
            rendered_levels[self.workspace.current_level_index].doorways[0].shape,
            DOORWAY_SHAPE_ARCH,
        )
