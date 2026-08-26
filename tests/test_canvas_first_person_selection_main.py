# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, RoomData, VertexData
from housemaker.surface_geometry import FixedSurface, build_fixed_surfaces
from housemaker.viewer import (
    NAVIGATION_MODE_FIRST_PERSON,
    NAVIGATION_MODE_ORBIT,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_square_level() -> LevelData:
    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
    ):
        vertex_data.add_edge(start_id, end_id)
    center = vertex_data.add_vertex(50.0, 50.0)
    room = RoomData(
        name="Living room",
        vertex_ids=boundary_ids,
        center_vertex_id=center.id,
        color_rgb=(140, 180, 220),
    )
    return LevelData(
        index=0,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[room],
        floor_contour_vertex_ids=boundary_ids,
    )


def _get_wall_and_ray(
    level: LevelData,
) -> tuple[FixedSurface, tuple[np.ndarray, np.ndarray]]:
    wall = next(
        surface
        for surface in build_fixed_surfaces([level])
        if surface.surface_type == "wall"
    )
    vertices = np.asarray(wall.mesh.vertices, dtype=float)
    faces = np.asarray(wall.mesh.faces, dtype=np.int64)
    triangle = vertices[faces[0]]
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    target = np.asarray(wall.mesh.centroid, dtype=float)
    return wall, (target - normal, normal)


def _camera_state(view) -> tuple[float, ...]:
    center = view.opts["center"]
    return (
        float(center.x()),
        float(center.y()),
        float(center.z()),
        float(view.opts["distance"]),
        float(view.opts["azimuth"]),
        float(view.opts["elevation"]),
        float(view.opts["fov"]),
    )


def _primary_screen():
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        raise RuntimeError("A QScreen is required for detached viewer tests.")
    return screen


# ### Main and detached Canvas integration tests ###
class CanvasFirstPersonSelectionMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        self.workspace.resize(1400, 850)
        self.workspace.show()
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.canvas_viewer_workspace
        )
        self.workspace.canvas_viewer_tabs.setCurrentIndex(
            self.workspace.canvas_3d_view_tab_index
        )
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.viewer.exit_first_person_mode()
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _install_wall_target(
        self,
    ) -> tuple[FixedSurface, tuple[np.ndarray, np.ndarray]]:
        level = _build_square_level()
        self.workspace.levels = [level]
        wall, ray = _get_wall_and_ray(level)
        self.workspace.viewer.set_wall_targets(
            tuple(build_fixed_surfaces([level]))
        )
        return wall, ray

    def test_released_first_person_view_selects_and_arms_a_wall_window(
        self,
    ) -> None:
        wall, ray = self._install_wall_target()
        viewer = self.workspace.viewer
        orbit_state = _camera_state(viewer.view)

        self.workspace.canvas_3d_navigation_shortcut.activated.emit()
        first_person_state = _camera_state(viewer.view)

        self.assertEqual(
            viewer.get_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )
        self.assertTrue(viewer.is_first_person_pointer_captured)
        self.assertNotEqual(first_person_state, orbit_state)

        click_position = QPoint(120, 90)
        QTest.mouseClick(
            viewer.view,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            click_position,
        )

        self.assertFalse(viewer.is_first_person_pointer_captured)
        self.assertEqual(
            viewer.get_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )
        self.assertEqual(_camera_state(viewer.view), first_person_state)
        self.assertEqual(
            self.workspace.canvas_viewer_tabs.tabText(
                self.workspace.canvas_3d_view_tab_index
            ),
            "3D view (first person)",
        )

        with patch.object(viewer.view, "build_camera_ray", return_value=ray):
            QTest.mouseClick(
                viewer.view,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                click_position,
            )

        self.assertEqual(viewer.get_selected_wall_surface_id(), wall.surface_id)
        assert viewer.add_window_button is not None
        self.assertTrue(viewer.add_window_button.isEnabled())
        viewer.add_window_button.click()

        self.assertTrue(viewer.is_window_placement_active())
        self.assertFalse(viewer.is_first_person_pointer_captured)
        self.assertEqual(
            viewer.get_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )
        self.assertEqual(_camera_state(viewer.view), first_person_state)

        QTest.keyClick(viewer.view, Qt.Key.Key_N)
        _qt_application.processEvents()

        self.assertEqual(viewer.get_navigation_mode(), NAVIGATION_MODE_ORBIT)
        self.assertFalse(viewer.is_window_placement_active())
        self.assertEqual(_camera_state(viewer.view), orbit_state)
        self.assertEqual(
            self.workspace.canvas_viewer_tabs.tabText(
                self.workspace.canvas_3d_view_tab_index
            ),
            "3D view",
        )

    def test_detached_viewer_keeps_first_person_selection_until_hotkey(
        self,
    ) -> None:
        wall, ray = self._install_wall_target()
        viewer = self.workspace.viewer
        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=_primary_screen(),
        ):
            self.workspace._apply_fullscreen_3d_viewer_screen(
                "first-person-selection-screen"
            )
        _qt_application.processEvents()

        host = self.workspace._external_viewer_host
        self.assertTrue(host.is_active)
        self.assertIs(host.viewer, viewer)

        QTest.keyClick(viewer.view, Qt.Key.Key_N)
        _qt_application.processEvents()
        self.assertEqual(
            viewer.get_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )

        click_position = QPoint(160, 110)
        QTest.mouseClick(
            viewer.view,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            click_position,
        )
        with patch.object(viewer.view, "build_camera_ray", return_value=ray):
            QTest.mouseClick(
                viewer.view,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                click_position,
            )

        self.assertIs(host.viewer, viewer)
        self.assertFalse(viewer.is_first_person_pointer_captured)
        self.assertEqual(
            viewer.get_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )
        self.assertEqual(viewer.get_selected_wall_surface_id(), wall.surface_id)

        QTest.keyClick(viewer.view, Qt.Key.Key_N)
        _qt_application.processEvents()

        self.assertIs(host.viewer, viewer)
        self.assertEqual(viewer.get_navigation_mode(), NAVIGATION_MODE_ORBIT)


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
