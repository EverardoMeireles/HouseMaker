# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, RoomData, VertexData
from housemaker.surface_geometry import (
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    FixedSurface,
    build_fixed_surfaces,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_level(index: int) -> LevelData:
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
    return LevelData(
        index=index,
        name=f"Level {index}",
        vertex_data=vertex_data,
        rooms=[
            RoomData(
                name="Room",
                vertex_ids=boundary_ids,
                center_vertex_id=center.id,
                color_rgb=(120, 160, 200),
            )
        ],
        floor_contour_vertex_ids=boundary_ids,
    )


def _surface(
    surfaces: tuple[FixedSurface, ...],
    surface_type: str,
    level_index: int,
) -> FixedSurface:
    return next(
        surface
        for surface in surfaces
        if surface.surface_type == surface_type
        and surface.level_index == level_index
    )


# ### Tests ###
class CanvasWallHighlightMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        self.workspace.resize(1400, 850)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _install_levels(
        self,
        levels: list[LevelData],
        *,
        current_level_index: int = 0,
    ) -> tuple[FixedSurface, ...]:
        self.workspace.levels = levels
        self.workspace.current_level_index = current_level_index
        self.workspace._reset_viewer_doorway_snapshots()
        self.workspace._sync_level_controls()
        self.workspace._sync_canvas_to_current_level()
        surfaces = tuple(build_fixed_surfaces(levels))
        self.workspace._set_canvas_viewer_targets(surfaces)
        return surfaces

    def test_3d_wall_selection_highlights_the_wall_on_the_2d_canvas(
        self,
    ) -> None:
        level = _build_level(0)
        surfaces = self._install_levels([level])
        wall = _surface(surfaces, SURFACE_TYPE_WALL, level.index)

        self.assertTrue(
            self.workspace.viewer.select_canvas_surface_target(
                wall.surface_id
            )
        )

        self.assertEqual(
            self.workspace.canvas.get_selected_wall_surface_id(),
            wall.surface_id,
        )

    def test_non_wall_selection_and_clear_remove_the_2d_highlight(self) -> None:
        level = _build_level(0)
        surfaces = self._install_levels([level])
        wall = _surface(surfaces, SURFACE_TYPE_WALL, level.index)
        floor = _surface(surfaces, SURFACE_TYPE_FLOOR, level.index)
        viewer = self.workspace.viewer

        viewer.select_canvas_surface_target(wall.surface_id)
        self.assertEqual(
            self.workspace.canvas.get_selected_wall_surface_id(),
            wall.surface_id,
        )

        viewer.select_canvas_surface_target(floor.surface_id)
        self.assertIsNone(
            self.workspace.canvas.get_selected_wall_surface_id()
        )

        viewer.select_canvas_surface_target(wall.surface_id)
        viewer.select_canvas_surface_target(None)
        self.assertIsNone(
            self.workspace.canvas.get_selected_wall_surface_id()
        )

    def test_target_refresh_preserves_the_selected_wall_highlight(self) -> None:
        level = _build_level(0)
        surfaces = self._install_levels([level])
        wall = _surface(surfaces, SURFACE_TYPE_WALL, level.index)
        self.workspace.viewer.select_canvas_surface_target(wall.surface_id)

        refreshed_surfaces = tuple(build_fixed_surfaces([level]))
        self.workspace._set_canvas_viewer_targets(refreshed_surfaces)

        self.assertEqual(
            self.workspace.viewer.get_active_canvas_surface_id(),
            wall.surface_id,
        )
        self.assertEqual(
            self.workspace.canvas.get_selected_wall_surface_id(),
            wall.surface_id,
        )

    def test_wall_highlight_is_filtered_to_the_current_2d_level(self) -> None:
        first_level = _build_level(0)
        second_level = _build_level(1)
        surfaces = self._install_levels([first_level, second_level])
        second_wall = _surface(
            surfaces,
            SURFACE_TYPE_WALL,
            second_level.index,
        )

        self.workspace.viewer.select_canvas_surface_target(
            second_wall.surface_id
        )
        self.assertIsNone(
            self.workspace.canvas.get_selected_wall_surface_id()
        )

        self.workspace.current_level_index = 1
        self.workspace._sync_canvas_to_current_level()
        self.assertEqual(
            self.workspace.canvas.get_selected_wall_surface_id(),
            second_wall.surface_id,
        )

        self.workspace.current_level_index = 0
        self.workspace._sync_canvas_to_current_level()
        self.assertIsNone(
            self.workspace.canvas.get_selected_wall_surface_id()
        )


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
