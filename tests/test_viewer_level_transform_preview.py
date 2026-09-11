# ### Environment setup ###
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication

from housemaker.surface_geometry import SURFACE_TYPE_FLOOR, FixedSurface
from housemaker.viewer import (
    CANVAS_SURFACE_SELECTION_COLOR,
    GlbViewerWidget,
    _DepthTestedOverlayLineItem,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_surface(
    surface_id: str,
    level_index: int,
    *,
    origin_x: float = 0.0,
) -> FixedSurface:
    vertices = np.asarray(
        (
            (origin_x, 0.0, 0.0),
            (origin_x + 1.0, 0.0, 0.0),
            (origin_x + 1.0, 1.0, 0.0),
            (origin_x, 1.0, 0.0),
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
        surface_type=SURFACE_TYPE_FLOOR,
        level_index=level_index,
        room_index=None,
        mesh=mesh,
        area_square_meters=1.0,
    )


# ### Preview tests ###
class ViewerLevelTransformPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.viewer = GlbViewerWidget(window_editing_enabled=True)

    def tearDown(self) -> None:
        self.viewer.clear_level_transform_preview(
            restore_canvas_tools=False
        )
        self.viewer.close()
        self.viewer.deleteLater()
        _qt_application.processEvents()

    def test_preview_transforms_only_the_requested_level_and_updates_in_place(
        self,
    ) -> None:
        level_surface = _build_surface("level:2/floor", 2)
        other_surface = _build_surface("level:3/floor", 3, origin_x=20.0)
        self.viewer.set_wall_targets((level_surface, other_surface))
        delta = np.asarray(
            (
                (2.0, 0.0, 0.0, 3.0),
                (0.0, 2.0, 0.0, -4.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype=float,
        )

        self.viewer.set_level_transform_preview(2, delta)

        source = self.viewer._level_transform_preview_source_positions
        positions = self.viewer._level_transform_preview_outline_positions
        item = self.viewer._level_transform_preview_outline_item
        self.assertIsNotNone(source)
        self.assertIsNotNone(positions)
        self.assertIsInstance(item, _DepthTestedOverlayLineItem)
        assert source is not None and positions is not None and item is not None
        expected = np.column_stack((source, np.ones(len(source)))) @ delta.T
        np.testing.assert_allclose(positions, expected[:, :3])
        self.assertLess(float(np.max(source[:, 0])), 20.0)
        self.assertEqual(tuple(item.color), CANVAS_SURFACE_SELECTION_COLOR)

        source_cache = source
        updated_delta = delta.copy()
        updated_delta[0, 3] = 7.0
        self.viewer.set_level_transform_preview(2, updated_delta)

        self.assertIs(
            self.viewer._level_transform_preview_source_positions,
            source_cache,
        )
        self.assertIs(self.viewer._level_transform_preview_outline_item, item)
        np.testing.assert_allclose(
            self.viewer._level_transform_preview_outline_positions[:, 0],
            source[:, 0] * 2.0 + 7.0,
        )

    def test_target_and_scene_rebuilds_preserve_active_preview_until_clear(
        self,
    ) -> None:
        self.viewer.set_wall_targets((_build_surface("level:2/floor", 2),))
        delta = np.eye(4, dtype=float)
        delta[0, 3] = 2.0
        self.viewer.set_level_transform_preview(2, delta)
        first_item = self.viewer._level_transform_preview_outline_item

        self.viewer.set_wall_targets(
            (_build_surface("level:2/floor", 2, origin_x=10.0),)
        )

        positions = self.viewer._level_transform_preview_outline_positions
        self.assertIsNotNone(positions)
        assert positions is not None
        self.assertGreaterEqual(float(np.min(positions[:, 0])), 12.0)

        self.viewer._populate_scene()

        rebuilt_item = self.viewer._level_transform_preview_outline_item
        self.assertIsNotNone(rebuilt_item)
        self.assertIsNot(rebuilt_item, first_item)
        assert rebuilt_item is not None
        self.assertIn(rebuilt_item, self.viewer.view.items)
        np.testing.assert_allclose(rebuilt_item.pos, positions)

        self.viewer.clear_level_transform_preview()

        self.assertIsNone(self.viewer._level_transform_preview_level_index)
        self.assertIsNone(self.viewer._level_transform_preview_source_positions)
        self.assertIsNone(self.viewer._level_transform_preview_delta_matrix)
        self.assertIsNone(self.viewer._level_transform_preview_outline_positions)
        self.assertIsNone(self.viewer._level_transform_preview_outline_item)
        self.assertNotIn(rebuilt_item, self.viewer.view.items)

    def test_preview_rejects_invalid_level_and_delta_values(self) -> None:
        with self.assertRaisesRegex(TypeError, "integer level index"):
            self.viewer.set_level_transform_preview(True, np.eye(4))
        with self.assertRaisesRegex(ValueError, "4 by 4"):
            self.viewer.set_level_transform_preview(2, np.eye(3))

        non_finite = np.eye(4)
        non_finite[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.viewer.set_level_transform_preview(2, non_finite)

        non_affine = np.eye(4)
        non_affine[3, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "affine"):
            self.viewer.set_level_transform_preview(2, non_affine)

    def test_preview_suspends_coordinate_dependent_canvas_tools(self) -> None:
        self.viewer.set_wall_targets((_build_surface("level:2/floor", 2),))
        self.assertTrue(self.viewer.add_surface_vertex_button.isEnabled())

        self.viewer.set_level_transform_preview(2, np.eye(4))

        self.assertFalse(self.viewer.add_surface_vertex_button.isEnabled())
        self.assertIn(
            "resumes after the level transform",
            self.viewer.surface_tools_status_label.text(),
        )

        self.viewer.clear_level_transform_preview()

        self.assertTrue(self.viewer.add_surface_vertex_button.isEnabled())


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
