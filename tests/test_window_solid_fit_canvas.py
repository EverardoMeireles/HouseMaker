"""Paired wall faces share one four-corner window placement transaction."""

# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.blueprint_canvas import PENDING_WINDOW_FILL_COLOR, BlueprintCanvas
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, VertexData
from housemaker.project_io import load_project, save_project
from housemaker.surface_geometry import WallWindowPlacement, build_fixed_surfaces

# ### Fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


def _solid_level(*, right: float = 130.0) -> LevelData:
    graph = VertexData()
    for point in ((10.0, 50.0), (right, 50.0), (10.0, 65.0), (right, 65.0)):
        graph.add_vertex(*point)
    for edge in ((1, 2), (3, 4), (1, 3), (2, 4)):
        graph.add_edge(*edge)
    return LevelData(index=2, name="Ground", vertex_data=graph)


def _gap_level_without_jambs() -> LevelData:
    graph = VertexData()
    for point in (
        (10.0, 50.0),
        (40.0, 50.0),
        (90.0, 50.0),
        (130.0, 50.0),
        (10.0, 65.0),
        (40.0, 65.0),
        (90.0, 65.0),
        (130.0, 65.0),
    ):
        graph.add_vertex(*point)
    for edge in ((1, 2), (3, 4), (5, 6), (7, 8)):
        graph.add_edge(*edge)
    return LevelData(index=2, name="Ground", vertex_data=graph)


def _configure_canvas(canvas: BlueprintCanvas, level: LevelData) -> None:
    canvas.resize(640, 520)
    canvas.set_level_data(
        vertex_data=level.vertex_data,
        rooms=level.rooms,
        image_path=None,
        doorways=level.doorways,
        windows=level.windows,
    )
    canvas.blueprint_image = QImage(160, 100, QImage.Format.Format_RGB32)
    canvas.blueprint_image.fill(Qt.GlobalColor.white)
    canvas.set_stair_context((), level)


def _hover(canvas: BlueprintCanvas, pointer=(65.0, 50.0)) -> WallWindowPlacement:
    assert canvas.start_window_placement()
    canvas._update_pending_window(QPointF(*pointer))
    placement = canvas._pending_window_placement
    assert isinstance(placement, WallWindowPlacement)
    return placement


def _assert_four_corner_supports(
    test: unittest.TestCase, level: LevelData, placement: WallWindowPlacement
) -> None:
    corner_ids = placement.bridge_corner_vertex_ids
    test.assertIsNotNone(corner_ids)
    assert corner_ids is not None
    test.assertEqual(len(set(corner_ids)), 4)
    for vertex_id in corner_ids:
        test.assertIsNotNone(level.vertex_data.get_vertex(vertex_id))
    # The physical face spans retain the wall below the sill and above the
    # lintel. The two jambs connect their four endpoints through its thickness.
    for first, second in (
        (corner_ids[0], corner_ids[1]),
        (corner_ids[2], corner_ids[3]),
        (corner_ids[0], corner_ids[2]),
        (corner_ids[1], corner_ids[3]),
    ):
        test.assertTrue(level.vertex_data.has_edge(first, second))


# ### Canvas preview regressions ###
class WindowSolidFitCanvasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.level = _solid_level()
        self.canvas = BlueprintCanvas()
        _configure_canvas(self.canvas, self.level)

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()
        _application.processEvents()

    def test_solid_wall_preview_fills_actual_thickness_without_mutating_graph(self):
        before = self.level.vertex_data.to_dict()
        revision = self.level.vertex_data.revision

        placement = _hover(self.canvas)

        self.assertIsNotNone(placement.opening_fit)
        self.assertEqual(self.level.vertex_data.to_dict(), before)
        self.assertEqual(self.level.vertex_data.revision, revision)
        corners = placement.bridge_corner_vertex_ids
        assert corners is not None
        self.assertTrue(
            all(self.level.vertex_data.get_vertex(i) is None for i in corners)
        )
        polygon = self.canvas._get_window_placement_widget_polygon(placement)
        self.assertIsNotNone(polygon)
        assert polygon is not None
        self.assertEqual(
            {(point.x(), point.y()) for point in polygon},
            {
                (point.x(), point.y())
                for point in (
                    self.canvas._image_to_widget(40.0, 50.0),
                    self.canvas._image_to_widget(90.0, 50.0),
                    self.canvas._image_to_widget(40.0, 65.0),
                    self.canvas._image_to_widget(90.0, 65.0),
                )
            },
        )
        image = QImage(self.canvas.size(), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        self.canvas._paint_pending_window(painter)
        painter.end()
        midpoint = self.canvas._image_to_widget(65.0, 57.5).toPoint()
        color = image.pixelColor(midpoint)
        self.assertGreater(color.alpha(), 0)
        self.assertAlmostEqual(color.red(), PENDING_WINDOW_FILL_COLOR.red(), delta=2)
        self.assertAlmostEqual(color.blue(), PENDING_WINDOW_FILL_COLOR.blue(), delta=2)

    def test_default_window_width_contracts_to_short_paired_wall(self):
        self.level = _solid_level(right=40.0)
        _configure_canvas(self.canvas, self.level)

        placement = _hover(self.canvas, (25.0, 50.0))

        self.assertIsNotNone(placement.opening_fit)
        assert placement.opening_fit is not None
        positions = placement.opening_fit.corner_positions
        self.assertEqual({round(point[0], 6) for point in positions}, {11.0, 39.0})
        self.assertEqual({round(point[1], 6) for point in positions}, {50.0, 65.0})
        self.assertEqual(len(self.level.vertex_data.vertices), 4)

    def test_hovering_either_face_previews_the_same_full_thickness_opening(self):
        before = self.level.vertex_data.to_dict()
        for depth in (50.0, 65.0):
            with self.subTest(depth=depth):
                placement = _hover(self.canvas, (65.0, depth))
                polygon = self.canvas._get_window_placement_widget_polygon(placement)
                self.assertIsNotNone(polygon)
                assert polygon is not None
                self.assertEqual(
                    {(point.x(), point.y()) for point in polygon},
                    {
                        (point.x(), point.y())
                        for point in (
                            self.canvas._image_to_widget(40.0, 50.0),
                            self.canvas._image_to_widget(90.0, 50.0),
                            self.canvas._image_to_widget(40.0, 65.0),
                            self.canvas._image_to_widget(90.0, 65.0),
                        )
                    },
                )
                self.assertEqual(self.level.vertex_data.to_dict(), before)

    def test_gap_without_preexisting_jambs_snaps_to_all_four_endpoints(self):
        self.level = _gap_level_without_jambs()
        _configure_canvas(self.canvas, self.level)
        before = self.level.vertex_data.to_dict()

        placement = _hover(self.canvas)

        self.assertEqual(set(placement.bridge_corner_vertex_ids or ()), {2, 3, 6, 7})
        self.assertEqual((placement.start_ratio, placement.end_ratio), (0.0, 1.0))
        self.assertEqual(self.level.vertex_data.to_dict(), before)


# ### Main transaction and persistence regressions ###
class WindowSolidFitMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self.temporary.name) / "settings.json"
            )
        )
        self.workspace._set_mesh_edit_update_delay_seconds(60.0)
        self.level = _solid_level()
        self.workspace.levels = [self.level]
        self.workspace.current_level_index = 0
        self.workspace._sync_canvas_to_current_level()
        self.workspace._reset_viewer_doorway_snapshots()
        _configure_canvas(self.workspace.canvas, self.level)

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _application.processEvents()
        self.temporary.cleanup()

    def test_solid_commit_nodes_both_faces_and_waits_for_shared_mesh_delay(self):
        placement = _hover(self.workspace.canvas)
        before = self.level.vertex_data.to_dict()
        with (
            patch.object(self.workspace, "_build_viewer_preview_model") as build,
            patch.object(
                self.workspace, "_reconcile_canvas_surface_edit_and_refresh"
            ) as refresh,
        ):
            self.workspace._handle_canvas_window_placement_requested(placement)
            self.assertEqual(len(self.level.windows), 1)
            self.assertEqual(len(self.level.vertex_data.vertices), 8)
            self.assertEqual(len(self.level.vertex_data.edges), 10)
            self.assertFalse(self.level.vertex_data.has_edge(1, 2))
            self.assertFalse(self.level.vertex_data.has_edge(3, 4))
            self.assertNotEqual(self.level.vertex_data.to_dict(), before)
            _assert_four_corner_supports(self, self.level, placement)
            self.assertTrue(self.workspace._pending_doorway_bridge_topology_change)
            self.assertTrue(self.workspace._doorway_mesh_update_timer.isActive())
            self.assertEqual(
                self.workspace._build_viewer_preview_levels()[0].windows, []
            )
            self.workspace._queue_viewer_preview_refresh()
            self.workspace._run_scheduled_viewer_preview_refresh()
            build.assert_not_called()
            refresh.assert_not_called()

            self.workspace._commit_pending_doorway_mesh_update()

            refresh.assert_called_once()
            self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())
            self.assertEqual(len(self.workspace._canvas_undo_stack), 1)

    def test_single_undo_restores_unsplit_edges_and_vertex_allocator(self):
        before = self.level.vertex_data.to_dict()
        placement = _hover(self.workspace.canvas)
        self.workspace._handle_canvas_window_placement_requested(placement)
        self.assertEqual(len(self.level.windows), 1)

        self.workspace._handle_canvas_undo_requested()

        self.assertEqual(self.level.vertex_data.to_dict(), before)
        self.assertEqual(self.level.windows, [])
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_failed_window_creation_restores_complete_graph_transaction(self):
        before = self.level.vertex_data.to_dict()
        placement = _hover(self.workspace.canvas)
        with patch(
            "housemaker.main.add_wall_window",
            side_effect=ValueError("rejected opening"),
        ):
            self.workspace._handle_canvas_window_placement_requested(placement)

        self.assertEqual(self.level.vertex_data.to_dict(), before)
        self.assertEqual(self.level.windows, [])
        self.assertEqual(self.workspace._canvas_undo_stack, [])
        self.assertFalse(self.workspace._pending_doorway_bridge_topology_change)
        self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())

    def test_unexpected_window_failure_restores_graph_before_propagating(self):
        before = self.level.vertex_data.to_dict()
        placement = _hover(self.workspace.canvas)
        with (
            patch(
                "housemaker.main.add_wall_window",
                side_effect=RuntimeError("mesh failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "mesh failure"),
        ):
            self.workspace._handle_canvas_window_placement_requested(placement)

        self.assertEqual(self.level.vertex_data.to_dict(), before)
        self.assertEqual(self.level.windows, [])
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_gap_without_jambs_commits_both_spans_and_both_jambs(self):
        self.level = _gap_level_without_jambs()
        self.workspace.levels = [self.level]
        self.workspace._sync_canvas_to_current_level()
        self.workspace._reset_viewer_doorway_snapshots()
        _configure_canvas(self.workspace.canvas, self.level)
        before = self.level.vertex_data.to_dict()
        placement = _hover(self.workspace.canvas)

        self.workspace._handle_canvas_window_placement_requested(placement)

        self.assertEqual(len(self.level.windows), 1)
        self.assertEqual(len(self.level.vertex_data.vertices), 8)
        self.assertEqual(len(self.level.vertex_data.edges), 8)
        _assert_four_corner_supports(self, self.level, placement)
        self.workspace._handle_canvas_undo_requested()
        self.assertEqual(self.level.vertex_data.to_dict(), before)
        self.assertEqual(self.level.windows, [])

    def test_save_load_preserves_complete_paired_window_topology_and_cut(self):
        placement = _hover(self.workspace.canvas)
        self.workspace._handle_canvas_window_placement_requested(placement)
        self.assertEqual(len(self.level.windows), 1)
        path = Path(self.temporary.name) / "paired-window.json"

        save_project(path, 0, [self.level])
        restored = next(
            level
            for level in load_project(path).levels
            if level.index == self.level.index
        )

        self.assertEqual(
            restored.vertex_data.to_dict(), self.level.vertex_data.to_dict()
        )
        self.assertEqual(restored.windows, self.level.windows)
        _assert_four_corner_supports(self, restored, placement)
        corners = placement.bridge_corner_vertex_ids
        assert corners is not None
        surfaces = {
            surface.surface_id: surface for surface in build_fixed_surfaces([restored])
        }
        for first, second in ((corners[0], corners[1]), (corners[2], corners[3])):
            surface_id = f"level:2/wall:{min(first, second)}:{max(first, second)}"
            self.assertIn(surface_id, surfaces)
            surface = surfaces[surface_id]
            face_mask = abs(surface.mesh.face_normals[:, 1]) > 1.0 - 1e-6
            self.assertAlmostEqual(float(surface.mesh.area_faces[face_mask].sum()), 2.0)


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
