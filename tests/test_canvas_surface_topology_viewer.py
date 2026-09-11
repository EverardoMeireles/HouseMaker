# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
from OpenGL import GL
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
import trimesh

from housemaker.architectural_surface_edits import (
    SurfaceDrawingEdgeTarget,
    SurfaceDrawingOverlay,
    SurfaceDrawingVertexTarget,
    SurfaceFaceExtrusionRequest,
    SurfaceVertexInsertionRequest,
)
from housemaker.canvas_surface_edits import (
    CANVAS_SURFACE_EDIT_WALL_VERTEX,
    CanvasSurfaceEditHandleTarget,
    CanvasSurfaceEditReference,
)
from housemaker.glb import GeneratedModel
from housemaker.surface_geometry import (
    FixedSurface,
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
)
from housemaker.viewer import (
    CANVAS_EXTRUDABLE_FACE_OUTLINE_COLOR,
    CANVAS_OPENING_OVERLAY_DEPTH_VALUE,
    CANVAS_SURFACE_DRAWING_EDGE_COLOR,
    CANVAS_SURFACE_DRAWING_VERTEX_COLOR,
    CANVAS_SURFACE_SELECTION_COLOR,
    GlbViewerWidget,
    _get_nearest_fixed_surface_ray_hit,
    _resolve_canvas_surface_vertex_preview,
    _snap_surface_point_from_active_vertex,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_wall(
    surface_id: str = "level:0/wall:1:2",
    *,
    minimum_x: float = 0.0,
    maximum_x: float = 4.0,
    source_surface_id: str | None = None,
    is_directly_drawn: bool = False,
) -> FixedSurface:
    vertices = np.asarray(
        (
            (minimum_x, 0.0, 0.0),
            (maximum_x, 0.0, 0.0),
            (maximum_x, 0.0, 3.0),
            (minimum_x, 0.0, 3.0),
        ),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    is_child = source_surface_id is not None
    return FixedSurface(
        surface_id=surface_id,
        surface_type=SURFACE_TYPE_WALL,
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=(maximum_x - minimum_x) * 3.0,
        wall_key=None if is_child else "1:2",
        wall_start_world=None if is_child else (minimum_x, 0.0, 0.0),
        wall_end_world=None if is_child else (maximum_x, 0.0, 0.0),
        wall_height_meters=None if is_child else 3.0,
        source_surface_id=source_surface_id,
        is_directly_drawn=is_directly_drawn,
    )


def _build_horizontal_surface(
    surface_type: str,
    z_meters: float,
) -> FixedSurface:
    vertices = np.asarray(
        (
            (0.0, 0.0, z_meters),
            (4.0, 0.0, z_meters),
            (4.0, 3.0, z_meters),
            (0.0, 3.0, z_meters),
        ),
        dtype=float,
    )
    faces = (
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        if surface_type == SURFACE_TYPE_FLOOR
        else np.asarray(((0, 2, 1), (0, 3, 2)), dtype=np.int64)
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return FixedSurface(
        surface_id=f"level:0/{surface_type}",
        surface_type=surface_type,
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=12.0,
    )


def _build_model() -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(4.0, 0.2, 3.0))
    mesh.apply_translation((2.0, 0.1, 1.5))
    return GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b"")


def _ray_at(x: float, z: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((x, -2.0, z), dtype=float),
        np.asarray((0.0, 1.0, 0.0), dtype=float),
    )


def _build_child_structural_target(
    child: FixedSurface,
) -> CanvasSurfaceEditHandleTarget:
    return CanvasSurfaceEditHandleTarget(
        reference=CanvasSurfaceEditReference(
            kind=CANVAS_SURFACE_EDIT_WALL_VERTEX,
            level_index=0,
            axis_index=0,
            vertex_id=1,
        ),
        surface_id=child.surface_id,
        origin_world=(0.0, 0.0, 0.0),
        axis_world=(1.0, 0.0, 0.0),
        minimum_delta_meters=-1.0,
        maximum_delta_meters=1.0,
        chain_vertex_ids=(1, 2),
        chain_vertex_ratios=(0.0, 1.0),
        chain_vertex_image_positions=((0.0, 0.0), (4.0, 0.0)),
    )


def _line_positions_with_color(
    viewer: GlbViewerWidget,
    color: tuple[float, float, float, float],
) -> tuple[np.ndarray, ...]:
    """Return rendered line positions whose uniform color matches ``color``."""

    matching: list[np.ndarray] = []
    for item in viewer.view.items:
        if getattr(item, "mode", None) != "lines":
            continue
        item_color = getattr(item, "color", None)
        if item_color is None:
            continue
        try:
            color_matches = bool(
                np.allclose(
                    np.asarray(item_color, dtype=float),
                    np.asarray(color, dtype=float),
                )
            )
        except (TypeError, ValueError):
            continue
        if not color_matches:
            continue
        positions = np.asarray(getattr(item, "pos", ()), dtype=float)
        if positions.ndim == 2 and positions.shape[1:] == (3,):
            matching.append(positions)
    return tuple(matching)


def _scatter_items_with_color(
    viewer: GlbViewerWidget,
    color: tuple[float, float, float, float],
) -> tuple[object, ...]:
    """Return rendered point-marker items whose uniform color matches."""

    matching: list[object] = []
    for item in viewer.view.items:
        if type(item).__name__ != "GLScatterPlotItem":
            continue
        item_color = getattr(item, "color", None)
        if item_color is None:
            continue
        try:
            color_matches = bool(
                np.allclose(
                    np.asarray(item_color, dtype=float),
                    np.asarray(color, dtype=float),
                )
            )
        except (TypeError, ValueError):
            continue
        if color_matches:
            matching.append(item)
    return tuple(matching)


def _canonical_segments(*position_groups: np.ndarray) -> tuple[object, ...]:
    """Normalize paired line positions into comparable undirected segments."""

    segments: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    for positions in position_groups:
        normalized = np.asarray(positions, dtype=float)
        if len(normalized) % 2:
            raise AssertionError("Line positions must contain paired endpoints.")
        for first, second in normalized.reshape((-1, 2, 3)):
            endpoints = tuple(
                sorted(
                    (
                        tuple(float(value) for value in np.round(first, 6)),
                        tuple(float(value) for value in np.round(second, 6)),
                    )
                )
            )
            segments.append(endpoints)
    return tuple(sorted(segments))


def _wall_rectangle_segments(
    minimum_x: float,
    maximum_x: float,
) -> tuple[object, ...]:
    """Return the four true boundary segments of a wall fixture."""

    corners = (
        (minimum_x, 0.0, 0.0),
        (maximum_x, 0.0, 0.0),
        (maximum_x, 0.0, 3.0),
        (minimum_x, 0.0, 3.0),
    )
    positions = np.asarray(
        tuple(
            point
            for index, point in enumerate(corners)
            for point in (point, corners[(index + 1) % len(corners)])
        ),
        dtype=float,
    )
    return _canonical_segments(positions)


def _horizontal_rectangle_segments(z_meters: float) -> tuple[object, ...]:
    """Return the four true boundary segments of a horizontal fixture."""

    corners = (
        (0.0, 0.0, z_meters),
        (4.0, 0.0, z_meters),
        (4.0, 3.0, z_meters),
        (0.0, 3.0, z_meters),
    )
    positions = np.asarray(
        tuple(
            point
            for index, point in enumerate(corners)
            for point in (point, corners[(index + 1) % len(corners)])
        ),
        dtype=float,
    )
    return _canonical_segments(positions)


# ### Canvas surface topology viewer tests ###
class CanvasSurfaceTopologyViewerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[GlbViewerWidget] = []

    def tearDown(self) -> None:
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        _qt_application.processEvents()

    def _build_viewer(
        self,
        surfaces: tuple[FixedSurface, ...],
    ) -> GlbViewerWidget:
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets(surfaces)
        viewer.set_model(_build_model())
        self.widgets.append(viewer)
        return viewer

    def test_child_faces_are_not_window_or_structural_targets(
        self,
    ) -> None:
        parent = _build_wall()
        child = _build_wall(
            "level:0/edit-face:11111111111111111111111111111111:wall",
            source_surface_id=parent.surface_id,
        )
        viewer = self._build_viewer((parent, child))

        viewer.set_canvas_surface_edit_targets(
            (_build_child_structural_target(child),)
        )
        viewer.select_canvas_surface_target(child.surface_id)

        self.assertNotIn(child.surface_id, viewer._window_wall_targets)
        self.assertEqual(viewer._canvas_surface_edit_targets, {})
        self.assertIsNone(viewer.get_selected_wall_surface_id())
        assert viewer.add_window_button is not None
        self.assertFalse(viewer.add_window_button.isEnabled())

    def test_only_directly_drawn_unselected_face_has_a_blue_boundary(
        self,
    ) -> None:
        parent = _build_wall()
        directly_drawn = _build_wall(
            "level:0/edit-face:11111111111111111111111111111111:wall",
            minimum_x=0.25,
            maximum_x=1.25,
            source_surface_id=parent.surface_id,
            is_directly_drawn=True,
        )
        automatically_created = _build_wall(
            "level:0/edit-face:22222222222222222222222222222222:wall",
            minimum_x=2.75,
            maximum_x=3.75,
            source_surface_id=parent.surface_id,
        )
        viewer = self._build_viewer(
            (parent, directly_drawn, automatically_created)
        )

        blue_positions = _line_positions_with_color(
            viewer,
            CANVAS_EXTRUDABLE_FACE_OUTLINE_COLOR,
        )
        self.assertEqual(
            _canonical_segments(*blue_positions),
            _wall_rectangle_segments(0.25, 1.25),
        )

    def test_direct_face_selection_swaps_blue_boundary_and_markers_for_yellow(
        self,
    ) -> None:
        parent = _build_wall()
        directly_drawn = _build_wall(
            "level:0/edit-face:11111111111111111111111111111111:wall",
            minimum_x=0.25,
            maximum_x=1.25,
            source_surface_id=parent.surface_id,
            is_directly_drawn=True,
        )
        viewer = self._build_viewer((parent, directly_drawn))
        direct_face_id = directly_drawn.surface_id
        marker_points = (
            (0.25, 0.0, 0.0),
            (1.25, 0.0, 0.0),
            (1.25, 0.0, 3.0),
            (0.25, 0.0, 3.0),
        )
        viewer.set_canvas_surface_drawing_overlay(
            SurfaceDrawingOverlay(
                vertices=tuple(
                    SurfaceDrawingVertexTarget(
                        vertex_id=str(index + 1) * 32,
                        source_surface_id=parent.surface_id,
                        world_point=world_point,
                        show_marker=True,
                        direct_face_surface_ids=(direct_face_id,),
                    )
                    for index, world_point in enumerate(marker_points)
                ),
                edges=(),
            )
        )

        self.assertEqual(
            _canonical_segments(
                *_line_positions_with_color(
                    viewer,
                    CANVAS_EXTRUDABLE_FACE_OUTLINE_COLOR,
                )
            ),
            _wall_rectangle_segments(0.25, 1.25),
        )
        blue_markers = _scatter_items_with_color(
            viewer,
            CANVAS_SURFACE_DRAWING_VERTEX_COLOR,
        )
        self.assertEqual(len(blue_markers), 1)
        np.testing.assert_allclose(
            np.asarray(getattr(blue_markers[0], "pos"), dtype=float),
            np.asarray(marker_points, dtype=float),
        )

        viewer.select_canvas_surface_target(direct_face_id)

        self.assertEqual(
            _line_positions_with_color(
                viewer,
                CANVAS_EXTRUDABLE_FACE_OUTLINE_COLOR,
            ),
            (),
        )
        self.assertEqual(
            _scatter_items_with_color(
                viewer,
                CANVAS_SURFACE_DRAWING_VERTEX_COLOR,
            ),
            (),
        )
        yellow_positions = _line_positions_with_color(
            viewer,
            CANVAS_SURFACE_SELECTION_COLOR,
        )
        self.assertEqual(len(yellow_positions), 1)
        self.assertEqual(
            _canonical_segments(*yellow_positions),
            _wall_rectangle_segments(0.25, 1.25),
        )

        viewer.select_canvas_surface_target(None)

        self.assertEqual(
            _line_positions_with_color(
                viewer,
                CANVAS_SURFACE_SELECTION_COLOR,
            ),
            (),
        )
        self.assertEqual(
            _canonical_segments(
                *_line_positions_with_color(
                    viewer,
                    CANVAS_EXTRUDABLE_FACE_OUTLINE_COLOR,
                )
            ),
            _wall_rectangle_segments(0.25, 1.25),
        )
        self.assertEqual(
            len(
                _scatter_items_with_color(
                    viewer,
                    CANVAS_SURFACE_DRAWING_VERTEX_COLOR,
                )
            ),
            1,
        )

    def test_direct_face_selection_keeps_unowned_manual_vertex_markers(
        self,
    ) -> None:
        parent = _build_wall()
        directly_drawn = _build_wall(
            "level:0/edit-face:11111111111111111111111111111111:wall",
            minimum_x=0.25,
            maximum_x=1.25,
            source_surface_id=parent.surface_id,
            is_directly_drawn=True,
        )
        viewer = self._build_viewer((parent, directly_drawn))
        owned_point = (0.25, 0.0, 0.0)
        unowned_point = (2.0, 0.0, 1.0)
        viewer.set_canvas_surface_drawing_overlay(
            SurfaceDrawingOverlay(
                vertices=(
                    SurfaceDrawingVertexTarget(
                        vertex_id="1" * 32,
                        source_surface_id=parent.surface_id,
                        world_point=owned_point,
                        show_marker=True,
                        direct_face_surface_ids=(directly_drawn.surface_id,),
                    ),
                    SurfaceDrawingVertexTarget(
                        vertex_id="2" * 32,
                        source_surface_id=parent.surface_id,
                        world_point=unowned_point,
                        show_marker=True,
                    ),
                ),
                edges=(),
            )
        )

        viewer.select_canvas_surface_target(directly_drawn.surface_id)

        blue_markers = _scatter_items_with_color(
            viewer,
            CANVAS_SURFACE_DRAWING_VERTEX_COLOR,
        )
        self.assertEqual(len(blue_markers), 1)
        np.testing.assert_allclose(
            np.asarray(getattr(blue_markers[0], "pos"), dtype=float),
            np.asarray((unowned_point,), dtype=float),
        )

    def test_non_extrudable_base_surfaces_have_no_blue_boundaries(self) -> None:
        wall = _build_wall()
        floor = _build_horizontal_surface(SURFACE_TYPE_FLOOR, 0.0)
        ceiling = _build_horizontal_surface(SURFACE_TYPE_CEILING, 3.0)
        viewer = self._build_viewer((wall, floor, ceiling))

        self.assertEqual(
            _line_positions_with_color(
                viewer,
                CANVAS_EXTRUDABLE_FACE_OUTLINE_COLOR,
            ),
            (),
        )

    def test_structural_selection_has_one_true_edge_contour_without_duplicates(
        self,
    ) -> None:
        wall = _build_wall()
        viewer = self._build_viewer((wall,))
        viewer.set_canvas_surface_edit_targets(
            (_build_child_structural_target(wall),)
        )

        viewer.select_canvas_surface_target(wall.surface_id)

        yellow_positions = _line_positions_with_color(
            viewer,
            CANVAS_SURFACE_SELECTION_COLOR,
        )
        self.assertEqual(len(yellow_positions), 1)
        segments = _canonical_segments(*yellow_positions)
        self.assertEqual(segments, _wall_rectangle_segments(0.0, 4.0))
        self.assertEqual(len(segments), len(set(segments)))

    def test_floor_and_ceiling_selections_use_one_true_edge_contour(self) -> None:
        for surface_type, z_meters in (
            (SURFACE_TYPE_FLOOR, 0.0),
            (SURFACE_TYPE_CEILING, 3.0),
        ):
            with self.subTest(surface_type=surface_type):
                surface = _build_horizontal_surface(surface_type, z_meters)
                viewer = self._build_viewer((surface,))

                viewer.select_canvas_surface_target(surface.surface_id)

                yellow_positions = _line_positions_with_color(
                    viewer,
                    CANVAS_SURFACE_SELECTION_COLOR,
                )
                self.assertEqual(len(yellow_positions), 1)
                segments = _canonical_segments(*yellow_positions)
                self.assertEqual(
                    segments,
                    _horizontal_rectangle_segments(z_meters),
                )
                self.assertEqual(len(segments), len(set(segments)))

    def test_add_vertex_uses_snapped_preview_and_emits_request_while_staying_armed(
        self,
    ) -> None:
        wall = _build_wall()
        viewer = self._build_viewer((wall,))
        emitted: list[object] = []
        viewer.canvas_surface_vertex_insertion_requested.connect(emitted.append)
        snapped = (1.0, 0.0, 1.0)

        self.assertTrue(viewer.begin_surface_vertex_placement())
        with (
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=_ray_at(1.01, 1.005),
            ),
            patch(
                "housemaker.architectural_surface_edits."
                "snap_surface_vertex_world_point",
                return_value=snapped,
            ) as snap,
        ):
            self.assertTrue(
                viewer._begin_surface_vertex_pointer_interaction(QPointF())
            )
            np.testing.assert_allclose(
                viewer._surface_vertex_preview_item.pos,
                np.asarray((snapped,), dtype=float),
            )
            self.assertTrue(
                viewer._finish_surface_vertex_pointer_interaction(QPointF())
            )

        self.assertGreaterEqual(snap.call_count, 1)
        self.assertEqual(
            emitted,
            [SurfaceVertexInsertionRequest(wall.surface_id, snapped)],
        )
        self.assertTrue(viewer.is_surface_vertex_placement_active())
        assert viewer.add_surface_vertex_button is not None
        self.assertTrue(viewer.add_surface_vertex_button.isChecked())
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)

    def test_add_vertex_hover_selects_existing_vertex_and_previews_next_edge(
        self,
    ) -> None:
        wall = _build_wall()
        viewer = self._build_viewer((wall,))
        first_id = "1" * 32
        second_id = "2" * 32
        overlay = SurfaceDrawingOverlay(
            vertices=(
                SurfaceDrawingVertexTarget(
                    vertex_id=first_id,
                    source_surface_id=wall.surface_id,
                    world_point=(0.5, 0.0, 1.0),
                ),
                SurfaceDrawingVertexTarget(
                    vertex_id=second_id,
                    source_surface_id=wall.surface_id,
                    world_point=(2.0, 0.0, 1.0),
                ),
            ),
            edges=(),
        )
        viewer.set_canvas_surface_drawing_overlay(
            overlay,
            active_vertex_id=first_id,
        )
        self.assertTrue(viewer.begin_surface_vertex_placement())

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.005, 1.004),
        ):
            self.assertTrue(
                viewer._handle_surface_vertex_pointer_hovered(QPointF())
            )

        preview = viewer._surface_vertex_hover_preview
        self.assertIsNotNone(preview)
        assert preview is not None
        self.assertEqual(preview.snapped_vertex_id, second_id)
        np.testing.assert_allclose(preview.world_point, (2.0, 0.0, 1.0))
        self.assertIsNotNone(viewer._surface_vertex_preview_item)
        self.assertIsNotNone(viewer._surface_vertex_preview_edge_item)
        assert viewer._surface_vertex_preview_edge_item is not None
        np.testing.assert_allclose(
            viewer._surface_vertex_preview_edge_item.pos,
            np.asarray(((0.5, 0.0, 1.0), (2.0, 0.0, 1.0))),
        )
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)

    def test_add_vertex_does_not_snap_to_a_vertex_over_one_centimeter_away(
        self,
    ) -> None:
        wall = _build_wall()
        vertex_id = "1" * 32

        preview = _resolve_canvas_surface_vertex_preview(
            wall,
            (2.015, 0.0, 1.0),
            (wall,),
            (
                SurfaceDrawingVertexTarget(
                    vertex_id=vertex_id,
                    source_surface_id=wall.surface_id,
                    world_point=(2.0, 0.0, 1.0),
                ),
            ),
            (),
            None,
        )

        self.assertIsNone(preview.snapped_vertex_id)
        self.assertEqual(preview.snap_kind, "surface")
        np.testing.assert_allclose(preview.world_point, (2.015, 0.0, 1.0))

    def test_click_commits_once_with_active_vertex_and_owns_only_that_click(
        self,
    ) -> None:
        wall = _build_wall()
        viewer = self._build_viewer((wall,))
        first_id = "1" * 32
        second_id = "2" * 32
        overlay = SurfaceDrawingOverlay(
            vertices=(
                SurfaceDrawingVertexTarget(
                    vertex_id=first_id,
                    source_surface_id=wall.surface_id,
                    world_point=(0.5, 0.0, 1.0),
                ),
                SurfaceDrawingVertexTarget(
                    vertex_id=second_id,
                    source_surface_id=wall.surface_id,
                    world_point=(2.0, 0.0, 1.0),
                ),
            ),
            edges=(),
        )
        viewer.set_canvas_surface_drawing_overlay(
            overlay,
            active_vertex_id=first_id,
        )
        emitted: list[object] = []
        viewer.canvas_surface_vertex_insertion_requested.connect(emitted.append)
        self.assertTrue(viewer.begin_surface_vertex_placement())

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.005, 1.004),
        ):
            self.assertTrue(
                viewer._begin_surface_vertex_pointer_interaction(QPointF())
            )

        self.assertEqual(
            emitted,
            [
                SurfaceVertexInsertionRequest(
                    surface_id=wall.surface_id,
                    world_point=(2.0, 0.0, 1.0),
                    active_vertex_id=first_id,
                )
            ],
        )
        self.assertTrue(viewer.view.is_primary_pointer_drag_reserved)
        self.assertTrue(
            viewer._finish_surface_vertex_pointer_interaction(QPointF())
        )
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)

    def test_hover_snaps_to_authored_edge_between_existing_vertices(self) -> None:
        wall = _build_wall()
        viewer = self._build_viewer((wall,))
        first_id = "1" * 32
        second_id = "2" * 32
        overlay = SurfaceDrawingOverlay(
            vertices=(
                SurfaceDrawingVertexTarget(
                    vertex_id=first_id,
                    source_surface_id=wall.surface_id,
                    world_point=(1.0, 0.0, 1.0),
                ),
                SurfaceDrawingVertexTarget(
                    vertex_id=second_id,
                    source_surface_id=wall.surface_id,
                    world_point=(3.0, 0.0, 1.0),
                ),
            ),
            edges=(
                SurfaceDrawingEdgeTarget(
                    source_surface_id=wall.surface_id,
                    vertex_ids=(first_id, second_id),
                    world_points=((1.0, 0.0, 1.0), (3.0, 0.0, 1.0)),
                ),
            ),
        )
        viewer.set_canvas_surface_drawing_overlay(overlay)
        self.assertTrue(viewer.begin_surface_vertex_placement())

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.0, 1.005),
        ):
            self.assertTrue(
                viewer._handle_surface_vertex_pointer_hovered(QPointF())
            )

        preview = viewer._surface_vertex_hover_preview
        self.assertIsNotNone(preview)
        assert preview is not None
        self.assertEqual(
            preview.snapped_edge_vertex_ids,
            (first_id, second_id),
        )
        np.testing.assert_allclose(preview.world_point, (2.0, 0.0, 1.0))
        orange_edges = _line_positions_with_color(
            viewer,
            CANVAS_SURFACE_DRAWING_EDGE_COLOR,
        )
        self.assertEqual(len(orange_edges), 1)
        np.testing.assert_allclose(
            orange_edges[0],
            np.asarray(((1.0, 0.0, 1.0), (3.0, 0.0, 1.0))),
        )

        distant_preview = _resolve_canvas_surface_vertex_preview(
            wall,
            (2.0, 0.0, 1.015),
            (wall,),
            tuple(viewer._canvas_surface_drawing_vertices.values()),
            viewer._canvas_surface_drawing_edges,
            None,
        )
        self.assertEqual(distant_preview.snap_kind, "surface")
        self.assertIsNone(distant_preview.snapped_edge_vertex_ids)
        np.testing.assert_allclose(
            distant_preview.world_point,
            (2.0, 0.0, 1.015),
        )

    def test_authored_subedge_wins_over_its_coincident_face_boundary(
        self,
    ) -> None:
        wall = _build_wall()
        viewer = self._build_viewer((wall,))
        boundary_start_id = "1" * 32
        authored_start_id = "2" * 32
        authored_end_id = "3" * 32
        boundary_end_id = "4" * 32
        vertices = tuple(
            SurfaceDrawingVertexTarget(
                vertex_id=vertex_id,
                source_surface_id=wall.surface_id,
                world_point=world_point,
            )
            for vertex_id, world_point in (
                (boundary_start_id, (0.0, 0.0, 0.0)),
                (authored_start_id, (1.0, 0.0, 0.0)),
                (authored_end_id, (3.0, 0.0, 0.0)),
                (boundary_end_id, (4.0, 0.0, 0.0)),
            )
        )
        edges = (
            SurfaceDrawingEdgeTarget(
                source_surface_id=wall.surface_id,
                vertex_ids=(authored_start_id, authored_end_id),
                world_points=((1.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
            ),
        )
        viewer.set_canvas_surface_drawing_overlay(
            SurfaceDrawingOverlay(vertices=vertices, edges=edges)
        )

        preview = _resolve_canvas_surface_vertex_preview(
            wall,
            (2.0, 0.0, 0.0),
            (wall,),
            tuple(viewer._canvas_surface_drawing_vertices.values()),
            viewer._canvas_surface_drawing_edges,
            None,
        )

        self.assertEqual(
            preview.snapped_edge_vertex_ids,
            (authored_start_id, authored_end_id),
        )

    def test_only_unfinished_open_chain_edges_render_while_armed(self) -> None:
        wall = _build_wall()
        viewer = self._build_viewer((wall,))
        first_id = "1" * 32
        second_id = "2" * 32
        vertices = (
            SurfaceDrawingVertexTarget(
                vertex_id=first_id,
                source_surface_id=wall.surface_id,
                world_point=(1.0, 0.0, 1.0),
                show_marker=False,
            ),
            SurfaceDrawingVertexTarget(
                vertex_id=second_id,
                source_surface_id=wall.surface_id,
                world_point=(3.0, 0.0, 1.0),
                show_marker=False,
            ),
        )
        open_chain = SurfaceDrawingOverlay(
            vertices=vertices,
            edges=(
                SurfaceDrawingEdgeTarget(
                    source_surface_id=wall.surface_id,
                    vertex_ids=(first_id, second_id),
                    world_points=((1.0, 0.0, 1.0), (3.0, 0.0, 1.0)),
                ),
            ),
        )
        viewer.set_canvas_surface_drawing_overlay(open_chain)

        self.assertEqual(
            _line_positions_with_color(
                viewer,
                CANVAS_SURFACE_DRAWING_EDGE_COLOR,
            ),
            (),
        )

        self.assertTrue(viewer.begin_surface_vertex_placement())

        orange_edges = _line_positions_with_color(
            viewer,
            CANVAS_SURFACE_DRAWING_EDGE_COLOR,
        )
        self.assertEqual(len(orange_edges), 1)
        np.testing.assert_allclose(
            orange_edges[0],
            np.asarray(((1.0, 0.0, 1.0), (3.0, 0.0, 1.0))),
        )

        viewer.set_canvas_surface_drawing_overlay(
            SurfaceDrawingOverlay(vertices=vertices, edges=())
        )

        self.assertEqual(
            _line_positions_with_color(
                viewer,
                CANVAS_SURFACE_DRAWING_EDGE_COLOR,
            ),
            (),
        )

    def test_edge_cut_refresh_keeps_drawing_endpoint_selection_and_status(
        self,
    ) -> None:
        parent = _build_wall()
        lower = _build_wall(
            "level:0/edit-face:11111111111111111111111111111111:wall",
            minimum_x=0.0,
            maximum_x=2.0,
            source_surface_id=parent.surface_id,
        )
        upper = _build_wall(
            "level:0/edit-face:22222222222222222222222222222222:wall",
            minimum_x=2.0,
            maximum_x=4.0,
            source_surface_id=parent.surface_id,
        )
        viewer = self._build_viewer((parent,))
        first_id = "1" * 32
        second_id = "2" * 32
        overlay = SurfaceDrawingOverlay(
            vertices=(
                SurfaceDrawingVertexTarget(
                    vertex_id=first_id,
                    source_surface_id=parent.surface_id,
                    world_point=(0.0, 0.0, 1.5),
                ),
                SurfaceDrawingVertexTarget(
                    vertex_id=second_id,
                    source_surface_id=parent.surface_id,
                    world_point=(4.0, 0.0, 1.5),
                ),
            ),
            edges=(
                SurfaceDrawingEdgeTarget(
                    source_surface_id=parent.surface_id,
                    vertex_ids=(first_id, second_id),
                    world_points=((0.0, 0.0, 1.5), (4.0, 0.0, 1.5)),
                ),
            ),
        )
        self.assertTrue(viewer.begin_surface_vertex_placement())
        viewer.set_canvas_surface_drawing_overlay(
            overlay,
            active_vertex_id=second_id,
        )
        viewer.set_surface_tools_status("Surface drawing updated.")

        viewer.set_wall_targets((lower, upper))
        viewer.set_canvas_surface_drawing_overlay(
            overlay,
            active_vertex_id=second_id,
        )
        viewer.set_selected_canvas_surface_ids(
            (lower.surface_id, upper.surface_id)
        )
        viewer.set_model(_build_model(), preserve_camera=True)

        self.assertTrue(viewer.is_surface_vertex_placement_active())
        self.assertEqual(viewer.get_active_surface_vertex_id(), second_id)
        self.assertEqual(
            viewer.get_selected_canvas_surface_ids(),
            (lower.surface_id, upper.surface_id),
        )
        self.assertIsNone(viewer._canvas_face_extrusion_target)
        self.assertEqual(
            viewer.surface_tools_status_label.text(),
            "Surface drawing updated.",
        )
        orange_edges = _line_positions_with_color(
            viewer,
            CANVAS_SURFACE_DRAWING_EDGE_COLOR,
        )
        self.assertEqual(len(orange_edges), 1)
        np.testing.assert_allclose(
            orange_edges[0],
            np.asarray(((0.0, 0.0, 1.5), (4.0, 0.0, 1.5))),
        )

    def test_hover_does_not_snap_to_a_hidden_render_triangle_diagonal(
        self,
    ) -> None:
        wall = _build_wall()

        preview = _resolve_canvas_surface_vertex_preview(
            wall,
            (2.0, 0.0, 1.5),
            (wall,),
            (),
            (),
            None,
        )

        self.assertEqual(preview.snap_kind, "surface")
        self.assertIsNone(preview.snapped_edge_vertex_ids)
        np.testing.assert_allclose(preview.world_point, (2.0, 0.0, 1.5))

    def test_add_vertex_picks_visible_floor_and_ceiling_front_faces(
        self,
    ) -> None:
        floor = _build_horizontal_surface(SURFACE_TYPE_FLOOR, 0.0)
        ceiling = _build_horizontal_surface(SURFACE_TYPE_CEILING, 3.0)
        viewer = self._build_viewer((floor, ceiling))
        emitted: list[object] = []
        viewer.canvas_surface_vertex_insertion_requested.connect(emitted.append)
        viewer.add_surface_vertex_button.setChecked(True)

        cases = (
            (
                floor,
                (
                    np.asarray((2.0, 1.5, 5.0), dtype=float),
                    np.asarray((0.0, 0.0, -1.0), dtype=float),
                ),
            ),
            (
                ceiling,
                (
                    np.asarray((2.0, 1.5, -2.0), dtype=float),
                    np.asarray((0.0, 0.0, 1.0), dtype=float),
                ),
            ),
        )
        for surface, camera_ray in cases:
            with self.subTest(surface_type=surface.surface_type):
                ordinary_hit = _get_nearest_fixed_surface_ray_hit(
                    (floor, ceiling),
                    *camera_ray,
                )
                self.assertIsNotNone(ordinary_hit)
                assert ordinary_hit is not None
                self.assertNotEqual(ordinary_hit[0].surface_id, surface.surface_id)

                with patch.object(
                    viewer.view,
                    "build_camera_ray",
                    return_value=camera_ray,
                ):
                    self.assertTrue(
                        viewer._handle_surface_vertex_pointer_hovered(QPointF())
                    )
                    preview = viewer._surface_vertex_hover_preview
                    self.assertIsNotNone(preview)
                    assert preview is not None
                    self.assertEqual(preview.surface_id, surface.surface_id)
                    self.assertTrue(
                        viewer._begin_surface_vertex_pointer_interaction(
                            QPointF()
                        )
                    )

                self.assertEqual(
                    emitted[-1],
                    SurfaceVertexInsertionRequest(
                        surface_id=surface.surface_id,
                        world_point=(2.0, 1.5, float(surface.mesh.centroid[2])),
                    ),
                )
                self.assertEqual(
                    viewer.get_selected_canvas_surface_ids(),
                    (surface.surface_id,),
                )
                self.assertTrue(
                    viewer._finish_surface_vertex_pointer_interaction(QPointF())
                )

    def test_active_angle_preview_near_boundary_stays_on_the_surface(self) -> None:
        wall = _build_wall()

        snapped = _snap_surface_point_from_active_vertex(
            wall,
            np.asarray((0.011, 0.0, 0.06), dtype=float),
            np.asarray((0.02, 0.0, 0.02), dtype=float),
            (wall,),
        )

        self.assertIsNotNone(snapped)
        assert snapped is not None
        self.assertGreaterEqual(float(snapped[0]), 0.0)
        self.assertGreaterEqual(float(snapped[2]), 0.0)

    def test_active_angle_snap_rejects_over_one_centimeter_displacement(
        self,
    ) -> None:
        wall = _build_wall()

        close = _snap_surface_point_from_active_vertex(
            wall,
            np.asarray((1.005, 0.0, 2.0), dtype=float),
            np.asarray((1.0, 0.0, 1.0), dtype=float),
            (wall,),
        )
        distant = _snap_surface_point_from_active_vertex(
            wall,
            np.asarray((1.015, 0.0, 2.0), dtype=float),
            np.asarray((1.0, 0.0, 1.0), dtype=float),
            (wall,),
        )

        self.assertIsNotNone(close)
        assert close is not None
        np.testing.assert_allclose(close, (1.0, 0.0, 2.0))
        self.assertIsNone(distant)

    def test_disarming_add_vertex_keeps_small_manual_marker(self) -> None:
        wall = _build_wall()
        viewer = self._build_viewer((wall,))
        vertex_id = "1" * 32
        overlay = SurfaceDrawingOverlay(
            vertices=(
                SurfaceDrawingVertexTarget(
                    vertex_id=vertex_id,
                    source_surface_id=wall.surface_id,
                    world_point=(1.0, 0.0, 1.0),
                ),
            ),
            edges=(),
        )
        viewer.set_canvas_surface_drawing_overlay(
            overlay,
            active_vertex_id=vertex_id,
        )
        reset_requests: list[bool] = []
        viewer.canvas_surface_vertex_chain_reset_requested.connect(
            lambda: reset_requests.append(True)
        )
        self.assertTrue(viewer.begin_surface_vertex_placement())

        viewer.cancel_surface_vertex_placement()

        self.assertIsNone(viewer.get_active_surface_vertex_id())
        blue_markers = _scatter_items_with_color(
            viewer,
            CANVAS_SURFACE_DRAWING_VERTEX_COLOR,
        )
        self.assertEqual(len(blue_markers), 1)
        marker = blue_markers[0]
        np.testing.assert_allclose(
            np.asarray(getattr(marker, "pos"), dtype=float),
            np.asarray(((1.0, 0.0, 1.0),), dtype=float),
        )
        self.assertEqual(float(getattr(marker, "size")), 9.0)
        self.assertTrue(bool(getattr(marker, "pxMode")))
        self.assertEqual(reset_requests, [True])

    def test_extrusion_is_available_only_after_add_vertex_is_disarmed(
        self,
    ) -> None:
        source_id = "level:0/wall:1:2"
        child = _build_wall(
            "level:0/edit-face:11111111111111111111111111111111:wall",
            source_surface_id=source_id,
        )
        viewer = self._build_viewer((child,))
        viewer.select_canvas_surface_target(child.surface_id)

        self.assertIsNotNone(viewer._canvas_face_extrusion_target)
        self.assertTrue(viewer._canvas_face_extrusion_gizmo_items)

        viewer.add_surface_vertex_button.setChecked(True)

        self.assertTrue(viewer.is_surface_vertex_placement_active())
        self.assertEqual(
            viewer.get_selected_canvas_surface_ids(),
            (child.surface_id,),
        )
        self.assertIsNone(viewer._canvas_face_extrusion_target)
        self.assertEqual(viewer._canvas_face_extrusion_gizmo_items, [])

        viewer.add_surface_vertex_button.setChecked(False)

        self.assertFalse(viewer.is_surface_vertex_placement_active())
        self.assertEqual(
            viewer.get_selected_canvas_surface_ids(),
            (child.surface_id,),
        )
        target = viewer._canvas_face_extrusion_target
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.surface_ids, (child.surface_id,))
        self.assertTrue(viewer._canvas_face_extrusion_gizmo_items)

        emitted: list[object] = []
        viewer.canvas_surface_face_extrusion_requested.connect(emitted.append)
        with (
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=_ray_at(2.0, 1.5),
            ),
            patch(
                "housemaker.viewer._intersect_ray_with_plane",
                side_effect=(
                    np.asarray(target.center_world, dtype=float),
                    np.asarray(target.center_world, dtype=float)
                    + np.asarray(target.normal_world, dtype=float) * 0.5,
                ),
            ),
        ):
            self.assertTrue(
                viewer._begin_canvas_face_extrusion_drag(target, QPointF())
            )
            self.assertTrue(
                viewer._finish_canvas_face_extrusion_drag(QPointF())
            )

        self.assertEqual(
            emitted,
            [SurfaceFaceExtrusionRequest((child.surface_id,), 0.5)],
        )

    def test_connected_coplanar_children_share_one_depth_free_normal_handle(
        self,
    ) -> None:
        source_id = "level:0/wall:1:2"
        first = _build_wall(
            "level:0/edit-face:11111111111111111111111111111111:wall",
            minimum_x=0.0,
            maximum_x=2.0,
            source_surface_id=source_id,
        )
        second = _build_wall(
            "level:0/edit-face:22222222222222222222222222222222:wall",
            minimum_x=2.0,
            maximum_x=4.0,
            source_surface_id=source_id,
        )
        viewer = self._build_viewer((first, second))

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            viewer.select_canvas_surface_target(first.surface_id)
            viewer.select_canvas_surface_target(second.surface_id, additive=True)

        target = viewer._canvas_face_extrusion_target
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.surface_ids, (first.surface_id, second.surface_id))
        np.testing.assert_allclose(target.center_world, (2.0, 0.0, 1.5))
        self.assertEqual(len(viewer._canvas_face_extrusion_gizmo_items), 2)
        for item in viewer._canvas_face_extrusion_gizmo_items:
            self.assertEqual(item.depthValue(), CANVAS_OPENING_OVERLAY_DEPTH_VALUE)
            gl_options = getattr(item, "_GLGraphicsItem__glOpts")
            self.assertFalse(gl_options[GL.GL_DEPTH_TEST])

    def test_extrusion_drag_previews_and_emits_one_signed_request(self) -> None:
        source_id = "level:0/wall:1:2"
        child = _build_wall(
            "level:0/edit-face:11111111111111111111111111111111:wall",
            source_surface_id=source_id,
        )
        viewer = self._build_viewer((child,))
        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            viewer.select_canvas_surface_target(child.surface_id)
        target = viewer._canvas_face_extrusion_target
        self.assertIsNotNone(target)
        assert target is not None
        emitted: list[object] = []
        viewer.canvas_surface_face_extrusion_requested.connect(emitted.append)

        with (
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=_ray_at(2.0, 1.5),
            ),
            patch(
                "housemaker.viewer._intersect_ray_with_plane",
                side_effect=(
                    np.asarray(target.center_world, dtype=float),
                    np.asarray(target.center_world, dtype=float)
                    + np.asarray(target.normal_world, dtype=float) * 0.5,
                ),
            ),
        ):
            self.assertTrue(
                viewer._begin_canvas_face_extrusion_drag(target, QPointF())
            )
            self.assertTrue(
                viewer._finish_canvas_face_extrusion_drag(QPointF())
            )

        self.assertEqual(
            emitted,
            [SurfaceFaceExtrusionRequest((child.surface_id,), 0.5)],
        )
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)
        self.assertIsNone(viewer._canvas_face_extrusion_drag)

    def test_disconnected_child_selection_keeps_selection_without_a_gizmo(
        self,
    ) -> None:
        source_id = "level:0/wall:1:2"
        first = _build_wall(
            "level:0/edit-face:11111111111111111111111111111111:wall",
            minimum_x=0.0,
            maximum_x=1.0,
            source_surface_id=source_id,
        )
        second = _build_wall(
            "level:0/edit-face:22222222222222222222222222222222:wall",
            minimum_x=3.0,
            maximum_x=4.0,
            source_surface_id=source_id,
        )
        viewer = self._build_viewer((first, second))

        viewer.select_canvas_surface_target(first.surface_id)
        viewer.select_canvas_surface_target(second.surface_id, additive=True)

        self.assertEqual(
            viewer.get_selected_canvas_surface_ids(),
            (first.surface_id, second.surface_id),
        )
        self.assertIsNone(viewer._canvas_face_extrusion_target)
        self.assertEqual(viewer._canvas_face_extrusion_gizmo_items, [])
        assert viewer.surface_tools_status_label is not None
        self.assertIn(
            "connected coplanar faces",
            viewer.surface_tools_status_label.text(),
        )

    def test_external_surface_status_survives_a_target_refresh(self) -> None:
        wall = _build_wall()
        viewer = self._build_viewer((wall,))

        viewer.set_surface_tools_status("Surface edit stopped: test failure")
        viewer.set_wall_targets((wall,))

        assert viewer.surface_tools_status_label is not None
        self.assertEqual(
            viewer.surface_tools_status_label.text(),
            "Surface edit stopped: test failure",
        )

    def test_only_a_manual_surface_selection_change_clears_external_status(
        self,
    ) -> None:
        first = _build_wall("level:0/wall:1:2")
        second = _build_wall("level:0/wall:2:3")
        viewer = self._build_viewer((first, second))
        viewer.set_selected_canvas_surface_ids((first.surface_id,))
        viewer.set_surface_tools_status("Faces extruded.")

        viewer.set_selected_canvas_surface_ids((second.surface_id,))
        assert viewer.surface_tools_status_label is not None
        self.assertEqual(
            viewer.surface_tools_status_label.text(),
            "Faces extruded.",
        )
        viewer.select_canvas_surface_target(second.surface_id)
        self.assertEqual(
            viewer.surface_tools_status_label.text(),
            "Faces extruded.",
        )

        viewer.select_canvas_surface_target(first.surface_id)

        self.assertEqual(
            viewer.surface_tools_status_label.text(),
            "Click Add vertex to draw vertices, split a surface between "
            "two edges, or close a face.",
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
