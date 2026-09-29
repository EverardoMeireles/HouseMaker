# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PySide6.QtCore import QPointF
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from housemaker.architectural_trim import (
    TRIM_HANDLE_CORNER_RADIUS,
    TRIM_HANDLE_DEPTH,
    TRIM_HANDLE_HEIGHT,
    TRIM_PART_FRONT,
    TRIM_SURFACE_TYPE,
    ArchitecturalTrimEditHandle,
    ArchitecturalTrimEditTarget,
    ArchitecturalTrimPart,
    ArchitecturalTrimPlacementRequest,
    build_architectural_trim_placement_preview_meshes,
    build_architectural_trim_surface_id,
)
from housemaker.glb import GeneratedModel
from housemaker.models import (
    TRIM_KIND_CORNICE,
    TRIM_KIND_EDGING_STRIP,
    TRIM_KIND_SKIRTING_BOARD,
)
from housemaker.surface_geometry import SURFACE_TYPE_WALL, FixedSurface
from housemaker.viewer import (
    ArchitecturalTrimDimensionEdit,
    GlbViewerWidget,
    _get_architectural_trim_handle_value_bounds,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)
_TRIM_ID = "12345678123456781234567812345678"


# ### Fixture helpers ###
def _build_wall(
    surface_id: str,
    wall_key: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    *,
    height: float = 3.0,
    reverse_faces: bool = False,
) -> FixedSurface:
    start_point = np.asarray(start, dtype=float)
    end_point = np.asarray(end, dtype=float)
    vertices = np.asarray(
        (
            start_point,
            end_point,
            end_point + (0.0, 0.0, height),
            start_point + (0.0, 0.0, height),
        ),
        dtype=float,
    )
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    if reverse_faces:
        faces = faces[:, ::-1]
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )
    return FixedSurface(
        surface_id=surface_id,
        surface_type=SURFACE_TYPE_WALL,
        level_index=2,
        room_index=None,
        mesh=mesh,
        area_square_meters=float(np.linalg.norm(end_point - start_point) * height),
        wall_key=wall_key,
        wall_start_world=tuple(start_point),
        wall_end_world=tuple(end_point),
        wall_height_meters=height,
    )


def _build_walls() -> tuple[FixedSurface, FixedSurface]:
    return (
        _build_wall(
            "level:2/wall:1:2",
            "1:2",
            (-2.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        ),
        _build_wall(
            "level:2/wall:2:3",
            "2:3",
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
        ),
    )


def _build_model(*walls: FixedSurface) -> GeneratedModel:
    mesh = trimesh.util.concatenate(tuple(wall.mesh for wall in walls))
    return GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b"")


def _build_trim_part() -> ArchitecturalTrimPart:
    mesh = trimesh.creation.box(extents=(2.0, 0.04, 0.12))
    mesh.apply_translation((0.0, -0.02, 0.06))
    semantic_id = build_architectural_trim_surface_id(
        _TRIM_ID,
        TRIM_PART_FRONT,
    )
    return ArchitecturalTrimPart(
        trim_id=_TRIM_ID,
        run_id=_TRIM_ID,
        semantic_id=semantic_id,
        part_kind=TRIM_PART_FRONT,
        surface_type=TRIM_SURFACE_TYPE,
        mesh=mesh,
        level_index=2,
        source_wall_surface_id="level:2/wall:1:2",
    )


def _build_edit_target(part: ArchitecturalTrimPart) -> ArchitecturalTrimEditTarget:
    return ArchitecturalTrimEditTarget(
        trim_id=part.trim_id,
        run_id=part.run_id,
        semantic_ids=(part.semantic_id,),
        kind=TRIM_KIND_SKIRTING_BOARD,
        level_index=part.level_index,
        anchor_world=(0.0, -0.04, 0.08),
        handles=(
            ArchitecturalTrimEditHandle(
                handle_kind=TRIM_HANDLE_HEIGHT,
                origin_world=(0.0, -0.04, 0.08),
                axis_world=(0.0, 0.0, 1.0),
                value_meters=0.10,
            ),
            ArchitecturalTrimEditHandle(
                handle_kind=TRIM_HANDLE_DEPTH,
                origin_world=(0.0, -0.04, 0.08),
                axis_world=(0.0, -1.0, 0.0),
                value_meters=0.04,
            ),
            ArchitecturalTrimEditHandle(
                handle_kind=TRIM_HANDLE_CORNER_RADIUS,
                origin_world=(0.0, -0.04, 0.08),
                axis_world=(0.0, 0.0, 1.0),
                value_meters=0.0,
            ),
        ),
    )


def _cornice_sloped_face_mask(mesh: trimesh.Trimesh) -> np.ndarray:
    """Select the visible profile faces spanning wall depth and ceiling height."""

    normals = np.asarray(mesh.face_normals, dtype=float)
    return (
        (np.abs(normals[:, 0]) < 0.25)
        & (np.abs(normals[:, 1]) > 0.08)
        & (np.abs(normals[:, 2]) > 0.08)
    )


def _cornice_slope_normal_count(mesh: trimesh.Trimesh) -> int:
    normals = np.asarray(mesh.face_normals, dtype=float)
    return len(
        {
            (round(float(normal[1]), 6), round(float(normal[2]), 6))
            for normal in normals[_cornice_sloped_face_mask(mesh)]
        }
    )


def _cornice_sloped_profile_extents(
    mesh: trimesh.Trimesh,
) -> tuple[float, float, float, float]:
    face_indices = np.flatnonzero(_cornice_sloped_face_mask(mesh))
    if not len(face_indices):
        return (0.0, 0.0, 0.0, 0.0)
    vertex_indices = np.unique(np.asarray(mesh.faces)[face_indices].reshape(-1))
    vertices = np.asarray(mesh.vertices, dtype=float)[vertex_indices]
    outward = -vertices[:, 1]
    downward = float(np.max(np.asarray(mesh.vertices)[:, 2])) - vertices[:, 2]
    return (
        float(np.min(outward)),
        float(np.max(outward)),
        float(np.min(downward)),
        float(np.max(downward)),
    )


# ### Viewer interaction tests ###
class ArchitecturalTrimViewerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.walls = _build_walls()
        self.viewer = GlbViewerWidget(window_editing_enabled=True)
        self.viewer.set_model(_build_model(*self.walls))
        self.viewer.set_wall_targets(self.walls)

    def tearDown(self) -> None:
        self.viewer.close()
        self.viewer.deleteLater()
        _qt_application.processEvents()

    def test_side_panel_modes_are_exclusive_and_require_visible_walls(self) -> None:
        buttons = self.viewer.architectural_trim_mode_buttons
        self.assertEqual(
            set(buttons),
            {
                TRIM_KIND_SKIRTING_BOARD,
                TRIM_KIND_EDGING_STRIP,
                TRIM_KIND_CORNICE,
            },
        )
        self.assertTrue(all(button.isEnabled() for button in buttons.values()))

        self.assertTrue(
            self.viewer.begin_architectural_trim_placement(TRIM_KIND_SKIRTING_BOARD)
        )
        self.assertTrue(buttons[TRIM_KIND_SKIRTING_BOARD].isChecked())
        self.assertTrue(
            self.viewer.begin_architectural_trim_placement(TRIM_KIND_CORNICE)
        )
        self.assertFalse(buttons[TRIM_KIND_SKIRTING_BOARD].isChecked())
        self.assertTrue(buttons[TRIM_KIND_CORNICE].isChecked())

    def test_passive_hover_exposes_only_location_appropriate_trim_actions(
        self,
    ) -> None:
        hover_cases = (
            (
                TRIM_KIND_SKIRTING_BOARD,
                (
                    np.asarray((0.0, -2.0, 0.03), dtype=float),
                    np.asarray((0.0, 1.0, 0.0), dtype=float),
                ),
            ),
            (
                TRIM_KIND_CORNICE,
                (
                    np.asarray((0.0, -2.0, 2.97), dtype=float),
                    np.asarray((0.0, 1.0, 0.0), dtype=float),
                ),
            ),
            (
                TRIM_KIND_EDGING_STRIP,
                (
                    np.asarray((1.99, -2.0, 1.0), dtype=float),
                    np.asarray((0.0, 1.0, 0.0), dtype=float),
                ),
            ),
        )

        for expected_kind, camera_ray in hover_cases:
            with (
                self.subTest(kind=expected_kind),
                patch.object(
                    self.viewer.view,
                    "build_camera_ray",
                    return_value=camera_ray,
                ),
                patch.object(
                    self.viewer,
                    "_get_architectural_trim_snap_tolerance",
                    return_value=0.05,
                ),
            ):
                self.assertFalse(self.viewer.is_architectural_trim_placement_active)
                self.viewer._update_architectural_trim_passive_hover(QPointF(32, 48))

                action_buttons = self.viewer.architectural_trim_hover_action_buttons
                self.assertEqual(
                    set(action_buttons),
                    {
                        TRIM_KIND_SKIRTING_BOARD,
                        TRIM_KIND_EDGING_STRIP,
                        TRIM_KIND_CORNICE,
                    },
                )
                self.assertTrue(action_buttons[expected_kind].isEnabled())
                self.assertFalse(action_buttons[expected_kind].isHidden())
                self.assertTrue(self.viewer._architectural_trim_preview_items)
                self.assertTrue(
                    all(
                        button.isHidden()
                        for kind, button in action_buttons.items()
                        if kind != expected_kind
                    )
                )

    def test_passive_hover_action_inserts_without_arming_persistent_mode(
        self,
    ) -> None:
        camera_ray = (
            np.asarray((0.0, -2.0, 0.03), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
        )
        emitted = QSignalSpy(self.viewer.architectural_trim_placement_requested)
        with (
            patch.object(
                self.viewer.view,
                "build_camera_ray",
                return_value=camera_ray,
            ),
            patch.object(
                self.viewer,
                "_get_architectural_trim_snap_tolerance",
                return_value=0.05,
            ),
        ):
            self.viewer._update_architectural_trim_passive_hover(QPointF(32, 48))
            action_button = self.viewer.architectural_trim_hover_action_buttons[
                TRIM_KIND_SKIRTING_BOARD
            ]
            self.assertFalse(action_button.isHidden())
            action_button.click()

        self.assertEqual(emitted.count(), 1)
        request = emitted.at(0)[0]
        self.assertEqual(request.kind, TRIM_KIND_SKIRTING_BOARD)
        self.assertEqual(request.wall_surface_ids, (self.walls[0].surface_id,))
        self.assertFalse(self.viewer.is_architectural_trim_placement_active)
        self.assertTrue(
            all(
                button.isHidden()
                for button in self.viewer.architectural_trim_hover_action_buttons.values()
            )
        )
        self.assertFalse(self.viewer._architectural_trim_preview_items)

    def test_derived_editable_wall_faces_are_not_trim_hosts(self) -> None:
        source_wall = self.walls[0]
        derived_wall = replace(
            source_wall,
            surface_id="editable:face:wall",
            source_surface_id=source_wall.surface_id,
        )

        self.viewer.set_wall_targets((derived_wall,))

        self.assertTrue(
            all(
                not button.isEnabled()
                for button in self.viewer.architectural_trim_mode_buttons.values()
            )
        )

    def test_skirting_and_cornice_snap_only_to_their_wall_edge(self) -> None:
        bottom_ray = (
            np.asarray((0.0, -2.0, 0.03), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
        )
        top_ray = (
            np.asarray((0.0, -2.0, 2.97), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
        )
        with patch.object(
            self.viewer,
            "_get_architectural_trim_snap_tolerance",
            return_value=0.05,
        ):
            self.viewer.begin_architectural_trim_placement(TRIM_KIND_SKIRTING_BOARD)
            bottom = self.viewer._resolve_architectural_trim_hover_candidate(
                *bottom_ray
            )
            self.assertIsNotNone(bottom)
            self.assertIsNone(
                self.viewer._resolve_architectural_trim_hover_candidate(*top_ray)
            )

            self.viewer.begin_architectural_trim_placement(TRIM_KIND_CORNICE)
            top = self.viewer._resolve_architectural_trim_hover_candidate(*top_ray)
            self.assertIsNotNone(top)
            self.assertIsNone(
                self.viewer._resolve_architectural_trim_hover_candidate(*bottom_ray)
            )
        assert bottom is not None and top is not None
        self.assertEqual(bottom.request.wall_surface_ids, (self.walls[0].surface_id,))
        self.assertEqual(top.request.kind, TRIM_KIND_CORNICE)
        self.assertTrue(bottom.preview_meshes)

    def test_cornice_hover_preview_uses_a_straight_profile_by_default(self) -> None:
        self.viewer.begin_architectural_trim_placement(TRIM_KIND_CORNICE)
        ray = (
            np.asarray((0.0, -2.0, 2.97), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
        )
        with patch.object(
            self.viewer,
            "_get_architectural_trim_snap_tolerance",
            return_value=0.05,
        ):
            candidate = self.viewer._resolve_architectural_trim_hover_candidate(*ray)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        preview = candidate.preview_meshes[0]
        minimum_outward, maximum_outward, minimum_down, maximum_down = (
            _cornice_sloped_profile_extents(preview)
        )
        depth = candidate.request.depth_meters
        height = candidate.request.height_meters

        self.assertEqual(_cornice_slope_normal_count(preview), 1)
        self.assertLessEqual(minimum_outward, depth * 0.25)
        self.assertGreaterEqual(maximum_outward, depth * 0.95)
        self.assertLessEqual(minimum_down, height * 0.05)
        self.assertGreaterEqual(maximum_down, height * 0.95)
        self.assertAlmostEqual(float(preview.bounds[1, 2]), 3.0)

    def test_cornice_preview_rounds_when_given_a_positive_radius(self) -> None:
        request = ArchitecturalTrimPlacementRequest(
            kind=TRIM_KIND_CORNICE,
            wall_surface_ids=(self.walls[0].surface_id,),
            height_meters=0.24,
            depth_meters=0.12,
            corner_radius_meters=0.04,
        )

        previews = build_architectural_trim_placement_preview_meshes(
            request,
            (self.walls[0],),
        )

        self.assertEqual(len(previews), 1)
        preview = previews[0]
        minimum_outward, maximum_outward, minimum_down, maximum_down = (
            _cornice_sloped_profile_extents(preview)
        )
        depth = request.depth_meters
        height = request.height_meters

        self.assertGreater(len(preview.vertices), 8)
        self.assertGreaterEqual(_cornice_slope_normal_count(preview), 4)
        self.assertLessEqual(minimum_outward, depth * 0.25)
        self.assertGreaterEqual(maximum_outward, depth * 0.95)
        self.assertLessEqual(minimum_down, height * 0.05)
        self.assertGreaterEqual(maximum_down, height * 0.95)
        self.assertAlmostEqual(float(preview.bounds[1, 2]), 3.0)

    def test_edging_strip_snaps_to_shared_right_angle_vertex(self) -> None:
        self.viewer.begin_architectural_trim_placement(TRIM_KIND_EDGING_STRIP)
        ray = (
            np.asarray((1.99, -2.0, 1.0), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
        )
        with patch.object(
            self.viewer,
            "_get_architectural_trim_snap_tolerance",
            return_value=0.05,
        ):
            candidate = self.viewer._resolve_architectural_trim_hover_candidate(*ray)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.request.kind, TRIM_KIND_EDGING_STRIP)
        self.assertEqual(candidate.request.corner_vertex_id, 2)
        self.assertEqual(candidate.request.height_meters, 3.0)
        self.assertEqual(
            set(candidate.request.wall_surface_ids),
            {wall.surface_id for wall in self.walls},
        )

    def test_edging_strip_hover_preview_wraps_both_walls(self) -> None:
        self.viewer.begin_architectural_trim_placement(TRIM_KIND_EDGING_STRIP)
        ray = (
            np.asarray((1.99, -2.0, 1.0), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
        )
        with patch.object(
            self.viewer,
            "_get_architectural_trim_snap_tolerance",
            return_value=0.05,
        ):
            candidate = self.viewer._resolve_architectural_trim_hover_candidate(*ray)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        preview = candidate.preview_meshes[0]
        footprint = np.unique(
            np.round(np.asarray(preview.vertices, dtype=float)[:, :2], decimals=8),
            axis=0,
        )
        width = candidate.request.width_meters
        first_wall_contacts = footprint[np.abs(footprint[:, 1]) <= 1e-6]
        second_wall_contacts = footprint[np.abs(footprint[:, 0] - 2.0) <= 1e-6]

        self.assertGreaterEqual(len(footprint), 6)
        self.assertGreaterEqual(len(first_wall_contacts), 2)
        self.assertGreaterEqual(len(second_wall_contacts), 2)
        self.assertGreater(
            float(np.ptp(first_wall_contacts[:, 0])),
            width * 0.75,
        )
        self.assertGreater(
            float(np.ptp(second_wall_contacts[:, 1])),
            width * 0.75,
        )

    def test_edging_strip_rejects_the_inner_face_pair(self) -> None:
        inward_walls = (
            _build_wall(
                "level:2/wall:4:5",
                "4:5",
                (-2.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                reverse_faces=True,
            ),
            _build_wall(
                "level:2/wall:5:6",
                "5:6",
                (2.0, 0.0, 0.0),
                (2.0, 2.0, 0.0),
                reverse_faces=True,
            ),
        )
        self.viewer.set_model(_build_model(*inward_walls))
        self.viewer.set_wall_targets(inward_walls)
        self.viewer.begin_architectural_trim_placement(TRIM_KIND_EDGING_STRIP)
        ray = (
            np.asarray((1.99, -2.0, 1.0), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
        )
        with patch.object(
            self.viewer,
            "_get_architectural_trim_snap_tolerance",
            return_value=0.05,
        ):
            candidate = self.viewer._resolve_architectural_trim_hover_candidate(*ray)

        self.assertIsNone(candidate)

    def test_primary_click_emits_the_retained_hover_request(self) -> None:
        self.viewer.begin_architectural_trim_placement(TRIM_KIND_SKIRTING_BOARD)
        ray = (
            np.asarray((0.0, -2.0, 0.03), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
        )
        emitted = QSignalSpy(self.viewer.architectural_trim_placement_requested)
        with (
            patch.object(self.viewer.view, "build_camera_ray", return_value=ray),
            patch.object(
                self.viewer,
                "_get_architectural_trim_snap_tolerance",
                return_value=0.05,
            ),
        ):
            self.viewer._handle_placed_object_pointer_pressed(QPointF())
            self.viewer._handle_canvas_gizmo_pointer_released(QPointF())

        self.assertEqual(emitted.count(), 1)
        request = emitted.at(0)[0]
        self.assertEqual(request.kind, TRIM_KIND_SKIRTING_BOARD)
        self.assertTrue(self.viewer.is_architectural_trim_placement_active)

    def test_camera_preserving_scene_refresh_rearms_insertion_mode(self) -> None:
        self.assertTrue(
            self.viewer.begin_architectural_trim_placement(TRIM_KIND_SKIRTING_BOARD)
        )

        self.viewer.set_model(
            _build_model(*self.walls),
            preserve_camera=True,
        )

        self.assertTrue(self.viewer.is_architectural_trim_placement_active)
        self.assertTrue(
            self.viewer.architectural_trim_mode_buttons[
                TRIM_KIND_SKIRTING_BOARD
            ].isChecked()
        )

        self.viewer.set_model(
            _build_model(*self.walls),
            preserve_camera=False,
        )
        self.assertFalse(self.viewer.is_architectural_trim_placement_active)

    def test_repeated_edit_preview_reuses_same_cardinality_mesh_items(self) -> None:
        """Drag frames should update existing GL items instead of recreating them."""

        part = _build_trim_part()
        self.assertTrue(self.viewer.set_architectural_trim_edit_preview_parts((part,)))
        self.assertEqual(len(self.viewer._architectural_trim_edit_preview_items), 1)
        original_item = self.viewer._architectural_trim_edit_preview_items[0]
        revised_mesh = part.mesh.copy()
        revised_mesh.apply_scale((1.0, 1.0, 1.25))
        revised_part = replace(part, mesh=revised_mesh)

        with (
            patch.object(self.viewer.view, "addItem") as add_item,
            patch.object(self.viewer.view, "removeItem") as remove_item,
            patch.object(
                original_item,
                "setMeshData",
                wraps=original_item.setMeshData,
            ) as set_mesh_data,
        ):
            self.assertTrue(
                self.viewer.set_architectural_trim_edit_preview_parts((revised_part,))
            )

        add_item.assert_not_called()
        remove_item.assert_not_called()
        set_mesh_data.assert_called_once()
        self.assertEqual(len(self.viewer._architectural_trim_edit_preview_items), 1)
        self.assertIs(
            self.viewer._architectural_trim_edit_preview_items[0],
            original_item,
        )

    def test_selection_highlight_and_delete_use_semantic_trim_owners(self) -> None:
        part = _build_trim_part()
        self.viewer.set_architectural_trim_parts((part,))
        selected = QSignalSpy(self.viewer.architectural_trim_part_selection_changed)
        deleted = QSignalSpy(self.viewer.architectural_trim_deletion_requested)

        self.assertTrue(self.viewer.select_architectural_trim_part(part.semantic_id))
        self.assertEqual(selected.count(), 1)
        self.assertTrue(self.viewer._architectural_trim_selection_items)
        self.assertTrue(
            self.viewer.set_highlighted_architectural_trim_part_ids((part.semantic_id,))
        )
        self.assertTrue(self.viewer._atlas_architectural_trim_part_highlight_items)

        self.viewer._handle_view_delete_requested()
        self.assertEqual(deleted.count(), 1)
        self.assertEqual(deleted.at(0)[0], (part.trim_id,))
        self.assertEqual(
            self.viewer.get_selected_architectural_trim_part_ids(),
            (),
        )

    def test_dimension_gizmo_emits_clamped_absolute_edits(self) -> None:
        part = _build_trim_part()
        target = _build_edit_target(part)
        self.viewer.set_architectural_trim_parts((part,))
        self.viewer.set_architectural_trim_edit_targets((target,))
        self.viewer.select_architectural_trim_part(part.semantic_id)
        self.assertEqual(len(self.viewer._architectural_trim_gizmo_items), 6)

        started = QSignalSpy(self.viewer.architectural_trim_edit_started)
        previews = QSignalSpy(self.viewer.architectural_trim_edit_preview_changed)
        finished = QSignalSpy(self.viewer.architectural_trim_edit_finished)
        start_ray = (
            np.asarray((0.0, -2.0, 0.08), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
        )
        moved_ray = (
            np.asarray((0.0, -2.0, 0.18), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
        )
        second_moved_ray = (
            np.asarray((0.0, -2.0, 0.23), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
        )
        height_handle = target.handles[0]
        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=start_ray,
        ):
            self.assertTrue(
                self.viewer._begin_architectural_trim_edit_drag(
                    target,
                    height_handle,
                    QPointF(),
                )
            )
        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=moved_ray,
        ):
            self.assertTrue(self.viewer._update_architectural_trim_edit_drag(QPointF()))

        revised_mesh = part.mesh.copy()
        revised_mesh.apply_scale((1.0, 1.0, 1.1))
        revised_part = replace(part, mesh=revised_mesh)
        revised_target = replace(
            target,
            anchor_world=(0.0, -0.04, 0.13),
            handles=tuple(
                replace(handle, origin_world=(0.0, -0.04, 0.13))
                for handle in target.handles
            ),
        )
        self.viewer.set_architectural_trim_edit_targets((revised_target,))
        self.viewer.set_architectural_trim_parts((revised_part,))
        self.assertIsNotNone(self.viewer._architectural_trim_edit_drag)
        self.assertTrue(self.viewer._architectural_trim_edit_preview_items)

        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=second_moved_ray,
        ):
            self.assertTrue(self.viewer._update_architectural_trim_edit_drag(QPointF()))
            self.assertTrue(self.viewer._finish_architectural_trim_edit_drag(QPointF()))

        self.assertEqual(started.count(), 1)
        self.assertEqual(previews.count(), 2)
        self.assertEqual(finished.count(), 1)
        edit = finished.at(0)[0]
        self.assertIsInstance(edit, ArchitecturalTrimDimensionEdit)
        self.assertAlmostEqual(edit.value_meters, 0.25)
        self.assertTrue(finished.at(0)[1])

        radius_handle = target.handles[2]
        self.assertEqual(
            _get_architectural_trim_handle_value_bounds(
                target,
                radius_handle,
            ),
            (0.0, 0.02),
        )

        rounded_target = replace(
            target,
            handles=(
                target.handles[0],
                target.handles[1],
                replace(target.handles[2], value_meters=0.015),
            ),
        )
        self.assertAlmostEqual(
            _get_architectural_trim_handle_value_bounds(
                rounded_target,
                rounded_target.handles[0],
            )[0],
            0.03,
        )
        self.assertAlmostEqual(
            _get_architectural_trim_handle_value_bounds(
                rounded_target,
                rounded_target.handles[1],
            )[0],
            0.03,
        )

    def test_ctrl_z_cancels_an_active_dimension_drag_before_history(self) -> None:
        part = _build_trim_part()
        target = _build_edit_target(part)
        self.viewer.set_architectural_trim_parts((part,))
        self.viewer.set_architectural_trim_edit_targets((target,))
        self.viewer.select_architectural_trim_part(part.semantic_id)
        cancelled = QSignalSpy(self.viewer.architectural_trim_edit_cancelled)
        start_ray = (
            np.asarray((0.0, -2.0, 0.08), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
        )
        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=start_ray,
        ):
            self.assertTrue(
                self.viewer._begin_architectural_trim_edit_drag(
                    target,
                    target.handles[0],
                    QPointF(),
                )
            )

        self.assertTrue(self.viewer.cancel_uncommitted_canvas_interaction_for_undo())
        self.assertIsNone(self.viewer._architectural_trim_edit_drag)
        self.assertFalse(self.viewer._architectural_trim_edit_preview_items)
        self.assertEqual(cancelled.count(), 1)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
