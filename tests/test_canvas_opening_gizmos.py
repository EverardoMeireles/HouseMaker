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
from OpenGL import GL
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from housemaker.canvas_openings import (
    CANVAS_OPENING_DOORWAY,
    CANVAS_OPENING_WINDOW,
    CanvasOpeningBounds,
    CanvasOpeningReference,
    CanvasOpeningTarget,
)
from housemaker.door_state import (
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
)
from housemaker.glb import GeneratedModel, PreviewPlacedObject
from housemaker.surface_geometry import SURFACE_TYPE_WALL, FixedSurface
from housemaker.viewer import (
    CANVAS_OPENING_ARCH_SIZE_PIXELS,
    CANVAS_OPENING_CREATE_SIZE_PIXELS,
    CANVAS_OPENING_GIZMO_ANCHOR,
    CANVAS_OPENING_GIZMO_ARCH,
    CANVAS_OPENING_GIZMO_SIDE,
    CANVAS_OPENING_HANDLE_HIT_RADIUS_PIXELS,
    CANVAS_OPENING_OVERLAY_DEPTH_VALUE,
    CANVAS_OPENING_SIDE_BOTTOM,
    CANVAS_OPENING_SIDE_LEFT,
    CANVAS_OPENING_SIDE_RIGHT,
    CANVAS_OPENING_SIDE_SIZE_PIXELS,
    CANVAS_OPENING_SIDE_TOP,
    CANVAS_SURFACE_SELECTION_COLOR,
    DOOR_PLACEMENT_PREVIEW_COLORS,
    DoorPlacementPreviewCandidate,
    DoorPlacementPreviewPart,
    GlbViewerWidget,
    _CanvasOpeningGizmoHandle,
    _get_canvas_opening_creation_local_position,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_target(
    kind: str = CANVAS_OPENING_WINDOW,
    *,
    item_index: int = 0,
    stable_id: str = "window-a",
    bounds: CanvasOpeningBounds | None = None,
    arch_amount: float | None = None,
) -> CanvasOpeningTarget:
    return CanvasOpeningTarget(
        reference=CanvasOpeningReference(
            kind=kind,
            level_index=0,
            item_index=item_index,
            stable_id=stable_id if kind == CANVAS_OPENING_WINDOW else None,
        ),
        wall_surface_id="level:0/wall:1:2",
        plane_start_world=(0.0, 0.0, 0.0),
        wall_tangent_world=(1.0, 0.0, 0.0),
        wall_normal_world=(0.0, -1.0, 0.0),
        wall_width_meters=4.0,
        wall_height_meters=3.0,
        minimum_width_meters=0.2,
        minimum_height_meters=0.3,
        bounds=bounds
        or CanvasOpeningBounds(
            start_ratio=0.25,
            end_ratio=0.75,
            bottom_ratio=0.2,
            top_ratio=0.8,
        ),
        arch_amount=arch_amount,
    )


def _build_wall(*, y_offset: float = 0.0) -> FixedSurface:
    vertices = np.asarray(
        (
            (0.0, y_offset, 0.0),
            (4.0, y_offset, 0.0),
            (4.0, y_offset, 3.0),
            (0.0, y_offset, 3.0),
        ),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id="level:0/wall:1:2",
        surface_type=SURFACE_TYPE_WALL,
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=12.0,
        wall_key="1:2",
        wall_start_world=(0.0, y_offset, 0.0),
        wall_end_world=(4.0, y_offset, 0.0),
        wall_height_meters=3.0,
    )


def _build_model() -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(4.0, 0.2, 3.0))
    mesh.apply_translation((2.0, 0.0, 1.5))
    return GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b"")


def _build_opening_fixture_selection_model() -> GeneratedModel:
    """Build one installed door and one ordinary selectable scene object."""

    base_mesh = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    base_mesh.apply_translation((100.0, 100.0, 100.0))
    door_mesh = trimesh.creation.box(extents=(0.8, 0.2, 1.8))
    door_mesh.apply_translation((2.0, 0.0, 1.5))
    ordinary_mesh = trimesh.creation.box(extents=(0.8, 0.2, 1.0))
    ordinary_mesh.apply_translation((3.5, 0.0, 1.0))
    identity = np.eye(4, dtype=float)
    return GeneratedModel(
        mesh=base_mesh,
        scene=trimesh.Scene(base_mesh),
        glb_bytes=b"",
        preview_base_mesh=base_mesh.copy(),
        preview_placed_objects=[
            PreviewPlacedObject(
                object_id="door-placement:placement-a",
                meshes=(door_mesh,),
                placement_transform=identity.copy(),
                world_position=(0.0, 0.0, 0.0),
                rotation_degrees=(0.0, 0.0, 0.0),
            ),
            PreviewPlacedObject(
                object_id="ordinary-object",
                meshes=(ordinary_mesh,),
                placement_transform=identity.copy(),
                world_position=(0.0, 0.0, 0.0),
                rotation_degrees=(0.0, 0.0, 0.0),
            ),
        ],
    )


def _ray_at(x: float, z: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((x, -2.0, z), dtype=float),
        np.asarray((0.0, 1.0, 0.0), dtype=float),
    )


def _bounds_values(bounds: CanvasOpeningBounds) -> tuple[float, ...]:
    return (
        bounds.start_ratio,
        bounds.end_ratio,
        bounds.bottom_ratio,
        bounds.top_ratio,
    )


def _build_door_preview_candidate(
    target: CanvasOpeningTarget,
) -> DoorPlacementPreviewCandidate:
    """Build visibly distinct profiles for all four wheel orientations."""

    roles = (DOOR_SLOT_BODY, DOOR_SLOT_KNOB)
    normal_parts: list[DoorPlacementPreviewPart] = []
    mirrored_parts: list[DoorPlacementPreviewPart] = []
    depth_mirrored_parts: list[DoorPlacementPreviewPart] = []
    depth_and_horizontal_parts: list[DoorPlacementPreviewPart] = []
    for index, role in enumerate(roles):
        normal_mesh = trimesh.creation.box(
            extents=(0.25 + index * 0.05, 0.1, 0.5),
        )
        normal_mesh.apply_translation((float(index), 0.0, 0.25))
        mirrored_mesh = normal_mesh.copy()
        mirrored_mesh.apply_translation((10.0, 0.0, 0.0))
        depth_mirrored_mesh = normal_mesh.copy()
        depth_mirrored_mesh.apply_translation((20.0, 0.0, 0.0))
        depth_and_horizontal_mesh = normal_mesh.copy()
        depth_and_horizontal_mesh.apply_translation((30.0, 0.0, 0.0))
        normal_parts.append(DoorPlacementPreviewPart(role, (normal_mesh,)))
        mirrored_parts.append(DoorPlacementPreviewPart(role, (mirrored_mesh,)))
        depth_mirrored_parts.append(
            DoorPlacementPreviewPart(role, (depth_mirrored_mesh,))
        )
        depth_and_horizontal_parts.append(
            DoorPlacementPreviewPart(role, (depth_and_horizontal_mesh,))
        )
    normal_transform = np.eye(4, dtype=float)
    normal_transform[:3, 3] = (2.0, 3.0, 4.0)
    mirrored_transform = np.eye(4, dtype=float)
    mirrored_transform[:3, 3] = (5.0, 6.0, 7.0)
    depth_mirrored_transform = np.eye(4, dtype=float)
    depth_mirrored_transform[:3, 3] = (8.0, 9.0, 10.0)
    depth_and_horizontal_transform = np.eye(4, dtype=float)
    depth_and_horizontal_transform[:3, 3] = (11.0, 12.0, 13.0)
    return DoorPlacementPreviewCandidate(
        opening_key=target.key,
        normal_parts=tuple(normal_parts),
        normal_transform=normal_transform,
        mirrored_parts=tuple(mirrored_parts),
        mirrored_transform=mirrored_transform,
        depth_mirrored_parts=tuple(depth_mirrored_parts),
        depth_mirrored_transform=depth_mirrored_transform,
        depth_and_horizontally_mirrored_parts=tuple(
            depth_and_horizontal_parts
        ),
        depth_and_horizontally_mirrored_transform=(
            depth_and_horizontal_transform
        ),
    )


# ### Canvas opening gizmo tests ###
class CanvasOpeningGizmoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[GlbViewerWidget] = []

    def tearDown(self) -> None:
        for widget in self.widgets:
            widget.exit_first_person_mode()
            widget.close()
            widget.deleteLater()
        _qt_application.processEvents()

    def _build_viewer(self, target: CanvasOpeningTarget) -> GlbViewerWidget:
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((_build_wall(),))
        viewer.set_canvas_opening_targets((target,))
        self.widgets.append(viewer)
        return viewer

    def test_hole_plane_selection_wins_a_coplanar_wall_pick(self) -> None:
        target = _build_target()
        viewer = self._build_viewer(target)
        selected: list[object] = []
        viewer.canvas_opening_selection_changed.connect(selected.append)

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.0, 1.5),
        ):
            viewer._handle_window_wall_pick_requested(QPointF(20.0, 20.0))

        self.assertEqual(
            viewer.get_selected_canvas_opening_reference(),
            target.reference,
        )
        self.assertIsNone(viewer.get_selected_wall_surface_id())
        self.assertEqual(selected, [target.reference])

    def test_installed_door_pick_selects_its_doorway_instead_of_the_mesh(
        self,
    ) -> None:
        """Installed doors redirect Canvas selection to their owning opening."""

        target = _build_target(CANVAS_OPENING_DOORWAY)
        viewer = self._build_viewer(target)
        viewer.set_model(_build_opening_fixture_selection_model())
        viewer.set_placed_door_opening_bindings(
            {"door-placement:placement-a": target.reference}
        )
        selected_openings: list[object] = []
        selected_objects: list[object] = []
        viewer.canvas_opening_selection_changed.connect(selected_openings.append)
        viewer.placed_object_selection_changed.connect(selected_objects.append)

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=(
                np.asarray((2.0, -5.0, 1.5), dtype=float),
                np.asarray((0.0, 1.0, 0.0), dtype=float),
            ),
        ):
            viewer._handle_window_wall_pick_requested(QPointF())

        self.assertEqual(
            viewer.get_selected_canvas_opening_reference(),
            target.reference,
        )
        self.assertEqual(viewer.get_selected_placed_object_ids(), ())
        self.assertEqual(selected_openings, [target.reference])
        self.assertNotIn("door-placement:placement-a", selected_objects)

    def test_ordinary_placed_object_remains_selectable_beside_installed_door(
        self,
    ) -> None:
        """Door redirection must not change ordinary Canvas object selection."""

        target = _build_target(CANVAS_OPENING_DOORWAY)
        viewer = self._build_viewer(target)
        viewer.set_model(_build_opening_fixture_selection_model())
        viewer.set_placed_door_opening_bindings(
            {"door-placement:placement-a": target.reference}
        )

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=(
                np.asarray((3.5, -5.0, 1.0), dtype=float),
                np.asarray((0.0, 1.0, 0.0), dtype=float),
            ),
        ):
            viewer._handle_window_wall_pick_requested(QPointF())

        self.assertIsNone(viewer.get_selected_canvas_opening_reference())
        self.assertEqual(
            viewer.get_selected_placed_object_ids(),
            ("ordinary-object",),
        )

    def test_opening_and_handles_remain_selectable_through_a_nearer_wall(
        self,
    ) -> None:
        target = _build_target()
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((_build_wall(y_offset=-1.0),))
        viewer.set_canvas_opening_targets((target,))
        self.widgets.append(viewer)

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.0, 1.5),
        ):
            viewer._handle_window_wall_pick_requested(QPointF())

        self.assertEqual(
            viewer.get_selected_canvas_opening_reference(),
            target.reference,
        )
        self.assertIsNone(viewer.get_selected_wall_surface_id())

        with (
            patch.object(viewer.view, "pixelSize", return_value=0.01),
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=_ray_at(1.0, 1.5),
            ),
        ):
            viewer._handle_placed_object_pointer_pressed(QPointF())

        self.assertIsNotNone(viewer._canvas_opening_edit_drag)
        assert viewer._canvas_opening_edit_drag is not None
        self.assertEqual(
            viewer._canvas_opening_edit_drag.handle,
            _CanvasOpeningGizmoHandle(
                CANVAS_OPENING_GIZMO_SIDE,
                CANVAS_OPENING_SIDE_LEFT,
            ),
        )
        self.assertTrue(viewer.view.is_primary_pointer_drag_reserved)
        viewer._cancel_canvas_opening_edit_drag()

    def test_selected_opening_renders_handles_and_survives_model(
        self,
    ) -> None:
        target = _build_target()
        viewer = self._build_viewer(target)
        viewer.select_canvas_opening(target.reference)
        first_items = tuple(viewer._canvas_opening_gizmo_items)

        self.assertEqual(len(first_items), 3)
        self.assertEqual(first_items[0].color, CANVAS_SURFACE_SELECTION_COLOR)
        for item in first_items:
            self.assertEqual(
                item.depthValue(),
                CANVAS_OPENING_OVERLAY_DEPTH_VALUE,
            )
            gl_options = getattr(item, "_GLGraphicsItem__glOpts")
            self.assertFalse(gl_options[GL.GL_DEPTH_TEST])
            self.assertTrue(gl_options[GL.GL_BLEND])
        side_positions = np.asarray(first_items[1].pos, dtype=float)
        anchor_positions = np.asarray(first_items[2].pos, dtype=float)
        self.assertEqual(side_positions.shape, (4, 3))
        self.assertEqual(anchor_positions.shape, (1, 3))
        np.testing.assert_allclose(
            side_positions[:, (0, 2)],
            ((1.0, 1.5), (3.0, 1.5), (2.0, 0.6), (2.0, 2.4)),
            atol=1e-9,
        )
        self.assertGreaterEqual(CANVAS_OPENING_SIDE_SIZE_PIXELS, 20.0)
        self.assertGreaterEqual(CANVAS_OPENING_HANDLE_HIT_RADIUS_PIXELS, 18.0)

        viewer.set_model(_build_model(), preserve_camera=True)

        self.assertEqual(
            viewer.get_selected_canvas_opening_reference(),
            target.reference,
        )
        self.assertEqual(len(viewer._canvas_opening_gizmo_items), 3)
        self.assertTrue(
            first_items[0] not in viewer._canvas_opening_gizmo_items
        )

    def test_create_fixture_plus_is_rendered_for_both_opening_types(self) -> None:
        window_target = _build_target(CANVAS_OPENING_WINDOW)
        viewer = self._build_viewer(window_target)
        viewer.select_canvas_opening(window_target.reference)

        self.assertEqual(len(viewer._canvas_opening_creation_items), 2)

        doorway_target = _build_target(
            CANVAS_OPENING_DOORWAY,
            item_index=1,
        )
        viewer.set_canvas_opening_targets((window_target, doorway_target))
        viewer.select_canvas_opening(doorway_target.reference)

        self.assertEqual(len(viewer._canvas_opening_creation_items), 2)
        background_item, icon_item = viewer._canvas_opening_creation_items
        self.assertEqual(background_item.size, CANVAS_OPENING_CREATE_SIZE_PIXELS)
        for item in (background_item, icon_item):
            self.assertEqual(item.depthValue(), CANVAS_OPENING_OVERLAY_DEPTH_VALUE)
            gl_options = getattr(item, "_GLGraphicsItem__glOpts")
            self.assertFalse(gl_options[GL.GL_DEPTH_TEST])
            self.assertTrue(gl_options[GL.GL_BLEND])

    def test_create_door_plus_emits_on_release_without_moving_doorway(self) -> None:
        target = _build_target(CANVAS_OPENING_DOORWAY)
        viewer = self._build_viewer(target)
        viewer.select_canvas_opening(target.reference)
        requested: list[object] = []
        viewer.door_creation_requested.connect(requested.append)
        local_position = _get_canvas_opening_creation_local_position(target)
        assert local_position is not None
        world_position = target.local_to_world(*local_position)

        with (
            patch.object(viewer.view, "pixelSize", return_value=0.01),
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=_ray_at(world_position[0], world_position[2]),
            ),
            patch.object(
                viewer,
                "_pick_canvas_opening_gizmo_handle",
                wraps=viewer._pick_canvas_opening_gizmo_handle,
            ) as opening_pick,
        ):
            viewer._handle_placed_object_pointer_pressed(QPointF())
            self.assertEqual(
                viewer._canvas_opening_creation_pressed_reference,
                target.reference,
            )
            self.assertIsNone(viewer._canvas_opening_edit_drag)
            self.assertFalse(opening_pick.called)
            viewer._handle_canvas_gizmo_pointer_released(QPointF())

        self.assertEqual(requested, [target.reference])
        self.assertEqual(viewer._canvas_opening_targets[target.key], target)
        self.assertIsNone(viewer._canvas_opening_creation_pressed_reference)
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)

    def test_create_door_plus_cancels_when_release_leaves_control(self) -> None:
        target = _build_target(CANVAS_OPENING_DOORWAY)
        viewer = self._build_viewer(target)
        viewer.select_canvas_opening(target.reference)
        requested: list[object] = []
        viewer.door_creation_requested.connect(requested.append)
        local_position = _get_canvas_opening_creation_local_position(target)
        assert local_position is not None
        world_position = target.local_to_world(*local_position)

        with (
            patch.object(viewer.view, "pixelSize", return_value=0.01),
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=_ray_at(world_position[0], world_position[2]),
            ),
        ):
            viewer._handle_placed_object_pointer_pressed(QPointF())
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(-10.0, -10.0),
        ):
            viewer._handle_canvas_gizmo_pointer_released(QPointF())

        self.assertEqual(requested, [])
        self.assertIsNone(viewer._canvas_opening_creation_pressed_reference)
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)

    def test_create_door_plus_press_is_cleared_when_selection_changes(self) -> None:
        doorway_target = _build_target(CANVAS_OPENING_DOORWAY)
        window_target = _build_target(
            CANVAS_OPENING_WINDOW,
            item_index=1,
            stable_id="window-b",
        )
        viewer = self._build_viewer(doorway_target)
        viewer.set_canvas_opening_targets((doorway_target, window_target))
        viewer.select_canvas_opening(doorway_target.reference)
        local_position = _get_canvas_opening_creation_local_position(doorway_target)
        assert local_position is not None
        world_position = doorway_target.local_to_world(*local_position)

        with (
            patch.object(viewer.view, "pixelSize", return_value=0.01),
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=_ray_at(world_position[0], world_position[2]),
            ),
        ):
            viewer._handle_placed_object_pointer_pressed(QPointF())
        viewer.select_canvas_opening(window_target.reference)

        self.assertIsNone(viewer._canvas_opening_creation_pressed_reference)
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)
        self.assertEqual(len(viewer._canvas_opening_creation_items), 2)

    def test_create_window_plus_emits_a_separate_fixture_request(self) -> None:
        target = _build_target(CANVAS_OPENING_WINDOW)
        viewer = self._build_viewer(target)
        viewer.select_canvas_opening(target.reference)
        requested: list[object] = []
        door_requests: list[object] = []
        viewer.window_fixture_creation_requested.connect(requested.append)
        viewer.door_creation_requested.connect(door_requests.append)
        local_position = _get_canvas_opening_creation_local_position(target)
        assert local_position is not None
        world_position = target.local_to_world(*local_position)

        with (
            patch.object(viewer.view, "pixelSize", return_value=0.01),
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=_ray_at(world_position[0], world_position[2]),
            ),
        ):
            viewer._handle_placed_object_pointer_pressed(QPointF())
            viewer._handle_canvas_gizmo_pointer_released(QPointF())

        self.assertEqual(requested, [target.reference])
        self.assertEqual(door_requests, [])

    def test_door_placement_click_emits_without_cancellation(self) -> None:
        target = _build_target(CANVAS_OPENING_DOORWAY)
        viewer = self._build_viewer(target)
        placements: list[tuple[str, object, bool, bool]] = []
        cancellations: list[str] = []
        viewer.door_placement_requested.connect(
            lambda door_id, reference, horizontal, depth: placements.append(
                (door_id, reference, horizontal, depth)
            )
        )
        viewer.door_placement_cancelled.connect(cancellations.append)

        self.assertTrue(viewer.begin_door_placement("door-a", (target.key,)))
        self.assertTrue(viewer.is_door_placement_active)
        self.assertTrue(viewer.view._primary_pointer_tool_active)
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.0, 1.5),
        ):
            viewer._handle_placed_object_pointer_pressed(QPointF(20.0, 20.0))
            self.assertTrue(viewer.view.is_primary_pointer_drag_reserved)
            viewer._handle_canvas_gizmo_pointer_released(QPointF(20.0, 20.0))

        self.assertEqual(
            placements,
            [("door-a", target.reference, False, False)],
        )
        self.assertEqual(cancellations, [])
        self.assertFalse(viewer.is_door_placement_active)
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)
        self.assertFalse(viewer.view._primary_pointer_tool_active)

    def test_ctrl_door_placement_still_places_the_complete_door(self) -> None:
        target = _build_target(CANVAS_OPENING_DOORWAY)
        viewer = self._build_viewer(target)
        candidate = _build_door_preview_candidate(target)
        placements: list[tuple[str, object, bool, bool]] = []
        viewer.door_placement_requested.connect(
            lambda door_id, reference, horizontal, depth: placements.append(
                (door_id, reference, horizontal, depth)
            )
        )

        self.assertTrue(
            viewer.begin_door_placement(
                "door-a",
                (target.key,),
                preview_candidates=(candidate,),
            )
        )
        viewer.view.control_modifier_changed.emit(True)
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(1.5, 1.5),
        ):
            viewer._handle_surface_vertex_pointer_hovered(QPointF())
            self.assertIsNotNone(viewer._door_placement_preview_root)
            viewer._handle_placed_object_pointer_pressed(QPointF())
            viewer._handle_canvas_gizmo_pointer_released(QPointF())

        self.assertEqual(
            placements,
            [("door-a", target.reference, False, False)],
        )
        self.assertFalse(viewer.is_door_placement_active)

    def test_ctrl_door_placement_does_not_block_an_arched_doorway(self) -> None:
        target = _build_target(
            CANVAS_OPENING_DOORWAY,
            arch_amount=0.6,
        )
        viewer = self._build_viewer(target)
        placements: list[tuple[str, object, bool, bool]] = []
        viewer.door_placement_requested.connect(
            lambda door_id, reference, horizontal, depth: placements.append(
                (door_id, reference, horizontal, depth)
            )
        )

        self.assertTrue(viewer.begin_door_placement("door-a", (target.key,)))
        viewer.view.control_modifier_changed.emit(True)
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(1.5, 1.5),
        ):
            viewer._handle_surface_vertex_pointer_hovered(QPointF())
            viewer._handle_placed_object_pointer_pressed(QPointF())
            viewer._handle_canvas_gizmo_pointer_released(QPointF())

        self.assertEqual(
            placements,
            [("door-a", target.reference, False, False)],
        )
        self.assertFalse(viewer.is_door_placement_active)

    def test_window_fixture_placement_reuses_preview_and_mirror_pipeline(
        self,
    ) -> None:
        target = _build_target(CANVAS_OPENING_WINDOW)
        viewer = self._build_viewer(target)
        candidate = _build_door_preview_candidate(target)
        placements: list[tuple[str, object, bool, bool]] = []
        viewer.window_fixture_placement_requested.connect(
            lambda window_id, reference, horizontal, depth: placements.append(
                (window_id, reference, horizontal, depth)
            )
        )

        self.assertTrue(
            viewer.begin_window_fixture_placement(
                "window-a",
                (target.key,),
                preview_candidates=(candidate,),
            )
        )
        self.assertTrue(viewer.is_window_fixture_placement_active)
        self.assertFalse(viewer.is_door_placement_active)
        viewer._handle_projection_camera_wheel_steps_requested(1)
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.0, 1.5),
        ):
            viewer._handle_surface_vertex_pointer_hovered(QPointF())
            self.assertIsNotNone(viewer._door_placement_preview_root)
            viewer._handle_placed_object_pointer_pressed(QPointF())
            viewer._handle_canvas_gizmo_pointer_released(QPointF())

        self.assertEqual(
            placements,
            [("window-a", target.reference, True, False)],
        )
        self.assertFalse(viewer.is_opening_fixture_placement_active)

    def test_door_placement_hover_shows_colored_component_profile(self) -> None:
        target = _build_target(CANVAS_OPENING_DOORWAY)
        viewer = self._build_viewer(target)
        candidate = _build_door_preview_candidate(target)

        self.assertNotIn("hinges", DOOR_PLACEMENT_PREVIEW_COLORS)

        self.assertTrue(
            viewer.begin_door_placement(
                "door-a",
                (target.key,),
                preview_candidates=(candidate,),
            )
        )
        self.assertIsNone(viewer._door_placement_preview_root)
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.0, 1.5),
        ):
            self.assertTrue(
                viewer._handle_surface_vertex_pointer_hovered(QPointF())
            )

        self.assertEqual(
            viewer._door_placement_hover_opening_key,
            target.key,
        )
        self.assertIsNotNone(viewer._door_placement_preview_root)
        self.assertEqual(len(viewer._door_placement_preview_items), 2)
        rendered_colors = tuple(
            tuple(float(component) for component in item.opts["color"])
            for item in viewer._door_placement_preview_items
        )
        expected_colors = tuple(
            DOOR_PLACEMENT_PREVIEW_COLORS[role]
            for role in (
                DOOR_SLOT_BODY,
                DOOR_SLOT_KNOB,
            )
        )
        self.assertEqual(rendered_colors, expected_colors)

    def test_door_placement_preview_clears_on_pointer_leave_and_cancel(
        self,
    ) -> None:
        target = _build_target(CANVAS_OPENING_DOORWAY)
        viewer = self._build_viewer(target)
        candidate = _build_door_preview_candidate(target)
        self.assertTrue(
            viewer.begin_door_placement(
                "door-a",
                (target.key,),
                preview_candidates=(candidate,),
            )
        )

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.0, 1.5),
        ):
            viewer._handle_surface_vertex_pointer_hovered(QPointF())
        self.assertIsNotNone(viewer._door_placement_preview_root)

        viewer._handle_surface_vertex_pointer_left()

        self.assertIsNone(viewer._door_placement_hover_opening_key)
        self.assertIsNone(viewer._door_placement_preview_root)
        self.assertEqual(viewer._door_placement_preview_items, [])

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.0, 1.5),
        ):
            viewer._handle_surface_vertex_pointer_hovered(QPointF())
        self.assertIsNotNone(viewer._door_placement_preview_root)

        self.assertTrue(viewer.cancel_door_placement())
        self.assertIsNone(viewer._door_placement_preview_root)
        self.assertEqual(viewer._door_placement_preview_items, [])
        self.assertEqual(viewer._door_placement_preview_candidates, {})

    def test_nearer_scene_geometry_occludes_door_placement_hover_and_click(
        self,
    ) -> None:
        target = _build_target(CANVAS_OPENING_DOORWAY)
        viewer = self._build_viewer(target)
        viewer.set_model(_build_model(), preserve_camera=True)
        placements: list[tuple[str, object, bool, bool]] = []
        viewer.door_placement_requested.connect(
            lambda door_id, reference, horizontal, depth: placements.append(
                (door_id, reference, horizontal, depth)
            )
        )
        self.assertTrue(
            viewer.begin_door_placement(
                "door-a",
                (target.key,),
                preview_candidates=(_build_door_preview_candidate(target),),
            )
        )

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.0, 1.5),
        ):
            self.assertFalse(
                viewer._handle_surface_vertex_pointer_hovered(QPointF())
            )
            viewer._handle_placed_object_pointer_pressed(QPointF())
            viewer._handle_canvas_gizmo_pointer_released(QPointF())

        self.assertIsNone(viewer._door_placement_hover_opening_key)
        self.assertIsNone(viewer._door_placement_preview_root)
        self.assertEqual(placements, [])
        self.assertTrue(viewer.is_door_placement_active)

    def test_changing_compatible_opening_cancels_stale_door_preview(
        self,
    ) -> None:
        target = _build_target(CANVAS_OPENING_DOORWAY)
        viewer = self._build_viewer(target)
        cancellations: list[str] = []
        viewer.door_placement_cancelled.connect(cancellations.append)
        self.assertTrue(
            viewer.begin_door_placement(
                "door-a",
                (target.key,),
                preview_candidates=(_build_door_preview_candidate(target),),
            )
        )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.0, 1.5),
        ):
            viewer._handle_surface_vertex_pointer_hovered(QPointF())
        self.assertIsNotNone(viewer._door_placement_preview_root)

        changed_target = replace(
            target,
            bounds=CanvasOpeningBounds(
                start_ratio=0.2,
                end_ratio=0.8,
                bottom_ratio=0.1,
                top_ratio=0.9,
            ),
        )
        self.assertEqual(changed_target.key, target.key)
        viewer.set_canvas_opening_targets((changed_target,))

        self.assertFalse(viewer.is_door_placement_active)
        self.assertEqual(cancellations, ["door-a"])
        self.assertIsNone(viewer._door_placement_hover_opening_key)
        self.assertIsNone(viewer._door_placement_preview_root)
        self.assertEqual(viewer._door_placement_preview_candidates, {})

    def test_door_placement_wheel_switches_hover_to_mirrored_profile(
        self,
    ) -> None:
        target = _build_target(CANVAS_OPENING_DOORWAY)
        viewer = self._build_viewer(target)
        candidate = _build_door_preview_candidate(target)
        placements: list[tuple[str, object, bool, bool]] = []
        viewer.door_placement_requested.connect(
            lambda door_id, reference, horizontal, depth: placements.append(
                (door_id, reference, horizontal, depth)
            )
        )
        self.assertTrue(
            viewer.begin_door_placement(
                "door-a",
                (target.key,),
                preview_candidates=(candidate,),
            )
        )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.0, 1.5),
        ):
            viewer._handle_surface_vertex_pointer_hovered(QPointF())
            normal_minimum_x = min(
                float(item.opts["meshdata"].vertexes()[:, 0].min())
                for item in viewer._door_placement_preview_items
            )

            viewer._handle_projection_camera_wheel_steps_requested(1)

            mirrored_minimum_x = min(
                float(item.opts["meshdata"].vertexes()[:, 0].min())
                for item in viewer._door_placement_preview_items
            )
            viewer._handle_placed_object_pointer_pressed(QPointF())
            viewer._handle_canvas_gizmo_pointer_released(QPointF())

        self.assertGreater(mirrored_minimum_x, normal_minimum_x + 5.0)
        self.assertEqual(
            placements,
            [("door-a", target.reference, True, False)],
        )
        self.assertFalse(viewer.is_door_placement_active)

    def test_door_placement_rejects_incompatible_openings_and_stays_armed(
        self,
    ) -> None:
        compatible = _build_target(
            CANVAS_OPENING_DOORWAY,
            item_index=0,
            bounds=CanvasOpeningBounds(0.05, 0.25, 0.2, 0.8),
        )
        incompatible = _build_target(
            CANVAS_OPENING_DOORWAY,
            item_index=1,
            bounds=CanvasOpeningBounds(0.75, 0.95, 0.2, 0.8),
        )
        viewer = self._build_viewer(compatible)
        viewer.set_canvas_opening_targets((compatible, incompatible))
        placements: list[object] = []
        viewer.door_placement_requested.connect(
            lambda *_args: placements.append(_args)
        )

        self.assertTrue(
            viewer.begin_door_placement("door-a", (compatible.key,))
        )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(3.4, 1.5),
        ):
            viewer._handle_placed_object_pointer_pressed(QPointF(10.0, 10.0))
            viewer._handle_canvas_gizmo_pointer_released(QPointF(10.0, 10.0))

        self.assertEqual(placements, [])
        self.assertTrue(viewer.is_door_placement_active)
        self.assertIsNone(viewer.window_tools_status_label)

    def test_door_placement_wheel_cycles_four_orientations_without_camera_edit(
        self,
    ) -> None:
        target = _build_target(CANVAS_OPENING_DOORWAY)
        viewer = self._build_viewer(target)
        camera_steps: list[tuple[str, int]] = []
        viewer.projection_camera_percentage_step_requested.connect(
            lambda camera_id, steps: camera_steps.append((camera_id, steps))
        )
        viewer._selected_projection_camera_id = "+x"
        self.assertTrue(viewer.begin_door_placement("door-a", (target.key,)))

        viewer._handle_projection_camera_wheel_steps_requested(1)
        self.assertTrue(viewer._door_placement_mirrored_horizontally)
        self.assertFalse(viewer._door_placement_mirrored_depthwise)
        viewer._handle_projection_camera_wheel_steps_requested(1)
        self.assertTrue(viewer._door_placement_mirrored_horizontally)
        self.assertTrue(viewer._door_placement_mirrored_depthwise)
        viewer._handle_projection_camera_wheel_steps_requested(1)
        self.assertFalse(viewer._door_placement_mirrored_horizontally)
        self.assertTrue(viewer._door_placement_mirrored_depthwise)
        viewer._handle_projection_camera_wheel_steps_requested(1)
        self.assertFalse(viewer._door_placement_mirrored_horizontally)
        self.assertFalse(viewer._door_placement_mirrored_depthwise)
        viewer._handle_projection_camera_wheel_steps_requested(-1)
        self.assertFalse(viewer._door_placement_mirrored_horizontally)
        self.assertTrue(viewer._door_placement_mirrored_depthwise)
        self.assertEqual(camera_steps, [])

        viewer.cancel_door_placement()
        viewer._handle_projection_camera_wheel_steps_requested(2)
        self.assertEqual(camera_steps, [("+x", 2)])

    def test_escape_and_right_click_cancel_door_placement(self) -> None:
        target = _build_target(CANVAS_OPENING_DOORWAY)
        viewer = self._build_viewer(target)
        cancellations: list[str] = []
        viewer.door_placement_cancelled.connect(cancellations.append)

        self.assertTrue(viewer.begin_door_placement("door-a", (target.key,)))
        viewer.view.keyPressEvent(
            QKeyEvent(
                QKeyEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        self.assertFalse(viewer.is_door_placement_active)

        self.assertTrue(viewer.begin_door_placement("door-b", (target.key,)))
        viewer.view.mousePressEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(4.0, 4.0),
                QPointF(4.0, 4.0),
                Qt.MouseButton.RightButton,
                Qt.MouseButton.RightButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )

        self.assertFalse(viewer.is_door_placement_active)
        self.assertEqual(cancellations, ["door-a", "door-b"])

    def test_cpu_handle_pick_distinguishes_each_side_from_the_anchor(self) -> None:
        target = _build_target()
        viewer = self._build_viewer(target)
        viewer.select_canvas_opening(target.reference)

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            side_handles = tuple(
                viewer._pick_canvas_opening_gizmo_handle(*_ray_at(x, z))
                for x, z in (
                    (1.0, 1.5),
                    (3.0, 1.5),
                    (2.0, 0.6),
                    (2.0, 2.4),
                )
            )
            anchor_handle = viewer._pick_canvas_opening_gizmo_handle(
                *_ray_at(2.0, 1.5)
            )

        self.assertEqual(
            side_handles,
            tuple(
                _CanvasOpeningGizmoHandle(CANVAS_OPENING_GIZMO_SIDE, side)
                for side in (
                    CANVAS_OPENING_SIDE_LEFT,
                    CANVAS_OPENING_SIDE_RIGHT,
                    CANVAS_OPENING_SIDE_BOTTOM,
                    CANVAS_OPENING_SIDE_TOP,
                )
            ),
        )
        self.assertEqual(
            anchor_handle,
            _CanvasOpeningGizmoHandle(CANVAS_OPENING_GIZMO_ANCHOR),
        )

    def test_arch_doorway_renders_and_picks_two_linked_shoulder_handles(
        self,
    ) -> None:
        target = _build_target(
            CANVAS_OPENING_DOORWAY,
            arch_amount=0.6,
        )
        viewer = self._build_viewer(target)
        viewer.select_canvas_opening(target.reference)
        items = tuple(viewer._canvas_opening_gizmo_items)

        self.assertEqual(len(items), 5)
        self.assertGreater(len(np.asarray(items[0].pos)), 20)
        guide_positions = np.asarray(items[3].pos, dtype=float)
        handle_positions = np.asarray(items[4].pos, dtype=float)
        np.testing.assert_allclose(
            guide_positions[:, (0, 2)],
            ((1.0, 1.8), (3.0, 1.8)),
            atol=1e-9,
        )
        np.testing.assert_allclose(handle_positions, guide_positions, atol=1e-9)
        self.assertGreaterEqual(CANVAS_OPENING_ARCH_SIZE_PIXELS, 20.0)

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            left_handle = viewer._pick_canvas_opening_gizmo_handle(
                *_ray_at(1.0, 1.8)
            )
            right_handle = viewer._pick_canvas_opening_gizmo_handle(
                *_ray_at(3.0, 1.8)
            )

        self.assertEqual(
            left_handle,
            _CanvasOpeningGizmoHandle(
                CANVAS_OPENING_GIZMO_ARCH,
                CANVAS_OPENING_SIDE_LEFT,
            ),
        )
        self.assertEqual(
            right_handle,
            _CanvasOpeningGizmoHandle(
                CANVAS_OPENING_GIZMO_ARCH,
                CANVAS_OPENING_SIDE_RIGHT,
            ),
        )

    def test_arch_shoulder_drag_changes_only_arch_amount(self) -> None:
        target = _build_target(
            CANVAS_OPENING_DOORWAY,
            arch_amount=0.6,
        )
        viewer = self._build_viewer(target)
        viewer.select_canvas_opening(target.reference)
        finished: list[tuple[object, bool]] = []
        viewer.canvas_opening_edit_finished.connect(
            lambda edit, changed: finished.append((edit, changed))
        )
        handle = _CanvasOpeningGizmoHandle(
            CANVAS_OPENING_GIZMO_ARCH,
            CANVAS_OPENING_SIDE_LEFT,
        )

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(1.0, 1.8),
        ):
            self.assertTrue(viewer._begin_canvas_opening_gizmo_drag(handle, QPointF()))
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(1.0, 2.2),
        ):
            self.assertTrue(viewer._finish_canvas_opening_gizmo_drag(QPointF()))

        self.assertEqual(len(finished), 1)
        final_edit, changed = finished[0]
        self.assertTrue(changed)
        self.assertEqual(final_edit.bounds, target.bounds)
        self.assertAlmostEqual(final_edit.arch_amount, 0.2)
        self.assertAlmostEqual(
            viewer._canvas_opening_targets[target.key].arch_amount,
            0.2,
        )

    def test_each_side_drag_changes_only_its_own_boundary(self) -> None:
        cases = (
            (
                CANVAS_OPENING_SIDE_LEFT,
                (1.0, 1.5),
                (0.4, 0.3),
                CanvasOpeningBounds(0.1, 0.75, 0.2, 0.8),
            ),
            (
                CANVAS_OPENING_SIDE_RIGHT,
                (3.0, 1.5),
                (3.6, 0.3),
                CanvasOpeningBounds(0.25, 0.9, 0.2, 0.8),
            ),
            (
                CANVAS_OPENING_SIDE_BOTTOM,
                (2.0, 0.6),
                (0.4, 0.3),
                CanvasOpeningBounds(0.25, 0.75, 0.1, 0.8),
            ),
            (
                CANVAS_OPENING_SIDE_TOP,
                (2.0, 2.4),
                (0.4, 2.7),
                CanvasOpeningBounds(0.25, 0.75, 0.2, 0.9),
            ),
        )
        for side, start_point, finish_point, expected_bounds in cases:
            with self.subTest(side=side):
                target = _build_target()
                viewer = self._build_viewer(target)
                viewer.select_canvas_opening(target.reference)
                started: list[object] = []
                previews: list[object] = []
                finished: list[tuple[object, bool]] = []
                viewer.canvas_opening_edit_started.connect(started.append)
                viewer.canvas_opening_edit_preview_changed.connect(previews.append)
                viewer.canvas_opening_edit_finished.connect(
                    lambda edit, changed: finished.append((edit, changed))
                )
                handle = _CanvasOpeningGizmoHandle(
                    CANVAS_OPENING_GIZMO_SIDE,
                    side,
                )

                with patch.object(
                    viewer.view,
                    "build_camera_ray",
                    return_value=_ray_at(*start_point),
                ):
                    self.assertTrue(
                        viewer._begin_canvas_opening_gizmo_drag(handle, QPointF())
                    )
                with patch.object(
                    viewer.view,
                    "build_camera_ray",
                    return_value=_ray_at(*finish_point),
                ):
                    self.assertTrue(
                        viewer._finish_canvas_opening_gizmo_drag(QPointF())
                    )

                self.assertEqual(len(started), 1)
                self.assertEqual(len(previews), 1)
                self.assertEqual(len(finished), 1)
                self.assertTrue(finished[0][1])
                np.testing.assert_allclose(
                    _bounds_values(finished[0][0].bounds),
                    _bounds_values(expected_bounds),
                    atol=1e-9,
                )
                self.assertEqual(
                    viewer._canvas_opening_targets[target.key].bounds,
                    finished[0][0].bounds,
                )
                self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)

    def test_window_anchor_moves_both_axes_and_doorway_only_sideways(
        self,
    ) -> None:
        handle = _CanvasOpeningGizmoHandle(CANVAS_OPENING_GIZMO_ANCHOR)
        for kind, expected_bounds in (
            (
                CANVAS_OPENING_WINDOW,
                CanvasOpeningBounds(0.375, 0.875, 0.3, 0.9),
            ),
            (
                CANVAS_OPENING_DOORWAY,
                CanvasOpeningBounds(0.375, 0.875, 0.2, 0.8),
            ),
        ):
            with self.subTest(kind=kind):
                target = _build_target(kind)
                viewer = self._build_viewer(target)
                viewer.select_canvas_opening(target.reference)
                with patch.object(
                    viewer.view,
                    "build_camera_ray",
                    return_value=_ray_at(2.0, 1.5),
                ):
                    viewer._begin_canvas_opening_gizmo_drag(handle, QPointF())
                with patch.object(
                    viewer.view,
                    "build_camera_ray",
                    return_value=_ray_at(2.5, 1.8),
                ):
                    viewer._finish_canvas_opening_gizmo_drag(QPointF())

                np.testing.assert_allclose(
                    _bounds_values(
                        viewer._canvas_opening_targets[target.key].bounds
                    ),
                    _bounds_values(expected_bounds),
                    atol=1e-9,
                )

    def test_side_drags_clamp_to_wall_and_minimum_size(self) -> None:
        cases = (
            (
                CANVAS_OPENING_SIDE_LEFT,
                (1.0, 1.5),
                (20.0, 20.0),
                CanvasOpeningBounds(0.7, 0.75, 0.2, 0.8),
            ),
            (
                CANVAS_OPENING_SIDE_RIGHT,
                (3.0, 1.5),
                (-20.0, -20.0),
                CanvasOpeningBounds(0.25, 0.3, 0.2, 0.8),
            ),
            (
                CANVAS_OPENING_SIDE_BOTTOM,
                (2.0, 0.6),
                (20.0, 20.0),
                CanvasOpeningBounds(0.25, 0.75, 0.7, 0.8),
            ),
            (
                CANVAS_OPENING_SIDE_TOP,
                (2.0, 2.4),
                (-20.0, -20.0),
                CanvasOpeningBounds(0.25, 0.75, 0.2, 0.3),
            ),
        )
        for side, start_point, finish_point, expected_bounds in cases:
            with self.subTest(side=side):
                target = _build_target()
                viewer = self._build_viewer(target)
                viewer.select_canvas_opening(target.reference)
                handle = _CanvasOpeningGizmoHandle(
                    CANVAS_OPENING_GIZMO_SIDE,
                    side,
                )
                with patch.object(
                    viewer.view,
                    "build_camera_ray",
                    return_value=_ray_at(*start_point),
                ):
                    viewer._begin_canvas_opening_gizmo_drag(handle, QPointF())
                with patch.object(
                    viewer.view,
                    "build_camera_ray",
                    return_value=_ray_at(*finish_point),
                ):
                    viewer._finish_canvas_opening_gizmo_drag(QPointF())

                np.testing.assert_allclose(
                    _bounds_values(
                        viewer._canvas_opening_targets[target.key].bounds
                    ),
                    _bounds_values(expected_bounds),
                    atol=1e-9,
                )

    def test_escape_restores_start_and_delete_is_consumed(self) -> None:
        target = _build_target()
        viewer = self._build_viewer(target)
        viewer.select_canvas_opening(target.reference)
        cancelled: list[object] = []
        finished: list[object] = []
        generic_deletions: list[bool] = []
        viewer.canvas_opening_edit_cancelled.connect(cancelled.append)
        viewer.canvas_opening_edit_finished.connect(
            lambda *_args: finished.append(True)
        )
        viewer.delete_requested.connect(lambda: generic_deletions.append(True))
        handle = _CanvasOpeningGizmoHandle(CANVAS_OPENING_GIZMO_ANCHOR)
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.0, 1.5),
        ):
            viewer._begin_canvas_opening_gizmo_drag(handle, QPointF())
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.5, 1.8),
        ):
            viewer._update_canvas_opening_gizmo_drag(QPointF())

        viewer.view.keyPressEvent(
            QKeyEvent(
                QKeyEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        viewer.view.keyPressEvent(
            QKeyEvent(
                QKeyEvent.Type.KeyPress,
                Qt.Key.Key_Delete,
                Qt.KeyboardModifier.NoModifier,
            )
        )

        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0].bounds, target.bounds)
        self.assertEqual(finished, [])
        self.assertEqual(generic_deletions, [])
        self.assertEqual(viewer._canvas_opening_targets[target.key], target)
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
