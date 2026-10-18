# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import copy
import math
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.canvas_openings import (
    CANVAS_OPENING_DOORWAY,
    CANVAS_OPENING_WINDOW,
)
from housemaker.door_geometry import (
    DoorBodyMirrorConfiguration,
    build_door_body_model,
)
from housemaker.door_state import (
    DOOR_SIDE_DUPLICATION_BACK,
    DOOR_SIDE_DUPLICATION_FRONT,
    DOOR_SLOT_BACK_BODY,
    DOOR_SLOT_BACK_KNOB,
    DOOR_SLOT_BACK_LEFT_BODY,
    DOOR_SLOT_BACK_LEFT_KNOB,
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    DOOR_SLOT_LEFT_BODY,
    DOOR_SLOT_LEFT_KNOB,
    DoorSideDuplication,
    DoorSideDuplicationMetadata,
    GlassAreaData,
)
from housemaker.doors_workspace import ASSET_KIND_DOOR, ASSET_KIND_WINDOW
from housemaker.doorway_geometry import build_double_door_leaf_source_outline
from housemaker.generation_workspace import (
    DOOR_SIDE_DUPLICATION_PIPELINE_KEY,
)
from housemaker.glass_material import is_housemaker_glass_material
from housemaker.glb import (
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
    build_gltf_mirror_plane_from_z_up_transform,
    build_placed_generated_model_gltf_transform,
    build_placed_generated_model_transform,
    import_generated_glb,
)
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
)
from housemaker.main import BlueprintWorkspace
from housemaker.models import (
    DOORWAY_SHAPE_ARCH,
    DoorwayData,
    LevelData,
    WindowData,
)
from housemaker.object_face_edit import ObjectFaceGeometry
from housemaker.project_io import ProjectData
from housemaker.surface_geometry import SURFACE_TYPE_WALL, FixedSurface
from housemaker.viewer import (
    PLACED_OBJECT_GIZMO_SCALE,
    PLACED_OBJECT_GIZMO_TRANSFORM,
    GlassAreaPlacementRequest,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_level_and_wall() -> tuple[LevelData, FixedSurface]:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (4.0, 0.0, 3.0),
            (0.0, 0.0, 3.0),
        ),
        dtype=float,
    )
    wall = FixedSurface(
        surface_id="level:2/wall:1:2",
        surface_type=SURFACE_TYPE_WALL,
        level_index=2,
        room_index=None,
        mesh=trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
            process=False,
        ),
        area_square_meters=12.0,
        wall_key="1:2",
        wall_start_world=(0.0, 0.0, 0.0),
        wall_end_world=(4.0, 0.0, 0.0),
        wall_height_meters=3.0,
    )
    return (
        LevelData(
            index=2,
            name="Ground",
            height_meters=3.0,
            doorways=[
                DoorwayData(
                    center_x=100.0,
                    center_y=0.0,
                    width_meters=1.0,
                    height_meters=2.0,
                    rotation_degrees=90.0,
                )
            ],
        ),
        wall,
    )


def _arch_profile_height_at_x(
    width_meters: float,
    height_meters: float,
    arch_amount: float,
    x_coordinate: float,
) -> float:
    """Return the curved profile height used by an arched doorway."""

    half_width = width_meters / 2.0
    arch_rise = min(half_width, height_meters) * arch_amount
    spring_height = height_meters - arch_rise
    normalized_x = min(1.0, abs(x_coordinate) / half_width)
    return spring_height + arch_rise * math.sqrt(
        max(0.0, 1.0 - normalized_x * normalized_x)
    )


# ### Main integration tests ###
class DoorMainIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        self.level, self.wall = _build_level_and_wall()
        self.workspace.levels = [self.level]
        self.workspace.current_level_index = 0
        self.workspace._reset_viewer_doorway_snapshots()
        self.workspace._sync_canvas_to_current_level()
        self.workspace._set_canvas_viewer_targets((self.wall,))

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _doorway_reference(self):
        return next(
            target.reference
            for target in self.workspace._canvas_opening_targets_by_key.values()
            if target.reference.kind == CANVAS_OPENING_DOORWAY
        )

    def _create_door(self):
        self.workspace._handle_door_creation_requested(
            self._doorway_reference()
        )
        return self.workspace.doors_workspace.doors()[0]

    def _install_window_frame(self):
        """Install one stable wall frame and return its viewer reference."""

        frame = WindowData(
            window_id="window-frame-a",
            wall_surface_id=self.wall.surface_id,
            start_ratio=0.25,
            end_ratio=0.75,
            bottom_ratio=0.2,
            top_ratio=0.8,
        )
        self.level.windows = [frame]
        self.workspace._set_canvas_viewer_targets((self.wall,))
        return next(
            target.reference
            for target in self.workspace._canvas_opening_targets_by_key.values()
            if target.reference.kind == CANVAS_OPENING_WINDOW
        )

    def _create_window(self):
        """Create one reusable window from the installed frame."""

        reference = self._install_window_frame()
        self.workspace._handle_window_creation_requested(reference)
        return self.workspace.doors_workspace.windows()[0], reference

    def _wait_for_generation_jobs(
        self,
        timeout_milliseconds: int = 15_000,
    ) -> None:
        """Process GUI events until every object mutation worker finishes."""

        deadline = time.monotonic() + timeout_milliseconds / 1000.0
        while self.workspace.generation._object_job_runtimes:
            if time.monotonic() >= deadline:
                self.fail("The opening-component edit did not finish in time.")
            _qt_application.processEvents()
            time.sleep(0.01)

    def _register_generated_component(self, door, slot_id):
        """Supply and join one hardware model that has no default mesh."""

        slot = door.get_slot(slot_id)
        assert slot is not None
        model = build_door_body_model(
            0.08,
            0.12,
            0.04,
            shape="rectangular",
            arch_amount=0.0,
            geometry_name=f"generated_{slot_id}_fixture",
        )
        self.workspace.generation.register_door_component_model(
            object_id=slot.source_object_id,
            object_name=f"{door.name} - {slot.display_name}",
            door_id=door.door_id,
            slot_id=slot.slot_id,
            model=model,
        )
        self.workspace.doors_workspace.replace_slot(
            door.door_id,
            slot.with_transform(joined=True),
        )
        self.workspace._sync_selected_door_slot_preview()
        return slot

    def _register_unjoined_generated_component(self, door, slot_id):
        """Supply a hardware asset without bypassing the completion handler."""

        slot = door.get_slot(slot_id)
        assert slot is not None
        model = build_door_body_model(
            0.08,
            0.12,
            0.04,
            shape="rectangular",
            arch_amount=0.0,
            geometry_name=f"generated_unjoined_{slot_id}_fixture",
        )
        record = self.workspace.generation.register_door_component_model(
            object_id=slot.source_object_id,
            object_name=f"{door.name} - {slot.display_name}",
            door_id=door.door_id,
            slot_id=slot.slot_id,
            model=model,
        )
        return slot, record, model

    def _register_generated_knob(self, door):
        """Supply the knob model used by the existing transform tests."""

        return self._register_generated_component(door, DOOR_SLOT_KNOB)

    def _register_asymmetric_handle(self, asset, *, asset_kind):
        """Join a handle whose overhang makes the assembly bounds asymmetric."""

        slot = asset.get_slot(DOOR_SLOT_KNOB)
        assert slot is not None
        model = build_door_body_model(
            0.08,
            0.12,
            0.04,
            shape="rectangular",
            arch_amount=0.0,
            geometry_name=f"asymmetric_{asset_kind}_placement_handle",
        )
        self.workspace.generation.register_door_component_model(
            object_id=slot.source_object_id,
            object_name=f"{asset.name} - Handle",
            door_id=(
                asset.door_id
                if asset_kind == ASSET_KIND_DOOR
                else asset.window_id
            ),
            slot_id=slot.slot_id,
            model=model,
            asset_kind=asset_kind,
        )
        updated_slot = slot.with_transform(
            position_meters=(
                asset.width_meters * 1.5,
                asset.thickness_meters + 0.5,
                asset.height_meters * 0.5,
            ),
            joined=True,
        )
        updated = asset.replace_slot(updated_slot)
        if asset_kind == ASSET_KIND_DOOR:
            self.workspace.doors_workspace.upsert_door(updated)
        else:
            self.workspace.doors_workspace.upsert_window(updated)
        return updated

    @staticmethod
    def _combine_door_preview_parts(parts):
        """Combine each preview component into one assertion-friendly mesh."""

        return {
            part.component_kind: trimesh.util.concatenate(part.meshes)
            for part in parts
        }

    def _assert_preview_body_bottom_center(
        self,
        parts,
        transform,
        expected_center,
    ) -> None:
        """Assert that one colored body stays centered on its opening anchor."""

        body_part = next(
            part for part in parts if part.component_kind == DOOR_SLOT_BODY
        )
        body = trimesh.util.concatenate(body_part.meshes)
        body.apply_transform(transform)
        minimum, maximum = np.asarray(body.bounds, dtype=float)
        bottom_center = np.asarray(
            (
                (minimum[0] + maximum[0]) * 0.5,
                (minimum[1] + maximum[1]) * 0.5,
                minimum[2],
            ),
            dtype=float,
        )
        np.testing.assert_allclose(bottom_center, expected_center, atol=1e-8)

    def test_plus_creation_builds_hidden_slot_records_and_selects_body(self) -> None:
        door = self._create_door()

        self.assertEqual(door.name, "New door 1")
        self.assertEqual(
            tuple(slot.slot_id for slot in door.slots),
            (DOOR_SLOT_BODY, DOOR_SLOT_KNOB),
        )
        self.assertEqual(self.workspace.canvas_door_list.count(), 1)
        self.assertEqual(
            self.workspace.doors_workspace.selected_slot_id(),
            DOOR_SLOT_BODY,
        )
        self.assertEqual(len(self.workspace.generation.get_data().generated_objects), 1)
        self.assertEqual(self.workspace.generation.get_generated_object_ids(), ())
        self.assertFalse(
            self.workspace.generation.generate_geometry_button.isEnabled()
        )
        self.assertFalse(self.workspace.generation.place_button.isEnabled())
        self.assertFalse(
            self.workspace.generation.object_3d_panel.projection_camera_controls.isEnabled()
        )
        self.assertTrue(
            self.workspace.merged_generation_workspace.infer_ceiling_height_button.isEnabled()
        )

    def test_combined_workspace_and_window_frame_labels_are_visible(self) -> None:
        doors_index = self.workspace.workspace_tabs.indexOf(
            self.workspace.doors_workspace
        )

        self.assertGreaterEqual(doors_index, 0)
        self.assertEqual(
            self.workspace.workspace_tabs.tabText(doors_index),
            "Doors and Windows",
        )
        self.assertEqual(
            self.workspace.add_window_button.text(),
            "Add window frame",
        )
        self.assertIsNone(self.workspace.viewer.window_tools_status_label)
        self.assertIsNone(self.workspace.viewer.undo_window_button)

    def test_delete_button_removes_and_undo_restores_a_door(self) -> None:
        door = self._create_door()
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        self.workspace._handle_door_placement_requested(
            door.door_id,
            self._doorway_reference(),
            False,
        )
        original_placement = self.workspace.doors_workspace.data().placements[0]
        scene_object_id = f"door-placement:{original_placement.placement_id}"
        self.assertIn(
            scene_object_id,
            self.workspace.viewer._placed_object_level_indices,
        )

        self.workspace.doors_workspace.delete_asset_button.click()

        deleted = self.workspace.doors_workspace.data()
        self.assertIsNone(deleted.get_door(door.door_id))
        self.assertEqual(deleted.placements, [])
        self.assertIsNone(
            self.workspace.generation.get_generated_object_model(
                body.source_object_id
            )
        )
        self.assertNotIn(
            scene_object_id,
            self.workspace.viewer._placed_object_level_indices,
        )

        self.workspace._handle_doors_undo_requested()

        restored = self.workspace.doors_workspace.data()
        self.assertEqual(restored.get_door(door.door_id), door)
        self.assertEqual(restored.placements, [original_placement])
        self.assertIsNotNone(
            self.workspace.generation.get_generated_object_model(
                body.source_object_id
            )
        )
        self.assertIn(
            scene_object_id,
            self.workspace.viewer._placed_object_level_indices,
        )
        self.assertEqual(
            self.workspace.doors_workspace.selected_door_id(),
            door.door_id,
        )

    def test_delete_button_removes_and_undo_restores_a_window(self) -> None:
        window, reference = self._create_window()
        body = window.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        self.workspace._handle_window_fixture_placement_requested(
            window.window_id,
            reference,
            False,
        )
        self.assertEqual(
            len(self.workspace.doors_workspace.data().window_placements),
            1,
        )

        self.workspace.doors_workspace.delete_asset_button.click()

        deleted = self.workspace.doors_workspace.data()
        self.assertIsNone(deleted.get_window(window.window_id))
        self.assertEqual(deleted.window_placements, [])
        self.assertIsNone(
            self.workspace.generation.get_generated_object_model(
                body.source_object_id
            )
        )

        self.workspace._handle_doors_undo_requested()

        restored = self.workspace.doors_workspace.data()
        self.assertEqual(restored.get_window(window.window_id), window)
        self.assertEqual(len(restored.window_placements), 1)
        self.assertIsNotNone(
            self.workspace.generation.get_generated_object_model(
                body.source_object_id
            )
        )
        self.assertEqual(
            self.workspace.doors_workspace.selected_window_id(),
            window.window_id,
        )

    def test_scene_delete_removes_installed_door_before_its_doorway(self) -> None:
        door = self._create_door()
        reference = self._doorway_reference()
        self.workspace._handle_door_placement_requested(
            door.door_id,
            reference,
            False,
        )

        self.assertTrue(self.workspace.viewer.select_canvas_opening(reference))
        self.workspace.viewer._handle_view_delete_requested()

        library = self.workspace.doors_workspace.data()
        self.assertEqual(library.placements, [])
        self.assertEqual(library.get_door(door.door_id), door)
        self.assertEqual(len(self.level.doorways), 1)

    def test_scene_door_selection_redirects_to_its_owning_doorway(self) -> None:
        """Canvas must expose an installed door through its structural opening."""

        door = self._create_door()
        reference = self._doorway_reference()
        self.workspace._handle_door_placement_requested(
            door.door_id,
            reference,
            False,
        )
        placement = self.workspace.doors_workspace.data().placements[0]
        scene_object_id = f"door-placement:{placement.placement_id}"

        self.workspace.viewer.select_placed_object(scene_object_id)

        self.assertEqual(
            self.workspace.viewer.get_selected_canvas_opening_reference(),
            reference,
        )
        self.assertEqual(
            self.workspace.viewer.get_selected_placed_object_ids(),
            (),
        )
        self.assertEqual(self.workspace.canvas.selected_doorway_index, 0)

    def test_scene_delete_detaches_window_without_deleting_reusable_asset(
        self,
    ) -> None:
        """Delete on a selected frame removes only its installed window."""

        window, reference = self._create_window()
        body = window.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        self.workspace._handle_window_fixture_placement_requested(
            window.window_id,
            reference,
            False,
        )
        self.assertEqual(
            len(self.workspace.doors_workspace.data().window_placements),
            1,
        )
        self.assertIsNotNone(
            self.workspace.generation.get_generated_object_model(
                body.source_object_id
            )
        )

        self.assertTrue(self.workspace.viewer.select_canvas_opening(reference))
        self.workspace.viewer._handle_view_delete_requested()

        detached = self.workspace.doors_workspace.data()
        self.assertEqual(detached.window_placements, [])
        self.assertEqual(detached.get_window(window.window_id), window)
        self.assertEqual(self.level.windows[0].window_id, reference.stable_id)
        self.assertIsNotNone(
            self.workspace.generation.get_generated_object_model(
                body.source_object_id
            )
        )

        self.workspace._handle_window_fixture_placement_requested(
            window.window_id,
            reference,
            False,
        )
        self.assertEqual(
            len(self.workspace.doors_workspace.data().window_placements),
            1,
        )

    def test_horizontal_mesh_flip_toggles_door_slot_and_is_undoable(self) -> None:
        """Flip one door component through the asset-aware main handler."""

        door = self._create_door()
        original_slot = door.get_slot(DOOR_SLOT_BODY)
        assert original_slot is not None

        self.workspace._handle_opening_asset_slot_horizontal_flip_requested(
            ASSET_KIND_DOOR,
            door.door_id,
            original_slot.slot_id,
        )

        flipped_door = self.workspace.doors_workspace.data().get_door(
            door.door_id
        )
        assert flipped_door is not None
        flipped_slot = flipped_door.get_slot(original_slot.slot_id)
        assert flipped_slot is not None
        self.assertEqual(
            flipped_slot.mirrored_horizontally,
            not original_slot.mirrored_horizontally,
        )

        self.workspace._handle_doors_undo_requested()

        restored = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert restored is not None
        self.assertEqual(restored.get_slot(original_slot.slot_id), original_slot)

    def test_horizontal_mesh_flip_toggles_window_slot_and_is_undoable(
        self,
    ) -> None:
        """Flip one window component through the shared opening handler."""

        window, _reference = self._create_window()
        original_slot = window.get_slot(DOOR_SLOT_KNOB)
        assert original_slot is not None

        self.workspace._handle_opening_asset_slot_horizontal_flip_requested(
            ASSET_KIND_WINDOW,
            window.window_id,
            original_slot.slot_id,
        )

        flipped_window = self.workspace.doors_workspace.data().get_window(
            window.window_id
        )
        assert flipped_window is not None
        flipped_slot = flipped_window.get_slot(original_slot.slot_id)
        assert flipped_slot is not None
        self.assertEqual(
            flipped_slot.mirrored_horizontally,
            not original_slot.mirrored_horizontally,
        )

        self.workspace._handle_doors_undo_requested()

        restored = self.workspace.doors_workspace.data().get_window(
            window.window_id
        )
        assert restored is not None
        self.assertEqual(restored.get_slot(original_slot.slot_id), original_slot)

    def test_proportional_horizontal_flip_toggles_both_door_leaves_and_undoes(
        self,
    ) -> None:
        """Flip paired double-door meshes as one undoable edit."""

        door = self._create_door()
        self.workspace._handle_door_make_double_changed(door.door_id, True)
        self.assertTrue(
            self.workspace.doors_workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_BODY,
            )
        )
        self.assertTrue(
            self.workspace.doors_workspace.is_proportional_editing_enabled()
        )
        original = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert original is not None
        original_right = original.get_slot(DOOR_SLOT_BODY)
        original_left = original.get_slot(DOOR_SLOT_LEFT_BODY)
        assert original_right is not None and original_left is not None

        self.workspace._handle_opening_asset_slot_horizontal_flip_requested(
            ASSET_KIND_DOOR,
            door.door_id,
            original_right.slot_id,
        )

        flipped = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert flipped is not None
        flipped_right = flipped.get_slot(DOOR_SLOT_BODY)
        flipped_left = flipped.get_slot(DOOR_SLOT_LEFT_BODY)
        assert flipped_right is not None and flipped_left is not None
        self.assertEqual(
            flipped_right.mirrored_horizontally,
            not original_right.mirrored_horizontally,
        )
        self.assertEqual(
            flipped_left.mirrored_horizontally,
            not original_left.mirrored_horizontally,
        )

        self.workspace._handle_doors_undo_requested()

        restored = self.workspace.doors_workspace.data().get_door(door.door_id)
        self.assertEqual(restored, original)

    def test_proportional_horizontal_flip_toggles_both_window_leaves_and_undoes(
        self,
    ) -> None:
        """Flip paired double-window meshes as one undoable edit."""

        window, _reference = self._create_window()
        self.workspace._handle_window_make_double_changed(
            window.window_id,
            True,
        )
        self.assertTrue(
            self.workspace.doors_workspace.select_window_slot(
                window.window_id,
                DOOR_SLOT_KNOB,
            )
        )
        self.assertTrue(
            self.workspace.doors_workspace.is_proportional_editing_enabled()
        )
        original = self.workspace.doors_workspace.data().get_window(
            window.window_id
        )
        assert original is not None
        original_right = original.get_slot(DOOR_SLOT_KNOB)
        original_left = original.get_slot(DOOR_SLOT_LEFT_KNOB)
        assert original_right is not None and original_left is not None

        self.workspace._handle_opening_asset_slot_horizontal_flip_requested(
            ASSET_KIND_WINDOW,
            window.window_id,
            original_right.slot_id,
        )

        flipped = self.workspace.doors_workspace.data().get_window(
            window.window_id
        )
        assert flipped is not None
        flipped_right = flipped.get_slot(DOOR_SLOT_KNOB)
        flipped_left = flipped.get_slot(DOOR_SLOT_LEFT_KNOB)
        assert flipped_right is not None and flipped_left is not None
        self.assertEqual(
            flipped_right.mirrored_horizontally,
            not original_right.mirrored_horizontally,
        )
        self.assertEqual(
            flipped_left.mirrored_horizontally,
            not original_left.mirrored_horizontally,
        )

        self.workspace._handle_doors_undo_requested()

        restored = self.workspace.doors_workspace.data().get_window(
            window.window_id
        )
        self.assertEqual(restored, original)

    def test_horizontal_flip_leaves_counterpart_when_proportional_is_off(
        self,
    ) -> None:
        """Flip only the selected double-door leaf when pairing is disabled."""

        door = self._create_door()
        self.workspace._handle_door_make_double_changed(door.door_id, True)
        self.workspace.doors_workspace.select_door_slot(
            door.door_id,
            DOOR_SLOT_BODY,
        )
        self.workspace.doors_workspace.proportional_editing_checkbox.setChecked(
            False
        )
        original = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert original is not None
        original_right = original.get_slot(DOOR_SLOT_BODY)
        original_left = original.get_slot(DOOR_SLOT_LEFT_BODY)
        assert original_right is not None and original_left is not None

        self.workspace._handle_opening_asset_slot_horizontal_flip_requested(
            ASSET_KIND_DOOR,
            door.door_id,
            original_right.slot_id,
        )

        flipped = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert flipped is not None
        flipped_right = flipped.get_slot(DOOR_SLOT_BODY)
        flipped_left = flipped.get_slot(DOOR_SLOT_LEFT_BODY)
        assert flipped_right is not None and flipped_left is not None
        self.assertEqual(
            flipped_right.mirrored_horizontally,
            not original_right.mirrored_horizontally,
        )
        self.assertEqual(flipped_left, original_left)

        self.workspace._handle_doors_undo_requested()

        restored = self.workspace.doors_workspace.data().get_door(door.door_id)
        self.assertEqual(restored, original)

    def test_canvas_delete_removes_installed_door_before_its_doorway(self) -> None:
        door = self._create_door()
        reference = self._doorway_reference()
        self.workspace._handle_door_placement_requested(
            door.door_id,
            reference,
            False,
        )
        self.workspace._sync_canvas_to_current_level(preserve_view=True)
        self.workspace.canvas._set_selected_doorway_index(reference.item_index)

        self.assertTrue(
            self.workspace.canvas._request_selected_opening_fixture_deletion()
        )

        library = self.workspace.doors_workspace.data()
        self.assertEqual(library.placements, [])
        self.assertEqual(library.get_door(door.door_id), door)
        self.assertEqual(len(self.level.doorways), 1)

    def test_canvas_delete_removes_installed_window_before_its_frame(self) -> None:
        window, reference = self._create_window()
        self.workspace._handle_window_fixture_placement_requested(
            window.window_id,
            reference,
            False,
        )
        self.workspace._sync_canvas_to_current_level(preserve_view=True)
        self.workspace.canvas._set_selected_window_id(
            reference.stable_id,
            notify=False,
        )

        self.assertTrue(
            self.workspace.canvas._request_selected_opening_fixture_deletion()
        )

        library = self.workspace.doors_workspace.data()
        self.assertEqual(library.window_placements, [])
        self.assertEqual(library.get_window(window.window_id), window)
        self.assertEqual(len(self.level.windows), 1)

    def test_delete_undo_preserves_assets_created_after_deletion(self) -> None:
        door = self._create_door()
        door_body = door.get_slot(DOOR_SLOT_BODY)
        assert door_body is not None
        self.workspace._handle_door_placement_requested(
            door.door_id,
            self._doorway_reference(),
            False,
        )
        original_door_placement = (
            self.workspace.doors_workspace.data().placements[0]
        )
        self.workspace.doors_workspace.delete_asset_button.click()

        window, reference = self._create_window()
        window_body = window.get_slot(DOOR_SLOT_BODY)
        assert window_body is not None
        self.workspace._handle_window_fixture_placement_requested(
            window.window_id,
            reference,
            False,
        )
        current_window_placement = (
            self.workspace.doors_workspace.data().window_placements[0]
        )

        self.workspace._handle_doors_undo_requested()

        restored = self.workspace.doors_workspace.data()
        self.assertEqual(restored.get_door(door.door_id), door)
        self.assertEqual(restored.get_window(window.window_id), window)
        self.assertEqual(restored.placements, [original_door_placement])
        self.assertEqual(
            restored.window_placements,
            [current_window_placement],
        )
        self.assertIsNotNone(
            self.workspace.generation.get_generated_object_model(
                door_body.source_object_id
            )
        )
        self.assertIsNotNone(
            self.workspace.generation.get_generated_object_model(
                window_body.source_object_id
            )
        )

    def test_reusable_window_is_created_and_refits_to_edited_frame(self) -> None:
        window, reference = self._create_window()

        self.assertEqual(window.name, "New window 1")
        self.assertEqual(
            tuple(slot.display_name for slot in window.slots),
            ("Body", "Handle"),
        )
        self.assertEqual(self.workspace.canvas_window_list.count(), 1)
        self.assertEqual(
            self.workspace.doors_workspace.selected_asset_kind(),
            ASSET_KIND_WINDOW,
        )

        self.workspace._handle_window_fixture_placement_requested(
            window.window_id,
            reference,
            False,
        )

        library = self.workspace.doors_workspace.data()
        self.assertEqual(len(library.window_placements), 1)
        placed = next(
            model
            for model in self.workspace._build_placed_generated_models(
                include_excluded_levels=True
            )
            if model.object_id.startswith("window-placement:")
        )
        np.testing.assert_allclose(
            placed.model.mesh.extents[[0, 2]],
            (2.0, 1.8),
            atol=1e-8,
        )

        self.level.windows[0] = replace(
            self.level.windows[0],
            start_ratio=0.125,
            end_ratio=0.875,
            bottom_ratio=0.1,
            top_ratio=0.9,
        )
        self.workspace._set_canvas_viewer_targets((self.wall,))
        resized = next(
            model
            for model in self.workspace._build_placed_generated_models(
                include_excluded_levels=True
            )
            if model.object_id.startswith("window-placement:")
        )
        np.testing.assert_allclose(
            resized.model.mesh.extents[[0, 2]],
            (3.0, 2.4),
            atol=1e-8,
        )

    def test_window_make_double_builds_independent_leaves_and_fits_frame(
        self,
    ) -> None:
        window, reference = self._create_window()

        self.workspace._handle_opening_asset_make_double_changed(
            ASSET_KIND_WINDOW,
            window.window_id,
            True,
        )

        updated = self.workspace.doors_workspace.data().get_window(
            window.window_id
        )
        assert updated is not None
        self.assertTrue(updated.make_double)
        self.assertEqual(
            tuple(slot.slot_id for slot in updated.slots),
            (
                DOOR_SLOT_BODY,
                DOOR_SLOT_LEFT_BODY,
                DOOR_SLOT_KNOB,
                DOOR_SLOT_LEFT_KNOB,
            ),
        )
        right_body = updated.get_slot(DOOR_SLOT_BODY)
        left_body = updated.get_slot(DOOR_SLOT_LEFT_BODY)
        assert right_body is not None and left_body is not None
        self.assertTrue(left_body.source_object_id.startswith("window:"))
        self.assertIsNotNone(
            self.workspace.generation.get_generated_object_model(
                left_body.source_object_id
            )
        )

        self.workspace._handle_window_fixture_placement_requested(
            updated.window_id,
            reference,
            False,
        )
        placed = next(
            model
            for model in self.workspace._build_placed_generated_models(
                include_excluded_levels=True
            )
            if model.object_id.startswith("window-placement:")
        )
        np.testing.assert_allclose(
            placed.model.mesh.extents[[0, 2]],
            (2.0, 1.8),
            atol=1e-8,
        )

    def test_window_double_sided_hierarchy_and_records_are_undoable(self) -> None:
        window, _reference = self._create_window()
        self.workspace._handle_opening_asset_make_double_changed(
            ASSET_KIND_WINDOW,
            window.window_id,
            True,
        )
        self.workspace._handle_opening_asset_make_double_sided_changed(
            ASSET_KIND_WINDOW,
            window.window_id,
            True,
        )

        updated = self.workspace.doors_workspace.data().get_window(
            window.window_id
        )
        assert updated is not None
        self.assertTrue(updated.make_double)
        self.assertTrue(updated.make_double_sided)
        self.assertEqual(
            {slot.slot_id for slot in updated.slots},
            {
                DOOR_SLOT_BODY,
                DOOR_SLOT_LEFT_BODY,
                DOOR_SLOT_KNOB,
                DOOR_SLOT_LEFT_KNOB,
                DOOR_SLOT_BACK_BODY,
                DOOR_SLOT_BACK_LEFT_BODY,
                DOOR_SLOT_BACK_KNOB,
                DOOR_SLOT_BACK_LEFT_KNOB,
            },
        )
        for slot_id in (DOOR_SLOT_BACK_BODY, DOOR_SLOT_BACK_LEFT_BODY):
            slot = updated.get_slot(slot_id)
            assert slot is not None
            self.assertIsNotNone(
                self.workspace.generation.get_generated_object_model(
                    slot.source_object_id
                )
            )

        self.workspace._handle_doors_undo_requested()
        restored_double = self.workspace.doors_workspace.data().get_window(
            window.window_id
        )
        assert restored_double is not None
        self.assertTrue(restored_double.make_double)
        self.assertFalse(restored_double.make_double_sided)

        self.workspace._handle_doors_undo_requested()
        restored_single = self.workspace.doors_workspace.data().get_window(
            window.window_id
        )
        assert restored_single is not None
        self.assertFalse(restored_single.make_double)
        self.assertEqual(
            tuple(slot.slot_id for slot in restored_single.slots),
            (DOOR_SLOT_BODY, DOOR_SLOT_KNOB),
        )

    def test_pasting_a_double_window_part_clones_and_mirrors_its_mesh(
        self,
    ) -> None:
        window, _reference = self._create_window()
        handle = window.get_slot(DOOR_SLOT_KNOB)
        assert handle is not None
        handle_model = build_door_body_model(
            0.08,
            0.12,
            0.04,
            shape="rectangular",
            arch_amount=0.0,
            geometry_name="generated_window_handle_fixture",
        )
        self.workspace.generation.register_door_component_model(
            object_id=handle.source_object_id,
            object_name=f"{window.name} - Handle",
            door_id=window.window_id,
            slot_id=handle.slot_id,
            model=handle_model,
            asset_kind=ASSET_KIND_WINDOW,
        )
        self.workspace.doors_workspace.replace_window_slot(
            window.window_id,
            handle.with_transform(joined=True),
        )
        self.workspace._handle_opening_asset_make_double_changed(
            ASSET_KIND_WINDOW,
            window.window_id,
            True,
        )
        before = self.workspace.doors_workspace.data().get_window(
            window.window_id
        )
        assert before is not None
        source = before.get_slot(DOOR_SLOT_KNOB)
        original_target = before.get_slot(DOOR_SLOT_LEFT_KNOB)
        assert source is not None and original_target is not None
        source = source.with_transform(
            position_meters=(0.22, -0.01, 0.64),
            rotation_degrees=(-5.0, 15.0, -25.0),
            axis_scales=(0.8, 1.3, 1.1),
        )
        self.workspace.doors_workspace.replace_window_slot(
            window.window_id,
            source,
        )

        self.workspace._handle_opening_asset_slot_mesh_paste_requested(
            ASSET_KIND_WINDOW,
            window.window_id,
            source.slot_id,
            original_target.slot_id,
        )

        updated = self.workspace.doors_workspace.data().get_window(
            window.window_id
        )
        assert updated is not None
        pasted = updated.get_slot(DOOR_SLOT_LEFT_KNOB)
        assert pasted is not None
        self.assertNotEqual(
            pasted.source_object_id,
            original_target.source_object_id,
        )
        self.assertTrue(pasted.source_object_id.startswith("window:"))
        self.assertEqual(pasted.position_meters, (-0.22, -0.01, 0.64))
        self.assertEqual(pasted.rotation_degrees, (-5.0, -15.0, 25.0))
        self.assertEqual(pasted.axis_scales, source.axis_scales)
        self.assertEqual(pasted.joined, source.joined)
        self.assertEqual(
            pasted.mirrored_horizontally,
            not source.mirrored_horizontally,
        )
        self.assertIsNotNone(
            self.workspace.generation.get_generated_object_model(
                pasted.source_object_id
            )
        )

    def test_make_double_builds_independent_leaves_and_one_full_placement(
        self,
    ) -> None:
        door = self._create_door()
        reference = self._doorway_reference()

        self.workspace._handle_door_make_double_changed(door.door_id, True)

        updated = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert updated is not None
        self.assertTrue(updated.make_double)
        self.assertEqual(
            tuple(slot.slot_id for slot in updated.slots),
            (
                DOOR_SLOT_BODY,
                DOOR_SLOT_LEFT_BODY,
                DOOR_SLOT_KNOB,
                DOOR_SLOT_LEFT_KNOB,
            ),
        )
        right_body = updated.get_slot(DOOR_SLOT_BODY)
        left_body = updated.get_slot(DOOR_SLOT_LEFT_BODY)
        assert right_body is not None and left_body is not None
        self.assertNotEqual(
            right_body.source_object_id,
            left_body.source_object_id,
        )
        self.assertIsNotNone(
            self.workspace.generation.get_generated_object_model(
                left_body.source_object_id
            )
        )

        self.workspace._handle_door_placement_requested(
            updated.door_id,
            reference,
            False,
        )

        placements = self.workspace.doors_workspace.data().placements
        self.assertEqual(len(placements), 1)
        placed = next(
            model
            for model in self.workspace._build_placed_generated_models(
                include_excluded_levels=True
            )
            if model.object_id.startswith("door-placement:")
        )
        np.testing.assert_allclose(
            placed.model.mesh.extents[[0, 2]],
            (1.0, 2.0),
            atol=1e-8,
        )

    def test_disabling_make_double_detaches_left_models_until_undo(self) -> None:
        door = self._create_door()
        self._register_generated_component(door, DOOR_SLOT_KNOB)
        self.workspace._handle_door_make_double_changed(door.door_id, True)
        doubled = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert doubled is not None
        left_object_ids = {
            slot.source_object_id
            for slot in doubled.slots
            if slot.slot_id in {DOOR_SLOT_LEFT_BODY, DOOR_SLOT_LEFT_KNOB}
        }
        self.assertEqual(len(left_object_ids), 2)

        self.workspace._handle_door_make_double_changed(door.door_id, False)

        current_record_ids = {
            record.object_id
            for record in self.workspace.generation.get_generated_object_records()
        }
        self.assertTrue(left_object_ids.isdisjoint(current_record_ids))

        self.workspace._handle_doors_undo_requested()

        restored = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert restored is not None
        self.assertTrue(restored.make_double)
        restored_record_ids = {
            record.object_id
            for record in self.workspace.generation.get_generated_object_records()
        }
        self.assertTrue(left_object_ids.issubset(restored_record_ids))

    def test_pasting_a_double_door_part_clones_and_mirrors_its_mesh(
        self,
    ) -> None:
        door = self._create_door()
        self._register_generated_component(door, DOOR_SLOT_KNOB)
        self.workspace._handle_door_make_double_changed(door.door_id, True)
        before = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert before is not None
        source = before.get_slot(DOOR_SLOT_KNOB)
        original_target = before.get_slot(DOOR_SLOT_LEFT_KNOB)
        assert source is not None and original_target is not None
        source = source.with_transform(
            position_meters=(0.18, 0.03, 0.92),
            rotation_degrees=(10.0, 20.0, 30.0),
            axis_scales=(1.1, 0.9, 1.2),
        )
        self.workspace.doors_workspace.replace_slot(door.door_id, source)

        self.workspace._handle_door_slot_mesh_paste_requested(
            door.door_id,
            source.slot_id,
            original_target.slot_id,
        )

        updated = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert updated is not None
        pasted = updated.get_slot(DOOR_SLOT_LEFT_KNOB)
        assert pasted is not None
        self.assertNotEqual(
            pasted.source_object_id,
            original_target.source_object_id,
        )
        self.assertEqual(pasted.position_meters, (-0.18, 0.03, 0.92))
        self.assertEqual(pasted.rotation_degrees, (10.0, -20.0, -30.0))
        self.assertEqual(pasted.axis_scales, source.axis_scales)
        self.assertEqual(pasted.joined, source.joined)
        self.assertEqual(
            pasted.mirrored_horizontally,
            not source.mirrored_horizontally,
        )
        self.assertIsNotNone(
            self.workspace.generation.get_generated_object_model(
                pasted.source_object_id
            )
        )
        current_record_ids = {
            record.object_id
            for record in self.workspace.generation.get_generated_object_records()
        }
        self.assertNotIn(original_target.source_object_id, current_record_ids)

        self.workspace._handle_doors_undo_requested()

        restored = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert restored is not None
        restored_target = restored.get_slot(DOOR_SLOT_LEFT_KNOB)
        assert restored_target is not None
        self.assertEqual(
            restored_target.source_object_id,
            original_target.source_object_id,
        )
        restored_record_ids = {
            record.object_id
            for record in self.workspace.generation.get_generated_object_records()
        }
        self.assertIn(original_target.source_object_id, restored_record_ids)
        self.assertNotIn(pasted.source_object_id, restored_record_ids)

    def test_pasting_a_double_door_body_replaces_stale_counterpart_transform(
        self,
    ) -> None:
        door = self._create_door()
        self.workspace._handle_door_make_double_changed(door.door_id, True)
        before = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert before is not None
        source = before.get_slot(DOOR_SLOT_BODY)
        target = before.get_slot(DOOR_SLOT_LEFT_BODY)
        assert source is not None and target is not None
        self.workspace.doors_workspace.replace_slot(
            door.door_id,
            target.with_transform(
                position_meters=(-0.9, 0.0, 0.0),
                axis_scales=(0.2, 1.0, 1.0),
            ),
        )

        self.workspace._handle_door_slot_mesh_paste_requested(
            door.door_id,
            source.slot_id,
            target.slot_id,
        )

        updated = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert updated is not None
        pasted = updated.get_slot(DOOR_SLOT_LEFT_BODY)
        assert pasted is not None
        self.assertEqual(
            pasted.position_meters,
            (-source.position_meters[0], *source.position_meters[1:]),
        )
        self.assertEqual(pasted.axis_scales, source.axis_scales)
        preview = self.workspace._build_door_assembly_preview(updated)
        body_bounds = {}
        for part in preview.preview_placed_objects:
            if part.object_id not in {
                source.source_object_id,
                pasted.source_object_id,
            }:
                continue
            mesh = part.meshes[0].copy()
            mesh.apply_transform(part.placement_transform)
            body_bounds[part.object_id] = mesh.bounds
        self.assertAlmostEqual(
            body_bounds[pasted.source_object_id][1, 0],
            body_bounds[source.source_object_id][0, 0],
        )

    def test_make_double_cleanup_preserves_records_owned_by_another_door(
        self,
    ) -> None:
        first = self._create_door()
        self._register_generated_component(first, DOOR_SLOT_KNOB)
        self.workspace._handle_door_make_double_changed(first.door_id, True)
        first = self.workspace.doors_workspace.data().get_door(first.door_id)
        assert first is not None
        shared = first.get_slot(DOOR_SLOT_LEFT_KNOB)
        source = first.get_slot(DOOR_SLOT_KNOB)
        assert shared is not None and source is not None

        second = replace(
            first,
            door_id="second-door",
            name="New door 2",
            slots=tuple(
                replace(slot, source_object_id=shared.source_object_id)
                if slot.slot_id == DOOR_SLOT_LEFT_KNOB
                else slot
                for slot in first.slots
            ),
        )
        self.workspace.doors_workspace.upsert_door(second)

        self.workspace._handle_door_slot_mesh_paste_requested(
            first.door_id,
            source.slot_id,
            shared.slot_id,
        )

        record_ids = {
            record.object_id
            for record in self.workspace.generation.get_generated_object_records()
        }
        self.assertIn(shared.source_object_id, record_ids)

    def test_window_glass_area_can_be_undone_and_survives_project_restore(
        self,
    ) -> None:
        window, _reference = self._create_window()
        body = window.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        request = GlassAreaPlacementRequest(
            asset_kind=ASSET_KIND_WINDOW,
            asset_id=window.window_id,
            body_object_id=body.source_object_id,
            normalized_x_bounds=(0.2, 0.8),
            normalized_z_bounds=(0.25, 0.75),
        )

        with patch("housemaker.main.QMessageBox.warning") as warning:
            self.workspace._handle_opening_glass_area_placement_requested(request)
        warning.assert_not_called()

        edited = self.workspace.doors_workspace.data().get_window(window.window_id)
        assert edited is not None
        self.assertEqual(len(edited.glass_areas), 1)
        self.assertEqual(edited.glass_areas[0].name, "Glass area 1")
        self.assertEqual(
            self.workspace.doors_workspace.selected_glass_area_id(),
            edited.glass_areas[0].glass_area_id,
        )

        self.workspace._handle_doors_undo_requested()

        restored = self.workspace.doors_workspace.data().get_window(window.window_id)
        assert restored is not None
        self.assertEqual(restored.glass_areas, ())

        with patch("housemaker.main.QMessageBox.warning") as warning:
            self.workspace._handle_opening_glass_area_placement_requested(request)
        warning.assert_not_called()
        generation = self.workspace.generation.get_data()
        doors = self.workspace.doors_workspace.data()
        self.workspace._apply_project_state(
            levels=[copy.deepcopy(self.level)],
            current_level_index=0,
            generation=generation,
            doors=doors,
        )

        persisted = self.workspace.doors_workspace.data().get_window(window.window_id)
        assert persisted is not None
        self.assertEqual(len(persisted.glass_areas), 1)
        self.assertEqual(persisted.glass_areas[0].name, "Glass area 1")

    def test_add_glass_area_arms_the_selected_window_body_profile(self) -> None:
        window, _reference = self._create_window()
        body = window.get_slot(DOOR_SLOT_BODY)
        assert body is not None

        with patch.object(
            self.workspace.doors_workspace.preview_viewer,
            "begin_glass_area_placement",
            return_value=True,
        ) as begin_placement:
            self.workspace._handle_opening_glass_area_creation_requested(
                ASSET_KIND_WINDOW,
                window.window_id,
                body.source_object_id,
            )

        begin_placement.assert_called_once_with(
            ASSET_KIND_WINDOW,
            window.window_id,
            body.source_object_id,
            profile_bounds=(
                -window.width_meters * 0.5,
                window.width_meters * 0.5,
                0.0,
                window.height_meters,
            ),
        )

    def test_door_glass_area_is_cut_and_restored_by_undo(self) -> None:
        door = self._create_door()
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        request = GlassAreaPlacementRequest(
            asset_kind=ASSET_KIND_DOOR,
            asset_id=door.door_id,
            body_object_id=body.source_object_id,
            normalized_x_bounds=(0.25, 0.75),
            normalized_z_bounds=(0.3, 0.7),
        )

        with patch("housemaker.main.QMessageBox.warning") as warning:
            self.workspace._handle_opening_glass_area_placement_requested(request)
        warning.assert_not_called()
        edited = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert edited is not None
        self.assertEqual(len(edited.glass_areas), 1)
        preview = self.workspace._build_door_assembly_preview(edited)
        self.assertGreater(len(preview.mesh.faces), 0)

        self.workspace._handle_doors_undo_requested()

        restored = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert restored is not None
        self.assertEqual(restored.glass_areas, ())

    def test_door_creation_shows_the_complete_assembly_in_doors_preview(self) -> None:
        door = self._create_door()
        preview_model = self.workspace.doors_workspace.preview_viewer.model
        self.assertIsNotNone(preview_model)
        assert preview_model is not None
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        body_model = self.workspace.generation.get_generated_object_model(
            body.source_object_id
        )
        assert body_model is not None

        expected_model = self.workspace._build_door_assembly_preview(door)

        self.assertEqual(
            len(preview_model.mesh.faces),
            len(expected_model.mesh.faces),
        )
        self.assertEqual(len(preview_model.mesh.faces), len(body_model.mesh.faces))
        np.testing.assert_allclose(
            preview_model.mesh.bounds,
            expected_model.mesh.bounds,
        )

    def test_double_clicking_a_slot_opens_generation_with_that_slot_active(
        self,
    ) -> None:
        door = self._create_door()
        self.assertTrue(
            self.workspace.doors_workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.doors_workspace
        )
        selected_item = self.workspace.doors_workspace.door_tree.currentItem()
        assert selected_item is not None

        self.workspace.doors_workspace.door_tree.itemDoubleClicked.emit(
            selected_item,
            0,
        )
        _qt_application.processEvents()

        self.assertIs(
            self.workspace.workspace_tabs.currentWidget(),
            self.workspace.merged_generation_workspace,
        )
        self.assertEqual(
            self.workspace.doors_workspace.selected_slot_id(),
            DOOR_SLOT_KNOB,
        )
        target = self.workspace.generation._door_slot_editing_target
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.door_id, door.door_id)
        self.assertEqual(target.slot_id, DOOR_SLOT_KNOB)
        self.assertTrue(target.generation_ready)
        indicator = (
            self.workspace.merged_generation_workspace
            .door_slot_generation_indicator
        )
        self.assertEqual(
            indicator.text(),
            "- Generating the door's Handle",
        )
        self.assertFalse(indicator.isHidden())
        self.assertTrue(
            self.workspace.merged_generation_workspace
            ._door_slot_generation_blink_timer.isActive()
        )

    def test_successful_door_part_generation_returns_to_doors_workspace(
        self,
    ) -> None:
        door = self._create_door()
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        self.workspace._handle_opening_asset_slot_edit_requested(
            ASSET_KIND_DOOR,
            door.door_id,
            body.slot_id,
        )
        record = next(
            record
            for record in self.workspace.generation.get_generated_object_records()
            if record.object_id == body.source_object_id
        )
        model = self.workspace.generation.get_generated_object_model(
            body.source_object_id
        )
        assert model is not None

        self.workspace.generation.generated_object_changed.emit(record, model)
        _qt_application.processEvents()

        self.assertIs(
            self.workspace.workspace_tabs.currentWidget(),
            self.workspace.merged_generation_workspace,
        )

        self.workspace.generation.opening_component_generation_completed.emit(
            record,
            model,
        )
        _qt_application.processEvents()

        self.assertIs(
            self.workspace.workspace_tabs.currentWidget(),
            self.workspace.doors_workspace,
        )
        self.assertIsNone(
            self.workspace.generation._door_slot_editing_target
        )
        self.assertTrue(
            self.workspace.generation.symmetric_division_checkbox.isEnabled()
        )

    def test_successful_window_part_generation_returns_to_doors_workspace(
        self,
    ) -> None:
        window, _reference = self._create_window()
        body = window.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        self.workspace._handle_opening_asset_slot_edit_requested(
            ASSET_KIND_WINDOW,
            window.window_id,
            body.slot_id,
        )
        record = next(
            record
            for record in self.workspace.generation.get_generated_object_records()
            if record.object_id == body.source_object_id
        )
        model = self.workspace.generation.get_generated_object_model(
            body.source_object_id
        )
        assert model is not None

        self.workspace.generation.opening_component_generation_completed.emit(
            record,
            model,
        )
        _qt_application.processEvents()

        self.assertIs(
            self.workspace.workspace_tabs.currentWidget(),
            self.workspace.doors_workspace,
        )
        self.assertIsNone(
            self.workspace.generation._door_slot_editing_target
        )

    def test_door_and_window_face_actions_target_the_exact_component(
        self,
    ) -> None:
        door = self._create_door()
        window, _reference = self._create_window()
        door_body = door.get_slot(DOOR_SLOT_BODY)
        window_body = window.get_slot(DOOR_SLOT_BODY)
        assert door_body is not None and window_body is not None

        with (
            patch.object(
                self.workspace.generation,
                "delete_object_faces",
                return_value=True,
            ) as delete_faces,
            patch.object(
                self.workspace.generation,
                "convert_object_faces_to_glass",
                return_value=True,
            ) as convert_faces,
            patch.object(
                self.workspace.generation,
                "create_object_face",
                return_value=True,
            ) as create_face,
            patch("housemaker.main.QMessageBox.warning") as warning,
        ):
            self.assertTrue(
                self.workspace.doors_workspace.select_door_slot(
                    door.door_id,
                    DOOR_SLOT_BODY,
                )
            )
            self.workspace.doors_workspace.face_deletion_requested.emit(
                ASSET_KIND_DOOR,
                door.door_id,
                door_body.source_object_id,
                (1, 3),
            )
            self.workspace.doors_workspace.face_glass_conversion_requested.emit(
                ASSET_KIND_DOOR,
                door.door_id,
                door_body.source_object_id,
                (0, 2),
            )
            self.workspace.doors_workspace.face_creation_requested.emit(
                ASSET_KIND_DOOR,
                door.door_id,
                door_body.source_object_id,
                (0, 1, 2),
            )

            self.assertTrue(
                self.workspace.doors_workspace.select_window_slot(
                    window.window_id,
                    DOOR_SLOT_BODY,
                )
            )
            self.workspace.doors_workspace.face_deletion_requested.emit(
                ASSET_KIND_WINDOW,
                window.window_id,
                window_body.source_object_id,
                (4,),
            )
            self.workspace.doors_workspace.face_glass_conversion_requested.emit(
                ASSET_KIND_WINDOW,
                window.window_id,
                window_body.source_object_id,
                (5, 7),
            )
            self.workspace.doors_workspace.face_creation_requested.emit(
                ASSET_KIND_WINDOW,
                window.window_id,
                window_body.source_object_id,
                (3, 4, 5),
            )
            # A queued action from the previously selected door must not edit
            # either source after the selected asset changes.
            self.workspace.doors_workspace.face_deletion_requested.emit(
                ASSET_KIND_DOOR,
                door.door_id,
                door_body.source_object_id,
                (6,),
            )
            self.workspace.doors_workspace.face_glass_conversion_requested.emit(
                ASSET_KIND_DOOR,
                door.door_id,
                door_body.source_object_id,
                (8,),
            )
            self.workspace.doors_workspace.face_creation_requested.emit(
                ASSET_KIND_DOOR,
                door.door_id,
                door_body.source_object_id,
                (4, 5, 6),
            )

        warning.assert_not_called()

        self.assertEqual(
            delete_faces.call_args_list,
            [
                call(door_body.source_object_id, (1, 3)),
                call(window_body.source_object_id, (4,)),
            ],
        )
        self.assertEqual(
            convert_faces.call_args_list,
            [
                call(door_body.source_object_id, (0, 2)),
                call(window_body.source_object_id, (5, 7)),
            ],
        )
        self.assertEqual(
            create_face.call_args_list,
            [
                call(door_body.source_object_id, (0, 1, 2)),
                call(window_body.source_object_id, (3, 4, 5)),
            ],
        )

    def test_opening_face_glass_is_local_and_refreshes_selected_preview(
        self,
    ) -> None:
        """The Doors action needs no Meshy key or video reference."""

        door = self._create_door()
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        self.assertIn(":", body.source_object_id)
        self.assertTrue(
            self.workspace.doors_workspace.select_door_slot(
                door.door_id,
                body.slot_id,
            )
        )
        viewer = self.workspace.doors_workspace.preview_viewer
        initial_preview = viewer.model
        self.assertIsNotNone(initial_preview)
        assert initial_preview is not None
        initial_body = next(
            preview
            for preview in initial_preview.preview_placed_objects
            if preview.object_id == body.source_object_id
        )
        self.assertFalse(
            any(
                is_housemaker_glass_material(mesh.visual.material)
                for mesh in initial_body.meshes
            )
        )
        self.assertFalse(self.workspace.generation._settings.meshy_api_key)
        self.assertFalse(
            self.workspace.generation
            ._has_current_object_generation_reference()
        )
        viewer.set_selected_face_indices((0,))
        self.assertEqual(
            self.workspace.doors_workspace.add_glass_area_button.text(),
            "Convert faces to glass",
        )

        with patch("housemaker.main.QMessageBox.warning") as warning:
            self.workspace.doors_workspace.add_glass_area_button.click()
            self._wait_for_generation_jobs()

        warning.assert_not_called()
        self.assertIn(
            "to glass",
            self.workspace.generation.status_label.text().lower(),
        )
        updated_preview = viewer.model
        self.assertIsNotNone(updated_preview)
        assert updated_preview is not None
        self.assertIsNot(updated_preview, initial_preview)
        updated_body = next(
            preview
            for preview in updated_preview.preview_placed_objects
            if preview.object_id == body.source_object_id
        )
        self.assertTrue(
            any(
                is_housemaker_glass_material(mesh.visual.material)
                for mesh in updated_body.meshes
            )
        )
        self.assertEqual(
            self.workspace.doors_workspace.selected_slot_id(),
            body.slot_id,
        )
        record = next(
            candidate
            for candidate in self.workspace.generation
            .get_generated_object_records()
            if candidate.object_id == body.source_object_id
        )
        self.assertFalse(
            set(Path(record.asset_path).name).intersection('<>:"/\\|?*')
        )

    def test_mismatched_slot_topology_disables_preview_face_editing(
        self,
    ) -> None:
        door = self._create_door()
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        mismatched_geometry = ObjectFaceGeometry(
            vertices=np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                dtype=float,
            ),
            faces=np.asarray(((0, 1, 2),), dtype=np.int64),
            uv_triangles=np.empty((0, 3, 2), dtype=float),
            uv_face_indices=np.empty((0,), dtype=np.int64),
        )
        viewer = self.workspace.doors_workspace.preview_viewer
        viewer.set_face_editing_enabled(True)
        viewer.set_face_edit_geometry(
            mismatched_geometry.vertices,
            mismatched_geometry.faces,
        )

        with patch.object(
            self.workspace.generation,
            "get_object_face_geometry",
            return_value=mismatched_geometry,
        ) as get_geometry:
            self.workspace._sync_selected_opening_asset_preview()

        get_geometry.assert_called_once_with(body.source_object_id)
        self.assertFalse(viewer._face_editing_enabled)
        self.assertEqual(viewer.face_edit_face_count, 0)
        self.assertEqual(viewer.get_selected_face_indices(), ())
        self.assertEqual(
            self.workspace.doors_workspace.add_glass_area_button.text(),
            "Add glass area",
        )

    def test_successful_part_generation_returns_from_detached_generation(
        self,
    ) -> None:
        door = self._create_door()
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        record = next(
            record
            for record in self.workspace.generation.get_generated_object_records()
            if record.object_id == body.source_object_id
        )
        model = self.workspace.generation.get_generated_object_model(
            body.source_object_id
        )
        assert model is not None
        screen = _qt_application.primaryScreen()
        assert screen is not None

        try:
            with patch(
                "housemaker.main.resolve_fullscreen_3d_viewer_screen",
                return_value=screen,
            ):
                self.workspace._apply_generation_display_screen(
                    "screen:opening-completion-test"
                )
            self.workspace._handle_opening_asset_slot_edit_requested(
                ASSET_KIND_DOOR,
                door.door_id,
                body.slot_id,
            )
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.canvas_viewer_workspace
            )

            with (
                patch.object(BlueprintWorkspace, "show", autospec=True) as show,
                patch.object(BlueprintWorkspace, "raise_", autospec=True) as raise_,
                patch.object(
                    BlueprintWorkspace,
                    "activateWindow",
                    autospec=True,
                ) as activate_window,
            ):
                self.workspace.generation.opening_component_generation_completed.emit(
                    record,
                    model,
                )
                _qt_application.processEvents()

            self.assertTrue(self.workspace._external_generation_host.is_active)
            self.assertIs(
                self.workspace.workspace_tabs.currentWidget(),
                self.workspace.doors_workspace,
            )
            show.assert_called_once_with()
            raise_.assert_called_once_with()
            activate_window.assert_called_once_with()
        finally:
            self.workspace._apply_generation_display_screen(None)

    def test_double_clicking_a_slot_keeps_detached_generation_active(self) -> None:
        door = self._create_door()
        screen = _qt_application.primaryScreen()
        assert screen is not None
        self.assertTrue(
            self.workspace.doors_workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )
        selected_item = self.workspace.doors_workspace.door_tree.currentItem()
        assert selected_item is not None

        try:
            with patch(
                "housemaker.main.resolve_fullscreen_3d_viewer_screen",
                return_value=screen,
            ):
                self.workspace._apply_generation_display_screen(
                    "screen:door-slot-test"
                )
            self.assertTrue(self.workspace._external_generation_host.is_active)

            self.workspace.doors_workspace.door_tree.itemDoubleClicked.emit(
                selected_item,
                0,
            )
            _qt_application.processEvents()

            self.assertTrue(self.workspace._external_generation_host.is_active)
            self.assertIs(
                self.workspace._external_generation_host.viewer,
                self.workspace.merged_generation_workspace,
            )
            target = self.workspace.generation._door_slot_editing_target
            self.assertIsNotNone(target)
            assert target is not None
            self.assertEqual(target.door_id, door.door_id)
            self.assertEqual(target.slot_id, DOOR_SLOT_KNOB)
        finally:
            self.workspace._apply_generation_display_screen(None)

    def test_side_door_duplication_requires_an_open_body_generation_target(
        self,
    ) -> None:
        door = self._create_door()
        checkbox = self.workspace.generation.side_door_duplication_checkbox

        self.assertTrue(checkbox.isHidden())
        self.assertFalse(checkbox.isEnabled())
        selected_item = self.workspace.doors_workspace.door_tree.currentItem()
        assert selected_item is not None

        self.workspace.doors_workspace.door_tree.itemDoubleClicked.emit(
            selected_item,
            0,
        )
        _qt_application.processEvents()

        self.assertFalse(checkbox.isHidden())
        self.assertTrue(checkbox.isEnabled())
        self.assertEqual(checkbox.text(), "Side door duplication")
        checkbox.click()

        target = self.workspace.generation._door_slot_editing_target
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(
            target.side_duplication,
            DoorSideDuplication(DOOR_SIDE_DUPLICATION_FRONT),
        )
        persisted = self.workspace.doors_workspace.data().get_door(
            door.door_id
        )
        assert persisted is not None
        self.assertEqual(
            persisted.side_duplication,
            DoorSideDuplication(DOOR_SIDE_DUPLICATION_FRONT),
        )

    def test_side_duplication_stays_editable_during_generation_jobs(
        self,
    ) -> None:
        self._create_door()
        checkbox = self.workspace.generation.side_door_duplication_checkbox
        selected_item = self.workspace.doors_workspace.door_tree.currentItem()
        assert selected_item is not None
        self.workspace.doors_workspace.door_tree.itemDoubleClicked.emit(
            selected_item,
            0,
        )
        _qt_application.processEvents()

        with (
            patch.object(
                self.workspace.generation,
                "_object_has_active_mutation_job",
                return_value=True,
            ),
            patch.object(
                self.workspace.generation,
                "_generation_thread",
                object(),
            ),
        ):
            self.workspace.generation._sync_controls()

            self.assertFalse(checkbox.isHidden())
            self.assertTrue(checkbox.isEnabled())

    def test_generate_displacement_persists_and_only_routes_body_slots(
        self,
    ) -> None:
        door = self._create_door()
        checkbox = self.workspace.generation.generate_displacement_checkbox

        self.assertTrue(checkbox.isChecked())
        self.assertTrue(checkbox.isEnabled())
        checkbox.click()
        _qt_application.processEvents()

        persisted = self.workspace.doors_workspace.data().get_door(
            door.door_id
        )
        assert persisted is not None
        self.assertFalse(persisted.generate_displacement)
        body_target = self.workspace.generation._door_slot_editing_target
        self.assertIsNotNone(body_target)
        assert body_target is not None
        self.assertEqual(body_target.slot_id, DOOR_SLOT_BODY)
        self.assertFalse(body_target.generate_displacement)

        self.assertTrue(
            self.workspace.doors_workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )
        self.workspace._sync_selected_door_slot_preview(
            generation_ready=True,
        )

        knob_target = self.workspace.generation._door_slot_editing_target
        self.assertIsNotNone(knob_target)
        assert knob_target is not None
        self.assertEqual(knob_target.slot_id, DOOR_SLOT_KNOB)
        self.assertFalse(knob_target.generate_displacement)
        self.assertFalse(checkbox.isChecked())
        self.assertFalse(checkbox.isEnabled())

    def test_preserve_profile_persists_and_routes_only_body_outline(
        self,
    ) -> None:
        door = self._create_door()
        checkbox = self.workspace.generation.preserve_door_profile_checkbox

        self.assertFalse(checkbox.isChecked())
        self.assertTrue(checkbox.isEnabled())
        checkbox.click()
        _qt_application.processEvents()

        persisted = self.workspace.doors_workspace.data().get_door(
            door.door_id
        )
        assert persisted is not None
        self.assertTrue(persisted.preserve_profile)
        body_target = self.workspace.generation._door_slot_editing_target
        self.assertIsNotNone(body_target)
        assert body_target is not None
        self.assertTrue(body_target.preserve_profile)
        self.assertIsNotNone(body_target.door_body_profile_outline)

        self.assertTrue(
            self.workspace.doors_workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )
        self.workspace._sync_selected_door_slot_preview(
            generation_ready=True,
        )

        knob_target = self.workspace.generation._door_slot_editing_target
        self.assertIsNotNone(knob_target)
        assert knob_target is not None
        self.assertFalse(knob_target.preserve_profile)
        self.assertIsNone(knob_target.door_body_profile_outline)
        self.assertFalse(checkbox.isChecked())
        self.assertFalse(checkbox.isEnabled())

    def test_double_arch_body_targets_use_one_leaf_source_profile(self) -> None:
        self.level.doorways[0] = replace(
            self.level.doorways[0],
            shape=DOORWAY_SHAPE_ARCH,
            arch_amount=0.75,
        )
        self.workspace._set_canvas_viewer_targets((self.wall,))
        door = self._create_door()
        self.assertTrue(door.preserve_profile)
        self.assertTrue(
            self.workspace.generation.preserve_door_profile_checkbox.isChecked()
        )
        self.workspace._handle_opening_asset_make_double_changed(
            ASSET_KIND_DOOR,
            door.door_id,
            True,
        )
        updated = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert updated is not None
        expected_profile = build_double_door_leaf_source_outline(
            updated.width_meters,
            updated.height_meters,
            updated.shape,
            updated.arch_amount,
        )

        for slot_id in (DOOR_SLOT_BODY, DOOR_SLOT_LEFT_BODY):
            with self.subTest(slot_id=slot_id):
                self.assertTrue(
                    self.workspace.doors_workspace.select_door_slot(
                        updated.door_id,
                        slot_id,
                    )
                )
                self.workspace._sync_selected_door_slot_preview(
                    generation_ready=True,
                )
                target = self.workspace.generation._door_slot_editing_target
                self.assertIsNotNone(target)
                assert target is not None
                self.assertTrue(target.preserve_profile)
                self.assertEqual(
                    target.door_body_profile_outline,
                    expected_profile,
                )
                self.assertEqual(
                    target.door_body_fit_dimensions,
                    (
                        updated.width_meters,
                        updated.thickness_meters,
                        updated.height_meters,
                    ),
                )

    def test_window_body_uses_generation_profile_options(self) -> None:
        window, _reference = self._create_window()
        displacement = self.workspace.generation.generate_displacement_checkbox
        preserve_profile = (
            self.workspace.generation.preserve_door_profile_checkbox
        )

        self.assertTrue(window.generate_displacement)
        self.assertTrue(displacement.isChecked())
        self.assertTrue(displacement.isEnabled())
        self.assertFalse(preserve_profile.isChecked())
        self.assertTrue(preserve_profile.isEnabled())

        preserve_profile.click()
        _qt_application.processEvents()

        persisted = self.workspace.doors_workspace.data().get_window(
            window.window_id
        )
        assert persisted is not None
        self.assertTrue(persisted.preserve_profile)
        target = self.workspace.generation._door_slot_editing_target
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.asset_kind, ASSET_KIND_WINDOW)
        self.assertTrue(target.preserve_profile)
        self.assertEqual(
            target.door_body_profile_outline,
            (
                (-window.width_meters / 2.0, 0.0),
                (window.width_meters / 2.0, 0.0),
                (window.width_meters / 2.0, window.height_meters),
                (-window.width_meters / 2.0, window.height_meters),
                (-window.width_meters / 2.0, 0.0),
            ),
        )

    def test_make_double_sided_builds_front_and_back_assets_and_preview(
        self,
    ) -> None:
        door = self._create_door()

        self.workspace.doors_workspace.make_double_sided_checkbox.click()
        _qt_application.processEvents()

        door = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert door is not None
        self.assertTrue(door.make_double_sided)
        self.assertEqual(
            tuple(slot.slot_id for slot in door.slots),
            (
                DOOR_SLOT_BODY,
                DOOR_SLOT_KNOB,
                DOOR_SLOT_BACK_BODY,
                DOOR_SLOT_BACK_KNOB,
            ),
        )
        for body_slot_id in (DOOR_SLOT_BODY, DOOR_SLOT_BACK_BODY):
            body_slot = door.get_slot(body_slot_id)
            assert body_slot is not None
            self.assertIsNotNone(
                self.workspace.generation.get_generated_object_model(
                    body_slot.source_object_id
                )
            )

        for knob_slot_id in (DOOR_SLOT_KNOB, DOOR_SLOT_BACK_KNOB):
            self._register_generated_component(door, knob_slot_id)
        door = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert door is not None

        preview_model = self.workspace.doors_workspace.preview_viewer.model
        self.assertIsNotNone(preview_model)
        assert preview_model is not None
        expected_source_ids = {slot.source_object_id for slot in door.slots}
        self.assertEqual(
            {
                preview.source_object_id
                for preview in preview_model.preview_placed_objects
            },
            expected_source_ids,
        )

        target = next(
            target
            for target in self.workspace._canvas_opening_targets_by_key.values()
            if target.reference.kind == CANVAS_OPENING_DOORWAY
        )
        candidates = self.workspace._build_door_placement_preview_candidates(
            door,
            (target,),
        )
        self.assertEqual(len(candidates), 1)
        parts_by_kind = {
            part.component_kind: part
            for part in candidates[0].normal_parts
        }
        self.assertEqual(set(parts_by_kind), {DOOR_SLOT_BODY, DOOR_SLOT_KNOB})
        self.assertGreaterEqual(len(parts_by_kind[DOOR_SLOT_BODY].meshes), 2)
        self.assertGreaterEqual(len(parts_by_kind[DOOR_SLOT_KNOB].meshes), 2)

    def test_double_sided_slots_open_individually_and_hide_side_duplication(
        self,
    ) -> None:
        door = self._create_door()
        self.workspace.doors_workspace.make_double_sided_checkbox.click()
        _qt_application.processEvents()
        door = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert door is not None

        for slot_id in (
            DOOR_SLOT_BODY,
            DOOR_SLOT_KNOB,
            DOOR_SLOT_BACK_BODY,
            DOOR_SLOT_BACK_KNOB,
        ):
            self.assertTrue(
                self.workspace.doors_workspace.select_door_slot(
                    door.door_id,
                    slot_id,
                )
            )
            selected_item = self.workspace.doors_workspace.door_tree.currentItem()
            assert selected_item is not None
            self.workspace.doors_workspace.door_tree.itemDoubleClicked.emit(
                selected_item,
                0,
            )
            _qt_application.processEvents()

            target = self.workspace.generation._door_slot_editing_target
            self.assertIsNotNone(target)
            assert target is not None
            self.assertEqual(target.door_id, door.door_id)
            self.assertEqual(target.slot_id, slot_id)
            self.assertTrue(target.generation_ready)
            self.assertTrue(
                self.workspace.generation.side_door_duplication_checkbox.isHidden()
            )
            self.assertFalse(
                self.workspace.generation.side_door_duplication_checkbox.isEnabled()
            )

    def test_ctrl_z_history_restores_single_sided_door_hierarchy(self) -> None:
        door = self._create_door()
        self.workspace.doors_workspace.make_double_sided_checkbox.click()
        _qt_application.processEvents()

        self.workspace._handle_doors_undo_requested()

        restored = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert restored is not None
        self.assertFalse(restored.make_double_sided)
        self.assertEqual(
            tuple(slot.slot_id for slot in restored.slots),
            (DOOR_SLOT_BODY, DOOR_SLOT_KNOB),
        )
        self.assertFalse(
            self.workspace.doors_workspace.make_double_sided_checkbox.isChecked()
        )

    def test_side_duplicated_door_mirrors_knob_in_preview_and_placement(
        self,
    ) -> None:
        door = self._create_door()
        knob = self._register_generated_knob(door)
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        metadata = DoorSideDuplicationMetadata(
            kept_side=DOOR_SIDE_DUPLICATION_FRONT,
            plane_coordinate=0.0,
        )
        record = next(
            record
            for record in self.workspace.generation._data.generated_objects
            if record.object_id == body.source_object_id
        )
        record_index = self.workspace.generation._data.generated_objects.index(
            record
        )
        self.workspace.generation._data.generated_objects[record_index] = replace(
            record,
            pipeline={
                **record.pipeline,
                DOOR_SIDE_DUPLICATION_PIPELINE_KEY: metadata.to_pipeline_dict(),
            },
        )
        self.workspace._sync_selected_door_slot_preview()
        door = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert door is not None

        preview_model = self.workspace.doors_workspace.preview_viewer.model
        self.assertIsNotNone(preview_model)
        assert preview_model is not None
        mirrored_ids = {
            preview.object_id
            for preview in preview_model.preview_symmetric_objects
        }
        self.assertIn(f"{body.source_object_id}:side", mirrored_ids)
        self.assertIn(f"{knob.source_object_id}:side", mirrored_ids)

        target = next(
            target
            for target in self.workspace._canvas_opening_targets_by_key.values()
            if target.reference.kind == CANVAS_OPENING_DOORWAY
        )
        candidates = self.workspace._build_door_placement_preview_candidates(
            door,
            (target,),
        )
        self.assertEqual(len(candidates), 1)
        parts_by_kind = {
            part.component_kind: part
            for part in candidates[0].normal_parts
        }
        self.assertGreaterEqual(len(parts_by_kind[DOOR_SLOT_KNOB].meshes), 2)

        self.workspace._handle_door_placement_requested(
            door.door_id,
            target.reference,
            False,
        )
        placed_door = next(
            model
            for model in self.workspace._build_placed_generated_models(
                include_excluded_levels=True
            )
            if model.object_id.startswith("door-placement:")
        )
        self.assertIn(
            f"{knob.source_object_id}:side",
            {
                preview.object_id
                for preview in placed_door.model.preview_symmetric_objects
            },
        )

    def test_persisted_side_duplication_does_not_lock_next_generation(self) -> None:
        door = self._create_door()
        metadata = DoorSideDuplicationMetadata(
            kept_side=DOOR_SIDE_DUPLICATION_BACK,
            plane_coordinate=0.0,
        )
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        record = next(
            record
            for record in self.workspace.generation._data.generated_objects
            if record.object_id == body.source_object_id
        )
        record_index = (
            self.workspace.generation._data.generated_objects.index(record)
        )
        self.workspace.generation._data.generated_objects[record_index] = replace(
            record,
            pipeline={
                **record.pipeline,
                DOOR_SIDE_DUPLICATION_PIPELINE_KEY: (
                    metadata.to_pipeline_dict()
                ),
            },
        )

        self.workspace._sync_selected_door_slot_preview(
            generation_ready=True,
        )

        selected = self.workspace.doors_workspace.selected_door()
        assert selected is not None
        self.assertEqual(selected.door_id, door.door_id)
        self.assertIsNone(selected.side_duplication)
        checkbox = self.workspace.generation.side_door_duplication_checkbox
        self.assertFalse(checkbox.isHidden())
        self.assertFalse(checkbox.isChecked())
        self.assertTrue(checkbox.isEnabled())
        target = self.workspace.generation._door_slot_editing_target
        self.assertIsNotNone(target)
        assert target is not None
        self.assertIsNone(target.side_duplication)

        checkbox.click()

        updated = self.workspace.doors_workspace.selected_door()
        assert updated is not None
        self.assertEqual(
            updated.side_duplication,
            DoorSideDuplication(DOOR_SIDE_DUPLICATION_FRONT),
        )
        self.assertTrue(checkbox.isEnabled())

    def test_door_preview_combines_persisted_vertical_and_depth_mirrors(
        self,
    ) -> None:
        door = self._create_door()
        expected_model = self.workspace.generation.get_generated_object_model(
            door.slots[0].source_object_id
        )
        assert expected_model is not None
        symmetry = SimpleNamespace(
            orientation="vertical",
            plane_coordinate=0.125,
        )
        side_duplication = DoorSideDuplicationMetadata(
            kept_side=DOOR_SIDE_DUPLICATION_BACK,
            plane_coordinate=-0.025,
        )

        with (
            patch.object(
                self.workspace.generation,
                "resolve_symmetric_division_for_record",
                return_value=symmetry,
            ),
            patch.object(
                self.workspace.generation,
                "resolve_door_side_duplication_for_record",
                return_value=side_duplication,
            ),
            patch(
                "housemaker.main.assemble_door_model",
                return_value=expected_model,
            ) as assemble,
        ):
            actual_model = self.workspace._build_door_assembly_preview(door)

        self.assertIs(actual_model, expected_model)
        self.assertEqual(
            assemble.call_args.kwargs["body_mirrors"],
            {
                DOOR_SLOT_BODY: DoorBodyMirrorConfiguration(
                    symmetric_orientation="vertical",
                    symmetric_plane_coordinate=0.125,
                    side_duplication_plane_coordinate=-0.025,
                )
            },
        )

    def test_matching_doorway_placement_builds_one_mirrored_scene_model(self) -> None:
        door = self._create_door()
        reference = self._doorway_reference()

        self.workspace._handle_door_placement_requested(
            door.door_id,
            reference,
            True,
        )

        library = self.workspace.doors_workspace.data()
        self.assertEqual(len(library.placements), 1)
        self.assertTrue(library.placements[0].mirrored_horizontally)
        placed = self.workspace._build_placed_generated_models(
            include_excluded_levels=True
        )
        door_models = [
            model
            for model in placed
            if model.object_id.startswith("door-placement:")
        ]
        self.assertEqual(len(door_models), 1)
        self.assertEqual(door_models[0].object_name, "New door 1")

    def test_doorway_fit_moves_hardware_without_scaling_hardware_meshes(self) -> None:
        door = self._create_door()
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        scaled_knob = knob.with_transform(axis_scales=(1.2, 0.8, 1.4))
        door = door.replace_slot(scaled_knob)
        destination = replace(
            self.level.doorways[0],
            width_meters=1.6,
            height_meters=2.4,
        )

        fitted = self.workspace._fit_door_definition_to_doorway(
            door,
            destination,
        )

        fitted_body = fitted.get_slot(DOOR_SLOT_BODY)
        fitted_knob = fitted.get_slot(DOOR_SLOT_KNOB)
        assert fitted_body is not None and fitted_knob is not None
        self.assertEqual(fitted_body.axis_scales, (1.6, 1.0, 1.2))
        self.assertEqual(fitted_knob.axis_scales, scaled_knob.axis_scales)
        self.assertAlmostEqual(
            fitted_knob.position_meters[0],
            scaled_knob.position_meters[0] * 1.6,
        )
        self.assertAlmostEqual(
            fitted_knob.position_meters[2],
            scaled_knob.position_meters[2] * 1.2,
        )

    def test_arch_door_preview_and_commit_use_destination_profile(self) -> None:
        """An arch can be assembled into a differently sized, deeper arch."""

        source_doorway = replace(
            self.level.doorways[0],
            shape=DOORWAY_SHAPE_ARCH,
            arch_amount=0.35,
        )
        self.level.doorways[0] = source_doorway
        self.workspace._set_canvas_viewer_targets((self.wall,))
        door = self._create_door()
        self._register_generated_component(door, DOOR_SLOT_KNOB)
        door = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert door is not None

        destination = replace(
            source_doorway,
            width_meters=1.6,
            height_meters=2.4,
            arch_amount=0.8,
        )
        self.level.doorways[0] = destination
        self.workspace._set_canvas_viewer_targets((self.wall,))
        target = next(
            target
            for target in self.workspace._canvas_opening_targets_by_key.values()
            if target.reference.kind == CANVAS_OPENING_DOORWAY
        )

        fitted, assembled = self.workspace._assemble_door_for_doorway(
            door,
            destination,
            include_unjoined=False,
        )
        candidates = self.workspace._build_door_placement_preview_candidates(
            door,
            (target,),
        )

        self.assertEqual(len(candidates), 1)
        assembled_meshes = self._combine_door_preview_parts(
            self.workspace._build_door_placement_preview_parts(
                fitted,
                assembled,
            )
        )
        preview_meshes = self._combine_door_preview_parts(
            candidates[0].normal_parts
        )
        self.assertEqual(
            set(preview_meshes),
            {DOOR_SLOT_BODY, DOOR_SLOT_KNOB},
        )
        body_mesh = preview_meshes[DOOR_SLOT_BODY]
        np.testing.assert_allclose(
            body_mesh.extents[[0, 2]],
            (destination.width_meters, destination.height_meters),
            atol=1e-8,
        )
        body_vertices = np.asarray(body_mesh.vertices, dtype=float)
        side_vertices = body_vertices[
            np.isclose(
                np.abs(body_vertices[:, 0]),
                destination.width_meters / 2.0,
                atol=1e-8,
            )
        ]
        self.assertGreater(len(side_vertices), 0)
        expected_spring_height = _arch_profile_height_at_x(
            destination.width_meters,
            destination.height_meters,
            destination.arch_amount,
            destination.width_meters / 2.0,
        )
        self.assertAlmostEqual(
            float(side_vertices[:, 2].max()),
            expected_spring_height,
        )

        for component_kind in (DOOR_SLOT_BODY, DOOR_SLOT_KNOB):
            with self.subTest(component_kind=component_kind):
                np.testing.assert_allclose(
                    preview_meshes[component_kind].bounds,
                    assembled_meshes[component_kind].bounds,
                    atol=1e-9,
                )

        self.workspace._handle_door_placement_requested(
            door.door_id,
            target.reference,
            False,
        )
        self.assertEqual(
            len(self.workspace.doors_workspace.data().placements),
            1,
        )
        placed_door = next(
            model
            for model in self.workspace._build_placed_generated_models(
                include_excluded_levels=True
            )
            if model.object_id.startswith("door-placement:")
        )
        committed_meshes = self._combine_door_preview_parts(
            self.workspace._build_door_placement_preview_parts(
                fitted,
                placed_door.model,
            )
        )
        for component_kind in (DOOR_SLOT_BODY, DOOR_SLOT_KNOB):
            with self.subTest(committed_component=component_kind):
                candidate_mesh = preview_meshes[component_kind]
                committed_mesh = committed_meshes[component_kind]
                np.testing.assert_allclose(
                    committed_mesh.bounds,
                    candidate_mesh.bounds,
                    atol=1e-9,
                )
                self.assertEqual(
                    len(committed_mesh.faces),
                    len(candidate_mesh.faces),
                )
                np.testing.assert_allclose(
                    np.unique(
                        np.round(committed_mesh.vertices, decimals=9),
                        axis=0,
                    ),
                    np.unique(
                        np.round(candidate_mesh.vertices, decimals=9),
                        axis=0,
                    ),
                    atol=1e-9,
                )

    def test_arch_door_fit_preserves_handle_height_below_curve(self) -> None:
        """The handle keeps its local height ratio under the changed arch."""

        source_doorway = replace(
            self.level.doorways[0],
            shape=DOORWAY_SHAPE_ARCH,
            arch_amount=0.35,
        )
        self.level.doorways[0] = source_doorway
        self.workspace._set_canvas_viewer_targets((self.wall,))
        door = self._create_door()
        source_handle = door.get_slot(DOOR_SLOT_KNOB)
        assert source_handle is not None
        destination = replace(
            source_doorway,
            width_meters=1.6,
            height_meters=2.4,
            arch_amount=0.8,
        )

        fitted = self.workspace._fit_door_definition_to_doorway(
            door,
            destination,
        )

        fitted_handle = fitted.get_slot(DOOR_SLOT_KNOB)
        assert fitted_handle is not None
        source_profile_height = _arch_profile_height_at_x(
            door.width_meters,
            door.height_meters,
            door.arch_amount,
            source_handle.position_meters[0],
        )
        destination_profile_height = _arch_profile_height_at_x(
            destination.width_meters,
            destination.height_meters,
            destination.arch_amount,
            fitted_handle.position_meters[0],
        )
        self.assertAlmostEqual(
            fitted_handle.position_meters[2] / destination_profile_height,
            source_handle.position_meters[2] / source_profile_height,
        )
        self.assertNotAlmostEqual(
            fitted_handle.position_meters[2],
            source_handle.position_meters[2]
            * destination.height_meters
            / door.height_meters,
        )

    def test_rectangular_door_remains_incompatible_with_arch_doorway(
        self,
    ) -> None:
        """A profile topology change removes and cannot recreate a binding."""

        door = self._create_door()
        reference = self._doorway_reference()
        self.workspace._handle_door_placement_requested(
            door.door_id,
            reference,
            False,
        )
        self.assertEqual(
            len(self.workspace.doors_workspace.data().placements),
            1,
        )

        self.level.doorways[0] = replace(
            self.level.doorways[0],
            shape=DOORWAY_SHAPE_ARCH,
            arch_amount=0.65,
        )

        self.assertTrue(self.workspace._reconcile_door_placements())
        self.assertEqual(
            self.workspace.doors_workspace.data().placements,
            [],
        )
        self.workspace._handle_door_placement_requested(
            door.door_id,
            reference,
            False,
        )
        self.assertEqual(
            self.workspace.doors_workspace.data().placements,
            [],
        )

    def test_door_body_refits_when_destination_doorway_size_changes(self) -> None:
        door = self._create_door()
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        self.workspace.doors_workspace.replace_slot(
            door.door_id,
            body.with_transform(axis_scales=(1.25, 1.5, 0.8)),
        )
        door = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert door is not None
        reference = self._doorway_reference()
        self.level.doorways[0] = replace(
            self.level.doorways[0],
            width_meters=1.6,
            height_meters=2.4,
        )

        self.workspace._handle_door_placement_requested(
            door.door_id,
            reference,
            False,
        )

        self.assertEqual(len(self.workspace.doors_workspace.data().placements), 1)
        first_placed = next(
            model
            for model in self.workspace._build_placed_generated_models(
                include_excluded_levels=True
            )
            if model.object_id.startswith("door-placement:")
        )
        np.testing.assert_allclose(
            first_placed.model.mesh.extents[[0, 2]],
            (1.6, 2.4),
            atol=1e-8,
        )
        self.assertAlmostEqual(first_placed.model.mesh.extents[1], 0.06)

        self.level.doorways[0] = replace(
            self.level.doorways[0],
            width_meters=0.8,
            height_meters=1.5,
        )

        self.assertFalse(self.workspace._reconcile_door_placements())
        self.assertEqual(len(self.workspace.doors_workspace.data().placements), 1)
        resized_placed = next(
            model
            for model in self.workspace._build_placed_generated_models(
                include_excluded_levels=True
            )
            if model.object_id.startswith("door-placement:")
        )
        np.testing.assert_allclose(
            resized_placed.model.mesh.extents[[0, 2]],
            (0.8, 1.5),
            atol=1e-8,
        )

    def test_place_door_supplies_preview_matching_committed_transforms(
        self,
    ) -> None:
        door = self._create_door()
        self._register_generated_component(door, DOOR_SLOT_KNOB)
        door = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert door is not None
        self.level.doorways[0] = replace(
            self.level.doorways[0],
            width_meters=1.6,
            height_meters=2.4,
        )
        target = next(
            target
            for target in self.workspace._canvas_opening_targets_by_key.values()
            if target.reference.kind == CANVAS_OPENING_DOORWAY
        )

        with patch.object(
            self.workspace.viewer,
            "begin_door_placement",
            return_value=True,
        ) as begin_placement, patch.object(
            self.workspace,
            "_sync_selected_door_slot_preview",
        ) as refresh_editor_preview:
            self.workspace._handle_place_selected_door_clicked()

        begin_placement.assert_called_once()
        refresh_editor_preview.assert_not_called()
        self.assertEqual(begin_placement.call_args.args[0], door.door_id)
        self.assertEqual(begin_placement.call_args.args[1], (target.key,))
        preview_candidates = begin_placement.call_args.kwargs[
            "preview_candidates"
        ]
        self.assertEqual(len(preview_candidates), 1)
        preview_candidate = preview_candidates[0]
        self.assertEqual(preview_candidate.opening_key, target.key)
        self.assertEqual(
            tuple(part.component_kind for part in preview_candidate.normal_parts),
            (DOOR_SLOT_BODY, DOOR_SLOT_KNOB),
        )
        self.assertEqual(
            tuple(
                part.component_kind
                for part in preview_candidate.mirrored_parts
            ),
            (DOOR_SLOT_BODY, DOOR_SLOT_KNOB),
        )

        self.workspace._handle_door_placement_requested(
            door.door_id,
            target.reference,
            False,
        )
        normal_placement = next(
            model
            for model in self.workspace._build_placed_generated_models(
                include_excluded_levels=True
            )
            if model.object_id.startswith("door-placement:")
        )
        np.testing.assert_allclose(
            preview_candidate.normal_transform,
            build_placed_generated_model_transform(normal_placement),
            atol=1e-9,
        )

        self.workspace._handle_door_placement_requested(
            door.door_id,
            target.reference,
            True,
        )
        mirrored_placement = next(
            model
            for model in self.workspace._build_placed_generated_models(
                include_excluded_levels=True
            )
            if model.object_id.startswith("door-placement:")
        )
        np.testing.assert_allclose(
            preview_candidate.mirrored_transform,
            build_placed_generated_model_transform(mirrored_placement),
            atol=1e-9,
        )

        self.workspace._handle_door_placement_requested(
            door.door_id,
            target.reference,
            False,
            True,
        )
        depth_mirrored_placement = next(
            model
            for model in self.workspace._build_placed_generated_models(
                include_excluded_levels=True
            )
            if model.object_id.startswith("door-placement:")
        )
        np.testing.assert_allclose(
            preview_candidate.depth_mirrored_transform,
            build_placed_generated_model_transform(depth_mirrored_placement),
            atol=1e-9,
        )

        self.workspace._handle_door_placement_requested(
            door.door_id,
            target.reference,
            True,
            True,
        )
        fully_mirrored_placement = next(
            model
            for model in self.workspace._build_placed_generated_models(
                include_excluded_levels=True
            )
            if model.object_id.startswith("door-placement:")
        )
        np.testing.assert_allclose(
            preview_candidate.depth_and_horizontally_mirrored_transform,
            build_placed_generated_model_transform(fully_mirrored_placement),
            atol=1e-9,
        )

    def test_door_wheel_states_keep_body_on_canonical_opening_anchor(
        self,
    ) -> None:
        """An asymmetric handle cannot pull any door state off its doorway."""

        door = self._register_asymmetric_handle(
            self._create_door(),
            asset_kind=ASSET_KIND_DOOR,
        )
        target = next(
            target
            for target in self.workspace._canvas_opening_targets_by_key.values()
            if target.reference.kind == CANVAS_OPENING_DOORWAY
        )
        candidate = self.workspace._build_door_placement_preview_candidates(
            door,
            (target,),
        )[0]
        center_x, center_y = level_image_to_world_xy(
            self.level,
            self.level.doorways[0].center_x,
            self.level.doorways[0].center_y,
        )
        expected_center = np.asarray(
            (
                center_x,
                center_y,
                build_level_base_z_lookup(self.workspace.levels)[self.level.index]
                + self.level.doorways[0].bottom_height_meters,
            ),
            dtype=float,
        )
        variants = (
            (candidate.normal_parts, candidate.normal_transform, False, False),
            (candidate.mirrored_parts, candidate.mirrored_transform, True, False),
            (
                candidate.depth_and_horizontally_mirrored_parts,
                candidate.depth_and_horizontally_mirrored_transform,
                True,
                True,
            ),
            (
                candidate.depth_mirrored_parts,
                candidate.depth_mirrored_transform,
                False,
                True,
            ),
        )

        for state_index, (
            preview_parts,
            preview_transform,
            mirrored_horizontally,
            mirrored_depthwise,
        ) in enumerate(variants):
            with self.subTest(state_index=state_index):
                np.testing.assert_allclose(
                    preview_transform,
                    candidate.normal_transform,
                    atol=1e-9,
                )
                self._assert_preview_body_bottom_center(
                    preview_parts,
                    preview_transform,
                    expected_center,
                )
                self.workspace._handle_door_placement_requested(
                    door.door_id,
                    target.reference,
                    mirrored_horizontally,
                    mirrored_depthwise,
                )
                placed = next(
                    model
                    for model in self.workspace._build_placed_generated_models(
                        include_excluded_levels=True
                    )
                    if model.object_id.startswith("door-placement:")
                )
                committed_transform = build_placed_generated_model_transform(
                    placed
                )
                np.testing.assert_allclose(
                    committed_transform,
                    candidate.normal_transform,
                    atol=1e-9,
                )
                self._assert_preview_body_bottom_center(
                    self.workspace._build_door_placement_preview_parts(
                        door,
                        placed.model,
                    ),
                    committed_transform,
                    expected_center,
                )

    def test_window_wheel_states_keep_body_on_canonical_opening_anchor(
        self,
    ) -> None:
        """An asymmetric handle cannot pull any window state through its wall."""

        window, _reference = self._create_window()
        window = self._register_asymmetric_handle(
            window,
            asset_kind=ASSET_KIND_WINDOW,
        )
        target = next(
            target
            for target in self.workspace._canvas_opening_targets_by_key.values()
            if target.reference.kind == CANVAS_OPENING_WINDOW
        )
        candidate = self.workspace._build_window_placement_preview_candidates(
            window,
            (target,),
        )[0]
        midpoint = (target.bounds.start_ratio + target.bounds.end_ratio) * 0.5
        expected_center = target.local_to_world(
            midpoint,
            target.bounds.bottom_ratio,
        )
        variants = (
            (candidate.normal_parts, candidate.normal_transform, False, False),
            (candidate.mirrored_parts, candidate.mirrored_transform, True, False),
            (
                candidate.depth_and_horizontally_mirrored_parts,
                candidate.depth_and_horizontally_mirrored_transform,
                True,
                True,
            ),
            (
                candidate.depth_mirrored_parts,
                candidate.depth_mirrored_transform,
                False,
                True,
            ),
        )

        for state_index, (
            preview_parts,
            preview_transform,
            mirrored_horizontally,
            mirrored_depthwise,
        ) in enumerate(variants):
            with self.subTest(state_index=state_index):
                np.testing.assert_allclose(
                    preview_transform,
                    candidate.normal_transform,
                    atol=1e-9,
                )
                self._assert_preview_body_bottom_center(
                    preview_parts,
                    preview_transform,
                    expected_center,
                )
                self.workspace._handle_window_fixture_placement_requested(
                    window.window_id,
                    target.reference,
                    mirrored_horizontally,
                    mirrored_depthwise,
                )
                placed = next(
                    model
                    for model in self.workspace._build_placed_generated_models(
                        include_excluded_levels=True
                    )
                    if model.object_id.startswith("window-placement:")
                )
                committed_transform = build_placed_generated_model_transform(
                    placed
                )
                np.testing.assert_allclose(
                    committed_transform,
                    candidate.normal_transform,
                    atol=1e-9,
                )
                self._assert_preview_body_bottom_center(
                    self.workspace._build_door_placement_preview_parts(
                        window,
                        placed.model,
                    ),
                    committed_transform,
                    expected_center,
                )

    def test_window_place_previews_joined_handle_in_all_wheel_states(self) -> None:
        """Window placement keeps its generated Handle blue in every state."""

        window, reference = self._create_window()
        handle = window.get_slot(DOOR_SLOT_KNOB)
        assert handle is not None
        handle_model = build_door_body_model(
            0.08,
            0.12,
            0.04,
            shape="rectangular",
            arch_amount=0.0,
            geometry_name="generated_window_placement_handle",
        )
        self.workspace.generation.register_door_component_model(
            object_id=handle.source_object_id,
            object_name=f"{window.name} - Handle",
            door_id=window.window_id,
            slot_id=handle.slot_id,
            model=handle_model,
            asset_kind=ASSET_KIND_WINDOW,
        )

        reflected_scene_error = AssertionError(
            "placement preview built a reflected model scene"
        )
        with patch.object(
            self.workspace.viewer,
            "begin_window_fixture_placement",
            return_value=True,
        ) as begin_placement, patch.object(
            self.workspace,
            "_sync_selected_opening_asset_preview",
        ) as refresh_editor_preview, patch(
            "housemaker.main.mirror_door_model_horizontally",
            side_effect=reflected_scene_error,
        ), patch(
            "housemaker.main.mirror_door_model_depthwise",
            side_effect=reflected_scene_error,
        ):
            self.workspace._handle_place_selected_window_clicked()

        begin_placement.assert_called_once()
        refresh_editor_preview.assert_not_called()
        preview_candidates = begin_placement.call_args.kwargs[
            "preview_candidates"
        ]
        self.assertEqual(len(preview_candidates), 1)
        candidate = preview_candidates[0]
        for parts in (
            candidate.normal_parts,
            candidate.mirrored_parts,
            candidate.depth_mirrored_parts,
            candidate.depth_and_horizontally_mirrored_parts,
        ):
            self.assertEqual(
                {part.component_kind for part in parts},
                {DOOR_SLOT_BODY, DOOR_SLOT_KNOB},
            )

        updated_window = self.workspace.doors_workspace.data().get_window(
            window.window_id
        )
        assert updated_window is not None
        updated_handle = updated_window.get_slot(DOOR_SLOT_KNOB)
        assert updated_handle is not None
        self.assertTrue(updated_handle.joined)
        self.workspace._handle_window_fixture_placement_requested(
            window.window_id,
            reference,
            True,
            True,
        )
        placement = self.workspace.doors_workspace.data().window_placements[0]
        self.assertTrue(placement.mirrored_horizontally)
        self.assertTrue(placement.mirrored_depthwise)

    def test_glass_placement_preview_never_serializes_a_glb(self) -> None:
        """The Place button may cut preview glass but never exports its scene."""

        door = self._create_door()
        door = replace(
            door,
            glass_areas=(
                GlassAreaData(
                    glass_area_id="glass-preview",
                    source_object_id="glass-preview-source",
                    name="Glass area 1",
                    minimum_x_ratio=0.25,
                    maximum_x_ratio=0.75,
                    minimum_z_ratio=0.25,
                    maximum_z_ratio=0.75,
                ),
            ),
        )
        self.workspace.doors_workspace.upsert_door(door)
        target = next(
            target
            for target in self.workspace._canvas_opening_targets_by_key.values()
            if target.reference.kind == CANVAS_OPENING_DOORWAY
        )
        self.workspace._handle_door_placement_requested(
            door.door_id,
            target.reference,
            True,
            True,
        )

        serialization_error = AssertionError(
            "placement preview serialized a GLB"
        )
        with patch.object(
            trimesh.Scene,
            "export",
            side_effect=serialization_error,
        ), patch(
            "housemaker.glb._serialize_scene_glb_with_half_mesh_extras",
            side_effect=serialization_error,
        ), patch(
            "housemaker.opening_glass._serialize_scene_glb_with_half_mesh_extras",
            side_effect=serialization_error,
        ):
            candidates = self.workspace._build_door_placement_preview_candidates(
                door,
                (target,),
            )
            placed = tuple(
                model
                for model in self.workspace._build_placed_generated_models(
                    include_excluded_levels=True,
                    preview_only=True,
                )
                if model.object_id.startswith("door-placement:")
            )

        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].normal_parts)
        self.assertEqual(len(placed), 1)
        self.assertTrue(placed[0].model.preview_only)

    def test_new_hardware_auto_joins_and_appears_in_preview_and_placement(
        self,
    ) -> None:
        door = self._create_door()
        hardware_source_ids: set[str] = set()
        for slot_id in (DOOR_SLOT_KNOB,):
            slot, record, model = self._register_unjoined_generated_component(
                door,
                slot_id,
            )
            hardware_source_ids.add(slot.source_object_id)
            self.assertFalse(slot.joined)

            self.workspace._handle_door_component_model_changed(record, model)

            updated_door = self.workspace.doors_workspace.data().get_door(
                door.door_id
            )
            assert updated_door is not None
            updated_slot = updated_door.get_slot(slot_id)
            assert updated_slot is not None
            self.assertTrue(updated_slot.joined)
            self.assertFalse(
                self.workspace.doors_workspace.is_slot_position_pending(
                    door.door_id,
                    slot_id,
                )
            )

        door = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert door is not None
        target = next(
            target
            for target in self.workspace._canvas_opening_targets_by_key.values()
            if target.reference.kind == CANVAS_OPENING_DOORWAY
        )
        preview_candidates = (
            self.workspace._build_door_placement_preview_candidates(
                door,
                (target,),
            )
        )

        self.assertEqual(len(preview_candidates), 1)
        self.assertEqual(
            {
                part.component_kind
                for part in preview_candidates[0].normal_parts
            },
            {DOOR_SLOT_BODY, DOOR_SLOT_KNOB},
        )

        self.workspace._handle_door_placement_requested(
            door.door_id,
            target.reference,
            False,
        )
        placed_door = next(
            model
            for model in self.workspace._build_placed_generated_models(
                include_excluded_levels=True
            )
            if model.object_id.startswith("door-placement:")
        )
        placed_source_ids = {
            preview.source_object_id
            for preview in placed_door.model.preview_placed_objects
        }
        self.assertTrue(hardware_source_ids.issubset(placed_source_ids))

    def test_place_reconciles_available_unjoined_imported_hardware(self) -> None:
        """Available generated parts must not disappear from door placement."""

        door = self._create_door()
        hardware_source_ids: set[str] = set()
        for slot_id in (DOOR_SLOT_KNOB,):
            slot = door.get_slot(slot_id)
            assert slot is not None
            hardware_source_ids.add(slot.source_object_id)
            authored_model = build_door_body_model(
                0.08,
                0.12,
                0.04,
                shape="rectangular",
                arch_amount=0.0,
                geometry_name=f"generated_imported_{slot_id}_fixture",
            )
            imported_model = import_generated_glb(authored_model.glb_bytes)
            self.workspace.generation.register_door_component_model(
                object_id=slot.source_object_id,
                object_name=f"{door.name} - {slot.display_name}",
                door_id=door.door_id,
                slot_id=slot.slot_id,
                model=imported_model,
            )

        stale_door = self.workspace.doors_workspace.data().get_door(
            door.door_id
        )
        assert stale_door is not None
        self.assertFalse(stale_door.get_slot(DOOR_SLOT_KNOB).joined)

        with patch.object(
            self.workspace.viewer,
            "begin_door_placement",
            return_value=True,
        ) as begin_placement:
            self.workspace._handle_place_selected_door_clicked()

        begin_placement.assert_called_once()
        preview_candidates = begin_placement.call_args.kwargs[
            "preview_candidates"
        ]
        self.assertEqual(len(preview_candidates), 1)
        self.assertEqual(
            {
                part.component_kind
                for part in preview_candidates[0].normal_parts
            },
            {DOOR_SLOT_BODY, DOOR_SLOT_KNOB},
        )
        reconciled_door = self.workspace.doors_workspace.data().get_door(
            door.door_id
        )
        assert reconciled_door is not None
        self.assertTrue(reconciled_door.get_slot(DOOR_SLOT_KNOB).joined)

        target = next(
            target
            for target in self.workspace._canvas_opening_targets_by_key.values()
            if target.reference.kind == CANVAS_OPENING_DOORWAY
        )
        self.workspace._handle_door_placement_requested(
            door.door_id,
            target.reference,
            False,
        )
        placed_door = next(
            model
            for model in self.workspace._build_placed_generated_models(
                include_excluded_levels=True
            )
            if model.object_id.startswith("door-placement:")
        )
        placed_source_ids = {
            preview.source_object_id
            for preview in placed_door.model.preview_placed_objects
        }
        self.assertTrue(hardware_source_ids.issubset(placed_source_ids))

    def test_place_confirms_pending_scaled_hardware_and_keeps_it_visible(
        self,
    ) -> None:
        """Placing a door commits its current generated slot transforms."""

        door = self._create_door()
        hardware_source_ids: set[str] = set()
        for index, slot_id in enumerate(
            (DOOR_SLOT_KNOB,),
            start=1,
        ):
            self._register_generated_component(door, slot_id)
            current_door = self.workspace.doors_workspace.data().get_door(
                door.door_id
            )
            assert current_door is not None
            slot = current_door.get_slot(slot_id)
            assert slot is not None
            hardware_source_ids.add(slot.source_object_id)
            self.workspace._handle_door_slot_transform_changed(
                door.door_id,
                slot.slot_id,
                (0.1 * index, 0.01 * index, 0.4 * index),
                (0.0, 0.0, 5.0 * index),
            )
            self.workspace._handle_door_slot_axis_scales_changed(
                door.door_id,
                slot.slot_id,
                (1.0 + index * 0.1, 0.8, 1.0 + index * 0.05),
            )

            pending_door = self.workspace.doors_workspace.data().get_door(
                door.door_id
            )
            assert pending_door is not None
            pending_slot = pending_door.get_slot(slot_id)
            assert pending_slot is not None
            self.assertFalse(pending_slot.joined)
            self.assertTrue(
                self.workspace.doors_workspace.is_slot_position_pending(
                    door.door_id,
                    slot.slot_id,
                )
            )

        with patch.object(
            self.workspace.viewer,
            "begin_door_placement",
            return_value=True,
        ) as begin_placement:
            self.workspace._handle_place_selected_door_clicked()

        begin_placement.assert_called_once()
        preview_candidates = begin_placement.call_args.kwargs[
            "preview_candidates"
        ]
        self.assertEqual(len(preview_candidates), 1)
        expected_components = {
            DOOR_SLOT_BODY,
            DOOR_SLOT_KNOB,
        }
        self.assertEqual(
            {
                part.component_kind
                for part in preview_candidates[0].normal_parts
            },
            expected_components,
        )
        self.assertEqual(
            {
                part.component_kind
                for part in preview_candidates[0].mirrored_parts
            },
            expected_components,
        )

        confirmed_door = self.workspace.doors_workspace.data().get_door(
            door.door_id
        )
        assert confirmed_door is not None
        for slot_id in (DOOR_SLOT_KNOB,):
            confirmed_slot = confirmed_door.get_slot(slot_id)
            assert confirmed_slot is not None
            self.assertTrue(confirmed_slot.joined)
            self.assertFalse(
                self.workspace.doors_workspace.is_slot_position_pending(
                    door.door_id,
                    slot_id,
                )
            )

        target = next(
            target
            for target in self.workspace._canvas_opening_targets_by_key.values()
            if target.reference.kind == CANVAS_OPENING_DOORWAY
        )
        self.workspace._handle_door_placement_requested(
            door.door_id,
            target.reference,
            False,
        )
        placed_door = next(
            model
            for model in self.workspace._build_placed_generated_models(
                include_excluded_levels=True
            )
            if model.object_id.startswith("door-placement:")
        )
        placed_source_ids = {
            preview.source_object_id
            for preview in placed_door.model.preview_placed_objects
        }
        self.assertTrue(hardware_source_ids.issubset(placed_source_ids))

    def test_model_change_does_not_auto_confirm_a_pending_hardware_edit(
        self,
    ) -> None:
        door = self._create_door()
        slot = door.get_slot(DOOR_SLOT_KNOB)
        assert slot is not None
        edited_position = (0.25, 0.02, 0.85)
        self.workspace._handle_door_slot_transform_changed(
            door.door_id,
            slot.slot_id,
            edited_position,
            (0.0, 0.0, 10.0),
        )
        _slot, record, model = self._register_unjoined_generated_component(
            door,
            slot.slot_id,
        )

        self.workspace._handle_door_component_model_changed(record, model)

        updated_door = self.workspace.doors_workspace.data().get_door(
            door.door_id
        )
        assert updated_door is not None
        updated_slot = updated_door.get_slot(slot.slot_id)
        assert updated_slot is not None
        self.assertFalse(updated_slot.joined)
        self.assertEqual(updated_slot.position_meters, edited_position)
        self.assertTrue(
            self.workspace.doors_workspace.is_slot_position_pending(
                door.door_id,
                slot.slot_id,
            )
        )

    def test_project_restore_joins_preexisting_generated_hardware(self) -> None:
        door = self._create_door()
        slot, _record, _model = self._register_unjoined_generated_component(
            door,
            DOOR_SLOT_KNOB,
        )
        self.assertFalse(slot.joined)

        self.workspace._apply_project_state(
            levels=[copy.deepcopy(self.level)],
            current_level_index=0,
            generation=self.workspace.generation.get_data(),
            doors=self.workspace.doors_workspace.data(),
        )

        restored_door = self.workspace.doors_workspace.data().get_door(
            door.door_id
        )
        assert restored_door is not None
        restored_slot = restored_door.get_slot(DOOR_SLOT_KNOB)
        assert restored_slot is not None
        self.assertTrue(restored_slot.joined)

    def test_export_describes_each_side_duplicated_door_body(self) -> None:
        door = self._create_door()
        knob_slot = self._register_generated_knob(door)
        door = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert door is not None
        body_slot = door.get_slot(DOOR_SLOT_BODY)
        assert body_slot is not None
        self.workspace.doors_workspace.replace_slot(
            door.door_id,
            body_slot.with_transform(axis_scales=(1.0, 2.0, 1.0)),
        )
        door = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert door is not None
        self.workspace._handle_door_placement_requested(
            door.door_id,
            self._doorway_reference(),
            False,
        )
        source_placements, _instances = (
            self.workspace._build_export_placed_models()
        )
        placed_door = next(
            placement
            for placement in source_placements
            if placement.object_id.startswith("door-placement:")
        )
        side_duplication = DoorSideDuplicationMetadata(
            kept_side=DOOR_SIDE_DUPLICATION_FRONT,
            plane_coordinate=0.015,
        )

        with patch.object(
            self.workspace.generation,
            "resolve_door_side_duplication_for_record",
            return_value=side_duplication,
        ):
            reconstructions = (
                self.workspace._build_export_door_body_reconstructions(
                    source_placements
                )
            )

        self.assertEqual(len(reconstructions), 1)
        reconstruction = reconstructions[0]
        body_slot = door.get_slot(DOOR_SLOT_BODY)
        assert body_slot is not None
        self.assertEqual(
            reconstruction.placement_object_id,
            placed_door.object_id,
        )
        self.assertEqual(
            reconstruction.body_object_id,
            body_slot.source_object_id,
        )
        self.assertEqual(
            reconstruction.kept_side,
            DOOR_SIDE_DUPLICATION_FRONT,
        )
        self.assertEqual(
            reconstruction.mirrored_component_object_ids,
            (knob_slot.source_object_id,),
        )
        body_preview = next(
            preview
            for preview in placed_door.model.preview_placed_objects
            if preview.source_object_id == body_slot.source_object_id
        )
        expected_plane = build_gltf_mirror_plane_from_z_up_transform(
            build_placed_generated_model_transform(placed_door)
            @ body_preview.placement_transform,
            axis=1,
            plane_coordinate=0.015,
        )
        np.testing.assert_allclose(
            reconstruction.mirror_point,
            expected_plane["point"],
        )
        np.testing.assert_allclose(
            reconstruction.mirror_normal,
            expected_plane["normal"],
        )

    def test_hidden_door_texture_completion_does_not_mark_an_atlas_row(self) -> None:
        door = self._create_door()
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        record = next(
            candidate
            for candidate in self.workspace.generation.get_data().generated_objects
            if candidate.object_id == body.source_object_id
        )
        model = self.workspace.generation.get_generated_object_model(
            body.source_object_id
        )
        assert model is not None

        self.workspace._handle_generated_object_generated_for_atlases(
            record,
            model,
        )

        self.assertNotIn(
            body.source_object_id,
            self.workspace.texture_atlas_workspace._new_source_attention_ids,
        )

    def test_component_transform_stays_pending_until_confirmed(self) -> None:
        door = self._create_door()
        knob = self._register_generated_knob(door)
        self.assertTrue(
            self.workspace.doors_workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )
        doors_view = self.workspace.doors_workspace.preview_viewer
        self.assertEqual(
            doors_view.get_placed_object_gizmo_mode(),
            PLACED_OBJECT_GIZMO_TRANSFORM,
        )
        self.assertEqual(
            doors_view.get_selected_placed_object_ids(),
            (knob.source_object_id,),
        )
        self.assertEqual(
            self.workspace.generation.result_view.get_selected_placed_object_ids(),
            (),
        )
        self.assertTrue(doors_view._toggle_placed_object_gizmo_mode())
        self.assertEqual(
            doors_view.get_placed_object_gizmo_mode(),
            PLACED_OBJECT_GIZMO_SCALE,
        )
        self.assertEqual(doors_view._placed_object_instance_gizmo_items, [])
        self.assertFalse(doors_view.view._object_scale_wheel_steps_enabled)
        self.assertTrue(doors_view._toggle_placed_object_gizmo_mode())

        self.workspace._handle_door_slot_transform_changed(
            door.door_id,
            DOOR_SLOT_KNOB,
            (0.2, 0.01, 0.9),
            (0.0, 0.0, 15.0),
        )

        edited = self.workspace.doors_workspace.selected_slot()
        assert edited is not None
        self.assertEqual(edited.position_meters, (0.2, 0.01, 0.9))
        self.assertFalse(edited.joined)
        self.assertTrue(
            self.workspace.doors_workspace.is_slot_position_pending(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )

        self.workspace._handle_door_slot_confirmation_requested(
            door.door_id,
            DOOR_SLOT_KNOB,
        )

        confirmed = self.workspace.doors_workspace.selected_slot()
        assert confirmed is not None
        self.assertTrue(confirmed.joined)
        self.assertFalse(
            self.workspace.doors_workspace.is_slot_position_pending(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )

    def test_proportional_editing_mirrors_double_door_slot_edits(self) -> None:
        """A double door's paired part follows transforms as one undoable edit."""

        door = self._create_door()
        self.workspace._handle_door_make_double_changed(door.door_id, True)
        self.assertTrue(
            self.workspace.doors_workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )
        self.assertTrue(
            self.workspace.doors_workspace.is_proportional_editing_enabled()
        )

        self.workspace._handle_door_slot_transform_changed(
            door.door_id,
            DOOR_SLOT_KNOB,
            (0.18, -0.03, 0.91),
            (12.0, 24.0, -36.0),
        )
        transformed = self.workspace.doors_workspace.data().get_door(
            door.door_id
        )
        assert transformed is not None
        right = transformed.get_slot(DOOR_SLOT_KNOB)
        left = transformed.get_slot(DOOR_SLOT_LEFT_KNOB)
        assert right is not None and left is not None
        self.assertEqual(right.position_meters, (0.18, -0.03, 0.91))
        self.assertEqual(left.position_meters, (-0.18, -0.03, 0.91))
        self.assertEqual(right.rotation_degrees, (12.0, 24.0, -36.0))
        self.assertEqual(left.rotation_degrees, (12.0, -24.0, 36.0))
        self.assertTrue(
            self.workspace.doors_workspace.is_slot_position_pending(
                door.door_id,
                DOOR_SLOT_LEFT_KNOB,
            )
        )

        self.workspace._handle_door_slot_axis_scales_changed(
            door.door_id,
            DOOR_SLOT_KNOB,
            (1.2, 0.8, 1.4),
        )
        scaled = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert scaled is not None
        self.assertEqual(
            scaled.get_slot(DOOR_SLOT_KNOB).axis_scales,
            (1.2, 0.8, 1.4),
        )
        self.assertEqual(
            scaled.get_slot(DOOR_SLOT_LEFT_KNOB).axis_scales,
            (1.2, 0.8, 1.4),
        )

        self.workspace._handle_door_slot_axis_scales_changed(
            door.door_id,
            DOOR_SLOT_BODY,
            (0.75, 1.1, 0.9),
        )
        body_scaled = self.workspace.doors_workspace.data().get_door(
            door.door_id
        )
        assert body_scaled is not None
        self.assertEqual(
            body_scaled.get_slot(DOOR_SLOT_BODY).axis_scales,
            (0.75, 1.1, 0.9),
        )
        self.assertEqual(
            body_scaled.get_slot(DOOR_SLOT_LEFT_BODY).axis_scales,
            (0.75, 1.1, 0.9),
        )

        self.workspace._handle_door_slot_confirmation_requested(
            door.door_id,
            DOOR_SLOT_KNOB,
        )
        confirmed = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert confirmed is not None
        self.assertTrue(confirmed.get_slot(DOOR_SLOT_KNOB).joined)
        self.assertTrue(confirmed.get_slot(DOOR_SLOT_LEFT_KNOB).joined)
        self.assertFalse(
            self.workspace.doors_workspace.is_slot_position_pending(
                door.door_id,
                DOOR_SLOT_LEFT_KNOB,
            )
        )

    def test_proportional_editing_can_be_disabled_for_double_door(self) -> None:
        door = self._create_door()
        self.workspace._handle_door_make_double_changed(door.door_id, True)
        self.workspace.doors_workspace.select_door_slot(
            door.door_id,
            DOOR_SLOT_KNOB,
        )
        self.workspace.doors_workspace.proportional_editing_checkbox.setChecked(
            False
        )
        before = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert before is not None
        left_before = before.get_slot(DOOR_SLOT_LEFT_KNOB)

        self.workspace._handle_door_slot_axis_scales_changed(
            door.door_id,
            DOOR_SLOT_KNOB,
            (1.3, 0.7, 1.1),
        )

        after = self.workspace.doors_workspace.data().get_door(door.door_id)
        assert after is not None
        self.assertEqual(
            after.get_slot(DOOR_SLOT_LEFT_KNOB),
            left_before,
        )

    def test_proportional_editing_applies_to_double_windows(self) -> None:
        window, _reference = self._create_window()
        self.workspace._handle_window_make_double_changed(
            window.window_id,
            True,
        )
        self.assertTrue(
            self.workspace.doors_workspace.select_window_slot(
                window.window_id,
                DOOR_SLOT_KNOB,
            )
        )

        self.workspace._handle_opening_asset_slot_transform_changed(
            ASSET_KIND_WINDOW,
            window.window_id,
            DOOR_SLOT_KNOB,
            (0.22, 0.01, 0.44),
            (-5.0, 15.0, 25.0),
        )

        updated = self.workspace.doors_workspace.data().get_window(
            window.window_id
        )
        assert updated is not None
        right = updated.get_slot(DOOR_SLOT_KNOB)
        left = updated.get_slot(DOOR_SLOT_LEFT_KNOB)
        assert right is not None and left is not None
        self.assertEqual(left.position_meters, (-0.22, 0.01, 0.44))
        self.assertEqual(left.rotation_degrees, (-5.0, -15.0, -25.0))

    def test_hardware_axis_scale_stays_pending_until_confirmed(self) -> None:
        door = self._create_door()
        self._register_generated_knob(door)
        self.assertTrue(
            self.workspace.doors_workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )

        self.workspace.doors_workspace.asset_slot_axis_scales_changed.emit(
            ASSET_KIND_DOOR,
            door.door_id,
            DOOR_SLOT_KNOB,
            (1.2, 0.8, 1.4),
        )

        edited = self.workspace.doors_workspace.selected_slot()
        assert edited is not None
        self.assertEqual(edited.axis_scales, (1.2, 0.8, 1.4))
        self.assertFalse(edited.joined)
        self.assertTrue(
            self.workspace.doors_workspace.is_slot_position_pending(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )

        self.workspace._handle_door_slot_confirmation_requested(
            door.door_id,
            DOOR_SLOT_KNOB,
        )

        confirmed = self.workspace.doors_workspace.selected_slot()
        assert confirmed is not None
        self.assertEqual(confirmed.axis_scales, (1.2, 0.8, 1.4))
        self.assertTrue(confirmed.joined)

    def test_body_axis_scale_applies_immediately_to_placed_doors(self) -> None:
        door = self._create_door()
        self.workspace._handle_door_placement_requested(
            door.door_id,
            self._doorway_reference(),
            False,
        )

        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ) as schedule_refresh:
            self.workspace._handle_door_slot_axis_scales_changed(
                door.door_id,
                DOOR_SLOT_BODY,
                (1.0, 1.5, 1.0),
            )

        body = self.workspace.doors_workspace.selected_slot()
        assert body is not None
        self.assertEqual(body.axis_scales, (1.0, 1.5, 1.0))
        self.assertTrue(body.joined)
        self.assertFalse(
            self.workspace.doors_workspace.is_slot_position_pending(
                door.door_id,
                DOOR_SLOT_BODY,
            )
        )
        schedule_refresh.assert_called_once_with(preserve_camera=True)

    def test_doors_undo_walks_transform_scale_and_confirmation_history(
        self,
    ) -> None:
        door = self._create_door()
        self._register_generated_knob(door)
        self.assertTrue(
            self.workspace.doors_workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )
        original_slot = self.workspace.doors_workspace.selected_slot()
        assert original_slot is not None

        self.workspace._handle_door_slot_transform_changed(
            door.door_id,
            DOOR_SLOT_KNOB,
            (0.2, 0.01, 0.9),
            (0.0, 0.0, 15.0),
        )
        self.workspace._handle_door_slot_axis_scales_changed(
            door.door_id,
            DOOR_SLOT_KNOB,
            (1.2, 0.8, 1.4),
        )
        self.workspace._handle_door_slot_confirmation_requested(
            door.door_id,
            DOOR_SLOT_KNOB,
        )

        self.assertEqual(len(self.workspace._door_undo_stack), 3)
        confirmed = self.workspace.doors_workspace.selected_slot()
        assert confirmed is not None
        self.assertTrue(confirmed.joined)
        self.assertFalse(
            self.workspace.doors_workspace.is_slot_position_pending(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )

        self.workspace.doors_workspace.undo_requested.emit()

        unconfirmed_scale = self.workspace.doors_workspace.selected_slot()
        assert unconfirmed_scale is not None
        self.assertEqual(unconfirmed_scale.axis_scales, (1.2, 0.8, 1.4))
        self.assertFalse(unconfirmed_scale.joined)
        self.assertTrue(
            self.workspace.doors_workspace.is_slot_position_pending(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )

        self.workspace.doors_workspace.undo_requested.emit()

        transformed = self.workspace.doors_workspace.selected_slot()
        assert transformed is not None
        self.assertEqual(transformed.axis_scales, original_slot.axis_scales)
        self.assertEqual(transformed.position_meters, (0.2, 0.01, 0.9))
        self.assertEqual(transformed.rotation_degrees, (0.0, 0.0, 15.0))
        self.assertFalse(transformed.joined)
        self.assertTrue(
            self.workspace.doors_workspace.is_slot_position_pending(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )

        self.workspace.doors_workspace.undo_requested.emit()

        restored = self.workspace.doors_workspace.selected_slot()
        self.assertEqual(restored, original_slot)
        self.assertFalse(
            self.workspace.doors_workspace.is_slot_position_pending(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )
        self.assertEqual(self.workspace._door_undo_stack, [])

    def test_no_op_door_slot_changes_do_not_add_undo_history(self) -> None:
        door = self._create_door()
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        initial_history_size = len(self.workspace._door_undo_stack)

        self.workspace._handle_door_slot_axis_scales_changed(
            door.door_id,
            DOOR_SLOT_BODY,
            body.axis_scales,
        )

        self.assertEqual(
            len(self.workspace._door_undo_stack),
            initial_history_size,
        )

    def test_loading_a_project_clears_doors_undo_history(self) -> None:
        door = self._create_door()
        self.workspace._handle_door_slot_axis_scales_changed(
            door.door_id,
            DOOR_SLOT_BODY,
            (1.0, 1.5, 1.0),
        )
        self.assertEqual(len(self.workspace._door_undo_stack), 1)
        project = ProjectData(
            blueprint_path=None,
            current_level_index=0,
            levels=[copy.deepcopy(self.level)],
            doors=self.workspace.doors_workspace.data(),
        )

        self.workspace._apply_loaded_project(project)

        self.assertEqual(self.workspace._door_undo_stack, [])

    def test_moving_a_slot_refreshes_the_complete_doors_preview(self) -> None:
        door = self._create_door()
        self._register_generated_knob(door)
        self.assertTrue(
            self.workspace.doors_workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )
        initial_model = self.workspace.doors_workspace.preview_viewer.model
        assert initial_model is not None
        initial_maximum_x = float(initial_model.mesh.bounds[1][0])

        self.workspace._handle_door_slot_transform_changed(
            door.door_id,
            DOOR_SLOT_KNOB,
            (2.0, 0.01, 0.9),
            (0.0, 0.0, 15.0),
        )

        updated_model = self.workspace.doors_workspace.preview_viewer.model
        self.assertIsNotNone(updated_model)
        assert updated_model is not None
        self.assertIsNot(updated_model, initial_model)
        self.assertGreater(
            float(updated_model.mesh.bounds[1][0]),
            initial_maximum_x + 1.0,
        )

    def test_moved_hardware_does_not_shift_door_body_from_doorway(self) -> None:
        door = self._create_door()
        self._register_generated_knob(door)
        self.assertTrue(
            self.workspace.doors_workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )
        self.workspace._handle_door_slot_transform_changed(
            door.door_id,
            DOOR_SLOT_KNOB,
            (2.0, 0.35, -0.25),
            (20.0, 30.0, 40.0),
        )
        self.workspace._handle_door_slot_confirmation_requested(
            door.door_id,
            DOOR_SLOT_KNOB,
        )
        self.workspace._handle_door_placement_requested(
            door.door_id,
            self._doorway_reference(),
            False,
        )

        placed = next(
            model
            for model in self.workspace._build_placed_generated_models(
                include_excluded_levels=True
            )
            if model.object_id.startswith("door-placement:")
        )
        minimum, maximum = placed.model.mesh.bounds
        bounds_bottom_center = np.asarray(
            (
                (minimum[0] + maximum[0]) / 2.0,
                (minimum[1] + maximum[1]) / 2.0,
                minimum[2],
            ),
            dtype=float,
        )
        self.assertGreater(np.linalg.norm(bounds_bottom_center), 0.1)

        doorway = self.level.doorways[0]
        center_x, center_y = level_image_to_world_xy(
            self.level,
            doorway.center_x,
            doorway.center_y,
        )
        expected_anchor_z_up = np.asarray(
            (
                center_x,
                center_y,
                build_level_base_z_lookup([self.level])[self.level.index]
                + doorway.bottom_height_meters,
                1.0,
            ),
            dtype=float,
        )
        expected_anchor_gltf = (
            Z_UP_TO_GLTF_Y_UP_TRANSFORM @ expected_anchor_z_up
        )[:3]
        placement_transform = build_placed_generated_model_gltf_transform(placed)
        actual_anchor_gltf = (
            placement_transform @ np.asarray((0.0, 0.0, 0.0, 1.0))
        )[:3]
        np.testing.assert_allclose(
            actual_anchor_gltf,
            expected_anchor_gltf,
            atol=1e-8,
        )

    def test_project_state_restore_reconnects_door_slots_and_placements(self) -> None:
        door = self._create_door()
        self.workspace._handle_door_placement_requested(
            door.door_id,
            self._doorway_reference(),
            False,
        )
        generation = self.workspace.generation.get_data()
        doors = self.workspace.doors_workspace.data()

        self.workspace._apply_project_state(
            levels=[copy.deepcopy(self.level)],
            current_level_index=0,
            generation=generation,
            doors=doors,
        )

        restored = self.workspace.doors_workspace.data()
        self.assertEqual(len(restored.doors), 1)
        self.assertEqual(len(restored.placements), 1)
        self.assertEqual(
            restored.placements[0].doorway_id,
            self.level.doorways[0].doorway_id,
        )
        self.assertEqual(
            len(self.workspace.generation.get_data().generated_objects),
            1,
        )


if __name__ == "__main__":
    unittest.main()
