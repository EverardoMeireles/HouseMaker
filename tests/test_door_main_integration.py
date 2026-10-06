# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.canvas_openings import CANVAS_OPENING_DOORWAY
from housemaker.door_geometry import DoorBodyMirrorConfiguration
from housemaker.door_state import (
    DOOR_SIDE_DUPLICATION_BACK,
    DOOR_SIDE_DUPLICATION_FRONT,
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    DoorSideDuplication,
    DoorSideDuplicationMetadata,
)
from housemaker.glb import (
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
    build_placed_generated_model_gltf_mirror_plane,
    build_placed_generated_model_gltf_transform,
)
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
)
from housemaker.main import BlueprintWorkspace
from housemaker.models import DoorwayData, LevelData
from housemaker.surface_geometry import SURFACE_TYPE_WALL, FixedSurface
from housemaker.viewer import PLACED_OBJECT_GIZMO_TRANSFORM

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

    def test_plus_creation_builds_hidden_slot_records_and_selects_body(self) -> None:
        door = self._create_door()

        self.assertEqual(door.name, "New door 1")
        self.assertEqual(self.workspace.canvas_door_list.count(), 1)
        self.assertEqual(
            self.workspace.doors_workspace.selected_slot_id(),
            DOOR_SLOT_BODY,
        )
        self.assertEqual(len(self.workspace.generation.get_data().generated_objects), 3)
        self.assertEqual(self.workspace.generation.get_generated_object_ids(), ())
        self.assertFalse(
            self.workspace.generation.generate_geometry_button.isEnabled()
        )
        self.assertFalse(self.workspace.generation.place_button.isEnabled())
        self.assertFalse(
            self.workspace.generation.object_3d_panel.projection_camera_controls.isEnabled()
        )
        self.assertFalse(
            self.workspace.merged_generation_workspace.infer_ceiling_height_button.isEnabled()
        )

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
        self.assertGreater(
            len(preview_model.mesh.faces),
            len(body_model.mesh.faces),
        )
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

    def test_side_duplication_controls_refresh_the_body_generation_target(
        self,
    ) -> None:
        self._create_door()

        self.workspace.doors_workspace.side_duplication_checkbox.click()

        target = self.workspace.generation._door_slot_editing_target
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(
            target.side_duplication,
            DoorSideDuplication(DOOR_SIDE_DUPLICATION_FRONT),
        )

        self.workspace.doors_workspace.side_duplication_back_radio.click()

        target = self.workspace.generation._door_slot_editing_target
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(
            target.side_duplication,
            DoorSideDuplication(DOOR_SIDE_DUPLICATION_BACK),
        )

    def test_persisted_side_duplication_is_authoritative_and_locked(self) -> None:
        door = self._create_door()
        metadata = DoorSideDuplicationMetadata(
            kept_side=DOOR_SIDE_DUPLICATION_BACK,
            plane_coordinate=0.0,
        )

        with patch.object(
            self.workspace.generation,
            "resolve_door_side_duplication_for_record",
            return_value=metadata,
        ):
            self.workspace._sync_selected_door_slot_preview()

        synchronized = self.workspace.doors_workspace.selected_door()
        assert synchronized is not None
        self.assertEqual(synchronized.door_id, door.door_id)
        self.assertEqual(
            synchronized.side_duplication,
            DoorSideDuplication(DOOR_SIDE_DUPLICATION_BACK),
        )
        self.assertTrue(
            self.workspace.doors_workspace.side_duplication_back_radio.isChecked()
        )
        self.assertFalse(
            self.workspace.doors_workspace.side_duplication_checkbox.isEnabled()
        )
        target = self.workspace.generation._door_slot_editing_target
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(
            target.side_duplication,
            DoorSideDuplication(DOOR_SIDE_DUPLICATION_BACK),
        )

        self.workspace.doors_workspace.side_duplication_front_radio.click()

        self.assertEqual(
            self.workspace.doors_workspace.selected_door(),
            synchronized,
        )

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
            assemble.call_args.kwargs["body_mirror"],
            DoorBodyMirrorConfiguration(
                symmetric_orientation="vertical",
                symmetric_plane_coordinate=0.125,
                side_duplication_plane_coordinate=-0.025,
            ),
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

    def test_export_describes_each_side_duplicated_door_body(self) -> None:
        door = self._create_door()
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
        expected_plane = build_placed_generated_model_gltf_mirror_plane(
            placed_door,
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
        self.assertTrue(
            self.workspace.doors_workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )
        result_view = self.workspace.generation.result_view
        self.assertEqual(
            result_view.get_placed_object_gizmo_mode(),
            PLACED_OBJECT_GIZMO_TRANSFORM,
        )
        self.assertFalse(result_view._toggle_placed_object_gizmo_mode())
        self.assertEqual(result_view._placed_object_instance_gizmo_items, [])
        self.assertFalse(result_view.view._object_scale_wheel_steps_enabled)

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

    def test_moving_a_slot_refreshes_the_complete_doors_preview(self) -> None:
        door = self._create_door()
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
            3,
        )


if __name__ == "__main__":
    unittest.main()
