# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.canvas_openings import CANVAS_OPENING_DOORWAY
from housemaker.door_geometry import (
    DoorBodyMirrorConfiguration,
    build_door_body_model,
)
from housemaker.door_state import (
    DOOR_SIDE_DUPLICATION_BACK,
    DOOR_SIDE_DUPLICATION_FRONT,
    DOOR_SLOT_BACK_BODY,
    DOOR_SLOT_BACK_KNOB,
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    DoorSideDuplication,
    DoorSideDuplicationMetadata,
)
from housemaker.generation_workspace import (
    DOOR_SIDE_DUPLICATION_PIPELINE_KEY,
)
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
from housemaker.models import DoorwayData, LevelData
from housemaker.project_io import ProjectData
from housemaker.surface_geometry import SURFACE_TYPE_WALL, FixedSurface
from housemaker.viewer import (
    PLACED_OBJECT_GIZMO_SCALE,
    PLACED_OBJECT_GIZMO_TRANSFORM,
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
            "- Generating the door's Door knob/handle",
        )
        self.assertFalse(indicator.isHidden())
        self.assertTrue(
            self.workspace.merged_generation_workspace
            ._door_slot_generation_blink_timer.isActive()
        )

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
        checkbox = self.workspace.doors_workspace.generate_displacement_checkbox

        self.assertFalse(checkbox.isChecked())
        checkbox.click()
        _qt_application.processEvents()

        persisted = self.workspace.doors_workspace.data().get_door(
            door.door_id
        )
        assert persisted is not None
        self.assertTrue(persisted.generate_displacement)
        body_target = self.workspace.generation._door_slot_editing_target
        self.assertIsNotNone(body_target)
        assert body_target is not None
        self.assertEqual(body_target.slot_id, DOOR_SLOT_BODY)
        self.assertTrue(body_target.generate_displacement)

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
        self.assertTrue(checkbox.isChecked())

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
        ) as begin_placement:
            self.workspace._handle_place_selected_door_clicked()

        begin_placement.assert_called_once()
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

    def test_hardware_axis_scale_stays_pending_until_confirmed(self) -> None:
        door = self._create_door()
        self._register_generated_knob(door)
        self.assertTrue(
            self.workspace.doors_workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )

        self.workspace.doors_workspace.slot_axis_scales_changed.emit(
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
