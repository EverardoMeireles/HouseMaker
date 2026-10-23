"""Keep ordinary per-object references without the removed multi-view workflow."""

# ### Environment and imports ###
from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QCheckBox, QPushButton

from housemaker.generation_state import GenerationData, MaskPoint, MaskStroke
from housemaker.generation_workspace import (
    _ArchitecturalTrimEditingTarget,
    _DoorSlotEditingTarget,
)
from housemaker.geometry_generation_models import GEOMETRY_MODEL_7_1
from housemaker.merged_generation_workspace import _ReferenceGenerationThread
from tests import test_reference_generation_workspace as fixtures


# ### Local-only controls and reference fixtures ###
class OrdinaryReferenceControlsTests(unittest.TestCase):
    setUp = fixtures.ReferenceGenerationWorkspaceTests.setUp
    tearDown = fixtures.ReferenceGenerationWorkspaceTests.tearDown
    _apply_reference_generation_model = fixtures.ReferenceGenerationWorkspaceTests._apply_reference_generation_model
    _load_video = fixtures.ReferenceGenerationWorkspaceTests._load_video
    _start = fixtures.ReferenceGenerationWorkspaceTests._start

    def _paint_two_objects(self) -> None:
        self._load_video()
        strokes = [
            MaskStroke("paint", 0.1, (MaskPoint(0.25, 0.5),)),
            MaskStroke("paint", 0.1, (MaskPoint(0.75, 0.5),)),
        ]
        self.workspace.video_view.set_strokes(strokes)
        self.workspace.video_view.strokes_changed.emit(strokes)
        self.workspace.sync_shared_controls()
        self.assertEqual(len(self.objects.inpaint_queue()), 2)

    def _select_geometry_model(self, model: str) -> None:
        combo = self.workspace.geometry_provider_combo
        combo.setCurrentIndex(combo.findData(model))

    @staticmethod
    def _opening_target(asset_kind: str, *, generation_ready: bool = False):
        return _DoorSlotEditingTarget(
            door_id="opening-1", slot_id="body", object_id="opening-body",
            object_name="Opening body", slot_display_name="body",
            asset_kind=asset_kind, generation_ready=generation_ready,
        )

    @staticmethod
    def _trim_target(*, generation_ready: bool = False):
        return _ArchitecturalTrimEditingTarget(
            level_index=0, trim_id="cornice-1", object_id="cornice-object",
            object_name="Cornice", fit_dimensions=(1.0, 0.1, 0.1),
            generation_ready=generation_ready,
        )

    # ### Removed controls and independently selectable geometry providers ###
    def test_multi_view_workflow_controls_are_absent(self) -> None:
        for name in (
            "multi_view_generation_checkbox", "multi_inpaint_reference_checkbox",
            "multi_view_reference_checkbox", "multi_view_slots_control",
            "multi_view_slot_buttons",
        ):
            self.assertFalse(hasattr(self.workspace, name), name)
        self.assertFalse(any(
            checkbox.text() in {
                "Multi-view generation", "Multi-inpaint reference", "Multi-view reference",
            }
            for checkbox in self.workspace.findChildren(QCheckBox)
        ))
        self.assertFalse(any(
            button.objectName().startswith("multi_view_")
            for button in self.workspace.findChildren(QPushButton)
        ))

    def test_meshy_7_1_keeps_standard_and_ultra_2k_without_provider_lock(self) -> None:
        self.assertEqual(
            [self.workspace.geometry_provider_combo.itemText(index) for index in range(3)],
            ["Meshy t2", "Meshy t2.5", "Meshy 7.1"],
        )
        self.assertEqual(
            [self.workspace.geometry_resolution_combo.itemText(index) for index in range(2)],
            ["Standard", "Ultra 2K"],
        )
        self._select_geometry_model(GEOMETRY_MODEL_7_1)
        self.assertTrue(self.workspace.geometry_provider_combo.isEnabled())
        self.assertFalse(self.workspace.geometry_resolution_control.isHidden())
        self.assertTrue(self.objects.meshy_target_polycount_control.isHidden())
        self.workspace.geometry_resolution_combo.setCurrentIndex(1)
        self.assertEqual(self.objects.geometry_generation_options(), (GEOMETRY_MODEL_7_1, "2k"))
        self._select_geometry_model("meshy-t2.5")
        self.assertTrue(self.workspace.geometry_provider_combo.isEnabled())
        self.assertTrue(self.workspace.geometry_resolution_control.isHidden())
        self.assertFalse(self.objects.meshy_target_polycount_control.isHidden())

    def test_target_tris_limit_tracks_geometry_model_and_effective_settings(self) -> None:
        spin = self.objects.meshy_target_polycount_spinbox
        self.assertEqual(spin.maximum(), 30_000)
        spin.setValue(25_000)
        self._select_geometry_model("meshy-t2")
        self.assertEqual(spin.maximum(), 15_000)
        self.assertEqual(spin.value(), 15_000)
        self.assertEqual(self.objects._settings.meshy_target_polycount, 15_000)
        self._select_geometry_model("meshy-t2.5")
        self.assertEqual(spin.maximum(), 30_000)
        self.assertEqual(spin.value(), 15_000)

    def test_saved_geometry_provider_restores_without_any_mode_or_seek(self) -> None:
        self._select_geometry_model(GEOMETRY_MODEL_7_1)
        self.workspace.geometry_resolution_combo.setCurrentIndex(1)
        saved = GenerationData.from_dict(self.objects.get_data().to_dict())
        self._select_geometry_model("meshy-t2.5")

        self.objects.set_data(saved)
        self.workspace.sync_shared_controls()

        self.assertEqual(self.workspace.geometry_provider_combo.currentData(), GEOMETRY_MODEL_7_1)
        self.assertEqual(self.workspace.geometry_resolution_combo.currentData(), "2k")
        self.assertTrue(self.workspace.geometry_provider_combo.isEnabled())
        self.assertFalse(self.workspace.geometry_resolution_control.isHidden())

    # ### Ordinary blobs retain independent references, prompts and materials ###
    def test_previous_next_cycle_objects_and_keep_their_reference_images(self) -> None:
        self._paint_two_objects()
        first_key = self.objects.current_inpaint_reference_key()
        first = fixtures._opaque_reference(40)
        second = fixtures._opaque_reference(90)
        self.workspace._activate_temporary_object_reference(first, source_label="First")
        self.workspace.next_inpaint_button.click()
        second_key = self.objects.current_inpaint_reference_key()
        self.assertNotEqual(first_key, second_key)
        self.assertIsNone(self.objects.get_temporary_object_reference())
        self.workspace._activate_temporary_object_reference(second, source_label="Second")

        self.workspace.previous_inpaint_button.click()
        self.assertEqual(self.objects.current_inpaint_reference_key(), first_key)
        np.testing.assert_array_equal(self.objects.get_temporary_object_reference(), first)
        self.workspace.next_inpaint_button.click()
        np.testing.assert_array_equal(self.objects.get_temporary_object_reference(), second)
        self.assertEqual(len(self.objects.inpaint_queue()), 2)
        self.assertIn("Inpaint", self.workspace.inpaint_position_label.text())

    def test_view_prompt_and_pbr_maps_remain_individual_per_blob(self) -> None:
        self._paint_two_objects()
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData("front"))
        self.workspace.reference_generation_prompt_edit.setText("First furniture")
        self.objects.pbr_map_checkboxes["normal"].setChecked(False)
        first_key = self.objects.current_inpaint_reference_key()
        self.workspace.next_inpaint_button.click()
        second_view = str(combo.itemData(0))
        combo.setCurrentIndex(0)
        self.workspace.reference_generation_prompt_edit.setText("Second furniture")
        self.objects.pbr_map_checkboxes["normal"].setChecked(True)

        self.workspace.previous_inpaint_button.click()
        self.assertEqual(combo.currentData(), "front")
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), "First furniture")
        self.assertFalse(self.objects.pbr_map_checkboxes["normal"].isChecked())
        self.assertEqual(
            self.objects.get_inpaint_reference_view_preferences(first_key),
            ("front", "First furniture"),
        )
        self.workspace.next_inpaint_button.click()
        self.assertEqual(combo.currentData(), second_view)
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), "Second furniture")
        self.assertTrue(self.objects.pbr_map_checkboxes["normal"].isChecked())

    def test_clear_mask_removes_only_the_selected_blob(self) -> None:
        self._paint_two_objects()
        removed_key = self.objects.current_inpaint_reference_key()
        self.workspace.clear_mask_button.click()

        remaining = self.objects.inpaint_queue()
        self.assertEqual(len(remaining), 1)
        self.assertNotEqual(self.objects.current_inpaint_reference_key(), removed_key)

    def test_generate_all_remains_visible_for_meshy_7_1(self) -> None:
        self._paint_two_objects()
        self._select_geometry_model(GEOMETRY_MODEL_7_1)
        self.assertFalse(self.workspace.generate_all_references_button.isHidden())
        self.assertTrue(self.workspace.generate_all_references_button.isEnabled())

    # ### One selected image per reference job, including retries ###
    def test_reference_job_uses_only_selected_object_and_replaces_its_reference(self) -> None:
        self._paint_two_objects()
        expected = self.objects.get_current_object_generation_reference()
        with patch(
            "housemaker.merged_generation_workspace._ReferenceGenerationThread",
            wraps=fixtures._ControlledReferenceThread,
        ) as worker:
            thread = self._start()
        np.testing.assert_array_equal(thread.source_bgra, expected)
        self.assertNotIn("additional_sources_bgra", worker.call_args.kwargs)
        self.assertNotIn("combine_partial_views", worker.call_args.kwargs)
        thread.complete(fixtures._opaque_reference(80))
        retry = self._start()
        np.testing.assert_array_equal(retry.source_bgra, expected)
        retry.complete(fixtures._opaque_reference(120))
        np.testing.assert_array_equal(
            self.objects.get_current_object_generation_reference(), fixtures._opaque_reference(120),
        )
        self.assertEqual(len(self.objects.inpaint_queue()), 2)

    def test_reference_worker_forwards_only_single_image_arguments(self) -> None:
        thread = _ReferenceGenerationThread(
            fixtures._opaque_reference(20), "A furniture reference", "test-model",
            api_key="", strength=0.5, parent=self.workspace,
        )
        with patch(
            "housemaker.reference_generation.generate_object_reference",
            return_value=fixtures._opaque_reference(30),
        ) as provider:
            thread.run()
        self.assertIsNone(thread.error_message)
        self.assertNotIn("additional_sources_bgra", provider.call_args.kwargs)
        self.assertNotIn("combine_partial_views", provider.call_args.kwargs)
        thread.deleteLater()

    def test_changing_geometry_provider_does_not_cancel_reference_image_work(self) -> None:
        thread = self._start(fixtures._opaque_reference(20))
        runtime = self.workspace._reference_generation_runtime

        self._select_geometry_model(GEOMETRY_MODEL_7_1)
        self.workspace.geometry_resolution_combo.setCurrentIndex(1)

        self.assertIs(self.workspace._reference_generation_runtime, runtime)
        self.assertFalse(thread.interrupted)
        thread.complete(fixtures._opaque_reference(30))
        self.assertEqual(self.job_manager.jobs()[-1].status, "completed")

    # ### Passive library previews are not explicit component generation modes ###
    def test_passive_opening_status_does_not_show_generation_indicator(self) -> None:
        for asset_kind in ("door", "window"):
            with self.subTest(asset_kind=asset_kind):
                target = self._opening_target(asset_kind)
                self.objects._door_slot_editing_target = target
                self.workspace._activate_temporary_object_reference(
                    fixtures._opaque_reference(20), source_label="Pasted image",
                )
                self.objects._emit_active_door_slot_generation_status()
                self.assertIs(self.objects._door_slot_editing_target, target)
                self.assertFalse(target.generation_ready)
                self.assertTrue(self.workspace.door_slot_generation_indicator.isHidden())
                self.assertFalse(self.workspace._door_slot_generation_blink_timer.isActive())

                explicit_target = replace(target, generation_ready=True)
                self.objects._door_slot_editing_target = explicit_target
                self.objects._door_slot_generation_prompt_visible = True
                self.workspace._activate_temporary_object_reference(
                    fixtures._opaque_reference(30), source_label="Loaded image",
                )
                self.assertIs(self.objects._door_slot_editing_target, explicit_target)
                self.objects._emit_active_door_slot_generation_status()
                self.assertFalse(self.workspace.door_slot_generation_indicator.isHidden())
                self.assertIn(
                    f"{asset_kind}'s body", self.workspace.door_slot_generation_indicator.text(),
                )
                self.assertTrue(self.workspace._door_slot_generation_blink_timer.isActive())
                self.objects._door_slot_generation_prompt_visible = False

    def test_passive_component_preview_does_not_bind_reference_job_context(self) -> None:
        self.workspace._activate_temporary_object_reference(
            fixtures._opaque_reference(20), source_label="Ordinary object",
        )
        ordinary_signature = self.workspace._reference_generation_context_signature()
        for kind in ("door", "window", "trim"):
            with self.subTest(kind=kind):
                self.objects._door_slot_editing_target = None
                self.objects._architectural_trim_editing_target = None
                attribute = (
                    "_architectural_trim_editing_target" if kind == "trim"
                    else "_door_slot_editing_target"
                )
                target = self._trim_target() if kind == "trim" else self._opening_target(kind)
                setattr(self.objects, attribute, target)
                self.assertEqual(
                    self.workspace._reference_generation_context_signature(), ordinary_signature,
                )
                setattr(self.objects, attribute, replace(target, generation_ready=True))
                self.assertNotEqual(
                    self.workspace._reference_generation_context_signature(), ordinary_signature,
                )

    def test_reference_batch_ignores_passive_component_previews(self) -> None:
        from housemaker.plan_image_correction import (
            PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
        )

        self._paint_two_objects()
        self.workspace.set_reference_generation_model(PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1)
        for kind in ("door", "window", "trim"):
            with self.subTest(kind=kind):
                self.objects._door_slot_editing_target = None
                self.objects._architectural_trim_editing_target = None
                attribute = (
                    "_architectural_trim_editing_target" if kind == "trim"
                    else "_door_slot_editing_target"
                )
                target = self._trim_target() if kind == "trim" else self._opening_target(kind)
                setattr(self.objects, attribute, target)
                self.workspace.sync_shared_controls()
                self.assertTrue(self.workspace.generate_all_references_button.isEnabled())
                with patch.object(self.workspace, "_start_shared_reference_batch") as batch:
                    self.workspace.generate_all_references_button.click()
                    batch.assert_called_once()
                    self.assertEqual(len(batch.call_args.args[0]), 2)
                self.assertTrue(self.workspace.door_slot_generation_indicator.isHidden())

                setattr(self.objects, attribute, replace(target, generation_ready=True))
                self.workspace.sync_shared_controls()
                self.assertFalse(self.workspace.generate_all_references_button.isEnabled())
                with patch.object(self.workspace, "_start_shared_reference_batch") as batch:
                    self.workspace._handle_generate_all_references_clicked()
                    batch.assert_not_called()

    def test_only_explicit_component_mode_overrides_selected_surface_texture_action(self) -> None:
        with patch.object(
            self.surfaces, "get_selected_surface_ids", return_value=("level:0/floor",),
        ):
            for kind in ("door", "window", "trim"):
                with self.subTest(kind=kind):
                    self.objects._door_slot_editing_target = None
                    self.objects._architectural_trim_editing_target = None
                    attribute = (
                        "_architectural_trim_editing_target" if kind == "trim"
                        else "_door_slot_editing_target"
                    )
                    target = self._trim_target() if kind == "trim" else self._opening_target(kind)
                    setattr(self.objects, attribute, target)
                    self.workspace._sync_generation_texture_context()
                    self.assertIs(
                        self.workspace.generate_texture_stack.currentWidget(), self.surfaces.generate_button,
                    )

                    setattr(self.objects, attribute, replace(target, generation_ready=True))
                    self.workspace._sync_generation_texture_context()
                    self.assertIs(
                        self.workspace.generate_texture_stack.currentWidget(), self.objects.generate_texture_button,
                    )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
