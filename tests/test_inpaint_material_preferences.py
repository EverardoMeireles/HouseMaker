"""Reference instructions and material selections belong to individual inpaints."""

# ### Environment and imports ###
from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np
import test_inpaint_reference_view_preferences as fixtures

from housemaker.generation_state import GeneratedObjectRecord, GenerationData
from housemaker.generation_workspace import PBR_MAP_TYPES
from housemaker.reference_generation import MAX_REFERENCE_PROMPT_CHARACTERS
from housemaker.settings_widget import GenerationServiceSettings


# ### Per-blob reference instructions ###
class InpaintMaterialPreferencesTests(unittest.TestCase):
    setUp = fixtures.InpaintReferenceViewPreferencesTests.setUp
    tearDown = fixtures.InpaintReferenceViewPreferencesTests.tearDown

    def test_reference_prompts_are_individual_and_preserve_editable_whitespace(self) -> None:
        first, second, third = self.keys
        self.assertIsNone(self.workspace.get_inpaint_reference_view_preferences(first))
        self.workspace.set_inpaint_reference_view_preferences(first, "front", "  first\nobject  ")
        self.workspace.set_inpaint_reference_view_preferences(third, "three-quarter", "last object")
        self.assertEqual(self.workspace.get_inpaint_reference_view_preferences(first), ("front", "  first\nobject  "))
        self.assertIsNone(self.workspace.get_inpaint_reference_view_preferences(second))
        self.assertEqual(self.workspace.get_inpaint_reference_view_preferences(third), ("three-quarter", "last object"))
        self.workspace.set_inpaint_reference_view_preferences(first, "front", "")
        self.assertEqual(self.workspace.get_inpaint_reference_view_preferences(first), ("front", ""))

    def test_reference_prompt_validates_type_and_limit_before_replacing_text(self) -> None:
        self.workspace.set_inpaint_reference_view_preferences(self.keys[0], "front", "saved")
        for value in (None, False, 4, [], {}):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.workspace.set_inpaint_reference_view_preferences(self.keys[0], "front", value)
        with self.assertRaises(ValueError):
            self.workspace.set_inpaint_reference_view_preferences(
                self.keys[0], "front", "x" * (MAX_REFERENCE_PROMPT_CHARACTERS + 1),
            )
        self.assertEqual(self.workspace.get_inpaint_reference_view_preferences(self.keys[0]), ("front", "saved"))
        boundary = "x" * MAX_REFERENCE_PROMPT_CHARACTERS
        self.workspace.set_inpaint_reference_view_preferences(self.keys[0], "front", boundary)
        self.assertEqual(self.workspace.get_inpaint_reference_view_preferences(self.keys[0]), ("front", boundary))

    def test_offscreen_preferences_do_not_seek_or_decode_or_mutate_project_data(self) -> None:
        original_data = self.workspace._data.to_dict()
        original_key = self.workspace.current_inpaint_reference_key()
        changed = Mock()
        self.workspace.inpaint_selection_changed.connect(changed)
        with patch.object(self.workspace, "show_frame", side_effect=AssertionError("No seek")), patch.object(
            self.source, "get_frame", side_effect=AssertionError("No decode"),
        ):
            self.workspace.set_inpaint_reference_view_preferences(self.keys[2], "front", "Keep the handles")
            self.workspace.set_inpaint_pbr_maps(self.keys[2], (PBR_MAP_TYPES[1],))
            self.assertEqual(self.workspace.get_inpaint_reference_view_preferences(self.keys[2]), ("front", "Keep the handles"))
            self.assertEqual(self.workspace.get_inpaint_pbr_maps(self.keys[2]), (PBR_MAP_TYPES[1],))
        self.assertEqual(self.workspace._data.to_dict(), original_data)
        self.assertEqual(self.workspace.current_inpaint_reference_key(), original_key)
        self.assertEqual(self.workspace._get_enabled_pbr_maps(), PBR_MAP_TYPES)
        changed.assert_not_called()

    # ### Material state restoration ###
    def test_manual_maps_are_stored_only_for_the_selected_blob(self) -> None:
        self.workspace.pbr_map_checkboxes[PBR_MAP_TYPES[0]].setChecked(False)
        expected = PBR_MAP_TYPES[1:]
        self.assertEqual(self.workspace.get_inpaint_pbr_maps(self.keys[0]), expected)
        self.assertIsNone(self.workspace.get_inpaint_pbr_maps(self.keys[1]))
        self.workspace.navigate_inpaints(1)
        self.assertEqual(self.workspace._get_enabled_pbr_maps(), PBR_MAP_TYPES)
        self.workspace.navigate_inpaints(-1)
        self.assertEqual(self.workspace._get_enabled_pbr_maps(), expected)

    def test_active_map_writes_notify_shared_views_after_restoring_controls(self) -> None:
        snapshots = []
        self.workspace.inpaint_selection_changed.connect(
            lambda: snapshots.append(self.workspace._get_enabled_pbr_maps()),
        )
        self.workspace.set_inpaint_pbr_maps(self.keys[0], ())
        self.assertEqual(snapshots, [()])
        self.workspace.set_inpaint_pbr_maps(self.keys[2], (PBR_MAP_TYPES[0],))
        self.assertEqual(snapshots, [()])

    def test_navigation_restores_all_maps_none_or_subset_without_toggle_signals(self) -> None:
        choices = ((), (PBR_MAP_TYPES[0],), PBR_MAP_TYPES)
        for key, maps in zip(self.keys, choices):
            self.workspace.set_inpaint_pbr_maps(key, maps)
        toggled = Mock()
        for checkbox in self.workspace.pbr_map_checkboxes.values():
            checkbox.toggled.connect(toggled)
        preview = Mock()
        self.workspace.result_view.set_pbr_maps_enabled = preview
        for key, maps in zip(self.keys, choices):
            while self.workspace.current_inpaint_reference_key() != key:
                self.workspace.navigate_inpaints(1)
            self.assertEqual(self.workspace._get_enabled_pbr_maps(), maps)
            self.assertEqual(self.workspace.get_inpaint_pbr_maps(key), maps)
        toggled.assert_not_called()
        preview.assert_called_with(PBR_MAP_TYPES)

    def test_map_values_are_validated_canonicalized_and_detached_from_input(self) -> None:
        maps = [PBR_MAP_TYPES[2], PBR_MAP_TYPES[0], PBR_MAP_TYPES[2]]
        self.workspace.set_inpaint_pbr_maps(self.keys[0], maps)
        maps.clear()
        expected = (PBR_MAP_TYPES[0], PBR_MAP_TYPES[2])
        self.assertEqual(self.workspace.get_inpaint_pbr_maps(self.keys[0]), expected)
        for value in (None, "normal", ("unknown",), 4):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.workspace.set_inpaint_pbr_maps(self.keys[0], value)
        self.assertEqual(self.workspace.get_inpaint_pbr_maps(self.keys[0]), expected)

    def test_component_and_standalone_controls_are_not_changed_by_blob_writes(self) -> None:
        for attribute in ("_door_slot_editing_target", "_architectural_trim_editing_target"):
            with self.subTest(attribute=attribute):
                setattr(self.workspace, attribute, object())
                try:
                    self.workspace.set_inpaint_reference_view_preferences(self.keys[0], "front", "normal object")
                    self.workspace.set_inpaint_pbr_maps(self.keys[0], ())
                    self.assertEqual(self.workspace._get_enabled_pbr_maps(), PBR_MAP_TYPES)
                finally:
                    setattr(self.workspace, attribute, None)
        self.workspace._restore_inpaint_pbr_maps(None)
        self.assertEqual(self.workspace._get_enabled_pbr_maps(), PBR_MAP_TYPES)

    # ### Cache lifecycle ###
    def test_deleted_or_changed_blobs_reject_both_preferences_and_prune(self) -> None:
        for key in (self.keys[0], self.keys[2]):
            self.workspace.set_inpaint_reference_view_preferences(key, "front", "saved")
            self.workspace.set_inpaint_pbr_maps(key, ())
        self.workspace.video_view.clear_mask()
        self.workspace._data.set_frame_strokes(2, [fixtures._stroke(0.6)])
        for key in (self.keys[0], self.keys[2], (3, "absent")):
            self.assertIsNone(self.workspace.get_inpaint_reference_view_preferences(key))
            self.assertIsNone(self.workspace.get_inpaint_pbr_maps(key))
            self.assertFalse(self.workspace.set_inpaint_reference_view_preferences(key, "front", "stale"))
            self.assertFalse(self.workspace.set_inpaint_pbr_maps(key, ()))
        self.assertNotIn(self.keys[0], self.workspace._inpaint_reference_view_preferences)
        self.assertNotIn(self.keys[0], self.workspace._inpaint_pbr_maps)

    def test_replacement_project_and_video_clear_both_session_caches(self) -> None:
        self.workspace.set_inpaint_reference_view_preferences(self.keys[0], "front", "saved")
        self.workspace.set_inpaint_pbr_maps(self.keys[0], ())
        self.workspace.set_data(GenerationData())
        self.assertFalse(self.workspace._inpaint_reference_view_preferences)
        self.assertFalse(self.workspace._inpaint_pbr_maps)
        self.workspace._inpaint_reference_view_preferences[self.keys[0]] = ("front", "stale")
        self.workspace._inpaint_pbr_maps[self.keys[0]] = ()
        with patch("housemaker.generation_workspace.probe_video", return_value=self.source.metadata), patch(
            "housemaker.generation_workspace.VideoFrameSource", return_value=fixtures._VideoSource(),
        ):
            self.workspace.load_video(self.source.metadata.path)
        self.assertFalse(self.workspace._inpaint_reference_view_preferences)
        self.assertFalse(self.workspace._inpaint_pbr_maps)

    # ### Generation and texture request snapshots ###
    def test_full_and_geometry_batches_snapshot_each_blobs_own_maps(self) -> None:
        self.workspace.set_runtime_settings(GenerationServiceSettings(meshy_api_key="test"))
        choices = ((), (PBR_MAP_TYPES[0],), PBR_MAP_TYPES)
        for key, maps in zip(self.keys, choices):
            self.workspace.set_inpaint_pbr_maps(key, maps)
        self.workspace.navigate_inpaints(1)
        for geometry_only in (False, True):
            with self.subTest(geometry_only=geometry_only):
                requests = self.workspace._build_generation_requests(geometry_only=geometry_only)
                self.assertEqual(tuple(request.enabled_pbr_maps for request in requests), choices)
                self.assertEqual(self.workspace.current_inpaint_reference_key(), self.keys[1])
                self.workspace.set_inpaint_pbr_maps(self.keys[0], PBR_MAP_TYPES)
                self.assertEqual(requests[0].enabled_pbr_maps, ())
                self.workspace.set_inpaint_pbr_maps(self.keys[0], ())

    def test_unconfigured_blobs_default_to_all_maps_not_the_visible_selection(self) -> None:
        self.workspace.set_runtime_settings(GenerationServiceSettings(meshy_api_key="test"))
        self.workspace.set_inpaint_pbr_maps(self.keys[0], ())
        requests = self.workspace._build_generation_requests()
        self.assertEqual(tuple(request.enabled_pbr_maps for request in requests), ((), PBR_MAP_TYPES, PBR_MAP_TYPES))

    def test_texture_batch_captures_own_maps_after_geometry_and_later_edits(self) -> None:
        self.workspace.set_runtime_settings(GenerationServiceSettings(meshy_api_key="test"))
        choices = ((), (PBR_MAP_TYPES[0],), PBR_MAP_TYPES)
        for key, maps in zip(self.keys, choices):
            self.workspace.set_inpaint_pbr_maps(key, maps)
        with patch.object(self.workspace, "_start_generation", side_effect=["a", "b", "c"]):
            self.workspace.generate_geometry()
        self.workspace._data.generated_objects = [
            GeneratedObjectRecord(
                object_id=object_id, frame_index=frame_index, object_name=object_id,
                pipeline={}, provider_task_id=object_id, asset_path=f"{object_id}.glb",
            ) for object_id, frame_index in (("a", 0), ("b", 0), ("c", 2))
        ]
        self.workspace.set_inpaint_pbr_maps(self.keys[2], (PBR_MAP_TYPES[2],))
        self.workspace.navigate_inpaints(1)
        with patch.object(self.workspace, "_build_texture_regeneration_request", side_effect=[Mock(), Mock(), Mock()]) as build, patch.object(
            self.workspace, "_start_texture_regeneration", return_value=True,
        ):
            self.assertTrue(self.workspace.generate_selected_object_texture())
        self.assertEqual(
            [call.kwargs["enabled_pbr_maps_override"] for call in build.call_args_list],
            [(), (PBR_MAP_TYPES[0],), (PBR_MAP_TYPES[2],)],
        )

    def test_texture_preflight_applies_explicit_empty_subset_and_default_maps(self) -> None:
        self.workspace._data.generated_objects = [GeneratedObjectRecord(
            object_id="object", frame_index=0, object_name="Object", pipeline={},
            provider_task_id="object", asset_path="object.glb",
        )]
        reference = np.full((24, 24, 4), 255, np.uint8)
        with patch.object(self.workspace, "_can_regenerate_object_texture", return_value=True), patch.object(
            self.workspace, "_object_has_complete_texture_uvs", return_value=False,
        ), patch(
            "housemaker.generation_workspace._build_generation_asset_revision",
            return_value=("object.glb", 1, 2, 3),
        ):
            for maps in (None, (), (PBR_MAP_TYPES[1],)):
                with self.subTest(maps=maps):
                    request = self.workspace._build_texture_regeneration_request(
                        object_id="object", reference_image_bgra=reference,
                        enabled_pbr_maps_override=maps,
                    )
                    self.assertIsNotNone(request, self.workspace.status_label.text())
                    self.assertEqual(request.enabled_pbr_maps, PBR_MAP_TYPES if maps is None else maps)


# ### Standalone runner ###
if __name__ == "__main__":
    unittest.main()
