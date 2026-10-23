"""Shared texture actions support reusable materials without a scene target."""

# ### Environment and imports ###
from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import trimesh

from housemaker.generation_state import (
    EXTERNAL_GLB_GENERATION_PROVIDER,
    GeneratedObjectRecord,
)
from housemaker.generation_workspace import (
    ARCHITECTURAL_TRIM_COMPONENT_PIPELINE_KEY,
    DOOR_COMPONENT_PIPELINE_KEY,
    _ArchitecturalTrimEditingTarget,
    _DoorSlotEditingTarget,
)
from housemaker.merged_generation_workspace import _bgra_array_to_qimage
from housemaker.models import LandingData
from housemaker.surface_geometry import FixedSurface
from tests import test_reference_generation_workspace as fixtures


# ### Local-only shared-workspace fixture ###
class SurfaceTextureLibraryRoutingTests(unittest.TestCase):
    setUp = fixtures.ReferenceGenerationWorkspaceTests.setUp
    tearDown = fixtures.ReferenceGenerationWorkspaceTests.tearDown
    _apply_reference_generation_model = (
        fixtures.ReferenceGenerationWorkspaceTests._apply_reference_generation_model
    )
    _load_video = fixtures.ReferenceGenerationWorkspaceTests._load_video

    def _configure_keys(self) -> None:
        settings = replace(
            self.objects.get_runtime_settings(),
            meshy_api_key="test-meshy-key",
            openai_api_key="test-openai-key",
        )
        self.objects.set_runtime_settings(settings)
        self.surfaces.set_runtime_settings(settings)

    def _select_record(self, object_id: str, pipeline=None) -> GeneratedObjectRecord:
        record = GeneratedObjectRecord(
            object_id=object_id,
            frame_index=0,
            object_name=object_id,
            pipeline=pipeline or {},
            provider=EXTERNAL_GLB_GENERATION_PROVIDER,
            asset_path=f"{object_id}.glb",
        )
        self.objects._data.generated_objects.append(record)
        self.objects._selected_object_id = object_id
        self.objects._sync_controls()
        return record

    def _selected_surface(self) -> FixedSurface:
        mesh = trimesh.creation.box(extents=(1.0, 0.1, 1.0))
        return FixedSurface(
            surface_id="level:0/room:0/wall:1:2",
            surface_type="wall",
            level_index=0,
            room_index=0,
            mesh=mesh,
            area_square_meters=float(mesh.area),
        )

    # ### Library-only Surface action must leave new Object generation available ###
    def test_unselected_reference_uses_surface_action_without_blocking_object_geometry(
        self,
    ) -> None:
        self._configure_keys()
        self.workspace._activate_temporary_object_reference(
            fixtures._opaque_reference(40),
            source_label="Loaded image",
        )

        self.assertIs(
            self.workspace.generate_texture_stack.currentWidget(),
            self.surfaces.generate_button,
        )
        self.assertTrue(self.surfaces.generate_button.isEnabled())
        self.assertFalse(self.objects._external_surface_generation_target_selected)
        self.assertTrue(self.objects.generate_button.isEnabled())
        self.assertTrue(self.objects.generate_geometry_button.isEnabled())

    def test_dangling_object_selection_does_not_hide_library_surface_generation(
        self,
    ) -> None:
        self._configure_keys()
        self.objects._selected_object_id = "deleted-object"
        self.workspace._activate_temporary_object_reference(
            fixtures._opaque_reference(40),
            source_label="Pasted image",
        )

        self.assertIs(
            self.workspace.generate_texture_stack.currentWidget(),
            self.surfaces.generate_button,
        )
        self.assertTrue(self.surfaces.generate_button.isEnabled())
        self.assertTrue(self.objects.generate_geometry_button.isEnabled())

    def test_object_selection_changes_texture_route_without_an_extra_shared_sync(
        self,
    ) -> None:
        self._select_record("chair")
        self.assertIs(
            self.workspace.generate_texture_stack.currentWidget(),
            self.objects.generate_texture_button,
        )

        self.objects._selected_object_id = None
        self.objects._sync_controls()

        self.assertIs(
            self.workspace.generate_texture_stack.currentWidget(),
            self.surfaces.generate_button,
        )
        self.assertFalse(self.objects._external_surface_generation_target_selected)

    def test_fixed_surface_takes_priority_over_ordinary_object_selection(self) -> None:
        self._configure_keys()
        self.workspace._activate_temporary_object_reference(
            fixtures._opaque_reference(40),
            source_label="Pasted image",
        )
        self._select_record("chair")
        self.surfaces.set_scene_surface_selection((self._selected_surface(),))

        self.assertIs(
            self.workspace.generate_texture_stack.currentWidget(),
            self.surfaces.generate_button,
        )
        self.assertTrue(self.objects._external_surface_generation_target_selected)
        self.assertFalse(self.objects.generate_button.isEnabled())
        self.assertFalse(self.objects.generate_geometry_button.isEnabled())

        self.surfaces.set_scene_surface_selection(())

        self.assertIs(
            self.workspace.generate_texture_stack.currentWidget(),
            self.objects.generate_texture_button,
        )
        self.assertFalse(self.objects._external_surface_generation_target_selected)

    # ### Explicit component generation wins; passive library previews do not ###
    def test_passive_opening_records_still_allow_library_generation(self) -> None:
        self._configure_keys()
        for asset_kind in ("door", "window"):
            with self.subTest(asset_kind=asset_kind):
                target = _DoorSlotEditingTarget(
                    door_id=f"{asset_kind}-1",
                    slot_id="body",
                    object_id=f"{asset_kind}-body",
                    object_name="Body",
                    slot_display_name="body",
                    asset_kind=asset_kind,
                    generation_ready=False,
                )
                self.objects._door_slot_editing_target = target
                self._select_record(
                    target.object_id, {DOOR_COMPONENT_PIPELINE_KEY: {"slot_id": "body"}}
                )
                self.workspace._activate_temporary_object_reference(
                    fixtures._opaque_reference(40),
                    source_label="Pasted image",
                )
                self.assertIs(
                    self.workspace.generate_texture_stack.currentWidget(),
                    self.surfaces.generate_button,
                )
                self.assertTrue(self.surfaces.generate_button.isEnabled())
                self.assertFalse(
                    self.objects._external_surface_generation_target_selected
                )

                self.objects._door_slot_editing_target = replace(
                    target, generation_ready=True
                )
                self.workspace._sync_generation_texture_context()
                self.assertIs(
                    self.workspace.generate_texture_stack.currentWidget(),
                    self.objects.generate_texture_button,
                )
                self.objects._door_slot_editing_target = None

    def test_passive_trim_record_and_active_trim_use_different_texture_routes(
        self,
    ) -> None:
        target = _ArchitecturalTrimEditingTarget(
            level_index=0,
            trim_id="cornice-1",
            object_id="cornice",
            object_name="Cornice",
            fit_dimensions=(1.0, 0.1, 0.1),
            generation_ready=False,
        )
        self.objects._architectural_trim_editing_target = target
        self._select_record(
            target.object_id,
            {
                ARCHITECTURAL_TRIM_COMPONENT_PIPELINE_KEY: {
                    "level_index": 0,
                    "trim_id": target.trim_id,
                },
            },
        )
        self.assertIs(
            self.workspace.generate_texture_stack.currentWidget(),
            self.surfaces.generate_button,
        )

        self.objects._architectural_trim_editing_target = replace(
            target, generation_ready=True
        )
        self.workspace._sync_generation_texture_context()
        self.assertIs(
            self.workspace.generate_texture_stack.currentWidget(),
            self.objects.generate_texture_button,
        )

    def test_active_landing_keeps_object_texture_action(self) -> None:
        self.objects._landing_editing_target = LandingData(center_x=0.0, center_y=0.0)
        self.workspace._sync_generation_texture_context()

        self.assertIs(
            self.workspace.generate_texture_stack.currentWidget(),
            self.objects.generate_texture_button,
        )
        self.assertFalse(self.objects._external_surface_generation_target_selected)

    # ### Actual shared-button requests, with provider dispatch replaced by a mock ###
    def test_pasted_and_loaded_images_are_single_library_surface_references(
        self,
    ) -> None:
        self._configure_keys()
        image = fixtures._opaque_reference(70)
        image[:, :, :3] = (30, 100, 180)
        qimage = _bgra_array_to_qimage(image)
        path = self.root / "material.png"
        self.assertTrue(qimage.save(str(path)))
        for source in ("Paste", "Load image"):
            with self.subTest(source=source):
                if source == "Paste":
                    with patch(
                        "housemaker.merged_generation_workspace._read_clipboard_image",
                        return_value=qimage,
                    ):
                        self.workspace._handle_paste_inpaint_clicked()
                else:
                    with patch(
                        "housemaker.merged_generation_workspace.QFileDialog.getOpenFileName",
                        return_value=(str(path), ""),
                    ):
                        self.workspace.load_reference_image_button.click()
                self.workspace.ai_prompt_edit.setText("Plain material without objects")
                with patch.object(
                    self.surfaces, "_start_generation", return_value=True
                ) as start:
                    self.workspace.generate_texture_stack.currentWidget().click()
                start.assert_called_once()
                request = start.call_args.args[0]
                self.assertEqual(request.surface_ids, ())
                self.assertEqual(request.surface_type, "surface")
                self.assertEqual(len(request.reference_pngs), 1)
                decoded = cv2.imdecode(
                    np.frombuffer(request.reference_pngs[0], dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED,
                )
                np.testing.assert_array_equal(decoded[0, 0, :3], image[0, 0, :3])
                self.assertIn("Plain material without objects", request.prompt)
                self.assertIsNone(start.call_args.kwargs["reference_input"])

    def test_video_mask_library_request_keeps_deferred_reference_frames(self) -> None:
        self._configure_keys()
        self._load_video()
        self.workspace.video_view.strokes_changed.emit(
            self.workspace.video_view.get_strokes()
        )
        self.workspace.sync_shared_controls()
        self.assertTrue(self.surfaces.generate_button.isEnabled())

        with patch.object(
            self.surfaces, "_start_generation", return_value=True
        ) as start:
            self.workspace.generate_texture_stack.currentWidget().click()

        start.assert_called_once()
        request = start.call_args.args[0]
        self.assertEqual(request.surface_ids, ())
        self.assertEqual(request.surface_type, "surface")
        reference_input = start.call_args.kwargs["reference_input"]
        self.assertIsNotNone(reference_input)
        self.assertEqual(len(reference_input.frame_strokes), 1)
        self.assertTrue(reference_input.frame_strokes[0][1])


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
