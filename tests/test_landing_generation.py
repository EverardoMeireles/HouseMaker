"""Landing generation reuses Meshy without publishing ordinary library objects."""

# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import cv2
import numpy as np
import pytest
import trimesh
from PIL import Image
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.generation_jobs import GenerationJobManager
from housemaker.generation_workspace import (
    GenerationRequest,
    GenerationWorkspace,
    MeshyImagePlanner,
    MeshyTextureRegenerator,
    TextureRegenerationRequest,
)
from housemaker.glb import GeneratedModel, import_generated_glb
from housemaker.landing_generation import (
    LandingGenerationThread,
    LandingMeshGenerationResult,
    LandingViewReference,
    landing_reference_signature,
)
from housemaker.landing_geometry import build_landing_mesh
from housemaker.merged_generation_workspace import MergedGenerationWorkspace
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.models import LandingData, LevelData
from housemaker.openai_batch import LUNA_BATCH_MODEL
from housemaker.reference_view_presets import ReferenceViewPreset
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.surface_texture_workspace import SurfaceTextureGenerationWorkspace

# ### Fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


def _landing():
    return LandingData(
        center_x=100.0,
        center_y=200.0,
        rotation_degrees=32.0,
        scale_x=1.5,
        elevation_meters=0.7,
    )


def _image(value=100):
    image = np.full((24, 32, 4), value, dtype=np.uint8)
    image[:, :, 3] = 255
    return image


def _shape():
    return {
        "footprint_meters": [[-2, -1], [2, -1], [2, 0], [0, 0], [0, 1], [-2, 1]],
        "thickness_meters": 0.2,
    }


def _views():
    return tuple(
        ReferenceViewPreset(f"view-{i}", f"View {i}", f"Use view {i}") for i in range(8)
    )


def _references():
    return tuple(
        LandingViewReference(view.preset_id, view.name, _image(i))
        for i, view in enumerate(_views())
    )


def _selected_view():
    return _views()[5]


def _selected_reference():
    return _references()[5]


def _settings():
    return GenerationServiceSettings(
        meshy_api_key="test-meshy-key",
        unused_face_removal=False,
        use_uv_raycast_for_object_generation=False,
    )


def _provider_result():
    scene = trimesh.Scene(trimesh.creation.box(extents=(2.0, 0.2, 1.0)))
    return MeshyGenerationResult("landing-task", bytes(scene.export(file_type="glb")))


def _mesh_request(*, geometry_only=True):
    return GenerationRequest(
        frame_index=3,
        selected_object_bgra=_selected_reference().image_bgra,
        settings=_settings(),
        geometry_only=geometry_only,
        reference_images_bgra=(_selected_reference().image_bgra,),
    )


# ### Meshy geometry routing ###
@pytest.mark.parametrize("action", ["geometry", "generate"])
def test_geometry_worker_reuses_meshy_and_never_infers_a_qwen_polygon(action):
    received = []
    original = _landing()
    before = original.to_dict()

    def plan(request):
        received.append(request)
        return _provider_result()

    request = _mesh_request(geometry_only=action == "geometry")
    thread = LandingGenerationThread(
        original,
        _image(),
        action=action,
        references=(_selected_reference(),),
        mesh_request=request,
        planner=plan,
        executor=lambda result: import_generated_glb(result.glb_bytes),
    )
    with patch(
        "housemaker.qwen_ceiling_height_estimation.QwenCeilingHeightServer"
    ) as qwen:
        thread.run()
    qwen.assert_not_called()
    assert thread.error_message is None
    assert isinstance(thread.result, LandingMeshGenerationResult)
    assert thread.result.task_id == "landing-task"
    assert thread.result.geometry_only is (action == "geometry")
    assert len(thread.result.model.mesh.faces) == 12
    assert received == [request]
    assert original.to_dict() == before


@pytest.mark.parametrize("remove_faces", [False, True])
def test_landing_meshy_geometry_respects_fourteen_camera_face_removal_setting(
    remove_faces,
):
    from housemaker.unused_face_removal import FACE_REMOVAL_CAMERA_IDS

    request = _mesh_request()
    request.settings = replace(_settings(), unused_face_removal=remove_faces)
    model = import_generated_glb(_provider_result().glb_bytes)
    removal = SimpleNamespace(
        glb_bytes=model.glb_bytes,
        original_face_count=12,
        retained_face_count=12,
        removed_face_count=0,
        protected_face_count=0,
        visibility_removed_face_count=0,
        stacked_face_removed_count=0,
    )
    thread = LandingGenerationThread(
        _landing(),
        _image(),
        action="geometry",
        references=(_selected_reference(),),
        mesh_request=request,
        planner=MeshyImagePlanner(),
        executor=lambda _result: model,
    )
    with (
        patch(
            "housemaker.generation_workspace.request_image_to_3d_model",
            return_value=_provider_result(),
        ),
        patch(
            "housemaker.generation_workspace.remove_unused_faces_from_glb",
            return_value=removal,
        ) as cleanup,
    ):
        thread.run()
    assert thread.error_message is None
    assert cleanup.call_count == int(remove_faces)
    if remove_faces:
        assert (
            cleanup.call_args.kwargs["options"].enabled_camera_ids
            == FACE_REMOVAL_CAMERA_IDS
        )


# ### Shared reference session and worker cancellation ###
def test_reference_worker_renders_only_selected_view_in_one_resident_session():
    calls = []
    sessions = []

    @contextmanager
    def session(*_args):
        sessions.append("opened")

        def generate(source, prompt, **_options):
            calls.append((source.copy(), prompt))
            return _image(len(calls))

        yield generate
        sessions.append("closed")

    thread = LandingGenerationThread(
        _landing(),
        _image(),
        action="references",
        views=(_selected_view(),),
        reference_session=session,
    )
    thread.run()
    assert thread.error_message is None
    assert len(thread.result) == 1
    assert thread.result[0].view_id == _selected_view().preset_id
    assert sessions == ["opened", "closed"]
    assert [prompt for _source, prompt in calls] == [_selected_view().prompt]
    assert all(np.array_equal(source, _image()) for source, _prompt in calls)


def test_reference_worker_rejects_multiple_views_before_provider_submission():
    thread = LandingGenerationThread(
        _landing(),
        _image(),
        action="references",
        views=_views(),
    )
    with patch.object(thread, "_reference_session") as session:
        thread.run()
    session.assert_not_called()
    assert thread.result is None
    assert thread.error_message is not None


def test_cancelled_reference_worker_publishes_no_partial_reference_set():
    cancelled = False

    @contextmanager
    def session(*_args):
        def generate(*_args, **_options):
            nonlocal cancelled
            cancelled = True
            return _image()

        yield generate

    thread = LandingGenerationThread(
        _landing(),
        _image(),
        action="references",
        views=(_selected_view(),),
        reference_session=session,
    )
    with patch.object(thread, "isInterruptionRequested", side_effect=lambda: cancelled):
        thread.run()
    assert thread.was_cancelled
    assert thread.result is None


def test_geometry_worker_requires_generated_references_before_submitting_meshy():
    calls = []
    thread = LandingGenerationThread(
        _landing(),
        _image(),
        action="geometry",
        mesh_request=_mesh_request(),
        planner=lambda request: calls.append(request),
    )
    thread.run()
    assert thread.result is None
    assert "reference" in thread.error_message.lower()
    assert not calls


@pytest.mark.parametrize("geometry_only", [True, False])
def test_single_selected_landing_reference_reaches_meshy_image_generation(geometry_only):
    request = _mesh_request(geometry_only=geometry_only)
    with (
        patch(
            "housemaker.generation_workspace.request_image_to_3d_model",
            return_value=_provider_result(),
        ) as geometry,
        patch(
            "housemaker.generation_workspace.request_retextured_model",
            return_value=_provider_result(),
        ) as retexture,
    ):
        result = MeshyImagePlanner().plan(request)
    assert result.task_id == "landing-task"
    geometry.assert_called_once()
    decoded = cv2.imdecode(
        np.frombuffer(geometry.call_args.kwargs["image_png"], dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    np.testing.assert_array_equal(decoded, _selected_reference().image_bgra)
    retexture.assert_not_called()


def test_meshy_multiview_generate_uses_primary_geometry_and_four_texture_images():
    request = _mesh_request(geometry_only=False)
    request.reference_images_bgra = tuple(
        item.image_bgra for item in _references()[:4]
    )
    with (
        patch(
            "housemaker.generation_workspace.request_image_to_3d_model",
            return_value=_provider_result(),
        ) as geometry,
        patch(
            "housemaker.generation_workspace.request_retextured_model",
            return_value=_provider_result(),
        ) as texture,
    ):
        result = MeshyImagePlanner().plan(request)
    assert result.task_id == "landing-task"
    assert geometry.call_args.kwargs["should_texture"] is False
    decoded_primary = cv2.imdecode(
        np.frombuffer(geometry.call_args.kwargs["image_png"], dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert np.array_equal(decoded_primary, request.selected_object_bgra)
    images = texture.call_args.kwargs["reference_images_png"]
    assert len(images) == 4
    for encoded, reference in zip(images, request.reference_images_bgra, strict=True):
        decoded = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        assert np.array_equal(decoded, reference)


def test_meshy_texture_adapter_submits_all_supported_views_and_requested_pbr():
    request = TextureRegenerationRequest(
        object_id="landing:test",
        reference_frame_index=0,
        reference_image_bgra=_image(),
        reference_images_bgra=tuple(item.image_bgra for item in _references()[:4]),
        model_glb=_provider_result().glb_bytes,
        settings=_settings(),
        enabled_pbr_maps=("normal", "roughness", "metallic"),
    )
    with patch(
        "housemaker.generation_workspace.request_retextured_model",
        return_value=_provider_result(),
    ) as texture:
        MeshyTextureRegenerator().regenerate(request)
    assert texture.call_args.kwargs["model_glb"] == request.model_glb
    assert len(texture.call_args.kwargs["reference_images_png"]) == 4
    assert texture.call_args.kwargs["enable_pbr"] is True


def test_extra_reference_views_are_not_silently_submitted_beyond_meshy_limit():
    with pytest.raises(ValueError, match="at most four"):
        GenerationRequest(
            frame_index=0,
            selected_object_bgra=_image(),
            settings=_settings(),
            reference_images_bgra=tuple(item.image_bgra for item in _references()),
        )


def test_luna_batch_submits_only_selected_view_in_one_existing_batch_request():
    _success, encoded = cv2.imencode(".png", _image())
    thread = LandingGenerationThread(
        _landing(),
        _image(),
        action="references",
        views=(_selected_view(),),
        model=LUNA_BATCH_MODEL,
        api_key="test-key",
    )
    with (
        patch("housemaker.openai_batch.run_responses_batch") as batch,
        patch(
            "housemaker.plan_image_correction._decode_openai_image_generation_result",
            return_value=encoded.tobytes(),
        ),
    ):
        batch.side_effect = lambda requests, **_kwargs: {
            key: SimpleNamespace(body={}, error=None) for key in requests
        }
        thread.run()
    assert thread.error_message is None
    assert len(thread.result) == 1
    batch.assert_called_once()
    assert set(batch.call_args.args[0]) == {_selected_view().preset_id}


# ### Workspace routing and late-result safety ###
class LandingGenerationWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = GenerationWorkspace(
            asset_directory=Path(self.temporary.name),
            job_manager=GenerationJobManager(),
        )
        self.workspace._settings = _settings()
        self.workspace._meshy_planner = lambda _request: _provider_result()
        self.workspace._meshy_executor = lambda result: import_generated_glb(
            result.glb_bytes
        )
        self.landing = _landing()
        self.workspace.set_landing_editing_target(self.landing)
        self.workspace.set_temporary_object_reference(_image())
        self.workspace._landing_references = (_selected_reference(),)
        self.workspace._landing_reference_signature = landing_reference_signature(
            _image()
        )

    def tearDown(self):
        self.workspace.shutdown()
        self.workspace.close()
        _application.processEvents()
        self.temporary.cleanup()

    def _prepare_completed_geometry(self):
        with patch.object(LandingGenerationThread, "start"):
            self.workspace.generate_geometry()
        thread = self.workspace._landing_thread
        self.assertIsNotNone(thread)
        thread.run()
        return thread

    def _finish(self, thread):
        with patch.object(self.workspace, "sender", return_value=thread):
            self.workspace._handle_landing_generation_finished()

    def test_geometry_applies_only_to_existing_landing_not_object_library(self):
        completed = []
        self.workspace.landing_generation_completed.connect(
            lambda *values: completed.append(values)
        )
        with patch.object(self.workspace, "_start_mask_blob_generations") as normal:
            thread = self._prepare_completed_geometry()
            self._finish(thread)
        normal.assert_not_called()
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0][0], self.landing.landing_id)
        self.assertIsInstance(completed[0][1], LandingMeshGenerationResult)
        self.assertTrue(completed[0][1].geometry_only)
        self.assertEqual(self.workspace.get_data().generated_objects, [])

    def test_generate_requests_textured_mesh_with_live_settings_and_selected_reference(
        self,
    ):
        self.workspace.meshy_target_polycount_spinbox.setValue(3456)
        with patch.object(LandingGenerationThread, "start"):
            self.workspace.generate()
        thread = self.workspace._landing_thread
        self.assertEqual(thread.action, "generate")
        self.assertFalse(thread.mesh_request.geometry_only)
        self.assertFalse(thread.mesh_request.symmetric_division_enabled)
        self.assertEqual(thread.mesh_request.settings.meshy_target_polycount, 3456)
        self.assertEqual(len(thread.mesh_request.reference_images_bgra), 1)
        self.assertTrue(
            np.array_equal(
                thread.mesh_request.selected_object_bgra,
                self.workspace._landing_references[0].image_bgra,
            )
        )
        self.assertEqual(
            thread.mesh_request.enabled_pbr_maps, self.workspace._get_enabled_pbr_maps()
        )
        thread.run()
        self._finish(thread)
        self.assertEqual(self.workspace.get_data().generated_objects, [])

    def test_texture_targets_current_landing_model_and_never_creates_library_record(
        self,
    ):
        model = import_generated_glb(_provider_result().glb_bytes)
        self.workspace.update_landing_editing_preview(self.landing, model)
        requested = []

        def retexture(request):
            requested.append(request)
            return _provider_result()

        self.workspace._meshy_texture_regenerator = retexture
        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())
        with patch.object(LandingGenerationThread, "start"):
            self.assertTrue(self.workspace.generate_selected_object_texture())
        thread = self.workspace._landing_thread
        self.assertEqual(thread.action, "texture")
        self.assertEqual(thread.mesh_request.model_glb, model.glb_bytes)
        self.assertEqual(
            thread.mesh_request.object_id, f"landing:{self.landing.landing_id}"
        )
        self.assertEqual(len(thread.mesh_request.reference_images_bgra), 1)
        np.testing.assert_array_equal(
            thread.mesh_request.reference_image_bgra,
            _selected_reference().image_bgra,
        )
        thread.run()
        self.assertIsNone(thread.error_message)
        self.assertTrue(thread.result.texture_only)
        self.assertEqual(requested[0].model_glb, model.glb_bytes)
        self._finish(thread)
        self.assertEqual(self.workspace.get_data().generated_objects, [])

    def test_texture_preserves_generated_landing_uvs_and_does_not_rotate_source_again(
        self,
    ):
        mesh = trimesh.creation.box(extents=(2.0, 0.2, 1.0))
        mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.asarray(mesh.vertices[:, (0, 2)]) / 2 + 0.5,
            image=Image.new("RGB", (2048, 2048), "tan"),
        )
        result = MeshyGenerationResult(
            "textured-landing", bytes(trimesh.Scene(mesh).export(file_type="glb"))
        )
        model = import_generated_glb(result.glb_bytes)
        self.workspace.update_landing_editing_preview(self.landing, model)
        requests = []

        def retexture(request):
            requests.append(request)
            return result

        self.workspace._meshy_texture_regenerator = retexture
        with patch.object(LandingGenerationThread, "start"):
            self.assertTrue(self.workspace.generate_selected_object_texture())
        thread = self.workspace._landing_thread
        thread.run()
        self.assertIsNone(thread.error_message)
        self.assertTrue(requests[0].enable_original_uv)
        self.assertIsNotNone(requests[0].submitted_uv_fingerprint)
        self.assertEqual(requests[0].model_glb, model.glb_bytes)
        np.testing.assert_allclose(thread.result.model.mesh.bounds, model.mesh.bounds)
        self._finish(thread)

    def test_cancelled_geometry_late_result_is_discarded(self):
        completed = []
        self.workspace.landing_generation_completed.connect(
            lambda *values: completed.append(values)
        )
        thread = self._prepare_completed_geometry()
        self.workspace.cancel_landing_generation()
        self._finish(thread)
        self.assertEqual(completed, [])

    def test_different_selected_landing_discards_old_geometry_result(self):
        completed = []
        self.workspace.landing_generation_completed.connect(
            lambda *values: completed.append(values)
        )
        thread = self._prepare_completed_geometry()
        self.workspace.set_landing_editing_target(
            replace(self.landing, landing_id="a" * 32)
        )
        self._finish(thread)
        self.assertEqual(completed, [])

    def test_changed_reference_discards_old_geometry_result(self):
        completed = []
        self.workspace.landing_generation_completed.connect(
            lambda *values: completed.append(values)
        )
        thread = self._prepare_completed_geometry()
        self.workspace.set_temporary_object_reference(_image(90))
        self._finish(thread)
        self.assertEqual(completed, [])

    def test_cancel_running_meshy_worker_propagates_cancel_event(self):
        started = threading.Event()
        cancelled = threading.Event()

        class Planner:
            def plan_with_progress(self, _request, _progress, cancel_event):
                started.set()
                cancel_event.wait(3.0)
                if cancel_event.is_set():
                    cancelled.set()
                return _provider_result()

        self.workspace._meshy_planner = Planner()
        completed = []
        self.workspace.landing_generation_completed.connect(
            lambda *args: completed.append(args)
        )
        self.workspace.generate_geometry()
        self.assertTrue(started.wait(3.0))
        self.assertTrue(self.workspace.cancel_landing_generation())
        deadline = time.monotonic() + 3.0
        while self.workspace.is_landing_generating and time.monotonic() < deadline:
            _application.processEvents()
            QTest.qWait(1)
        self.assertTrue(cancelled.wait(1.0))
        self.assertFalse(self.workspace.is_landing_generating)
        self.assertEqual(completed, [])

    def test_normal_geometry_dispatch_returns_after_leaving_landing_mode(self):
        self.workspace.clear_landing_editing_target()
        with patch.object(
            self.workspace, "_build_generation_requests", return_value=()
        ) as normal:
            self.workspace.generate_geometry()
        normal.assert_called_once_with(geometry_only=True)

    def test_updated_landing_preview_preserves_reference_session_and_camera(self):
        updated = replace(
            self.landing,
            footprint_meters=tuple(
                tuple(point) for point in _shape()["footprint_meters"]
            ),
            thickness_meters=0.2,
        )
        level = LevelData(index=2, name="Ground")
        mesh = build_landing_mesh(level, updated, [level])
        scene = trimesh.Scene(mesh)
        preview = GeneratedModel(
            mesh=mesh, scene=scene, glb_bytes=bytes(scene.export(file_type="glb"))
        )
        references = self.workspace._landing_references
        source_signature = self.workspace._landing_reference_signature
        token = self.workspace._landing_generation_token
        mode_changes = []
        self.workspace.landing_editing_changed.connect(mode_changes.append)

        with patch.object(
            self.workspace.result_view,
            "set_model",
            wraps=self.workspace.result_view.set_model,
        ) as display:
            self.assertTrue(
                self.workspace.update_landing_editing_preview(updated, preview)
            )

        display.assert_called_once_with(preview, preserve_camera=True)
        self.assertEqual(self.workspace.get_landing_editing_target(), updated)
        self.assertIs(self.workspace.result_view.model, preview)
        self.assertIs(self.workspace._generated_model, preview)
        self.assertIs(self.workspace._landing_references, references)
        self.assertEqual(self.workspace._landing_reference_signature, source_signature)
        self.assertEqual(self.workspace._landing_generation_token, token)
        self.assertEqual(mode_changes, [])
        self.assertEqual(self.workspace.get_data().generated_objects, [])

    def test_preview_refresh_for_different_landing_is_ignored(self):
        other = replace(self.landing, landing_id="b" * 32)
        self.assertFalse(self.workspace.update_landing_editing_preview(other, None))
        self.assertEqual(self.workspace.get_landing_editing_target(), self.landing)

    def test_tab_refresh_and_inpaint_navigation_do_not_clear_landing_preview(self):
        model = import_generated_glb(_provider_result().glb_bytes)
        self.workspace.update_landing_editing_preview(self.landing, model)
        self.workspace.refresh_file_backed_previews()
        self.assertFalse(self.workspace._select_current_inpaint_generated_object())
        self.assertIs(self.workspace.result_view.model, model)
        self.assertEqual(self.workspace.get_landing_editing_target(), self.landing)
        self.assertIsNone(self.workspace.current_inpaint_reference_key())
        self.assertEqual(self.workspace.snapshot_inpaint_reference_inputs(), ())

    def test_real_background_threads_complete_selected_view_then_meshy_job(self):
        @contextmanager
        def session(*_args):
            yield lambda *_args, **_options: _image()

        def wait_for_worker():
            deadline = time.monotonic() + 3.0
            while self.workspace.is_landing_generating and time.monotonic() < deadline:
                _application.processEvents()
                QTest.qWait(1)
            self.assertFalse(self.workspace.is_landing_generating)

        self.workspace._landing_references = ()
        with patch("housemaker.landing_generation._reference_session", session):
            self.assertTrue(
                self.workspace.generate_landing_references(
                    (_selected_view(),), model="test"
                )
            )
            wait_for_worker()
        self.assertEqual(len(self.workspace._landing_references), 1)
        self.assertEqual(
            self.workspace._landing_references[0].view_id,
            _selected_view().preset_id,
        )
        completed = []
        self.workspace.landing_generation_completed.connect(
            lambda *values: completed.append(values)
        )
        self.workspace.generate_geometry()
        wait_for_worker()
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0][0], self.landing.landing_id)
        self.assertIsInstance(completed[0][1], LandingMeshGenerationResult)
        self.assertEqual(self.workspace._landing_runtimes, {})

    def test_reference_api_rejects_multiple_views_without_launching_provider(self):
        with patch.object(LandingGenerationThread, "start") as start:
            self.assertFalse(
                self.workspace.generate_landing_references(_views(), model="test")
            )
        start.assert_not_called()


# ### Merged reference controls ###
class LandingMergedReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        directory = Path(self.temporary.name)
        self.manager = GenerationJobManager()
        self.objects = GenerationWorkspace(
            asset_directory=directory / "objects", job_manager=self.manager
        )
        self.surfaces = SurfaceTextureGenerationWorkspace(
            asset_directory=directory / "surfaces",
            job_manager=self.manager,
            shared_controls=self.objects.get_shared_controls(),
        )
        self.workspace = MergedGenerationWorkspace(
            self.surfaces, self.objects, self.manager
        )
        self.workspace.set_reference_view_presets(_views())
        self.objects.set_landing_editing_target(_landing())
        self.objects.set_temporary_object_reference(_image())
        self.workspace.reference_generation_view_combo.setCurrentIndex(0)
        self.workspace.reference_generation_prompt_edit.setText(_views()[0].prompt)

    def tearDown(self):
        self.workspace.shutdown()
        self.surfaces.shutdown()
        self.objects.shutdown()
        self.workspace.close()
        _application.processEvents()
        self.temporary.cleanup()

    def _generate_selected_reference(self):
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(_selected_view().preset_id))
        with patch.object(LandingGenerationThread, "start"):
            self.workspace._handle_generate_reference_clicked()
        thread = self.objects._landing_thread
        self.assertIsNotNone(thread)

        @contextmanager
        def session(*_args):
            yield lambda *_args, **_options: _selected_reference().image_bgra

        thread._reference_session = session
        thread.run()
        self.assertIsNone(thread.error_message)
        with patch.object(self.objects, "sender", return_value=thread):
            self.objects._handle_landing_generation_finished()
        return thread

    def test_reference_button_dispatches_only_nonfirst_selected_view_and_visible_prompt(
        self,
    ):
        self.assertFalse(self.workspace.landing_mode_label.isHidden())
        self.assertFalse(self.objects.regenerate_texture_button.isEnabled())
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(_selected_view().preset_id))
        self.assertEqual(
            self.workspace.reference_generation_prompt_edit.text(),
            _selected_view().prompt,
        )
        prompt = "Use view 5, preserve the landing's irregular perimeter"
        self.workspace.reference_generation_prompt_edit.setText(prompt)
        with patch.object(LandingGenerationThread, "start"):
            self.workspace._handle_generate_reference_clicked()
        thread = self.objects._landing_thread
        self.assertIsNotNone(thread)
        self.assertEqual(len(thread.views), 1)
        self.assertEqual(thread.views[0].preset_id, _selected_view().preset_id)
        effective_prompt = ", ".join(
            part for part in (thread.views[0].prompt, thread.prompt) if part
        )
        self.assertEqual(effective_prompt, prompt)
        self.assertEqual(len(self.manager.jobs()), 1)
        self.assertEqual(self.workspace.generate_reference_button.text(), "Cancel")

    def test_custom_reference_prompt_generates_one_reference_without_a_preset(self):
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(""))
        prompt = "Show the complete landing from a low three-quarter angle"
        self.workspace.reference_generation_prompt_edit.setText(prompt)
        self.assertTrue(self.workspace.generate_reference_button.isEnabled())
        with patch.object(LandingGenerationThread, "start"):
            self.workspace._handle_generate_reference_clicked()
        thread = self.objects._landing_thread
        self.assertIsNotNone(thread)
        self.assertEqual(len(thread.views), 1)
        effective_prompt = ", ".join(
            part for part in (thread.views[0].prompt, thread.prompt) if part
        )
        self.assertEqual(effective_prompt, prompt)

    def test_custom_reference_prompt_must_not_be_empty(self):
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(""))
        self.workspace.reference_generation_prompt_edit.clear()
        self.assertFalse(self.workspace.generate_reference_button.isEnabled())
        with patch.object(LandingGenerationThread, "start") as start:
            self.workspace._handle_generate_reference_clicked()
        start.assert_not_called()

    def test_custom_reference_generation_does_not_require_configured_views(self):
        self.workspace.set_reference_view_presets(())
        self.workspace.reference_generation_prompt_edit.setText(
            "Recreate this landing from the same angle"
        )
        self.assertTrue(self.workspace.generate_reference_button.isEnabled())
        with patch.object(LandingGenerationThread, "start"):
            self.workspace._handle_generate_reference_clicked()
        thread = self.objects._landing_thread
        self.assertIsNotNone(thread)
        self.assertEqual(len(thread.views), 1)

    def test_leaving_landing_mode_restores_regular_reference_controls(self):
        self.workspace._sync_reference_generation_controls()
        self.assertEqual(
            self.workspace.generate_reference_button.text(),
            "Generate reference",
        )

        self.objects.clear_landing_editing_target()

        self.assertEqual(
            self.workspace.generate_reference_button.text(), "Generate reference"
        )
        self.assertFalse(self.workspace.generate_all_references_button.isHidden())

    def test_changing_selected_view_requires_new_reference_before_meshy_generation(self):
        self._generate_selected_reference()
        self.assertTrue(self.objects.generate_geometry_button.isEnabled())
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(_views()[2].preset_id))
        self.assertEqual(self.objects._landing_references, ())
        self.assertFalse(self.objects.generate_geometry_button.isEnabled())
        self.assertFalse(self.objects.generate_button.isEnabled())

    def test_changed_selected_view_discards_late_reference_result(self):
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(_selected_view().preset_id))
        with patch.object(LandingGenerationThread, "start"):
            self.workspace._handle_generate_reference_clicked()
        thread = self.objects._landing_thread
        self.assertIsNotNone(thread)

        @contextmanager
        def session(*_args):
            yield lambda *_args, **_options: _selected_reference().image_bgra

        thread._reference_session = session
        thread.run()
        self.assertEqual(len(thread.result), 1)
        combo.setCurrentIndex(combo.findData(_views()[2].preset_id))
        with patch.object(self.objects, "sender", return_value=thread):
            self.objects._handle_landing_generation_finished()
        self.assertEqual(self.objects._landing_references, ())
        self.assertFalse(self.objects.generate_geometry_button.isEnabled())

    def test_meshy_uses_only_selected_generated_reference(self):
        self._generate_selected_reference()
        self.objects.update_landing_editing_preview(
            self.objects.get_landing_editing_target(),
            import_generated_glb(_provider_result().glb_bytes),
        )
        for action in ("geometry", "generate", "texture"):
            with self.subTest(action=action):
                request = self.objects._build_landing_mesh_request(action)
                self.assertIsNotNone(request)
                self.assertEqual(len(request.reference_images_bgra), 1)
                source = (
                    request.reference_image_bgra
                    if action == "texture"
                    else request.selected_object_bgra
                )
                np.testing.assert_array_equal(source, _selected_reference().image_bgra)
                np.testing.assert_array_equal(
                    request.reference_images_bgra[0],
                    _selected_reference().image_bgra,
                )

    def test_adding_unrelated_view_preserves_selected_landing_reference(self):
        self._generate_selected_reference()
        references = self.objects._landing_references
        self.workspace.set_reference_view_presets(
            (*_views(), ReferenceViewPreset("extra", "Extra view", "Use an extra view"))
        )
        self.assertIs(self.objects._landing_references, references)
        self.assertTrue(self.objects.generate_geometry_button.isEnabled())

    def test_editing_selected_view_updates_prompt_and_invalidates_old_reference(self):
        self._generate_selected_reference()
        updated_views = tuple(
            replace(view, prompt="Use a newly defined view 5")
            if view.preset_id == _selected_view().preset_id
            else view
            for view in _views()
        )
        self.workspace.set_reference_view_presets(updated_views)
        self.assertEqual(
            self.workspace.reference_generation_prompt_edit.text(),
            "Use a newly defined view 5",
        )
        self.assertEqual(self.objects._landing_references, ())
        self.assertFalse(self.objects.generate_geometry_button.isEnabled())


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
