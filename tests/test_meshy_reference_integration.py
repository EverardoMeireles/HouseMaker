"""Meshy reference choices, isolated images, and provider-key routing."""

# ### Environment setup ###
from __future__ import annotations

import gc
import os
from io import BytesIO
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import cv2
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from housemaker import reference_generation as reference
from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_jobs import GenerationJobManager
from housemaker.generation_state import MASK_MODE_PAINT, MaskPoint, MaskStroke
from housemaker.generation_workspace import GenerationWorkspace
from housemaker.landing_generation import LandingGenerationThread
from housemaker.merged_generation_workspace import (
    MergedGenerationWorkspace,
    _ReferenceGenerationThread,
)
from housemaker.meshy_reference_generation import (
    MeshyReferenceGenerationCancelled,
    MeshyReferenceGenerationError,
)
from housemaker.models import LandingData
from housemaker.openai_batch import LUNA_BATCH_MODEL
from housemaker.reference_generation_batch import APIReferenceBatchThread
from housemaker.reference_generation_models import (
    REFERENCE_GENERATION_MODEL_MESHY,
    REFERENCE_GENERATION_MODEL_OPTIONS,
    REFERENCE_GENERATION_MODEL_SETTINGS_KEY,
    REFERENCE_GENERATION_MODELS,
)
from housemaker.reference_symmetry_inference import (
    SYMMETRY_MODEL_LUNA,
    SymmetryInferenceResult,
)
from housemaker.settings_widget import (
    GenerationServiceSettings,
    SettingsWidget,
    read_reference_generation_model,
)
from housemaker.surface_texture_workspace import SurfaceTextureGenerationWorkspace

# ### Image and application fixtures ###
APP = QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)


def _masked_source():
    source = np.zeros((24, 48, 4), dtype=np.uint8)
    source[:, :, :3] = (250, 10, 90)
    source[6:18, 12:36] = (30, 70, 190, 255)
    return source


def _opaque_reference(value=110):
    result = np.full((32, 48, 4), value, dtype=np.uint8)
    result[:, :, 3] = 255
    return result


def _png(size=(1024, 1024)):
    output = BytesIO()
    Image.new("RGBA", size, (30, 80, 190, 255)).save(output, format="PNG")
    return output.getvalue()


def _dispose(*widgets):
    """Delete this fixture's ownership trees without flushing unrelated objects."""

    for widget in widgets:
        widget.close()
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(widget, QEvent.Type.DeferredDelete)


@pytest.fixture(autouse=True)
def collect_qt_wrappers_outside_native_construction():
    """Avoid PySide collecting old ownership cycles inside QObject.__init__."""

    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        if enabled:
            gc.enable()


@pytest.fixture
def merged_workspace(tmp_path):
    store = ApplicationSettingsStore(tmp_path / "settings.json")
    manager = GenerationJobManager()
    objects = GenerationWorkspace(
        asset_directory=tmp_path / "objects",
        job_manager=manager,
    )
    surfaces = SurfaceTextureGenerationWorkspace(
        asset_directory=tmp_path / "surfaces",
        application_settings=store,
        job_manager=manager,
        shared_controls=objects.get_shared_controls(),
    )
    workspace = MergedGenerationWorkspace(
        surfaces,
        objects,
        manager,
        application_settings=store,
    )
    objects.set_runtime_settings(
        GenerationServiceSettings(
            meshy_api_key="  mesh-provider-key  ",
            openai_api_key="  openai-analysis-key  ",
        )
    )
    workspace.set_reference_generation_model(REFERENCE_GENERATION_MODEL_MESHY)
    video_path = tmp_path / "references.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for value in (40, 80):
            writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    finally:
        writer.release()
    objects.load_video(str(video_path))
    surfaces.load_video(str(video_path))
    workspace.video_view.set_strokes(
        [
            MaskStroke(MASK_MODE_PAINT, 0.08, (MaskPoint(0.25, 0.5),)),
            MaskStroke(MASK_MODE_PAINT, 0.08, (MaskPoint(0.75, 0.5),)),
        ]
    )
    workspace.video_view.strokes_changed.emit(workspace.video_view.get_strokes())
    workspace.video_view.select_blob(0)
    workspace.sync_shared_controls()
    try:
        yield workspace, objects, manager
    finally:
        workspace.shutdown()
        objects.shutdown()
        surfaces.shutdown()
        _dispose(workspace)


# ### Settings choice and persistence ###
def test_meshy_is_a_reference_model_and_valid_service_setting():
    assert REFERENCE_GENERATION_MODEL_MESHY in REFERENCE_GENERATION_MODELS
    assert any(
        "Meshy" in label and model == REFERENCE_GENERATION_MODEL_MESHY
        for label, model in REFERENCE_GENERATION_MODEL_OPTIONS
    )
    settings = GenerationServiceSettings(
        reference_generation_model=REFERENCE_GENERATION_MODEL_MESHY,
    )
    assert settings.reference_generation_model == REFERENCE_GENERATION_MODEL_MESHY


def test_meshy_choice_is_available_and_persists_in_settings(tmp_path):
    store = ApplicationSettingsStore(tmp_path / "settings.json")
    widget = SettingsWidget(application_settings=store, environment={})
    reopened = None
    try:
        combo = widget.reference_generation_model_combo
        index = combo.findData(REFERENCE_GENERATION_MODEL_MESHY)
        assert index >= 0
        combo.setCurrentIndex(index)
        assert (
            widget.get_settings().reference_generation_model
            == REFERENCE_GENERATION_MODEL_MESHY
        )
        assert (
            store.get(REFERENCE_GENERATION_MODEL_SETTINGS_KEY)
            == REFERENCE_GENERATION_MODEL_MESHY
        )
        assert (
            read_reference_generation_model(store) == REFERENCE_GENERATION_MODEL_MESHY
        )
        reopened = SettingsWidget(
            application_settings=ApplicationSettingsStore(store.path),
            environment={},
        )
        assert (
            reopened.reference_generation_model_combo.currentData()
            == REFERENCE_GENERATION_MODEL_MESHY
        )
    finally:
        widget.dispose()
        if reopened is not None:
            reopened.dispose()
        _dispose(*(item for item in (widget, reopened) if item is not None))


# ### Standalone image preparation and provider isolation ###
def test_meshy_reference_receives_masked_png_prompt_and_normalizes_output():
    source = _masked_source()
    before = source.copy()
    progress = Mock()
    cancelled = Mock(return_value=False)
    with (
        patch(
            "housemaker.meshy_reference_generation.request_meshy_reference_image",
            return_value=_png((800, 400)),
        ) as generate,
        patch.object(reference, "_create_reference_image_editor") as editor,
    ):
        result = reference.generate_object_reference(
            source,
            "Show the object from directly in front.",
            REFERENCE_GENERATION_MODEL_MESHY,
            api_key=" mesh-provider-key ",
            progress_callback=progress,
            cancellation_check=cancelled,
        )
    editor.assert_not_called()
    generate.assert_called_once()
    sent = generate.call_args
    assert sent.args[0] == reference.prepare_reference_generation_input(source)
    assert sent.kwargs["api_key"] == "mesh-provider-key"
    assert "Show the object from directly in front." in sent.kwargs["prompt"]
    assert "NEW standalone 1024x1024" in sent.kwargs["prompt"]
    assert sent.kwargs["output_size"] == (1024, 1024)
    assert sent.kwargs["cancellation_check"] is cancelled
    assert sent.kwargs["progress_callback"] is progress
    with Image.open(BytesIO(sent.args[0])) as prepared:
        assert prepared.size == (1024, 1024)
        assert prepared.getpixel((0, 0)) == (255, 255, 255)
        assert prepared.getpixel((512, 512)) == (190, 70, 30)
    assert result.shape == (1024, 1024, 4)
    assert np.all(result[256:768, :, 3] == 255)
    assert np.all(result[:256, :, 3] == 0)
    np.testing.assert_array_equal(source, before)
    assert progress.call_args.args == ("Reference ready", 100)


def test_meshy_reference_missing_api_key_does_not_start_a_provider():
    with (
        patch(
            "housemaker.meshy_reference_generation.request_meshy_reference_image"
        ) as generate,
        pytest.raises(reference.ReferenceGenerationError, match="Meshy API key"),
    ):
        reference.generate_object_reference(
            _masked_source(),
            "A front view",
            REFERENCE_GENERATION_MODEL_MESHY,
            api_key=" ",
        )
    generate.assert_not_called()


@pytest.mark.parametrize("payload", [b"not an image", b""])
def test_meshy_unusable_image_fails_without_using_another_provider(payload):
    with (
        patch(
            "housemaker.meshy_reference_generation.request_meshy_reference_image",
            return_value=payload,
        ) as generate,
        patch.object(reference, "_create_reference_image_editor") as fallback,
        pytest.raises(reference.ReferenceGenerationError),
    ):
        reference.generate_object_reference(
            _masked_source(),
            "A front view",
            REFERENCE_GENERATION_MODEL_MESHY,
            api_key="mesh-provider-key",
        )
    generate.assert_called_once()
    fallback.assert_not_called()


def test_cancelled_meshy_reference_does_not_start_a_provider():
    with (
        patch(
            "housemaker.meshy_reference_generation.request_meshy_reference_image"
        ) as generate,
        pytest.raises(reference.ReferenceGenerationCancelled),
    ):
        reference.generate_object_reference(
            _masked_source(),
            "A front view",
            REFERENCE_GENERATION_MODEL_MESHY,
            api_key="mesh-provider-key",
            cancellation_check=lambda: True,
        )
    generate.assert_not_called()


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (
            MeshyReferenceGenerationError("Meshy provider unavailable"),
            reference.ReferenceGenerationError,
        ),
        (
            MeshyReferenceGenerationCancelled("Meshy request cancelled"),
            reference.ReferenceGenerationCancelled,
        ),
    ],
)
def test_meshy_failure_or_cancellation_preserves_source_without_fallback(
    failure,
    expected,
):
    source = _masked_source()
    before = source.copy()
    with (
        patch(
            "housemaker.meshy_reference_generation.request_meshy_reference_image",
            side_effect=failure,
        ) as generate,
        patch.object(reference, "_create_reference_image_editor") as fallback,
        pytest.raises(expected, match=str(failure)),
    ):
        reference.generate_object_reference(
            source,
            "A front view",
            REFERENCE_GENERATION_MODEL_MESHY,
            api_key="mesh-provider-key",
        )
    generate.assert_called_once()
    fallback.assert_not_called()
    np.testing.assert_array_equal(source, before)


# ### UI dispatch and independent symmetry credentials ###
@pytest.mark.parametrize("with_luna_symmetry", [False, True])
def test_single_reference_dispatch_uses_meshy_key_and_luna_uses_openai_key(
    merged_workspace,
    with_luna_symmetry,
):
    workspace, _objects, _manager = merged_workspace
    workspace.set_object_symmetry_inference(with_luna_symmetry, SYMMETRY_MODEL_LUNA)
    generated = _opaque_reference()
    decision = SymmetryInferenceResult(symmetric=True, model=SYMMETRY_MODEL_LUNA)
    with patch.object(_ReferenceGenerationThread, "start"):
        workspace.generate_reference_button.click()
    runtime = workspace._reference_generation_runtime
    assert runtime is not None
    with (
        patch(
            "housemaker.reference_generation.generate_object_reference",
            return_value=generated,
        ) as generate,
        patch(
            "housemaker.reference_symmetry_inference.infer_reference_symmetry",
            return_value=decision,
        ) as infer,
    ):
        runtime.thread.run()
    assert runtime.thread.error_message is None
    assert generate.call_args.args[2] == REFERENCE_GENERATION_MODEL_MESHY
    assert generate.call_args.kwargs["api_key"] == "mesh-provider-key"
    if with_luna_symmetry:
        assert infer.call_args.args[0] is generated
        assert infer.call_args.kwargs["api_key"] == "openai-analysis-key"
    else:
        infer.assert_not_called()
    runtime.thread.finished.emit()
    assert workspace._reference_generation_runtime is None


@pytest.mark.parametrize("with_luna_symmetry", [False, True])
def test_generate_all_meshy_references_route_each_provider_key_independently(
    merged_workspace,
    with_luna_symmetry,
):
    workspace, _objects, manager = merged_workspace
    workspace.set_object_symmetry_inference(with_luna_symmetry, SYMMETRY_MODEL_LUNA)
    generated = _opaque_reference()
    decision = SymmetryInferenceResult(symmetric=False, model=SYMMETRY_MODEL_LUNA)
    with (
        patch.object(_ReferenceGenerationThread, "start"),
        patch(
            "housemaker.reference_generation.generate_object_reference",
            return_value=generated,
        ) as generate,
        patch(
            "housemaker.reference_symmetry_inference.infer_reference_symmetry",
            return_value=decision,
        ) as infer,
    ):
        workspace.generate_all_references_button.click()
        runtimes = tuple(workspace._batch_reference_generation_runtimes.values())
        assert len(runtimes) == 2
        for runtime in runtimes:
            runtime.thread.run()
            assert runtime.thread.error_message is None
            runtime.thread.finished.emit()
    assert generate.call_count == 2
    assert all(
        call.args[2] == REFERENCE_GENERATION_MODEL_MESHY
        for call in generate.call_args_list
    )
    assert all(
        call.kwargs["api_key"] == "mesh-provider-key"
        for call in generate.call_args_list
    )
    if with_luna_symmetry:
        assert infer.call_count == 2
        assert all(
            call.kwargs["api_key"] == "openai-analysis-key"
            for call in infer.call_args_list
        )
    else:
        infer.assert_not_called()
    assert all(job.status == "completed" for job in manager.jobs())


def test_landing_reference_dispatch_uses_meshy_key_for_only_selected_view(
    merged_workspace,
):
    workspace, objects, _manager = merged_workspace
    objects.set_landing_editing_target(LandingData(center_x=100.0, center_y=200.0))
    objects.set_temporary_object_reference(_opaque_reference())
    selected_id = workspace.reference_generation_view_combo.currentData()
    with patch.object(LandingGenerationThread, "start"):
        workspace.generate_reference_button.click()
    worker = objects._landing_thread
    assert worker is not None
    assert worker.model == REFERENCE_GENERATION_MODEL_MESHY
    assert worker._api_key == "mesh-provider-key"
    assert len(worker.views) == 1
    assert worker.views[0].preset_id == selected_id
    with patch(
        "housemaker.reference_generation.generate_object_reference",
        return_value=_opaque_reference(),
    ) as generate:
        worker.run()
    assert worker.error_message is None
    assert generate.call_args.kwargs["model"] == REFERENCE_GENERATION_MODEL_MESHY
    assert generate.call_args.kwargs["api_key"] == "mesh-provider-key"
    worker.finished.emit()
    assert objects._landing_thread is None
    assert not objects._landing_runtimes


def test_meshy_generate_all_with_luna_batch_symmetry_keeps_both_credentials(
    merged_workspace,
):
    workspace, objects, manager = merged_workspace
    workspace.set_object_symmetry_inference(True, LUNA_BATCH_MODEL)
    generated = _opaque_reference()
    decision = SymmetryInferenceResult(symmetric=True, model=LUNA_BATCH_MODEL)

    def decisions(images, **_kwargs):
        return {job_id: decision for job_id in images}

    with patch.object(APIReferenceBatchThread, "start"):
        workspace.generate_all_references_button.click()
    worker = workspace._local_reference_batch_thread
    assert isinstance(worker, APIReferenceBatchThread)
    assert len(worker.items) == 2
    with (
        patch(
            "housemaker.reference_generation.generate_object_reference",
            return_value=generated,
        ) as generate,
        patch(
            "housemaker.reference_symmetry_inference.infer_reference_symmetry_batch",
            side_effect=decisions,
        ) as infer,
    ):
        worker.run()
        assert worker.error_message is None
        worker.finished.emit()
    assert generate.call_count == 2
    assert all(
        call.kwargs["api_key"] == "mesh-provider-key"
        for call in generate.call_args_list
    )
    infer.assert_called_once()
    assert infer.call_args.kwargs["api_key"] == "openai-analysis-key"
    assert len(infer.call_args.args[0]) == 2
    assert all(image is generated for image in infer.call_args.args[0].values())
    assert all(job.status == "completed" for job in manager.jobs())
    assert all(
        objects.get_inpaint_symmetric_division(key) is True
        for key in objects._current_inpaint_keys()
    )
