"""Owned geometry choices reach both Meshy generation paths unchanged."""

# ### Environment and imports ###
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from housemaker.generation_workspace import (
    GenerationRequest,
    MeshyImagePlanner,
    StagedMeshyGenerationResult,
    _geometry_provider_request_options,
)
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.settings_widget import GenerationServiceSettings

# ### Qt-backed settings lifetime ###
_application = QApplication.instance() or QApplication([])


# ### Owned image fixtures ###
def _image(value=60):
    return np.full((8, 12, 4), (value, value + 1, value + 2, 255), np.uint8)


def _request(**options):
    return GenerationRequest(
        frame_index=0,
        selected_object_bgra=_image(),
        settings=GenerationServiceSettings(
            meshy_api_key="test-only",
            unused_face_removal=False,
            use_uv_raycast_for_object_generation=False,
        ),
        **options,
    )


# ### Request validation and thread-safe ownership ###
def test_default_request_preserves_existing_meshy_call_options():
    request = _request()
    assert request.geometry_model == "meshy-t2.5"
    assert request.geometry_resolution == "standard"
    assert _geometry_provider_request_options(request) == {}


def test_geometry_options_forward_native_model_and_resolution():
    request = _request(
        geometry_model="meshy-7.1",
        geometry_resolution="2k",
    )
    options = _geometry_provider_request_options(request)
    assert options == {"ai_model": "meshy-7.1", "geometry_resolution": "2k"}


@pytest.mark.parametrize(
    "options",
    (
        {"geometry_model": "latest"},
        {"geometry_model": "meshy-7.1", "geometry_resolution": "4k"},
        {"geometry_model": "meshy-t2", "geometry_resolution": "2k"},
    ),
)
def test_invalid_or_unsupported_geometry_options_fail_before_submission(options):
    with pytest.raises(ValueError):
        _request(**options)


# ### Direct and staged planners use the selected geometry model ###
@pytest.mark.parametrize("geometry_only", (False, True))
@pytest.mark.parametrize("model", ("meshy-t2", "meshy-t2.5", "meshy-7.1"))
def test_planner_forwards_model_for_both_generation_actions(model, geometry_only):
    options = {"geometry_model": model, "geometry_only": geometry_only}
    if model == "meshy-7.1":
        options.update(geometry_resolution="2k")
    request = _request(**options)
    result = MeshyGenerationResult("task-a", b"owned-glb")
    cleanup = SimpleNamespace(removed_face_count=0, duplicate_group_count=0)
    with (
        patch(
            "housemaker.generation_workspace.request_image_to_3d_model",
            return_value=result,
        ) as submit,
        patch(
            "housemaker.generation_workspace._remove_safe_duplicates_from_meshy_result",
            return_value=(result, cleanup),
        ),
        patch(
            "housemaker.generation_workspace._scan_project_provider_result",
            side_effect=lambda _request, result, *_args: result,
        ),
    ):
        output = MeshyImagePlanner().plan(request)
    assert output.task_id == "task-a"
    kwargs = submit.call_args.kwargs
    if model == "meshy-t2.5":
        assert "ai_model" not in kwargs
    else:
        assert kwargs["ai_model"] == model
    if model == "meshy-7.1":
        assert kwargs["geometry_resolution"] == "2k"
    assert "reference_images_png" not in kwargs
    if geometry_only:
        assert kwargs["should_texture"] is False
        assert isinstance(output, StagedMeshyGenerationResult)
