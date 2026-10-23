"""Geometry provider contracts, with all HTTP requests mocked locally."""

# ### Imports ###
from __future__ import annotations

import base64
import io
import json
import threading
from urllib.error import HTTPError

import pytest

from housemaker.geometry_generation_models import (
    DEFAULT_GEOMETRY_MODEL,
    DEFAULT_GEOMETRY_RESOLUTION,
    GEOMETRY_MODEL_OPTIONS,
    GEOMETRY_RESOLUTION_OPTIONS,
    normalize_geometry_model,
    normalize_geometry_resolution,
)
from housemaker.meshy_generation import (
    MESHY_IMAGE_TO_3D_ENDPOINT,
    MeshyRequestError,
    MeshyTaskError,
    build_image_to_3d_request_body,
    create_image_to_3d_task,
    get_image_to_3d_task,
    request_image_to_3d_model,
)


# ### Transport fixtures ###
class _Response(io.BytesIO):
    pass


class _Opener:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, timeout: float):
        self.requests.append(request)
        assert timeout > 0
        assert self.responses, "Unexpected network request"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _Response(
            response if isinstance(response, bytes) else json.dumps(response).encode()
        )


def _successful_task():
    return {
        "status": "SUCCEEDED",
        "progress": 100,
        "model_urls": {"glb": "https://assets.meshy.ai/test.glb"},
    }


# ### Model and payload contracts ###
def test_geometry_model_choices_preserve_smart_topology_default():
    assert GEOMETRY_MODEL_OPTIONS == (
        ("Meshy t2", "meshy-t2"),
        ("Meshy t2.5", "meshy-t2.5"),
        ("Meshy 7.1", "meshy-7.1"),
    )
    assert DEFAULT_GEOMETRY_MODEL == "meshy-t2.5"
    assert DEFAULT_GEOMETRY_RESOLUTION == "standard"
    assert GEOMETRY_RESOLUTION_OPTIONS == (("Standard", "standard"), ("Ultra 2K", "2k"))


@pytest.mark.parametrize("value", [None, "latest", "meshy-7", "not-a-model", True])
def test_geometry_model_validation_rejects_unknown_or_changing_aliases(value):
    with pytest.raises(ValueError, match="geometry model"):
        normalize_geometry_model(value)


@pytest.mark.parametrize("value", [None, "4k", "8k", True])
def test_geometry_resolution_excludes_unrequested_options(value):
    with pytest.raises(ValueError, match="Standard or Ultra 2K"):
        normalize_geometry_resolution(value)


@pytest.mark.parametrize(
    "model, maximum", [("meshy-t2", 15_000), ("meshy-t2.5", 30_000)]
)
def test_smart_topology_models_have_model_specific_face_budgets(model, maximum):
    for value in (100, maximum):
        body = build_image_to_3d_request_body(
            b"front", ai_model=model, target_polycount=value
        )
        assert body["model_type"] == "smart-topology"
        assert body["ai_model"] == model
        assert body["target_polycount"] == value
        assert "should_remesh" not in body
        assert "geometry_resolution" not in body
    with pytest.raises(ValueError, match="target polycount"):
        build_image_to_3d_request_body(
            b"front", ai_model=model, target_polycount=maximum + 1
        )


@pytest.mark.parametrize("resolution", ["standard", "2k"])
def test_meshy_71_uses_native_geometry_resolution_without_triangle_budget(resolution):
    body = build_image_to_3d_request_body(
        b"front",
        ai_model="meshy-7.1",
        geometry_resolution=resolution,
        target_polycount=99_999,
    )
    assert body["ai_model"] == "meshy-7.1"
    assert body["model_type"] == "standard"
    assert body["geometry_resolution"] == resolution
    assert body["should_remesh"] is False
    assert "target_polycount" not in body
    assert base64.b64decode(body["image_url"].split(",", 1)[1]) == b"front"
    assert "image_urls" not in body


@pytest.mark.parametrize("model", ["meshy-t2", "meshy-t2.5"])
def test_smart_topology_does_not_silently_ignore_native_resolution(model):
    with pytest.raises(ValueError, match="Ultra 2K geometry requires Meshy 7.1"):
        build_image_to_3d_request_body(
            b"front", ai_model=model, geometry_resolution="2k"
        )


@pytest.mark.parametrize(
    "request_function, arguments",
    (
        (build_image_to_3d_request_body, (b"front",)),
        (create_image_to_3d_task, ("test-key", b"front")),
        (request_image_to_3d_model, ("test-key", b"front")),
    ),
)
def test_geometry_api_no_longer_accepts_multiple_reference_images(request_function, arguments):
    with pytest.raises(TypeError, match="reference_images_png"):
        request_function(*arguments, reference_images_png=(b"front", b"back"))


# ### Endpoint and lifecycle contracts ###
@pytest.mark.parametrize(
    "model, resolution",
    (("meshy-t2", "standard"), ("meshy-t2.5", "standard"),
     ("meshy-7.1", "standard"), ("meshy-7.1", "2k")),
)
def test_single_image_model_request_posts_and_polls_matching_endpoint(model, resolution):
    opener = _Opener({"result": "task-1"}, _successful_task(), b"glb fixture")
    result = request_image_to_3d_model(
        "secret",
        b"front",
        ai_model=model,
        should_texture=False,
        geometry_resolution=resolution,
        opener=opener,
        max_polls=1,
    )
    endpoint = MESHY_IMAGE_TO_3D_ENDPOINT
    assert result.task_id == "task-1"
    assert result.glb_bytes == b"glb fixture"
    assert [request.full_url for request in opener.requests] == [
        endpoint,
        endpoint + "/task-1",
        "https://assets.meshy.ai/test.glb",
    ]
    assert [request.method for request in opener.requests] == ["POST", "GET", "GET"]
    assert opener.requests[0].get_header("Authorization") == "Bearer secret"
    assert opener.requests[2].get_header("Authorization") is None
    body = json.loads(opener.requests[0].data)
    assert body["ai_model"] == model
    assert base64.b64decode(body["image_url"].split(",", 1)[1]) == b"front"
    assert "image_urls" not in body
    assert body["should_texture"] is False
    assert "enable_pbr" not in body
    assert "texture_resolution" not in body


def test_single_image_get_encodes_task_identifier_without_changing_endpoint():
    opener = _Opener({"status": "PENDING"})
    get_image_to_3d_task("secret", "task/with spaces", opener=opener)
    assert (
        opener.requests[0].full_url
        == MESHY_IMAGE_TO_3D_ENDPOINT + "/task%2Fwith%20spaces"
    )


@pytest.mark.parametrize("model", ("meshy-t2", "meshy-t2.5", "meshy-7.1"))
def test_canceled_before_submit_never_makes_paid_geometry_request(model):
    canceled = threading.Event()
    canceled.set()
    opener = _Opener()
    with pytest.raises(MeshyTaskError, match="before task submission"):
        request_image_to_3d_model(
            "secret",
            b"front",
            ai_model=model,
            cancel_event=canceled,
            opener=opener,
        )
    assert not opener.requests


def test_cancel_during_single_image_poll_prevents_further_poll_and_download():
    canceled = threading.Event()
    opener = _Opener({"result": "task-1"}, {"status": "IN_PROGRESS", "progress": 25})
    with pytest.raises(MeshyTaskError, match="canceled locally"):
        request_image_to_3d_model(
            "secret",
            b"front",
            ai_model="meshy-7.1",
            cancel_event=canceled,
            opener=opener,
            progress_callback=lambda *_args: canceled.set(),
            sleep=lambda _seconds: None,
        )
    assert len(opener.requests) == 2
    assert opener.requests[-1].full_url == MESHY_IMAGE_TO_3D_ENDPOINT + "/task-1"


def test_single_image_http_failures_redact_secret():
    error = HTTPError(
        MESHY_IMAGE_TO_3D_ENDPOINT,
        429,
        "error",
        None,
        io.BytesIO(json.dumps({"message": "secret was rate limited"}).encode()),
    )
    with pytest.raises(MeshyRequestError) as raised:
        request_image_to_3d_model(
            "secret",
            b"front",
            ai_model="meshy-7.1",
            opener=_Opener(error),
        )
    assert raised.value.retryable
    assert raised.value.status_code == 429
    assert "secret" not in str(raised.value)
