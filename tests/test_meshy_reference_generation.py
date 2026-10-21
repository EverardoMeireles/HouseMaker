# ### Imports ###
from __future__ import annotations

import base64
import io
import json
from urllib.error import HTTPError, URLError

import pytest

import housemaker.meshy_reference_generation as meshy_reference
from housemaker.meshy_reference_generation import (
    MESHY_IMAGE_TO_IMAGE_ENDPOINT,
    MeshyReferenceGenerationCancelled,
    MeshyReferenceGenerationError,
    request_meshy_reference_image,
)

# ### Constants and test doubles ###
PNG_BYTES = b"\x89PNG\r\n\x1a\nprepared-reference"
API_KEY = "private-meshy-api-key"
IMAGE_URL = "https://assets.meshy.ai/reference.png?Expires=123&signature=private"


class FakeResponse:
    def __init__(self, payload: bytes, *, content_type: str = "", on_read=None):
        self.stream = io.BytesIO(payload)
        self.headers = {"Content-Type": content_type} if content_type else {}
        self.on_read = on_read

    def read(self, amount=-1):
        if self.on_read is not None:
            self.on_read()
        return self.stream.read(amount)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class SequentialOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.timeouts = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        if not self.responses:
            raise AssertionError("Unexpected additional Meshy request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict):
            response = json.dumps(response).encode()
        return FakeResponse(response) if isinstance(response, bytes) else response


# ### Fixtures ###
def success_responses():
    return [
        {"result": "image-task-id"},
        {"status": "SUCCEEDED", "progress": 100, "image_urls": [IMAGE_URL]},
        FakeResponse(PNG_BYTES, content_type="image/png"),
    ]


def generate(opener, **kwargs):
    return request_meshy_reference_image(
        PNG_BYTES,
        api_key=API_KEY,
        prompt="Show the complete object in front view.",
        opener=opener,
        poll_interval_seconds=0,
        **kwargs,
    )


# ### Request and polling tests ###
def test_single_reference_request_and_download_do_not_create_a_3d_model():
    opener = SequentialOpener(success_responses())
    assert generate(opener) == PNG_BYTES
    create, poll, download = opener.requests
    assert create.full_url == MESHY_IMAGE_TO_IMAGE_ENDPOINT
    assert create.get_method() == "POST"
    payload = json.loads(create.data)
    assert payload == {
        "ai_model": "nano-banana",
        "prompt": "Show the complete object in front view.",
        "reference_image_urls": [
            "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()
        ],
        "generate_multi_view": False,
        "aspect_ratio": "1:1",
        "remove_background": False,
    }
    assert poll.full_url == MESHY_IMAGE_TO_IMAGE_ENDPOINT + "/image-task-id"
    assert poll.get_method() == "GET"
    assert create.get_header("Authorization") == f"Bearer {API_KEY}"
    assert poll.get_header("Authorization") == f"Bearer {API_KEY}"
    assert download.full_url == IMAGE_URL
    assert download.get_header("Authorization") is None
    assert opener.timeouts == [30.0, 30.0, 30.0]


def test_reports_all_poll_states_and_escapes_opaque_task_id():
    opener = SequentialOpener(
        [
            {"result": "image / task"},
            {"status": "PENDING", "progress": None},
            {"status": "IN_PROGRESS", "progress": 54},
            {"status": "SUCCEEDED", "progress": 120, "image_urls": [IMAGE_URL]},
            PNG_BYTES,
        ]
    )
    progress = []
    assert (
        generate(opener, progress_callback=lambda *args: progress.append(args))
        == PNG_BYTES
    )
    assert progress == [("PENDING", 0), ("IN_PROGRESS", 54), ("SUCCEEDED", 100)]
    assert opener.requests[1].full_url.endswith("/image%20%2F%20task")


def test_timeout_is_bounded_and_does_not_create_again():
    opener = SequentialOpener(
        [{"result": "task"}, {"status": "PENDING"}, {"status": "IN_PROGRESS"}]
    )
    with pytest.raises(MeshyReferenceGenerationError, match="timed out"):
        generate(opener, max_polls=2)
    assert [request.get_method() for request in opener.requests] == [
        "POST",
        "GET",
        "GET",
    ]


@pytest.mark.parametrize("status", ["FAILED", "CANCELED"])
def test_provider_failure_redacts_key_and_signed_urls(status):
    opener = SequentialOpener(
        [
            {"result": "task"},
            {
                "status": status,
                "task_error": {
                    "message": f"Unable to edit {IMAGE_URL} using {API_KEY}"
                },
            },
        ]
    )
    with pytest.raises(MeshyReferenceGenerationError) as captured:
        generate(opener)
    assert API_KEY not in str(captured.value)
    assert "signature=private" not in str(captured.value)
    assert "[redacted" in str(captured.value)


@pytest.mark.parametrize("payload", [{}, {"result": None}, {"result": ""}])
def test_missing_task_id_is_rejected_without_polling(payload):
    opener = SequentialOpener([payload])
    with pytest.raises(MeshyReferenceGenerationError, match="identifier"):
        generate(opener)
    assert len(opener.requests) == 1


@pytest.mark.parametrize("status", [None, "BROKEN", 10])
def test_unknown_task_state_is_rejected(status):
    opener = SequentialOpener([{"result": "task"}, {"status": status}])
    with pytest.raises(
        MeshyReferenceGenerationError, match="invalid image task status"
    ):
        generate(opener)


# ### Validation tests ###
@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_key": ""},
        {"api_key": None},
        {"api_key": "secret\r\nInjected: header"},
        {"image_png": b""},
        {"image_png": b"not a PNG"},
        {"image_png": "image bytes"},
        {"prompt": " "},
        {"prompt": None},
        {"output_size": (1024, 512)},
        {"output_size": (False, False)},
        {"output_size": (0, 0)},
        {"max_polls": 0},
        {"max_polls": True},
        {"max_polls": 1.5},
        {"poll_interval_seconds": float("inf")},
        {"poll_interval_seconds": -1},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("nan")},
    ],
)
def test_invalid_inputs_never_launch_a_paid_task(kwargs):
    opener = SequentialOpener([])
    arguments = {
        "image_png": PNG_BYTES,
        "api_key": API_KEY,
        "prompt": "Reference prompt",
        "opener": opener,
        **kwargs,
    }
    with pytest.raises((ValueError, TypeError)):
        request_meshy_reference_image(**arguments)
    assert not opener.requests


def test_oversized_source_is_rejected_before_a_paid_task(monkeypatch):
    monkeypatch.setattr(
        meshy_reference, "MAX_REFERENCE_IMAGE_BYTES", len(PNG_BYTES) - 1
    )
    opener = SequentialOpener([])
    with pytest.raises(ValueError, match="20 MB"):
        generate(opener)
    assert not opener.requests


# ### Download and transport error tests ###
@pytest.mark.parametrize(
    "image_urls",
    [
        None,
        [],
        [IMAGE_URL, IMAGE_URL],
        ["http://assets.meshy.ai/image.png"],
        ["file:///private.png"],
        ["https://user:pass@assets.meshy.ai/image.png"],
        ["https://"],
    ],
)
def test_invalid_image_url_never_downloads(image_urls):
    opener = SequentialOpener(
        [{"result": "task"}, {"status": "SUCCEEDED", "image_urls": image_urls}]
    )
    with pytest.raises(MeshyReferenceGenerationError):
        generate(opener)
    assert len(opener.requests) == 2


@pytest.mark.parametrize("payload", [b"not json", b"[]"])
def test_invalid_json_response_has_safe_error(payload):
    opener = SequentialOpener([payload])
    with pytest.raises(MeshyReferenceGenerationError, match="invalid"):
        generate(opener)


def test_http_error_is_not_retried_and_redacts_sensitive_detail():
    error = HTTPError(
        MESHY_IMAGE_TO_IMAGE_ENDPOINT,
        402,
        "Payment Required",
        {},
        io.BytesIO(
            json.dumps({"message": f"No credits for {API_KEY} {IMAGE_URL}"}).encode()
        ),
    )
    opener = SequentialOpener([error])
    with pytest.raises(MeshyReferenceGenerationError) as captured:
        generate(opener)
    assert "402" in str(captured.value)
    assert API_KEY not in str(captured.value)
    assert "signature=private" not in str(captured.value)
    assert len(opener.requests) == 1


def test_download_failure_redacts_the_signed_url_and_key():
    responses = success_responses()
    responses[-1] = URLError(f"Cannot reach {IMAGE_URL} {API_KEY}")
    with pytest.raises(MeshyReferenceGenerationError) as captured:
        generate(SequentialOpener(responses))
    assert "download failed" in str(captured.value)
    assert API_KEY not in str(captured.value)
    assert "signature=private" not in str(captured.value)


@pytest.mark.parametrize(
    "payload", [b"", FakeResponse(b"HTML", content_type="text/html")]
)
def test_empty_and_nonimage_downloads_are_rejected(payload):
    responses = success_responses()
    responses[-1] = payload
    with pytest.raises(MeshyReferenceGenerationError):
        generate(SequentialOpener(responses))


def test_download_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(meshy_reference, "MAX_IMAGE_DOWNLOAD_BYTES", 10)
    with pytest.raises(MeshyReferenceGenerationError, match="oversized"):
        generate(SequentialOpener(success_responses()))


def test_download_rejects_insecure_redirect():
    response = FakeResponse(PNG_BYTES)
    response.geturl = lambda: "http://assets.meshy.ai/reference.png"
    responses = success_responses()
    responses[-1] = response
    with pytest.raises(MeshyReferenceGenerationError, match="unsafe URL"):
        generate(SequentialOpener(responses))


# ### Cancellation tests ###
def test_cancelled_before_create_never_makes_a_paid_request():
    opener = SequentialOpener([])
    with pytest.raises(MeshyReferenceGenerationCancelled):
        generate(opener, cancellation_check=lambda: True)
    assert not opener.requests


def test_cancellation_after_create_does_not_poll_or_delete_remote_assets():
    opener = SequentialOpener([{"result": "task"}])
    with pytest.raises(MeshyReferenceGenerationCancelled):
        generate(opener, cancellation_check=lambda: bool(opener.requests))
    assert [request.get_method() for request in opener.requests] == ["POST"]


def test_cancellation_after_poll_prevents_download():
    opener = SequentialOpener(success_responses())
    with pytest.raises(MeshyReferenceGenerationCancelled):
        generate(opener, cancellation_check=lambda: len(opener.requests) >= 2)
    assert len(opener.requests) == 2


def test_cancellation_during_download_is_checked_after_each_read():
    state = {"cancelled": False}
    responses = success_responses()
    responses[-1] = FakeResponse(
        PNG_BYTES, on_read=lambda: state.update(cancelled=True)
    )
    with pytest.raises(MeshyReferenceGenerationCancelled):
        generate(
            SequentialOpener(responses), cancellation_check=lambda: state["cancelled"]
        )


def test_poll_sleep_checks_cancellation_in_small_intervals():
    state = {"cancelled": False}
    waits = []
    opener = SequentialOpener([{"result": "task"}, {"status": "PENDING"}])

    def sleep(seconds):
        waits.append(seconds)
        state["cancelled"] = True

    with pytest.raises(MeshyReferenceGenerationCancelled):
        request_meshy_reference_image(
            PNG_BYTES,
            api_key=API_KEY,
            prompt="Reference prompt",
            opener=opener,
            sleep=sleep,
            poll_interval_seconds=5,
            cancellation_check=lambda: state["cancelled"],
        )
    assert waits == [0.1]
    assert len(opener.requests) == 2
