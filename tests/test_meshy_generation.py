# ### Imports ###
from __future__ import annotations

import base64
import io
import json
import unittest
from urllib.error import HTTPError, URLError

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

try:
    from PIL import Image as PillowImage
except ImportError:
    PillowImage = None

from housemaker.glb import GeneratedModel, import_generated_glb
from housemaker.meshy_generation import (
    DEFAULT_SMART_TOPOLOGY_TARGET_POLYCOUNT,
    MAX_SMART_TOPOLOGY_TARGET_POLYCOUNT,
    MESHY_IMAGE_TO_3D_ENDPOINT,
    MIN_SMART_TOPOLOGY_TARGET_POLYCOUNT,
    MeshyRequestError,
    MeshyTaskError,
    build_image_to_3d_request_body,
    create_image_to_3d_task,
    download_glb,
    get_image_to_3d_task,
    request_image_to_3d_model,
    wait_for_image_to_3d_task,
)
from housemaker.viewer import _build_texture_mesh_data


# ### Test doubles ###
class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._position = 0

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            amount = len(self._payload) - self._position
        start = self._position
        end = min(len(self._payload), start + amount)
        self._position = end
        return self._payload[start:end]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class SequentialOpener:
    def __init__(self, responses: list[bytes | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[object] = []
        self.timeouts: list[float] = []

    def __call__(self, request, timeout: float) -> FakeResponse:
        self.requests.append(request)
        self.timeouts.append(timeout)
        if not self._responses:
            raise AssertionError("The fake Meshy opener received an extra request")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeResponse(response)


# ### Fixture helpers ###
def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _valid_glb_bytes() -> bytes:
    mesh = trimesh.creation.box(extents=(1.0, 0.5, 0.25))
    vertex_colors = np.tile(
        np.asarray([215, 70, 35, 255], dtype=np.uint8),
        (len(mesh.vertices), 1),
    )
    vertex_colors[0] = np.asarray([20, 180, 90, 255], dtype=np.uint8)
    mesh.visual.vertex_colors = vertex_colors
    scene = trimesh.Scene(mesh)
    return bytes(scene.export(file_type="glb"))


def _embedded_texture_glb_bytes(texture_color: tuple[int, int, int]) -> bytes:
    if PillowImage is None:
        raise RuntimeError("Pillow is required to build the textured GLB fixture.")
    texture_rgba = np.empty((2, 2, 4), dtype=np.uint8)
    texture_rgba[:, :, :3] = np.asarray(texture_color, dtype=np.uint8)
    texture_rgba[:, :, 3] = 255
    texture = PillowImage.fromarray(texture_rgba, mode="RGBA")
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            [
                [-0.5, 0.0, -0.5],
                [0.5, 0.0, -0.5],
                [0.5, 0.0, 0.5],
                [-0.5, 0.0, 0.5],
            ],
            dtype=float,
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=np.asarray(
            [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],
            dtype=float,
        ),
        material=PBRMaterial(baseColorTexture=texture),
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _http_error(status_code: int, message: str) -> HTTPError:
    return HTTPError(
        url=MESHY_IMAGE_TO_3D_ENDPOINT,
        code=status_code,
        msg="Meshy request failed",
        hdrs=None,
        fp=io.BytesIO(_json_bytes({"message": message})),
    )


# ### Request construction tests ###
class MeshyRequestConstructionTests(unittest.TestCase):
    def test_request_body_embeds_png_as_data_uri_and_requests_glb(self) -> None:
        image_png = b"\x89PNG\r\n\x1a\nselected object"

        body = build_image_to_3d_request_body(image_png)

        image_url = body["image_url"]
        self.assertIsInstance(image_url, str)
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        encoded_image = image_url.removeprefix("data:image/png;base64,")
        self.assertEqual(base64.b64decode(encoded_image), image_png)
        self.assertIn("glb", body["target_formats"])

    def test_request_body_uses_compatible_smart_topology_t2_options(self) -> None:
        body = build_image_to_3d_request_body(b"png bytes")

        self.assertEqual(body["model_type"], "smart-topology")
        self.assertEqual(body["ai_model"], "meshy-t2")
        self.assertEqual(body["texture_resolution"], "2k")
        self.assertTrue(body["should_texture"])
        self.assertFalse(body["moderation"])
        self.assertEqual(
            body["target_polycount"],
            DEFAULT_SMART_TOPOLOGY_TARGET_POLYCOUNT,
        )
        for incompatible_option in (
            "should_remesh",
            "topology",
            "image_enhancement",
        ):
            self.assertNotIn(incompatible_option, body)

    def test_request_body_accepts_only_the_smart_topology_t2_polycount_range(
        self,
    ) -> None:
        body = build_image_to_3d_request_body(
            b"png bytes",
            target_polycount=12_345,
        )

        self.assertEqual(body["target_polycount"], 12_345)
        for invalid_polycount in (
            MIN_SMART_TOPOLOGY_TARGET_POLYCOUNT - 1,
            MAX_SMART_TOPOLOGY_TARGET_POLYCOUNT + 1,
            True,
            4_000.0,
        ):
            with self.subTest(target_polycount=invalid_polycount):
                with self.assertRaisesRegex(ValueError, "target polycount"):
                    build_image_to_3d_request_body(
                        b"png bytes",
                        target_polycount=invalid_polycount,
                    )

    def test_create_posts_json_with_bearer_auth_and_returns_task_id(self) -> None:
        opener = SequentialOpener([_json_bytes({"result": "task-123"})])

        task_id = create_image_to_3d_task(
            api_key="msy-test-secret",
            image_png=b"png bytes",
            timeout_seconds=8.5,
            opener=opener,
            target_polycount=6_789,
        )

        self.assertEqual(task_id, "task-123")
        self.assertEqual(opener.timeouts, [8.5])
        request = opener.requests[0]
        self.assertEqual(request.full_url, MESHY_IMAGE_TO_3D_ENDPOINT)
        self.assertEqual(request.method, "POST")
        self.assertEqual(
            request.get_header("Authorization"),
            "Bearer msy-test-secret",
        )
        self.assertEqual(request.get_header("Content-type"), "application/json")
        sent_body = json.loads(request.data.decode("utf-8"))
        self.assertTrue(sent_body["image_url"].startswith("data:image/png;base64,"))
        self.assertIn("glb", sent_body["target_formats"])
        self.assertEqual(sent_body["target_polycount"], 6_789)

    def test_get_task_uses_same_endpoint_and_bearer_auth(self) -> None:
        task_payload = {
            "id": "task-123",
            "status": "IN_PROGRESS",
            "progress": 37,
        }
        opener = SequentialOpener([_json_bytes(task_payload)])

        result = get_image_to_3d_task(
            api_key="msy-test-secret",
            task_id="task-123",
            timeout_seconds=6.0,
            opener=opener,
        )

        self.assertEqual(result, task_payload)
        request = opener.requests[0]
        self.assertEqual(
            request.full_url,
            f"{MESHY_IMAGE_TO_3D_ENDPOINT}/task-123",
        )
        self.assertEqual(request.method, "GET")
        self.assertEqual(
            request.get_header("Authorization"),
            "Bearer msy-test-secret",
        )


# ### Task polling tests ###
class MeshyTaskPollingTests(unittest.TestCase):
    def test_wait_reports_progress_and_returns_succeeded_task(self) -> None:
        responses = [
            _json_bytes(
                {"id": "task-123", "status": "PENDING", "progress": 0}
            ),
            _json_bytes(
                {
                    "id": "task-123",
                    "status": "IN_PROGRESS",
                    "progress": 54,
                }
            ),
            _json_bytes(
                {
                    "id": "task-123",
                    "status": "SUCCEEDED",
                    "progress": 100,
                    "model_urls": {"glb": "https://assets.meshy.ai/model.glb"},
                }
            ),
        ]
        opener = SequentialOpener(responses)
        sleeps: list[float] = []
        progress_updates: list[tuple[str, int]] = []

        result = wait_for_image_to_3d_task(
            api_key="msy-test-secret",
            task_id="task-123",
            poll_interval_seconds=2.5,
            max_polls=3,
            opener=opener,
            sleep=sleeps.append,
            progress_callback=lambda status, progress: progress_updates.append(
                (status, progress)
            ),
        )

        self.assertEqual(result["status"], "SUCCEEDED")
        self.assertEqual(
            result["model_urls"]["glb"],
            "https://assets.meshy.ai/model.glb",
        )
        self.assertEqual(
            progress_updates,
            [("PENDING", 0), ("IN_PROGRESS", 54), ("SUCCEEDED", 100)],
        )
        self.assertEqual(sleeps, [2.5, 2.5])

    def test_wait_raises_task_error_with_provider_failure_message(self) -> None:
        api_key = "msy-must-be-redacted"
        opener = SequentialOpener(
            [
                _json_bytes(
                    {
                        "id": "task-failed",
                        "status": "FAILED",
                        "progress": 31,
                        "task_error": {
                            "message": f"The source was rejected for {api_key}"
                        },
                    }
                )
            ]
        )

        with self.assertRaises(MeshyTaskError) as raised:
            wait_for_image_to_3d_task(
                api_key=api_key,
                task_id="task-failed",
                max_polls=1,
                opener=opener,
                sleep=lambda _seconds: None,
            )

        self.assertIn("source was rejected", str(raised.exception))
        self.assertIn("[redacted]", str(raised.exception))
        self.assertNotIn(api_key, str(raised.exception))

    def test_wait_is_bounded_when_task_never_finishes(self) -> None:
        opener = SequentialOpener(
            [
                _json_bytes(
                    {
                        "id": "task-slow",
                        "status": "IN_PROGRESS",
                        "progress": progress,
                    }
                )
                for progress in (1, 2, 3)
            ]
        )

        with self.assertRaisesRegex(MeshyTaskError, "timed out"):
            wait_for_image_to_3d_task(
                api_key="msy-test-secret",
                task_id="task-slow",
                max_polls=3,
                opener=opener,
                sleep=lambda _seconds: None,
            )

    def test_wait_rejects_unknown_status_and_invalid_progress(self) -> None:
        invalid_tasks = (
            {"id": "task-1", "status": "MYSTERY", "progress": 10},
            {"id": "task-2", "status": "IN_PROGRESS", "progress": 101},
        )

        for task in invalid_tasks:
            with self.subTest(task=task):
                with self.assertRaises(MeshyTaskError):
                    wait_for_image_to_3d_task(
                        api_key="msy-test-secret",
                        task_id=str(task["id"]),
                        max_polls=1,
                        opener=SequentialOpener([_json_bytes(task)]),
                        sleep=lambda _seconds: None,
                    )


# ### Artifact download tests ###
class MeshyArtifactDownloadTests(unittest.TestCase):
    def test_download_returns_glb_without_sending_meshy_authorization(self) -> None:
        glb_bytes = _valid_glb_bytes()
        opener = SequentialOpener([glb_bytes])

        result = download_glb(
            "https://assets.meshy.ai/signed/model.glb?Expires=123",
            max_bytes=len(glb_bytes) + 1,
            timeout_seconds=7.0,
            opener=opener,
        )

        self.assertEqual(result, glb_bytes)
        request = opener.requests[0]
        self.assertEqual(request.method, "GET")
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(opener.timeouts, [7.0])

    def test_download_rejects_oversized_payload_and_non_https_url(self) -> None:
        opener = SequentialOpener([b"x" * 17])

        with self.assertRaisesRegex(MeshyRequestError, "too large"):
            download_glb(
                "https://assets.meshy.ai/model.glb",
                max_bytes=16,
                opener=opener,
            )
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            download_glb(
                "http://assets.meshy.ai/model.glb",
                opener=SequentialOpener([]),
            )


# ### End-to-end client tests ###
class MeshyImageTo3DClientTests(unittest.TestCase):
    def test_client_creates_polls_and_downloads_successful_model(self) -> None:
        glb_bytes = _valid_glb_bytes()
        opener = SequentialOpener(
            [
                _json_bytes({"result": "task-123"}),
                _json_bytes(
                    {
                        "id": "task-123",
                        "status": "SUCCEEDED",
                        "progress": 100,
                        "model_urls": {
                            "glb": "https://assets.meshy.ai/model.glb"
                        },
                    }
                ),
                glb_bytes,
            ]
        )

        result = request_image_to_3d_model(
            api_key="msy-test-secret",
            image_png=b"png bytes",
            poll_interval_seconds=0.0,
            max_polls=1,
            opener=opener,
            sleep=lambda _seconds: None,
            target_polycount=5_432,
        )

        self.assertEqual(result.task_id, "task-123")
        self.assertEqual(result.glb_bytes, glb_bytes)
        self.assertEqual(len(opener.requests), 3)
        submitted_body = json.loads(opener.requests[0].data.decode("utf-8"))
        self.assertEqual(submitted_body["target_polycount"], 5_432)

    def test_http_and_network_errors_are_safe_and_retryable_when_expected(
        self,
    ) -> None:
        api_key = "msy-never-display-this"
        opener = SequentialOpener(
            [_http_error(429, f"Rate limited for {api_key}")]
        )

        with self.assertRaises(MeshyRequestError) as raised:
            create_image_to_3d_task(
                api_key=api_key,
                image_png=b"png bytes",
                opener=opener,
            )

        self.assertEqual(raised.exception.status_code, 429)
        self.assertTrue(raised.exception.retryable)
        self.assertIn("[redacted]", str(raised.exception))
        self.assertNotIn(api_key, str(raised.exception))

        offline_opener = SequentialOpener([URLError("offline")])
        with self.assertRaises(MeshyRequestError) as offline:
            create_image_to_3d_task(
                api_key=api_key,
                image_png=b"png bytes",
                opener=offline_opener,
            )
        self.assertTrue(offline.exception.retryable)


# ### GLB import tests ###
class GeneratedGlbImportTests(unittest.TestCase):
    def test_import_builds_generated_model_from_meshy_glb_bytes(self) -> None:
        glb_bytes = _valid_glb_bytes()

        model = import_generated_glb(glb_bytes)

        self.assertIsInstance(model, GeneratedModel)
        self.assertEqual(model.glb_bytes, glb_bytes)
        self.assertGreater(len(model.mesh.vertices), 0)
        self.assertGreater(len(model.mesh.faces), 0)
        self.assertIsInstance(model.scene, trimesh.Scene)
        self.assertTrue(np.all(np.isfinite(model.mesh.vertices)))
        np.testing.assert_allclose(
            model.mesh.extents,
            np.asarray([1.0, 0.25, 0.5]),
            atol=1e-6,
        )
        self.assertIn(model.mesh.visual.kind, {"face", "vertex"})
        imported_colors = np.asarray(model.mesh.visual.vertex_colors)
        self.assertGreaterEqual(len(np.unique(imported_colors, axis=0)), 2)

    @unittest.skipUnless(
        PillowImage is not None,
        "Pillow is required to verify embedded GLB textures.",
    )
    def test_import_retains_visible_color_from_embedded_texture(self) -> None:
        expected_rgb = np.asarray([24, 176, 92], dtype=np.uint8)
        glb_bytes = _embedded_texture_glb_bytes(tuple(expected_rgb))

        model = import_generated_glb(glb_bytes)

        self.assertIn(b"\x89PNG\r\n\x1a\n", glb_bytes)
        self.assertEqual(model.mesh.visual.kind, "texture")
        texture_image = model.mesh.visual.material.baseColorTexture
        imported_rgb = np.asarray(texture_image.convert("RGB"))[0, 0]
        np.testing.assert_allclose(imported_rgb, expected_rgb, atol=1)

        texture_mesh_data = _build_texture_mesh_data(model.mesh)
        self.assertIsNotNone(texture_mesh_data)
        assert texture_mesh_data is not None
        self.assertEqual(len(texture_mesh_data.vertices), 6)
        self.assertEqual(texture_mesh_data.texture_rgba.shape, (2, 2, 4))

    def test_import_rejects_empty_or_invalid_glb(self) -> None:
        for payload in (b"", b"not a glb file"):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    import_generated_glb(payload)


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
