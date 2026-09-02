# ### Imports ###
from __future__ import annotations

import base64
import io
import json
import struct
import threading
import unittest
import zlib
from urllib.error import HTTPError, URLError

from PIL import Image, features

from housemaker.pbr_maps import (
    PBR_MAP_METALLIC,
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
    PBR_MAP_TYPES,
)
from housemaker.surface_texture_providers import (
    DEFAULT_MESHY_IMAGE_MODEL,
    MESHY_IMAGE_TO_IMAGE_ENDPOINT,
    MESHY_RETEXTURE_ENDPOINT,
    OPENAI_IMAGE_EDITS_ENDPOINT,
    OPENAI_RESPONSES_ENDPOINT,
    SurfaceTextureProviderSettings,
    SurfaceTextureRequest,
    SurfaceTextureRequestError,
    SurfaceTextureResult,
    SurfaceTextureTaskError,
    _align_surface_pbr_primary_region,
    build_meshy_image_to_image_request_body,
    build_meshy_surface_pbr_request_body,
    build_openai_analysis_request_body,
    build_openai_image_edit_multipart,
    build_openai_responses_request_body,
    generate_surface_texture,
    request_surface_texture,
)


# ### Test doubles ###
class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        content_type: str | None = None,
    ) -> None:
        self._payload = payload
        self._position = 0
        self._content_type = content_type

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

    def getheader(self, name: str) -> str | None:
        if name.lower() == "content-type":
            return self._content_type
        return None


class SequentialOpener:
    def __init__(
        self,
        responses: list[bytes | Exception | FakeResponse],
    ) -> None:
        self._responses = list(responses)
        self.requests: list[object] = []
        self.timeouts: list[float] = []

    def __call__(self, request, timeout: float) -> FakeResponse:
        self.requests.append(request)
        self.timeouts.append(timeout)
        if not self._responses:
            raise AssertionError("The fake provider received an extra request.")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, FakeResponse):
            return response
        return FakeResponse(response)


# ### Fixture helpers ###
def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", checksum)
    )


def _png_bytes(red: int = 40, green: int = 120, blue: int = 210) -> bytes:
    header = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    pixels = zlib.compress(bytes((0, red, green, blue, 255)))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", pixels)
        + _png_chunk(b"IEND", b"")
    )


def _encoded_image_bytes(
    image_format: str,
    color: tuple[int, int, int] = (70, 130, 190),
) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (3, 2), color).save(output, format=image_format)
    return output.getvalue()


def _surface_pbr_map_png(
    primary_color: tuple[int, int, int],
    support_color: tuple[int, int, int],
    *,
    size: int = 20,
) -> bytes:
    """Build a square fixture whose right 5% represents helper-face pixels."""

    image = Image.new("RGB", (size, size), primary_color)
    support_width = max(1, round(size * 0.05))
    image.paste(
        support_color,
        (size - support_width, 0, size, size),
    )
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _jpeg_with_dimensions(width: int, height: int) -> bytes:
    payload = bytearray(_encoded_image_bytes("JPEG"))
    for marker in (b"\xff\xc0", b"\xff\xc1", b"\xff\xc2"):
        marker_index = payload.find(marker)
        if marker_index < 0:
            continue
        payload[marker_index + 5 : marker_index + 7] = height.to_bytes(
            2,
            "big",
        )
        payload[marker_index + 7 : marker_index + 9] = width.to_bytes(
            2,
            "big",
        )
        return bytes(payload)
    raise AssertionError("JPEG fixture has no supported frame header")


def _meshy_settings(api_key: str = "msy-test-secret") -> SurfaceTextureProviderSettings:
    return SurfaceTextureProviderSettings(
        provider="meshy",
        meshy_api_key=api_key,
    )


def _openai_settings(
    model: str,
    api_key: str = "sk-test-secret",
) -> SurfaceTextureProviderSettings:
    return SurfaceTextureProviderSettings(
        provider="openai",
        model=model,
        openai_api_key=api_key,
    )


def _request(
    settings: SurfaceTextureProviderSettings,
    images: tuple[bytes, ...] | None = None,
) -> SurfaceTextureRequest:
    return SurfaceTextureRequest(
        reference_pngs=images or (_png_bytes(),),
        prompt="old limestone wall with subtle warm variation",
        settings=settings,
    )


def _http_error(url: str, status_code: int, message: str) -> HTTPError:
    return HTTPError(
        url=url,
        code=status_code,
        msg="Provider request failed",
        hdrs=None,
        fp=io.BytesIO(_json_bytes({"error": {"message": message}})),
    )


# ### Validation tests ###
class SurfaceTextureValidationTests(unittest.TestCase):
    def test_settings_normalize_provider_model_and_hide_keys_from_repr(self) -> None:
        settings = SurfaceTextureProviderSettings(
            provider=" MESHY ",
            meshy_api_key=" msy-private-key ",
        )

        self.assertEqual(settings.provider, "meshy")
        self.assertEqual(settings.model, DEFAULT_MESHY_IMAGE_MODEL)
        self.assertEqual(settings.active_api_key, "msy-private-key")
        self.assertNotIn("msy-private-key", repr(settings))

    def test_settings_require_supported_model_and_active_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "Meshy API key"):
            SurfaceTextureProviderSettings(provider="meshy")
        with self.assertRaisesRegex(ValueError, "Unsupported OpenAI"):
            SurfaceTextureProviderSettings(
                provider="openai",
                model="gpt-made-up",
                openai_api_key="secret",
            )

    def test_request_accepts_one_to_five_complete_pngs(self) -> None:
        settings = _meshy_settings()
        images = tuple(_png_bytes(index, 20, 30) for index in range(1, 6))

        request = _request(settings, images)

        self.assertEqual(request.reference_pngs, images)
        with self.assertRaisesRegex(ValueError, "between 1 and 5"):
            SurfaceTextureRequest((), "stone", settings)
        with self.assertRaisesRegex(ValueError, "between 1 and 5"):
            SurfaceTextureRequest(images + (_png_bytes(),), "stone", settings)

    def test_request_rejects_bad_signature_checksum_and_empty_prompt(self) -> None:
        settings = _meshy_settings()
        corrupt = bytearray(_png_bytes())
        corrupt[-1] ^= 0x01

        for image in (b"not png", bytes(corrupt)):
            with self.subTest(image=image[:8]):
                with self.assertRaisesRegex(ValueError, "PNG"):
                    SurfaceTextureRequest((image,), "stone", settings)
        with self.assertRaisesRegex(ValueError, "prompt"):
            SurfaceTextureRequest((_png_bytes(),), "  ", settings)

    def test_request_normalizes_meshy_pbr_maps_and_rejects_them_for_openai(
        self,
    ) -> None:
        request = SurfaceTextureRequest(
            (_png_bytes(),),
            "stone",
            _meshy_settings(),
            (PBR_MAP_METALLIC, PBR_MAP_NORMAL, PBR_MAP_NORMAL),
        )

        self.assertEqual(
            request.enabled_pbr_maps,
            (PBR_MAP_NORMAL, PBR_MAP_METALLIC),
        )
        with self.assertRaisesRegex(ValueError, "requires the Meshy"):
            SurfaceTextureRequest(
                (_png_bytes(),),
                "stone",
                _openai_settings("gpt-5.6-luna"),
                (PBR_MAP_NORMAL,),
            )

    def test_result_contract_has_provider_texture_and_optional_task(self) -> None:
        result = SurfaceTextureResult(
            provider="meshy",
            texture_png=_png_bytes(),
            task_id="task-1",
            pbr_texture_pngs={
                PBR_MAP_METALLIC: _png_bytes(3, 3, 3),
                PBR_MAP_NORMAL: _png_bytes(128, 128, 255),
            },
            pbr_task_id="pbr-task-1",
        )

        self.assertEqual(result.provider, "meshy")
        self.assertEqual(result.task_id, "task-1")
        self.assertEqual(
            result.available_pbr_maps,
            (PBR_MAP_NORMAL, PBR_MAP_METALLIC),
        )
        self.assertEqual(result.pbr_task_id, "pbr-task-1")

# ### Request construction tests ###
class SurfaceTextureRequestConstructionTests(unittest.TestCase):
    def test_meshy_body_uses_multi_reference_image_to_image(self) -> None:
        images = (_png_bytes(1, 2, 3), _png_bytes(4, 5, 6))
        request = _request(_meshy_settings(), images)

        body = build_meshy_image_to_image_request_body(request)

        self.assertEqual(
            set(body),
            {
                "ai_model",
                "prompt",
                "reference_image_urls",
                "generate_multi_view",
                "aspect_ratio",
            },
        )
        self.assertEqual(body["ai_model"], DEFAULT_MESHY_IMAGE_MODEL)
        self.assertEqual(body["aspect_ratio"], "1:1")
        self.assertFalse(body["generate_multi_view"])
        self.assertIn("seamless", body["prompt"])
        self.assertIn(request.prompt, body["prompt"])
        decoded_images = [
            base64.b64decode(url.removeprefix("data:image/png;base64,"))
            for url in body["reference_image_urls"]
        ]
        self.assertEqual(decoded_images, list(images))

    def test_meshy_surface_pbr_body_uses_isolated_non_overlapping_uvs(
        self,
    ) -> None:
        base_color = _png_bytes(30, 60, 90)

        first_body = build_meshy_surface_pbr_request_body(base_color)
        second_body = build_meshy_surface_pbr_request_body(base_color)

        self.assertEqual(first_body, second_body)
        self.assertEqual(first_body["ai_model"], "meshy-6")
        self.assertTrue(first_body["enable_original_uv"])
        self.assertTrue(first_body["enable_pbr"])
        self.assertEqual(first_body["texture_resolution"], "2k")
        self.assertEqual(first_body["target_formats"], ["glb"])
        self.assertEqual(
            base64.b64decode(
                first_body["image_style_url"].removeprefix(
                    "data:image/png;base64,"
                )
            ),
            base_color,
        )
        slab_glb = base64.b64decode(
            first_body["model_url"].removeprefix(
                "data:application/octet-stream;base64,"
            )
        )
        self.assertEqual(slab_glb[:4], b"glTF")
        json_length = struct.unpack("<I", slab_glb[12:16])[0]
        self.assertEqual(slab_glb[16:20], b"JSON")
        document = json.loads(
            slab_glb[20 : 20 + json_length].decode("utf-8")
        )
        primitive = document["meshes"][0]["primitives"][0]
        position_accessor = document["accessors"][
            primitive["attributes"]["POSITION"]
        ]
        uv_accessor = document["accessors"][
            primitive["attributes"]["TEXCOORD_0"]
        ]
        self.assertEqual(position_accessor["count"], 24)
        self.assertLess(position_accessor["min"][2], 0.0)
        self.assertGreater(position_accessor["max"][2], 0.0)
        self.assertEqual(uv_accessor["min"], [0.0, 0.0])
        self.assertEqual(uv_accessor["max"], [1.0, 1.0])
        self.assertEqual(
            document["accessors"][primitive["indices"]]["count"],
            36,
        )
        binary_offset = 20 + json_length + 8
        uv_view = document["bufferViews"][uv_accessor["bufferView"]]
        uv_offset = binary_offset + uv_view.get("byteOffset", 0)
        uv_values = struct.unpack_from("<48f", slab_glb, uv_offset)
        uv_pairs = tuple(zip(uv_values[::2], uv_values[1::2]))
        self.assertEqual(
            tuple(round(u, 2) for u, _v in uv_pairs[:4]),
            (0.0, 0.95, 0.95, 0.0),
        )
        self.assertTrue(
            all(round(u, 2) >= 0.95 for u, _v in uv_pairs[4:])
        )

    def test_luna_and_terra_use_responses_image_generation_tool(self) -> None:
        for model in ("gpt-5.6-luna", "gpt-5.6-terra"):
            with self.subTest(model=model):
                request = _request(_openai_settings(model))

                body = build_openai_responses_request_body(request)

                self.assertEqual(body["model"], model)
                self.assertEqual(
                    body["tools"],
                    [
                        {
                            "type": "image_generation",
                            "action": "generate",
                            "quality": "high",
                            "size": "2048x2048",
                        }
                    ],
                )
                content = body["input"][0]["content"]
                self.assertEqual(content[0]["type"], "input_text")
                self.assertEqual(content[1]["type"], "input_image")
                self.assertEqual(content[1]["detail"], "high")

    def test_mini_analysis_then_image_edit_bodies_are_well_formed(self) -> None:
        images = (_png_bytes(10, 20, 30), _png_bytes(40, 50, 60))
        request = _request(_openai_settings("gpt-4o-mini"), images)

        analysis_body = build_openai_analysis_request_body(request)
        multipart, content_type = build_openai_image_edit_multipart(
            images,
            "Generate a seamless warm stone texture.",
        )

        self.assertEqual(analysis_body["model"], "gpt-4o-mini")
        self.assertNotIn("tools", analysis_body)
        self.assertEqual(len(analysis_body["input"][0]["content"]), 3)
        self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
        self.assertEqual(multipart.count(b'name="image[]"'), 2)
        self.assertIn(b'name="model"', multipart)
        self.assertIn(b"gpt-image-2", multipart)
        self.assertIn(b"Generate a seamless warm stone texture.", multipart)
        for image in images:
            self.assertIn(image, multipart)


# ### Meshy adapter tests ###
class MeshySurfaceTextureAdapterTests(unittest.TestCase):
    def test_public_adapter_creates_polls_and_downloads_texture(self) -> None:
        texture_png = _png_bytes(200, 170, 90)
        opener = SequentialOpener(
            [
                _json_bytes({"result": "task-texture-1"}),
                _json_bytes(
                    {
                        "id": "task-texture-1",
                        "status": "PENDING",
                        "progress": 0,
                    }
                ),
                _json_bytes(
                    {
                        "id": "task-texture-1",
                        "status": "SUCCEEDED",
                        "progress": 100,
                        "image_urls": [
                            "https://assets.meshy.ai/texture.png?Expires=123"
                        ],
                    }
                ),
                texture_png,
            ]
        )
        sleeps: list[float] = []
        updates: list[tuple[str, int]] = []

        result = request_surface_texture(
            "meshy",
            "msy-secret",
            (_png_bytes(),),
            "warm plaster",
            opener=opener,
            sleep=sleeps.append,
            progress_callback=lambda status, progress: updates.append(
                (status, progress)
            ),
        )

        self.assertEqual(result.provider, "meshy")
        self.assertEqual(result.task_id, "task-texture-1")
        self.assertEqual(result.texture_png, texture_png)
        self.assertEqual(sleeps, [5.0])
        self.assertEqual(updates[:2], [("PENDING", 0), ("SUCCEEDED", 100)])
        self.assertEqual(
            [request.full_url for request in opener.requests],
            [
                MESHY_IMAGE_TO_IMAGE_ENDPOINT,
                f"{MESHY_IMAGE_TO_IMAGE_ENDPOINT}/task-texture-1",
                f"{MESHY_IMAGE_TO_IMAGE_ENDPOINT}/task-texture-1",
                "https://assets.meshy.ai/texture.png?Expires=123",
            ],
        )
        self.assertEqual(
            opener.requests[0].get_header("Authorization"),
            "Bearer msy-secret",
        )
        self.assertIsNone(opener.requests[-1].get_header("Authorization"))
        sent_body = json.loads(opener.requests[0].data.decode("utf-8"))
        self.assertEqual(sent_body["ai_model"], DEFAULT_MESHY_IMAGE_MODEL)

    def test_generic_image_processing_failure_retries_with_backoff(self) -> None:
        texture_png = _png_bytes(80, 120, 160)
        opener = SequentialOpener(
            [
                _json_bytes({"result": "image-failed"}),
                _json_bytes(
                    {
                        "status": "FAILED",
                        "progress": 25,
                        "task_error": {
                            "type": "invalid_input",
                            "code": "invalid_input",
                            "message": (
                                "The input file or parameters could not be "
                                "processed. Please check your input and try "
                                "again."
                            ),
                        },
                    }
                ),
                _json_bytes({"result": "image-recovered"}),
                _json_bytes(
                    {
                        "status": "SUCCEEDED",
                        "progress": 100,
                        "image_urls": ["https://assets.meshy.ai/recovered.png"],
                    }
                ),
                texture_png,
            ]
        )
        sleeps: list[float] = []
        updates: list[tuple[str, int]] = []

        result = request_surface_texture(
            "meshy",
            "msy-secret",
            (_png_bytes(),),
            "warm plaster",
            opener=opener,
            sleep=sleeps.append,
            progress_callback=lambda status, progress: updates.append(
                (status, progress)
            ),
        )

        self.assertEqual(result.task_id, "image-recovered")
        self.assertEqual(result.texture_png, texture_png)
        self.assertEqual(sleeps, [5.0])
        self.assertIn(("RETRYING_1", 25), updates)
        self.assertEqual(
            [progress for _status, progress in updates],
            sorted(progress for _status, progress in updates),
        )

    def test_pbr_server_failure_retries_without_repeating_image_stage(
        self,
    ) -> None:
        style_png = _png_bytes(20, 40, 60)
        normal_png = _png_bytes(128, 128, 255)
        roughness_png = _png_bytes(180, 180, 180)
        metallic_png = _png_bytes(5, 5, 5)
        texture_urls = {
            "base_color": "https://assets.meshy.ai/final-base.png",
            PBR_MAP_NORMAL: "https://assets.meshy.ai/final-normal.png",
            PBR_MAP_ROUGHNESS: "https://assets.meshy.ai/final-roughness.png",
            PBR_MAP_METALLIC: "https://assets.meshy.ai/final-metallic.png",
        }
        opener = SequentialOpener(
            [
                _json_bytes({"result": "image-task"}),
                _json_bytes(
                    {
                        "status": "SUCCEEDED",
                        "progress": 100,
                        "image_urls": ["https://assets.meshy.ai/style.png"],
                    }
                ),
                style_png,
                _json_bytes({"result": "pbr-failed"}),
                _json_bytes(
                    {
                        "status": "FAILED",
                        "progress": 30,
                        "task_error": {
                            "type": "server_error",
                            "message": "An unexpected error occurred.",
                        },
                    }
                ),
                _json_bytes({"result": "pbr-recovered"}),
                _json_bytes(
                    {
                        "status": "SUCCEEDED",
                        "progress": 100,
                        "texture_urls": [texture_urls],
                    }
                ),
                normal_png,
                roughness_png,
                metallic_png,
            ]
        )
        sleeps: list[float] = []

        result = request_surface_texture(
            "meshy",
            "msy-secret",
            (_png_bytes(),),
            "warm stone",
            enabled_pbr_maps=(PBR_MAP_NORMAL,),
            opener=opener,
            sleep=sleeps.append,
        )

        self.assertEqual(result.task_id, "image-task")
        self.assertEqual(result.pbr_task_id, "pbr-recovered")
        self.assertEqual(result.texture_png, style_png)
        self.assertEqual(sleeps, [5.0])
        image_posts = [
            request
            for request in opener.requests
            if request.full_url == MESHY_IMAGE_TO_IMAGE_ENDPOINT
        ]
        pbr_posts = [
            request
            for request in opener.requests
            if request.full_url == MESHY_RETEXTURE_ENDPOINT
        ]
        self.assertEqual(len(image_posts), 1)
        self.assertEqual(len(pbr_posts), 2)

    def test_pbr_generation_retextures_a_slab_and_downloads_aligned_maps(
        self,
    ) -> None:
        style_png = _png_bytes(20, 40, 60)
        normal_png = _surface_pbr_map_png(
            (128, 128, 255),
            (255, 0, 255),
        )
        roughness_png = _png_bytes(180, 180, 180)
        metallic_png = _png_bytes(5, 5, 5)
        texture_urls = {
            "base_color": "https://assets.meshy.ai/final-base.png",
            PBR_MAP_NORMAL: "https://assets.meshy.ai/final-normal.png",
            PBR_MAP_ROUGHNESS: "https://assets.meshy.ai/final-roughness.png",
            PBR_MAP_METALLIC: "https://assets.meshy.ai/final-metallic.png",
        }
        opener = SequentialOpener(
            [
                _json_bytes({"result": "image-task"}),
                _json_bytes(
                    {
                        "status": "SUCCEEDED",
                        "progress": 100,
                        "image_urls": ["https://assets.meshy.ai/style.png"],
                    }
                ),
                style_png,
                _json_bytes({"result": "pbr-task"}),
                _json_bytes({"status": "PENDING", "progress": 0}),
                _json_bytes(
                    {
                        "status": "SUCCEEDED",
                        "progress": 100,
                        "texture_urls": [texture_urls],
                    }
                ),
                normal_png,
                roughness_png,
                metallic_png,
            ]
        )
        updates: list[tuple[str, int]] = []

        result = request_surface_texture(
            "meshy",
            "msy-secret",
            (_png_bytes(),),
            "warm stone",
            enabled_pbr_maps=(PBR_MAP_NORMAL,),
            opener=opener,
            sleep=lambda _seconds: None,
            progress_callback=lambda status, progress: updates.append(
                (status, progress)
            ),
        )

        self.assertEqual(result.task_id, "image-task")
        self.assertEqual(result.pbr_task_id, "pbr-task")
        self.assertEqual(result.texture_png, style_png)
        self.assertEqual(result.available_pbr_maps, PBR_MAP_TYPES)
        self.assertEqual(result.pbr_texture_pngs[PBR_MAP_ROUGHNESS], roughness_png)
        self.assertEqual(result.pbr_texture_pngs[PBR_MAP_METALLIC], metallic_png)
        with Image.open(
            io.BytesIO(result.pbr_texture_pngs[PBR_MAP_NORMAL])
        ) as isolated_normal:
            isolated_normal.load()
            self.assertEqual(isolated_normal.size, (20, 20))
            self.assertEqual(
                isolated_normal.convert("RGB").getpixel((19, 10)),
                (127, 128, 255),
            )
        self.assertEqual(
            [request.full_url for request in opener.requests],
            [
                MESHY_IMAGE_TO_IMAGE_ENDPOINT,
                f"{MESHY_IMAGE_TO_IMAGE_ENDPOINT}/image-task",
                "https://assets.meshy.ai/style.png",
                MESHY_RETEXTURE_ENDPOINT,
                f"{MESHY_RETEXTURE_ENDPOINT}/pbr-task",
                f"{MESHY_RETEXTURE_ENDPOINT}/pbr-task",
                *(texture_urls[map_type] for map_type in PBR_MAP_TYPES),
            ],
        )
        retexture_body = json.loads(opener.requests[3].data.decode("utf-8"))
        self.assertEqual(retexture_body["ai_model"], "meshy-6")
        self.assertEqual(
            base64.b64decode(
                retexture_body["image_style_url"].removeprefix(
                    "data:image/png;base64,"
                )
            ),
            style_png,
        )
        self.assertEqual(updates[-1], ("SUCCEEDED", 100))
        self.assertEqual(
            [progress for _status, progress in updates],
            sorted(progress for _status, progress in updates),
        )

    def test_surface_pbr_alignment_mirrors_pixels_and_normal_tangent_x(
        self,
    ) -> None:
        source = Image.new("RGBA", (20, 10), (0, 0, 0, 255))
        source.paste((32, 70, 220, 255), (0, 0, 9, 10))
        source.paste((210, 90, 180, 255), (9, 0, 19, 10))
        source.paste((255, 0, 255, 255), (19, 0, 20, 10))
        payload = io.BytesIO()
        source.save(payload, format="PNG")

        aligned_normal = _align_surface_pbr_primary_region(
            payload.getvalue(),
            map_type=PBR_MAP_NORMAL,
            label="normal fixture",
        )

        with Image.open(io.BytesIO(aligned_normal)) as aligned:
            aligned.load()
            rgba = aligned.convert("RGBA")
            self.assertEqual(rgba.getpixel((0, 5)), (45, 90, 180, 255))
            self.assertEqual(rgba.getpixel((19, 5)), (223, 70, 220, 255))

    def test_pbr_generation_fails_if_a_requested_map_is_missing(self) -> None:
        opener = SequentialOpener(
            [
                _json_bytes({"result": "image-task"}),
                _json_bytes(
                    {
                        "status": "SUCCEEDED",
                        "progress": 100,
                        "image_urls": ["https://assets.meshy.ai/style.png"],
                    }
                ),
                _png_bytes(),
                _json_bytes({"result": "pbr-task"}),
                _json_bytes(
                    {
                        "status": "SUCCEEDED",
                        "progress": 100,
                        "texture_urls": [
                            {
                                "base_color": (
                                    "https://assets.meshy.ai/base.png"
                                ),
                                PBR_MAP_ROUGHNESS: (
                                    "https://assets.meshy.ai/roughness.png"
                                ),
                            }
                        ],
                    }
                ),
            ]
        )

        with self.assertRaisesRegex(
            SurfaceTextureTaskError,
            "omitted requested maps: normal",
        ):
            request_surface_texture(
                "meshy",
                "msy-secret",
                (_png_bytes(),),
                "stone",
                enabled_pbr_maps=(PBR_MAP_NORMAL,),
                opener=opener,
                sleep=lambda _seconds: None,
            )

    def test_jpeg_and_webp_artifacts_are_safely_normalized_to_png(self) -> None:
        fixture_formats = [("JPEG", "image/jpeg", "texture.jpg")]
        if features.check("webp"):
            fixture_formats.append(("WEBP", "image/webp", "texture.webp"))

        for image_format, content_type, file_name in fixture_formats:
            with self.subTest(image_format=image_format):
                source_image = _encoded_image_bytes(image_format)
                opener = SequentialOpener(
                    [
                        _json_bytes({"result": f"task-{image_format.lower()}"}),
                        _json_bytes(
                            {
                                "status": "SUCCEEDED",
                                "progress": 100,
                                "image_urls": [
                                    f"https://assets.meshy.ai/{file_name}"
                                ],
                            }
                        ),
                        FakeResponse(source_image, content_type),
                    ]
                )

                result = request_surface_texture(
                    "meshy",
                    "msy-secret",
                    (_png_bytes(),),
                    "warm stone",
                    opener=opener,
                    sleep=lambda _seconds: None,
                )

                self.assertTrue(result.texture_png.startswith(b"\x89PNG"))
                self.assertNotEqual(result.texture_png, source_image)
                with Image.open(io.BytesIO(result.texture_png)) as normalized:
                    self.assertEqual(normalized.format, "PNG")
                    self.assertEqual(normalized.size, (3, 2))
                    normalized.load()
                self.assertIn(
                    "image/jpeg",
                    opener.requests[-1].get_header("Accept"),
                )
                self.assertIsNone(
                    opener.requests[-1].get_header("Authorization")
                )

    def test_non_image_wrapper_and_invalid_image_bytes_are_rejected_safely(
        self,
    ) -> None:
        api_key = "msy-artifact-secret"
        artifacts = (
            (
                FakeResponse(
                    _json_bytes(
                        {
                            "url": (
                                "https://assets.meshy.ai/next.png?token="
                                f"{api_key}"
                            )
                        }
                    ),
                    "application/json; charset=utf-8",
                ),
                "non-image",
            ),
            (FakeResponse(b"not an image", "image/jpeg"), "not a valid"),
        )
        for artifact, expected_message in artifacts:
            with self.subTest(expected_message=expected_message):
                opener = SequentialOpener(
                    [
                        _json_bytes({"result": "task-invalid-artifact"}),
                        _json_bytes(
                            {
                                "status": "SUCCEEDED",
                                "progress": 100,
                                "image_urls": [
                                    "https://assets.meshy.ai/texture.png"
                                ],
                            }
                        ),
                        artifact,
                    ]
                )

                with self.assertRaisesRegex(
                    SurfaceTextureTaskError,
                    expected_message,
                ) as raised:
                    request_surface_texture(
                        "meshy",
                        api_key,
                        (_png_bytes(),),
                        "stone",
                        opener=opener,
                        sleep=lambda _seconds: None,
                    )

                self.assertNotIn(api_key, str(raised.exception))
                self.assertEqual(len(opener.requests), 3)

    def test_downloaded_image_dimensions_are_bounded_before_decode(self) -> None:
        oversized_jpeg = _jpeg_with_dimensions(32_769, 1)
        opener = SequentialOpener(
            [
                _json_bytes({"result": "task-oversized-image"}),
                _json_bytes(
                    {
                        "status": "SUCCEEDED",
                        "progress": 100,
                        "image_urls": [
                            "https://assets.meshy.ai/oversized.jpg"
                        ],
                    }
                ),
                FakeResponse(oversized_jpeg, "image/jpeg"),
            ]
        )

        with self.assertRaisesRegex(
            SurfaceTextureTaskError,
            "dimensions",
        ):
            request_surface_texture(
                "meshy",
                "msy-secret",
                (_png_bytes(),),
                "stone",
                opener=opener,
                sleep=lambda _seconds: None,
            )

    def test_failed_task_and_http_errors_redact_api_key(self) -> None:
        api_key = "msy-never-show"
        failed_opener = SequentialOpener(
            [
                _json_bytes({"result": "task-failed"}),
                _json_bytes(
                    {
                        "status": "FAILED",
                        "progress": 20,
                        "task_error": {
                            "type": "invalid_input",
                            "code": "bad_reference",
                            "message": f"bad reference for {api_key}",
                            "doc_url": (
                                "https://docs.meshy.ai/en/api/errors"
                                "#invalid-input"
                            ),
                        },
                    }
                ),
            ]
        )

        with self.assertRaises(SurfaceTextureTaskError) as failed:
            request_surface_texture(
                "meshy",
                api_key,
                (_png_bytes(),),
                "stone",
                opener=failed_opener,
                sleep=lambda _seconds: None,
            )
        self.assertIn("[redacted]", str(failed.exception))
        self.assertNotIn(api_key, str(failed.exception))
        self.assertFalse(failed.exception.retryable)
        self.assertEqual(failed.exception.error_type, "invalid_input")
        self.assertEqual(failed.exception.error_code, "bad_reference")
        self.assertEqual(
            failed.exception.doc_url,
            "https://docs.meshy.ai/en/api/errors#invalid-input",
        )

        http_opener = SequentialOpener(
            [
                _http_error(
                    MESHY_IMAGE_TO_IMAGE_ENDPOINT,
                    429,
                    f"limited {api_key}",
                ),
                _http_error(
                    MESHY_IMAGE_TO_IMAGE_ENDPOINT,
                    429,
                    f"limited {api_key}",
                ),
                _http_error(
                    MESHY_IMAGE_TO_IMAGE_ENDPOINT,
                    429,
                    f"limited {api_key}",
                ),
            ]
        )
        retry_sleeps: list[float] = []
        with self.assertRaises(SurfaceTextureRequestError) as rejected:
            request_surface_texture(
                "meshy",
                api_key,
                (_png_bytes(),),
                "stone",
                opener=http_opener,
                sleep=retry_sleeps.append,
            )
        self.assertEqual(rejected.exception.status_code, 429)
        self.assertTrue(rejected.exception.retryable)
        self.assertNotIn(api_key, str(rejected.exception))
        self.assertEqual(retry_sleeps, [5.0, 15.0])
        self.assertEqual(len(http_opener.requests), 3)


# ### OpenAI adapter tests ###
class OpenAISurfaceTextureAdapterTests(unittest.TestCase):
    def test_terra_returns_responses_image_generation_png(self) -> None:
        texture_png = _png_bytes(90, 100, 110)
        opener = SequentialOpener(
            [
                _json_bytes(
                    {
                        "id": "resp-1",
                        "output": [
                            {
                                "type": "image_generation_call",
                                "result": base64.b64encode(texture_png).decode(
                                    "ascii"
                                ),
                            }
                        ],
                    }
                )
            ]
        )

        result = request_surface_texture(
            "gpt-5.6-terra",
            "sk-secret",
            (_png_bytes(),),
            "pale oak floor",
            opener=opener,
        )

        self.assertEqual(result.provider, "gpt-5.6-terra")
        self.assertIsNone(result.task_id)
        self.assertEqual(result.texture_png, texture_png)
        self.assertEqual(opener.requests[0].full_url, OPENAI_RESPONSES_ENDPOINT)
        body = json.loads(opener.requests[0].data.decode("utf-8"))
        self.assertEqual(body["model"], "gpt-5.6-terra")
        self.assertEqual(body["tools"][0]["type"], "image_generation")

    def test_mini_analyzes_then_renders_with_gpt_image_2_edits(self) -> None:
        texture_png = _png_bytes(210, 210, 205)
        references = (_png_bytes(1, 2, 3), _png_bytes(4, 5, 6))
        opener = SequentialOpener(
            [
                _json_bytes(
                    {
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": (
                                            "A seamless square limestone "
                                            "material texture."
                                        ),
                                    }
                                ],
                            }
                        ]
                    }
                ),
                _json_bytes(
                    {
                        "data": [
                            {
                                "b64_json": base64.b64encode(
                                    texture_png
                                ).decode("ascii")
                            }
                        ]
                    }
                ),
            ]
        )

        result = request_surface_texture(
            "gpt-4o-mini",
            "sk-secret",
            references,
            "limestone wall",
            opener=opener,
        )

        self.assertEqual(result.texture_png, texture_png)
        self.assertEqual(
            [request.full_url for request in opener.requests],
            [OPENAI_RESPONSES_ENDPOINT, OPENAI_IMAGE_EDITS_ENDPOINT],
        )
        analysis_body = json.loads(opener.requests[0].data.decode("utf-8"))
        self.assertEqual(analysis_body["model"], "gpt-4o-mini")
        edit_request = opener.requests[1]
        self.assertTrue(
            edit_request.get_header("Content-type").startswith(
                "multipart/form-data; boundary="
            )
        )
        self.assertEqual(edit_request.data.count(b'name="image[]"'), 2)
        self.assertIn(b"gpt-image-2", edit_request.data)
        self.assertIn(b"A seamless square limestone", edit_request.data)
        for reference in references:
            self.assertIn(reference, edit_request.data)

    def test_openai_error_is_retryable_and_redacts_nested_message(self) -> None:
        api_key = "sk-never-show"
        opener = SequentialOpener(
            [
                _http_error(
                    OPENAI_RESPONSES_ENDPOINT,
                    503,
                    f"temporarily unavailable for {api_key}",
                )
            ]
        )

        with self.assertRaises(SurfaceTextureRequestError) as raised:
            request_surface_texture(
                "gpt-5.6-luna",
                api_key,
                (_png_bytes(),),
                "paint",
                opener=opener,
            )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertTrue(raised.exception.retryable)
        self.assertIn("[redacted]", str(raised.exception))
        self.assertNotIn(api_key, str(raised.exception))

    def test_invalid_generated_base64_and_network_errors_are_safe(self) -> None:
        invalid_opener = SequentialOpener(
            [
                _json_bytes(
                    {
                        "output": [
                            {
                                "type": "image_generation_call",
                                "result": "not valid base64!",
                            }
                        ]
                    }
                )
            ]
        )
        with self.assertRaisesRegex(SurfaceTextureTaskError, "base64"):
            request_surface_texture(
                "gpt-5.6-luna",
                "sk-secret",
                (_png_bytes(),),
                "paint",
                opener=invalid_opener,
            )

        offline = SequentialOpener([URLError("offline")])
        with self.assertRaises(SurfaceTextureRequestError) as raised:
            request_surface_texture(
                "gpt-5.6-luna",
                "sk-secret",
                (_png_bytes(),),
                "paint",
                opener=offline,
            )
        self.assertTrue(raised.exception.retryable)


# ### Dispatch and cancellation tests ###
class SurfaceTextureDispatchTests(unittest.TestCase):
    def test_low_level_dispatch_returns_png_bytes(self) -> None:
        texture_png = _png_bytes()
        request = _request(_openai_settings("gpt-5.6-luna"))
        opener = SequentialOpener(
            [
                _json_bytes(
                    {
                        "output": [
                            {
                                "type": "image_generation_call",
                                "result": base64.b64encode(texture_png).decode(
                                    "ascii"
                                ),
                            }
                        ]
                    }
                )
            ]
        )

        result = generate_surface_texture(request, opener=opener)

        self.assertEqual(result, texture_png)

    def test_cancellation_stops_before_network_and_unknown_choice_fails(self) -> None:
        cancel_event = threading.Event()
        cancel_event.set()
        opener = SequentialOpener([])

        with self.assertRaisesRegex(SurfaceTextureTaskError, "canceled"):
            request_surface_texture(
                "gpt-5.6-luna",
                "sk-secret",
                (_png_bytes(),),
                "paint",
                opener=opener,
                cancel_event=cancel_event,
            )
        self.assertEqual(opener.requests, [])

        with self.assertRaisesRegex(ValueError, "Unsupported"):
            request_surface_texture(
                "unknown-provider",
                "secret",
                (_png_bytes(),),
                "paint",
                opener=opener,
            )


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
