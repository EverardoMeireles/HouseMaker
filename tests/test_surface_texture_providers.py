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

from housemaker.surface_texture_providers import (
    DEFAULT_MESHY_IMAGE_MODEL,
    MESHY_IMAGE_TO_IMAGE_ENDPOINT,
    OPENAI_FILES_ENDPOINT,
    OPENAI_IMAGE_EDITS_ENDPOINT,
    OPENAI_RESPONSES_ENDPOINT,
    SurfaceTextureProviderSettings,
    SurfaceTextureRequest,
    SurfaceTextureRequestError,
    SurfaceTextureResult,
    SurfaceTextureTaskError,
    build_meshy_image_to_image_request_body,
    build_openai_analysis_request_body,
    build_openai_file_upload_multipart,
    build_openai_image_edit_multipart,
    build_openai_responses_request_body,
    composite_partial_texture_edit,
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


def _pil_png(
    mode: str,
    size: tuple[int, int],
    values: list[object] | object,
) -> bytes:
    image = Image.new(mode, size, None if isinstance(values, list) else values)
    if isinstance(values, list):
        image.putdata(values)
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

    def test_result_contract_has_provider_texture_and_optional_task(self) -> None:
        result = SurfaceTextureResult(
            provider="meshy",
            texture_png=_png_bytes(),
            task_id="task-1",
        )

        self.assertEqual(result.provider, "meshy")
        self.assertEqual(result.task_id, "task-1")

    def test_partial_edit_requires_matching_nonempty_texture_space_mask(self) -> None:
        settings = _meshy_settings()
        base = _pil_png("RGBA", (2, 2), (10, 20, 30, 255))
        mask = _pil_png("L", (2, 2), [0, 255, 0, 0])

        request = SurfaceTextureRequest(
            (_png_bytes(),),
            "stone",
            settings,
            existing_texture_png=base,
            edit_mask_png=mask,
        )

        self.assertTrue(request.is_partial_edit)
        with self.assertRaisesRegex(ValueError, "provided together"):
            SurfaceTextureRequest(
                (_png_bytes(),),
                "stone",
                settings,
                existing_texture_png=base,
            )
        with self.assertRaisesRegex(ValueError, "identical dimensions"):
            SurfaceTextureRequest(
                (_png_bytes(),),
                "stone",
                settings,
                existing_texture_png=base,
                edit_mask_png=_pil_png("L", (1, 2), 255),
            )
        with self.assertRaisesRegex(ValueError, "no editable pixels"):
            SurfaceTextureRequest(
                (_png_bytes(),),
                "stone",
                settings,
                existing_texture_png=base,
                edit_mask_png=_pil_png("L", (2, 2), 0),
            )


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

    def test_partial_meshy_request_reserves_base_and_control_references(self) -> None:
        base = _pil_png("RGBA", (2, 2), (10, 20, 30, 255))
        mask = _pil_png("L", (2, 2), [0, 255, 0, 0])
        references = tuple(_png_bytes(index, 2, 3) for index in range(1, 6))
        request = SurfaceTextureRequest(
            references,
            "repair the wall finish",
            _meshy_settings(),
            existing_texture_png=base,
            edit_mask_png=mask,
        )

        body = build_meshy_image_to_image_request_body(request)
        decoded = [
            base64.b64decode(value.removeprefix("data:image/png;base64,"))
            for value in body["reference_image_urls"]
        ]

        self.assertEqual(len(decoded), 5)
        self.assertEqual(decoded[0], base)
        with Image.open(io.BytesIO(decoded[1])) as control:
            self.assertEqual(control.convert("RGB").getpixel((1, 0)), (211, 4, 214))
            self.assertEqual(control.convert("RGB").getpixel((0, 0)), (10, 20, 30))
        packed_colors: set[tuple[int, int, int, int]] = set()
        for packed_reference in decoded[2:]:
            with Image.open(io.BytesIO(packed_reference)) as image:
                rgba = image.convert("RGBA")
                packed_colors.update(
                    rgba.getpixel((column, row))
                    for row in range(rgba.height)
                    for column in range(rgba.width)
                    if rgba.getpixel((column, row))[3] > 0
                )
        self.assertEqual(
            packed_colors,
            {(index, 2, 3, 255) for index in range(1, 6)},
        )
        self.assertIn("bright magenta", body["prompt"])

    def test_partial_openai_responses_request_forces_masked_edit(self) -> None:
        request = SurfaceTextureRequest(
            (_png_bytes(),),
            "repair the wall finish",
            _openai_settings("gpt-5.6-luna"),
            existing_texture_png=_pil_png("RGBA", (2, 2), (1, 2, 3, 255)),
            edit_mask_png=_pil_png("L", (2, 2), [255, 0, 0, 0]),
        )

        body = build_openai_responses_request_body(
            request,
            existing_texture_file_id="file-base",
            edit_mask_file_id="file-mask",
        )

        content = body["input"][0]["content"]
        self.assertEqual(content[1]["file_id"], "file-base")
        self.assertEqual(body["tools"][0]["action"], "edit")
        self.assertEqual(
            body["tools"][0]["input_image_mask"],
            {"file_id": "file-mask"},
        )

    def test_openai_multipart_mask_has_transparent_editable_pixels(self) -> None:
        base = _pil_png("RGBA", (2, 1), [(10, 20, 30, 255)] * 2)
        edit_mask = _pil_png("L", (2, 1), [255, 0])

        body, content_type = build_openai_image_edit_multipart(
            (_png_bytes(),),
            "repair only the marked area",
            existing_texture_png=base,
            edit_mask_png=edit_mask,
        )

        boundary = content_type.removeprefix("multipart/form-data; boundary=")
        mask_part = next(
            part
            for part in body.split(("--" + boundary).encode("ascii"))
            if b'name="mask"' in part
        )
        encoded_mask = mask_part.split(b"\r\n\r\n", 1)[1].removesuffix(b"\r\n")
        with Image.open(io.BytesIO(encoded_mask)) as mask_image:
            alpha = mask_image.convert("RGBA").getchannel("A")
            self.assertEqual(
                [alpha.getpixel((0, 0)), alpha.getpixel((1, 0))],
                [0, 255],
            )
        self.assertEqual(body.count(b'name="image[]"'), 2)

    def test_openai_vision_upload_is_a_png_file_with_vision_purpose(self) -> None:
        image_png = _png_bytes()

        body, content_type = build_openai_file_upload_multipart(
            image_png,
            "existing_texture.png",
        )

        self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
        self.assertIn(b'name="purpose"', body)
        self.assertIn(b"vision", body)
        self.assertIn(b'name="file"', body)
        self.assertIn(image_png, body)

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


# ### Partial compositing tests ###
class SurfaceTexturePartialCompositingTests(unittest.TestCase):
    def test_composite_preserves_unmasked_pixels_and_resizes_generation(self) -> None:
        base_pixels = [
            (10, 20, 30, 40),
            (50, 60, 70, 80),
            (90, 100, 110, 120),
            (130, 140, 150, 160),
        ]
        base = _pil_png("RGBA", (2, 2), base_pixels)
        generated = _pil_png("RGBA", (1, 1), (200, 210, 220, 230))
        mask = _pil_png("L", (2, 2), [0, 255, 0, 255])

        output = composite_partial_texture_edit(generated, base, mask)

        with Image.open(io.BytesIO(output)) as image:
            rgba = image.convert("RGBA")
            pixels = [
                rgba.getpixel((column, row))
                for row in range(2)
                for column in range(2)
            ]
        self.assertEqual(pixels[0], base_pixels[0])
        self.assertEqual(pixels[2], base_pixels[2])
        self.assertEqual(pixels[1], (200, 210, 220, 230))
        self.assertEqual(pixels[3], (200, 210, 220, 230))

# ### Meshy adapter tests ###
class MeshySurfaceTextureAdapterTests(unittest.TestCase):
    def test_partial_meshy_result_is_composited_with_the_exact_base_pixels(self) -> None:
        base_pixels = [(10, 20, 30, 40), (50, 60, 70, 80)]
        base = _pil_png("RGBA", (2, 1), base_pixels)
        edit_mask = _pil_png("L", (2, 1), [0, 255])
        generated = _pil_png("RGBA", (2, 1), [(200, 210, 220, 230)] * 2)
        opener = SequentialOpener(
            [
                _json_bytes({"result": "task-partial"}),
                _json_bytes(
                    {
                        "id": "task-partial",
                        "status": "SUCCEEDED",
                        "progress": 100,
                        "image_urls": [
                            "https://assets.meshy.ai/partial-texture.png"
                        ],
                    }
                ),
                generated,
            ]
        )

        result = request_surface_texture(
            "meshy",
            "msy-secret",
            (_png_bytes(),),
            "repair the marked plaster",
            existing_texture_png=base,
            edit_mask_png=edit_mask,
            opener=opener,
            sleep=lambda _seconds: None,
        )

        with Image.open(io.BytesIO(result.texture_png)) as output:
            rgba = output.convert("RGBA")
            self.assertEqual(rgba.getpixel((0, 0)), base_pixels[0])
            self.assertEqual(rgba.getpixel((1, 0)), (200, 210, 220, 230))

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
                            "message": f"bad reference for {api_key}"
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

        http_opener = SequentialOpener(
            [
                _http_error(
                    MESHY_IMAGE_TO_IMAGE_ENDPOINT,
                    429,
                    f"limited {api_key}",
                )
            ]
        )
        with self.assertRaises(SurfaceTextureRequestError) as rejected:
            request_surface_texture(
                "meshy",
                api_key,
                (_png_bytes(),),
                "stone",
                opener=http_opener,
            )
        self.assertEqual(rejected.exception.status_code, 429)
        self.assertTrue(rejected.exception.retryable)
        self.assertNotIn(api_key, str(rejected.exception))


# ### OpenAI adapter tests ###
class OpenAISurfaceTextureAdapterTests(unittest.TestCase):
    def test_direct_partial_edit_uploads_mask_and_preserves_outside_pixels(self) -> None:
        base_pixels = [(10, 20, 30, 40), (50, 60, 70, 80)]
        base = _pil_png("RGBA", (2, 1), base_pixels)
        edit_mask = _pil_png("L", (2, 1), [0, 255])
        generated = _pil_png("RGBA", (2, 1), [(200, 210, 220, 230)] * 2)
        opener = SequentialOpener(
            [
                _json_bytes({"id": "file-base"}),
                _json_bytes({"id": "file-mask"}),
                _json_bytes(
                    {
                        "output": [
                            {
                                "type": "image_generation_call",
                                "result": base64.b64encode(generated).decode("ascii"),
                            }
                        ]
                    }
                ),
                _json_bytes({"deleted": True, "id": "file-mask"}),
                _json_bytes({"deleted": True, "id": "file-base"}),
            ]
        )

        result = request_surface_texture(
            "gpt-5.6-terra",
            "sk-secret",
            (_png_bytes(),),
            "repair the marked plaster",
            existing_texture_png=base,
            edit_mask_png=edit_mask,
            opener=opener,
        )

        with Image.open(io.BytesIO(result.texture_png)) as output:
            rgba = output.convert("RGBA")
            self.assertEqual(rgba.size, (2, 1))
            self.assertEqual(rgba.getpixel((0, 0)), base_pixels[0])
            self.assertEqual(rgba.getpixel((1, 0)), (200, 210, 220, 230))
        self.assertEqual(
            [request.full_url for request in opener.requests],
            [
                OPENAI_FILES_ENDPOINT,
                OPENAI_FILES_ENDPOINT,
                OPENAI_RESPONSES_ENDPOINT,
                f"{OPENAI_FILES_ENDPOINT}/file-mask",
                f"{OPENAI_FILES_ENDPOINT}/file-base",
            ],
        )
        response_body = json.loads(opener.requests[2].data.decode("utf-8"))
        self.assertEqual(response_body["tools"][0]["action"], "edit")
        self.assertEqual(
            response_body["tools"][0]["input_image_mask"]["file_id"],
            "file-mask",
        )

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
