# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from PIL import Image
from PySide6.QtCore import QThread
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QMessageBox
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.uv_integrity import (
    UV_FINGERPRINT_VERSION,
    UvFingerprint,
    build_uv_fingerprint,
)
from housemaker.generation_state import (
    MASK_MODE_PAINT,
    GeneratedObjectRecord,
    GenerationData,
    MaskPoint,
    MaskStroke,
)
from housemaker.generation_workspace import (
    TEXTURE_INPAINT_STROKES_PIPELINE_KEY,
    TextureRegenerationRequest,
    TextureRegenerationWorker,
    GenerationWorkspace,
    MeshyModelExecutor,
    MeshyTextureRegenerator,
    _TextureRegenerationPreflight,
    _GenerationCancelled,
    _materialize_texture_regeneration_preflight,
    _persist_generated_named_asset,
)
from housemaker.glb import (
    GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    GeneratedModel,
    import_generated_glb,
)
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.object_symmetry import (
    AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION,
    SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL,
    SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
    SYMMETRIC_DIVISION_SIDE_LEFT,
    SYMMETRIC_DIVISION_SIDE_RIGHT,
    SYMMETRIC_SELECTION_MODE_FEWEST_TRIANGLES_RANDOM_TIE,
    SYMMETRIC_TEXTURE_CONTENT_HALF_LEFT,
    SYMMETRIC_TEXTURE_PACKING_MODE_PAIR,
    SymmetricDivisionMetadata,
    build_symmetric_retexture_proxy_glb,
)
from housemaker.object_texture_variants import (
    TEXTURE_RESOLUTIONS,
    ObjectTextureVariants,
)
from housemaker.settings_widget import GenerationServiceSettings


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])


# ### Fixture helpers ###
def _box_glb(scale: float = 1.0) -> bytes:
    mesh = trimesh.creation.box(
        extents=(float(scale), float(scale) * 0.5, float(scale) * 0.75)
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _record(object_id: str, asset_path: str) -> GeneratedObjectRecord:
    return GeneratedObjectRecord(
        object_id=object_id,
        frame_index=0,
        object_name=f"Object {object_id}",
        pipeline={},
        provider_task_id=f"task-{object_id}",
        asset_path=asset_path,
    )


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (80, 60),
    )
    if not writer.isOpened():
        raise unittest.SkipTest("MJPG writer is unavailable")
    try:
        for frame_index in range(2):
            writer.write(
                np.full(
                    (60, 80, 3),
                    (frame_index * 40, 40, 160),
                    dtype=np.uint8,
                )
            )
    finally:
        writer.release()


def _uv_glb(
    *,
    mutate_uv: bool = False,
    subdivide_faces: bool = False,
    left_packed: bool = False,
    texture_resolution: int | None = None,
    texture_color: tuple[int, int, int, int] = (25, 80, 160, 255),
    texture_right_color: tuple[int, int, int, int] | None = None,
) -> bytes:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=float,
    )
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    uv = np.asarray(
        ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
        dtype=float,
    )
    if mutate_uv:
        uv[-1] = (0.3, 0.25)
    if left_packed:
        uv[:, 0] *= 0.5
    if subdivide_faces:
        vertices = np.vstack((vertices, np.mean(vertices, axis=0)))
        uv = np.vstack((uv, np.mean(uv, axis=0)))
        faces = np.asarray(
            ((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
            dtype=np.int64,
        )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    texture_image = None
    if texture_resolution is not None:
        texture_image = Image.new(
            "RGBA",
            (texture_resolution, texture_resolution),
            texture_color,
        )
        if texture_right_color is not None:
            texture_image.paste(
                texture_right_color,
                (
                    texture_resolution // 2,
                    0,
                    texture_resolution,
                    texture_resolution,
                ),
            )
    material = PBRMaterial(baseColorTexture=texture_image)
    mesh.visual = TextureVisuals(uv=uv, material=material)
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _png_bytes(color: tuple[int, int, int, int]) -> bytes:
    pixels = np.full((8, 8, 4), color, dtype=np.uint8)
    encoded, payload = cv2.imencode(".png", pixels)
    if not encoded:
        raise RuntimeError("Test PNG encoding failed.")
    return bytes(payload)


def _texture_variants(
    generation_index: int,
) -> ObjectTextureVariants:
    glb_by_resolution = {
        resolution: _box_glb(
            1.0 + generation_index + resolution / 10_000.0
        )
        for resolution in TEXTURE_RESOLUTIONS
    }
    texture_png_by_resolution = {
        resolution: _png_bytes(
            (
                20 + generation_index * 30,
                resolution // 16 % 255,
                140,
                255,
            )
        )
        for resolution in TEXTURE_RESOLUTIONS
    }
    preview_rgba_by_resolution = {
        resolution: np.full(
            (8, 8, 4),
            (20 + generation_index * 30, 80, 140, 255),
            dtype=np.uint8,
        )
        for resolution in TEXTURE_RESOLUTIONS
    }
    return ObjectTextureVariants(
        glb_by_resolution=glb_by_resolution,
        texture_png_by_resolution=texture_png_by_resolution,
        preview_rgba_by_resolution=preview_rgba_by_resolution,
    )


def _model_with_variants(variants: ObjectTextureVariants) -> GeneratedModel:
    model = import_generated_glb(variants.glb_by_resolution[1024])
    model.object_texture_variants = variants
    return model


def _record_variant_paths(record: GeneratedObjectRecord) -> set[str]:
    raw_variants = record.pipeline["texture_variants"]
    return {
        str(path)
        for variant in raw_variants.values()
        for path in variant.values()
    }


def _asset_bytes(
    asset_directory: Path,
    raw_paths: set[str],
) -> dict[str, bytes]:
    return {
        raw_path: asset_directory.joinpath(raw_path).read_bytes()
        for raw_path in raw_paths
    }


# ### Provider fixtures ###
class _SequenceTextureRegenerator:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[TextureRegenerationRequest] = []

    def regenerate(
        self,
        request: TextureRegenerationRequest,
    ) -> MeshyGenerationResult:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, MeshyGenerationResult)
        return outcome


class _SequenceExecutor:
    def __init__(self, models: list[GeneratedModel]) -> None:
        self.models = list(models)
        self.results: list[MeshyGenerationResult] = []

    def execute(self, result: MeshyGenerationResult) -> GeneratedModel:
        self.results.append(result)
        return self.models.pop(0)


class _BlockingTextureRegenerator:
    def __init__(self, result: MeshyGenerationResult) -> None:
        self.result = result
        self.requests: list[TextureRegenerationRequest] = []
        self.started = threading.Event()
        self.release = threading.Event()

    def regenerate(
        self,
        request: TextureRegenerationRequest,
    ) -> MeshyGenerationResult:
        self.requests.append(request)
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise RuntimeError("Blocking texture test timed out.")
        return self.result


# ### Request and provider tests ###
class TextureRegenerationRequestTests(unittest.TestCase):
    def test_request_owns_reference_and_model_and_validates_uv_pairing(
        self,
    ) -> None:
        reference = np.full((6, 8, 4), (10, 20, 30, 255), dtype=np.uint8)
        model_buffer = bytearray(b"owned model")
        settings = GenerationServiceSettings(meshy_api_key="msy-key")

        request = TextureRegenerationRequest(
            object_id="  chair  ",
            reference_frame_index=7,
            reference_image_bgra=reference,
            model_glb=model_buffer,
            settings=settings,
        )
        reference[:, :] = 0
        model_buffer[:] = b"changedxxxx"

        self.assertEqual(request.object_id, "chair")
        self.assertEqual(request.reference_frame_index, 7)
        self.assertTrue(
            np.all(
                request.reference_image_bgra
                == np.asarray((10, 20, 30, 255), dtype=np.uint8)
            )
        )
        self.assertTrue(request.reference_image_bgra.flags.c_contiguous)
        self.assertEqual(request.model_glb, b"owned model")
        self.assertIs(request.settings, settings)

        fingerprint = UvFingerprint(
            UV_FINGERPRINT_VERSION,
            "a" * 64,
            2,
        )
        with self.assertRaisesRegex(ValueError, "requires a UV fingerprint"):
            TextureRegenerationRequest(
                object_id="chair",
                reference_frame_index=0,
                reference_image_bgra=np.zeros((2, 2, 4), dtype=np.uint8),
                model_glb=b"model",
                settings=settings,
                enable_original_uv=True,
            )
        with self.assertRaisesRegex(ValueError, "only valid"):
            TextureRegenerationRequest(
                object_id="chair",
                reference_frame_index=0,
                reference_image_bgra=np.zeros((2, 2, 4), dtype=np.uint8),
                model_glb=b"model",
                settings=settings,
                submitted_uv_fingerprint=fingerprint,
            )
        with self.assertRaisesRegex(ValueError, "only supported for symmetric"):
            TextureRegenerationRequest(
                object_id="chair",
                reference_frame_index=0,
                reference_image_bgra=np.zeros((2, 2, 4), dtype=np.uint8),
                model_glb=b"model",
                settings=settings,
                enable_original_uv=True,
                submitted_uv_fingerprint=fingerprint,
            )

    def test_request_rejects_empty_identity_image_and_model(self) -> None:
        settings = GenerationServiceSettings(meshy_api_key="msy-key")
        valid_image = np.zeros((2, 2, 4), dtype=np.uint8)
        invalid_cases = (
            {"object_id": "", "reference_image_bgra": valid_image, "model_glb": b"m"},
            {"object_id": "chair", "reference_image_bgra": np.zeros((2, 2, 3)), "model_glb": b"m"},
            {"object_id": "chair", "reference_image_bgra": valid_image, "model_glb": b""},
        )
        for invalid in invalid_cases:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    TextureRegenerationRequest(
                        reference_frame_index=0,
                        settings=settings,
                        **invalid,
                    )


# ### Symmetric provider-proxy tests ###
class SymmetricRetextureProxyTests(unittest.TestCase):
    def test_proxy_mirrors_geometry_and_uses_both_texture_halves(self) -> None:
        retained_color = (145, 35, 210, 255)
        retained_glb = _uv_glb(
            left_packed=True,
            texture_resolution=1024,
            texture_color=retained_color,
            texture_right_color=(0, 0, 0, 255),
        )

        for orientation, axis in (
            (SYMMETRIC_DIVISION_ORIENTATION_VERTICAL, 0),
            (SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL, 2),
        ):
            with self.subTest(orientation=orientation):
                proxy_glb = build_symmetric_retexture_proxy_glb(
                    retained_glb,
                    orientation,
                    0.0,
                )
                proxy_scene = trimesh.load(
                    BytesIO(proxy_glb),
                    file_type="glb",
                    force="scene",
                    process=False,
                )

                self.assertEqual(len(proxy_scene.geometry), 2)
                self.assertEqual(
                    sum(
                        len(mesh.faces)
                        for mesh in proxy_scene.geometry.values()
                    ),
                    4,
                )
                z_up_vertices: list[np.ndarray] = []
                uv_ranges: list[tuple[float, float]] = []
                for node_name in proxy_scene.graph.nodes_geometry:
                    transform, geometry_name = proxy_scene.graph.get(node_name)
                    mesh = proxy_scene.geometry[geometry_name]
                    gltf_vertices = trimesh.transform_points(
                        mesh.vertices,
                        transform,
                    )
                    z_up_vertices.append(
                        trimesh.transform_points(
                            gltf_vertices,
                            GLTF_Y_UP_TO_Z_UP_TRANSFORM,
                        )
                    )
                    uv = np.asarray(mesh.visual.uv, dtype=float)
                    uv_ranges.append(
                        (
                            float(np.min(uv[:, 0])),
                            float(np.max(uv[:, 0])),
                        )
                    )
                    texture = np.asarray(
                        mesh.visual.material.baseColorTexture.convert("RGBA"),
                        dtype=np.uint8,
                    )
                    half_width = texture.shape[1] // 2
                    self.assertTrue(
                        np.all(texture[:, :half_width] == retained_color)
                    )
                    self.assertTrue(
                        np.all(texture[:, half_width:] == retained_color)
                    )
                combined_vertices = np.vstack(z_up_vertices)
                self.assertAlmostEqual(
                    float(np.min(combined_vertices[:, axis])),
                    -1.0,
                )
                self.assertAlmostEqual(
                    float(np.max(combined_vertices[:, axis])),
                    1.0,
                )
                self.assertTrue(
                    any(
                        maximum <= 0.5
                        for _minimum, maximum in uv_ranges
                    )
                )
                self.assertTrue(
                    any(
                        minimum >= 0.5
                        for minimum, _maximum in uv_ranges
                    )
                )

    def test_proxy_also_mirrors_untextured_geometry(self) -> None:
        retained_scene = trimesh.load(
            BytesIO(
                _uv_glb(
                    left_packed=True,
                    texture_resolution=1024,
                )
            ),
            file_type="glb",
            force="scene",
            process=False,
        )
        untextured_mesh = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
        untextured_transform = np.eye(4, dtype=float)
        untextured_transform[:3, 3] = (2.0, 0.25, 0.0)
        retained_scene.add_geometry(
            untextured_mesh,
            geom_name="untextured-part",
            node_name="untextured-node",
            transform=untextured_transform,
        )
        retained_glb = bytes(retained_scene.export(file_type="glb"))

        proxy_glb = build_symmetric_retexture_proxy_glb(
            retained_glb,
            SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
            0.0,
        )
        proxy_scene = trimesh.load(
            BytesIO(proxy_glb),
            file_type="glb",
            force="scene",
            process=False,
        )

        textured_mesh_count = sum(
            getattr(
                getattr(mesh.visual, "material", None),
                "baseColorTexture",
                None,
            )
            is not None
            for mesh in proxy_scene.geometry.values()
        )
        self.assertEqual(len(proxy_scene.geometry), 4)
        self.assertEqual(textured_mesh_count, 2)
        self.assertEqual(
            sum(len(mesh.faces) for mesh in proxy_scene.geometry.values()),
            28,
        )


class MeshyTextureRegeneratorTests(unittest.TestCase):
    def test_provider_sends_owned_model_reference_and_uv_flag(self) -> None:
        reference = np.zeros((5, 7, 4), dtype=np.uint8)
        reference[:, :] = (11, 22, 33, 177)
        model_glb = _uv_glb()
        fingerprint = build_uv_fingerprint(model_glb)
        request = TextureRegenerationRequest(
            object_id="chair",
            reference_frame_index=3,
            reference_image_bgra=reference,
            model_glb=model_glb,
            settings=GenerationServiceSettings(meshy_api_key="msy-secret"),
            enable_original_uv=True,
            submitted_uv_fingerprint=fingerprint,
            preserve_symmetric_uvs=True,
        )
        expected = MeshyGenerationResult("texture-task", model_glb, "Chair")
        progress_messages: list[str] = []

        def fake_retexture(**kwargs: object) -> MeshyGenerationResult:
            callback = kwargs["progress_callback"]
            callback("PENDING", 0)  # type: ignore[operator]
            callback("IN_PROGRESS", 47)  # type: ignore[operator]
            callback("SUCCEEDED", 100)  # type: ignore[operator]
            return expected

        with patch(
            "housemaker.generation_workspace.request_retextured_model",
            side_effect=fake_retexture,
        ) as retexture:
            actual = MeshyTextureRegenerator().regenerate(
                request,
                progress_messages.append,
                threading.Event(),
            )

        self.assertIs(actual, expected)
        self.assertEqual(retexture.call_args.kwargs["api_key"], "msy-secret")
        self.assertEqual(retexture.call_args.kwargs["model_glb"], model_glb)
        self.assertTrue(retexture.call_args.kwargs["enable_original_uv"])
        references = retexture.call_args.kwargs["reference_images_png"]
        self.assertEqual(len(references), 1)
        decoded = cv2.imdecode(
            np.frombuffer(references[0], dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        np.testing.assert_array_equal(decoded, reference)
        self.assertIn("queued", " ".join(progress_messages).lower())
        self.assertIn("47%", " ".join(progress_messages))
        self.assertIn("complete", " ".join(progress_messages).lower())

    def test_provider_does_not_preserve_uvs_when_request_disables_it(
        self,
    ) -> None:
        request = TextureRegenerationRequest(
            object_id="chair",
            reference_frame_index=0,
            reference_image_bgra=np.zeros((2, 2, 4), dtype=np.uint8),
            model_glb=b"ordinary model",
            settings=GenerationServiceSettings(meshy_api_key="msy-secret"),
        )
        expected = MeshyGenerationResult(
            "texture-task",
            b"regenerated model",
            "Chair",
        )

        with patch(
            "housemaker.generation_workspace.request_retextured_model",
            return_value=expected,
        ) as retexture:
            actual = MeshyTextureRegenerator().regenerate(request)

        self.assertIs(actual, expected)
        self.assertFalse(retexture.call_args.kwargs["enable_original_uv"])
        self.assertEqual(
            retexture.call_args.kwargs["model_glb"],
            b"ordinary model",
        )

    def test_provider_honors_pre_cancel_before_paid_request(self) -> None:
        request = TextureRegenerationRequest(
            object_id="chair",
            reference_frame_index=0,
            reference_image_bgra=np.zeros((2, 2, 4), dtype=np.uint8),
            model_glb=b"model",
            settings=GenerationServiceSettings(meshy_api_key="msy-secret"),
        )
        cancel_event = threading.Event()
        cancel_event.set()

        with patch(
            "housemaker.generation_workspace.request_retextured_model"
        ) as retexture, self.assertRaises(_GenerationCancelled):
            MeshyTextureRegenerator().regenerate(
                request,
                cancel_event=cancel_event,
            )

        retexture.assert_not_called()

    def test_worker_preserves_symmetric_geometry_when_meshy_remeshes(self) -> None:
        submitted_glb = _uv_glb(
            left_packed=True,
            texture_resolution=1024,
        )
        submitted_fingerprint = build_uv_fingerprint(submitted_glb)
        request = TextureRegenerationRequest(
            object_id="chair",
            reference_frame_index=0,
            reference_image_bgra=np.zeros((2, 2, 4), dtype=np.uint8),
            model_glb=submitted_glb,
            settings=GenerationServiceSettings(meshy_api_key="msy-secret"),
            enable_original_uv=True,
            submitted_uv_fingerprint=submitted_fingerprint,
            preserve_symmetric_uvs=True,
        )
        generated_color = (180, 35, 90, 255)
        provider_glb = build_symmetric_retexture_proxy_glb(
            _uv_glb(
                subdivide_faces=True,
                left_packed=True,
                texture_resolution=2048,
                texture_color=generated_color,
            ),
            SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
            0.0,
        )
        self.assertNotEqual(
            build_uv_fingerprint(provider_glb).face_count,
            submitted_fingerprint.face_count,
        )
        result = MeshyGenerationResult(
            "changed-task",
            provider_glb,
            "Changed",
        )
        regenerator = _SequenceTextureRegenerator([result])
        executor = MeshyModelExecutor()
        symmetry = SymmetricDivisionMetadata(
            orientation=SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
            kept_side=SYMMETRIC_DIVISION_SIDE_RIGHT,
            plane_coordinate=0.0,
        )
        worker = TextureRegenerationWorker(
            regenerator,
            executor,
            request,
            symmetry=symmetry,
        )
        succeeded = QSignalSpy(worker.succeeded)
        failed = QSignalSpy(worker.failed)
        finished = QSignalSpy(worker.finished)

        worker.run()

        self.assertEqual(succeeded.count(), 1)
        self.assertEqual(failed.count(), 0)
        self.assertEqual(finished.count(), 1)
        self.assertEqual(len(regenerator.requests), 1)
        provider_request = regenerator.requests[0]
        self.assertEqual(
            build_uv_fingerprint(provider_request.model_glb).face_count,
            submitted_fingerprint.face_count * 2,
        )
        np.testing.assert_array_equal(
            provider_request.reference_image_bgra,
            request.reference_image_bgra,
        )
        outcome = succeeded.at(0)[0]
        preserved_result = outcome.result
        self.assertEqual(
            build_uv_fingerprint(preserved_result.glb_bytes),
            submitted_fingerprint,
        )
        preserved_model = import_generated_glb(preserved_result.glb_bytes)
        self.assertEqual(len(preserved_model.mesh.faces), 2)
        preserved_texture = (
            preserved_model.mesh.visual.material.baseColorTexture
        )
        self.assertEqual(preserved_texture.size, (2048, 2048))
        self.assertEqual(preserved_texture.getpixel((0, 0)), generated_color)
        self.assertEqual(outcome.final_uv_fingerprint, submitted_fingerprint)
        generated_model = succeeded.at(0)[1]
        generated_variants = generated_model.object_texture_variants
        self.assertIsNotNone(generated_variants)
        assert generated_variants is not None
        for variant_glb in generated_variants.glb_by_resolution.values():
            self.assertEqual(
                build_uv_fingerprint(variant_glb),
                submitted_fingerprint,
            )

    def test_worker_rejects_symmetric_result_without_a_texture(self) -> None:
        submitted_glb = _uv_glb(texture_resolution=1024)
        request = TextureRegenerationRequest(
            object_id="chair",
            reference_frame_index=0,
            reference_image_bgra=np.zeros((2, 2, 4), dtype=np.uint8),
            model_glb=submitted_glb,
            settings=GenerationServiceSettings(meshy_api_key="msy-secret"),
            enable_original_uv=True,
            submitted_uv_fingerprint=build_uv_fingerprint(submitted_glb),
            preserve_symmetric_uvs=True,
        )
        result = MeshyGenerationResult(
            "untextured-task",
            _uv_glb(mutate_uv=True),
            "Untextured",
        )
        regenerator = _SequenceTextureRegenerator([result])
        executor = _SequenceExecutor(
            [_model_with_variants(_texture_variants(10))]
        )
        worker = TextureRegenerationWorker(regenerator, executor, request)
        succeeded = QSignalSpy(worker.succeeded)
        failed = QSignalSpy(worker.failed)

        worker.run()

        self.assertEqual(succeeded.count(), 0)
        self.assertEqual(failed.count(), 1)
        self.assertIn("no embedded base-color texture", failed.at(0)[0])
        self.assertEqual(executor.results, [])


# ### Regenerate-texture UI tests ###
class TextureRegenerationUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = (
            Path(self.temporary_directory.name) / "generated"
        )
        self.asset_directory.mkdir(parents=True)
        self.video_path = Path(self.temporary_directory.name) / "source.avi"
        _write_test_video(self.video_path)
        self.workspace = GenerationWorkspace(
            asset_directory=self.asset_directory,
        )
        self.workspace.resize(1400, 850)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _write_record(self, object_id: str) -> GeneratedObjectRecord:
        asset_name = f"{object_id}.glb"
        self.asset_directory.joinpath(asset_name).write_bytes(_box_glb())
        return _record(object_id, asset_name)

    def _enable_meshy(self) -> None:
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(meshy_api_key="msy-test-key")
        )

    def _load_video(self) -> None:
        self.workspace.load_video(str(self.video_path))

    def _paint_current_frame_mask(self) -> None:
        stroke = MaskStroke(
            mode=MASK_MODE_PAINT,
            radius_normalized=0.12,
            points=(MaskPoint(0.5, 0.5),),
        )
        self.workspace.video_view.set_strokes([stroke])
        self.workspace.video_view.strokes_changed.emit([stroke])
        _qt_application.processEvents()

    def test_button_is_beside_generate_and_tracks_selected_id(self) -> None:
        first = self._write_record("first")
        second = self._write_record("second")
        self._enable_meshy()
        self.workspace.set_data(
            GenerationData(generated_objects=[first, second])
        )
        self._load_video()
        self._paint_current_frame_mask()

        self.assertEqual(
            self.workspace.regenerate_texture_button.objectName(),
            "generate_texture_button",
        )
        self.assertEqual(
            self.workspace.regenerate_texture_button.text(),
            "Generate texture",
        )
        self.assertLess(
            self.workspace.generate_button.geometry().left(),
            self.workspace.regenerate_texture_button.geometry().left(),
        )
        self.assertIn(
            "credits",
            self.workspace.regenerate_texture_button.toolTip().lower(),
        )
        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())
        self.assertEqual(self.workspace._selected_object_id, "second")

        self.workspace.generated_objects_list.setCurrentRow(0)
        self.assertEqual(self.workspace._selected_object_id, "first")
        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())

    def test_eligibility_tracks_key_selection_video_mask_busy_and_delete(
        self,
    ) -> None:
        valid = self._write_record("valid")
        self.workspace.set_data(GenerationData(generated_objects=[valid]))

        self.assertFalse(self.workspace.regenerate_texture_button.isEnabled())

        self._enable_meshy()
        self.assertFalse(self.workspace.regenerate_texture_button.isEnabled())

        self._load_video()
        self.assertFalse(self.workspace.regenerate_texture_button.isEnabled())

        self._paint_current_frame_mask()
        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())

        self.workspace.generated_objects_list.setCurrentRow(-1)
        self.assertFalse(self.workspace.regenerate_texture_button.isEnabled())

        self.workspace.generated_objects_list.setCurrentRow(0)
        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())
        self.workspace.video_view.clear_mask()
        self.assertFalse(self.workspace.regenerate_texture_button.isEnabled())
        self._paint_current_frame_mask()
        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())
        with patch.object(
            self.workspace,
            "_generation_thread",
            object(),
        ):
            self.workspace._sync_controls()
            self.assertFalse(
                self.workspace.regenerate_texture_button.isEnabled()
            )
        self.workspace._sync_controls()

        self.assertTrue(self.workspace.delete_generated_object("valid"))
        self.assertFalse(self.workspace.regenerate_texture_button.isEnabled())

    def test_asset_location_is_not_a_ui_eligibility_gate(self) -> None:
        self._enable_meshy()
        missing = _record("missing", "missing.glb")
        self.workspace.set_data(GenerationData(generated_objects=[missing]))
        self._load_video()
        self._paint_current_frame_mask()

        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())

        outside_path = Path(self.temporary_directory.name) / "outside.glb"
        outside_path.write_bytes(_box_glb())
        unsafe = _record("unsafe", "../outside.glb")
        self.workspace.set_data(GenerationData(generated_objects=[unsafe]))
        self._load_video()
        self._paint_current_frame_mask()

        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())

    def test_external_mode_keeps_local_button_and_selected_id_current(self) -> None:
        first = self._write_record("first")
        second = self._write_record("second")
        self._enable_meshy()
        self.workspace.set_data(
            GenerationData(generated_objects=[first, second])
        )
        self._load_video()
        self._paint_current_frame_mask()

        self.assertFalse(hasattr(self.workspace, "ambient_light_slider"))
        self.assertEqual(
            self.workspace.result_view.get_ambient_light_intensity(),
            1.0,
        )

        self.workspace.set_external_3d_viewer_active(True)
        _qt_application.processEvents()

        self.assertTrue(
            self.workspace.object_3d_panel.is_external_presentation_active
        )
        self.assertIs(
            self.workspace.right_view_stack.currentWidget(),
            self.workspace.texture_view_page,
        )
        self.assertTrue(
            self.workspace.regenerate_texture_button.isVisibleTo(
                self.workspace
            )
        )
        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())
        self.assertEqual(
            self.workspace.result_view.get_ambient_light_intensity(),
            1.0,
        )

        self.workspace.generated_objects_list.setCurrentRow(0)
        self.assertEqual(self.workspace._selected_object_id, "first")
        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())

        self.workspace.set_external_3d_viewer_active(False)
        _qt_application.processEvents()

        self.assertEqual(
            self.workspace.result_view.get_ambient_light_intensity(),
            1.0,
        )
        self.assertIs(
            self.workspace.right_view_stack.currentWidget(),
            self.workspace.object_3d_page,
        )


# ### Regeneration pipeline tests ###
class TextureRegenerationPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = (
            Path(self.temporary_directory.name) / "generated"
        )
        self.asset_directory.mkdir(parents=True)
        self.video_path = Path(self.temporary_directory.name) / "source.avi"
        _write_test_video(self.video_path)
        self.workspace = GenerationWorkspace(
            asset_directory=self.asset_directory,
        )
        self.workspace.resize(1400, 850)
        self.workspace.show()
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(meshy_api_key="msy-test-key")
        )
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _seed_object(
        self,
        generation_index: int,
        *,
        name: str,
        task_id: str,
    ) -> tuple[GeneratedObjectRecord, ObjectTextureVariants]:
        variants = _texture_variants(generation_index)
        model = _model_with_variants(variants)
        result = MeshyGenerationResult(
            task_id,
            variants.glb_by_resolution[1024],
            name,
        )
        self.workspace._handle_generation_succeeded(result, model)
        return self.workspace.get_data().generated_objects[-1], variants

    def _replace_record(
        self,
        record: GeneratedObjectRecord,
        **pipeline_updates: object,
    ) -> GeneratedObjectRecord:
        pipeline = dict(record.pipeline)
        pipeline.update(pipeline_updates)
        replacement = replace(record, pipeline=pipeline)
        record_index = self.workspace._data.generated_objects.index(record)
        self.workspace._data.generated_objects[record_index] = replacement
        selected_id = self.workspace._selected_object_id
        self.workspace._refresh_generated_objects_list(selected_id)
        return replacement

    def _load_reference(self, frame_index: int = 0) -> np.ndarray:
        self.workspace.load_video(str(self.video_path))
        if frame_index:
            self.workspace.show_frame(frame_index)
        stroke = MaskStroke(
            mode=MASK_MODE_PAINT,
            radius_normalized=0.12,
            points=(MaskPoint(0.5, 0.5),),
        )
        self.workspace.video_view.set_strokes([stroke])
        self.workspace.video_view.strokes_changed.emit([stroke])
        _qt_application.processEvents()
        return self.workspace.video_view.build_selected_object_crop()

    def _wait_until_idle(self, timeout_seconds: float = 8.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while self.workspace.is_generating and time.monotonic() < deadline:
            _qt_application.processEvents()
            QTest.qWait(10)
        _qt_application.processEvents()
        self.assertFalse(
            self.workspace.is_generating,
            "Texture-regeneration worker did not finish.",
        )

    def _wait_for_event(
        self,
        event: threading.Event,
        timeout_seconds: float = 5.0,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while not event.is_set() and time.monotonic() < deadline:
            _qt_application.processEvents()
            QTest.qWait(10)
        self.assertTrue(event.is_set(), "Blocking provider did not start.")

    def test_global_resolution_selection_updates_object_generation_once(
        self,
    ) -> None:
        record, variants = self._seed_object(
            0,
            name="Chair",
            task_id="geometry-task",
        )
        data_changed = QSignalSpy(self.workspace.data_changed)
        content_changed = QSignalSpy(self.workspace.generated_object_changed)

        self.assertTrue(
            self.workspace.select_object_texture_resolution(
                record.object_id,
                2048,
            )
        )

        updated = self.workspace.get_data().generated_objects[0]
        self.assertEqual(updated.pipeline["selected_texture_resolution"], 2048)
        self.assertEqual(
            updated.asset_path,
            updated.pipeline["texture_variants"]["2048"]["glb_asset_path"],
        )
        self.assertEqual(
            self.workspace.result_view.model.glb_bytes,
            variants.glb_by_resolution[2048],
        )
        self.assertEqual(
            self.workspace.texture_view.selected_atlas_id,
            f"{record.object_id}:resolution:2048",
        )
        self.assertEqual(data_changed.count(), 1)
        self.assertEqual(content_changed.count(), 0)

        self.assertTrue(
            self.workspace.select_object_texture_resolution(
                record.object_id,
                2048,
            )
        )
        self.assertEqual(data_changed.count(), 1)
        self.assertFalse(
            self.workspace.select_object_texture_resolution(
                record.object_id,
                4096,
            )
        )
        self.assertEqual(data_changed.count(), 1)

    def test_legacy_camera_uv_metadata_is_preserved_but_not_activated(
        self,
    ) -> None:
        record, _variants = self._seed_object(
            0,
            name="Chair",
            task_id="geometry-task",
        )
        self._load_reference(frame_index=1)
        postprocessed_glb = _uv_glb()
        postprocessed_name = f"{record.object_id}.postprocessed.glb"
        self.asset_directory.joinpath(postprocessed_name).write_bytes(
            postprocessed_glb
        )
        legacy_pipeline = {
            "postprocessed_asset_path": postprocessed_name,
            "camera_uv_projection_applied": True,
            "camera_uv_submitted_fingerprint": "a" * 64,
            "camera_uv_integrity_face_count": 2,
            "retained_provenance": "keep-me",
        }
        replacement = self._replace_record(record, **legacy_pipeline)

        request = self.workspace._build_texture_regeneration_request()

        self.assertIsNotNone(request)
        assert request is not None
        self.assertIsInstance(request, _TextureRegenerationPreflight)
        self.assertEqual(request.source_asset_path, postprocessed_name)
        materialized = _materialize_texture_regeneration_preflight(
            request,
            self.asset_directory,
        ).request
        self.assertEqual(materialized.model_glb, postprocessed_glb)
        self.assertFalse(materialized.enable_original_uv)
        self.assertFalse(materialized.preserve_symmetric_uvs)
        self.assertIsNone(materialized.submitted_uv_fingerprint)
        restored = GenerationData.from_dict(self.workspace.get_data().to_dict())
        for key, value in legacy_pipeline.items():
            self.assertEqual(replacement.pipeline[key], value)
            self.assertEqual(restored.generated_objects[0].pipeline[key], value)

    def test_source_read_import_and_symmetric_fingerprint_run_off_gui(
        self,
    ) -> None:
        record, _variants = self._seed_object(
            0,
            name="Chair",
            task_id="geometry-task",
        )
        full_reference = self._load_reference()
        source_asset_path = record.pipeline["texture_variants"]["1024"][
            "glb_asset_path"
        ]
        source_glb = _uv_glb(
            left_packed=True,
            texture_resolution=1024,
        )
        self.asset_directory.joinpath(source_asset_path).write_bytes(
            source_glb
        )
        record = self._replace_record(
            record,
            symmetric_division={
                "version": 1,
                "orientation": "vertical",
                "kept_side": "left",
                "plane_coordinate": 0.0,
                "texture_content_half": "left",
            },
        )
        regenerator = _SequenceTextureRegenerator(
            [RuntimeError("stop after preflight")]
        )
        self.workspace.set_texture_regenerator(regenerator)
        self.workspace.set_meshy_executor(_SequenceExecutor([]))
        gui_thread = _qt_application.thread()
        read_threads: list[QThread] = []
        import_threads: list[QThread] = []
        fingerprint_threads: list[QThread] = []
        original_read_bytes = Path.read_bytes
        original_import = import_generated_glb
        original_fingerprint = build_uv_fingerprint

        def read_bytes_with_thread(path: Path) -> bytes:
            read_threads.append(QThread.currentThread())
            return original_read_bytes(path)

        def import_with_thread(payload: bytes) -> GeneratedModel:
            import_threads.append(QThread.currentThread())
            return original_import(payload)

        def fingerprint_with_thread(payload: bytes) -> UvFingerprint:
            fingerprint_threads.append(QThread.currentThread())
            return original_fingerprint(payload)

        with (
            patch.object(Path, "read_bytes", read_bytes_with_thread),
            patch(
                "housemaker.generation_workspace.import_generated_glb",
                side_effect=import_with_thread,
            ),
            patch(
                "housemaker.generation_workspace.build_uv_fingerprint",
                side_effect=fingerprint_with_thread,
            ),
            patch.object(QMessageBox, "warning"),
        ):
            self.assertTrue(
                self.workspace.generate_selected_object_texture()
            )
            self._wait_until_idle()

        self.assertEqual(len(regenerator.requests), 1)
        self.assertNotEqual(regenerator.requests[0].model_glb, source_glb)
        self.assertEqual(
            build_uv_fingerprint(
                regenerator.requests[0].model_glb
            ).face_count,
            build_uv_fingerprint(source_glb).face_count * 2,
        )
        np.testing.assert_array_equal(
            regenerator.requests[0].reference_image_bgra,
            full_reference,
        )
        self.assertIsNotNone(
            regenerator.requests[0].submitted_uv_fingerprint
        )
        self.assertTrue(read_threads)
        self.assertTrue(import_threads)
        self.assertTrue(fingerprint_threads)
        self.assertTrue(
            all(thread != gui_thread for thread in read_threads)
        )
        self.assertTrue(
            all(thread != gui_thread for thread in import_threads)
        )
        self.assertTrue(
            all(thread != gui_thread for thread in fingerprint_threads)
        )
        self.assertEqual(
            self.workspace.get_data().generated_objects[0],
            record,
        )

    def test_symmetric_remesh_saves_only_the_new_texture_on_old_geometry(
        self,
    ) -> None:
        record, _variants = self._seed_object(
            0,
            name="Chair",
            task_id="geometry-task",
        )
        full_reference = self._load_reference()
        source_glb = _uv_glb(
            left_packed=True,
            texture_resolution=1024,
        )
        source_fingerprint = build_uv_fingerprint(source_glb)
        source_asset_path = record.pipeline["texture_variants"]["1024"][
            "glb_asset_path"
        ]
        self.asset_directory.joinpath(source_asset_path).write_bytes(
            source_glb
        )
        symmetry = SymmetricDivisionMetadata(
            version=AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION,
            orientation=SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
            kept_side=SYMMETRIC_DIVISION_SIDE_LEFT,
            plane_coordinate=0.0,
            packing_mode=SYMMETRIC_TEXTURE_PACKING_MODE_PAIR,
            texture_content_half=SYMMETRIC_TEXTURE_CONTENT_HALF_LEFT,
            selection_mode=(
                SYMMETRIC_SELECTION_MODE_FEWEST_TRIANGLES_RANDOM_TIE
            ),
            triangle_count_by_side=(("left", 2), ("right", 5)),
            tie_broken_randomly=False,
        )
        record = self._replace_record(
            record,
            postprocessed_asset_path=source_asset_path,
            symmetric_division=symmetry.to_pipeline_dict(),
            selected_texture_resolution=1024,
        )
        generated_color = (210, 65, 25, 255)
        provider_glb = build_symmetric_retexture_proxy_glb(
            _uv_glb(
                subdivide_faces=True,
                left_packed=True,
                texture_resolution=2048,
                texture_color=generated_color,
            ),
            SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
            0.0,
        )
        self.assertEqual(build_uv_fingerprint(provider_glb).face_count, 8)
        regenerator = _SequenceTextureRegenerator(
            [
                MeshyGenerationResult(
                    "remeshed-texture-task",
                    provider_glb,
                    "Chair",
                )
            ]
        )
        self.workspace.set_texture_regenerator(regenerator)
        self.workspace.set_meshy_executor(MeshyModelExecutor())

        self.assertTrue(self.workspace.generate_selected_object_texture())
        self._wait_until_idle()

        replacement = self.workspace.get_data().generated_objects[0]
        self.assertEqual(len(regenerator.requests), 1)
        np.testing.assert_array_equal(
            regenerator.requests[0].reference_image_bgra,
            full_reference,
        )
        self.assertEqual(
            build_uv_fingerprint(
                regenerator.requests[0].model_glb
            ).face_count,
            source_fingerprint.face_count * 2,
        )
        self.assertEqual(
            replacement.provider_task_id,
            "remeshed-texture-task",
        )
        self.assertEqual(
            replacement.pipeline["symmetric_division"],
            record.pipeline["symmetric_division"],
        )
        self.assertEqual(
            set(replacement.pipeline["texture_variants"]),
            {"512", "1024"},
        )
        self.assertEqual(
            replacement.pipeline["texture_regeneration_uv_face_count"],
            source_fingerprint.face_count,
        )
        for variant in replacement.pipeline["texture_variants"].values():
            variant_glb = self.asset_directory.joinpath(
                variant["glb_asset_path"]
            ).read_bytes()
            self.assertEqual(
                build_uv_fingerprint(variant_glb),
                source_fingerprint,
            )
            texture = np.asarray(
                Image.open(
                    self.asset_directory / variant["texture_asset_path"]
                ).convert("RGBA"),
                dtype=np.uint8,
            )
            half_width = texture.shape[1] // 2
            self.assertTrue(
                np.all(texture[:, half_width:] == (0, 0, 0, 255))
            )
            self.assertGreater(
                np.count_nonzero(
                    np.all(texture[:, :half_width] == generated_color, axis=2)
                ),
                0,
            )

    def test_success_replaces_variants_repeats_and_refreshes_external_view(
        self,
    ) -> None:
        original, original_variants = self._seed_object(
            0,
            name="Chair",
            task_id="original-task",
        )
        original = self._replace_record(
            original,
            texture_inpaint_strokes=[
                {
                    "mode": "paint",
                    "radius_normalized": 0.04,
                    "points": [{"u": 0.25, "v": 0.75}],
                }
            ],
            retained_provenance={"geometry": "unchanged"},
        )
        self.workspace.texture_view.select_atlas(
            f"{original.object_id}:resolution:2048"
        )
        original = self.workspace.get_data().generated_objects[0]
        original_paths = _record_variant_paths(original)
        original_bytes = _asset_bytes(self.asset_directory, original_paths)
        expected_reference = self._load_reference(frame_index=1)
        first_variants = _texture_variants(1)
        second_variants = _texture_variants(2)
        first_result = MeshyGenerationResult(
            "regenerated-task-1",
            first_variants.glb_by_resolution[1024],
            "Chair",
        )
        second_result = MeshyGenerationResult(
            "regenerated-task-2",
            second_variants.glb_by_resolution[1024],
            "Chair",
        )
        regenerator = _SequenceTextureRegenerator(
            [first_result, second_result]
        )
        executor = _SequenceExecutor(
            [
                _model_with_variants(first_variants),
                _model_with_variants(second_variants),
            ]
        )
        self.workspace.set_texture_regenerator(regenerator)
        self.workspace.set_meshy_executor(executor)
        self.workspace.wireframe_checkbox.setChecked(True)
        self.workspace.set_external_3d_viewer_active(True)
        changed = QSignalSpy(self.workspace.data_changed)
        completed = QSignalSpy(
            self.workspace.texture_regeneration_completed
        )
        generated = QSignalSpy(self.workspace.generation_completed)

        self.assertTrue(self.workspace.regenerate_selected_object_texture())
        self._wait_until_idle()

        self.assertEqual(len(regenerator.requests), 1)
        first_request = regenerator.requests[0]
        self.assertEqual(first_request.object_id, original.object_id)
        self.assertEqual(
            first_request.model_glb,
            original_variants.glb_by_resolution[1024],
        )
        np.testing.assert_array_equal(
            first_request.reference_image_bgra,
            expected_reference,
        )
        first_record = self.workspace.get_data().generated_objects[0]
        first_paths = _record_variant_paths(first_record)
        self.assertEqual(first_record.object_id, original.object_id)
        self.assertEqual(first_record.object_name, original.object_name)
        self.assertEqual(first_record.frame_index, original.frame_index)
        self.assertEqual(first_record.provider_task_id, "regenerated-task-1")
        self.assertEqual(
            first_record.pipeline["retained_provenance"],
            {"geometry": "unchanged"},
        )
        self.assertNotIn(
            TEXTURE_INPAINT_STROKES_PIPELINE_KEY,
            first_record.pipeline,
        )
        self.assertEqual(
            first_record.pipeline["selected_texture_resolution"],
            2048,
        )
        self.assertEqual(first_record.pipeline["texture_regeneration_count"], 1)
        self.assertEqual(
            first_record.pipeline["latest_texture_task_id"],
            "regenerated-task-1",
        )
        self.assertTrue(first_paths.isdisjoint(original_paths))
        self.assertTrue(
            all(path.startswith("regenerated-") for path in first_paths)
        )
        self.assertTrue(
            all(self.asset_directory.joinpath(path).is_file() for path in original_paths)
        )
        self.assertNotEqual(
            _asset_bytes(self.asset_directory, first_paths),
            original_bytes,
        )
        self.assertEqual(
            self.workspace.result_view.model.glb_bytes,
            first_variants.glb_by_resolution[2048],
        )
        self.assertEqual(len(self.workspace.texture_view.entries), 3)
        self.assertEqual(
            self.workspace.texture_view.selected_atlas_id,
            f"{original.object_id}:resolution:2048",
        )
        self.assertTrue(self.workspace.texture_view.uv_overlay_enabled)
        self.assertTrue(self.workspace.result_view.get_wireframe_enabled())
        self.assertIs(
            self.workspace.right_view_stack.currentWidget(),
            self.workspace.texture_view_page,
        )

        self.assertTrue(self.workspace.regenerate_selected_object_texture())
        self._wait_until_idle()

        self.assertEqual(len(regenerator.requests), 2)
        self.assertEqual(
            regenerator.requests[1].model_glb,
            first_variants.glb_by_resolution[1024],
        )
        second_record = self.workspace.get_data().generated_objects[0]
        second_paths = _record_variant_paths(second_record)
        self.assertEqual(second_record.provider_task_id, "regenerated-task-2")
        self.assertEqual(second_record.pipeline["texture_regeneration_count"], 2)
        self.assertEqual(
            [
                entry["task_id"]
                for entry in second_record.pipeline["texture_regeneration_history"]
            ],
            ["regenerated-task-1", "regenerated-task-2"],
        )
        self.assertTrue(second_paths.isdisjoint(first_paths))
        self.assertTrue(
            all(self.asset_directory.joinpath(path).is_file() for path in first_paths)
        )
        self.assertTrue(
            all(self.asset_directory.joinpath(path).is_file() for path in second_paths)
        )
        self.assertEqual(changed.count(), 2)
        self.assertEqual(completed.count(), 2)
        self.assertEqual(generated.count(), 0)

        saved = self.workspace.get_data()
        self.workspace.set_data(saved)
        restored = self.workspace.get_data().generated_objects[0]
        self.assertEqual(restored, second_record)
        active = self.workspace.get_active_texture_variant(restored.object_id)
        self.assertIsNotNone(active)
        assert active is not None
        self.assertEqual(active.resolution, 2048)
        self.assertEqual(active.glb_asset_path.read_bytes(), second_variants.glb_by_resolution[2048])

    def test_undo_restores_texture_record_resolution_and_provenance(self) -> None:
        original, original_variants = self._seed_object(
            0,
            name="Chair",
            task_id="geometry-task",
        )
        self.workspace.texture_view.select_atlas(
            f"{original.object_id}:resolution:2048"
        )
        original = self.workspace.get_data().generated_objects[0]
        original_paths = _record_variant_paths(original)
        self._load_reference()

        generated_variants = _texture_variants(9)
        self.workspace.set_texture_regenerator(
            _SequenceTextureRegenerator(
                [
                    MeshyGenerationResult(
                        "texture-task",
                        generated_variants.glb_by_resolution[1024],
                        "Chair",
                    )
                ]
            )
        )
        self.workspace.set_meshy_executor(
            _SequenceExecutor([_model_with_variants(generated_variants)])
        )
        changed = QSignalSpy(self.workspace.generated_object_changed)

        self.assertTrue(self.workspace.generate_selected_object_texture())
        self._wait_until_idle()

        generated = self.workspace.get_data().generated_objects[0]
        generated_paths = _record_variant_paths(generated)
        self.assertEqual(generated.provider_task_id, "texture-task")
        self.assertTrue(self.workspace.undo_object_change_button.isEnabled())
        self.assertTrue(
            all(self.asset_directory.joinpath(path).is_file() for path in original_paths)
        )

        self.assertTrue(self.workspace.undo_selected_object_change())

        restored = self.workspace.get_data().generated_objects[0]
        self.assertEqual(restored, original)
        self.assertEqual(
            self.workspace.result_view.model.glb_bytes,
            original_variants.glb_by_resolution[2048],
        )
        self.assertTrue(
            all(self.asset_directory.joinpath(path).is_file() for path in original_paths)
        )
        self.assertTrue(
            all(not self.asset_directory.joinpath(path).exists() for path in generated_paths)
        )
        self.assertFalse(self.workspace.undo_object_change_button.isEnabled())
        self.assertEqual(changed.count(), 2)

    def test_selection_change_during_task_updates_target_without_hijacking_view(
        self,
    ) -> None:
        first, _first_variants = self._seed_object(
            0,
            name="First",
            task_id="first-task",
        )
        second, second_variants = self._seed_object(
            5,
            name="Second",
            task_id="second-task",
        )
        self.workspace.generated_objects_list.setCurrentRow(0)
        self.assertEqual(self.workspace._selected_object_id, first.object_id)
        first = self.workspace.get_data().generated_objects[0]
        second = self.workspace.get_data().generated_objects[1]
        first_old_paths = _record_variant_paths(first)
        second_old_paths = _record_variant_paths(second)
        second_old_bytes = _asset_bytes(self.asset_directory, second_old_paths)
        self._load_reference()

        new_variants = _texture_variants(6)
        result = MeshyGenerationResult(
            "first-regenerated",
            new_variants.glb_by_resolution[1024],
            "First",
        )
        regenerator = _BlockingTextureRegenerator(result)
        executor = _SequenceExecutor([_model_with_variants(new_variants)])
        self.workspace.set_texture_regenerator(regenerator)
        self.workspace.set_meshy_executor(executor)
        self.workspace.set_external_3d_viewer_active(True)
        completed = QSignalSpy(
            self.workspace.texture_regeneration_completed
        )

        self.assertTrue(self.workspace.regenerate_selected_object_texture())
        self._wait_for_event(regenerator.started)
        self.workspace.generated_objects_list.setCurrentRow(1)
        self.assertEqual(self.workspace._selected_object_id, second.object_id)
        regenerator.release.set()
        self._wait_until_idle()

        records = self.workspace.get_data().generated_objects
        self.assertEqual([record.object_id for record in records], [first.object_id, second.object_id])
        self.assertEqual(records[0].provider_task_id, "first-regenerated")
        self.assertEqual(records[1], second)
        self.assertEqual(self.workspace._selected_object_id, second.object_id)
        self.assertEqual(
            self.workspace.result_view.model.glb_bytes,
            second_variants.glb_by_resolution[1024],
        )
        self.assertEqual(
            _asset_bytes(self.asset_directory, second_old_paths),
            second_old_bytes,
        )
        self.assertTrue(
            all(self.asset_directory.joinpath(path).is_file() for path in first_old_paths)
        )
        self.assertEqual(completed.count(), 1)
        self.assertEqual(completed.at(0)[0].object_id, first.object_id)

    def test_source_revision_change_rejects_prepared_texture_commit(
        self,
    ) -> None:
        original, _original_variants = self._seed_object(
            0,
            name="Chair",
            task_id="original-task",
        )
        self._load_reference()
        original = self.workspace.get_data().generated_objects[0]
        original_paths = _record_variant_paths(original)
        new_variants = _texture_variants(8)
        regenerator = _BlockingTextureRegenerator(
            MeshyGenerationResult(
                "new-texture-task",
                new_variants.glb_by_resolution[1024],
                "Chair",
            )
        )
        self.workspace.set_texture_regenerator(regenerator)
        self.workspace.set_meshy_executor(
            _SequenceExecutor([_model_with_variants(new_variants)])
        )

        with patch.object(QMessageBox, "warning"):
            self.assertTrue(
                self.workspace.generate_selected_object_texture()
            )
            self._wait_for_event(regenerator.started)
            source_asset_path = original.pipeline["texture_variants"][
                "1024"
            ]["glb_asset_path"]
            self.asset_directory.joinpath(source_asset_path).write_bytes(
                _box_glb(99.0)
            )
            regenerator.release.set()
            self._wait_until_idle()

        self.assertEqual(
            self.workspace.get_data().generated_objects[0],
            original,
        )
        self.assertEqual(
            {path.name for path in self.asset_directory.iterdir()},
            original_paths,
        )
        self.assertIn(
            "source model changed",
            self.workspace.status_label.text(),
        )

    def test_provider_failure_keeps_record_files_cache_and_display(self) -> None:
        record, _variants = self._seed_object(
            0,
            name="Chair",
            task_id="original-task",
        )
        self._load_reference()
        record = self.workspace.get_data().generated_objects[0]
        old_paths = _record_variant_paths(record)
        old_bytes = _asset_bytes(self.asset_directory, old_paths)
        old_model = self.workspace.result_view.model
        regenerator = _SequenceTextureRegenerator(
            [RuntimeError("provider rejected reference")]
        )
        executor = _SequenceExecutor([])
        self.workspace.set_texture_regenerator(regenerator)
        self.workspace.set_meshy_executor(executor)
        changed = QSignalSpy(self.workspace.data_changed)
        completed = QSignalSpy(
            self.workspace.texture_regeneration_completed
        )

        with patch.object(QMessageBox, "warning") as warning:
            self.assertTrue(self.workspace.regenerate_selected_object_texture())
            self._wait_until_idle()

        self.assertEqual(self.workspace.get_data().generated_objects[0], record)
        self.assertEqual(_asset_bytes(self.asset_directory, old_paths), old_bytes)
        self.assertEqual(
            {path.name for path in self.asset_directory.iterdir()},
            old_paths,
        )
        self.assertIs(self.workspace.result_view.model, old_model)
        self.assertEqual(executor.results, [])
        self.assertEqual(changed.count(), 0)
        self.assertEqual(completed.count(), 0)
        warning.assert_called_once()
        self.assertIn("provider rejected", self.workspace.status_label.text())

    def test_partial_write_and_post_persist_import_failures_roll_back(self) -> None:
        failure_modes = ("write", "import")
        for failure_mode in failure_modes:
            with self.subTest(failure_mode=failure_mode):
                self.workspace.set_data(None)
                for path in tuple(self.asset_directory.iterdir()):
                    path.unlink()
                record, _variants = self._seed_object(
                    0,
                    name="Chair",
                    task_id="original-task",
                )
                self._load_reference()
                record = self.workspace.get_data().generated_objects[0]
                old_paths = _record_variant_paths(record)
                old_bytes = _asset_bytes(self.asset_directory, old_paths)
                new_variants = _texture_variants(4)
                result = MeshyGenerationResult(
                    "regenerated-task",
                    new_variants.glb_by_resolution[1024],
                    "Chair",
                )
                regenerator = _SequenceTextureRegenerator([result])
                executor = _SequenceExecutor(
                    [_model_with_variants(new_variants)]
                )
                self.workspace.set_texture_regenerator(regenerator)
                self.workspace.set_meshy_executor(executor)
                original_persist = _persist_generated_named_asset
                write_count = 0
                import_count = 0

                def persist_or_fail(
                    asset_directory: Path,
                    file_name: str,
                    payload: bytes,
                ) -> str:
                    nonlocal write_count
                    write_count += 1
                    if failure_mode == "write" and write_count == 4:
                        raise OSError("injected fourth write failure")
                    return original_persist(
                        asset_directory,
                        file_name,
                        payload,
                    )

                def import_or_fail(payload: bytes) -> GeneratedModel:
                    nonlocal import_count
                    import_count += 1
                    if failure_mode == "import" and import_count == 2:
                        raise RuntimeError("injected reimport failure")
                    return import_generated_glb(payload)

                import_patch = patch(
                    "housemaker.generation_workspace.import_generated_glb",
                    side_effect=import_or_fail,
                )
                with (
                    patch(
                        "housemaker.generation_workspace."
                        "_persist_generated_named_asset",
                        side_effect=persist_or_fail,
                    ),
                    import_patch,
                    patch.object(QMessageBox, "warning") as warning,
                ):
                    self.assertTrue(
                        self.workspace.regenerate_selected_object_texture()
                    )
                    self._wait_until_idle()

                self.assertEqual(
                    self.workspace.get_data().generated_objects[0],
                    record,
                )
                self.assertEqual(
                    _asset_bytes(self.asset_directory, old_paths),
                    old_bytes,
                )
                self.assertEqual(
                    {path.name for path in self.asset_directory.iterdir()},
                    old_paths,
                )
                warning.assert_called_once()

    def test_shutdown_cancellation_keeps_existing_transaction(self) -> None:
        record, _variants = self._seed_object(
            0,
            name="Chair",
            task_id="original-task",
        )
        self._load_reference()
        record = self.workspace.get_data().generated_objects[0]
        old_paths = _record_variant_paths(record)
        old_bytes = _asset_bytes(self.asset_directory, old_paths)
        new_variants = _texture_variants(8)
        result = MeshyGenerationResult(
            "late-task",
            new_variants.glb_by_resolution[1024],
            "Chair",
        )
        regenerator = _BlockingTextureRegenerator(result)
        executor = _SequenceExecutor([_model_with_variants(new_variants)])
        self.workspace.set_texture_regenerator(regenerator)
        self.workspace.set_meshy_executor(executor)
        changed = QSignalSpy(self.workspace.data_changed)
        completed = QSignalSpy(
            self.workspace.texture_regeneration_completed
        )

        self.assertTrue(self.workspace.regenerate_selected_object_texture())
        self._wait_for_event(regenerator.started)
        self.workspace.shutdown()
        regenerator.release.set()
        QTest.qWait(30)
        _qt_application.processEvents()

        self.assertFalse(self.workspace.is_generating)
        self.assertEqual(self.workspace.get_data().generated_objects[0], record)
        self.assertEqual(_asset_bytes(self.asset_directory, old_paths), old_bytes)
        self.assertEqual(
            {path.name for path in self.asset_directory.iterdir()},
            old_paths,
        )
        self.assertEqual(executor.results, [])
        self.assertEqual(changed.count(), 0)
        self.assertEqual(completed.count(), 0)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
