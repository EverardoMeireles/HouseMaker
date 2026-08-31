# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from housemaker.uv_integrity import (
    UV_FINGERPRINT_VERSION,
    UvFingerprint,
)
from housemaker.generation_state import GeneratedObjectRecord
from housemaker.generation_workspace import (
    LOCALLY_AUTHORED_UVS_PIPELINE_KEY,
    OBJECT_OPERATION_UNDO_STACK_PIPELINE_KEY,
    SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY,
    SCAN_PROJECTION_PIPELINE_KEY,
    SYMMETRIC_DIVISION_PIPELINE_KEY,
    TEXTURE_VARIANTS_PIPELINE_KEY,
    VISIBILITY_UV_UNWRAP_PIPELINE_KEY,
    GenerationRequest,
    GenerationWorkspace,
    StagedMeshyGenerationResult,
    TextureRegenerationOutcome,
    TextureRegenerationRequest,
    _prepare_and_persist_object_generation,
)
from housemaker.glb import import_generated_glb
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.object_symmetry import (
    AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION,
    LEGACY_SYMMETRIC_PAIR_METADATA_VERSION,
    SYMMETRIC_QUARTER_METADATA_VERSION,
    SYMMETRIC_DIVISION_METADATA_VERSION,
    SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL,
    SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
    SYMMETRIC_DIVISION_SIDE_BOTTOM,
    SYMMETRIC_DIVISION_SIDE_LEFT,
    SYMMETRIC_SELECTION_MODE_FEWEST_TRIANGLES_RANDOM_TIE,
    SYMMETRIC_TEXTURE_CONTENT_QUADRANT_TOP_LEFT,
    SYMMETRIC_TEXTURE_CONTENT_HALF_LEFT,
    SYMMETRIC_TEXTURE_PACKING_MODE_PAIR,
    SYMMETRIC_TEXTURE_PACKING_MODE_TOP_LEFT_QUARTER,
    SymmetricDivisionMetadata,
    SymmetricDivisionResult,
    SymmetricPairTextureVariants,
    SymmetricQuarterTextureVariants,
    SymmetricSquarePairTextureVariants,
)
from housemaker.object_texture_variants import (
    TEXTURE_RESOLUTIONS,
    ObjectTextureVariants,
)
from housemaker.object_uv_scan_projection import (
    LEFT_HALF_OUTER_SAFETY_INSET_PIXELS,
    SCAN_PROJECTION_TARGET_LEFT_HALF,
    SCAN_PROJECTION_VERSION,
    ScanProjectionStats,
)
from housemaker.settings_widget import GenerationServiceSettings


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])


# ### Projection fixtures ###
_TEST_CAMERA_PERCENTAGES = (30, 20, 15, 15, 10, 10)


# ### Fixture helpers ###
def _box_glb(scale: float) -> bytes:
    mesh = trimesh.creation.box(
        extents=(float(scale), float(scale) * 0.7, float(scale) * 0.8)
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _png_bytes(color: tuple[int, int, int, int]) -> bytes:
    pixels = np.full((8, 8, 4), color, dtype=np.uint8)
    encoded, payload = cv2.imencode(".png", pixels)
    if not encoded:
        raise RuntimeError("Test PNG encoding failed.")
    return bytes(payload)


def _ordinary_variants(seed: int) -> ObjectTextureVariants:
    return ObjectTextureVariants(
        glb_by_resolution={
            resolution: _box_glb(seed + resolution / 10_000.0)
            for resolution in TEXTURE_RESOLUTIONS
        },
        texture_png_by_resolution={
            resolution: _png_bytes(
                (seed * 23 % 255, resolution // 16 % 255, 80, 255)
            )
            for resolution in TEXTURE_RESOLUTIONS
        },
        preview_rgba_by_resolution={
            resolution: np.full(
                (8, 8, 4),
                (seed * 23 % 255, 90, resolution // 16 % 255, 255),
                dtype=np.uint8,
            )
            for resolution in TEXTURE_RESOLUTIONS
        },
    )


def _quarter_variants(seed: int) -> SymmetricQuarterTextureVariants:
    return SymmetricQuarterTextureVariants(
        glb_by_resolution={
            512: _box_glb(seed + 0.512),
            1024: _box_glb(seed + 1.024),
        },
        texture_png_by_resolution={
            512: _png_bytes((seed * 29 % 255, 50, 120, 255)),
            1024: _png_bytes((seed * 31 % 255, 80, 150, 255)),
        },
        preview_rgba_by_resolution={
            512: np.zeros((1024, 1024, 4), dtype=np.uint8),
            1024: np.zeros((2048, 2048, 4), dtype=np.uint8),
        },
    )


def _legacy_pair_variants(seed: int) -> SymmetricPairTextureVariants:
    return SymmetricPairTextureVariants(
        glb_by_resolution={
            512: _box_glb(seed + 0.512),
            1024: _box_glb(seed + 1.024),
        },
        texture_png_by_resolution={
            512: _png_bytes((seed * 37 % 255, 60, 130, 255)),
            1024: _png_bytes((seed * 41 % 255, 90, 160, 255)),
        },
        preview_rgba_by_resolution={
            512: np.zeros((1024, 1024, 4), dtype=np.uint8),
            1024: np.zeros((2048, 2048, 4), dtype=np.uint8),
        },
    )


def _square_pair_variants(seed: int) -> SymmetricSquarePairTextureVariants:
    return SymmetricSquarePairTextureVariants(
        glb_by_resolution={
            512: _box_glb(seed + 0.512),
            1024: _box_glb(seed + 1.024),
        },
        texture_png_by_resolution={
            512: _png_bytes((seed * 43 % 255, 70, 140, 255)),
            1024: _png_bytes((seed * 47 % 255, 100, 170, 255)),
        },
        preview_rgba_by_resolution={
            512: np.zeros((512, 512, 4), dtype=np.uint8),
            1024: np.zeros((1024, 1024, 4), dtype=np.uint8),
        },
    )


def _automatic_result(
    variants: SymmetricSquarePairTextureVariants,
    *,
    orientation: str = SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
    scan_projection_stats: ScanProjectionStats | None = None,
) -> SymmetricDivisionResult:
    if orientation == SYMMETRIC_DIVISION_ORIENTATION_VERTICAL:
        kept_side = SYMMETRIC_DIVISION_SIDE_LEFT
        counts = (("left", 4), ("right", 7))
    else:
        kept_side = SYMMETRIC_DIVISION_SIDE_BOTTOM
        counts = (("bottom", 3), ("top", 6))
    metadata = SymmetricDivisionMetadata(
        version=AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION,
        orientation=orientation,
        kept_side=kept_side,
        plane_coordinate=0.125,
        packing_mode=SYMMETRIC_TEXTURE_PACKING_MODE_PAIR,
        texture_content_half=SYMMETRIC_TEXTURE_CONTENT_HALF_LEFT,
        selection_mode=(
            SYMMETRIC_SELECTION_MODE_FEWEST_TRIANGLES_RANDOM_TIE
        ),
        triangle_count_by_side=counts,
        tie_broken_randomly=False,
    )
    return SymmetricDivisionResult(
        variants=variants,
        orientation=orientation,
        kept_side=kept_side,
        plane_coordinate=metadata.plane_coordinate,
        metadata=metadata,
        scan_projection_stats=scan_projection_stats,
    )


def _symmetric_scan_projection_stats() -> ScanProjectionStats:
    return ScanProjectionStats(
        version=SCAN_PROJECTION_VERSION,
        camera_percentages=_TEST_CAMERA_PERCENTAGES,
        view_face_counts=(4, 3, 2, 1, 1, 1),
        view_pixel_counts=(300, 200, 150, 150, 100, 100),
        face_count=12,
        source_vertex_count=8,
        output_vertex_count=36,
        texture_resolution=2_048,
        target_domain=SCAN_PROJECTION_TARGET_LEFT_HALF,
        target_width=1_024,
        target_height=2_048,
        island_padding_pixels=0,
        outer_safety_inset_pixels=LEFT_HALF_OUTER_SAFETY_INSET_PIXELS,
        usable_pixel_count=1_000,
        covered_pixel_count=985,
        triangle_occupancy=0.985,
    )


def _model_with_variants(variants: ObjectTextureVariants):
    model = import_generated_glb(variants.glb_by_resolution[1024])
    model.object_texture_variants = variants
    return model


def _fingerprint() -> UvFingerprint:
    return UvFingerprint(
        version=UV_FINGERPRINT_VERSION,
        sha256="a" * 64,
        face_count=12,
    )


def _asset_snapshot(asset_directory: Path) -> dict[str, bytes]:
    if not asset_directory.is_dir():
        return {}
    return {
        path.name: path.read_bytes()
        for path in asset_directory.iterdir()
        if path.is_file()
    }


# ### Generation-time symmetry tests ###
class GenerationSymmetricDivisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = Path(self.temporary_directory.name) / "assets"
        self.workspace = GenerationWorkspace(
            asset_directory=self.asset_directory
        )

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.temporary_directory.cleanup()

    def _request(
        self,
        *,
        orientation: str = SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
        use_weighted_projection: bool = False,
    ) -> GenerationRequest:
        return GenerationRequest(
            frame_index=7,
            selected_object_bgra=np.full((4, 4, 4), 255, dtype=np.uint8),
            settings=GenerationServiceSettings(
                meshy_api_key="key",
                use_uv_raycast_for_object_generation=use_weighted_projection,
            ),
            symmetric_division_enabled=True,
            symmetric_division_orientation=orientation,
            projection_camera_percentages=_TEST_CAMERA_PERCENTAGES,
        )

    def _generate_symmetric_record(
        self,
        *,
        source_seed: int = 1,
        pair_seed: int = 2,
        orientation: str = SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
    ) -> tuple[GeneratedObjectRecord, SymmetricSquarePairTextureVariants]:
        source_variants = _ordinary_variants(source_seed)
        pair_variants = _square_pair_variants(pair_seed)
        self.workspace._active_generation_request = self._request(
            orientation=orientation
        )
        with patch(
            "housemaker.generation_workspace."
            "build_automatic_symmetric_object_variants",
            return_value=_automatic_result(
                pair_variants,
                orientation=orientation,
            ),
        ):
            self.workspace._handle_generation_succeeded(
                MeshyGenerationResult(
                    "provider-task",
                    source_variants.glb_by_resolution[2048],
                    "Chair",
                ),
                _model_with_variants(source_variants),
            )
        record = self.workspace.get_data().generated_objects[0]
        return record, pair_variants

    def test_ui_captures_checkbox_and_orientation_only_for_full_generate(
        self,
    ) -> None:
        self.assertEqual(
            self.workspace.symmetric_division_checkbox.text(),
            "Symmetric division",
        )
        self.assertFalse(
            hasattr(
                self.workspace,
                "apply_symmetric_division_to_selected_object",
            )
        )
        self.assertFalse(
            hasattr(
                self.workspace.result_view,
                "begin_symmetric_division_selection",
            )
        )

        self.workspace.symmetric_division_checkbox.setChecked(True)
        horizontal_index = (
            self.workspace.symmetric_division_orientation_combo.findData(
                SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL
            )
        )
        self.workspace.symmetric_division_orientation_combo.setCurrentIndex(
            horizontal_index
        )
        with (
            patch.object(
                self.workspace.video_view,
                "get_frame_bgr",
                return_value=np.zeros((8, 8, 3), dtype=np.uint8),
            ),
            patch.object(
                self.workspace.video_view,
                "has_selection",
                return_value=True,
            ),
            patch.object(
                self.workspace.video_view,
                "build_selected_object_crop",
                return_value=np.full((4, 4, 4), 255, dtype=np.uint8),
            ),
        ):
            request = self.workspace._build_generation_request()
            geometry_request = self.workspace._build_generation_request(
                geometry_only=True
            )
        assert request is not None
        assert geometry_request is not None
        self.assertTrue(request.symmetric_division_enabled)
        self.assertEqual(
            request.symmetric_division_orientation,
            SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL,
        )
        self.assertFalse(geometry_request.symmetric_division_enabled)

        with patch.object(self.workspace, "_start_generation") as start:
            self.workspace.generate_geometry()
        start.assert_not_called()
        self.assertIn(
            "requires the full Generate workflow",
            self.workspace.status_label.text(),
        )

    def test_request_rejects_geometry_only_division_and_bad_orientation(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "Geometry-only"):
            GenerationRequest(
                frame_index=0,
                selected_object_bgra=np.ones((2, 2, 4), dtype=np.uint8),
                settings=GenerationServiceSettings(),
                geometry_only=True,
                symmetric_division_enabled=True,
            )
        with self.assertRaisesRegex(ValueError, "orientation"):
            GenerationRequest(
                frame_index=0,
                selected_object_bgra=np.ones((2, 2, 4), dtype=np.uint8),
                settings=GenerationServiceSettings(),
                symmetric_division_enabled=True,
                symmetric_division_orientation="diagonal",
            )

    def test_new_generation_auto_divides_before_one_pair_record_commit(
        self,
    ) -> None:
        source_variants = _ordinary_variants(3)
        pair_variants = _square_pair_variants(4)
        generated_spy = QSignalSpy(self.workspace.generation_completed)
        self.workspace._active_generation_request = self._request(
            orientation=SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL
        )
        automatic_result = _automatic_result(
            pair_variants,
            orientation=SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL,
        )
        with patch(
            "housemaker.generation_workspace."
            "build_automatic_symmetric_object_variants",
            return_value=automatic_result,
        ) as transform:
            self.workspace._handle_generation_succeeded(
                MeshyGenerationResult(
                    "provider-task",
                    source_variants.glb_by_resolution[2048],
                    "Lamp",
                ),
                _model_with_variants(source_variants),
            )

        transform.assert_called_once_with(
            source_variants.glb_by_resolution[2048],
            SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL,
        )
        self.assertEqual(generated_spy.count(), 1)
        record = self.workspace.get_data().generated_objects[0]
        self.assertEqual(record.frame_index, 7)
        variants = record.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY]
        self.assertEqual(set(variants), {"512", "1024"})
        self.assertEqual(
            record.pipeline[SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY],
            1024,
        )
        self.assertEqual(
            record.asset_path,
            variants["1024"]["glb_asset_path"],
        )
        self.assertEqual(
            record.pipeline["postprocessed_asset_path"],
            variants["1024"]["glb_asset_path"],
        )
        self.assertNotIn(
            OBJECT_OPERATION_UNDO_STACK_PIPELINE_KEY,
            record.pipeline,
        )
        metadata = record.pipeline[SYMMETRIC_DIVISION_PIPELINE_KEY]
        self.assertEqual(
            metadata,
            automatic_result.metadata.to_pipeline_dict(),
        )
        self.assertEqual(
            metadata["version"],
            AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION,
        )
        self.assertEqual(
            metadata["packing_mode"],
            SYMMETRIC_TEXTURE_PACKING_MODE_PAIR,
        )
        self.assertEqual(metadata["texture_content_half"], "left")
        self.assertNotIn("texture_content_quadrant", metadata)
        self.assertEqual(set(_asset_snapshot(self.asset_directory)), {
            Path(entry[path_key]).name
            for entry in variants.values()
            for path_key in ("glb_asset_path", "texture_asset_path")
        })
        self.assertFalse(
            self.workspace.select_object_texture_resolution(
                record.object_id,
                2048,
            )
        )
        self.assertTrue(
            self.workspace.select_object_texture_resolution(
                record.object_id,
                512,
            )
        )
        self.assertEqual(
            {entry.display_name for entry in self.workspace.texture_view.entries},
            {"512 x 512", "1024 x 1024"},
        )

    def test_symmetric_staged_generation_uses_weighted_left_half_scan(
        self,
    ) -> None:
        source_variants = _ordinary_variants(5)
        pair_variants = _square_pair_variants(6)
        scan_stats = _symmetric_scan_projection_stats()
        automatic_result = _automatic_result(
            pair_variants,
            scan_projection_stats=scan_stats,
        )
        self.workspace._active_generation_request = self._request(
            use_weighted_projection=True
        )
        staged = StagedMeshyGenerationResult(
            task_id="texture-task",
            glb_bytes=source_variants.glb_by_resolution[2048],
            name="Table",
            geometry_task_id="geometry-task",
            source_glb_bytes=_box_glb(2.0),
            postprocessed_glb_bytes=_box_glb(3.0),
            original_face_count=20,
            retained_face_count=12,
            removed_face_count=8,
            unused_face_removal_applied=True,
        )
        with patch(
            "housemaker.generation_workspace."
            "build_automatic_symmetric_object_variants",
            return_value=automatic_result,
        ) as divide_and_project:
            self.workspace._handle_generation_succeeded(
                staged,
                _model_with_variants(source_variants),
            )

        divide_and_project.assert_called_once_with(
            source_variants.glb_by_resolution[2048],
            SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
            projection_camera_percentages=_TEST_CAMERA_PERCENTAGES,
            cancellation_check=None,
        )
        record = self.workspace.get_data().generated_objects[0]
        self.assertTrue(
            (self.asset_directory / record.pipeline["source_asset_path"]).is_file()
        )
        self.assertFalse(
            any(
                path.name.endswith(".postprocessed.glb")
                for path in self.asset_directory.iterdir()
            )
        )
        self.assertIn(
            ".texture-1024.glb",
            record.pipeline["postprocessed_asset_path"],
        )
        self.assertEqual(
            record.pipeline[SYMMETRIC_DIVISION_PIPELINE_KEY],
            automatic_result.metadata.to_pipeline_dict(),
        )
        self.assertTrue(record.pipeline[LOCALLY_AUTHORED_UVS_PIPELINE_KEY])
        self.assertEqual(
            record.pipeline[SCAN_PROJECTION_PIPELINE_KEY],
            scan_stats.to_pipeline_dict(),
        )
        self.assertNotIn(VISIBILITY_UV_UNWRAP_PIPELINE_KEY, record.pipeline)

    def test_weighted_projection_is_forwarded_in_gui_persistence(
        self,
    ) -> None:
        source_variants = _ordinary_variants(17)
        pair_variants = _square_pair_variants(17)
        scan_stats = _symmetric_scan_projection_stats()
        automatic_result = _automatic_result(
            pair_variants,
            orientation=SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL,
            scan_projection_stats=scan_stats,
        )
        self.workspace._active_generation_request = self._request(
            orientation=SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL,
            use_weighted_projection=True,
        )
        provider_result = MeshyGenerationResult(
            task_id="texture-task",
            glb_bytes=source_variants.glb_by_resolution[2048],
            name="Dense symmetric table",
        )

        with patch(
            "housemaker.generation_workspace."
            "build_automatic_symmetric_object_variants",
            return_value=automatic_result,
        ) as divide_and_project:
            self.workspace._handle_generation_succeeded(
                provider_result,
                _model_with_variants(source_variants),
            )

        divide_and_project.assert_called_once_with(
            source_variants.glb_by_resolution[2048],
            SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL,
            projection_camera_percentages=_TEST_CAMERA_PERCENTAGES,
            cancellation_check=None,
        )
        record = self.workspace.get_data().generated_objects[0]
        self.assertEqual(
            record.pipeline[SYMMETRIC_DIVISION_PIPELINE_KEY],
            automatic_result.metadata.to_pipeline_dict(),
        )
        self.assertEqual(
            set(record.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY]),
            {"512", "1024"},
        )
        self.assertEqual(
            record.pipeline["postprocessed_asset_path"],
            record.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY]["1024"][
                "glb_asset_path"
            ],
        )
        self.assertFalse(
            any(
                path.name.endswith(".postprocessed.glb")
                for path in self.asset_directory.iterdir()
            )
        )
        self.assertTrue(record.pipeline[LOCALLY_AUTHORED_UVS_PIPELINE_KEY])
        self.assertEqual(
            record.pipeline[SCAN_PROJECTION_PIPELINE_KEY],
            scan_stats.to_pipeline_dict(),
        )
        self.assertNotIn(VISIBILITY_UV_UNWRAP_PIPELINE_KEY, record.pipeline)

    def test_weighted_projection_is_forwarded_in_background_persistence(
        self,
    ) -> None:
        source_variants = _ordinary_variants(19)
        pair_variants = _square_pair_variants(19)
        scan_stats = _symmetric_scan_projection_stats()
        automatic_result = _automatic_result(
            pair_variants,
            scan_projection_stats=scan_stats,
        )
        provider_result = MeshyGenerationResult(
            task_id="texture-task",
            glb_bytes=source_variants.glb_by_resolution[2048],
            name="Dense symmetric cupboard",
        )

        with patch(
            "housemaker.generation_workspace."
            "build_automatic_symmetric_object_variants",
            return_value=automatic_result,
        ) as divide_and_project:
            saved = _prepare_and_persist_object_generation(
                self.asset_directory,
                "background-object",
                self._request(use_weighted_projection=True),
                provider_result,
                _model_with_variants(source_variants),
            )

        divide_and_project.assert_called_once_with(
            source_variants.glb_by_resolution[2048],
            SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
            projection_camera_percentages=_TEST_CAMERA_PERCENTAGES,
            cancellation_check=None,
        )
        self.assertIsNotNone(saved.symmetry)
        assert saved.symmetry is not None
        self.assertEqual(
            saved.symmetry.to_pipeline_dict(),
            automatic_result.metadata.to_pipeline_dict(),
        )
        self.assertEqual(
            saved.pipeline[SYMMETRIC_DIVISION_PIPELINE_KEY],
            automatic_result.metadata.to_pipeline_dict(),
        )
        self.assertEqual(
            set(saved.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY]),
            {"512", "1024"},
        )
        self.assertTrue(saved.pipeline[LOCALLY_AUTHORED_UVS_PIPELINE_KEY])
        self.assertEqual(
            saved.pipeline[SCAN_PROJECTION_PIPELINE_KEY],
            scan_stats.to_pipeline_dict(),
        )
        self.assertNotIn(VISIBILITY_UV_UNWRAP_PIPELINE_KEY, saved.pipeline)

    def test_transform_failure_is_transactional(self) -> None:
        source_variants = _ordinary_variants(7)
        self.workspace._active_generation_request = self._request()
        generated_spy = QSignalSpy(self.workspace.generation_completed)
        with (
            patch(
                "housemaker.generation_workspace."
                "build_automatic_symmetric_object_variants",
                side_effect=ValueError("unsupported material"),
            ),
            patch("housemaker.generation_workspace.QMessageBox.warning"),
        ):
            self.workspace._handle_generation_succeeded(
                MeshyGenerationResult(
                    "provider-task",
                    source_variants.glb_by_resolution[2048],
                    "Broken",
                ),
                _model_with_variants(source_variants),
            )
        self.assertEqual(self.workspace.get_data().generated_objects, [])
        self.assertEqual(self.workspace._generated_model_cache, {})
        self.assertEqual(_asset_snapshot(self.asset_directory), {})
        self.assertEqual(generated_spy.count(), 0)
        self.assertIsNone(self.workspace._active_generation_request)

    def test_v4_retexture_preserves_pair_layout_and_undo(self) -> None:
        original, _initial_variants = self._generate_symmetric_record()
        before = self.workspace.get_data().generated_objects[0]
        fingerprint = _fingerprint()
        provider_variants = _ordinary_variants(8)
        regenerated_variants = _square_pair_variants(9)
        outcome = TextureRegenerationOutcome(
            request=TextureRegenerationRequest(
                object_id=before.object_id,
                reference_frame_index=2,
                reference_image_bgra=np.full(
                    (3, 3, 4), 255, dtype=np.uint8
                ),
                model_glb=_box_glb(4.0),
                settings=GenerationServiceSettings(),
                enable_original_uv=True,
                submitted_uv_fingerprint=fingerprint,
                preserve_symmetric_uvs=True,
            ),
            result=MeshyGenerationResult(
                "retexture-task",
                provider_variants.glb_by_resolution[2048],
            ),
            final_uv_fingerprint=fingerprint,
        )
        with patch(
            "housemaker.generation_workspace."
            "build_symmetric_square_pair_texture_variants",
            return_value=regenerated_variants,
        ) as pair_repack:
            self.workspace._handle_texture_regeneration_succeeded(
                outcome,
                _model_with_variants(provider_variants),
            )
        pair_repack.assert_called_once_with(
            provider_variants.glb_by_resolution[2048],
            uvs_already_left_packed=True,
        )
        regenerated = self.workspace.get_data().generated_objects[0]
        self.assertEqual(
            regenerated.pipeline[SYMMETRIC_DIVISION_PIPELINE_KEY],
            before.pipeline[SYMMETRIC_DIVISION_PIPELINE_KEY],
        )
        self.assertEqual(
            set(regenerated.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY]),
            {"512", "1024"},
        )
        self.assertTrue(self.workspace.undo_selected_object_change())
        restored = self.workspace.get_data().generated_objects[0]
        self.assertEqual(restored.pipeline, before.pipeline)
        self.assertEqual(restored.asset_path, before.asset_path)
        self.assertEqual(restored.provider_task_id, before.provider_task_id)

    def test_v2_quarter_record_retexture_remains_compatible(
        self,
    ) -> None:
        quarter_variants = _quarter_variants(14)
        variant_metadata = self.workspace._persist_object_texture_variants(
            "quarter-v2",
            quarter_variants,
        )
        symmetry = SymmetricDivisionMetadata(
            version=SYMMETRIC_QUARTER_METADATA_VERSION,
            orientation=SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
            kept_side=SYMMETRIC_DIVISION_SIDE_LEFT,
            plane_coordinate=0.0,
            packing_mode=SYMMETRIC_TEXTURE_PACKING_MODE_TOP_LEFT_QUARTER,
            texture_content_quadrant=(
                SYMMETRIC_TEXTURE_CONTENT_QUADRANT_TOP_LEFT
            ),
            selection_mode=(
                SYMMETRIC_SELECTION_MODE_FEWEST_TRIANGLES_RANDOM_TIE
            ),
            triangle_count_by_side=(("left", 2), ("right", 5)),
            tie_broken_randomly=False,
        )
        record = GeneratedObjectRecord(
            object_id="quarter-v2",
            frame_index=0,
            object_name="Compatible quarter",
            pipeline={
                TEXTURE_VARIANTS_PIPELINE_KEY: variant_metadata,
                SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY: 1024,
                "postprocessed_asset_path": variant_metadata["1024"][
                    "glb_asset_path"
                ],
                SYMMETRIC_DIVISION_PIPELINE_KEY: symmetry.to_pipeline_dict(),
            },
            provider="meshy",
            provider_task_id="quarter-task",
            asset_path=variant_metadata["1024"]["glb_asset_path"],
        )
        self.workspace._data.generated_objects.append(record)
        self.workspace._refresh_generated_objects_list(record.object_id)

        provider_variants = _ordinary_variants(15)
        next_quarter_variants = _quarter_variants(16)
        fingerprint = _fingerprint()
        outcome = TextureRegenerationOutcome(
            request=TextureRegenerationRequest(
                object_id=record.object_id,
                reference_frame_index=1,
                reference_image_bgra=np.ones((2, 2, 4), dtype=np.uint8),
                model_glb=_box_glb(2.0),
                settings=GenerationServiceSettings(),
                enable_original_uv=True,
                submitted_uv_fingerprint=fingerprint,
                preserve_symmetric_uvs=True,
            ),
            result=MeshyGenerationResult(
                "quarter-retexture",
                provider_variants.glb_by_resolution[2048],
            ),
            final_uv_fingerprint=fingerprint,
        )
        with patch(
            "housemaker.generation_workspace."
            "build_symmetric_quarter_texture_variants",
            return_value=next_quarter_variants,
        ) as quarter_repack:
            self.workspace._handle_texture_regeneration_succeeded(
                outcome,
                _model_with_variants(provider_variants),
            )
        quarter_repack.assert_called_once_with(
            provider_variants.glb_by_resolution[2048],
            uvs_already_top_left_quarter=True,
        )
        regenerated = self.workspace.get_data().generated_objects[0]
        self.assertEqual(
            regenerated.pipeline[SYMMETRIC_DIVISION_PIPELINE_KEY],
            symmetry.to_pipeline_dict(),
        )
        self.assertEqual(
            set(regenerated.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY]),
            {"512", "1024"},
        )

    def test_v3_double_sized_pair_retexture_remains_compatible(self) -> None:
        legacy_pair_variants = _legacy_pair_variants(18)
        variant_metadata = self.workspace._persist_object_texture_variants(
            "pair-v3",
            legacy_pair_variants,
        )
        symmetry = SymmetricDivisionMetadata(
            version=LEGACY_SYMMETRIC_PAIR_METADATA_VERSION,
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
        record = GeneratedObjectRecord(
            object_id="pair-v3",
            frame_index=0,
            object_name="Compatible pair",
            pipeline={
                TEXTURE_VARIANTS_PIPELINE_KEY: variant_metadata,
                SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY: 1024,
                "postprocessed_asset_path": variant_metadata["1024"][
                    "glb_asset_path"
                ],
                SYMMETRIC_DIVISION_PIPELINE_KEY: symmetry.to_pipeline_dict(),
            },
            provider="meshy",
            provider_task_id="pair-v3-task",
            asset_path=variant_metadata["1024"]["glb_asset_path"],
        )
        self.workspace._data.generated_objects.append(record)
        self.workspace._refresh_generated_objects_list(record.object_id)

        provider_variants = _ordinary_variants(19)
        next_pair_variants = _legacy_pair_variants(20)
        fingerprint = _fingerprint()
        outcome = TextureRegenerationOutcome(
            request=TextureRegenerationRequest(
                object_id=record.object_id,
                reference_frame_index=1,
                reference_image_bgra=np.ones((2, 2, 4), dtype=np.uint8),
                model_glb=_box_glb(2.0),
                settings=GenerationServiceSettings(),
                enable_original_uv=True,
                submitted_uv_fingerprint=fingerprint,
                preserve_symmetric_uvs=True,
            ),
            result=MeshyGenerationResult(
                "pair-v3-retexture",
                provider_variants.glb_by_resolution[2048],
            ),
            final_uv_fingerprint=fingerprint,
        )
        with patch(
            "housemaker.generation_workspace."
            "build_symmetric_pair_texture_variants",
            return_value=next_pair_variants,
        ) as pair_repack:
            self.workspace._handle_texture_regeneration_succeeded(
                outcome,
                _model_with_variants(provider_variants),
            )
        pair_repack.assert_called_once_with(
            provider_variants.glb_by_resolution[2048],
            uvs_already_left_packed=True,
        )
        replacement = self.workspace.get_data().generated_objects[0]
        self.assertEqual(
            replacement.pipeline[SYMMETRIC_DIVISION_PIPELINE_KEY],
            symmetry.to_pipeline_dict(),
        )
        self.assertEqual(
            set(replacement.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY]),
            {"512", "1024"},
        )

    def test_legacy_v1_record_keeps_half_metadata_and_three_resolutions(
        self,
    ) -> None:
        legacy_variants = _ordinary_variants(11)
        variant_metadata = self.workspace._persist_object_texture_variants(
            "legacy",
            legacy_variants,
        )
        record = GeneratedObjectRecord(
            object_id="legacy",
            frame_index=0,
            object_name="Legacy half",
            pipeline={
                TEXTURE_VARIANTS_PIPELINE_KEY: variant_metadata,
                SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY: 2048,
                "postprocessed_asset_path": variant_metadata["2048"][
                    "glb_asset_path"
                ],
                SYMMETRIC_DIVISION_PIPELINE_KEY: {
                    "version": SYMMETRIC_DIVISION_METADATA_VERSION,
                    "orientation": SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
                    "kept_side": SYMMETRIC_DIVISION_SIDE_LEFT,
                    "plane_coordinate": 0.0,
                    "texture_content_half": "left",
                },
            },
            provider="meshy",
            provider_task_id="legacy-task",
            asset_path=variant_metadata["2048"]["glb_asset_path"],
        )
        self.workspace._data.generated_objects.append(record)
        self.workspace._refresh_generated_objects_list(record.object_id)
        metadata = self.workspace.get_object_symmetric_division(
            record.object_id
        )
        assert metadata is not None
        self.assertEqual(metadata.version, SYMMETRIC_DIVISION_METADATA_VERSION)
        self.assertEqual(metadata.texture_content_half, "left")
        self.assertEqual(
            {entry.display_name for entry in self.workspace.texture_view.entries},
            {"512 x 512", "1024 x 1024", "2048 x 2048"},
        )

        provider_variants = _ordinary_variants(12)
        next_legacy_variants = _ordinary_variants(13)
        fingerprint = _fingerprint()
        outcome = TextureRegenerationOutcome(
            request=TextureRegenerationRequest(
                object_id=record.object_id,
                reference_frame_index=1,
                reference_image_bgra=np.ones((2, 2, 4), dtype=np.uint8),
                model_glb=_box_glb(2.0),
                settings=GenerationServiceSettings(),
                enable_original_uv=True,
                submitted_uv_fingerprint=fingerprint,
                preserve_symmetric_uvs=True,
            ),
            result=MeshyGenerationResult(
                "legacy-retexture",
                provider_variants.glb_by_resolution[2048],
            ),
            final_uv_fingerprint=fingerprint,
        )
        with patch(
            "housemaker.generation_workspace."
            "build_symmetric_half_texture_variants",
            return_value=next_legacy_variants,
        ) as half_repack:
            self.workspace._handle_texture_regeneration_succeeded(
                outcome,
                _model_with_variants(provider_variants),
            )
        half_repack.assert_called_once_with(
            provider_variants.glb_by_resolution[2048],
            uvs_already_left_packed=True,
        )
        replacement = self.workspace.get_data().generated_objects[0]
        self.assertEqual(
            set(replacement.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY]),
            {str(resolution) for resolution in TEXTURE_RESOLUTIONS},
        )
        self.assertEqual(
            replacement.pipeline[SYMMETRIC_DIVISION_PIPELINE_KEY],
            record.pipeline[SYMMETRIC_DIVISION_PIPELINE_KEY],
        )


if __name__ == "__main__":
    unittest.main()
