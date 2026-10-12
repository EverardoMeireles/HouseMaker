# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtWidgets import QApplication

from housemaker.architectural_trim_generation import (
    ArchitecturalTrimRepeatLayout,
    prepare_architectural_trim_repeat_module,
    retarget_architectural_trim_repeat_module_preview_model,
)
from housemaker.architectural_trim_uv import (
    ARCHITECTURAL_TRIM_UV_LONGITUDINAL_AXIS_X,
    ARCHITECTURAL_TRIM_UV_LONGITUDINAL_AXIS_Z,
    ARCHITECTURAL_TRIM_UV_TEXTURE_DOMAIN_LEFT_HALF,
    CORNICE_UV_TEXTURE_DOMAIN_FULL,
    CORNICE_UV_TEXTURE_DOMAIN_LEFT_HALF,
    build_architectural_trim_segmented_uv_glb,
    build_cornice_segmented_uv_glb,
)
from housemaker.generation_workspace import (
    ARCHITECTURAL_TRIM_COMPONENT_PIPELINE_KEY,
    ARCHITECTURAL_TRIM_REPEAT_PIPELINE_KEY,
    ARCHITECTURAL_TRIM_SEGMENTED_UV_PIPELINE_KEY,
    ARCHITECTURAL_TRIM_SEGMENTED_UV_PIPELINE_VERSION,
    CORNICE_SEGMENTED_UV_PIPELINE_KEY,
    CORNICE_SEGMENTED_UV_PIPELINE_VERSION,
    LOCALLY_AUTHORED_UVS_PIPELINE_KEY,
    OBJECT_OPERATION_GENERATE_MODEL,
    SYMMETRIC_DIVISION_PIPELINE_KEY,
    GenerationRequest,
    GenerationWorkspace,
    MeshyImagePlanner,
    StagedMeshyGenerationResult,
    TextureRegenerationOutcome,
    TextureRegenerationRequest,
    _ActiveObjectOperation,
    _materialize_texture_regeneration_preflight,
    _prepare_and_persist_texture_regeneration,
    get_architectural_trim_uv_texture_domain,
    get_cornice_uv_texture_domain,
)
from housemaker.glb import GeneratedModel, import_generated_glb
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.models import (
    TRIM_KIND_CORNICE,
    TRIM_KIND_EDGING_STRIP,
    TRIM_KIND_SKIRTING_BOARD,
)
from housemaker.object_symmetry import build_symmetric_half_texture_variants
from housemaker.object_texture_variants import (
    TEXTURE_RESOLUTIONS,
    ObjectTextureVariants,
)
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.uv_integrity import build_uv_fingerprint

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _trim_component_model(
    extents: tuple[float, float, float] = (1.0, 0.1, 0.2),
) -> GeneratedModel:
    mesh = trimesh.creation.box(extents=extents)
    glb_bytes = bytes(trimesh.Scene(mesh).export(file_type="glb"))
    return import_generated_glb(glb_bytes)


def _trim_generation_request(
    *,
    geometry_only: bool,
    fit_dimensions: tuple[float, float, float],
    trim_kind: str | None = None,
    cornice_uv_texture_domain: str = CORNICE_UV_TEXTURE_DOMAIN_FULL,
) -> GenerationRequest:
    reference = np.zeros((16, 16, 4), dtype=np.uint8)
    reference[:, :, 3] = 255
    return GenerationRequest(
        frame_index=0,
        selected_object_bgra=reference,
        settings=GenerationServiceSettings(meshy_api_key="meshy-key"),
        geometry_only=geometry_only,
        architectural_trim_fit_dimensions=fit_dimensions,
        architectural_trim_kind=trim_kind,
        cornice_uv_texture_domain=cornice_uv_texture_domain,
    )


def _module_texture_variants(module_glb: bytes) -> ObjectTextureVariants:
    """Build lightweight variants around one authoritative trim module."""

    texture_pngs: dict[int, bytes] = {}
    previews: dict[int, np.ndarray] = {}
    for resolution in TEXTURE_RESOLUTIONS:
        preview = np.full((4, 4, 4), (180, 190, 200, 255), dtype=np.uint8)
        image_buffer = BytesIO()
        Image.fromarray(preview, mode="RGBA").save(image_buffer, format="PNG")
        texture_pngs[resolution] = image_buffer.getvalue()
        previews[resolution] = preview
    return ObjectTextureVariants(
        glb_by_resolution={
            resolution: module_glb for resolution in TEXTURE_RESOLUTIONS
        },
        texture_png_by_resolution=texture_pngs,
        preview_rgba_by_resolution=previews,
    )


def _left_packed_textured_cornice_glb() -> bytes:
    """Build one real 2048 texture source for half-packing persistence."""

    mesh = trimesh.creation.box(extents=(1.0, 0.1, 0.2))
    texture = np.full((2048, 2048, 4), (96, 144, 208, 255), dtype=np.uint8)
    mesh.visual = TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=np.float64),
        material=PBRMaterial(
            baseColorTexture=Image.fromarray(texture, mode="RGBA")
        ),
    )
    source = bytes(trimesh.Scene(mesh).export(file_type="glb"))
    return build_cornice_segmented_uv_glb(
        source,
        texture_resolution=2048,
        texture_domain=CORNICE_UV_TEXTURE_DOMAIN_LEFT_HALF,
    )


# ### Architectural-trim planner tests ###
class ArchitecturalTrimPlannerTests(unittest.TestCase):
    def test_half_texture_domain_accepts_every_textured_trim_kind(self) -> None:
        for trim_kind in (
            TRIM_KIND_CORNICE,
            TRIM_KIND_SKIRTING_BOARD,
            TRIM_KIND_EDGING_STRIP,
        ):
            with self.subTest(trim_kind=trim_kind):
                request = _trim_generation_request(
                    geometry_only=False,
                    fit_dimensions=(1.0, 0.1, 0.2),
                    trim_kind=trim_kind,
                    cornice_uv_texture_domain=(
                        ARCHITECTURAL_TRIM_UV_TEXTURE_DOMAIN_LEFT_HALF
                    ),
                )
                self.assertEqual(
                    request.cornice_uv_texture_domain,
                    ARCHITECTURAL_TRIM_UV_TEXTURE_DOMAIN_LEFT_HALF,
                )
        with self.assertRaisesRegex(ValueError, "requires a trim"):
            _trim_generation_request(
                geometry_only=False,
                fit_dimensions=(1.0, 0.1, 0.2),
                trim_kind=None,
                cornice_uv_texture_domain=(
                    ARCHITECTURAL_TRIM_UV_TEXTURE_DOMAIN_LEFT_HALF
                ),
            )
        with self.assertRaisesRegex(ValueError, "Geometry-only"):
            _trim_generation_request(
                geometry_only=True,
                fit_dimensions=(1.0, 0.1, 0.2),
                trim_kind=TRIM_KIND_CORNICE,
                cornice_uv_texture_domain=(
                    ARCHITECTURAL_TRIM_UV_TEXTURE_DOMAIN_LEFT_HALF
                ),
            )

    def test_textured_generation_retextures_one_module_then_repeats_it(self) -> None:
        source_glb = _trim_component_model().glb_bytes
        fit_dimensions = (6.0, 0.2, 0.4)
        geometry_result = MeshyGenerationResult(
            task_id="geometry-task",
            glb_bytes=source_glb,
            name="Generated trim",
        )
        submitted_glbs: list[bytes] = []

        def retexture_model(**kwargs: object) -> MeshyGenerationResult:
            submitted_glb = bytes(kwargs["model_glb"])
            submitted_glbs.append(submitted_glb)
            return MeshyGenerationResult(
                task_id="texture-task",
                glb_bytes=source_glb,
                name="Textured trim",
            )

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=geometry_result,
            ) as generate_model,
            patch(
                "housemaker.generation_workspace.request_retextured_model",
                side_effect=retexture_model,
            ) as retexture,
        ):
            result = MeshyImagePlanner().plan(
                _trim_generation_request(
                    geometry_only=False,
                    fit_dimensions=fit_dimensions,
                )
            )

        self.assertIsInstance(result, StagedMeshyGenerationResult)
        self.assertTrue(generate_model.call_args.kwargs["should_texture"] is False)
        retexture.assert_called_once()
        self.assertEqual(len(submitted_glbs), 1)
        self.assertIsNotNone(result.architectural_trim_repeat_layout)
        assert result.architectural_trim_repeat_layout is not None
        self.assertEqual(result.architectural_trim_repeat_layout.repeat_count, 3)
        module_model = import_generated_glb(submitted_glbs[0])
        np.testing.assert_allclose(
            module_model.mesh.extents,
            result.architectural_trim_repeat_layout.module_extents,
        )
        np.testing.assert_allclose(
            import_generated_glb(result.postprocessed_glb_bytes).mesh.extents,
            result.architectural_trim_repeat_layout.module_extents,
        )
        np.testing.assert_allclose(
            import_generated_glb(result.glb_bytes).mesh.extents,
            fit_dimensions,
        )

    def test_geometry_only_generation_exposes_full_repeated_trim(self) -> None:
        source_glb = _trim_component_model().glb_bytes
        fit_dimensions = (6.0, 0.2, 0.4)
        geometry_result = MeshyGenerationResult(
            task_id="geometry-task",
            glb_bytes=source_glb,
            name="Generated trim",
        )

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=geometry_result,
            ),
            patch(
                "housemaker.generation_workspace.request_retextured_model",
            ) as retexture,
        ):
            result = MeshyImagePlanner().plan(
                _trim_generation_request(
                    geometry_only=True,
                    fit_dimensions=fit_dimensions,
                )
            )

        self.assertIsInstance(result, StagedMeshyGenerationResult)
        self.assertTrue(result.geometry_only)
        retexture.assert_not_called()
        self.assertIsNotNone(result.architectural_trim_repeat_layout)
        assert result.architectural_trim_repeat_layout is not None
        self.assertEqual(result.architectural_trim_repeat_layout.repeat_count, 3)
        np.testing.assert_allclose(
            import_generated_glb(result.postprocessed_glb_bytes).mesh.extents,
            result.architectural_trim_repeat_layout.module_extents,
        )
        np.testing.assert_allclose(
            import_generated_glb(result.glb_bytes).mesh.extents,
            fit_dimensions,
        )

    def test_geometry_only_cornice_does_not_build_segmented_texture_uvs(
        self,
    ) -> None:
        source_glb = _trim_component_model().glb_bytes
        geometry_result = MeshyGenerationResult(
            task_id="cornice-geometry-task",
            glb_bytes=source_glb,
            name="Generated cornice",
        )

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=geometry_result,
            ),
            patch(
                "housemaker.generation_workspace."
                "build_architectural_trim_segmented_uv_glb",
                side_effect=lambda model_glb, **_kwargs: model_glb,
            ) as build_segmented_uv,
            patch(
                "housemaker.generation_workspace.request_retextured_model",
            ) as retexture,
        ):
            result = MeshyImagePlanner().plan(
                _trim_generation_request(
                    geometry_only=True,
                    fit_dimensions=(6.0, 0.2, 0.4),
                    trim_kind=TRIM_KIND_CORNICE,
                )
            )

        build_segmented_uv.assert_not_called()
        retexture.assert_not_called()
        self.assertIsInstance(result, StagedMeshyGenerationResult)
        self.assertTrue(result.geometry_only)
        self.assertFalse(result.cornice_segmented_uv)

    def test_cornice_generation_preserves_segmented_uv_module(self) -> None:
        source_glb = _trim_component_model().glb_bytes
        fit_dimensions = (6.0, 0.2, 0.4)
        submitted_glbs: list[bytes] = []

        def retexture_model(**kwargs: object) -> MeshyGenerationResult:
            submitted_glbs.append(bytes(kwargs["model_glb"]))
            self.assertTrue(kwargs["enable_original_uv"])
            return MeshyGenerationResult(
                task_id="cornice-texture-task",
                glb_bytes=source_glb,
                name="Textured cornice",
            )

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=MeshyGenerationResult(
                    task_id="cornice-geometry-task",
                    glb_bytes=source_glb,
                    name="Generated cornice",
                ),
            ),
            patch(
                "housemaker.generation_workspace.request_retextured_model",
                side_effect=retexture_model,
            ),
            patch(
                "housemaker.generation_workspace."
                "replace_object_base_color_texture_from_glb",
                side_effect=lambda model_glb, _texture_glb: model_glb,
            ),
        ):
            result = MeshyImagePlanner().plan(
                _trim_generation_request(
                    geometry_only=False,
                    fit_dimensions=fit_dimensions,
                    trim_kind=TRIM_KIND_CORNICE,
                )
            )

        self.assertIsInstance(result, StagedMeshyGenerationResult)
        self.assertTrue(result.cornice_segmented_uv)
        self.assertEqual(len(submitted_glbs), 1)
        self.assertIsNotNone(result.architectural_trim_repeat_layout)
        assert result.architectural_trim_repeat_layout is not None
        self.assertGreaterEqual(
            result.architectural_trim_repeat_layout.repeat_count,
            4,
        )
        submitted_model = import_generated_glb(submitted_glbs[0])
        self.assertIsNotNone(getattr(submitted_model.mesh.visual, "uv", None))
        self.assertEqual(
            submitted_glbs[0],
            result.postprocessed_glb_bytes,
        )

    def test_skirting_generation_repeats_one_bounded_uv_module(self) -> None:
        fit_dimensions = (6.0, 0.1, 0.2)
        source_glb = _trim_component_model(fit_dimensions).glb_bytes
        submitted_glbs: list[bytes] = []

        def retexture_model(**kwargs: object) -> MeshyGenerationResult:
            submitted_glbs.append(bytes(kwargs["model_glb"]))
            return MeshyGenerationResult(
                task_id="skirting-texture-task",
                glb_bytes=source_glb,
                name="Textured skirting board",
            )

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=MeshyGenerationResult(
                    task_id="skirting-geometry-task",
                    glb_bytes=source_glb,
                    name="Generated skirting board",
                ),
            ),
            patch(
                "housemaker.generation_workspace.request_retextured_model",
                side_effect=retexture_model,
            ),
            patch(
                "housemaker.generation_workspace."
                "replace_object_base_color_texture_from_glb",
                side_effect=lambda model_glb, _texture_glb: model_glb,
            ),
        ):
            result = MeshyImagePlanner().plan(
                _trim_generation_request(
                    geometry_only=False,
                    fit_dimensions=fit_dimensions,
                    trim_kind=TRIM_KIND_SKIRTING_BOARD,
                )
            )

        layout = result.architectural_trim_repeat_layout
        self.assertIsNotNone(layout)
        assert layout is not None
        self.assertEqual(layout.repeat_count, 8)
        self.assertLessEqual(
            layout.module_extents[0],
            max(layout.module_extents[1:]) * 4.0,
        )
        self.assertEqual(len(submitted_glbs), 1)
        np.testing.assert_allclose(
            import_generated_glb(submitted_glbs[0]).mesh.extents,
            layout.module_extents,
        )
        np.testing.assert_allclose(
            import_generated_glb(result.glb_bytes).mesh.extents,
            fit_dimensions,
        )

    def test_cornice_half_texture_uses_left_domain_without_clipping(self) -> None:
        source_glb = _trim_component_model().glb_bytes
        fit_dimensions = (6.0, 0.2, 0.4)
        request = _trim_generation_request(
            geometry_only=False,
            fit_dimensions=fit_dimensions,
            trim_kind=TRIM_KIND_CORNICE,
            cornice_uv_texture_domain=(
                CORNICE_UV_TEXTURE_DOMAIN_LEFT_HALF
            ),
        )
        self.assertFalse(request.symmetric_division_enabled)

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=MeshyGenerationResult(
                    task_id="cornice-half-geometry-task",
                    glb_bytes=source_glb,
                    name="Generated cornice",
                ),
            ),
            patch(
                "housemaker.generation_workspace.request_retextured_model",
                return_value=MeshyGenerationResult(
                    task_id="cornice-half-texture-task",
                    glb_bytes=source_glb,
                    name="Textured cornice",
                ),
            ),
            patch(
                "housemaker.generation_workspace."
                "replace_object_base_color_texture_from_glb",
                side_effect=lambda model_glb, _texture_glb: model_glb,
            ),
            patch(
                "housemaker.generation_workspace."
                "build_architectural_trim_segmented_uv_glb",
                wraps=build_architectural_trim_segmented_uv_glb,
            ) as build_segmented_uv,
        ):
            result = MeshyImagePlanner().plan(request)

        self.assertTrue(result.cornice_segmented_uv)
        self.assertEqual(
            result.cornice_uv_texture_domain,
            CORNICE_UV_TEXTURE_DOMAIN_LEFT_HALF,
        )
        self.assertEqual(
            build_segmented_uv.call_args.kwargs["texture_domain"],
            CORNICE_UV_TEXTURE_DOMAIN_LEFT_HALF,
        )
        self.assertEqual(
            build_segmented_uv.call_args.kwargs["longitudinal_axis"],
            ARCHITECTURAL_TRIM_UV_LONGITUDINAL_AXIS_X,
        )
        for geometry in import_generated_glb(
            result.postprocessed_glb_bytes
        ).scene.geometry.values():
            uvs = np.asarray(geometry.visual.uv, dtype=np.float64)
            self.assertLessEqual(float(np.max(uvs[:, 0])), 0.5 + 1e-7)
        np.testing.assert_allclose(
            import_generated_glb(result.glb_bytes).mesh.extents,
            fit_dimensions,
        )


# ### Architectural-trim generation routing tests ###
class ArchitecturalTrimGenerationWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = GenerationWorkspace(
            asset_directory=Path(self._temporary_directory.name) / "generated"
        )
        self.trim_id = "a" * 32
        self.object_id = "architectural-trim-test-object"
        self.fit_dimensions = (6.0, 0.2, 0.4)
        self.record = self.workspace.register_architectural_trim_component_model(
            level_index=2,
            trim_id=self.trim_id,
            object_id=self.object_id,
            object_name="Cornice",
            fit_dimensions=self.fit_dimensions,
            model=_trim_component_model(),
            trim_kind=TRIM_KIND_CORNICE,
        )
        self.assertTrue(
            self.workspace.set_architectural_trim_editing_target(
                level_index=2,
                trim_id=self.trim_id,
                object_id=self.object_id,
                object_name="Cornice",
                fit_dimensions=self.fit_dimensions,
                trim_kind=TRIM_KIND_CORNICE,
            )
        )

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_every_trim_kind_disables_model_creation_but_allows_retexture(
        self,
    ) -> None:
        for trim_kind in (
            TRIM_KIND_CORNICE,
            TRIM_KIND_SKIRTING_BOARD,
            TRIM_KIND_EDGING_STRIP,
        ):
            with self.subTest(trim_kind=trim_kind):
                self.assertTrue(
                    self.workspace.set_architectural_trim_editing_target(
                        level_index=2,
                        trim_id=self.trim_id,
                        object_id=self.object_id,
                        object_name="Trim",
                        fit_dimensions=self.fit_dimensions,
                        trim_kind=trim_kind,
                    )
                )
                with patch.object(
                    self.workspace,
                    "_can_regenerate_object_texture",
                    return_value=True,
                ):
                    self.workspace._sync_controls()

                self.assertFalse(self.workspace.generate_button.isEnabled())
                self.assertFalse(
                    self.workspace.generate_geometry_button.isEnabled()
                )
                self.assertTrue(
                    self.workspace.generate_texture_button.isEnabled()
                )

    def test_symmetric_division_is_available_for_every_trim_kind(
        self,
    ) -> None:
        for trim_kind in (
            TRIM_KIND_CORNICE,
            TRIM_KIND_SKIRTING_BOARD,
            TRIM_KIND_EDGING_STRIP,
        ):
            with self.subTest(trim_kind=trim_kind):
                self.assertTrue(
                    self.workspace.set_architectural_trim_editing_target(
                        level_index=2,
                        trim_id=self.trim_id,
                        object_id=self.object_id,
                        object_name="Trim",
                        fit_dimensions=self.fit_dimensions,
                        trim_kind=trim_kind,
                    )
                )
                with patch.object(
                    self.workspace,
                    "_can_regenerate_object_texture",
                    return_value=True,
                ):
                    self.workspace._sync_controls()
                self.assertEqual(
                    self.workspace.symmetric_division_checkbox.isEnabled(),
                    True,
                )

    def test_selecting_cornice_restores_its_saved_texture_domain(self) -> None:
        self.record.pipeline[CORNICE_SEGMENTED_UV_PIPELINE_KEY] = {
            "version": CORNICE_SEGMENTED_UV_PIPELINE_VERSION,
            "texture_domain": CORNICE_UV_TEXTURE_DOMAIN_LEFT_HALF,
        }
        self.workspace.symmetric_division_checkbox.setChecked(False)

        self.assertTrue(
            self.workspace.set_architectural_trim_editing_target(
                level_index=2,
                trim_id=self.trim_id,
                object_id=self.object_id,
                object_name="Cornice",
                fit_dimensions=self.fit_dimensions,
                trim_kind=TRIM_KIND_CORNICE,
            )
        )
        self.assertTrue(
            self.workspace.symmetric_division_checkbox.isChecked()
        )

        self.record.pipeline[CORNICE_SEGMENTED_UV_PIPELINE_KEY] = {
            "version": CORNICE_SEGMENTED_UV_PIPELINE_VERSION,
            "texture_domain": CORNICE_UV_TEXTURE_DOMAIN_FULL,
        }
        self.workspace.symmetric_division_checkbox.setChecked(True)

        self.assertTrue(
            self.workspace.set_architectural_trim_editing_target(
                level_index=2,
                trim_id=self.trim_id,
                object_id=self.object_id,
                object_name="Cornice",
                fit_dimensions=self.fit_dimensions,
                trim_kind=TRIM_KIND_CORNICE,
            )
        )
        self.assertFalse(
            self.workspace.symmetric_division_checkbox.isChecked()
        )

    def test_legacy_cornice_domain_getter_ignores_other_trim_kinds(
        self,
    ) -> None:
        component_metadata = self.record.pipeline[
            ARCHITECTURAL_TRIM_COMPONENT_PIPELINE_KEY
        ]
        assert isinstance(component_metadata, dict)
        component_metadata["trim_kind"] = TRIM_KIND_SKIRTING_BOARD
        self.record.pipeline[ARCHITECTURAL_TRIM_SEGMENTED_UV_PIPELINE_KEY] = {
            "version": ARCHITECTURAL_TRIM_SEGMENTED_UV_PIPELINE_VERSION,
            "texture_domain": (
                ARCHITECTURAL_TRIM_UV_TEXTURE_DOMAIN_LEFT_HALF
            ),
        }

        self.assertEqual(
            get_architectural_trim_uv_texture_domain(self.record),
            ARCHITECTURAL_TRIM_UV_TEXTURE_DOMAIN_LEFT_HALF,
        )
        self.assertEqual(
            get_cornice_uv_texture_domain(self.record),
            CORNICE_UV_TEXTURE_DOMAIN_FULL,
        )

    def test_generate_geometry_replaces_selected_trim_component(self) -> None:
        generation_request = object()
        with (
            patch.object(
                self.workspace,
                "_build_generation_request",
                return_value=generation_request,
            ) as build_request,
            patch.object(
                self.workspace,
                "_start_generation",
                return_value="trim-generation-operation",
            ) as start_generation,
        ):
            self.workspace.generate_geometry()

        build_request.assert_called_once_with(geometry_only=True)
        start_generation.assert_called_once()
        self.assertIs(start_generation.call_args.args[0], generation_request)
        target = start_generation.call_args.kwargs["architectural_trim_target"]
        self.assertEqual(target.level_index, 2)
        self.assertEqual(target.trim_id, self.trim_id)
        self.assertEqual(target.object_id, self.object_id)
        self.assertEqual(target.fit_dimensions, self.fit_dimensions)
        self.assertIs(
            start_generation.call_args.kwargs["replaced_object_record"],
            self.record,
        )
        self.assertNotIn("door_slot_target", start_generation.call_args.kwargs)

    def test_generate_texture_uses_selected_trim_instead_of_other_object(
        self,
    ) -> None:
        texture_request = object()
        selected_ids: list[str | None] = []

        def build_request() -> object:
            selected_ids.append(self.workspace._selected_object_id)
            return texture_request

        with (
            patch.object(
                self.workspace,
                "_build_texture_regeneration_request",
                side_effect=build_request,
            ) as build_texture_request,
            patch.object(
                self.workspace,
                "_start_texture_regeneration",
                return_value=True,
            ) as start_texture_regeneration,
        ):
            self.assertTrue(
                self.workspace.generate_selected_object_texture()
            )

        build_texture_request.assert_called_once_with()
        start_texture_regeneration.assert_called_once_with(texture_request)
        self.assertEqual(selected_ids, [self.object_id])
        target = self.workspace._architectural_trim_editing_target
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.object_id, self.object_id)

    def test_trim_texture_regeneration_refits_even_with_authored_uvs(self) -> None:
        reference_bgra = np.zeros((16, 16, 4), dtype=np.uint8)
        reference_bgra[:, :, 3] = 255
        self.record.pipeline[LOCALLY_AUTHORED_UVS_PIPELINE_KEY] = True
        with (
            patch.object(
                self.workspace,
                "_can_regenerate_object_texture",
                return_value=True,
            ),
            patch.object(
                self.workspace,
                "_build_current_object_generation_reference",
                return_value=reference_bgra,
            ),
        ):
            preflight = self.workspace._build_texture_regeneration_request()

        self.assertIsNotNone(preflight)
        assert preflight is not None
        self.assertTrue(preflight.enable_original_uv)
        self.assertEqual(
            preflight.architectural_trim_fit_dimensions,
            self.fit_dimensions,
        )
        materialized = _materialize_texture_regeneration_preflight(
            preflight,
            self.workspace._asset_directory,
        )

        self.assertEqual(materialized.request.object_id, self.object_id)
        self.assertTrue(materialized.request.model_glb)
        repeat_layout = materialized.request.architectural_trim_repeat_layout
        self.assertIsNotNone(repeat_layout)
        assert repeat_layout is not None
        self.assertEqual(repeat_layout.repeat_count, 4)
        np.testing.assert_allclose(
            import_generated_glb(materialized.request.model_glb).mesh.extents,
            repeat_layout.module_extents,
        )
        np.testing.assert_allclose(
            repeat_layout.target_extents,
            self.fit_dimensions,
        )
        self.assertEqual(
            materialized.request.architectural_trim_kind,
            TRIM_KIND_CORNICE,
        )
        self.assertIsNotNone(materialized.request.submitted_uv_fingerprint)

    def test_skirting_retexture_rebuilds_legacy_continuous_layout(self) -> None:
        skirting_object_id = "continuous-skirting-test-object"
        fit_dimensions = (6.0, 0.1, 0.2)
        record = self.workspace.register_architectural_trim_component_model(
            level_index=2,
            trim_id="b" * 32,
            object_id=skirting_object_id,
            object_name="Skirting board",
            fit_dimensions=fit_dimensions,
            model=_trim_component_model(fit_dimensions),
            trim_kind=TRIM_KIND_SKIRTING_BOARD,
        )
        record.pipeline[ARCHITECTURAL_TRIM_REPEAT_PIPELINE_KEY] = (
            ArchitecturalTrimRepeatLayout(
                target_extents=fit_dimensions,
                module_extents=fit_dimensions,
                repeat_count=1,
            ).to_pipeline_dict()
        )
        self.assertTrue(
            self.workspace.set_architectural_trim_editing_target(
                level_index=2,
                trim_id="b" * 32,
                object_id=skirting_object_id,
                object_name="Skirting board",
                fit_dimensions=fit_dimensions,
                trim_kind=TRIM_KIND_SKIRTING_BOARD,
            )
        )
        reference_bgra = np.zeros((16, 16, 4), dtype=np.uint8)
        reference_bgra[:, :, 3] = 255
        with (
            patch.object(
                self.workspace,
                "_can_regenerate_object_texture",
                return_value=True,
            ),
            patch.object(
                self.workspace,
                "_build_current_object_generation_reference",
                return_value=reference_bgra,
            ),
        ):
            preflight = self.workspace._build_texture_regeneration_request()

        self.assertIsNotNone(preflight)
        assert preflight is not None
        materialized = _materialize_texture_regeneration_preflight(
            preflight,
            self.workspace._asset_directory,
        )
        layout = materialized.request.architectural_trim_repeat_layout
        self.assertIsNotNone(layout)
        assert layout is not None
        self.assertEqual(layout.repeat_count, 8)
        self.assertLessEqual(
            layout.module_extents[0],
            max(layout.module_extents[1:]) * 4.0,
        )
        np.testing.assert_allclose(
            import_generated_glb(materialized.request.model_glb).mesh.extents,
            layout.module_extents,
        )

    def test_cornice_symmetric_option_builds_texture_only_half_preflight(
        self,
    ) -> None:
        reference_bgra = np.zeros((16, 16, 4), dtype=np.uint8)
        reference_bgra[:, :, 3] = 255
        self.workspace.symmetric_division_checkbox.setChecked(True)
        with (
            patch.object(
                self.workspace,
                "_can_regenerate_object_texture",
                return_value=True,
            ),
            patch.object(
                self.workspace,
                "_build_current_object_generation_reference",
                return_value=reference_bgra,
            ),
        ):
            preflight = self.workspace._build_texture_regeneration_request()

        self.assertIsNotNone(preflight)
        assert preflight is not None
        self.assertEqual(
            preflight.cornice_uv_texture_domain,
            CORNICE_UV_TEXTURE_DOMAIN_LEFT_HALF,
        )
        self.assertFalse(preflight.preserve_symmetric_uvs)
        self.assertIsNone(preflight.new_symmetric_division_orientation)
        with patch(
            "housemaker.generation_workspace."
            "build_architectural_trim_segmented_uv_glb",
            wraps=build_architectural_trim_segmented_uv_glb,
        ) as build_segmented_uv:
            materialized = _materialize_texture_regeneration_preflight(
                preflight,
                self.workspace._asset_directory,
            )

        self.assertEqual(
            materialized.request.cornice_uv_texture_domain,
            CORNICE_UV_TEXTURE_DOMAIN_LEFT_HALF,
        )
        self.assertEqual(
            build_segmented_uv.call_args.kwargs["texture_domain"],
            CORNICE_UV_TEXTURE_DOMAIN_LEFT_HALF,
        )
        for geometry in import_generated_glb(
            materialized.request.model_glb
        ).scene.geometry.values():
            uvs = np.asarray(geometry.visual.uv, dtype=np.float64)
            self.assertLessEqual(float(np.max(uvs[:, 0])), 0.5 + 1e-7)

    def test_non_cornice_trims_build_texture_only_half_preflights(self) -> None:
        reference_bgra = np.zeros((16, 16, 4), dtype=np.uint8)
        reference_bgra[:, :, 3] = 255
        for trim_kind, fit_dimensions, expects_repeat, expected_axis in (
            (
                TRIM_KIND_SKIRTING_BOARD,
                (6.0, 0.2, 0.4),
                True,
                ARCHITECTURAL_TRIM_UV_LONGITUDINAL_AXIS_X,
            ),
            (
                TRIM_KIND_EDGING_STRIP,
                (0.2, 0.2, 6.0),
                False,
                ARCHITECTURAL_TRIM_UV_LONGITUDINAL_AXIS_Z,
            ),
        ):
            with self.subTest(trim_kind=trim_kind):
                component_metadata = self.record.pipeline[
                    ARCHITECTURAL_TRIM_COMPONENT_PIPELINE_KEY
                ]
                assert isinstance(component_metadata, dict)
                component_metadata["trim_kind"] = trim_kind
                self.assertTrue(
                    self.workspace.set_architectural_trim_editing_target(
                        level_index=2,
                        trim_id=self.trim_id,
                        object_id=self.object_id,
                        object_name="Trim",
                        fit_dimensions=fit_dimensions,
                        trim_kind=trim_kind,
                    )
                )
                self.workspace.symmetric_division_checkbox.setChecked(True)
                with (
                    patch.object(
                        self.workspace,
                        "_can_regenerate_object_texture",
                        return_value=True,
                    ),
                    patch.object(
                        self.workspace,
                        "_build_current_object_generation_reference",
                        return_value=reference_bgra,
                    ),
                ):
                    preflight = (
                        self.workspace._build_texture_regeneration_request()
                    )

                self.assertIsNotNone(preflight)
                assert preflight is not None
                self.assertTrue(preflight.enable_original_uv)
                self.assertEqual(
                    preflight.cornice_uv_texture_domain,
                    ARCHITECTURAL_TRIM_UV_TEXTURE_DOMAIN_LEFT_HALF,
                )
                with patch(
                    "housemaker.generation_workspace."
                    "build_architectural_trim_segmented_uv_glb",
                    wraps=build_architectural_trim_segmented_uv_glb,
                ) as build_segmented_uv:
                    materialized = (
                        _materialize_texture_regeneration_preflight(
                            preflight,
                            self.workspace._asset_directory,
                        )
                    )
                self.assertEqual(
                    materialized.request.architectural_trim_kind,
                    trim_kind,
                )
                self.assertEqual(
                    build_segmented_uv.call_args.kwargs[
                        "longitudinal_axis"
                    ],
                    expected_axis,
                )
                self.assertEqual(
                    materialized.request.architectural_trim_repeat_layout
                    is not None,
                    expects_repeat,
                )
                for geometry in import_generated_glb(
                    materialized.request.model_glb
                ).scene.geometry.values():
                    uvs = np.asarray(geometry.visual.uv, dtype=np.float64)
                    self.assertLessEqual(
                        float(np.max(uvs[:, 0])),
                        0.5 + 1e-7,
                    )
                np.testing.assert_allclose(
                    import_generated_glb(
                        materialized.request.model_glb
                    ).mesh.extents,
                    (
                        materialized.request
                        .architectural_trim_repeat_layout.module_extents
                        if expects_repeat
                        else fit_dimensions
                    ),
                )

    def test_trim_texture_regeneration_requires_scene_trim_target(self) -> None:
        reference_bgra = np.zeros((16, 16, 4), dtype=np.uint8)
        reference_bgra[:, :, 3] = 255
        self.workspace.clear_architectural_trim_editing_target()
        self.workspace._selected_object_id = self.object_id
        with (
            patch.object(
                self.workspace,
                "_can_regenerate_object_texture",
                return_value=True,
            ),
            patch.object(
                self.workspace,
                "_build_current_object_generation_reference",
                return_value=reference_bgra,
            ),
        ):
            preflight = self.workspace._build_texture_regeneration_request()

        self.assertIsNone(preflight)
        self.assertIn(
            "Select the architectural trim in the 3D scene",
            self.workspace.status_label.text(),
        )

    def test_trim_texture_persistence_repeats_variants_and_keeps_module_source(
        self,
    ) -> None:
        module_glb, repeat_layout = prepare_architectural_trim_repeat_module(
            _trim_component_model().glb_bytes,
            self.fit_dimensions,
        )
        provider_glb = _trim_component_model().glb_bytes
        variants = _module_texture_variants(provider_glb)
        generated_model = import_generated_glb(provider_glb)
        generated_model.object_texture_variants = variants
        reference = np.zeros((16, 16, 4), dtype=np.uint8)
        reference[:, :, 3] = 255
        request = TextureRegenerationRequest(
            object_id=self.object_id,
            reference_frame_index=0,
            reference_image_bgra=reference,
            model_glb=module_glb,
            settings=GenerationServiceSettings(),
            architectural_trim_repeat_layout=repeat_layout,
        )
        outcome = TextureRegenerationOutcome(
            request=request,
            result=MeshyGenerationResult(
                task_id="trim-texture-task",
                glb_bytes=module_glb,
            ),
        )

        saved = _prepare_and_persist_texture_regeneration(
            self.workspace._asset_directory,
            outcome,
            generated_model,
            None,
            1024,
            self.record,
        )

        self.assertEqual(
            saved.next_pipeline[ARCHITECTURAL_TRIM_REPEAT_PIPELINE_KEY],
            repeat_layout.to_pipeline_dict(),
        )
        module_asset = self.workspace._asset_directory.joinpath(
            saved.next_pipeline["postprocessed_asset_path"]
        )
        expected_module_bounds = np.asarray(
            (
                (
                    -repeat_layout.module_extents[0] * 0.5,
                    -repeat_layout.module_extents[1] * 0.5,
                    0.0,
                ),
                (
                    repeat_layout.module_extents[0] * 0.5,
                    repeat_layout.module_extents[1] * 0.5,
                    repeat_layout.module_extents[2],
                ),
            ),
            dtype=float,
        )
        np.testing.assert_allclose(
            import_generated_glb(module_asset.read_bytes()).mesh.bounds,
            expected_module_bounds,
        )
        for variant in saved.next_pipeline["texture_variants"].values():
            repeated_asset = self.workspace._asset_directory.joinpath(
                variant["glb_asset_path"]
            )
            np.testing.assert_allclose(
                import_generated_glb(repeated_asset.read_bytes()).mesh.extents,
                self.fit_dimensions,
            )
        np.testing.assert_allclose(
            saved.preview_model.mesh.extents,
            self.fit_dimensions,
        )

    def test_trim_preview_transfer_reuses_in_memory_retarget(self) -> None:
        module_glb, repeat_layout = prepare_architectural_trim_repeat_module(
            _trim_component_model().glb_bytes,
            self.fit_dimensions,
        )
        module_asset_path = self.workspace._persist_meshy_named_asset(
            "preview-repeat-module.glb",
            module_glb,
        )
        record_index = self.workspace._data.generated_objects.index(self.record)
        self.workspace._data.generated_objects[record_index] = replace(
            self.record,
            pipeline={
                **self.record.pipeline,
                ARCHITECTURAL_TRIM_REPEAT_PIPELINE_KEY: (
                    repeat_layout.to_pipeline_dict()
                ),
                "postprocessed_asset_path": module_asset_path,
            },
        )
        target_extents = (4.5, 0.3, 0.6)

        with (
            patch(
                "housemaker.generation_workspace."
                "retarget_architectural_trim_repeat_module_preview_model",
                wraps=retarget_architectural_trim_repeat_module_preview_model,
            ) as preview_retarget,
            patch(
                "housemaker.generation_workspace."
                "retarget_architectural_trim_repeat_module_glb",
                side_effect=AssertionError("preview used authoritative GLB path"),
            ),
        ):
            first_model = (
                self.workspace
                .build_architectural_trim_texture_transfer_preview_model(
                    self.object_id,
                    target_extents,
                )
            )
            second_model = (
                self.workspace
                .build_architectural_trim_texture_transfer_preview_model(
                    self.object_id,
                    target_extents,
                )
            )

        self.assertIsNotNone(first_model)
        assert first_model is not None
        self.assertIs(first_model, second_model)
        self.assertEqual(first_model.glb_bytes, b"")
        np.testing.assert_allclose(first_model.mesh.extents, target_extents)
        preview_retarget.assert_called_once()
        self.assertEqual(
            len(self.workspace._architectural_trim_preview_source_model_cache),
            1,
        )
        self.assertEqual(
            len(self.workspace._architectural_trim_preview_transfer_model_cache),
            1,
        )

    def test_cornice_half_texture_persistence_keeps_full_geometry(self) -> None:
        textured_glb = _left_packed_textured_cornice_glb()
        module_glb, repeat_layout = prepare_architectural_trim_repeat_module(
            textured_glb,
            self.fit_dimensions,
        )
        variants = _module_texture_variants(module_glb)
        generated_model = import_generated_glb(module_glb)
        generated_model.object_texture_variants = variants
        reference = np.zeros((16, 16, 4), dtype=np.uint8)
        reference[:, :, 3] = 255
        fingerprint = build_uv_fingerprint(module_glb)
        request = TextureRegenerationRequest(
            object_id=self.object_id,
            reference_frame_index=0,
            reference_image_bgra=reference,
            model_glb=module_glb,
            settings=GenerationServiceSettings(),
            enable_original_uv=True,
            submitted_uv_fingerprint=fingerprint,
            architectural_trim_repeat_layout=repeat_layout,
            architectural_trim_kind=TRIM_KIND_CORNICE,
            cornice_uv_texture_domain=(
                CORNICE_UV_TEXTURE_DOMAIN_LEFT_HALF
            ),
        )
        outcome = TextureRegenerationOutcome(
            request=request,
            result=MeshyGenerationResult(
                task_id="cornice-half-texture-task",
                glb_bytes=module_glb,
            ),
            final_uv_fingerprint=fingerprint,
        )

        with patch(
            "housemaker.generation_workspace."
            "build_symmetric_half_texture_variants",
            wraps=build_symmetric_half_texture_variants,
        ) as build_half_variants:
            saved = _prepare_and_persist_texture_regeneration(
                self.workspace._asset_directory,
                outcome,
                generated_model,
                None,
                1024,
                self.record,
            )

        build_half_variants.assert_called_once()
        self.assertTrue(
            build_half_variants.call_args.kwargs["uvs_already_left_packed"]
        )
        self.assertEqual(
            saved.next_pipeline[
                ARCHITECTURAL_TRIM_SEGMENTED_UV_PIPELINE_KEY
            ],
            {
                "version": ARCHITECTURAL_TRIM_SEGMENTED_UV_PIPELINE_VERSION,
                "texture_domain": (
                    ARCHITECTURAL_TRIM_UV_TEXTURE_DOMAIN_LEFT_HALF
                ),
            },
        )
        self.assertNotIn(
            SYMMETRIC_DIVISION_PIPELINE_KEY,
            saved.next_pipeline,
        )
        variant_metadata = saved.next_pipeline["texture_variants"]
        self.assertEqual(set(variant_metadata), {"512", "1024", "2048"})
        for raw_resolution, variant in variant_metadata.items():
            resolution = int(raw_resolution)
            texture_path = self.workspace._asset_directory.joinpath(
                variant["texture_asset_path"]
            )
            with Image.open(texture_path) as texture_image:
                pixels = np.asarray(texture_image.convert("RGBA"))
            self.assertEqual(pixels.shape, (resolution, resolution, 4))
            self.assertTrue(
                np.all(
                    pixels[:, resolution // 2 :]
                    == np.asarray((0, 0, 0, 255), dtype=np.uint8)
                )
            )
            self.assertTrue(
                np.any(pixels[:, : resolution // 2, :3] != 0)
            )
            model_path = self.workspace._asset_directory.joinpath(
                variant["glb_asset_path"]
            )
            model = import_generated_glb(model_path.read_bytes())
            np.testing.assert_allclose(
                model.mesh.extents,
                self.fit_dimensions,
            )
            for geometry in model.scene.geometry.values():
                uvs = np.asarray(geometry.visual.uv, dtype=np.float64)
                self.assertLessEqual(
                    float(np.max(uvs[:, 0])),
                    0.5 + 1e-7,
                )
        np.testing.assert_allclose(
            saved.preview_model.mesh.extents,
            self.fit_dimensions,
        )

    def test_edging_half_texture_persistence_never_adds_mirror_metadata(
        self,
    ) -> None:
        model_glb = _left_packed_textured_cornice_glb()
        variants = _module_texture_variants(model_glb)
        generated_model = import_generated_glb(model_glb)
        generated_model.object_texture_variants = variants
        reference = np.zeros((16, 16, 4), dtype=np.uint8)
        reference[:, :, 3] = 255
        fingerprint = build_uv_fingerprint(model_glb)
        request = TextureRegenerationRequest(
            object_id=self.object_id,
            reference_frame_index=0,
            reference_image_bgra=reference,
            model_glb=model_glb,
            settings=GenerationServiceSettings(),
            enable_original_uv=True,
            submitted_uv_fingerprint=fingerprint,
            architectural_trim_kind=TRIM_KIND_EDGING_STRIP,
            cornice_uv_texture_domain=(
                ARCHITECTURAL_TRIM_UV_TEXTURE_DOMAIN_LEFT_HALF
            ),
        )
        saved = _prepare_and_persist_texture_regeneration(
            self.workspace._asset_directory,
            TextureRegenerationOutcome(
                request=request,
                result=MeshyGenerationResult(
                    task_id="edging-half-texture-task",
                    glb_bytes=model_glb,
                ),
                final_uv_fingerprint=fingerprint,
            ),
            generated_model,
            None,
            1024,
            self.record,
        )

        self.assertEqual(
            saved.next_pipeline[
                ARCHITECTURAL_TRIM_SEGMENTED_UV_PIPELINE_KEY
            ]["texture_domain"],
            ARCHITECTURAL_TRIM_UV_TEXTURE_DOMAIN_LEFT_HALF,
        )
        self.assertNotIn(
            SYMMETRIC_DIVISION_PIPELINE_KEY,
            saved.next_pipeline,
        )
        texture_path = self.workspace._asset_directory.joinpath(
            saved.next_pipeline["texture_variants"]["1024"]
            ["texture_asset_path"]
        )
        with Image.open(texture_path) as texture_image:
            pixels = np.asarray(texture_image.convert("RGBA"))
        self.assertTrue(
            np.all(
                pixels[:, 512:]
                == np.asarray((0, 0, 0, 255), dtype=np.uint8)
            )
        )

    def test_older_trim_generation_completion_keeps_newer_trim_selected(
        self,
    ) -> None:
        first_target = self.workspace._architectural_trim_editing_target
        self.assertIsNotNone(first_target)
        assert first_target is not None
        operation = _ActiveObjectOperation(
            kind=OBJECT_OPERATION_GENERATE_MODEL,
            target_object_id=self.object_id,
            architectural_trim_target=first_target,
            replaced_object_record=self.record,
        )
        self.workspace._active_object_operation = operation
        self.addCleanup(
            setattr,
            self.workspace,
            "_active_object_operation",
            None,
        )

        second_trim_id = "b" * 32
        second_object_id = "architectural-trim-second-object"
        second_model = _trim_component_model()
        second_record = self.workspace.register_architectural_trim_component_model(
            level_index=2,
            trim_id=second_trim_id,
            object_id=second_object_id,
            object_name="Second cornice",
            fit_dimensions=self.fit_dimensions,
            model=second_model,
            trim_kind=TRIM_KIND_CORNICE,
        )
        self.assertTrue(
            self.workspace.set_architectural_trim_editing_target(
                level_index=2,
                trim_id=second_trim_id,
                object_id=second_object_id,
                object_name=second_record.object_name,
                fit_dimensions=self.fit_dimensions,
                trim_kind=TRIM_KIND_CORNICE,
            )
        )

        replacement_model = _trim_component_model()
        replacement_asset_path = self.workspace._persist_meshy_named_asset(
            "first-cornice-replacement.glb",
            replacement_model.glb_bytes,
        )
        replacement_record = replace(
            self.record,
            asset_path=replacement_asset_path,
        )

        self.assertTrue(
            self.workspace._commit_generated_model_record(
                replacement_record,
                replacement_model,
                self.record,
                operation_id=None,
            )
        )

        active_target = self.workspace._architectural_trim_editing_target
        self.assertIsNotNone(active_target)
        assert active_target is not None
        self.assertEqual(active_target.object_id, second_object_id)
        self.assertEqual(self.workspace._selected_object_id, second_object_id)
        self.assertIs(self.workspace.result_view.model, second_model)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
