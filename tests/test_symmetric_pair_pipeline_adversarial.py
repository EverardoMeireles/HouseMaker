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
from PIL import Image
from PySide6.QtWidgets import QApplication

from housemaker.generation_workspace import (
    SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY,
    SYMMETRIC_DIVISION_PIPELINE_KEY,
    TEXTURE_VARIANTS_PIPELINE_KEY,
    GenerationRequest,
    GenerationWorkspace,
)
from housemaker.glb import import_generated_glb
from housemaker.main import BlueprintWorkspace
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.object_symmetry import (
    AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION,
    SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL,
    SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
    SYMMETRIC_DIVISION_SIDE_BOTTOM,
    SYMMETRIC_DIVISION_SIDE_LEFT,
    SYMMETRIC_SELECTION_MODE_FEWEST_TRIANGLES_RANDOM_TIE,
    SYMMETRIC_TEXTURE_CONTENT_HALF_LEFT,
    SYMMETRIC_TEXTURE_PACKING_MODE_PAIR,
    SymmetricDivisionMetadata,
    SymmetricDivisionResult,
    SymmetricSquarePairTextureVariants,
)
from housemaker.object_texture_variants import (
    TEXTURE_RESOLUTIONS,
    ObjectTextureVariants,
)
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.texture_atlas_state import (
    ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
    ATLAS_STATE_SCHEMA_VERSION,
    ATLAS_SLOT_HALF_LEFT,
    ATLAS_SLOT_HALF_RIGHT,
    TextureAtlasData,
)
from housemaker.texture_atlas_workspace import TextureAtlasWorkspace


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])


# ### Fixture helpers ###
_OPAQUE_BLACK = np.asarray((0, 0, 0, 255), dtype=np.uint8)


def _box_glb(scale: float = 1.0) -> bytes:
    mesh = trimesh.creation.box(extents=(scale, scale * 0.8, scale * 0.6))
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _encode_rgba_png(rgba: np.ndarray) -> bytes:
    bgra = cv2.cvtColor(np.asarray(rgba, dtype=np.uint8), cv2.COLOR_RGBA2BGRA)
    encoded, payload = cv2.imencode(".png", bgra)
    if not encoded:
        raise RuntimeError("The adversarial fixture PNG could not be encoded.")
    return bytes(payload)


def _ordinary_variants(seed: int) -> ObjectTextureVariants:
    glb = _box_glb(1.0 + seed / 100.0)
    return ObjectTextureVariants(
        glb_by_resolution={resolution: glb for resolution in TEXTURE_RESOLUTIONS},
        texture_png_by_resolution={
            resolution: _encode_rgba_png(
                np.full(
                    (8, 8, 4),
                    (seed * 31 % 255, resolution // 16 % 255, 90, 255),
                    dtype=np.uint8,
                )
            )
            for resolution in TEXTURE_RESOLUTIONS
        },
        preview_rgba_by_resolution={
            resolution: np.full(
                (8, 8, 4),
                (seed * 31 % 255, 80, resolution // 16 % 255, 255),
                dtype=np.uint8,
            )
            for resolution in TEXTURE_RESOLUTIONS
        },
    )


def _model_with_variants(variants: ObjectTextureVariants):
    model = import_generated_glb(variants.glb_by_resolution[1024])
    model.object_texture_variants = variants
    return model


def _pair_atlas(
    content_resolution: int,
    color: tuple[int, int, int, int],
) -> np.ndarray:
    physical_resolution = content_resolution
    atlas = np.empty(
        (physical_resolution, physical_resolution, 4),
        dtype=np.uint8,
    )
    atlas[:] = _OPAQUE_BLACK
    atlas[:, : content_resolution // 2] = color
    return atlas


def _pair_variants(
    color: tuple[int, int, int, int],
) -> SymmetricSquarePairTextureVariants:
    glb = _box_glb(0.75)
    preview_by_resolution = {
        resolution: _pair_atlas(resolution, color)
        for resolution in (512, 1024)
    }
    return SymmetricSquarePairTextureVariants(
        glb_by_resolution={512: glb, 1024: glb},
        texture_png_by_resolution={
            resolution: _encode_rgba_png(preview)
            for resolution, preview in preview_by_resolution.items()
        },
        preview_rgba_by_resolution=preview_by_resolution,
    )


def _automatic_result(
    variants: SymmetricSquarePairTextureVariants,
    orientation: str,
) -> SymmetricDivisionResult:
    if orientation == SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL:
        kept_side = SYMMETRIC_DIVISION_SIDE_BOTTOM
        counts = (("bottom", 3), ("top", 8))
    else:
        kept_side = SYMMETRIC_DIVISION_SIDE_LEFT
        counts = (("left", 2), ("right", 7))
    metadata = SymmetricDivisionMetadata(
        version=AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION,
        orientation=orientation,
        kept_side=kept_side,
        plane_coordinate=0.0,
        packing_mode=SYMMETRIC_TEXTURE_PACKING_MODE_PAIR,
        texture_content_half=SYMMETRIC_TEXTURE_CONTENT_HALF_LEFT,
        selection_mode=SYMMETRIC_SELECTION_MODE_FEWEST_TRIANGLES_RANDOM_TIE,
        triangle_count_by_side=counts,
        tie_broken_randomly=False,
    )
    return SymmetricDivisionResult(
        variants=variants,
        orientation=orientation,
        kept_side=kept_side,
        plane_coordinate=metadata.plane_coordinate,
        metadata=metadata,
    )


def _request(*, enabled: bool, orientation: str) -> GenerationRequest:
    return GenerationRequest(
        frame_index=3,
        selected_object_bgra=np.full((4, 4, 4), 255, dtype=np.uint8),
        settings=GenerationServiceSettings(meshy_api_key="key"),
        symmetric_division_enabled=enabled,
        symmetric_division_orientation=orientation,
    )


# ### Generation lifecycle adversaries ###
class SymmetricGenerationRequestSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = GenerationWorkspace(
            asset_directory=Path(self.temporary_directory.name) / "generated"
        )

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.temporary_directory.cleanup()

    def test_in_flight_request_not_later_checkbox_state_controls_division(
        self,
    ) -> None:
        source_variants = _ordinary_variants(1)
        pair_variants = _pair_variants((211, 31, 67, 255))
        self.workspace._active_generation_request = _request(
            enabled=True,
            orientation=SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL,
        )

        self.workspace.symmetric_division_checkbox.setChecked(False)
        self.workspace.symmetric_division_orientation_combo.setCurrentIndex(
            self.workspace.symmetric_division_orientation_combo.findData(
                SYMMETRIC_DIVISION_ORIENTATION_VERTICAL
            )
        )
        with patch(
            "housemaker.generation_workspace."
            "build_automatic_symmetric_object_variants",
            return_value=_automatic_result(
                pair_variants,
                SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL,
            ),
        ) as automatic_transform:
            self.workspace._handle_generation_succeeded(
                MeshyGenerationResult(
                    "snapshot-symmetric",
                    source_variants.glb_by_resolution[2048],
                    "Captured symmetric",
                ),
                _model_with_variants(source_variants),
            )

        automatic_transform.assert_called_once_with(
            source_variants.glb_by_resolution[2048],
            SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL,
        )
        first_record = self.workspace.get_data().generated_objects[0]
        self.assertEqual(
            first_record.pipeline[SYMMETRIC_DIVISION_PIPELINE_KEY][
                "orientation"
            ],
            SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL,
        )

        ordinary_source = _ordinary_variants(2)
        self.workspace._active_generation_request = _request(
            enabled=False,
            orientation=SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
        )
        self.workspace.symmetric_division_checkbox.setChecked(True)
        self.workspace.symmetric_division_orientation_combo.setCurrentIndex(
            self.workspace.symmetric_division_orientation_combo.findData(
                SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL
            )
        )
        with patch(
            "housemaker.generation_workspace."
            "build_automatic_symmetric_object_variants"
        ) as automatic_transform:
            self.workspace._handle_generation_succeeded(
                MeshyGenerationResult(
                    "snapshot-ordinary",
                    ordinary_source.glb_by_resolution[2048],
                    "Captured ordinary",
                ),
                _model_with_variants(ordinary_source),
            )

        automatic_transform.assert_not_called()
        second_record = self.workspace.get_data().generated_objects[1]
        self.assertNotIn(SYMMETRIC_DIVISION_PIPELINE_KEY, second_record.pipeline)
        self.assertEqual(
            set(second_record.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY]),
            {"512", "1024", "2048"},
        )
        self.assertIsNone(
            self.workspace.result_view._symmetric_preview_orientation
        )


# ### Persisted Generation-to-Atlas integration ###
class SymmetricPairAtlasPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.generation = GenerationWorkspace(
            asset_directory=root / "generated"
        )
        self.atlas = TextureAtlasWorkspace(
            asset_directory=root / "atlases"
        )

    def tearDown(self) -> None:
        self.generation.shutdown()
        self.generation.close()
        self.atlas.close()
        self.temporary_directory.cleanup()

    def test_two_persisted_halves_fill_one_ordinary_slot_left_to_right(
        self,
    ) -> None:
        colors = (
            (221, 37, 53, 255),
            (41, 193, 79, 255),
        )
        records = []
        for index, color in enumerate(colors):
            source_variants = _ordinary_variants(20 + index)
            pair_variants = _pair_variants(color)
            orientation = (
                SYMMETRIC_DIVISION_ORIENTATION_VERTICAL
                if index == 0
                else SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL
            )
            self.generation._active_generation_request = _request(
                enabled=True,
                orientation=orientation,
            )
            with patch(
                "housemaker.generation_workspace."
                "build_automatic_symmetric_object_variants",
                return_value=_automatic_result(
                    pair_variants,
                    orientation,
                ),
            ):
                self.generation._handle_generation_succeeded(
                    MeshyGenerationResult(
                        f"pair-{index}",
                        source_variants.glb_by_resolution[2048],
                        f"Pair {index}",
                    ),
                    _model_with_variants(source_variants),
                )
            records.append(self.generation.get_data().generated_objects[-1])

        sources = []
        for record in records:
            variant = self.generation.get_active_texture_variant(
                record.object_id
            )
            symmetry = self.generation.get_object_symmetric_division(
                record.object_id
            )
            source = BlueprintWorkspace._build_atlas_object_texture_source(
                variant,
                symmetry,
            )
            self.assertIsNotNone(source)
            assert source is not None
            self.assertEqual(source.texture_resolution, 1024)
            self.assertEqual(
                source.packing_mode,
                ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
            )
            self.assertEqual(source.load_texture_rgba().shape, (1024, 1024, 4))
            sources.append(source)

        self.atlas.set_object_texture_sources(sources)
        data = TextureAtlasData()
        data.create_atlas("Two halves", 2048, atlas_id="pair-atlas")
        self.atlas.set_data(data)
        selected_atlas = self.atlas.selected_atlas
        assert selected_atlas is not None
        for source in sources:
            self.atlas._data.assign_object(
                selected_atlas.atlas_id,
                source.object_id,
                source.texture_path,
                source.texture_resolution,
                source.packing_mode,
            )
        self.atlas._materialize_atlas(selected_atlas)

        placements = {
            placement.object_id: placement
            for placement in selected_atlas.placements
        }
        self.assertEqual(
            [placements[source.object_id].slot_half for source in sources],
            [ATLAS_SLOT_HALF_LEFT, ATLAS_SLOT_HALF_RIGHT],
        )
        self.assertEqual(
            {
                (
                    placement.x,
                    placement.y,
                    placement.size,
                    placement.texture_resolution,
                )
                for placement in placements.values()
            },
            {(0, 0, 1024, 1024)},
        )
        serialized = self.atlas.get_data().to_dict()
        self.assertEqual(
            serialized["schema_version"],
            ATLAS_STATE_SCHEMA_VERSION,
        )
        round_tripped = TextureAtlasData.from_dict(serialized)
        self.assertEqual(round_tripped.to_dict(), serialized)

        output_path = Path(self.atlas._asset_directory) / "pair-atlas.png"
        with Image.open(output_path) as image:
            composite = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        self.assertEqual(composite.shape, (2048, 2048, 4))
        sample_points = (
            (512, 256),
            (512, 768),
        )
        for color, (sample_y, sample_x) in zip(colors, sample_points):
            self.assertEqual(
                tuple(int(value) for value in composite[sample_y, sample_x]),
                color,
            )
        for record in records:
            self.assertEqual(
                record.pipeline[SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY],
                1024,
            )
            self.assertEqual(
                set(record.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY]),
                {"512", "1024"},
            )


if __name__ == "__main__":
    unittest.main()
