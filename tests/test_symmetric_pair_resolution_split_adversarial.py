# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from housemaker.texture_atlas_state import (
    ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
    ATLAS_SLOT_HALF_LEFT,
    TextureAtlasData,
)
from housemaker.texture_atlas_workspace import (
    AtlasObjectTextureSource,
    TextureAtlasWorkspace,
)


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])


# ### Fixture helpers ###
_OPAQUE_BLACK = np.asarray((0, 0, 0, 255), dtype=np.uint8)


def _pair_source(
    directory: Path,
    object_id: str,
    resolution: int,
    color: tuple[int, int, int, int],
) -> AtlasObjectTextureSource:
    physical_resolution = resolution
    texture_path = directory / f"{object_id}-{resolution}.png"
    texture = Image.new(
        "RGBA",
        (physical_resolution, physical_resolution),
        tuple(int(value) for value in _OPAQUE_BLACK),
    )
    texture.paste(color, (0, 0, resolution // 2, physical_resolution))
    texture.save(texture_path)
    preview = np.empty((8, 8, 4), dtype=np.uint8)
    preview[:] = np.asarray(color, dtype=np.uint8)
    return AtlasObjectTextureSource(
        object_id=object_id,
        object_name=object_id.title(),
        texture_path=f"generated/{texture_path.name}",
        texture_resolution=resolution,
        physical_texture_path=texture_path,
        preview_rgba=preview,
        packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
        symmetric_preview_orientation="vertical",
        symmetric_preview_plane_coordinate=0.0,
    )


# ### Resolution split adversary ###
class SymmetricPairResolutionSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.workspace = TextureAtlasWorkspace(
            asset_directory=self.root / "atlases"
        )

    def tearDown(self) -> None:
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def test_resolution_change_splits_pair_and_compacts_survivor_left(
        self,
    ) -> None:
        target_512 = _pair_source(
            self.root,
            "target",
            512,
            (211, 37, 73, 255),
        )
        target_1024 = _pair_source(
            self.root,
            "target",
            1024,
            (53, 101, 223, 255),
        )
        partner = _pair_source(
            self.root,
            "partner",
            512,
            (31, 197, 89, 255),
        )
        variants = {
            (source.object_id, source.texture_resolution): source
            for source in (target_512, target_1024, partner)
        }
        data = TextureAtlasData()
        atlas = data.create_atlas("Split pair", 2048, atlas_id="pair-split")
        for source in (target_512, partner):
            data.assign_object(
                atlas.atlas_id,
                source.object_id,
                source.texture_path,
                source.texture_resolution,
                source.packing_mode,
            )
        initial_target = atlas.placement_for_object(target_512.object_id)
        initial_partner = atlas.placement_for_object(partner.object_id)
        assert initial_target is not None and initial_partner is not None
        self.assertEqual(
            (initial_target.x, initial_target.y, initial_target.size),
            (initial_partner.x, initial_partner.y, initial_partner.size),
        )

        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [target_512, partner],
            variant_resolver=lambda object_id, resolution: variants.get(
                (object_id, resolution)
            ),
        )
        commit = Mock(return_value=True)

        changed = self.workspace.set_object_texture_resolution(
            target_512.object_id,
            1024,
            commit_callback=commit,
        )

        self.assertTrue(changed)
        commit.assert_called_once_with()
        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        resized_target = updated.placement_for_object(target_512.object_id)
        surviving_partner = updated.placement_for_object(partner.object_id)
        assert resized_target is not None and surviving_partner is not None
        self.assertEqual(resized_target.texture_resolution, 1024)
        self.assertEqual(resized_target.size, 1024)
        self.assertEqual(resized_target.slot_half, ATLAS_SLOT_HALF_LEFT)
        self.assertEqual(surviving_partner.texture_resolution, 512)
        self.assertEqual(surviving_partner.size, 512)
        self.assertEqual(surviving_partner.slot_half, ATLAS_SLOT_HALF_LEFT)
        self.assertNotEqual(
            (resized_target.x, resized_target.y),
            (surviving_partner.x, surviving_partner.y),
        )

        assert updated.image_path is not None
        with Image.open(self.root / "atlases" / updated.image_path) as image:
            composite = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        target_left = composite[
            resized_target.y + 768,
            resized_target.x + 256,
        ]
        target_right = composite[
            resized_target.y + 768,
            resized_target.x + 768,
        ]
        partner_left = composite[
            surviving_partner.y + 384,
            surviving_partner.x + 128,
        ]
        partner_right = composite[
            surviving_partner.y + 384,
            surviving_partner.x + 384,
        ]
        self.assertTrue(np.any(target_left[:3] != 0))
        self.assertTrue(np.any(partner_left[:3] != 0))
        np.testing.assert_array_equal(target_right, _OPAQUE_BLACK)
        np.testing.assert_array_equal(partner_right, _OPAQUE_BLACK)

    def test_rejected_resolution_change_restores_square_pair_and_png(self) -> None:
        target_512 = _pair_source(
            self.root,
            "rollback-target",
            512,
            (211, 37, 73, 255),
        )
        target_1024 = _pair_source(
            self.root,
            "rollback-target",
            1024,
            (53, 101, 223, 255),
        )
        partner = _pair_source(
            self.root,
            "rollback-partner",
            512,
            (31, 197, 89, 255),
        )
        variants = {
            (source.object_id, source.texture_resolution): source
            for source in (target_512, target_1024, partner)
        }
        data = TextureAtlasData()
        atlas = data.create_atlas("Rollback pair", 2048, atlas_id="rollback")
        for source in (target_512, partner):
            data.assign_object(
                atlas.atlas_id,
                source.object_id,
                source.texture_path,
                source.texture_resolution,
                source.packing_mode,
            )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [target_512, partner],
            variant_resolver=lambda object_id, resolution: variants.get(
                (object_id, resolution)
            ),
        )
        self.assertEqual(self.workspace.materialize_missing_atlases(), 1)
        before = self.workspace.get_data()
        before_atlas = before.atlas_by_id(atlas.atlas_id)
        assert before_atlas is not None and before_atlas.image_path is not None
        png_path = self.root / "atlases" / before_atlas.image_path
        png_before = png_path.read_bytes()
        commit = Mock(return_value=False)

        changed = self.workspace.set_object_texture_resolution(
            target_512.object_id,
            1024,
            commit_callback=commit,
        )

        self.assertFalse(changed)
        commit.assert_called_once_with()
        self.assertEqual(self.workspace.get_data(), before)
        self.assertEqual(png_path.read_bytes(), png_before)


if __name__ == "__main__":
    unittest.main()
