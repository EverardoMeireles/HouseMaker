# ### Imports ###
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from housemaker.camera_models import CameraPose, InitialFirstPersonCamera
from housemaker.generation_state import GenerationData
from housemaker.models import create_default_doorway_presets, create_default_levels
from housemaker.project_io import ProjectData, load_project, save_project
from housemaker.surface_texture_state import SurfaceTextureData
from housemaker.texture_atlas_state import (
    TextureAtlasData,
    write_texture_atlas_metadata,
    write_texture_atlas_png,
)


# ### Texture atlas state tests ###
class TextureAtlasStateTests(unittest.TestCase):
    def test_four_512_textures_share_one_1024_quadrant(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Furniture", 4096, atlas_id="atlas-a")

        for object_id in ("chair-d", "chair-b", "chair-c", "chair-a"):
            data.assign_object(
                atlas.atlas_id,
                object_id,
                f"textures/{object_id}-512.png",
                512,
            )

        positions = {
            placement.object_id: (placement.x, placement.y)
            for placement in atlas.placements
        }
        self.assertEqual(
            positions,
            {
                "chair-d": (0, 0),
                "chair-b": (512, 0),
                "chair-c": (0, 512),
                "chair-a": (512, 512),
            },
        )
        self.assertTrue(
            all(
                placement.x + placement.size <= 1024
                and placement.y + placement.size <= 1024
                for placement in atlas.placements
            )
        )

    def test_mixed_sizes_preserve_existing_slots_after_add_and_removal(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Mixed", 4096, atlas_id="atlas-a")
        data.assign_object("atlas-a", "small", "textures/small.png", 512)
        data.assign_object("atlas-a", "large", "textures/large.png", 2048)
        data.assign_object("atlas-a", "medium", "textures/medium.png", 1024)

        self.assertEqual(
            [(item.object_id, item.x, item.y) for item in atlas.placements],
            [
                ("small", 0, 0),
                ("large", 2048, 0),
                ("medium", 1024, 0),
            ],
        )
        self.assertTrue(data.unassign_object("atlas-a", "large"))
        self.assertEqual(
            [(item.object_id, item.x, item.y) for item in atlas.placements],
            [("small", 0, 0), ("medium", 1024, 0)],
        )

    def test_assignment_failure_does_not_change_existing_placements(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Full", 2048, atlas_id="atlas-a")
        for index in range(4):
            data.assign_object(
                "atlas-a",
                f"large-{index}",
                f"textures/large-{index}.png",
                1024,
            )
        before = list(atlas.placements)

        with self.assertRaisesRegex(ValueError, "has no space"):
            data.assign_object(
                "atlas-a",
                "overflow",
                "textures/overflow.png",
                512,
            )

        self.assertEqual(atlas.placements, before)

    def test_exact_manual_placement_preserves_slots_and_round_trips(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Manual", 4096, atlas_id="atlas-a")
        data.place_object_at(
            atlas.atlas_id,
            "large",
            "textures/large.png",
            1024,
            2048,
            1024,
        )
        moved = data.place_object_at(
            atlas.atlas_id,
            "small",
            "textures/small.png",
            512,
            3584,
            3584,
        )

        self.assertEqual((moved.x, moved.y), (3584, 3584))
        self.assertEqual(
            [
                (placement.object_id, placement.x, placement.y)
                for placement in atlas.placements
            ],
            [("large", 2048, 1024), ("small", 3584, 3584)],
        )
        self.assertEqual(TextureAtlasData.from_dict(data.to_dict()), data)

    def test_manual_placement_rejects_grid_bounds_and_collisions_atomically(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Manual", 2048, atlas_id="atlas-a")
        data.place_object_at(
            atlas.atlas_id,
            "existing",
            "textures/existing.png",
            1024,
            0,
            0,
        )
        before = data.clone()

        with self.assertRaisesRegex(ValueError, "texture-size grid"):
            data.place_object_at(
                atlas.atlas_id,
                "misaligned",
                "textures/misaligned.png",
                512,
                256,
                1024,
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            data.place_object_at(
                atlas.atlas_id,
                "collision",
                "textures/collision.png",
                512,
                512,
                0,
            )
        with self.assertRaisesRegex(ValueError, "bounds"):
            data.place_object_at(
                atlas.atlas_id,
                "outside",
                "textures/outside.png",
                512,
                2048,
                0,
            )

        self.assertEqual(data, before)

    def test_selection_and_serialization_round_trip(self) -> None:
        data = TextureAtlasData()
        data.create_atlas("First", 2048, atlas_id="atlas-a")
        data.create_atlas("Second", 4096, atlas_id="atlas-b")
        data.select_atlas("atlas-b")
        data.assign_object(
            "atlas-b",
            "table",
            "textures/table-2048.png",
            2048,
        )

        loaded = TextureAtlasData.from_dict(data.to_dict())

        self.assertEqual(loaded, data)
        self.assertEqual(loaded.selected_atlas_id, "atlas-b")
        self.assertTrue(loaded.remove_atlas("atlas-b"))
        self.assertEqual(loaded.selected_atlas_id, "atlas-a")

    def test_clone_has_no_shared_mutable_records_or_placements(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Original", 2048, atlas_id="atlas-a")
        data.assign_object(
            atlas.atlas_id,
            "chair",
            "textures/chair.png",
            512,
        )

        cloned = data.clone()
        cloned.atlases[0].name = "Changed"
        cloned.atlases[0].placements.clear()

        self.assertEqual(data.atlases[0].name, "Original")
        self.assertEqual(len(data.atlases[0].placements), 1)

    def test_atlas_names_are_trimmed_and_case_insensitively_unique(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("  Furniture  ", 2048)

        self.assertEqual(atlas.name, "Furniture")
        with self.assertRaisesRegex(ValueError, "name already exists"):
            data.create_atlas("furniture", 4096)

    def test_rejects_unsafe_or_unsupported_values(self) -> None:
        data = TextureAtlasData()
        with self.assertRaisesRegex(ValueError, "2048 or 4096"):
            data.create_atlas("Wrong", 1024)
        atlas = data.create_atlas("Safe", 2048, atlas_id="atlas-a")
        with self.assertRaisesRegex(ValueError, "project-relative"):
            data.assign_object(
                atlas.atlas_id,
                "chair",
                "../outside.png",
                512,
            )
        with self.assertRaisesRegex(ValueError, "512, 1024, or 2048"):
            data.assign_object(
                atlas.atlas_id,
                "chair",
                "textures/chair.png",
                256,
            )

    def test_loaded_atlas_ids_must_be_bounded_filename_safe_identifiers(
        self,
    ) -> None:
        valid_payload = {
            "atlases": [
                {
                    "atlas_id": "atlas-a.01_safe",
                    "name": "Safe",
                    "resolution": 2048,
                    "placements": [],
                    "image_path": None,
                }
            ],
            "selected_atlas_id": "atlas-a.01_safe",
        }
        self.assertEqual(
            TextureAtlasData.from_dict(valid_payload).selected_atlas_id,
            "atlas-a.01_safe",
        )

        for malicious_id in (
            "../../outside",
            "folder/atlas",
            r"folder\atlas",
            "C:atlas",
            ".",
            "..",
            "a" * 129,
        ):
            with self.subTest(atlas_id=malicious_id):
                payload = json.loads(json.dumps(valid_payload))
                payload["atlases"][0]["atlas_id"] = malicious_id
                payload["selected_atlas_id"] = malicious_id
                with self.assertRaisesRegex(ValueError, "filename-safe"):
                    TextureAtlasData.from_dict(payload)

    def test_png_and_metadata_output_match_placements(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Colors", 2048, atlas_id="atlas-a")
        data.assign_object("atlas-a", "blue", "textures/blue.png", 512)
        data.assign_object("atlas-a", "green", "textures/green.png", 512)

        colors = {
            "blue": np.array([255, 0, 0, 255], dtype=np.uint8),
            "green": np.array([0, 255, 0, 255], dtype=np.uint8),
        }

        def load_color(placement):
            return np.broadcast_to(
                colors[placement.object_id],
                (placement.size, placement.size, 4),
            ).copy()

        with tempfile.TemporaryDirectory() as directory:
            png_path = Path(directory) / "colors.png"
            metadata_path = Path(directory) / "colors.json"
            write_texture_atlas_png(
                atlas,
                png_path,
                source_loader=load_color,
                project_relative_image_path="atlases/colors.png",
            )
            write_texture_atlas_metadata(atlas, metadata_path)

            image = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(tuple(image[100, 100]), (255, 0, 0, 255))
        self.assertEqual(tuple(image[100, 600]), (0, 255, 0, 255))
        self.assertEqual(metadata["atlas_id"], "atlas-a")
        self.assertEqual(metadata["image_path"], "atlases/colors.png")
        self.assertEqual(len(metadata["placements"]), 2)

    def test_default_png_loader_resolves_relative_to_explicit_asset_root(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Disk", 2048, atlas_id="atlas-a")
        data.assign_object("atlas-a", "red", "textures/red.png", 512)

        with tempfile.TemporaryDirectory() as directory:
            asset_root = Path(directory) / "assets"
            texture_path = asset_root / "textures" / "red.png"
            output_path = Path(directory) / "output" / "disk.png"
            texture_path.parent.mkdir(parents=True)
            source = np.zeros((512, 512, 3), dtype=np.uint8)
            source[:, :] = (0, 0, 255)
            self.assertTrue(cv2.imwrite(str(texture_path), source))

            with self.assertRaisesRegex(ValueError, "asset root"):
                write_texture_atlas_png(atlas, output_path)
            write_texture_atlas_png(
                atlas,
                output_path,
                asset_root=asset_root,
                project_relative_image_path="atlases/disk.png",
            )
            rendered = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)

        self.assertEqual(tuple(rendered[100, 100]), (0, 0, 255, 255))
        self.assertEqual(atlas.image_path, "atlases/disk.png")

    def test_project_file_round_trip_and_legacy_default(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Project Atlas", 4096, atlas_id="atlas-a")
        data.assign_object(
            atlas.atlas_id,
            "sofa",
            "textures/sofa-1024.png",
            1024,
        )

        with tempfile.TemporaryDirectory() as directory:
            project_path = Path(directory) / "project.json"
            save_project(
                project_path,
                current_level_index=2,
                levels=create_default_levels(),
                texture_atlases=data,
            )
            loaded = load_project(project_path)
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload.pop("texture_atlases")
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            legacy_loaded = load_project(project_path)

        self.assertEqual(loaded.texture_atlases, data)
        self.assertEqual(legacy_loaded.texture_atlases, TextureAtlasData())

    def test_new_atlas_field_preserves_prior_positional_api_order(self) -> None:
        levels = create_default_levels()
        camera = InitialFirstPersonCamera(
            level_index=2,
            pose=CameraPose(x=1.0, y=2.0, z=1.7),
        )
        generation = GenerationData()
        surface_generation = SurfaceTextureData()
        presets = create_default_doorway_presets()

        project_data = ProjectData(
            None,
            2,
            levels,
            [],
            presets,
            generation,
            surface_generation,
            camera,
            [],
        )
        self.assertEqual(project_data.initial_first_person_camera, camera)
        self.assertEqual(project_data.stairs, [])
        self.assertEqual(project_data.texture_atlases, TextureAtlasData())

        with tempfile.TemporaryDirectory() as directory:
            project_path = Path(directory) / "positional-project.json"
            save_project(
                project_path,
                2,
                levels,
                [],
                presets,
                generation,
                surface_generation,
                camera,
                [],
            )
            loaded = load_project(project_path)

        self.assertEqual(loaded.initial_first_person_camera, camera)
        self.assertEqual(loaded.stairs, [])
        self.assertEqual(loaded.texture_atlases, TextureAtlasData())


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
