# ### Imports ###
from __future__ import annotations

from dataclasses import fields
import json
import tempfile
import unittest
from pathlib import Path

from housemaker.models import LevelData, create_default_levels
from housemaker.project_io import load_project, save_project


# ### Constants ###
RETIRED_IMAGE_TRANSFORM_FIELDS = frozenset(
    ("image_scale", "image_offset_x", "image_offset_y")
)


# ### Tests ###
class LevelImageTransformRemovalTests(unittest.TestCase):
    def test_level_model_has_no_per_image_transform_fields(self) -> None:
        field_names = {model_field.name for model_field in fields(LevelData)}

        self.assertTrue(RETIRED_IMAGE_TRANSFORM_FIELDS.isdisjoint(field_names))

    def test_projects_omit_and_ignore_legacy_image_transform_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.json"
            save_project(
                project_path,
                current_level_index=2,
                levels=create_default_levels(),
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            for raw_level in payload["levels"]:
                self.assertTrue(
                    RETIRED_IMAGE_TRANSFORM_FIELDS.isdisjoint(raw_level)
                )
                raw_level.update(
                    {
                        "image_scale": 2.5,
                        "image_offset_x": 145.0,
                        "image_offset_y": -93.0,
                    }
                )
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded_project = load_project(project_path)

        for level in loaded_project.levels:
            self.assertTrue(
                all(
                    not hasattr(level, field_name)
                    for field_name in RETIRED_IMAGE_TRANSFORM_FIELDS
                )
            )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
