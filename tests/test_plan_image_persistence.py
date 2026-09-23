# ### Imports ###
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from housemaker.models import LevelData
from housemaker.project_io import load_project, save_project


# ### Tests ###
class PlanImagePersistenceTests(unittest.TestCase):
    def test_original_plan_image_path_round_trips_with_corrected_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project_path = root / "project.json"
            source_path = root / "source-photo.jpg"
            corrected_path = root / "corrected-plan.png"
            level = LevelData(
                index=2,
                name="Ground",
                image_path=str(corrected_path),
                original_image_path=str(source_path),
                image_size_pixels=(1200.0, 800.0),
            )

            save_project(project_path, 2, [level])
            restored = load_project(project_path).levels[2]

            self.assertEqual(restored.image_path, str(corrected_path.resolve()))
            self.assertEqual(
                restored.original_image_path,
                str(source_path.resolve()),
            )

    def test_legacy_plan_without_original_path_keeps_field_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project_path = root / "project.json"
            image_path = root / "plan.png"
            save_project(
                project_path,
                2,
                [
                    LevelData(
                        index=2,
                        name="Ground",
                        image_path=str(image_path),
                    )
                ],
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["levels"][0].pop("original_image_path")
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_project(project_path).levels[2]

            self.assertIsNone(restored.original_image_path)


if __name__ == "__main__":
    unittest.main()
