# ### Imports ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.project_io import save_project


# ### Fixture helpers ###
def _save_minimal_project(path: Path) -> None:
    save_project(
        path=path,
        current_level_index=GROUND_LEVEL_INDEX,
        levels=create_default_levels(),
    )


# ### Atomic project save tests ###
class AtomicProjectSaveTests(unittest.TestCase):
    def test_temporary_write_failure_preserves_existing_project_and_cleans_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.json"
            previous_contents = '{"existing": "project"}\n'
            project_path.write_text(previous_contents, encoding="utf-8")

            with (
                patch.object(os, "fsync", side_effect=OSError("write failed")),
                self.assertRaises(ValueError),
            ):
                _save_minimal_project(project_path)

            self.assertEqual(
                project_path.read_text(encoding="utf-8"),
                previous_contents,
            )
            self.assertEqual(
                list(project_path.parent.glob(f".{project_path.name}.*.tmp")),
                [],
            )

    def test_replace_failure_preserves_existing_project_and_cleans_temporary_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.json"
            previous_contents = '{"existing": "project"}\n'
            project_path.write_text(previous_contents, encoding="utf-8")

            with (
                patch.object(os, "replace", side_effect=OSError("replace failed")),
                self.assertRaises(ValueError),
            ):
                _save_minimal_project(project_path)

            self.assertEqual(
                project_path.read_text(encoding="utf-8"),
                previous_contents,
            )
            self.assertEqual(
                list(project_path.parent.glob(f".{project_path.name}.*.tmp")),
                [],
            )


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
