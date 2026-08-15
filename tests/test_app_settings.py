# ### Imports ###
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from housemaker.app_settings import ApplicationSettingsStore


# ### Application settings tests ###
class ApplicationSettingsStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.settings_path = (
            Path(self._temporary_directory.name) / "settings.json"
        )
        self.store = ApplicationSettingsStore(self.settings_path)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_missing_file_returns_default_and_values_round_trip(self) -> None:
        self.assertEqual(self.store.get("missing", "fallback"), "fallback")

        self.assertTrue(self.store.set("last_project_path", "project.json"))
        self.assertEqual(self.store.get("last_project_path"), "project.json")
        self.assertEqual(
            json.loads(self.settings_path.read_text(encoding="utf-8")),
            {"last_project_path": "project.json"},
        )

    def test_malformed_or_non_object_json_falls_back_and_can_recover(self) -> None:
        for malformed_payload in ("{not-json", "[1, 2, 3]"):
            with self.subTest(payload=malformed_payload):
                self.settings_path.write_text(
                    malformed_payload,
                    encoding="utf-8",
                )
                self.assertEqual(self.store.get("missing", 42), 42)
                self.assertTrue(self.store.set("recovered", True))
                self.assertEqual(
                    json.loads(self.settings_path.read_text(encoding="utf-8")),
                    {"recovered": True},
                )

    def test_unserializable_value_does_not_overwrite_existing_settings(self) -> None:
        self.assertTrue(self.store.set("stable", "value"))
        original_contents = self.settings_path.read_text(encoding="utf-8")

        self.assertFalse(self.store.set("invalid", object()))

        self.assertEqual(
            self.settings_path.read_text(encoding="utf-8"),
            original_contents,
        )

    def test_failed_atomic_replace_preserves_existing_settings(self) -> None:
        self.assertTrue(self.store.set("stable", "value"))
        original_contents = self.settings_path.read_text(encoding="utf-8")

        with patch("housemaker.app_settings.os.replace", side_effect=OSError):
            self.assertFalse(self.store.set("new", "value"))

        self.assertEqual(
            self.settings_path.read_text(encoding="utf-8"),
            original_contents,
        )
        self.assertEqual(list(self.settings_path.parent.glob("*.tmp")), [])

    def test_remove_preserves_unrelated_values(self) -> None:
        self.assertTrue(self.store.set("one", 1))
        self.assertTrue(self.store.set("two", 2))

        self.assertTrue(self.store.remove("one"))

        self.assertIsNone(self.store.get("one"))
        self.assertEqual(self.store.get("two"), 2)


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
