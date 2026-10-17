# ### Imports ###
from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.reference_view_presets import (
    DEFAULT_REFERENCE_VIEW_PRESETS,
    REFERENCE_VIEW_PRESETS_SETTINGS_KEY,
    ReferenceViewPreset,
    read_reference_view_presets,
    save_reference_view_presets,
)


# ### Preset model tests ###
class ReferenceViewPresetTests(unittest.TestCase):
    def test_defaults_have_six_unique_views_and_complete_generic_prompts(self) -> None:
        self.assertEqual(len(DEFAULT_REFERENCE_VIEW_PRESETS), 6)
        self.assertEqual(
            len({preset.preset_id for preset in DEFAULT_REFERENCE_VIEW_PRESETS}), 6
        )
        for preset in DEFAULT_REFERENCE_VIEW_PRESETS:
            with self.subTest(view=preset.name):
                self.assertIn("same object", preset.prompt)
                self.assertIn("proportions", preset.prompt)
                self.assertIn("materials, colors", preset.prompt)
                self.assertIn("white background", preset.prompt)
                self.assertIn("90–95%", preset.prompt)
                self.assertIn("without cropping", preset.prompt)
                self.assertNotIn("cupboard", preset.prompt)
        self.assertIn(
            "straight-on front view", DEFAULT_REFERENCE_VIEW_PRESETS[1].prompt
        )
        self.assertIn("upper surface", DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt)
        self.assertEqual(
            [preset.preset_id for preset in DEFAULT_REFERENCE_VIEW_PRESETS],
            [
                "elevated-three-quarter",
                "front",
                "three-quarter",
                "side",
                "top",
                "low-three-quarter",
            ],
        )
        self.assertIn("side/profile view", DEFAULT_REFERENCE_VIEW_PRESETS[3].prompt)
        self.assertIn("vertically downward", DEFAULT_REFERENCE_VIEW_PRESETS[4].prompt)
        self.assertIn("underside", DEFAULT_REFERENCE_VIEW_PRESETS[5].prompt)

    def test_presets_and_defaults_cannot_be_mutated(self) -> None:
        self.assertIsInstance(DEFAULT_REFERENCE_VIEW_PRESETS, tuple)
        with self.assertRaises(FrozenInstanceError):
            DEFAULT_REFERENCE_VIEW_PRESETS[0].name = "Changed"

    def test_values_are_trimmed_and_boundary_lengths_allowed(self) -> None:
        preset = ReferenceViewPreset(" custom ", " Custom view ", " A custom view. ")
        self.assertEqual(
            preset, ReferenceViewPreset("custom", "Custom view", "A custom view.")
        )
        self.assertEqual(
            len(ReferenceViewPreset("id", "n" * 80, "p" * 4000).prompt), 4000
        )

    def test_invalid_values_are_rejected(self) -> None:
        for values in (
            ("", "View", "Prompt"),
            ("id", "   ", "Prompt"),
            ("id", "View", "\n"),
            (None, "View", "Prompt"),
            ("id", 42, "Prompt"),
            ("id", "View", []),
            ("id", "n" * 81, "Prompt"),
            ("id", "View", "p" * 4001),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ReferenceViewPreset(*values)

    def test_write_prompt_name_is_reserved_case_insensitively(self) -> None:
        for name in ("(Write prompt)", "(write prompt)", " (WRITE PROMPT) "):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "reserved"),
            ):
                ReferenceViewPreset("custom-id", name, "A reusable prompt.")


# ### Persistence tests ###
class ReferenceViewPresetPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.store = ApplicationSettingsStore(
            Path(self._temporary_directory.name) / "settings.json"
        )

    def test_missing_settings_return_defaults_without_creating_file(self) -> None:
        self.assertEqual(
            read_reference_view_presets(None), DEFAULT_REFERENCE_VIEW_PRESETS
        )
        self.assertEqual(
            read_reference_view_presets(self.store), DEFAULT_REFERENCE_VIEW_PRESETS
        )
        self.assertFalse(self.store.path.exists())

    def test_custom_views_round_trip_without_losing_unrelated_settings(self) -> None:
        self.store.set("unrelated", 123)
        presets = (
            ReferenceViewPreset(
                "stable-custom-id", "Renamable view", "A top-right view."
            ),
            ReferenceViewPreset("other", "Another view", "A low view."),
        )
        self.assertTrue(save_reference_view_presets(self.store, presets))
        self.assertEqual(read_reference_view_presets(self.store), presets)
        self.assertEqual(self.store.get("unrelated"), 123)
        self.assertEqual(
            self.store.get(REFERENCE_VIEW_PRESETS_SETTINGS_KEY)[0],
            {
                "preset_id": "stable-custom-id",
                "name": "Renamable view",
                "prompt": "A top-right view.",
            },
        )
        renamed = (
            ReferenceViewPreset(presets[0].preset_id, "New name", presets[0].prompt),
        )
        self.assertTrue(save_reference_view_presets(self.store, renamed))
        self.assertEqual(
            read_reference_view_presets(self.store)[0].preset_id, "stable-custom-id"
        )

    def test_intentionally_empty_list_survives_reload(self) -> None:
        self.assertTrue(save_reference_view_presets(self.store, []))
        self.assertEqual(read_reference_view_presets(self.store), ())
        self.assertEqual(self.store.get(REFERENCE_VIEW_PRESETS_SETTINGS_KEY), [])

    def test_saved_views_move_elevated_first_without_losing_custom_edits(self) -> None:
        custom = ReferenceViewPreset("custom", "My custom view", "My custom prompt.")
        front = ReferenceViewPreset("front", "Renamed front", "Edited front prompt.")
        elevated = ReferenceViewPreset(
            "elevated-three-quarter", "Renamed elevated", "Edited elevated prompt."
        )
        saved = (custom, front, elevated)
        self.assertTrue(save_reference_view_presets(self.store, saved))

        self.assertEqual(
            read_reference_view_presets(self.store), (elevated, custom, front)
        )
        self.assertEqual(
            [
                item["preset_id"]
                for item in self.store.get(REFERENCE_VIEW_PRESETS_SETTINGS_KEY)
            ],
            ["custom", "front", "elevated-three-quarter"],
        )

    def test_removed_elevated_view_is_not_restored_or_other_views_reordered(self) -> None:
        front = ReferenceViewPreset("front", "Front", "A front view.")
        custom = ReferenceViewPreset("custom", "Custom", "A custom view.")
        self.assertTrue(save_reference_view_presets(self.store, (custom, front)))
        self.assertEqual(read_reference_view_presets(self.store), (custom, front))

    def test_malformed_payloads_fall_back_to_defaults(self) -> None:
        for payload in (
            None,
            {},
            "[]",
            5,
            ["invalid"],
            [{"preset_id": "missing-fields"}],
            [{"preset_id": "id", "name": "", "prompt": "Prompt"}],
            [{"preset_id": "id", "name": "Name", "prompt": "p" * 4001}],
        ):
            with self.subTest(payload=payload):
                self.store.set(REFERENCE_VIEW_PRESETS_SETTINGS_KEY, payload)
                self.assertEqual(
                    read_reference_view_presets(self.store),
                    DEFAULT_REFERENCE_VIEW_PRESETS,
                )

    def test_saved_duplicate_ids_are_rejected_and_read_duplicates_keep_first(
        self,
    ) -> None:
        first = ReferenceViewPreset("same", "First", "First prompt.")
        duplicate = ReferenceViewPreset("same", "Duplicate", "Second prompt.")
        other = ReferenceViewPreset("other", "Other", "Other prompt.")
        self.assertFalse(
            save_reference_view_presets(self.store, [first, duplicate, other])
        )
        self.assertTrue(save_reference_view_presets(self.store, [first, other]))
        payload = self.store.get(REFERENCE_VIEW_PRESETS_SETTINGS_KEY)
        payload.append({"preset_id": "same", "name": "Again", "prompt": "Again."})
        self.store.set(REFERENCE_VIEW_PRESETS_SETTINGS_KEY, payload)
        self.assertEqual(read_reference_view_presets(self.store), (first, other))

    def test_case_insensitive_duplicate_names_are_rejected(self) -> None:
        first = ReferenceViewPreset("one", "View", "First prompt.")
        second = ReferenceViewPreset("two", "VIEW", "Second prompt.")
        self.assertFalse(save_reference_view_presets(self.store, [first, second]))
        self.assertFalse(self.store.path.exists())
        self.store.set(
            REFERENCE_VIEW_PRESETS_SETTINGS_KEY,
            [
                {"preset_id": item.preset_id, "name": item.name, "prompt": item.prompt}
                for item in (first, second)
            ],
        )
        self.assertEqual(
            read_reference_view_presets(self.store), DEFAULT_REFERENCE_VIEW_PRESETS
        )

    def test_reserved_custom_prompt_name_cannot_be_imported_or_saved(self) -> None:
        payload = [
            {"preset_id": "custom-id", "name": "(Write prompt)", "prompt": "A view."}
        ]
        self.assertFalse(save_reference_view_presets(self.store, payload))
        self.assertFalse(self.store.path.exists())
        self.store.set(REFERENCE_VIEW_PRESETS_SETTINGS_KEY, payload)
        self.assertEqual(
            read_reference_view_presets(self.store), DEFAULT_REFERENCE_VIEW_PRESETS
        )

    def test_invalid_save_preserves_previous_settings(self) -> None:
        self.assertTrue(
            save_reference_view_presets(self.store, DEFAULT_REFERENCE_VIEW_PRESETS)
        )
        original_contents = self.store.path.read_text(encoding="utf-8")
        for invalid in ([{"name": "not a preset"}], [None], None):
            with self.subTest(invalid=invalid):
                self.assertFalse(save_reference_view_presets(self.store, invalid))
                self.assertEqual(
                    self.store.path.read_text(encoding="utf-8"), original_contents
                )
        self.assertFalse(
            save_reference_view_presets(None, DEFAULT_REFERENCE_VIEW_PRESETS)
        )

    def test_failed_write_is_reported(self) -> None:
        with patch.object(self.store, "set", return_value=False):
            self.assertFalse(
                save_reference_view_presets(self.store, DEFAULT_REFERENCE_VIEW_PRESETS)
            )


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
