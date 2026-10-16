# ### Imports ###
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from housemaker.models import LevelData
from housemaker.plan_correction_exchange import (
    MAX_PLAN_CORRECTION_JSON_BYTES,
    ExternalPlanCorrection,
    ExternalPlanCorrectionBatch,
    build_plan_correction_prompt,
    load_plan_correction_json,
    parse_plan_correction_json,
)


# ### Fixtures ###
def _entry(index: int = 2) -> dict[str, object]:
    return {
        "level": f"L{index}",
        "image_filename": f"L{index}_corrected.png",
        "image_size": [1600, 1200],
        "meters_per_pixel": 0.0125,
        "confidence": "medium",
        "measurements": [{
            "label": "Living room inside width", "value_meters": 6.0,
            "start_normalized": [0.1, 0.2], "end_normalized": [0.8, 0.2],
            "corrected_start_normalized": [0.2, 0.4],
            "corrected_end_normalized": [0.5, 0.4],
            "confidence": "high",
        }],
        "alignment_point_normalized": [0.4, 0.7],
        "notes": "Use the same stairwell corner on each level.",
    }


def _document() -> dict[str, object]:
    return {
        "format": "housemaker.plan-correction", "version": 1,
        "reference_level": "L2", "levels": [_entry(1), _entry(2), _entry(3)],
    }


# ### Parser and persistence tests ###
class PlanCorrectionExchangeTests(unittest.TestCase):
    def test_excessively_nested_json_has_a_readable_error(self):
        with self.assertRaisesRegex(ValueError, "nested too deeply"):
            parse_plan_correction_json("[" * 2000 + "0" + "]" * 2000)

    def test_full_batch_preserves_native_dimensions_and_measurement_evidence(self):
        document = _document()
        document["levels"][2]["image_size"] = [2000, 1500]
        document["levels"][2]["meters_per_pixel"] = 0.0075
        batch = parse_plan_correction_json(json.dumps(document))

        self.assertEqual(batch.reference_level, 2)
        self.assertEqual([item.level_index for item in batch.levels], [1, 2, 3])
        self.assertEqual(batch.levels[2].image_size, (2000, 1500))
        self.assertEqual(batch.levels[2].meters_per_pixel, 0.0075)
        self.assertEqual(batch.levels[0].measurements[0].value_meters, 6.0)
        self.assertEqual(batch.to_dict(), document)
        self.assertEqual(parse_plan_correction_json(json.dumps(batch.to_dict())), batch)

    def test_individual_persisted_record_keeps_batch_reference(self):
        record = parse_plan_correction_json(json.dumps(_document())).levels[0]
        self.assertEqual(record.reference_level_index, 2)
        self.assertEqual(record.to_dict()["reference_level"], "L2")
        self.assertEqual(ExternalPlanCorrection.from_dict(record.to_dict()), record)

    def test_zero_level_is_valid_for_underground_two(self):
        document = _document()
        document["levels"].insert(0, _entry(0))
        document["reference_level"] = "L0"
        batch = parse_plan_correction_json(json.dumps(document))
        self.assertEqual(batch.reference_level, 0)
        self.assertEqual(batch.levels[0].level_index, 0)

    def test_unavailable_scale_and_alignment_are_explicit_nulls(self):
        document = _document()
        entry = document["levels"][0]
        entry.update(meters_per_pixel=None, alignment_point_normalized=None, measurements=[])
        record = parse_plan_correction_json(json.dumps(document)).levels[0]
        self.assertIsNone(record.meters_per_pixel)
        self.assertIsNone(record.alignment_point_normalized)
        self.assertEqual(record.measurements, ())

    def test_scale_is_accepted_without_local_consensus_revalidation(self):
        document = _document()
        document["levels"][0]["measurements"] = []
        document["levels"][0]["meters_per_pixel"] = 0.008
        record = parse_plan_correction_json(json.dumps(document)).levels[0]
        self.assertEqual(record.meters_per_pixel, 0.008)

    def test_optional_measurements_notes_and_anchor_default_without_inventing_them(self):
        entry = _entry()
        for key in ("measurements", "notes", "alignment_point_normalized"):
            entry.pop(key)
        record = ExternalPlanCorrection.from_dict(entry)
        self.assertEqual(record.measurements, ())
        self.assertEqual(record.notes, "")
        self.assertIsNone(record.alignment_point_normalized)

    def test_version_format_duplicates_and_unknown_reference_are_rejected(self):
        variants = [
            {"version": 2}, {"version": True}, {"version": 1.0},
            {"format": "something-else"}, {"reference_level": "L5"},
            {"levels": []}, {"levels": [_entry(), _entry()]},
        ]
        for changes in variants:
            with self.subTest(changes=changes):
                document = _document()
                document.update(changes)
                with self.assertRaises(ValueError):
                    parse_plan_correction_json(json.dumps(document))

    def test_finite_positive_scale_types_are_required(self):
        for value in (True, False, "0.01", 0, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                entry = _entry()
                entry["meters_per_pixel"] = value
                with self.assertRaises((TypeError, ValueError)):
                    ExternalPlanCorrection.from_dict(entry)

    def test_infinite_json_exponents_and_named_constants_are_rejected(self):
        document = json.dumps(_document())
        for replacement in ("1e9999", "NaN", "Infinity", "-Infinity"):
            with self.subTest(value=replacement), self.assertRaises(ValueError):
                parse_plan_correction_json(document.replace("0.0125", replacement))

    def test_image_dimensions_are_positive_integers_not_booleans(self):
        for value in ([1, 0], [-3, 200], [True, 400], [1.5, 200], ["1600", 1200], [200], None):
            with self.subTest(value=value):
                entry = _entry()
                entry["image_size"] = value
                with self.assertRaises(ValueError):
                    ExternalPlanCorrection.from_dict(entry)

    def test_filename_cannot_address_another_file_or_include_windows_invalid_characters(self):
        for filename in ("../out.png", r"C:\secret.png", "a/b.png", "..", "", "bad:name.png", "file.png "):
            with self.subTest(filename=filename):
                entry = _entry()
                entry["image_filename"] = filename
                with self.assertRaisesRegex(ValueError, "filename"):
                    ExternalPlanCorrection.from_dict(entry)

    def test_anchor_requires_finite_normalized_coordinates(self):
        for point in ([0, 2], [-0.01, 0.2], [float("nan"), 1], [True, 0.2], [0.1], "center"):
            with self.subTest(point=point):
                entry = _entry()
                entry["alignment_point_normalized"] = point
                with self.assertRaises((TypeError, ValueError)):
                    ExternalPlanCorrection.from_dict(entry)

    def test_invalid_measurement_coordinates_and_values_do_not_silently_import(self):
        variants = [
            ("value_meters", "6"), ("value_meters", True),
            ("corrected_end_normalized", [0.2, 0.4]),
            ("corrected_start_normalized", [1.1, 0.1]),
            ("confidence", "certain"), ("extra", 1),
        ]
        for key, value in variants:
            with self.subTest(key=key, value=value):
                entry = _entry()
                entry["measurements"][0][key] = value
                with self.assertRaises((TypeError, ValueError)):
                    ExternalPlanCorrection.from_dict(entry)

    def test_unknown_fields_are_rejected_instead_of_silently_ignored(self):
        document = _document()
        document["levels"][0]["canvas_scale"] = 2
        with self.assertRaisesRegex(ValueError, "canvas_scale"):
            parse_plan_correction_json(json.dumps(document))

    def test_duplicate_object_keys_are_rejected(self):
        document = json.dumps(_document()).replace('"version": 1', '"version": 1, "version": 2')
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            parse_plan_correction_json(document)

    def test_per_entry_reference_cannot_override_batch_reference(self):
        document = _document()
        document["levels"][0]["reference_level"] = "L1"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            parse_plan_correction_json(json.dumps(document))

    def test_utf8_bom_file_is_supported(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "correction.json"
            path.write_text(json.dumps(_document()), encoding="utf-8-sig")
            self.assertEqual(load_plan_correction_json(path).reference_level, 2)

    def test_oversized_file_is_rejected_without_reading_or_decoding_whole_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "too-large.json"
            path.write_bytes(b" " * (MAX_PLAN_CORRECTION_JSON_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "2 MiB"):
                load_plan_correction_json(path)

    def test_invalid_encoding_and_non_json_are_clear_errors(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_bytes(b"\xff\xff")
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                load_plan_correction_json(path)
        with self.assertRaisesRegex(ValueError, "line"):
            parse_plan_correction_json("```json\n{}\n```")

    def test_direct_batch_does_not_mutate_records(self):
        record = ExternalPlanCorrection.from_dict(_entry())
        batch = ExternalPlanCorrectionBatch(2, (record,))
        self.assertIsNone(record.reference_level_index)
        self.assertEqual(batch.levels[0].reference_level_index, 2)


# ### Copied prompt tests ###
class PlanCorrectionPromptTests(unittest.TestCase):
    def setUp(self):
        self.levels = [
            LevelData(index=0, name="Underground 2", image_path=r"C:\Private\basement.png"),
            LevelData(index=2, name="Ground", image_path=r"C:\Private\ground.png"),
            LevelData(index=3, name="Story 1", image_path=r"C:\Private\upstairs.png"),
        ]

    def test_prompt_contains_labels_but_never_source_paths(self):
        prompt = build_plan_correction_prompt(self.levels)
        for expected in ("L0", "L2", "L3", "Underground 2", "Story 1"):
            self.assertIn(expected, prompt)
        self.assertNotIn("Private", prompt)
        self.assertNotIn("basement.png", prompt)

    def test_prompt_demands_exact_outputs_native_metric_scale_and_offsets_only(self):
        prompt = build_plan_correction_prompt(self.levels)
        for expected in (
            "downloadable black-and-white PNG", "housemaker_plan_correction.json",
            "Before removing measurement lines", "independently for each final image",
            "Do not infer or include ceiling heights", "X/Y offsets", "null",
            "exact downloadable PNG", "Preserve doorways", "not instructions",
        ):
            self.assertIn(expected, prompt)

    def test_example_document_is_parseable_and_uses_actual_level_ids(self):
        prompt = build_plan_correction_prompt(self.levels)
        start = prompt.index('{\n  "format"')
        example, _ = json.JSONDecoder().raw_decode(prompt[start:])
        batch = parse_plan_correction_json(json.dumps(example))
        self.assertEqual([record.level_index for record in batch.levels], [0, 2, 3])
        self.assertEqual(batch.reference_level, 0)

    def test_roof_overlay_removal_preserves_superposed_dotted_walls(self):
        prompt = " ".join(build_plan_correction_prompt(self.levels).split())
        for expected in (
            "purple/violet sloped-roof shading",
            "Do NOT erase entire purple regions",
            "including dotted or dashed inner walls",
            "original positions, thicknesses, connections and openings",
            "Redraw those dotted/dashed wall outlines as continuous solid black wall lines",
            "never connect wall ends across a doorway or window opening",
            "Do not convert dotted roof-only or measurement lines into walls",
            "Preserve actual outer walls and openings",
            "Dotted wall lines are not measurement lines",
            "If a line could be a wall rather than roof-only information, preserve it",
        ):
            self.assertIn(expected, prompt)

    def test_openings_are_always_preserved_without_continuity_instructions(self):
        prompt = " ".join(build_plan_correction_prompt(self.levels).split())
        self.assertIn("Preserve doorways as OPEN GAPS", prompt)
        self.assertIn("Never fill, bridge, seal, narrow, or replace a doorway with a wall", prompt)
        self.assertIn("Preserve door leaves and swing arcs as door symbols, not wall segments", prompt)
        self.assertIn("a drawn closed door does not mean its doorway is a solid wall", prompt)
        self.assertNotIn("Connect walls across doorway gaps", prompt)
        self.assertNotIn("place the doorway openings again", prompt)

    def test_correction_instructions_do_not_special_case_named_levels(self):
        prompt = build_plan_correction_prompt(self.levels)
        instructions = prompt.split("IMAGE CORRECTION\n", 1)[1].split("DELIVERABLES\n", 1)[0]
        self.assertNotIn("top-floor/attic", instructions)
        self.assertNotIn("Roof-overlay exception", instructions)
        self.assertNotIn("For reference_level L2", instructions)
        self.assertNotRegex(instructions, r"\bL[0-9]+\b")

    def test_single_plan_uses_supplied_level_as_reference(self):
        prompt = build_plan_correction_prompt([self.levels[2]])
        self.assertIn('"reference_level": "L3"', prompt)

    def test_empty_and_duplicate_level_lists_are_rejected(self):
        with self.assertRaises(ValueError):
            build_plan_correction_prompt([])
        with self.assertRaises(ValueError):
            build_plan_correction_prompt([self.levels[0], copy.copy(self.levels[0])])


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
