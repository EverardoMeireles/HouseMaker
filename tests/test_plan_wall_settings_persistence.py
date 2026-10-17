# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.main import BlueprintWorkspace
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.plan_wall_options import (
    PlanWallDetectionError,
    PlanWallDetectionOptions,
)
from housemaker.project_io import load_project, save_project

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _custom_options() -> PlanWallDetectionOptions:
    return PlanWallDetectionOptions(
        minimum_wall_separation_pixels=9,
        maximum_wall_separation_pixels=63,
        parallel_angle_tolerance_degrees=12,
        minimum_parallel_overlap_ratio=0.42,
        maximum_gap_bridge_pixels=59,
        endpoint_snap_distance_pixels=17,
        maximum_vertex_distance_pixels=11,
        minimum_wall_length_pixels=35,
        confidence_threshold=0.67,
        endpoint_alignment_tolerance_pixels=27,
        endpoint_alignment_search_distance_pixels=2300,
    )


# ### Project serialization tests ###
class PlanWallSettingsSerializationTests(unittest.TestCase):
    def test_every_reconstruction_option_round_trips_with_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "project.json"
            save_project(
                path,
                GROUND_LEVEL_INDEX,
                create_default_levels(),
                plan_wall_generation=_custom_options(),
            )

            restored = load_project(path)

            self.assertEqual(restored.plan_wall_generation, _custom_options())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["plan_wall_generation"],
                _custom_options().to_dict(),
            )

    def test_legacy_and_malformed_settings_use_valid_defaults(self) -> None:
        malformed_values = (
            None,
            [],
            {"confidence_threshold": "invalid"},
            {"confidence_threshold": True},
            {"endpoint_snap_distance_pixels": float("nan")},
            {"maximum_gap_bridge_pixels": float("inf")},
            {"minimum_wall_length_pixels": -1},
            {"endpoint_alignment_tolerance_pixels": -1},
            {"endpoint_alignment_search_distance_pixels": -1},
            {"endpoint_alignment_tolerance_pixels": float("inf")},
            {"endpoint_alignment_search_distance_pixels": float("nan")},
            {
                "minimum_wall_separation_pixels": 90,
                "maximum_wall_separation_pixels": 20,
            },
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "project.json"
            save_project(path, GROUND_LEVEL_INDEX, create_default_levels())
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.pop("plan_wall_generation")
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                load_project(path).plan_wall_generation,
                PlanWallDetectionOptions(),
            )

            for raw_settings in malformed_values:
                with self.subTest(settings=raw_settings):
                    payload["plan_wall_generation"] = raw_settings
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    self.assertEqual(
                        load_project(path).plan_wall_generation,
                        PlanWallDetectionOptions(),
                    )

    def test_partial_future_compatible_settings_keep_other_defaults(self) -> None:
        restored = PlanWallDetectionOptions.from_dict(
            {"endpoint_snap_distance_pixels": 15, "unknown_future_option": 123}
        )

        self.assertEqual(restored.endpoint_snap_distance_pixels, 15)
        self.assertEqual(restored.confidence_threshold, 0.45)
        self.assertEqual(restored.endpoint_alignment_tolerance_pixels, 16)
        self.assertEqual(restored.endpoint_alignment_search_distance_pixels, 1024)

    def test_legacy_positional_options_keep_their_original_meanings(self) -> None:
        restored = PlanWallDetectionOptions(9, 63, 12, 0.42, 59, 17, 11, 35, 0.67)

        self.assertEqual(restored.confidence_threshold, 0.67)
        self.assertEqual(restored.endpoint_alignment_tolerance_pixels, 16)
        self.assertEqual(restored.endpoint_alignment_search_distance_pixels, 1024)

    def test_alignment_ranges_allow_zero_but_reject_invalid_values(self) -> None:
        for name in (
            "endpoint_alignment_tolerance_pixels",
            "endpoint_alignment_search_distance_pixels",
        ):
            with self.subTest(name=name):
                self.assertEqual(getattr(PlanWallDetectionOptions(**{name: 0}), name), 0)
                for invalid in (-1, float("nan"), float("inf")):
                    with (
                        self.subTest(value=invalid),
                        self.assertRaises(PlanWallDetectionError),
                    ):
                        PlanWallDetectionOptions(**{name: invalid})


# ### Main-workspace persistence tests ###
class PlanWallSettingsMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary_directory.name)
        self.store = ApplicationSettingsStore(self.directory / "settings.json")
        self.workspaces: list[BlueprintWorkspace] = []

    def tearDown(self) -> None:
        for workspace in self.workspaces:
            workspace.shutdown()
            workspace.close()
            workspace.deleteLater()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _workspace(self) -> BlueprintWorkspace:
        workspace = BlueprintWorkspace(application_settings=self.store)
        self.workspaces.append(workspace)
        return workspace

    def test_save_and_restart_restore_all_sliders_and_labels(self) -> None:
        workspace = self._workspace()
        expected_values = (9, 63, 12, 42, 59, 17, 11, 35, 67, 27, 2300)
        self.assertEqual(len(workspace._get_plan_wall_generation_sliders()), 11)
        for slider, value in zip(
            workspace._get_plan_wall_generation_sliders(),
            expected_values,
        ):
            slider.setValue(value)
        path = self.directory / "saved.json"
        with (
            patch(
                "housemaker.main.QFileDialog.getSaveFileName",
                return_value=(str(path), ""),
            ),
            patch("housemaker.main.QMessageBox.information"),
        ):
            workspace._handle_save_clicked()
        self.assertEqual(load_project(path).plan_wall_generation, _custom_options())
        workspace.shutdown()

        restored = self._workspace()
        self.assertTrue(restored.restore_last_project())

        self.assertEqual(
            tuple(
                slider.value()
                for slider in restored._get_plan_wall_generation_sliders()
            ),
            expected_values,
        )
        self.assertEqual(restored.wall_endpoint_snap_value_label.text(), "17 px")
        self.assertEqual(
            restored.wall_endpoint_alignment_tolerance_value_label.text(), "27 px"
        )
        self.assertEqual(
            restored.wall_endpoint_alignment_search_distance_value_label.text(),
            "2300 px",
        )
        self.assertEqual(restored.parallel_wall_overlap_value_label.text(), "42%")
        self.assertEqual(
            restored._build_plan_wall_detection_options(),
            _custom_options(),
        )
        self.assertIsNone(restored._plan_wall_preview_session)
        self.assertFalse(restored._plan_wall_preview_refresh_timer.isActive())

    def test_loading_legacy_project_resets_previous_project_controls(self) -> None:
        workspace = self._workspace()
        workspace._restore_plan_wall_detection_options(_custom_options())
        path = self.directory / "legacy.json"
        save_project(path, GROUND_LEVEL_INDEX, create_default_levels())
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("plan_wall_generation")
        path.write_text(json.dumps(payload), encoding="utf-8")

        workspace._load_project_path(path)

        self.assertEqual(
            tuple(
                slider.value()
                for slider in workspace._get_plan_wall_generation_sliders()
            ),
            (6, 48, 8, 55, 36, 8, 5, 20, 45, 16, 1024),
        )
        self.assertEqual(workspace.wall_endpoint_snap_value_label.text(), "8 px")


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
