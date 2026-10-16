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

from PIL import Image
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.door_state import (
    DoorLibraryData,
    WindowPlacement,
    create_window_definition_for_frame,
)
from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, SlopedCeilingData, WindowData
from housemaker.surface_geometry import build_fixed_surfaces

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Main-workspace integration tests ###
class PlanCorrectionExchangeMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.temporary_path = Path(self._temporary_directory.name)
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                self.temporary_path / "settings.json"
            )
        )
        self.workspace.resize(1200, 800)
        self.workspace.show()
        _qt_application.processEvents()
        self._thread_patch = patch("housemaker.main._PlanImageCorrectionThread")
        self.api_thread = self._thread_patch.start()
        self.addCleanup(self._thread_patch.stop)

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    # ### Fixtures ###
    def _image(self, name: str, size: tuple[int, int] = (120, 80)) -> Path:
        path = self.temporary_path / name
        Image.new("RGB", size, "white").save(path)
        return path.resolve()

    def _load_image(self, name: str = "original.png") -> Path:
        path = self._image(name)
        self.workspace._set_current_level_image(str(path))
        return path

    def _record(self, plan_level: LevelData, **changes: object) -> dict[str, object]:
        record: dict[str, object] = {
            "level": f"L{plan_level.index}",
            "image_filename": f"L{plan_level.index}_corrected.png",
            "image_size": [120, 80],
            "meters_per_pixel": 0.0125,
            "confidence": "medium",
            "measurements": [],
            "alignment_point_normalized": [0.5, 0.5],
            "notes": "Imported independent scale; no ceiling height changes.",
        }
        record.update(changes)
        return record

    def _manifest(self, *records: dict[str, object]) -> Path:
        path = self.temporary_path / "corrections.json"
        path.write_text(
            json.dumps({
                "format": "housemaker.plan-correction",
                "version": 1,
                "reference_level": records[0]["level"],
                "levels": records,
            }),
            encoding="utf-8",
        )
        return path

    def _wall_window(self, level: LevelData) -> WindowData:
        first = level.vertex_data.add_vertex(10.0, 10.0)
        second = level.vertex_data.add_vertex(110.0, 10.0)
        level.vertex_data.add_edge(first.id, second.id)
        wall = next(
            surface for surface in build_fixed_surfaces((level,))
            if surface.surface_type == "wall"
        )
        window = WindowData(
            window_id="live-window",
            wall_surface_id=wall.surface_id,
            start_ratio=0.35,
            end_ratio=0.65,
            bottom_ratio=0.25,
            top_ratio=0.75,
        )
        level.windows.append(window)
        self.workspace._reset_viewer_doorway_snapshots()
        return window

    @staticmethod
    def _orphan_window(level: LevelData) -> WindowData:
        window = WindowData(
            window_id="orphan-window",
            wall_surface_id=f"level:{level.index}/wall:998:999",
            start_ratio=0.25,
            end_ratio=0.75,
            bottom_ratio=0.25,
            top_ratio=0.75,
        )
        level.windows.append(window)
        return window

    @staticmethod
    def _state(level: LevelData) -> tuple[object, ...]:
        return (
            level.image_path,
            level.original_image_path,
            level.image_size_pixels,
            level.external_plan_correction,
            level.plan_metric_calibration,
            level.plan_meters_per_pixel,
            level.height_meters,
            level.canvas_level_scale,
            level.canvas_offset_x_pixels,
            level.canvas_offset_y_pixels,
            level.scale,
            tuple((vertex.id, vertex.x, vertex.y) for vertex in level.vertex_data.vertices),
            tuple(level.vertex_data.edges),
            tuple(level.windows),
            tuple(level.sloped_ceilings),
        )

    # ### Choice and clipboard routing ###
    def test_choice_dispatches_exactly_one_workflow(self) -> None:
        for action, target in (
            ("api", "_handle_api_image_correction_clicked"),
            ("copy_prompt", "_copy_external_plan_correction_prompt"),
            ("browse_json", "_browse_external_plan_correction_json"),
        ):
            with (
                self.subTest(action=action),
                patch("housemaker.main.PlanCorrectionDialog") as dialog_type,
                patch.object(self.workspace, "_handle_api_image_correction_clicked") as api,
                patch.object(self.workspace, "_copy_external_plan_correction_prompt") as copy,
                patch.object(self.workspace, "_browse_external_plan_correction_json") as browse,
            ):
                dialog_type.return_value.exec.return_value = QDialog.DialogCode.Accepted
                dialog_type.return_value.chosen_action = action
                self.workspace._handle_image_correction_clicked()
                for name, handler in (
                    ("_handle_api_image_correction_clicked", api),
                    ("_copy_external_plan_correction_prompt", copy),
                    ("_browse_external_plan_correction_json", browse),
                ):
                    self.assertEqual(handler.call_count, int(name == target))
                dialog_type.assert_called_once_with(self.workspace)
        self.api_thread.assert_not_called()

    def test_cancelled_choice_does_not_start_any_workflow(self) -> None:
        with (
            patch("housemaker.main.PlanCorrectionDialog") as dialog_type,
            patch.object(self.workspace, "_handle_api_image_correction_clicked") as api,
            patch.object(self.workspace, "_copy_external_plan_correction_prompt") as copy,
            patch.object(self.workspace, "_browse_external_plan_correction_json") as browse,
        ):
            dialog_type.return_value.exec.return_value = QDialog.DialogCode.Rejected
            self.workspace._handle_image_correction_clicked()
        api.assert_not_called()
        copy.assert_not_called()
        browse.assert_not_called()

    def test_copy_prompt_uses_loaded_levels_without_creating_jobs(self) -> None:
        self._load_image()
        self.workspace.settings_widget.make_walls_continuous_checkbox.setChecked(True)
        self.assertTrue(self.workspace.settings_widget.get_settings().make_walls_continuous)
        level = self.workspace.current_level
        before = self._state(level)
        with patch("housemaker.main.QMessageBox.information") as information:
            self.workspace._copy_external_plan_correction_prompt()
        prompt = QApplication.clipboard().text()
        self.assertIn(f'"level": "L{level.index}"', prompt)
        self.assertIn("housemaker_plan_correction.json", prompt)
        self.assertIn("black-and-white PNG", prompt)
        self.assertIn("meters_per_pixel", prompt)
        self.assertIn("Preserve doorways as OPEN GAPS", prompt)
        self.assertIn("Never fill, bridge, seal, narrow, or replace a", prompt)
        self.assertNotIn("Connect walls across doorway gaps", prompt)
        self.assertNotIn("For reference_level L2", prompt)
        self.assertNotIn("top-floor/attic", prompt)
        self.assertEqual(self._state(level), before)
        self.assertEqual(self.workspace.job_manager.jobs(), ())
        self.api_thread.assert_not_called()
        information.assert_called_once()

    def test_cancelled_file_picker_leaves_state_unchanged(self) -> None:
        level = self.workspace.current_level
        before = self._state(level)
        with (
            patch("housemaker.main.QFileDialog.getOpenFileName", return_value=("", "")),
            patch.object(self.workspace, "_import_external_plan_correction_json") as import_json,
        ):
            self.workspace._browse_external_plan_correction_json()
        import_json.assert_not_called()
        self.assertEqual(self._state(level), before)

    def test_browse_import_reports_pending_images_without_api_calls(self) -> None:
        self._load_image()
        level = self.workspace.current_level
        record = self._record(level)
        path = self._manifest(record)
        with (
            patch("housemaker.main.QFileDialog.getOpenFileName", return_value=(str(path), "")),
            patch("housemaker.main.QMessageBox.information") as information,
        ):
            self.workspace._browse_external_plan_correction_json()
        self.assertIsNotNone(level.external_plan_correction)
        self.assertIn(record["image_filename"], information.call_args.args[2])
        self.assertEqual(self.workspace.job_manager.jobs(), ())
        self.api_thread.assert_not_called()

    # ### Deferred and immediate application ###
    def test_json_can_be_imported_before_loading_any_plan_images(self) -> None:
        level = self.workspace.current_level
        record = self._record(level)
        self.assertIsNone(level.image_path)
        self.assertTrue(self.workspace.image_correction_button.isEnabled())
        applied, pending = self.workspace._import_external_plan_correction_json(
            self._manifest(record)
        )
        self.assertEqual(applied, 0)
        self.assertEqual(pending, (f"L{level.index}: {record['image_filename']}",))
        self.assertIsNone(level.image_path)
        corrected = self._image(record["image_filename"])
        self.workspace._set_current_level_image(str(corrected))
        self.assertEqual(level.image_path, str(corrected))
        self.assertEqual(level.meters_per_pixel, 0.0125)
        self.api_thread.assert_not_called()

    def test_json_first_applies_only_when_matching_image_is_loaded(self) -> None:
        original = self._load_image()
        level = self.workspace.current_level
        level.plan_meters_per_pixel = 0.018
        level.height_meters = 2.45
        level.canvas_level_scale = 1.25
        level.canvas_offset_x_pixels = 17.0
        level.canvas_offset_y_pixels = -9.0
        level.scale = 1.5
        record = self._record(level)
        result = self.workspace._import_external_plan_correction_json(self._manifest(record))
        self.assertEqual(result, (0, (f"L{level.index}: {record['image_filename']}",)))
        self.assertEqual(level.image_path, str(original))
        self.assertEqual(level.meters_per_pixel, 0.018)
        self.assertEqual(level.scale, 1.5)

        corrected = self._image(record["image_filename"])
        self.workspace._set_current_level_image(str(corrected))

        self.assertEqual(level.image_path, str(corrected))
        self.assertEqual(level.original_image_path, str(original))
        self.assertEqual(level.meters_per_pixel, 0.0125)
        self.assertEqual(level.scale, 1.0)
        self.assertEqual(level.height_meters, 2.45)
        self.assertEqual(level.canvas_level_scale, 1.25)
        self.assertEqual((level.canvas_offset_x_pixels, level.canvas_offset_y_pixels), (17.0, -9.0))
        self.assertEqual(level.image_size_pixels, (120.0, 80.0))
        self.assertIsNotNone(level.external_plan_correction)
        self.api_thread.assert_not_called()

    def test_images_first_apply_independent_scales_and_offsets_only(self) -> None:
        level = self.workspace.current_level
        other = self.workspace.levels[level.index + 1]
        records = (
            self._record(level, alignment_point_normalized=[0.2, 0.7]),
            self._record(other, meters_per_pixel=0.03, alignment_point_normalized=[0.8, 0.3]),
        )
        self._load_image(records[0]["image_filename"])
        other_path = self._image(records[1]["image_filename"])
        other.image_path = str(other_path)
        other.original_image_path = str(other_path)
        other.image_size_pixels = (120.0, 80.0)
        level.height_meters, other.height_meters = 2.45, 2.7
        level.canvas_level_scale, other.canvas_level_scale = 1.25, 0.8
        level.canvas_offset_x_pixels, level.canvas_offset_y_pixels = 17.0, -9.0
        other.canvas_offset_x_pixels, other.canvas_offset_y_pixels = 33.0, 42.0

        applied, pending = self.workspace._import_external_plan_correction_json(self._manifest(*records))

        self.assertEqual((applied, pending), (2, ()))
        self.assertEqual((level.meters_per_pixel, other.meters_per_pixel), (0.0125, 0.03))
        self.assertEqual((level.height_meters, other.height_meters), (2.45, 2.7))
        self.assertEqual((level.canvas_level_scale, other.canvas_level_scale), (1.25, 0.8))
        self.assertEqual((level.canvas_offset_x_pixels, level.canvas_offset_y_pixels), (17.0, -9.0))
        width, height = self.workspace.canvas.get_plan_alignment_viewport_size()
        fit = min(width / 120.0, height / 80.0)
        self.assertAlmostEqual(other.canvas_offset_x_pixels, 17.0 + (-0.3 * 1.25 - 0.3 * 0.8) * 120.0 * fit)
        self.assertAlmostEqual(other.canvas_offset_y_pixels, -9.0 + (0.2 * 1.25 + 0.2 * 0.8) * 80.0 * fit)
        self.api_thread.assert_not_called()

    # ### Metadata-only recalibration of existing geometry ###
    def test_loaded_corrected_raster_can_be_recalibrated_with_walls_and_windows(self) -> None:
        level = self.workspace.current_level
        record = self._record(level)
        self._load_image(record["image_filename"])
        level.height_meters = 2.45
        window = self._wall_window(level)
        vertices = tuple((vertex.id, vertex.x, vertex.y) for vertex in level.vertex_data.vertices)
        edges = tuple(level.vertex_data.edges)
        image_path = level.image_path
        prior_revision = self.workspace._viewer_preview_revision
        previous_dimensions = self.workspace._capture_canvas_window_dimension_snapshots((level.index,))
        self.assertEqual(len(previous_dimensions), 1)

        applied, pending = self.workspace._import_external_plan_correction_json(self._manifest(record))

        self.assertEqual((applied, pending), (1, ()))
        self.assertEqual(level.image_path, image_path)
        self.assertEqual(tuple((v.id, v.x, v.y) for v in level.vertex_data.vertices), vertices)
        self.assertEqual(tuple(level.vertex_data.edges), edges)
        self.assertEqual(level.height_meters, 2.45)
        self.assertEqual(level.meters_per_pixel, 0.0125)
        first = level_image_to_world_xy(level, vertices[0][1], vertices[0][2])
        second = level_image_to_world_xy(level, vertices[1][1], vertices[1][2])
        self.assertAlmostEqual(abs(second[0] - first[0]), 1.25)
        self.assertEqual([item.window_id for item in level.windows], [window.window_id])
        current_dimensions = self.workspace._capture_canvas_window_dimension_snapshots((level.index,))
        self.assertEqual(len(current_dimensions), 1)
        self.assertAlmostEqual(current_dimensions[0].width_meters, previous_dimensions[0].width_meters)
        self.assertAlmostEqual(current_dimensions[0].height_meters, previous_dimensions[0].height_meters)
        self.assertGreater(self.workspace._viewer_preview_revision, prior_revision)
        self.assertEqual(self.workspace.job_manager.jobs(), ())
        self.api_thread.assert_not_called()

    def test_import_removes_orphan_frame_after_pending_wall_deletion(self) -> None:
        level = self.workspace.current_level
        record = self._record(level)
        self._load_image(record["image_filename"])
        window = self._wall_window(level)
        deleted_ids = tuple(vertex.id for vertex in level.vertex_data.vertices)
        level.vertex_data.delete_vertices(deleted_ids)
        self.workspace._handle_canvas_wall_vertices_deleted(deleted_ids)
        self.assertTrue(self.workspace._pending_deleted_wall_cleanup)
        self.assertEqual(level.windows, [window])

        self.assertEqual(self.workspace._import_external_plan_correction_json(self._manifest(record)), (1, ()))

        self.assertEqual(level.windows, [])
        self.assertFalse(self.workspace._pending_deleted_wall_cleanup)
        self.assertFalse(self.workspace._pending_wall_vertex_mesh_update)

    def test_one_undo_restores_imported_scale_windows_and_all_level_offsets(self) -> None:
        level = self.workspace.current_level
        other = self.workspace.levels[level.index + 1]
        records = (
            self._record(level, alignment_point_normalized=[0.2, 0.7]),
            self._record(other, meters_per_pixel=0.03, alignment_point_normalized=[0.8, 0.3]),
        )
        self._load_image(records[0]["image_filename"])
        other.image_path = str(self._image(records[1]["image_filename"]))
        other.original_image_path = other.image_path
        other.image_size_pixels = (120.0, 80.0)
        level.plan_meters_per_pixel = 0.018
        level.scale = 1.5
        level.height_meters = 2.45
        level.canvas_offset_x_pixels, level.canvas_offset_y_pixels = 17.0, -9.0
        other.canvas_offset_x_pixels, other.canvas_offset_y_pixels = 33.0, 42.0
        self._wall_window(level)
        self.workspace._canvas_undo_stack.clear()
        before = tuple(self._state(item) for item in self.workspace.levels)

        self.assertEqual(self.workspace._import_external_plan_correction_json(self._manifest(*records)), (2, ()))
        self.assertNotEqual(tuple(self._state(item) for item in self.workspace.levels), before)
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
        self.workspace._handle_canvas_undo_requested()

        self.assertEqual(tuple(self._state(item) for item in self.workspace.levels), before)
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_stale_orphan_without_pending_timer_does_not_block_future_image_import(self) -> None:
        self._load_image()
        level = self.workspace.current_level
        self._orphan_window(level)
        self.assertFalse(self.workspace._pending_deleted_wall_cleanup)
        record = self._record(level)

        applied, pending = self.workspace._import_external_plan_correction_json(self._manifest(record))

        self.assertEqual(applied, 0)
        self.assertEqual(pending, (f"L{level.index}: {record['image_filename']}",))
        self.assertEqual(level.windows, [])

    def test_import_reconciles_orphan_even_when_vertex_chain_paused_the_timer(self) -> None:
        level = self.workspace.current_level
        record = self._record(level)
        self._load_image(record["image_filename"])
        vertex = level.vertex_data.add_vertex(10.0, 10.0)
        self.workspace.canvas._set_active_vertex_id(vertex.id)
        self._orphan_window(level)
        self.workspace._handle_canvas_wall_vertices_deleted((998, 999))
        self.assertFalse(self.workspace._wall_vertex_update_timer.isActive())

        self.assertEqual(self.workspace._import_external_plan_correction_json(self._manifest(record)), (1, ()))

        self.assertEqual(level.windows, [])
        self.assertIsNotNone(level.vertex_data.get_vertex(vertex.id))
        self.assertFalse(self.workspace._pending_deleted_wall_cleanup)

    def test_hidden_level_sloped_window_is_not_mistaken_for_orphan(self) -> None:
        level = self.workspace.current_level
        record = self._record(level)
        self._load_image(record["image_filename"])
        window = self._wall_window(level)
        first, second = level.vertex_data.vertices
        third = level.vertex_data.add_vertex(110.0, 70.0)
        fourth = level.vertex_data.add_vertex(10.0, 70.0)
        level.vertex_data.add_edge(second.id, third.id)
        level.vertex_data.add_edge(third.id, fourth.id)
        level.vertex_data.add_edge(fourth.id, first.id)
        slope = SlopedCeilingData(
            slope_id="a" * 32,
            wall_surface_id=window.wall_surface_id,
            depth_meters=0.2,
            wall_drop_meters=0.2,
            transition_radius_meters=0.0,
        )
        level.sloped_ceilings = [slope]
        sloped_window = WindowData(
            window_id=window.window_id,
            wall_surface_id=window.wall_surface_id,
            start_ratio=window.start_ratio,
            end_ratio=window.end_ratio,
            bottom_ratio=window.bottom_ratio,
            top_ratio=window.top_ratio,
            sloped_ceiling_id=slope.slope_id,
        )
        level.windows = [sloped_window]
        level.include_in_export = False
        self.workspace._reset_viewer_doorway_snapshots()

        self.assertEqual(self.workspace._import_external_plan_correction_json(self._manifest(record)), (1, ()))

        self.assertEqual(len(level.windows), 1)
        self.assertEqual(level.windows[0].window_id, sloped_window.window_id)
        self.assertEqual(level.windows[0].sloped_ceiling_id, slope.slope_id)
        self.assertEqual(level.sloped_ceilings, [slope])
        self.assertFalse(level.include_in_export)

    def test_same_filename_replaced_pixels_are_not_treated_as_metadata_only(self) -> None:
        level = self.workspace.current_level
        record = self._record(level)
        image_path = self._load_image(record["image_filename"])
        self._wall_window(level)
        Image.new("RGB", (120, 80), "black").save(image_path)
        before = self._state(level)

        with self.assertRaises(ValueError):
            self.workspace._import_external_plan_correction_json(self._manifest(record))

        self.assertEqual(self._state(level), before)

    # ### Loading images after metadata import ###
    def test_reloading_same_corrected_image_with_metadata_keeps_existing_geometry(self) -> None:
        level = self.workspace.current_level
        record = self._record(level)
        corrected = self._load_image(record["image_filename"])
        self.workspace._import_external_plan_correction_json(self._manifest(record))
        self._wall_window(level)
        level.height_meters = 2.45
        before = self._state(level)
        prior_revision = self.workspace._viewer_preview_revision

        with patch.object(self.workspace.canvas, "load_blueprint") as reload_raster:
            self.workspace._set_current_level_image(str(corrected))

        reload_raster.assert_not_called()
        self.assertEqual(self._state(level), before)
        self.assertGreater(self.workspace._viewer_preview_revision, prior_revision)
        self.api_thread.assert_not_called()

    def test_different_corrected_image_path_cannot_replace_raster_with_geometry(self) -> None:
        level = self.workspace.current_level
        record = self._record(level)
        self._load_image(record["image_filename"])
        self.workspace._import_external_plan_correction_json(self._manifest(record))
        self._wall_window(level)
        replacement_directory = self.temporary_path / "new-correction"
        replacement_directory.mkdir()
        replacement = replacement_directory / record["image_filename"]
        Image.new("RGB", (120, 80), "black").save(replacement)
        before = self._state(level)

        with self.assertRaisesRegex(ValueError, "wall vert"):
            self.workspace._set_current_level_image(str(replacement))

        self.assertEqual(self._state(level), before)
        self.api_thread.assert_not_called()

    def test_loading_pending_corrected_image_removes_orphan_placement_but_keeps_asset(self) -> None:
        original = self._load_image()
        level = self.workspace.current_level
        record = self._record(level)
        self.workspace._import_external_plan_correction_json(self._manifest(record))
        frame = self._orphan_window(level)
        window = create_window_definition_for_frame(
            frame.window_id, 1.0, 1.0,
            name="Reusable window", window_id="reusable-window",
        )
        placement = WindowPlacement(
            placement_id="orphan-window-placement",
            window_id=window.window_id,
            level_index=level.index,
            window_frame_id=frame.window_id,
        )
        self.workspace.doors_workspace.set_data(DoorLibraryData(
            windows=[window], window_placements=[placement],
        ))
        corrected = self._image(record["image_filename"])

        self.workspace._set_current_level_image(str(corrected))

        library = self.workspace.doors_workspace.data()
        self.assertEqual(level.windows, [])
        self.assertEqual(library.window_placements, [])
        self.assertEqual(library.get_window(window.window_id), window)
        self.assertEqual(level.image_path, str(corrected))
        self.assertEqual(level.original_image_path, str(original))
        self.assertEqual(level.meters_per_pixel, 0.0125)
        self.api_thread.assert_not_called()

    # ### Atomic validation and geometry safety ###
    def test_malformed_json_is_rejected_without_mutating_levels(self) -> None:
        self._load_image()
        self._orphan_window(self.workspace.current_level)
        before = tuple(self._state(level) for level in self.workspace.levels)
        path = self.temporary_path / "invalid.json"
        path.write_text('{"levels": [', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.workspace._import_external_plan_correction_json(path)
        self.assertEqual(tuple(self._state(level) for level in self.workspace.levels), before)

    def test_unknown_level_rejects_whole_batch(self) -> None:
        level = self.workspace.current_level
        self._load_image(self._record(level)["image_filename"])
        self._orphan_window(level)
        records = (self._record(level), self._record(level, level="L999", image_filename="L999_corrected.png"))
        before = self._state(level)
        with self.assertRaisesRegex(ValueError, "no level L999"):
            self.workspace._import_external_plan_correction_json(self._manifest(*records))
        self.assertEqual(self._state(level), before)

    def test_one_matching_image_with_wrong_size_rejects_whole_batch(self) -> None:
        level = self.workspace.current_level
        other = self.workspace.levels[level.index + 1]
        records = (self._record(level), self._record(other))
        self._load_image(records[0]["image_filename"])
        self._orphan_window(level)
        other.image_path = str(self._image(records[1]["image_filename"], (60, 40)))
        other.image_size_pixels = (60.0, 40.0)
        before = (self._state(level), self._state(other))
        with self.assertRaisesRegex(ValueError, "120 x 80"):
            self.workspace._import_external_plan_correction_json(self._manifest(*records))
        self.assertEqual((self._state(level), self._state(other)), before)

    def test_existing_geometry_prevents_import_for_a_different_raster(self) -> None:
        self._load_image()
        level = self.workspace.current_level
        level.vertex_data.add_vertex(10.0, 20.0)
        before = self._state(level)
        with self.assertRaisesRegex(ValueError, "1 wall vertex"):
            self.workspace._import_external_plan_correction_json(self._manifest(self._record(level)))
        self.assertEqual(self._state(level), before)

    def test_new_geometry_after_staging_blocks_corrected_image_loading(self) -> None:
        self._load_image()
        level = self.workspace.current_level
        record = self._record(level)
        self.workspace._import_external_plan_correction_json(self._manifest(record))
        level.vertex_data.add_vertex(10.0, 20.0)
        before = self._state(level)
        corrected = self._image(record["image_filename"])
        with self.assertRaisesRegex(ValueError, "1 wall vertex"):
            self.workspace._set_current_level_image(str(corrected))
        self.assertEqual(self._state(level), before)

    def test_wrong_size_manual_image_keeps_pending_metadata_and_original(self) -> None:
        self._load_image()
        level = self.workspace.current_level
        record = self._record(level)
        self.workspace._import_external_plan_correction_json(self._manifest(record))
        before = self._state(level)
        corrected = self._image(record["image_filename"], (240, 160))
        with self.assertRaisesRegex(ValueError, "120 x 80"):
            self.workspace._set_current_level_image(str(corrected))
        self.assertEqual(self._state(level), before)

    def test_loading_an_unrelated_image_clears_pending_metadata(self) -> None:
        self._load_image()
        level = self.workspace.current_level
        self.workspace._import_external_plan_correction_json(self._manifest(self._record(level)))
        self._load_image("different-plan.png")
        self.assertIsNone(level.external_plan_correction)
        self.assertIsNone(level.plan_meters_per_pixel)

    def test_load_button_can_cancel_discarding_a_pending_correction(self) -> None:
        self._load_image()
        level = self.workspace.current_level
        self.workspace._import_external_plan_correction_json(self._manifest(self._record(level)))
        before = self._state(level)
        unrelated = self._image("different-plan.png")
        with (
            patch.object(self.workspace, "_get_image_file_path", return_value=str(unrelated)),
            patch("housemaker.main.QMessageBox.question", return_value=QMessageBox.StandardButton.No),
        ):
            self.workspace._handle_load_image_clicked()
        self.assertEqual(self._state(level), before)

    def test_browse_import_surfaces_parser_errors_without_jobs(self) -> None:
        path = self.temporary_path / "invalid.json"
        path.write_text("not json", encoding="utf-8")
        with (
            patch("housemaker.main.QFileDialog.getOpenFileName", return_value=(str(path), "")),
            patch("housemaker.main.QMessageBox.critical") as critical,
        ):
            self.workspace._browse_external_plan_correction_json()
        critical.assert_called_once()
        self.assertEqual(critical.call_args.args[1], "Plan correction import failed")
        self.assertEqual(self.workspace.job_manager.jobs(), ())
        self.api_thread.assert_not_called()
