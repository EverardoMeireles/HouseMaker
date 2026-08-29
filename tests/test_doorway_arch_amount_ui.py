# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import json
import math
import tempfile
import unittest
from pathlib import Path

from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.main import BlueprintWorkspace
from housemaker.models import (
    DEFAULT_DOORWAY_ARCH_AMOUNT,
    DOORWAY_SHAPE_ARCH,
    DOORWAY_SHAPE_RECTANGULAR,
    DoorwayData,
    VertexData,
    create_default_levels,
    normalize_doorway_arch_amount,
)
from housemaker.project_io import load_project, save_project


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_doorway(
    *,
    shape: str = DOORWAY_SHAPE_ARCH,
    arch_amount: float = DEFAULT_DOORWAY_ARCH_AMOUNT,
) -> DoorwayData:
    return DoorwayData(
        center_x=50.0,
        center_y=50.0,
        width_meters=0.9,
        height_meters=2.1,
        depth_meters=0.2,
        rotation_degrees=0.0,
        shape=shape,
        arch_amount=arch_amount,
    )


def _build_canvas(doorway: DoorwayData) -> BlueprintCanvas:
    canvas = BlueprintCanvas()
    canvas.set_level_data(
        vertex_data=VertexData(),
        rooms=[],
        image_path=None,
        doorways=[doorway],
    )
    return canvas


# ### Model and persistence tests ###
class DoorwayArchAmountStateTests(unittest.TestCase):
    def test_arch_amount_is_normalized_and_strictly_bounded(self) -> None:
        for raw_amount, expected_amount in (
            (0, 0.0),
            (0.375, 0.375),
            ("1", 1.0),
        ):
            with self.subTest(raw_amount=raw_amount):
                doorway = _build_doorway(
                    arch_amount=raw_amount,  # type: ignore[arg-type]
                )
                self.assertEqual(doorway.arch_amount, expected_amount)
                self.assertEqual(
                    normalize_doorway_arch_amount(raw_amount),
                    expected_amount,
                )

        for invalid_amount in (
            None,
            True,
            -0.01,
            1.01,
            math.nan,
            math.inf,
            "curved",
        ):
            with self.subTest(invalid_amount=invalid_amount):
                with self.assertRaisesRegex(ValueError, "Doorway arch amount"):
                    _build_doorway(
                        arch_amount=invalid_amount,  # type: ignore[arg-type]
                    )

    def test_arch_amount_round_trips_and_malformed_values_fall_back(self) -> None:
        levels = create_default_levels()
        levels[2].doorways = [_build_doorway(arch_amount=0.375)]

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "arch-amount.json"
            save_project(
                project_path,
                current_level_index=2,
                levels=levels,
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            raw_doorway = payload["levels"][2]["doorways"][0]
            self.assertEqual(raw_doorway["arch_amount"], 0.375)
            self.assertEqual(
                load_project(project_path).levels[2].doorways[0].arch_amount,
                0.375,
            )

            raw_doorway.pop("arch_amount")
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                load_project(project_path).levels[2].doorways[0].arch_amount,
                DEFAULT_DOORWAY_ARCH_AMOUNT,
            )

            for malformed_amount in (None, True, -1.0, 2.0, "curved"):
                with self.subTest(malformed_amount=malformed_amount):
                    raw_doorway["arch_amount"] = malformed_amount
                    project_path.write_text(
                        json.dumps(payload),
                        encoding="utf-8",
                    )
                    self.assertEqual(
                        load_project(project_path)
                        .levels[2]
                        .doorways[0]
                        .arch_amount,
                        DEFAULT_DOORWAY_ARCH_AMOUNT,
                    )

    def test_canvas_arch_amount_edit_is_undoable_and_survives_copies(self) -> None:
        doorway = _build_doorway(arch_amount=0.8)
        canvas = _build_canvas(doorway)
        preview_changes: list[None] = []
        canvas.doorway_dimension_preview_changed.connect(
            lambda: preview_changes.append(None)
        )

        try:
            self.assertFalse(canvas.set_selected_doorway_arch_amount(0.4))
            canvas._set_selected_doorway_index(0)
            self.assertTrue(canvas.set_selected_doorway_arch_amount(0.4))
            self.assertEqual(canvas.doorways[0].arch_amount, 0.4)
            self.assertEqual(len(canvas.undo_stack), 1)
            self.assertEqual(preview_changes, [None])
            self.assertFalse(canvas.set_selected_doorway_arch_amount(0.4))

            copied_doorway = canvas._copy_doorway_with(
                canvas.doorways[0],
                width_meters=1.2,
                height_meters=2.4,
                depth_meters=0.5,
                rotation_degrees=90.0,
            )
            self.assertEqual(copied_doorway.arch_amount, 0.4)

            canvas.undo_last_step()
            self.assertEqual(canvas.doorways[0].arch_amount, 0.8)
        finally:
            canvas.close()
            _qt_application.processEvents()

    def test_canvas_rejects_arch_amount_edits_for_rectangular_doorways(self) -> None:
        canvas = _build_canvas(
            _build_doorway(
                shape=DOORWAY_SHAPE_RECTANGULAR,
                arch_amount=0.25,
            )
        )
        try:
            canvas._set_selected_doorway_index(0)
            self.assertFalse(canvas.set_selected_doorway_arch_amount(0.5))
            self.assertEqual(canvas.doorways[0].arch_amount, 0.25)
            self.assertEqual(canvas.undo_stack, [])
        finally:
            canvas.close()
            _qt_application.processEvents()


# ### Canvas workspace UI tests ###
class DoorwayArchAmountUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        settings = ApplicationSettingsStore(
            Path(self._temporary_directory.name) / "settings.json"
        )
        self.workspace = BlueprintWorkspace(application_settings=settings)
        self.workspace.resize(1500, 900)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_arch_amount_control_tracks_selection_and_previews_one_edit(self) -> None:
        rectangular = _build_doorway(
            shape=DOORWAY_SHAPE_RECTANGULAR,
            arch_amount=0.2,
        )
        arch = _build_doorway(arch_amount=0.375)
        self.workspace.current_level.doorways[:] = [rectangular, arch]
        self.workspace._reset_viewer_doorway_snapshots()
        preview_changes: list[None] = []
        self.workspace.canvas.doorway_dimension_preview_changed.connect(
            lambda: preview_changes.append(None)
        )

        self.workspace.canvas._set_selected_doorway_index(0)
        self.assertFalse(
            self.workspace.selected_doorway_arch_amount_spinbox.isEnabled()
        )
        self.assertEqual(
            self.workspace.selected_doorway_arch_amount_spinbox.value(),
            20.0,
        )

        self.workspace.canvas._set_selected_doorway_index(1)
        self.assertTrue(
            self.workspace.selected_doorway_arch_amount_spinbox.isEnabled()
        )
        self.assertEqual(
            self.workspace.selected_doorway_arch_amount_spinbox.value(),
            37.5,
        )
        self.assertFalse(
            self.workspace.selected_doorway_arch_amount_spinbox.keyboardTracking()
        )

        undo_count = len(self.workspace.canvas.undo_stack)
        self.workspace.selected_doorway_arch_amount_spinbox.setValue(62.5)
        _qt_application.processEvents()

        self.assertEqual(
            self.workspace.current_level.doorways[1].arch_amount,
            0.625,
        )
        self.assertEqual(len(self.workspace.canvas.undo_stack), undo_count + 1)
        self.assertEqual(preview_changes, [None])
        self.assertTrue(self.workspace._doorway_mesh_update_timer.isActive())
        self.assertTrue(
            self.workspace.canvas._get_doorway_label_text(
                self.workspace.current_level.doorways[1]
            ).startswith("Arch 62.5%\n")
        )

        self.workspace.canvas._set_selected_doorway_index(None)
        self.assertFalse(
            self.workspace.selected_doorway_arch_amount_spinbox.isEnabled()
        )


if __name__ == "__main__":
    unittest.main()
