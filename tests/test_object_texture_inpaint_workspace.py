# ### Environment setup ###
from __future__ import annotations

import inspect
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path

import trimesh
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QHBoxLayout

from housemaker.generation_state import GeneratedObjectRecord, GenerationData
from housemaker.generation_workspace import GenerationWorkspace


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _box_glb() -> bytes:
    return bytes(trimesh.Scene(trimesh.creation.box()).export(file_type="glb"))


def _action_layout(workspace: GenerationWorkspace) -> QHBoxLayout:
    controls_layout = workspace.generate_button.parentWidget().layout()
    assert controls_layout is not None
    for index in range(controls_layout.count()):
        candidate = controls_layout.itemAt(index).layout()
        if isinstance(candidate, QHBoxLayout):
            widgets = [
                candidate.itemAt(item_index).widget()
                for item_index in range(candidate.count())
            ]
            if workspace.generate_button in widgets:
                return candidate
    raise AssertionError("Object Generation action row was not found.")


# ### Retired Object Generation inpaint tests ###
class RetiredObjectTextureInpaintWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = Path(self.temporary_directory.name)
        self.asset_directory.joinpath("chair.glb").write_bytes(_box_glb())
        self.workspace = GenerationWorkspace(
            asset_directory=self.asset_directory,
        )
        self.workspace.resize(2600, 900)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def test_inpaint_controls_signals_and_provider_path_are_removed(self) -> None:
        retired_attributes = (
            "paint_texture_mask_button",
            "texture_inpaint_instructions_edit",
            "undo_texture_stroke_button",
            "clear_texture_mask_button",
            "inpaint_texture_button",
            "texture_inpaint_completed",
            "inpaint_selected_object_texture",
            "set_object_texture_inpaint_provider",
            "_object_texture_inpaint_provider",
        )
        for attribute in retired_attributes:
            self.assertFalse(hasattr(self.workspace, attribute), attribute)
        self.assertNotIn(
            "object_texture_inpaint_provider",
            inspect.signature(GenerationWorkspace).parameters,
        )

    def test_legacy_strokes_load_without_rendering_or_reactivating_input(
        self,
    ) -> None:
        legacy_pipeline = {
            "texture_inpaint_strokes": [
                {
                    "mode": "paint",
                    "radius_normalized": 0.04,
                    "points": [{"u": 0.25, "v": 0.75}],
                }
            ],
            "texture_inpaint_history": [{"task_id": "legacy-task"}],
        }
        record = GeneratedObjectRecord(
            object_id="chair",
            frame_index=0,
            object_name="Chair",
            pipeline=legacy_pipeline,
            provider="meshy",
            provider_task_id="legacy-object",
            asset_path="chair.glb",
        )

        self.workspace.set_data(GenerationData(generated_objects=[record]))
        _qt_application.processEvents()

        restored = self.workspace.get_data().generated_objects[0]
        self.assertEqual(restored.pipeline, legacy_pipeline)
        self.assertIsNone(self.workspace.result_view._texture_edit_mask)
        self.assertFalse(self.workspace.texture_view.edit_mask_enabled)
        self.assertFalse(
            hasattr(self.workspace.result_view.view, "texture_inpaint_enabled")
        )

    def test_action_order_spacer_and_far_right_operation_controls(self) -> None:
        layout = _action_layout(self.workspace)
        widgets = [
            layout.itemAt(index).widget()
            for index in range(layout.count())
        ]
        action_widgets = (
            self.workspace.generate_button,
            self.workspace.generate_geometry_button,
            self.workspace.generate_texture_button,
            self.workspace.purge_faces_button,
            self.workspace.place_object_button,
            self.workspace.undo_object_change_button,
            self.workspace.cancel_operation_button,
        )
        action_indices = tuple(widgets.index(widget) for widget in action_widgets)
        self.assertEqual(action_indices, tuple(sorted(action_indices)))

        generate_index = action_indices[0]
        spacer = layout.itemAt(generate_index + 1).spacerItem()
        self.assertIsNotNone(spacer)
        assert spacer is not None
        self.assertGreaterEqual(spacer.sizeHint().width(), 30)

        place_index = action_indices[-3]
        stretch = layout.itemAt(place_index - 1).spacerItem()
        self.assertIsNotNone(stretch)
        assert stretch is not None
        self.assertTrue(
            stretch.expandingDirections() & Qt.Orientation.Horizontal
        )
        self.assertEqual(action_indices[-1], layout.count() - 1)
        self.assertEqual(action_indices[-1] - action_indices[-2], 1)
        self.assertGreater(
            self.workspace.undo_object_change_button.geometry().left(),
            self.workspace.purge_faces_button.geometry().right(),
        )


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
