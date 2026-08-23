# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import trimesh
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_state import GeneratedObjectRecord, GenerationData
from housemaker.generation_workspace import (
    GENERATION_BACKEND_MESHY,
    GenerationWorkspace,
)
from housemaker.main import BlueprintWorkspace
from housemaker.unused_face_removal import ALL_CAMERA_IDS


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _generated_box_glb() -> bytes:
    mesh = trimesh.creation.box(extents=(1.0, 0.5, 0.75))
    return trimesh.Scene(mesh).export(file_type="glb")


def _record(
    object_id: str,
    asset_path: str,
    *,
    source_asset_path: str | None = None,
    postprocessed_asset_path: str | None = None,
) -> GeneratedObjectRecord:
    pipeline: dict[str, object] = {}
    if source_asset_path is not None or postprocessed_asset_path is not None:
        pipeline = {
            "mode": "unused_face_removal",
            "source_asset_path": source_asset_path,
            "postprocessed_asset_path": postprocessed_asset_path,
        }
    return GeneratedObjectRecord(
        object_id=object_id,
        frame_index=0,
        object_name=f"Object {object_id}",
        pipeline=pipeline,
        provider=GENERATION_BACKEND_MESHY,
        provider_task_id=f"task-{object_id}",
        asset_path=asset_path,
    )


def _write_asset(asset_directory: Path, relative_path: str, data: bytes) -> Path:
    destination = asset_directory / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return destination


def _primary_screen():
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        raise RuntimeError("A QScreen is required for external viewer tests.")
    return screen


# ### Asset cleanup tests ###
class GeneratedObjectAssetDeletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = (
            Path(self._temporary_directory.name) / "generation_assets"
        )
        self.asset_directory.mkdir()
        self.workspace = GenerationWorkspace(
            asset_directory=self.asset_directory
        )
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_delete_staged_object_removes_all_revisions_and_emits_data(
        self,
    ) -> None:
        final_path = _write_asset(
            self.asset_directory,
            "staged.glb",
            _generated_box_glb(),
        )
        source_path = _write_asset(
            self.asset_directory,
            "staged.geometry.glb",
            b"source geometry",
        )
        postprocessed_path = _write_asset(
            self.asset_directory,
            "staged.postprocessed.glb",
            b"postprocessed geometry",
        )
        self.workspace.set_data(
            GenerationData(
                generated_objects=[
                    _record(
                        "staged",
                        final_path.name,
                        source_asset_path=source_path.name,
                        postprocessed_asset_path=postprocessed_path.name,
                    )
                ]
            )
        )
        data_changed_spy = QSignalSpy(self.workspace.data_changed)

        deleted = self.workspace.delete_generated_object("staged")

        self.assertTrue(deleted)
        self.assertFalse(final_path.exists())
        self.assertFalse(source_path.exists())
        self.assertFalse(postprocessed_path.exists())
        self.assertEqual(self.workspace.get_data().generated_objects, [])
        self.assertEqual(data_changed_spy.count(), 1)
        emitted_data = data_changed_spy.at(0)[0]
        self.assertIsInstance(emitted_data, GenerationData)
        self.assertEqual(emitted_data.generated_objects, [])

    def test_delete_preserves_assets_referenced_by_a_live_object(self) -> None:
        shared_final_path = _write_asset(
            self.asset_directory,
            "shared-final.glb",
            _generated_box_glb(),
        )
        shared_revision_path = _write_asset(
            self.asset_directory,
            "shared-revision.glb",
            b"shared revision",
        )
        deleted_only_path = _write_asset(
            self.asset_directory,
            "deleted-only.glb",
            b"deleted revision",
        )
        live_final_path = _write_asset(
            self.asset_directory,
            "live.glb",
            _generated_box_glb(),
        )
        deleted_record = _record(
            "deleted",
            shared_final_path.name,
            source_asset_path=shared_revision_path.name,
            postprocessed_asset_path=deleted_only_path.name,
        )
        live_record = _record(
            "live",
            live_final_path.name,
            source_asset_path=shared_final_path.name,
            postprocessed_asset_path=shared_revision_path.name,
        )
        self.workspace.set_data(
            GenerationData(generated_objects=[deleted_record, live_record])
        )

        self.assertTrue(self.workspace.delete_generated_object("deleted"))

        self.assertTrue(shared_final_path.exists())
        self.assertTrue(shared_revision_path.exists())
        self.assertTrue(live_final_path.exists())
        self.assertFalse(deleted_only_path.exists())
        self.assertEqual(
            self.workspace.get_data().generated_objects,
            [live_record],
        )

    def test_delete_skips_unsafe_and_missing_asset_paths(self) -> None:
        outside_path = (
            Path(self._temporary_directory.name) / "outside.glb"
        )
        outside_path.write_bytes(_generated_box_glb())
        non_glb_path = _write_asset(
            self.asset_directory,
            "revision-notes.txt",
            b"not a generated model",
        )
        record = _record(
            "unsafe",
            "../outside.glb",
            source_asset_path="missing.geometry.glb",
            postprocessed_asset_path=non_glb_path.name,
        )
        self.workspace.set_data(
            GenerationData(generated_objects=[record])
        )

        deleted = self.workspace.delete_generated_object("unsafe")

        self.assertTrue(deleted)
        self.assertTrue(outside_path.exists())
        self.assertTrue(non_glb_path.exists())
        self.assertEqual(self.workspace.get_data().generated_objects, [])
        self.assertEqual(self.workspace.generated_objects_list.count(), 0)

    def test_asset_unlink_error_does_not_restore_deleted_record(self) -> None:
        asset_path = _write_asset(
            self.asset_directory,
            "locked.glb",
            _generated_box_glb(),
        )
        self.workspace.set_data(
            GenerationData(
                generated_objects=[_record("locked", asset_path.name)]
            )
        )
        data_changed_spy = QSignalSpy(self.workspace.data_changed)

        with patch.object(
            Path,
            "unlink",
            side_effect=PermissionError("asset is locked"),
        ):
            deleted = self.workspace.delete_generated_object("locked")

        self.assertTrue(deleted)
        self.assertTrue(asset_path.exists())
        self.assertEqual(self.workspace.get_data().generated_objects, [])
        self.assertEqual(data_changed_spy.count(), 1)


# ### External viewer deletion tests ###
class ExternalGeneratedObjectDeletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        temporary_path = Path(self._temporary_directory.name)
        self.asset_directory = temporary_path / "generation_assets"
        self.asset_directory.mkdir()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                temporary_path / "settings.json"
            )
        )
        self.workspace.generation._asset_directory = self.asset_directory
        self.workspace.resize(1400, 850)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_delete_clears_the_detached_object_panel_in_place(self) -> None:
        asset_path = _write_asset(
            self.asset_directory,
            "external.glb",
            _generated_box_glb(),
        )
        generation = self.workspace.generation
        self.workspace.settings_widget.unused_face_removal_checkbox.setChecked(
            True
        )
        generation.set_data(
            GenerationData(
                generated_objects=[_record("external", asset_path.name)]
            )
        )
        screen_id = "screen:external-deletion-test"
        combo = self.workspace.settings_widget.fullscreen_3d_viewer_screen_combo
        combo.addItem("External deletion display", screen_id)

        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=_primary_screen(),
        ):
            combo.setCurrentIndex(combo.findData(screen_id))
            self.workspace.workspace_tabs.setCurrentWidget(generation)
            _qt_application.processEvents()

        panel = generation.object_3d_panel
        self.assertIs(self.workspace._external_viewer_host.viewer, panel)
        self.assertIsNotNone(panel.viewer.model)
        self.assertEqual(
            tuple(panel.viewer.unused_face_camera_indicator_items),
            ALL_CAMERA_IDS,
        )

        self.assertTrue(generation.delete_generated_object("external"))
        _qt_application.processEvents()

        self.assertTrue(self.workspace._external_viewer_host.is_active)
        self.assertIs(self.workspace._external_viewer_host.viewer, panel)
        self.assertIsNone(panel.viewer.model)
        self.assertEqual(panel.viewer.unused_face_camera_indicator_items, {})
        self.assertEqual(panel.viewer.unused_face_camera_indicator_labels, {})
        self.assertEqual(generation.texture_view.entries, ())
        self.assertFalse(generation.delete_generated_object_button.isEnabled())
        self.assertFalse(asset_path.exists())


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
