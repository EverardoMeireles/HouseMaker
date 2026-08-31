# ### Environment setup ###
from __future__ import annotations

import copy
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QWidget
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.generation_state import GeneratedObjectRecord, GenerationData
from housemaker.generation_workspace import (
    FACE_EDIT_ATLAS_PLACEHOLDERS_PIPELINE_KEY,
    FACE_EDIT_TEXTURE_STALE_PIPELINE_KEY,
    LOCALLY_AUTHORED_UVS_PIPELINE_KEY,
    OBJECT_OPERATION_DELETE_FACES,
    TEXTURE_VARIANTS_PIPELINE_KEY,
    VISIBILITY_UV_UNWRAP_PIPELINE_KEY,
    GenerationWorkspace,
    _get_object_operation_undo_stack,
)
from housemaker.object_face_edit import load_object_face_geometry


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])


# ### Fixture helpers ###
def _textured_box_glb(
    texture_size: int,
    color: tuple[int, int, int, int],
) -> bytes:
    mesh = trimesh.creation.box(extents=(1.0, 0.8, 0.6))
    vertices = np.asarray(mesh.vertices, dtype=float)
    uvs = (vertices[:, :2] - np.min(vertices[:, :2], axis=0)) / np.ptp(
        vertices[:, :2],
        axis=0,
    )
    mesh.visual = TextureVisuals(
        uv=uvs,
        material=PBRMaterial(
            name="cabinet-material",
            baseColorTexture=Image.new("RGBA", (texture_size, texture_size), color),
        ),
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


# ### Tests ###
class ObjectFaceEditWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = Path(self.temporary_directory.name)
        self.variant_paths: dict[str, dict[str, str]] = {}
        self.original_glb_paths: set[str] = set()
        self.original_png_payloads: dict[str, bytes] = {}
        for resolution, color in (
            (512, (80, 120, 180, 255)),
            (1024, (110, 140, 190, 255)),
            (2048, (140, 160, 200, 255)),
        ):
            glb_name = f"cabinet.{resolution}.glb"
            png_name = f"cabinet.{resolution}.png"
            glb_bytes = _textured_box_glb(16, color)
            (self.asset_directory / glb_name).write_bytes(glb_bytes)
            Image.new("RGBA", (16, 16), color).save(
                self.asset_directory / png_name
            )
            self.variant_paths[str(resolution)] = {
                "glb_asset_path": glb_name,
                "texture_asset_path": png_name,
            }
            self.original_glb_paths.add(glb_name)
            self.original_png_payloads[png_name] = (
                self.asset_directory / png_name
            ).read_bytes()
        self.original_record = GeneratedObjectRecord(
            object_id="cabinet",
            frame_index=0,
            object_name="Cabinet",
            pipeline={
                TEXTURE_VARIANTS_PIPELINE_KEY: copy.deepcopy(self.variant_paths),
                "selected_texture_resolution": 512,
                "postprocessed_asset_path": "cabinet.2048.glb",
                VISIBILITY_UV_UNWRAP_PIPELINE_KEY: {
                    "version": 1,
                    "face_count": 12,
                },
                "texture_regeneration_uv_fingerprint_version": "test-v1",
                "texture_regeneration_submitted_uv_fingerprint": "before",
                "texture_regeneration_final_uv_fingerprint": "before",
                "texture_regeneration_uv_face_count": 12,
            },
            provider="meshy",
            provider_task_id="task-cabinet",
            asset_path="cabinet.512.glb",
        )
        self.workspace = GenerationWorkspace(asset_directory=self.asset_directory)
        self.workspace.resize(960, 700)
        self.workspace.show()
        self.workspace.set_data(
            GenerationData(generated_objects=[self.original_record])
        )
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _wait_for_jobs(self, timeout_milliseconds: int = 15_000) -> None:
        deadline = time.monotonic() + timeout_milliseconds / 1000.0
        while self.workspace._object_job_runtimes:
            if time.monotonic() >= deadline:
                self.fail("The face-edit worker did not finish in time.")
            _qt_application.processEvents()
            time.sleep(0.01)

    def _wait_for_event(
        self,
        event: threading.Event,
        timeout_milliseconds: int = 2_000,
    ) -> None:
        deadline = time.monotonic() + timeout_milliseconds / 1000.0
        while not event.is_set():
            if time.monotonic() >= deadline:
                self.fail("The background face-edit fixture did not respond.")
            QTest.qWait(5)
            _qt_application.processEvents()

    def test_delete_preserves_texture_variants_and_undo_restores(self) -> None:
        self.assertEqual(self.workspace.result_view.face_edit_face_count, 12)
        self.workspace.result_view.set_selected_face_indices((0, 1))
        changed_spy = QSignalSpy(self.workspace.generated_object_changed)
        QTest.mousePress(
            self.workspace.result_view.view,
            Qt.MouseButton.MiddleButton,
            pos=QPoint(100, 100),
        )
        self.assertTrue(
            self.workspace.result_view.view.is_middle_navigation_active
        )

        started_at = time.monotonic()
        self.assertTrue(self.workspace.delete_selected_object_faces())
        self.assertLess(time.monotonic() - started_at, 0.5)
        self.assertTrue(self.workspace._object_job_runtimes)
        self.assertFalse(
            self.workspace.result_view.view.is_middle_navigation_active
        )
        self.assertIsNot(
            QWidget.mouseGrabber(),
            self.workspace.result_view.view,
        )
        QTest.mouseRelease(
            self.workspace.result_view.view,
            Qt.MouseButton.MiddleButton,
            pos=QPoint(100, 100),
        )
        self._wait_for_jobs()

        edited_record = self.workspace._data.generated_objects[0]
        edited_variants = edited_record.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY]
        self.assertNotIn(FACE_EDIT_TEXTURE_STALE_PIPELINE_KEY, edited_record.pipeline)
        self.assertNotIn(
            FACE_EDIT_ATLAS_PLACEHOLDERS_PIPELINE_KEY,
            edited_record.pipeline,
        )
        self.assertTrue(edited_record.pipeline[LOCALLY_AUTHORED_UVS_PIPELINE_KEY])
        self.assertNotIn(
            VISIBILITY_UV_UNWRAP_PIPELINE_KEY,
            edited_record.pipeline,
        )
        for stale_key in (
            "texture_regeneration_uv_fingerprint_version",
            "texture_regeneration_submitted_uv_fingerprint",
            "texture_regeneration_final_uv_fingerprint",
            "texture_regeneration_uv_face_count",
        ):
            self.assertNotIn(stale_key, edited_record.pipeline)
        self.assertTrue(edited_record.pipeline["face_edit_texture_preserved"])
        self.assertEqual(
            edited_record.pipeline["selected_texture_resolution"],
            512,
        )

        edited_glb_paths: set[str] = set()
        for resolution, original_variant in self.variant_paths.items():
            edited_variant = edited_variants[resolution]
            self.assertEqual(
                edited_variant["texture_asset_path"],
                original_variant["texture_asset_path"],
            )
            edited_glb_path = edited_variant["glb_asset_path"]
            edited_glb_paths.add(edited_glb_path)
            self.assertNotEqual(
                edited_glb_path,
                original_variant["glb_asset_path"],
            )
            self.assertEqual(
                load_object_face_geometry(
                    (self.asset_directory / edited_glb_path).read_bytes()
                ).face_count,
                10,
            )
        self.assertEqual(edited_record.asset_path, edited_variants["512"]["glb_asset_path"])
        self.assertEqual(
            edited_record.pipeline["postprocessed_asset_path"],
            edited_variants["2048"]["glb_asset_path"],
        )
        for png_name, original_payload in self.original_png_payloads.items():
            self.assertEqual(
                (self.asset_directory / png_name).read_bytes(),
                original_payload,
            )
        active_variant = self.workspace.get_active_texture_variant("cabinet")
        self.assertIsNotNone(active_variant)
        assert active_variant is not None
        self.assertEqual(active_variant.texture_asset_relative_path, "cabinet.512.png")
        atlas_variant = self.workspace.get_atlas_texture_image_variant(
            "cabinet",
            512,
        )
        self.assertIsNotNone(atlas_variant)
        assert atlas_variant is not None
        self.assertEqual(atlas_variant.texture_asset_relative_path, "cabinet.512.png")
        self.assertEqual(changed_spy.count(), 1)
        self.assertEqual(
            self.workspace.result_view.get_selected_face_indices(),
            (),
        )
        self.assertEqual(
            self.workspace.texture_view.selected_uv_face_indices,
            (),
        )
        self.assertTrue(self.workspace.result_view._face_editing_enabled)
        undo_stack = _get_object_operation_undo_stack(edited_record)
        self.assertEqual(undo_stack[-1]["operation"], OBJECT_OPERATION_DELETE_FACES)
        self.assertTrue(
            self.workspace.select_object_texture_resolution("cabinet", 1024)
        )
        self.assertEqual(self.workspace.result_view.face_edit_face_count, 10)

        self.assertTrue(self.workspace.undo_selected_object_change())
        restored = self.workspace._data.generated_objects[0]
        self.assertEqual(restored.asset_path, "cabinet.512.glb")
        self.assertEqual(
            restored.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY],
            self.variant_paths,
        )
        self.assertIn(
            VISIBILITY_UV_UNWRAP_PIPELINE_KEY,
            restored.pipeline,
        )
        self.assertEqual(
            restored.pipeline["texture_regeneration_uv_face_count"],
            12,
        )
        self.assertEqual(self.workspace.result_view.face_edit_face_count, 12)
        for edited_glb_path in edited_glb_paths:
            self.assertFalse((self.asset_directory / edited_glb_path).exists())
        for png_name, original_payload in self.original_png_payloads.items():
            self.assertEqual(
                (self.asset_directory / png_name).read_bytes(),
                original_payload,
            )

    def test_symmetric_edit_preserves_existing_layout_metadata(self) -> None:
        symmetry_metadata = {
            "version": 2,
            "orientation": "vertical",
            "kept_side": "left",
            "plane_coordinate": 0.0,
            "packing_mode": "symmetric_quarter",
            "texture_content_quadrant": "top_left",
            "selection_mode": "fewest_triangles_random_tie",
            "triangle_count_by_side": {"left": 12, "right": 12},
            "tie_broken_randomly": True,
        }
        legacy_pipeline = copy.deepcopy(self.original_record.pipeline)
        legacy_pipeline["symmetric_division"] = copy.deepcopy(symmetry_metadata)
        legacy_record = GeneratedObjectRecord(
            object_id=self.original_record.object_id,
            frame_index=self.original_record.frame_index,
            object_name=self.original_record.object_name,
            pipeline=legacy_pipeline,
            provider=self.original_record.provider,
            provider_task_id=self.original_record.provider_task_id,
            asset_path=self.original_record.asset_path,
        )
        self.workspace.set_data(GenerationData(generated_objects=[legacy_record]))
        self.workspace.result_view.set_selected_face_indices((0, 1))

        self.assertTrue(self.workspace.delete_selected_object_faces())
        self._wait_for_jobs()

        edited_record = self.workspace._data.generated_objects[0]
        self.assertEqual(
            edited_record.pipeline["symmetric_division"],
            symmetry_metadata,
        )
        edited_variants = edited_record.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY]
        self.assertEqual(
            {
                resolution: variant["texture_asset_path"]
                for resolution, variant in edited_variants.items()
            },
            {
                resolution: variant["texture_asset_path"]
                for resolution, variant in self.variant_paths.items()
            },
        )
        for variant in edited_variants.values():
            self.assertEqual(
                load_object_face_geometry(
                    (
                        self.asset_directory / variant["glb_asset_path"]
                    ).read_bytes()
                ).face_count,
                10,
            )

    def test_mismatched_saved_revision_keeps_the_existing_object(self) -> None:
        mismatched_mesh = trimesh.creation.box(extents=(1.2, 0.8, 0.6))
        (self.asset_directory / "cabinet.1024.glb").write_bytes(
            bytes(trimesh.Scene(mismatched_mesh).export(file_type="glb"))
        )
        self.workspace.result_view.set_selected_face_indices((0, 1))

        self.assertTrue(self.workspace.delete_selected_object_faces())
        self._wait_for_jobs()

        self.assertEqual(
            self.workspace._data.generated_objects,
            [self.original_record],
        )
        self.assertIn(
            "do not share the displayed face index layout",
            self.workspace.status_label.text(),
        )
        self.assertEqual(
            tuple(self.asset_directory.glob("*.face-edit-*.glb")),
            (),
        )
        self.assertEqual(
            self.workspace.result_view.get_selected_face_indices(),
            (0, 1),
        )
        self.assertEqual(
            self.workspace.texture_view.selected_uv_face_indices,
            (0, 1),
        )
        self.assertTrue(self.workspace.result_view._face_editing_enabled)
        self.assertFalse(
            self.workspace.result_view.view.is_face_selection_gesture_active
        )
        self.assertFalse(
            self.workspace.result_view.view.is_middle_navigation_active
        )

    def test_cancel_does_not_wait_for_blocked_local_edit(self) -> None:
        started = threading.Event()
        release = threading.Event()
        completed = threading.Event()

        def block_face_edit(*_args: object, **_kwargs: object) -> object:
            started.set()
            release.wait(timeout=5.0)
            completed.set()
            return object()

        self.workspace.result_view.set_selected_face_indices((0, 1))
        with patch(
            "housemaker.generation_workspace._prepare_object_face_deletion",
            side_effect=block_face_edit,
        ):
            self.assertTrue(self.workspace.delete_selected_object_faces())
            self._wait_for_event(started)
            cancelled_at = time.monotonic()
            self.assertTrue(self.workspace.cancel_current_operation())
            self._wait_for_jobs(timeout_milliseconds=1_000)
            self.assertLess(time.monotonic() - cancelled_at, 0.5)
            release.set()
            self._wait_for_event(completed)

        self.assertEqual(
            self.workspace._data.generated_objects,
            [self.original_record],
        )


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
