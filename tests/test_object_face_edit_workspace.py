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
    OBJECT_OPERATION_CREATE_FACE,
    OBJECT_OPERATION_DELETE_FACES,
    TEXTURE_VARIANTS_PIPELINE_KEY,
    VISIBILITY_UV_UNWRAP_PIPELINE_KEY,
    GenerationWorkspace,
    _get_object_operation_undo_stack,
    _map_face_edit_vertex_indices,
)
from housemaker.object_face_edit import ObjectFaceGeometry, load_object_face_geometry

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


def _textured_box_with_triangle_hole_glb(
    texture_size: int,
    color: tuple[int, int, int, int],
) -> bytes:
    """Return the textured box fixture with one repairable open triangle."""

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
    keep_faces = np.ones(len(mesh.faces), dtype=bool)
    keep_faces[0] = False
    mesh.update_faces(keep_faces)
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _face_expanded_textured_box_with_triangle_hole_glb(
    texture_size: int,
    color: tuple[int, int, int, int],
) -> bytes:
    """Return a generated-style mesh with UV-seam duplicates per face."""

    source = trimesh.creation.box(extents=(1.0, 0.8, 0.6))
    keep_faces = np.ones(len(source.faces), dtype=bool)
    keep_faces[0] = False
    retained_faces = np.asarray(source.faces[keep_faces], dtype=np.int64)
    expanded_vertices = np.asarray(source.vertices, dtype=float)[
        retained_faces
    ].reshape((-1, 3))
    expanded_faces = np.arange(
        len(expanded_vertices),
        dtype=np.int64,
    ).reshape((-1, 3))
    expanded_uvs = np.tile(
        np.asarray(((0.05, 0.05), (0.95, 0.05), (0.5, 0.95)), dtype=float),
        (len(expanded_faces), 1),
    )
    mesh = trimesh.Trimesh(
        vertices=expanded_vertices,
        faces=expanded_faces,
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=expanded_uvs,
        material=PBRMaterial(
            name="cabinet-material",
            baseColorTexture=Image.new("RGBA", (texture_size, texture_size), color),
        ),
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _triangle_hole_vertex_indices(glb_bytes: bytes) -> tuple[int, int, int]:
    """Return the canonical vertices around the fixture's triangular hole."""

    geometry = load_object_face_geometry(glb_bytes)
    edge_counts: dict[tuple[int, int], int] = {}
    for face in geometry.faces:
        face_indices = tuple(int(index) for index in face)
        for start, end in (
            (face_indices[0], face_indices[1]),
            (face_indices[1], face_indices[2]),
            (face_indices[2], face_indices[0]),
        ):
            edge = tuple(sorted((start, end)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary_vertices = tuple(
        sorted(
            {
                vertex_index
                for edge, count in edge_counts.items()
                if count == 1
                for vertex_index in edge
            }
        )
    )
    if len(boundary_vertices) != 3:
        raise AssertionError("The face-creation fixture has no triangular hole.")
    return boundary_vertices


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

    def test_selected_vertices_map_across_reindexed_texture_variants(
        self,
    ) -> None:
        """Use ordered face corners instead of assuming shared vertex IDs."""

        reference_vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=float,
        )
        reference_faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        permutation = np.asarray((2, 0, 3, 1), dtype=np.int64)
        inverse_permutation = np.argsort(permutation)

        def geometry(vertices: np.ndarray, faces: np.ndarray) -> ObjectFaceGeometry:
            return ObjectFaceGeometry(
                vertices=vertices,
                faces=faces,
                uv_triangles=np.empty((0, 3, 2), dtype=float),
                uv_face_indices=np.empty((0,), dtype=np.int64),
            )

        mapped = _map_face_edit_vertex_indices(
            geometry(reference_vertices, reference_faces),
            geometry(
                reference_vertices[permutation],
                inverse_permutation[reference_faces],
            ),
            (3, 0, 2),
        )

        self.assertEqual(
            mapped,
            tuple(int(inverse_permutation[index]) for index in (3, 0, 2)),
        )

    def _install_triangle_hole_variants(
        self,
        *,
        face_expanded: bool = False,
        symmetry_metadata: dict[str, object] | None = None,
    ) -> tuple[GeneratedObjectRecord, dict[str, dict[str, str]]]:
        """Display equivalent textured variants with one missing triangle."""

        hole_variants: dict[str, dict[str, str]] = {}
        build_glb = (
            _face_expanded_textured_box_with_triangle_hole_glb
            if face_expanded
            else _textured_box_with_triangle_hole_glb
        )
        for resolution, color in (
            (512, (80, 120, 180, 255)),
            (1024, (110, 140, 190, 255)),
            (2048, (140, 160, 200, 255)),
        ):
            glb_name = f"cabinet-hole.{resolution}.glb"
            (self.asset_directory / glb_name).write_bytes(
                build_glb(16, color)
            )
            hole_variants[str(resolution)] = {
                "glb_asset_path": glb_name,
                "texture_asset_path": f"cabinet.{resolution}.png",
            }

        pipeline = copy.deepcopy(self.original_record.pipeline)
        pipeline[TEXTURE_VARIANTS_PIPELINE_KEY] = copy.deepcopy(hole_variants)
        pipeline["postprocessed_asset_path"] = "cabinet-hole.2048.glb"
        pipeline[VISIBILITY_UV_UNWRAP_PIPELINE_KEY]["face_count"] = 11
        pipeline["texture_regeneration_uv_face_count"] = 11
        if symmetry_metadata is not None:
            pipeline["symmetric_division"] = copy.deepcopy(symmetry_metadata)
        hole_record = GeneratedObjectRecord(
            object_id=self.original_record.object_id,
            frame_index=self.original_record.frame_index,
            object_name=self.original_record.object_name,
            pipeline=pipeline,
            provider=self.original_record.provider,
            provider_task_id=self.original_record.provider_task_id,
            asset_path="cabinet-hole.512.glb",
        )
        self.workspace.set_data(GenerationData(generated_objects=[hole_record]))
        _qt_application.processEvents()
        return hole_record, hole_variants

    def test_f_fills_textured_hole_in_every_variant_and_ctrl_z_restores(
        self,
    ) -> None:
        hole_record, hole_variants = self._install_triangle_hole_variants()
        source_glb = (self.asset_directory / hole_record.asset_path).read_bytes()
        selected_vertices = _triangle_hole_vertex_indices(source_glb)
        self.assertEqual(load_object_face_geometry(source_glb).face_count, 11)
        self.workspace.result_view.set_selected_vertex_indices(
            selected_vertices
        )
        creation_requested_spy = QSignalSpy(
            self.workspace.result_view.object_face_creation_requested
        )
        changed_spy = QSignalSpy(self.workspace.generated_object_changed)

        QTest.keyClick(self.workspace.result_view.view, Qt.Key.Key_F)
        self.assertEqual(creation_requested_spy.count(), 1)
        self._wait_for_jobs()

        edited_record = self.workspace._data.generated_objects[0]
        edited_variants = edited_record.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY]
        self.assertTrue(edited_record.pipeline[LOCALLY_AUTHORED_UVS_PIPELINE_KEY])
        self.assertTrue(edited_record.pipeline["face_edit_texture_preserved"])
        self.assertEqual(edited_record.pipeline["face_edit_original_face_count"], 11)
        self.assertEqual(edited_record.pipeline["face_edit_result_face_count"], 12)
        self.assertEqual(
            edited_record.pipeline["face_edit_added_triangle_count"],
            1,
        )
        self.assertEqual(
            edited_record.pipeline["face_edit_created_polygon_count"],
            1,
        )
        self.assertNotIn(
            VISIBILITY_UV_UNWRAP_PIPELINE_KEY,
            edited_record.pipeline,
        )
        self.assertEqual(
            self.workspace.result_view.get_selected_vertex_indices(),
            (),
        )

        edited_glb_paths: set[str] = set()
        for resolution, original_variant in hole_variants.items():
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
                12,
            )
        self.assertEqual(
            edited_record.asset_path,
            edited_variants["512"]["glb_asset_path"],
        )
        self.assertEqual(
            edited_record.pipeline["postprocessed_asset_path"],
            edited_variants["2048"]["glb_asset_path"],
        )
        for png_name, original_payload in self.original_png_payloads.items():
            self.assertEqual(
                (self.asset_directory / png_name).read_bytes(),
                original_payload,
            )
        undo_stack = _get_object_operation_undo_stack(edited_record)
        self.assertEqual(undo_stack[-1]["operation"], OBJECT_OPERATION_CREATE_FACE)
        self.assertEqual(changed_spy.count(), 1)

        QTest.keyClick(
            self.workspace.result_view.view,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier,
        )
        _qt_application.processEvents()

        restored_record = self.workspace._data.generated_objects[0]
        self.assertEqual(restored_record.asset_path, hole_record.asset_path)
        self.assertEqual(
            restored_record.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY],
            hole_variants,
        )
        self.assertIn(
            VISIBILITY_UV_UNWRAP_PIPELINE_KEY,
            restored_record.pipeline,
        )
        self.assertEqual(self.workspace.result_view.face_edit_face_count, 11)
        self.assertEqual(changed_spy.count(), 2)
        for edited_glb_path in edited_glb_paths:
            self.assertFalse((self.asset_directory / edited_glb_path).exists())
        for png_name, original_payload in self.original_png_payloads.items():
            self.assertEqual(
                (self.asset_directory / png_name).read_bytes(),
                original_payload,
            )

    def test_face_expanded_uv_seam_selection_fills_logical_hole(self) -> None:
        """Collapse drag-style coincident IDs before the async face fill."""

        hole_record, _hole_variants = self._install_triangle_hole_variants(
            face_expanded=True
        )
        expanded_geometry = load_object_face_geometry(
            (self.asset_directory / hole_record.asset_path).read_bytes()
        )
        shared_glb = _textured_box_with_triangle_hole_glb(
            16,
            (80, 120, 180, 255),
        )
        shared_geometry = load_object_face_geometry(shared_glb)
        boundary_indices = _triangle_hole_vertex_indices(shared_glb)
        coincident_groups: list[tuple[int, ...]] = []
        for boundary_index in boundary_indices:
            position = shared_geometry.vertices[boundary_index]
            matches = tuple(
                int(index)
                for index in np.flatnonzero(
                    np.linalg.norm(
                        expanded_geometry.vertices - position,
                        axis=1,
                    )
                    <= 1e-8
                )
            )
            self.assertGreater(len(matches), 1)
            coincident_groups.append(matches)
        drag_style_vertex_ids = tuple(
            vertex_index
            for group in coincident_groups
            for vertex_index in group
        )
        self.assertGreater(len(drag_style_vertex_ids), 3)

        self.workspace.result_view.set_selected_vertex_indices(
            drag_style_vertex_ids
        )

        selected_vertices = (
            self.workspace.result_view.get_selected_vertex_indices()
        )
        self.assertEqual(len(selected_vertices), 3)
        self.assertEqual(
            len(
                {
                    tuple(
                        np.round(expanded_geometry.vertices[index], decimals=8)
                    )
                    for index in selected_vertices
                }
            ),
            3,
        )

        QTest.keyClick(self.workspace.result_view.view, Qt.Key.Key_F)
        self._wait_for_jobs()

        edited_record = self.workspace._data.generated_objects[0]
        self.assertNotEqual(edited_record.asset_path, hole_record.asset_path)
        self.assertTrue(edited_record.pipeline["face_edit_texture_preserved"])
        self.assertEqual(
            edited_record.pipeline["face_edit_added_triangle_count"],
            1,
        )
        self.assertEqual(
            edited_record.pipeline["face_edit_created_polygon_count"],
            1,
        )
        for variant in edited_record.pipeline[
            TEXTURE_VARIANTS_PIPELINE_KEY
        ].values():
            self.assertEqual(
                load_object_face_geometry(
                    (self.asset_directory / variant["glb_asset_path"]).read_bytes()
                ).face_count,
                12,
            )

    def test_face_creation_preserves_symmetric_layout_metadata(self) -> None:
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
        hole_record, hole_variants = self._install_triangle_hole_variants(
            symmetry_metadata=symmetry_metadata
        )
        selected_vertices = _triangle_hole_vertex_indices(
            (self.asset_directory / hole_record.asset_path).read_bytes()
        )
        self.workspace.result_view.set_selected_vertex_indices(
            selected_vertices
        )

        self.assertTrue(self.workspace.create_selected_object_face())
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
                for resolution, variant in hole_variants.items()
            },
        )
        for variant in edited_variants.values():
            self.assertEqual(
                load_object_face_geometry(
                    (
                        self.asset_directory / variant["glb_asset_path"]
                    ).read_bytes()
                ).face_count,
                12,
            )

    def test_invalid_vertex_selection_is_retained_without_starting_job(
        self,
    ) -> None:
        selected_vertices = (0, 1)
        self.workspace.result_view.set_selected_vertex_indices(
            selected_vertices
        )

        self.assertFalse(self.workspace.create_selected_object_face())

        self.assertEqual(self.workspace._object_job_runtimes, {})
        self.assertEqual(
            self.workspace.result_view.get_selected_vertex_indices(),
            selected_vertices,
        )
        self.assertEqual(
            self.workspace._data.generated_objects,
            [self.original_record],
        )
        self.assertIn("at least three", self.workspace.status_label.text())

    def test_failed_face_creation_retains_vertices_and_existing_object(
        self,
    ) -> None:
        geometry = load_object_face_geometry(
            (self.asset_directory / self.original_record.asset_path).read_bytes()
        )
        selected_vertices = tuple(int(index) for index in geometry.faces[0])
        self.workspace.result_view.set_selected_vertex_indices(
            selected_vertices
        )

        QTest.keyClick(self.workspace.result_view.view, Qt.Key.Key_F)
        self._wait_for_jobs()

        self.assertEqual(
            self.workspace._data.generated_objects,
            [self.original_record],
        )
        self.assertEqual(
            self.workspace.result_view.get_selected_vertex_indices(),
            selected_vertices,
        )
        self.assertTrue(self.workspace.result_view._face_editing_enabled)
        self.assertIn("Face creation failed:", self.workspace.status_label.text())
        self.assertEqual(
            tuple(self.asset_directory.glob("*.face-edit-*.glb")),
            (),
        )

    def test_cancel_does_not_wait_for_blocked_local_face_creation(self) -> None:
        started = threading.Event()
        release = threading.Event()
        completed = threading.Event()

        def block_face_creation(*_args: object, **_kwargs: object) -> object:
            started.set()
            release.wait(timeout=5.0)
            completed.set()
            return object()

        hole_record, _hole_variants = self._install_triangle_hole_variants()
        selected_vertices = _triangle_hole_vertex_indices(
            (self.asset_directory / hole_record.asset_path).read_bytes()
        )
        self.workspace.result_view.set_selected_vertex_indices(
            selected_vertices
        )
        with patch(
            "housemaker.generation_workspace._prepare_object_face_creation",
            side_effect=block_face_creation,
        ):
            self.assertTrue(self.workspace.create_selected_object_face())
            self._wait_for_event(started)
            cancelled_at = time.monotonic()
            self.assertTrue(self.workspace.cancel_current_operation())
            self._wait_for_jobs(timeout_milliseconds=1_000)
            self.assertLess(time.monotonic() - cancelled_at, 0.5)
            release.set()
            self._wait_for_event(completed)

        self.assertEqual(
            self.workspace._data.generated_objects,
            [hole_record],
        )
        self.assertEqual(
            self.workspace.result_view.get_selected_vertex_indices(),
            selected_vertices,
        )
        self.assertEqual(
            tuple(self.asset_directory.glob("*.face-edit-*.glb")),
            (),
        )

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
