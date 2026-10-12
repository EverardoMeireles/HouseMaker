# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.glb import GeneratedModel, PreviewTexturedSurface
from housemaker.main import (
    BlueprintWorkspace,
    _build_local_file_revision,
    _load_progressive_texture_request,
    _ProgressiveTextureLoadRequest,
)
from housemaker.pbr_maps import ATLAS_MAP_BASE_COLOR, PBR_MAP_NORMAL
from housemaker.viewer import build_texture_preview_bundle

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _write_rgba(path: Path, size: int, color: tuple[int, int, int, int]) -> None:
    Image.new("RGBA", (size, size), color).save(path)


def _textured_surface_model() -> GeneratedModel:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=float,
    )
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    material = PBRMaterial(
        baseColorTexture=Image.new("RGBA", (16, 16), (40, 50, 60, 255))
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=TextureVisuals(
            uv=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            material=material,
        ),
        process=False,
    )
    return GeneratedModel(
        mesh=mesh.copy(),
        scene=trimesh.Scene(),
        glb_bytes=b"",
        preview_textured_surfaces=[
            PreviewTexturedSurface(
                surface_id="wall-surface",
                surface_type="wall",
                mesh=mesh,
                level_index=0,
            )
        ],
        preview_untextured_mesh=trimesh.Trimesh(process=False),
    )


# ### Test doubles ###
class _RecordingProgressiveViewer:
    """Small viewer double that records proxy and full-resolution commits."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.applied: list[tuple[object, object]] = []
        self._token_index = 0

    def begin_progressive_texture_update(
        self,
        update_id: str,
        *,
        surface_ids: tuple[str, ...],
        placed_object_ids: tuple[str, ...],
    ) -> object:
        self._token_index += 1
        return (
            self.name,
            update_id,
            self._token_index,
            surface_ids,
            placed_object_ids,
        )

    def apply_progressive_texture_update(
        self,
        token: object,
        bundle: object,
    ) -> int:
        self.applied.append((token, bundle))
        return 1


# ### Progressive decode tests ###
class ProgressiveTextureDecodeTests(unittest.TestCase):
    def test_full_bundle_decodes_aligned_base_and_normal_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base.png"
            normal_path = root / "normal.png"
            _write_rgba(base_path, 512, (210, 30, 20, 255))
            _write_rgba(normal_path, 512, (128, 128, 255, 255))
            paths = (
                (ATLAS_MAP_BASE_COLOR, base_path),
                (PBR_MAP_NORMAL, normal_path),
            )
            request = _ProgressiveTextureLoadRequest(
                request_key="material",
                map_paths=paths,
                source_revisions=tuple(
                    _build_local_file_revision(path) for _map_type, path in paths
                ),
                surface_ids=("wall-surface",),
            )

            bundle = _load_progressive_texture_request(request)

        self.assertEqual(bundle.base_color_rgba.shape, (512, 512, 4))
        self.assertIsNotNone(bundle.normal_rgba)
        assert bundle.normal_rgba is not None
        self.assertEqual(bundle.normal_rgba.shape, (512, 512, 4))
        self.assertEqual(tuple(bundle.base_color_rgba[0, 0]), (210, 30, 20, 255))

    def test_changed_file_revision_is_rejected_before_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.png"
            _write_rgba(path, 32, (10, 20, 30, 255))
            request = _ProgressiveTextureLoadRequest(
                request_key="stale",
                map_paths=((ATLAS_MAP_BASE_COLOR, path),),
                source_revisions=(('obsolete',),),
                surface_ids=("wall-surface",),
            )

            with self.assertRaisesRegex(RuntimeError, "changed before"):
                _load_progressive_texture_request(request)


# ### Main-window integration tests ###
class ProgressiveTextureMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        self.workspace.resize(1200, 760)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_proxy_is_visible_before_background_full_texture_commit(self) -> None:
        full_path = Path(self._temporary_directory.name) / "full.png"
        _write_rgba(full_path, 512, (230, 20, 10, 255))
        request = _ProgressiveTextureLoadRequest(
            request_key="surface-family",
            map_paths=((ATLAS_MAP_BASE_COLOR, full_path),),
            source_revisions=(_build_local_file_revision(full_path),),
            surface_ids=("wall-surface",),
            priority=0,
        )
        proxy = build_texture_preview_bundle(
            np.full((256, 256, 4), (10, 30, 220, 255), dtype=np.uint8)
        )
        model = _textured_surface_model()
        self.workspace.viewer.set_model(model)

        with patch.object(
            self.workspace,
            "_collect_progressive_scene_texture_requests",
            return_value=((request, proxy),),
        ):
            self.workspace._start_progressive_scene_texture_load(
                model,
                (self.workspace.viewer,),
            )

        item = self.workspace.viewer.textured_surface_items[0]
        self.assertEqual(item._texture_rgba.shape, (256, 256, 4))
        self.assertEqual(tuple(item._texture_rgba[0, 0]), (10, 30, 220, 255))

        runtime = next(
            iter(self.workspace._progressive_texture_load_runtimes.values())
        )
        future = next(iter(runtime.futures_by_request_key.values()))
        future.result(timeout=10.0)
        deadline = time.monotonic() + 10.0
        while item._texture_rgba.shape != (512, 512, 4):
            _qt_application.processEvents()
            if time.monotonic() >= deadline:
                self.fail("The full progressive texture was not committed.")
            time.sleep(0.01)

        self.assertEqual(item._texture_rgba.shape, (512, 512, 4))
        self.assertEqual(tuple(item._texture_rgba[0, 0]), (230, 20, 10, 255))

    def test_repeated_refresh_reuses_one_pool_and_retires_stale_batches(self) -> None:
        request = _ProgressiveTextureLoadRequest(
            request_key="surface-family",
            map_paths=(),
            source_revisions=(),
            surface_ids=("wall-surface",),
        )
        proxy = build_texture_preview_bundle(
            np.full((16, 16, 4), (10, 30, 220, 255), dtype=np.uint8)
        )
        viewer = _RecordingProgressiveViewer("canvas")
        futures: tuple[Future, ...] = (Future(), Future(), Future())
        self.assertTrue(futures[0].set_running_or_notify_cancel())
        executor = self.workspace._progressive_texture_load_executor
        assert executor is not None

        with (
            patch.object(
                self.workspace,
                "_collect_progressive_scene_texture_requests",
                return_value=((request, proxy),),
            ),
            patch.object(executor, "submit", side_effect=futures) as submit,
        ):
            for _index in range(3):
                self.workspace._start_progressive_scene_texture_load(
                    _textured_surface_model(),
                    (viewer,),
                )

        self.assertIs(self.workspace._progressive_texture_load_executor, executor)
        self.assertEqual(submit.call_count, 3)
        self.assertFalse(futures[0].cancelled())
        self.assertTrue(futures[1].cancelled())
        self.assertFalse(futures[2].cancelled())
        self.assertEqual(len(self.workspace._progressive_texture_load_runtimes), 1)

        applied_count = len(viewer.applied)
        futures[0].set_result(proxy)
        _qt_application.processEvents()
        self.assertEqual(len(viewer.applied), applied_count)

    def test_canvas_only_refresh_preserves_active_tour_promotion(self) -> None:
        request = _ProgressiveTextureLoadRequest(
            request_key="surface-family",
            map_paths=(),
            source_revisions=(),
            surface_ids=("wall-surface",),
        )
        proxy = build_texture_preview_bundle(
            np.full((16, 16, 4), (10, 30, 220, 255), dtype=np.uint8)
        )
        full = build_texture_preview_bundle(
            np.full((32, 32, 4), (230, 20, 10, 255), dtype=np.uint8)
        )
        canvas = _RecordingProgressiveViewer("canvas")
        tour = _RecordingProgressiveViewer("tour")
        shared_future: Future = Future()
        canvas_future: Future = Future()
        executor = self.workspace._progressive_texture_load_executor
        assert executor is not None

        with (
            patch.object(
                self.workspace,
                "_collect_progressive_scene_texture_requests",
                return_value=((request, proxy),),
            ),
            patch.object(
                executor,
                "submit",
                side_effect=(shared_future, canvas_future),
            ),
        ):
            self.workspace._start_progressive_scene_texture_load(
                _textured_surface_model(),
                (canvas, tour),
            )
            first_request_id = min(
                self.workspace._progressive_texture_load_runtimes
            )
            self.workspace._start_progressive_scene_texture_load(
                _textured_surface_model(),
                (canvas,),
            )

        first_runtime = self.workspace._progressive_texture_load_runtimes[
            first_request_id
        ]
        retained_viewers = tuple(
            viewer
            for viewer, _token in first_runtime.tokens_by_request_key[
                request.request_key
            ]
        )
        self.assertEqual(retained_viewers, (tour,))
        self.assertFalse(shared_future.cancelled())

        shared_future.set_result(full)
        _qt_application.processEvents()

        self.assertEqual(len(canvas.applied), 2)
        self.assertEqual(len(tour.applied), 2)
        self.assertIs(tour.applied[-1][1], full)

    def test_shutdown_processes_events_while_draining_owned_workers(self) -> None:
        request = _ProgressiveTextureLoadRequest(
            request_key="surface-family",
            map_paths=(),
            source_revisions=(),
            surface_ids=("wall-surface",),
        )
        bundle = build_texture_preview_bundle(
            np.full((16, 16, 4), (10, 30, 220, 255), dtype=np.uint8)
        )
        viewer = _RecordingProgressiveViewer("canvas")
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def load_texture(
            _request: _ProgressiveTextureLoadRequest,
        ) -> object:
            started.set()
            if not release.wait(timeout=5.0):
                raise TimeoutError("The shutdown test worker was not released.")
            finished.set()
            return bundle

        with (
            patch.object(
                self.workspace,
                "_collect_progressive_scene_texture_requests",
                return_value=((request, bundle),),
            ),
            patch(
                "housemaker.main._load_progressive_texture_request",
                side_effect=load_texture,
            ),
        ):
            self.workspace._start_progressive_scene_texture_load(
                _textured_surface_model(),
                (viewer,),
            )
            self.assertTrue(started.wait(timeout=2.0))
            QTimer.singleShot(25, release.set)
            self.workspace._shutdown_progressive_texture_loads()

        self.assertTrue(release.is_set())
        self.assertTrue(finished.is_set())
        self.assertIsNone(self.workspace._progressive_texture_load_executor)

    def test_project_finalize_does_not_eagerly_materialize_full_atlas(self) -> None:
        with (
            patch.object(self.workspace, "_schedule_viewer_preview_refresh"),
            patch.object(self.workspace, "_sync_atlas_object_texture_sources"),
            patch.object(
                self.workspace.texture_atlas_workspace,
                "materialize_missing_atlases",
            ) as materialize,
        ):
            self.workspace._finalize_project_state_apply()

        materialize.assert_not_called()


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
