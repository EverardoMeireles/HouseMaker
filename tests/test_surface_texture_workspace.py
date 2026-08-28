# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import copy
import io
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import (
    QApplication,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
)

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.camera_models import CameraPose, InitialFirstPersonCamera
from housemaker.generation_state import MASK_MODE_PAINT, MaskPoint, MaskStroke
from housemaker.glb import GeneratedModel
from housemaker.models import LevelData, RoomData, VertexData
from housemaker.settings_widget import (
    SURFACE_TEXTURE_PROVIDER_GPT_4O_MINI,
    SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA,
    SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA,
    SURFACE_TEXTURE_PROVIDER_MESHY,
    SURFACE_TEXTURE_PROVIDER_SETTING_KEY,
    GenerationServiceSettings,
)
from housemaker.surface_texture_providers import SurfaceTextureResult
from housemaker.surface_texture_state import (
    SURFACE_TEXTURE_RESOLUTIONS,
    SurfaceTextureAssignment,
    SurfaceTextureData,
    SurfaceTextureInpaintUndoSnapshot,
    SurfaceTextureVariant,
)
from housemaker.surface_texture_workspace import (
    DefaultSurfaceTextureProvider,
    SurfaceTextureGenerationWorkspace,
    SurfaceTextureRequest,
    _build_masked_crop,
    _build_surface_asset_revision,
    _build_surface_texture_outputs,
    _decode_png_rgba,
    _encode_png,
    _encode_rgba_png,
)
from housemaker.texture_atlas_view import TextureAtlasEntry


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _test_level() -> LevelData:
    vertex_data = VertexData()
    for point_x, point_y in (
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 100.0),
        (0.0, 100.0),
        (50.0, 50.0),
    ):
        vertex_data.add_vertex(point_x, point_y)
    for start_vertex_id, end_vertex_id in ((1, 2), (2, 3), (3, 4), (4, 1)):
        vertex_data.add_edge(start_vertex_id, end_vertex_id)
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[
            RoomData(
                name="Room",
                vertex_ids=(1, 2, 3, 4),
                center_vertex_id=5,
                color_rgb=(120, 150, 180),
            )
        ],
        image_size_pixels=(100.0, 100.0),
        floor_contour_vertex_ids=(1, 2, 3, 4),
    )


def _test_preview_model() -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    return GeneratedModel(
        mesh=mesh,
        scene=trimesh.Scene(mesh.copy()),
        glb_bytes=b"",
    )


def _test_stroke(x: float = 0.5) -> MaskStroke:
    return MaskStroke(
        mode=MASK_MODE_PAINT,
        radius_normalized=0.12,
        points=(MaskPoint(x=x, y=0.5),),
    )


def _write_test_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (80, 60),
    )
    if not writer.isOpened():
        raise unittest.SkipTest("MJPG writer is unavailable")
    try:
        for frame_index in range(frame_count):
            writer.write(
                np.full(
                    (60, 80, 3),
                    (20 + frame_index * 30, 70, 150),
                    dtype=np.uint8,
                )
            )
    finally:
        writer.release()


def _texture_png() -> bytes:
    did_encode, encoded = cv2.imencode(
        ".png",
        np.full((8, 12, 3), (30, 120, 210), dtype=np.uint8),
    )
    if not did_encode:
        raise AssertionError("Unable to build PNG fixture")
    return bytes(encoded)


def _colored_texture_png(color: tuple[int, int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", (10, 7), color).save(output, format="PNG")
    return output.getvalue()


def _set_current_mask(
    workspace: SurfaceTextureGenerationWorkspace,
    stroke: MaskStroke,
) -> None:
    workspace.video_view.set_strokes([stroke])
    workspace._handle_video_strokes_changed([stroke])


def _surface_assignment(
    assignment_id: str,
    surface_ids: tuple[str, ...],
    asset_path: str,
) -> SurfaceTextureAssignment:
    return SurfaceTextureAssignment(
        assignment_id=assignment_id,
        surface_type="wall",
        surface_ids=surface_ids,
        provider="meshy",
        asset_path=asset_path,
    )


def _surface_assignment_with_variants(
    asset_directory: Path,
    assignment_id: str,
    surface_ids: tuple[str, ...],
    *,
    surface_type: str = "wall",
    selected_resolution: int = 1024,
    color: tuple[int, int, int, int] = (180, 80, 30, 255),
) -> SurfaceTextureAssignment:
    asset_directory.mkdir(parents=True, exist_ok=True)
    variants: list[SurfaceTextureVariant] = []
    for resolution in SURFACE_TEXTURE_RESOLUTIONS:
        asset_path = f"{assignment_id}.texture-{resolution}.png"
        Image.new("RGBA", (resolution, resolution), color).save(
            asset_directory / asset_path,
            format="PNG",
        )
        variants.append(SurfaceTextureVariant(resolution, asset_path))
    active_path = next(
        variant.asset_path
        for variant in variants
        if variant.resolution == selected_resolution
    )
    return SurfaceTextureAssignment(
        assignment_id=assignment_id,
        surface_type=surface_type,
        surface_ids=surface_ids,
        provider="meshy",
        asset_path=active_path,
        texture_width=selected_resolution,
        texture_height=selected_resolution,
        texture_variants=tuple(variants),
        selected_texture_resolution=selected_resolution,
    )


def _other_texture_entry_ids(
    workspace: SurfaceTextureGenerationWorkspace,
) -> list[str]:
    return [
        str(
            workspace.other_texture_list.item(row).data(
                Qt.ItemDataRole.UserRole
            )
        )
        for row in range(workspace.other_texture_list.count())
    ]


def _other_texture_item(
    workspace: SurfaceTextureGenerationWorkspace,
    assignment_id: str,
) -> QListWidgetItem:
    expected_id = f"{assignment_id}:other-texture"
    for row in range(workspace.other_texture_list.count()):
        item = workspace.other_texture_list.item(row)
        if item.data(Qt.ItemDataRole.UserRole) == expected_id:
            return item
    raise AssertionError(f"Missing other texture item: {assignment_id}")


# ### Provider fixtures ###
class _FakeProvider:
    def __init__(self) -> None:
        self.requests: list[SurfaceTextureRequest] = []
        self.result = SurfaceTextureResult(
            provider=SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA,
            texture_png=_texture_png(),
            task_id="texture-task-1",
        )

    def generate(self, request: SurfaceTextureRequest) -> SurfaceTextureResult:
        self.requests.append(request)
        return self.result


class _BlockingProvider:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, _request: SurfaceTextureRequest) -> SurfaceTextureResult:
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise RuntimeError("Blocking surface provider timed out")
        return SurfaceTextureResult(
            provider="meshy",
            texture_png=_texture_png(),
            task_id="late-task",
        )


# ### Provider integration tests ###
class SurfaceTextureDefaultProviderIntegrationTests(unittest.TestCase):
    def test_default_adapter_forwards_partial_texture_edit_inputs(self) -> None:
        base = _texture_png()
        edit_mask = _texture_png()
        request = SurfaceTextureRequest(
            provider=SURFACE_TEXTURE_PROVIDER_MESHY,
            api_key="meshy-test-key",
            reference_pngs=(_texture_png(),),
            reference_frame_indices=(4,),
            surface_type="wall",
            surface_ids=("wall-one",),
            combined_area_m2=2.5,
            prompt="Repair the marked plaster",
            existing_texture_png=base,
            edit_mask_png=edit_mask,
        )
        expected = SurfaceTextureResult(
            provider=SURFACE_TEXTURE_PROVIDER_MESHY,
            texture_png=_texture_png(),
        )

        with patch(
            "housemaker.surface_texture_workspace.request_surface_texture",
            return_value=expected,
        ) as generate:
            result = DefaultSurfaceTextureProvider().generate(request)

        self.assertIs(result, expected)
        generate.assert_called_once_with(
            provider=SURFACE_TEXTURE_PROVIDER_MESHY,
            api_key="meshy-test-key",
            reference_pngs=request.reference_pngs,
            prompt=request.prompt,
            existing_texture_png=base,
            edit_mask_png=edit_mask,
            progress_callback=None,
            cancel_event=None,
        )


# ### Workspace tests ###
class SurfaceTextureGenerationWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_path = Path(tempfile.mkdtemp())
        self.provider = _FakeProvider()
        self.initial_camera = InitialFirstPersonCamera(
            level_index=2,
            pose=CameraPose(
                x=0.5,
                y=0.75,
                z=1.7,
                yaw_degrees=35.0,
                pitch_degrees=-4.0,
            ),
        )
        self.workspace = SurfaceTextureGenerationWorkspace(
            provider=self.provider,
            asset_directory=self._temporary_path / "surface_assets",
            application_settings=ApplicationSettingsStore(
                self._temporary_path / "settings.json"
            ),
        )
        self.workspace.set_levels([_test_level()], self.initial_camera)
        self.workspace.resize(1000, 700)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        shutil.rmtree(self._temporary_path, ignore_errors=True)

    def test_plane_controls_are_absent(self) -> None:
        self.assertFalse(hasattr(self.workspace, "add_plane_button"))
        self.assertFalse(hasattr(self.workspace, "remove_plane_button"))
        self.assertIsNone(
            self.workspace.findChild(
                QPushButton,
                "add_surface_texture_plane_button",
            )
        )
        self.assertIsNone(
            self.workspace.findChild(
                QPushButton,
                "remove_surface_texture_plane_button",
            )
        )

    def test_equivalent_levels_skip_all_surface_rebuilding(self) -> None:
        equivalent_level = copy.deepcopy(self.workspace._levels[0])
        equivalent_camera = InitialFirstPersonCamera.from_dict(
            self.initial_camera.to_dict()
        )

        with (
            patch.object(self.workspace, "_store_viewer_state") as store_state,
            patch.object(self.workspace.surface_view, "set_levels") as set_levels,
            patch.object(
                self.workspace,
                "_restore_viewer_state",
            ) as restore_state,
            patch.object(
                self.workspace,
                "_restore_assignment_textures",
            ) as restore_textures,
            patch.object(
                self.workspace,
                "_refresh_texture_atlases",
            ) as refresh_atlases,
            patch.object(
                self.workspace,
                "_sync_selection_status",
            ) as sync_selection,
            patch.object(self.workspace, "_sync_controls") as sync_controls,
        ):
            self.workspace.set_levels(
                [equivalent_level],
                equivalent_camera,
            )

        store_state.assert_not_called()
        set_levels.assert_not_called()
        restore_state.assert_not_called()
        restore_textures.assert_not_called()
        refresh_atlases.assert_not_called()
        sync_selection.assert_not_called()
        sync_controls.assert_not_called()
        self.assertIs(self.workspace._levels[0], equivalent_level)

    def test_changed_levels_preserve_viewer_state_without_data_signal(self) -> None:
        custom_pose = CameraPose(
            x=9.0,
            y=8.0,
            z=7.0,
            yaw_degrees=61.0,
            pitch_degrees=-12.0,
        )
        selected_wall = "level:2/room:5/wall:1:2"
        self.workspace.surface_view.set_camera_pose(custom_pose)
        self.workspace.surface_view.set_selected_surface_ids((selected_wall,))
        changed = QSignalSpy(self.workspace.data_changed)
        changed_level = copy.deepcopy(self.workspace._levels[0])
        changed_level.height_meters += 0.25

        self.workspace.set_levels([changed_level], self.initial_camera)

        self.assertEqual(
            self.workspace.surface_view.get_camera_pose(),
            custom_pose,
        )
        self.assertEqual(
            self.workspace.surface_view.get_selected_surface_ids(),
            (selected_wall,),
        )
        restored_data = self.workspace.get_data()
        self.assertEqual(restored_data.camera_pose, custom_pose)
        self.assertEqual(restored_data.selected_surface_ids, (selected_wall,))
        self.assertEqual(changed.count(), 0)

    def test_in_place_level_change_rebuilds_surface_content(self) -> None:
        mutable_level = self.workspace._levels[0]
        mutable_level.height_meters += 0.5

        with (
            patch.object(self.workspace, "_store_viewer_state") as store_state,
            patch.object(self.workspace.surface_view, "set_levels") as set_levels,
            patch.object(
                self.workspace,
                "_restore_viewer_state",
            ) as restore_state,
            patch.object(
                self.workspace,
                "_restore_assignment_textures",
            ) as restore_textures,
            patch.object(
                self.workspace,
                "_refresh_texture_atlases",
            ) as refresh_atlases,
            patch.object(
                self.workspace,
                "_sync_selection_status",
            ) as sync_selection,
            patch.object(self.workspace, "_sync_controls") as sync_controls,
        ):
            self.workspace.set_levels([mutable_level], self.initial_camera)

        store_state.assert_called_once_with()
        set_levels.assert_called_once_with(
            [mutable_level],
            self.initial_camera,
        )
        restore_state.assert_called_once_with()
        restore_textures.assert_called_once_with()
        refresh_atlases.assert_called_once_with()
        sync_selection.assert_called_once_with()
        sync_controls.assert_called_once_with()

    def test_initial_camera_change_rebuilds_surface_content(self) -> None:
        changed_camera = InitialFirstPersonCamera(
            level_index=self.initial_camera.level_index,
            pose=CameraPose(
                x=self.initial_camera.pose.x + 1.0,
                y=self.initial_camera.pose.y,
                z=self.initial_camera.pose.z,
                yaw_degrees=self.initial_camera.pose.yaw_degrees,
                pitch_degrees=self.initial_camera.pose.pitch_degrees,
                roll_degrees=self.initial_camera.pose.roll_degrees,
                fov_degrees=self.initial_camera.pose.fov_degrees,
            ),
            light_intensity=self.initial_camera.light_intensity,
        )

        with (
            patch.object(self.workspace.surface_view, "set_levels") as set_levels,
            patch.object(self.workspace, "_restore_viewer_state"),
            patch.object(
                self.workspace,
                "_restore_assignment_textures",
            ) as restore_textures,
            patch.object(
                self.workspace,
                "_refresh_texture_atlases",
            ) as refresh_atlases,
        ):
            self.workspace.set_levels(
                list(self.workspace._levels),
                changed_camera,
            )

        set_levels.assert_called_once_with(
            self.workspace._levels,
            changed_camera,
        )
        restore_textures.assert_called_once_with()
        refresh_atlases.assert_called_once_with()

    def test_changed_preview_context_populates_the_gl_scene_once(self) -> None:
        mutable_level = self.workspace._levels[0]
        mutable_level.height_meters += 0.5
        preview_model = _test_preview_model()

        with patch.object(
            self.workspace.surface_view,
            "_populate_scene",
            wraps=self.workspace.surface_view._populate_scene,
        ) as populate_scene:
            self.workspace.set_preview_context(
                [mutable_level],
                self.initial_camera,
                preview_model,
            )

        populate_scene.assert_called_once_with()
        self.assertIs(
            self.workspace.surface_view.get_scene_model(),
            preview_model,
        )

    def test_provider_selector_has_exact_options_and_persists_selection(self) -> None:
        combo = self.workspace.surface_texture_provider_combo
        options = [
            (combo.itemText(index), combo.itemData(index))
            for index in range(combo.count())
        ]
        self.assertEqual(
            options,
            [
                ("Meshy", SURFACE_TEXTURE_PROVIDER_MESHY),
                ("GPT-4o-mini", SURFACE_TEXTURE_PROVIDER_GPT_4O_MINI),
                ("GPT-5.6 Luna", SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA),
                ("GPT-5.6 Terra", SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA),
            ],
        )

        terra_index = combo.findData(SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA)
        combo.setCurrentIndex(terra_index)

        settings_store = ApplicationSettingsStore(
            self._temporary_path / "settings.json"
        )
        self.assertEqual(
            settings_store.get(SURFACE_TEXTURE_PROVIDER_SETTING_KEY),
            SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA,
        )
        self.assertEqual(
            self.workspace.get_runtime_settings().surface_texture_provider,
            SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA,
        )

        restored = SurfaceTextureGenerationWorkspace(
            asset_directory=self._temporary_path / "restored_assets",
            application_settings=settings_store,
        )
        try:
            self.assertEqual(
                restored.surface_texture_provider_combo.currentData(),
                SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA,
            )
        finally:
            restored.shutdown()
            restored.close()

    def test_equal_views_initial_camera_and_homogeneous_shift_selection(self) -> None:
        view_sizes = self.workspace.views_splitter.sizes()
        self.assertEqual(len(view_sizes), 2)
        self.assertLessEqual(abs(view_sizes[0] - view_sizes[1]), 2)
        self.assertEqual(
            self.workspace.surface_view.get_camera_pose(),
            self.initial_camera.pose,
        )

        viewer = self.workspace.surface_view
        first_wall = "level:2/room:5/wall:1:2"
        second_wall = "level:2/room:5/wall:2:3"
        floor = "level:2/room:5/floor"
        self.assertTrue(viewer.select_surface(first_wall))
        self.assertTrue(viewer.select_surface(second_wall, shift_pressed=True))
        self.assertEqual(viewer.get_selected_surface_ids(), (first_wall, second_wall))

        self.assertFalse(viewer.select_surface(floor, shift_pressed=True))
        self.assertEqual(viewer.get_selected_surface_ids(), (first_wall, second_wall))
        self.assertTrue(viewer.select_surface(first_wall, shift_pressed=True))
        self.assertEqual(viewer.get_selected_surface_ids(), (second_wall,))
        self.assertTrue(viewer.select_surface(floor))
        self.assertEqual(viewer.get_selected_surface_ids(), (floor,))
        self.assertEqual(viewer.get_selected_surface_type(), "floor")

    def test_video_view_is_left_of_fixed_surfaces_view(self) -> None:
        splitter = self.workspace.views_splitter

        self.assertIs(splitter.widget(0), self.workspace.video_view.parentWidget())
        self.assertIs(splitter.widget(1), self.workspace.right_view_stack)
        self.assertIs(
            self.workspace.surface_view.parentWidget(),
            self.workspace.surface_3d_page,
        )
        self.assertLessEqual(abs(splitter.sizes()[0] - splitter.sizes()[1]), 2)

    def test_texture_view_shows_only_three_resolutions_for_selected_family(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        surface_id = "level:2/room:5/floor"
        assignment = _surface_assignment_with_variants(
            asset_directory,
            "oak-floor",
            (surface_id,),
            surface_type="floor",
            color=(25, 190, 50, 255),
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="floor",
                selected_surface_ids=(surface_id,),
                assignments=[assignment],
            )
        )

        self.assertEqual(
            [entry.atlas_id for entry in self.workspace.texture_view.entries],
            [
                f"oak-floor:resolution:{resolution}"
                for resolution in SURFACE_TEXTURE_RESOLUTIONS
            ],
        )
        self.assertEqual(
            [entry.display_name for entry in self.workspace.texture_view.entries],
            ["512 x 512", "1024 x 1024", "2048 x 2048"],
        )
        self.assertEqual(
            self.workspace.texture_view.selected_atlas_id,
            "oak-floor:resolution:1024",
        )
        selected_image = self.workspace.texture_view.selected_entry.get_image()
        self.assertEqual(selected_image.pixelColor(0, 0).green(), 190)
        cached_entries = self.workspace.texture_view.entries

        self.workspace._handle_surface_selection_changed((surface_id,))

        self.assertEqual(len(self.workspace.texture_view.entries), 3)
        self.assertTrue(
            all(
                current is cached
                for current, cached in zip(
                    self.workspace.texture_view.entries,
                    cached_entries,
                    strict=True,
                )
            )
        )

        self.workspace.set_external_3d_viewer_active(True)
        self.assertIs(
            self.workspace.right_view_stack.currentWidget(),
            self.workspace.texture_view_page,
        )
        self.workspace.set_external_3d_viewer_active(False)
        self.assertIs(
            self.workspace.right_view_stack.currentWidget(),
            self.workspace.surface_3d_page,
        )

    def test_same_path_texture_replacement_rebuilds_thumbnail_and_signature(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        surface_id = "level:2/room:5/floor"
        assignment = _surface_assignment_with_variants(
            asset_directory,
            "changing-floor",
            (surface_id,),
            surface_type="floor",
            color=(25, 190, 50, 255),
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="floor",
                selected_surface_ids=(surface_id,),
                assignments=[assignment],
            )
        )
        original_entry = self.workspace.texture_view.selected_entry
        original_signature = self.workspace.get_preview_dependency_signature()
        texture_path = asset_directory / assignment.asset_path
        Image.new("RGBA", (1024, 1024), (210, 30, 40, 255)).save(
            texture_path
        )
        current_stat = texture_path.stat()
        os.utime(
            texture_path,
            ns=(current_stat.st_atime_ns, current_stat.st_mtime_ns + 1_000_000),
        )

        self.workspace._refresh_texture_atlases()

        replacement_entry = self.workspace.texture_view.selected_entry
        self.assertIsNot(replacement_entry, original_entry)
        self.assertEqual(
            replacement_entry.get_image().pixelColor(0, 0).red(),
            210,
        )
        self.assertNotEqual(
            self.workspace.get_preview_dependency_signature(),
            original_signature,
        )

    def test_file_backed_preview_refresh_is_revision_guarded(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        surface_id = "level:2/room:5/floor"
        assignment = _surface_assignment_with_variants(
            asset_directory,
            "guarded-floor",
            (surface_id,),
            surface_type="floor",
            color=(25, 190, 50, 255),
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="floor",
                selected_surface_ids=(surface_id,),
                assignments=[assignment],
            )
        )

        with (
            patch.object(
                self.workspace,
                "_restore_assignment_textures",
                wraps=self.workspace._restore_assignment_textures,
            ) as restore_textures,
            patch.object(
                self.workspace,
                "_refresh_texture_atlases",
                wraps=self.workspace._refresh_texture_atlases,
            ) as refresh_atlases,
        ):
            self.workspace.refresh_file_backed_previews()
            restore_textures.assert_not_called()
            refresh_atlases.assert_not_called()

            texture_path = asset_directory / assignment.asset_path
            Image.new("RGBA", (1024, 1024), (210, 30, 40, 255)).save(
                texture_path
            )
            current_stat = texture_path.stat()
            os.utime(
                texture_path,
                ns=(
                    current_stat.st_atime_ns,
                    current_stat.st_mtime_ns + 1_000_000,
                ),
            )
            self.workspace.refresh_file_backed_previews()

        restore_textures.assert_called_once_with()
        refresh_atlases.assert_called_once_with()

    def test_file_backed_refresh_stats_each_texture_path_once(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        surface_id = "level:2/room:5/floor"
        assignment = _surface_assignment_with_variants(
            asset_directory,
            "single-stat-floor",
            (surface_id,),
            surface_type="floor",
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="floor",
                selected_surface_ids=(surface_id,),
                assignments=[assignment],
            )
        )

        with patch(
            "housemaker.surface_texture_workspace."
            "_build_surface_asset_revision",
            wraps=_build_surface_asset_revision,
        ) as build_revision:
            self.workspace.refresh_file_backed_previews()

        self.assertEqual(build_revision.call_count, 3)

    def test_known_missing_texture_revision_is_cached_until_reappearance(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        surface_id = "level:2/room:5/wall:1:2"
        valid_assignment = _surface_assignment_with_variants(
            asset_directory,
            "available-wall",
            (surface_id,),
            surface_type="wall",
            color=(25, 190, 50, 255),
        )
        missing_assignment = SurfaceTextureAssignment(
            assignment_id="missing-wall",
            surface_type="wall",
            surface_ids=("level:2/room:5/wall:2:3",),
            provider="test",
            asset_path="missing-wall.png",
            texture_width=32,
            texture_height=32,
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(surface_id,),
                assignments=[valid_assignment, missing_assignment],
            )
        )

        with (
            patch.object(
                self.workspace,
                "_restore_assignment_textures",
                wraps=self.workspace._restore_assignment_textures,
            ) as restore_textures,
            patch.object(
                self.workspace,
                "_refresh_texture_atlases",
                wraps=self.workspace._refresh_texture_atlases,
            ) as refresh_atlases,
        ):
            self.workspace.refresh_file_backed_previews()
            self.workspace.refresh_file_backed_previews()
            restore_textures.assert_not_called()
            refresh_atlases.assert_not_called()

            Image.new("RGBA", (32, 32), (210, 30, 40, 255)).save(
                asset_directory / missing_assignment.asset_path
            )
            self.workspace.refresh_file_backed_previews()

        restore_textures.assert_called_once_with()
        refresh_atlases.assert_called_once_with()

    def test_failed_surface_install_retries_the_same_revision(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        assignment = _surface_assignment_with_variants(
            asset_directory,
            "retry-floor",
            ("level:2/room:5/floor",),
            surface_type="floor",
            color=(25, 190, 50, 255),
        )
        self.workspace.set_data(SurfaceTextureData(assignments=[assignment]))
        self.workspace._restored_assignment_texture_signature = None

        with patch.object(
            self.workspace.surface_view,
            "set_surface_texture",
            side_effect=(ValueError("temporary decode failure"), None),
        ) as install_texture:
            self.workspace.refresh_file_backed_previews()
            self.assertIsNone(
                self.workspace._restored_assignment_texture_signature
            )
            self.workspace.refresh_file_backed_previews()

        self.assertEqual(install_texture.call_count, 2)
        self.assertIsNotNone(
            self.workspace._restored_assignment_texture_signature
        )

    def test_failed_surface_thumbnail_decode_retries_same_revision(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        assignment = _surface_assignment_with_variants(
            asset_directory,
            "retry-thumbnail",
            ("level:2/room:5/floor",),
            surface_type="floor",
            color=(25, 190, 50, 255),
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="floor",
                selected_surface_ids=("level:2/room:5/floor",),
                assignments=[assignment],
            )
        )
        self.workspace._texture_atlas_entry_cache.clear()
        self.workspace._texture_catalog_dependency_signature = None
        failed_once = False

        def build_entry(*args, **kwargs):
            nonlocal failed_once
            if not failed_once and str(kwargs.get("atlas_id", "")).endswith(
                ":resolution:512"
            ):
                failed_once = True
                raise ValueError("temporary thumbnail decode failure")
            return TextureAtlasEntry(*args, **kwargs)

        with patch(
            "housemaker.surface_texture_workspace.TextureAtlasEntry",
            side_effect=build_entry,
        ) as entry_builder:
            self.workspace.refresh_file_backed_previews()
            self.assertIsNone(
                self.workspace._texture_catalog_dependency_signature
            )
            self.workspace.refresh_file_backed_previews()

        self.assertEqual(entry_builder.call_count, 4)
        self.assertEqual(len(self.workspace.texture_view.entries), 3)
        self.assertIsNotNone(
            self.workspace._texture_catalog_dependency_signature
        )

    def test_surface_material_revision_change_during_load_retries(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        assignment = _surface_assignment_with_variants(
            asset_directory,
            "material-race",
            ("level:2/room:5/floor",),
            surface_type="floor",
            color=(25, 190, 50, 255),
        )
        self.workspace.set_data(SurfaceTextureData(assignments=[assignment]))
        self.workspace._restored_assignment_texture_signature = None
        revision_before = (("material", "before"),)
        revision_after = (("material", "after"),)

        with patch.object(
            self.workspace,
            "get_preview_dependency_signature",
            side_effect=(
                revision_before,
                revision_before,
                revision_after,
                revision_after,
                revision_after,
                revision_after,
            ),
        ):
            self.workspace.refresh_file_backed_previews()
            self.assertIsNone(
                self.workspace._restored_assignment_texture_signature
            )
            self.workspace.refresh_file_backed_previews()

        self.assertEqual(
            self.workspace._restored_assignment_texture_signature,
            revision_after,
        )

    def test_surface_catalog_revision_change_during_decode_retries(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        assignment = _surface_assignment_with_variants(
            asset_directory,
            "catalog-race",
            ("level:2/room:5/floor",),
            surface_type="floor",
            color=(25, 190, 50, 255),
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="floor",
                selected_surface_ids=("level:2/room:5/floor",),
                assignments=[assignment],
            )
        )
        self.workspace._texture_atlas_entry_cache.clear()
        self.workspace._texture_catalog_dependency_signature = None
        revision_before = (("catalog", "before"),)
        revision_after = (("catalog", "after"),)

        with patch.object(
            self.workspace,
            "_build_texture_catalog_dependency_signature",
            side_effect=(
                revision_before,
                revision_before,
                revision_after,
                revision_after,
                revision_after,
                revision_after,
            ),
        ):
            self.workspace.refresh_file_backed_previews()
            self.assertIsNone(
                self.workspace._texture_catalog_dependency_signature
            )
            self.workspace.refresh_file_backed_previews()

        self.assertEqual(
            self.workspace._texture_catalog_dependency_signature,
            revision_after,
        )

    def test_assignment_apply_reloads_a_concurrently_replaced_png(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        asset_directory.mkdir(exist_ok=True)
        texture_path = asset_directory / "apply-race.png"
        original_png = _colored_texture_png((180, 40, 30, 255))
        replacement_png = _colored_texture_png((20, 170, 90, 255))
        texture_path.write_bytes(original_png)
        first_wall = "level:2/room:5/wall:1:2"
        second_wall = "level:2/room:5/wall:2:3"
        assignment = _surface_assignment(
            "apply-race",
            (first_wall,),
            texture_path.name,
        )
        self.workspace.set_data(SurfaceTextureData(assignments=[assignment]))
        install_texture = self.workspace.surface_view.set_surface_texture

        def install_then_replace(
            surface_ids: tuple[str, ...],
            texture: object,
        ) -> None:
            install_texture(surface_ids, texture)  # type: ignore[arg-type]
            texture_path.write_bytes(replacement_png)

        with patch.object(
            self.workspace.surface_view,
            "set_surface_texture",
            side_effect=install_then_replace,
        ):
            self.assertTrue(
                self.workspace.apply_assignment_texture(
                    assignment.assignment_id,
                    (second_wall,),
                )
            )

        self.assertIsNone(
            self.workspace._restored_assignment_texture_signature
        )
        installed_before_refresh = (
            self.workspace.surface_view.get_surface_texture_rgba(second_wall)
        )
        assert installed_before_refresh is not None
        self.assertEqual(tuple(installed_before_refresh[0, 0]), (180, 40, 30, 255))

        self.workspace.refresh_file_backed_previews()

        installed_after_refresh = (
            self.workspace.surface_view.get_surface_texture_rgba(second_wall)
        )
        assert installed_after_refresh is not None
        self.assertEqual(tuple(installed_after_refresh[0, 0]), (20, 170, 90, 255))
        self.assertIsNotNone(
            self.workspace._restored_assignment_texture_signature
        )

    def test_generation_commit_reloads_a_concurrently_replaced_png(self) -> None:
        wall_id = "level:2/room:5/wall:1:2"
        generated_png = _colored_texture_png((180, 40, 30, 255))
        replacement_output = io.BytesIO()
        Image.new("RGBA", (1024, 1024), (20, 170, 90, 255)).save(
            replacement_output,
            format="PNG",
        )
        replacement_png = replacement_output.getvalue()
        request = SurfaceTextureRequest(
            provider="meshy",
            api_key="test-key",
            reference_pngs=(_texture_png(),),
            reference_frame_indices=(0,),
            surface_type="wall",
            surface_ids=(wall_id,),
            combined_area_m2=6.0,
            prompt="Generate one wall",
        )
        install_texture = self.workspace.surface_view.set_surface_texture

        def install_then_replace(
            surface_ids: tuple[str, ...],
            texture: object,
        ) -> None:
            install_texture(surface_ids, texture)  # type: ignore[arg-type]
            active_paths = tuple(
                (self._temporary_path / "surface_assets").glob(
                    "*.texture-1024.png"
                )
            )
            self.assertEqual(len(active_paths), 1)
            active_paths[0].write_bytes(replacement_png)

        with patch.object(
            self.workspace.surface_view,
            "set_surface_texture",
            side_effect=install_then_replace,
        ):
            self.workspace._handle_generation_succeeded(
                request,
                SurfaceTextureResult(
                    provider="meshy",
                    texture_png=generated_png,
                ),
            )

        self.assertIsNone(
            self.workspace._restored_assignment_texture_signature
        )
        installed_before_refresh = (
            self.workspace.surface_view.get_surface_texture_rgba(wall_id)
        )
        assert installed_before_refresh is not None
        self.assertEqual(tuple(installed_before_refresh[0, 0]), (180, 40, 30, 255))

        self.workspace.refresh_file_backed_previews()

        installed_after_refresh = (
            self.workspace.surface_view.get_surface_texture_rgba(wall_id)
        )
        assert installed_after_refresh is not None
        self.assertEqual(tuple(installed_after_refresh[0, 0]), (20, 170, 90, 255))
        self.assertIsNotNone(
            self.workspace._restored_assignment_texture_signature
        )

    def test_single_clicked_variant_changes_the_assignment_globally(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        surface_ids = (
            "level:2/room:5/wall:1:2",
            "level:2/room:5/wall:2:3",
        )
        assignment = _surface_assignment_with_variants(
            asset_directory,
            "global-brick",
            surface_ids,
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(surface_ids[0],),
                assignments=[assignment],
            )
        )

        self.assertTrue(
            self.workspace.texture_view.select_atlas(
                "global-brick:resolution:2048"
            )
        )
        _qt_application.processEvents()

        selected_assignment = self.workspace.get_data().assignments[0]
        self.assertEqual(selected_assignment.selected_texture_resolution, 2048)
        self.assertEqual(
            selected_assignment.asset_path,
            "global-brick.texture-2048.png",
        )
        for surface_id in surface_ids:
            texture = self.workspace.surface_view.get_surface_texture_rgba(
                surface_id
            )
            self.assertIsNotNone(texture)
            self.assertEqual(texture.shape[:2], (2048, 2048))

    def test_rejected_single_click_restores_the_active_resolution(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        surface_id = "level:2/room:5/floor"
        assignment = _surface_assignment_with_variants(
            asset_directory,
            "blocked-floor",
            (surface_id,),
            surface_type="floor",
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="floor",
                selected_surface_ids=(surface_id,),
                assignments=[assignment],
            )
        )
        requests: list[tuple[str, int]] = []
        self.workspace.set_texture_resolution_change_handler(
            lambda assignment_id, resolution: (
                requests.append((assignment_id, resolution)) or False
            )
        )

        self.assertTrue(
            self.workspace.texture_view.select_atlas(
                "blocked-floor:resolution:2048"
            )
        )
        _qt_application.processEvents()

        self.assertEqual(requests, [("blocked-floor", 2048)])
        selected_assignment = self.workspace.get_data().assignments[0]
        self.assertEqual(selected_assignment.selected_texture_resolution, 1024)
        self.assertEqual(
            self.workspace.texture_view.selected_atlas_id,
            "blocked-floor:resolution:1024",
        )
        self.assertIn(
            "previous resolution was kept",
            self.workspace.status_label.text(),
        )

    def test_texture_view_clears_stale_family_for_untextured_selection(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        textured_surface_id = "level:2/room:5/floor"
        untextured_surface_id = "level:2/room:5/ceiling"
        assignment = _surface_assignment_with_variants(
            asset_directory,
            "valid-family",
            (textured_surface_id,),
            surface_type="floor",
            color=(85, 120, 210, 255),
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="floor",
                selected_surface_ids=(textured_surface_id,),
                assignments=[assignment],
            )
        )
        self.assertEqual(len(self.workspace.texture_view.entries), 3)

        self.workspace.surface_view.set_selected_surface_ids(
            (untextured_surface_id,)
        )

        self.assertEqual(self.workspace.texture_view.entries, ())
        self.assertIsNone(self.workspace.texture_view.selected_atlas_id)

    def test_other_texture_library_filters_and_tracks_surface_selection(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        first_wall = "level:2/room:5/wall:1:2"
        second_wall = "level:2/room:5/wall:2:3"
        untextured_wall = "level:2/room:5/wall:3:4"
        legacy_wall = "level:2/room:5/wall:1:4"
        floor = "level:2/room:5/floor"
        current_wall_assignment = _surface_assignment_with_variants(
            asset_directory,
            "current-wall",
            (first_wall,),
            selected_resolution=1024,
        )
        other_wall_assignment = _surface_assignment_with_variants(
            asset_directory,
            "other-wall",
            (second_wall,),
            selected_resolution=2048,
        )
        legacy_asset_path = "legacy-wall.png"
        Image.new("RGBA", (13, 9), (70, 90, 210, 255)).save(
            asset_directory / legacy_asset_path,
            format="PNG",
        )
        legacy_wall_assignment = _surface_assignment(
            "legacy-wall",
            (legacy_wall,),
            legacy_asset_path,
        )
        floor_assignment = _surface_assignment_with_variants(
            asset_directory,
            "floor-texture",
            (floor,),
            surface_type="floor",
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(first_wall,),
                assignments=[
                    current_wall_assignment,
                    other_wall_assignment,
                    legacy_wall_assignment,
                    floor_assignment,
                ],
            )
        )

        self.assertEqual(self.workspace.texture_views_splitter.count(), 2)
        self.assertEqual(
            _other_texture_entry_ids(self.workspace),
            [
                "other-wall:other-texture",
                "legacy-wall:other-texture",
            ],
        )
        other_item = _other_texture_item(self.workspace, "other-wall")
        self.assertEqual(other_item.text(), "Wall texture - 2048 x 2048")
        self.assertIn("Provider: Meshy", other_item.toolTip())
        self.assertIn("Double-click to apply", other_item.toolTip())
        legacy_item = _other_texture_item(self.workspace, "legacy-wall")
        self.assertEqual(legacy_item.text(), "Wall texture - fixed image")
        self.assertIn(
            "Fixed image: 13 x 9 (no resolution variants)",
            legacy_item.toolTip(),
        )

        self.workspace.surface_view.set_selected_surface_ids((second_wall,))

        self.assertEqual(
            self.workspace.texture_view.selected_atlas_id,
            "other-wall:resolution:2048",
        )
        self.assertEqual(
            _other_texture_entry_ids(self.workspace),
            [
                "current-wall:other-texture",
                "legacy-wall:other-texture",
            ],
        )

        self.workspace.surface_view.set_selected_surface_ids(
            (untextured_wall,)
        )

        self.assertEqual(self.workspace.texture_view.entries, ())
        self.assertEqual(
            _other_texture_entry_ids(self.workspace),
            [
                "current-wall:other-texture",
                "other-wall:other-texture",
                "legacy-wall:other-texture",
            ],
        )

        self.workspace.surface_view.set_selected_surface_ids((legacy_wall,))

        self.assertEqual(self.workspace.texture_view.entries, ())
        self.assertEqual(
            _other_texture_entry_ids(self.workspace),
            [
                "current-wall:other-texture",
                "other-wall:other-texture",
            ],
        )

    def test_delete_current_texture_family_clears_state_assets_and_viewer(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        target_surface = "level:2/room:5/wall:1:2"
        retained_surface = "level:2/room:5/wall:2:3"
        target = _surface_assignment_with_variants(
            asset_directory,
            "delete-plaster",
            (target_surface,),
        )
        retained = _surface_assignment_with_variants(
            asset_directory,
            "keep-brick",
            (retained_surface,),
        )
        target_stroke = _test_stroke(0.25)
        retained_stroke = _test_stroke(0.75)
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(target_surface,),
                assignments=[target, retained],
                texture_mask_strokes={
                    target_surface: [target_stroke],
                    retained_surface: [retained_stroke],
                },
                localized_inpaint_undo_stack=[
                    SurfaceTextureInpaintUndoSnapshot(
                        previous_assignments=(target,),
                        replacement_assignment_ids=("localized-edit",),
                        affected_surface_ids=(target_surface,),
                    )
                ],
            )
        )
        event_order: list[str] = []
        removed = QSignalSpy(self.workspace.assignments_removed)
        changed = QSignalSpy(self.workspace.data_changed)
        content_changed = QSignalSpy(self.workspace.surface_content_changed)
        self.workspace.assignments_removed.connect(
            lambda _ids: event_order.append("removed")
        )
        self.workspace.data_changed.connect(
            lambda _data: event_order.append("data")
        )
        self.workspace.surface_content_changed.connect(
            lambda: event_order.append("content")
        )

        self.assertTrue(
            self.workspace.delete_assignment_texture("delete-plaster")
        )

        data = self.workspace.get_data()
        self.assertEqual(
            [assignment.assignment_id for assignment in data.assignments],
            ["keep-brick"],
        )
        self.assertEqual(
            data.texture_mask_strokes,
            {retained_surface: [retained_stroke]},
        )
        self.assertEqual(data.localized_inpaint_undo_stack, [])
        self.assertEqual(tuple(removed.at(0)[0]), ("delete-plaster",))
        self.assertEqual(changed.count(), 1)
        self.assertEqual(content_changed.count(), 1)
        self.assertEqual(event_order, ["removed", "data", "content"])
        self.assertIsNone(
            self.workspace.surface_view.get_surface_texture_rgba(
                target_surface
            )
        )
        self.assertIsNotNone(
            self.workspace.surface_view.get_surface_texture_rgba(
                retained_surface
            )
        )
        self.assertEqual(self.workspace.texture_view.entries, ())
        self.assertTrue(
            all(
                not (
                    asset_directory
                    / f"delete-plaster.texture-{resolution}.png"
                ).exists()
                for resolution in SURFACE_TEXTURE_RESOLUTIONS
            )
        )
        self.assertTrue(
            all(
                (
                    asset_directory
                    / f"keep-brick.texture-{resolution}.png"
                ).is_file()
                for resolution in SURFACE_TEXTURE_RESOLUTIONS
            )
        )

    def test_delete_key_targets_explicit_other_texture_after_confirmation(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        current_surface = "level:2/room:5/wall:1:2"
        other_surface = "level:2/room:5/wall:2:3"
        current = _surface_assignment_with_variants(
            asset_directory,
            "current-wall",
            (current_surface,),
        )
        other = _surface_assignment_with_variants(
            asset_directory,
            "other-wall",
            (other_surface,),
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(current_surface,),
                assignments=[current, other],
            )
        )
        item = _other_texture_item(self.workspace, "other-wall")
        self.workspace.other_texture_list.setCurrentItem(item)
        item.setSelected(True)

        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Cancel,
        ) as question:
            QTest.keyClick(
                self.workspace.other_texture_list,
                Qt.Key.Key_Delete,
            )
        self.assertEqual(len(self.workspace.get_data().assignments), 2)
        self.assertIn("other-wall", question.call_args.args[2])

        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            QTest.keyClick(
                self.workspace.other_texture_list,
                Qt.Key.Key_Delete,
            )

        self.assertEqual(
            [
                assignment.assignment_id
                for assignment in self.workspace.get_data().assignments
            ],
            ["current-wall"],
        )
        self.assertEqual(
            self.workspace.texture_view.selected_atlas_id,
            "current-wall:resolution:1024",
        )

    def test_surface_change_clears_stale_other_texture_deletion_target(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        surfaces = (
            "level:2/room:5/wall:1:2",
            "level:2/room:5/wall:2:3",
            "level:2/room:5/wall:3:4",
        )
        assignments = [
            _surface_assignment_with_variants(
                asset_directory,
                assignment_id,
                (surface_id,),
            )
            for assignment_id, surface_id in zip(
                ("first-wall", "stale-other", "latest-wall"),
                surfaces,
                strict=True,
            )
        ]
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(surfaces[0],),
                assignments=assignments,
            )
        )
        stale_item = _other_texture_item(self.workspace, "stale-other")
        self.workspace.other_texture_list.setCurrentItem(stale_item)
        stale_item.setSelected(True)
        self.assertEqual(
            self.workspace._selected_texture_assignment_id_for_deletion(),
            "stale-other",
        )

        self.workspace.surface_view.set_selected_surface_ids((surfaces[2],))

        self.assertFalse(self.workspace.other_texture_list.selectedItems())
        self.assertEqual(
            self.workspace._selected_texture_assignment_id_for_deletion(),
            "latest-wall",
        )

    def test_delete_legacy_texture_keeps_shared_asset_and_overlap_mask(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        asset_directory.mkdir(parents=True, exist_ok=True)
        first_surface = "level:2/room:5/wall:1:2"
        shared_surface = "level:2/room:5/wall:2:3"
        shared_path = "shared-legacy.png"
        Image.new("RGBA", (13, 9), (70, 90, 210, 255)).save(
            asset_directory / shared_path,
            format="PNG",
        )
        deleted = _surface_assignment(
            "legacy-delete",
            (first_surface, shared_surface),
            shared_path,
        )
        retained = _surface_assignment(
            "legacy-retained",
            (shared_surface,),
            shared_path,
        )
        first_stroke = _test_stroke(0.25)
        shared_stroke = _test_stroke(0.75)
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(first_surface,),
                assignments=[deleted, retained],
                texture_mask_strokes={
                    first_surface: [first_stroke],
                    shared_surface: [shared_stroke],
                },
            )
        )
        self.assertEqual(self.workspace.texture_view.entries, ())
        self.assertTrue(self.workspace.delete_texture_button.isEnabled())

        self.assertTrue(
            self.workspace.delete_assignment_texture("legacy-delete")
        )

        self.assertTrue((asset_directory / shared_path).is_file())
        self.assertEqual(
            self.workspace.get_data().texture_mask_strokes,
            {shared_surface: [shared_stroke]},
        )
        self.assertIsNone(
            self.workspace.surface_view.get_surface_texture_rgba(first_surface)
        )
        self.assertIsNotNone(
            self.workspace.surface_view.get_surface_texture_rgba(shared_surface)
        )

        self.assertTrue(
            self.workspace.delete_assignment_texture("legacy-retained")
        )
        self.assertFalse((asset_directory / shared_path).exists())

    def test_delete_texture_rolls_back_viewer_failure_and_gates_invalid_calls(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        surface_id = "level:2/room:5/wall:1:2"
        assignment = _surface_assignment_with_variants(
            asset_directory,
            "rollback-wall",
            (surface_id,),
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(surface_id,),
                assignments=[assignment],
            )
        )
        changed = QSignalSpy(self.workspace.data_changed)
        removed = QSignalSpy(self.workspace.assignments_removed)

        with patch.object(
            self.workspace.surface_view,
            "clear_surface_textures",
            side_effect=[ValueError("viewer failed"), None],
        ):
            self.assertFalse(
                self.workspace.delete_assignment_texture("rollback-wall")
            )

        self.assertEqual(self.workspace.get_data().assignments, [assignment])
        self.assertEqual(changed.count(), 0)
        self.assertEqual(removed.count(), 0)
        self.assertIsNotNone(
            self.workspace.surface_view.get_surface_texture_rgba(surface_id)
        )
        self.assertTrue(
            all(
                (
                    asset_directory
                    / f"rollback-wall.texture-{resolution}.png"
                ).is_file()
                for resolution in SURFACE_TEXTURE_RESOLUTIONS
            )
        )
        self.assertFalse(
            self.workspace.delete_assignment_texture("missing-family")
        )
        with patch.object(
            SurfaceTextureGenerationWorkspace,
            "is_generating",
            property(lambda _workspace: True),
        ):
            self.assertFalse(
                self.workspace.delete_assignment_texture("rollback-wall")
            )

        self.workspace.surface_view.set_selected_surface_ids(())
        self.assertFalse(self.workspace.delete_texture_button.isEnabled())

    def test_delete_texture_asset_cleanup_failure_is_nonfatal(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        asset_directory.mkdir(parents=True, exist_ok=True)
        surface_id = "level:2/room:5/wall:1:2"
        asset_path = "locked-legacy.png"
        Image.new("RGBA", (13, 9), (70, 90, 210, 255)).save(
            asset_directory / asset_path,
            format="PNG",
        )
        assignment = _surface_assignment(
            "locked-legacy",
            (surface_id,),
            asset_path,
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(surface_id,),
                assignments=[assignment],
            )
        )

        with patch.object(Path, "unlink", side_effect=PermissionError("locked")):
            self.assertTrue(
                self.workspace.delete_assignment_texture("locked-legacy")
            )

        self.assertEqual(self.workspace.get_data().assignments, [])
        self.assertTrue((asset_directory / asset_path).is_file())
        self.assertIn(
            "1 unused texture file(s) could not be deleted",
            self.workspace.status_label.text(),
        )

    def test_double_clicked_other_texture_uses_assignment_transaction(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        target_surface = "level:2/room:5/wall:1:2"
        source_surface = "level:2/room:5/wall:2:3"
        replaced_assignment = _surface_assignment_with_variants(
            asset_directory,
            "replaced-plaster",
            (target_surface,),
            color=(180, 40, 30, 255),
        )
        reusable_assignment = _surface_assignment_with_variants(
            asset_directory,
            "reusable-brick",
            (source_surface,),
            color=(20, 170, 90, 255),
        )
        target_stroke = _test_stroke(0.25)
        source_stroke = _test_stroke(0.75)
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(target_surface,),
                texture_mask_strokes={
                    target_surface: [target_stroke],
                    source_surface: [source_stroke],
                },
                assignments=[replaced_assignment, reusable_assignment],
                localized_inpaint_undo_stack=[
                    SurfaceTextureInpaintUndoSnapshot(
                        previous_assignments=(replaced_assignment,),
                        replacement_assignment_ids=("localized-edit",),
                        affected_surface_ids=(target_surface,),
                        previous_texture_mask_strokes={
                            target_surface: (target_stroke,),
                        },
                    )
                ],
            )
        )
        removed = QSignalSpy(self.workspace.assignments_removed)
        changed = QSignalSpy(self.workspace.data_changed)
        content_changed = QSignalSpy(self.workspace.surface_content_changed)
        item = _other_texture_item(self.workspace, "reusable-brick")

        self.workspace.other_texture_list.itemDoubleClicked.emit(item)
        _qt_application.processEvents()

        data = self.workspace.get_data()
        self.assertEqual(len(data.assignments), 1)
        selected_assignment = data.assignments[0]
        self.assertEqual(selected_assignment.assignment_id, "reusable-brick")
        self.assertEqual(
            selected_assignment.surface_ids,
            (source_surface, target_surface),
        )
        self.assertEqual(
            data.texture_mask_strokes,
            {source_surface: [source_stroke]},
        )
        self.assertEqual(data.localized_inpaint_undo_stack, [])
        self.assertEqual(tuple(removed.at(0)[0]), ("replaced-plaster",))
        self.assertEqual(changed.count(), 1)
        self.assertEqual(content_changed.count(), 1)
        target_texture = self.workspace.surface_view.get_surface_texture_rgba(
            target_surface
        )
        self.assertIsNotNone(target_texture)
        self.assertEqual(target_texture.shape[:2], (1024, 1024))
        self.assertEqual(
            self.workspace.texture_view.selected_atlas_id,
            "reusable-brick:resolution:1024",
        )
        self.assertEqual(self.workspace.other_texture_list.count(), 0)
        self.assertTrue(
            all(
                not (
                    asset_directory
                    / f"replaced-plaster.texture-{resolution}.png"
                ).exists()
                for resolution in SURFACE_TEXTURE_RESOLUTIONS
            )
        )

    def test_double_clicked_legacy_texture_preserves_fixed_asset_metadata(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        asset_directory.mkdir(parents=True, exist_ok=True)
        target_surface = "level:2/room:5/wall:1:2"
        source_surface = "level:2/room:5/wall:2:3"
        replaced_assignment = _surface_assignment_with_variants(
            asset_directory,
            "replaced-wall",
            (target_surface,),
        )
        legacy_asset_path = "legacy-stone.png"
        Image.new("RGBA", (11, 6), (25, 140, 190, 255)).save(
            asset_directory / legacy_asset_path,
            format="PNG",
        )
        legacy_assignment = _surface_assignment(
            "legacy-stone",
            (source_surface,),
            legacy_asset_path,
        )
        target_stroke = _test_stroke(0.25)
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(target_surface,),
                texture_mask_strokes={target_surface: [target_stroke]},
                assignments=[replaced_assignment, legacy_assignment],
                localized_inpaint_undo_stack=[
                    SurfaceTextureInpaintUndoSnapshot(
                        previous_assignments=(replaced_assignment,),
                        replacement_assignment_ids=("localized-edit",),
                        affected_surface_ids=(target_surface,),
                        previous_texture_mask_strokes={
                            target_surface: (target_stroke,),
                        },
                    )
                ],
            )
        )
        removed = QSignalSpy(self.workspace.assignments_removed)
        changed = QSignalSpy(self.workspace.data_changed)
        content_changed = QSignalSpy(self.workspace.surface_content_changed)
        item = _other_texture_item(self.workspace, "legacy-stone")

        self.workspace.other_texture_list.itemDoubleClicked.emit(item)
        _qt_application.processEvents()

        data = self.workspace.get_data()
        self.assertEqual(len(data.assignments), 1)
        applied = data.assignments[0]
        self.assertEqual(applied.assignment_id, "legacy-stone")
        self.assertEqual(
            applied.surface_ids,
            (source_surface, target_surface),
        )
        self.assertEqual(applied.asset_path, legacy_asset_path)
        self.assertEqual(applied.texture_variants, ())
        self.assertIsNone(applied.selected_texture_resolution)
        self.assertIsNone(applied.texture_width)
        self.assertIsNone(applied.texture_height)
        self.assertEqual(data.texture_mask_strokes, {})
        self.assertEqual(data.localized_inpaint_undo_stack, [])
        self.assertEqual(tuple(removed.at(0)[0]), ("replaced-wall",))
        self.assertEqual(changed.count(), 1)
        self.assertEqual(content_changed.count(), 1)
        texture = self.workspace.surface_view.get_surface_texture_rgba(
            target_surface
        )
        self.assertIsNotNone(texture)
        self.assertEqual(texture.shape[:2], (6, 11))
        self.assertEqual(self.workspace.texture_view.entries, ())
        self.assertEqual(self.workspace.other_texture_list.count(), 0)
        self.assertTrue((asset_directory / legacy_asset_path).is_file())
        self.assertTrue(
            all(
                not (
                    asset_directory
                    / f"replaced-wall.texture-{resolution}.png"
                ).exists()
                for resolution in SURFACE_TEXTURE_RESOLUTIONS
            )
        )

    def test_other_texture_busy_and_missing_asset_are_no_ops(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        target_surface = "level:2/room:5/wall:1:2"
        source_surface = "level:2/room:5/wall:2:3"
        target_assignment = _surface_assignment_with_variants(
            asset_directory,
            "target-wall",
            (target_surface,),
        )
        source_assignment = _surface_assignment_with_variants(
            asset_directory,
            "source-wall",
            (source_surface,),
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(target_surface,),
                assignments=[target_assignment, source_assignment],
            )
        )
        item = _other_texture_item(self.workspace, "source-wall")
        before = self.workspace.get_data().to_dict()
        changed = QSignalSpy(self.workspace.data_changed)

        with patch.object(
            SurfaceTextureGenerationWorkspace,
            "is_generating",
            property(lambda _workspace: True),
        ):
            self.workspace._handle_other_texture_activated(item)

        self.assertEqual(self.workspace.get_data().to_dict(), before)
        self.assertEqual(changed.count(), 0)

        (asset_directory / source_assignment.asset_path).unlink()
        self.workspace._handle_other_texture_activated(item)

        self.assertEqual(self.workspace.get_data().to_dict(), before)
        self.assertEqual(changed.count(), 0)
        self.assertIn("could not be applied", self.workspace.status_label.text())

    def test_other_texture_decode_error_is_a_no_op(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        target_surface = "level:2/room:5/wall:1:2"
        source_surface = "level:2/room:5/wall:2:3"
        target_assignment = _surface_assignment_with_variants(
            asset_directory,
            "target-wall",
            (target_surface,),
        )
        source_assignment = _surface_assignment_with_variants(
            asset_directory,
            "source-wall",
            (source_surface,),
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(target_surface,),
                assignments=[target_assignment, source_assignment],
            )
        )
        item = _other_texture_item(self.workspace, "source-wall")
        before = self.workspace.get_data().to_dict()
        changed = QSignalSpy(self.workspace.data_changed)

        with patch(
            "housemaker.surface_texture_workspace._decode_png_rgba",
            side_effect=ValueError("invalid texture"),
        ):
            self.workspace._handle_other_texture_activated(item)

        self.assertEqual(self.workspace.get_data().to_dict(), before)
        self.assertEqual(changed.count(), 0)
        self.assertIn("could not be applied", self.workspace.status_label.text())

    def test_double_clicked_variant_applies_family_to_selected_surfaces(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        source_surface_id = "level:2/room:5/wall:1:2"
        target_surface_id = "level:2/room:5/wall:2:3"
        target_assignment = _surface_assignment_with_variants(
            asset_directory,
            "old-plaster",
            (target_surface_id,),
            color=(180, 40, 30, 255),
        )
        source_assignment = _surface_assignment_with_variants(
            asset_directory,
            "new-brick",
            (source_surface_id,),
            color=(20, 170, 90, 255),
        )
        source_stroke = _test_stroke(0.25)
        target_stroke = _test_stroke(0.75)
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(source_surface_id, target_surface_id),
                texture_mask_strokes={
                    source_surface_id: [source_stroke],
                    target_surface_id: [target_stroke],
                },
                assignments=[target_assignment, source_assignment],
                localized_inpaint_undo_stack=[
                    SurfaceTextureInpaintUndoSnapshot(
                        previous_assignments=(target_assignment,),
                        replacement_assignment_ids=(
                            source_assignment.assignment_id,
                        ),
                        affected_surface_ids=(target_surface_id,),
                        previous_texture_mask_strokes={
                            target_surface_id: (target_stroke,),
                        },
                    )
                ],
            )
        )
        removed = QSignalSpy(self.workspace.assignments_removed)
        target_entry = next(
            entry
            for entry in self.workspace.texture_view.entries
            if entry.atlas_id == "new-brick:resolution:2048"
        )

        self.workspace.texture_view.atlas_activated.emit(target_entry)
        _qt_application.processEvents()

        data = self.workspace.get_data()
        self.assertEqual(len(data.assignments), 1)
        selected_assignment = data.assignments[0]
        self.assertEqual(selected_assignment.assignment_id, "new-brick")
        self.assertEqual(
            selected_assignment.surface_ids,
            (source_surface_id, target_surface_id),
        )
        self.assertEqual(selected_assignment.selected_texture_resolution, 2048)
        self.assertEqual(
            selected_assignment.asset_path,
            "new-brick.texture-2048.png",
        )
        self.assertEqual(
            data.texture_mask_strokes,
            {source_surface_id: [source_stroke]},
        )
        self.assertEqual(data.localized_inpaint_undo_stack, [])
        self.assertEqual(tuple(removed.at(0)[0]), ("old-plaster",))
        self.assertEqual(
            self.workspace.surface_view.get_surface_texture_rgba(
                target_surface_id
            ).shape[:2],
            (2048, 2048),
        )
        self.assertEqual(
            self.workspace.texture_view.selected_atlas_id,
            "new-brick:resolution:2048",
        )
        self.assertEqual(len(self.workspace.texture_view.entries), 3)
        self.assertTrue(
            all(
                not (
                    asset_directory
                    / f"old-plaster.texture-{resolution}.png"
                ).exists()
                for resolution in SURFACE_TEXTURE_RESOLUTIONS
            )
        )

    def test_multiframe_masks_build_request_with_selected_surface_area(self) -> None:
        video_path = self._temporary_path / "walkthrough.avi"
        _write_test_video(video_path)
        self.workspace.load_video(str(video_path))
        _set_current_mask(self.workspace, _test_stroke(0.25))
        self.workspace.show_frame(2)
        _set_current_mask(self.workspace, _test_stroke(0.75))
        self.workspace.show_frame(0)

        self.assertEqual(
            self.workspace.video_view.get_strokes(),
            [_test_stroke(0.25)],
        )
        self.workspace.surface_view.select_surface("level:2/room:5/floor")
        self.workspace.material_notes_edit.setText("pale oak")
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(
                openai_api_key="openai-test-key",
                surface_texture_provider=SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA,
            )
        )

        request = self.workspace._build_request()

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.provider, SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA)
        self.assertEqual(request.api_key, "openai-test-key")
        self.assertEqual(request.reference_frame_indices, (0, 2))
        self.assertEqual(len(request.reference_pngs), 2)
        self.assertTrue(all(image.startswith(b"\x89PNG") for image in request.reference_pngs))
        self.assertEqual(request.surface_type, "floor")
        self.assertEqual(request.surface_ids, ("level:2/room:5/floor",))
        self.assertAlmostEqual(request.combined_area_m2, 4.0)
        self.assertIn("pale oak", request.prompt)

    def test_3d_inpaint_request_targets_only_painted_surfaces(self) -> None:
        video_path = self._temporary_path / "partial-walkthrough.avi"
        _write_test_video(video_path, frame_count=1)
        self.workspace.load_video(str(video_path))
        _set_current_mask(self.workspace, _test_stroke())
        first_wall = "level:2/room:5/wall:1:2"
        second_wall = "level:2/room:5/wall:2:3"
        texture = np.full((32, 32, 4), (15, 35, 75, 255), dtype=np.uint8)
        self.workspace.surface_view.set_surface_texture(
            (first_wall, second_wall),
            texture,
        )
        self.workspace.surface_view.set_selected_surface_ids(
            (first_wall, second_wall)
        )
        self.workspace.surface_view.add_texture_mask_stroke(
            first_wall,
            _test_stroke(0.25),
        )
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(
                openai_api_key="openai-test-key",
                surface_texture_provider=SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA,
            )
        )

        request = self.workspace._build_request()

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.surface_ids, (first_wall,))
        self.assertIsNotNone(request.existing_texture_png)
        self.assertIsNotNone(request.edit_mask_png)
        self.assertEqual(
            [surface_id for surface_id, _mask in request.surface_edit_mask_pngs],
            [first_wall],
        )
        self.assertAlmostEqual(request.combined_area_m2, 6.0)

    def test_localized_inpaint_uses_2048_base_and_preserves_active_resolution(
        self,
    ) -> None:
        surface_id = "level:2/room:5/wall:1:2"
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(
                openai_api_key="openai-test-key",
                surface_texture_provider=SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA,
            )
        )
        for selected_resolution in (512, 2048):
            with self.subTest(selected_resolution=selected_resolution):
                assignment = _surface_assignment_with_variants(
                    self._temporary_path / "surface_assets",
                    f"prior-{selected_resolution}",
                    (surface_id,),
                    selected_resolution=selected_resolution,
                    color=(40, 90, 180, 255),
                )
                texture_stroke = _test_stroke(0.4)
                self.workspace.set_data(
                    SurfaceTextureData(
                        selected_surface_type="wall",
                        selected_surface_ids=(surface_id,),
                        texture_mask_strokes={surface_id: [texture_stroke]},
                        assignments=[assignment],
                    )
                )
                video_path = (
                    self._temporary_path
                    / f"inpaint-{selected_resolution}.avi"
                )
                _write_test_video(video_path, frame_count=1)
                self.workspace.load_video(str(video_path))
                _set_current_mask(self.workspace, _test_stroke(0.2))

                request = self.workspace._build_request()

                self.assertIsNotNone(request)
                assert request is not None
                self.assertIsNotNone(request.existing_texture_png)
                self.assertIsNotNone(request.edit_mask_png)
                assert request.existing_texture_png is not None
                assert request.edit_mask_png is not None
                self.assertEqual(
                    _decode_png_rgba(
                        request.existing_texture_png,
                        "Canonical inpaint base",
                    ).shape[:2],
                    (2048, 2048),
                )
                self.assertEqual(
                    _decode_png_rgba(
                        request.edit_mask_png,
                        "Canonical inpaint mask",
                    ).shape[:2],
                    (2048, 2048),
                )

                self.workspace._handle_generation_succeeded(
                    request,
                    SurfaceTextureResult(
                        provider="meshy",
                        texture_png=_colored_texture_png(
                            (20, 170, 90, 255)
                        ),
                        task_id=f"inpaint-{selected_resolution}",
                    ),
                )

                generated_data = self.workspace.get_data()
                self.assertEqual(len(generated_data.assignments), 1)
                replacement = generated_data.assignments[0]
                self.assertEqual(
                    replacement.selected_texture_resolution,
                    selected_resolution,
                )
                self.assertEqual(
                    self.workspace.surface_view.get_surface_texture_rgba(
                        surface_id
                    ).shape[:2],
                    (selected_resolution, selected_resolution),
                )
                self.assertEqual(
                    len(generated_data.localized_inpaint_undo_stack),
                    1,
                )
                replacement_paths = tuple(
                    self._temporary_path
                    / "surface_assets"
                    / variant.asset_path
                    for variant in replacement.texture_variants
                )

                self.assertTrue(
                    self.workspace.undo_localized_texture_inpaint()
                )

                restored = self.workspace.get_data()
                self.assertEqual(restored.assignments, [assignment])
                self.assertEqual(restored.localized_inpaint_undo_stack, [])
                self.assertEqual(
                    restored.assignments[0].selected_texture_resolution,
                    selected_resolution,
                )
                self.assertEqual(
                    self.workspace.surface_view.get_surface_texture_rgba(
                        surface_id
                    ).shape[:2],
                    (selected_resolution, selected_resolution),
                )
                self.assertTrue(
                    all(
                        (
                            self._temporary_path
                            / "surface_assets"
                            / variant.asset_path
                        ).is_file()
                        for variant in assignment.texture_variants
                    )
                )
                self.assertTrue(
                    all(not path.exists() for path in replacement_paths)
                )

    def test_async_result_is_persisted_applied_and_restored(self) -> None:
        video_path = self._temporary_path / "walkthrough.avi"
        _write_test_video(video_path, frame_count=1)
        self.workspace.load_video(str(video_path))
        _set_current_mask(self.workspace, _test_stroke())
        surface_id = "level:2/room:5/floor"
        self.workspace.surface_view.select_surface(surface_id)
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(
                openai_api_key="openai-test-key",
                surface_texture_provider=SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA,
            )
        )
        completed = QSignalSpy(self.workspace.generation_completed)

        self.workspace.generate()
        deadline = time.monotonic() + 3.0
        while completed.count() == 0 and time.monotonic() < deadline:
            _qt_application.processEvents()
            QTest.qWait(5)
        while self.workspace.is_generating and time.monotonic() < deadline:
            _qt_application.processEvents()
            QTest.qWait(5)

        self.assertEqual(completed.count(), 1)
        self.assertFalse(self.workspace.is_generating)
        self.assertEqual(len(self.provider.requests), 1)
        saved_data = self.workspace.get_data()
        self.assertEqual(len(saved_data.assignments), 1)
        assignment = saved_data.assignments[0]
        self.assertEqual(assignment.surface_ids, (surface_id,))
        self.assertEqual(assignment.provider_task_id, "texture-task-1")
        self.assertAlmostEqual(assignment.combined_area_m2, 4.0)
        self.assertEqual(assignment.reference_frame_indices, (0,))
        self.assertEqual(
            (assignment.texture_width, assignment.texture_height),
            (1024, 1024),
        )
        self.assertEqual(assignment.selected_texture_resolution, 1024)
        self.assertEqual(
            [variant.resolution for variant in assignment.texture_variants],
            list(SURFACE_TEXTURE_RESOLUTIONS),
        )
        self.assertEqual(
            assignment.asset_path,
            next(
                variant.asset_path
                for variant in assignment.texture_variants
                if variant.resolution == 1024
            ),
        )
        for variant in assignment.texture_variants:
            with self.subTest(resolution=variant.resolution):
                variant_path = (
                    self._temporary_path
                    / "surface_assets"
                    / variant.asset_path
                )
                self.assertTrue(variant_path.is_file())
                with Image.open(variant_path) as image:
                    self.assertEqual(
                        image.size,
                        (variant.resolution, variant.resolution),
                    )
        self.assertEqual(
            sorted(
                path.name
                for path in (
                    self._temporary_path / "surface_assets"
                ).glob("*.png")
            ),
            sorted(
                variant.asset_path
                for variant in assignment.texture_variants
            ),
        )
        self.assertIn(surface_id, self.workspace.surface_view._surface_textures)

        restored = SurfaceTextureGenerationWorkspace(
            asset_directory=self._temporary_path / "surface_assets"
        )
        try:
            restored.set_levels([_test_level()], self.initial_camera)
            restored.set_data(saved_data)
            self.assertIn(surface_id, restored.surface_view._surface_textures)
        finally:
            restored.shutdown()
            restored.close()

    def test_generation_replaces_fully_covered_assignment_and_deletes_asset(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        asset_directory.mkdir(exist_ok=True)
        old_asset_path = asset_directory / "old-walls.png"
        old_asset_path.write_bytes(_colored_texture_png((180, 40, 30, 255)))
        first_wall = "level:2/room:5/wall:1:2"
        second_wall = "level:2/room:5/wall:2:3"
        self.workspace.set_data(
            SurfaceTextureData(
                assignments=[
                    _surface_assignment(
                        "old-walls",
                        (first_wall, second_wall),
                        old_asset_path.name,
                    )
                ]
            )
        )
        removed = QSignalSpy(self.workspace.assignments_removed)
        completed = QSignalSpy(self.workspace.generation_completed)
        request = SurfaceTextureRequest(
            provider="meshy",
            api_key="test-key",
            reference_pngs=(_texture_png(),),
            reference_frame_indices=(0,),
            surface_type="wall",
            surface_ids=(first_wall, second_wall),
            combined_area_m2=12.0,
            prompt="Replace both walls",
        )

        self.workspace._handle_generation_succeeded(
            request,
            SurfaceTextureResult(
                provider="meshy",
                texture_png=_colored_texture_png((20, 170, 90, 255)),
                task_id="replacement-task",
            ),
        )

        assignments = self.workspace.get_data().assignments
        self.assertEqual(len(assignments), 1)
        self.assertNotEqual(assignments[0].assignment_id, "old-walls")
        self.assertEqual(assignments[0].surface_ids, (first_wall, second_wall))
        self.assertFalse(old_asset_path.exists())
        self.assertEqual(removed.count(), 1)
        self.assertEqual(tuple(removed.at(0)[0]), ("old-walls",))
        self.assertEqual(completed.count(), 1)

    def test_generation_trims_partially_covered_assignment_and_keeps_asset(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        asset_directory.mkdir(exist_ok=True)
        old_asset_path = asset_directory / "shared-walls.png"
        old_asset_path.write_bytes(_colored_texture_png((180, 40, 30, 255)))
        first_wall = "level:2/room:5/wall:1:2"
        second_wall = "level:2/room:5/wall:2:3"
        self.workspace.set_data(
            SurfaceTextureData(
                assignments=[
                    _surface_assignment(
                        "shared-walls",
                        (first_wall, second_wall),
                        old_asset_path.name,
                    )
                ]
            )
        )
        removed = QSignalSpy(self.workspace.assignments_removed)
        request = SurfaceTextureRequest(
            provider="meshy",
            api_key="test-key",
            reference_pngs=(_texture_png(),),
            reference_frame_indices=(0,),
            surface_type="wall",
            surface_ids=(first_wall,),
            combined_area_m2=6.0,
            prompt="Replace one wall",
        )

        self.workspace._handle_generation_succeeded(
            request,
            SurfaceTextureResult(
                provider="meshy",
                texture_png=_colored_texture_png((20, 170, 90, 255)),
            ),
        )

        assignments = self.workspace.get_data().assignments
        self.assertEqual(len(assignments), 2)
        retained, replacement = assignments
        self.assertEqual(retained.assignment_id, "shared-walls")
        self.assertEqual(retained.surface_ids, (second_wall,))
        self.assertAlmostEqual(retained.combined_area_m2, 6.0)
        self.assertIn("1 wall surface(s)", retained.area_description)
        self.assertEqual(replacement.surface_ids, (first_wall,))
        self.assertTrue(old_asset_path.exists())
        self.assertEqual(removed.count(), 0)

    def test_localized_inpaint_undo_restores_assignment_asset_mask_and_signals(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        asset_directory.mkdir(exist_ok=True)
        old_png = _colored_texture_png((180, 40, 30, 255))
        old_asset_path = asset_directory / "old-wall.png"
        old_asset_path.write_bytes(old_png)
        wall_id = "level:2/room:5/wall:1:2"
        old_assignment = _surface_assignment(
            "old-wall",
            (wall_id,),
            old_asset_path.name,
        )
        original_stroke = _test_stroke(0.3)
        self.workspace.set_data(
            SurfaceTextureData(
                texture_mask_strokes={wall_id: [original_stroke]},
                assignments=[old_assignment],
            )
        )
        mask = np.zeros((7, 10), dtype=np.uint8)
        mask[:, :4] = 255
        request = SurfaceTextureRequest(
            provider="meshy",
            api_key="test-key",
            reference_pngs=(_texture_png(),),
            reference_frame_indices=(0,),
            surface_type="wall",
            surface_ids=(wall_id,),
            combined_area_m2=6.0,
            prompt="Localized edit",
            existing_texture_png=old_png,
            edit_mask_png=_encode_png(mask),
            surface_edit_mask_pngs=((wall_id, _encode_png(mask)),),
        )
        removed = QSignalSpy(self.workspace.assignments_removed)
        undone = QSignalSpy(self.workspace.localized_inpaint_undone)

        self.workspace._handle_generation_succeeded(
            request,
            SurfaceTextureResult(
                provider="meshy",
                texture_png=_colored_texture_png((20, 170, 90, 255)),
                task_id="localized-task",
            ),
        )

        generated_data = self.workspace.get_data()
        self.assertEqual(len(generated_data.assignments), 1)
        replacement = generated_data.assignments[0]
        replacement_path = asset_directory / replacement.asset_path
        self.assertTrue(replacement_path.is_file())
        self.assertTrue(old_asset_path.is_file())
        self.assertEqual(len(generated_data.localized_inpaint_undo_stack), 1)
        self.assertEqual(removed.count(), 0)
        self.assertTrue(self.workspace.undo_inpaint_button.isEnabled())
        self.assertEqual(
            SurfaceTextureData.from_dict(generated_data.to_dict()),
            generated_data,
        )

        self.assertTrue(self.workspace.undo_localized_texture_inpaint())

        restored = self.workspace.get_data()
        self.assertEqual(restored.assignments, [old_assignment])
        self.assertEqual(restored.localized_inpaint_undo_stack, [])
        self.assertEqual(
            restored.texture_mask_strokes,
            {wall_id: [original_stroke]},
        )
        self.assertTrue(old_asset_path.is_file())
        self.assertFalse(replacement_path.exists())
        self.assertEqual(removed.count(), 1)
        self.assertEqual(tuple(removed.at(0)[0]), (replacement.assignment_id,))
        self.assertEqual(undone.count(), 1)
        self.assertFalse(self.workspace.undo_inpaint_button.isEnabled())
        np.testing.assert_array_equal(
            self.workspace.surface_view.get_surface_texture_rgba(wall_id),
            _decode_png_rgba(old_png, "Old texture"),
        )

    def test_localized_inpaint_undo_blocks_atomically_when_old_asset_is_missing(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        asset_directory.mkdir(exist_ok=True)
        old_png = _colored_texture_png((180, 40, 30, 255))
        old_asset_path = asset_directory / "old-wall.png"
        old_asset_path.write_bytes(old_png)
        wall_id = "level:2/room:5/wall:1:2"
        old_assignment = _surface_assignment(
            "old-wall",
            (wall_id,),
            old_asset_path.name,
        )
        mask = np.full((7, 10), 255, dtype=np.uint8)
        self.workspace.set_data(SurfaceTextureData(assignments=[old_assignment]))
        request = SurfaceTextureRequest(
            provider="meshy",
            api_key="test-key",
            reference_pngs=(_texture_png(),),
            reference_frame_indices=(0,),
            surface_type="wall",
            surface_ids=(wall_id,),
            combined_area_m2=6.0,
            prompt="Localized edit",
            existing_texture_png=old_png,
            edit_mask_png=_encode_png(mask),
            surface_edit_mask_pngs=((wall_id, _encode_png(mask)),),
        )
        self.workspace._handle_generation_succeeded(
            request,
            SurfaceTextureResult(
                provider="meshy",
                texture_png=_colored_texture_png((20, 170, 90, 255)),
            ),
        )
        generated_data = self.workspace.get_data()
        old_asset_path.unlink()
        removed = QSignalSpy(self.workspace.assignments_removed)

        self.assertFalse(self.workspace.undo_localized_texture_inpaint())

        self.assertEqual(self.workspace.get_data(), generated_data)
        self.assertEqual(removed.count(), 0)
        self.assertIn("missing", self.workspace.status_label.text())

    def test_full_generation_invalidates_localized_inpaint_history(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        asset_directory.mkdir(exist_ok=True)
        old_png = _colored_texture_png((180, 40, 30, 255))
        old_asset_path = asset_directory / "old-wall.png"
        old_asset_path.write_bytes(old_png)
        wall_id = "level:2/room:5/wall:1:2"
        old_assignment = _surface_assignment(
            "old-wall",
            (wall_id,),
            old_asset_path.name,
        )
        self.workspace.set_data(SurfaceTextureData(assignments=[old_assignment]))
        mask = np.full((7, 10), 255, dtype=np.uint8)
        localized_request = SurfaceTextureRequest(
            provider="meshy",
            api_key="test-key",
            reference_pngs=(_texture_png(),),
            reference_frame_indices=(0,),
            surface_type="wall",
            surface_ids=(wall_id,),
            combined_area_m2=6.0,
            prompt="Localized edit",
            existing_texture_png=old_png,
            edit_mask_png=_encode_png(mask),
            surface_edit_mask_pngs=((wall_id, _encode_png(mask)),),
        )
        self.workspace._handle_generation_succeeded(
            localized_request,
            SurfaceTextureResult(
                provider="meshy",
                texture_png=_colored_texture_png((20, 170, 90, 255)),
            ),
        )
        localized_assignment = self.workspace.get_data().assignments[0]
        removed = QSignalSpy(self.workspace.assignments_removed)
        full_request = SurfaceTextureRequest(
            provider="meshy",
            api_key="test-key",
            reference_pngs=(_texture_png(),),
            reference_frame_indices=(0,),
            surface_type="wall",
            surface_ids=(wall_id,),
            combined_area_m2=6.0,
            prompt="Full replacement",
        )

        self.workspace._handle_generation_succeeded(
            full_request,
            SurfaceTextureResult(
                provider="meshy",
                texture_png=_colored_texture_png((30, 80, 210, 255)),
            ),
        )

        self.assertEqual(
            self.workspace.get_data().localized_inpaint_undo_stack,
            [],
        )
        self.assertFalse(self.workspace.undo_localized_texture_inpaint())
        removed_ids = {
            str(value)
            for signal_arguments in (
                removed.at(index) for index in range(removed.count())
            )
            for value in signal_arguments[0]
        }
        self.assertEqual(
            removed_ids,
            {old_assignment.assignment_id, localized_assignment.assignment_id},
        )
        self.assertFalse(old_asset_path.exists())

    def test_removed_assignment_asset_is_kept_while_another_record_uses_it(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        asset_directory.mkdir(exist_ok=True)
        shared_asset_path = asset_directory / "shared.png"
        shared_asset_path.write_bytes(_colored_texture_png((180, 40, 30, 255)))
        first_wall = "level:2/room:5/wall:1:2"
        second_wall = "level:2/room:5/wall:2:3"
        self.workspace.set_data(
            SurfaceTextureData(
                assignments=[
                    _surface_assignment(
                        "first-record",
                        (first_wall,),
                        shared_asset_path.name,
                    ),
                    _surface_assignment(
                        "second-record",
                        (second_wall,),
                        shared_asset_path.name,
                    ),
                ]
            )
        )
        removed = QSignalSpy(self.workspace.assignments_removed)
        request = SurfaceTextureRequest(
            provider="meshy",
            api_key="test-key",
            reference_pngs=(_texture_png(),),
            reference_frame_indices=(0,),
            surface_type="wall",
            surface_ids=(first_wall,),
            combined_area_m2=6.0,
            prompt="Replace one record",
        )

        self.workspace._handle_generation_succeeded(
            request,
            SurfaceTextureResult(
                provider="meshy",
                texture_png=_colored_texture_png((20, 170, 90, 255)),
            ),
        )

        assignments = self.workspace.get_data().assignments
        self.assertEqual(
            [assignment.assignment_id for assignment in assignments[:-1]],
            ["second-record"],
        )
        self.assertTrue(shared_asset_path.exists())
        self.assertEqual(tuple(removed.at(0)[0]), ("first-record",))

    @unittest.skipUnless(os.name == "nt", "Windows path casing regression")
    def test_orphan_cleanup_keeps_case_aliased_active_asset(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        asset_directory.mkdir(exist_ok=True)
        shared_asset_path = asset_directory / "Shared.PNG"
        shared_asset_path.write_bytes(_colored_texture_png((180, 40, 30, 255)))
        first_wall = "level:2/room:5/wall:1:2"
        second_wall = "level:2/room:5/wall:2:3"
        retained = _surface_assignment(
            "retained",
            (second_wall,),
            "shared.png",
        )
        removed = _surface_assignment(
            "removed",
            (first_wall,),
            "Shared.PNG",
        )
        self.workspace.set_data(SurfaceTextureData(assignments=[retained]))

        failure_count = self.workspace._delete_orphaned_assignment_assets(
            [removed]
        )

        self.assertEqual(failure_count, 0)
        self.assertTrue(shared_asset_path.exists())

    def test_multi_output_apply_failure_restores_assignments_and_assets(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        asset_directory.mkdir(exist_ok=True)
        old_texture = _colored_texture_png((180, 40, 30, 255))
        old_asset_path = asset_directory / "old.png"
        old_asset_path.write_bytes(old_texture)
        first_wall = "level:2/room:5/wall:1:2"
        second_wall = "level:2/room:5/wall:2:3"
        original_assignment = _surface_assignment(
            "old",
            (first_wall, second_wall),
            old_asset_path.name,
        )
        self.workspace.set_data(
            SurfaceTextureData(assignments=[original_assignment])
        )
        first_texture = _colored_texture_png((20, 170, 90, 255))
        second_texture = _colored_texture_png((30, 80, 210, 255))
        request = SurfaceTextureRequest(
            provider="meshy",
            api_key="test-key",
            reference_pngs=(_texture_png(),),
            reference_frame_indices=(0,),
            surface_type="wall",
            surface_ids=(first_wall, second_wall),
            combined_area_m2=12.0,
            prompt="Replace both walls",
        )
        original_apply = self.workspace.surface_view.set_surface_texture
        apply_count = 0

        def apply_with_second_failure(
            surface_ids: tuple[str, ...],
            texture: object,
        ) -> None:
            nonlocal apply_count
            apply_count += 1
            if apply_count == 2:
                raise ValueError("simulated second output failure")
            original_apply(surface_ids, texture)  # type: ignore[arg-type]

        with (
            patch(
                "housemaker.surface_texture_workspace."
                "_build_surface_texture_outputs",
                return_value=[
                    ((first_wall,), first_texture),
                    ((second_wall,), second_texture),
                ],
            ),
            patch.object(
                self.workspace.surface_view,
                "set_surface_texture",
                side_effect=apply_with_second_failure,
            ),
        ):
            self.workspace._handle_generation_succeeded(
                request,
                SurfaceTextureResult(
                    provider="meshy",
                    texture_png=first_texture,
                ),
            )

        self.assertEqual(
            self.workspace.get_data().assignments,
            [original_assignment],
        )
        self.assertEqual(
            sorted(path.name for path in asset_directory.iterdir()),
            [old_asset_path.name],
        )
        expected = _decode_png_rgba(old_texture, "Old texture")
        np.testing.assert_array_equal(
            self.workspace.surface_view.get_surface_texture_rgba(first_wall),
            expected,
        )
        np.testing.assert_array_equal(
            self.workspace.surface_view.get_surface_texture_rgba(second_wall),
            expected,
        )
        self.assertIn("could not be applied", self.workspace.status_label.text())

    def test_legacy_room_index_assignment_is_not_retargeted_on_restore(self) -> None:
        asset_path = self._temporary_path / "surface_assets" / "legacy.png"
        asset_path.parent.mkdir(parents=True)
        asset_path.write_bytes(_texture_png())
        legacy_assignment = SurfaceTextureAssignment(
            assignment_id="legacy-floor",
            surface_type="floor",
            surface_ids=("level:2/room:0/floor",),
            provider="meshy",
            asset_path=asset_path.name,
        )

        self.workspace.set_data(
            SurfaceTextureData(assignments=[legacy_assignment])
        )

        self.assertEqual(self.workspace.surface_view._surface_textures, {})
        self.assertNotIn(
            "level:2/room:5/floor",
            self.workspace.surface_view._surface_textures,
        )

    def test_shutdown_discards_inflight_result_and_busy_state_blocks_replacement(self) -> None:
        blocker = _BlockingProvider()
        self.workspace.set_provider(blocker)
        request = SurfaceTextureRequest(
            provider="meshy",
            api_key="meshy-test-key",
            reference_pngs=(_texture_png(),),
            reference_frame_indices=(0,),
            surface_type="floor",
            surface_ids=("level:2/room:5/floor",),
            combined_area_m2=4.0,
            prompt="Test floor material",
        )
        completed = QSignalSpy(self.workspace.generation_completed)

        self.workspace._start_generation(request)
        self.assertTrue(blocker.started.wait(timeout=1.0))
        active_thread = self.workspace._generation_thread
        self.assertIsNotNone(active_thread)
        with self.assertRaises(RuntimeError):
            self.workspace.set_data(SurfaceTextureData())
        with self.assertRaises(RuntimeError):
            self.workspace.load_video(str(self._temporary_path / "unused.avi"))

        self.workspace.shutdown()
        blocker.release.set()
        assert active_thread is not None
        self.assertTrue(active_thread.wait(2_000))
        _qt_application.processEvents()
        self.assertEqual(completed.count(), 0)
        self.assertEqual(self.workspace.get_data().assignments, [])


# ### Reference crop tests ###
class SurfaceTextureReferenceCropTests(unittest.TestCase):
    def test_pixels_outside_the_painted_mask_are_fully_transparent_black(self) -> None:
        frame = np.full((20, 20, 3), (25, 75, 125), dtype=np.uint8)

        crop = _build_masked_crop(frame, [_test_stroke()])

        self.assertGreater(crop.size, 0)
        transparent_pixels = crop[:, :, 3] == 0
        self.assertTrue(np.any(transparent_pixels))
        self.assertTrue(np.all(crop[transparent_pixels, :3] == 0))

    def test_partial_output_composites_each_surface_with_its_own_mask(self) -> None:
        base = np.full((8, 8, 4), (10, 20, 30, 255), dtype=np.uint8)
        generated = np.full((8, 8, 4), (200, 150, 100, 255), dtype=np.uint8)
        left_mask = np.zeros((8, 8), dtype=np.uint8)
        left_mask[:, :2] = 255
        right_mask = np.zeros((8, 8), dtype=np.uint8)
        right_mask[:, -2:] = 255
        request = SurfaceTextureRequest(
            provider="meshy",
            api_key="test",
            reference_pngs=(_texture_png(),),
            reference_frame_indices=(0,),
            surface_type="wall",
            surface_ids=("wall-one", "wall-two"),
            combined_area_m2=12.0,
            prompt="Test",
            existing_texture_png=_encode_rgba_png(base),
            edit_mask_png=_encode_png(np.maximum(left_mask, right_mask)),
            surface_edit_mask_pngs=(
                ("wall-one", _encode_png(left_mask)),
                ("wall-two", _encode_png(right_mask)),
            ),
        )

        outputs = _build_surface_texture_outputs(
            request,
            _encode_rgba_png(generated),
        )

        self.assertEqual([surface_ids for surface_ids, _png in outputs], [
            ("wall-one",),
            ("wall-two",),
        ])
        first = _decode_png_rgba(outputs[0][1], "First output")
        second = _decode_png_rgba(outputs[1][1], "Second output")
        np.testing.assert_array_equal(first[:, :2], generated[:, :2])
        np.testing.assert_array_equal(first[:, 2:], base[:, 2:])
        np.testing.assert_array_equal(second[:, -2:], generated[:, -2:])
        np.testing.assert_array_equal(second[:, :-2], base[:, :-2])


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
