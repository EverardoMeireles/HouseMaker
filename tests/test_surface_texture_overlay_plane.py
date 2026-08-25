# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from PIL import Image
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.glb import convert_to_glb
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, RoomData, VertexData
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.surface_geometry import (
    DEFAULT_SURFACE_OVERLAY_OFFSET_METERS,
    SURFACE_TYPE_WALL,
    FixedSurface,
    build_fixed_surfaces,
    build_surface_overlay_plane,
    get_surface_overlay_offset_toward_point,
)
from housemaker.surface_texture_state import (
    SurfaceTextureAssignment,
    SurfaceTextureData,
    SurfaceTextureOverlayPlane,
)
from housemaker.surface_texture_viewer import SurfaceTextureViewer
from housemaker.surface_texture_workspace import (
    SurfaceTextureGenerationWorkspace,
)
from housemaker.texture_atlas_workspace import (
    build_atlas_wall_texture_source_id,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_level() -> LevelData:
    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
    ):
        vertex_data.add_edge(start_id, end_id)
    center = vertex_data.add_vertex(50.0, 50.0)
    room = RoomData(
        name="Room",
        vertex_ids=boundary_ids,
        center_vertex_id=center.id,
        color_rgb=(120, 150, 180),
    )
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[room],
        image_size_pixels=(100.0, 100.0),
        floor_contour_vertex_ids=boundary_ids,
    )


def _wall_surface() -> FixedSurface:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 0.0, 3.0),
            (0.0, 0.0, 3.0),
            (2.0, 0.0, 1.0),
            (2.0, 0.2, 1.0),
            (2.0, 0.2, 2.0),
            (2.0, 0.0, 2.0),
        ),
        dtype=float,
    )
    faces = np.asarray(
        ((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)),
        dtype=np.int64,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return FixedSurface(
        surface_id="level:2/wall:1:2",
        surface_type=SURFACE_TYPE_WALL,
        level_index=2,
        room_index=None,
        wall_key="1:2",
        mesh=mesh,
        area_square_meters=float(mesh.area),
    )


def _texture_png() -> bytes:
    output = BytesIO()
    Image.new("RGBA", (8, 8), (80, 130, 200, 255)).save(
        output,
        format="PNG",
    )
    return output.getvalue()


# ### Geometry and viewer tests ###
class SurfaceTextureOverlayGeometryTests(unittest.TestCase):
    def test_overlay_uses_dominant_plane_and_is_picked_before_parent(self) -> None:
        parent = _wall_surface()
        offset = get_surface_overlay_offset_toward_point(
            parent,
            (1.0, -2.0, 1.5),
        )
        plane = build_surface_overlay_plane(
            parent,
            f"{parent.surface_id}/overlay:1",
            offset,
        )

        self.assertEqual(len(plane.mesh.faces), 2)
        self.assertEqual(plane.overlay_parent_surface_id, parent.surface_id)
        self.assertAlmostEqual(abs(offset), DEFAULT_SURFACE_OVERLAY_OFFSET_METERS)
        np.testing.assert_allclose(plane.mesh.bounds[:, 1], (-0.003, -0.003))

        viewer = SurfaceTextureViewer()
        try:
            viewer.set_surfaces((parent, plane))
            self.assertEqual(
                viewer.pick_surface_from_ray(
                    (1.0, -2.0, 1.5),
                    (0.0, 1.0, 0.0),
                ),
                plane.surface_id,
            )
        finally:
            viewer.close()
            viewer.deleteLater()
            _qt_application.processEvents()

    def test_negative_offset_is_not_cancelled_during_glb_texturing(self) -> None:
        level = _build_level()
        parent = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        plane_definition = SurfaceTextureOverlayPlane(
            parent_surface_id=parent.surface_id,
            normal_offset_meters=-DEFAULT_SURFACE_OVERLAY_OFFSET_METERS,
        )
        plane = build_surface_overlay_plane(
            parent,
            plane_definition.surface_id,
            plane_definition.normal_offset_meters,
        )

        model = convert_to_glb(
            [level],
            surface_materials={
                parent.surface_id: _texture_png(),
                plane.surface_id: _texture_png(),
            },
            surface_overlay_planes=(plane,),
        )
        preview_by_id = {
            preview.surface_id: preview
            for preview in model.preview_textured_surfaces
        }

        self.assertIn(parent.surface_id, preview_by_id)
        self.assertIn(plane.surface_id, preview_by_id)
        np.testing.assert_allclose(
            preview_by_id[plane.surface_id].mesh.bounds,
            plane.mesh.bounds,
        )
        self.assertFalse(
            np.allclose(
                preview_by_id[parent.surface_id].mesh.bounds,
                preview_by_id[plane.surface_id].mesh.bounds,
            )
        )

    def test_existing_positional_stairs_argument_keeps_its_meaning(self) -> None:
        level = _build_level()

        model = convert_to_glb([level], 2.5, None, None, 2.0, ())

        self.assertGreater(len(model.mesh.faces), 0)


# ### Persistence tests ###
class SurfaceTextureOverlayPersistenceTests(unittest.TestCase):
    def test_overlay_definition_and_references_round_trip(self) -> None:
        parent_id = "level:2/room:5/wall:1:2"
        plane = SurfaceTextureOverlayPlane(
            parent_surface_id=parent_id,
            normal_offset_meters=-0.003,
        )
        assignment = SurfaceTextureAssignment(
            assignment_id="overlay-material",
            surface_type="wall",
            surface_ids=(plane.surface_id,),
            provider="meshy",
            asset_path="overlay-material.png",
        )
        data = SurfaceTextureData(
            selected_surface_type="wall",
            selected_surface_ids=(plane.surface_id,),
            overlay_planes=[plane],
            assignments=[assignment],
        )

        restored = SurfaceTextureData.from_dict(data.to_dict())

        self.assertEqual(restored, data)
        self.assertEqual(restored.overlay_planes[0].surface_id, plane.surface_id)

    def test_malformed_and_duplicate_overlay_records_are_isolated(self) -> None:
        parent_id = "level:2/wall:1:2"
        valid = SurfaceTextureOverlayPlane(parent_id, 0.003).to_dict()

        restored = SurfaceTextureData.from_dict(
            {
                "overlay_planes": [
                    valid,
                    valid,
                    {"parent_surface_id": parent_id, "normal_offset_meters": 0},
                    {"parent_surface_id": "bad", "normal_offset_meters": 0.003},
                ]
            }
        )

        self.assertEqual(
            restored.overlay_planes,
            [SurfaceTextureOverlayPlane(parent_id, 0.003)],
        )

    def test_existing_positional_assignments_argument_keeps_its_meaning(
        self,
    ) -> None:
        parent_id = "level:2/wall:1:2"
        assignment = SurfaceTextureAssignment(
            assignment_id="legacy-positional",
            surface_type="wall",
            surface_ids=(parent_id,),
            provider="meshy",
            asset_path="legacy-positional.png",
        )

        data = SurfaceTextureData(
            None,
            0,
            {},
            {},
            None,
            "wall",
            (parent_id,),
            [assignment],
        )

        self.assertEqual(data.assignments, [assignment])
        self.assertEqual(data.overlay_planes, [])


# ### Workspace lifecycle tests ###
class SurfaceTextureOverlayWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = Path(self.temporary_directory.name)
        self.level = _build_level()
        self.workspace = SurfaceTextureGenerationWorkspace(
            asset_directory=self.asset_directory
        )
        self.workspace.set_levels([self.level])

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def test_add_plane_selects_it_and_prevents_overlay_chaining(self) -> None:
        parent = next(
            surface
            for surface in self.workspace.surface_view.get_surfaces()
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        content_changes = QSignalSpy(self.workspace.surface_content_changed)
        self.workspace.surface_view.set_selected_surface_ids((parent.surface_id,))

        self.workspace.add_plane_button.click()

        data = self.workspace.get_data()
        self.assertEqual(len(data.overlay_planes), 1)
        plane = data.overlay_planes[0]
        self.assertEqual(
            self.workspace.surface_view.get_selected_surface_ids(),
            (plane.surface_id,),
        )
        self.assertFalse(self.workspace.add_plane_button.isEnabled())
        self.assertTrue(self.workspace.remove_plane_button.isEnabled())
        self.assertEqual(content_changes.count(), 1)

    def test_remove_plane_trims_shared_assignment_and_keeps_live_asset(self) -> None:
        parent = next(
            surface
            for surface in self.workspace.surface_view.get_surfaces()
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        self.workspace.surface_view.set_selected_surface_ids((parent.surface_id,))
        self.workspace.add_plane_button.click()
        plane = self.workspace.get_data().overlay_planes[0]
        texture_path = self.asset_directory / "shared.png"
        texture_path.write_bytes(_texture_png())
        self.workspace._data.assignments = [
            SurfaceTextureAssignment(
                assignment_id="shared",
                surface_type="wall",
                surface_ids=(parent.surface_id, plane.surface_id),
                provider="meshy",
                asset_path=texture_path.name,
            )
        ]
        self.workspace.surface_view.set_surface_texture(
            (parent.surface_id, plane.surface_id),
            texture_path,
        )

        self.workspace.remove_plane_button.click()

        data = self.workspace.get_data()
        self.assertEqual(data.overlay_planes, [])
        self.assertEqual(data.assignments[0].surface_ids, (parent.surface_id,))
        self.assertTrue(texture_path.is_file())
        self.assertEqual(
            self.workspace.surface_view.get_selected_surface_ids(),
            (parent.surface_id,),
        )

    def test_plane_generation_request_uses_plane_identity_type_and_area(
        self,
    ) -> None:
        parent = next(
            surface
            for surface in self.workspace.surface_view.get_surfaces()
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        self.workspace.surface_view.set_selected_surface_ids((parent.surface_id,))
        self.workspace.add_plane_button.click()
        plane = self.workspace.get_data().overlay_planes[0]
        plane_surface = self.workspace.surface_view.get_surface(plane.surface_id)
        assert plane_surface is not None
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(meshy_api_key="meshy-test-key")
        )
        self.workspace._data.frame_strokes = {0: []}

        with patch.object(
            self.workspace,
            "_build_reference_pngs",
            return_value=((_texture_png(),), (0,)),
        ):
            request = self.workspace._build_request()

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.surface_ids, (plane.surface_id,))
        self.assertEqual(request.surface_type, SURFACE_TYPE_WALL)
        self.assertAlmostEqual(
            request.combined_area_m2,
            plane_surface.area_square_meters,
        )

    def test_remove_plane_only_assignment_deletes_asset_and_emits_removal(
        self,
    ) -> None:
        parent = next(
            surface
            for surface in self.workspace.surface_view.get_surfaces()
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        self.workspace.surface_view.set_selected_surface_ids((parent.surface_id,))
        self.workspace.add_plane_button.click()
        plane = self.workspace.get_data().overlay_planes[0]
        texture_path = self.asset_directory / "plane-only.png"
        texture_path.write_bytes(_texture_png())
        self.workspace._data.assignments = [
            SurfaceTextureAssignment(
                assignment_id="plane-only",
                surface_type="wall",
                surface_ids=(plane.surface_id,),
                provider="meshy",
                asset_path=texture_path.name,
            )
        ]
        self.workspace.surface_view.set_surface_texture(
            (plane.surface_id,),
            texture_path,
        )
        assignment_removals = QSignalSpy(self.workspace.assignments_removed)

        self.workspace.remove_plane_button.click()

        self.assertFalse(texture_path.exists())
        self.assertEqual(self.workspace.get_data().assignments, [])
        self.assertEqual(assignment_removals.count(), 1)
        self.assertEqual(assignment_removals.at(0)[0], ("plane-only",))


# ### Main integration tests ###
class SurfaceTextureOverlayMainIntegrationTests(unittest.TestCase):
    def test_main_forwards_overlay_geometry_and_refreshes_on_content_change(
        self,
    ) -> None:
        level = _build_level()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = ApplicationSettingsStore(root / "settings.json")
            workspace = BlueprintWorkspace(application_settings=settings)
            try:
                workspace.levels = [level]
                surface_workspace = workspace.surface_texture_generation
                surface_workspace.set_levels([level])
                parent = next(
                    surface
                    for surface in surface_workspace.surface_view.get_surfaces()
                    if surface.surface_type == SURFACE_TYPE_WALL
                )
                plane = SurfaceTextureOverlayPlane(
                    parent_surface_id=parent.surface_id,
                    normal_offset_meters=0.003,
                )
                surface_asset_directory = root / "surface_textures"
                surface_asset_directory.mkdir(exist_ok=True)
                texture_path = surface_asset_directory / "overlay.png"
                texture_path.write_bytes(_texture_png())
                surface_workspace.set_data(
                    SurfaceTextureData(
                        overlay_planes=[plane],
                        assignments=[
                            SurfaceTextureAssignment(
                                assignment_id="overlay",
                                surface_type="wall",
                                surface_ids=(plane.surface_id,),
                                provider="meshy",
                                asset_path=texture_path.name,
                            )
                        ],
                    )
                )
                atlas_source_id = build_atlas_wall_texture_source_id(
                    "overlay"
                )
                self.assertIn(
                    atlas_source_id,
                    workspace.texture_atlas_workspace._sources_by_object_id,
                )
                generated_model = object()
                with patch(
                    "housemaker.main.convert_to_glb",
                    return_value=generated_model,
                ) as convert_mock:
                    self.assertIs(
                        workspace._build_generated_model(None),
                        generated_model,
                    )

                call_kwargs = convert_mock.call_args.kwargs
                self.assertEqual(
                    call_kwargs["surface_materials"],
                    {plane.surface_id: texture_path.resolve()},
                )
                forwarded_planes = call_kwargs["surface_overlay_planes"]
                self.assertEqual(len(forwarded_planes), 1)
                self.assertEqual(
                    forwarded_planes[0].surface_id,
                    plane.surface_id,
                )
                with patch.object(
                    workspace,
                    "_schedule_viewer_preview_refresh",
                ) as refresh:
                    surface_workspace.surface_content_changed.emit()
                refresh.assert_called_once_with(preserve_camera=True)
            finally:
                workspace.surface_texture_generation.shutdown()
                workspace.generation.shutdown()
                workspace.close()
                workspace.deleteLater()
                _qt_application.processEvents()


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
