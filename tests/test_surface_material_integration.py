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
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.glb import GLTF_Y_UP_TO_Z_UP_TRANSFORM, convert_to_glb
from housemaker.main import BlueprintWorkspace
from housemaker.models import (
    DoorwayData,
    LevelData,
    RoomData,
    VertexData,
    WallTextureData,
)
from housemaker.surface_geometry import build_fixed_surfaces
from housemaker.surface_materials import (
    build_assignment_surface_material_source_map,
)
from housemaker.surface_texture_state import (
    SurfaceTextureAssignment,
    SurfaceTextureData,
)
from housemaker.surface_texture_workspace import (
    SurfaceTextureGenerationWorkspace,
)
from housemaker.viewer import GlbViewerWidget


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _add_closed_room(
    vertex_data: VertexData,
    points: tuple[tuple[float, float], ...],
    name: str,
) -> RoomData:
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id for point in points
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
    ):
        vertex_data.add_edge(start_id, end_id)
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    center = vertex_data.add_vertex(center_x, center_y)
    return RoomData(
        name=name,
        vertex_ids=boundary_ids,
        center_vertex_id=center.id,
        color_rgb=(140, 180, 220),
    )


def _build_one_room_level() -> tuple[LevelData, RoomData]:
    vertex_data = VertexData()
    room = _add_closed_room(
        vertex_data,
        ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),
        "Living room",
    )
    return (
        LevelData(
            index=2,
            name="Ground",
            vertex_data=vertex_data,
            rooms=[room],
            floor_contour_vertex_ids=room.vertex_ids,
        ),
        room,
    )


def _build_two_separate_room_level() -> tuple[LevelData, RoomData, RoomData]:
    vertex_data = VertexData()
    first_room = _add_closed_room(
        vertex_data,
        ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),
        "First room",
    )
    second_room = _add_closed_room(
        vertex_data,
        (
            (200.0, 0.0),
            (300.0, 0.0),
            (300.0, 100.0),
            (200.0, 100.0),
        ),
        "Second room",
    )
    return (
        LevelData(
            index=2,
            name="Ground",
            vertex_data=vertex_data,
            rooms=[first_room, second_room],
        ),
        first_room,
        second_room,
    )


def _build_adjacent_room_level_with_doorway() -> tuple[LevelData, str]:
    vertex_data = VertexData()
    bottom_left = vertex_data.add_vertex(0.0, 0.0)
    shared_bottom = vertex_data.add_vertex(100.0, 0.0)
    shared_top = vertex_data.add_vertex(100.0, 100.0)
    top_left = vertex_data.add_vertex(0.0, 100.0)
    bottom_right = vertex_data.add_vertex(200.0, 0.0)
    top_right = vertex_data.add_vertex(200.0, 100.0)
    for start_id, end_id in (
        (bottom_left.id, shared_bottom.id),
        (shared_bottom.id, shared_top.id),
        (shared_top.id, top_left.id),
        (top_left.id, bottom_left.id),
        (shared_bottom.id, bottom_right.id),
        (bottom_right.id, top_right.id),
        (top_right.id, shared_top.id),
    ):
        vertex_data.add_edge(start_id, end_id)
    first_center = vertex_data.add_vertex(50.0, 50.0)
    second_center = vertex_data.add_vertex(150.0, 50.0)
    rooms = [
        RoomData(
            name="Left",
            vertex_ids=(
                bottom_left.id,
                shared_bottom.id,
                shared_top.id,
                top_left.id,
            ),
            center_vertex_id=first_center.id,
            color_rgb=(150, 170, 210),
        ),
        RoomData(
            name="Right",
            vertex_ids=(
                shared_bottom.id,
                bottom_right.id,
                top_right.id,
                shared_top.id,
            ),
            center_vertex_id=second_center.id,
            color_rgb=(210, 170, 150),
        ),
    ]
    level = LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=rooms,
        doorways=[
            DoorwayData(
                center_x=100.0,
                center_y=50.0,
                width_meters=0.8,
                height_meters=2.0,
                depth_meters=0.2,
                rotation_degrees=0.0,
            )
        ],
    )
    shared_wall_key = (
        f"{min(shared_bottom.id, shared_top.id)}:"
        f"{max(shared_bottom.id, shared_top.id)}"
    )
    return level, shared_wall_key


def _build_concave_u_room_level() -> tuple[LevelData, RoomData]:
    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (300.0, 0.0),
            (300.0, 300.0),
            (200.0, 300.0),
            (200.0, 100.0),
            (100.0, 100.0),
            (100.0, 300.0),
            (0.0, 300.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
    ):
        vertex_data.add_edge(start_id, end_id)
    center_vertex = vertex_data.add_vertex(50.0, 150.0)
    room = RoomData(
        name="U room",
        vertex_ids=boundary_ids,
        center_vertex_id=center_vertex.id,
        color_rgb=(140, 180, 220),
    )
    return (
        LevelData(
            index=2,
            name="Ground",
            vertex_data=vertex_data,
            rooms=[room],
            floor_contour_vertex_ids=room.vertex_ids,
        ),
        room,
    )


def _build_reversed_plain_wall_level() -> LevelData:
    vertex_data = VertexData()
    left_vertex = vertex_data.add_vertex(0.0, 0.0)
    right_vertex = vertex_data.add_vertex(100.0, 0.0)
    vertex_data.add_edge(right_vertex.id, left_vertex.id)
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
    )


def _solid_png(color: tuple[int, int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGBA", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def _surface_by_id(level: LevelData, surface_id: str):
    return next(
        surface
        for surface in build_fixed_surfaces([level])
        if surface.surface_id == surface_id
    )


def _assignment(
    assignment_id: str,
    surface_id: str,
    asset_path: str,
) -> SurfaceTextureAssignment:
    surface_type = surface_id.rsplit("/", maxsplit=1)[-1].split(":", maxsplit=1)[0]
    return SurfaceTextureAssignment(
        assignment_id=assignment_id,
        surface_type=surface_type,
        surface_ids=(surface_id,),
        provider="meshy",
        asset_path=asset_path,
    )


# ### Stable identity tests ###
class StableRoomSurfaceIdentityTests(unittest.TestCase):
    def test_assignment_keeps_targeting_room_after_reorder_and_prior_deletion(
        self,
    ) -> None:
        level, first_room, target_room = _build_two_separate_room_level()
        target_surface_id = (
            f"level:2/room:{target_room.center_vertex_id}/floor"
        )
        original_bounds = _surface_by_id(
            level,
            target_surface_id,
        ).mesh.bounds.copy()

        level.rooms.reverse()
        reordered_surface = _surface_by_id(level, target_surface_id)
        np.testing.assert_allclose(reordered_surface.mesh.bounds, original_bounds)

        level.rooms = [target_room]
        retained_surface = _surface_by_id(level, target_surface_id)
        np.testing.assert_allclose(retained_surface.mesh.bounds, original_bounds)
        self.assertNotEqual(
            target_room.center_vertex_id,
            first_room.center_vertex_id,
        )

        model = convert_to_glb(
            [level],
            surface_materials={target_surface_id: _solid_png((190, 80, 40, 255))},
        )
        self.assertEqual(
            [surface.surface_id for surface in model.preview_textured_surfaces],
            [target_surface_id],
        )


# ### GLB material tests ###
class SurfaceMaterialGlbTests(unittest.TestCase):
    def test_generated_floor_texture_preserves_legacy_room_wall_texture(
        self,
    ) -> None:
        level, room = _build_one_room_level()
        legacy_color = (18, 202, 76, 255)
        generated_color = (196, 62, 31, 255)
        floor_surface_id = (
            f"level:2/room:{room.center_vertex_id}/floor"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            legacy_texture_path = Path(temporary_directory) / "legacy-wall.png"
            legacy_texture_path.write_bytes(_solid_png(legacy_color))
            room.wall_textures["1:2"] = WallTextureData(
                image_path=str(legacy_texture_path),
                source_x=0.0,
                source_y=0.0,
                source_width=8.0,
                source_height=8.0,
            )

            model = convert_to_glb(
                [level],
                surface_materials={
                    floor_surface_id: _solid_png(generated_color),
                },
            )

        legacy_wall = next(
            wall
            for wall in model.preview_textured_walls
            if wall.wall_key == "1:2"
        )
        self.assertTrue(
            np.any(
                np.all(
                    legacy_wall.texture_rgba
                    == np.asarray(legacy_color, dtype=np.uint8),
                    axis=2,
                )
            )
        )
        self.assertEqual(
            [surface.surface_id for surface in model.preview_textured_surfaces],
            [floor_surface_id],
        )
        self.assertEqual(model.preview_textured_surfaces[0].mesh.visual.kind, "texture")

        exported_scene = trimesh.load(
            BytesIO(model.glb_bytes),
            file_type="glb",
            force="scene",
            process=False,
        )
        self.assertIsInstance(exported_scene, trimesh.Scene)
        material_names = {
            getattr(getattr(mesh.visual, "material", None), "name", None)
            for mesh in exported_scene.geometry.values()
            if mesh.visual.kind == "texture"
        }
        self.assertIn(f"Surface {floor_surface_id}", material_names)
        self.assertTrue(
            any(
                name is not None and name.startswith("L2 Ground Living room")
                for name in material_names
            )
        )

    def test_unknown_surface_material_is_ignored_without_changing_legacy_model(
        self,
    ) -> None:
        level, _room = _build_one_room_level()
        legacy_model = convert_to_glb([level])

        model = convert_to_glb(
            [level],
            surface_materials={
                "level:2/room:999999/floor": Path("missing-texture.png"),
            },
        )

        self.assertEqual(model.preview_textured_surfaces, [])
        self.assertEqual(len(model.mesh.faces), len(legacy_model.mesh.faces))
        self.assertEqual(set(model.scene.geometry), set(legacy_model.scene.geometry))

    def test_legacy_plain_wall_id_textures_both_non_coplanar_sides(self) -> None:
        level = _build_reversed_plain_wall_level()
        surface_id = "level:2/wall:1:2"

        model = convert_to_glb(
            [level],
            surface_materials={surface_id: _solid_png((180, 70, 40, 255))},
        )

        preview_sides = [
            surface.mesh
            for surface in model.preview_textured_surfaces
            if surface.surface_id == surface_id
        ]
        self.assertEqual(len(preview_sides), 2)
        np.testing.assert_allclose(
            sorted(float(mesh.vertices[:, 1].mean()) for mesh in preview_sides),
            (-0.002, 0.002),
        )
        np.testing.assert_allclose(
            sorted(float(mesh.face_normals[:, 1].mean()) for mesh in preview_sides),
            (-1.0, 1.0),
        )

        exported_scene = trimesh.load(
            BytesIO(model.glb_bytes),
            file_type="glb",
            force="scene",
            process=False,
        )
        self.assertIsInstance(exported_scene, trimesh.Scene)
        exported_sides = [
            mesh.copy()
            for mesh in exported_scene.geometry.values()
            if getattr(getattr(mesh.visual, "material", None), "name", "")
            == f"Surface {surface_id}"
        ]
        self.assertEqual(len(exported_sides), 2)
        for mesh in exported_sides:
            mesh.apply_transform(GLTF_Y_UP_TO_Z_UP_TRANSFORM)
        np.testing.assert_allclose(
            sorted(float(mesh.vertices[:, 1].mean()) for mesh in exported_sides),
            (-0.002, 0.002),
        )
        np.testing.assert_allclose(
            sorted(float(mesh.face_normals[:, 1].mean()) for mesh in exported_sides),
            (-1.0, 1.0),
        )

    def test_two_textured_sides_of_shared_wall_use_separate_overlay_planes(
        self,
    ) -> None:
        level, shared_wall_key = _build_adjacent_room_level_with_doorway()
        shared_surface_ids = [
            (
                f"level:2/room:{room.center_vertex_id}/"
                f"wall:{shared_wall_key}"
            )
            for room in level.rooms
        ]
        legacy_model = convert_to_glb([level])

        model = convert_to_glb(
            [level],
            surface_materials={
                shared_surface_ids[0]: _solid_png((180, 70, 40, 255)),
                shared_surface_ids[1]: _solid_png((40, 90, 190, 255)),
            },
        )

        overlays = {
            surface.surface_id: surface.mesh
            for surface in model.preview_textured_surfaces
            if surface.surface_id in shared_surface_ids
        }
        self.assertEqual(set(overlays), set(shared_surface_ids))
        primary_wall_planes: list[float] = []
        for surface_id in shared_surface_ids:
            overlay = overlays[surface_id]
            primary_face_indices = np.flatnonzero(
                np.abs(overlay.face_normals[:, 0]) > 0.9
            )
            primary_vertices = overlay.vertices[
                overlay.faces[primary_face_indices].reshape(-1)
            ]
            plane_x_values = np.unique(np.round(primary_vertices[:, 0], 6))
            self.assertEqual(len(plane_x_values), 1)
            primary_wall_planes.append(float(plane_x_values[0]))

        self.assertAlmostEqual(
            abs(primary_wall_planes[0] - primary_wall_planes[1]),
            0.004,
            places=6,
        )
        self.assertLess(primary_wall_planes[0], 2.0)
        self.assertGreater(primary_wall_planes[1], 2.0)
        self.assertIsNotNone(model.preview_untextured_mesh)
        assert model.preview_untextured_mesh is not None
        self.assertEqual(
            len(model.preview_untextured_mesh.faces),
            len(legacy_model.mesh.faces),
        )
        self.assertTrue(
            np.any(np.isclose(model.preview_untextured_mesh.vertices[:, 0], 2.0))
        )

    def test_concave_room_exports_generated_texture_on_its_interior_face(
        self,
    ) -> None:
        level, room = _build_concave_u_room_level()
        surface_id = f"level:2/room:{room.center_vertex_id}/wall:4:5"

        model = convert_to_glb(
            [level],
            surface_materials={surface_id: _solid_png((180, 70, 40, 255))},
        )

        overlay = next(
            surface.mesh
            for surface in model.preview_textured_surfaces
            if surface.surface_id == surface_id
        )
        self.assertTrue(np.all(overlay.face_normals[:, 0] > 0.9))
        self.assertTrue(np.all(overlay.vertices[:, 0] > 4.0))

        exported_scene = trimesh.load(
            BytesIO(model.glb_bytes),
            file_type="glb",
            force="scene",
            process=False,
        )
        self.assertIsInstance(exported_scene, trimesh.Scene)
        room_mesh = next(
            mesh
            for mesh in exported_scene.geometry.values()
            if getattr(getattr(mesh.visual, "material", None), "name", "")
            == "L2 Ground U room 1"
        )
        inner_wall_faces = np.flatnonzero(
            np.all(np.isclose(room_mesh.triangles[:, :, 0], 4.0), axis=1)
        )
        self.assertEqual(len(inner_wall_faces), 2)
        self.assertTrue(
            np.all(room_mesh.face_normals[inner_wall_faces, 0] > 0.9)
        )


# ### Workspace material routing tests ###
class SurfaceMaterialWorkspaceTests(unittest.TestCase):
    def test_assignment_source_helper_uses_last_valid_and_rejects_bad_paths(
        self,
    ) -> None:
        surface_id = "level:2/room:5/floor"
        ignored_surface_id = "level:2/room:5/ceiling"
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            asset_directory = temporary_root / "assets"
            asset_directory.mkdir()
            first_path = asset_directory / "first.png"
            winning_path = asset_directory / "winning.png"
            non_png_path = asset_directory / "texture.jpg"
            outside_path = temporary_root / "outside.png"
            first_path.write_bytes(_solid_png((20, 40, 60, 255)))
            winning_path.write_bytes(_solid_png((160, 120, 80, 255)))
            non_png_path.write_bytes(b"not a PNG")
            outside_path.write_bytes(_solid_png((220, 30, 30, 255)))

            result = build_assignment_surface_material_source_map(
                [
                    {
                        "asset_path": first_path.name,
                        "surface_ids": (surface_id,),
                    },
                    {
                        "asset_path": "../outside.png",
                        "surface_ids": (surface_id,),
                    },
                    {
                        "asset_path": "missing.png",
                        "surface_ids": (surface_id, ignored_surface_id),
                    },
                    {
                        "asset_path": non_png_path.name,
                        "surface_ids": (surface_id,),
                    },
                    {
                        "asset_path": winning_path.name,
                        "surface_ids": (surface_id,),
                    },
                ],
                asset_directory,
            )

        self.assertEqual(result, {surface_id: winning_path.resolve()})

    def test_material_sources_are_last_assignment_wins_and_skip_missing_assets(
        self,
    ) -> None:
        level, room = _build_one_room_level()
        floor_surface_id = (
            f"level:2/room:{room.center_vertex_id}/floor"
        )
        ceiling_surface_id = (
            f"level:2/room:{room.center_vertex_id}/ceiling"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            asset_directory = Path(temporary_directory)
            first_path = asset_directory / "first.png"
            winning_path = asset_directory / "winning.png"
            first_path.write_bytes(_solid_png((30, 60, 90, 255)))
            winning_path.write_bytes(_solid_png((180, 120, 40, 255)))
            workspace = SurfaceTextureGenerationWorkspace(
                asset_directory=asset_directory
            )
            try:
                workspace.set_levels([level])
                workspace.set_data(
                    SurfaceTextureData(
                        assignments=[
                            _assignment("first", floor_surface_id, first_path.name),
                            _assignment(
                                "winner",
                                floor_surface_id,
                                winning_path.name,
                            ),
                            _assignment(
                                "missing",
                                ceiling_surface_id,
                                "missing.png",
                            ),
                        ]
                    )
                )

                self.assertEqual(
                    workspace.get_surface_material_sources(),
                    {floor_surface_id: winning_path.resolve()},
                )
            finally:
                workspace.shutdown()
                workspace.close()
                workspace.deleteLater()
                _qt_application.processEvents()

    def test_main_routes_material_sources_and_refreshes_viewer_after_generation(
        self,
    ) -> None:
        level, room = _build_one_room_level()
        floor_surface_id = (
            f"level:2/room:{room.center_vertex_id}/floor"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings_path = Path(temporary_directory) / "settings.json"
            asset_directory = Path(temporary_directory) / "surface_textures"
            asset_directory.mkdir()
            texture_path = asset_directory / "floor.png"
            texture_path.write_bytes(_solid_png((70, 110, 190, 255)))
            workspace = BlueprintWorkspace(
                application_settings=ApplicationSettingsStore(settings_path)
            )
            try:
                workspace.levels = [level]
                workspace.surface_texture_generation.set_levels([level])
                assignment = _assignment(
                    "floor",
                    floor_surface_id,
                    texture_path.name,
                )
                workspace.surface_texture_generation.set_data(
                    SurfaceTextureData(assignments=[assignment])
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

                convert_mock.assert_called_once_with(
                    workspace.levels,
                    stairs=[],
                    surface_materials={
                        floor_surface_id: texture_path.resolve(),
                    },
                )
                with patch.object(
                    workspace,
                    "_schedule_viewer_preview_refresh",
                ) as refresh_mock:
                    workspace.surface_texture_generation.generation_completed.emit(
                        assignment
                    )
                refresh_mock.assert_called_once_with(preserve_camera=True)
            finally:
                workspace.surface_texture_generation.shutdown()
                workspace.generation.shutdown()
                workspace.close()
                workspace.deleteLater()
                _qt_application.processEvents()


# ### Ordinary viewer tests ###
class SurfaceMaterialOrdinaryViewerTests(unittest.TestCase):
    def test_legacy_plain_wall_texture_draws_both_sides_in_canvas(self) -> None:
        level = _build_reversed_plain_wall_level()
        surface_id = "level:2/wall:1:2"
        model = convert_to_glb(
            [level],
            surface_materials={surface_id: _solid_png((205, 75, 30, 255))},
        )

        viewer = GlbViewerWidget()
        try:
            viewer.set_model(model)

            self.assertEqual(len(viewer.textured_surface_items), 2)
            self.assertTrue(
                all(item.visible() for item in viewer.textured_surface_items)
            )
        finally:
            viewer.close()
            viewer.deleteLater()
            _qt_application.processEvents()

    def test_neighbor_legacy_wall_stays_on_its_side_of_generated_shared_wall(
        self,
    ) -> None:
        level, shared_wall_key = _build_adjacent_room_level_with_doorway()
        generated_surface_id = (
            f"level:2/room:{level.rooms[0].center_vertex_id}/"
            f"wall:{shared_wall_key}"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            legacy_texture_path = Path(temporary_directory) / "legacy-shared.png"
            legacy_texture_path.write_bytes(_solid_png((35, 180, 80, 255)))
            for room in level.rooms:
                room.wall_textures[shared_wall_key] = WallTextureData(
                    image_path=str(legacy_texture_path),
                    source_x=0.0,
                    source_y=0.0,
                    source_width=8.0,
                    source_height=8.0,
                )
            model = convert_to_glb(
                [level],
                surface_materials={
                    generated_surface_id: _solid_png((205, 75, 30, 255)),
                },
            )

        viewer = GlbViewerWidget()
        try:
            viewer.set_model(model)

            shared_legacy_items = [
                item
                for item in viewer.textured_wall_items
                if viewer.wall_by_item_id[id(item)].wall_key == shared_wall_key
            ]
            self.assertEqual(len(shared_legacy_items), 3)
            self.assertEqual(
                {
                    viewer.wall_by_item_id[id(item)].room_index
                    for item in shared_legacy_items
                },
                {1},
            )
            legacy_plane_x_values = {
                round(float(np.asarray(item.transform().matrix())[0, 3]), 6)
                for item in shared_legacy_items
            }
            self.assertEqual(legacy_plane_x_values, {2.01})

            generated_overlay = model.preview_textured_surfaces[0].mesh
            primary_faces = np.flatnonzero(
                np.abs(generated_overlay.face_normals[:, 0]) > 0.9
            )
            generated_plane_x_values = np.unique(
                np.round(
                    generated_overlay.vertices[
                        generated_overlay.faces[primary_faces].reshape(-1)
                    ][:, 0],
                    6,
                )
            )
            wall_plane_x_values = generated_plane_x_values[
                np.abs(generated_plane_x_values - 2.0) < 0.01
            ]
            np.testing.assert_allclose(wall_plane_x_values, (1.998,))
            self.assertGreater(
                min(legacy_plane_x_values),
                float(wall_plane_x_values[0]),
            )
        finally:
            viewer.close()
            viewer.deleteLater()
            _qt_application.processEvents()

    def test_generated_wall_suppresses_only_its_legacy_wall_preview(self) -> None:
        level, room = _build_one_room_level()
        assigned_wall_key = "1:2"
        assigned_surface_id = (
            f"level:2/room:{room.center_vertex_id}/"
            f"wall:{assigned_wall_key}"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            legacy_texture_path = Path(temporary_directory) / "legacy-wall.png"
            legacy_texture_path.write_bytes(_solid_png((30, 190, 70, 255)))
            room.wall_textures[assigned_wall_key] = WallTextureData(
                image_path=str(legacy_texture_path),
                source_x=0.0,
                source_y=0.0,
                source_width=8.0,
                source_height=8.0,
            )
            model = convert_to_glb(
                [level],
                surface_materials={
                    assigned_surface_id: _solid_png((200, 80, 35, 255)),
                },
            )

        viewer = GlbViewerWidget()
        try:
            viewer.set_model(model)

            self.assertEqual(len(viewer.textured_surface_items), 1)
            remaining_legacy_wall_keys = [
                wall.wall_key for wall in viewer.wall_by_item_id.values()
            ]
            self.assertNotIn(assigned_wall_key, remaining_legacy_wall_keys)
            self.assertEqual(len(remaining_legacy_wall_keys), 3)
            self.assertEqual(
                set(remaining_legacy_wall_keys),
                {"1:4", "2:3", "3:4"},
            )
        finally:
            viewer.close()
            viewer.deleteLater()
            _qt_application.processEvents()

    def test_generated_surface_texture_is_visible_and_obeys_texture_toggle(
        self,
    ) -> None:
        level, room = _build_one_room_level()
        floor_surface_id = (
            f"level:2/room:{room.center_vertex_id}/floor"
        )
        model = convert_to_glb(
            [level],
            surface_materials={
                floor_surface_id: _solid_png((80, 130, 210, 255)),
            },
        )
        viewer = GlbViewerWidget()
        try:
            viewer.set_model(model)

            self.assertEqual(len(viewer.textured_surface_items), 1)
            self.assertTrue(viewer.textured_surface_items[0].visible())
            viewer.set_textures_enabled(False)
            self.assertFalse(viewer.textured_surface_items[0].visible())
            viewer.set_textures_enabled(True)
            self.assertTrue(viewer.textured_surface_items[0].visible())
        finally:
            viewer.close()
            viewer.deleteLater()
            _qt_application.processEvents()


# ### Doorway ownership tests ###
class SharedDoorwaySurfaceTests(unittest.TestCase):
    def test_shared_doorway_reveals_are_owned_by_only_one_room_surface(self) -> None:
        level, shared_wall_key = _build_adjacent_room_level_with_doorway()
        shared_surfaces = [
            surface
            for surface in build_fixed_surfaces([level])
            if surface.wall_key == shared_wall_key
        ]

        self.assertEqual(len(shared_surfaces), 2)
        self.assertEqual(
            sorted(len(surface.mesh.faces) for surface in shared_surfaces),
            [6, 12],
        )
        self.assertAlmostEqual(
            sum(surface.area_square_meters for surface in shared_surfaces),
            9.76,
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
