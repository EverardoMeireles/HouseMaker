# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from PIL import Image
from PySide6.QtWidgets import QApplication
from shapely.geometry import Polygon

import housemaker.glb as glb_module
from housemaker.app_settings import ApplicationSettingsStore
from housemaker.atlas_export import (
    MaterializedTextureAtlas,
    apply_texture_atlases_to_export,
)
from housemaker.glb import (
    convert_to_glb,
    convert_to_preview_model,
    import_generated_glb,
)
from housemaker.models import LevelData, RoomData, SlopedCeilingData, VertexData
from housemaker.pbr_maps import ATLAS_MAP_BASE_COLOR, ATLAS_MAP_TYPES
from housemaker.surface_geometry import (
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_WALL,
    FixedSurface,
    build_base_fixed_surfaces,
    build_fixed_surfaces,
)
from housemaker.surface_texture_state import (
    SurfaceTextureAssignment,
    SurfaceTextureData,
)
from housemaker.surface_texture_workspace import (
    SurfaceTextureGenerationWorkspace,
)
from housemaker.texture_atlas_state import (
    TextureAtlasPlacement,
    TextureAtlasRecord,
)

# ### Constants ###
SLOPE_ID = "a" * 32
JOINED_FIRST_SLOPE_ID = "b" * 32
JOINED_SECOND_SLOPE_ID = "c" * 32
JOINED_SLOPE_GROUP_ID = "d" * 32


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_sloped_room_level() -> LevelData:
    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (400.0, 0.0),
            (400.0, 400.0),
            (0.0, 400.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
        strict=True,
    ):
        vertex_data.add_edge(start_id, end_id)
    center = vertex_data.add_vertex(200.0, 200.0)
    level = LevelData(
        index=2,
        name="Ground",
        height_meters=3.0,
        floor_thickness_meters=0.3,
        vertex_data=vertex_data,
        rooms=[
            RoomData(
                name="Room",
                vertex_ids=boundary_ids,
                center_vertex_id=center.id,
                color_rgb=(140, 180, 220),
            )
        ],
    )
    flat_surfaces = build_base_fixed_surfaces(
        (level,),
        apply_sloped_ceilings=False,
    )
    host_wall = next(
        surface
        for surface in flat_surfaces
        if surface.surface_type == "wall"
        and np.allclose(surface.wall_start_world[1], 0.0)
        and np.allclose(surface.wall_end_world[1], 0.0)
    )
    level.sloped_ceilings = [
        SlopedCeilingData(
            slope_id=SLOPE_ID,
            wall_surface_id=host_wall.surface_id,
            depth_meters=1.0,
            wall_drop_meters=0.5,
            transition_radius_meters=0.0,
            wall_transition_radius_meters=0.0,
            side_transition_radius_meters=0.1,
            start_ratio=0.2,
            end_ratio=0.8,
        )
    ]
    return level


def _build_joined_sloped_room_level() -> LevelData:
    """Build two opposing partial slopes with shared physical side faces."""

    level = _build_sloped_room_level()
    flat_surfaces = build_base_fixed_surfaces(
        (level,),
        apply_sloped_ceilings=False,
    )
    first_wall = next(
        surface
        for surface in flat_surfaces
        if surface.surface_type == SURFACE_TYPE_WALL
        and np.allclose(surface.wall_start_world[1], 0.0)
        and np.allclose(surface.wall_end_world[1], 0.0)
    )
    second_wall = next(
        surface
        for surface in flat_surfaces
        if surface.surface_type == SURFACE_TYPE_WALL
        and np.allclose(surface.wall_start_world[1], -8.0)
        and np.allclose(surface.wall_end_world[1], -8.0)
    )
    level.sloped_ceilings = [
        SlopedCeilingData(
            slope_id=slope_id,
            wall_surface_id=wall.surface_id,
            joined_group_id=JOINED_SLOPE_GROUP_ID,
            depth_meters=4.5,
            wall_drop_meters=0.6,
            transition_radius_meters=0.1,
            wall_transition_radius_meters=0.1,
            side_transition_radius_meters=0.1,
            start_ratio=0.2,
            end_ratio=0.8,
        )
        for slope_id, wall in (
            (JOINED_FIRST_SLOPE_ID, first_wall),
            (JOINED_SECOND_SLOPE_ID, second_wall),
        )
    ]
    return level


def _slope_surfaces(level: LevelData) -> tuple[FixedSurface, ...]:
    return tuple(
        surface
        for surface in build_fixed_surfaces((level,))
        if surface.sloped_ceiling_id == SLOPE_ID
    )


def _solid_png(color: tuple[int, int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGBA", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def _write_atlas_maps(directory: Path, resolution: int) -> dict[str, Path]:
    """Materialize the complete map set required by one export Atlas."""

    paths: dict[str, Path] = {}
    for map_type in ATLAS_MAP_TYPES:
        path = directory / f"slope-atlas.{map_type}.png"
        Image.new("RGBA", (resolution, resolution), (128, 128, 128, 255)).save(
            path
        )
        paths[map_type] = path
    return paths


def _triangle_key(triangle: object) -> tuple[tuple[float, ...], ...]:
    # GLB stores mesh positions as float32, so compare at a tolerance that
    # distinguishes faces without treating serialization noise as geometry.
    points = np.round(np.asarray(triangle, dtype=float), 4)
    return tuple(sorted(tuple(float(value) for value in point) for point in points))


def _canonical_triangle_plane_key(
    triangle: object,
) -> tuple[float, ...] | None:
    """Return an orientation-independent key for one triangle plane."""

    points = np.asarray(triangle, dtype=float)
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    length = float(np.linalg.norm(normal))
    if length <= 1e-12:
        return None
    normal /= length
    plane_offset = float(np.dot(normal, points[0]))
    coefficients = np.append(normal, plane_offset)
    first_nonzero = next(
        (value for value in coefficients if abs(float(value)) > 1e-10),
        0.0,
    )
    if first_nonzero < 0.0:
        coefficients *= -1.0
    return tuple(float(value) for value in np.round(coefficients, 7))


def _projected_triangle(
    triangle: object,
    plane_key: tuple[float, ...],
) -> Polygon:
    """Project one triangle onto the most stable axes of its shared plane."""

    drop_axis = int(np.argmax(np.abs(np.asarray(plane_key[:3], dtype=float))))
    return Polygon(np.delete(np.asarray(triangle, dtype=float), drop_axis, axis=1))


def _coplanar_overlap_pairs(
    first_mesh: trimesh.Trimesh,
    second_mesh: trimesh.Trimesh,
) -> list[tuple[int, int]]:
    """Find positive-area overlaps between two independently drawn meshes."""

    second_by_plane: dict[
        tuple[float, ...],
        list[tuple[int, Polygon]],
    ] = {}
    for second_index, triangle in enumerate(second_mesh.triangles):
        plane_key = _canonical_triangle_plane_key(triangle)
        if plane_key is None:
            continue
        second_by_plane.setdefault(plane_key, []).append(
            (second_index, _projected_triangle(triangle, plane_key))
        )

    overlaps: list[tuple[int, int]] = []
    for first_index, triangle in enumerate(first_mesh.triangles):
        plane_key = _canonical_triangle_plane_key(triangle)
        if plane_key is None:
            continue
        first_polygon = _projected_triangle(triangle, plane_key)
        first_bounds = first_polygon.bounds
        for second_index, second_polygon in second_by_plane.get(plane_key, ()):
            second_bounds = second_polygon.bounds
            if (
                first_bounds[2] <= second_bounds[0] + 1e-9
                or second_bounds[2] <= first_bounds[0] + 1e-9
                or first_bounds[3] <= second_bounds[1] + 1e-9
                or second_bounds[3] <= first_bounds[1] + 1e-9
            ):
                continue
            if first_polygon.intersection(second_polygon).area > 1e-9:
                overlaps.append((first_index, second_index))
    return overlaps


def _scene_world_meshes(
    scene: trimesh.Scene,
) -> tuple[tuple[str, str, trimesh.Trimesh], ...]:
    """Return each glTF primitive in world space with its semantic surface ID."""

    meshes: list[tuple[str, str, trimesh.Trimesh]] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        meshes.append(
            (
                str(node_name),
                str(mesh.metadata.get("housemaker_surface_id", "")),
                mesh,
            )
        )
    return tuple(meshes)


# ### Texture-state regressions ###
class SlopedCeilingTextureStateTests(unittest.TestCase):
    def test_each_part_id_round_trips_in_independent_assignments(self) -> None:
        level = _build_sloped_room_level()
        slope_surfaces = _slope_surfaces(level)
        assignments = tuple(
            SurfaceTextureAssignment(
                assignment_id=f"slope-{surface.sloped_ceiling_part}",
                surface_type=SURFACE_TYPE_CEILING,
                surface_ids=(surface.surface_id,),
                provider="meshy",
                asset_path=f"{surface.sloped_ceiling_part}.png",
                combined_area_m2=surface.area_square_meters,
            )
            for surface in slope_surfaces
        )

        restored = SurfaceTextureData.from_dict(
            SurfaceTextureData(assignments=list(assignments)).to_dict()
        )

        self.assertEqual(tuple(restored.assignments), assignments)
        self.assertEqual(
            {assignment.surface_ids[0] for assignment in restored.assignments},
            {surface.surface_id for surface in slope_surfaces},
        )

    def test_assignment_survives_slope_geometry_revision(self) -> None:
        level = _build_sloped_room_level()
        main_surface = next(
            surface
            for surface in _slope_surfaces(level)
            if surface.sloped_ceiling_part == "main"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            asset_directory = temporary_path / "surface_assets"
            asset_directory.mkdir(parents=True)
            (asset_directory / "slope.png").write_bytes(
                _solid_png((180, 120, 80, 255))
            )
            workspace = SurfaceTextureGenerationWorkspace(
                asset_directory=asset_directory,
                application_settings=ApplicationSettingsStore(
                    temporary_path / "settings.json"
                ),
            )
            try:
                workspace.set_levels((level,))
                workspace.set_data(
                    SurfaceTextureData(
                        assignments=[
                            SurfaceTextureAssignment(
                                assignment_id="slope-main",
                                surface_type=SURFACE_TYPE_CEILING,
                                surface_ids=(main_surface.surface_id,),
                                provider="meshy",
                                asset_path="slope.png",
                                combined_area_m2=main_surface.area_square_meters,
                            )
                        ]
                    )
                )
                level.sloped_ceilings[0] = replace(
                    level.sloped_ceilings[0],
                    depth_meters=1.35,
                )

                changed = workspace.reconcile_assignments_with_levels((level,))

                updated_surface = next(
                    surface
                    for surface in _slope_surfaces(level)
                    if surface.surface_id == main_surface.surface_id
                )
                updated_assignment = workspace.get_assignments()[0]
                self.assertTrue(changed)
                self.assertEqual(
                    updated_assignment.surface_ids,
                    (main_surface.surface_id,),
                )
                self.assertAlmostEqual(
                    updated_assignment.combined_area_m2,
                    updated_surface.area_square_meters,
                )
            finally:
                workspace.shutdown()
                workspace.close()
                _qt_application.processEvents()

    def test_retired_joined_side_assignment_migrates_to_complete_side(
        self,
    ) -> None:
        level = _build_joined_sloped_room_level()
        target = min(
            (
                surface
                for surface in build_fixed_surfaces((level,))
                if surface.sloped_ceiling_part
                in {"start-return", "end-return"}
            ),
            key=lambda surface: float(surface.mesh.bounds[0, 0]),
        )
        self.assertEqual(len(target.surface_alias_ids), 1)
        retired_id = target.surface_alias_ids[0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            asset_directory = temporary_path / "surface_assets"
            asset_directory.mkdir(parents=True)
            (asset_directory / "joined-side.png").write_bytes(
                _solid_png((180, 120, 80, 255))
            )
            workspace = SurfaceTextureGenerationWorkspace(
                asset_directory=asset_directory,
                application_settings=ApplicationSettingsStore(
                    temporary_path / "settings.json"
                ),
            )
            try:
                workspace.set_levels((level,))
                workspace.set_data(
                    SurfaceTextureData(
                        assignments=[
                            SurfaceTextureAssignment(
                                assignment_id="retired-joined-side",
                                surface_type=SURFACE_TYPE_CEILING,
                                surface_ids=(retired_id,),
                                provider="meshy",
                                asset_path="joined-side.png",
                                combined_area_m2=target.area_square_meters / 2.0,
                            )
                        ]
                    )
                )

                self.assertTrue(
                    workspace.reconcile_assignments_with_levels((level,))
                )
                assignment = workspace.get_assignments()[0]
                self.assertEqual(assignment.surface_ids, (target.surface_id,))
                self.assertAlmostEqual(
                    assignment.combined_area_m2,
                    target.area_square_meters,
                )
                model = convert_to_preview_model(
                    (level,),
                    surface_materials=workspace.get_surface_material_sources(),
                )
                self.assertEqual(
                    tuple(
                        surface.surface_id
                        for surface in model.preview_textured_surfaces
                    ),
                    (target.surface_id,),
                )
                np.testing.assert_allclose(
                    model.preview_textured_surfaces[0].mesh.bounds,
                    target.mesh.bounds,
                )
            finally:
                workspace.shutdown()
                workspace.close()
                _qt_application.processEvents()

    def test_slope_parts_accept_wall_and_ceiling_texture_families(self) -> None:
        level = _build_sloped_room_level()
        all_surfaces = tuple(build_fixed_surfaces((level,)))
        slope_surfaces = _slope_surfaces(level)
        ordinary_surfaces = {
            surface_type: next(
                surface
                for surface in all_surfaces
                if surface.surface_type == surface_type
                and surface.sloped_ceiling_id is None
            )
            for surface_type in (SURFACE_TYPE_WALL, SURFACE_TYPE_CEILING)
        }

        for texture_type in (SURFACE_TYPE_WALL, SURFACE_TYPE_CEILING):
            with (
                self.subTest(texture_type=texture_type),
                tempfile.TemporaryDirectory() as temporary_directory,
            ):
                temporary_path = Path(temporary_directory)
                asset_directory = temporary_path / "surface_assets"
                asset_directory.mkdir(parents=True)
                asset_path = asset_directory / f"{texture_type}.png"
                asset_path.write_bytes(_solid_png((120, 160, 200, 255)))
                assignment = SurfaceTextureAssignment(
                    assignment_id=f"{texture_type}-family",
                    surface_type=texture_type,
                    surface_ids=(),
                    provider="meshy",
                    asset_path=asset_path.name,
                )
                workspace = SurfaceTextureGenerationWorkspace(
                    asset_directory=asset_directory,
                    application_settings=ApplicationSettingsStore(
                        temporary_path / "settings.json"
                    )
                )
                try:
                    workspace.set_levels((level,))
                    workspace.set_data(
                        SurfaceTextureData(assignments=[assignment])
                    )
                    targets = (
                        ordinary_surfaces[texture_type].surface_id,
                        *(surface.surface_id for surface in slope_surfaces),
                    )

                    self.assertTrue(
                        workspace.assignment_targets_are_valid(
                            assignment.assignment_id,
                            targets,
                        )
                    )
                    self.assertTrue(
                        workspace.apply_assignment_texture(
                            assignment.assignment_id,
                            targets,
                        )
                    )

                    applied = workspace.get_assignment(assignment.assignment_id)
                    assert applied is not None
                    self.assertEqual(applied.surface_ids, targets)
                    restored = SurfaceTextureData.from_dict(
                        workspace.get_data().to_dict()
                    )
                    self.assertEqual(
                        restored.assignments[0].surface_ids,
                        targets,
                    )
                finally:
                    workspace.shutdown()
                    workspace.close()
                    _qt_application.processEvents()

    def test_slope_compatibility_does_not_relax_ordinary_surface_types(
        self,
    ) -> None:
        level = _build_sloped_room_level()
        all_surfaces = tuple(build_fixed_surfaces((level,)))
        ordinary_wall = next(
            surface
            for surface in all_surfaces
            if surface.surface_type == SURFACE_TYPE_WALL
            and surface.sloped_ceiling_id is None
        )
        ordinary_ceiling = next(
            surface
            for surface in all_surfaces
            if surface.surface_type == SURFACE_TYPE_CEILING
            and surface.sloped_ceiling_id is None
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            asset_directory = temporary_path / "surface_assets"
            asset_directory.mkdir(parents=True)
            (asset_directory / "wall.png").write_bytes(
                _solid_png((120, 160, 200, 255))
            )
            assignment = SurfaceTextureAssignment(
                assignment_id="wall-family",
                surface_type=SURFACE_TYPE_WALL,
                surface_ids=(),
                provider="meshy",
                asset_path="wall.png",
            )
            workspace = SurfaceTextureGenerationWorkspace(
                asset_directory=asset_directory,
                application_settings=ApplicationSettingsStore(
                    temporary_path / "settings.json"
                ),
            )
            try:
                workspace.set_levels((level,))
                workspace.set_data(
                    SurfaceTextureData(assignments=[assignment])
                )

                self.assertTrue(
                    workspace.assignment_targets_are_valid(
                        assignment.assignment_id,
                        (ordinary_wall.surface_id,),
                    )
                )
                self.assertFalse(
                    workspace.assignment_targets_are_valid(
                        assignment.assignment_id,
                        (ordinary_ceiling.surface_id,),
                    )
                )
                self.assertFalse(
                    workspace.apply_assignment_texture(
                        assignment.assignment_id,
                        (ordinary_ceiling.surface_id,),
                    )
                )
                retained = workspace.get_assignment(assignment.assignment_id)
                assert retained is not None
                self.assertEqual(retained.surface_ids, ())
            finally:
                workspace.shutdown()
                workspace.close()
                _qt_application.processEvents()


# ### GLB export regressions ###
class SlopedCeilingTextureExportTests(unittest.TestCase):
    def test_one_joined_side_assignment_textures_the_complete_physical_side(
        self,
    ) -> None:
        level = _build_joined_sloped_room_level()
        side_surfaces = tuple(
            surface
            for surface in build_fixed_surfaces((level,))
            if surface.sloped_ceiling_part
            in {"start-return", "end-return"}
        )
        self.assertEqual(len(side_surfaces), 2)
        target = min(
            side_surfaces,
            key=lambda surface: float(surface.mesh.bounds[0, 0]),
        )
        target_bounds = np.asarray(target.mesh.bounds, dtype=float)
        self.assertAlmostEqual(float(target_bounds[0, 1]), -8.0, places=8)
        self.assertAlmostEqual(float(target_bounds[1, 1]), 0.0, places=8)

        model = convert_to_glb(
            (level,),
            surface_materials={
                target.surface_id: _solid_png((200, 80, 40, 255))
            },
        )
        self.assertEqual(
            tuple(
                surface.surface_id
                for surface in model.preview_textured_surfaces
            ),
            (target.surface_id,),
        )
        textured = model.preview_textured_surfaces[0].mesh
        np.testing.assert_allclose(textured.bounds, target.mesh.bounds)
        textured_uv = np.asarray(getattr(textured.visual, "uv", None))
        self.assertEqual(textured_uv.shape, (len(textured.vertices), 2))
        self.assertTrue(np.all(np.isfinite(textured_uv)))
        assert model.preview_untextured_mesh is not None
        self.assertEqual(
            _coplanar_overlap_pairs(textured, model.preview_untextured_mesh),
            [],
        )

        source_id = "surface-texture:joined-slope-side"
        resolution = 2048
        atlas = TextureAtlasRecord(
            atlas_id="joined-slope-side",
            name="Joined slope side",
            resolution=resolution,
            placements=[
                TextureAtlasPlacement(
                    object_id=source_id,
                    texture_path="joined-slope-side.png",
                    texture_resolution=512,
                    x=0,
                    y=0,
                    size=512,
                )
            ],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            exported = apply_texture_atlases_to_export(
                model,
                (
                    MaterializedTextureAtlas(
                        atlas,
                        _write_atlas_maps(
                            Path(temporary_directory),
                            resolution,
                        ),
                        active_map_types=frozenset(
                            {ATLAS_MAP_BASE_COLOR}
                        ),
                    ),
                ),
                surface_source_ids={target.surface_id: source_id},
            )

        atlas_meshes = tuple(
            mesh
            for mesh in exported.scene.geometry.values()
            if getattr(getattr(mesh.visual, "material", None), "name", None)
            == atlas.name
        )
        self.assertEqual(len(atlas_meshes), 1)
        atlas_uv = np.asarray(
            getattr(atlas_meshes[0].visual, "uv", None),
            dtype=float,
        )
        self.assertEqual(atlas_uv.shape, (len(atlas_meshes[0].vertices), 2))
        self.assertTrue(np.all(np.isfinite(atlas_uv)))
        self.assertGreater(len(exported.glb_bytes), 20)

    def test_texturing_one_part_replaces_only_that_part(self) -> None:
        level = _build_sloped_room_level()
        slope_surfaces = _slope_surfaces(level)
        target = next(
            surface
            for surface in slope_surfaces
            if surface.sloped_ceiling_part == "start-return"
        )
        sibling = next(
            surface
            for surface in slope_surfaces
            if surface.sloped_ceiling_part == "main"
        )

        model = convert_to_preview_model(
            (level,),
            surface_materials={
                target.surface_id: _solid_png((200, 80, 40, 255))
            },
        )

        self.assertEqual(
            tuple(
                surface.surface_id for surface in model.preview_textured_surfaces
            ),
            (target.surface_id,),
        )
        assert model.preview_untextured_mesh is not None
        untextured_face_keys = {
            _triangle_key(triangle)
            for triangle in model.preview_untextured_mesh.triangles
        }
        self.assertTrue(
            all(
                _triangle_key(triangle) not in untextured_face_keys
                for triangle in target.mesh.triangles
            )
        )
        self.assertTrue(
            all(
                _triangle_key(triangle) in untextured_face_keys
                for triangle in sibling.mesh.triangles
            )
        )

    def test_textured_side_returns_have_no_coplanar_render_duplicates(
        self,
    ) -> None:
        """Side skins must replace, rather than overlay, their base geometry."""

        level = _build_sloped_room_level()
        side_surfaces = tuple(
            surface
            for surface in _slope_surfaces(level)
            if surface.sloped_ceiling_part in {"start-return", "end-return"}
        )
        side_surface_ids = {surface.surface_id for surface in side_surfaces}
        self.assertEqual(len(side_surfaces), 2)

        model = convert_to_glb(
            (level,),
            surface_materials={
                surface.surface_id: _solid_png((200, 80, 40, 255))
                for surface in side_surfaces
            },
        )
        assert model.preview_untextured_mesh is not None
        preview_meshes = [
            ("untextured preview", "", model.preview_untextured_mesh),
            *(
                (surface.surface_id, surface.surface_id, surface.mesh)
                for surface in model.preview_textured_surfaces
            ),
        ]
        for target_name, target_id, target_mesh in preview_meshes:
            if target_id not in side_surface_ids:
                continue
            for other_name, _other_id, other_mesh in preview_meshes:
                if target_name == other_name:
                    continue
                with self.subTest(
                    representation="preview",
                    target=target_name,
                    other=other_name,
                ):
                    self.assertEqual(
                        _coplanar_overlap_pairs(target_mesh, other_mesh),
                        [],
                    )

        serialized_scene = trimesh.load(
            BytesIO(model.glb_bytes),
            file_type="glb",
            force="scene",
        )
        self.assertIsInstance(serialized_scene, trimesh.Scene)
        glb_meshes = _scene_world_meshes(serialized_scene)
        self.assertEqual(
            {
                surface_id
                for _name, surface_id, _mesh in glb_meshes
                if surface_id in side_surface_ids
            },
            side_surface_ids,
        )
        for target_name, target_id, target_mesh in glb_meshes:
            if target_id not in side_surface_ids:
                continue
            for other_name, _other_id, other_mesh in glb_meshes:
                if target_name == other_name:
                    continue
                with self.subTest(
                    representation="serialized glb",
                    target=target_name,
                    other=other_name,
                ):
                    self.assertEqual(
                        _coplanar_overlap_pairs(target_mesh, other_mesh),
                        [],
                    )

    def test_tagged_side_source_is_replaced_before_atlas_export(self) -> None:
        """A semantic source tag must outrank small geometric drift."""

        level = _build_sloped_room_level()
        target = next(
            surface
            for surface in _slope_surfaces(level)
            if surface.sloped_ceiling_part == "start-return"
        )
        original_apply = glb_module._apply_sloped_ceiling_surface_geometry

        def build_offset_tagged_source(*args, **kwargs):
            named_meshes = original_apply(*args, **kwargs)
            for named_mesh in named_meshes:
                if (
                    named_mesh.mesh.metadata.get("housemaker_surface_id")
                    != target.surface_id
                ):
                    continue
                named_mesh.mesh = named_mesh.mesh.copy()
                named_mesh.mesh.apply_translation((0.002, 0.0, 0.0))
            return named_meshes

        with patch.object(
            glb_module,
            "_apply_sloped_ceiling_surface_geometry",
            side_effect=build_offset_tagged_source,
        ):
            model = convert_to_glb(
                (level,),
                surface_materials={
                    target.surface_id: _solid_png((200, 80, 40, 255))
                },
            )

        tagged_geometry = tuple(
            mesh
            for mesh in model.scene.geometry.values()
            if mesh.metadata.get("housemaker_surface_id") == target.surface_id
        )
        self.assertEqual(len(tagged_geometry), 1)
        tagged_uv = np.asarray(getattr(tagged_geometry[0].visual, "uv", None))
        self.assertEqual(tagged_uv.shape, (len(tagged_geometry[0].vertices), 2))

        source_id = "surface-texture:slope-side"
        resolution = 2048
        atlas = TextureAtlasRecord(
            atlas_id="slope-side",
            name="Slope side",
            resolution=resolution,
            placements=[
                TextureAtlasPlacement(
                    object_id=source_id,
                    texture_path="slope-side.png",
                    texture_resolution=512,
                    x=0,
                    y=0,
                    size=512,
                )
            ],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            exported = apply_texture_atlases_to_export(
                model,
                (
                    MaterializedTextureAtlas(
                        atlas,
                        _write_atlas_maps(
                            Path(temporary_directory),
                            resolution,
                        ),
                        active_map_types=frozenset({ATLAS_MAP_BASE_COLOR}),
                    ),
                ),
                surface_source_ids={target.surface_id: source_id},
            )

        atlas_meshes = tuple(
            mesh
            for mesh in exported.scene.geometry.values()
            if getattr(getattr(mesh.visual, "material", None), "name", None)
            == atlas.name
        )
        self.assertEqual(len(atlas_meshes), 1)
        self.assertGreater(len(exported.glb_bytes), 20)

    def test_all_slope_parts_share_one_double_sided_atlas_source(self) -> None:
        """Every semantic slope part must reach one Atlas with valid UVs."""

        level = _build_sloped_room_level()
        slope_surfaces = _slope_surfaces(level)
        slope_surface_ids = {
            surface.surface_id for surface in slope_surfaces
        }
        model = convert_to_glb(
            (level,),
            surface_materials={
                surface.surface_id: _solid_png((200, 80, 40, 255))
                for surface in slope_surfaces
            },
            make_ceilings_double_sided=True,
        )

        bound_geometry = tuple(
            mesh
            for mesh in model.scene.geometry.values()
            if mesh.metadata.get("housemaker_surface_id") in slope_surface_ids
        )
        self.assertEqual(len(bound_geometry), len(slope_surfaces))
        for mesh in bound_geometry:
            uv = np.asarray(getattr(mesh.visual, "uv", None), dtype=float)
            self.assertEqual(uv.shape, (len(mesh.vertices), 2))
            self.assertTrue(np.all(np.isfinite(uv)))

        source_id = "surface-texture:all-slope-parts"
        resolution = 2048
        atlas = TextureAtlasRecord(
            atlas_id="all-slope-parts",
            name="All slope parts",
            resolution=resolution,
            placements=[
                TextureAtlasPlacement(
                    object_id=source_id,
                    texture_path="all-slope-parts.png",
                    texture_resolution=512,
                    x=0,
                    y=0,
                    size=512,
                )
            ],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            exported = apply_texture_atlases_to_export(
                model,
                (
                    MaterializedTextureAtlas(
                        atlas,
                        _write_atlas_maps(
                            Path(temporary_directory),
                            resolution,
                        ),
                        active_map_types=frozenset({ATLAS_MAP_BASE_COLOR}),
                    ),
                ),
                surface_source_ids={
                    surface.surface_id: source_id for surface in slope_surfaces
                },
            )

        atlas_meshes = tuple(
            mesh
            for mesh in exported.scene.geometry.values()
            if getattr(getattr(mesh.visual, "material", None), "name", None)
            == atlas.name
        )
        self.assertEqual(len(atlas_meshes), 1)
        atlas_uv = np.asarray(
            getattr(atlas_meshes[0].visual, "uv", None),
            dtype=float,
        )
        self.assertEqual(atlas_uv.shape, (len(atlas_meshes[0].vertices), 2))
        self.assertTrue(np.all(np.isfinite(atlas_uv)))
        self.assertGreater(len(exported.glb_bytes), 20)

    def test_assigned_only_glb_contains_exact_slope_part_and_metadata(self) -> None:
        level = _build_sloped_room_level()
        target = next(
            surface
            for surface in _slope_surfaces(level)
            if surface.sloped_ceiling_part == "end-return"
        )

        model = convert_to_glb(
            (level,),
            surface_materials={
                target.surface_id: _solid_png((40, 120, 200, 255))
            },
            export_untextured_surfaces=False,
        )
        exported_mesh = import_generated_glb(model.glb_bytes).mesh

        self.assertEqual(len(exported_mesh.faces), len(target.mesh.faces))
        self.assertEqual(
            {
                str(mesh.metadata.get("housemaker_surface_id", ""))
                for mesh in model.scene.geometry.values()
            },
            {target.surface_id},
        )
        np.testing.assert_allclose(
            exported_mesh.bounds,
            target.mesh.bounds,
            atol=1e-5,
        )
        self.assertAlmostEqual(exported_mesh.area, target.mesh.area, places=5)


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
