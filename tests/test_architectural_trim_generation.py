# ### Imports ###
from __future__ import annotations

import unittest
from collections import Counter
from dataclasses import replace

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.architectural_trim import (
    build_architectural_trim_geometry,
    build_cornice_miter_descriptors,
)
from housemaker.architectural_trim_generation import (
    ArchitecturalTrimComponentFrame,
    ArchitecturalTrimRepeatLayout,
    build_architectural_trim_component_frame,
    build_architectural_trim_component_placement,
    build_architectural_trim_component_seed_model,
    fit_architectural_trim_glb_to_dimensions,
    prepare_architectural_trim_repeat_module,
    repeat_architectural_trim_module_glb,
    retarget_architectural_trim_repeat_module_glb,
)
from housemaker.architectural_trim_uv import build_cornice_segmented_uv_glb
from housemaker.glb import (
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
    build_placed_generated_model_transform,
    import_generated_glb,
)
from housemaker.models import (
    TRIM_KIND_CORNICE,
    ArchitecturalTrimData,
    LevelData,
)
from housemaker.surface_geometry import FixedSurface


# ### Fixture helpers ###
def _translated_box_glb() -> bytes:
    z_up_mesh = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
    z_up_mesh.apply_translation((5.0, -3.0, 7.0))
    gltf_mesh = z_up_mesh.copy()
    gltf_mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    scene = trimesh.Scene()
    scene.add_geometry(
        gltf_mesh,
        geom_name="generated_cornice",
        node_name="generated_cornice",
    )
    exported = scene.export(file_type="glb")
    if not isinstance(exported, bytes):
        raise TypeError("The trim test fixture could not be serialized.")
    return exported


def _full_span_trim_glb() -> bytes:
    z_up_mesh = trimesh.creation.box(extents=(6.0, 0.1, 0.1))
    z_up_mesh.apply_translation((0.0, 0.0, 0.05))
    gltf_mesh = z_up_mesh.copy()
    gltf_mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    exported = trimesh.Scene(gltf_mesh).export(file_type="glb")
    if not isinstance(exported, bytes):
        raise TypeError("The full-span trim fixture could not be serialized.")
    return exported


def _textured_two_primitive_glb() -> bytes:
    scene = trimesh.Scene()
    uv_coordinates = np.asarray(
        (
            (0.0, 0.0),
            (0.0, 1.0),
            (0.25, 0.0),
            (0.25, 1.0),
            (0.75, 0.0),
            (0.75, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        ),
        dtype=float,
    )
    for index, center_y in enumerate((-0.25, 0.25), start=1):
        z_up_mesh = trimesh.creation.box(extents=(2.0, 0.5, 2.0))
        z_up_mesh.apply_translation((3.0, center_y, 4.0))
        z_up_mesh.visual = TextureVisuals(
            uv=uv_coordinates.copy(),
            material=PBRMaterial(
                name=f"trim_material_{index}",
                baseColorTexture=Image.new(
                    "RGBA",
                    (2, 2),
                    (40 * index, 80, 160, 255),
                ),
            ),
        )
        gltf_mesh = z_up_mesh.copy()
        gltf_mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
        scene.add_geometry(
            gltf_mesh,
            geom_name=f"trim_primitive_{index}",
            node_name=f"trim_primitive_{index}",
        )
    exported = scene.export(file_type="glb")
    if not isinstance(exported, bytes):
        raise TypeError("The repeated-trim test fixture could not be serialized.")
    return exported


def _textured_box_glb(
    extents: tuple[float, float, float],
) -> bytes:
    z_up_mesh = trimesh.creation.box(extents=extents)
    z_up_mesh.apply_translation((0.0, 0.0, extents[2] * 0.5))
    z_up_mesh.visual = TextureVisuals(
        uv=np.asarray(
            (
                (0.0, 0.0),
                (0.0, 1.0),
                (0.25, 0.0),
                (0.25, 1.0),
                (0.75, 0.0),
                (0.75, 1.0),
                (1.0, 0.0),
                (1.0, 1.0),
            ),
            dtype=float,
        ),
        material=PBRMaterial(
            name="mitered_cornice_material",
            baseColorTexture=Image.new("RGBA", (2, 2), (80, 120, 160, 255)),
        ),
    )
    gltf_mesh = z_up_mesh.copy()
    gltf_mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    exported = trimesh.Scene(gltf_mesh).export(file_type="glb")
    if not isinstance(exported, bytes):
        raise TypeError("The mitered-cornice fixture could not be serialized.")
    return exported


def _corner_wall_surface(
    surface_id: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> FixedSurface:
    start_vertex = np.asarray(start, dtype=float)
    end_vertex = np.asarray(end, dtype=float)
    top_offset = np.asarray((0.0, 0.0, 3.0), dtype=float)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            (
                start_vertex,
                end_vertex,
                end_vertex + top_offset,
                start_vertex + top_offset,
            ),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id=surface_id,
        surface_type="wall",
        level_index=2,
        room_index=None,
        mesh=mesh,
        area_square_meters=float(mesh.area),
        wall_key=surface_id.rsplit("wall:", 1)[1],
        wall_start_world=start,
        wall_end_world=end,
        wall_height_meters=3.0,
    )


def _world_mesh_for_placement(placement: object) -> trimesh.Trimesh:
    mesh = placement.model.mesh.copy()  # type: ignore[attr-defined]
    mesh.apply_transform(build_placed_generated_model_transform(placement))
    return mesh


def _reverse_surface_faces(surface: FixedSurface) -> FixedSurface:
    """Return the same wall face with its rendered side reversed."""

    mesh = surface.mesh.copy()
    mesh.faces = np.ascontiguousarray(
        np.asarray(mesh.faces, dtype=np.int64)[:, ::-1]
    )
    return replace(surface, mesh=mesh)


def _apply_test_texture(glb_bytes: bytes) -> bytes:
    """Attach one visible material without changing authored UV coordinates."""

    scene = import_generated_glb(glb_bytes).scene
    for geometry in scene.geometry.values():
        uv_coordinates = np.asarray(geometry.visual.uv, dtype=float)
        geometry.visual = TextureVisuals(
            uv=uv_coordinates.copy(),
            material=PBRMaterial(
                name="rounded_cornice_material",
                baseColorTexture=Image.new(
                    "RGBA",
                    (8, 8),
                    (90, 130, 175, 255),
                ),
            ),
        )
    exported = scene.export(file_type="glb")
    if not isinstance(exported, bytes):
        raise TypeError("The rounded-cornice fixture could not be serialized.")
    return exported


def _geometry_by_material_name(
    scene: trimesh.Scene,
) -> dict[str, trimesh.Trimesh]:
    result: dict[str, trimesh.Trimesh] = {}
    for geometry in scene.geometry.values():
        material = getattr(geometry.visual, "material", None)
        material_name = str(getattr(material, "name", ""))
        result[material_name] = geometry
    return result


def _uv_counts(mesh: trimesh.Trimesh) -> Counter[tuple[float, float]]:
    uv_coordinates = np.asarray(mesh.visual.uv, dtype=float)
    return Counter(
        tuple(float(value) for value in row)
        for row in np.round(uv_coordinates, decimals=7)
    )


# ### GLB fitting tests ###
class ArchitecturalTrimGlbFitTests(unittest.TestCase):
    def test_fit_uses_requested_extents_and_bottom_center_origin(self) -> None:
        target_extents = np.asarray((8.0, 0.5, 2.5), dtype=float)

        fitted_glb = fit_architectural_trim_glb_to_dimensions(
            _translated_box_glb(),
            target_extents,
        )

        fitted_model = import_generated_glb(fitted_glb)
        expected_bounds = np.asarray(
            (
                (-4.0, -0.25, 0.0),
                (4.0, 0.25, 2.5),
            ),
            dtype=float,
        )
        np.testing.assert_allclose(
            fitted_model.mesh.bounds,
            expected_bounds,
            rtol=1e-6,
            atol=1e-7,
        )
        for node_name in fitted_model.scene.graph.nodes_geometry:
            node_transform, _geometry_name = fitted_model.scene.graph.get(
                node_name
            )
            np.testing.assert_allclose(
                node_transform,
                np.eye(4, dtype=float),
                rtol=0.0,
                atol=1e-9,
            )

    def test_fitted_model_placement_uses_identity_axis_scale(self) -> None:
        target_extents = (8.0, 0.5, 2.5)
        fitted_model = import_generated_glb(
            fit_architectural_trim_glb_to_dimensions(
                _translated_box_glb(),
                target_extents,
            )
        )
        frame = ArchitecturalTrimComponentFrame(
            level_index=2,
            trim_id="trim-one",
            world_bottom_center=(10.0, 20.0, 3.0),
            yaw_degrees=90.0,
            target_extents=target_extents,
        )

        placement = build_architectural_trim_component_placement(
            fitted_model,
            frame,
            object_id="generated-cornice",
            object_name="Cornice",
        )

        self.assertEqual(placement.axis_scales, (1.0, 1.0, 1.0))
        self.assertEqual(placement.world_position, frame.world_bottom_center)
        self.assertEqual(placement.rotation_degrees, (0.0, 0.0, 90.0))


# ### Repeated-module tests ###
class ArchitecturalTrimRepeatModuleTests(unittest.TestCase):
    def test_prepare_module_preserves_aspect_and_round_trips_layout(self) -> None:
        target_extents = (6.0, 0.5, 1.0)

        module_glb, layout = prepare_architectural_trim_repeat_module(
            _textured_two_primitive_glb(),
            target_extents,
        )

        self.assertEqual(layout.target_extents, target_extents)
        self.assertEqual(layout.module_extents, (1.0, 0.5, 1.0))
        self.assertEqual(layout.repeat_count, 6)
        self.assertEqual(
            layout.to_pipeline_dict(),
            {
                "version": 1,
                "targetExtents": [6.0, 0.5, 1.0],
                "moduleExtents": [1.0, 0.5, 1.0],
                "repeatCount": 6,
            },
        )
        self.assertEqual(
            ArchitecturalTrimRepeatLayout.from_pipeline_dict(
                layout.to_pipeline_dict()
            ),
            layout,
        )
        module_model = import_generated_glb(module_glb)
        np.testing.assert_allclose(
            module_model.mesh.bounds,
            ((-0.5, -0.25, 0.0), (0.5, 0.25, 1.0)),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_prepare_module_clamps_pathological_repeat_count(self) -> None:
        target_extents = (1_000.0, 0.5, 1.0)

        _module_glb, layout = prepare_architectural_trim_repeat_module(
            _textured_two_primitive_glb(),
            target_extents,
        )

        self.assertEqual(layout.repeat_count, 256)
        self.assertAlmostEqual(
            layout.module_extents[0] * layout.repeat_count,
            target_extents[0],
        )

    def test_prepare_module_can_bound_longitudinal_aspect_ratio(self) -> None:
        module_glb, layout = prepare_architectural_trim_repeat_module(
            _full_span_trim_glb(),
            (6.0, 0.1, 0.1),
            maximum_module_aspect_ratio=4.0,
        )

        self.assertEqual(layout.repeat_count, 15)
        self.assertLessEqual(
            layout.module_extents[0],
            max(layout.module_extents[1:]) * 4.0,
        )
        np.testing.assert_allclose(
            import_generated_glb(module_glb).mesh.bounds,
            ((-0.2, -0.05, 0.0), (0.2, 0.05, 0.1)),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_repeat_merges_each_primitive_and_reuses_uvs_and_materials(
        self,
    ) -> None:
        module_glb, layout = prepare_architectural_trim_repeat_module(
            _textured_two_primitive_glb(),
            (6.0, 0.5, 1.0),
        )
        module_model = import_generated_glb(module_glb)
        module_by_material = _geometry_by_material_name(module_model.scene)

        repeated_glb = repeat_architectural_trim_module_glb(
            module_glb,
            layout,
        )

        repeated_model = import_generated_glb(repeated_glb)
        repeated_by_material = _geometry_by_material_name(
            repeated_model.scene
        )
        np.testing.assert_allclose(
            repeated_model.mesh.bounds,
            ((-3.0, -0.25, 0.0), (3.0, 0.25, 1.0)),
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertEqual(
            set(repeated_by_material),
            {"trim_material_1", "trim_material_2"},
        )
        self.assertEqual(len(repeated_by_material), len(module_by_material))
        for material_name, module_geometry in module_by_material.items():
            with self.subTest(material_name=material_name):
                repeated_geometry = repeated_by_material[material_name]
                self.assertEqual(
                    len(repeated_geometry.faces),
                    len(module_geometry.faces) * layout.repeat_count,
                )
                np.testing.assert_array_equal(
                    np.asarray(
                        repeated_geometry.visual.material.baseColorTexture
                    ),
                    np.asarray(
                        module_geometry.visual.material.baseColorTexture
                    ),
                )
                self.assertEqual(
                    _uv_counts(repeated_geometry),
                    Counter(
                        {
                            coordinate: count * layout.repeat_count
                            for coordinate, count in _uv_counts(
                                module_geometry
                            ).items()
                        }
                    ),
                )

    def test_repeat_refits_provider_geometry_before_expanding_span(self) -> None:
        provider_glb = _textured_two_primitive_glb()
        provider_model = import_generated_glb(provider_glb)
        provider_by_material = _geometry_by_material_name(
            provider_model.scene
        )
        layout = ArchitecturalTrimRepeatLayout(
            target_extents=(6.0, 0.5, 1.0),
            module_extents=(1.0, 0.5, 1.0),
            repeat_count=6,
        )

        repeated_glb = repeat_architectural_trim_module_glb(
            provider_glb,
            layout,
        )

        repeated_model = import_generated_glb(repeated_glb)
        repeated_by_material = _geometry_by_material_name(
            repeated_model.scene
        )
        np.testing.assert_allclose(
            repeated_model.mesh.bounds,
            ((-3.0, -0.25, 0.0), (3.0, 0.25, 1.0)),
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertEqual(
            set(repeated_by_material),
            set(provider_by_material),
        )
        for material_name, provider_geometry in provider_by_material.items():
            with self.subTest(material_name=material_name):
                repeated_geometry = repeated_by_material[material_name]
                np.testing.assert_array_equal(
                    np.asarray(
                        repeated_geometry.visual.material.baseColorTexture
                    ),
                    np.asarray(
                        provider_geometry.visual.material.baseColorTexture
                    ),
                )
                self.assertEqual(
                    _uv_counts(repeated_geometry),
                    Counter(
                        {
                            coordinate: count * layout.repeat_count
                            for coordinate, count in _uv_counts(
                                provider_geometry
                            ).items()
                        }
                    ),
                )

    def test_repeated_model_placement_uses_identity_axis_scale(self) -> None:
        target_extents = (6.0, 0.5, 1.0)
        module_glb, layout = prepare_architectural_trim_repeat_module(
            _textured_two_primitive_glb(),
            target_extents,
        )
        repeated_model = import_generated_glb(
            repeat_architectural_trim_module_glb(module_glb, layout)
        )
        frame = ArchitecturalTrimComponentFrame(
            level_index=2,
            trim_id="repeated-trim",
            world_bottom_center=(1.0, 2.0, 3.0),
            yaw_degrees=45.0,
            target_extents=target_extents,
        )

        placement = build_architectural_trim_component_placement(
            repeated_model,
            frame,
            object_id="repeated-cornice",
            object_name="Repeated cornice",
        )

        self.assertEqual(placement.axis_scales, (1.0, 1.0, 1.0))

    def test_retarget_uses_nearest_source_module_count_and_preserves_uvs(
        self,
    ) -> None:
        source_module_glb, source_layout = prepare_architectural_trim_repeat_module(
            _textured_two_primitive_glb(),
            (6.0, 0.5, 1.0),
        )
        source_module = import_generated_glb(source_module_glb)
        source_by_material = _geometry_by_material_name(source_module.scene)
        target_extents = (4.6, 0.75, 1.5)

        target_glb, target_layout = (
            retarget_architectural_trim_repeat_module_glb(
                source_module_glb,
                source_layout,
                target_extents,
            )
        )

        self.assertEqual(target_layout.repeat_count, 5)
        np.testing.assert_allclose(
            target_layout.module_extents,
            (0.92, 0.75, 1.5),
        )
        target_model = import_generated_glb(target_glb)
        np.testing.assert_allclose(target_model.mesh.extents, target_extents)
        target_by_material = _geometry_by_material_name(target_model.scene)
        self.assertEqual(set(target_by_material), set(source_by_material))
        for material_name, source_geometry in source_by_material.items():
            with self.subTest(material_name=material_name):
                target_geometry = target_by_material[material_name]
                source_faces = np.asarray(source_geometry.faces, dtype=np.int64)
                target_faces = np.asarray(target_geometry.faces, dtype=np.int64)
                source_uvs = np.asarray(source_geometry.visual.uv, dtype=float)
                target_uvs = np.asarray(target_geometry.visual.uv, dtype=float)
                self.assertEqual(
                    len(target_faces),
                    len(source_faces) * target_layout.repeat_count,
                )
                self.assertEqual(
                    _uv_counts(target_geometry),
                    Counter(
                        {
                            coordinate: count * target_layout.repeat_count
                            for coordinate, count in _uv_counts(
                                source_geometry
                            ).items()
                        }
                    ),
                )
                for repeat_index in range(target_layout.repeat_count):
                    face_start = repeat_index * len(source_faces)
                    face_end = face_start + len(source_faces)
                    vertex_start = repeat_index * len(source_geometry.vertices)
                    vertex_end = vertex_start + len(source_geometry.vertices)
                    np.testing.assert_array_equal(
                        target_faces[face_start:face_end] - vertex_start,
                        source_faces,
                    )
                    np.testing.assert_allclose(
                        target_uvs[vertex_start:vertex_end],
                        source_uvs,
                        rtol=0.0,
                        atol=0.0,
                    )

    def test_retarget_clamps_repeat_count_and_rejects_invalid_dimensions(
        self,
    ) -> None:
        source_module_glb, source_layout = prepare_architectural_trim_repeat_module(
            _textured_two_primitive_glb(),
            (6.0, 0.5, 1.0),
        )

        target_glb, target_layout = (
            retarget_architectural_trim_repeat_module_glb(
                source_module_glb,
                source_layout,
                (1_000.0, 0.5, 1.0),
            )
        )

        self.assertEqual(target_layout.repeat_count, 256)
        np.testing.assert_allclose(
            import_generated_glb(target_glb).mesh.extents,
            target_layout.target_extents,
        )
        with self.assertRaisesRegex(ValueError, "three positive numbers"):
            retarget_architectural_trim_repeat_module_glb(
                source_module_glb,
                source_layout,
                (6.0, float("nan"), 1.0),
            )


# ### Generated corner-miter tests ###
class ArchitecturalTrimGeneratedMiterTests(unittest.TestCase):
    def test_textured_outer_corner_closes_without_rewriting_uvs(self) -> None:
        surfaces = (
            _corner_wall_surface(
                "level:2/wall:1:2",
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
            ),
            _corner_wall_surface(
                "level:2/wall:2:3",
                (2.0, 0.0, 0.0),
                (2.0, 2.0, 0.0),
            ),
        )
        trims = (
            ArchitecturalTrimData(
                trim_id="a" * 32,
                kind=TRIM_KIND_CORNICE,
                wall_surface_ids=(surfaces[0].surface_id,),
                depth_meters=0.5,
                height_meters=1.0,
            ),
            ArchitecturalTrimData(
                trim_id="b" * 32,
                kind=TRIM_KIND_CORNICE,
                wall_surface_ids=(surfaces[1].surface_id,),
                depth_meters=0.5,
                height_meters=1.0,
            ),
        )
        level = LevelData(index=2, name="Ground")
        level.architectural_trims.extend(trims)
        descriptors = {
            descriptor.trim_id: descriptor
            for descriptor in build_cornice_miter_descriptors((level,), surfaces)
        }
        source_model = import_generated_glb(
            _textured_box_glb((2.0, 0.5, 1.0))
        )
        source_geometry = next(iter(source_model.scene.geometry.values()))
        source_uv_counts = _uv_counts(source_geometry)

        placements = tuple(
            build_architectural_trim_component_placement(
                source_model,
                build_architectural_trim_component_frame(
                    level,
                    trim,
                    (surface,),
                ),
                object_id=f"cornice-{index}",
                object_name=f"Cornice {index}",
                cornice_miter=descriptors[trim.trim_id],
            )
            for index, (trim, surface) in enumerate(
                zip(trims, surfaces, strict=True),
                start=1,
            )
        )

        world_meshes = tuple(
            _world_mesh_for_placement(placement) for placement in placements
        )
        expected_outer_top = np.asarray((2.5, -0.5, 3.0), dtype=float)
        expected_outer_bottom = np.asarray((2.5, -0.5, 2.0), dtype=float)
        for placement, world_mesh in zip(placements, world_meshes, strict=True):
            with self.subTest(object_id=placement.object_id):
                vertices = np.asarray(world_mesh.vertices, dtype=float)
                self.assertLess(
                    float(np.min(np.linalg.norm(vertices - expected_outer_top, axis=1))),
                    1e-6,
                )
                self.assertLess(
                    float(
                        np.min(
                            np.linalg.norm(
                                vertices - expected_outer_bottom,
                                axis=1,
                            )
                        )
                    ),
                    1e-6,
                )
                geometry = next(iter(placement.model.scene.geometry.values()))
                self.assertEqual(
                    getattr(geometry.visual.material, "name", ""),
                    "mitered_cornice_material",
                )
                self.assertEqual(_uv_counts(geometry), source_uv_counts)

        vertex_blocks: list[np.ndarray] = []
        face_blocks: list[np.ndarray] = []
        vertex_offset = 0
        for world_mesh in world_meshes:
            vertex_blocks.append(np.asarray(world_mesh.vertices, dtype=float))
            face_blocks.append(
                np.asarray(world_mesh.faces, dtype=np.int64) + vertex_offset
            )
            vertex_offset += len(world_mesh.vertices)
        joined_mesh = trimesh.Trimesh(
            vertices=np.vstack(vertex_blocks),
            faces=np.vstack(face_blocks),
            process=True,
        )
        self.assertTrue(joined_mesh.is_watertight)
        self.assertTrue(joined_mesh.is_winding_consistent)

    def test_rounded_repeated_cornice_matches_both_wall_orientations(
        self,
    ) -> None:
        base_surfaces = (
            _corner_wall_surface(
                "level:2/wall:1:2",
                (0.0, 0.0, 0.0),
                (6.0, 0.0, 0.0),
            ),
            _corner_wall_surface(
                "level:2/wall:2:3",
                (6.0, 0.0, 0.0),
                (6.0, 2.25, 0.0),
            ),
        )
        for reverse_faces in (False, True):
            with self.subTest(reverse_faces=reverse_faces):
                surfaces = (
                    tuple(_reverse_surface_faces(item) for item in base_surfaces)
                    if reverse_faces
                    else base_surfaces
                )
                trims = (
                    ArchitecturalTrimData(
                        trim_id="c" * 32,
                        kind=TRIM_KIND_CORNICE,
                        wall_surface_ids=(surfaces[0].surface_id,),
                        depth_meters=0.5,
                        height_meters=1.0,
                        corner_radius_meters=0.2,
                    ),
                    ArchitecturalTrimData(
                        trim_id="d" * 32,
                        kind=TRIM_KIND_CORNICE,
                        wall_surface_ids=(surfaces[1].surface_id,),
                        depth_meters=0.5,
                        height_meters=1.0,
                        corner_radius_meters=0.2,
                    ),
                )
                level = LevelData(index=2, name="Ground")
                level.architectural_trims.extend(trims)
                descriptors = {
                    descriptor.trim_id: descriptor
                    for descriptor in build_cornice_miter_descriptors(
                        (level,),
                        surfaces,
                    )
                }

                seed_model, source_frame = (
                    build_architectural_trim_component_seed_model(
                        level,
                        trims[0],
                        (surfaces[0],),
                    )
                )
                module_glb, source_layout = (
                    prepare_architectural_trim_repeat_module(
                        seed_model.glb_bytes,
                        source_frame.target_extents,
                        maximum_module_aspect_ratio=4.0,
                    )
                )
                self.assertGreater(source_layout.repeat_count, 1)
                textured_module_glb = _apply_test_texture(
                    build_cornice_segmented_uv_glb(
                        module_glb,
                        texture_resolution=512,
                    )
                )
                source_model = import_generated_glb(
                    repeat_architectural_trim_module_glb(
                        textured_module_glb,
                        source_layout,
                    )
                )
                target_frame = build_architectural_trim_component_frame(
                    level,
                    trims[1],
                    (surfaces[1],),
                )
                target_glb, target_layout = (
                    retarget_architectural_trim_repeat_module_glb(
                        textured_module_glb,
                        source_layout,
                        target_frame.target_extents,
                    )
                )
                self.assertEqual(target_layout.repeat_count, 1)
                target_model = import_generated_glb(target_glb)
                models_and_frames = (
                    (source_model, source_frame),
                    (target_model, target_frame),
                )

                placements = tuple(
                    build_architectural_trim_component_placement(
                        model,
                        frame,
                        object_id=f"rounded-cornice-{index}",
                        object_name=f"Rounded cornice {index}",
                        cornice_miter=descriptors[trim.trim_id],
                    )
                    for index, (trim, (model, frame)) in enumerate(
                        zip(trims, models_and_frames, strict=True),
                        start=1,
                    )
                )

                for source, placement in zip(
                    (source_model, target_model),
                    placements,
                    strict=True,
                ):
                    original_uvs: Counter[tuple[float, float]] = Counter()
                    for geometry in source.scene.geometry.values():
                        original_uvs.update(_uv_counts(geometry))
                    retained_uvs: Counter[tuple[float, float]] = Counter()
                    for geometry in placement.model.scene.geometry.values():
                        self.assertEqual(
                            getattr(geometry.visual.material, "name", ""),
                            "rounded_cornice_material",
                        )
                        retained_uvs.update(_uv_counts(geometry))
                    self.assertTrue(retained_uvs)
                    self.assertEqual(retained_uvs & original_uvs, retained_uvs)

                world_meshes = tuple(
                    _world_mesh_for_placement(placement)
                    for placement in placements
                )
                vertex_blocks: list[np.ndarray] = []
                face_blocks: list[np.ndarray] = []
                vertex_offset = 0
                for world_mesh in world_meshes:
                    vertex_blocks.append(
                        np.asarray(world_mesh.vertices, dtype=float)
                    )
                    face_blocks.append(
                        np.asarray(world_mesh.faces, dtype=np.int64)
                        + vertex_offset
                    )
                    vertex_offset += len(world_mesh.vertices)
                joined_mesh = trimesh.Trimesh(
                    vertices=np.vstack(vertex_blocks),
                    faces=np.vstack(face_blocks),
                    process=True,
                )
                joined_edges = np.sort(joined_mesh.edges, axis=1)
                _unique_edges, edge_counts = np.unique(
                    joined_edges,
                    axis=0,
                    return_counts=True,
                )
                # Repeated modules intentionally retain coincident internal
                # caps, so those edges have four incident triangles. A visual
                # corner gap would instead leave one-triangle boundary edges.
                self.assertFalse(np.any(edge_counts == 1))
                self.assertTrue(joined_mesh.is_winding_consistent)

                procedural = build_architectural_trim_geometry(
                    (level,),
                    surfaces,
                )
                procedural_mesh = trimesh.util.concatenate(
                    tuple(part.mesh for part in procedural.parts)
                )
                procedural_mesh.merge_vertices()
                self.assertTrue(procedural_mesh.is_watertight)
                np.testing.assert_allclose(
                    joined_mesh.bounds,
                    procedural_mesh.bounds,
                    atol=1e-6,
                )
                self.assertAlmostEqual(
                    abs(float(joined_mesh.volume)),
                    abs(float(procedural_mesh.volume)),
                    places=5,
                )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
