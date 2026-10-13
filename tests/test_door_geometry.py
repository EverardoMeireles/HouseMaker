# ### Imports ###
from __future__ import annotations

import copy
import unittest
from dataclasses import replace

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.door_geometry import (
    DoorBodyMirrorConfiguration,
    assemble_door_model,
    assemble_window_model,
    build_default_door_slot_models,
    build_default_window_slot_models,
    build_door_body_model,
    mirror_door_model_horizontally,
)
from housemaker.door_state import (
    DOOR_SLOT_BACK_BODY,
    DOOR_SLOT_BACK_KNOB,
    DOOR_SLOT_BACK_LEFT_BODY,
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    DOOR_SLOT_LEFT_BODY,
    DOOR_SLOT_LEFT_KNOB,
    create_door_definition_for_doorway,
    create_glass_area_data,
    create_window_definition_for_frame,
)
from housemaker.doorway_geometry import build_doorway_cross_section_outline
from housemaker.glass_material import is_housemaker_glass_material
from housemaker.glb import (
    GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
    GeneratedModel,
    PlacedGeneratedModel,
    compose_placed_generated_models,
    import_generated_glb,
)
from housemaker.models import (
    DOORWAY_SHAPE_ARCH,
    DOORWAY_SHAPE_RECTANGULAR,
    DoorwayData,
)


# ### Geometry fixtures ###
def _doorway(*, shape: str, arch_amount: float = 1.0) -> DoorwayData:
    return DoorwayData(
        doorway_id=f"{shape}-doorway",
        center_x=0.0,
        center_y=0.0,
        width_meters=1.0,
        height_meters=2.2,
        depth_meters=0.2,
        shape=shape,
        arch_amount=arch_amount,
    )


def _with_solid_texture(
    model: GeneratedModel,
    rgba: tuple[int, int, int, int],
) -> GeneratedModel:
    """Return one equivalent model carrying an embedded base-color image."""

    textured_mesh = copy.deepcopy(model.mesh)
    textured_mesh.visual = TextureVisuals(
        uv=np.asarray(model.mesh.visual.uv, dtype=float).copy(),
        material=PBRMaterial(
            name="door_body_test_texture",
            baseColorTexture=Image.new("RGBA", (8, 8), rgba),
        ),
    )
    export_mesh = copy.deepcopy(textured_mesh)
    export_mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    scene = trimesh.Scene()
    scene.add_geometry(
        export_mesh,
        geom_name="textured_door_body",
        node_name="textured_door_body",
    )
    return import_generated_glb(bytes(scene.export(file_type="glb")))


def _generated_hardware_fixture() -> GeneratedModel:
    """Build a small stand-in for hardware that Generation would supply."""

    return build_door_body_model(
        0.08,
        0.12,
        0.04,
        shape=DOORWAY_SHAPE_RECTANGULAR,
        arch_amount=0.0,
        geometry_name="generated_hardware_fixture",
    )


def _translated_model_x(
    model: GeneratedModel,
    offset_meters: float,
) -> GeneratedModel:
    """Return an intentionally asymmetric model for mirror assertions."""

    translated_mesh = copy.deepcopy(model.mesh)
    translated_mesh.apply_translation((offset_meters, 0.0, 0.0))
    translated_scene = copy.deepcopy(model.scene)
    translated_scene.apply_translation((offset_meters, 0.0, 0.0))
    return replace(
        model,
        mesh=translated_mesh,
        scene=translated_scene,
        glb_bytes=bytes(translated_scene.export(file_type="glb")),
    )


def _translated_model_y(
    model: GeneratedModel,
    offset_meters: float,
) -> GeneratedModel:
    """Return a model whose source anchor is intentionally noncanonical."""

    translated_mesh = copy.deepcopy(model.mesh)
    translated_mesh.apply_translation((0.0, offset_meters, 0.0))
    translation = np.eye(4, dtype=float)
    translation[1, 3] = offset_meters
    translated_scene = copy.deepcopy(model.scene)
    translated_scene.apply_transform(
        Z_UP_TO_GLTF_Y_UP_TRANSFORM
        @ translation
        @ GLTF_Y_UP_TO_Z_UP_TRANSFORM
    )
    return replace(
        model,
        mesh=translated_mesh,
        scene=translated_scene,
        glb_bytes=bytes(translated_scene.export(file_type="glb")),
    )


def _front_facing_hardware_fixture() -> GeneratedModel:
    """Build a one-sided marker whose face normal points toward local -Y."""

    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            (
                (-0.02, 0.0, 0.0),
                (0.02, 0.0, 0.0),
                (0.02, 0.0, 0.04),
                (-0.02, 0.0, 0.04),
            ),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    export_mesh = mesh.copy()
    export_mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    scene = trimesh.Scene(export_mesh)
    return GeneratedModel(
        mesh=mesh,
        scene=scene,
        glb_bytes=bytes(scene.export(file_type="glb")),
    )


# ### UV assertions ###
def _assert_complete_uvs(
    case: unittest.TestCase,
    mesh: object,
) -> None:
    uv = np.asarray(getattr(getattr(mesh, "visual", None), "uv", None))
    vertices = np.asarray(getattr(mesh, "vertices", ()))
    case.assertEqual(uv.shape, (len(vertices), 2))
    case.assertTrue(np.all(np.isfinite(uv)))
    case.assertTrue(np.all(uv >= -1e-9))
    case.assertTrue(np.all(uv <= 1.0 + 1e-9))


def _assert_geometrically_watertight(
    case: unittest.TestCase,
    mesh: object,
) -> None:
    welded = mesh.copy()
    welded.merge_vertices(merge_tex=True)
    case.assertTrue(welded.is_watertight)


# ### Door body tests ###
class DoorBodyGeometryTests(unittest.TestCase):
    def test_rectangular_body_has_exact_bounds_and_is_watertight(self) -> None:
        model = build_door_body_model(
            1.0,
            2.2,
            0.04,
            shape=DOORWAY_SHAPE_RECTANGULAR,
            arch_amount=1.0,
        )

        np.testing.assert_allclose(
            model.mesh.bounds,
            ((-0.5, -0.02, 0.0), (0.5, 0.02, 2.2)),
            atol=1e-9,
        )
        _assert_geometrically_watertight(self, model.mesh)
        self.assertGreater(len(model.glb_bytes), 0)
        imported = import_generated_glb(model.glb_bytes)
        np.testing.assert_allclose(imported.mesh.bounds, model.mesh.bounds, atol=1e-7)
        _assert_complete_uvs(self, model.mesh)
        _assert_complete_uvs(self, imported.mesh)

    def test_arch_body_retains_requested_silhouette_and_dimensions(self) -> None:
        model = build_door_body_model(
            1.0,
            2.2,
            0.05,
            shape=DOORWAY_SHAPE_ARCH,
            arch_amount=0.75,
        )
        vertices = np.asarray(model.mesh.vertices, dtype=float)

        np.testing.assert_allclose(
            model.mesh.bounds,
            ((-0.5, -0.025, 0.0), (0.5, 0.025, 2.2)),
            atol=1e-9,
        )
        _assert_geometrically_watertight(self, model.mesh)
        top_vertices = vertices[np.isclose(vertices[:, 2], 2.2, atol=1e-9)]
        self.assertTrue(np.allclose(top_vertices[:, 0], 0.0, atol=1e-7))
        _assert_complete_uvs(self, model.mesh)

    def test_only_default_body_model_has_complete_finite_uvs(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_ARCH, arch_amount=0.7),
            door_id="door-with-uvs",
            name="New door 1",
        )

        slot_models = build_default_door_slot_models(door)

        self.assertEqual(set(slot_models), {"body"})
        for model in slot_models.values():
            with self.subTest(model_bounds=model.mesh.bounds.tolist()):
                _assert_complete_uvs(self, model.mesh)
                imported = import_generated_glb(model.glb_bytes)
                _assert_complete_uvs(self, imported.mesh)


# ### Door assembly and mirror tests ###
class DoorAssemblyGeometryTests(unittest.TestCase):
    def test_glass_area_cuts_and_tags_the_complete_body_assembly(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="door-with-glass",
            name="New door 1",
        )
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        area = create_glass_area_data(
            body.source_object_id,
            (),
            minimum_x_ratio=0.25,
            maximum_x_ratio=0.75,
            minimum_z_ratio=0.3,
            maximum_z_ratio=0.7,
            glass_area_id="glass-a",
        )
        door = replace(door, glass_areas=(area,))

        assembled = assemble_door_model(door)

        self.assertGreater(len(assembled.glb_bytes), 0)
        self.assertTrue(
            any(
                is_housemaker_glass_material(geometry.visual.material)
                for geometry in assembled.scene.geometry.values()
                if isinstance(geometry, trimesh.Trimesh)
                and isinstance(geometry.visual, TextureVisuals)
            )
        )
        self.assertEqual(
            sum(
                int(is_housemaker_glass_material(geometry.visual.material))
                for geometry in assembled.scene.geometry.values()
                if isinstance(geometry, trimesh.Trimesh)
                and isinstance(geometry.visual, TextureVisuals)
            ),
            1,
        )

    def test_glass_area_is_visible_in_preview_only_body_assembly(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="preview-door-with-glass",
            name="New door 1",
        )
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        door = replace(
            door,
            glass_areas=(
                create_glass_area_data(
                    body.source_object_id,
                    (),
                    minimum_x_ratio=0.2,
                    maximum_x_ratio=0.8,
                    minimum_z_ratio=0.2,
                    maximum_z_ratio=0.8,
                    glass_area_id="preview-glass",
                ),
            ),
        )
        source = build_default_door_slot_models(door)[DOOR_SLOT_BODY]
        preview_source = replace(source, preview_only=True)

        assembled = assemble_door_model(
            door,
            {body.source_object_id: preview_source},
        )

        self.assertTrue(assembled.preview_only)
        self.assertTrue(
            any(
                is_housemaker_glass_material(mesh.visual.material)
                for preview in assembled.preview_placed_objects
                for mesh in preview.meshes
                if isinstance(mesh.visual, TextureVisuals)
            )
        )

    def test_double_sided_glass_omits_both_coincident_center_caps(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="double-sided-glass-door",
            name="New door 1",
        ).with_double_sided(True)
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        door = replace(
            door,
            glass_areas=(
                create_glass_area_data(
                    body.source_object_id,
                    (),
                    minimum_x_ratio=0.25,
                    maximum_x_ratio=0.75,
                    minimum_z_ratio=0.25,
                    maximum_z_ratio=0.75,
                    glass_area_id="shared-pane",
                ),
            ),
        )

        assembled = assemble_door_model(door, include_unjoined=True)

        glass_meshes: list[trimesh.Trimesh] = []
        for preview in assembled.preview_placed_objects:
            for mesh in preview.meshes:
                if not is_housemaker_glass_material(mesh.visual.material):
                    continue
                transformed = mesh.copy()
                transformed.apply_transform(preview.placement_transform)
                glass_meshes.append(transformed)
        self.assertEqual(len(glass_meshes), 2)
        self.assertTrue(all(len(mesh.faces) == 10 for mesh in glass_meshes))
        for mesh in glass_meshes:
            vertices = np.asarray(mesh.vertices, dtype=float)
            face_vertices = vertices[np.asarray(mesh.faces, dtype=np.int64)]
            center_faces = np.all(
                np.isclose(face_vertices[:, :, 1], 0.0, atol=1e-9),
                axis=1,
            )
            self.assertFalse(np.any(center_faces))
        self.assertAlmostEqual(
            min(float(mesh.bounds[0, 1]) for mesh in glass_meshes),
            -door.thickness_meters * 0.5,
        )
        self.assertAlmostEqual(
            max(float(mesh.bounds[1, 1]) for mesh in glass_meshes),
            door.thickness_meters * 0.5,
        )

    def test_double_door_assembles_two_independent_half_width_bodies(
        self,
    ) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="double-door",
            name="New door 1",
        ).with_double(True)
        right_body = door.get_slot(DOOR_SLOT_BODY)
        left_body = door.get_slot(DOOR_SLOT_LEFT_BODY)
        assert right_body is not None and left_body is not None

        assembled = assemble_door_model(door)

        body_previews = {
            preview.object_id: preview
            for preview in assembled.preview_placed_objects
        }
        self.assertEqual(
            set(body_previews),
            {right_body.source_object_id, left_body.source_object_id},
        )
        right_mesh = body_previews[right_body.source_object_id].meshes[0].copy()
        right_mesh.apply_transform(
            body_previews[right_body.source_object_id].placement_transform
        )
        left_mesh = body_previews[left_body.source_object_id].meshes[0].copy()
        left_mesh.apply_transform(
            body_previews[left_body.source_object_id].placement_transform
        )
        np.testing.assert_allclose(
            right_mesh.bounds[:, 0],
            (0.0, door.width_meters * 0.5),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            left_mesh.bounds[:, 0],
            (-door.width_meters * 0.5, 0.0),
            atol=1e-8,
        )
        self.assertAlmostEqual(assembled.mesh.extents[0], door.width_meters)

    def test_double_arch_bodies_reconstruct_one_global_profile(self) -> None:
        doorway = _doorway(shape=DOORWAY_SHAPE_ARCH, arch_amount=0.75)
        door = create_door_definition_for_doorway(
            doorway,
            door_id="double-arch-door",
            name="New door 1",
        ).with_double(True)

        assembled = assemble_door_model(door)

        transformed_vertices: list[np.ndarray] = []
        for preview in assembled.preview_placed_objects:
            for mesh in preview.meshes:
                transformed = mesh.copy()
                transformed.apply_transform(preview.placement_transform)
                transformed_vertices.append(
                    np.asarray(transformed.vertices, dtype=float)
                )
        vertices = np.concatenate(transformed_vertices, axis=0)
        profile_vertices = vertices[:, (0, 2)]
        expected_outline = build_doorway_cross_section_outline(
            door.width_meters,
            door.height_meters,
            door.shape,
            door.arch_amount,
        )
        for expected_point in expected_outline[:-1]:
            with self.subTest(expected_point=expected_point):
                self.assertTrue(
                    np.any(
                        np.all(
                            np.isclose(
                                profile_vertices,
                                expected_point,
                                atol=1e-7,
                            ),
                            axis=1,
                        )
                    )
                )

        half_width = door.width_meters / 2.0
        arch_rise = min(half_width, door.height_meters) * door.arch_amount
        spring_height = door.height_meters - arch_rise
        normalized_widths = np.clip(vertices[:, 0] / half_width, -1.0, 1.0)
        arch_ceiling = spring_height + arch_rise * np.sqrt(
            np.maximum(0.0, 1.0 - normalized_widths**2)
        )
        self.assertTrue(np.all(vertices[:, 2] <= arch_ceiling + 1e-7))

    def test_double_symmetric_half_bodies_reconstruct_without_a_center_gap(
        self,
    ) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="symmetric-half-double-door",
            name="New door 1",
        ).with_double(True)
        right_body = door.get_slot(DOOR_SLOT_BODY)
        left_body = door.get_slot(DOOR_SLOT_LEFT_BODY)
        assert right_body is not None and left_body is not None
        retained_half = _translated_model_x(
            build_door_body_model(
                door.width_meters * 0.5,
                door.height_meters,
                door.thickness_meters,
                shape=DOORWAY_SHAPE_RECTANGULAR,
                arch_amount=0.0,
            ),
            -door.width_meters * 0.25,
        )
        mirrors = {
            slot_id: DoorBodyMirrorConfiguration(
                symmetric_orientation="vertical",
                symmetric_plane_coordinate=0.0,
            )
            for slot_id in (DOOR_SLOT_BODY, DOOR_SLOT_LEFT_BODY)
        }

        assembled = assemble_door_model(
            door,
            {
                right_body.source_object_id: retained_half,
                left_body.source_object_id: retained_half,
            },
            body_mirrors=mirrors,
        )

        reconstructed_bounds: dict[str, np.ndarray] = {}
        for preview in assembled.preview_symmetric_objects:
            if preview.object_id not in {
                right_body.source_object_id,
                left_body.source_object_id,
            }:
                continue
            vertices = np.concatenate(
                tuple(
                    np.asarray(mesh.vertices, dtype=float)
                    for mesh in (*preview.meshes, *preview.mirrored_meshes)
                ),
                axis=0,
            )
            reconstructed_bounds[preview.object_id] = np.stack(
                (np.min(vertices, axis=0), np.max(vertices, axis=0))
            )

        np.testing.assert_allclose(
            reconstructed_bounds[left_body.source_object_id][:, 0],
            (-door.width_meters * 0.5, 0.0),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            reconstructed_bounds[right_body.source_object_id][:, 0],
            (0.0, door.width_meters * 0.5),
            atol=1e-8,
        )

    def test_double_door_pasted_handle_mirror_is_applied_during_assembly(
        self,
    ) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="double-door",
            name="New door 1",
        ).with_double(True)
        right_handle = door.get_slot(DOOR_SLOT_KNOB)
        left_handle = door.get_slot(DOOR_SLOT_LEFT_KNOB)
        assert right_handle is not None and left_handle is not None
        door = door.replace_slot(right_handle.with_transform(joined=True))
        door = door.replace_slot(left_handle.with_transform(joined=True))
        right_handle = door.get_slot(DOOR_SLOT_KNOB)
        left_handle = door.get_slot(DOOR_SLOT_LEFT_KNOB)
        assert right_handle is not None and left_handle is not None
        asymmetric_hardware = _translated_model_x(
            _generated_hardware_fixture(),
            0.025,
        )
        slot_models = build_default_door_slot_models(door)
        slot_models[right_handle.source_object_id] = asymmetric_hardware
        slot_models[left_handle.source_object_id] = asymmetric_hardware

        assembled = assemble_door_model(door, slot_models)

        previews = {
            preview.object_id: preview
            for preview in assembled.preview_placed_objects
        }
        right_bounds = previews[right_handle.source_object_id].meshes[0].bounds
        left_bounds = previews[left_handle.source_object_id].meshes[0].bounds
        np.testing.assert_allclose(
            left_bounds[:, 0],
            -right_bounds[::-1, 0],
            atol=1e-8,
        )

    def test_double_double_sided_door_assembles_four_leaf_bodies(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="complete-double-door",
            name="New door 1",
        ).with_double(True).with_double_sided(True)
        body_slot_ids = {
            DOOR_SLOT_BODY,
            DOOR_SLOT_LEFT_BODY,
            DOOR_SLOT_BACK_BODY,
            DOOR_SLOT_BACK_LEFT_BODY,
        }

        assembled = assemble_door_model(door)

        expected_object_ids = {
            door.get_slot(slot_id).source_object_id  # type: ignore[union-attr]
            for slot_id in body_slot_ids
        }
        self.assertEqual(
            {
                preview.object_id
                for preview in assembled.preview_placed_objects
            },
            expected_object_ids,
        )
        self.assertAlmostEqual(assembled.mesh.extents[0], door.width_meters)
        self.assertAlmostEqual(
            assembled.mesh.extents[1],
            door.thickness_meters,
        )
        transformed_bounds = {}
        for preview in assembled.preview_placed_objects:
            if preview.object_id not in expected_object_ids:
                continue
            mesh = preview.meshes[0].copy()
            mesh.apply_transform(preview.placement_transform)
            transformed_bounds[preview.object_id] = mesh.bounds
        right_front = door.get_slot(DOOR_SLOT_BODY)
        left_front = door.get_slot(DOOR_SLOT_LEFT_BODY)
        right_back = door.get_slot(DOOR_SLOT_BACK_BODY)
        left_back = door.get_slot(DOOR_SLOT_BACK_LEFT_BODY)
        assert right_front is not None and left_front is not None
        assert right_back is not None and left_back is not None
        self.assertAlmostEqual(
            transformed_bounds[left_front.source_object_id][1, 0],
            transformed_bounds[right_front.source_object_id][0, 0],
        )
        self.assertAlmostEqual(
            transformed_bounds[right_front.source_object_id][1, 1],
            transformed_bounds[right_back.source_object_id][0, 1],
        )
        self.assertAlmostEqual(
            transformed_bounds[left_front.source_object_id][1, 1],
            transformed_bounds[left_back.source_object_id][0, 1],
        )

    def test_legacy_front_bodies_are_fitted_before_double_leaf_placement(
        self,
    ) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="legacy-double-door",
            name="New door 1",
        ).with_double(True).with_double_sided(True)
        legacy_front = build_door_body_model(
            door.width_meters * 0.4,
            door.height_meters * 0.8,
            door.thickness_meters * 0.6,
            shape=DOORWAY_SHAPE_RECTANGULAR,
            arch_amount=0.0,
        )
        slot_models = build_default_door_slot_models(door)
        slot_models[DOOR_SLOT_BODY] = legacy_front
        slot_models[DOOR_SLOT_LEFT_BODY] = legacy_front

        assembled = assemble_door_model(door, slot_models)

        bounds_by_slot = {}
        for slot_id in (
            DOOR_SLOT_BODY,
            DOOR_SLOT_LEFT_BODY,
            DOOR_SLOT_BACK_BODY,
            DOOR_SLOT_BACK_LEFT_BODY,
        ):
            slot = door.get_slot(slot_id)
            assert slot is not None
            preview = next(
                candidate
                for candidate in assembled.preview_placed_objects
                if candidate.object_id == slot.source_object_id
            )
            mesh = preview.meshes[0].copy()
            mesh.apply_transform(preview.placement_transform)
            bounds_by_slot[slot_id] = mesh.bounds

        self.assertAlmostEqual(
            bounds_by_slot[DOOR_SLOT_LEFT_BODY][1, 0],
            bounds_by_slot[DOOR_SLOT_BODY][0, 0],
        )
        np.testing.assert_allclose(
            (
                bounds_by_slot[DOOR_SLOT_LEFT_BODY][0, 0],
                bounds_by_slot[DOOR_SLOT_BODY][1, 0],
            ),
            (-door.width_meters * 0.5, door.width_meters * 0.5),
            atol=1e-8,
        )
        self.assertAlmostEqual(
            bounds_by_slot[DOOR_SLOT_BACK_LEFT_BODY][1, 0],
            bounds_by_slot[DOOR_SLOT_BACK_BODY][0, 0],
        )

    def test_double_door_glass_is_limited_to_its_leaf_on_both_sides(
        self,
    ) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="double-door-glass",
            name="New door 1",
        ).with_double(True).with_double_sided(True)
        right_body = door.get_slot(DOOR_SLOT_BODY)
        left_body = door.get_slot(DOOR_SLOT_LEFT_BODY)
        assert right_body is not None and left_body is not None
        door = replace(
            door,
            glass_areas=(
                create_glass_area_data(
                    door.door_id,
                    (),
                    minimum_x_ratio=0.2,
                    maximum_x_ratio=0.8,
                    minimum_z_ratio=0.2,
                    maximum_z_ratio=0.8,
                    glass_area_id="right-pane",
                    body_slot_id=DOOR_SLOT_BODY,
                ),
            ),
        )

        assembled = assemble_door_model(door)

        glass_owner_ids = {
            preview.object_id
            for preview in assembled.preview_placed_objects
            if any(
                is_housemaker_glass_material(mesh.visual.material)
                for mesh in preview.meshes
            )
        }
        self.assertEqual(
            glass_owner_ids,
            {
                right_body.source_object_id,
                door.get_slot(DOOR_SLOT_BACK_BODY).source_object_id,  # type: ignore[union-attr]
            },
        )
        self.assertNotIn(left_body.source_object_id, glass_owner_ids)

    def test_arch_glass_rectangle_must_fit_complete_body_silhouette(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_ARCH, arch_amount=1.0),
            door_id="arched-glass-door",
            name="New door 1",
        )
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        door = replace(
            door,
            glass_areas=(
                create_glass_area_data(
                    body.source_object_id,
                    (),
                    minimum_x_ratio=0.0,
                    maximum_x_ratio=0.3,
                    minimum_z_ratio=0.82,
                    maximum_z_ratio=0.98,
                    glass_area_id="outside-arch",
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "inside the body silhouette"):
            assemble_door_model(door)

    def test_double_sided_bodies_share_depth_without_overlapping(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="double-sided-door",
            name="New door 1",
        ).with_double_sided(True)
        slot_models = build_default_door_slot_models(door)

        assembled = assemble_door_model(
            door,
            slot_models,
            include_unjoined=True,
        )

        self.assertEqual(
            set(slot_models),
            {DOOR_SLOT_BODY, DOOR_SLOT_BACK_BODY},
        )
        transformed_bounds: dict[str, np.ndarray] = {}
        for preview in assembled.preview_placed_objects:
            if preview.object_id not in {
                door.get_slot(DOOR_SLOT_BODY).source_object_id,
                door.get_slot(DOOR_SLOT_BACK_BODY).source_object_id,
            }:
                continue
            mesh = preview.meshes[0].copy()
            mesh.apply_transform(preview.placement_transform)
            transformed_bounds[preview.object_id] = mesh.bounds
        front = door.get_slot(DOOR_SLOT_BODY)
        back = door.get_slot(DOOR_SLOT_BACK_BODY)
        assert front is not None and back is not None
        self.assertAlmostEqual(
            transformed_bounds[front.source_object_id][1, 1],
            transformed_bounds[back.source_object_id][0, 1],
        )
        self.assertAlmostEqual(
            assembled.mesh.extents[1],
            door.thickness_meters,
        )

    def test_double_sided_bodies_ignore_source_anchor_offsets(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="offset-double-sided-door",
            name="New door 1",
        ).with_double_sided(True)
        front = door.get_slot(DOOR_SLOT_BODY)
        back = door.get_slot(DOOR_SLOT_BACK_BODY)
        assert front is not None and back is not None
        defaults = build_default_door_slot_models(door)
        assembled = assemble_door_model(
            door,
            {
                front.source_object_id: _translated_model_y(
                    defaults[DOOR_SLOT_BODY],
                    -0.6,
                ),
                back.source_object_id: _translated_model_y(
                    defaults[DOOR_SLOT_BACK_BODY],
                    0.8,
                ),
            },
        )

        bounds_by_id = {}
        for preview in assembled.preview_placed_objects:
            mesh = preview.meshes[0].copy()
            mesh.apply_transform(preview.placement_transform)
            bounds_by_id[preview.object_id] = mesh.bounds

        self.assertAlmostEqual(
            bounds_by_id[front.source_object_id][1, 1],
            bounds_by_id[back.source_object_id][0, 1],
        )
        self.assertAlmostEqual(
            assembled.mesh.extents[1],
            door.thickness_meters,
        )

    def test_double_sided_back_hardware_faces_outward(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="oriented-double-sided-door",
            name="New door 1",
        ).with_double_sided(True)
        front = door.get_slot(DOOR_SLOT_KNOB)
        back = door.get_slot(DOOR_SLOT_BACK_KNOB)
        assert front is not None and back is not None
        door = door.replace_slot(
            front.with_transform(joined=True)
        ).replace_slot(back.with_transform(joined=True))
        marker = _front_facing_hardware_fixture()
        slot_models = build_default_door_slot_models(door)
        slot_models[front.source_object_id] = marker
        slot_models[back.source_object_id] = marker

        assembled = assemble_door_model(door, slot_models)

        normals_by_id = {}
        bounds_by_id = {}
        for preview in assembled.preview_placed_objects:
            if preview.object_id not in {
                front.source_object_id,
                back.source_object_id,
            }:
                continue
            mesh = preview.meshes[0].copy()
            mesh.apply_transform(preview.placement_transform)
            normals_by_id[preview.object_id] = np.mean(mesh.face_normals, axis=0)
            bounds_by_id[preview.object_id] = mesh.bounds
        self.assertLess(normals_by_id[front.source_object_id][1], -0.99)
        self.assertGreater(normals_by_id[back.source_object_id][1], 0.99)
        self.assertAlmostEqual(
            bounds_by_id[front.source_object_id][0, 1],
            -door.thickness_meters * 0.5,
        )
        self.assertAlmostEqual(
            bounds_by_id[back.source_object_id][0, 1],
            door.thickness_meters * 0.5,
        )

    def test_slot_axis_scales_are_applied_to_assembled_geometry(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="scaled-door",
            name="New door 1",
        )
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        door = door.replace_slot(body.with_transform(axis_scales=(1.5, 2.0, 0.5)))
        body_model = build_default_door_slot_models(door)[DOOR_SLOT_BODY]

        assembled = assemble_door_model(
            door,
            {body.source_object_id: body_model},
        )

        body_preview = next(
            preview
            for preview in assembled.preview_placed_objects
            if preview.object_id == body.source_object_id
        )
        self.assertEqual(body_preview.axis_scales, (1.5, 2.0, 0.5))
        np.testing.assert_allclose(
            assembled.mesh.extents,
            body_model.mesh.extents * np.asarray((1.5, 2.0, 0.5)),
            atol=1e-8,
        )

    def test_textured_body_remains_textured_in_complete_preview(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="door-with-textured-body",
            name="New door 1",
        )
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        slot_models = build_default_door_slot_models(door)
        slot_models[DOOR_SLOT_BODY] = _with_solid_texture(
            slot_models[DOOR_SLOT_BODY],
            (210, 40, 25, 255),
        )

        assembled = assemble_door_model(
            door,
            slot_models,
            include_unjoined=True,
        )

        body_surfaces = [
            surface
            for surface in assembled.preview_textured_surfaces
            if body.source_object_id in surface.surface_id
        ]
        self.assertEqual(len(body_surfaces), 1)
        texture = body_surfaces[0].mesh.visual.material.baseColorTexture
        self.assertIsNotNone(texture)
        np.testing.assert_array_equal(
            np.asarray(texture.convert("RGBA"), dtype=np.uint8)[0, 0],
            (210, 40, 25, 255),
        )
        self.assertIsNotNone(assembled.preview_untextured_mesh)
        assert assembled.preview_untextured_mesh is not None
        self.assertEqual(len(assembled.preview_untextured_mesh.faces), 0)
        self.assertEqual(
            len(assembled.mesh.faces),
            len(slot_models[DOOR_SLOT_BODY].mesh.faces),
        )

    def test_body_only_assembly_keeps_its_embedded_texture(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="body-only-door",
            name="New door 1",
        )
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        slot_models = build_default_door_slot_models(door)
        slot_models[DOOR_SLOT_BODY] = _with_solid_texture(
            slot_models[DOOR_SLOT_BODY],
            (25, 80, 220, 255),
        )

        assembled = assemble_door_model(door, slot_models)

        self.assertEqual(len(assembled.preview_textured_surfaces), 1)
        self.assertIn(
            body.source_object_id,
            assembled.preview_textured_surfaces[0].surface_id,
        )
        self.assertEqual(
            len(assembled.mesh.faces),
            len(slot_models[DOOR_SLOT_BODY].mesh.faces),
        )
        self.assertEqual(
            [preview.object_id for preview in assembled.preview_placed_objects],
            [body.source_object_id],
        )

    def test_confirmed_components_are_assembled_with_source_object_ids(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="door-a",
            name="New door 1",
        )
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        door = door.replace_slot(
            knob.with_transform(
                position_meters=(0.7, 0.0, 0.9),
                joined=True,
            )
        )
        slot_models = build_default_door_slot_models(door)
        slot_models[knob.source_object_id] = _generated_hardware_fixture()

        assembled = assemble_door_model(door, slot_models)

        self.assertGreater(
            len(assembled.mesh.faces),
            len(slot_models["body"].mesh.faces),
        )
        self.assertTrue(
            any(
                preview.object_id == knob.source_object_id
                for preview in assembled.preview_placed_objects
            )
        )

    def test_horizontal_mirror_reflects_asymmetric_hardware_and_keeps_glb(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="door-a",
            name="New door 1",
        )
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        door = door.replace_slot(
            knob.with_transform(position_meters=(0.7, 0.0, 0.9), joined=True)
        )
        assembled = assemble_door_model(
            door,
            {
                **build_default_door_slot_models(door),
                knob.source_object_id: _generated_hardware_fixture(),
            },
        )

        mirrored = mirror_door_model_horizontally(assembled)

        self.assertAlmostEqual(
            mirrored.mesh.bounds[0, 0],
            -assembled.mesh.bounds[1, 0],
        )
        self.assertAlmostEqual(
            mirrored.mesh.bounds[1, 0],
            -assembled.mesh.bounds[0, 0],
        )
        self.assertEqual(len(mirrored.mesh.faces), len(assembled.mesh.faces))
        self.assertGreater(len(mirrored.glb_bytes), 0)

    def test_nested_body_and_knob_mirrors_follow_complete_door_placement(
        self,
    ) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="door-with-two-mirrors",
            name="New door 1",
        )
        body = door.get_slot(DOOR_SLOT_BODY)
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert body is not None and knob is not None
        door = door.replace_slot(
            knob.with_transform(
                position_meters=(0.3, -0.06, 0.9),
                axis_scales=(1.0, 1.5, 1.0),
                joined=True,
            )
        )
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        slot_models = build_default_door_slot_models(door)
        slot_models[DOOR_SLOT_BODY] = _with_solid_texture(
            slot_models[DOOR_SLOT_BODY],
            (45, 130, 220, 255),
        )
        body_face_count = len(slot_models[DOOR_SLOT_BODY].mesh.faces)
        slot_models[knob.source_object_id] = _generated_hardware_fixture()
        hardware_face_count = sum(
            len(model.mesh.faces)
            for slot_id, model in slot_models.items()
            if slot_id != DOOR_SLOT_BODY
        )
        assembled = assemble_door_model(
            door,
            slot_models,
            include_unjoined=True,
            body_mirror=DoorBodyMirrorConfiguration(
                symmetric_orientation="vertical",
                symmetric_plane_coordinate=0.0,
                side_duplication_plane_coordinate=0.0,
            ),
        )

        previews = {
            preview.object_id: preview
            for preview in assembled.preview_symmetric_objects
        }
        body_preview_ids = {
            body.source_object_id,
            f"{body.source_object_id}:side",
            f"{body.source_object_id}:symmetric-side",
        }
        knob_preview_id = f"{knob.source_object_id}:side"
        self.assertEqual(set(previews), {*body_preview_ids, knob_preview_id})
        self.assertTrue(
            all(
                not preview.fade_enabled
                for preview in assembled.preview_symmetric_objects
            )
        )
        self.assertGreater(hardware_face_count, 0)
        self.assertEqual(
            len(assembled.mesh.faces),
            body_face_count + hardware_face_count,
        )
        for preview_id in body_preview_ids:
            preview = previews[preview_id]
            self.assertEqual(
                sum(len(mesh.faces) for mesh in preview.meshes),
                body_face_count,
            )
            self.assertEqual(
                sum(len(mesh.faces) for mesh in preview.mirrored_meshes),
                body_face_count,
            )
        knob_preview = previews[knob_preview_id]
        self.assertEqual(
            sum(len(mesh.faces) for mesh in knob_preview.meshes),
            hardware_face_count,
        )
        self.assertEqual(
            sum(len(mesh.faces) for mesh in knob_preview.mirrored_meshes),
            hardware_face_count,
        )
        retained_knob_vertices = np.asarray(
            knob_preview.meshes[0].vertices,
            dtype=float,
        )
        mirrored_knob_vertices = np.asarray(
            knob_preview.mirrored_meshes[0].vertices,
            dtype=float,
        )
        np.testing.assert_allclose(
            np.sort(mirrored_knob_vertices[:, 0]),
            np.sort(retained_knob_vertices[:, 0]),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            np.sort(mirrored_knob_vertices[:, 1]),
            np.sort(
                knob_preview.plane_coordinate * 2.0
                - retained_knob_vertices[:, 1]
            ),
            atol=1e-9,
        )
        mirrored_assembly = mirror_door_model_horizontally(assembled)
        self.assertEqual(len(mirrored_assembly.preview_symmetric_objects), 4)
        self.assertTrue(
            all(
                not preview.fade_enabled
                for preview in mirrored_assembly.preview_symmetric_objects
            )
        )
        for original, reflected in zip(
            assembled.preview_symmetric_objects,
            mirrored_assembly.preview_symmetric_objects,
            strict=True,
        ):
            self.assertAlmostEqual(
                reflected.meshes[0].bounds[0, 0],
                -original.meshes[0].bounds[1, 0],
            )
            self.assertAlmostEqual(
                reflected.mirrored_meshes[0].bounds[1, 0],
                -original.mirrored_meshes[0].bounds[0, 0],
            )

        empty_base = GeneratedModel(
            mesh=trimesh.Trimesh(process=False),
            scene=trimesh.Scene(),
            glb_bytes=b"",
        )
        composed = compose_placed_generated_models(
            empty_base,
            (
                PlacedGeneratedModel(
                    object_id="placed-door",
                    model=assembled,
                    world_position=(4.0, 5.0, 1.0),
                    rotation_degrees=(0.0, 0.0, 90.0),
                ),
            ),
        )

        self.assertEqual(len(composed.preview_symmetric_objects), 4)
        for preview in composed.preview_symmetric_objects:
            self.assertTrue(preview.object_id.startswith("placed-door:nested:"))
            self.assertFalse(preview.fade_enabled)
        composed_body_previews = tuple(
            preview
            for preview in composed.preview_symmetric_objects
            if not preview.object_id.endswith(knob_preview_id)
        )
        self.assertEqual(len(composed_body_previews), 3)
        for preview in composed_body_previews:
            self.assertEqual(
                sum(len(mesh.faces) for mesh in preview.mirrored_meshes),
                body_face_count,
            )
            texture = preview.mirrored_meshes[0].visual.material.baseColorTexture
            self.assertIsNotNone(texture)
            np.testing.assert_array_equal(
                np.asarray(texture.convert("RGBA"), dtype=np.uint8)[0, 0],
                (45, 130, 220, 255),
            )
        composed_knob_preview = next(
            preview
            for preview in composed.preview_symmetric_objects
            if preview.object_id.endswith(knob_preview_id)
        )
        self.assertEqual(
            sum(
                len(mesh.faces)
                for mesh in composed_knob_preview.mirrored_meshes
            ),
            hardware_face_count,
        )

    def test_nested_body_mirror_planes_follow_body_axis_scales(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="scaled-mirror-door",
            name="New door 1",
        )
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        door = door.replace_slot(
            body.with_transform(axis_scales=(1.5, 2.0, 0.75))
        )
        body_model = build_default_door_slot_models(door)[DOOR_SLOT_BODY]

        assembled = assemble_door_model(
            door,
            {body.source_object_id: body_model},
            body_mirror=DoorBodyMirrorConfiguration(
                symmetric_orientation="vertical",
                symmetric_plane_coordinate=0.1,
                side_duplication_plane_coordinate=0.01,
            ),
        )

        previews = {
            preview.object_id: preview
            for preview in assembled.preview_symmetric_objects
        }
        symmetric = previews[body.source_object_id]
        side = previews[f"{body.source_object_id}:side"]
        symmetric_side = previews[f"{body.source_object_id}:symmetric-side"]
        self.assertAlmostEqual(symmetric.plane_coordinate, 0.15)
        self.assertAlmostEqual(side.plane_coordinate, 0.02)
        self.assertAlmostEqual(symmetric_side.plane_coordinate, 0.02)

        retained_vertices = np.asarray(side.meshes[0].vertices, dtype=float)
        side_vertices = np.asarray(side.mirrored_meshes[0].vertices, dtype=float)
        np.testing.assert_allclose(
            np.sort(side_vertices[:, 1]),
            np.sort(0.04 - retained_vertices[:, 1]),
            atol=1e-9,
        )
        symmetric_vertices = np.asarray(
            symmetric_side.meshes[0].vertices,
            dtype=float,
        )
        np.testing.assert_allclose(
            np.sort(symmetric_vertices[:, 0]),
            np.sort(0.3 - retained_vertices[:, 0]),
            atol=1e-9,
        )


# ### Window assembly tests ###
class WindowAssemblyGeometryTests(unittest.TestCase):
    def test_double_sided_double_window_has_one_fallback_per_body(self) -> None:
        window = create_window_definition_for_frame(
            "frame-a",
            1.2,
            1.0,
            window_id="window-a",
            name="New window 1",
        ).with_double(True).with_double_sided(True)

        slot_models = build_default_window_slot_models(window)
        assembled = assemble_window_model(window, slot_models)

        expected_body_ids = {
            DOOR_SLOT_BODY,
            DOOR_SLOT_LEFT_BODY,
            DOOR_SLOT_BACK_BODY,
            DOOR_SLOT_BACK_LEFT_BODY,
        }
        self.assertEqual(set(slot_models), expected_body_ids)
        bounds_by_slot = {}
        for slot_id in expected_body_ids:
            slot = window.get_slot(slot_id)
            assert slot is not None
            preview = next(
                candidate
                for candidate in assembled.preview_placed_objects
                if candidate.object_id == slot.source_object_id
            )
            mesh = preview.meshes[0].copy()
            mesh.apply_transform(preview.placement_transform)
            bounds_by_slot[slot_id] = mesh.bounds
        self.assertAlmostEqual(
            bounds_by_slot[DOOR_SLOT_LEFT_BODY][1, 0],
            bounds_by_slot[DOOR_SLOT_BODY][0, 0],
        )
        self.assertAlmostEqual(
            bounds_by_slot[DOOR_SLOT_BODY][1, 1],
            bounds_by_slot[DOOR_SLOT_BACK_BODY][0, 1],
        )
        self.assertAlmostEqual(assembled.mesh.extents[0], window.width_meters)
        self.assertAlmostEqual(
            assembled.mesh.extents[1],
            window.thickness_meters,
        )

    def test_legacy_front_window_bodies_remain_flush_when_made_double(
        self,
    ) -> None:
        window = create_window_definition_for_frame(
            "frame-a",
            1.2,
            1.0,
            window_id="legacy-window",
            name="New window 1",
        ).with_double(True).with_double_sided(True)
        legacy_front = build_door_body_model(
            0.35,
            0.7,
            0.04,
            shape=DOORWAY_SHAPE_RECTANGULAR,
            arch_amount=0.0,
        )
        slot_models = build_default_window_slot_models(window)
        slot_models[DOOR_SLOT_BODY] = legacy_front
        slot_models[DOOR_SLOT_LEFT_BODY] = legacy_front

        assembled = assemble_window_model(window, slot_models)

        bounds_by_slot = {}
        for slot_id in (DOOR_SLOT_BODY, DOOR_SLOT_LEFT_BODY):
            slot = window.get_slot(slot_id)
            assert slot is not None
            preview = next(
                candidate
                for candidate in assembled.preview_placed_objects
                if candidate.object_id == slot.source_object_id
            )
            mesh = preview.meshes[0].copy()
            mesh.apply_transform(preview.placement_transform)
            bounds_by_slot[slot_id] = mesh.bounds
        self.assertAlmostEqual(
            bounds_by_slot[DOOR_SLOT_LEFT_BODY][1, 0],
            bounds_by_slot[DOOR_SLOT_BODY][0, 0],
        )
        np.testing.assert_allclose(
            (
                bounds_by_slot[DOOR_SLOT_LEFT_BODY][0, 0],
                bounds_by_slot[DOOR_SLOT_BODY][1, 0],
            ),
            (-window.width_meters * 0.5, window.width_meters * 0.5),
            atol=1e-8,
        )

    def test_window_glass_follows_its_leaf_across_front_and_back(self) -> None:
        window = create_window_definition_for_frame(
            "frame-a",
            1.2,
            1.0,
            window_id="window-glass",
            name="New window 1",
        ).with_double(True).with_double_sided(True)
        left_body = window.get_slot(DOOR_SLOT_LEFT_BODY)
        assert left_body is not None
        window = replace(
            window,
            glass_areas=(
                create_glass_area_data(
                    window.window_id,
                    (),
                    minimum_x_ratio=0.2,
                    maximum_x_ratio=0.8,
                    minimum_z_ratio=0.2,
                    maximum_z_ratio=0.8,
                    glass_area_id="left-pane",
                    body_slot_id=DOOR_SLOT_LEFT_BODY,
                ),
            ),
        )

        assembled = assemble_window_model(window)

        glass_owner_ids = {
            preview.object_id
            for preview in assembled.preview_placed_objects
            if any(
                is_housemaker_glass_material(mesh.visual.material)
                for mesh in preview.meshes
            )
        }
        back_left_body = window.get_slot(DOOR_SLOT_BACK_LEFT_BODY)
        right_body = window.get_slot(DOOR_SLOT_BODY)
        assert back_left_body is not None and right_body is not None
        self.assertEqual(
            glass_owner_ids,
            {left_body.source_object_id, back_left_body.source_object_id},
        )
        self.assertNotIn(right_body.source_object_id, glass_owner_ids)

    def test_window_forwards_per_body_mirror_configurations(self) -> None:
        window = create_window_definition_for_frame(
            "frame-a",
            1.2,
            1.0,
            window_id="window-mirror",
            name="New window 1",
        ).with_double(True)
        left_body = window.get_slot(DOOR_SLOT_LEFT_BODY)
        assert left_body is not None

        assembled = assemble_window_model(
            window,
            body_mirrors={
                DOOR_SLOT_LEFT_BODY: DoorBodyMirrorConfiguration(
                    symmetric_orientation="vertical",
                    symmetric_plane_coordinate=0.0,
                )
            },
        )

        self.assertIn(
            left_body.source_object_id,
            {
                preview.object_id
                for preview in assembled.preview_symmetric_objects
            },
        )


if __name__ == "__main__":
    unittest.main()
