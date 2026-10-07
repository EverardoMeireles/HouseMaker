# ### Imports ###
from __future__ import annotations

import copy
import unittest

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.door_geometry import (
    DoorBodyMirrorConfiguration,
    assemble_door_model,
    build_default_door_slot_models,
    build_door_body_model,
    mirror_door_model_horizontally,
)
from housemaker.door_state import (
    DOOR_SLOT_BACK_BODY,
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    create_door_definition_for_doorway,
)
from housemaker.glb import (
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


if __name__ == "__main__":
    unittest.main()
