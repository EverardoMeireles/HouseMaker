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

    def test_every_default_slot_model_has_complete_finite_uvs(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_ARCH, arch_amount=0.7),
            door_id="door-with-uvs",
            name="New door 1",
        )

        slot_models = build_default_door_slot_models(door)

        self.assertEqual(set(slot_models), {"body", "hinges", "door_knob"})
        for model in slot_models.values():
            with self.subTest(model_bounds=model.mesh.bounds.tolist()):
                _assert_complete_uvs(self, model.mesh)
                imported = import_generated_glb(model.glb_bytes)
                _assert_complete_uvs(self, imported.mesh)


# ### Door assembly and mirror tests ###
class DoorAssemblyGeometryTests(unittest.TestCase):
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
        self.assertEqual(
            len(assembled.preview_untextured_mesh.faces),
            len(slot_models["hinges"].mesh.faces)
            + len(slot_models["door_knob"].mesh.faces),
        )
        self.assertEqual(
            len(assembled.mesh.faces),
            sum(len(model.mesh.faces) for model in slot_models.values()),
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
            build_default_door_slot_models(door),
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

    def test_nested_door_body_mirrors_follow_the_complete_door_placement(
        self,
    ) -> None:
        door = create_door_definition_for_doorway(
            _doorway(shape=DOORWAY_SHAPE_RECTANGULAR),
            door_id="door-with-two-mirrors",
            name="New door 1",
        )
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        slot_models = build_default_door_slot_models(door)
        slot_models[DOOR_SLOT_BODY] = _with_solid_texture(
            slot_models[DOOR_SLOT_BODY],
            (45, 130, 220, 255),
        )
        body_face_count = len(slot_models[DOOR_SLOT_BODY].mesh.faces)
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

        self.assertEqual(len(assembled.preview_symmetric_objects), 3)
        self.assertGreater(hardware_face_count, 0)
        self.assertEqual(
            len(assembled.mesh.faces),
            body_face_count + hardware_face_count,
        )
        for preview in assembled.preview_symmetric_objects:
            self.assertEqual(
                sum(len(mesh.faces) for mesh in preview.meshes),
                body_face_count,
            )
            self.assertEqual(
                sum(len(mesh.faces) for mesh in preview.mirrored_meshes),
                body_face_count,
            )
        mirrored_assembly = mirror_door_model_horizontally(assembled)
        self.assertEqual(len(mirrored_assembly.preview_symmetric_objects), 3)
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

        self.assertEqual(len(composed.preview_symmetric_objects), 3)
        for preview in composed.preview_symmetric_objects:
            self.assertTrue(preview.object_id.startswith("placed-door:nested:"))
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


if __name__ == "__main__":
    unittest.main()
