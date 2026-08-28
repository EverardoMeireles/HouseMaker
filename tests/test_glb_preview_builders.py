# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import copy
import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.glb import (
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
    GeneratedModel,
    PlacedGeneratedModel,
    compose_placed_generated_models,
    compose_placed_generated_models_preview,
    convert_to_glb,
    convert_to_preview_model,
    import_generated_glb,
)
from housemaker.models import LevelData, RoomData, VertexData
from housemaker.surface_geometry import build_fixed_surfaces


# ### Fixture helpers ###
def _build_room_level() -> LevelData:
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
        name="Preview room",
        vertex_ids=boundary_ids,
        center_vertex_id=center.id,
        color_rgb=(120, 160, 200),
    )
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[room],
        floor_contour_vertex_ids=boundary_ids,
    )


def _png_bytes(color: tuple[int, int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGBA", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def _build_textured_object_model() -> GeneratedModel:
    vertices = np.asarray(
        (
            (-0.5, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.5, 0.0, 1.0),
            (-0.5, 0.0, 1.0),
        ),
        dtype=float,
    )
    texture = Image.new("RGBA", (4, 4), (220, 40, 30, 255))
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        visual=TextureVisuals(
            uv=np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=float),
            material=PBRMaterial(
                name="Preview object material",
                baseColorTexture=texture,
            ),
        ),
        process=False,
    )
    gltf_mesh = copy.deepcopy(mesh)
    gltf_mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    scene = trimesh.Scene()
    scene.add_geometry(
        gltf_mesh,
        geom_name="textured_object",
        node_name="textured_object",
    )
    return import_generated_glb(bytes(scene.export(file_type="glb")))


def _assert_models_have_matching_geometry(
    test_case: unittest.TestCase,
    first: GeneratedModel,
    second: GeneratedModel,
) -> None:
    np.testing.assert_allclose(first.mesh.vertices, second.mesh.vertices)
    np.testing.assert_array_equal(first.mesh.faces, second.mesh.faces)
    test_case.assertEqual(
        tuple(first.scene.geometry),
        tuple(second.scene.geometry),
    )
    test_case.assertEqual(
        set(first.scene.graph.nodes_geometry),
        set(second.scene.graph.nodes_geometry),
    )
    for node_name in first.scene.graph.nodes_geometry:
        first_transform, first_geometry = first.scene.graph.get(node_name)
        second_transform, second_geometry = second.scene.graph.get(node_name)
        test_case.assertEqual(first_geometry, second_geometry)
        np.testing.assert_allclose(first_transform, second_transform)


# ### Preview conversion tests ###
class GlbPreviewConversionTests(unittest.TestCase):
    def test_surface_material_preview_matches_export_without_serializing(
        self,
    ) -> None:
        level = _build_room_level()
        textured_surface = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_type == "wall"
        )
        surface_materials = {
            textured_surface.surface_id: _png_bytes((20, 110, 210, 255))
        }
        exported = convert_to_glb(
            [level],
            surface_materials=surface_materials,
        )

        with patch.object(
            trimesh.Scene,
            "export",
            side_effect=AssertionError("Preview builders must not serialize."),
        ):
            preview = convert_to_preview_model(
                [level],
                surface_materials=surface_materials,
            )

        self.assertEqual(preview.glb_bytes, b"")
        self.assertTrue(exported.glb_bytes)
        _assert_models_have_matching_geometry(self, preview, exported)
        self.assertEqual(len(preview.preview_textured_surfaces), 1)
        self.assertEqual(
            preview.preview_textured_surfaces[0].surface_id,
            textured_surface.surface_id,
        )
        self.assertEqual(
            preview.preview_textured_surfaces[0].mesh.visual.kind,
            "texture",
        )
        self.assertIsNotNone(preview.preview_untextured_mesh)
        np.testing.assert_allclose(
            preview.preview_untextured_mesh.vertices,
            exported.preview_untextured_mesh.vertices,
        )


# ### Preview composition tests ###
class GlbPreviewCompositionTests(unittest.TestCase):
    def test_placed_preview_matches_export_without_serializing(self) -> None:
        level = _build_room_level()
        object_model = _build_textured_object_model()
        placement = PlacedGeneratedModel(
            object_id="half-lamp",
            model=object_model,
            world_position=(4.0, 5.0, 6.0),
            rotation_degrees=(0.0, 0.0, 35.0),
            symmetric_preview_orientation="vertical",
            symmetric_preview_plane_coordinate=0.0,
        )
        exported = compose_placed_generated_models(
            convert_to_glb([level]),
            [placement],
        )
        preview_base = convert_to_preview_model([level])

        with patch.object(
            trimesh.Scene,
            "export",
            side_effect=AssertionError("Preview builders must not serialize."),
        ):
            preview = compose_placed_generated_models_preview(
                preview_base,
                [placement],
            )

        self.assertEqual(preview.glb_bytes, b"")
        self.assertTrue(exported.glb_bytes)
        _assert_models_have_matching_geometry(self, preview, exported)
        self.assertEqual(len(preview.preview_placed_objects), 1)
        self.assertEqual(len(preview.preview_symmetric_objects), 1)
        self.assertEqual(len(preview.preview_textured_surfaces), 1)
        self.assertEqual(
            preview.preview_textured_surfaces[0].surface_type,
            "generated_object",
        )
        np.testing.assert_allclose(
            preview.preview_placed_objects[0].placement_transform,
            exported.preview_placed_objects[0].placement_transform,
        )
        np.testing.assert_allclose(
            preview.preview_symmetric_objects[0].mirrored_meshes[0].vertices,
            exported.preview_symmetric_objects[0].mirrored_meshes[0].vertices,
        )
        np.testing.assert_allclose(
            preview.preview_base_mesh.vertices,
            exported.preview_base_mesh.vertices,
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
