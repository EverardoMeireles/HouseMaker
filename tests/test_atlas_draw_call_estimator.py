# ### Imports ###
from __future__ import annotations

import unittest

import numpy as np
import trimesh
from trimesh.visual.material import MultiMaterial, PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.atlas_export import (
    AtlasDrawCallEstimate,
    estimate_texture_atlas_draw_calls,
)
from housemaker.glass_material import build_housemaker_glass_material
from housemaker.glb import GeneratedModel
from housemaker.texture_atlas_state import (
    TextureAtlasPlacement,
    TextureAtlasRecord,
)


# ### Fixture helpers ###
def _placement(object_id: str, *, x: int = 0, y: int = 0) -> TextureAtlasPlacement:
    return TextureAtlasPlacement(
        object_id=object_id,
        texture_path=f"textures/{object_id}.png",
        texture_resolution=512,
        x=x,
        y=y,
        size=512,
    )


def _mesh(
    *,
    object_id: str | None = None,
    surface_id: str | None = None,
    material: object | None = None,
) -> trimesh.Trimesh:
    mesh = trimesh.creation.box()
    mesh.visual = TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=float),
        material=material or PBRMaterial(name="Opaque"),
    )
    if object_id is not None:
        mesh.metadata["housemaker_object_id"] = object_id
    if surface_id is not None:
        mesh.metadata["housemaker_surface_id"] = surface_id
    return mesh


def _multi_material_mesh(object_id: str) -> trimesh.Trimesh:
    mesh = trimesh.creation.box()
    face_materials = np.arange(len(mesh.faces), dtype=np.int64) % 2
    mesh.visual = TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=float),
        material=MultiMaterial(
            (PBRMaterial(name="First"), PBRMaterial(name="Second"))
        ),
        face_materials=face_materials,
    )
    mesh.metadata["housemaker_object_id"] = object_id
    return mesh


def _model(scene: trimesh.Scene) -> GeneratedModel:
    return GeneratedModel(
        mesh=trimesh.creation.box(),
        scene=scene,
        glb_bytes=b"",
    )


# ### Estimator tests ###
class AtlasDrawCallEstimatorTests(unittest.TestCase):
    def test_ordinary_objects_and_surfaces_share_one_used_atlas_batch(self) -> None:
        scene = trimesh.Scene()
        scene.add_geometry(_mesh(object_id="chair"), node_name="chair")
        scene.add_geometry(_mesh(object_id="table"), node_name="table")
        scene.add_geometry(_mesh(surface_id="wall-a"), node_name="wall")
        used = TextureAtlasRecord(
            atlas_id="used",
            name="Used",
            resolution=2048,
            placements=[
                _placement("chair"),
                _placement("table", x=512),
                _placement("plaster", x=1024),
            ],
        )
        unused = TextureAtlasRecord(
            atlas_id="unused",
            name="Unused",
            resolution=2048,
            placements=[_placement("unused-object")],
        )

        estimate = estimate_texture_atlas_draw_calls(
            _model(scene),
            (used, unused),
            surface_source_ids={"wall-a": "plaster"},
        )

        self.assertEqual(estimate.atlas_batch_count, 1)
        self.assertEqual(estimate.exported_count, 1)
        self.assertEqual(estimate.mirrored_runtime_count, 1)

    def test_duplicate_source_uses_only_the_first_atlas_binding(self) -> None:
        scene = trimesh.Scene(_mesh(object_id="chair"))
        first = TextureAtlasRecord(
            atlas_id="first",
            name="First",
            resolution=2048,
            placements=[_placement("chair")],
        )
        second = TextureAtlasRecord(
            atlas_id="second",
            name="Second",
            resolution=2048,
            placements=[_placement("chair")],
        )

        estimate = estimate_texture_atlas_draw_calls(_model(scene), (first, second))

        self.assertEqual(estimate.atlas_batch_count, 1)
        self.assertEqual(estimate.exported_count, 1)

    def test_half_glass_and_passthrough_groups_remain_independent(self) -> None:
        scene = trimesh.Scene()
        scene.add_geometry(_mesh(object_id="ordinary"), node_name="ordinary")
        scene.graph.update(
            frame_from=scene.graph.base_frame,
            frame_to="half-root",
            metadata={"halfMesh": {"uvMode": "reuse"}},
        )
        scene.add_geometry(
            _multi_material_mesh("half"),
            node_name="half-mesh",
            parent_node_name="half-root",
        )
        scene.add_geometry(
            _mesh(
                object_id="glass",
                material=build_housemaker_glass_material(False),
            ),
            node_name="glass",
        )
        scene.add_geometry(
            _multi_material_mesh("unpacked"),
            node_name="unpacked",
        )
        atlas = TextureAtlasRecord(
            atlas_id="scene",
            name="Scene",
            resolution=2048,
            placements=[_placement("ordinary"), _placement("half", x=512)],
        )

        estimate = estimate_texture_atlas_draw_calls(_model(scene), (atlas,))

        self.assertEqual(
            estimate,
            AtlasDrawCallEstimate(
                atlas_batch_count=1,
                half_primitive_count=2,
                glass_primitive_count=1,
                passthrough_primitive_count=2,
            ),
        )
        self.assertEqual(estimate.exported_count, 6)
        self.assertEqual(estimate.mirrored_runtime_count, 8)

    def test_estimate_counts_each_instance_of_shared_geometry(self) -> None:
        scene = trimesh.Scene()
        scene.add_geometry(
            _mesh(object_id="unpacked"),
            geom_name="shared",
            node_name="first",
        )
        scene.graph.update(
            frame_from=scene.graph.base_frame,
            frame_to="second",
            geometry="shared",
        )

        estimate = estimate_texture_atlas_draw_calls(_model(scene), ())

        self.assertEqual(estimate.passthrough_primitive_count, 2)
        self.assertEqual(estimate.exported_count, 2)


if __name__ == "__main__":
    unittest.main()
