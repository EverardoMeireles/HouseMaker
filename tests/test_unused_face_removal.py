# ### Imports ###
from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.glb import import_generated_glb
from housemaker.unused_face_removal import (
    ALL_CAMERA_IDS,
    CAMERA_ID_BOTTOM,
    CAMERA_ID_NEG_X,
    CAMERA_ID_NEG_Y,
    CAMERA_ID_POS_X,
    CAMERA_ID_POS_Y,
    CAMERA_ID_TOP,
    CAMERA_OPTIONS,
    UnusedFaceRemovalCancelled,
    UnusedFaceRemovalOptions,
    remove_unused_faces,
    remove_unused_faces_from_glb,
)


# ### Fixture helpers ###
def _scene_glb(*meshes: tuple[str, trimesh.Trimesh, np.ndarray | None]) -> bytes:
    scene = trimesh.Scene()
    for name, mesh, transform in meshes:
        scene.add_geometry(
            mesh,
            geom_name=name,
            node_name=name,
            transform=transform,
        )
    return bytes(scene.export(file_type="glb"))


def _nested_box_glb(*, textured_outer: bool = False) -> bytes:
    outer = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    if textured_outer:
        texture = Image.new("RGBA", (2, 2), (210, 80, 35, 255))
        outer.visual = TextureVisuals(
            uv=np.zeros((len(outer.vertices), 2), dtype=float),
            material=PBRMaterial(baseColorTexture=texture),
        )
    inner = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    return _scene_glb(
        ("outer", outer, None),
        ("inner", inner, None),
    )


# ### Camera metadata tests ###
class UnusedFaceCameraTests(unittest.TestCase):
    def test_camera_metadata_contains_the_six_canonical_views(self) -> None:
        self.assertEqual(
            ALL_CAMERA_IDS,
            (
                CAMERA_ID_POS_X,
                CAMERA_ID_NEG_X,
                CAMERA_ID_POS_Y,
                CAMERA_ID_NEG_Y,
                CAMERA_ID_TOP,
                CAMERA_ID_BOTTOM,
            ),
        )
        self.assertEqual(tuple(option[0] for option in CAMERA_OPTIONS), ALL_CAMERA_IDS)

    def test_camera_selection_is_validated_and_canonically_ordered(self) -> None:
        options = UnusedFaceRemovalOptions(
            enabled_camera_ids=(CAMERA_ID_BOTTOM, CAMERA_ID_POS_X, CAMERA_ID_BOTTOM)
        )
        self.assertEqual(
            options.enabled_camera_ids,
            (CAMERA_ID_POS_X, CAMERA_ID_BOTTOM),
        )
        with self.assertRaisesRegex(ValueError, "Select at least one"):
            UnusedFaceRemovalOptions(enabled_camera_ids=())
        with self.assertRaisesRegex(ValueError, "Unknown unused-face camera"):
            UnusedFaceRemovalOptions(enabled_camera_ids=("diagonal",))


# ### Visibility processing tests ###
class UnusedFaceProcessingTests(unittest.TestCase):
    def test_six_views_remove_an_enclosed_mesh_and_keep_the_outer_shell(self) -> None:
        result = remove_unused_faces_from_glb(_nested_box_glb())

        self.assertEqual(result.original_face_count, 24)
        self.assertEqual(result.retained_face_count, 12)
        self.assertEqual(result.protected_face_count, 12)
        self.assertEqual(result.removed_face_count, 12)
        self.assertEqual(len(result.model.mesh.faces), 12)
        self.assertEqual(result.enabled_camera_ids, ALL_CAMERA_IDS)

    def test_only_checked_cameras_protect_faces(self) -> None:
        source_glb = _scene_glb(
            ("box", trimesh.creation.box(extents=(1.0, 1.0, 1.0)), None)
        )

        one_side = remove_unused_faces_from_glb(
            source_glb,
            options=UnusedFaceRemovalOptions(
                enabled_camera_ids=(CAMERA_ID_POS_X,),
                image_size=64,
            ),
        )
        opposite_sides = remove_unused_faces_from_glb(
            source_glb,
            options=UnusedFaceRemovalOptions(
                enabled_camera_ids=(CAMERA_ID_POS_X, CAMERA_ID_NEG_X),
                image_size=64,
            ),
        )

        self.assertEqual(one_side.retained_face_count, 2)
        self.assertEqual(opposite_sides.retained_face_count, 4)

    def test_pixel_comparison_uses_opencv(self) -> None:
        with patch(
            "housemaker.unused_face_removal.cv2.absdiff",
            wraps=cv2.absdiff,
        ) as absdiff:
            result = remove_unused_faces_from_glb(_nested_box_glb())

        self.assertEqual(result.removed_face_count, 12)
        self.assertGreater(absdiff.call_count, 0)

    def test_generated_model_wrapper_returns_a_generated_model_result(self) -> None:
        source_model = import_generated_glb(_nested_box_glb())

        result = remove_unused_faces(source_model)

        self.assertEqual(result.glb_bytes, result.model.glb_bytes)
        self.assertEqual(len(result.model.mesh.faces), 12)


# ### Asset preservation tests ###
class UnusedFaceAssetPreservationTests(unittest.TestCase):
    def test_filtered_glb_preserves_embedded_outer_texture(self) -> None:
        result = remove_unused_faces_from_glb(_nested_box_glb(textured_outer=True))

        self.assertEqual(tuple(result.model.scene.geometry), ("outer",))
        output_mesh = result.model.scene.geometry["outer"]
        self.assertIsInstance(output_mesh.visual, TextureVisuals)
        material = output_mesh.visual.material
        self.assertIsInstance(material, PBRMaterial)
        self.assertIsNotNone(material.baseColorTexture)
        self.assertEqual(material.baseColorTexture.size, (2, 2))

    def test_filtered_glb_preserves_retained_node_transform(self) -> None:
        transform = trimesh.transformations.translation_matrix((3.0, 4.0, 5.0))
        outer = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        inner = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        source_glb = _scene_glb(
            ("outer", outer, transform),
            ("inner", inner, transform),
        )

        result = remove_unused_faces_from_glb(source_glb)

        output_transform, output_geometry = result.model.scene.graph.get("outer")
        self.assertEqual(output_geometry, "outer")
        np.testing.assert_allclose(output_transform, transform, atol=1e-6)


# ### Bounds and callback tests ###
class UnusedFaceBoundsAndCallbackTests(unittest.TestCase):
    def test_face_limit_is_enforced_before_capture(self) -> None:
        source_glb = _scene_glb(
            ("box", trimesh.creation.box(extents=(1.0, 1.0, 1.0)), None)
        )

        with self.assertRaisesRegex(ValueError, "configured unused-face limit is 11"):
            remove_unused_faces_from_glb(
                source_glb,
                options=UnusedFaceRemovalOptions(max_face_count=11),
            )

    def test_cancellation_interrupts_processing(self) -> None:
        source_glb = _nested_box_glb()

        with self.assertRaises(UnusedFaceRemovalCancelled):
            remove_unused_faces_from_glb(
                source_glb,
                cancel_requested=lambda: True,
            )

    def test_progress_reports_checking_and_completion(self) -> None:
        events = []

        result = remove_unused_faces_from_glb(
            _nested_box_glb(),
            progress_callback=events.append,
        )

        self.assertEqual(result.removed_face_count, 12)
        self.assertIn("checking", {event.stage for event in events})
        self.assertEqual(events[-1].stage, "complete")
        self.assertEqual(events[-1].completed_face_count, 24)
        self.assertEqual(events[-1].total_face_count, 24)


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
