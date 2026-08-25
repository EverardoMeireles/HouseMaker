# ### Imports ###
from __future__ import annotations

import unittest

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.unused_face_removal import (
    CAMERA_ID_BOTTOM,
    CAMERA_ID_NEG_X,
    CAMERA_ID_POS_X,
    UncheckedCameraFacePurgeOptions,
    UnusedFaceRemovalCancelled,
    purge_faces_visible_from_unchecked_cameras_from_glb,
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


def _box_glb() -> bytes:
    return _scene_glb(
        ("box", trimesh.creation.box(extents=(1.0, 1.0, 1.0)), None)
    )


# ### Selection and visibility tests ###
class UncheckedCameraFacePurgeTests(unittest.TestCase):
    def test_one_unchecked_camera_removes_only_its_visible_box_side(self) -> None:
        result = purge_faces_visible_from_unchecked_cameras_from_glb(
            _box_glb(),
            unchecked_camera_ids=(CAMERA_ID_POS_X,),
            options=UncheckedCameraFacePurgeOptions(image_size=64),
        )

        self.assertEqual(result.original_face_count, 12)
        self.assertEqual(result.removed_face_count, 2)
        self.assertEqual(result.retained_face_count, 10)
        self.assertEqual(len(result.model.mesh.faces), 10)

    def test_visible_faces_are_removed_once_across_the_camera_union(self) -> None:
        result = purge_faces_visible_from_unchecked_cameras_from_glb(
            _box_glb(),
            unchecked_camera_ids=(
                CAMERA_ID_NEG_X,
                CAMERA_ID_POS_X,
                CAMERA_ID_NEG_X,
            ),
            options=UncheckedCameraFacePurgeOptions(image_size=64),
        )

        self.assertEqual(
            result.unchecked_camera_ids,
            (CAMERA_ID_POS_X, CAMERA_ID_NEG_X),
        )
        self.assertEqual(result.removed_face_count, 4)
        self.assertEqual(result.retained_face_count, 8)

    def test_depth_hidden_faces_are_not_removed(self) -> None:
        outer = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        inner = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

        result = purge_faces_visible_from_unchecked_cameras_from_glb(
            _scene_glb(("outer", outer, None), ("inner", inner, None)),
            unchecked_camera_ids=(CAMERA_ID_POS_X,),
            options=UncheckedCameraFacePurgeOptions(image_size=64),
        )

        self.assertEqual(result.original_face_count, 24)
        self.assertEqual(result.removed_face_count, 2)
        self.assertEqual(result.retained_face_count, 22)
        self.assertEqual(len(result.model.scene.geometry["inner"].faces), 12)

    def test_no_unchecked_cameras_is_an_exact_no_op(self) -> None:
        source_glb = _box_glb()

        result = purge_faces_visible_from_unchecked_cameras_from_glb(
            source_glb,
            unchecked_camera_ids=(),
        )

        self.assertEqual(result.glb_bytes, source_glb)
        self.assertEqual(result.unchecked_camera_ids, ())
        self.assertEqual(result.removed_face_count, 0)
        self.assertEqual(result.retained_face_count, 12)

    def test_unknown_camera_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown unused-face camera"):
            purge_faces_visible_from_unchecked_cameras_from_glb(
                _box_glb(),
                unchecked_camera_ids=("diagonal",),
            )


# ### Asset preservation tests ###
class UncheckedCameraFacePurgeAssetTests(unittest.TestCase):
    def test_retained_mesh_preserves_texture_and_node_transform(self) -> None:
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        texture = Image.new("RGBA", (2, 2), (45, 120, 210, 255))
        mesh.visual = TextureVisuals(
            uv=np.zeros((len(mesh.vertices), 2), dtype=float),
            material=PBRMaterial(baseColorTexture=texture),
        )
        transform = trimesh.transformations.translation_matrix((3.0, 4.0, 5.0))

        result = purge_faces_visible_from_unchecked_cameras_from_glb(
            _scene_glb(("box", mesh, transform)),
            unchecked_camera_ids=(CAMERA_ID_BOTTOM,),
            options=UncheckedCameraFacePurgeOptions(image_size=64),
        )

        output_mesh = result.model.scene.geometry["box"]
        self.assertEqual(len(output_mesh.faces), 10)
        self.assertIsInstance(output_mesh.visual, TextureVisuals)
        self.assertIsInstance(output_mesh.visual.material, PBRMaterial)
        self.assertIsNotNone(output_mesh.visual.material.baseColorTexture)
        output_transform, output_geometry_name = result.model.scene.graph.get("box")
        self.assertEqual(output_geometry_name, "box")
        np.testing.assert_allclose(output_transform, transform, atol=1e-6)


# ### Bounds and callback tests ###
class UncheckedCameraFacePurgeCallbackTests(unittest.TestCase):
    def test_face_limit_is_checked_before_capture(self) -> None:
        with self.assertRaisesRegex(ValueError, "camera-face purge limit is 11"):
            purge_faces_visible_from_unchecked_cameras_from_glb(
                _box_glb(),
                unchecked_camera_ids=(CAMERA_ID_POS_X,),
                options=UncheckedCameraFacePurgeOptions(max_face_count=11),
            )

    def test_cancellation_interrupts_processing(self) -> None:
        with self.assertRaises(UnusedFaceRemovalCancelled):
            purge_faces_visible_from_unchecked_cameras_from_glb(
                _box_glb(),
                unchecked_camera_ids=(CAMERA_ID_POS_X,),
                cancel_requested=lambda: True,
            )

    def test_progress_reports_capture_check_export_and_completion(self) -> None:
        events = []

        result = purge_faces_visible_from_unchecked_cameras_from_glb(
            _box_glb(),
            unchecked_camera_ids=(CAMERA_ID_POS_X,),
            progress_callback=events.append,
        )

        self.assertEqual(result.removed_face_count, 2)
        self.assertTrue(
            {"capturing", "checking", "exporting", "complete"}.issubset(
                event.stage for event in events
            )
        )
        self.assertEqual(events[-1].stage, "complete")
        self.assertEqual(events[-1].completed_face_count, 12)
        self.assertEqual(events[-1].total_face_count, 12)

    def test_purging_every_visible_face_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "would remove every face"):
            purge_faces_visible_from_unchecked_cameras_from_glb(
                _box_glb(),
                unchecked_camera_ids=(
                    CAMERA_ID_POS_X,
                    CAMERA_ID_NEG_X,
                    "pos_y",
                    "neg_y",
                    "top",
                    CAMERA_ID_BOTTOM,
                ),
                options=UncheckedCameraFacePurgeOptions(image_size=64),
            )


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
