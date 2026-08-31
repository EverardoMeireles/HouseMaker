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
    DEFAULT_MINIMUM_PROJECTED_SAMPLES,
    DEFAULT_MINIMUM_VISIBLE_FRACTION,
    UnusedFaceRemovalCancelled,
    UnusedFaceRemovalOptions,
    capture_visible_face_indices,
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


def _triangle_mesh(
    vertices: np.ndarray,
    *,
    reverse_winding: bool = False,
) -> trimesh.Trimesh:
    face = [0, 2, 1] if reverse_winding else [0, 1, 2]
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray([face], dtype=np.int64),
        process=False,
    )


def _wafer_triangle(
    depth: float,
    *,
    scale: float = 1.0,
    reverse_winding: bool = False,
) -> trimesh.Trimesh:
    half_extent = 0.5 * scale
    return _triangle_mesh(
        np.asarray(
            (
                (depth, -half_extent, -half_extent),
                (depth, half_extent, -half_extent),
                (depth, -half_extent, half_extent),
            ),
            dtype=float,
        ),
        reverse_winding=reverse_winding,
    )


def _hidden_skinny_triangle() -> trimesh.Trimesh:
    return _triangle_mesh(
        np.asarray(
            (
                (0.47250977, 0.01021223, 0.19674428),
                (0.47636201, -0.33276139, -0.07404630),
                (0.47444922, -0.16127259, 0.06135040),
            ),
            dtype=float,
        )
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

    def test_visibility_threshold_defaults_and_validation(self) -> None:
        options = UnusedFaceRemovalOptions()

        self.assertEqual(
            options.minimum_visible_fraction,
            DEFAULT_MINIMUM_VISIBLE_FRACTION,
        )
        self.assertEqual(
            options.minimum_projected_samples,
            DEFAULT_MINIMUM_PROJECTED_SAMPLES,
        )
        for invalid_fraction in (-0.01, 1.01, float("nan"), True, "0.05"):
            with self.subTest(invalid_fraction=invalid_fraction):
                with self.assertRaisesRegex(ValueError, "visible fraction"):
                    UnusedFaceRemovalOptions(
                        minimum_visible_fraction=invalid_fraction,  # type: ignore[arg-type]
                    )
        for invalid_samples in (0, -1, 1.5, True, "4"):
            with self.subTest(invalid_samples=invalid_samples):
                with self.assertRaisesRegex(ValueError, "projected samples"):
                    UnusedFaceRemovalOptions(
                        minimum_projected_samples=invalid_samples,  # type: ignore[arg-type]
                    )


# ### Visibility processing tests ###
class UnusedFaceProcessingTests(unittest.TestCase):
    def test_six_views_remove_an_enclosed_mesh_and_keep_the_outer_shell(self) -> None:
        result = remove_unused_faces_from_glb(_nested_box_glb())

        self.assertEqual(result.original_face_count, 24)
        self.assertEqual(result.retained_face_count, 12)
        self.assertEqual(result.protected_face_count, 12)
        self.assertEqual(result.removed_face_count, 12)
        self.assertEqual(result.visibility_removed_face_count, 12)
        self.assertEqual(result.stacked_face_removed_count, 0)
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

    def test_exact_rasterization_uses_opencv_as_a_candidate_mask(self) -> None:
        with patch(
            "housemaker.unused_face_removal.cv2.fillConvexPoly",
            wraps=cv2.fillConvexPoly,
        ) as fill_polygon:
            result = remove_unused_faces_from_glb(_nested_box_glb())

        self.assertEqual(result.removed_face_count, 12)
        self.assertGreater(fill_polygon.call_count, 0)

    def test_generated_model_wrapper_returns_a_generated_model_result(self) -> None:
        source_model = import_generated_glb(_nested_box_glb())

        result = remove_unused_faces(source_model)

        self.assertEqual(result.glb_bytes, result.model.glb_bytes)
        self.assertEqual(len(result.model.mesh.faces), 12)


# ### Visibility and stacked-layer regression tests ###
class UnusedFaceVisibilityRegressionTests(unittest.TestCase):
    def test_enclosed_skinny_triangle_cannot_create_extrapolated_depth(self) -> None:
        outer = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        hidden = _hidden_skinny_triangle()
        vertices = np.vstack((outer.vertices, hidden.vertices))
        faces = np.vstack((outer.faces, hidden.faces + len(outer.vertices)))

        positive_x_visible = capture_visible_face_indices(
            vertices,
            faces,
            CAMERA_ID_POS_X,
            image_size=256,
        )
        result = remove_unused_faces_from_glb(
            _scene_glb(("outer", outer, None), ("hidden", hidden, None))
        )

        self.assertNotIn(12, positive_x_visible)
        self.assertEqual(result.original_face_count, 13)
        self.assertEqual(result.retained_face_count, 12)
        self.assertEqual(result.visibility_removed_face_count, 1)
        self.assertEqual(result.stacked_face_removed_count, 0)

    def test_aligned_same_facing_wafer_keeps_only_the_front_layer(self) -> None:
        result = remove_unused_faces_from_glb(
            _scene_glb(
                ("rear", _wafer_triangle(0.0), None),
                ("front", _wafer_triangle(0.002), None),
            ),
            options=UnusedFaceRemovalOptions(image_size=128),
        )

        self.assertEqual(result.original_face_count, 2)
        self.assertEqual(result.retained_face_count, 1)
        self.assertEqual(result.visibility_removed_face_count, 0)
        self.assertEqual(result.stacked_face_removed_count, 1)
        self.assertEqual(tuple(result.model.scene.geometry), ("front",))

    def test_tiny_wafer_protrusion_does_not_protect_the_rear_layer(self) -> None:
        result = remove_unused_faces_from_glb(
            _scene_glb(
                ("rear", _wafer_triangle(0.0), None),
                ("front", _wafer_triangle(0.002, scale=0.99), None),
            ),
            options=UnusedFaceRemovalOptions(image_size=256),
        )

        self.assertEqual(result.retained_face_count, 1)
        self.assertEqual(result.visibility_removed_face_count, 0)
        self.assertEqual(result.stacked_face_removed_count, 1)

    def test_opposite_facing_thin_shell_sides_are_not_stack_duplicates(self) -> None:
        result = remove_unused_faces_from_glb(
            _scene_glb(
                (
                    "negative_side",
                    _wafer_triangle(0.0, reverse_winding=True),
                    None,
                ),
                ("positive_side", _wafer_triangle(0.002), None),
            ),
            options=UnusedFaceRemovalOptions(image_size=128),
        )

        self.assertEqual(result.original_face_count, 2)
        self.assertEqual(result.retained_face_count, 2)
        self.assertEqual(result.removed_face_count, 0)
        self.assertEqual(result.stacked_face_removed_count, 0)

    def test_zero_visible_fraction_still_rejects_fully_occluded_faces(self) -> None:
        result = remove_unused_faces_from_glb(
            _nested_box_glb(),
            options=UnusedFaceRemovalOptions(minimum_visible_fraction=0.0),
        )

        self.assertEqual(result.retained_face_count, 12)
        self.assertEqual(result.visibility_removed_face_count, 12)

    def test_visible_fraction_controls_a_real_small_protrusion(self) -> None:
        source_glb = _scene_glb(
            ("rear", _wafer_triangle(0.0), None),
            ("front", _wafer_triangle(0.1, scale=0.99), None),
        )

        default_threshold = remove_unused_faces_from_glb(
            source_glb,
            options=UnusedFaceRemovalOptions(
                enabled_camera_ids=(CAMERA_ID_POS_X,),
                image_size=256,
            ),
        )
        any_visible_sample = remove_unused_faces_from_glb(
            source_glb,
            options=UnusedFaceRemovalOptions(
                enabled_camera_ids=(CAMERA_ID_POS_X,),
                image_size=256,
                minimum_visible_fraction=0.0,
            ),
        )

        self.assertEqual(default_threshold.retained_face_count, 1)
        self.assertEqual(default_threshold.visibility_removed_face_count, 1)
        self.assertEqual(default_threshold.stacked_face_removed_count, 0)
        self.assertEqual(any_visible_sample.retained_face_count, 2)
        self.assertEqual(any_visible_sample.visibility_removed_face_count, 0)

    def test_subpixel_non_edge_on_face_is_conservatively_protected(self) -> None:
        outer = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        tiny = _triangle_mesh(
            np.asarray(
                (
                    (1.01, -0.0005, -0.0005),
                    (1.01, 0.0005, -0.0005),
                    (1.01, -0.0005, 0.0005),
                ),
                dtype=float,
            )
        )
        result = remove_unused_faces_from_glb(
            _scene_glb(("outer", outer, None), ("tiny", tiny, None)),
            options=UnusedFaceRemovalOptions(
                enabled_camera_ids=(CAMERA_ID_POS_X,),
                image_size=32,
            ),
        )

        self.assertEqual(result.retained_face_count, 3)
        self.assertIn("tiny", result.model.scene.geometry)


# ### Asset preservation tests ###
class UnusedFaceAssetPreservationTests(unittest.TestCase):
    def test_partial_face_filter_preserves_one_mesh_texture_and_uvs(self) -> None:
        outer = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        inner = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        vertices = np.vstack((outer.vertices, inner.vertices))
        faces = np.vstack(
            (outer.faces, inner.faces + len(outer.vertices))
        )
        combined = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )
        texture = Image.new("RGBA", (4, 4), (45, 120, 205, 255))
        combined.visual = TextureVisuals(
            uv=np.zeros((len(vertices), 2), dtype=float),
            material=PBRMaterial(baseColorTexture=texture),
        )

        result = remove_unused_faces_from_glb(
            _scene_glb(("combined", combined, None))
        )

        self.assertEqual(result.retained_face_count, 12)
        output = result.model.scene.geometry["combined"]
        self.assertIsInstance(output.visual, TextureVisuals)
        self.assertEqual(len(output.visual.uv), len(output.vertices))
        self.assertEqual(
            output.visual.material.baseColorTexture.getpixel((0, 0)),
            (45, 120, 205, 255),
        )

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
