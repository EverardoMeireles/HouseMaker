# ### Imports ###
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import trimesh
from trimesh.visual.texture import TextureVisuals

from housemaker import object_ao_baking, surface_ao_baking
from housemaker.surface_ao_baking import (
    SurfaceAmbientOcclusionBakeCancelled,
    SurfaceAmbientOcclusionReceiver,
    SurfaceAmbientOcclusionReceiverLayout,
    SurfaceAmbientOcclusionUvLayout,
    bake_surface_ambient_occlusion,
    build_and_bake_surface_ambient_occlusion,
    build_surface_ao_geometry_signature,
    build_surface_ao_uv_layout,
)


# ### Fixture helpers ###
def _horizontal_quad(
    *,
    minimum_x: float,
    maximum_x: float,
    minimum_y: float = 0.0,
    maximum_y: float = 1.0,
    height: float = 0.0,
) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            (
                (minimum_x, minimum_y, height),
                (maximum_x, minimum_y, height),
                (maximum_x, maximum_y, height),
                (minimum_x, maximum_y, height),
            ),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=np.asarray(
            ((0.1, 0.2), (0.9, 0.2), (0.9, 0.8), (0.1, 0.8)),
            dtype=float,
        )
    )
    return mesh


def _parallel_blocker(
    *,
    minimum_x: float,
    maximum_x: float,
    minimum_y: float = -1.0,
    maximum_y: float = 2.0,
    height: float = 0.1,
) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(
            (
                (minimum_x, minimum_y, height),
                (maximum_x, minimum_y, height),
                (maximum_x, maximum_y, height),
                (minimum_x, maximum_y, height),
            ),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )


def _deindexed_grid(cell_count: int = 4) -> trimesh.Trimesh:
    """Return connected geometry expressed as independent UV0 face corners."""

    shared_vertices = np.asarray(
        tuple(
            (float(column), float(row), 0.0)
            for row in range(cell_count + 1)
            for column in range(cell_count + 1)
        ),
        dtype=float,
    )
    shared_faces: list[tuple[int, int, int]] = []
    row_width = cell_count + 1
    for row in range(cell_count):
        for column in range(cell_count):
            bottom_left = row * row_width + column
            bottom_right = bottom_left + 1
            top_left = bottom_left + row_width
            top_right = top_left + 1
            shared_faces.extend(
                (
                    (bottom_left, bottom_right, top_right),
                    (bottom_left, top_right, top_left),
                )
            )
    shared_face_array = np.asarray(shared_faces, dtype=np.int64)
    vertices = np.ascontiguousarray(
        shared_vertices[shared_face_array].reshape((-1, 3)),
    )
    faces = np.arange(len(vertices), dtype=np.int64).reshape((-1, 3))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = TextureVisuals(
        uv=np.ascontiguousarray(vertices[:, :2] / float(cell_count)),
    )
    return mesh


def _coverage_mask(uv: np.ndarray, faces: np.ndarray, resolution: int) -> np.ndarray:
    output = np.full((resolution, resolution), 255, dtype=np.uint8)
    coverage = np.zeros((resolution, resolution), dtype=bool)
    pixels = object_ao_baking._uv_to_top_origin_pixels(uv, resolution)
    for face in faces:
        object_ao_baking._rasterize_scalar_triangle(
            output,
            coverage,
            pixels[face],
            np.zeros(3, dtype=float),
        )
    return coverage


def _manual_uv1_layout(
    uv_sets: tuple[np.ndarray, ...],
    resolution: int,
    *,
    mirrored_receiver_index: int | None = None,
) -> SurfaceAmbientOcclusionUvLayout:
    receiver_layouts = []
    for receiver_index, uv in enumerate(uv_sets):
        receiver_layouts.append(
            SurfaceAmbientOcclusionReceiverLayout(
                key=f"receiver-{receiver_index}",
                source_world_vertices=np.asarray(
                    (
                        (float(receiver_index) * 2.0, 0.0, 0.0),
                        (float(receiver_index) * 2.0 + 1.0, 0.0, 0.0),
                        (float(receiver_index) * 2.0, 1.0, 0.0),
                    ),
                    dtype=float,
                ),
                vertex_mapping=np.asarray((0, 1, 2), dtype=np.int64),
                faces=np.asarray(((0, 1, 2),), dtype=np.int64),
                uv1=np.asarray(uv, dtype=float),
                mirror_plane_point=(
                    (0.0, 0.0, 0.0)
                    if receiver_index == mirrored_receiver_index
                    else None
                ),
                mirror_plane_normal=(
                    (1.0, 0.0, 0.0)
                    if receiver_index == mirrored_receiver_index
                    else None
                ),
            )
        )
    return SurfaceAmbientOcclusionUvLayout(
        resolution=resolution,
        requested_padding_pixels=2,
        effective_padding_pixels=2.0,
        atlas_width=resolution,
        atlas_height=resolution,
        packing_strategy="test",
        receivers=tuple(receiver_layouts),
        layout_signature="0" * 64,
    )


# ### UV1 layout tests ###
class SurfaceAmbientOcclusionUvLayoutTests(unittest.TestCase):
    def test_globally_packs_disjoint_receivers_and_preserves_uv0_mapping(
        self,
    ) -> None:
        first_mesh = _horizontal_quad(minimum_x=0.0, maximum_x=1.0)
        second_mesh = _horizontal_quad(minimum_x=3.0, maximum_x=4.0)
        receivers = (
            SurfaceAmbientOcclusionReceiver("first", first_mesh),
            SurfaceAmbientOcclusionReceiver("second", second_mesh),
        )

        layout = build_surface_ao_uv_layout(
            receivers,
            64,
            padding_pixels=2,
        )

        self.assertEqual(
            tuple(receiver.key for receiver in layout.receivers),
            ("first", "second"),
        )
        self.assertGreaterEqual(layout.effective_padding_pixels, 2.0)
        combined_coverage = np.zeros((64, 64), dtype=bool)
        for source, receiver_layout in zip(
            (first_mesh, second_mesh),
            layout.receivers,
            strict=True,
        ):
            np.testing.assert_array_equal(
                receiver_layout.vertex_mapping[receiver_layout.faces],
                source.faces,
            )
            np.testing.assert_array_equal(
                receiver_layout.map_vertex_attribute(source.visual.uv),
                np.asarray(source.visual.uv)[receiver_layout.vertex_mapping],
            )
            np.testing.assert_allclose(
                receiver_layout.gltf_uv1[:, 0],
                receiver_layout.uv1[:, 0],
            )
            np.testing.assert_allclose(
                receiver_layout.gltf_uv1[:, 1],
                1.0 - receiver_layout.uv1[:, 1],
                atol=1e-7,
            )
            coverage = _coverage_mask(
                receiver_layout.uv1,
                receiver_layout.faces,
                64,
            )
            self.assertFalse(np.any(combined_coverage & coverage))
            combined_coverage |= coverage
        self.assertTrue(np.any(combined_coverage))

    def test_extreme_aspect_layout_retries_until_padding_is_real(self) -> None:
        mesh = _horizontal_quad(
            minimum_x=0.0,
            maximum_x=10.0,
            maximum_y=1.0,
        )

        layout = build_surface_ao_uv_layout(
            (SurfaceAmbientOcclusionReceiver("wide", mesh),),
            64,
            padding_pixels=2,
        )

        self.assertGreaterEqual(layout.effective_padding_pixels, 2.0)
        self.assertGreater(layout.atlas_width, 0)
        self.assertGreater(layout.atlas_height, 0)

    def test_deindexed_half_mesh_rebuilds_safe_geometric_adjacency(self) -> None:
        mesh = _deindexed_grid()
        receiver = SurfaceAmbientOcclusionReceiver(
            "half-grid",
            mesh,
            mirror_plane_point=(0.0, 0.0, 0.0),
            mirror_plane_normal=(1.0, 0.0, 0.0),
        )

        proxy = surface_ao_baking._build_geometric_adjacency_proxy(
            mesh,
            cancellation_check=None,
        )
        layout = build_surface_ao_uv_layout(
            (receiver,),
            64,
            padding_pixels=4,
        )

        self.assertLess(len(proxy.vertices), len(mesh.vertices))
        receiver_layout = layout.receivers[0]
        np.testing.assert_array_equal(
            receiver_layout.vertex_mapping[receiver_layout.faces],
            mesh.faces,
        )
        self.assertEqual(receiver_layout.mirror_plane_point, (0.0, 0.0, 0.0))
        self.assertEqual(receiver_layout.mirror_plane_normal, (1.0, 0.0, 0.0))

    def test_adjacency_proxy_does_not_weld_duplicate_sheets(self) -> None:
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=float,
        )
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64),
            process=False,
        )
        mesh.visual = TextureVisuals(uv=vertices[:, :2])

        proxy = surface_ao_baking._build_geometric_adjacency_proxy(
            mesh,
            cancellation_check=None,
        )

        self.assertEqual(len(proxy.vertices), 6)
        np.testing.assert_array_equal(
            proxy.faces,
            np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64),
        )

    def test_numerically_thin_half_face_does_not_abort_the_layout(self) -> None:
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (2.0, 1e-7, 0.0),
                (4.0, 0.0, 0.0),
                (4.001, 0.0, 0.0),
                (4.0005, 0.000866, 0.0),
            ),
            dtype=float,
        )
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(((0, 1, 2), (3, 4, 5), (6, 7, 8)), dtype=np.int64),
            process=False,
        )
        mesh.visual = TextureVisuals(uv=vertices[:, :2] / 5.0)

        layout = build_surface_ao_uv_layout(
            (
                SurfaceAmbientOcclusionReceiver(
                    "half-with-sliver",
                    mesh,
                    mirror_plane_point=(0.0, 0.0, 0.0),
                    mirror_plane_normal=(1.0, 0.0, 0.0),
                ),
            ),
            4096,
        )

        receiver_layout = layout.receivers[0]
        np.testing.assert_array_equal(
            receiver_layout.vertex_mapping[receiver_layout.faces],
            mesh.faces,
        )
        self.assertTrue(np.all(np.isfinite(receiver_layout.uv1)))
        triangles = receiver_layout.uv1[receiver_layout.faces]
        vectors_a = triangles[:, 1] - triangles[:, 0]
        vectors_b = triangles[:, 2] - triangles[:, 0]
        areas = np.abs(
            vectors_a[:, 0] * vectors_b[:, 1]
            - vectors_a[:, 1] * vectors_b[:, 0]
        )
        self.assertTrue(np.all(areas > surface_ao_baking._GEOMETRY_EPSILON))

    def test_large_world_translation_is_removed_before_float32_packing(self) -> None:
        origin = 1e9
        vertices = np.asarray(
            (
                (origin, origin, origin),
                (origin + 1.0, origin, origin),
                (origin, origin + 1.0, origin),
            ),
            dtype=float,
        )
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(((0, 1, 2),), dtype=np.int64),
            process=False,
        )
        mesh.visual = TextureVisuals(
            uv=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        )

        layout = build_surface_ao_uv_layout(
            (SurfaceAmbientOcclusionReceiver("translated", mesh),),
            64,
            padding_pixels=2,
        )

        np.testing.assert_array_equal(
            layout.receivers[0].vertex_mapping[layout.receivers[0].faces],
            mesh.faces,
        )

    def test_pack_search_stops_after_first_attempt_with_valid_padding(self) -> None:
        receiver = SurfaceAmbientOcclusionReceiver(
            "surface",
            _horizontal_quad(minimum_x=0.0, maximum_x=1.0),
        )
        effective_paddings = (1.0, 2.0, 2.0, 1.5)
        triangle_areas = (0.9, 0.4, 0.8, 0.7)

        def fake_candidate(
            _receivers: object,
            **kwargs: object,
        ) -> surface_ao_baking._PackingCandidate:
            strategy_index = int(kwargs["strategy_index"])
            return surface_ao_baking._PackingCandidate(
                receivers=(),
                internal_padding_pixels=int(kwargs["internal_padding_pixels"]),
                effective_padding_pixels=effective_paddings[strategy_index],
                atlas_width=64,
                atlas_height=64,
                packing_strategy=str(kwargs["strategy_name"]),
                uv_triangle_area=triangle_areas[strategy_index],
                strategy_index=strategy_index,
            )

        with patch.object(
            surface_ao_baking,
            "_build_packing_candidate",
            side_effect=fake_candidate,
        ) as build_candidate:
            selected = surface_ao_baking._build_best_packing_candidate(
                (receiver,),
                resolution=64,
                requested_padding_pixels=2,
                cancellation_check=None,
            )

        self.assertEqual(build_candidate.call_count, 4)
        self.assertEqual(selected.packing_strategy, "align_charts")
        self.assertEqual(selected.uv_triangle_area, 0.8)

    def test_pack_search_retains_a_safe_candidate_when_later_retries_fail(
        self,
    ) -> None:
        receiver = SurfaceAmbientOcclusionReceiver(
            "dense",
            _horizontal_quad(minimum_x=0.0, maximum_x=1.0),
        )

        def fake_candidate(
            _receivers: object,
            **kwargs: object,
        ) -> surface_ao_baking._PackingCandidate:
            if int(kwargs["internal_padding_pixels"]) > 4:
                raise surface_ao_baking._PackingCandidateError("multiple atlases")
            strategy_index = int(kwargs["strategy_index"])
            return surface_ao_baking._PackingCandidate(
                receivers=(),
                internal_padding_pixels=4,
                effective_padding_pixels=2.5,
                atlas_width=96,
                atlas_height=96,
                packing_strategy=str(kwargs["strategy_name"]),
                uv_triangle_area=float(strategy_index + 1),
                strategy_index=strategy_index,
            )

        with patch.object(
            surface_ao_baking,
            "_build_packing_candidate",
            side_effect=fake_candidate,
        ) as build_candidate:
            selected = surface_ao_baking._build_best_packing_candidate(
                (receiver,),
                resolution=64,
                requested_padding_pixels=4,
                cancellation_check=None,
            )

        self.assertEqual(build_candidate.call_count, 8)
        self.assertEqual(selected.effective_padding_pixels, 2.5)
        self.assertEqual(selected.packing_strategy, "rotate_and_align_charts")

    def test_pack_search_retries_the_safe_minimum_after_an_initial_fit_failure(
        self,
    ) -> None:
        receiver = SurfaceAmbientOcclusionReceiver(
            "minimum-padding",
            _horizontal_quad(minimum_x=0.0, maximum_x=1.0),
        )

        def fake_candidate(
            _receivers: object,
            **kwargs: object,
        ) -> surface_ao_baking._PackingCandidate:
            internal_padding = int(kwargs["internal_padding_pixels"])
            if internal_padding > 2:
                raise surface_ao_baking._PackingCandidateError("multiple atlases")
            strategy_index = int(kwargs["strategy_index"])
            return surface_ao_baking._PackingCandidate(
                receivers=(),
                internal_padding_pixels=internal_padding,
                effective_padding_pixels=2.0,
                atlas_width=64,
                atlas_height=64,
                packing_strategy=str(kwargs["strategy_name"]),
                uv_triangle_area=float(strategy_index + 1),
                strategy_index=strategy_index,
            )

        with patch.object(
            surface_ao_baking,
            "_build_packing_candidate",
            side_effect=fake_candidate,
        ) as build_candidate:
            selected = surface_ao_baking._build_best_packing_candidate(
                (receiver,),
                resolution=64,
                requested_padding_pixels=4,
                cancellation_check=None,
            )

        self.assertEqual(build_candidate.call_count, 8)
        self.assertEqual(selected.internal_padding_pixels, 2)
        self.assertEqual(selected.effective_padding_pixels, 2.0)

    def test_pack_search_uses_detached_faces_after_topology_policies_fail(
        self,
    ) -> None:
        receiver = SurfaceAmbientOcclusionReceiver(
            "topology-fallback",
            _horizontal_quad(minimum_x=0.0, maximum_x=1.0),
        )

        def fake_candidate(
            _receivers: object,
            **kwargs: object,
        ) -> surface_ao_baking._PackingCandidate:
            strategy_name = str(kwargs["strategy_name"])
            if not strategy_name.startswith("detached_faces+"):
                raise surface_ao_baking._PackingCandidateError("invalid topology")
            strategy_index = int(kwargs["strategy_index"])
            return surface_ao_baking._PackingCandidate(
                receivers=(),
                internal_padding_pixels=int(kwargs["internal_padding_pixels"]),
                effective_padding_pixels=4.0,
                atlas_width=64,
                atlas_height=64,
                packing_strategy=strategy_name,
                uv_triangle_area=float(strategy_index + 1),
                strategy_index=strategy_index,
            )

        with patch.object(
            surface_ao_baking,
            "_build_packing_candidate",
            side_effect=fake_candidate,
        ) as build_candidate:
            selected = surface_ao_baking._build_best_packing_candidate(
                (receiver,),
                resolution=64,
                requested_padding_pixels=4,
                cancellation_check=None,
            )

        self.assertEqual(build_candidate.call_count, 12)
        self.assertEqual(
            selected.packing_strategy,
            "detached_faces+rotate_and_align_charts",
        )

    def test_layout_and_geometry_signatures_are_deterministic_and_scoped(
        self,
    ) -> None:
        receiver = SurfaceAmbientOcclusionReceiver(
            "surface",
            _horizontal_quad(minimum_x=0.0, maximum_x=1.0),
        )
        first = build_surface_ao_uv_layout((receiver,), 64, padding_pixels=2)
        second = build_surface_ao_uv_layout((receiver,), 64, padding_pixels=2)
        occluder = _parallel_blocker(minimum_x=0.0, maximum_x=1.0)

        self.assertEqual(first.layout_signature, second.layout_signature)
        self.assertEqual(
            build_surface_ao_geometry_signature(first, (occluder,)),
            build_surface_ao_geometry_signature(second, (occluder,)),
        )
        moved_occluder = occluder.copy()
        moved_occluder.apply_translation((1.0, 0.0, 0.0))
        self.assertNotEqual(
            build_surface_ao_geometry_signature(first, (occluder,)),
            build_surface_ao_geometry_signature(first, (moved_occluder,)),
        )

    def test_uv0_only_edits_do_not_invalidate_the_ao_layout(self) -> None:
        original_mesh = _horizontal_quad(minimum_x=0.0, maximum_x=1.0)
        retextured_mesh = original_mesh.copy()
        retextured_mesh.visual = TextureVisuals(
            uv=np.asarray(
                ((0.2, 0.9), (0.4, 0.1), (0.7, 0.3), (0.1, 0.6)),
                dtype=float,
            )
        )

        original = build_surface_ao_uv_layout(
            (SurfaceAmbientOcclusionReceiver("surface", original_mesh),),
            64,
            padding_pixels=2,
        )
        retextured = build_surface_ao_uv_layout(
            (SurfaceAmbientOcclusionReceiver("surface", retextured_mesh),),
            64,
            padding_pixels=2,
        )

        self.assertEqual(original.layout_signature, retextured.layout_signature)
        self.assertEqual(
            build_surface_ao_geometry_signature(original, (original_mesh,)),
            build_surface_ao_geometry_signature(retextured, (original_mesh,)),
        )

    def test_receiver_geometry_topology_and_mirror_plane_invalidate_layout(
        self,
    ) -> None:
        original_mesh = _horizontal_quad(minimum_x=0.0, maximum_x=1.0)
        original = build_surface_ao_uv_layout(
            (SurfaceAmbientOcclusionReceiver("surface", original_mesh),),
            64,
            padding_pixels=2,
        )
        moved_mesh = original_mesh.copy()
        moved_mesh.apply_translation((0.25, 0.0, 0.0))
        moved = build_surface_ao_uv_layout(
            (SurfaceAmbientOcclusionReceiver("surface", moved_mesh),),
            64,
            padding_pixels=2,
        )
        retopologized_mesh = original_mesh.copy()
        retopologized_mesh.faces = np.asarray(
            ((0, 1, 3), (1, 2, 3)),
            dtype=np.int64,
        )
        retopologized = build_surface_ao_uv_layout(
            (SurfaceAmbientOcclusionReceiver("surface", retopologized_mesh),),
            64,
            padding_pixels=2,
        )
        mirrored = build_surface_ao_uv_layout(
            (
                SurfaceAmbientOcclusionReceiver(
                    "surface",
                    original_mesh,
                    mirror_plane_point=(0.0, 0.0, 0.0),
                    mirror_plane_normal=(1.0, 0.0, 0.0),
                ),
            ),
            64,
            padding_pixels=2,
        )

        self.assertNotEqual(original.layout_signature, moved.layout_signature)
        self.assertNotEqual(
            original.layout_signature,
            retopologized.layout_signature,
        )
        self.assertNotEqual(original.layout_signature, mirrored.layout_signature)

    def test_layout_rejects_missing_uv0_and_duplicate_keys(self) -> None:
        mesh = _horizontal_quad(minimum_x=0.0, maximum_x=1.0)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
        with self.assertRaisesRegex(ValueError, "existing UV0"):
            build_surface_ao_uv_layout(
                (SurfaceAmbientOcclusionReceiver("missing-uv", mesh),),
                64,
            )

        valid = _horizontal_quad(minimum_x=0.0, maximum_x=1.0)
        with self.assertRaisesRegex(ValueError, "keys must be unique"):
            build_surface_ao_uv_layout(
                (
                    SurfaceAmbientOcclusionReceiver("same", valid),
                    SurfaceAmbientOcclusionReceiver("same", valid.copy()),
                ),
                64,
            )

    def test_layout_cancellation_is_explicit(self) -> None:
        receiver = SurfaceAmbientOcclusionReceiver(
            "cancelled",
            _horizontal_quad(minimum_x=0.0, maximum_x=1.0),
        )
        with self.assertRaises(SurfaceAmbientOcclusionBakeCancelled):
            build_surface_ao_uv_layout(
                (receiver,),
                64,
                cancellation_check=lambda: True,
            )

    def test_geometry_signature_can_cancel_during_chunked_hashing(self) -> None:
        receiver = SurfaceAmbientOcclusionReceiver(
            "cancelled-signature",
            _horizontal_quad(minimum_x=0.0, maximum_x=1.0),
        )
        layout = build_surface_ao_uv_layout(
            (receiver,),
            64,
            padding_pixels=2,
        )
        occluder = _parallel_blocker(minimum_x=0.0, maximum_x=1.0)
        poll_count = 0

        def cancellation_check() -> bool:
            nonlocal poll_count
            poll_count += 1
            return poll_count >= 4

        with (
            patch.object(
                surface_ao_baking.object_ao_baking,
                "_build_occluder_mesh",
                return_value=occluder,
            ),
            self.assertRaises(SurfaceAmbientOcclusionBakeCancelled),
        ):
            build_surface_ao_geometry_signature(
                layout,
                (occluder,),
                cancellation_check=cancellation_check,
            )


# ### Baking tests ###
class SurfaceAmbientOcclusionBakingTests(unittest.TestCase):
    def test_empty_occluder_scene_returns_white_without_losing_layout(self) -> None:
        receiver = SurfaceAmbientOcclusionReceiver(
            "surface",
            _horizontal_quad(minimum_x=0.0, maximum_x=1.0),
        )

        result = build_and_bake_surface_ambient_occlusion(
            (receiver,),
            64,
            (),
            padding_pixels=2,
        )

        np.testing.assert_array_equal(
            result.ambient_occlusion,
            np.full((64, 64), 255, dtype=np.uint8),
        )
        self.assertEqual(
            result.geometry_signature,
            build_surface_ao_geometry_signature(result.layout, ()),
        )

    def test_independent_tight_targets_are_composited_not_averaged(
        self,
    ) -> None:
        receivers = (
            SurfaceAmbientOcclusionReceiver(
                "first",
                _horizontal_quad(minimum_x=0.0, maximum_x=1.0),
            ),
            SurfaceAmbientOcclusionReceiver(
                "second",
                _horizontal_quad(minimum_x=3.0, maximum_x=4.0),
            ),
        )
        layout = build_surface_ao_uv_layout(receivers, 64, padding_pixels=2)
        blocker = _parallel_blocker(minimum_x=-1.0, maximum_x=5.0)
        call_index = 0

        def fake_bake(
            target: object,
            _intersector: object,
            _ray_bias: float,
            _resolution: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            nonlocal call_index
            left, top, right, bottom = target.atlas_pixel_bounds
            values = np.full((bottom - top, right - left), 255, dtype=np.uint8)
            coverage = np.zeros(values.shape, dtype=bool)
            values[0, 0] = 40 if call_index == 0 else 90
            coverage[0, 0] = True
            call_index += 1
            return values, coverage

        with (
            patch.object(
                surface_ao_baking.object_ao_baking,
                "_build_ray_intersector",
                return_value=object(),
            ),
            patch.object(
                surface_ao_baking.object_ao_baking,
                "_bake_target_mesh",
                side_effect=fake_bake,
            ),
        ):
            result = bake_surface_ambient_occlusion(layout, (blocker,))

        self.assertEqual(call_index, 2)
        first_bounds = surface_ao_baking._receiver_pixel_bounds(
            layout.receivers[0],
            layout.resolution,
        )
        second_bounds = surface_ao_baking._receiver_pixel_bounds(
            layout.receivers[1],
            layout.resolution,
        )
        self.assertEqual(
            int(result.ambient_occlusion[first_bounds[1], first_bounds[0]]),
            40,
        )
        self.assertEqual(
            int(result.ambient_occlusion[second_bounds[1], second_bounds[0]]),
            90,
        )

    def test_4096_edge_islands_use_small_clamped_target_buffers(self) -> None:
        resolution = 4096
        layout = _manual_uv1_layout(
            (
                np.asarray(((0.0, 0.0), (0.01, 0.0), (0.0, 0.01))),
                np.asarray(((0.99, 1.0), (1.0, 1.0), (1.0, 0.99))),
            ),
            resolution,
            mirrored_receiver_index=0,
        )
        blocker = _parallel_blocker(minimum_x=-1.0, maximum_x=5.0)
        targets = []

        def fake_bake(
            target: object,
            _intersector: object,
            _ray_bias: float,
            _resolution: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            targets.append(target)
            left, top, right, bottom = target.atlas_pixel_bounds
            values = np.full((bottom - top, right - left), 255, dtype=np.uint8)
            coverage = np.zeros(values.shape, dtype=bool)
            if len(targets) == 1:
                values[-1, 0] = 35
                coverage[-1, 0] = True
            else:
                values[0, -1] = 75
                coverage[0, -1] = True
            return values, coverage

        with (
            patch.object(
                surface_ao_baking.object_ao_baking,
                "_build_ray_intersector",
                return_value=object(),
            ),
            patch.object(
                surface_ao_baking.object_ao_baking,
                "_bake_target_mesh",
                side_effect=fake_bake,
            ),
        ):
            result = bake_surface_ambient_occlusion(layout, (blocker,))

        self.assertEqual(len(targets), 2)
        first_bounds = targets[0].atlas_pixel_bounds
        second_bounds = targets[1].atlas_pixel_bounds
        self.assertEqual(first_bounds[0], 0)
        self.assertEqual(first_bounds[3], resolution)
        self.assertEqual(second_bounds[1], 0)
        self.assertEqual(second_bounds[2], resolution)
        self.assertLess(first_bounds[2] - first_bounds[0], 128)
        self.assertLess(first_bounds[3] - first_bounds[1], 128)
        self.assertLess(second_bounds[2] - second_bounds[0], 128)
        self.assertLess(second_bounds[3] - second_bounds[1], 128)
        self.assertLessEqual(first_bounds[2], second_bounds[0])
        self.assertEqual(targets[0].mirror_plane_point, (0.0, 0.0, 0.0))
        self.assertEqual(targets[0].mirror_plane_normal, (1.0, 0.0, 0.0))
        self.assertEqual(int(result.ambient_occlusion[-1, 0]), 35)
        self.assertEqual(int(result.ambient_occlusion[0, -1]), 75)

    def test_one_receiver_can_be_occluded_without_darkening_another(
        self,
    ) -> None:
        first_mesh = _horizontal_quad(minimum_x=0.0, maximum_x=1.0)
        second_mesh = _horizontal_quad(minimum_x=4.0, maximum_x=5.0)
        blocker = _parallel_blocker(minimum_x=-1.0, maximum_x=2.0)
        layout = build_surface_ao_uv_layout(
            (
                SurfaceAmbientOcclusionReceiver("blocked", first_mesh),
                SurfaceAmbientOcclusionReceiver("clear", second_mesh),
            ),
            64,
            padding_pixels=2,
        )

        result = bake_surface_ambient_occlusion(
            layout,
            (first_mesh, second_mesh, blocker),
        )

        blocked_layout, clear_layout = layout.receivers
        blocked_mask = _coverage_mask(
            blocked_layout.uv1,
            blocked_layout.faces,
            layout.resolution,
        )
        clear_mask = _coverage_mask(
            clear_layout.uv1,
            clear_layout.faces,
            layout.resolution,
        )
        self.assertLess(int(np.min(result.ambient_occlusion[blocked_mask])), 64)
        self.assertEqual(int(np.min(result.ambient_occlusion[clear_mask])), 255)

    def test_mirror_plane_averages_authored_and_runtime_receiver_ao(self) -> None:
        authored_mesh = _horizontal_quad(
            minimum_x=-2.0,
            maximum_x=-1.0,
            minimum_y=-2.0,
            maximum_y=-1.0,
        )
        blocker = _parallel_blocker(
            minimum_x=0.5,
            maximum_x=2.5,
            minimum_y=-3.0,
            maximum_y=0.0,
        )
        without_mirror = build_and_bake_surface_ambient_occlusion(
            (SurfaceAmbientOcclusionReceiver("authored", authored_mesh),),
            64,
            (authored_mesh, blocker),
            padding_pixels=2,
        )
        with_mirror = build_and_bake_surface_ambient_occlusion(
            (
                SurfaceAmbientOcclusionReceiver(
                    "mirrored",
                    authored_mesh,
                    mirror_plane_point=(0.0, 0.0, 0.0),
                    mirror_plane_normal=(1.0, 0.0, 0.0),
                ),
            ),
            64,
            (authored_mesh, blocker),
            padding_pixels=2,
        )
        without_layout = without_mirror.layout.receivers[0]
        with_layout = with_mirror.layout.receivers[0]
        without_mask = _coverage_mask(
            without_layout.uv1,
            without_layout.faces,
            64,
        )
        with_mask = _coverage_mask(
            with_layout.uv1,
            with_layout.faces,
            64,
        )

        self.assertEqual(
            int(np.min(without_mirror.ambient_occlusion[without_mask])),
            255,
        )
        self.assertLess(int(np.max(with_mirror.ambient_occlusion[with_mask])), 170)
        self.assertGreater(int(np.min(with_mirror.ambient_occlusion[with_mask])), 90)


if __name__ == "__main__":
    unittest.main()
