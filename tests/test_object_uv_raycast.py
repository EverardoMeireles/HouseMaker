# ### Imports ###
from __future__ import annotations

import random
from io import BytesIO

import numpy as np
import pytest
import trimesh
from PIL import Image
from shapely import Polygon
from trimesh.visual.color import ColorVisuals
from trimesh.visual.material import MultiMaterial, PBRMaterial
from trimesh.visual.texture import TextureVisuals

import housemaker.object_uv_raycast as object_uv_raycast
from housemaker.object_symmetry import (
    SYMMETRIC_LEFT_PACKED_UV_SAFETY_INSET,
    build_automatic_symmetric_geometry,
    build_symmetric_retexture_proxy_glb,
    build_symmetric_square_pair_texture_variants,
)
from housemaker.object_uv_raycast import (
    UV_TARGET_DOMAIN_FULL,
    UV_TARGET_DOMAIN_LEFT_HALF,
    VISIBILITY_UV_UNWRAP_VERSION,
    VisibilityUvUnwrapCancelled,
    _AtlasOutput,
    _AtlasSubmission,
    _rebuild_geometry_with_uvs,
    unwrap_object_uvs_by_visibility,
)


# ### Test scene builders ###
def _nested_boxes_glb() -> bytes:
    outer = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    inner = trimesh.creation.box(extents=(3.0, 3.0, 3.0))
    mesh = trimesh.util.concatenate((outer, inner))
    normals = np.asarray(mesh.vertices, dtype=np.float64).copy()
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    mesh.vertex_normals = normals
    mesh.visual = TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=np.float64),
        material=PBRMaterial(
            name="untextured-shell-material",
            baseColorFactor=(80, 120, 180, 255),
        ),
    )
    scene = trimesh.Scene()
    transform = trimesh.transformations.translation_matrix((2.0, 3.0, 4.0))
    scene.add_geometry(
        mesh,
        geom_name="nested-shells",
        node_name="nested-shell-node",
        transform=transform,
    )
    return scene.export(file_type="glb")


def _textured_box_glb() -> bytes:
    mesh = trimesh.creation.box()
    mesh.visual = TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=np.float64),
        material=PBRMaterial(
            baseColorTexture=Image.new("RGBA", (4, 4), (30, 60, 90, 255))
        ),
    )
    return trimesh.Scene(mesh).export(file_type="glb")


def _attach_2048_texture(glb_bytes: bytes) -> bytes:
    scene = _load_scene(glb_bytes)
    texture = Image.new("RGBA", (2048, 2048), (45, 90, 135, 255))
    for geometry in scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        uvs = np.asarray(geometry.visual.uv, dtype=np.float64).copy()
        geometry.visual = TextureVisuals(
            uv=uvs,
            material=PBRMaterial(baseColorTexture=texture),
        )
    return bytes(scene.export(file_type="glb"))


def _vertex_colored_box_glb() -> tuple[bytes, dict[tuple[float, ...], np.ndarray]]:
    mesh = trimesh.creation.box()
    colors = np.column_stack(
        (
            np.linspace(25, 200, len(mesh.vertices)),
            np.linspace(200, 25, len(mesh.vertices)),
            np.full(len(mesh.vertices), 100),
            np.full(len(mesh.vertices), 255),
        )
    ).astype(np.uint8)
    mesh.visual = TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=np.float64),
        material=PBRMaterial(name="vertex-colored-material"),
    )
    mesh.visual.vertex_attributes["color"] = colors
    expected = {
        tuple(vertex): color
        for vertex, color in zip(mesh.vertices, colors, strict=True)
    }
    return trimesh.Scene(mesh).export(file_type="glb"), expected


def _untextured_box_mesh(extents: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.visual = TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=np.float64),
        material=PBRMaterial(name="visibility-reference-material"),
    )
    return mesh


def _occluded_target_and_reference_glbs() -> tuple[bytes, bytes]:
    target_scene = trimesh.Scene()
    target_scene.add_geometry(
        _untextured_box_mesh((2.0, 2.0, 2.0)),
        geom_name="visibility-target",
        node_name="visibility-target-node",
    )
    target_glb = target_scene.export(file_type="glb")
    reference_scene = _load_scene(target_glb).copy()
    reference_scene.add_geometry(
        _untextured_box_mesh((4.0, 4.0, 4.0)),
        geom_name="visibility-occluder",
        node_name="visibility-occluder-node",
    )
    return target_glb, reference_scene.export(file_type="glb")


class _FakeAtlas:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        chart_count: int,
        utilization: float,
    ) -> None:
        self.width = width
        self.height = height
        self.chart_count = chart_count
        self.atlas_count = 1
        self._utilization = utilization

    def get_utilization(self, _atlas_index: int) -> float:
        return self._utilization


def _synthetic_generated_atlas(
    rectangles: tuple[tuple[float, float, float, float], ...],
    *,
    packing_strategy: str = "synthetic",
):
    uvs: list[tuple[float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for x, y, width, height in rectangles:
        first_vertex = len(uvs)
        uvs.extend(
            (
                (x, y),
                (x + width, y),
                (x + width, y + height),
                (x, y + height),
            )
        )
        faces.extend(
            (
                (first_vertex, first_vertex + 1, first_vertex + 2),
                (first_vertex, first_vertex + 2, first_vertex + 3),
            )
        )
    face_array = np.asarray(faces, dtype=np.uint32)
    submission = _AtlasSubmission(
        submission_id=0,
        geometry_name="synthetic",
        source_face_indices=np.arange(len(faces), dtype=np.int64),
        source_vertices=np.zeros((len(uvs), 3), dtype=np.float32),
        source_faces=face_array,
        exterior=True,
    )
    output = _AtlasOutput(
        submission=submission,
        vertex_mapping=np.arange(len(uvs), dtype=np.int64),
        faces=face_array.astype(np.int64),
        uvs=np.asarray(uvs, dtype=np.float64),
    )
    return object_uv_raycast._GeneratedAtlas(
        atlas=_FakeAtlas(
            width=2_048,
            height=2_048,
            chart_count=len(rectangles),
            utilization=sum(width * height for _, _, width, height in rectangles),
        ),
        outputs=(output,),
        effective_gutter_pixels=16.0,
        packing_gutter_pixels=16,
        packing_strategy=packing_strategy,
    )


def _synthetic_radial_generated_atlas():
    segment_count = 32
    angles = np.linspace(0.0, 2.0 * np.pi, segment_count, endpoint=False)
    uvs = np.vstack(
        (
            (0.5, 0.5),
            np.column_stack(
                (0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles))
            ),
        )
    )
    faces = np.asarray(
        [
            (0, segment_index + 1, (segment_index + 1) % segment_count + 1)
            for segment_index in range(segment_count)
        ],
        dtype=np.int64,
    )
    submission = _AtlasSubmission(
        submission_id=0,
        geometry_name="radial",
        source_face_indices=np.arange(len(faces), dtype=np.int64),
        source_vertices=np.zeros((len(uvs), 3), dtype=np.float32),
        source_faces=faces.astype(np.uint32),
        exterior=True,
    )
    return object_uv_raycast._GeneratedAtlas(
        atlas=_FakeAtlas(
            width=2_048,
            height=2_048,
            chart_count=1,
            utilization=0.5,
        ),
        outputs=(
            _AtlasOutput(
                submission=submission,
                vertex_mapping=np.arange(len(uvs), dtype=np.int64),
                faces=faces,
                uvs=uvs,
            ),
        ),
        effective_gutter_pixels=16.0,
        packing_gutter_pixels=16,
        packing_strategy="organic",
    )


# ### Inspection helpers ###
def _load_scene(payload: bytes) -> trimesh.Scene:
    loaded = trimesh.load(
        BytesIO(payload),
        file_type="glb",
        force="scene",
        process=False,
    )
    assert isinstance(loaded, trimesh.Scene)
    return loaded


def _all_uv_triangles(scene: trimesh.Scene) -> np.ndarray:
    triangles = []
    for mesh in scene.geometry.values():
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        uv = np.asarray(mesh.visual.uv, dtype=np.float64)
        triangles.append(uv[np.asarray(mesh.faces, dtype=np.int64)])
    return np.concatenate(triangles, axis=0)


def _triangle_areas_2d(triangles: np.ndarray) -> np.ndarray:
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    return np.abs(
        first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    ) / 2.0


def _assert_uv_triangles_do_not_overlap(triangles: np.ndarray) -> None:
    polygons = [Polygon(triangle) for triangle in triangles]
    tolerance = 1e-10
    for first_index, first in enumerate(polygons):
        for second in polygons[first_index + 1 :]:
            assert first.intersection(second).area <= tolerance


# ### Visibility and UV quality tests ###
def test_visibility_unwrap_prioritizes_exterior_and_packs_densely() -> None:
    result = unwrap_object_uvs_by_visibility(
        _nested_boxes_glb(),
        raycast_resolution=64,
    )
    stats = result.stats

    assert VISIBILITY_UV_UNWRAP_VERSION == 4
    assert stats.face_count == 24
    assert stats.instance_face_count == 24
    assert stats.exterior_face_count == 12
    assert stats.hidden_face_count == 12
    assert stats.camera_count == 14
    assert stats.ray_sample_count > 0
    assert stats.chart_count < stats.face_count
    assert stats.achieved_exterior_uv_share == pytest.approx(0.95, abs=0.01)
    assert stats.uv_triangle_occupancy > 0.70
    assert stats.atlas_utilization > 0.75
    assert stats.gutter_pixels == 8
    assert stats.effective_gutter_pixels >= 8.0
    assert stats.target_domain == UV_TARGET_DOMAIN_FULL
    assert stats.effective_horizontal_gutter_pixels >= 8.0
    assert stats.effective_vertical_gutter_pixels >= 8.0
    assert stats.packing_strategy in {
        strategy[0]
        for strategy in object_uv_raycast._XATLAS_PACKING_STRATEGIES
    }
    assert stats.atlas_width != stats.atlas_height

    scene = _load_scene(result.glb_bytes)
    uv_triangles = _all_uv_triangles(scene)
    _assert_uv_triangles_do_not_overlap(uv_triangles)
    assert np.min(uv_triangles) >= 0.0
    assert np.max(uv_triangles) <= 1.0

    mesh = next(iter(scene.geometry.values()))
    positions = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    local_extent = np.max(
        np.abs(positions - np.mean(positions, axis=(0, 1))),
        axis=(1, 2),
    )
    uv_areas = _triangle_areas_2d(
        np.asarray(mesh.visual.uv)[np.asarray(mesh.faces)]
    )
    world_areas = np.linalg.norm(
        np.cross(
            positions[:, 1] - positions[:, 0],
            positions[:, 2] - positions[:, 0],
        ),
        axis=1,
    ) / 2.0
    exterior_density = np.mean(
        uv_areas[local_extent > 1.75] / world_areas[local_extent > 1.75]
    )
    hidden_density = np.mean(
        uv_areas[local_extent < 1.75] / world_areas[local_extent < 1.75]
    )
    assert exterior_density > hidden_density * 8.0


def test_left_half_target_packs_densely_with_effective_gutters() -> None:
    payload = _nested_boxes_glb()
    result = unwrap_object_uvs_by_visibility(
        payload,
        raycast_resolution=48,
        target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
    )
    stats = result.stats
    triangles = _all_uv_triangles(_load_scene(result.glb_bytes))

    assert stats.target_domain == UV_TARGET_DOMAIN_LEFT_HALF
    assert stats.uv_triangle_occupancy > 0.70
    assert stats.atlas_utilization > 0.75
    assert stats.effective_horizontal_gutter_pixels >= 8.0
    assert stats.effective_vertical_gutter_pixels >= 16.0
    assert stats.effective_gutter_pixels == pytest.approx(
        min(
            stats.effective_horizontal_gutter_pixels,
            stats.effective_vertical_gutter_pixels,
        )
    )
    tolerance = 1e-7
    inset = SYMMETRIC_LEFT_PACKED_UV_SAFETY_INSET
    assert np.min(triangles[:, :, 0]) >= inset - tolerance
    assert np.max(triangles[:, :, 0]) <= 0.5 - inset + tolerance
    assert np.min(triangles[:, :, 1]) >= inset - tolerance
    assert np.max(triangles[:, :, 1]) <= 1.0 - inset + tolerance
    _assert_uv_triangles_do_not_overlap(triangles)

    source_mesh = next(iter(_load_scene(payload).geometry.values()))
    output_mesh = next(iter(_load_scene(result.glb_bytes).geometry.values()))
    np.testing.assert_allclose(
        np.asarray(output_mesh.vertices)[np.asarray(output_mesh.faces)],
        np.asarray(source_mesh.vertices)[np.asarray(source_mesh.faces)],
    )


def test_left_half_target_is_deterministic() -> None:
    payload = _nested_boxes_glb()
    first = unwrap_object_uvs_by_visibility(
        payload,
        raycast_resolution=48,
        target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
    )
    second = unwrap_object_uvs_by_visibility(
        payload,
        raycast_resolution=48,
        target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
    )

    assert first.stats == second.stats
    np.testing.assert_allclose(
        _all_uv_triangles(_load_scene(first.glb_bytes)),
        _all_uv_triangles(_load_scene(second.glb_bytes)),
    )


def test_true_aspect_maxrects_improves_fragmented_half_atlas() -> None:
    rectangles = tuple(
        (
            0.02 + column * 0.15,
            0.02 + row * 0.22,
            0.09,
            0.025,
        )
        for row in range(4)
        for column in range(6)
    )
    generated = _synthetic_generated_atlas(rectangles)
    direct_outputs = object_uv_raycast._map_outputs_to_target_domain(
        generated.outputs,
        target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
    )
    _share, direct_occupancy = object_uv_raycast._measure_uv_area(
        direct_outputs,
        target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
    )

    first = object_uv_raycast._build_target_domain_layout(
        generated,
        texture_resolution=2_048,
        gutter_pixels=8,
        target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
        cancellation_check=None,
    )
    second = object_uv_raycast._build_target_domain_layout(
        generated,
        texture_resolution=2_048,
        gutter_pixels=8,
        target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
        cancellation_check=None,
    )
    _share, packed_occupancy = object_uv_raycast._measure_uv_area(
        first.outputs,
        target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
    )

    assert packed_occupancy > direct_occupancy * 5.0
    assert first.chart_count == 24
    assert first.layout_width == 1_024
    assert first.layout_height == 2_048
    assert first.horizontal_gutter_pixels == 8.0
    assert first.vertical_gutter_pixels == 8.0
    assert "true_aspect_maxrects" in first.packing_strategy
    assert 0.0 < first.utilization <= 1.0
    np.testing.assert_array_equal(
        first.outputs[0].faces,
        generated.outputs[0].faces,
    )
    np.testing.assert_array_equal(
        first.outputs[0].vertex_mapping,
        generated.outputs[0].vertex_mapping,
    )
    np.testing.assert_allclose(first.outputs[0].uvs, second.outputs[0].uvs)
    assert first.packing_strategy == second.packing_strategy
    triangles = first.outputs[0].uvs[first.outputs[0].faces]
    assert np.min(triangles) >= 0.0
    assert np.max(triangles[:, :, 0]) <= 0.5
    assert np.max(triangles[:, :, 1]) <= 1.0
    _assert_uv_triangles_do_not_overlap(triangles)


def test_true_aspect_maxrects_skips_dense_and_low_chart_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense_rectangles = tuple(
        (
            0.01 + column * 0.25,
            0.005 + row / 6.0,
            0.23,
            0.15,
        )
        for row in range(6)
        for column in range(4)
    )
    dense = _synthetic_generated_atlas(dense_rectangles)
    organic = _synthetic_radial_generated_atlas()

    def unexpected_repack(*_args: object, **_kwargs: object):
        raise AssertionError("The optional MaxRects path should be skipped.")

    monkeypatch.setattr(
        object_uv_raycast,
        "_build_true_aspect_layout",
        unexpected_repack,
    )
    for generated in (dense, organic):
        layout = object_uv_raycast._build_target_domain_layout(
            generated,
            texture_resolution=2_048,
            gutter_pixels=8,
            target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
            cancellation_check=None,
        )
        expected = generated.outputs[0].uvs.copy()
        expected[:, 0] *= 0.5
        np.testing.assert_allclose(layout.outputs[0].uvs, expected)
        assert "true_aspect_maxrects" not in layout.packing_strategy


def test_true_aspect_maxrects_honors_cancellation() -> None:
    rectangles = tuple(
        (0.02 + column * 0.15, 0.02 + row * 0.22, 0.09, 0.025)
        for row in range(4)
        for column in range(6)
    )
    generated = _synthetic_generated_atlas(rectangles)
    charts = object_uv_raycast._extract_connected_uv_charts(
        generated,
        cancellation_check=None,
    )

    with pytest.raises(VisibilityUvUnwrapCancelled):
        object_uv_raycast._pack_connected_uv_charts(
            charts,
            target_width=1_024.0,
            target_height=2_048.0,
            gutter_pixels=8.0,
            cancellation_check=lambda: True,
        )


def test_true_aspect_maxrects_failure_falls_back_to_direct_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rectangles = tuple(
        (0.02 + column * 0.15, 0.02 + row * 0.22, 0.09, 0.025)
        for row in range(4)
        for column in range(6)
    )
    generated = _synthetic_generated_atlas(rectangles)

    def fail_packing(*_args: object, **_kwargs: object):
        raise object_uv_raycast._TrueAspectPackingError("bounded failure")

    monkeypatch.setattr(
        object_uv_raycast,
        "_build_true_aspect_layout",
        fail_packing,
    )
    layout = object_uv_raycast._build_target_domain_layout(
        generated,
        texture_resolution=2_048,
        gutter_pixels=8,
        target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
        cancellation_check=None,
    )
    expected = generated.outputs[0].uvs.copy()
    expected[:, 0] *= 0.5

    np.testing.assert_allclose(layout.outputs[0].uvs, expected)
    assert layout.packing_strategy == "synthetic"


def test_packing_strategy_search_beats_the_previous_fixed_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _reference = _occluded_target_and_reference_glbs()
    strategies = object_uv_raycast._XATLAS_PACKING_STRATEGIES
    monkeypatch.setattr(
        object_uv_raycast,
        "_XATLAS_PACKING_STRATEGIES",
        (("previous_fixed", True, True),),
    )
    previous = unwrap_object_uvs_by_visibility(
        payload,
        raycast_resolution=48,
    )
    monkeypatch.setattr(
        object_uv_raycast,
        "_XATLAS_PACKING_STRATEGIES",
        strategies,
    )
    optimized = unwrap_object_uvs_by_visibility(
        payload,
        raycast_resolution=48,
    )

    assert optimized.stats.uv_triangle_occupancy > (
        previous.stats.uv_triangle_occupancy
    )
    assert optimized.stats.packing_strategy != "previous_fixed"


def test_failed_packing_policy_does_not_abort_other_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = object_uv_raycast._generate_atlas_candidate

    def generate_candidate(*args: object, **kwargs: object):
        if kwargs["strategy_name"] == "unavailable":
            raise object_uv_raycast._AtlasCandidatePackingError(
                "unavailable test policy"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        object_uv_raycast,
        "_XATLAS_PACKING_STRATEGIES",
        (
            ("unavailable", False, False),
            ("fixed", False, False),
        ),
    )
    monkeypatch.setattr(
        object_uv_raycast,
        "_generate_atlas_candidate",
        generate_candidate,
    )

    result = unwrap_object_uvs_by_visibility(
        _nested_boxes_glb(),
        raycast_resolution=32,
    )

    assert result.stats.packing_strategy == "fixed"


def test_relaxed_organic_charting_is_gated_and_keeps_only_a_denser_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _synthetic_generated_atlas(
        ((0.0, 0.0, 1.0, 0.4),),
        packing_strategy="baseline",
    )
    baseline.atlas.chart_count = 10
    relaxed = _synthetic_generated_atlas(
        ((0.0, 0.0, 1.0, 0.65),),
        packing_strategy="relaxed",
    )
    relaxed.atlas.chart_count = 8
    calls: list[tuple[float | None, float | None]] = []

    def generate_policy(*_args: object, **kwargs: object):
        policy = (
            kwargs["chart_max_cost"],
            kwargs["chart_straightness_weight"],
        )
        calls.append(policy)
        return baseline if policy[0] is None else relaxed

    monkeypatch.setattr(
        object_uv_raycast,
        "_generate_atlas_for_chart_policy",
        generate_policy,
    )
    submission = _AtlasSubmission(
        submission_id=0,
        geometry_name="gate",
        source_face_indices=np.arange(100, dtype=np.int64),
        source_vertices=np.zeros((3, 3), dtype=np.float32),
        source_faces=np.zeros((100, 3), dtype=np.uint32),
        exterior=True,
    )

    selected = object_uv_raycast._generate_atlas(
        (submission,),
        texture_resolution=2_048,
        gutter_pixels=8,
        cancellation_check=None,
    )

    assert selected is relaxed
    assert calls == [
        (None, None),
        (
            object_uv_raycast._RELAXED_CHART_MAX_COST,
            object_uv_raycast._RELAXED_CHART_STRAIGHTNESS_WEIGHT,
        ),
    ]


def test_complete_target_layouts_are_compared_after_alternate_charting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _synthetic_generated_atlas(
        ((0.0, 0.0, 1.0, 0.4),),
        packing_strategy="baseline",
    )
    relaxed = _synthetic_generated_atlas(
        ((0.0, 0.0, 1.0, 0.5),),
        packing_strategy="relaxed",
    )

    def build_layout(generated: object, **_kwargs: object):
        is_baseline = generated is baseline
        return object_uv_raycast._TargetDomainLayout(
            outputs=(baseline if is_baseline else relaxed).outputs,
            chart_count=24,
            layout_width=1_024,
            layout_height=2_048,
            utilization=0.82 if is_baseline else 0.74,
            horizontal_gutter_pixels=8.0,
            vertical_gutter_pixels=8.0,
            packing_strategy=(
                "baseline+true_aspect_maxrects"
                if is_baseline
                else "relaxed+true_aspect_maxrects"
            ),
        )

    monkeypatch.setattr(
        object_uv_raycast,
        "_build_target_domain_layout",
        build_layout,
    )

    selected = object_uv_raycast._build_best_target_domain_layout(
        (baseline, relaxed),
        texture_resolution=2_048,
        gutter_pixels=8,
        target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
        cancellation_check=None,
    )

    assert selected.packing_strategy.startswith("baseline+")


def test_relaxed_charting_skips_fragmented_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _synthetic_generated_atlas(
        ((0.0, 0.0, 1.0, 0.4),),
        packing_strategy="fragmented",
    )
    baseline.atlas.chart_count = 25
    calls = 0

    def generate_policy(*_args: object, **_kwargs: object):
        nonlocal calls
        calls += 1
        return baseline

    monkeypatch.setattr(
        object_uv_raycast,
        "_generate_atlas_for_chart_policy",
        generate_policy,
    )
    submission = _AtlasSubmission(
        submission_id=0,
        geometry_name="fragmented-gate",
        source_face_indices=np.arange(100, dtype=np.int64),
        source_vertices=np.zeros((3, 3), dtype=np.float32),
        source_faces=np.zeros((100, 3), dtype=np.uint32),
        exterior=True,
    )

    assert (
        object_uv_raycast._generate_atlas(
            (submission,),
            texture_resolution=2_048,
            gutter_pixels=8,
            cancellation_check=None,
        )
        is baseline
    )
    assert calls == 1


# ### Preservation and determinism tests ###
def test_visibility_unwrap_preserves_geometry_normals_material_and_transform() -> None:
    source = _load_scene(_nested_boxes_glb())
    result = unwrap_object_uvs_by_visibility(
        source.export(file_type="glb"),
        raycast_resolution=48,
    )
    output = _load_scene(result.glb_bytes)

    source_transform, _source_name = source.graph.get("nested-shell-node")
    output_transform, _output_name = output.graph.get("nested-shell-node")
    np.testing.assert_allclose(output_transform, source_transform)

    source_mesh = next(iter(source.geometry.values()))
    output_mesh = next(iter(output.geometry.values()))
    assert len(output_mesh.faces) == len(source_mesh.faces)
    source_triangles = np.asarray(source_mesh.vertices)[source_mesh.faces]
    output_triangles = np.asarray(output_mesh.vertices)[output_mesh.faces]
    np.testing.assert_allclose(output_triangles, source_triangles)

    source_normal_by_position = {
        tuple(position): normal
        for position, normal in zip(
            np.asarray(source_mesh.vertices),
            np.asarray(source_mesh.vertex_normals),
            strict=True,
        )
    }
    for position, normal in zip(
        np.asarray(output_mesh.vertices),
        np.asarray(output_mesh.vertex_normals),
        strict=True,
    ):
        np.testing.assert_allclose(normal, source_normal_by_position[tuple(position)])
    assert output_mesh.visual.material.name == source_mesh.visual.material.name


def test_visibility_unwrap_preserves_authored_vertex_colors() -> None:
    source, expected_by_position = _vertex_colored_box_glb()

    result = unwrap_object_uvs_by_visibility(
        source,
        raycast_resolution=32,
    )
    output = next(iter(_load_scene(result.glb_bytes).geometry.values()))
    colors = np.asarray(output.visual.vertex_attributes["color"])

    assert len(colors) == len(output.vertices)
    for vertex, color in zip(output.vertices, colors, strict=True):
        np.testing.assert_allclose(
            color,
            expected_by_position[tuple(vertex)],
            atol=1e-6,
        )


def test_visibility_unwrap_is_deterministic() -> None:
    payload = _nested_boxes_glb()
    first = unwrap_object_uvs_by_visibility(payload, raycast_resolution=48)
    second = unwrap_object_uvs_by_visibility(payload, raycast_resolution=48)

    assert first.stats == second.stats
    np.testing.assert_allclose(
        _all_uv_triangles(_load_scene(first.glb_bytes)),
        _all_uv_triangles(_load_scene(second.glb_bytes)),
    )


def test_visibility_reference_transfers_only_authoritative_target_hits() -> None:
    target_glb, reference_glb = _occluded_target_and_reference_glbs()
    without_reference = unwrap_object_uvs_by_visibility(
        target_glb,
        raycast_resolution=48,
    )
    with_reference = unwrap_object_uvs_by_visibility(
        target_glb,
        raycast_resolution=48,
        visibility_reference_glb=reference_glb,
    )

    assert without_reference.stats.exterior_face_count == 12
    assert with_reference.stats.face_count == 12
    assert with_reference.stats.instance_face_count == 12
    assert with_reference.stats.exterior_face_count == 0
    assert with_reference.stats.hidden_face_count == 12
    assert with_reference.stats.visibility_hits == (0,) * 12
    assert with_reference.stats.ray_sample_count > 0


def test_symmetric_geometry_uses_full_proxy_for_left_half_unwrap() -> None:
    source = trimesh.Scene(
        _untextured_box_mesh((4.0, 3.0, 2.0))
    ).export(file_type="glb")
    divided = build_automatic_symmetric_geometry(
        source,
        "vertical",
        rng=random.Random(3),
    )
    visibility_proxy = build_symmetric_retexture_proxy_glb(
        divided.glb_bytes,
        divided.orientation,
        divided.plane_coordinate,
    )

    result = unwrap_object_uvs_by_visibility(
        divided.glb_bytes,
        target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
        visibility_reference_glb=visibility_proxy,
        raycast_resolution=32,
    )
    texture_proxy = build_symmetric_retexture_proxy_glb(
        result.glb_bytes,
        divided.orientation,
        divided.plane_coordinate,
    )
    proxy_scene = _load_scene(texture_proxy)
    proxy_triangles = _all_uv_triangles(proxy_scene)

    assert result.stats.target_domain == UV_TARGET_DOMAIN_LEFT_HALF
    assert result.stats.face_count == 14
    assert result.stats.uv_triangle_occupancy > 0.70
    assert sum(
        len(geometry.faces) for geometry in proxy_scene.geometry.values()
    ) == 2 * result.stats.face_count
    assert np.min(proxy_triangles) >= 0.0
    assert np.max(proxy_triangles[:, :, 0]) <= 1.0
    assert np.any(proxy_triangles[:, :, 0] < 0.5)
    assert np.any(proxy_triangles[:, :, 0] > 0.5)


def test_direct_symmetric_unwrap_survives_strict_pair_persistence() -> None:
    source = trimesh.Scene(
        _untextured_box_mesh((4.0, 3.0, 2.0))
    ).export(file_type="glb")
    divided = build_automatic_symmetric_geometry(
        source,
        "vertical",
        rng=random.Random(3),
    )
    visibility_proxy = build_symmetric_retexture_proxy_glb(
        divided.glb_bytes,
        divided.orientation,
        divided.plane_coordinate,
    )
    result = unwrap_object_uvs_by_visibility(
        divided.glb_bytes,
        target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
        visibility_reference_glb=visibility_proxy,
        raycast_resolution=32,
    )

    assert "true_aspect_maxrects" not in result.stats.packing_strategy
    textured_glb = _attach_2048_texture(result.glb_bytes)
    variants = build_symmetric_square_pair_texture_variants(
        textured_glb,
        uvs_already_left_packed=True,
    )

    assert set(variants.glb_by_resolution) == {512, 1024}
    assert set(variants.texture_png_by_resolution) == {512, 1024}


def test_symmetric_proxy_preserves_transformed_multi_mesh_subset() -> None:
    scene = trimesh.Scene()
    scene.add_geometry(
        _untextured_box_mesh((4.0, 1.0, 1.5)),
        geom_name="wide-part",
        node_name="wide-part-node",
        transform=trimesh.transformations.translation_matrix(
            (0.0, -1.5, 0.0)
        ),
    )
    scene.add_geometry(
        _untextured_box_mesh((2.0, 1.5, 2.0)),
        geom_name="tall-part",
        node_name="tall-part-node",
        transform=trimesh.transformations.translation_matrix(
            (0.0, 1.5, 0.5)
        ),
    )
    divided = build_automatic_symmetric_geometry(
        scene.export(file_type="glb"),
        "vertical",
        rng=random.Random(4),
    )
    visibility_proxy = build_symmetric_retexture_proxy_glb(
        divided.glb_bytes,
        divided.orientation,
        divided.plane_coordinate,
    )

    result = unwrap_object_uvs_by_visibility(
        divided.glb_bytes,
        target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
        visibility_reference_glb=visibility_proxy,
        raycast_resolution=32,
    )
    retained_scene = _load_scene(divided.glb_bytes)
    proxy_scene = _load_scene(visibility_proxy)

    assert result.stats.face_count == sum(
        len(geometry.faces) for geometry in retained_scene.geometry.values()
    )
    assert sum(
        len(geometry.faces) for geometry in proxy_scene.geometry.values()
    ) == 2 * result.stats.face_count
    assert len(retained_scene.graph.nodes_geometry) == 2
    assert len(proxy_scene.graph.nodes_geometry) == 4


@pytest.mark.parametrize("mismatch", ("vertices", "topology", "transform"))
def test_visibility_reference_rejects_changed_authoritative_geometry(
    mismatch: str,
) -> None:
    target_glb, _reference_glb = _occluded_target_and_reference_glbs()
    reference = _load_scene(target_glb)
    geometry_name, geometry = next(iter(reference.geometry.items()))
    if mismatch == "vertices":
        vertices = np.asarray(geometry.vertices, dtype=np.float64).copy()
        vertices[0, 0] += 0.1
        geometry.vertices = vertices
        message = "local vertices"
    elif mismatch == "topology":
        geometry.faces = np.asarray(geometry.faces, dtype=np.int64)[::-1].copy()
        message = "face topology"
    else:
        node_name = next(iter(reference.graph.nodes_geometry))
        transform, attached_name = reference.graph.get(node_name)
        changed_transform = np.asarray(transform, dtype=np.float64).copy()
        changed_transform[0, 3] += 1.0
        reference.graph.update(
            frame_from=reference.graph.base_frame,
            frame_to=node_name,
            matrix=changed_transform,
            geometry=attached_name,
        )
        message = "node transforms"

    with pytest.raises(ValueError, match=message):
        unwrap_object_uvs_by_visibility(
            target_glb,
            raycast_resolution=32,
            visibility_reference_glb=reference.export(file_type="glb"),
        )


def test_visibility_reference_face_work_is_bounded_to_twice_the_target() -> None:
    target_glb, reference_glb = _occluded_target_and_reference_glbs()
    reference = _load_scene(reference_glb)
    reference.add_geometry(
        _untextured_box_mesh((6.0, 6.0, 6.0)),
        geom_name="second-visibility-occluder",
        node_name="second-visibility-occluder-node",
    )

    with pytest.raises(ValueError, match="at most twice"):
        unwrap_object_uvs_by_visibility(
            target_glb,
            raycast_resolution=32,
            visibility_reference_glb=reference.export(file_type="glb"),
        )


def test_geometry_rebuild_preserves_face_material_indices() -> None:
    source = trimesh.creation.box()
    face_materials = np.arange(len(source.faces), dtype=np.int64) % 2
    source.visual = TextureVisuals(
        uv=np.zeros((len(source.vertices), 2), dtype=np.float64),
        material=MultiMaterial(
            (PBRMaterial(name="first"), PBRMaterial(name="second"))
        ),
        face_materials=face_materials,
    )
    source_faces = np.asarray(source.faces, dtype=np.uint32)
    submission = _AtlasSubmission(
        submission_id=0,
        geometry_name="box",
        source_face_indices=np.arange(len(source.faces), dtype=np.int64),
        source_vertices=np.asarray(source.vertices, dtype=np.float32),
        source_faces=source_faces,
        exterior=True,
    )
    output = _AtlasOutput(
        submission=submission,
        vertex_mapping=np.arange(len(source.vertices), dtype=np.int64),
        faces=np.asarray(source.faces, dtype=np.int64),
        uvs=np.zeros((len(source.vertices), 2), dtype=np.float64),
    )

    rebuilt = _rebuild_geometry_with_uvs(source, (output,))

    np.testing.assert_array_equal(rebuilt.visual.face_materials, face_materials)
    assert [material.name for material in rebuilt.visual.material.materials] == [
        "first",
        "second",
    ]


def test_geometry_rebuild_converts_face_colors_to_authored_vertex_colors() -> None:
    source = trimesh.creation.box()
    face_colors = np.column_stack(
        (
            np.arange(len(source.faces), dtype=np.uint8),
            np.full(len(source.faces), 80, dtype=np.uint8),
            np.full(len(source.faces), 160, dtype=np.uint8),
            np.full(len(source.faces), 255, dtype=np.uint8),
        )
    )
    source.visual = ColorVisuals(mesh=source, face_colors=face_colors)
    submission = _AtlasSubmission(
        submission_id=0,
        geometry_name="box",
        source_face_indices=np.arange(len(source.faces), dtype=np.int64),
        source_vertices=np.asarray(source.vertices, dtype=np.float32),
        source_faces=np.asarray(source.faces, dtype=np.uint32),
        exterior=True,
    )
    output = _AtlasOutput(
        submission=submission,
        vertex_mapping=np.arange(len(source.vertices), dtype=np.int64),
        faces=np.asarray(source.faces, dtype=np.int64),
        uvs=np.zeros((len(source.vertices), 2), dtype=np.float64),
    )

    rebuilt = _rebuild_geometry_with_uvs(source, (output,))
    rebuilt_colors = np.asarray(rebuilt.visual.vertex_attributes["color"])

    assert len(rebuilt.vertices) == len(source.faces) * 3
    np.testing.assert_array_equal(
        rebuilt_colors[np.asarray(rebuilt.faces)][:, 0],
        face_colors,
    )
    np.testing.assert_array_equal(
        rebuilt_colors[np.asarray(rebuilt.faces)][:, 1],
        face_colors,
    )
    np.testing.assert_array_equal(
        rebuilt_colors[np.asarray(rebuilt.faces)][:, 2],
        face_colors,
    )


# ### Failure and cancellation tests ###
def test_visibility_unwrap_rejects_already_textured_objects() -> None:
    with pytest.raises(ValueError, match="before object texturing"):
        unwrap_object_uvs_by_visibility(_textured_box_glb())


def test_visibility_unwrap_enforces_face_limit() -> None:
    with pytest.raises(ValueError, match="at most 10"):
        unwrap_object_uvs_by_visibility(_nested_boxes_glb(), max_face_count=10)


def test_visibility_unwrap_validates_target_domain() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        unwrap_object_uvs_by_visibility(
            _nested_boxes_glb(),
            target_domain=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="'full' or 'left_half'"):
        unwrap_object_uvs_by_visibility(
            _nested_boxes_glb(),
            target_domain="right_half",  # type: ignore[arg-type]
        )


def test_visibility_unwrap_honors_cancellation() -> None:
    with pytest.raises(VisibilityUvUnwrapCancelled):
        unwrap_object_uvs_by_visibility(
            _nested_boxes_glb(),
            cancellation_check=lambda: True,
        )


def test_visibility_reference_honors_cancellation_before_raycast() -> None:
    target_glb, reference_glb = _occluded_target_and_reference_glbs()
    checks = 0

    def cancel_after_reference_validation() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(VisibilityUvUnwrapCancelled):
        unwrap_object_uvs_by_visibility(
            target_glb,
            target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
            visibility_reference_glb=reference_glb,
            cancellation_check=cancel_after_reference_validation,
        )
    assert checks == 3
