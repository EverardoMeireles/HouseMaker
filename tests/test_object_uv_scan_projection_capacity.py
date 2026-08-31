# ### Imports ###
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.object_uv_scan_projection import (
    _BARYCENTRIC_EPSILON,
    _DESTINATION_PIXELS_PER_GROUP,
    SCAN_PROJECTION_TARGET_FULL,
    SCAN_PROJECTION_TARGET_LEFT_HALF,
    _FaceGroup,
    _PixelRectangle,
    _ProjectedFace,
    _bake_scanlines,
    _build_face_placements,
    _group_camera_faces,
    _partition_group_rectangles,
    _triangle_barycentric_weights,
    scan_project_textured_glb,
)


# ### Test constants ###
STRESS_TEXTURE_RESOLUTION = 32
EXTREME_PERCENTAGES = (1, 1, 1, 1, 1, 95)
LEFT_HALF_TARGET = _PixelRectangle(4, 4, 8, 24)


# ### Synthetic layout helpers ###
def _build_projected_faces(
    face_counts: tuple[int, ...],
) -> list[_ProjectedFace]:
    faces: list[_ProjectedFace] = []
    face_index = 0
    total = sum(face_counts)
    for camera_index, count in enumerate(face_counts):
        for camera_face_index in range(count):
            exponent = -12.0 + 24.0 * face_index / max(total - 1, 1)
            faces.append(
                _ProjectedFace(
                    geometry_name="capacity-stress",
                    local_face_index=face_index,
                    camera_index=camera_index,
                    source_positions=np.asarray(
                        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                        dtype=float,
                    ),
                    source_normals=np.asarray(
                        ((0.0, 0.0, 1.0),) * 3,
                        dtype=float,
                    ),
                    source_uvs=np.asarray(
                        ((0.1, 0.1), (0.9, 0.1), (0.1, 0.9)),
                        dtype=float,
                    ),
                    face_material_index=None,
                    projected_area=10.0**exponent,
                )
            )
            face_index += 1
    return faces


def _group_count(faces: list[_ProjectedFace]) -> int:
    return sum(
        len(
            _group_camera_faces(
                [face for face in faces if face.camera_index == camera_index]
            )
        )
        for camera_index in range(6)
    )


def _assert_dense_layout(
    faces: list[_ProjectedFace],
    percentages: tuple[int, ...],
    target: _PixelRectangle,
) -> None:
    placements = _build_face_placements(faces, percentages, target)
    source = np.full(
        (STRESS_TEXTURE_RESOLUTION, STRESS_TEXTURE_RESOLUTION, 4),
        127,
        dtype=np.uint8,
    )
    _texture, owner = _bake_scanlines(
        placements,
        source,
        STRESS_TEXTURE_RESOLUTION,
        None,
    )
    target_owner = owner[target.y : target.bottom, target.x : target.right]

    assert len(placements) == 2 * len(faces)
    assert np.count_nonzero(target_owner >= 0) / target_owner.size >= 0.98
    fragment_owner = np.full(
        (STRESS_TEXTURE_RESOLUTION, STRESS_TEXTURE_RESOLUTION),
        -1,
        dtype=np.int32,
    )
    for placement_index, placement in enumerate(placements):
        points = np.asarray(placement.destination_points, dtype=float)
        minimum_x = max(0, int(np.floor(np.min(points[:, 0]))))
        maximum_x = min(
            STRESS_TEXTURE_RESOLUTION,
            int(np.ceil(np.max(points[:, 0]))),
        )
        minimum_y = max(0, int(np.floor(np.min(points[:, 1]))))
        maximum_y = min(
            STRESS_TEXTURE_RESOLUTION,
            int(np.ceil(np.max(points[:, 1]))),
        )
        for row in range(minimum_y, maximum_y):
            columns = np.arange(minimum_x, maximum_x, dtype=float)
            sample_points = np.column_stack(
                (
                    columns + 0.5,
                    np.full(len(columns), row + 0.5),
                )
            )
            barycentric = _triangle_barycentric_weights(sample_points, points)
            inside = np.all(
                barycentric >= -_BARYCENTRIC_EPSILON,
                axis=1,
            )
            integer_columns = columns.astype(np.int64)
            inside &= fragment_owner[row, integer_columns] < 0
            fragment_owner[row, integer_columns[inside]] = placement_index
    fragment_counts = np.bincount(
        fragment_owner[fragment_owner >= 0],
        minlength=len(placements),
    )
    assert np.all(fragment_counts >= 1)
    repeated = _build_face_placements(faces, percentages, target)
    assert len(repeated) == len(placements)
    for first, second in zip(placements, repeated, strict=True):
        np.testing.assert_array_equal(
            first.destination_points,
            second.destination_points,
        )


# ### Private capacity regression tests ###
@pytest.mark.parametrize("seed", range(12))
def test_scan_row_tiler_never_rejects_a_group_set_that_fits(seed: int) -> None:
    generator = np.random.default_rng(seed)
    for _case_index in range(80):
        width = int(generator.integers(2, 49))
        height = int(generator.integers(1, 49))
        group_capacity = (
            width * height // _DESTINATION_PIXELS_PER_GROUP
        )
        group_count = int(generator.integers(1, group_capacity + 1))
        weights = 10.0 ** generator.uniform(-12.0, 12.0, group_count)
        groups = tuple(
            _FaceGroup(faces=(), weight=float(weight)) for weight in weights
        )
        target = _PixelRectangle(3, 7, width, height)

        rectangles = _partition_group_rectangles(target, groups)
        repeated = _partition_group_rectangles(target, groups)

        assert rectangles == repeated
        assert len(rectangles) == group_count
        coverage = np.zeros((height, width), dtype=np.uint8)
        for rectangle in rectangles:
            assert rectangle.width >= 1
            assert rectangle.height >= 1
            assert target.x <= rectangle.x < rectangle.right <= target.right
            assert target.y <= rectangle.y < rectangle.bottom <= target.bottom
            local_x = rectangle.x - target.x
            local_y = rectangle.y - target.y
            coverage[
                local_y : local_y + rectangle.height,
                local_x : local_x + rectangle.width,
            ] += 1
        assert np.all(coverage == 1)


@pytest.mark.parametrize(
    "width, height",
    ((3, 28), (5, 25), (7, 34), (17, 35)),
)
def test_scan_row_tiler_reaches_exact_odd_rectangle_capacity(
    width: int,
    height: int,
) -> None:
    group_count = width * height // _DESTINATION_PIXELS_PER_GROUP
    groups = tuple(
        _FaceGroup(faces=(), weight=(1e-12 if index % 2 else 1e12))
        for index in range(group_count)
    )
    target = _PixelRectangle(3, 7, width, height)

    rectangles = _partition_group_rectangles(target, groups)

    coverage = np.zeros((height, width), dtype=np.uint8)
    for rectangle in rectangles:
        local_x = rectangle.x - target.x
        local_y = rectangle.y - target.y
        coverage[
            local_y : local_y + rectangle.height,
            local_x : local_x + rectangle.width,
        ] += 1
    assert len(rectangles) == group_count
    assert np.all(coverage == 1)


@pytest.mark.parametrize(
    "face_counts, percentages, target",
    (
        (
            (76, 4, 4, 4, 4, 4),
            EXTREME_PERCENTAGES,
            LEFT_HALF_TARGET,
        ),
        (
            (17, 17, 16, 16, 15, 15),
            EXTREME_PERCENTAGES,
            LEFT_HALF_TARGET,
        ),
        (
            (144, 120, 80, 70, 50, 48),
            EXTREME_PERCENTAGES,
            _PixelRectangle(0, 0, 32, 32),
        ),
        (
            (14, 14, 14, 14, 16, 15),
            EXTREME_PERCENTAGES,
            _PixelRectangle(0, 0, 7, 25),
        ),
    ),
)
def test_camera_bands_honor_global_capacity_under_extreme_inputs(
    face_counts: tuple[int, ...],
    percentages: tuple[int, ...],
    target: _PixelRectangle,
) -> None:
    faces = _build_projected_faces(face_counts)
    assert (
        _group_count(faces) * _DESTINATION_PIXELS_PER_GROUP
        <= target.width * target.height
    )

    _assert_dense_layout(faces, percentages, target)


# ### End-to-end asset helpers ###
def _directional_face_basis(
    camera_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        (
            (np.asarray((0.0, 1.0, 0.0)), np.asarray((0.0, 0.0, 1.0))),
            (np.asarray((0.0, 0.0, 1.0)), np.asarray((0.0, 1.0, 0.0))),
            (np.asarray((0.0, 1.0, 0.0)), np.asarray((1.0, 0.0, 0.0))),
            (np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 1.0, 0.0))),
            (np.asarray((0.0, 0.0, 1.0)), np.asarray((1.0, 0.0, 0.0))),
            (np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 0.0, 1.0))),
        )[camera_index]
    )


def _build_directional_stress_glb(
    face_counts: tuple[int, ...],
    texture_resolution: int = STRESS_TEXTURE_RESOLUTION,
) -> bytes:
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    uvs: list[tuple[float, float]] = []
    total = sum(face_counts)
    face_index = 0
    for camera_index, count in enumerate(face_counts):
        first_axis, second_axis = _directional_face_basis(camera_index)
        for _camera_face_index in range(count):
            scale = 10.0 ** (-2.0 + 4.0 * face_index / max(total - 1, 1))
            vertex_offset = len(vertices)
            vertices.extend(
                (
                    np.zeros(3, dtype=float),
                    first_axis * scale,
                    second_axis * scale,
                )
            )
            faces.append(
                (vertex_offset, vertex_offset + 1, vertex_offset + 2)
            )
            uvs.extend(((0.1, 0.1), (0.9, 0.1), (0.1, 0.9)))
            face_index += 1
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    columns = np.linspace(
        20,
        230,
        texture_resolution,
        dtype=np.uint8,
    )
    texture = np.empty(
        (texture_resolution, texture_resolution, 4),
        dtype=np.uint8,
    )
    texture[:, :, 0] = columns[np.newaxis, :]
    texture[:, :, 1] = columns[:, np.newaxis]
    texture[:, :, 2] = 117
    texture[:, :, 3] = 255
    mesh.visual = TextureVisuals(
        uv=np.asarray(uvs, dtype=float),
        material=PBRMaterial(
            baseColorTexture=Image.fromarray(texture, mode="RGBA")
        ),
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


# ### End-to-end capacity tests ###
@pytest.mark.parametrize(
    "target_domain, face_counts, texture_resolution",
    (
        (
            SCAN_PROJECTION_TARGET_FULL,
            (300, 16, 16, 16, 16, 16),
            STRESS_TEXTURE_RESOLUTION,
        ),
        (
            SCAN_PROJECTION_TARGET_LEFT_HALF,
            (17, 17, 16, 16, 15, 15),
            STRESS_TEXTURE_RESOLUTION,
        ),
        (
            SCAN_PROJECTION_TARGET_LEFT_HALF,
            (20, 20, 20, 20, 19, 18),
            34,
        ),
    ),
)
def test_public_projection_is_dense_and_deterministic_at_high_capacity(
    target_domain: str,
    face_counts: tuple[int, ...],
    texture_resolution: int,
) -> None:
    source = _build_directional_stress_glb(
        face_counts,
        texture_resolution,
    )

    first = scan_project_textured_glb(
        source,
        EXTREME_PERCENTAGES,
        target_domain=target_domain,
        texture_resolution=texture_resolution,
    )
    second = scan_project_textured_glb(
        source,
        EXTREME_PERCENTAGES,
        target_domain=target_domain,
        texture_resolution=texture_resolution,
    )

    assert first.stats.triangle_occupancy >= 0.98
    assert first.stats == second.stats
    assert hashlib.sha256(first.glb_bytes).digest() == hashlib.sha256(
        second.glb_bytes
    ).digest()
