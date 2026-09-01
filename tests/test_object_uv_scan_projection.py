# ### Imports ###
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest
import trimesh
from PIL import Image
from trimesh.visual.material import MultiMaterial, PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.glass_material import (
    HOUSEMAKER_GLASS_BASE_COLOR_FACTOR,
    is_housemaker_glass_material,
)
from housemaker.object_texture_variants import (
    MATERIAL_TEXTURE_METALLIC_ROUGHNESS,
    MATERIAL_TEXTURE_NORMAL,
    TEXTURE_RESOLUTIONS,
    _collect_material_texture_maps,
    _collect_material_textures,
    _load_glb_scene,
    _replace_material_textures,
    _validate_shared_texture,
    build_object_texture_variants,
)
from housemaker.object_face_edit import (
    delete_object_faces_preserving_uvs,
    load_object_face_geometry,
)
from housemaker.object_uv_scan_projection import (
    DEFAULT_PROJECTION_CAMERA_PERCENTAGES,
    LEFT_HALF_OUTER_SAFETY_INSET_PIXELS,
    SCAN_PROJECTION_ISLAND_PADDING_PIXELS,
    SCAN_PROJECTION_MINIMUM_VISIBLE_FRACTION,
    SCAN_PROJECTION_TARGET_FULL,
    SCAN_PROJECTION_TARGET_LEFT_HALF,
    SCAN_PROJECTION_TARGET_TOP_LEFT_QUARTER,
    SCAN_PROJECTION_VERSION,
    TOP_LEFT_QUARTER_OUTER_SAFETY_INSET_PIXELS,
    ScanProjectionCancelled,
    _FALLBACK_CAMERA_INDEX,
    _FaceGroup,
    _FacePlacement,
    _PixelRectangle,
    _ProjectedFace,
    _SceneGeometry,
    _assign_faces_to_cameras,
    _build_face_placements,
    _build_effective_group_percentages,
    _build_geometry_glass_material,
    _fit_selected_faces_rectangle,
    _normalize_glass_face_indices,
    _partition_group_rectangles,
    _rasterize_visibility_face_samples,
    _select_face_camera,
    normalize_projection_camera_percentages,
    scan_project_textured_glb,
)
from housemaker.unused_face_removal import ALL_CAMERA_IDS
from housemaker.scan_projection_layout import (
    SCAN_PROJECTION_LAYOUT_METADATA_KEY,
    remap_scan_projection_scene_uvs,
)


# ### Test constants ###
TEST_TEXTURE_RESOLUTION = 128
CUSTOM_PERCENTAGES = (30, 10, 20, 10, 20, 10)


# ### Test asset helpers ###
def _gradient_rgba(resolution: int = TEST_TEXTURE_RESOLUTION) -> np.ndarray:
    columns = np.linspace(31, 231, resolution, dtype=np.uint8)
    rows = np.linspace(37, 237, resolution, dtype=np.uint8)
    red = np.broadcast_to(columns[np.newaxis, :], (resolution, resolution))
    green = np.broadcast_to(rows[:, np.newaxis], (resolution, resolution))
    blue = np.full((resolution, resolution), 113, dtype=np.uint8)
    alpha = np.full((resolution, resolution), 255, dtype=np.uint8)
    return np.ascontiguousarray(np.dstack((red, green, blue, alpha)))


def _build_textured_cube_glb(
    *,
    texture: np.ndarray | None = None,
    transform: np.ndarray | None = None,
) -> bytes:
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh.unmerge_vertices()
    face_uvs = np.asarray(
        (
            (0.08, 0.12),
            (0.91, 0.18),
            (0.21, 0.89),
        ),
        dtype=float,
    )
    uvs = np.vstack([face_uvs for _face in mesh.faces])
    material = PBRMaterial(
        name="scan-gradient",
        baseColorTexture=Image.fromarray(
            _gradient_rgba() if texture is None else texture,
            mode="RGBA",
        ),
        metallicFactor=0.15,
        roughnessFactor=0.72,
    )
    mesh.visual = TextureVisuals(uv=uvs, material=material)
    scene = trimesh.Scene()
    scene.add_geometry(
        mesh,
        geom_name="cube-geometry",
        node_name="cube-node",
        transform=np.eye(4) if transform is None else transform,
    )
    return bytes(scene.export(file_type="glb"))


def _build_textured_triangle_glb() -> bytes:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=np.asarray(((0.08, 0.12), (0.91, 0.18), (0.21, 0.89))),
        material=PBRMaterial(
            baseColorTexture=Image.fromarray(_gradient_rgba(), mode="RGBA")
        ),
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _build_subdivided_glass_panel_glb(
    *,
    transform: np.ndarray | None = None,
) -> bytes:
    """Build four triangles whose slightly uneven surface fits one panel."""

    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            (
                (-1.0, -0.5, 0.00),
                (1.0, -0.5, 0.02),
                (1.0, 0.5, 0.00),
                (-1.0, 0.5, -0.02),
                (0.0, 0.0, 0.04),
            ),
            dtype=float,
        ),
        faces=np.asarray(
            ((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
            dtype=np.int64,
        ),
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=np.asarray(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (0.5, 0.5),
            ),
            dtype=float,
        ),
        material=PBRMaterial(
            baseColorTexture=Image.fromarray(_gradient_rgba(), mode="RGBA")
        ),
    )
    scene = trimesh.Scene()
    scene.add_geometry(
        mesh,
        geom_name="panel-geometry",
        node_name="panel-node",
        transform=np.eye(4) if transform is None else transform,
    )
    retained = trimesh.Trimesh(
        vertices=np.asarray(
            ((3.0, 0.0, 0.0), (4.0, 0.0, 0.0), (3.0, 1.0, 0.0)),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
    )
    retained.visual = TextureVisuals(
        uv=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        material=PBRMaterial(
            baseColorTexture=Image.fromarray(_gradient_rgba(), mode="RGBA")
        ),
    )
    scene.add_geometry(
        retained,
        geom_name="retained-geometry",
        node_name="retained-node",
        transform=np.eye(4) if transform is None else transform,
    )
    return bytes(scene.export(file_type="glb"))


def _build_split_geometry_glass_panel_glb() -> bytes:
    """Build one rectangle split across two independently named meshes."""

    scene = trimesh.Scene()
    texture = Image.fromarray(_gradient_rgba(), mode="RGBA")
    triangle_specs = (
        (
            "panel-a",
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0)),
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        ),
        (
            "panel-b",
            ((0.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        ),
    )
    for name, vertices, uvs in triangle_specs:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=float),
            faces=np.asarray(((0, 1, 2),), dtype=np.int64),
            process=False,
        )
        mesh.visual = TextureVisuals(
            uv=np.asarray(uvs, dtype=float),
            material=PBRMaterial(baseColorTexture=texture.copy()),
        )
        scene.add_geometry(mesh, geom_name=name, node_name=f"{name}-node")
    retained = trimesh.Trimesh(
        vertices=np.asarray(
            ((3.0, 0.0, 0.0), (4.0, 0.0, 0.0), (3.0, 1.0, 0.0)),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
    )
    retained.visual = TextureVisuals(
        uv=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        material=PBRMaterial(baseColorTexture=texture.copy()),
    )
    scene.add_geometry(
        retained,
        geom_name="retained",
        node_name="retained-node",
    )
    return bytes(scene.export(file_type="glb"))


def _build_low_percentage_many_faces_glb(
    *,
    positive_x_face_count: int = 200,
    texture_resolution: int = 64,
) -> bytes:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    uvs: list[tuple[float, float]] = []
    source_triangle_uvs = ((0.08, 0.12), (0.91, 0.18), (0.21, 0.89))
    for face_index in range(positive_x_face_count):
        offset = face_index * 0.002
        next_vertex = len(vertices)
        vertices.extend(
            (
                (0.0, offset, 0.0),
                (0.0, offset + 1.0, 0.0),
                (0.0, offset, 1.0),
            )
        )
        faces.append((next_vertex, next_vertex + 1, next_vertex + 2))
        uvs.extend(source_triangle_uvs)
    next_vertex = len(vertices)
    vertices.extend(((2.0, 0.0, 0.0), (2.0, 0.0, 1.0), (2.0, 1.0, 0.0)))
    faces.append((next_vertex, next_vertex + 1, next_vertex + 2))
    uvs.extend(source_triangle_uvs)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=np.asarray(uvs, dtype=float),
        material=PBRMaterial(
            baseColorTexture=Image.fromarray(
                _gradient_rgba(texture_resolution),
                mode="RGBA",
            )
        ),
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _projected_face(
    local_face_index: int,
    camera_index: int,
    *,
    projected_area: float = 1.0,
) -> _ProjectedFace:
    return _ProjectedFace(
        geometry_name="geometry",
        local_face_index=local_face_index,
        camera_index=camera_index,
        source_positions=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        ),
        source_normals=np.asarray(((0.0, 0.0, 1.0),) * 3, dtype=float),
        source_uvs=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=float),
        face_material_index=None,
        projected_area=projected_area,
    )


def _scene_geometry_from_triangles(
    triangles: np.ndarray,
) -> _SceneGeometry:
    vertices = np.asarray(triangles, dtype=float).reshape((-1, 3))
    faces = np.arange(len(vertices), dtype=np.int64).reshape((-1, 3))
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=np.tile(
            np.asarray(((0.1, 0.1), (0.9, 0.1), (0.1, 0.9))),
            (len(faces), 1),
        ),
        material=PBRMaterial(
            baseColorTexture=Image.fromarray(_gradient_rgba(), mode="RGBA")
        ),
    )
    return _SceneGeometry(
        geometry_name="visibility-test",
        mesh=mesh,
        world_vertices=vertices.copy(),
    )


def _load_single_mesh_and_texture(
    payload: bytes,
) -> tuple[trimesh.Scene, trimesh.Trimesh, np.ndarray]:
    scene = _load_glb_scene(payload)
    meshes = [
        geometry
        for geometry in scene.geometry.values()
        if isinstance(geometry, trimesh.Trimesh)
    ]
    assert len(meshes) == 1
    texture = _validate_shared_texture(_collect_material_textures(scene))
    return scene, meshes[0], texture


def _sample_source_gradient(uv: np.ndarray) -> np.ndarray:
    source = _gradient_rgba()
    height, width = source.shape[:2]
    u = float(uv[0] % 1.0)
    v = float(uv[1] % 1.0)
    pixel_x = u * width - 0.5
    pixel_y = (1.0 - v) * height - 0.5
    x0_unwrapped = int(np.floor(pixel_x))
    y0_unwrapped = int(np.floor(pixel_y))
    fraction_x = pixel_x - x0_unwrapped
    fraction_y = pixel_y - y0_unwrapped
    x0 = x0_unwrapped % width
    y0 = y0_unwrapped % height
    x1 = (x0_unwrapped + 1) % width
    y1 = (y0_unwrapped + 1) % height
    top = source[y0, x0].astype(float) * (1.0 - fraction_x)
    top += source[y0, x1].astype(float) * fraction_x
    bottom = source[y1, x0].astype(float) * (1.0 - fraction_x)
    bottom += source[y1, x1].astype(float) * fraction_x
    return top * (1.0 - fraction_y) + bottom * fraction_y


def _sample_clamp_bilinear_rgba(
    texture: np.ndarray,
    uv: np.ndarray,
) -> np.ndarray:
    height, width = texture.shape[:2]
    pixel_x = float(uv[0]) * width - 0.5
    pixel_y = (1.0 - float(uv[1])) * height - 0.5
    x0 = int(np.floor(pixel_x))
    y0 = int(np.floor(pixel_y))
    fraction_x = pixel_x - x0
    fraction_y = pixel_y - y0
    x0 = min(max(x0, 0), width - 1)
    y0 = min(max(y0, 0), height - 1)
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    top = texture[y0, x0].astype(float) * (1.0 - fraction_x)
    top += texture[y0, x1].astype(float) * fraction_x
    bottom = texture[y1, x0].astype(float) * (1.0 - fraction_x)
    bottom += texture[y1, x1].astype(float) * fraction_x
    return top * (1.0 - fraction_y) + bottom * fraction_y


def _face_scan_cell(
    face_uvs: np.ndarray,
    *,
    texture_resolution: int,
) -> tuple[int, int, int, int]:
    canonical_resolution = 2048
    reduction = canonical_resolution // texture_resolution
    points = np.column_stack(
        (
            face_uvs[:, 0] * canonical_resolution,
            (1.0 - face_uvs[:, 1]) * canonical_resolution,
        )
    )
    inset = reduction * 0.5
    minimums = np.rint(np.min(points, axis=0) - inset).astype(int)
    maximums = np.rint(np.max(points, axis=0) + inset).astype(int)
    return (
        int(minimums[0]),
        int(minimums[1]),
        int(maximums[0] - minimums[0]),
        int(maximums[1] - minimums[1]),
    )


# ### Percentage validation tests ###
def test_normalizes_sequence_and_mapping_in_canonical_camera_order() -> None:
    assert sum(DEFAULT_PROJECTION_CAMERA_PERCENTAGES) == 100
    assert normalize_projection_camera_percentages(CUSTOM_PERCENTAGES) == (
        CUSTOM_PERCENTAGES
    )
    mapping = {
        camera_id: CUSTOM_PERCENTAGES[index]
        for index, camera_id in reversed(tuple(enumerate(ALL_CAMERA_IDS)))
    }
    assert normalize_projection_camera_percentages(mapping) == CUSTOM_PERCENTAGES


@pytest.mark.parametrize(
    "values, message",
    (
        ((20, 20), "six values"),
        ((20, 20, 20, 20, 10, 5), "exactly 100"),
        ((20, 20, 20, 20, 10, 10.0), "integers"),
        ((20, 20, 20, 20, 10, -10), "between 0 and 100"),
        ({camera_id: 20 for camera_id in ALL_CAMERA_IDS[:-1]}, "each canonical"),
    ),
)
def test_rejects_invalid_camera_percentages(values: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_projection_camera_percentages(values)  # type: ignore[arg-type]


# ### Projection quality tests ###
def test_dense_scan_bake_fills_cube_without_dots_and_preserves_gradients() -> None:
    source_glb = _build_textured_cube_glb()

    result = scan_project_textured_glb(
        source_glb,
        DEFAULT_PROJECTION_CAMERA_PERCENTAGES,
        texture_resolution=TEST_TEXTURE_RESOLUTION,
    )

    _source_scene, source_mesh, _source_texture = _load_single_mesh_and_texture(
        source_glb
    )
    _scene, output_mesh, output_texture = _load_single_mesh_and_texture(
        result.glb_bytes
    )
    assert result.stats.triangle_occupancy >= 0.999
    assert result.stats.covered_pixel_count == result.stats.usable_pixel_count
    assert not np.any(np.all(output_texture[:, :, :3] == 0, axis=2))
    assert len(np.unique(output_texture[:, :, :3].reshape((-1, 3)), axis=0)) > 500

    output_uvs = np.asarray(output_mesh.visual.uv, dtype=float)
    output_face = np.asarray(output_mesh.faces[0], dtype=np.int64)
    output_triangle_uvs = output_uvs[output_face]
    destination_points = np.column_stack(
        (
            output_triangle_uvs[:, 0] * TEST_TEXTURE_RESOLUTION,
            (1.0 - output_triangle_uvs[:, 1]) * TEST_TEXTURE_RESOLUTION,
        )
    )
    centroid = np.mean(destination_points, axis=0)
    column = int(np.floor(centroid[0]))
    row = int(np.floor(centroid[1]))
    pixel_center = np.asarray((column + 0.5, row + 0.5))
    matrix = np.vstack((destination_points.T, np.ones(3)))
    barycentric = np.linalg.solve(
        matrix,
        np.asarray((pixel_center[0], pixel_center[1], 1.0)),
    )
    surface_position = barycentric @ np.asarray(
        output_mesh.vertices[output_face],
        dtype=float,
    )
    source_face = np.asarray(source_mesh.faces[0], dtype=np.int64)
    source_positions = np.asarray(source_mesh.vertices[source_face], dtype=float)
    source_barycentric = np.linalg.lstsq(
        np.vstack((source_positions.T, np.ones(3))),
        np.append(surface_position, 1.0),
        rcond=None,
    )[0]
    source_uv = source_barycentric @ np.asarray(
        source_mesh.visual.uv[source_face],
        dtype=float,
    )
    expected = _sample_source_gradient(source_uv)
    np.testing.assert_allclose(
        output_texture[row, column].astype(float),
        expected,
        atol=1.5,
    )


def test_texture_variants_keep_every_scan_edge_inside_its_own_texels() -> None:
    canonical_resolution = 2048
    source_texture = _gradient_rgba(canonical_resolution)
    projected = scan_project_textured_glb(
        _build_textured_cube_glb(texture=source_texture),
        texture_resolution=canonical_resolution,
    )
    scene, mesh, _texture = _load_single_mesh_and_texture(projected.glb_bytes)
    source_uvs = np.asarray(mesh.visual.uv, dtype=float)
    source_faces = np.asarray(mesh.faces, dtype=np.int64)
    cells = {
        _face_scan_cell(
            source_uvs[face],
            texture_resolution=canonical_resolution,
        )
        for face in source_faces
    }
    color_by_cell = {
        cell: np.asarray(
            (
                31 + cell_index * 13,
                47 + cell_index * 7,
                71 + cell_index * 5,
                255,
            ),
            dtype=np.uint8,
        )
        for cell_index, cell in enumerate(sorted(cells))
    }
    cell_texture = np.zeros(
        (canonical_resolution, canonical_resolution, 4),
        dtype=np.uint8,
    )
    for (x, y, width, height), color in color_by_cell.items():
        cell_texture[y : y + height, x : x + width] = color
    _replace_material_textures(scene, [cell_texture])
    tagged_glb = bytes(scene.export(file_type="glb"))

    variants = build_object_texture_variants(tagged_glb)

    assert variants is not None
    for resolution in TEXTURE_RESOLUTIONS:
        _variant_scene, variant_mesh, variant_texture = (
            _load_single_mesh_and_texture(variants.glb_by_resolution[resolution])
        )
        layout = variant_mesh.metadata[SCAN_PROJECTION_LAYOUT_METADATA_KEY]
        assert layout["uv_texture_resolution"] == resolution
        variant_uvs = np.asarray(variant_mesh.visual.uv, dtype=float)
        variant_faces = np.asarray(variant_mesh.faces, dtype=np.int64)
        variant_points = np.column_stack(
            (
                variant_uvs[:, 0] * resolution,
                (1.0 - variant_uvs[:, 1]) * resolution,
            )
        )
        np.testing.assert_allclose(
            np.mod(variant_points, 1.0),
            0.5,
            rtol=0.0,
            atol=1e-6,
        )
        for face in variant_faces:
            triangle_uvs = variant_uvs[face]
            expected = color_by_cell[
                _face_scan_cell(
                    triangle_uvs,
                    texture_resolution=resolution,
                )
            ]
            centroid = np.mean(triangle_uvs, axis=0)
            for edge_index in range(3):
                edge_midpoint = (
                    triangle_uvs[edge_index]
                    + triangle_uvs[(edge_index + 1) % 3]
                ) * 0.5
                just_inside = edge_midpoint * 0.9999 + centroid * 0.0001
                np.testing.assert_allclose(
                    _sample_clamp_bilinear_rgba(
                        variant_texture,
                        just_inside,
                    ),
                    expected,
                    rtol=0.0,
                    atol=0.01,
                )

    edited = delete_object_faces_preserving_uvs(tagged_glb, {0})
    edited_variants = build_object_texture_variants(edited.glb_bytes)
    assert edited_variants is not None
    for resolution in TEXTURE_RESOLUTIONS:
        _edited_scene, edited_mesh, _edited_texture = (
            _load_single_mesh_and_texture(
                edited_variants.glb_by_resolution[resolution]
            )
        )
        assert len(edited_mesh.faces) == len(source_faces) - 1
        edited_layout = edited_mesh.metadata[
            SCAN_PROJECTION_LAYOUT_METADATA_KEY
        ]
        assert edited_layout["uv_texture_resolution"] == resolution

    multi_scene = trimesh.Scene()
    canonical_faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    for instance_index in range(2):
        multi_scene.add_geometry(
            mesh.copy(),
            geom_name=f"scan-geometry-{instance_index}",
            node_name=f"scan-node-{instance_index}",
        )
    assert remap_scan_projection_scene_uvs(multi_scene, 512)
    for geometry in multi_scene.geometry.values():
        np.testing.assert_array_equal(geometry.faces, canonical_faces)
        assert geometry.visual.material.name == mesh.visual.material.name
        geometry_uvs = np.asarray(geometry.visual.uv, dtype=float)
        geometry_points = np.column_stack(
            (
                geometry_uvs[:, 0] * 512,
                (1.0 - geometry_uvs[:, 1]) * 512,
            )
        )
        np.testing.assert_allclose(
            np.mod(geometry_points, 1.0),
            0.5,
            rtol=0.0,
            atol=1e-6,
        )


def test_custom_view_percentages_control_pixel_area_and_are_auditable() -> None:
    result = scan_project_textured_glb(
        _build_textured_cube_glb(),
        CUSTOM_PERCENTAGES,
        texture_resolution=TEST_TEXTURE_RESOLUTION,
    )

    stats = result.stats
    assert stats.version == SCAN_PROJECTION_VERSION
    assert stats.camera_percentages == CUSTOM_PERCENTAGES
    assert stats.view_face_counts == (2, 2, 2, 2, 2, 2)
    achieved_percentages = np.asarray(stats.view_pixel_counts, dtype=float)
    achieved_percentages *= 100.0 / stats.covered_pixel_count
    np.testing.assert_allclose(achieved_percentages, CUSTOM_PERCENTAGES, atol=2.0)
    pipeline = stats.to_pipeline_dict()
    assert pipeline["camera_percentages"] == dict(
        zip(ALL_CAMERA_IDS, CUSTOM_PERCENTAGES, strict=True)
    )
    assert pipeline["triangle_occupancy"] >= 0.999
    assert pipeline["output_face_count"] == stats.output_face_count


def test_left_half_keeps_outer_safety_inset_but_no_island_padding() -> None:
    result = scan_project_textured_glb(
        _build_textured_cube_glb(),
        target_domain=SCAN_PROJECTION_TARGET_LEFT_HALF,
        texture_resolution=TEST_TEXTURE_RESOLUTION,
    )
    _scene, mesh, texture = _load_single_mesh_and_texture(result.glb_bytes)
    uvs = np.asarray(mesh.visual.uv, dtype=float)
    inset = LEFT_HALF_OUTER_SAFETY_INSET_PIXELS / TEST_TEXTURE_RESOLUTION

    assert result.stats.target_domain == SCAN_PROJECTION_TARGET_LEFT_HALF
    assert result.stats.island_padding_pixels == SCAN_PROJECTION_ISLAND_PADDING_PIXELS
    assert (
        result.stats.outer_safety_inset_pixels
        == LEFT_HALF_OUTER_SAFETY_INSET_PIXELS
    )
    assert result.stats.target_width == (
        TEST_TEXTURE_RESOLUTION // 2
        - 2 * LEFT_HALF_OUTER_SAFETY_INSET_PIXELS
    )
    assert np.min(uvs[:, 0]) >= inset - 1e-9
    assert np.max(uvs[:, 0]) <= 0.5 - inset + 1e-9
    assert np.min(uvs[:, 1]) >= inset - 1e-9
    assert np.max(uvs[:, 1]) <= 1.0 - inset + 1e-9
    assert np.all(texture[:, TEST_TEXTURE_RESOLUTION // 2 :, :3] == 0)
    assert result.stats.triangle_occupancy >= 0.999


def test_top_left_quarter_uses_the_legacy_content_gutter() -> None:
    result = scan_project_textured_glb(
        _build_textured_cube_glb(
            texture=_gradient_rgba(TEST_TEXTURE_RESOLUTION)
        ),
        target_domain=SCAN_PROJECTION_TARGET_TOP_LEFT_QUARTER,
        texture_resolution=TEST_TEXTURE_RESOLUTION,
    )
    _scene, mesh, texture = _load_single_mesh_and_texture(result.glb_bytes)
    uvs = np.asarray(mesh.visual.uv, dtype=float)
    inset = (
        TOP_LEFT_QUARTER_OUTER_SAFETY_INSET_PIXELS
        / TEST_TEXTURE_RESOLUTION
    )

    assert (
        result.stats.target_domain
        == SCAN_PROJECTION_TARGET_TOP_LEFT_QUARTER
    )
    assert (
        result.stats.outer_safety_inset_pixels
        == TOP_LEFT_QUARTER_OUTER_SAFETY_INSET_PIXELS
    )
    assert result.stats.target_width == (
        TEST_TEXTURE_RESOLUTION // 2
        - 2 * TOP_LEFT_QUARTER_OUTER_SAFETY_INSET_PIXELS
    )
    assert result.stats.target_height == result.stats.target_width
    assert np.min(uvs[:, 0]) >= inset - 1e-9
    assert np.max(uvs[:, 0]) <= 0.5 - inset + 1e-9
    assert np.min(uvs[:, 1]) >= 0.5 + inset - 1e-9
    assert np.max(uvs[:, 1]) <= 1.0 - inset + 1e-9
    target_right = (
        TEST_TEXTURE_RESOLUTION // 2
        - TOP_LEFT_QUARTER_OUTER_SAFETY_INSET_PIXELS
    )
    assert np.all(texture[:, target_right:, :3] == 0)
    assert np.all(texture[target_right:, :, :3] == 0)
    assert result.stats.triangle_occupancy >= 0.999


def test_repeat_source_uvs_are_inverse_sampled_without_an_atlas_error() -> None:
    scene, mesh, _texture = _load_single_mesh_and_texture(
        _build_textured_cube_glb()
    )
    repeated_uvs = np.asarray(mesh.visual.uv, dtype=float).copy()
    repeated_uvs[:, 0] = repeated_uvs[:, 0] * 2.5 - 0.75
    repeated_uvs[:, 1] = repeated_uvs[:, 1] * 3.0 + 0.4
    mesh.visual.uv = repeated_uvs

    result = scan_project_textured_glb(
        bytes(scene.export(file_type="glb")),
        target_domain=SCAN_PROJECTION_TARGET_LEFT_HALF,
        texture_resolution=TEST_TEXTURE_RESOLUTION,
    )

    _output_scene, output_mesh, output_texture = _load_single_mesh_and_texture(
        result.glb_bytes
    )
    output_uvs = np.asarray(output_mesh.visual.uv, dtype=float)
    assert result.stats.triangle_occupancy >= 0.999
    assert np.all(np.isfinite(output_uvs))
    assert np.min(output_uvs) >= 0.0
    assert np.max(output_uvs[:, 0]) <= 0.5
    assert np.any(output_texture[:, :, :3] != 0)


def test_single_populated_view_redistributes_space_and_splits_one_face() -> None:
    result = scan_project_textured_glb(
        _build_textured_triangle_glb(),
        CUSTOM_PERCENTAGES,
        target_domain=SCAN_PROJECTION_TARGET_LEFT_HALF,
        texture_resolution=TEST_TEXTURE_RESOLUTION,
    )
    _scene, output_mesh, _texture = _load_single_mesh_and_texture(
        result.glb_bytes
    )

    assert result.stats.face_count == 1
    assert result.stats.output_face_count == 2
    assert result.stats.output_vertex_count == 6
    assert len(output_mesh.faces) == 2
    assert result.stats.triangle_occupancy >= 0.999
    assert sum(result.stats.view_pixel_counts) == result.stats.usable_pixel_count
    np.testing.assert_allclose(
        output_mesh.bounds,
        np.asarray(((0.0, 0.0, 0.0), (2.0, 1.0, 0.0))),
    )


def test_projection_is_deterministic_and_preserves_transform_and_faces() -> None:
    transform = trimesh.transformations.translation_matrix((3.0, -2.0, 1.5))
    source_glb = _build_textured_cube_glb(transform=transform)
    first = scan_project_textured_glb(
        source_glb,
        CUSTOM_PERCENTAGES,
        texture_resolution=TEST_TEXTURE_RESOLUTION,
    )
    second = scan_project_textured_glb(
        source_glb,
        CUSTOM_PERCENTAGES,
        texture_resolution=TEST_TEXTURE_RESOLUTION,
    )

    assert first.stats == second.stats
    assert hashlib.sha256(first.glb_bytes).digest() == hashlib.sha256(
        second.glb_bytes
    ).digest()
    source_scene, source_mesh, _texture = _load_single_mesh_and_texture(source_glb)
    output_scene, output_mesh, _texture = _load_single_mesh_and_texture(
        first.glb_bytes
    )
    source_node = next(iter(source_scene.graph.nodes_geometry))
    output_node = next(iter(output_scene.graph.nodes_geometry))
    source_transform, _source_geometry = source_scene.graph.get(source_node)
    output_transform, _output_geometry = output_scene.graph.get(output_node)
    np.testing.assert_allclose(output_transform, source_transform)
    assert len(source_mesh.faces) <= len(output_mesh.faces) <= 2 * len(
        source_mesh.faces
    )
    np.testing.assert_allclose(output_mesh.bounds, source_mesh.bounds)
    np.testing.assert_allclose(
        np.unique(np.asarray(output_mesh.vertex_normals), axis=0),
        np.unique(np.asarray(source_mesh.vertex_normals), axis=0),
        atol=1e-6,
    )
    assert output_mesh.visual.material.roughnessFactor == pytest.approx(
        source_mesh.visual.material.roughnessFactor
    )
    assert first.stats.output_vertex_count == 3 * len(output_mesh.faces)
    assert first.stats.output_face_count == len(output_mesh.faces)


def test_reversed_coplanar_faces_choose_high_percentage_camera() -> None:
    geometry = _scene_geometry_from_triangles(
        np.asarray(
            (
                ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
                ((2.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 2.0, 0.0)),
                ((1.0, 1.0, 0.0), (0.0, 2.0, 0.0), (2.0, 2.0, 0.0)),
            ),
            dtype=float,
        )
    )
    percentages = (0, 0, 0, 0, 90, 10)

    faces = _assign_faces_to_cameras((geometry,), percentages, None)

    assert [face.camera_index for face in faces] == [4, 4, 4]
    assert all(
        face.visible_fraction >= SCAN_PROJECTION_MINIMUM_VISIBLE_FRACTION
        for face in faces
    )
    np.testing.assert_allclose(
        [face.projected_area for face in faces],
        (2.0, 1.0, 1.0),
    )


def test_same_camera_faces_receive_exact_projected_area_ratios() -> None:
    faces = tuple(
        _projected_face(
            face_index,
            4,
            projected_area=projected_area,
        )
        for face_index, projected_area in enumerate((2.0, 1.0, 1.0))
    )

    placements = _build_face_placements(
        faces,
        (0, 0, 0, 0, 100, 0),
        _PixelRectangle(0, 0, 120, 100),
    )

    allocated_areas = np.zeros(3, dtype=float)
    for placement in placements:
        points = placement.destination_points
        first_edge = points[1] - points[0]
        second_edge = points[2] - points[0]
        allocated_areas[placement.face.local_face_index] += abs(
            first_edge[0] * second_edge[1]
            - first_edge[1] * second_edge[0]
        ) * 0.5
    np.testing.assert_allclose(
        allocated_areas / allocated_areas[2],
        (2.0, 1.0, 1.0),
        atol=0.15,
    )
    assert len(placements) == 6


def test_face_below_half_visibility_uses_minimum_fallback_pool() -> None:
    occluder_scale = float(np.sqrt(0.6))
    geometry = _scene_geometry_from_triangles(
        np.asarray(
            (
                ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
                (
                    (0.0, 0.0, 1.0),
                    (2.0 * occluder_scale, 0.0, 1.0),
                    (0.0, 2.0 * occluder_scale, 1.0),
                ),
            ),
            dtype=float,
        )
    )
    percentages = (0, 0, 0, 0, 100, 0)

    faces = _assign_faces_to_cameras((geometry,), percentages, None)
    placements = _build_face_placements(
        faces,
        percentages,
        _PixelRectangle(0, 0, 128, 128),
    )

    assert faces[0].camera_index == _FALLBACK_CAMERA_INDEX
    assert faces[0].visible_fraction < SCAN_PROJECTION_MINIMUM_VISIBLE_FRACTION
    assert faces[0].visible_fraction == pytest.approx(0.4, abs=0.03)
    assert faces[1].camera_index == 4
    fallback_rows = {
        int(point[1])
        for placement in placements
        if placement.face.camera_index == _FALLBACK_CAMERA_INDEX
        for point in placement.destination_points
    }
    assert max(fallback_rows) - min(fallback_rows) >= 3


def test_visibility_raster_keeps_exact_centers_near_rounded_edges() -> None:
    screen_points = np.asarray(
        (
            (343.15534591, 34.92550811),
            (344.58902511, 32.40573958),
            (345.63132859, 35.53510883),
        ),
        dtype=float,
    )

    rows, columns, depths = _rasterize_visibility_face_samples(
        screen_points,
        np.asarray((1.0, 1.0, 1.0), dtype=float),
    )

    assert len(rows) == len(columns) == len(depths) == 4
    np.testing.assert_allclose(depths, 1.0)


def test_camera_visibility_threshold_is_inclusive_at_half() -> None:
    projected_areas = np.asarray((0.0, 0.0, 0.0, 0.0, 1.0, 1.0))
    enabled_indices = np.asarray((4, 5), dtype=np.int64)
    percentages = (0, 0, 0, 0, 90, 10)

    exactly_half = _select_face_camera(
        projected_areas,
        np.asarray((0.0, 0.0, 0.0, 0.0, 0.5, 0.0)),
        percentages,
        enabled_indices,
    )
    below_half = _select_face_camera(
        projected_areas,
        np.asarray((0.0, 0.0, 0.0, 0.0, 0.499, 0.0)),
        percentages,
        enabled_indices,
    )

    assert exactly_half == 4
    assert below_half == _FALLBACK_CAMERA_INDEX


# ### Projection capacity tests ###
def test_scan_rows_handle_extreme_group_weights_in_a_tiny_exact_atlas() -> None:
    groups = tuple(
        _FaceGroup(faces=(), weight=weight)
        for weight in (100.0, 1.0, 1.0, 1.0)
    )

    rectangles = _partition_group_rectangles(
        _PixelRectangle(0, 0, 8, 8),
        groups,
    )

    assert len(rectangles) == 4
    assert all(rectangle.width == 4 for rectangle in rectangles)
    assert all(rectangle.height == 4 for rectangle in rectangles)
    assert {
        (rectangle.x, rectangle.y)
        for rectangle in rectangles
    } == {(0, 0), (4, 0), (0, 4), (4, 4)}


def test_scan_rows_preserve_reduction_alignment_near_capacity() -> None:
    rectangle = _PixelRectangle(0, 0, 44, 40)
    groups = tuple(
        _FaceGroup(faces=(), weight=float(weight))
        for weight in np.geomspace(1.0, 1e-8, 103)
    )

    rectangles = _partition_group_rectangles(rectangle, groups)

    coverage = np.zeros((rectangle.height, rectangle.width), dtype=bool)
    assert len(rectangles) == len(groups)
    for item in rectangles:
        assert item.width >= 4
        assert item.height >= 4
        assert not any(
            value % 4 for value in (item.x, item.y, item.width, item.height)
        )
        assert not np.any(
            coverage[item.y : item.bottom, item.x : item.right]
        )
        coverage[item.y : item.bottom, item.x : item.right] = True
    assert np.all(coverage)


def test_winding_independent_tie_uses_percentage_and_preserves_capacity() -> None:
    result = scan_project_textured_glb(
        _build_low_percentage_many_faces_glb(),
        (1, 99, 0, 0, 0, 0),
        texture_resolution=64,
    )

    _scene, output_mesh, _texture = _load_single_mesh_and_texture(
        result.glb_bytes
    )
    assert result.stats.face_count == 201
    assert result.stats.output_face_count == 402
    assert result.stats.view_face_counts[:2] == (1, 200)
    assert result.stats.fallback_face_count == 0
    assert result.stats.view_pixel_counts[0] >= 2
    assert result.stats.covered_pixel_count == result.stats.usable_pixel_count
    assert result.stats.triangle_occupancy == pytest.approx(1.0)
    assert len(output_mesh.faces) == 402


def test_camera_groups_share_rows_when_full_width_bands_do_not_fit() -> None:
    faces = tuple(
        _projected_face(face_index, face_index)
        for face_index in range(4)
    )

    placements = _build_face_placements(
        faces,
        (25, 25, 25, 25, 0, 0),
        _PixelRectangle(0, 0, 8, 8),
    )

    assert len(placements) == 8
    assert {
        (placement.face.geometry_name, placement.face.local_face_index)
        for placement in placements
    } == {("geometry", face_index) for face_index in range(4)}


def test_scan_layout_reports_only_true_global_group_overcapacity() -> None:
    faces = tuple(
        _projected_face(face_index, 0)
        for face_index in range(2)
    )

    with pytest.raises(
        ValueError,
        match=r"2 groups require at least 32 pixels.*contains 16",
    ):
        _build_face_placements(
            faces,
            (100, 0, 0, 0, 0, 0),
            _PixelRectangle(0, 0, 4, 4),
        )


def test_atlas_independent_glass_does_not_reduce_opaque_camera_share() -> None:
    camera_sparse = _build_effective_group_percentages(
        DEFAULT_PROJECTION_CAMERA_PERCENTAGES,
        (1, 0, 0, 0, 0, 0, 0),
    )
    fallback_only = _build_effective_group_percentages(
        DEFAULT_PROJECTION_CAMERA_PERCENTAGES,
        (0, 0, 0, 0, 0, 0, 1),
    )

    assert camera_sparse[0] == 100.0
    assert sum(camera_sparse) == 100.0
    assert fallback_only[-1] == 100.0
    assert sum(fallback_only) == 100.0


@pytest.mark.parametrize("invalid_index", (True, 1.5, "1"))
def test_glass_face_indices_require_exact_integers(invalid_index: object) -> None:
    with pytest.raises(ValueError, match="must be integers"):
        _normalize_glass_face_indices((invalid_index,))  # type: ignore[arg-type]


def test_glass_rectangle_fit_ignores_zero_area_selected_faces() -> None:
    triangles = np.asarray(
        (
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0)),
            ((0.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            ((100.0, 0.0, 0.0), (110.0, 0.0, 0.0), (120.0, 0.0, 0.0)),
        ),
        dtype=float,
    )

    rectangle = _fit_selected_faces_rectangle(triangles)

    np.testing.assert_allclose(np.min(rectangle, axis=0), (0.0, 0.0, 0.0))
    np.testing.assert_allclose(np.max(rectangle, axis=0), (2.0, 1.0, 0.0))


def test_glass_material_changes_only_the_selected_material_leaf() -> None:
    selected_material = PBRMaterial(
        name="selected",
        metallicFactor=0.2,
        roughnessFactor=0.3,
    )
    retained_material = PBRMaterial(
        name="retained",
        metallicFactor=0.25,
        roughnessFactor=0.75,
    )
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        material=MultiMaterial((selected_material, retained_material)),
        face_materials=np.asarray((0,), dtype=np.int64),
    )
    selected_face = replace(
        _projected_face(0, 0),
        face_material_index=0,
        is_glass=True,
    )
    retained_face = replace(
        _projected_face(1, 0),
        face_material_index=1,
    )
    placements = tuple(
        _FacePlacement(
            face=face,
            fragment_index=0,
            source_positions=face.source_positions,
            source_normals=face.source_normals,
            source_uvs=face.source_uvs,
            destination_points=np.asarray(
                ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
                dtype=float,
            ),
        )
        for face in (selected_face, retained_face)
    )

    material, glass_indices = _build_geometry_glass_material(
        mesh,
        placements,
    )

    selected_output, retained_output, glass_output = material.materials
    assert glass_indices == {(0, False): 2}
    assert selected_output.name == "selected"
    assert selected_output.metallicFactor == pytest.approx(0.2)
    assert selected_output.roughnessFactor == pytest.approx(0.3)
    assert retained_output.name == "retained"
    assert retained_output.metallicFactor == pytest.approx(0.25)
    assert retained_output.roughnessFactor == pytest.approx(0.75)
    assert glass_output.name == "HouseMaker Glass"
    assert glass_output.alphaMode == "BLEND"
    assert glass_output.metallicFactor == pytest.approx(1.0)
    assert glass_output.roughnessFactor == pytest.approx(10.0 / 255.0)
    assert glass_output.doubleSided is False
    assert glass_output.baseColorTexture is None
    np.testing.assert_array_equal(
        glass_output.baseColorFactor,
        HOUSEMAKER_GLASS_BASE_COLOR_FACTOR,
    )


# ### Failure and cancellation tests ###
def test_rejects_untextured_wrong_resolution_and_distinct_atlases() -> None:
    untextured = trimesh.Scene(trimesh.creation.box()).export(file_type="glb")
    with pytest.raises(ValueError, match="textured"):
        scan_project_textured_glb(
            bytes(untextured),
            texture_resolution=TEST_TEXTURE_RESOLUTION,
        )

    wrong_texture = _gradient_rgba(TEST_TEXTURE_RESOLUTION // 2)
    with pytest.raises(ValueError, match="must match"):
        scan_project_textured_glb(
            _build_textured_cube_glb(texture=wrong_texture),
            texture_resolution=TEST_TEXTURE_RESOLUTION,
        )

    first_scene = _load_glb_scene(
        _build_textured_cube_glb(texture=_gradient_rgba())
    )
    _other_scene, other_mesh, _texture = _load_single_mesh_and_texture(
        _build_textured_cube_glb(texture=np.flipud(_gradient_rgba()).copy())
    )
    first_scene.add_geometry(
        other_mesh,
        geom_name="second-geometry",
        node_name="second-node",
        transform=trimesh.transformations.translation_matrix((4.0, 0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="more than one distinct"):
        scan_project_textured_glb(
            bytes(first_scene.export(file_type="glb")),
            texture_resolution=TEST_TEXTURE_RESOLUTION,
        )


def test_rebakes_pbr_material_maps_with_the_same_scan_layout() -> None:
    scene, mesh, _texture = _load_single_mesh_and_texture(
        _build_textured_cube_glb()
    )
    mesh.visual.material.normalTexture = Image.fromarray(
        _gradient_rgba(),
        mode="RGBA",
    )
    metallic_roughness = np.empty(
        (TEST_TEXTURE_RESOLUTION, TEST_TEXTURE_RESOLUTION, 4),
        dtype=np.uint8,
    )
    metallic_roughness[:, :, :] = (0, 72, 190, 255)
    mesh.visual.material.metallicRoughnessTexture = Image.fromarray(
        metallic_roughness,
        mode="RGBA",
    )

    result = scan_project_textured_glb(
        bytes(scene.export(file_type="glb")),
        texture_resolution=TEST_TEXTURE_RESOLUTION,
    )

    output_scene = _load_glb_scene(result.glb_bytes)
    output_maps = _collect_material_texture_maps(output_scene)
    assert set(output_maps) == {
        "base_color",
        MATERIAL_TEXTURE_NORMAL,
        MATERIAL_TEXTURE_METALLIC_ROUGHNESS,
    }
    assert all(
        texture.shape
        == (TEST_TEXTURE_RESOLUTION, TEST_TEXTURE_RESOLUTION, 4)
        for textures in output_maps.values()
        for texture in textures
    )


def test_strips_unrequested_uv_maps_without_rejecting_pbr_generation() -> None:
    scene, mesh, _texture = _load_single_mesh_and_texture(
        _build_textured_cube_glb()
    )
    normal = np.full(
        (TEST_TEXTURE_RESOLUTION, TEST_TEXTURE_RESOLUTION, 4),
        (128, 128, 255, 255),
        dtype=np.uint8,
    )
    metallic_roughness = np.full(
        (TEST_TEXTURE_RESOLUTION, TEST_TEXTURE_RESOLUTION, 4),
        (31, 72, 190, 255),
        dtype=np.uint8,
    )
    mesh.visual.material.normalTexture = Image.fromarray(normal, mode="RGBA")
    mesh.visual.material.metallicRoughnessTexture = Image.fromarray(
        metallic_roughness,
        mode="RGBA",
    )
    mesh.visual.material.occlusionTexture = Image.fromarray(
        np.flipud(_gradient_rgba()).copy(),
        mode="RGBA",
    )
    mesh.visual.material.emissiveTexture = Image.fromarray(
        _gradient_rgba(),
        mode="RGBA",
    )
    mesh.visual.material.emissiveFactor = (0.2, 0.4, 0.6)

    result = scan_project_textured_glb(
        bytes(scene.export(file_type="glb")),
        texture_resolution=TEST_TEXTURE_RESOLUTION,
    )

    output_scene = _load_glb_scene(result.glb_bytes)
    output_material = next(
        iter(output_scene.geometry.values())
    ).visual.material
    output_maps = _collect_material_texture_maps(output_scene)
    assert set(output_maps) == {
        "base_color",
        MATERIAL_TEXTURE_NORMAL,
        MATERIAL_TEXTURE_METALLIC_ROUGHNESS,
    }
    assert output_material.occlusionTexture is None
    assert output_material.emissiveTexture is None
    np.testing.assert_allclose(output_material.emissiveFactor, (0.0, 0.0, 0.0))
    packed_output = output_maps[MATERIAL_TEXTURE_METALLIC_ROUGHNESS][0]
    assert np.any(np.all(packed_output == (31, 72, 190, 255), axis=2))


def test_rebaked_packed_metallic_roughness_retains_its_occlusion_alias() -> None:
    scene, mesh, _texture = _load_single_mesh_and_texture(
        _build_textured_cube_glb()
    )
    metallic_roughness = Image.fromarray(
        np.full(
            (TEST_TEXTURE_RESOLUTION, TEST_TEXTURE_RESOLUTION, 4),
            (91, 72, 190, 255),
            dtype=np.uint8,
        ),
        mode="RGBA",
    )
    mesh.visual.material.metallicRoughnessTexture = metallic_roughness
    mesh.visual.material.occlusionTexture = metallic_roughness

    result = scan_project_textured_glb(
        bytes(scene.export(file_type="glb")),
        texture_resolution=TEST_TEXTURE_RESOLUTION,
    )

    output_scene = _load_glb_scene(result.glb_bytes)
    output_material = next(
        iter(output_scene.geometry.values())
    ).visual.material
    assert output_material.occlusionTexture is not None
    assert output_material.metallicRoughnessTexture is not None
    np.testing.assert_array_equal(
        np.asarray(output_material.occlusionTexture.convert("RGBA")),
        np.asarray(output_material.metallicRoughnessTexture.convert("RGBA")),
    )


@pytest.mark.parametrize("double_sided", (False, True))
def test_glass_faces_use_atlas_independent_prefab_and_preserve_side(
    double_sided: bool,
) -> None:
    first = scan_project_textured_glb(
        _build_textured_cube_glb(),
        texture_resolution=TEST_TEXTURE_RESOLUTION,
        glass_face_indices=(0,),
        glass_double_sided=double_sided,
    )

    first_scene = _load_glb_scene(first.glb_bytes)
    first_maps = {
        map_type: textures[0]
        for map_type, textures in _collect_material_texture_maps(
            first_scene
        ).items()
    }
    assert first.stats.glass_face_count == 2
    assert first.stats.glass_pixel_count == 0
    assert first.stats.covered_pixel_count == first.stats.usable_pixel_count
    assert set(first_maps) == {"base_color"}
    glass_meshes = tuple(
        geometry
        for geometry in first_scene.geometry.values()
        if is_housemaker_glass_material(geometry.visual.material)
    )
    opaque_meshes = tuple(
        geometry
        for geometry in first_scene.geometry.values()
        if not is_housemaker_glass_material(geometry.visual.material)
    )
    assert len(glass_meshes) == 1
    assert opaque_meshes
    glass_mesh = glass_meshes[0]
    glass_material = glass_mesh.visual.material
    assert all(
        geometry.visual.material.alphaMode != "BLEND"
        for geometry in opaque_meshes
    )
    assert glass_material.alphaMode == "BLEND"
    assert glass_material.doubleSided is double_sided
    assert glass_material.metallicFactor == pytest.approx(1.0)
    assert glass_material.roughnessFactor == pytest.approx(10.0 / 255.0)
    assert glass_material.baseColorTexture is None
    assert glass_material.normalTexture is None
    assert glass_material.metallicRoughnessTexture is None
    np.testing.assert_array_equal(
        glass_material.baseColorFactor,
        HOUSEMAKER_GLASS_BASE_COLOR_FACTOR,
    )
    assert len(np.unique(glass_mesh.visual.uv, axis=0)) == 1
    assert SCAN_PROJECTION_LAYOUT_METADATA_KEY not in glass_mesh.metadata
    assert all(
        SCAN_PROJECTION_LAYOUT_METADATA_KEY in geometry.metadata
        for geometry in opaque_meshes
    )

    second = scan_project_textured_glb(
        first.glb_bytes,
        texture_resolution=TEST_TEXTURE_RESOLUTION,
    )

    assert second.stats.glass_face_count == 2
    assert second.stats.glass_pixel_count == 0
    second_scene = _load_glb_scene(second.glb_bytes)
    second_glass = next(
        geometry.visual.material
        for geometry in second_scene.geometry.values()
        if is_housemaker_glass_material(geometry.visual.material)
    )
    assert second_glass.doubleSided is double_sided
    assert second_glass.baseColorTexture is None


def test_selected_glass_faces_join_into_one_approximated_rectangle() -> None:
    transform = trimesh.transformations.rotation_matrix(
        0.37,
        (0.0, 0.0, 1.0),
    )
    transform[:3, 3] = (2.5, -1.25, 0.75)
    source_glb = _build_subdivided_glass_panel_glb(transform=transform)
    source_geometry = load_object_face_geometry(source_glb)
    expected_center = np.mean(
        np.unique(
            source_geometry.vertices[
                source_geometry.faces[:4]
            ].reshape((-1, 3)),
            axis=0,
        ),
        axis=0,
    )
    projected = scan_project_textured_glb(
        source_glb,
        texture_resolution=TEST_TEXTURE_RESOLUTION,
        glass_face_indices=(0, 1, 2, 3),
    )

    scene = _load_glb_scene(projected.glb_bytes)
    meshes = [
        geometry
        for geometry in scene.geometry.values()
        if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces)
    ]
    assert len(meshes) == 2
    mesh = next(
        geometry
        for geometry in meshes
        if is_housemaker_glass_material(geometry.visual.material)
    )
    assert len(mesh.faces) == 2
    rectangle_points = np.unique(
        np.round(np.asarray(mesh.vertices)[mesh.faces].reshape((-1, 3)), 6),
        axis=0,
    )
    assert len(rectangle_points) == 4
    centered = rectangle_points - np.mean(rectangle_points, axis=0)
    assert np.linalg.matrix_rank(centered, tol=1e-5) == 2
    output_geometry = load_object_face_geometry(projected.glb_bytes)
    output_center = np.mean(
        np.unique(
            output_geometry.vertices[
                output_geometry.faces[:2]
            ].reshape((-1, 3)),
            axis=0,
        ),
        axis=0,
    )
    np.testing.assert_allclose(output_center, expected_center, atol=2e-4)
    assert projected.stats.glass_face_count == 2


def test_selected_glass_faces_join_across_mesh_parts() -> None:
    projected = scan_project_textured_glb(
        _build_split_geometry_glass_panel_glb(),
        texture_resolution=TEST_TEXTURE_RESOLUTION,
        glass_face_indices=(0, 1),
    )

    geometry = load_object_face_geometry(projected.glb_bytes)
    assert geometry.face_count == 4
    assert projected.stats.glass_face_count == 2
    scene = _load_glb_scene(projected.glb_bytes)
    glass_geometry = next(
        mesh
        for mesh in scene.geometry.values()
        if is_housemaker_glass_material(mesh.visual.material)
    )
    assert len(
        np.unique(
            np.round(
                np.asarray(glass_geometry.vertices)[
                    glass_geometry.faces
                ].reshape((-1, 3)),
                6,
            ),
            axis=0,
        )
    ) == 4


def test_validates_target_and_honors_cancellation() -> None:
    source_glb = _build_textured_cube_glb()
    with pytest.raises(ValueError, match="target domain"):
        scan_project_textured_glb(
            source_glb,
            target_domain="diagonal",
            texture_resolution=TEST_TEXTURE_RESOLUTION,
        )
    with pytest.raises(ValueError, match="even"):
        scan_project_textured_glb(source_glb, texture_resolution=127)
    with pytest.raises(ValueError, match="sidedness"):
        scan_project_textured_glb(
            source_glb,
            texture_resolution=TEST_TEXTURE_RESOLUTION,
            glass_double_sided=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ScanProjectionCancelled):
        scan_project_textured_glb(
            source_glb,
            target_domain=SCAN_PROJECTION_TARGET_FULL,
            texture_resolution=TEST_TEXTURE_RESOLUTION,
            cancellation_check=lambda: True,
        )


def test_rejects_an_all_glass_result_with_a_clear_message() -> None:
    with pytest.raises(ValueError, match="Leave at least one non-glass face"):
        scan_project_textured_glb(
            _build_textured_triangle_glb(),
            texture_resolution=TEST_TEXTURE_RESOLUTION,
            glass_face_indices=(0,),
        )
