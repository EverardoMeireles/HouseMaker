# ### Imports ###
from __future__ import annotations

import io
import unittest

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.object_symmetry import (
    SymmetricDivisionOptions,
    build_symmetric_half_texture_variants,
    build_symmetric_object_variants,
)


# ### Constants ###
_RESOLUTION = 2048
_OPAQUE_BLACK = np.asarray((0, 0, 0, 255), dtype=np.uint8)
_POISON_MAGENTA = np.asarray((255, 0, 255, 255), dtype=np.uint8)


# ### Texture fixture helpers ###
def _gradient_texture() -> np.ndarray:
    columns = np.rint(np.linspace(20.0, 220.0, _RESOLUTION)).astype(
        np.uint8
    )
    rows = np.rint(np.linspace(30.0, 210.0, _RESOLUTION)).astype(np.uint8)
    texture = np.empty((_RESOLUTION, _RESOLUTION, 4), dtype=np.uint8)
    texture[:, :, 0] = columns[np.newaxis, :]
    texture[:, :, 1] = rows[:, np.newaxis]
    texture[:, :, 2] = np.asarray(
        (
            columns.astype(np.uint16)[np.newaxis, :]
            + rows.astype(np.uint16)[:, np.newaxis]
        )
        // 2,
        dtype=np.uint8,
    )
    texture[:, :, 3] = 255
    return texture


def _rigid_fixture_texture(
    retained_rectangle: tuple[int, int, int, int],
) -> np.ndarray:
    texture = np.empty((_RESOLUTION, _RESOLUTION, 4), dtype=np.uint8)
    texture[:] = _POISON_MAGENTA
    gradient = _gradient_texture()
    x0, y0, x1, y1 = retained_rectangle
    texture[y0 : y1 + 1, x0 : x1 + 1] = gradient[
        y0 : y1 + 1,
        x0 : x1 + 1,
    ]
    return texture


def _sample_texture_bilinear(
    texture: np.ndarray,
    uv: np.ndarray,
) -> np.ndarray:
    height, width = texture.shape[:2]
    pixel_x = float(uv[0]) * width - 0.5
    pixel_y = (1.0 - float(uv[1])) * height - 0.5
    x0 = min(max(int(np.floor(pixel_x)), 0), width - 1)
    y0 = min(max(int(np.floor(pixel_y)), 0), height - 1)
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    fraction_x = pixel_x - np.floor(pixel_x)
    fraction_y = pixel_y - np.floor(pixel_y)
    top = (
        texture[y0, x0].astype(float) * (1.0 - fraction_x)
        + texture[y0, x1].astype(float) * fraction_x
    )
    bottom = (
        texture[y1, x0].astype(float) * (1.0 - fraction_x)
        + texture[y1, x1].astype(float) * fraction_x
    )
    return top * (1.0 - fraction_y) + bottom * fraction_y


# ### Mesh fixture helpers ###
def _pixel_center_uv(pixel_x: int, pixel_y: int) -> tuple[float, float]:
    return (
        (float(pixel_x) + 0.5) / _RESOLUTION,
        1.0 - (float(pixel_y) + 0.5) / _RESOLUTION,
    )


def _quad_uvs(rectangle: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rectangle
    return np.asarray(
        (
            _pixel_center_uv(x0, y1),
            _pixel_center_uv(x1, y1),
            _pixel_center_uv(x1, y0),
            _pixel_center_uv(x0, y0),
        ),
        dtype=float,
    )


def _quad_mesh(
    *,
    x_minimum: float,
    x_maximum: float,
    z_offset: float,
    uvs: np.ndarray,
    texture: np.ndarray,
) -> trimesh.Trimesh:
    vertices = np.asarray(
        (
            (x_minimum, 0.0, z_offset),
            (x_maximum, 0.0, z_offset),
            (x_maximum, 0.0, z_offset + 1.0),
            (x_minimum, 0.0, z_offset + 1.0),
        ),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=np.asarray(uvs, dtype=float),
        material=PBRMaterial(
            name="uniform-fallback-material",
            baseColorTexture=Image.fromarray(texture, mode="RGBA"),
        ),
    )
    return mesh


def _scene_glb(
    geometries: tuple[tuple[str, trimesh.Trimesh], ...],
) -> bytes:
    scene = trimesh.Scene()
    for node_name, geometry in geometries:
        scene.add_geometry(
            geometry,
            geom_name=f"{node_name}-geometry",
            node_name=node_name,
        )
    return bytes(scene.export(file_type="glb"))


def _rigid_fit_glb() -> bytes:
    retained_rectangle = (1200, 1300, 1599, 1699)
    texture = _rigid_fixture_texture(retained_rectangle)
    return _scene_glb(
        (
            (
                "retained-rigid",
                _quad_mesh(
                    x_minimum=-2.0,
                    x_maximum=-1.0,
                    z_offset=0.0,
                    uvs=_quad_uvs(retained_rectangle),
                    texture=texture,
                ),
            ),
            (
                "removed-rigid",
                _quad_mesh(
                    x_minimum=1.0,
                    x_maximum=2.0,
                    z_offset=0.0,
                    uvs=_quad_uvs((100, 100, 400, 400)),
                    texture=texture,
                ),
            ),
        )
    )


def _uniform_fallback_glb() -> bytes:
    first_rectangle = (50, 100, 900, 1600)
    second_rectangle = (1100, 350, 1950, 1850)
    poison_rectangle = (960, 120, 1040, 220)
    texture = _gradient_texture()
    x0, y0, x1, y1 = poison_rectangle
    texture[y0 : y1 + 1, x0 : x1 + 1] = _POISON_MAGENTA
    return _scene_glb(
        (
            (
                "retained-first",
                _quad_mesh(
                    x_minimum=-3.0,
                    x_maximum=-2.0,
                    z_offset=0.0,
                    uvs=_quad_uvs(first_rectangle),
                    texture=texture,
                ),
            ),
            (
                "retained-second",
                _quad_mesh(
                    x_minimum=-2.0,
                    x_maximum=-1.0,
                    z_offset=2.0,
                    uvs=_quad_uvs(second_rectangle),
                    texture=texture,
                ),
            ),
            (
                "removed-poison",
                _quad_mesh(
                    x_minimum=1.0,
                    x_maximum=2.0,
                    z_offset=0.0,
                    uvs=_quad_uvs(poison_rectangle),
                    texture=texture,
                ),
            ),
        )
    )


# ### Scene inspection helpers ###
def _load_scene(payload: bytes) -> trimesh.Scene:
    loaded = trimesh.load(
        io.BytesIO(payload),
        file_type="glb",
        force="scene",
        process=False,
    )
    assert isinstance(loaded, trimesh.Scene)
    return loaded


def _geometry_for_node(
    scene: trimesh.Scene,
    node_name: str,
) -> trimesh.Trimesh:
    _transform, geometry_name = scene.graph.get(node_name)
    geometry = scene.geometry[geometry_name]
    assert isinstance(geometry, trimesh.Trimesh)
    return geometry


def _texture_for_geometry(geometry: trimesh.Trimesh) -> np.ndarray:
    texture = geometry.visual.material.baseColorTexture
    return np.asarray(texture.convert("RGBA"), dtype=np.uint8)


def _uv_by_position(mesh: trimesh.Trimesh) -> dict[tuple[float, ...], np.ndarray]:
    return {
        tuple(np.round(vertex, 7)): np.asarray(mesh.visual.uv[index], dtype=float)
        for index, vertex in enumerate(np.asarray(mesh.vertices, dtype=float))
    }


def _assert_color_is_bounded_to_uv_neighborhood(
    test: unittest.TestCase,
    texture: np.ndarray,
    mesh: trimesh.Trimesh,
    color: np.ndarray,
    *,
    maximum_margin: int,
) -> None:
    rows, columns = np.nonzero(np.all(texture == color, axis=2))
    test.assertGreater(len(rows), 0)
    uvs = np.asarray(mesh.visual.uv, dtype=float)
    uv_columns = uvs[:, 0] * _RESOLUTION - 0.5
    uv_rows = (1.0 - uvs[:, 1]) * _RESOLUTION - 0.5
    test.assertGreaterEqual(
        int(np.min(columns)),
        int(np.floor(np.min(uv_columns))) - maximum_margin,
    )
    test.assertLessEqual(
        int(np.max(columns)),
        int(np.ceil(np.max(uv_columns))) + maximum_margin,
    )
    test.assertGreaterEqual(
        int(np.min(rows)),
        int(np.floor(np.min(uv_rows))) - maximum_margin,
    )
    test.assertLessEqual(
        int(np.max(rows)),
        int(np.ceil(np.max(uv_rows))) + maximum_margin,
    )


def _fit_uv_affine(
    source_geometry: trimesh.Trimesh,
    output_geometry: trimesh.Trimesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_by_position = _uv_by_position(source_geometry)
    output_by_position = _uv_by_position(output_geometry)
    if source_by_position.keys() != output_by_position.keys():
        raise AssertionError("The UV transform changed the chart's vertices.")
    positions = sorted(source_by_position)
    source_uvs = np.asarray([source_by_position[key] for key in positions])
    output_uvs = np.asarray([output_by_position[key] for key in positions])
    design = np.column_stack((source_uvs, np.ones(len(source_uvs))))
    coefficients = np.linalg.lstsq(design, output_uvs, rcond=None)[0]
    matrix = coefficients[:2, :].T
    translation = coefficients[2, :]
    fitted = source_uvs @ matrix.T + translation
    return matrix, translation, fitted, output_uvs


def _assert_similarity_transform(
    test_case: unittest.TestCase,
    source_geometry: trimesh.Trimesh,
    output_geometry: trimesh.Trimesh,
    *,
    expected_scale: float | None = None,
) -> float:
    matrix, _translation, fitted, output_uvs = _fit_uv_affine(
        source_geometry,
        output_geometry,
    )
    np.testing.assert_allclose(fitted, output_uvs, atol=2e-7)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    np.testing.assert_allclose(
        singular_values,
        np.repeat(np.mean(singular_values), 2),
        atol=2e-7,
    )
    scale = float(np.mean(singular_values))
    test_case.assertGreater(scale, 0.0)
    test_case.assertLessEqual(scale, 1.0 + 2e-7)
    test_case.assertGreater(float(np.linalg.det(matrix)), 0.0)
    np.testing.assert_allclose(
        matrix.T @ matrix,
        np.eye(2) * scale**2,
        atol=2e-7,
    )
    if expected_scale is not None:
        test_case.assertAlmostEqual(scale, expected_scale, delta=2e-7)
    return scale


def _assert_face_samples_preserved(
    source_geometry: trimesh.Trimesh,
    output_geometry: trimesh.Trimesh,
    source_texture: np.ndarray,
    output_texture: np.ndarray,
    *,
    tolerance: float,
) -> None:
    output_uvs = _uv_by_position(output_geometry)
    source_vertices = np.asarray(source_geometry.vertices, dtype=float)
    source_uvs = np.asarray(source_geometry.visual.uv, dtype=float)
    source_faces = np.asarray(source_geometry.faces, dtype=np.int64)
    samples = (
        (0, np.asarray((0.17, 0.29, 0.54), dtype=float)),
        (0, np.asarray((0.61, 0.13, 0.26), dtype=float)),
        (1, np.asarray((0.23, 0.68, 0.09), dtype=float)),
    )
    for face_index, weights in samples:
        face = source_faces[face_index]
        source_uv = weights @ source_uvs[face]
        output_triangle = np.asarray(
            [
                output_uvs[tuple(np.round(source_vertices[index], 7))]
                for index in face
            ],
            dtype=float,
        )
        output_uv = weights @ output_triangle
        np.testing.assert_allclose(
            _sample_texture_bilinear(output_texture, output_uv),
            _sample_texture_bilinear(source_texture, source_uv),
            atol=tolerance,
        )


def _assert_right_half_is_opaque_black(texture: np.ndarray) -> None:
    expected = np.empty((_RESOLUTION, _RESOLUTION // 2, 4), dtype=np.uint8)
    expected[:] = _OPAQUE_BLACK
    np.testing.assert_array_equal(texture[:, _RESOLUTION // 2 :], expected)


# ### Uniform fallback tests ###
class SymmetricUvUniformFallbackTests(unittest.TestCase):
    def test_rigid_fit_path_remains_unscaled_and_pixel_exact(self) -> None:
        source = _rigid_fit_glb()
        source_scene = _load_scene(source)
        source_geometry = _geometry_for_node(source_scene, "retained-rigid")
        source_texture = _texture_for_geometry(source_geometry)

        result = build_symmetric_object_variants(
            source,
            SymmetricDivisionOptions("vertical", "left"),
        )
        output_scene = _load_scene(result.variants.glb_by_resolution[2048])
        output_geometry = _geometry_for_node(output_scene, "retained-rigid")
        output_texture = result.variants.preview_rgba_by_resolution[2048]

        _assert_similarity_transform(
            self,
            source_geometry,
            output_geometry,
            expected_scale=1.0,
        )
        _assert_face_samples_preserved(
            source_geometry,
            output_geometry,
            source_texture,
            output_texture,
            tolerance=1e-6,
        )
        _assert_color_is_bounded_to_uv_neighborhood(
            self,
            output_texture,
            output_geometry,
            _POISON_MAGENTA,
            maximum_margin=19,
        )
        _assert_right_half_is_opaque_black(output_texture)

    def test_uniform_fallback_is_proper_similarity_and_deterministic(self) -> None:
        source = _uniform_fallback_glb()
        source_scene = _load_scene(source)

        first = build_symmetric_object_variants(
            source,
            SymmetricDivisionOptions("vertical", "left"),
        )
        second = build_symmetric_object_variants(
            source,
            SymmetricDivisionOptions("vertical", "left"),
        )
        output_scene = _load_scene(first.variants.glb_by_resolution[2048])
        output_texture = first.variants.preview_rgba_by_resolution[2048]
        scales: list[float] = []
        for node_name in ("retained-first", "retained-second"):
            source_geometry = _geometry_for_node(source_scene, node_name)
            output_geometry = _geometry_for_node(output_scene, node_name)
            scale = _assert_similarity_transform(
                self,
                source_geometry,
                output_geometry,
            )
            self.assertLess(scale, 1.0 - 1e-4)
            scales.append(scale)
            _assert_face_samples_preserved(
                source_geometry,
                output_geometry,
                _texture_for_geometry(source_geometry),
                output_texture,
                tolerance=2.0,
            )
        self.assertAlmostEqual(scales[0], scales[1], delta=2e-7)
        self.assertNotIn("removed-poison", output_scene.graph.nodes_geometry)
        self.assertEqual(
            np.count_nonzero(
                np.all(output_texture == _POISON_MAGENTA, axis=2)
            ),
            0,
        )
        _assert_right_half_is_opaque_black(output_texture)

        for resolution in (512, 1024, 2048):
            with self.subTest(resolution=resolution):
                np.testing.assert_array_equal(
                    first.variants.preview_rgba_by_resolution[resolution],
                    second.variants.preview_rgba_by_resolution[resolution],
                )
                self.assertEqual(
                    first.variants.texture_png_by_resolution[resolution],
                    second.variants.texture_png_by_resolution[resolution],
                )
                self.assertEqual(
                    first.variants.glb_by_resolution[resolution],
                    second.variants.glb_by_resolution[resolution],
                )

    def test_preserved_left_retexture_does_not_scale_fallback_twice(self) -> None:
        source = _uniform_fallback_glb()
        divided = build_symmetric_object_variants(
            source,
            SymmetricDivisionOptions("vertical", "left"),
        )
        divided_scene = _load_scene(divided.variants.glb_by_resolution[2048])
        divided_texture = divided.variants.preview_rgba_by_resolution[2048]

        preserved = build_symmetric_half_texture_variants(
            divided.variants.glb_by_resolution[2048],
            uvs_already_left_packed=True,
        )
        preserved_scene = _load_scene(preserved.glb_by_resolution[2048])
        preserved_texture = preserved.preview_rgba_by_resolution[2048]
        for node_name in ("retained-first", "retained-second"):
            divided_geometry = _geometry_for_node(divided_scene, node_name)
            preserved_geometry = _geometry_for_node(preserved_scene, node_name)
            divided_uvs = _uv_by_position(divided_geometry)
            preserved_uvs = _uv_by_position(preserved_geometry)
            self.assertEqual(divided_uvs.keys(), preserved_uvs.keys())
            for position in divided_uvs:
                np.testing.assert_allclose(
                    preserved_uvs[position],
                    divided_uvs[position],
                    atol=2e-7,
                )
            _assert_similarity_transform(
                self,
                divided_geometry,
                preserved_geometry,
                expected_scale=1.0,
            )
            _assert_face_samples_preserved(
                divided_geometry,
                preserved_geometry,
                divided_texture,
                preserved_texture,
                tolerance=1e-6,
            )
        _assert_right_half_is_opaque_black(preserved_texture)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
