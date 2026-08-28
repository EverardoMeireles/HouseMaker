# ### Imports ###
from __future__ import annotations

import math
import random
import time
import unittest
from dataclasses import FrozenInstanceError
from io import BytesIO
from unittest import mock

import cv2
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker import object_symmetry
from housemaker.glb import GLTF_Y_UP_TO_Z_UP_TRANSFORM
from housemaker.object_symmetry import (
    AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION,
    LEGACY_SYMMETRIC_PAIR_METADATA_VERSION,
    SYMMETRIC_PAIR_ATLAS_RESOLUTION_BY_CONTENT_RESOLUTION,
    SYMMETRIC_QUARTER_METADATA_VERSION,
    SYMMETRIC_QUARTER_ATLAS_RESOLUTION_BY_CONTENT_RESOLUTION,
    SYMMETRIC_SQUARE_PAIR_ATLAS_RESOLUTION_BY_CONTENT_RESOLUTION,
    SYMMETRIC_SELECTION_MODE_FEWEST_TRIANGLES_RANDOM_TIE,
    SYMMETRIC_TEXTURE_CONTENT_QUADRANT_TOP_LEFT,
    SYMMETRIC_TEXTURE_CONTENT_HALF_LEFT,
    SYMMETRIC_TEXTURE_PACKING_MODE_PAIR,
    SYMMETRIC_TEXTURE_PACKING_MODE_TOP_LEFT_QUARTER,
    SymmetricDivisionMetadata,
    SymmetricDivisionOptions,
    SymmetricDivisionResult,
    SymmetricPairTextureVariants,
    SymmetricQuarterTextureVariants,
    SymmetricSquarePairTextureVariants,
    build_automatic_symmetric_object_variants,
    build_symmetric_half_texture_variants,
    build_symmetric_object_variants,
    build_symmetric_pair_texture_variants,
    build_symmetric_quarter_texture_variants,
    build_symmetric_square_pair_texture_variants,
)


# ### Fixture helpers ###
def _solid_texture(color: tuple[int, int, int, int]) -> np.ndarray:
    return np.full((2048, 2048, 4), color, dtype=np.uint8)


def _textured_mesh(
    mesh: trimesh.Trimesh,
    texture: np.ndarray,
    *,
    uvs: np.ndarray | None = None,
) -> trimesh.Trimesh:
    result = mesh.copy()
    if uvs is None:
        u_values = np.linspace(0.1, 0.35, len(result.vertices))
        v_values = np.linspace(0.15, 0.4, len(result.vertices))
        uvs = np.column_stack((u_values, v_values[::-1]))
    result.visual = TextureVisuals(
        uv=np.asarray(uvs, dtype=float),
        material=PBRMaterial(
            name="shared-material",
            baseColorTexture=Image.fromarray(texture, mode="RGBA"),
        ),
    )
    return result


def _scene_glb(
    meshes: list[tuple[str, trimesh.Trimesh, np.ndarray]],
) -> bytes:
    scene = trimesh.Scene()
    for node_name, mesh, transform in meshes:
        scene.add_geometry(
            mesh,
            geom_name=f"{node_name}-geometry",
            node_name=node_name,
            transform=transform,
        )
    return bytes(scene.export(file_type="glb"))


def _box_glb(*, transform: np.ndarray | None = None) -> bytes:
    mesh = _textured_mesh(
        trimesh.creation.box(extents=(4.0, 6.0, 2.0)),
        _solid_texture((31, 79, 151, 207)),
    )
    return _scene_glb(
        [("box-node", mesh, transform if transform is not None else np.eye(4))]
    )


def _load_scene(payload: bytes) -> trimesh.Scene:
    loaded = trimesh.load(
        BytesIO(payload),
        file_type="glb",
        force="scene",
        process=False,
    )
    assert isinstance(loaded, trimesh.Scene)
    return loaded


def _z_up_world_mesh(payload: bytes) -> trimesh.Trimesh:
    mesh = _load_scene(payload).to_geometry()
    assert isinstance(mesh, trimesh.Trimesh)
    mesh = mesh.copy()
    mesh.apply_transform(GLTF_Y_UP_TO_Z_UP_TRANSFORM)
    return mesh


def _nonuniform_texture() -> np.ndarray:
    block = np.asarray(
        (
            (5, 7, 5, 5),
            (7, 5, 0, 6),
            (7, 4, 6, 0),
            (1, 1, 6, 6),
        ),
        dtype=np.uint8,
    )
    channel = np.tile(block, (512, 512))
    rgba = np.empty((2048, 2048, 4), dtype=np.uint8)
    rgba[:, :, 0] = channel
    rgba[:, :, 1] = 255 - channel
    rgba[:, :, 2] = channel * 31
    rgba[:, :, 3] = channel * 37
    return rgba


def _coordinate_texture() -> np.ndarray:
    rows, columns = np.indices((2048, 2048))
    rgba = np.empty((2048, 2048, 4), dtype=np.uint8)
    rgba[:, :, 0] = columns % 251
    rgba[:, :, 1] = rows % 253
    rgba[:, :, 2] = (columns // 251 + rows // 253) % 255
    rgba[:, :, 3] = 255
    return rgba


def _smooth_texture() -> np.ndarray:
    rows, columns = np.indices((2048, 2048))
    rgba = np.empty((2048, 2048, 4), dtype=np.uint8)
    rgba[:, :, 0] = np.rint(columns * (255.0 / 2047.0)).astype(np.uint8)
    rgba[:, :, 1] = np.rint(rows * (255.0 / 2047.0)).astype(np.uint8)
    rgba[:, :, 2] = 100
    rgba[:, :, 3] = 255
    return rgba


def _coprime_period_texture() -> np.ndarray:
    rows, columns = np.indices((2048, 2048))
    rgba = np.empty((2048, 2048, 4), dtype=np.uint8)
    rgba[:, :, 0] = (columns % 5) * 51
    rgba[:, :, 1] = (rows % 7) * 37
    rgba[:, :, 2] = ((columns + 2 * rows) % 11) * 23
    rgba[:, :, 3] = 255
    return rgba


def _quad_mesh(
    pixel_coordinates: tuple[tuple[int, int], ...],
) -> trimesh.Trimesh:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=float,
    )
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    uvs = np.asarray(
        [
            (
                (column + 0.5) / 2048.0,
                1.0 - (row + 0.5) / 2048.0,
            )
            for column, row in pixel_coordinates
        ],
        dtype=float,
    )
    return _textured_mesh(
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
        _coordinate_texture(),
        uvs=uvs,
    )


def _custom_textured_glb(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    *,
    texture: np.ndarray | None = None,
) -> bytes:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh = _textured_mesh(
        mesh,
        _coordinate_texture() if texture is None else texture,
        uvs=np.asarray(uvs, dtype=float),
    )
    return _scene_glb([("custom-node", mesh, np.eye(4))])


def _automatic_asymmetric_glb() -> bytes:
    vertices = np.asarray(
        (
            (-2.0, -1.0, 0.0),
            (-1.0, -1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (1.0, -1.0, 0.0),
            (2.0, -1.0, 0.0),
            (2.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
        dtype=float,
    )
    faces = np.asarray(
        ((0, 1, 2), (3, 4, 5), (3, 5, 6)),
        dtype=np.int64,
    )
    uvs = np.asarray(
        (
            (0.1, 0.2),
            (0.3, 0.2),
            (0.3, 0.4),
            (0.55, 0.2),
            (0.85, 0.2),
            (0.85, 0.5),
            (0.55, 0.5),
        ),
        dtype=float,
    )
    return _custom_textured_glb(
        vertices,
        faces,
        uvs,
        texture=_coordinate_texture(),
    )


def _first_texture_rgba(payload: bytes) -> np.ndarray:
    geometry = next(iter(_load_scene(payload).geometry.values()))
    texture = geometry.visual.material.baseColorTexture
    return np.asarray(texture.convert("RGBA"), dtype=np.uint8)


def _sample_texture(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    height, width = texture.shape[:2]
    column = int(np.clip(round(float(uv[0]) * width - 0.5), 0, width - 1))
    row = int(
        np.clip(round((1.0 - float(uv[1])) * height - 0.5), 0, height - 1)
    )
    return texture[row, column]


def _sample_texture_bilinear(
    texture: np.ndarray,
    uv: np.ndarray,
) -> np.ndarray:
    height, width = texture.shape[:2]
    column = float(uv[0]) * width - 0.5
    row = (1.0 - float(uv[1])) * height - 0.5
    column_floor = math.floor(column)
    row_floor = math.floor(row)
    column_fraction = column - column_floor
    row_fraction = row - row_floor
    column0 = int(np.clip(column_floor, 0, width - 1))
    column1 = int(np.clip(column_floor + 1, 0, width - 1))
    row0 = int(np.clip(row_floor, 0, height - 1))
    row1 = int(np.clip(row_floor + 1, 0, height - 1))
    top = (
        texture[row0, column0] * (1.0 - column_fraction)
        + texture[row0, column1] * column_fraction
    )
    bottom = (
        texture[row1, column0] * (1.0 - column_fraction)
        + texture[row1, column1] * column_fraction
    )
    return top * (1.0 - row_fraction) + bottom * row_fraction


def _sample_texture_repeat_bilinear(
    texture: np.ndarray,
    uv: np.ndarray,
) -> np.ndarray:
    height, width = texture.shape[:2]
    wrapped = uv - np.floor(uv)
    column = float(wrapped[0]) * width - 0.5
    row = (1.0 - float(wrapped[1])) * height - 0.5
    column_floor = math.floor(column)
    row_floor = math.floor(row)
    column_fraction = column - column_floor
    row_fraction = row - row_floor
    column0 = column_floor % width
    column1 = (column_floor + 1) % width
    row0 = row_floor % height
    row1 = (row_floor + 1) % height
    top = (
        texture[row0, column0] * (1.0 - column_fraction)
        + texture[row0, column1] * column_fraction
    )
    bottom = (
        texture[row1, column0] * (1.0 - column_fraction)
        + texture[row1, column1] * column_fraction
    )
    return top * (1.0 - row_fraction) + bottom * row_fraction


def _uvs_by_position(mesh: trimesh.Trimesh) -> dict[tuple[float, ...], np.ndarray]:
    return {
        tuple(np.round(position, 7)): uv.copy()
        for position, uv in zip(
            np.asarray(mesh.vertices, dtype=float),
            np.asarray(mesh.visual.uv, dtype=float),
            strict=True,
        )
    }


def _grid_glb(grid_size: int = 40) -> bytes:
    coordinates = np.linspace(-2.0, 2.0, grid_size)
    vertices = np.asarray(
        [(x, 0.0, z) for z in coordinates for x in coordinates],
        dtype=float,
    )
    faces: list[tuple[int, int, int]] = []
    for row in range(grid_size - 1):
        for column in range(grid_size - 1):
            first = row * grid_size + column
            second = first + 1
            third = first + grid_size
            fourth = third + 1
            faces.extend(
                ((first, second, fourth), (first, fourth, third))
            )
    uvs = np.asarray(
        [
            (
                0.05 + ((x + 2.0) / 4.0) * 0.4,
                0.05 + ((z + 2.0) / 4.0) * 0.9,
            )
            for z in coordinates
            for x in coordinates
        ],
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh = _textured_mesh(
        mesh,
        _solid_texture((31, 79, 151, 255)),
        uvs=uvs,
    )
    return _scene_glb([("grid-node", mesh, np.eye(4))])


# ### Metadata tests ###
class SymmetricDivisionMetadataTests(unittest.TestCase):
    def test_options_and_metadata_are_validated_and_frozen(self) -> None:
        options = SymmetricDivisionOptions("vertical", "left")
        metadata = SymmetricDivisionMetadata(
            orientation=options.orientation,
            kept_side=options.kept_side,
            plane_coordinate=1.25,
        )

        self.assertEqual(
            metadata.to_pipeline_dict(),
            {
                "version": 1,
                "orientation": "vertical",
                "kept_side": "left",
                "plane_coordinate": 1.25,
            },
        )
        self.assertFalse(metadata.cut_plane_capped)
        with self.assertRaises(FrozenInstanceError):
            options.kept_side = "right"  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "match"):
            SymmetricDivisionOptions("vertical", "top")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            SymmetricDivisionOptions("diagonal", "left")  # type: ignore[arg-type]

    def test_version_two_quarter_metadata_remains_load_compatible(self) -> None:
        variants = build_symmetric_quarter_texture_variants(_box_glb())
        metadata = SymmetricDivisionMetadata(
            orientation="vertical",
            kept_side="left",
            plane_coordinate=0.0,
            version=SYMMETRIC_QUARTER_METADATA_VERSION,
            packing_mode=SYMMETRIC_TEXTURE_PACKING_MODE_TOP_LEFT_QUARTER,
            texture_content_quadrant=(
                SYMMETRIC_TEXTURE_CONTENT_QUADRANT_TOP_LEFT
            ),
            selection_mode=(
                SYMMETRIC_SELECTION_MODE_FEWEST_TRIANGLES_RANDOM_TIE
            ),
            triangle_count_by_side=(("left", 1), ("right", 1)),
            tie_broken_randomly=True,
        )

        result = SymmetricDivisionResult(
            variants=variants,
            orientation="vertical",
            kept_side="left",
            plane_coordinate=0.0,
            metadata=metadata,
        )

        self.assertIs(result.metadata, metadata)
        self.assertEqual(
            metadata.to_pipeline_dict()["packing_mode"],
            "symmetric_quarter",
        )

    def test_version_three_pair_metadata_remains_load_compatible(self) -> None:
        variants = build_symmetric_pair_texture_variants(_box_glb())
        metadata = SymmetricDivisionMetadata(
            orientation="vertical",
            kept_side="left",
            plane_coordinate=0.0,
            version=LEGACY_SYMMETRIC_PAIR_METADATA_VERSION,
            packing_mode=SYMMETRIC_TEXTURE_PACKING_MODE_PAIR,
            texture_content_half=SYMMETRIC_TEXTURE_CONTENT_HALF_LEFT,
            selection_mode=(
                SYMMETRIC_SELECTION_MODE_FEWEST_TRIANGLES_RANDOM_TIE
            ),
            triangle_count_by_side=(("left", 1), ("right", 1)),
            tie_broken_randomly=True,
        )

        result = SymmetricDivisionResult(
            variants=variants,
            orientation="vertical",
            kept_side="left",
            plane_coordinate=0.0,
            metadata=metadata,
        )

        self.assertIs(result.metadata, metadata)
        self.assertEqual(
            variants.preview_rgba_by_resolution[1024].shape,
            (2048, 2048, 4),
        )


# ### Automatic symmetry tests ###
class AutomaticSymmetricDivisionTests(unittest.TestCase):
    def test_fewer_clipped_triangles_choose_the_retained_side(self) -> None:
        result = build_automatic_symmetric_object_variants(
            _automatic_asymmetric_glb(),
            "vertical",
            rng=random.Random(9),
        )
        retained = _z_up_world_mesh(result.variants.glb_by_resolution[1024])

        self.assertIsInstance(
            result.variants,
            SymmetricSquarePairTextureVariants,
        )
        self.assertEqual(result.kept_side, "left")
        self.assertEqual(len(retained.faces), 1)
        np.testing.assert_allclose(retained.bounds[:, 0], (-2.0, -1.0))
        self.assertEqual(
            result.metadata.to_pipeline_dict(),
            {
                "version": AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION,
                "orientation": "vertical",
                "kept_side": "left",
                "plane_coordinate": 0.0,
                "packing_mode": SYMMETRIC_TEXTURE_PACKING_MODE_PAIR,
                "texture_content_half": SYMMETRIC_TEXTURE_CONTENT_HALF_LEFT,
                "selection_mode": (
                    SYMMETRIC_SELECTION_MODE_FEWEST_TRIANGLES_RANDOM_TIE
                ),
                "triangle_count_by_side": {"left": 1, "right": 2},
                "tie_broken_randomly": False,
            },
        )

    def test_exact_triangle_tie_uses_the_seeded_random_source(self) -> None:
        source = _box_glb()
        for seed in (0, 1):
            with self.subTest(seed=seed):
                expected_side = random.Random(seed).choice(("left", "right"))
                result = build_automatic_symmetric_object_variants(
                    source,
                    "vertical",
                    rng=random.Random(seed),
                )
                counts = dict(result.metadata.triangle_count_by_side)

                self.assertEqual(counts["left"], counts["right"])
                self.assertEqual(result.kept_side, expected_side)
                self.assertTrue(result.metadata.tie_broken_randomly)

    def test_square_pair_uses_selected_physical_texture_size(self) -> None:
        source = _automatic_asymmetric_glb()
        result = build_automatic_symmetric_object_variants(
            source,
            "vertical",
            rng=random.Random(3),
        )

        self.assertEqual(
            set(result.variants.glb_by_resolution),
            {512, 1024},
        )
        self.assertEqual(result.variants.selectable_resolutions, (512, 1024))
        opaque_black = np.asarray((0, 0, 0, 255), dtype=np.uint8)
        for content_resolution in (1024, 512):
            with self.subTest(content_resolution=content_resolution):
                atlas_resolution = (
                    SYMMETRIC_SQUARE_PAIR_ATLAS_RESOLUTION_BY_CONTENT_RESOLUTION[
                        content_resolution
                    ]
                )
                preview = result.variants.preview_rgba_by_resolution[
                    content_resolution
                ]
                embedded = _first_texture_rgba(
                    result.variants.glb_by_resolution[content_resolution]
                )
                output_geometry = _load_scene(
                    result.variants.glb_by_resolution[content_resolution]
                ).to_geometry()
                output_uvs = np.asarray(output_geometry.visual.uv, dtype=float)

                self.assertEqual(
                    preview.shape,
                    (atlas_resolution, atlas_resolution, 4),
                )
                np.testing.assert_array_equal(embedded, preview)
                np.testing.assert_array_equal(
                    preview[:, atlas_resolution // 2 :],
                    np.broadcast_to(
                        opaque_black,
                        (atlas_resolution, atlas_resolution // 2, 4),
                    ),
                )
                self.assertGreaterEqual(float(np.min(output_uvs)), 0.0)
                self.assertLessEqual(float(np.max(output_uvs[:, 0])), 0.5)
                self.assertLessEqual(float(np.max(output_uvs[:, 1])), 1.0)

        chained_512 = cv2.resize(
            result.variants.preview_rgba_by_resolution[1024],
            (512, 512),
            interpolation=cv2.INTER_AREA,
        )
        self.assertLessEqual(
            int(
                np.max(
                    np.abs(
                        result.variants.preview_rgba_by_resolution[512].astype(
                            np.int16
                        )
                        - chained_512.astype(np.int16)
                    )
                )
            ),
            1,
        )

    def test_square_pair_reductions_each_use_the_canonical_source(self) -> None:
        with mock.patch.object(
            object_symmetry,
            "_resize_rgba",
            wraps=object_symmetry._resize_rgba,
        ) as resize_rgba:
            build_symmetric_square_pair_texture_variants(_box_glb())

        calls = resize_rgba.call_args_list
        self.assertEqual(
            {call.args[1] for call in calls},
            {512, 1024},
        )
        self.assertTrue(
            all(call.args[0].shape == (2048, 2048, 4) for call in calls)
        )

    def test_square_pair_preserves_a_packed_1024_operation_source(self) -> None:
        fresh = build_symmetric_square_pair_texture_variants(_box_glb())
        packed_source = fresh.glb_by_resolution[1024]
        source_texture = _first_texture_rgba(packed_source)
        source_geometry = _load_scene(packed_source).to_geometry()

        rebuilt = build_symmetric_square_pair_texture_variants(
            packed_source,
            uvs_already_left_packed=True,
        )

        np.testing.assert_array_equal(
            rebuilt.preview_rgba_by_resolution[1024],
            source_texture,
        )
        np.testing.assert_array_equal(
            rebuilt.preview_rgba_by_resolution[512],
            cv2.resize(
                source_texture,
                (512, 512),
                interpolation=cv2.INTER_AREA,
            ),
        )
        rebuilt_geometry = _load_scene(
            rebuilt.glb_by_resolution[1024]
        ).to_geometry()
        np.testing.assert_allclose(
            np.asarray(rebuilt_geometry.visual.uv),
            np.asarray(source_geometry.visual.uv),
            atol=1e-7,
        )
        with self.assertRaisesRegex(ValueError, "verified left-packed"):
            build_symmetric_square_pair_texture_variants(packed_source)

    def test_pair_rigid_pack_retains_normal_1024_texel_density(self) -> None:
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        source_uvs = np.asarray(
            ((0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4)),
            dtype=float,
        )
        source = _custom_textured_glb(
            vertices,
            faces,
            source_uvs,
            texture=_smooth_texture(),
        )
        source_geometry = _load_scene(source).to_geometry()
        source_uv_by_position = _uvs_by_position(source_geometry)
        variants = build_symmetric_pair_texture_variants(source)
        output = _load_scene(variants.glb_by_resolution[512]).to_geometry()
        output_uv_by_position = _uvs_by_position(output)
        positions = sorted(output_uv_by_position)
        source_ordered = np.asarray(
            [source_uv_by_position[position] for position in positions]
        )
        output_ordered = np.asarray(
            [output_uv_by_position[position] for position in positions]
        )
        source_texel_distances = 1024.0 * np.linalg.norm(
            source_ordered[:, np.newaxis] - source_ordered[np.newaxis, :],
            axis=2,
        )
        output_texel_distances = 1024.0 * np.linalg.norm(
            output_ordered[:, np.newaxis] - output_ordered[np.newaxis, :],
            axis=2,
        )

        np.testing.assert_allclose(
            output_texel_distances,
            source_texel_distances,
            atol=1e-5,
        )
        normal_1024 = cv2.resize(
            _first_texture_rgba(source),
            (1024, 1024),
            interpolation=cv2.INTER_AREA,
        )
        pair_512 = variants.preview_rgba_by_resolution[512]
        for face in np.asarray(source_geometry.faces, dtype=np.int64):
            face_positions = np.asarray(source_geometry.vertices)[face]
            source_uv = np.mean(np.asarray(source_geometry.visual.uv)[face], axis=0)
            output_uv = np.mean(
                [
                    output_uv_by_position[tuple(np.round(position, 7))]
                    for position in face_positions
                ],
                axis=0,
            )
            expected = _sample_texture_repeat_bilinear(normal_1024, source_uv)
            actual = _sample_texture_repeat_bilinear(pair_512, output_uv)
            np.testing.assert_allclose(actual, expected, atol=3.0)

    def test_pair_uses_uniform_isotropic_fallback_when_required(self) -> None:
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        source_uvs = np.asarray(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            dtype=float,
        )
        source = _custom_textured_glb(vertices, faces, source_uvs)
        variants = build_symmetric_pair_texture_variants(source)
        output = _load_scene(variants.glb_by_resolution[1024]).to_geometry()
        output_uvs = _uvs_by_position(output)
        bottom_left = output_uvs[(0.0, 0.0, 0.0)]
        bottom_right = output_uvs[(1.0, 0.0, 0.0)]
        top_left = output_uvs[(0.0, 1.0, 0.0)]
        horizontal_scale = float(np.linalg.norm(bottom_right - bottom_left))
        vertical_scale = float(np.linalg.norm(top_left - bottom_left))

        self.assertLess(horizontal_scale, 0.5)
        self.assertLess(vertical_scale, 0.5)
        self.assertAlmostEqual(horizontal_scale, vertical_scale, places=6)
        self.assertLessEqual(
            float(np.max(np.asarray(output.visual.uv)[:, 0])),
            0.5,
        )

    def test_pair_repeat_seam_survives_both_physical_resolutions(self) -> None:
        texture = _solid_texture((40, 90, 130, 255))
        texture[:, :16] = (240, 20, 40, 255)
        texture[:, -16:] = (10, 200, 220, 255)
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        source_uvs = np.asarray(
            ((1.0, 0.2), (1.5, 0.2), (1.5, 0.8), (1.0, 0.8)),
            dtype=float,
        )
        source = _custom_textured_glb(
            vertices,
            faces,
            source_uvs,
            texture=texture,
        )
        provider_texture = _first_texture_rgba(source)
        variants = build_symmetric_pair_texture_variants(source)

        for content_resolution, physical_resolution in ((1024, 2048), (512, 1024)):
            output = _load_scene(
                variants.glb_by_resolution[content_resolution]
            ).to_geometry()
            output_uv_by_position = _uvs_by_position(output)
            resized_provider = cv2.resize(
                provider_texture,
                (physical_resolution, physical_resolution),
                interpolation=cv2.INTER_AREA,
            )
            output_texture = variants.preview_rgba_by_resolution[
                content_resolution
            ]
            for position, source_uv in zip(vertices, source_uvs, strict=True):
                expected = _sample_texture_repeat_bilinear(
                    resized_provider,
                    source_uv,
                )
                actual = _sample_texture_repeat_bilinear(
                    output_texture,
                    output_uv_by_position[tuple(position)],
                )
                np.testing.assert_allclose(actual, expected, atol=3.0)

    def test_pair_preserved_rebuild_does_not_transform_uvs_twice(self) -> None:
        source = _box_glb()
        fresh = build_symmetric_pair_texture_variants(source)
        packed_source = fresh.glb_by_resolution[1024]
        packed_geometry = _load_scene(packed_source).to_geometry()
        packed_uv_by_position = _uvs_by_position(packed_geometry)

        rebuilt = build_symmetric_pair_texture_variants(
            packed_source,
            uvs_already_left_packed=True,
        )
        rebuilt_geometry = _load_scene(
            rebuilt.glb_by_resolution[1024]
        ).to_geometry()

        self.assertEqual(
            _uvs_by_position(rebuilt_geometry).keys(),
            packed_uv_by_position.keys(),
        )
        for position, packed_uv in packed_uv_by_position.items():
            np.testing.assert_allclose(
                _uvs_by_position(rebuilt_geometry)[position],
                packed_uv,
                atol=1e-7,
            )
        np.testing.assert_array_equal(
            rebuilt.preview_rgba_by_resolution[1024],
            fresh.preview_rgba_by_resolution[1024],
        )

    def test_nonclipping_rebuild_preserves_already_quarter_uvs(self) -> None:
        source = _box_glb()
        source_geometry = _load_scene(source).to_geometry()
        source_face_count = len(source_geometry.faces)
        fresh = build_symmetric_quarter_texture_variants(source)
        packed_source = fresh.glb_by_resolution[1024]
        packed_geometry = _load_scene(packed_source).to_geometry()
        packed_uv_by_position = _uvs_by_position(packed_geometry)

        rebuilt = build_symmetric_quarter_texture_variants(
            packed_source,
            uvs_already_top_left_quarter=True,
        )
        rebuilt_geometry = _load_scene(
            rebuilt.glb_by_resolution[1024]
        ).to_geometry()

        self.assertEqual(len(packed_geometry.faces), source_face_count)
        self.assertEqual(len(rebuilt_geometry.faces), source_face_count)
        for position, packed_uv in packed_uv_by_position.items():
            np.testing.assert_allclose(
                _uvs_by_position(rebuilt_geometry)[position],
                packed_uv,
                atol=1e-7,
            )
        np.testing.assert_array_equal(
            rebuilt.preview_rgba_by_resolution[1024],
            fresh.preview_rgba_by_resolution[1024],
        )
        with self.assertRaisesRegex(ValueError, "top-left"):
            build_symmetric_quarter_texture_variants(
                source,
                uvs_already_top_left_quarter=True,
            )

    def test_quarter_gutters_preserve_boundary_and_repeated_samples(self) -> None:
        texture = _solid_texture((40, 90, 130, 255))
        texture[:, :16] = (240, 20, 40, 255)
        texture[:, -16:] = (10, 200, 220, 255)
        texture[:16, :] = (210, 30, 180, 255)
        texture[-16:, :] = (30, 210, 60, 255)
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        cases = (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            np.asarray(((0.8, 0.8), (1.2, 0.8), (0.8, 1.2))),
        )

        for source_uvs in cases:
            with self.subTest(source_uvs=source_uvs):
                source = _custom_textured_glb(
                    vertices,
                    faces,
                    source_uvs,
                    texture=texture,
                )
                variants = build_symmetric_quarter_texture_variants(source)
                for content_resolution, gutter in ((1024, 8), (512, 4)):
                    output = _load_scene(
                        variants.glb_by_resolution[content_resolution]
                    ).to_geometry()
                    atlas = variants.preview_rgba_by_resolution[
                        content_resolution
                    ]
                    inner = atlas[
                        gutter : content_resolution - gutter,
                        gutter : content_resolution - gutter,
                    ]
                    for position, output_uv in zip(
                        np.asarray(output.vertices, dtype=float),
                        np.asarray(output.visual.uv, dtype=float),
                        strict=True,
                    ):
                        source_uv = (
                            source_uvs[0]
                            + position[0] * (source_uvs[1] - source_uvs[0])
                            + position[1] * (source_uvs[2] - source_uvs[0])
                        )
                        expected = _sample_texture_repeat_bilinear(
                            inner,
                            source_uv,
                        )
                        actual = _sample_texture_repeat_bilinear(
                            atlas,
                            output_uv,
                        )
                        np.testing.assert_allclose(actual, expected, atol=3.0)


# ### Geometry tests ###
class SymmetricDivisionGeometryTests(unittest.TestCase):
    def test_vertical_division_retains_only_requested_world_half(self) -> None:
        source = _box_glb()
        expected_bounds = {
            "left": (-2.0, 0.0),
            "right": (0.0, 2.0),
        }

        for side, expected_x_bounds in expected_bounds.items():
            with self.subTest(side=side):
                result = build_symmetric_object_variants(
                    source,
                    SymmetricDivisionOptions("vertical", side),
                )
                retained = _z_up_world_mesh(
                    result.variants.glb_by_resolution[512]
                )

                self.assertAlmostEqual(result.plane_coordinate, 0.0)
                np.testing.assert_allclose(
                    retained.bounds[:, 0],
                    expected_x_bounds,
                    atol=1e-7,
                )
                self.assertLess(len(retained.faces), 24)
                self.assertLess(len(retained.vertices), len(retained.faces))

    def test_horizontal_division_retains_only_requested_world_half(self) -> None:
        source = _box_glb()
        expected_bounds = {
            "bottom": (-3.0, 0.0),
            "top": (0.0, 3.0),
        }

        for side, expected_z_bounds in expected_bounds.items():
            with self.subTest(side=side):
                result = build_symmetric_object_variants(
                    source,
                    SymmetricDivisionOptions("horizontal", side),
                )
                retained = _z_up_world_mesh(
                    result.variants.glb_by_resolution[512]
                )

                self.assertAlmostEqual(result.plane_coordinate, 0.0)
                np.testing.assert_allclose(
                    retained.bounds[:, 2],
                    expected_z_bounds,
                    atol=1e-7,
                )
                self.assertLess(len(retained.faces), 24)

    def test_world_plane_accounts_for_node_transform_and_is_preserved(
        self,
    ) -> None:
        transform = trimesh.transformations.rotation_matrix(
            np.pi * 0.5,
            (0.0, 0.0, 1.0),
        )
        transform[:3, 3] = (7.5, 11.0, -4.0)
        result = build_symmetric_object_variants(
            _box_glb(transform=transform),
            SymmetricDivisionOptions("vertical", "right"),
        )
        scene = _load_scene(result.variants.glb_by_resolution[1024])
        retained = _z_up_world_mesh(result.variants.glb_by_resolution[1024])

        self.assertAlmostEqual(result.plane_coordinate, 7.5)
        self.assertIn("box-node", scene.graph.nodes_geometry)
        node_transform, _geometry_name = scene.graph.get("box-node")
        np.testing.assert_allclose(node_transform, transform, atol=1e-7)
        np.testing.assert_allclose(
            retained.bounds[:, 0],
            (7.5, 10.5),
            atol=1e-7,
        )

    def test_crossing_triangle_is_clipped_and_uvs_are_interpolated(self) -> None:
        vertices = np.asarray(
            ((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (-2.0, 0.0, 2.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        uvs = np.asarray(((0.1, 0.1), (0.5, 0.1), (0.1, 0.5)))
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=np.asarray(
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            ),
            process=False,
        )
        mesh = _textured_mesh(mesh, _solid_texture((90, 60, 30, 255)), uvs=uvs)
        source = _scene_glb([("triangle-node", mesh, np.eye(4))])

        result = build_symmetric_object_variants(
            source,
            SymmetricDivisionOptions("vertical", "left"),
        )
        output_scene = _load_scene(
            result.variants.glb_by_resolution[2048]
        )
        output_geometry = next(iter(output_scene.geometry.values()))
        retained = _z_up_world_mesh(result.variants.glb_by_resolution[2048])
        retained_uvs = np.asarray(retained.visual.uv, dtype=float)
        boundary_uvs = retained_uvs[np.isclose(retained.vertices[:, 0], 0.0)]

        self.assertEqual(len(retained.faces), 2)
        self.assertTrue(np.all(retained.vertices[:, 0] <= 1e-8))
        self.assertEqual(len(boundary_uvs), 2)
        self.assertAlmostEqual(
            float(np.linalg.norm(boundary_uvs[1] - boundary_uvs[0])),
            0.2,
            places=6,
        )
        exported_normals = output_geometry._cache.cache.get("vertex_normals")
        self.assertIsNotNone(exported_normals)
        normal_lengths = np.linalg.norm(exported_normals, axis=1)
        np.testing.assert_allclose(normal_lengths, 1.0, atol=1e-7)

    def test_indexed_grid_stays_compact_and_preserves_download_savings(
        self,
    ) -> None:
        source = _grid_glb(90)
        source_mesh = _z_up_world_mesh(source)
        started_at = time.perf_counter()
        result = build_symmetric_object_variants(
            source,
            SymmetricDivisionOptions("vertical", "left"),
        )
        elapsed_seconds = time.perf_counter() - started_at
        output_payload = result.variants.glb_by_resolution[2048]
        retained = _z_up_world_mesh(output_payload)

        self.assertLess(len(retained.faces), len(source_mesh.faces) * 0.55)
        self.assertLess(len(retained.vertices), len(source_mesh.vertices) * 0.6)
        self.assertLess(len(retained.vertices), len(retained.faces))
        self.assertLess(len(output_payload), len(source) * 0.8)
        self.assertLess(elapsed_seconds, 8.0)

    def test_vertex_reuse_does_not_merge_uv_seams(self) -> None:
        vertices = np.asarray(
            (
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 1.0),
                (-1.0, 0.0, 0.0),
                (-1.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
        uvs = np.asarray(
            (
                (0.05, 0.05),
                (0.2, 0.05),
                (0.05, 0.2),
                (0.45, 0.05),
                (0.45, 0.2),
                (0.3, 0.2),
            )
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh = _textured_mesh(
            mesh,
            _solid_texture((80, 90, 100, 255)),
            uvs=uvs,
        )
        source = _scene_glb([("seam-node", mesh, np.eye(4))])
        result = build_symmetric_object_variants(
            source,
            SymmetricDivisionOptions("vertical", "left"),
        )
        retained = _z_up_world_mesh(result.variants.glb_by_resolution[2048])
        coincident_indices = np.flatnonzero(
            np.all(
                np.isclose(retained.vertices, (-1.0, 0.0, 0.0)),
                axis=1,
            )
        )
        seam_u_values = np.asarray(retained.visual.uv)[coincident_indices, 0]

        self.assertGreaterEqual(len(coincident_indices), 2)
        self.assertGreater(float(np.ptp(seam_u_values)), 0.05)


# ### Texture tests ###
class SymmetricDivisionTextureTests(unittest.TestCase):
    def test_uvs_and_texture_fit_exactly_in_left_half(self) -> None:
        result = build_symmetric_object_variants(
            _box_glb(),
            SymmetricDivisionOptions("vertical", "left"),
        )

        for resolution, payload in result.variants.glb_by_resolution.items():
            with self.subTest(resolution=resolution):
                mesh = _load_scene(payload).to_geometry()
                uvs = np.asarray(mesh.visual.uv, dtype=float)
                preview = result.variants.preview_rgba_by_resolution[resolution]
                self.assertGreaterEqual(float(np.min(uvs[:, 0])), 0.0)
                self.assertLessEqual(float(np.max(uvs[:, 0])), 0.5)
                expected_black = np.zeros(
                    (resolution, resolution // 2, 4),
                    dtype=np.uint8,
                )
                expected_black[:, :, 3] = 255
                np.testing.assert_array_equal(
                    preview[:, resolution // 2 :],
                    expected_black,
                )

    def test_lower_resolutions_are_direct_area_resizes_of_packed_2048(
        self,
    ) -> None:
        source_texture = _nonuniform_texture()
        mesh = _textured_mesh(trimesh.creation.box(), source_texture)
        source = _scene_glb([("box-node", mesh, np.eye(4))])
        result = build_symmetric_object_variants(
            source,
            SymmetricDivisionOptions("vertical", "left"),
        )
        canonical = result.variants.preview_rgba_by_resolution[2048]
        expected_black = np.zeros((2048, 1024, 4), dtype=np.uint8)
        expected_black[:, :, 3] = 255
        np.testing.assert_array_equal(canonical[:, 1024:], expected_black)
        self.assertGreater(np.count_nonzero(canonical[:, :1024, :3]), 0)
        self.assertGreater(
            np.count_nonzero(
                np.all(canonical[:, :1024] == expected_black, axis=2)
            ),
            0,
        )

        for resolution in (1024, 512):
            with self.subTest(resolution=resolution):
                expected = cv2.resize(
                    canonical,
                    (resolution, resolution),
                    interpolation=cv2.INTER_AREA,
                )
                np.testing.assert_array_equal(
                    result.variants.preview_rgba_by_resolution[resolution],
                    expected,
                )
        chained_512 = cv2.resize(
            result.variants.preview_rgba_by_resolution[1024],
            (512, 512),
            interpolation=cv2.INTER_AREA,
        )
        direct_512 = result.variants.preview_rgba_by_resolution[512]
        self.assertGreater(np.count_nonzero(direct_512 != chained_512), 0)

    def test_fresh_retexture_uvs_and_pixels_are_left_packed_without_clip(
        self,
    ) -> None:
        source_texture = _nonuniform_texture()
        source_mesh = _textured_mesh(trimesh.creation.box(), source_texture)
        transform = trimesh.transformations.rotation_matrix(
            np.pi / 3.0,
            (0.0, 0.0, 1.0),
        )
        transform[:3, 3] = (3.0, 4.0, 5.0)
        source = _scene_glb([("half-node", source_mesh, transform)])
        loaded_source = _load_scene(source)
        source_geometry = next(iter(loaded_source.geometry.values()))
        provider_texture = _first_texture_rgba(source)

        variants = build_symmetric_half_texture_variants(source)
        output_scene = _load_scene(variants.glb_by_resolution[2048])
        output_geometry = next(iter(output_scene.geometry.values()))
        output_transform, _geometry_name = output_scene.graph.get("half-node")
        output_texture = variants.preview_rgba_by_resolution[2048]
        source_uv_by_position = _uvs_by_position(source_geometry)
        output_uv_by_position = _uvs_by_position(output_geometry)

        self.assertEqual(len(output_geometry.faces), len(source_geometry.faces))
        self.assertEqual(
            len(output_geometry.vertices),
            len(source_geometry.vertices),
        )
        np.testing.assert_allclose(
            output_geometry.bounds,
            source_geometry.bounds,
            atol=1e-7,
        )
        np.testing.assert_allclose(output_transform, transform, atol=1e-7)
        self.assertEqual(source_uv_by_position.keys(), output_uv_by_position.keys())
        ordered_positions = sorted(source_uv_by_position)
        source_ordered = np.asarray(
            [source_uv_by_position[position] for position in ordered_positions]
        )
        output_ordered = np.asarray(
            [output_uv_by_position[position] for position in ordered_positions]
        )
        source_distances = np.linalg.norm(
            source_ordered[:, np.newaxis] - source_ordered[np.newaxis, :],
            axis=2,
        )
        output_distances = np.linalg.norm(
            output_ordered[:, np.newaxis] - output_ordered[np.newaxis, :],
            axis=2,
        )
        np.testing.assert_allclose(output_distances, source_distances, atol=1e-7)
        self.assertFalse(np.allclose(output_ordered, source_ordered * (0.5, 1.0)))
        for source_uv, output_uv in zip(
            source_ordered,
            output_ordered,
            strict=True,
        ):
            np.testing.assert_array_equal(
                _sample_texture(output_texture, output_uv),
                _sample_texture(provider_texture, source_uv),
            )

    def test_preserved_left_uvs_and_pixels_are_not_scaled_twice(self) -> None:
        source_texture = _nonuniform_texture()
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        source_uvs = np.asarray(
            (
                (0.1, 0.2),
                (0.4, 0.2),
                (0.4, 0.8),
                (0.1, 0.8),
            ),
            dtype=float,
        )
        source = _custom_textured_glb(
            vertices,
            faces,
            source_uvs,
            texture=source_texture,
        )
        loaded_source = _load_scene(source)
        loaded_geometry = next(iter(loaded_source.geometry.values()))
        provider_uvs = np.asarray(loaded_geometry.visual.uv, dtype=float)
        provider_texture = _first_texture_rgba(source)

        variants = build_symmetric_half_texture_variants(
            source,
            uvs_already_left_packed=True,
        )
        output_geometry = next(
            iter(_load_scene(variants.glb_by_resolution[2048]).geometry.values())
        )
        canonical = variants.preview_rgba_by_resolution[2048]

        np.testing.assert_allclose(
            np.asarray(output_geometry.visual.uv),
            provider_uvs,
            atol=1e-7,
        )
        expected_black = np.zeros((2048, 1024, 4), dtype=np.uint8)
        expected_black[:, :, 3] = 255
        np.testing.assert_array_equal(canonical[:, 1024:], expected_black)
        self.assertGreater(
            np.count_nonzero(
                np.all(canonical[:, :1024] == expected_black, axis=2)
            ),
            0,
        )
        provider_faces = np.asarray(loaded_geometry.faces, dtype=np.int64)
        provider_centroids = np.mean(provider_uvs[provider_faces], axis=1)
        for provider_uv in provider_centroids:
            np.testing.assert_array_equal(
                _sample_texture(canonical, provider_uv),
                _sample_texture(provider_texture, provider_uv),
            )
        expected_512 = cv2.resize(
            canonical,
            (512, 512),
            interpolation=cv2.INTER_AREA,
        )
        np.testing.assert_array_equal(
            variants.preview_rgba_by_resolution[512],
            expected_512,
        )

    def test_preserved_boundary_uvs_use_appearance_safe_repacking(self) -> None:
        texture = _solid_texture((40, 80, 120, 255))
        texture[:, :8] = (240, 20, 40, 255)
        texture[:, -8:] = (10, 200, 220, 255)
        texture[:, 1016:1024] = (240, 220, 20, 255)
        texture[:, 1024:1032] = (20, 220, 240, 255)
        texture[:8, :] = (210, 30, 180, 255)
        texture[-8:, :] = (30, 210, 60, 255)
        texture[504:512, :] = (250, 120, 10, 255)
        texture[512:520, :] = (10, 80, 250, 255)
        texture[1528:1536, :] = (230, 40, 80, 255)
        texture[1536:1544, :] = (40, 230, 170, 255)
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        cases = {
            "minimum-u": np.asarray(
                ((0.0, 0.3), (0.2, 0.3), (0.2, 0.7), (0.0, 0.7))
            ),
            "left-half-edge": np.asarray(
                ((0.3, 0.3), (0.5, 0.3), (0.5, 0.7), (0.3, 0.7))
            ),
            "minimum-v": np.asarray(
                ((0.1, 0.0), (0.4, 0.0), (0.4, 0.2), (0.1, 0.2))
            ),
            "maximum-v": np.asarray(
                ((0.1, 0.8), (0.4, 0.8), (0.4, 1.0), (0.1, 1.0))
            ),
            "uniform-left-half-edge": np.asarray(
                ((0.01, 0.01), (0.5, 0.01), (0.5, 0.99), (0.01, 0.99))
            ),
            "uniform-internal-u-v-edges": np.asarray(
                ((0.01, 0.25), (0.5, 0.25), (0.5, 0.75), (0.01, 0.75))
            ),
        }

        for name, source_uvs in cases.items():
            with self.subTest(boundary=name):
                source = _custom_textured_glb(
                    vertices,
                    faces,
                    source_uvs,
                    texture=texture,
                )
                provider_texture = _first_texture_rgba(source)
                provider_geometry = _load_scene(source).to_geometry()
                provider_uv_by_position = _uvs_by_position(provider_geometry)
                fresh = build_symmetric_half_texture_variants(source)
                compatible = build_symmetric_half_texture_variants(
                    source,
                    uvs_already_left_packed=True,
                )

                self.assertEqual(
                    compatible.glb_by_resolution,
                    fresh.glb_by_resolution,
                )
                self.assertEqual(
                    compatible.texture_png_by_resolution,
                    fresh.texture_png_by_resolution,
                )
                for resolution in (2048, 1024, 512):
                    output_geometry = _load_scene(
                        compatible.glb_by_resolution[resolution]
                    ).to_geometry()
                    output_uv_by_position = _uvs_by_position(output_geometry)
                    resized_provider = cv2.resize(
                        provider_texture,
                        (resolution, resolution),
                        interpolation=cv2.INTER_AREA,
                    )
                    output_texture = (
                        compatible.preview_rgba_by_resolution[resolution]
                    )
                    for position, provider_uv in provider_uv_by_position.items():
                        expected = _sample_texture_repeat_bilinear(
                            resized_provider,
                            provider_uv,
                        )
                        actual = _sample_texture_repeat_bilinear(
                            output_texture,
                            output_uv_by_position[position],
                        )
                        np.testing.assert_allclose(actual, expected, atol=3.0)

    def test_smart_repack_does_not_copy_pixels_beyond_chart_gutter(self) -> None:
        texture = _solid_texture((30, 80, 140, 255))
        poison = np.asarray((250, 1, 249, 255), dtype=np.uint8)
        texture[100:300, 1500:1700] = poison
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        source_uvs = np.asarray(
            ((0.2, 0.4), (0.3, 0.4), (0.3, 0.6), (0.2, 0.6)),
            dtype=float,
        )
        source = _custom_textured_glb(
            vertices,
            faces,
            source_uvs,
            texture=texture,
        )

        variants = build_symmetric_half_texture_variants(
            source,
            uvs_already_left_packed=False,
        )
        canonical = variants.preview_rgba_by_resolution[2048]

        self.assertFalse(np.any(np.all(canonical == poison, axis=2)))

    def test_preserved_uv_mode_repacks_legacy_full_atlas_uvs(self) -> None:
        mesh = trimesh.creation.box()
        full_uvs = np.column_stack(
            (
                np.linspace(0.0, 1.0, len(mesh.vertices)),
                np.linspace(1.0, 0.0, len(mesh.vertices)),
            )
        )
        source = _scene_glb(
            [
                (
                    "full-node",
                    _textured_mesh(
                        mesh,
                        _solid_texture((1, 2, 3, 255)),
                        uvs=full_uvs,
                    ),
                    np.eye(4),
                )
            ]
        )
        variants = build_symmetric_half_texture_variants(
            source,
            uvs_already_left_packed=True,
        )
        output = _load_scene(variants.glb_by_resolution[2048]).to_geometry()
        output_uvs = np.asarray(output.visual.uv, dtype=float)
        expected_black = _solid_texture((0, 0, 0, 255))[:, :1024]

        self.assertGreaterEqual(float(np.min(output_uvs)), 0.0)
        self.assertLessEqual(float(np.max(output_uvs[:, 0])), 0.5)
        self.assertLessEqual(float(np.max(output_uvs[:, 1])), 1.0)
        np.testing.assert_array_equal(
            variants.preview_rgba_by_resolution[2048][:, 1024:],
            expected_black,
        )

    def test_wide_island_rotates_without_uv_or_pixel_scaling(self) -> None:
        source_mesh = _quad_mesh(
            ((100, 400), (1500, 400), (1500, 600), (100, 600))
        )
        source = _scene_glb([("wide-node", source_mesh, np.eye(4))])
        source_geometry = _load_scene(source).to_geometry()
        source_texture = _first_texture_rgba(source)

        variants = build_symmetric_half_texture_variants(source)
        output_geometry = _load_scene(
            variants.glb_by_resolution[2048]
        ).to_geometry()
        output_texture = variants.preview_rgba_by_resolution[2048]
        source_by_position = _uvs_by_position(source_geometry)
        output_by_position = _uvs_by_position(output_geometry)
        positions = sorted(source_by_position)
        source_uvs = np.asarray([source_by_position[value] for value in positions])
        output_uvs = np.asarray([output_by_position[value] for value in positions])
        source_edges = np.diff(source_uvs[[0, 1, 3]], axis=0)
        output_edges = np.diff(output_uvs[[0, 1, 3]], axis=0)

        np.testing.assert_allclose(
            np.linalg.norm(output_edges, axis=1),
            np.linalg.norm(source_edges, axis=1),
            atol=1e-7,
        )
        self.assertAlmostEqual(
            float(np.linalg.det(output_edges)),
            float(np.linalg.det(source_edges)),
            places=7,
        )
        self.assertLessEqual(float(np.max(output_uvs[:, 0])), 0.5)
        self.assertLess(float(np.ptp(output_uvs[:, 0])), 0.2)
        self.assertGreater(float(np.ptp(output_uvs[:, 1])), 0.6)
        for source_uv, output_uv in zip(source_uvs, output_uvs, strict=True):
            np.testing.assert_array_equal(
                _sample_texture(output_texture, output_uv),
                _sample_texture(source_texture, source_uv),
            )
        opaque_black = np.all(output_texture == _solid_texture((0, 0, 0, 255)), axis=2)
        self.assertGreater(np.count_nonzero(opaque_black[:, :1024]), 1_000_000)

    def test_rigid_odd_chart_moves_preserve_lower_resolution_phase(self) -> None:
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (3.0, 1.0, 0.0),
                (2.0, 1.0, 0.0),
            ),
            dtype=float,
        )
        faces = np.asarray(
            ((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)),
            dtype=np.int64,
        )
        pixel_coordinates = (
            (100, 100),
            (300, 100),
            (300, 300),
            (100, 300),
            (400, 100),
            (600, 100),
            (600, 300),
            (400, 300),
        )
        uvs = np.asarray(
            [
                (
                    (column + 0.5) / 2048.0,
                    1.0 - (row + 0.5) / 2048.0,
                )
                for column, row in pixel_coordinates
            ],
            dtype=float,
        )
        source = _custom_textured_glb(
            vertices,
            faces,
            uvs,
            texture=_coprime_period_texture(),
        )
        canonical_texture = _first_texture_rgba(source)
        source_by_position = _uvs_by_position(_load_scene(source).to_geometry())
        variants = build_symmetric_half_texture_variants(source)
        position_groups = (vertices[:4], vertices[4:])

        for resolution in (1024, 512):
            with self.subTest(resolution=resolution):
                output = _load_scene(
                    variants.glb_by_resolution[resolution]
                ).to_geometry()
                output_by_position = _uvs_by_position(output)
                source_texture = cv2.resize(
                    canonical_texture,
                    (resolution, resolution),
                    interpolation=cv2.INTER_AREA,
                )
                for positions in position_groups:
                    source_uv = np.mean(
                        [source_by_position[tuple(position)] for position in positions],
                        axis=0,
                    )
                    output_uv = np.mean(
                        [output_by_position[tuple(position)] for position in positions],
                        axis=0,
                    )
                    np.testing.assert_allclose(
                        _sample_texture_bilinear(
                            variants.preview_rgba_by_resolution[resolution],
                            output_uv,
                        ),
                        _sample_texture_bilinear(source_texture, source_uv),
                        atol=2.0,
                    )

    def test_rotated_rigid_chart_preserves_lower_resolution_phase(self) -> None:
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        pixel_coordinates = (
            (200, 400),
            (1700, 400),
            (1700, 600),
            (200, 600),
        )
        uvs = np.asarray(
            [
                (
                    (column + 0.5) / 2048.0,
                    1.0 - (row + 0.5) / 2048.0,
                )
                for column, row in pixel_coordinates
            ],
            dtype=float,
        )
        source = _custom_textured_glb(
            vertices,
            faces,
            uvs,
            texture=_coprime_period_texture(),
        )
        canonical_texture = _first_texture_rgba(source)
        source_by_position = _uvs_by_position(_load_scene(source).to_geometry())
        variants = build_symmetric_half_texture_variants(source)

        for resolution in (1024, 512):
            with self.subTest(resolution=resolution):
                output = _load_scene(
                    variants.glb_by_resolution[resolution]
                ).to_geometry()
                output_by_position = _uvs_by_position(output)
                source_texture = cv2.resize(
                    canonical_texture,
                    (resolution, resolution),
                    interpolation=cv2.INTER_AREA,
                )
                source_uv = np.mean(list(source_by_position.values()), axis=0)
                output_uv = np.mean(list(output_by_position.values()), axis=0)
                output_edge = (
                    output_by_position[(1.0, 0.0, 0.0)]
                    - output_by_position[(0.0, 0.0, 0.0)]
                )

                self.assertLess(abs(float(output_edge[0])), 1e-6)
                np.testing.assert_allclose(
                    _sample_texture_bilinear(
                        variants.preview_rgba_by_resolution[resolution],
                        output_uv,
                    ),
                    _sample_texture_bilinear(source_texture, source_uv),
                    atol=2.0,
                )

    def test_removed_half_texels_are_not_copied_into_packed_texture(self) -> None:
        texture = _solid_texture((0, 0, 0, 255))
        texture[300:700, 1300:1600] = (20, 220, 30, 255)
        texture[1100:1500, 100:500] = (245, 15, 25, 255)
        kept = _quad_mesh(
            ((1300, 300), (1599, 300), (1599, 699), (1300, 699))
        )
        removed = _quad_mesh(
            ((100, 1100), (499, 1100), (499, 1499), (100, 1499))
        )
        kept.visual.material.baseColorTexture = Image.fromarray(texture, mode="RGBA")
        removed.visual.material.baseColorTexture = Image.fromarray(
            texture,
            mode="RGBA",
        )
        kept_transform = np.eye(4)
        kept_transform[0, 3] = -2.0
        removed_transform = np.eye(4)
        removed_transform[0, 3] = 1.0
        source = _scene_glb(
            [
                ("kept-node", kept, kept_transform),
                ("removed-node", removed, removed_transform),
            ]
        )

        result = build_symmetric_object_variants(
            source,
            SymmetricDivisionOptions("vertical", "left"),
        )
        output = result.variants.preview_rgba_by_resolution[2048]
        output_scene = _load_scene(result.variants.glb_by_resolution[2048])

        self.assertEqual(len(output_scene.geometry), 1)
        self.assertGreater(
            np.count_nonzero(np.all(output == (20, 220, 30, 255), axis=2)),
            0,
        )
        self.assertEqual(
            np.count_nonzero(np.all(output == (245, 15, 25, 255), axis=2)),
            0,
        )

    def test_exact_half_chart_uses_small_uniform_scale_for_gutters(self) -> None:
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        uvs = np.asarray(((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)))
        source_mesh = _textured_mesh(
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
            _nonuniform_texture(),
            uvs=uvs,
        )
        source = _scene_glb([("exact-half-node", source_mesh, np.eye(4))])
        source_geometry = _load_scene(source).to_geometry()
        variants = build_symmetric_half_texture_variants(source)
        output_geometry = _load_scene(
            variants.glb_by_resolution[2048]
        ).to_geometry()
        output_texture = variants.preview_rgba_by_resolution[2048]

        self.assertEqual(
            _uvs_by_position(output_geometry).keys(),
            _uvs_by_position(source_geometry).keys(),
        )
        positions = tuple(sorted(_uvs_by_position(source_geometry)))
        source_by_position = _uvs_by_position(source_geometry)
        output_by_position = _uvs_by_position(output_geometry)
        source_edges = np.asarray(
            (
                source_by_position[(1.0, 0.0, 0.0)]
                - source_by_position[(0.0, 0.0, 0.0)],
                source_by_position[(0.0, 1.0, 0.0)]
                - source_by_position[(0.0, 0.0, 0.0)],
            )
        )
        output_edges = np.asarray(
            (
                output_by_position[(1.0, 0.0, 0.0)]
                - output_by_position[(0.0, 0.0, 0.0)],
                output_by_position[(0.0, 1.0, 0.0)]
                - output_by_position[(0.0, 0.0, 0.0)],
            )
        )
        scales = np.linalg.norm(output_edges, axis=1) / np.linalg.norm(
            source_edges,
            axis=1,
        )

        self.assertEqual(set(positions), set(output_by_position))
        np.testing.assert_allclose(scales[0], scales[1], atol=1e-7)
        self.assertGreater(float(scales[0]), 0.98)
        self.assertLess(float(scales[0]), 1.0)
        expected_black = _solid_texture((0, 0, 0, 255))[:, :1024]
        np.testing.assert_array_equal(output_texture[:, 1024:], expected_black)

    def test_many_charts_share_one_maximized_uniform_scale_quickly(self) -> None:
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        uvs: list[tuple[float, float]] = []
        chart_positions: list[tuple[tuple[float, float, float], ...]] = []
        for chart_index in range(160):
            row, column = divmod(chart_index, 16)
            base_x = float(chart_index * 2)
            positions = (
                (base_x, 0.0, 0.0),
                (base_x + 1.0, 0.0, 0.0),
                (base_x + 1.0, 1.0, 0.0),
                (base_x, 1.0, 0.0),
            )
            chart_positions.append(positions)
            first_vertex = len(vertices)
            vertices.extend(positions)
            faces.extend(
                (
                    (first_vertex, first_vertex + 1, first_vertex + 2),
                    (first_vertex, first_vertex + 2, first_vertex + 3),
                )
            )
            u0 = 0.005 + column * 0.062
            v0 = 0.005 + row * 0.099
            uvs.extend(
                (
                    (u0, v0),
                    (u0 + 0.055, v0),
                    (u0 + 0.055, v0 + 0.09),
                    (u0, v0 + 0.09),
                )
            )
        mesh = _textured_mesh(
            trimesh.Trimesh(
                vertices=np.asarray(vertices, dtype=float),
                faces=np.asarray(faces, dtype=np.int64),
                process=False,
            ),
            _smooth_texture(),
            uvs=np.asarray(uvs, dtype=float),
        )
        source = _scene_glb([("many-charts-node", mesh, np.eye(4))])
        source_geometry = _load_scene(source).to_geometry()
        started_at = time.perf_counter()

        variants = build_symmetric_half_texture_variants(source)
        elapsed_seconds = time.perf_counter() - started_at
        output_geometry = _load_scene(
            variants.glb_by_resolution[2048]
        ).to_geometry()
        source_by_position = _uvs_by_position(source_geometry)
        output_by_position = _uvs_by_position(output_geometry)
        scales: list[float] = []
        area_scales: list[float] = []
        for positions in chart_positions:
            source_chart = np.asarray(
                [source_by_position[position] for position in positions]
            )
            output_chart = np.asarray(
                [output_by_position[position] for position in positions]
            )
            source_edges = source_chart[[1, 3]] - source_chart[0]
            output_edges = output_chart[[1, 3]] - output_chart[0]
            chart_scale = float(
                np.linalg.norm(output_edges[0])
                / np.linalg.norm(source_edges[0])
            )
            scales.append(chart_scale)
            area_scales.append(
                float(
                    np.linalg.det(output_edges)
                    / np.linalg.det(source_edges)
                )
            )

        self.assertLess(scales[0], 1.0)
        np.testing.assert_allclose(scales, scales[0], atol=2e-6)
        np.testing.assert_allclose(
            area_scales,
            np.asarray(scales) ** 2,
            atol=1e-6,
        )
        self.assertLessEqual(
            float(np.max(np.asarray(output_geometry.visual.uv)[:, 0])),
            0.5,
        )
        self.assertLess(elapsed_seconds, 8.0)

    def test_uniform_fallback_resamples_transparency_without_color_fringe(
        self,
    ) -> None:
        texture = _solid_texture((255, 0, 0, 0))
        texture[:, :900] = (0, 240, 40, 255)
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        uvs = np.asarray(
            ((0.05, 0.05), (0.8, 0.05), (0.8, 0.8), (0.05, 0.8)),
            dtype=float,
        )
        mesh = _textured_mesh(
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
            texture,
            uvs=uvs,
        )
        source = _scene_glb([("alpha-node", mesh, np.eye(4))])

        variants = build_symmetric_half_texture_variants(source)
        output = variants.preview_rgba_by_resolution[2048][:, :1024]
        transparent = output[:, :, 3] <= 5
        opaque_green = (
            (output[:, :, 1] >= 235)
            & (output[:, :, 2] >= 35)
            & (output[:, :, 3] >= 250)
        )

        self.assertTrue(np.any(transparent))
        self.assertTrue(np.any(opaque_green))
        self.assertLessEqual(int(np.max(output[:, :, 0])), 1)


# ### Repeated UV normalization tests ###
class SymmetricDivisionRepeatedUvTests(unittest.TestCase):
    def _build_repeated_triangle(
        self,
        uvs: np.ndarray,
    ) -> tuple[object, trimesh.Trimesh]:
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        source = _custom_textured_glb(
            vertices,
            np.asarray(((0, 1, 2),), dtype=np.int64),
            uvs,
        )
        variants = build_symmetric_half_texture_variants(source)
        output = _load_scene(variants.glb_by_resolution[2048]).to_geometry()
        self.assertIsInstance(output, trimesh.Trimesh)
        return variants, output

    def _assert_triangle_appearance(
        self,
        source_uvs: np.ndarray,
        variants: object,
        output: trimesh.Trimesh,
    ) -> None:
        source_texture = _coordinate_texture()
        output_texture = variants.preview_rgba_by_resolution[2048]
        vertices = np.asarray(output.vertices, dtype=float)
        output_uvs = np.asarray(output.visual.uv, dtype=float)
        for face in np.asarray(output.faces, dtype=np.int64):
            position = np.mean(vertices[face], axis=0)
            output_uv = np.mean(output_uvs[face], axis=0)
            source_uv = (
                source_uvs[0]
                + position[0] * (source_uvs[1] - source_uvs[0])
                + position[1] * (source_uvs[2] - source_uvs[0])
            )
            expected = _sample_texture_repeat_bilinear(
                source_texture,
                source_uv,
            )
            actual = _sample_texture_bilinear(output_texture, output_uv)
            np.testing.assert_allclose(actual, expected, atol=2.0)

    def test_positive_and_negative_offset_tiles_wrap_without_subdivision(
        self,
    ) -> None:
        cases = (
            np.asarray(((2.1, 3.1), (2.4, 3.1), (2.1, 3.4))),
            np.asarray(((-0.9, -1.9), (-0.6, -1.9), (-0.9, -1.6))),
        )
        for source_uvs in cases:
            with self.subTest(source_uvs=source_uvs):
                variants, output = self._build_repeated_triangle(source_uvs)

                self.assertEqual(len(output.faces), 1)
                self.assertAlmostEqual(float(output.area), 0.5, places=6)
                self._assert_triangle_appearance(source_uvs, variants, output)

    def test_preserved_compatibility_repacks_negative_and_repeated_uvs(
        self,
    ) -> None:
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        source_uvs = np.asarray(
            ((-0.2, 0.8), (1.2, 0.8), (-0.2, 1.2)),
            dtype=float,
        )
        source = _custom_textured_glb(vertices, faces, source_uvs)

        fresh = build_symmetric_half_texture_variants(source)
        compatible = build_symmetric_half_texture_variants(
            source,
            uvs_already_left_packed=True,
        )
        output = _load_scene(
            compatible.glb_by_resolution[2048]
        ).to_geometry()
        output_uvs = np.asarray(output.visual.uv, dtype=float)

        self.assertEqual(compatible.glb_by_resolution, fresh.glb_by_resolution)
        self.assertEqual(
            compatible.texture_png_by_resolution,
            fresh.texture_png_by_resolution,
        )
        self.assertGreaterEqual(float(np.min(output_uvs)), 0.0)
        self.assertLessEqual(float(np.max(output_uvs[:, 0])), 0.5)
        self.assertLessEqual(float(np.max(output_uvs[:, 1])), 1.0)

    def test_crossing_u_and_both_axes_are_split_without_area_loss(self) -> None:
        cases = (
            np.asarray(((0.8, 0.2), (1.2, 0.2), (0.8, 0.8))),
            np.asarray(((0.8, 0.8), (1.2, 0.8), (0.8, 1.2))),
        )
        for source_uvs in cases:
            with self.subTest(source_uvs=source_uvs):
                first, output = self._build_repeated_triangle(source_uvs)
                second, _second_output = self._build_repeated_triangle(source_uvs)
                output_uvs = np.asarray(output.visual.uv, dtype=float)
                triangles = np.asarray(output.triangles, dtype=float)
                winding = np.cross(
                    triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0],
                )

                self.assertGreater(len(output.faces), 1)
                self.assertAlmostEqual(float(output.area), 0.5, places=6)
                self.assertTrue(np.all(winding[:, 2] > 0.0))
                self.assertGreaterEqual(float(np.min(output_uvs)), 0.0)
                self.assertLessEqual(float(np.max(output_uvs[:, 0])), 0.5)
                self.assertLessEqual(float(np.max(output_uvs[:, 1])), 1.0)
                np.testing.assert_array_equal(
                    first.preview_rgba_by_resolution[2048],
                    second.preview_rgba_by_resolution[2048],
                )
                self._assert_triangle_appearance(source_uvs, first, output)

    def test_two_axis_clip_normalizes_one_shot_barycentric_normals(self) -> None:
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        source_normals = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=float,
        )
        source_uvs = np.asarray(
            ((0.8, 0.8), (1.2, 0.8), (0.8, 1.2)),
            dtype=float,
        )
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(((0, 1, 2),), dtype=np.int64),
            process=False,
        )
        mesh = _textured_mesh(
            mesh,
            _coordinate_texture(),
            uvs=source_uvs,
        )
        mesh.vertex_normals = source_normals
        source = _scene_glb(
            [
                (
                    "normal-node",
                    mesh,
                    np.eye(4),
                )
            ]
        )
        variants = build_symmetric_half_texture_variants(source)
        output_geometry = next(
            iter(_load_scene(variants.glb_by_resolution[2048]).geometry.values())
        )
        output_normals = output_geometry._cache.cache.get("vertex_normals")
        self.assertIsNotNone(output_normals)

        for position, output_normal in zip(
            np.asarray(output_geometry.vertices, dtype=float),
            np.asarray(output_normals, dtype=float),
            strict=True,
        ):
            weights = np.asarray(
                (1.0 - position[0] - position[1], position[0], position[1])
            )
            expected = weights @ source_normals
            expected /= np.linalg.norm(expected)
            np.testing.assert_allclose(output_normal, expected, atol=2e-6)

    def test_exact_upper_seam_has_one_positive_area_owner(self) -> None:
        cases = (
            np.asarray(
                ((-0.2, 0.1), (0.0, 0.1), (-0.2, 0.3)),
                dtype=float,
            ),
            np.asarray(
                ((-0.2, 0.1), (5e-8, 0.1), (-0.2, 0.3)),
                dtype=float,
            ),
            np.asarray(
                ((-1.0 - 5e-8, 0.1), (-0.8, 0.1), (-0.8, 0.3)),
                dtype=float,
            ),
        )
        for source_uvs in cases:
            with self.subTest(source_uvs=source_uvs):
                variants, output = self._build_repeated_triangle(source_uvs)

                self.assertEqual(len(output.faces), 1)
                self.assertAlmostEqual(float(output.area), 0.5, places=6)
                self._assert_triangle_appearance(source_uvs, variants, output)

    def test_full_half_chart_preserves_repeat_boundary_bilinear_sample(
        self,
    ) -> None:
        texture = _solid_texture((0, 255, 0, 255))
        texture[:, 0] = (255, 0, 0, 255)
        texture[:, -1] = (0, 0, 255, 255)
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        source_uvs = np.asarray(
            ((1.0, 0.2), (1.5, 0.2), (1.5, 0.8), (1.0, 0.8)),
            dtype=float,
        )
        source = _custom_textured_glb(
            vertices,
            faces,
            source_uvs,
            texture=texture,
        )
        canonical_texture = _first_texture_rgba(source)
        variants = build_symmetric_half_texture_variants(source)
        for resolution in (2048, 1024, 512):
            with self.subTest(resolution=resolution):
                output = _load_scene(
                    variants.glb_by_resolution[resolution]
                ).to_geometry()
                output_by_position = _uvs_by_position(output)
                output_uv = 0.5 * (
                    output_by_position[(0.0, 0.0, 0.0)]
                    + output_by_position[(0.0, 1.0, 0.0)]
                )
                source_texture = cv2.resize(
                    canonical_texture,
                    (resolution, resolution),
                    interpolation=cv2.INTER_AREA,
                )
                expected = _sample_texture_repeat_bilinear(
                    source_texture,
                    np.asarray((1.0, 0.5)),
                )
                actual = _sample_texture_bilinear(
                    variants.preview_rgba_by_resolution[resolution],
                    output_uv,
                )

                np.testing.assert_allclose(actual, expected, atol=3.0)

    def test_repeat_boundary_neighborhood_does_not_expose_hidden_alpha_color(
        self,
    ) -> None:
        texture = _solid_texture((0, 180, 0, 255))
        texture[:, :8] = (255, 0, 0, 0)
        texture[:, -8:] = (0, 0, 255, 255)
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        for maximum_u in (1.2, 1.5):
            source_uvs = np.asarray(
                (
                    (1.0, 0.2),
                    (maximum_u, 0.2),
                    (maximum_u, 0.8),
                    (1.0, 0.8),
                ),
                dtype=float,
            )
            variants = build_symmetric_half_texture_variants(
                _custom_textured_glb(
                    vertices,
                    faces,
                    source_uvs,
                    texture=texture,
                )
            )

            for resolution in (2048, 1024, 512):
                with self.subTest(maximum_u=maximum_u, resolution=resolution):
                    output = _load_scene(
                        variants.glb_by_resolution[resolution]
                    ).to_geometry()
                    output_by_position = _uvs_by_position(output)
                    output_uv = 0.5 * (
                        output_by_position[(0.0, 0.0, 0.0)]
                        + output_by_position[(0.0, 1.0, 0.0)]
                    )
                    sampled = _sample_texture_bilinear(
                        variants.preview_rgba_by_resolution[resolution],
                        output_uv,
                    )

                    self.assertLess(float(sampled[0]), 5.0)
                    self.assertLess(float(sampled[1]), 5.0)
                    self.assertGreater(float(sampled[2]), 150.0)
                    self.assertAlmostEqual(
                        float(sampled[3]),
                        127.5,
                        delta=4.0,
                    )

    def test_out_of_range_point_and_line_uvs_reach_collapsed_repair(self) -> None:
        cases = (
            np.asarray(((2.25, -1.4),) * 3, dtype=float),
            np.asarray(
                ((-0.2, 0.5), (0.2, 0.5), (1.2, 0.5)),
                dtype=float,
            ),
        )
        for source_uvs in cases:
            with self.subTest(source_uvs=source_uvs):
                _variants, output = self._build_repeated_triangle(source_uvs)
                output_uvs = np.asarray(output.visual.uv, dtype=float)

                self.assertAlmostEqual(float(output.area), 0.5, places=6)
                self.assertTrue(np.all(np.isfinite(output_uvs)))
                self.assertGreaterEqual(float(np.min(output_uvs)), 0.0)
                self.assertLessEqual(float(np.max(output_uvs[:, 0])), 0.5)
                self.assertLessEqual(float(np.max(output_uvs[:, 1])), 1.0)

    def test_mixed_in_range_and_repeated_faces_remain_present(self) -> None:
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
        uvs = np.asarray(
            (
                (0.1, 0.1),
                (0.4, 0.1),
                (0.1, 0.4),
                (2.1, -1.9),
                (2.4, -1.9),
                (2.1, -1.6),
            ),
            dtype=float,
        )
        variants = build_symmetric_half_texture_variants(
            _custom_textured_glb(vertices, faces, uvs)
        )
        output = _load_scene(variants.glb_by_resolution[2048]).to_geometry()
        output_uvs = np.asarray(output.visual.uv, dtype=float)

        self.assertEqual(len(output.faces), 2)
        self.assertAlmostEqual(float(output.area), 1.0, places=6)
        np.testing.assert_allclose(
            np.asarray(output.bounds)[:, 0],
            (0.0, 3.0),
            atol=1e-7,
        )
        self.assertGreaterEqual(float(np.min(output_uvs)), 0.0)
        self.assertLessEqual(float(np.max(output_uvs[:, 0])), 0.5)
        self.assertLessEqual(float(np.max(output_uvs[:, 1])), 1.0)

    def test_full_division_preserves_rotated_node_with_repeated_uvs(self) -> None:
        transform = trimesh.transformations.rotation_matrix(
            np.pi * 0.25,
            (0.0, 0.0, 1.0),
        )
        transform[:3, 3] = (4.0, -3.0, 2.0)
        mesh = trimesh.creation.box(extents=(4.0, 2.0, 3.0))
        uvs = np.column_stack(
            (
                np.linspace(2.1, 2.4, len(mesh.vertices)),
                np.linspace(-1.9, -1.6, len(mesh.vertices)),
            )
        )
        source = _scene_glb(
            [
                (
                    "repeated-node",
                    _textured_mesh(mesh, _coordinate_texture(), uvs=uvs),
                    transform,
                )
            ]
        )

        result = build_symmetric_object_variants(
            source,
            SymmetricDivisionOptions("vertical", "left"),
        )
        output_scene = _load_scene(result.variants.glb_by_resolution[2048])
        output_transform, _geometry_name = output_scene.graph.get(
            "repeated-node"
        )
        output_uvs = np.asarray(output_scene.to_geometry().visual.uv)

        np.testing.assert_allclose(output_transform, transform, atol=1e-7)
        self.assertGreaterEqual(float(np.min(output_uvs)), 0.0)
        self.assertLessEqual(float(np.max(output_uvs[:, 0])), 0.5)
        self.assertLessEqual(float(np.max(output_uvs[:, 1])), 1.0)

    def test_repeat_tile_work_and_output_growth_are_bounded(self) -> None:
        crossing_uvs = np.asarray(
            ((0.2, 0.2), (1.2, 0.2), (0.2, 1.2)),
            dtype=float,
        )
        with mock.patch(
            "housemaker.object_symmetry._MAX_REPEAT_TILE_SPAN",
            1,
        ):
            with self.assertRaisesRegex(ValueError, "bounded tile processing"):
                self._build_repeated_triangle(crossing_uvs)
        with mock.patch(
            "housemaker.object_symmetry._MAX_REPEAT_OUTPUT_FACES",
            1,
        ):
            with self.assertRaisesRegex(ValueError, "too many retained faces"):
                self._build_repeated_triangle(crossing_uvs)
        with mock.patch(
            "housemaker.object_symmetry._MAX_REPEAT_CLIP_WORK",
            3,
        ):
            with self.assertRaisesRegex(ValueError, "bounded tile processing"):
                self._build_repeated_triangle(crossing_uvs)
        single_tile_uvs = np.asarray(
            ((2.1, 3.1), (2.4, 3.1), (2.1, 3.4)),
            dtype=float,
        )
        with mock.patch(
            "housemaker.object_symmetry._MAX_REPEAT_CLIP_WORK",
            0,
        ):
            _variants, output = self._build_repeated_triangle(single_tile_uvs)
            self.assertEqual(len(output.faces), 1)
        magnitude_uvs = np.asarray(
            (
                (1_000_001.1, 0.1),
                (1_000_001.4, 0.1),
                (1_000_001.1, 0.4),
            ),
            dtype=float,
        )
        with self.assertRaisesRegex(ValueError, "bounded tile processing"):
            self._build_repeated_triangle(magnitude_uvs)


# ### Degenerate UV repair tests ###
class SymmetricDivisionDegenerateUvRepairTests(unittest.TestCase):
    def test_point_chart_becomes_nonzero_and_keeps_constant_appearance(
        self,
    ) -> None:
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        original_uv = np.asarray((0.25, 0.4), dtype=float)
        uvs = np.repeat(original_uv[np.newaxis, :], 3, axis=0)
        texture = _smooth_texture()
        source = _custom_textured_glb(
            vertices,
            faces,
            uvs,
            texture=texture,
        )

        first = build_symmetric_half_texture_variants(source)
        second = build_symmetric_half_texture_variants(source)
        output_scene = _load_scene(first.glb_by_resolution[2048])
        output_mesh = next(iter(output_scene.geometry.values()))
        output_uvs = np.asarray(output_mesh.visual.uv, dtype=float)
        output_face_uvs = output_uvs[np.asarray(output_mesh.faces)[0]]
        output_texture = first.preview_rgba_by_resolution[2048]
        expected = _sample_texture_bilinear(texture, original_uv)

        self.assertGreater(
            abs(float(np.linalg.det(
                np.vstack(
                    (
                        output_face_uvs[1] - output_face_uvs[0],
                        output_face_uvs[2] - output_face_uvs[0],
                    )
                )
            ))),
            0.0,
        )
        for output_uv in output_face_uvs:
            np.testing.assert_allclose(
                _sample_texture_bilinear(output_texture, output_uv),
                expected,
                atol=2.0,
            )
        self.assertEqual(
            first.texture_png_by_resolution,
            second.texture_png_by_resolution,
        )

    def test_line_chart_preserves_its_one_dimensional_gradient(self) -> None:
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.5, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        uvs = np.asarray(
            ((0.1, 0.4), (0.4, 0.4), (0.25, 0.4)),
            dtype=float,
        )
        texture = _smooth_texture()
        variants = build_symmetric_half_texture_variants(
            _custom_textured_glb(
                vertices,
                faces,
                uvs,
                texture=texture,
            )
        )
        output_mesh = next(
            iter(_load_scene(variants.glb_by_resolution[2048]).geometry.values())
        )
        output_by_position = _uvs_by_position(output_mesh)
        output_texture = variants.preview_rgba_by_resolution[2048]

        for position, original_uv in zip(vertices, uvs, strict=True):
            output_uv = output_by_position[tuple(np.round(position, 7))]
            np.testing.assert_allclose(
                _sample_texture_bilinear(output_texture, output_uv),
                _sample_texture_bilinear(texture, original_uv),
                atol=3.0,
            )

    def test_subpixel_border_chart_keeps_its_original_uv_shape(self) -> None:
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        uvs = np.asarray(
            ((0.0, 0.0), (1e-7, 0.0), (0.0, 1e-7)),
            dtype=float,
        )

        variants = build_symmetric_half_texture_variants(
            _custom_textured_glb(vertices, faces, uvs)
        )
        output_mesh = next(
            iter(_load_scene(variants.glb_by_resolution[2048]).geometry.values())
        )
        output_uvs = np.asarray(output_mesh.visual.uv, dtype=float)
        output_face_uvs = output_uvs[np.asarray(output_mesh.faces)[0]]
        source_edges = uvs[1:] - uvs[0]
        output_edges = output_face_uvs[1:] - output_face_uvs[0]

        np.testing.assert_allclose(output_edges, source_edges, atol=3e-8)
        self.assertGreater(abs(float(np.linalg.det(output_edges))), 0.0)

    def test_long_subpixel_thin_chart_retains_span_and_gradient(self) -> None:
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        uvs = np.asarray(
            ((0.1, 0.25), (0.9, 0.25), (0.9, 0.25000012)),
            dtype=float,
        )
        texture = _smooth_texture()

        variants = build_symmetric_half_texture_variants(
            _custom_textured_glb(
                vertices,
                faces,
                uvs,
                texture=texture,
            )
        )
        output_mesh = next(
            iter(_load_scene(variants.glb_by_resolution[2048]).geometry.values())
        )
        output_by_position = _uvs_by_position(output_mesh)
        output_uvs = np.asarray(
            [
                output_by_position[tuple(np.round(position, 7))]
                for position in vertices
            ]
        )
        output_texture = variants.preview_rgba_by_resolution[2048]

        source_distances = np.linalg.norm(uvs[1:] - uvs[0], axis=1)
        output_distances = np.linalg.norm(
            output_uvs[1:] - output_uvs[0],
            axis=1,
        )
        np.testing.assert_allclose(
            output_distances,
            source_distances,
            atol=2e-7,
        )
        for output_uv, original_uv in zip(output_uvs, uvs, strict=True):
            np.testing.assert_allclose(
                _sample_texture_bilinear(output_texture, output_uv),
                _sample_texture_bilinear(texture, original_uv),
                atol=3.0,
            )

    def test_long_taper_keeps_vertex_bounds_beyond_raster_coverage(
        self,
    ) -> None:
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        uvs = np.asarray(
            (
                (0.848756015, 0.171862006),
                (0.850714028, 0.000185012817),
                (0.850714028, 0.173541009),
            ),
            dtype=float,
        )
        texture = _smooth_texture()

        variants = build_symmetric_pair_texture_variants(
            _custom_textured_glb(
                vertices,
                faces,
                uvs,
                texture=texture,
            )
        )
        output_mesh = next(
            iter(_load_scene(variants.glb_by_resolution[1024]).geometry.values())
        )
        output_by_position = _uvs_by_position(output_mesh)
        output_uvs = np.asarray(
            [
                output_by_position[tuple(np.round(position, 7))]
                for position in vertices
            ]
        )
        output_texture = variants.preview_rgba_by_resolution[1024]

        self.assertGreaterEqual(float(np.min(output_uvs)), 0.0)
        self.assertLessEqual(float(np.max(output_uvs[:, 0])), 0.5)
        self.assertLessEqual(float(np.max(output_uvs[:, 1])), 1.0)
        source_distances = np.linalg.norm(
            uvs[:, np.newaxis] - uvs[np.newaxis, :],
            axis=2,
        )
        output_distances = np.linalg.norm(
            output_uvs[:, np.newaxis] - output_uvs[np.newaxis, :],
            axis=2,
        )
        np.testing.assert_allclose(
            output_distances,
            source_distances,
            atol=2e-7,
        )
        for output_uv, original_uv in zip(output_uvs, uvs, strict=True):
            np.testing.assert_allclose(
                _sample_texture_repeat_bilinear(output_texture, output_uv),
                _sample_texture_repeat_bilinear(texture, original_uv),
                atol=3.0,
            )

    def test_border_chart_preserves_default_repeat_seam_sampling(self) -> None:
        texture = _solid_texture((0, 180, 0, 255))
        texture[:, 0] = (255, 0, 0, 255)
        texture[:, -1] = (0, 0, 255, 255)
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        uvs = np.asarray(
            ((1.0, 0.4), (1.0 - 1e-7, 0.4), (1.0, 0.4000001)),
            dtype=float,
        )

        variants = build_symmetric_half_texture_variants(
            _custom_textured_glb(
                vertices,
                faces,
                uvs,
                texture=texture,
            )
        )
        output_mesh = next(
            iter(_load_scene(variants.glb_by_resolution[2048]).geometry.values())
        )
        output_uv = _uvs_by_position(output_mesh)[tuple(vertices[0])]
        sampled = _sample_texture_bilinear(
            variants.preview_rgba_by_resolution[2048],
            output_uv,
        )

        np.testing.assert_allclose(
            sampled,
            np.asarray((128, 0, 128, 255), dtype=float),
            atol=2.0,
        )

    def test_repeat_seam_sampling_survives_uniform_scale_fallback(self) -> None:
        texture = _solid_texture((0, 180, 0, 255))
        texture[:, 0] = (255, 0, 0, 255)
        texture[:, -1] = (0, 0, 255, 255)
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
            ),
            dtype=float,
        )
        faces = np.asarray(
            ((0, 1, 2), (0, 2, 3), (4, 5, 6)),
            dtype=np.int64,
        )
        uvs = np.asarray(
            (
                (0.05, 0.05),
                (0.95, 0.05),
                (0.95, 0.95),
                (0.05, 0.95),
                (1.0, 0.4),
                (1.0 - 1e-7, 0.4),
                (1.0, 0.4000001),
            ),
            dtype=float,
        )

        variants = build_symmetric_half_texture_variants(
            _custom_textured_glb(
                vertices,
                faces,
                uvs,
                texture=texture,
            )
        )
        output_mesh = next(
            iter(_load_scene(variants.glb_by_resolution[2048]).geometry.values())
        )
        output_by_position = _uvs_by_position(output_mesh)
        output_uv = output_by_position[tuple(vertices[4])]
        sampled = _sample_texture_bilinear(
            variants.preview_rgba_by_resolution[2048],
            output_uv,
        )
        source_span = float(np.linalg.norm(uvs[1] - uvs[0]))
        output_span = float(
            np.linalg.norm(
                output_by_position[tuple(vertices[1])]
                - output_by_position[tuple(vertices[0])]
            )
        )

        self.assertLess(output_span, source_span)
        np.testing.assert_allclose(
            sampled,
            np.asarray((128, 0, 128, 255), dtype=float),
            atol=4.0,
        )

    def test_collapsed_face_is_repaired_inside_a_mixed_component(self) -> None:
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        uvs = np.asarray(
            ((0.1, 0.1), (0.3, 0.1), (0.5, 0.1), (0.1, 0.6)),
            dtype=float,
        )

        variants = build_symmetric_half_texture_variants(
            _custom_textured_glb(vertices, faces, uvs)
        )
        output_mesh = next(
            iter(_load_scene(variants.glb_by_resolution[2048]).geometry.values())
        )
        output_uvs = np.asarray(output_mesh.visual.uv, dtype=float)

        self.assertEqual(len(output_mesh.faces), 2)
        self.assertEqual(len(output_mesh.vertices), 6)
        for face in np.asarray(output_mesh.faces, dtype=np.int64):
            triangle = output_uvs[face]
            self.assertGreater(
                abs(float(np.linalg.det(
                    np.vstack(
                        (triangle[1] - triangle[0], triangle[2] - triangle[0])
                    )
                ))),
                0.0,
            )

    def test_zero_area_3d_face_is_dropped_without_removing_valid_face(
        self,
    ) -> None:
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.5, 0.0, 0.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (0, 1, 3)), dtype=np.int64)
        uvs = np.asarray(
            ((0.1, 0.1), (0.4, 0.1), (0.1, 0.4), (0.25, 0.1)),
            dtype=float,
        )

        variants = build_symmetric_half_texture_variants(
            _custom_textured_glb(vertices, faces, uvs)
        )
        output_mesh = next(
            iter(_load_scene(variants.glb_by_resolution[2048]).geometry.values())
        )

        self.assertEqual(len(output_mesh.faces), 1)


# ### Uniform fallback and rejection tests ###
class SymmetricDivisionFallbackAndRejectionTests(unittest.TestCase):
    def test_oversized_island_uses_deterministic_uniform_scale_fallback(
        self,
    ) -> None:
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        uvs = np.asarray(((0.05, 0.05), (0.8, 0.05), (0.05, 0.8)))
        mesh = _textured_mesh(
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
            _smooth_texture(),
            uvs=uvs,
        )
        source = _scene_glb([("oversized-node", mesh, np.eye(4))])
        source_geometry = _load_scene(source).to_geometry()
        source_texture = _first_texture_rgba(source)

        first = build_symmetric_half_texture_variants(source)
        second = build_symmetric_half_texture_variants(source)
        output_geometry = _load_scene(
            first.glb_by_resolution[2048]
        ).to_geometry()
        output_texture = first.preview_rgba_by_resolution[2048]
        source_by_position = _uvs_by_position(source_geometry)
        output_by_position = _uvs_by_position(output_geometry)
        positions = sorted(source_by_position)
        source_uvs = np.asarray([source_by_position[value] for value in positions])
        output_uvs = np.asarray([output_by_position[value] for value in positions])
        source_edges = source_uvs[1:] - source_uvs[0]
        output_edges = output_uvs[1:] - output_uvs[0]
        edge_scales = np.linalg.norm(output_edges, axis=1) / np.linalg.norm(
            source_edges,
            axis=1,
        )

        self.assertLess(float(edge_scales[0]), 1.0)
        self.assertGreater(float(edge_scales[0]), 0.65)
        np.testing.assert_allclose(edge_scales, edge_scales[0], atol=1e-7)
        self.assertAlmostEqual(
            float(np.linalg.det(output_edges)),
            float(np.linalg.det(source_edges) * edge_scales[0] ** 2),
            places=7,
        )
        for barycentric in (
            np.asarray((0.2, 0.3, 0.5)),
            np.asarray((0.7, 0.15, 0.15)),
            np.asarray((1.0 / 3.0,) * 3),
        ):
            np.testing.assert_allclose(
                _sample_texture_bilinear(
                    output_texture,
                    barycentric @ output_uvs,
                ),
                _sample_texture_bilinear(
                    source_texture,
                    barycentric @ source_uvs,
                ),
                atol=1.0,
            )
        self.assertEqual(first.glb_by_resolution, second.glb_by_resolution)
        self.assertEqual(
            first.texture_png_by_resolution,
            second.texture_png_by_resolution,
        )
        expected_black = _solid_texture((0, 0, 0, 255))[:, :1024]
        np.testing.assert_array_equal(output_texture[:, 1024:], expected_black)

    def test_auxiliary_material_maps_are_rejected_before_uv_changes(self) -> None:
        mesh = trimesh.creation.box()
        uvs = np.column_stack(
            (
                np.linspace(0.1, 0.3, len(mesh.vertices)),
                np.linspace(0.2, 0.4, len(mesh.vertices)),
            )
        )
        mesh.visual = TextureVisuals(
            uv=uvs,
            material=PBRMaterial(
                baseColorTexture=Image.fromarray(
                    _solid_texture((10, 20, 30, 255)),
                    mode="RGBA",
                ),
                normalTexture=Image.fromarray(
                    _solid_texture((128, 128, 255, 255)),
                    mode="RGBA",
                ),
            ),
        )
        source = _scene_glb([("normal-map-node", mesh, np.eye(4))])

        with self.assertRaisesRegex(ValueError, "auxiliary material"):
            build_symmetric_half_texture_variants(source)

    def test_uv_island_count_is_bounded_before_mask_growth(self) -> None:
        vertices = np.asarray(
            [
                (float(face_index * 2 + corner), float(corner == 2), 0.0)
                for face_index in range(3)
                for corner in range(3)
            ],
            dtype=float,
        )
        faces = np.arange(9, dtype=np.int64).reshape((3, 3))
        uvs = np.tile(
            np.asarray(((0.1, 0.1), (0.2, 0.1), (0.1, 0.2))),
            (3, 1),
        )
        mesh = _textured_mesh(
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
            _solid_texture((1, 2, 3, 255)),
            uvs=uvs,
        )
        source = _scene_glb([("islands-node", mesh, np.eye(4))])

        with mock.patch(
            "housemaker.object_symmetry._MAX_UV_ISLAND_COUNT",
            2,
        ):
            with self.assertRaisesRegex(ValueError, "too many"):
                build_symmetric_half_texture_variants(source)

    def test_uv_overlap_pixel_work_is_bounded_before_mask_comparison(
        self,
    ) -> None:
        vertices = np.asarray(
            [
                (float(face_index * 2 + corner), float(corner == 2), 0.0)
                for face_index in range(2)
                for corner in range(3)
            ],
            dtype=float,
        )
        faces = np.arange(6, dtype=np.int64).reshape((2, 3))
        uvs = np.tile(
            np.asarray(((0.1, 0.1), (0.4, 0.1), (0.1, 0.4))),
            (2, 1),
        )
        mesh = _textured_mesh(
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
            _solid_texture((1, 2, 3, 255)),
            uvs=uvs,
        )
        source = _scene_glb([("overlap-node", mesh, np.eye(4))])

        with mock.patch(
            "housemaker.object_symmetry._MAX_UV_OVERLAP_TEST_PIXELS",
            1,
        ):
            with self.assertRaisesRegex(ValueError, "packing work"):
                build_symmetric_half_texture_variants(source)

    def test_distinct_texture_atlases_are_rejected(self) -> None:
        first = _textured_mesh(
            trimesh.creation.box(),
            _solid_texture((255, 0, 0, 255)),
        )
        second = _textured_mesh(
            trimesh.creation.icosphere(subdivisions=1),
            _solid_texture((0, 0, 255, 255)),
        )
        transform = np.eye(4, dtype=float)
        transform[0, 3] = 3.0
        source = _scene_glb(
            [
                ("first-node", first, np.eye(4)),
                ("second-node", second, transform),
            ]
        )

        with self.assertRaisesRegex(ValueError, "more than one distinct"):
            build_symmetric_object_variants(
                source,
                SymmetricDivisionOptions("vertical", "left"),
            )

    def test_shared_geometry_nodes_are_rejected_before_output(self) -> None:
        mesh = _textured_mesh(
            trimesh.creation.box(),
            _solid_texture((10, 20, 30, 255)),
        )
        scene = trimesh.Scene()
        scene.add_geometry(mesh, geom_name="shared", node_name="first-node")
        second_transform = np.eye(4, dtype=float)
        second_transform[0, 3] = 3.0
        scene.graph.update(
            frame_to="second-node",
            matrix=second_transform,
            geometry="shared",
        )
        source = bytes(scene.export(file_type="glb"))

        with self.assertRaisesRegex(ValueError, "shared"):
            build_symmetric_object_variants(
                source,
                SymmetricDivisionOptions("vertical", "left"),
            )

    def test_missing_texture_is_rejected(self) -> None:
        source = bytes(
            trimesh.Scene(trimesh.creation.box()).export(file_type="glb")
        )
        with self.assertRaisesRegex(ValueError, "base-color texture"):
            build_symmetric_object_variants(
                source,
                SymmetricDivisionOptions("vertical", "left"),
            )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
