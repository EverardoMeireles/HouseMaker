# ### Imports ###
from __future__ import annotations

import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from housemaker.glb import convert_to_glb
from housemaker.models import LevelData, RoomData, VertexData
from housemaker.surface_geometry import (
    SURFACE_TYPE_CEILING,
    FixedSurface,
    build_fixed_surfaces,
)


# ### Constants ###
EXPECTED_CEILING_CLEARANCE_METERS = 0.001


# ### Fixture helpers ###
def _add_square_boundary(vertex_data: VertexData) -> tuple[int, ...]:
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
        strict=True,
    ):
        vertex_data.add_edge(start_id, end_id)
    return boundary_ids


def _build_square_level(
    index: int,
    name: str,
    *,
    floor_thickness_meters: float,
    with_room: bool = False,
) -> LevelData:
    vertex_data = VertexData()
    boundary_ids = _add_square_boundary(vertex_data)
    rooms: list[RoomData] = []
    if with_room:
        center = vertex_data.add_vertex(50.0, 50.0)
        rooms.append(
            RoomData(
                name="Room",
                vertex_ids=boundary_ids,
                center_vertex_id=center.id,
                color_rgb=(140, 180, 220),
            )
        )
    return LevelData(
        index=index,
        name=name,
        height_meters=3.0,
        floor_thickness_meters=floor_thickness_meters,
        vertex_data=vertex_data,
        rooms=rooms,
    )


def _solid_png() -> bytes:
    output = BytesIO()
    Image.new("RGBA", (8, 8), (170, 190, 210, 255)).save(
        output,
        format="PNG",
    )
    return output.getvalue()


def _get_ceiling_surface(level: LevelData) -> FixedSurface:
    return next(
        surface
        for surface in build_fixed_surfaces([level])
        if surface.surface_type == SURFACE_TYPE_CEILING
    )


# ### Ceiling geometry tests ###
class CeilingGeometryTests(unittest.TestCase):
    def test_untextured_ceiling_is_an_opaque_downward_facing_plane(self) -> None:
        level = _build_square_level(
            2,
            "Ground",
            floor_thickness_meters=0.2,
        )

        model = convert_to_glb([level])

        self.assertIn("l2_ground_ceiling", model.scene.geometry)
        ceiling_mesh = model.scene.geometry["l2_ground_ceiling"]
        self.assertFalse(ceiling_mesh.is_volume)
        self.assertGreater(len(ceiling_mesh.faces), 0)
        self.assertTrue(np.all(ceiling_mesh.face_normals[:, 1] < -0.999))
        self.assertAlmostEqual(float(ceiling_mesh.area), 4.0)
        self.assertTrue(
            np.all(np.asarray(ceiling_mesh.visual.face_colors)[:, 3] == 255)
        )

    def test_ceiling_has_one_millimeter_clearance_below_upper_floor(self) -> None:
        ground = _build_square_level(
            2,
            "Ground",
            floor_thickness_meters=0.2,
        )
        upper = _build_square_level(
            3,
            "First floor",
            floor_thickness_meters=0.5,
        )

        model = convert_to_glb([ground, upper])

        ceiling_mesh = model.scene.geometry["l2_ground_ceiling"]
        upper_floor_mesh = model.scene.geometry["l3_first_floor_floor"]
        ceiling_height = float(ceiling_mesh.bounds[1, 1])
        upper_floor_bottom = float(upper_floor_mesh.bounds[0, 1])
        self.assertAlmostEqual(
            upper_floor_bottom - ceiling_height,
            EXPECTED_CEILING_CLEARANCE_METERS,
            places=9,
        )
        self.assertAlmostEqual(
            ceiling_height,
            (
                ground.floor_thickness_meters
                + ground.height_meters
                - EXPECTED_CEILING_CLEARANCE_METERS
            ),
            places=9,
        )

        semantic_ceiling = _get_ceiling_surface(ground)
        np.testing.assert_allclose(
            semantic_ceiling.mesh.bounds[:, 2],
            ceiling_height,
            atol=1e-9,
        )

    def test_ceiling_texture_replaces_the_opaque_base_plane(self) -> None:
        level = _build_square_level(
            2,
            "Ground",
            floor_thickness_meters=0.2,
            with_room=True,
        )
        ceiling_surface = _get_ceiling_surface(level)
        base_model = convert_to_glb([level])

        textured_model = convert_to_glb(
            [level],
            surface_materials={ceiling_surface.surface_id: _solid_png()},
        )

        self.assertEqual(
            len(textured_model.mesh.faces),
            len(base_model.mesh.faces),
        )
        self.assertIsNotNone(textured_model.preview_untextured_mesh)
        assert textured_model.preview_untextured_mesh is not None
        self.assertEqual(
            len(textured_model.preview_untextured_mesh.faces),
            len(base_model.mesh.faces) - len(ceiling_surface.mesh.faces),
        )
        self.assertEqual(len(textured_model.preview_textured_surfaces), 1)
        textured_ceiling = textured_model.preview_textured_surfaces[0]
        self.assertEqual(textured_ceiling.surface_id, ceiling_surface.surface_id)
        np.testing.assert_allclose(
            textured_ceiling.mesh.triangles,
            ceiling_surface.mesh.triangles,
            atol=1e-9,
        )


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
