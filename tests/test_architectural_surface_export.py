# ### Imports ###
from __future__ import annotations

import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from housemaker.architectural_surface_edits import (
    extrude_surface_faces,
    insert_surface_vertex,
)
from housemaker.glb import convert_to_preview_model
from housemaker.models import LevelData, VertexData
from housemaker.surface_geometry import (
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    FixedSurface,
    build_fixed_surfaces,
)


# ### Fixture helpers ###
def _build_square_level() -> LevelData:
    vertex_data = VertexData()
    vertex_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        )
    )
    for start_id, end_id in zip(
        vertex_ids,
        (*vertex_ids[1:], vertex_ids[0]),
        strict=True,
    ):
        vertex_data.add_edge(start_id, end_id)
    return LevelData(
        index=2,
        name="Ground",
        height_meters=3.0,
        vertex_data=vertex_data,
    )


def _floor_surface(level: LevelData) -> FixedSurface:
    return next(
        surface
        for surface in build_fixed_surfaces((level,))
        if surface.surface_type == SURFACE_TYPE_FLOOR
        and surface.source_surface_id is None
    )


def _triangle_key(triangle: object) -> tuple[tuple[float, ...], ...]:
    points = np.round(np.asarray(triangle, dtype=float), 8)
    return tuple(sorted(tuple(float(value) for value in point) for point in points))


def _solid_png() -> bytes:
    output = BytesIO()
    Image.new("RGBA", (8, 8), (130, 180, 220, 255)).save(
        output,
        format="PNG",
    )
    return output.getvalue()


def _build_extruded_floor(
    level: LevelData,
) -> tuple[FixedSurface, tuple[str, ...]]:
    source = _floor_surface(level)
    insertion_point = tuple(
        float(value) for value in source.mesh.triangles[0].mean(axis=0)
    )
    insertion = insert_surface_vertex(
        (level,),
        source.surface_id,
        insertion_point,
    )
    extrusion = extrude_surface_faces(
        (level,),
        (insertion.selected_surface_ids[0],),
        0.2,
    )
    return source, extrusion.selected_surface_ids


# ### Editable surface export tests ###
class ArchitecturalSurfaceExportTests(unittest.TestCase):
    def test_extrusion_replaces_the_source_skin_without_overlap(self) -> None:
        level = _build_square_level()
        source = _floor_surface(level)
        original_triangle_key = _triangle_key(source.mesh.triangles[0])
        base_model = convert_to_preview_model((level,))

        _source, retained_ids = _build_extruded_floor(level)
        surfaces = build_fixed_surfaces((level,))
        edited_surfaces = tuple(
            surface
            for surface in surfaces
            if surface.source_surface_id == source.surface_id
        )
        model = convert_to_preview_model((level,))

        self.assertNotIn(source.surface_id, {item.surface_id for item in surfaces})
        self.assertEqual(len(retained_ids), 1)
        self.assertEqual(len(model.mesh.faces), len(base_model.mesh.faces) + 8)
        model_keys = [_triangle_key(triangle) for triangle in model.mesh.triangles]
        self.assertNotIn(original_triangle_key, model_keys)
        for surface in edited_surfaces:
            if surface.surface_type != SURFACE_TYPE_FLOOR:
                continue
            for triangle in surface.mesh.triangles:
                self.assertEqual(model_keys.count(_triangle_key(triangle)), 1)

    def test_extruded_side_can_be_textured_independently(self) -> None:
        level = _build_square_level()
        source, _retained_ids = _build_extruded_floor(level)
        edited_surfaces = tuple(
            surface
            for surface in build_fixed_surfaces((level,))
            if surface.source_surface_id == source.surface_id
        )
        side = next(
            surface
            for surface in edited_surfaces
            if surface.surface_type == SURFACE_TYPE_WALL
        )

        model = convert_to_preview_model(
            (level,),
            surface_materials={side.surface_id: _solid_png()},
        )

        self.assertEqual(
            tuple(
                surface.surface_id
                for surface in model.preview_textured_surfaces
            ),
            (side.surface_id,),
        )
        textured_mesh = model.preview_textured_surfaces[0].mesh
        texture_uv = np.asarray(textured_mesh.visual.uv, dtype=float)
        self.assertTrue(np.all(np.isfinite(texture_uv)))
        self.assertGreater(float(np.ptp(texture_uv[:, 0])), 0.0)
        self.assertGreater(float(np.ptp(texture_uv[:, 1])), 0.0)
        self.assertIsNotNone(model.preview_untextured_mesh)
        assert model.preview_untextured_mesh is not None
        untextured_keys = {
            _triangle_key(triangle)
            for triangle in model.preview_untextured_mesh.triangles
        }
        self.assertTrue(
            all(
                _triangle_key(triangle) not in untextured_keys
                for triangle in side.mesh.triangles
            )
        )


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
