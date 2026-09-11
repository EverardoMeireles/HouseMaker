# ### Imports ###
from __future__ import annotations

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from housemaker.architectural_surface_edits import (
    build_surface_drawing_overlay,
    insert_surface_vertex,
    place_surface_vertex,
)
from housemaker.glb import convert_to_glb, convert_to_preview_model
from housemaker.models import LevelData, VertexData
from housemaker.project_io import load_project, save_project
from housemaker.surface_geometry import FixedSurface, build_fixed_surfaces


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
    return LevelData(index=2, name="Ground", vertex_data=vertex_data)


def _get_root_floor(level: LevelData) -> FixedSurface:
    return next(
        surface
        for surface in build_fixed_surfaces((level,))
        if surface.surface_id == "level:2/floor"
    )


def _get_inner_triangle_points(
    surface: FixedSurface,
) -> tuple[tuple[float, float, float], ...]:
    first, second, third = np.asarray(surface.mesh.triangles[0], dtype=float)
    return tuple(
        tuple(float(component) for component in point)
        for point in (
            (0.6 * first) + (0.2 * second) + (0.2 * third),
            (0.2 * first) + (0.6 * second) + (0.2 * third),
            (0.2 * first) + (0.2 * second) + (0.6 * third),
        )
    )


def _draw_closed_inner_face(level: LevelData):
    source = _get_root_floor(level)
    points = _get_inner_triangle_points(source)
    active_vertex_id = None
    for point in points:
        result = place_surface_vertex(
            (level,),
            source.surface_id,
            point,
            active_vertex_id,
        )
        active_vertex_id = result.active_vertex_id
    return source, place_surface_vertex(
        (level,),
        source.surface_id,
        points[0],
        active_vertex_id,
    )


def _solid_png() -> bytes:
    output = BytesIO()
    Image.new("RGBA", (8, 8), (110, 160, 210, 255)).save(
        output,
        format="PNG",
    )
    return output.getvalue()


def _find_level(levels: list[LevelData], level_index: int) -> LevelData:
    return next(level for level in levels if level.index == level_index)


def _triangle_multiset(mesh: trimesh.Trimesh) -> tuple[object, ...]:
    """Return orientation-independent rounded triangles for exact comparisons."""

    return tuple(
        sorted(
            tuple(
                sorted(
                    tuple(round(float(component), 8) for component in point)
                    for point in triangle
                )
            )
            for triangle in np.asarray(mesh.triangles, dtype=float)
        )
    )


def _load_export_mesh(glb_bytes: bytes) -> trimesh.Trimesh:
    scene = trimesh.load(BytesIO(glb_bytes), file_type="glb", force="scene")
    if not isinstance(scene, trimesh.Scene):
        raise TypeError("GLB fixture did not load as a scene.")
    return scene.to_geometry()


# ### Draft persistence and export tests ###
class ArchitecturalSurfaceDraftIntegrationTests(unittest.TestCase):
    def test_open_chain_round_trips_and_does_not_replace_export_geometry(
        self,
    ) -> None:
        level = _build_square_level()
        root = _get_root_floor(level)
        points = _get_inner_triangle_points(root)
        baseline_preview = convert_to_preview_model((level,))
        baseline_export = convert_to_glb((level,))
        baseline_geometry_names = set(baseline_preview.scene.geometry)
        first = place_surface_vertex(
            (level,),
            root.surface_id,
            points[0],
        )
        second = place_surface_vertex(
            (level,),
            root.surface_id,
            points[1],
            first.active_vertex_id,
        )

        self.assertFalse(second.requires_mesh_refresh)
        self.assertFalse(level.editable_surfaces[0].replaces_source_surface)
        self.assertEqual(len(level.editable_surfaces[0].edges), 1)
        self.assertIn(
            root.surface_id,
            {surface.surface_id for surface in build_fixed_surfaces((level,))},
        )
        draft_preview = convert_to_preview_model((level,))
        draft_export = convert_to_glb((level,))
        self.assertEqual(set(draft_preview.scene.geometry), baseline_geometry_names)
        self.assertEqual(
            _triangle_multiset(draft_preview.mesh),
            _triangle_multiset(baseline_preview.mesh),
        )
        self.assertEqual(
            _triangle_multiset(draft_export.mesh),
            _triangle_multiset(baseline_export.mesh),
        )
        self.assertEqual(
            _triangle_multiset(_load_export_mesh(draft_export.glb_bytes)),
            _triangle_multiset(_load_export_mesh(baseline_export.glb_bytes)),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "open-chain.json"
            save_project(project_path, level.index, [level])
            loaded = load_project(project_path)

        loaded_level = _find_level(loaded.levels, level.index)
        self.assertEqual(loaded_level.editable_surfaces, level.editable_surfaces)
        self.assertFalse(loaded_level.editable_surfaces[0].replaces_source_surface)
        self.assertEqual(
            build_surface_drawing_overlay(loaded.levels),
            build_surface_drawing_overlay((level,)),
        )
        self.assertIn(
            root.surface_id,
            {surface.surface_id for surface in build_fixed_surfaces((loaded_level,))},
        )

    def test_legacy_edit_records_default_to_replacing_with_no_drawn_edges(
        self,
    ) -> None:
        level = _build_square_level()
        root = _get_root_floor(level)
        point = tuple(
            float(component)
            for component in np.asarray(root.mesh.triangles[0]).mean(axis=0)
        )
        insert_surface_vertex((level,), root.surface_id, point)

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "legacy-edit.json"
            save_project(project_path, level.index, [level])
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            raw_level = next(
                item for item in payload["levels"] if item["index"] == level.index
            )
            raw_edit = raw_level["editable_surfaces"][0]
            raw_edit.pop("edges")
            raw_edit.pop("replaces_source_surface")
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_project(project_path)

        loaded_level = _find_level(loaded.levels, level.index)
        loaded_edit = loaded_level.editable_surfaces[0]
        self.assertTrue(loaded_edit.replaces_source_surface)
        self.assertEqual(loaded_edit.edges, ())
        self.assertNotIn(
            root.surface_id,
            {surface.surface_id for surface in build_fixed_surfaces((loaded_level,))},
        )


# ### Closed-face lineage and material tests ###
class ArchitecturalSurfaceClosedFaceIntegrationTests(unittest.TestCase):
    def test_closed_loop_partitions_parent_and_exports_every_textured_child(
        self,
    ) -> None:
        level = _build_square_level()
        source, result = _draw_closed_inner_face(level)
        child_ids = result.replacements[source.surface_id]
        current_surfaces = {
            surface.surface_id: surface for surface in build_fixed_surfaces((level,))
        }

        self.assertTrue(result.requires_mesh_refresh)
        self.assertTrue(result.state_changed)
        self.assertTrue(result.created_surface_ids)
        self.assertTrue(set(result.created_surface_ids).issubset(child_ids))
        self.assertNotIn(source.surface_id, current_surfaces)
        self.assertEqual(
            set(child_ids),
            {
                surface_id
                for surface_id, surface in current_surfaces.items()
                if surface.source_surface_id == source.surface_id
            },
        )
        self.assertAlmostEqual(
            sum(
                current_surfaces[surface_id].area_square_meters
                for surface_id in child_ids
            ),
            source.area_square_meters,
        )

        model = convert_to_glb(
            (level,),
            surface_materials={surface_id: _solid_png() for surface_id in child_ids},
            export_untextured_surfaces=False,
        )
        self.assertEqual(
            {surface.surface_id for surface in model.preview_textured_surfaces},
            set(child_ids),
        )
        self.assertGreater(len(model.glb_bytes), 20)
        exported_scene = trimesh.load(
            BytesIO(model.glb_bytes),
            file_type="glb",
            force="scene",
        )
        self.assertIsInstance(exported_scene, trimesh.Scene)
        self.assertAlmostEqual(
            sum(float(mesh.area) for mesh in exported_scene.geometry.values()),
            source.area_square_meters,
        )


if __name__ == "__main__":
    unittest.main()
