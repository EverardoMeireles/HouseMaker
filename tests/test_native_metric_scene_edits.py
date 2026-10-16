# ### Imports ###
from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from housemaker.canvas_surface_edits import (
    CANVAS_SURFACE_EDIT_AXIS_X,
    CANVAS_SURFACE_EDIT_WALL_TRANSLATION,
    CanvasSurfaceEdit,
    apply_canvas_surface_edit,
    build_canvas_surface_edit_targets,
)
from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.models import LevelData, VertexData
from housemaker.outer_wall_marking import project_world_points_to_level_image
from housemaker.plan_metric_state import PlanMeasurementEvidence, PlanMetricCalibration
from housemaker.surface_geometry import build_fixed_surfaces


# ### Native-density fixtures ###
def _level(density: float, *, scale: float = 1.0) -> LevelData:
    size = round(8.0 / density)
    calibration = PlanMetricCalibration(
        estimated_meters_per_pixel=density,
        target_meters_per_pixel=density,
        normalization_scale=1.0,
        relative_residual=0.0,
        confidence="medium",
        source_image_size=(size, size),
        pre_normalization_size=(size, size),
        normalized_image_size=(size, size),
        measurements=tuple(
            PlanMeasurementEvidence(
                f"width {y}", 4.0, (0.2, y), (0.7, y),
                (0.2, y), (0.7, y), "high",
            )
            for y in (0.2, 0.7)
        ),
    )
    vertices = VertexData()
    corners = tuple(
        vertices.add_vertex(x / density, y / density)
        for x, y in ((2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0))
    )
    for start, end in zip(corners, (*corners[1:], corners[0]), strict=True):
        vertices.add_edge(start.id, end.id)
    return LevelData(
        index=2,
        name="Ground",
        scale=scale,
        height_meters=2.45,
        image_size_pixels=(float(size), float(size)),
        vertex_data=vertices,
        plan_metric_calibration=calibration,
    )


def _wall_translation_target(level: LevelData):
    surfaces = build_fixed_surfaces((level,))
    return next(
        target
        for target in build_canvas_surface_edit_targets((level,), surfaces)
        if target.reference.kind == CANVAS_SURFACE_EDIT_WALL_TRANSLATION
        and target.reference.axis_index == CANVAS_SURFACE_EDIT_AXIS_X
        and target.surface_id == "level:2/wall:1:2"
    )


# ### Metric editing and cross-level alignment regressions ###
class NativeMetricSceneEditTests(unittest.TestCase):
    def test_wall_gizmo_moves_one_metre_at_each_native_density(self) -> None:
        for density in (0.01, 0.04):
            for scale in (1.0, 1.7):
                with self.subTest(density=density, scale=scale):
                    level = _level(density, scale=scale)
                    target = _wall_translation_target(level)
                    vertex = level.vertex_data.get_vertex(1)
                    before_pixel_x = vertex.x
                    before = level_image_to_world_xy(level, vertex.x, vertex.y)
                    apply_canvas_surface_edit(
                        (level,), target,
                        CanvasSurfaceEdit(target.reference, target.surface_id, 1.0),
                    )
                    vertex = level.vertex_data.get_vertex(1)
                    after = level_image_to_world_xy(level, vertex.x, vertex.y)
                    np.testing.assert_allclose(np.subtract(after, before), (1.0, 0.0))
                    self.assertAlmostEqual(vertex.x - before_pixel_x, 1.0 / density / scale)
                    self.assertEqual(level.height_meters, 2.45)

    def test_changed_calibration_invalidates_in_progress_gizmo(self) -> None:
        level = _level(0.01)
        target = _wall_translation_target(level)
        level.plan_metric_calibration = replace(
            level.plan_metric_calibration,
            estimated_meters_per_pixel=0.02,
            target_meters_per_pixel=0.02,
        )
        with self.assertRaisesRegex(ValueError, "calibration changed"):
            apply_canvas_surface_edit(
                (level,), target,
                CanvasSurfaceEdit(target.reference, target.surface_id, 1.0),
            )

    def test_outer_wall_projection_preserves_world_alignment_across_densities(self) -> None:
        source = _level(0.01)
        target = _level(0.04, scale=1.3)
        target.offset_x_meters = 0.75
        target.offset_y_meters = -0.5
        world_points = tuple(
            level_image_to_world_xy(source, vertex.x, vertex.y)
            for vertex in source.vertex_data.vertices
        )
        image_points = project_world_points_to_level_image(target, world_points)
        for point in image_points:
            target.vertex_data.add_vertex(*point)
        projected = tuple(level_image_to_world_xy(target, *point) for point in image_points)
        np.testing.assert_allclose(projected, world_points, atol=1e-8)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
