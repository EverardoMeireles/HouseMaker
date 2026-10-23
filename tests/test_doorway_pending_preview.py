# ### Imports ###
from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from housemaker.canvas_openings import (
    CanvasOpeningBounds,
    CanvasOpeningEdit,
    apply_canvas_opening_edit,
    build_canvas_opening_preview_targets,
    build_canvas_opening_targets,
)
from housemaker.models import DoorwayData, LevelData
from housemaker.surface_geometry import build_fixed_surfaces


# ### Fixture helpers ###
def _gapped_level(angle: float, reverse_edges: bool) -> tuple[LevelData, DoorwayData]:
    level = LevelData(index=2, name="Ground")
    radians = math.radians(angle)
    width = np.asarray((math.cos(radians), math.sin(radians)))
    depth = np.asarray((-width[1], width[0]))
    for along, across in (
        (0, 0), (100, 0), (0, 10), (100, 10),
        (-100, 0), (200, 0), (-100, 10), (200, 10),
    ):
        point = np.asarray((200.0, 200.0)) + width * along + depth * across
        level.vertex_data.add_vertex(*point)
    for first, second in ((5, 1), (2, 6), (7, 3), (4, 8), (1, 3), (2, 4)):
        level.vertex_data.add_edge(second, first) if reverse_edges else level.vertex_data.add_edge(first, second)
    center = np.asarray((200.0, 200.0)) + width * 50.0 + depth * 5.0
    doorway = DoorwayData(
        doorway_id="pending-gap",
        center_x=float(center[0]),
        center_y=float(center[1]),
        width_meters=2.0,
        height_meters=2.1,
        depth_meters=0.2,
        rotation_degrees=angle + 90.0,
        corner_vertex_ids=(1, 2, 3, 4),
    )
    return level, doorway


# ### Pure metadata preview regressions ###
@pytest.mark.parametrize("angle", (0.0, 27.0, 90.0))
@pytest.mark.parametrize("reverse_edges", (False, True))
def test_new_gap_doorway_preview_uses_live_spans_without_rebuilding_meshes(
    angle: float, reverse_edges: bool,
) -> None:
    level, doorway = _gapped_level(angle, reverse_edges)
    old_surfaces = build_fixed_surfaces((level,))
    level.doorways = [doorway]
    graph_before = level.vertex_data.to_dict()

    with patch("housemaker.surface_geometry.build_fixed_surfaces", side_effect=AssertionError("mesh rebuild")), patch(
        "housemaker.surface_geometry._build_mesh", side_effect=AssertionError("mesh construction")
    ):
        (target,) = build_canvas_opening_preview_targets((level,), old_surfaces)

    assert 0.0 <= target.bounds.start_ratio < target.bounds.end_ratio <= 1.0
    assert target.wall_width_meters == pytest.approx(6.0)
    assert target.bounds.start_ratio == pytest.approx(1 / 3)
    assert target.bounds.end_ratio == pytest.approx(2 / 3)
    assert target.wall_width_meters * target.bounds.horizontal_span == pytest.approx(2.0)
    assert level.vertex_data.to_dict() == graph_before
    assert level.doorways == [doorway]

    # The frame is usable immediately by the normal drag/apply pipeline.
    edit = CanvasOpeningEdit(
        reference=target.reference,
        wall_surface_id=target.wall_surface_id,
        bounds=CanvasOpeningBounds(
            target.bounds.start_ratio + 0.01,
            target.bounds.end_ratio + 0.01,
            target.bounds.bottom_ratio,
            target.bounds.top_ratio,
        ),
    )
    applied = apply_canvas_opening_edit((level,), target, edit)
    assert applied.current.doorway_id == doorway.doorway_id


def test_preview_corrects_out_of_range_target_from_disconnected_old_walls() -> None:
    level, doorway = _gapped_level(0.0, False)
    old_surfaces = build_fixed_surfaces((level,))
    level.doorways = [doorway]
    (old_target,) = build_canvas_opening_targets((level,), old_surfaces)
    assert old_target.bounds.start_ratio < 0.0 or old_target.bounds.end_ratio > 1.0
    (target,) = build_canvas_opening_preview_targets((level,), old_surfaces)
    assert 0.0 <= target.bounds.start_ratio < target.bounds.end_ratio <= 1.0
    assert target.wall_surface_id != old_target.wall_surface_id
