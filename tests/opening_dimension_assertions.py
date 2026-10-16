# ### Imports ###
from __future__ import annotations

from housemaker.canvas_openings import build_canvas_opening_targets
from housemaker.models import LevelData
from housemaker.surface_geometry import build_fixed_surfaces


# ### Opening dimension helpers ###
def get_opening_physical_size(
    level: LevelData,
    stable_id: str,
) -> tuple[float, float]:
    """Return one Canvas opening's physical width and height in meters."""

    target = next(
        target
        for target in build_canvas_opening_targets(
            (level,),
            tuple(build_fixed_surfaces((level,))),
        )
        if target.reference.stable_id == stable_id
    )
    return (
        target.bounds.horizontal_span * target.wall_width_meters,
        target.bounds.vertical_span * target.wall_height_meters,
    )
