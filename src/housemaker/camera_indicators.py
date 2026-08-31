# ### Imports ###
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont

from housemaker.unused_face_removal import (
    ALL_CAMERA_IDS,
    CAMERA_ID_BOTTOM,
    CAMERA_ID_NEG_X,
    CAMERA_ID_NEG_Y,
    CAMERA_ID_POS_X,
    CAMERA_ID_POS_Y,
    CAMERA_ID_TOP,
)


# ### Constants ###
INDICATOR_COLOR = (1.0, 0.54, 0.12, 0.96)
INDICATOR_LINE_WIDTH = 2.0
INDICATOR_DEPTH_VALUE = 10
INDICATOR_LABEL_DEPTH_VALUE = 11
INDICATOR_LABEL_COLOR = QColor(255, 225, 176, 255)
INDICATOR_LABEL_FONT_POINT_SIZE = 12
INDICATOR_SELECTED_COLOR = (0.24, 0.82, 1.0, 1.0)
INDICATOR_SELECTED_LINE_WIDTH = 3.5
INDICATOR_SELECTED_LABEL_COLOR = QColor(190, 238, 255, 255)
INDICATOR_BAR_TRACK_COLOR = (0.12, 0.14, 0.18, 0.92)
INDICATOR_BAR_TRACK_SELECTED_COLOR = (0.12, 0.30, 0.38, 0.96)
INDICATOR_BAR_FILL_COLOR = (1.0, 0.54, 0.12, 1.0)
INDICATOR_BAR_FILL_SELECTED_COLOR = (0.24, 0.82, 1.0, 1.0)
INDICATOR_BAR_TRACK_WIDTH = 7.0
INDICATOR_BAR_FILL_WIDTH = 5.0
MINIMUM_INDICATOR_SCALE = 0.1
CAMERA_CLEARANCE_RATIO = 0.55
CAMERA_BODY_HALF_WIDTH_RATIO = 0.10
CAMERA_BODY_HALF_HEIGHT_RATIO = 0.07
CAMERA_BODY_HALF_DEPTH_RATIO = 0.06
CAMERA_LENS_LENGTH_RATIO = 0.18
CAMERA_ARROW_LENGTH_RATIO = 0.09
CAMERA_ARROW_WIDTH_RATIO = 0.045
CAMERA_SURFACE_CLEARANCE_RATIO = 0.025
CAMERA_LABEL_OFFSET_RATIO = 0.06
CAMERA_PERCENTAGE_BAR_OFFSET_RATIO = 0.055
CAMERA_PERCENTAGE_BAR_HALF_HEIGHT_RATIO = 0.12
MINIMUM_CAMERA_PERCENTAGE = 1
MAXIMUM_CAMERA_PERCENTAGE = 95
MAXIMUM_TOTAL_CAMERA_PERCENTAGE = 100
DEFAULT_CAMERA_INDICATOR_PERCENTAGES = (17, 17, 17, 17, 16, 16)
PROJECTION_CAMERA_INDICATOR_LABELS = {
    CAMERA_ID_POS_X: "+X",
    CAMERA_ID_NEG_X: "-X",
    CAMERA_ID_POS_Y: "+Y",
    CAMERA_ID_NEG_Y: "-Y",
    CAMERA_ID_TOP: "Top",
    CAMERA_ID_BOTTOM: "Bottom",
}


# ### Data models ###
@dataclass(frozen=True)
class ProjectionCameraIndicatorGeometry:
    """Line geometry for one illustrative projection camera."""

    camera_id: str
    camera_position: np.ndarray
    label_position: np.ndarray
    aim_endpoint: np.ndarray
    line_positions: np.ndarray
    selection_line_positions: np.ndarray
    percentage_bar_start: np.ndarray
    percentage_bar_end: np.ndarray


@dataclass(frozen=True)
class _CameraAxes:
    outward: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]


# ### Camera axes ###
_CAMERA_AXES = {
    CAMERA_ID_POS_X: _CameraAxes(
        outward=(1.0, 0.0, 0.0),
        right=(0.0, 1.0, 0.0),
        up=(0.0, 0.0, 1.0),
    ),
    CAMERA_ID_NEG_X: _CameraAxes(
        outward=(-1.0, 0.0, 0.0),
        right=(0.0, -1.0, 0.0),
        up=(0.0, 0.0, 1.0),
    ),
    CAMERA_ID_POS_Y: _CameraAxes(
        outward=(0.0, 1.0, 0.0),
        right=(-1.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
    ),
    CAMERA_ID_NEG_Y: _CameraAxes(
        outward=(0.0, -1.0, 0.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
    ),
    CAMERA_ID_TOP: _CameraAxes(
        outward=(0.0, 0.0, 1.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
    ),
    CAMERA_ID_BOTTOM: _CameraAxes(
        outward=(0.0, 0.0, -1.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, -1.0, 0.0),
    ),
}


# ### Public builders ###
def build_projection_camera_indicator_geometries(
    bounds: np.ndarray,
) -> dict[str, ProjectionCameraIndicatorGeometry]:
    """Build six camera outlines outside ``bounds`` and aimed inward."""

    minimum, maximum = _normalize_bounds(bounds)
    center = (minimum + maximum) / 2.0
    half_extents = (maximum - minimum) / 2.0
    scale = max(float(np.max(maximum - minimum)), MINIMUM_INDICATOR_SCALE)
    return {
        camera_id: _build_camera_indicator_geometry(
            camera_id=camera_id,
            center=center,
            half_extents=half_extents,
            scale=scale,
        )
        for camera_id in ALL_CAMERA_IDS
    }


def create_projection_camera_indicator_items(
    bounds: np.ndarray,
    camera_percentages: Sequence[int] | Mapping[str, int] = (
        DEFAULT_CAMERA_INDICATOR_PERCENTAGES
    ),
    *,
    selected_camera_id: str | None = None,
) -> dict[str, tuple[GLGraphicsItem, ...]]:
    """Create camera outlines, allocation bars, and percentage labels."""

    items: dict[str, tuple[GLGraphicsItem, ...]] = {}
    label_font = QFont("Helvetica", INDICATOR_LABEL_FONT_POINT_SIZE)
    label_font.setBold(True)
    geometries = build_projection_camera_indicator_geometries(bounds)
    percentages = normalize_projection_camera_indicator_percentages(
        camera_percentages
    )
    selected_id = _normalize_selected_camera_id(selected_camera_id)
    for camera_index, (camera_id, geometry) in enumerate(geometries.items()):
        is_selected = camera_id == selected_id
        line_item = gl.GLLinePlotItem(
            pos=geometry.line_positions,
            color=(INDICATOR_SELECTED_COLOR if is_selected else INDICATOR_COLOR),
            width=(
                INDICATOR_SELECTED_LINE_WIDTH
                if is_selected
                else INDICATOR_LINE_WIDTH
            ),
            antialias=True,
            mode="lines",
            glOptions="translucent",
        )
        line_item.setDepthValue(INDICATOR_DEPTH_VALUE)
        track_item = gl.GLLinePlotItem(
            pos=np.asarray(
                (geometry.percentage_bar_start, geometry.percentage_bar_end),
                dtype=np.float32,
            ),
            color=(
                INDICATOR_BAR_TRACK_SELECTED_COLOR
                if is_selected
                else INDICATOR_BAR_TRACK_COLOR
            ),
            width=INDICATOR_BAR_TRACK_WIDTH,
            antialias=True,
            mode="lines",
            glOptions="translucent",
        )
        track_item.setDepthValue(INDICATOR_DEPTH_VALUE + 2)
        fill_item = gl.GLLinePlotItem(
            pos=_build_percentage_bar_fill_positions(
                geometry,
                percentages[camera_index],
            ),
            color=(
                INDICATOR_BAR_FILL_SELECTED_COLOR
                if is_selected
                else INDICATOR_BAR_FILL_COLOR
            ),
            width=INDICATOR_BAR_FILL_WIDTH,
            antialias=True,
            mode="lines",
            glOptions="translucent",
        )
        # Draw the narrower fill before the wider track.  Their shared center
        # keeps the fill visible, while the later track contributes only the
        # uncovered border instead of rejecting the fill at equal depth.
        fill_item.setDepthValue(INDICATOR_DEPTH_VALUE + 1)
        label_item = gl.GLTextItem(
            pos=geometry.label_position,
            color=(
                INDICATOR_SELECTED_LABEL_COLOR
                if is_selected
                else INDICATOR_LABEL_COLOR
            ),
            text=_build_percentage_label(
                camera_id,
                percentages[camera_index],
            ),
            font=label_font,
            alignment=(
                Qt.AlignmentFlag.AlignHCenter
                | Qt.AlignmentFlag.AlignBottom
            ),
            glOptions="translucent",
        )
        label_item.setDepthValue(INDICATOR_LABEL_DEPTH_VALUE)
        items[camera_id] = (
            line_item,
            track_item,
            fill_item,
            label_item,
        )
    return items


def update_projection_camera_indicator_items(
    items: Mapping[str, Sequence[GLGraphicsItem]],
    geometries: Mapping[str, ProjectionCameraIndicatorGeometry],
    camera_percentages: Sequence[int] | Mapping[str, int],
    *,
    selected_camera_id: str | None = None,
) -> None:
    """Update allocation bars, exact labels, and selection highlighting."""

    percentages = normalize_projection_camera_indicator_percentages(
        camera_percentages
    )
    selected_id = _normalize_selected_camera_id(selected_camera_id)
    if tuple(items) != ALL_CAMERA_IDS or tuple(geometries) != ALL_CAMERA_IDS:
        raise ValueError("Projection camera indicators require all six cameras.")
    for camera_index, camera_id in enumerate(ALL_CAMERA_IDS):
        camera_items = tuple(items[camera_id])
        if len(camera_items) != 4:
            raise ValueError(
                "Projection camera indicators require outline, track, fill, "
                "and label items."
            )
        line_item, track_item, fill_item, label_item = camera_items
        if not (
            isinstance(line_item, gl.GLLinePlotItem)
            and isinstance(track_item, gl.GLLinePlotItem)
            and isinstance(fill_item, gl.GLLinePlotItem)
            and isinstance(label_item, gl.GLTextItem)
        ):
            raise TypeError("Projection camera indicator items have invalid types.")
        is_selected = camera_id == selected_id
        line_item.setData(
            color=(INDICATOR_SELECTED_COLOR if is_selected else INDICATOR_COLOR),
            width=(
                INDICATOR_SELECTED_LINE_WIDTH
                if is_selected
                else INDICATOR_LINE_WIDTH
            ),
        )
        track_item.setData(
            color=(
                INDICATOR_BAR_TRACK_SELECTED_COLOR
                if is_selected
                else INDICATOR_BAR_TRACK_COLOR
            )
        )
        fill_item.setData(
            pos=_build_percentage_bar_fill_positions(
                geometries[camera_id],
                percentages[camera_index],
            ),
            color=(
                INDICATOR_BAR_FILL_SELECTED_COLOR
                if is_selected
                else INDICATOR_BAR_FILL_COLOR
            ),
        )
        label_item.setData(
            color=(
                INDICATOR_SELECTED_LABEL_COLOR
                if is_selected
                else INDICATOR_LABEL_COLOR
            ),
            text=_build_percentage_label(
                camera_id,
                percentages[camera_index],
            ),
        )


# ### Geometry helpers ###
def _build_camera_indicator_geometry(
    *,
    camera_id: str,
    center: np.ndarray,
    half_extents: np.ndarray,
    scale: float,
) -> ProjectionCameraIndicatorGeometry:
    axes = _CAMERA_AXES[camera_id]
    outward = np.asarray(axes.outward, dtype=float)
    right = np.asarray(axes.right, dtype=float)
    up = np.asarray(axes.up, dtype=float)
    forward = -outward
    depth_axis = int(np.argmax(np.abs(outward)))
    half_depth = float(half_extents[depth_axis])
    camera_position = center + outward * (
        half_depth + scale * CAMERA_CLEARANCE_RATIO
    )

    body_half_width = scale * CAMERA_BODY_HALF_WIDTH_RATIO
    body_half_height = scale * CAMERA_BODY_HALF_HEIGHT_RATIO
    body_half_depth = scale * CAMERA_BODY_HALF_DEPTH_RATIO
    label_position = camera_position + up * (
        body_half_height + scale * CAMERA_LABEL_OFFSET_RATIO
    )
    front_center = camera_position + forward * body_half_depth
    back_center = camera_position - forward * body_half_depth
    front_corners = _rectangle_corners(
        front_center,
        right,
        up,
        body_half_width,
        body_half_height,
    )
    back_corners = _rectangle_corners(
        back_center,
        right,
        up,
        body_half_width,
        body_half_height,
    )
    lens_tip = front_center + forward * (scale * CAMERA_LENS_LENGTH_RATIO)
    aim_endpoint = center + outward * (
        half_depth + scale * CAMERA_SURFACE_CLEARANCE_RATIO
    )

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    _append_rectangle_segments(segments, front_corners)
    _append_rectangle_segments(segments, back_corners)
    segments.extend(zip(front_corners, back_corners))
    segments.extend((corner, lens_tip) for corner in front_corners)
    selection_segments = tuple(segments)
    segments.append((lens_tip, aim_endpoint))

    arrow_base = aim_endpoint - forward * (scale * CAMERA_ARROW_LENGTH_RATIO)
    arrow_width = scale * CAMERA_ARROW_WIDTH_RATIO
    segments.extend(
        (
            (aim_endpoint, arrow_base + right * arrow_width),
            (aim_endpoint, arrow_base - right * arrow_width),
            (aim_endpoint, arrow_base + up * arrow_width),
            (aim_endpoint, arrow_base - up * arrow_width),
        )
    )
    percentage_bar_center = camera_position + right * (
        body_half_width + scale * CAMERA_PERCENTAGE_BAR_OFFSET_RATIO
    )
    percentage_bar_half_height = (
        scale * CAMERA_PERCENTAGE_BAR_HALF_HEIGHT_RATIO
    )
    percentage_bar_start = (
        percentage_bar_center - up * percentage_bar_half_height
    )
    percentage_bar_end = percentage_bar_center + up * percentage_bar_half_height
    selection_segments = (
        *selection_segments,
        (percentage_bar_start, percentage_bar_end),
    )
    return ProjectionCameraIndicatorGeometry(
        camera_id=camera_id,
        camera_position=np.ascontiguousarray(camera_position, dtype=np.float32),
        label_position=np.ascontiguousarray(label_position, dtype=np.float32),
        aim_endpoint=np.ascontiguousarray(aim_endpoint, dtype=np.float32),
        line_positions=np.ascontiguousarray(
            np.asarray(segments, dtype=np.float32).reshape(-1, 3)
        ),
        selection_line_positions=np.ascontiguousarray(
            np.asarray(selection_segments, dtype=np.float32).reshape(-1, 3)
        ),
        percentage_bar_start=np.ascontiguousarray(
            percentage_bar_start,
            dtype=np.float32,
        ),
        percentage_bar_end=np.ascontiguousarray(
            percentage_bar_end,
            dtype=np.float32,
        ),
    )


# ### Percentage helpers ###
def normalize_projection_camera_indicator_percentages(
    values: Sequence[int] | Mapping[str, int],
) -> tuple[int, ...]:
    """Return six safe UI percentages in canonical camera order."""

    if isinstance(values, Mapping):
        if set(values) != set(ALL_CAMERA_IDS):
            raise ValueError(
                "Projection camera percentages require every canonical camera."
            )
        raw_values = tuple(values[camera_id] for camera_id in ALL_CAMERA_IDS)
    elif isinstance(values, str | bytes | bytearray):
        raise ValueError("Projection camera percentages must be a sequence.")
    else:
        try:
            raw_values = tuple(values)
        except TypeError as error:
            raise ValueError(
                "Projection camera percentages must be a sequence."
            ) from error
    if len(raw_values) != len(ALL_CAMERA_IDS):
        raise ValueError("Projection camera percentages require six values.")
    normalized: list[int] = []
    for value in raw_values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Projection camera percentages must be integers.")
        if not MINIMUM_CAMERA_PERCENTAGE <= value <= MAXIMUM_CAMERA_PERCENTAGE:
            raise ValueError(
                "Projection camera percentages must be between 1 and 95."
            )
        normalized.append(int(value))
    if sum(normalized) > MAXIMUM_TOTAL_CAMERA_PERCENTAGE:
        raise ValueError(
            "Projection camera percentages cannot total more than 100%."
        )
    return tuple(normalized)


def _normalize_selected_camera_id(camera_id: str | None) -> str | None:
    if camera_id is None:
        return None
    normalized = str(camera_id).strip()
    if normalized not in ALL_CAMERA_IDS:
        raise ValueError("Unknown projection camera ID.")
    return normalized


def _build_percentage_bar_fill_positions(
    geometry: ProjectionCameraIndicatorGeometry,
    percentage: int,
) -> np.ndarray:
    fraction = float(percentage) / 100.0
    fill_end = geometry.percentage_bar_start + (
        geometry.percentage_bar_end - geometry.percentage_bar_start
    ) * fraction
    return np.ascontiguousarray(
        np.asarray((geometry.percentage_bar_start, fill_end), dtype=np.float32)
    )


def _build_percentage_label(camera_id: str, percentage: int) -> str:
    return f"{PROJECTION_CAMERA_INDICATOR_LABELS[camera_id]} {percentage}%"


# ### Geometry helpers ###
def _rectangle_corners(
    center: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    half_width: float,
    half_height: float,
) -> tuple[np.ndarray, ...]:
    return (
        center - right * half_width - up * half_height,
        center + right * half_width - up * half_height,
        center + right * half_width + up * half_height,
        center - right * half_width + up * half_height,
    )


def _append_rectangle_segments(
    segments: list[tuple[np.ndarray, np.ndarray]],
    corners: tuple[np.ndarray, ...],
) -> None:
    segments.extend(
        (corners[index], corners[(index + 1) % len(corners)])
        for index in range(len(corners))
    )


def _normalize_bounds(bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized_bounds = np.asarray(bounds, dtype=float)
    if (
        normalized_bounds.shape != (2, 3)
        or not np.all(np.isfinite(normalized_bounds))
        or np.any(normalized_bounds[1] < normalized_bounds[0])
    ):
        raise ValueError("Camera indicator bounds must be two finite 3D corners.")
    return normalized_bounds[0], normalized_bounds[1]
