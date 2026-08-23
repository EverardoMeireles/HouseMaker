# ### Imports ###
from __future__ import annotations

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
CAMERA_INDICATOR_LABELS = {
    CAMERA_ID_POS_X: "+X",
    CAMERA_ID_NEG_X: "-X",
    CAMERA_ID_POS_Y: "+Y",
    CAMERA_ID_NEG_Y: "-Y",
    CAMERA_ID_TOP: "Top",
    CAMERA_ID_BOTTOM: "Bottom",
}


# ### Data models ###
@dataclass(frozen=True)
class CameraIndicatorGeometry:
    """Line geometry for one illustrative camera aimed at a model."""

    camera_id: str
    camera_position: np.ndarray
    label_position: np.ndarray
    aim_endpoint: np.ndarray
    line_positions: np.ndarray


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
def build_unused_face_camera_indicator_geometries(
    bounds: np.ndarray,
) -> dict[str, CameraIndicatorGeometry]:
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


def create_unused_face_camera_indicator_items(
    bounds: np.ndarray,
) -> dict[str, tuple[GLGraphicsItem, ...]]:
    """Create non-interactive camera outlines and labels for all six axes."""

    items: dict[str, tuple[GLGraphicsItem, ...]] = {}
    label_font = QFont("Helvetica", INDICATOR_LABEL_FONT_POINT_SIZE)
    label_font.setBold(True)
    for camera_id, geometry in (
        build_unused_face_camera_indicator_geometries(bounds).items()
    ):
        line_item = gl.GLLinePlotItem(
            pos=geometry.line_positions,
            color=INDICATOR_COLOR,
            width=INDICATOR_LINE_WIDTH,
            antialias=True,
            mode="lines",
            glOptions="translucent",
        )
        line_item.setDepthValue(INDICATOR_DEPTH_VALUE)
        label_item = gl.GLTextItem(
            pos=geometry.label_position,
            color=INDICATOR_LABEL_COLOR,
            text=CAMERA_INDICATOR_LABELS[camera_id],
            font=label_font,
            alignment=(
                Qt.AlignmentFlag.AlignHCenter
                | Qt.AlignmentFlag.AlignBottom
            ),
            glOptions="translucent",
        )
        label_item.setDepthValue(INDICATOR_LABEL_DEPTH_VALUE)
        items[camera_id] = (line_item, label_item)
    return items


def normalize_unused_face_camera_ids(
    camera_ids: object,
) -> tuple[str, ...]:
    """Return unique known camera IDs in the canonical axis order."""

    if isinstance(camera_ids, str):
        values = (camera_ids,)
    else:
        try:
            values = tuple(camera_ids)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("Unused-face camera IDs must be iterable.") from error
    unknown_ids = [
        camera_id
        for camera_id in values
        if not isinstance(camera_id, str) or camera_id not in ALL_CAMERA_IDS
    ]
    if unknown_ids:
        raise ValueError(
            "Unknown unused-face camera IDs: "
            + ", ".join(sorted(repr(camera_id) for camera_id in unknown_ids))
        )
    selected_ids = set(values)
    return tuple(
        camera_id for camera_id in ALL_CAMERA_IDS if camera_id in selected_ids
    )


# ### Geometry helpers ###
def _build_camera_indicator_geometry(
    *,
    camera_id: str,
    center: np.ndarray,
    half_extents: np.ndarray,
    scale: float,
) -> CameraIndicatorGeometry:
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
    return CameraIndicatorGeometry(
        camera_id=camera_id,
        camera_position=np.ascontiguousarray(camera_position, dtype=np.float32),
        label_position=np.ascontiguousarray(label_position, dtype=np.float32),
        aim_endpoint=np.ascontiguousarray(aim_endpoint, dtype=np.float32),
        line_positions=np.ascontiguousarray(
            np.asarray(segments, dtype=np.float32).reshape(-1, 3)
        ),
    )


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
