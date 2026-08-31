# ### Imports ###
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from io import BytesIO
import math
import threading
import weakref

import numpy as np
import trimesh
from OpenGL import GL
from OpenGL.GL import shaders as opengl_shaders
from pyqtgraph import Transform3D
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import shaders as gl_shaders
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from PySide6.QtCore import QPoint, QPointF, QRect, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QCursor, QKeyEvent, QMouseEvent, QVector3D
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRubberBand,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)
from PIL import Image

from housemaker.camera_indicators import (
    DEFAULT_CAMERA_INDICATOR_PERCENTAGES,
    ProjectionCameraIndicatorGeometry,
    build_projection_camera_indicator_geometries,
    create_projection_camera_indicator_items,
    normalize_projection_camera_indicator_percentages,
    update_projection_camera_indicator_items,
)
from housemaker.camera_models import CameraPose
from housemaker.glb import (
    SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION,
    GeneratedModel,
    PreviewPlacedObject,
    PreviewTexturedWall,
)
from housemaker.surface_geometry import (
    FixedSurface,
    SURFACE_TYPE_WALL,
    WallWindowPlacement,
    build_wall_window_placement,
    get_wall_window_world_corners,
)
from housemaker.unused_face_removal import ALL_CAMERA_IDS

# ### Constants ###
EDGE_COLOR = (0.12, 0.12, 0.16, 1.0)
FACE_COLOR = np.array([0.78, 0.80, 0.84, 1.0], dtype=float)
TEXTURE_PREVIEW_OFFSET_METERS = 0.01
CAMERA_STATE_KEYS = ("center", "distance", "elevation", "azimuth", "fov")
CLICK_SELECTION_TOLERANCE = 4.0
PROJECTION_CAMERA_SELECTION_TOLERANCE_PIXELS = 10.0
PROJECTION_CAMERA_LABEL_SELECTION_TOLERANCE_PIXELS = 16.0
MOUSE_WHEEL_DELTA_PER_STEP = 120
DEFAULT_AMBIENT_LIGHT_INTENSITY = 0.35
MIN_AMBIENT_LIGHT_INTENSITY = 0.0
MAX_AMBIENT_LIGHT_INTENSITY = 1.0
MAX_TEXTURE_PREVIEW_DIMENSION = 4096
DEFAULT_TEXTURES_ENABLED = True
DEFAULT_WIREFRAME_ENABLED = True
DEFAULT_WIREFRAME_ONLY = False
NAVIGATION_MODE_ORBIT = "orbit"
NAVIGATION_MODE_FIRST_PERSON = "first_person"
VIEWER_NAVIGATION_MODES = frozenset(
    (NAVIGATION_MODE_ORBIT, NAVIGATION_MODE_FIRST_PERSON)
)
DEFAULT_FIRST_PERSON_HEIGHT_METERS = 1.65
DEFAULT_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND = 2.5
DEFAULT_MOUSE_LOOK_SENSITIVITY_DEGREES = 0.16
FIRST_PERSON_UPDATE_INTERVAL_MILLISECONDS = 16
FIRST_PERSON_LOOK_DISTANCE_METERS = 1.0
MAX_FIRST_PERSON_PITCH_DEGREES = 89.0
WINDOW_EDITOR_PANEL_WIDTH = 190
WINDOW_PREVIEW_OFFSET_METERS = 0.006
WINDOW_SELECTION_COLOR = (0.20, 0.72, 1.0, 1.0)
WINDOW_VALID_PREVIEW_COLOR = (0.20, 0.86, 0.38, 0.34)
WINDOW_INVALID_PREVIEW_COLOR = (1.0, 0.24, 0.20, 0.34)
DOORWAY_PREVIEW_OUTLINE_COLOR = (1.0, 0.72, 0.18, 0.98)
DOORWAY_PREVIEW_OUTLINE_WIDTH = 3.0
SYMMETRIC_PREVIEW_MIN_OPACITY = 0.12
SYMMETRIC_PREVIEW_MAX_OPACITY = 0.72
SYMMETRIC_PREVIEW_FADE_PERIOD_MILLISECONDS = 2_000
SYMMETRIC_PREVIEW_UPDATE_INTERVAL_MILLISECONDS = 50
TRANSFORM_GIZMO_AXIS_COLORS = (
    (0.96, 0.22, 0.20, 1.0),
    (0.24, 0.86, 0.32, 1.0),
    (0.20, 0.48, 1.0, 1.0),
)
TRANSFORM_GIZMO_RING_POINT_COUNT = 72
TRANSFORM_GIZMO_SCREEN_SIZE_PIXELS = 92.0
TRANSFORM_GIZMO_MIN_SIZE_METERS = 0.35
TRANSFORM_GIZMO_AXIS_HIT_RATIO = 0.09
TRANSFORM_GIZMO_RING_RADIUS_RATIO = 0.72
TRANSFORM_GIZMO_RING_HIT_RATIO = 0.085
TRANSFORM_GIZMO_SELECTION_COLOR = (1.0, 0.72, 0.18, 0.95)
TRANSFORM_GIZMO_TRANSLATE = "translate"
TRANSFORM_GIZMO_ROTATE = "rotate"
FACE_SELECTION_COLOR = (1.0, 0.36, 0.08, 0.72)
FACE_SELECTION_EDGE_COLOR = (1.0, 0.78, 0.18, 1.0)
FACE_SELECTION_MAX_RASTER_DIMENSION = 768
FACE_SELECTION_REPLACE = "replace"
FACE_SELECTION_TOGGLE = "toggle"
FACE_SELECTION_ADD = "add"
FACE_SELECTION_UPDATE_MODES = frozenset(
    (FACE_SELECTION_REPLACE, FACE_SELECTION_TOGGLE, FACE_SELECTION_ADD)
)
# ### Shader source ###
AMBIENT_LIT_VERTEX_SHADER = """
    uniform mat4 u_mvp;
    uniform mat3 u_normal;
    attribute vec4 a_position;
    attribute vec3 a_normal;
    attribute vec4 a_color;
    varying vec4 v_color;
    varying vec3 v_normal;
    void main() {
        v_normal = normalize(u_normal * a_normal);
        v_color = a_color;
        gl_Position = u_mvp * a_position;
    }
"""
AMBIENT_LIT_FRAGMENT_SHADER = """
    #ifdef GL_ES
    precision mediump float;
    #endif
    uniform float u_ambient_light;
    varying vec4 v_color;
    varying vec3 v_normal;
    void main() {
        float diffuse = max(dot(v_normal, normalize(vec3(1.0, -1.0, -1.0))), 0.0);
        float illumination = min(1.0, u_ambient_light + diffuse * 0.65);
        gl_FragColor = vec4(v_color.rgb * illumination, v_color.a);
    }
"""
TEXTURED_AMBIENT_VERTEX_SHADER = """
    uniform mat4 u_mvp;
    uniform mat3 u_normal;
    attribute vec3 a_position;
    attribute vec3 a_normal;
    attribute vec2 a_texcoord;
    varying vec3 v_normal;
    varying vec2 v_texcoord;
    void main() {
        v_normal = normalize(u_normal * a_normal);
        v_texcoord = a_texcoord;
        gl_Position = u_mvp * vec4(a_position, 1.0);
    }
"""
TEXTURED_AMBIENT_FRAGMENT_SHADER = """
    #ifdef GL_ES
    precision mediump float;
    #endif
    uniform sampler2D u_texture;
    uniform sampler2D u_edit_mask;
    uniform float u_edit_mask_enabled;
    uniform float u_ambient_light;
    uniform float u_opacity;
    varying vec3 v_normal;
    varying vec2 v_texcoord;
    void main() {
        vec4 base_color = texture2D(u_texture, v_texcoord);
        float edit_amount = texture2D(u_edit_mask, v_texcoord).r
            * u_edit_mask_enabled * 0.58;
        base_color.rgb = mix(
            base_color.rgb,
            vec3(1.0, 0.49411765, 0.12549020),
            edit_amount
        );
        float diffuse = max(dot(v_normal, normalize(vec3(1.0, -1.0, -1.0))), 0.0);
        float illumination = min(1.0, u_ambient_light + diffuse * 0.65);
        gl_FragColor = vec4(
            base_color.rgb * illumination,
            base_color.a * u_opacity
        );
    }
"""


# ### Data models ###
@dataclass(frozen=True)
class TextureMeshData:
    """Face-expanded geometry and one embedded base-color texture."""

    vertices: np.ndarray
    normals: np.ndarray
    texture_coordinates: np.ndarray
    texture_rgba: np.ndarray


@dataclass(frozen=True)
class _TransformGizmoHandle:
    """One global-axis translation arrow or rotation ring."""

    kind: str
    axis_index: int

    def __post_init__(self) -> None:
        if self.kind not in {
            TRANSFORM_GIZMO_TRANSLATE,
            TRANSFORM_GIZMO_ROTATE,
        }:
            raise ValueError("Unknown placed-object gizmo handle kind.")
        if self.axis_index not in {0, 1, 2}:
            raise ValueError("Placed-object gizmo axes must be X, Y, or Z.")

# ### Widgets ###
class SelectableGLViewWidget(gl.GLViewWidget):
    """3D viewport with selectable items and two explicit navigation modes."""

    items_clicked = Signal(object)
    viewport_clicked = Signal(object)
    rectangle_pointer_pressed = Signal(object)
    rectangle_pointer_moved = Signal(object)
    rectangle_pointer_released = Signal(object)
    rectangle_drawing_cancel_requested = Signal()
    primary_pointer_pressed = Signal(object)
    primary_pointer_moved = Signal(object)
    primary_pointer_released = Signal(object)
    primary_pointer_cancel_requested = Signal()
    face_selection_pointer_pressed = Signal(object)
    face_selection_pointer_moved = Signal(object)
    face_selection_pointer_released = Signal(object)
    face_selection_pointer_cancel_requested = Signal()
    overlay_selection_requested = Signal(object)
    overlay_wheel_steps_requested = Signal(int)
    delete_requested = Signal()
    navigation_mode_changed = Signal(str)
    first_person_active_changed = Signal(bool)
    first_person_camera_pose_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.click_press_position = QPointF()
        self._is_middle_navigation_active = False
        self._rectangle_drawing_enabled = False
        self._primary_pointer_drag_reserved = False
        self._item_click_selection_enabled = True
        self._overlay_selection_enabled = False
        self._overlay_wheel_steps_enabled = False
        self._overlay_wheel_delta_remainder = 0
        self._face_selection_gestures_enabled = False
        self._face_selection_gesture_active = False
        self._face_selection_release_suppressed = False
        self._navigation_mode = NAVIGATION_MODE_ORBIT
        self._first_person_pointer_captured = False
        self._first_person_camera_pose = CameraPose(
            z=DEFAULT_FIRST_PERSON_HEIGHT_METERS
        )
        self._has_custom_first_person_camera_pose = False
        self._orbit_camera_state = self._capture_camera_state()
        self._pressed_movement_keys: set[int] = set()
        self._ignore_center_mouse_move = False
        self._mouse_look_sensitivity = DEFAULT_MOUSE_LOOK_SENSITIVITY_DEGREES
        self._move_speed = DEFAULT_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND
        self._movement_timer = QTimer(self)
        self._movement_timer.setInterval(FIRST_PERSON_UPDATE_INTERVAL_MILLISECONDS)
        self._movement_timer.timeout.connect(self._advance_from_pressed_keys)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._update_navigation_tooltip()

    @property
    def is_rectangle_drawing_enabled(self) -> bool:
        """Whether primary-pointer input is reserved for rectangle drawing."""

        return self._rectangle_drawing_enabled

    def set_rectangle_drawing_enabled(self, enabled: bool) -> None:
        """Reserve or release primary-pointer input for a transient tool."""

        normalized_enabled = bool(enabled)
        if normalized_enabled == self._rectangle_drawing_enabled:
            return
        if normalized_enabled:
            self.release_first_person_pointer_capture()
        self._rectangle_drawing_enabled = normalized_enabled
        if normalized_enabled:
            self.focus_navigation()

    @property
    def is_primary_pointer_drag_reserved(self) -> bool:
        """Whether a Canvas gizmo currently owns primary-pointer input."""

        return self._primary_pointer_drag_reserved

    def reserve_primary_pointer_drag(self) -> None:
        """Prevent navigation and click selection until the drag finishes."""

        self._primary_pointer_drag_reserved = True

    def release_primary_pointer_drag(self) -> None:
        """Return primary-pointer input to ordinary selection."""

        self._primary_pointer_drag_reserved = False

    def set_item_click_selection_enabled(self, enabled: bool) -> None:
        """Enable the legacy item-pick pass only for viewers that use it."""

        self._item_click_selection_enabled = bool(enabled)

    def set_overlay_selection_enabled(self, enabled: bool) -> None:
        """Enable safe click requests for CPU-picked viewport overlays."""

        self._overlay_selection_enabled = bool(enabled)

    def set_overlay_wheel_steps_enabled(self, enabled: bool) -> None:
        """Route wheel ticks to a selected overlay instead of camera zoom."""

        normalized_enabled = bool(enabled)
        if normalized_enabled == self._overlay_wheel_steps_enabled:
            return
        self._overlay_wheel_steps_enabled = normalized_enabled
        self._overlay_wheel_delta_remainder = 0

    @property
    def is_face_selection_gesture_active(self) -> bool:
        """Whether Ctrl+left selection currently owns pointer movement."""

        return self._face_selection_gesture_active

    @property
    def is_middle_navigation_active(self) -> bool:
        """Whether an orbit/pan gesture currently owns the middle button."""

        return self._is_middle_navigation_active

    def cancel_face_selection_gesture(self) -> None:
        """Release a pending Ctrl+left gesture after a context change."""

        self._cancel_face_selection_gesture()

    def cancel_transient_pointer_interactions(self) -> None:
        """Release selection/navigation ownership that cannot cross contexts."""

        self._cancel_face_selection_gesture()
        self._is_middle_navigation_active = False
        if QWidget.mouseGrabber() is self:
            self.releaseMouse()

    def set_face_selection_gestures_enabled(self, enabled: bool) -> None:
        """Enable Ctrl+primary-pointer gestures for object face editing."""

        self._face_selection_gestures_enabled = bool(enabled)
        if not self._face_selection_gestures_enabled:
            self._cancel_face_selection_gesture()

    @property
    def navigation_mode(self) -> str:
        """Return ``orbit`` or ``first_person`` for settings/UI synchronization."""

        return self._navigation_mode

    @property
    def is_first_person_active(self) -> bool:
        """Whether the viewport currently accepts first-person input."""

        return self._navigation_mode == NAVIGATION_MODE_FIRST_PERSON

    @property
    def is_first_person_pointer_captured(self) -> bool:
        """Whether mouse movement is currently captured for first-person look."""

        return bool(
            self.is_first_person_active
            and self._first_person_pointer_captured
        )

    @property
    def has_custom_first_person_camera_pose(self) -> bool:
        """Whether the first-person pose was supplied by the application/user."""

        return self._has_custom_first_person_camera_pose

    def focus_navigation(self) -> None:
        """Focus this viewport after it is moved into another window."""

        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        if self.is_first_person_pointer_captured and self.isVisible():
            self.grabMouse()
            self._center_pointer()

    def release_first_person_pointer_capture(self) -> None:
        """Release mouse-look without changing the first-person camera mode."""

        was_pointer_captured = self._first_person_pointer_captured
        self._pressed_movement_keys.clear()
        if not was_pointer_captured:
            return
        self._first_person_pointer_captured = False
        if self.isVisible():
            self.releaseMouse()
        self.unsetCursor()
        self._update_navigation_tooltip()

    def build_camera_ray(
        self,
        position: QPointF,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return one world-space ray through a viewport position."""

        width = max(float(self.width()), 1.0)
        height = max(float(self.height()), 1.0)
        normalized_x = 2.0 * float(position.x()) / width - 1.0
        normalized_y = 1.0 - 2.0 * float(position.y()) / height
        viewport = (0, 0, int(width), int(height))
        inverse, invertible = (
            self.projectionMatrix(viewport, viewport) * self.viewMatrix()
        ).inverted()
        if not invertible:
            return None
        near = inverse.map(QVector3D(normalized_x, normalized_y, -1.0))
        far = inverse.map(QVector3D(normalized_x, normalized_y, 1.0))
        origin = np.asarray((near.x(), near.y(), near.z()), dtype=float)
        direction = np.asarray(
            (far.x() - near.x(), far.y() - near.y(), far.z() - near.z()),
            dtype=float,
        )
        length = float(np.linalg.norm(direction))
        if not np.isfinite(length) or length <= 1e-12:
            return None
        return origin, direction / length

    def get_navigation_mode(self) -> str:
        """Return the active navigation mode."""

        return self.navigation_mode

    def set_navigation_mode(self, mode: str) -> None:
        """Switch explicitly between Blender orbit and captured first-person input."""

        normalized_mode = _normalize_navigation_mode(mode)
        if normalized_mode == self._navigation_mode:
            return
        if normalized_mode == NAVIGATION_MODE_FIRST_PERSON:
            self._enter_first_person_mode()
            return
        self._exit_first_person_mode()

    def toggle_navigation_mode(self) -> str:
        """Toggle modes and return the newly active mode for hotkey handlers."""

        next_mode = (
            NAVIGATION_MODE_FIRST_PERSON
            if self._navigation_mode == NAVIGATION_MODE_ORBIT
            else NAVIGATION_MODE_ORBIT
        )
        self.set_navigation_mode(next_mode)
        return self._navigation_mode

    def enter_first_person_mode(self) -> None:
        """Enter first-person navigation without requiring a click in the view."""

        self.set_navigation_mode(NAVIGATION_MODE_FIRST_PERSON)

    def exit_first_person_mode(self) -> None:
        """Return from first-person navigation to the saved orbit camera."""

        self.set_navigation_mode(NAVIGATION_MODE_ORBIT)

    def get_first_person_camera_pose(self) -> CameraPose:
        """Return the no-gravity first-person camera pose."""

        return self._first_person_camera_pose

    def set_first_person_camera_pose(self, pose: CameraPose) -> None:
        """Set a persistent first-person pose, for example from the Canvas camera."""

        self._set_first_person_camera_pose(pose, is_custom=True)

    def set_default_first_person_camera_pose(self, pose: CameraPose) -> None:
        """Set a model-derived pose only until an explicit pose has been supplied."""

        if self._has_custom_first_person_camera_pose:
            return
        self._set_first_person_camera_pose(pose, is_custom=False, emit_signal=False)

    def step_first_person_movement(self, elapsed_seconds: float) -> None:
        """Advance held Z/Q/S/D/R/F movement for deterministic callers/tests."""

        elapsed = max(0.0, float(elapsed_seconds))
        if (
            not self.is_first_person_active
            or elapsed <= 0.0
            or not self._pressed_movement_keys
        ):
            return
        forward_amount = float(
            (Qt.Key.Key_Z in self._pressed_movement_keys)
            - (Qt.Key.Key_S in self._pressed_movement_keys)
        )
        right_amount = float(
            (Qt.Key.Key_D in self._pressed_movement_keys)
            - (Qt.Key.Key_Q in self._pressed_movement_keys)
        )
        vertical_amount = float(
            (Qt.Key.Key_F in self._pressed_movement_keys)
            - (Qt.Key.Key_R in self._pressed_movement_keys)
        )
        if (
            forward_amount == 0.0
            and right_amount == 0.0
            and vertical_amount == 0.0
        ):
            return
        magnitude = math.sqrt(
            forward_amount**2 + right_amount**2 + vertical_amount**2
        )
        forward_amount /= magnitude
        right_amount /= magnitude
        vertical_amount /= magnitude
        yaw_radians = math.radians(self._first_person_camera_pose.yaw_degrees)
        forward_x = math.cos(yaw_radians)
        forward_y = math.sin(yaw_radians)
        # In this Z-up coordinate system, camera-right is forward cross up.
        # This makes French-layout Q move left and D move right.
        right_x = math.sin(yaw_radians)
        right_y = -math.cos(yaw_radians)
        distance = self._move_speed * elapsed
        pose = self._first_person_camera_pose
        self._set_first_person_camera_pose(
            CameraPose(
                x=pose.x
                + (forward_x * forward_amount + right_x * right_amount)
                * distance,
                y=pose.y
                + (forward_y * forward_amount + right_y * right_amount)
                * distance,
                z=pose.z + vertical_amount * distance,
                yaw_degrees=pose.yaw_degrees,
                pitch_degrees=pose.pitch_degrees,
                roll_degrees=pose.roll_degrees,
                fov_degrees=pose.fov_degrees,
            ),
            is_custom=True,
        )

    def step_movement(self, elapsed_seconds: float) -> None:
        """Compatibility shorthand for first-person movement stepping."""

        self.step_first_person_movement(elapsed_seconds)

    def remember_orbit_camera_state(self) -> None:
        """Record the current orbit pose before first-person rendering replaces it."""

        if self.is_first_person_active:
            return
        self._orbit_camera_state = self._capture_camera_state()

    def apply_navigation_camera(self) -> None:
        """Reapply the first-person camera after a scene refresh, when needed."""

        if self.is_first_person_active:
            self._sync_view_to_first_person_camera_pose()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._face_selection_release_suppressed = False
        if self._rectangle_drawing_enabled:
            if event.button() == Qt.MouseButton.LeftButton:
                self.click_press_position = event.position()
                self.rectangle_pointer_pressed.emit(event.position())
            elif event.button() == Qt.MouseButton.RightButton:
                self.rectangle_drawing_cancel_requested.emit()
            event.accept()
            return

        if (
            self._face_selection_gestures_enabled
            and event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and not self.is_first_person_pointer_captured
        ):
            self.click_press_position = event.position()
            self._face_selection_gesture_active = True
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self.face_selection_pointer_pressed.emit(event.position())
            event.accept()
            return

        if (
            (
                self._face_selection_gestures_enabled
                or not self._item_click_selection_enabled
                or self._overlay_selection_enabled
            )
            and event.button() == Qt.MouseButton.LeftButton
        ):
            # Plain clicks have no meaning in non-item-picking viewers such
            # as the object face editor.  Do not run GLViewWidget's legacy
            # OpenGL item-pick pass: custom textured items can leave that
            # selection render in a bad state and make later camera/model
            # updates appear frozen.
            self.click_press_position = event.position()
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            event.accept()
            return

        if (
            event.button() == Qt.MouseButton.RightButton
            and self._face_selection_gesture_active
        ):
            self._cancel_face_selection_gesture()
            event.accept()
            return

        if (
            event.button() == Qt.MouseButton.RightButton
            and self._primary_pointer_drag_reserved
        ):
            self.primary_pointer_cancel_requested.emit()
            event.accept()
            return

        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self.is_first_person_pointer_captured
        ):
            self.click_press_position = event.position()
            self.primary_pointer_pressed.emit(event.position())
            if self._primary_pointer_drag_reserved:
                event.accept()
                return

        if self.is_first_person_active:
            if event.button() == Qt.MouseButton.RightButton:
                self.release_first_person_pointer_capture()
            elif not self.is_first_person_pointer_captured:
                self.click_press_position = event.position()
                self.setFocus(Qt.FocusReason.MouseFocusReason)
            event.accept()
            return

        self.click_press_position = event.position()
        if event.button() == Qt.MouseButton.MiddleButton:
            self.mousePos = event.position()
            self._is_middle_navigation_active = True
            self.focus_navigation()
            if self.isVisible():
                self.grabMouse()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._rectangle_drawing_enabled:
            if event.button() == Qt.MouseButton.LeftButton:
                self.rectangle_pointer_released.emit(event.position())
            elif event.button() == Qt.MouseButton.RightButton:
                self.rectangle_drawing_cancel_requested.emit()
            event.accept()
            return

        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._face_selection_release_suppressed
        ):
            self._face_selection_release_suppressed = False
            event.accept()
            return

        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._face_selection_gesture_active
        ):
            self._face_selection_gesture_active = False
            self.face_selection_pointer_released.emit(event.position())
            event.accept()
            return

        if (
            (
                self._face_selection_gestures_enabled
                or not self._item_click_selection_enabled
                or self._overlay_selection_enabled
            )
            and event.button() == Qt.MouseButton.LeftButton
        ):
            # Match the inert press above without invoking itemsAt().  In the
            # face editor, 3D selection is deliberately Ctrl+click/drag only.
            if (
                self._overlay_selection_enabled
                and _get_point_distance(
                    self.click_press_position,
                    event.position(),
                )
                <= CLICK_SELECTION_TOLERANCE
            ):
                self.overlay_selection_requested.emit(event.position())
            event.accept()
            return

        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._primary_pointer_drag_reserved
        ):
            self.primary_pointer_released.emit(event.position())
            self._primary_pointer_drag_reserved = False
            event.accept()
            return

        if self.is_first_person_active:
            if (
                not self.is_first_person_pointer_captured
                and event.button() == Qt.MouseButton.LeftButton
                and _get_point_distance(
                    self.click_press_position,
                    event.position(),
                )
                <= CLICK_SELECTION_TOLERANCE
            ):
                clicked_items = self._get_clicked_items(event.position())
                if clicked_items:
                    self.items_clicked.emit(clicked_items)
                self.viewport_clicked.emit(event.position())
            event.accept()
            return

        if (
            event.button() == Qt.MouseButton.MiddleButton
            and self._is_middle_navigation_active
        ):
            self._is_middle_navigation_active = False
            if self.isVisible():
                self.releaseMouse()
            event.accept()
            return

        if (
            event.button() == Qt.MouseButton.LeftButton
            and _get_point_distance(self.click_press_position, event.position())
            <= CLICK_SELECTION_TOLERANCE
        ):
            clicked_items = self._get_clicked_items(event.position())
            if clicked_items:
                self.items_clicked.emit(clicked_items)
            self.viewport_clicked.emit(event.position())

        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        """Use Blender-style middle-button navigation without rotating on left drag."""

        if self._rectangle_drawing_enabled:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.rectangle_pointer_moved.emit(event.position())
            event.accept()
            return

        if self._face_selection_gesture_active:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.face_selection_pointer_moved.emit(event.position())
            event.accept()
            return

        if self._primary_pointer_drag_reserved:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.primary_pointer_moved.emit(event.position())
            event.accept()
            return

        if self.is_first_person_pointer_captured:
            self._handle_first_person_mouse_look(event)
            return

        if self.is_first_person_active:
            event.accept()
            return

        position = event.position()
        previous_position = getattr(self, "mousePos", position)
        delta = position - previous_position
        self.mousePos = position
        buttons = event.buttons()

        if buttons & Qt.MouseButton.MiddleButton:
            self._is_middle_navigation_active = True
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.pan(delta.x(), delta.y(), 0.0, relative="view")
            else:
                self.orbit(-delta.x(), delta.y())
            event.accept()
            return

        if buttons & Qt.MouseButton.LeftButton:
            # Left drag remains reserved for item selection and must not orbit.
            event.accept()
            return

        super().mouseMoveEvent(event)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        """Zoom with the wheel, including when a modifier key is held."""

        if self._overlay_wheel_steps_enabled:
            delta = event.angleDelta().y()
            if delta == 0:
                delta = event.angleDelta().x()
            self._overlay_wheel_delta_remainder += int(delta)
            wheel_steps = math.trunc(
                self._overlay_wheel_delta_remainder
                / MOUSE_WHEEL_DELTA_PER_STEP
            )
            if wheel_steps:
                self._overlay_wheel_delta_remainder -= (
                    wheel_steps * MOUSE_WHEEL_DELTA_PER_STEP
                )
                self.overlay_wheel_steps_requested.emit(wheel_steps)
            event.accept()
            return

        if self.is_first_person_active:
            event.accept()
            return

        delta = event.angleDelta().x()
        if delta == 0:
            delta = event.angleDelta().y()
        if delta != 0:
            self.opts["distance"] *= 0.999**delta
            self.update()
            event.accept()
            return

        super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Delete:
            self.delete_requested.emit()
            event.accept()
            return
        if (
            self._face_selection_gesture_active
            and event.key() == Qt.Key.Key_Escape
        ):
            self._cancel_face_selection_gesture()
            event.accept()
            return
        if (
            self._primary_pointer_drag_reserved
            and event.key() == Qt.Key.Key_Escape
        ):
            self.primary_pointer_cancel_requested.emit()
            event.accept()
            return
        if (
            self._rectangle_drawing_enabled
            and event.key() == Qt.Key.Key_Escape
        ):
            self.rectangle_drawing_cancel_requested.emit()
            event.accept()
            return
        if self.is_first_person_active and event.key() in _first_person_movement_keys():
            self._pressed_movement_keys.add(event.key())
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() in _first_person_movement_keys():
            self._pressed_movement_keys.discard(event.key())
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event) -> None:  # type: ignore[override]
        self.cancel_transient_pointer_interactions()
        if self._primary_pointer_drag_reserved:
            self.primary_pointer_cancel_requested.emit()
        self.release_first_person_pointer_capture()
        super().focusOutEvent(event)

    def hideEvent(self, event) -> None:  # type: ignore[override]
        self.cancel_transient_pointer_interactions()
        self.release_first_person_pointer_capture()
        super().hideEvent(event)

    def _cancel_face_selection_gesture(self) -> None:
        """Cancel one active face-selection gesture, if present."""

        if not self._face_selection_gesture_active:
            return
        self._face_selection_gesture_active = False
        self._face_selection_release_suppressed = True
        self.face_selection_pointer_cancel_requested.emit()

    def _enter_first_person_mode(self) -> None:
        self._orbit_camera_state = self._capture_camera_state()
        self._navigation_mode = NAVIGATION_MODE_FIRST_PERSON
        self._is_middle_navigation_active = False
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        self._sync_view_to_first_person_camera_pose()
        self._capture_first_person_pointer()
        self._movement_timer.start()
        self._update_navigation_tooltip()
        self.navigation_mode_changed.emit(self._navigation_mode)
        self.first_person_active_changed.emit(True)

    def _capture_first_person_pointer(self) -> None:
        """Capture the pointer for mouse-look while first-person mode is active."""

        if not self.is_first_person_active:
            return
        self._first_person_pointer_captured = True
        self.setCursor(Qt.CursorShape.BlankCursor)
        if self.isVisible():
            self.grabMouse()
            self._center_pointer()

    def _exit_first_person_mode(self) -> None:
        was_first_person_active = self.is_first_person_active
        if not was_first_person_active:
            return
        self.release_first_person_pointer_capture()
        self._navigation_mode = NAVIGATION_MODE_ORBIT
        self._pressed_movement_keys.clear()
        self._movement_timer.stop()
        self.opts.update(self._orbit_camera_state)
        self.update()
        self._update_navigation_tooltip()
        self.navigation_mode_changed.emit(self._navigation_mode)
        self.first_person_active_changed.emit(False)

    def _set_first_person_camera_pose(
        self,
        pose: CameraPose,
        *,
        is_custom: bool,
        emit_signal: bool = True,
    ) -> None:
        if not isinstance(pose, CameraPose):
            raise TypeError("The first-person camera pose must be a CameraPose.")
        self._first_person_camera_pose = CameraPose(
            x=pose.x,
            y=pose.y,
            z=pose.z,
            yaw_degrees=pose.yaw_degrees,
            pitch_degrees=min(
                max(pose.pitch_degrees, -MAX_FIRST_PERSON_PITCH_DEGREES),
                MAX_FIRST_PERSON_PITCH_DEGREES,
            ),
            roll_degrees=pose.roll_degrees,
            fov_degrees=pose.fov_degrees,
        )
        if is_custom:
            self._has_custom_first_person_camera_pose = True
        if self.is_first_person_active:
            self._sync_view_to_first_person_camera_pose()
        if emit_signal:
            self.first_person_camera_pose_changed.emit(self._first_person_camera_pose)

    def _handle_first_person_mouse_look(self, event: QMouseEvent) -> None:
        center = QPointF(self.rect().center())
        delta = event.position() - center
        if self._ignore_center_mouse_move and _point_is_near(delta, 0.5):
            self._ignore_center_mouse_move = False
            event.accept()
            return
        if not _point_is_near(delta, 0.0):
            pose = self._first_person_camera_pose
            self._set_first_person_camera_pose(
                CameraPose(
                    x=pose.x,
                    y=pose.y,
                    z=pose.z,
                    yaw_degrees=(
                        pose.yaw_degrees
                        - delta.x() * self._mouse_look_sensitivity
                    ),
                    pitch_degrees=(
                        pose.pitch_degrees
                        - delta.y() * self._mouse_look_sensitivity
                    ),
                    roll_degrees=pose.roll_degrees,
                    fov_degrees=pose.fov_degrees,
                ),
                is_custom=True,
            )
            if self.isVisible():
                self._center_pointer()
        event.accept()

    def _sync_view_to_first_person_camera_pose(self) -> None:
        pose = self._first_person_camera_pose
        yaw_radians = math.radians(pose.yaw_degrees)
        pitch_radians = math.radians(pose.pitch_degrees)
        forward = np.array(
            (
                math.cos(pitch_radians) * math.cos(yaw_radians),
                math.cos(pitch_radians) * math.sin(yaw_radians),
                math.sin(pitch_radians),
            ),
            dtype=float,
        )
        target = np.array((pose.x, pose.y, pose.z), dtype=float) + forward
        self.opts["center"] = QVector3D(*target.tolist())
        self.opts["distance"] = FIRST_PERSON_LOOK_DISTANCE_METERS
        self.opts["azimuth"] = pose.yaw_degrees + 180.0
        self.opts["elevation"] = -pose.pitch_degrees
        self.opts["fov"] = pose.fov_degrees
        self.update()

    def _advance_from_pressed_keys(self) -> None:
        self.step_first_person_movement(
            FIRST_PERSON_UPDATE_INTERVAL_MILLISECONDS / 1000.0
        )

    def _center_pointer(self) -> None:
        self._ignore_center_mouse_move = True
        QCursor.setPos(self.mapToGlobal(self.rect().center()))

    def _update_navigation_tooltip(self) -> None:
        if self.is_first_person_pointer_captured:
            self.setToolTip(
                "First-person controls: Z/Q/S/D to move, R/F to move down/up, "
                "move the mouse to look, right-click to release the pointer "
                "for selection."
            )
            return
        if self.is_first_person_active:
            self.setToolTip(
                "First-person view: click to select, Z/Q/S/D to move, and use "
                "the Canvas 3D navigation hotkey to return to orbit controls."
            )
            return
        self.setToolTip(
            "Blender controls: middle-drag to orbit, "
            "Shift+middle-drag to pan, mouse wheel to zoom."
        )

    def _capture_camera_state(self) -> dict[str, object]:
        camera_state: dict[str, object] = {}
        for key in CAMERA_STATE_KEYS:
            if key not in self.opts:
                continue
            value = self.opts[key]
            camera_state[key] = QVector3D(value) if isinstance(value, QVector3D) else value
        return camera_state

    def _get_clicked_items(self, position: QPointF) -> list[object]:
        region_size = int(CLICK_SELECTION_TOLERANCE * 2.0)
        region = (
            int(position.x() - CLICK_SELECTION_TOLERANCE),
            int(position.y() - CLICK_SELECTION_TOLERANCE),
            region_size,
            region_size,
        )
        try:
            return list(self.itemsAt(region))
        except Exception:
            return []


class TexturedMeshItem(GLGraphicsItem):
    """Draw a textured triangle mesh with a configurable ambient baseline."""

    def __init__(
        self,
        texture_mesh_data: TextureMeshData,
        ambient_light_intensity: float,
        *,
        texture_repeat: bool = False,
        double_sided: bool = True,
        opacity: float = 1.0,
        translucent: bool = False,
    ) -> None:
        super().__init__()
        self.setGLOptions("translucent" if translucent else "opaque")
        self._vertices = np.ascontiguousarray(
            texture_mesh_data.vertices,
            dtype=np.float32,
        )
        self._normals = np.ascontiguousarray(
            texture_mesh_data.normals,
            dtype=np.float32,
        )
        self._texture_coordinates = np.ascontiguousarray(
            texture_mesh_data.texture_coordinates,
            dtype=np.float32,
        )
        self._texture_rgba = np.ascontiguousarray(
            texture_mesh_data.texture_rgba,
            dtype=np.uint8,
        )
        self._edit_mask = np.zeros((1, 1), dtype=np.uint8)
        self._edit_mask_enabled = False
        self._ambient_light_intensity = _normalize_ambient_light_intensity(
            ambient_light_intensity
        )
        self._texture_repeat = bool(texture_repeat)
        self._double_sided = bool(double_sided)
        self._opacity = _normalize_preview_opacity(opacity)
        self._position_buffer: int | None = None
        self._normal_buffer: int | None = None
        self._texture_coordinate_buffer: int | None = None
        self._texture_id: int | None = None
        self._edit_mask_texture_id: int | None = None
        self._shader_program: int | None = None
        self._resources_uploaded = False
        self._edit_mask_dirty = False

    def set_ambient_light_intensity(self, intensity: float) -> None:
        self._ambient_light_intensity = _normalize_ambient_light_intensity(
            intensity
        )
        self.update()

    def set_opacity(self, opacity: float) -> None:
        """Update a preview item's opacity without rebuilding its buffers."""

        normalized = _normalize_preview_opacity(opacity)
        if normalized == self._opacity:
            return
        self._opacity = normalized
        self.update()

    def set_edit_mask(self, mask: np.ndarray | None) -> None:
        """Overlay the editable texels in orange without changing the model."""

        if mask is None:
            if not self._edit_mask_enabled:
                return
            self._edit_mask_enabled = False
            self.update()
            return
        raw_mask = np.asarray(mask)
        if raw_mask.ndim != 2 or raw_mask.size == 0:
            raise ValueError("A texture edit mask must be a non-empty image.")
        normalized_mask = np.ascontiguousarray(
            raw_mask > 0,
            dtype=np.uint8,
        ) * 255
        self._edit_mask_enabled = True
        if np.array_equal(normalized_mask, self._edit_mask):
            self.update()
            return
        self._edit_mask = normalized_mask
        self._edit_mask_dirty = True
        self.update()

    def initializeGL(self) -> None:
        self._resources_uploaded = False

    def paint(self) -> None:  # type: ignore[override]
        self.setupGLState()
        culling_was_enabled = bool(GL.glIsEnabled(GL.GL_CULL_FACE))
        if not self._double_sided:
            GL.glEnable(GL.GL_CULL_FACE)
            GL.glCullFace(GL.GL_BACK)
        self._ensure_gl_resources()
        if self._shader_program is None or self._texture_id is None:
            if not self._double_sided and not culling_was_enabled:
                GL.glDisable(GL.GL_CULL_FACE)
            return

        model_view_projection = np.asarray(
            self.mvpMatrix().data(),
            dtype=np.float32,
        )
        normal_matrix = np.asarray(
            self.modelViewMatrix().normalMatrix().data(),
            dtype=np.float32,
        )

        GL.glUseProgram(self._shader_program)
        _set_matrix_uniform(
            self._shader_program,
            "u_mvp",
            model_view_projection,
            4,
        )
        _set_matrix_uniform(
            self._shader_program,
            "u_normal",
            normal_matrix,
            3,
        )
        _set_float_uniform(
            self._shader_program,
            "u_ambient_light",
            self._ambient_light_intensity,
        )
        _set_float_uniform(
            self._shader_program,
            "u_opacity",
            self._opacity,
        )
        _set_integer_uniform(self._shader_program, "u_texture", 0)
        _set_integer_uniform(self._shader_program, "u_edit_mask", 1)
        _set_float_uniform(
            self._shader_program,
            "u_edit_mask_enabled",
            float(self._edit_mask_enabled),
        )

        enabled_locations: list[int] = []
        try:
            _bind_float_attribute(
                self._shader_program,
                "a_position",
                self._position_buffer,
                3,
                enabled_locations,
            )
            _bind_float_attribute(
                self._shader_program,
                "a_normal",
                self._normal_buffer,
                3,
                enabled_locations,
            )
            _bind_float_attribute(
                self._shader_program,
                "a_texcoord",
                self._texture_coordinate_buffer,
                2,
                enabled_locations,
            )
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_id)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._edit_mask_texture_id)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(self._vertices))
        finally:
            for location in enabled_locations:
                GL.glDisableVertexAttribArray(location)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glUseProgram(0)
            if not self._double_sided and not culling_was_enabled:
                GL.glDisable(GL.GL_CULL_FACE)

    def _ensure_gl_resources(self) -> None:
        if self._resources_uploaded:
            if (
                self._edit_mask_dirty
                and self._edit_mask_texture_id is not None
            ):
                _replace_uploaded_mask_texture(
                    self._edit_mask_texture_id,
                    self._edit_mask,
                )
                self._edit_mask_dirty = False
            return
        self._shader_program = opengl_shaders.compileProgram(
            opengl_shaders.compileShader(
                TEXTURED_AMBIENT_VERTEX_SHADER,
                GL.GL_VERTEX_SHADER,
            ),
            opengl_shaders.compileShader(
                TEXTURED_AMBIENT_FRAGMENT_SHADER,
                GL.GL_FRAGMENT_SHADER,
            ),
        )
        self._position_buffer = _upload_array_buffer(self._vertices)
        self._normal_buffer = _upload_array_buffer(self._normals)
        self._texture_coordinate_buffer = _upload_array_buffer(
            self._texture_coordinates
        )
        self._texture_id = _upload_texture(
            self._texture_rgba,
            repeat=self._texture_repeat,
        )
        self._edit_mask_texture_id = _upload_mask_texture(self._edit_mask)
        self._resources_uploaded = True
        self._edit_mask_dirty = False


class _WireframeOverlayMeshItem(gl.GLMeshItem):
    """Keep coplanar mesh edges visible over faces and texture previews."""

    def __init__(
        self,
        *args,
        cull_back_faces: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if cull_back_faces:
            # GLMeshItem calls setupGLState inside paint(), so culling must be
            # part of its configured state rather than enabled beforehand.
            self.setGLOptions(
                {
                    GL.GL_DEPTH_TEST: True,
                    GL.GL_BLEND: False,
                    GL.GL_CULL_FACE: True,
                    "glCullFace": (GL.GL_BACK,),
                }
            )

    def parseMeshData(self):
        """Prepare edge buffers even when the first frame hides wireframe."""

        draw_edges = self.opts["drawEdges"]
        self.opts["drawEdges"] = True
        try:
            return super().parseMeshData()
        finally:
            self.opts["drawEdges"] = draw_edges

    def paint(self) -> None:
        if not self.opts["drawEdges"]:
            super().paint()
            return

        previous_depth_function = int(GL.glGetIntegerv(GL.GL_DEPTH_FUNC))
        GL.glDepthFunc(GL.GL_LEQUAL)
        try:
            super().paint()
        finally:
            GL.glDepthFunc(previous_depth_function)


@dataclass
class _SymmetricPreviewRenderGroup:
    """The textured and fallback draw items for one mirrored retained mesh."""

    textured_item: TexturedMeshItem | None
    mesh_item: _WireframeOverlayMeshItem
    vertices: np.ndarray
    faces: np.ndarray
    face_colors: np.ndarray


@dataclass
class _PlacedObjectMeshRender:
    """Retained textured and fallback items for one local object mesh."""

    textured_item: TexturedMeshItem | None
    mesh_item: _WireframeOverlayMeshItem


@dataclass
class _PlacedObjectRenderGroup:
    """One independently transformable placed-object preview hierarchy."""

    preview: PreviewPlacedObject
    root_item: GLGraphicsItem
    retained_parts: list[_PlacedObjectMeshRender]
    symmetric_groups: list[_SymmetricPreviewRenderGroup]
    pick_meshes: tuple[object, ...]
    selection_item: gl.GLLinePlotItem
    current_transform: np.ndarray


@dataclass
class _PlacedObjectTransformDrag:
    """Stable geometry and accumulated state for one gizmo drag."""

    object_id: str
    handle: _TransformGizmoHandle
    start_world_position: np.ndarray
    start_rotation_degrees: tuple[float, float, float]
    start_transform: np.ndarray
    local_pivot: np.ndarray
    axis: np.ndarray
    drag_plane_normal: np.ndarray
    start_axis_parameter: float | None = None
    previous_rotation_vector: np.ndarray | None = None
    accumulated_rotation_degrees: float = 0.0
    preview_world_position: tuple[float, float, float] | None = None
    preview_rotation_degrees: tuple[float, float, float] | None = None


# ### Face-selection background models ###
@dataclass(frozen=True)
class _FaceRectangleSelectionTask:
    """Immutable projected input owned by one background raster pass."""

    request_revision: int
    geometry_revision: int
    projected_geometry: tuple[tuple[np.ndarray, np.ndarray], ...]
    rectangle: tuple[int, int, int, int]
    xray: bool


@dataclass(frozen=True)
class _FaceRectangleSelectionResult:
    """Logical face IDs produced for one exact request and geometry."""

    request_revision: int
    geometry_revision: int
    face_indices: frozenset[int]


class GlbViewerWidget(QWidget):
    """Generated-model viewer with Blender orbit and first-person navigation."""

    wall_selected = Signal(int, int, str)
    window_placement_requested = Signal(object)
    window_undo_requested = Signal()
    placed_object_transform_changed = Signal(str, object, object)
    face_selection_changed = Signal(object)
    projection_camera_selection_changed = Signal(object)
    projection_camera_percentage_step_requested = Signal(str, int)
    _face_rectangle_selection_completed = Signal(object)
    delete_requested = Signal()
    navigation_mode_changed = Signal(str)
    first_person_active_changed = Signal(bool)
    first_person_camera_pose_changed = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        textures_enabled: bool = DEFAULT_TEXTURES_ENABLED,
        wireframe_enabled: bool = DEFAULT_WIREFRAME_ENABLED,
        wireframe_only: bool = DEFAULT_WIREFRAME_ONLY,
        window_editing_enabled: bool = False,
        face_editing_enabled: bool = False,
    ) -> None:
        super().__init__(parent)
        self.model: GeneratedModel | None = None
        self.grid_item: gl.GLGridItem | None = None
        self.mesh_item: gl.GLMeshItem | None = None
        self.textured_mesh_item: TexturedMeshItem | None = None
        self.symmetric_preview_textured_mesh_item: TexturedMeshItem | None = None
        self.symmetric_preview_mesh_item: gl.GLMeshItem | None = None
        self.textured_surface_items: list[TexturedMeshItem] = []
        self.textured_wall_items: list[gl.GLImageItem] = []
        self.wall_by_item_id: dict[int, PreviewTexturedWall] = {}
        self.projection_camera_indicator_items: dict[
            str,
            tuple[GLGraphicsItem, ...],
        ] = {}
        self.projection_camera_indicator_geometries: dict[
            str,
            ProjectionCameraIndicatorGeometry,
        ] = {}
        self._projection_camera_percentages = (
            DEFAULT_CAMERA_INDICATOR_PERCENTAGES
        )
        self._selected_projection_camera_id: str | None = None
        self._doorway_preview_outline_positions: np.ndarray | None = None
        self._doorway_preview_outline_item: gl.GLLinePlotItem | None = None
        self._ambient_light_intensity = DEFAULT_AMBIENT_LIGHT_INTENSITY
        self._textures_enabled = bool(textures_enabled)
        self._wireframe_enabled = bool(wireframe_enabled)
        self._wireframe_only = bool(wireframe_only)
        self._window_editing_enabled = bool(window_editing_enabled)
        self._projection_camera_indicators_visible = False
        self._face_editing_enabled = bool(face_editing_enabled)
        self._face_selection_xray_enabled = False
        self._face_edit_vertices: np.ndarray | None = None
        self._face_edit_faces: np.ndarray | None = None
        self._selected_face_indices: set[int] = set()
        self._face_selection_press_position: QPointF | None = None
        self._face_selection_rubber_band: QRubberBand | None = None
        self._face_selection_item: gl.GLMeshItem | None = None
        self._mirrored_face_selection_item: gl.GLMeshItem | None = None
        self._face_selection_geometry_revision = 0
        self._face_rectangle_selection_request_revision = 0
        self._face_rectangle_selection_cancel_event: (
            threading.Event | None
        ) = None
        self._placed_object_editing_enabled = self._window_editing_enabled
        self._placed_object_render_groups: dict[
            str,
            _PlacedObjectRenderGroup,
        ] = {}
        self._selected_placed_object_id: str | None = None
        self._placed_object_transform_drag: (
            _PlacedObjectTransformDrag | None
        ) = None
        self._transform_gizmo_items: list[GLGraphicsItem] = []
        self._transform_gizmo_size = TRANSFORM_GIZMO_MIN_SIZE_METERS
        self.object_transform_status_label: QLabel | None = None
        self._window_wall_targets: dict[str, FixedSurface] = {}
        self._selected_window_wall_surface_id: str | None = None
        self._window_drag_first_world: tuple[float, float, float] | None = None
        self._window_preview_placement: WallWindowPlacement | None = None
        self._window_preview_is_valid = False
        self._window_undo_available = False
        self._window_selection_item: gl.GLLinePlotItem | None = None
        self._window_preview_item: gl.GLMeshItem | None = None
        self.window_tools_panel: QWidget | None = None
        self.window_tools_status_label: QLabel | None = None
        self.add_window_button: QPushButton | None = None
        self.undo_window_button: QPushButton | None = None
        self._texture_edit_mask: np.ndarray | None = None
        self._symmetric_preview_orientation: str | None = None
        self._symmetric_preview_plane_coordinate: float | None = None
        self._symmetric_preview_phase = 0.0
        self._symmetric_preview_vertices: np.ndarray | None = None
        self._symmetric_preview_faces: np.ndarray | None = None
        self._symmetric_preview_face_colors: np.ndarray | None = None
        self._explicit_symmetric_preview_group: (
            _SymmetricPreviewRenderGroup | None
        ) = None
        self._embedded_symmetric_preview_groups: list[
            _SymmetricPreviewRenderGroup
        ] = []
        self._last_set_model_preserved_camera = False
        self._symmetric_preview_timer = QTimer(self)
        self._symmetric_preview_timer.setInterval(
            SYMMETRIC_PREVIEW_UPDATE_INTERVAL_MILLISECONDS
        )
        self._symmetric_preview_timer.timeout.connect(
            self._advance_symmetric_preview_fade
        )
        self._ambient_shader = _build_ambient_shader(
            self._ambient_light_intensity
        )

        self._build_ui()
        if self._window_editing_enabled:
            self._connect_window_editor_input()
        if self._placed_object_editing_enabled:
            self._connect_placed_object_editor_input()
        self._connect_face_editor_input()
        self._populate_scene()

    # ### Doorway preview outline API ###
    def set_doorway_preview_outline(self, positions: object | None) -> None:
        """Set or explicitly clear one transient doorway wireframe outline."""

        if positions is None:
            self._doorway_preview_outline_positions = None
            self._remove_doorway_preview_outline_item()
            return

        self._doorway_preview_outline_positions = (
            _normalize_doorway_preview_outline_positions(positions)
        )
        self._refresh_doorway_preview_outline_item()

    # ### Viewer UI ###
    def _build_ui(self) -> None:
        if not self._window_editing_enabled:
            layout = QStackedLayout(self)
            self._populate_viewport_stack(layout)
            return

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        viewport_container = QWidget()
        viewport_container.setObjectName("canvas-window-editor-viewport")
        viewport_layout = QStackedLayout(viewport_container)
        self._populate_viewport_stack(viewport_layout)
        root_layout.addWidget(viewport_container, 1)
        root_layout.addWidget(self._build_window_tools_panel())

    def _populate_viewport_stack(self, layout: QStackedLayout) -> None:
        """Create the shared viewport and crosshair inside *layout*."""

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setStackingMode(QStackedLayout.StackingMode.StackAll)

        self.view = SelectableGLViewWidget()
        self.view.setBackgroundColor((24, 24, 28))
        self.view.set_item_click_selection_enabled(
            self._window_editing_enabled
        )
        self.view.items_clicked.connect(self._handle_view_items_clicked)
        self.view.delete_requested.connect(self.delete_requested.emit)
        self.view.navigation_mode_changed.connect(self.navigation_mode_changed.emit)
        self.view.first_person_active_changed.connect(
            self.first_person_active_changed.emit
        )
        self.view.first_person_camera_pose_changed.connect(
            self.first_person_camera_pose_changed.emit
        )
        self.view.set_face_selection_gestures_enabled(
            self._face_editing_enabled
        )
        self.view.overlay_selection_requested.connect(
            self._handle_projection_camera_selection_requested
        )
        self.view.overlay_wheel_steps_requested.connect(
            self._handle_projection_camera_wheel_steps_requested
        )
        layout.addWidget(self.view)

        self.first_person_crosshair_label = QLabel("+")
        self.first_person_crosshair_label.setObjectName(
            "generated_model_first_person_crosshair_label"
        )
        self.first_person_crosshair_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.first_person_crosshair_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
        )
        self.first_person_crosshair_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.first_person_crosshair_label.setStyleSheet(
            "color: rgba(255, 255, 255, 220); font-size: 22px; "
            "font-weight: 600; background: transparent;"
        )
        self.first_person_crosshair_label.hide()
        self.view.first_person_active_changed.connect(
            self.first_person_crosshair_label.setVisible
        )
        layout.addWidget(self.first_person_crosshair_label)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self._get_symmetric_preview_groups():
            self._symmetric_preview_timer.start()

    def hideEvent(self, event) -> None:  # type: ignore[override]
        self._symmetric_preview_timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Invalidate daemon work before this viewer closes or is destroyed."""

        self._invalidate_face_rectangle_selection_requests()
        super().closeEvent(event)

    def _build_window_tools_panel(self) -> QWidget:
        """Build the Canvas-only controls that travel with the 3D viewer."""

        panel = QWidget()
        panel.setObjectName("canvas-window-tools-panel")
        panel.setFixedWidth(WINDOW_EDITOR_PANEL_WIDTH)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(10)

        title_label = QLabel("Window tools")
        title_label.setObjectName("canvas-window-tools-title")
        panel_layout.addWidget(title_label)

        self.add_window_button = QPushButton("Add window")
        self.add_window_button.setObjectName("canvas-add-window-button")
        self.add_window_button.setCheckable(True)
        self.add_window_button.setEnabled(False)
        self.add_window_button.toggled.connect(
            self._handle_add_window_button_toggled
        )
        panel_layout.addWidget(self.add_window_button)

        self.undo_window_button = QPushButton("Undo window")
        self.undo_window_button.setObjectName("canvas-undo-window-button")
        self.undo_window_button.setEnabled(False)
        self.undo_window_button.clicked.connect(
            self._handle_undo_window_button_clicked
        )
        panel_layout.addWidget(self.undo_window_button)

        self.window_tools_status_label = QLabel("Select one wall.")
        self.window_tools_status_label.setObjectName(
            "canvas-window-tools-status"
        )
        self.window_tools_status_label.setWordWrap(True)
        panel_layout.addWidget(self.window_tools_status_label)

        object_title_label = QLabel("Object transforms")
        object_title_label.setObjectName("canvas-object-transform-title")
        panel_layout.addWidget(object_title_label)

        self.object_transform_status_label = QLabel(
            "Select a placed object to show its gizmo."
        )
        self.object_transform_status_label.setObjectName(
            "canvas-object-transform-status"
        )
        self.object_transform_status_label.setWordWrap(True)
        panel_layout.addWidget(self.object_transform_status_label)
        panel_layout.addStretch(1)
        self.window_tools_panel = panel
        return panel

    # ### Canvas window editor API ###
    @property
    def window_editing_enabled(self) -> bool:
        """Whether this viewer owns the Canvas wall-window tools."""

        return self._window_editing_enabled

    def set_wall_targets(self, surfaces: tuple[FixedSurface, ...]) -> None:
        """Replace the immutable semantic walls available for selection."""

        if not self._window_editing_enabled:
            return
        if not isinstance(surfaces, tuple):
            raise TypeError("Canvas wall targets must be supplied as a tuple.")

        targets: dict[str, FixedSurface] = {}
        for surface in surfaces:
            if not isinstance(surface, FixedSurface):
                raise TypeError("Canvas wall targets must be FixedSurface values.")
            if surface.surface_type != SURFACE_TYPE_WALL:
                continue
            if surface.surface_id in targets:
                raise ValueError(
                    f"Duplicate Canvas wall target: {surface.surface_id!r}."
                )
            targets[surface.surface_id] = surface

        selected_id = self._selected_window_wall_surface_id
        self._window_wall_targets = targets
        self._selected_window_wall_surface_id = (
            selected_id if selected_id in targets else None
        )
        self.cancel_window_placement(status_message=None)
        self._refresh_window_selection_outline()
        self._sync_window_tools_controls()

    def get_selected_wall_surface_id(self) -> str | None:
        """Return the one selected semantic wall, if any."""

        return self._selected_window_wall_surface_id

    def select_wall_target(self, surface_id: str | None) -> bool:
        """Select one semantic wall, or clear selection with ``None``."""

        if not self._window_editing_enabled:
            return False
        normalized_id = None if surface_id is None else str(surface_id).strip()
        if normalized_id is not None and normalized_id not in self._window_wall_targets:
            return False
        if normalized_id is not None:
            self._set_selected_placed_object(None)
        if normalized_id == self._selected_window_wall_surface_id:
            return False

        self.cancel_window_placement(status_message=None)
        self._selected_window_wall_surface_id = normalized_id
        self._refresh_window_selection_outline()
        self._sync_window_tools_controls()
        return True

    def is_window_placement_active(self) -> bool:
        """Whether pointer input is currently reserved for a new window."""

        return bool(
            self._window_editing_enabled
            and self.view.is_rectangle_drawing_enabled
        )

    def set_window_tools_status(self, message: str) -> None:
        """Show a commit result without exposing the panel's label internals."""

        if not isinstance(message, str):
            raise TypeError("A window-tools status message must be text.")
        self._set_window_tools_status(message)

    def set_window_undo_available(self, available: bool) -> None:
        """Enable undo only while Main owns a committed window history item."""

        self._window_undo_available = bool(available)
        self._sync_window_undo_button()

    # ### Placed-object transform API ###
    def get_selected_placed_object_id(self) -> str | None:
        """Return the stable ID currently controlled by the Canvas gizmo."""

        return self._selected_placed_object_id

    def select_placed_object(self, object_id: str | None) -> bool:
        """Select one rendered placed object, or clear the transform gizmo."""

        if not self._placed_object_editing_enabled:
            return False
        normalized_id = None if object_id is None else str(object_id).strip()
        if normalized_id is not None and normalized_id not in (
            self._placed_object_render_groups
        ):
            return False
        if normalized_id is not None:
            self.select_wall_target(None)
        return self._set_selected_placed_object(normalized_id)

    def _set_selected_placed_object(self, object_id: str | None) -> bool:
        if object_id == self._selected_placed_object_id:
            return False
        self._cancel_placed_object_gizmo_drag()
        self._selected_placed_object_id = object_id
        self._sync_placed_object_selection_rendering()
        return True

    def begin_window_placement(self) -> bool:
        """Arm rectangle drawing when exactly one wall is selected."""

        if (
            not self._window_editing_enabled
            or self._selected_window_wall_surface_id
            not in self._window_wall_targets
        ):
            self._set_window_tools_status("Select one wall.")
            self._set_add_window_button_checked(False)
            return False

        self._set_selected_placed_object(None)
        self._window_drag_first_world = None
        self._window_preview_placement = None
        self._window_preview_is_valid = False
        self._remove_window_preview_item()
        self.view.set_rectangle_drawing_enabled(True)
        self._set_add_window_button_checked(True)
        self._sync_window_undo_button()
        self._set_window_tools_status(
            "Drag on the selected wall. Escape or right-click cancels."
        )
        return True

    def cancel_window_placement(
        self,
        *,
        status_message: str | None = "Window placement cancelled.",
    ) -> None:
        """Safely clear every transient rectangle-drawing resource."""

        if not self._window_editing_enabled:
            return
        self.view.set_rectangle_drawing_enabled(False)
        self._window_drag_first_world = None
        self._window_preview_placement = None
        self._window_preview_is_valid = False
        self._remove_window_preview_item()
        self._set_add_window_button_checked(False)
        self._sync_window_undo_button()
        if status_message is not None:
            self._set_window_tools_status(status_message)

    # ### Canvas window editor input ###
    def _connect_window_editor_input(self) -> None:
        self.view.viewport_clicked.connect(
            self._handle_window_wall_pick_requested
        )
        self.view.rectangle_pointer_pressed.connect(
            self._handle_window_pointer_pressed
        )
        self.view.rectangle_pointer_moved.connect(
            self._handle_window_pointer_moved
        )
        self.view.rectangle_pointer_released.connect(
            self._handle_window_pointer_released
        )
        self.view.rectangle_drawing_cancel_requested.connect(
            self.cancel_window_placement
        )
        self.navigation_mode_changed.connect(
            self._handle_window_navigation_mode_changed
        )

    def _connect_placed_object_editor_input(self) -> None:
        """Connect Canvas-only pointer events used by transform gizmos."""

        self.view.primary_pointer_pressed.connect(
            self._handle_placed_object_pointer_pressed
        )
        self.view.primary_pointer_moved.connect(
            self._update_placed_object_gizmo_drag
        )
        self.view.primary_pointer_released.connect(
            self._finish_placed_object_gizmo_drag
        )
        self.view.primary_pointer_cancel_requested.connect(
            self._cancel_placed_object_gizmo_drag
        )
        self.navigation_mode_changed.connect(
            self._cancel_placed_object_gizmo_drag
        )

    def _handle_add_window_button_toggled(self, checked: bool) -> None:
        if checked:
            self.begin_window_placement()
            return
        if self.is_window_placement_active():
            self.cancel_window_placement()

    def _handle_undo_window_button_clicked(self) -> None:
        self.cancel_window_placement(status_message=None)
        self.window_undo_requested.emit()

    def _handle_window_navigation_mode_changed(self, _mode: str) -> None:
        if self.is_window_placement_active():
            self.cancel_window_placement()

    def _handle_window_wall_pick_requested(self, position: QPointF) -> None:
        if not self._window_editing_enabled or self.is_window_placement_active():
            return
        camera_ray = self.view.build_camera_ray(position)
        if camera_ray is None:
            self._set_selected_placed_object(None)
            self.select_wall_target(None)
            return
        ray_origin, ray_direction = camera_ray
        object_hit = _get_nearest_preview_placed_object_ray_hit(
            tuple(
                group.preview
                for group in self._placed_object_render_groups.values()
            ),
            ray_origin,
            ray_direction,
        )
        wall_hit = _get_nearest_fixed_surface_ray_hit(
            tuple(self._window_wall_targets.values()),
            ray_origin,
            ray_direction,
        )
        if object_hit is not None and (
            wall_hit is None or object_hit[2] <= wall_hit[2] + 1e-9
        ):
            self.select_placed_object(object_hit[0].object_id)
            return
        self._set_selected_placed_object(None)
        self.select_wall_target(
            None if wall_hit is None else wall_hit[0].surface_id
        )

    # ### Placed-object gizmo input ###
    def _handle_placed_object_pointer_pressed(self, position: QPointF) -> None:
        if (
            not self._placed_object_editing_enabled
            or self.is_window_placement_active()
            or self.view.is_first_person_pointer_captured
        ):
            return
        camera_ray = self.view.build_camera_ray(position)
        if camera_ray is None:
            return
        handle = self._pick_transform_gizmo_handle(*camera_ray)
        if handle is None:
            return
        self._begin_placed_object_gizmo_drag(handle, position)

    def _handle_window_pointer_pressed(self, position: QPointF) -> None:
        surface = self._get_selected_window_wall()
        camera_ray = self.view.build_camera_ray(position)
        if surface is None or camera_ray is None:
            self._set_window_tools_status("Start the drag on the selected wall.")
            return
        ray_origin, ray_direction = camera_ray
        hit = _get_nearest_fixed_surface_ray_hit(
            (surface,),
            ray_origin,
            ray_direction,
        )
        if hit is None:
            self._set_window_tools_status("Start the drag on the selected wall.")
            return
        self._window_drag_first_world = _world_point_tuple(hit[1])
        self._window_preview_placement = None
        self._window_preview_is_valid = False

    def _handle_window_pointer_moved(self, position: QPointF) -> None:
        self._update_window_drag_preview(position)

    def _handle_window_pointer_released(self, position: QPointF) -> None:
        if self._window_drag_first_world is None:
            self.cancel_window_placement(
                status_message="Start the drag on the selected wall."
            )
            return

        self._update_window_drag_preview(position)
        placement = self._window_preview_placement
        is_valid = self._window_preview_is_valid and placement is not None
        if not is_valid:
            message = (
                self.window_tools_status_label.text()
                if self.window_tools_status_label is not None
                else "Window placement is invalid."
            )
            self.cancel_window_placement(status_message=message)
            return

        self.cancel_window_placement(status_message="Window placement ready.")
        self.window_placement_requested.emit(placement)

    def _update_window_drag_preview(self, position: QPointF) -> None:
        surface = self._get_selected_window_wall()
        first_world = self._window_drag_first_world
        camera_ray = self.view.build_camera_ray(position)
        if surface is None or first_world is None:
            self._invalidate_window_preview(
                "Start the drag on the selected wall."
            )
            return
        if camera_ray is None:
            self._invalidate_window_preview(
                "Keep the pointer over the wall plane."
            )
            return
        ray_origin, ray_direction = camera_ray
        second_world = _intersect_ray_with_fixed_surface_plane(
            surface,
            ray_origin,
            ray_direction,
        )
        if second_world is None:
            self._invalidate_window_preview(
                "Keep the pointer over the wall plane."
            )
            return

        raw_corners = _build_wall_rectangle_corners(
            surface,
            first_world,
            second_world,
        )
        placement: object | None = None
        error_message: str | None = None
        try:
            placement = _build_validated_wall_window_placement(
                surface,
                first_world,
                second_world,
            )
            corners = _get_validated_wall_window_world_corners(
                surface,
                placement,
            )
        except (TypeError, ValueError) as error:
            corners = raw_corners
            error_message = str(error)

        self._window_preview_placement = placement
        self._window_preview_is_valid = placement is not None
        if corners is not None:
            self._set_window_preview_item(
                surface,
                corners,
                ray_origin,
                is_valid=self._window_preview_is_valid,
            )
        if self._window_preview_is_valid:
            self._set_window_tools_status("Release to add this window.")
        else:
            self._set_window_tools_status(
                error_message or "Window placement is invalid."
            )

    def _invalidate_window_preview(self, message: str) -> None:
        """Drop stale valid data whenever the pointer cannot be resolved."""

        self._window_preview_placement = None
        self._window_preview_is_valid = False
        self._remove_window_preview_item()
        self._set_window_tools_status(message)

    # ### Canvas window editor rendering ###
    def _get_selected_window_wall(self) -> FixedSurface | None:
        surface = self._window_wall_targets.get(
            self._selected_window_wall_surface_id or ""
        )
        return surface if isinstance(surface, FixedSurface) else None

    def _refresh_window_selection_outline(self) -> None:
        self._remove_window_selection_item()
        surface = self._get_selected_window_wall()
        if surface is None or self.model is None:
            return
        positions = _build_fixed_surface_boundary_line_positions(surface)
        if positions is None:
            return
        positions = _offset_points_toward_camera(
            positions,
            _get_fixed_surface_plane_normal(surface),
            self.view.cameraPosition(),
            WINDOW_PREVIEW_OFFSET_METERS,
        )
        self._window_selection_item = gl.GLLinePlotItem(
            pos=np.asarray(positions, dtype=float),
            color=WINDOW_SELECTION_COLOR,
            width=2.0,
            antialias=True,
            mode="lines",
        )
        self._window_selection_item.setGLOptions("translucent")
        self.view.addItem(self._window_selection_item)
        self.view.update()

    def _remove_window_selection_item(self) -> None:
        item = self._window_selection_item
        self._window_selection_item = None
        if item is not None and item in self.view.items:
            self.view.removeItem(item)

    def _set_window_preview_item(
        self,
        surface: FixedSurface,
        corners: object,
        camera_position: object,
        *,
        is_valid: bool,
    ) -> None:
        raw_corners = np.asarray(corners, dtype=float)
        if raw_corners.shape != (4, 3) or not np.isfinite(raw_corners).all():
            self._remove_window_preview_item()
            return
        normal = _get_fixed_surface_plane_normal(surface)
        offset_corners = _offset_points_toward_camera(
            raw_corners,
            normal,
            camera_position,
            WINDOW_PREVIEW_OFFSET_METERS * 1.5,
        )
        color = (
            WINDOW_VALID_PREVIEW_COLOR
            if is_valid
            else WINDOW_INVALID_PREVIEW_COLOR
        )
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
        face_colors = np.tile(np.asarray(color, dtype=float), (2, 1))
        self._remove_window_preview_item()
        self._window_preview_item = gl.GLMeshItem(
            vertexes=np.asarray(offset_corners, dtype=np.float32),
            faces=faces,
            faceColors=face_colors,
            smooth=False,
            drawFaces=True,
            drawEdges=True,
            edgeColor=(color[0], color[1], color[2], 0.95),
            glOptions="translucent",
        )
        self.view.addItem(self._window_preview_item)
        self.view.update()

    def _remove_window_preview_item(self) -> None:
        item = self._window_preview_item
        self._window_preview_item = None
        if item is not None and item in self.view.items:
            self.view.removeItem(item)
        if hasattr(self, "view"):
            self.view.update()

    # ### Doorway preview outline rendering ###
    def _refresh_doorway_preview_outline_item(self) -> None:
        positions = self._doorway_preview_outline_positions
        if positions is None:
            self._remove_doorway_preview_outline_item()
            return

        item = self._doorway_preview_outline_item
        if item is not None and item in self.view.items:
            item.setData(pos=positions)
            self.view.update()
            return

        self._remove_doorway_preview_outline_item()
        item = gl.GLLinePlotItem(
            pos=positions,
            color=DOORWAY_PREVIEW_OUTLINE_COLOR,
            width=DOORWAY_PREVIEW_OUTLINE_WIDTH,
            antialias=True,
            mode="lines",
        )
        item.setGLOptions("translucent")
        self._doorway_preview_outline_item = item
        self.view.addItem(item)
        self.view.update()

    def _remove_doorway_preview_outline_item(self) -> None:
        item = self._doorway_preview_outline_item
        self._doorway_preview_outline_item = None
        if item is not None and item in self.view.items:
            self.view.removeItem(item)
        if hasattr(self, "view"):
            self.view.update()

    # ### Canvas window editor controls ###
    def _sync_window_tools_controls(self) -> None:
        selected = self._get_selected_window_wall() is not None
        if self.add_window_button is not None:
            self.add_window_button.setEnabled(selected)
        self._sync_window_undo_button()
        if self.is_window_placement_active():
            return
        self._set_window_tools_status(
            "Wall selected. Click Add window."
            if selected
            else "Select one wall."
        )

    def _set_add_window_button_checked(self, checked: bool) -> None:
        if self.add_window_button is None:
            return
        was_blocked = self.add_window_button.blockSignals(True)
        self.add_window_button.setChecked(bool(checked))
        self.add_window_button.blockSignals(was_blocked)

    def _sync_window_undo_button(self) -> None:
        if self.undo_window_button is None:
            return
        self.undo_window_button.setEnabled(
            self._window_undo_available
            and not self.is_window_placement_active()
        )

    def _set_window_tools_status(self, message: str) -> None:
        if self.window_tools_status_label is not None:
            self.window_tools_status_label.setText(str(message))

    def focus_navigation(self) -> None:
        """Give the OpenGL viewport input focus after external reparenting."""

        self.view.focus_navigation()

    @property
    def navigation_mode(self) -> str:
        """Expose the active navigation mode without reaching into ``view``."""

        return self.view.navigation_mode

    @property
    def is_first_person_active(self) -> bool:
        """Whether the generated-model viewport is in first-person mode."""

        return self.view.is_first_person_active

    @property
    def is_first_person_pointer_captured(self) -> bool:
        """Whether first-person mouse-look currently owns the pointer."""

        return self.view.is_first_person_pointer_captured

    def release_first_person_pointer_capture(self) -> None:
        """Release mouse-look while preserving the first-person camera view."""

        self.view.release_first_person_pointer_capture()

    def get_navigation_mode(self) -> str:
        """Return the active ``orbit`` or ``first_person`` navigation mode."""

        return self.view.get_navigation_mode()

    def set_navigation_mode(self, mode: str) -> None:
        """Select the active navigation mode for the generated-model viewport."""

        self.view.set_navigation_mode(mode)

    def toggle_navigation_mode(self) -> str:
        """Toggle navigation mode and return the newly active mode."""

        return self.view.toggle_navigation_mode()

    def enter_first_person_mode(self) -> None:
        """Start captured Z/Q/S/D first-person navigation."""

        self.view.enter_first_person_mode()

    def exit_first_person_mode(self) -> None:
        """Return to the saved Blender orbit camera."""

        self.view.exit_first_person_mode()

    def get_first_person_camera_pose(self) -> CameraPose:
        """Return the stored no-gravity first-person camera pose."""

        return self.view.get_first_person_camera_pose()

    def set_first_person_camera_pose(self, pose: CameraPose) -> None:
        """Provide an explicit first-person pose, such as the Canvas camera."""

        self.view.set_first_person_camera_pose(pose)

    def set_model(self, model: GeneratedModel, preserve_camera: bool = False) -> None:
        self.view.cancel_transient_pointer_interactions()
        self._cancel_placed_object_gizmo_drag()
        self.clear_face_edit_geometry()
        if self._window_editing_enabled:
            self.cancel_window_placement(status_message=None)
        self._clear_symmetric_preview()
        self._last_set_model_preserved_camera = bool(preserve_camera)
        camera_state = self._capture_camera_state() if preserve_camera else None
        self._texture_edit_mask = None
        self.model = model
        self._populate_scene()
        if camera_state is not None:
            self._restore_camera_state(camera_state)
            self._refresh_window_selection_outline()
        if self._window_editing_enabled:
            self._sync_window_tools_controls()

    def clear_model(self) -> None:
        self.view.cancel_transient_pointer_interactions()
        self._cancel_placed_object_gizmo_drag()
        self.clear_face_edit_geometry()
        self._selected_placed_object_id = None
        if self._window_editing_enabled:
            self.cancel_window_placement(status_message=None)
        self._texture_edit_mask = None
        self._clear_symmetric_preview()
        self._last_set_model_preserved_camera = False
        self.model = None
        self._populate_scene()
        if self._window_editing_enabled:
            self._sync_window_tools_controls()

    # ### Face editor API ###
    def set_face_editing_enabled(self, enabled: bool) -> None:
        """Enable or disable Ctrl-based face-selection input."""

        normalized_enabled = bool(enabled)
        if normalized_enabled == self._face_editing_enabled:
            return
        self._face_editing_enabled = normalized_enabled
        self.view.set_face_selection_gestures_enabled(
            self._face_editing_enabled
        )
        if not self._face_editing_enabled:
            self._cancel_face_selection_gesture()

    def set_face_edit_geometry(
        self,
        vertices: object,
        faces: object,
    ) -> None:
        """Set the authoritative global face order used by editing/export."""

        normalized_vertices = np.asarray(vertices, dtype=np.float32)
        normalized_faces = np.asarray(faces, dtype=np.int64)
        if (
            normalized_vertices.ndim != 2
            or normalized_vertices.shape[1:] != (3,)
            or normalized_faces.ndim != 2
            or normalized_faces.shape[1:] != (3,)
            or not np.all(np.isfinite(normalized_vertices))
            or np.any(normalized_faces < 0)
            or (
                normalized_faces.size
                and np.any(normalized_faces >= len(normalized_vertices))
            )
        ):
            raise ValueError(
                "Face-edit geometry requires finite vertices and triangle faces."
            )
        self._face_edit_vertices = np.ascontiguousarray(
            normalized_vertices
        ).copy()
        self._face_edit_faces = np.ascontiguousarray(normalized_faces).copy()
        self._face_selection_geometry_revision += 1
        self.clear_face_selection()

    def clear_face_edit_geometry(self) -> None:
        """Forget editable geometry and clear every transient selection."""

        self.cancel_face_selection_interaction()
        self._face_edit_vertices = None
        self._face_edit_faces = None
        self._face_selection_geometry_revision += 1
        self.clear_face_selection()

    def cancel_face_selection_interaction(self) -> None:
        """Cancel both viewport and controller layers of one selection drag."""

        self.view.cancel_face_selection_gesture()
        self._cancel_face_selection_gesture()

    def cancel_transient_pointer_interactions(self) -> None:
        """Release orbit, pan, and face-selection pointer ownership."""

        self.view.cancel_transient_pointer_interactions()
        self._cancel_face_selection_gesture()

    def set_face_selection_xray_enabled(self, enabled: bool) -> None:
        """Select occluded faces during rectangle gestures when enabled."""

        normalized_enabled = bool(enabled)
        if normalized_enabled == self._face_selection_xray_enabled:
            return
        self._face_selection_xray_enabled = normalized_enabled
        self._invalidate_face_rectangle_selection_requests()

    def get_selected_face_indices(self) -> tuple[int, ...]:
        """Return stable global face indices in ascending order."""

        return tuple(sorted(self._selected_face_indices))

    def set_selected_face_indices(self, face_indices: object) -> None:
        """Set a validated selection, primarily for controller/tests."""

        self.update_face_selection(
            face_indices,
            mode=FACE_SELECTION_REPLACE,
        )

    def update_face_selection(
        self,
        face_indices: object,
        *,
        mode: str,
    ) -> None:
        """Apply every 2D, 3D, and programmatic selection through one path."""

        face_count = (
            0 if self._face_edit_faces is None else len(self._face_edit_faces)
        )
        try:
            normalized = {int(index) for index in face_indices}  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Selected face indices must be integers.") from error
        if any(index < 0 or index >= face_count for index in normalized):
            raise ValueError("A selected face index is outside the object.")
        normalized_mode = str(mode)
        if normalized_mode not in FACE_SELECTION_UPDATE_MODES:
            raise ValueError("Unknown face-selection update mode.")
        self._invalidate_face_rectangle_selection_requests()
        if normalized_mode == FACE_SELECTION_REPLACE:
            self._selected_face_indices = normalized
        elif normalized_mode == FACE_SELECTION_TOGGLE:
            self._selected_face_indices.symmetric_difference_update(normalized)
        else:
            self._selected_face_indices.update(normalized)
        self._refresh_face_selection_items()
        self.face_selection_changed.emit(self.get_selected_face_indices())

    def clear_face_selection(self) -> None:
        """Clear selected faces and hide the drag/selection overlays."""

        self._invalidate_face_rectangle_selection_requests()
        had_selection = bool(self._selected_face_indices)
        self._selected_face_indices.clear()
        self._face_selection_press_position = None
        self._hide_face_selection_rubber_band()
        self._remove_face_selection_items()
        if had_selection:
            self.face_selection_changed.emit(())

    @property
    def face_edit_face_count(self) -> int:
        """Return the number of authoritative editable triangles."""

        return 0 if self._face_edit_faces is None else len(self._face_edit_faces)

    # ### Face editor input ###
    def _connect_face_editor_input(self) -> None:
        """Connect opt-in pointer gestures without affecting other viewers."""

        self.view.face_selection_pointer_pressed.connect(
            self._handle_face_selection_pointer_pressed
        )
        self.view.face_selection_pointer_moved.connect(
            self._handle_face_selection_pointer_moved
        )
        self.view.face_selection_pointer_released.connect(
            self._handle_face_selection_pointer_released
        )
        self.view.face_selection_pointer_cancel_requested.connect(
            self._cancel_face_selection_gesture
        )
        self._face_rectangle_selection_completed.connect(
            self._apply_face_rectangle_selection_result,
            Qt.ConnectionType.QueuedConnection,
        )

    @Slot(object)
    def _handle_face_selection_pointer_pressed(self, position: object) -> None:
        self._invalidate_face_rectangle_selection_requests()
        if not self._can_select_faces():
            return
        self._face_selection_press_position = QPointF(position)

    @Slot(object)
    def _handle_face_selection_pointer_moved(self, position: object) -> None:
        start = self._face_selection_press_position
        if start is None or not self._can_select_faces():
            return
        current = QPointF(position)
        if _get_point_distance(start, current) <= CLICK_SELECTION_TOLERANCE:
            return
        rubber_band = self._ensure_face_selection_rubber_band()
        rubber_band.setGeometry(
            QRect(start.toPoint(), current.toPoint()).normalized()
        )
        rubber_band.show()

    @Slot(object)
    def _handle_face_selection_pointer_released(self, position: object) -> None:
        start = self._face_selection_press_position
        self._face_selection_press_position = None
        self._hide_face_selection_rubber_band()
        if start is None or not self._can_select_faces():
            return
        end = QPointF(position)
        if _get_point_distance(start, end) <= CLICK_SELECTION_TOLERANCE:
            face_index = self._pick_editable_face(end)
            if face_index is None:
                return
            self.update_face_selection(
                (face_index,),
                mode=FACE_SELECTION_TOGGLE,
            )
        else:
            self._start_editable_face_rectangle_selection(start, end)
            return

    @Slot()
    def _cancel_face_selection_gesture(self) -> None:
        self._invalidate_face_rectangle_selection_requests()
        self._face_selection_press_position = None
        self._hide_face_selection_rubber_band()

    def _can_select_faces(self) -> bool:
        return bool(
            self._face_editing_enabled
            and self.model is not None
            and self._face_edit_vertices is not None
            and self._face_edit_faces is not None
            and len(self._face_edit_faces)
        )

    def _pick_editable_face(self, position: QPointF) -> int | None:
        """Return the closest retained logical face on either preview half."""

        if self._face_edit_vertices is None or self._face_edit_faces is None:
            return None
        ray = self.view.build_camera_ray(position)
        if ray is None:
            return None
        ray_origin, ray_direction = ray
        hits: list[tuple[int, float]] = []
        retained_hit = _get_nearest_triangle_ray_face_index(
            self._face_edit_vertices,
            self._face_edit_faces,
            ray_origin,
            ray_direction,
        )
        if retained_hit is not None:
            hits.append(retained_hit)
        if (
            self._symmetric_preview_orientation is not None
            and self._symmetric_preview_plane_coordinate is not None
        ):
            mirrored_vertices = _mirror_preview_vertices(
                self._face_edit_vertices,
                self._symmetric_preview_orientation,
                self._symmetric_preview_plane_coordinate,
            )
            mirrored_hit = _get_nearest_triangle_ray_face_index(
                mirrored_vertices,
                self._face_edit_faces[:, (0, 2, 1)],
                ray_origin,
                ray_direction,
            )
            if mirrored_hit is not None:
                hits.append(mirrored_hit)
        if not hits:
            return None
        return min(hits, key=lambda item: item[1])[0]

    def _start_editable_face_rectangle_selection(
        self,
        start: QPointF,
        end: QPointF,
    ) -> bool:
        """Capture the camera and start one daemon raster task."""

        if self._face_edit_vertices is None or self._face_edit_faces is None:
            return False
        geometry = [(self._face_edit_vertices, self._face_edit_faces)]
        if (
            self._symmetric_preview_orientation is not None
            and self._symmetric_preview_plane_coordinate is not None
        ):
            geometry.append(
                (
                    _mirror_preview_vertices(
                        self._face_edit_vertices,
                        self._symmetric_preview_orientation,
                        self._symmetric_preview_plane_coordinate,
                    ),
                    self._face_edit_faces[:, (0, 2, 1)],
                )
            )
        captured = _capture_face_selection_raster_input(
            self.view,
            geometry,
            QRect(start.toPoint(), end.toPoint()).normalized(),
        )
        if captured is None:
            return False
        projected_geometry, rectangle = captured
        self._invalidate_face_rectangle_selection_requests()
        cancel_event = threading.Event()
        self._face_rectangle_selection_cancel_event = cancel_event
        task = _FaceRectangleSelectionTask(
            request_revision=(
                self._face_rectangle_selection_request_revision
            ),
            geometry_revision=self._face_selection_geometry_revision,
            projected_geometry=projected_geometry,
            rectangle=rectangle,
            xray=self._face_selection_xray_enabled,
        )
        viewer_reference = weakref.ref(self)
        worker = threading.Thread(
            target=_run_face_rectangle_selection_task,
            args=(task, cancel_event, viewer_reference),
            name=(
                "housemaker-face-selection-"
                f"{task.request_revision}"
            ),
            daemon=True,
        )
        worker.start()
        return True

    def _invalidate_face_rectangle_selection_requests(self) -> None:
        """Cancel delivery from the active raster task and advance its ID."""

        cancel_event = self._face_rectangle_selection_cancel_event
        if cancel_event is not None:
            cancel_event.set()
        self._face_rectangle_selection_cancel_event = None
        self._face_rectangle_selection_request_revision += 1

    @Slot(object)
    def _apply_face_rectangle_selection_result(self, raw_result: object) -> None:
        """Apply one current background result on the Qt GUI thread."""

        if not isinstance(raw_result, _FaceRectangleSelectionResult):
            return
        if (
            raw_result.request_revision
            != self._face_rectangle_selection_request_revision
            or raw_result.geometry_revision
            != self._face_selection_geometry_revision
            or not self._can_select_faces()
        ):
            return
        face_count = self.face_edit_face_count
        if any(
            face_index < 0 or face_index >= face_count
            for face_index in raw_result.face_indices
        ):
            return
        self._face_rectangle_selection_cancel_event = None
        self.update_face_selection(
            raw_result.face_indices,
            mode=FACE_SELECTION_ADD,
        )

    def _ensure_face_selection_rubber_band(self) -> QRubberBand:
        if self._face_selection_rubber_band is None:
            self._face_selection_rubber_band = QRubberBand(
                QRubberBand.Shape.Rectangle,
                self.view,
            )
        return self._face_selection_rubber_band

    def _hide_face_selection_rubber_band(self) -> None:
        if self._face_selection_rubber_band is not None:
            self._face_selection_rubber_band.hide()

    def _refresh_face_selection_items(self) -> None:
        self._remove_face_selection_items()
        if (
            not self._selected_face_indices
            or self._face_edit_vertices is None
            or self._face_edit_faces is None
            or not hasattr(self, "view")
        ):
            return
        selected = np.asarray(
            sorted(self._selected_face_indices),
            dtype=np.int64,
        )
        selected_triangles = self._face_edit_vertices[
            self._face_edit_faces[selected]
        ]
        selected_vertices = np.ascontiguousarray(
            selected_triangles.reshape((-1, 3)),
            dtype=np.float32,
        )
        selected_faces = np.arange(
            len(selected_vertices),
            dtype=np.int32,
        ).reshape((-1, 3))
        self._face_selection_item = _build_face_selection_item(
            selected_vertices,
            selected_faces,
        )
        self.view.addItem(self._face_selection_item)
        if (
            self._symmetric_preview_orientation is not None
            and self._symmetric_preview_plane_coordinate is not None
        ):
            mirrored_vertices = _mirror_preview_vertices(
                selected_vertices,
                self._symmetric_preview_orientation,
                self._symmetric_preview_plane_coordinate,
            )
            self._mirrored_face_selection_item = _build_face_selection_item(
                mirrored_vertices,
                selected_faces[:, (0, 2, 1)],
            )
            self.view.addItem(self._mirrored_face_selection_item)
        self.view.update()

    def _remove_face_selection_items(self) -> None:
        for item in (
            self._face_selection_item,
            self._mirrored_face_selection_item,
        ):
            if item is not None and hasattr(self, "view"):
                try:
                    self.view.removeItem(item)
                except ValueError:
                    pass
        self._face_selection_item = None
        self._mirrored_face_selection_item = None

    # ### Symmetric divided-object preview API ###
    def set_symmetric_division_preview(
        self,
        orientation: str | None,
        plane_coordinate: float | None = None,
    ) -> None:
        """Configure a viewer-only mirror without changing export geometry."""

        if orientation is None:
            if plane_coordinate is not None:
                raise ValueError("A cleared symmetric preview cannot keep a plane.")
            self._clear_symmetric_preview()
            return
        normalized_orientation = str(orientation).strip().lower()
        if normalized_orientation not in SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION:
            raise ValueError(
                "Symmetric preview orientation must be vertical or horizontal."
            )
        if plane_coordinate is None or not math.isfinite(float(plane_coordinate)):
            raise ValueError("Symmetric preview plane must be finite.")
        normalized_plane = float(plane_coordinate)
        if (
            self._symmetric_preview_orientation == normalized_orientation
            and self._symmetric_preview_plane_coordinate == normalized_plane
            and (
                self.symmetric_preview_textured_mesh_item is not None
                or self.symmetric_preview_mesh_item is not None
            )
        ):
            return
        self._face_selection_geometry_revision += 1
        self._invalidate_face_rectangle_selection_requests()
        self._remove_symmetric_preview_items()
        self._symmetric_preview_orientation = normalized_orientation
        self._symmetric_preview_plane_coordinate = normalized_plane
        self._build_symmetric_preview_items()
        self._refresh_face_selection_items()
        self._rebuild_projection_camera_indicators()
        self.view.update()

    def _build_symmetric_preview_items(self) -> None:
        """Build transient reflected draw items from the retained preview mesh."""

        if (
            self.model is None
            or self._symmetric_preview_orientation is None
            or self._symmetric_preview_plane_coordinate is None
        ):
            return
        display_mesh = self._get_display_mesh()
        assert display_mesh is not None
        group = self._create_symmetric_preview_group(
            display_mesh,
            self._symmetric_preview_orientation,
            self._symmetric_preview_plane_coordinate,
        )
        if group is None:
            return
        self._explicit_symmetric_preview_group = group
        self.symmetric_preview_textured_mesh_item = group.textured_item
        self.symmetric_preview_mesh_item = group.mesh_item
        self._symmetric_preview_vertices = group.vertices
        self._symmetric_preview_faces = group.faces
        self._symmetric_preview_face_colors = group.face_colors
        retained_vertices = np.asarray(display_mesh.vertices, dtype=np.float32)
        self._start_symmetric_preview_animation()
        self._apply_render_display_options()
        if not self._last_set_model_preserved_camera:
            self._frame_symmetric_preview(retained_vertices, group.vertices)

    def _build_embedded_symmetric_preview_items(self) -> None:
        """Build mirrors recorded by placed symmetric-object composition."""

        if self.model is None:
            return
        for preview_object in self.model.preview_symmetric_objects:
            source_meshes = (
                preview_object.mirrored_meshes
                if preview_object.mirrored_meshes
                else preview_object.meshes
            )
            for source_mesh in source_meshes:
                if preview_object.mirrored_meshes:
                    group = self._create_symmetric_preview_mesh_group(
                        source_mesh
                    )
                else:
                    group = self._create_symmetric_preview_group(
                        source_mesh,
                        preview_object.orientation,
                        preview_object.plane_coordinate,
                    )
                if group is not None:
                    self._embedded_symmetric_preview_groups.append(group)
        if self._embedded_symmetric_preview_groups:
            self._start_symmetric_preview_animation()

    def _create_symmetric_preview_group(
        self,
        retained_mesh,
        orientation: str,
        plane_coordinate: float,
        parent_item: GLGraphicsItem | None = None,
    ) -> _SymmetricPreviewRenderGroup | None:
        """Reflect one retained mesh using the shared symmetric-preview rules."""

        vertices = np.asarray(retained_mesh.vertices, dtype=np.float32)
        faces = np.asarray(retained_mesh.faces, dtype=np.int32)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or faces.ndim != 2
            or faces.shape[1] != 3
            or not len(vertices)
            or not len(faces)
        ):
            return None
        mirrored_vertices = _mirror_preview_vertices(
            vertices,
            orientation,
            plane_coordinate,
        )
        mirrored_faces = np.ascontiguousarray(faces[:, (0, 2, 1)])
        opacity = SYMMETRIC_PREVIEW_MIN_OPACITY
        texture_mesh_data = _build_texture_mesh_data(retained_mesh)
        textured_item = None
        if texture_mesh_data is not None:
            mirrored_texture_data = _mirror_texture_mesh_data(
                texture_mesh_data,
                orientation,
                plane_coordinate,
            )
            textured_item = TexturedMeshItem(
                mirrored_texture_data,
                self._ambient_light_intensity,
                opacity=opacity,
                translucent=True,
            )
            self._attach_preview_item(textured_item, parent_item)
        face_colors = (
            np.tile(FACE_COLOR, (faces.shape[0], 1))
            if texture_mesh_data is not None
            else _get_mesh_face_colors(retained_mesh, faces)
        )
        face_colors = np.ascontiguousarray(face_colors, dtype=float)
        faded_colors = face_colors.copy()
        faded_colors[:, 3] *= opacity
        mesh_item = _WireframeOverlayMeshItem(
            vertexes=mirrored_vertices,
            faces=mirrored_faces,
            faceColors=faded_colors,
            smooth=False,
            drawFaces=True,
            drawEdges=True,
            edgeColor=(*EDGE_COLOR[:3], opacity),
            shader=self._ambient_shader,
        )
        mesh_item.setGLOptions("translucent")
        self._attach_preview_item(mesh_item, parent_item)
        return _SymmetricPreviewRenderGroup(
            textured_item=textured_item,
            mesh_item=mesh_item,
            vertices=mirrored_vertices,
            faces=mirrored_faces,
            face_colors=face_colors,
        )

    def _create_symmetric_preview_mesh_group(
        self,
        mirrored_mesh,
        parent_item: GLGraphicsItem | None = None,
    ) -> _SymmetricPreviewRenderGroup | None:
        """Render geometry that was already mirrored in object-local space."""

        vertices = np.asarray(mirrored_mesh.vertices, dtype=np.float32)
        faces = np.asarray(mirrored_mesh.faces, dtype=np.int32)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or faces.ndim != 2
            or faces.shape[1:] != (3,)
            or not len(vertices)
            or not len(faces)
        ):
            return None
        opacity = SYMMETRIC_PREVIEW_MIN_OPACITY
        texture_mesh_data = _build_texture_mesh_data(mirrored_mesh)
        textured_item = None
        if texture_mesh_data is not None:
            textured_item = TexturedMeshItem(
                texture_mesh_data,
                self._ambient_light_intensity,
                opacity=opacity,
                translucent=True,
            )
            self._attach_preview_item(textured_item, parent_item)
        face_colors = (
            np.tile(FACE_COLOR, (faces.shape[0], 1))
            if texture_mesh_data is not None
            else _get_mesh_face_colors(mirrored_mesh, faces)
        )
        face_colors = np.ascontiguousarray(face_colors, dtype=float)
        faded_colors = face_colors.copy()
        faded_colors[:, 3] *= opacity
        mesh_item = _WireframeOverlayMeshItem(
            vertexes=vertices,
            faces=faces,
            faceColors=faded_colors,
            smooth=False,
            drawFaces=True,
            drawEdges=True,
            edgeColor=(*EDGE_COLOR[:3], opacity),
            shader=self._ambient_shader,
        )
        mesh_item.setGLOptions("translucent")
        self._attach_preview_item(mesh_item, parent_item)
        return _SymmetricPreviewRenderGroup(
            textured_item=textured_item,
            mesh_item=mesh_item,
            vertices=np.ascontiguousarray(vertices),
            faces=np.ascontiguousarray(faces),
            face_colors=face_colors,
        )

    def _attach_preview_item(
        self,
        item: GLGraphicsItem,
        parent_item: GLGraphicsItem | None,
    ) -> None:
        if parent_item is None:
            self.view.addItem(item)
            return
        item.setParentItem(parent_item)

    def _start_symmetric_preview_animation(self) -> None:
        """Synchronize all current mirror groups to one fade phase."""

        self._symmetric_preview_phase = -math.pi / 2.0
        if self.isVisible():
            self._symmetric_preview_timer.start()

    def _get_symmetric_preview_groups(
        self,
    ) -> tuple[_SymmetricPreviewRenderGroup, ...]:
        groups = list(self._embedded_symmetric_preview_groups)
        for placed_group in self._placed_object_render_groups.values():
            groups.extend(placed_group.symmetric_groups)
        if self._explicit_symmetric_preview_group is not None:
            groups.append(self._explicit_symmetric_preview_group)
        return tuple(groups)

    def _advance_symmetric_preview_fade(self) -> None:
        """Advance the translucent mirror through one smooth pulse sample."""

        groups = self._get_symmetric_preview_groups()
        if not groups:
            self._symmetric_preview_timer.stop()
            return
        self._symmetric_preview_phase = (
            self._symmetric_preview_phase
            + 2.0
            * math.pi
            * SYMMETRIC_PREVIEW_UPDATE_INTERVAL_MILLISECONDS
            / SYMMETRIC_PREVIEW_FADE_PERIOD_MILLISECONDS
        ) % (2.0 * math.pi)
        blend = (math.sin(self._symmetric_preview_phase) + 1.0) / 2.0
        opacity = SYMMETRIC_PREVIEW_MIN_OPACITY + blend * (
            SYMMETRIC_PREVIEW_MAX_OPACITY - SYMMETRIC_PREVIEW_MIN_OPACITY
        )
        for group in groups:
            if group.textured_item is not None:
                group.textured_item.set_opacity(opacity)
            faded_colors = group.face_colors.copy()
            faded_colors[:, 3] *= opacity
            group.mesh_item.setMeshData(
                vertexes=group.vertices,
                faces=group.faces,
                faceColors=faded_colors,
            )
            group.mesh_item.opts["edgeColor"] = (
                *EDGE_COLOR[:3],
                opacity,
            )
            group.mesh_item.update()
        self.view.update()

    def _frame_symmetric_preview(
        self,
        retained_vertices: np.ndarray,
        mirrored_vertices: np.ndarray,
    ) -> None:
        combined = np.vstack((retained_vertices, mirrored_vertices))
        if combined.size == 0 or not np.all(np.isfinite(combined)):
            return
        minimum = np.min(combined, axis=0)
        maximum = np.max(combined, axis=0)
        center = (minimum + maximum) / 2.0
        extent = float(max(np.max(maximum - minimum), 1.0))
        self.view.opts["center"] = QVector3D(
            float(center[0]),
            float(center[1]),
            float(center[2]),
        )
        self.view.setCameraPosition(
            distance=extent * 3.0,
            elevation=28.0,
            azimuth=-40.0,
        )
        self.view.remember_orbit_camera_state()
        self.view.apply_navigation_camera()

    def _remove_symmetric_preview_items(self) -> None:
        group = self._explicit_symmetric_preview_group
        if group is not None:
            for item in (group.textured_item, group.mesh_item):
                if item is not None and item in self.view.items:
                    self.view.removeItem(item)
        self._reset_symmetric_preview_item_state()
        if not self._get_symmetric_preview_groups():
            self._symmetric_preview_timer.stop()

    def _reset_symmetric_preview_item_state(self) -> None:
        """Forget transient mirror items after either targeted or full clear."""

        self.symmetric_preview_textured_mesh_item = None
        self.symmetric_preview_mesh_item = None
        self._symmetric_preview_vertices = None
        self._symmetric_preview_faces = None
        self._symmetric_preview_face_colors = None
        self._explicit_symmetric_preview_group = None

    def _clear_symmetric_preview(self) -> None:
        had_preview = (
            self._symmetric_preview_orientation is not None
            or self._symmetric_preview_plane_coordinate is not None
        )
        if had_preview:
            self._face_selection_geometry_revision += 1
            self._invalidate_face_rectangle_selection_requests()
        self._remove_symmetric_preview_items()
        self._symmetric_preview_orientation = None
        self._symmetric_preview_plane_coordinate = None
        if hasattr(self, "view"):
            self._refresh_face_selection_items()
            self._rebuild_projection_camera_indicators()
            self.view.update()

    def set_texture_edit_mask(self, mask: np.ndarray | None) -> None:
        """Preview editable UV texels on the generated object's material."""

        if mask is None:
            self._texture_edit_mask = None
        else:
            raw_mask = np.asarray(mask)
            if raw_mask.ndim != 2 or raw_mask.size == 0:
                raise ValueError("A texture edit mask must be a non-empty image.")
            self._texture_edit_mask = np.ascontiguousarray(
                raw_mask > 0,
                dtype=np.uint8,
            )
        if self.textured_mesh_item is not None:
            self.textured_mesh_item.set_edit_mask(self._texture_edit_mask)
        self.view.update()

    def set_ambient_light_intensity(self, intensity: float) -> None:
        """Set the view-wide ambient baseline from 0 (black) to 1 (full)."""

        self._ambient_light_intensity = _normalize_ambient_light_intensity(
            intensity
        )
        self._ambient_shader.setUniformData(
            "u_ambient_light",
            [self._ambient_light_intensity],
        )
        if self.textured_mesh_item is not None:
            self.textured_mesh_item.set_ambient_light_intensity(
                self._ambient_light_intensity
            )
        for group in self._get_symmetric_preview_groups():
            if group.textured_item is not None:
                group.textured_item.set_ambient_light_intensity(
                    self._ambient_light_intensity
                )
        for textured_surface_item in self.textured_surface_items:
            textured_surface_item.set_ambient_light_intensity(
                self._ambient_light_intensity
            )
        for placed_group in self._placed_object_render_groups.values():
            for retained_part in placed_group.retained_parts:
                if retained_part.textured_item is not None:
                    retained_part.textured_item.set_ambient_light_intensity(
                        self._ambient_light_intensity
                    )
        self.view.update()

    def get_ambient_light_intensity(self) -> float:
        return self._ambient_light_intensity

    def set_textures_enabled(self, enabled: bool) -> None:
        """Show or hide source textures while keeping the generated geometry."""

        self._textures_enabled = bool(enabled)
        self._apply_render_display_options()

    def get_textures_enabled(self) -> bool:
        return self._textures_enabled

    def set_wireframe_enabled(self, enabled: bool) -> None:
        """Show or hide wireframe edges over the current surface display."""

        self._wireframe_enabled = bool(enabled)
        self._apply_render_display_options()

    def get_wireframe_enabled(self) -> bool:
        return self._wireframe_enabled

    def set_wireframe_only(self, enabled: bool) -> None:
        """Draw geometry as edges only, regardless of texture visibility."""

        self._wireframe_only = bool(enabled)
        self._apply_render_display_options()

    def get_wireframe_only(self) -> bool:
        return self._wireframe_only

    # ### Projection camera indicator API ###
    def set_projection_camera_indicators_visible(self, visible: bool) -> None:
        """Show or hide the six illustrative texture-projection cameras."""

        self._projection_camera_indicators_visible = bool(visible)
        if self._projection_camera_indicators_visible:
            self._ensure_projection_camera_indicators()
        else:
            self.set_selected_projection_camera_id(None)
            self._remove_projection_camera_indicators()
        self._sync_projection_camera_input_state()

    def get_projection_camera_indicators_visible(self) -> bool:
        """Return whether this viewer requests projection camera markers."""

        return self._projection_camera_indicators_visible

    def set_projection_camera_percentages(
        self,
        percentages: Sequence[int] | Mapping[str, int],
    ) -> None:
        """Update all six allocation bars and labels immediately."""

        normalized = normalize_projection_camera_indicator_percentages(
            percentages
        )
        if normalized == self._projection_camera_percentages:
            return
        self._projection_camera_percentages = normalized
        self._refresh_projection_camera_indicator_items()

    def get_projection_camera_percentages(self) -> tuple[int, ...]:
        """Return the displayed percentages in canonical camera order."""

        return self._projection_camera_percentages

    def set_selected_projection_camera_id(
        self,
        camera_id: str | None,
    ) -> bool:
        """Select one camera indicator, or clear the current selection."""

        if camera_id is None:
            normalized_id = None
        else:
            normalized_id = str(camera_id).strip()
            if normalized_id not in ALL_CAMERA_IDS:
                raise ValueError("Unknown projection camera ID.")
        if normalized_id == self._selected_projection_camera_id:
            return False
        # A partial high-resolution wheel gesture belongs to the camera that
        # was selected when it began.  Disable routing before changing IDs so
        # that its sub-tick remainder cannot leak into the next camera.
        self.view.set_overlay_wheel_steps_enabled(False)
        self._selected_projection_camera_id = normalized_id
        self._refresh_projection_camera_indicator_items()
        self._sync_projection_camera_input_state()
        self.projection_camera_selection_changed.emit(normalized_id)
        return True

    def get_selected_projection_camera_id(self) -> str | None:
        """Return the selected canonical camera ID, if any."""

        return self._selected_projection_camera_id

    @Slot(object)
    def _handle_projection_camera_selection_requested(
        self,
        position: object,
    ) -> None:
        if not self._projection_camera_indicators_visible:
            return
        selected_id = self._pick_projection_camera_indicator(QPointF(position))
        self.set_selected_projection_camera_id(selected_id)

    @Slot(int)
    def _handle_projection_camera_wheel_steps_requested(
        self,
        steps: int,
    ) -> None:
        selected_id = self._selected_projection_camera_id
        normalized_steps = int(steps)
        if selected_id is None or normalized_steps == 0:
            return
        self.projection_camera_percentage_step_requested.emit(
            selected_id,
            normalized_steps,
        )

    def _pick_projection_camera_indicator(
        self,
        position: QPointF,
    ) -> str | None:
        """Pick the nearest camera body/bar in screen space without GL_SELECT."""

        if not self.projection_camera_indicator_geometries:
            return None
        viewport_width = max(int(self.view.width()), 1)
        viewport_height = max(int(self.view.height()), 1)
        viewport = (0, 0, viewport_width, viewport_height)
        view_projection = (
            self.view.projectionMatrix(viewport, viewport)
            * self.view.viewMatrix()
        )
        point = np.asarray((float(position.x()), float(position.y())), dtype=float)
        candidates: list[tuple[float, float, int, str]] = []
        for camera_index, camera_id in enumerate(ALL_CAMERA_IDS):
            geometry = self.projection_camera_indicator_geometries.get(camera_id)
            if geometry is None:
                continue
            projected_lines = _project_vertices_to_view(
                geometry.selection_line_positions,
                view_projection,
                viewport_width,
                viewport_height,
            )
            line_hit = _get_nearest_projected_line_hit(point, projected_lines)
            if (
                line_hit is not None
                and line_hit[0] <= PROJECTION_CAMERA_SELECTION_TOLERANCE_PIXELS
            ):
                candidates.append(
                    (line_hit[0], line_hit[1], camera_index, camera_id)
                )
            projected_label = _project_vertices_to_view(
                geometry.label_position[np.newaxis, :],
                view_projection,
                viewport_width,
                viewport_height,
            )[0]
            if _is_usable_projected_point(projected_label):
                label_distance = float(
                    np.linalg.norm(point - projected_label[:2])
                )
                if (
                    label_distance
                    <= PROJECTION_CAMERA_LABEL_SELECTION_TOLERANCE_PIXELS
                ):
                    candidates.append(
                        (
                            label_distance,
                            float(projected_label[2]),
                            camera_index,
                            camera_id,
                        )
                    )
        if not candidates:
            return None
        return min(candidates)[3]

    def _sync_projection_camera_input_state(self) -> None:
        has_indicators = bool(
            self._projection_camera_indicators_visible
            and self.projection_camera_indicator_items
        )
        self.view.set_overlay_selection_enabled(has_indicators)
        self.view.set_overlay_wheel_steps_enabled(
            has_indicators
            and self._selected_projection_camera_id is not None
        )

    def _refresh_projection_camera_indicator_items(self) -> None:
        if (
            not self.projection_camera_indicator_items
            or not self.projection_camera_indicator_geometries
        ):
            return
        update_projection_camera_indicator_items(
            self.projection_camera_indicator_items,
            self.projection_camera_indicator_geometries,
            self._projection_camera_percentages,
            selected_camera_id=self._selected_projection_camera_id,
        )
        self.view.update()

    def _ensure_projection_camera_indicators(self) -> None:
        """Lazily add camera markers once model bounds are available."""

        if (
            not self._projection_camera_indicators_visible
            or self.projection_camera_indicator_items
        ):
            return
        bounds = self._get_projection_camera_indicator_bounds()
        if bounds is None:
            self._sync_projection_camera_input_state()
            return
        self.projection_camera_indicator_geometries = (
            build_projection_camera_indicator_geometries(bounds)
        )
        self.projection_camera_indicator_items = (
            create_projection_camera_indicator_items(
                bounds,
                self._projection_camera_percentages,
                selected_camera_id=self._selected_projection_camera_id,
            )
        )
        for indicator_items in self.projection_camera_indicator_items.values():
            for indicator_item in indicator_items:
                self.view.addItem(indicator_item)
        self._sync_projection_camera_input_state()
        self.view.update()

    def _remove_projection_camera_indicators(self) -> None:
        """Remove every projection camera marker from the live scene."""

        for indicator_items in self.projection_camera_indicator_items.values():
            for indicator_item in indicator_items:
                indicator_item.setVisible(False)
                if indicator_item in self.view.items:
                    self.view.removeItem(indicator_item)
        self.projection_camera_indicator_items = {}
        self.projection_camera_indicator_geometries = {}
        self._sync_projection_camera_input_state()
        self.view.update()

    def _rebuild_projection_camera_indicators(self) -> None:
        """Reframe markers after the displayed model bounds change."""

        if not self._projection_camera_indicators_visible:
            return
        self._remove_projection_camera_indicators()
        self._ensure_projection_camera_indicators()

    def _get_projection_camera_indicator_bounds(
        self,
    ) -> np.ndarray | None:
        """Return bounds covering both retained and mirrored preview halves."""

        display_mesh = self._get_display_mesh()
        if display_mesh is None:
            return None
        retained_vertices = np.asarray(display_mesh.vertices, dtype=float)
        if retained_vertices.ndim != 2 or retained_vertices.shape[1:] != (3,):
            return None
        displayed_vertices = retained_vertices
        mirrored_vertices = self._symmetric_preview_vertices
        if mirrored_vertices is not None and len(mirrored_vertices):
            displayed_vertices = np.vstack(
                (retained_vertices, np.asarray(mirrored_vertices, dtype=float))
            )
        if not len(displayed_vertices) or not np.all(np.isfinite(displayed_vertices)):
            return None
        return np.asarray(
            (
                np.min(displayed_vertices, axis=0),
                np.max(displayed_vertices, axis=0),
            ),
            dtype=float,
        )

    def _populate_scene(self) -> None:
        self._clear_scene()
        self._add_grid()
        if self.model is None:
            self._set_default_camera()
            self._refresh_window_selection_outline()
            self._refresh_doorway_preview_outline_item()
            return

        display_mesh = self._get_display_mesh()
        assert display_mesh is not None
        vertices = np.asarray(display_mesh.vertices, dtype=np.float32)
        faces = np.asarray(display_mesh.faces, dtype=np.int32)
        if vertices.size and faces.size:
            texture_mesh_data = _build_texture_mesh_data(display_mesh)
            face_colors = (
                np.tile(FACE_COLOR, (faces.shape[0], 1))
                if texture_mesh_data is not None
                else _get_mesh_face_colors(display_mesh, faces)
            )

            if texture_mesh_data is not None:
                self.textured_mesh_item = TexturedMeshItem(
                    texture_mesh_data,
                    self._ambient_light_intensity,
                )
                self.textured_mesh_item.set_edit_mask(self._texture_edit_mask)
                self.view.addItem(self.textured_mesh_item)

            self.mesh_item = _WireframeOverlayMeshItem(
                vertexes=vertices,
                faces=faces,
                faceColors=face_colors,
                smooth=False,
                drawFaces=texture_mesh_data is None,
                drawEdges=True,
                edgeColor=EDGE_COLOR,
                shader=self._ambient_shader,
                cull_back_faces=bool(self.model.preview_textured_surfaces),
            )
            self.view.addItem(self.mesh_item)
        self._add_textured_surface_items()
        self._add_textured_wall_items()
        if self._placed_object_editing_enabled:
            self._add_placed_object_items()
        else:
            self._build_embedded_symmetric_preview_items()
        self._apply_render_display_options()
        self._ensure_projection_camera_indicators()

        bounding_box = self.model.mesh.bounding_box
        center = np.asarray(bounding_box.centroid, dtype=float)
        extent = float(max(bounding_box.extents.max(), 1.0))

        self.view.opts["center"] = QVector3D(
            float(center[0]),
            float(center[1]),
            float(center[2]),
        )
        self.view.setCameraPosition(distance=extent * 3.0, elevation=28.0, azimuth=-40.0)
        self.view.remember_orbit_camera_state()
        self._set_default_first_person_camera_pose_from_bounding_box(bounding_box)
        self.view.apply_navigation_camera()
        self._refresh_window_selection_outline()
        self._sync_placed_object_selection_rendering()
        self._refresh_doorway_preview_outline_item()
        self.view.update()

    def _get_display_mesh(self):
        """Return the mesh shared by the retained and mirrored previews."""

        if self.model is None:
            return None
        if (
            self._placed_object_editing_enabled
            and self.model.preview_placed_objects
            and self.model.preview_base_mesh is not None
        ):
            return self.model.preview_base_mesh
        if (
            self.model.preview_textured_surfaces
            and self.model.preview_untextured_mesh is not None
        ):
            return self.model.preview_untextured_mesh
        return self.model.mesh

    def _add_grid(self) -> None:
        self.grid_item = gl.GLGridItem()
        self.grid_item.setSize(x=20.0, y=20.0)
        self.grid_item.setSpacing(x=1.0, y=1.0)
        self.view.addItem(self.grid_item)

    def _add_textured_wall_items(self) -> None:
        if self.model is None:
            return

        assigned_wall_keys = {
            (
                surface.level_index,
                surface.room_index,
                surface.wall_key,
            )
            for surface in self.model.preview_textured_surfaces
            if surface.surface_type == "wall"
            and surface.level_index is not None
            and surface.room_index is not None
            and surface.wall_key is not None
        }
        for textured_wall in self.model.preview_textured_walls:
            if (
                textured_wall.level_index,
                textured_wall.room_index,
                textured_wall.wall_key,
            ) in assigned_wall_keys:
                continue
            # Room walls are oriented so the negative offset points toward
            # that room's interior. Rendering the legacy material on only its
            # owning side prevents an adjacent room's texture from crossing a
            # shared wall and covering a generated material on the other side.
            self._add_textured_wall_item(textured_wall, offset_sign=-1.0)

    def _add_textured_surface_items(self) -> None:
        if self.model is None:
            return
        for textured_surface in self.model.preview_textured_surfaces:
            if (
                self._placed_object_editing_enabled
                and self.model.preview_placed_objects
                and textured_surface.surface_type == "generated_object"
            ):
                continue
            texture_data = _build_texture_mesh_data(textured_surface.mesh)
            if texture_data is None:
                continue
            texture_item = TexturedMeshItem(
                texture_data,
                self._ambient_light_intensity,
                texture_repeat=True,
                double_sided=textured_surface.double_sided,
            )
            self.view.addItem(texture_item)
            self.textured_surface_items.append(texture_item)

    # ### Placed-object rendering ###
    def _add_placed_object_items(self) -> None:
        """Render each Canvas object below one independently movable root."""

        if self.model is None:
            return
        has_symmetric_preview = False
        for preview in self.model.preview_placed_objects:
            if preview.object_id in self._placed_object_render_groups:
                continue
            root_item = GLGraphicsItem()
            root_item.setTransform(
                _numpy_transform_to_qt(preview.placement_transform)
            )
            self.view.addItem(root_item)
            retained_parts = [
                part
                for mesh in preview.meshes
                if (part := self._create_placed_object_mesh_render(
                    mesh,
                    root_item,
                ))
                is not None
            ]
            mirrored_meshes = _build_local_symmetric_preview_meshes(preview)
            symmetric_groups = [
                group
                for mesh in mirrored_meshes
                if (group := self._create_symmetric_preview_mesh_group(
                    mesh,
                    root_item,
                ))
                is not None
            ]
            pick_meshes = (*preview.meshes, *mirrored_meshes)
            bounds = _get_combined_mesh_bounds(pick_meshes)
            if not retained_parts or bounds is None:
                self.view.removeItem(root_item)
                continue
            selection_item = gl.GLLinePlotItem(
                pos=_build_bounds_line_positions(bounds),
                color=TRANSFORM_GIZMO_SELECTION_COLOR,
                width=2.0,
                antialias=True,
                mode="lines",
            )
            selection_item.setGLOptions("translucent")
            selection_item.setVisible(False)
            selection_item.setParentItem(root_item)
            self._placed_object_render_groups[preview.object_id] = (
                _PlacedObjectRenderGroup(
                    preview=preview,
                    root_item=root_item,
                    retained_parts=retained_parts,
                    symmetric_groups=symmetric_groups,
                    pick_meshes=pick_meshes,
                    selection_item=selection_item,
                    current_transform=np.asarray(
                        preview.placement_transform,
                        dtype=float,
                    ).copy(),
                )
            )
            has_symmetric_preview = (
                has_symmetric_preview or bool(symmetric_groups)
            )
        if has_symmetric_preview:
            self._start_symmetric_preview_animation()

    def _create_placed_object_mesh_render(
        self,
        mesh,
        root_item: GLGraphicsItem,
    ) -> _PlacedObjectMeshRender | None:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or faces.ndim != 2
            or faces.shape[1:] != (3,)
            or not len(vertices)
            or not len(faces)
        ):
            return None
        texture_data = _build_texture_mesh_data(mesh)
        textured_item = None
        if texture_data is not None:
            textured_item = TexturedMeshItem(
                texture_data,
                self._ambient_light_intensity,
                texture_repeat=True,
            )
            textured_item.setParentItem(root_item)
        face_colors = (
            np.tile(FACE_COLOR, (faces.shape[0], 1))
            if texture_data is not None
            else _get_mesh_face_colors(mesh, faces)
        )
        mesh_item = _WireframeOverlayMeshItem(
            vertexes=vertices,
            faces=faces,
            faceColors=face_colors,
            smooth=False,
            drawFaces=texture_data is None,
            drawEdges=True,
            edgeColor=EDGE_COLOR,
            shader=self._ambient_shader,
        )
        mesh_item.setParentItem(root_item)
        return _PlacedObjectMeshRender(
            textured_item=textured_item,
            mesh_item=mesh_item,
        )

    # ### Placed-object selection and gizmo rendering ###
    def _sync_placed_object_selection_rendering(self) -> None:
        selected_id = self._selected_placed_object_id
        if selected_id not in self._placed_object_render_groups:
            selected_id = None
            self._selected_placed_object_id = None
        for object_id, group in self._placed_object_render_groups.items():
            group.selection_item.setVisible(object_id == selected_id)
        self._remove_transform_gizmo_items()
        if selected_id is None:
            if self.object_transform_status_label is not None:
                self.object_transform_status_label.setText(
                    "Select a placed object to show its gizmo."
                )
            if hasattr(self, "view"):
                self.view.update()
            return
        self._build_transform_gizmo_items(
            self._placed_object_render_groups[selected_id]
        )
        if self.object_transform_status_label is not None:
            self.object_transform_status_label.setText(
                "Drag an RGB arrow to move, or an RGB ring to rotate."
            )
        self.view.update()

    def _build_transform_gizmo_items(
        self,
        group: _PlacedObjectRenderGroup,
    ) -> None:
        pivot = _get_render_group_world_pivot(group)
        local_bounds = _get_combined_mesh_bounds(group.pick_meshes)
        fallback_size = TRANSFORM_GIZMO_MIN_SIZE_METERS
        if local_bounds is not None:
            fallback_size = max(
                fallback_size,
                float(np.max(local_bounds[1] - local_bounds[0])) * 0.65,
            )
        try:
            pixel_size = float(
                self.view.pixelSize(
                    QVector3D(*[float(value) for value in pivot])
                )
            )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pixel_size = 0.0
        gizmo_size = (
            pixel_size * TRANSFORM_GIZMO_SCREEN_SIZE_PIXELS
            if math.isfinite(pixel_size) and pixel_size > 0.0
            else fallback_size
        )
        self._transform_gizmo_size = max(
            TRANSFORM_GIZMO_MIN_SIZE_METERS,
            gizmo_size,
        )
        axes = np.eye(3, dtype=float)
        for axis_index, axis in enumerate(axes):
            color = TRANSFORM_GIZMO_AXIS_COLORS[axis_index]
            endpoint = pivot + axis * self._transform_gizmo_size
            axis_item = gl.GLLinePlotItem(
                pos=np.asarray((pivot, endpoint), dtype=float),
                color=color,
                width=3.0,
                antialias=True,
                mode="lines",
            )
            axis_item.setGLOptions("translucent")
            self.view.addItem(axis_item)
            self._transform_gizmo_items.append(axis_item)

            endpoint_item = gl.GLScatterPlotItem(
                pos=np.asarray((endpoint,), dtype=float),
                color=color,
                size=10.0,
                pxMode=True,
            )
            endpoint_item.setGLOptions("translucent")
            self.view.addItem(endpoint_item)
            self._transform_gizmo_items.append(endpoint_item)

            ring_positions = _build_rotation_ring_positions(
                pivot,
                axis_index,
                self._transform_gizmo_size
                * TRANSFORM_GIZMO_RING_RADIUS_RATIO,
            )
            ring_item = gl.GLLinePlotItem(
                pos=ring_positions,
                color=color,
                width=2.0,
                antialias=True,
                mode="line_strip",
            )
            ring_item.setGLOptions("translucent")
            self.view.addItem(ring_item)
            self._transform_gizmo_items.append(ring_item)

    def _remove_transform_gizmo_items(self) -> None:
        if not hasattr(self, "view"):
            self._transform_gizmo_items = []
            return
        for item in self._transform_gizmo_items:
            if item in self.view.items:
                self.view.removeItem(item)
        self._transform_gizmo_items = []

    def _pick_transform_gizmo_handle(
        self,
        ray_origin: object,
        ray_direction: object,
    ) -> _TransformGizmoHandle | None:
        selected_id = self._selected_placed_object_id
        group = self._placed_object_render_groups.get(selected_id or "")
        if group is None:
            return None
        origin, direction = _normalize_ray(ray_origin, ray_direction)
        if origin is None or direction is None:
            return None
        pivot = _get_render_group_world_pivot(group)
        candidates: list[tuple[float, _TransformGizmoHandle]] = []
        for axis_index, axis in enumerate(np.eye(3, dtype=float)):
            segment_end = pivot + axis * self._transform_gizmo_size
            segment_distance = _get_ray_segment_distance(
                origin,
                direction,
                pivot,
                segment_end,
            )
            axis_tolerance = (
                self._transform_gizmo_size
                * TRANSFORM_GIZMO_AXIS_HIT_RATIO
            )
            if segment_distance is not None and segment_distance <= axis_tolerance:
                candidates.append(
                    (
                        segment_distance / max(axis_tolerance, 1e-12),
                        _TransformGizmoHandle(
                            TRANSFORM_GIZMO_TRANSLATE,
                            axis_index,
                        ),
                    )
                )

            ring_hit = _intersect_ray_with_plane(
                origin,
                direction,
                pivot,
                axis,
            )
            if ring_hit is None:
                continue
            ring_radius = (
                self._transform_gizmo_size
                * TRANSFORM_GIZMO_RING_RADIUS_RATIO
            )
            ring_error = abs(float(np.linalg.norm(ring_hit - pivot)) - ring_radius)
            ring_tolerance = (
                self._transform_gizmo_size
                * TRANSFORM_GIZMO_RING_HIT_RATIO
            )
            if ring_error <= ring_tolerance:
                candidates.append(
                    (
                        ring_error / max(ring_tolerance, 1e-12),
                        _TransformGizmoHandle(
                            TRANSFORM_GIZMO_ROTATE,
                            axis_index,
                        ),
                    )
                )
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda candidate: (
                candidate[0],
                candidate[1].kind != TRANSFORM_GIZMO_TRANSLATE,
                candidate[1].axis_index,
            ),
        )[1]

    # ### Placed-object gizmo dragging ###
    def _begin_placed_object_gizmo_drag(
        self,
        handle: _TransformGizmoHandle,
        position: QPointF,
    ) -> bool:
        selected_id = self._selected_placed_object_id
        group = self._placed_object_render_groups.get(selected_id or "")
        camera_ray = self.view.build_camera_ray(position)
        if group is None or camera_ray is None:
            return False
        origin, direction = camera_ray
        axis = np.eye(3, dtype=float)[handle.axis_index]
        pivot = _get_render_group_world_pivot(group)
        start_transform = np.asarray(group.current_transform, dtype=float).copy()
        try:
            local_pivot = _transform_point(
                np.linalg.inv(start_transform),
                pivot,
            )
        except np.linalg.LinAlgError:
            return False
        if handle.kind == TRANSFORM_GIZMO_TRANSLATE:
            drag_plane_normal = _build_axis_drag_plane_normal(axis, direction)
            hit = _intersect_ray_with_plane(
                origin,
                direction,
                pivot,
                drag_plane_normal,
            )
            if hit is None:
                return False
            start_axis_parameter = float(np.dot(hit - pivot, axis))
            previous_rotation_vector = None
        else:
            drag_plane_normal = axis
            hit = _intersect_ray_with_plane(
                origin,
                direction,
                pivot,
                drag_plane_normal,
            )
            if hit is None:
                return False
            previous_rotation_vector = _normalize_vector(hit - pivot)
            if previous_rotation_vector is None:
                return False
            start_axis_parameter = None
        self._placed_object_transform_drag = _PlacedObjectTransformDrag(
            object_id=group.preview.object_id,
            handle=handle,
            start_world_position=np.asarray(
                group.preview.world_position,
                dtype=float,
            ),
            start_rotation_degrees=group.preview.rotation_degrees,
            start_transform=start_transform,
            local_pivot=local_pivot,
            axis=axis,
            drag_plane_normal=drag_plane_normal,
            start_axis_parameter=start_axis_parameter,
            previous_rotation_vector=previous_rotation_vector,
            preview_world_position=group.preview.world_position,
            preview_rotation_degrees=group.preview.rotation_degrees,
        )
        self.view.reserve_primary_pointer_drag()
        if self.object_transform_status_label is not None:
            action = (
                "Moving"
                if handle.kind == TRANSFORM_GIZMO_TRANSLATE
                else "Rotating"
            )
            self.object_transform_status_label.setText(
                f"{action} on {'XYZ'[handle.axis_index]}. "
                "Release to save; Escape cancels."
            )
        return True

    def _update_placed_object_gizmo_drag(self, position: QPointF) -> bool:
        drag = self._placed_object_transform_drag
        if drag is None:
            return False
        group = self._placed_object_render_groups.get(drag.object_id)
        camera_ray = self.view.build_camera_ray(position)
        if group is None or camera_ray is None:
            return False
        origin, direction = camera_ray
        pivot = drag.start_world_position
        hit = _intersect_ray_with_plane(
            origin,
            direction,
            pivot,
            drag.drag_plane_normal,
        )
        if hit is None:
            return False

        if drag.handle.kind == TRANSFORM_GIZMO_TRANSLATE:
            assert drag.start_axis_parameter is not None
            parameter = float(np.dot(hit - pivot, drag.axis))
            delta = drag.axis * (parameter - drag.start_axis_parameter)
            world_position = drag.start_world_position + delta
            rotation_degrees = drag.start_rotation_degrees
            transform = np.eye(4, dtype=float)
            transform[:3, 3] = delta
            transform = transform @ drag.start_transform
        else:
            current_vector = _normalize_vector(hit - pivot)
            previous_vector = drag.previous_rotation_vector
            if current_vector is None or previous_vector is None:
                return False
            drag.accumulated_rotation_degrees += _get_signed_rotation_degrees(
                drag.axis,
                previous_vector,
                current_vector,
            )
            drag.previous_rotation_vector = current_vector
            delta_rotation = trimesh.transformations.rotation_matrix(
                math.radians(drag.accumulated_rotation_degrees),
                drag.axis,
            )[:3, :3]
            world_rotation = delta_rotation @ drag.start_transform[:3, :3]
            rotation_degrees = _rotation_matrix_to_degrees(world_rotation)
            world_position = drag.start_world_position
            transform = _build_pivoted_world_transform(
                world_position,
                world_rotation,
                drag.local_pivot,
            )

        group.current_transform = np.asarray(transform, dtype=float)
        group.root_item.setTransform(_numpy_transform_to_qt(transform))
        drag.preview_world_position = tuple(
            float(value) for value in world_position
        )
        drag.preview_rotation_degrees = tuple(
            float(value) for value in rotation_degrees
        )
        self._remove_transform_gizmo_items()
        self._build_transform_gizmo_items(group)
        self.view.update()
        return True

    def _finish_placed_object_gizmo_drag(self, position: QPointF) -> bool:
        drag = self._placed_object_transform_drag
        if drag is None:
            return False
        self._update_placed_object_gizmo_drag(position)
        world_position = drag.preview_world_position
        rotation_degrees = drag.preview_rotation_degrees
        changed = bool(
            world_position is not None
            and rotation_degrees is not None
            and (
                not np.allclose(
                    world_position,
                    drag.start_world_position,
                    atol=1e-9,
                    rtol=0.0,
                )
                or not np.allclose(
                    rotation_degrees,
                    drag.start_rotation_degrees,
                    atol=1e-9,
                    rtol=0.0,
                )
            )
        )
        self._placed_object_transform_drag = None
        self.view.release_primary_pointer_drag()
        if changed:
            assert world_position is not None and rotation_degrees is not None
            group = self._placed_object_render_groups.get(drag.object_id)
            if group is None:
                changed = False
            else:
                self._remember_placed_object_preview_transform(
                    group,
                    world_position,
                    rotation_degrees,
                )
        if changed:
            assert world_position is not None and rotation_degrees is not None
            self.placed_object_transform_changed.emit(
                drag.object_id,
                world_position,
                rotation_degrees,
            )
        self._sync_placed_object_selection_rendering()
        return changed

    def _remember_placed_object_preview_transform(
        self,
        group: _PlacedObjectRenderGroup,
        world_position: tuple[float, float, float],
        rotation_degrees: tuple[float, float, float],
    ) -> None:
        """Make a committed live transform the baseline for the next drag."""

        preview = replace(
            group.preview,
            placement_transform=group.current_transform,
            world_position=world_position,
            rotation_degrees=rotation_degrees,
        )
        group.preview = preview
        group.current_transform = np.asarray(
            preview.placement_transform,
            dtype=float,
        ).copy()
        if self.model is None:
            return
        self.model.preview_placed_objects = [
            preview if candidate.object_id == preview.object_id else candidate
            for candidate in self.model.preview_placed_objects
        ]

    def _cancel_placed_object_gizmo_drag(self, *_args: object) -> None:
        drag = self._placed_object_transform_drag
        self._placed_object_transform_drag = None
        self.view.release_primary_pointer_drag()
        if drag is not None:
            group = self._placed_object_render_groups.get(drag.object_id)
            if group is not None:
                group.current_transform = drag.start_transform.copy()
                group.root_item.setTransform(
                    _numpy_transform_to_qt(drag.start_transform)
                )
        self._sync_placed_object_selection_rendering()

    def _add_textured_wall_item(
        self,
        textured_wall: PreviewTexturedWall,
        offset_sign: float,
    ) -> None:
        texture_rgba = np.asarray(textured_wall.texture_rgba, dtype=np.ubyte)
        if texture_rgba.ndim != 3 or texture_rgba.shape[2] != 4:
            return
        texture_is_opaque = bool(np.all(texture_rgba[:, :, 3] == 255))

        image_item = gl.GLImageItem(
            texture_rgba,
            smooth=True,
            glOptions="opaque" if texture_is_opaque else "translucent",
        )
        image_item.setTransform(
            _build_textured_wall_transform(
                textured_wall=textured_wall,
                offset_sign=offset_sign,
            )
        )
        self.view.addItem(image_item)
        self.textured_wall_items.append(image_item)
        self.wall_by_item_id[id(image_item)] = textured_wall

    def _apply_render_display_options(self) -> None:
        """Apply texture and wireframe state without requiring an OpenGL paint."""

        textures_visible = self._textures_enabled and not self._wireframe_only
        if self.textured_mesh_item is not None:
            self.textured_mesh_item.setVisible(textures_visible)
        symmetric_groups = self._get_symmetric_preview_groups()
        for group in symmetric_groups:
            if group.textured_item is not None:
                group.textured_item.setVisible(textures_visible)
        for textured_surface_item in self.textured_surface_items:
            textured_surface_item.setVisible(textures_visible)
        for textured_wall_item in self.textured_wall_items:
            textured_wall_item.setVisible(textures_visible)

        for placed_group in self._placed_object_render_groups.values():
            for retained_part in placed_group.retained_parts:
                if retained_part.textured_item is not None:
                    retained_part.textured_item.setVisible(textures_visible)
                retained_part.mesh_item.opts["drawFaces"] = (
                    not self._wireframe_only
                    and (
                        retained_part.textured_item is None
                        or not self._textures_enabled
                    )
                )
                retained_part.mesh_item.opts["drawEdges"] = (
                    self._wireframe_enabled or self._wireframe_only
                )
                retained_part.mesh_item.update()

        if self.mesh_item is not None:
            has_textured_surface = self.textured_mesh_item is not None
            self.mesh_item.opts["drawFaces"] = (
                not self._wireframe_only
                and (not has_textured_surface or not self._textures_enabled)
            )
            self.mesh_item.opts["drawEdges"] = (
                self._wireframe_enabled or self._wireframe_only
            )
            self.mesh_item.update()
        for group in symmetric_groups:
            group.mesh_item.opts["drawFaces"] = (
                not self._wireframe_only
                and (
                    group.textured_item is None
                    or not self._textures_enabled
                )
            )
            group.mesh_item.opts["drawEdges"] = (
                self._wireframe_enabled or self._wireframe_only
            )
            group.mesh_item.update()

        if hasattr(self, "view"):
            self.view.update()

    def _handle_view_items_clicked(self, clicked_items: list[object]) -> None:
        for clicked_item in clicked_items:
            textured_wall = self.wall_by_item_id.get(id(clicked_item))
            if textured_wall is None:
                continue

            self.wall_selected.emit(
                textured_wall.level_index,
                textured_wall.room_index,
                textured_wall.wall_key,
            )
            return

    def _set_default_camera(self) -> None:
        self.view.opts["center"] = QVector3D(0.0, 0.0, 0.0)
        self.view.setCameraPosition(distance=18.0, elevation=28.0, azimuth=-40.0)
        self.view.remember_orbit_camera_state()
        self.view.apply_navigation_camera()

    def _set_default_first_person_camera_pose_from_bounding_box(
        self,
        bounding_box,
    ) -> None:
        """Place an unset camera at a sensible indoor eye height near the model center."""

        if self.view.has_custom_first_person_camera_pose:
            return
        bounds = np.asarray(bounding_box.bounds, dtype=float)
        if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
            return
        minimum, maximum = bounds
        center = (minimum + maximum) / 2.0
        height = max(0.0, float(maximum[2] - minimum[2]))
        eye_height = (
            DEFAULT_FIRST_PERSON_HEIGHT_METERS
            if height <= 0.0
            else min(
                DEFAULT_FIRST_PERSON_HEIGHT_METERS,
                max(0.5, height * 0.65),
            )
        )
        self.view.set_default_first_person_camera_pose(
            CameraPose(
                x=float(center[0]),
                y=float(center[1]),
                z=float(minimum[2] + eye_height),
            )
        )

    def _clear_scene(self) -> None:
        if not hasattr(self, "view"):
            return

        self._symmetric_preview_timer.stop()
        self.view.clear()
        self.grid_item = None
        self.mesh_item = None
        self.textured_mesh_item = None
        self._face_selection_item = None
        self._mirrored_face_selection_item = None
        self._reset_symmetric_preview_item_state()
        self._embedded_symmetric_preview_groups = []
        self._placed_object_render_groups = {}
        self._remove_transform_gizmo_items()
        self.textured_surface_items = []
        self.textured_wall_items = []
        self.wall_by_item_id = {}
        self.projection_camera_indicator_items = {}
        self.projection_camera_indicator_geometries = {}
        self._sync_projection_camera_input_state()
        self._window_selection_item = None
        self._window_preview_item = None
        self._doorway_preview_outline_item = None

    def _capture_camera_state(self) -> dict[str, object]:
        camera_state: dict[str, object] = {}
        for key in CAMERA_STATE_KEYS:
            if key not in self.view.opts:
                continue

            value = self.view.opts[key]
            if isinstance(value, QVector3D):
                camera_state[key] = QVector3D(value)
            else:
                camera_state[key] = value

        return camera_state

    def _restore_camera_state(self, camera_state: dict[str, object]) -> None:
        self.view.opts.update(camera_state)
        self.view.remember_orbit_camera_state()
        self.view.apply_navigation_camera()
        self.view.update()


# ### Face selection helpers ###
def _build_face_selection_item(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> _WireframeOverlayMeshItem:
    """Build one translucent overlay for selected logical triangles."""

    face_colors = np.tile(
        np.asarray(FACE_SELECTION_COLOR, dtype=float),
        (len(faces), 1),
    )
    item = _WireframeOverlayMeshItem(
        vertexes=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        faceColors=face_colors,
        smooth=False,
        drawFaces=True,
        drawEdges=True,
        edgeColor=FACE_SELECTION_EDGE_COLOR,
        shader="shaded",
    )
    item.setGLOptions("translucent")
    return item


def _get_nearest_triangle_ray_face_index(
    vertices: object,
    faces: object,
    ray_origin: object,
    ray_direction: object,
) -> tuple[int, float] | None:
    """Return the closest double-sided triangle index and ray distance."""

    normalized_vertices = np.asarray(vertices, dtype=float)
    normalized_faces = np.asarray(faces, dtype=np.int64)
    origin = np.asarray(ray_origin, dtype=float)
    direction = np.asarray(ray_direction, dtype=float)
    if (
        normalized_vertices.ndim != 2
        or normalized_vertices.shape[1:] != (3,)
        or normalized_faces.ndim != 2
        or normalized_faces.shape[1:] != (3,)
        or origin.shape != (3,)
        or direction.shape != (3,)
        or not np.all(np.isfinite(normalized_vertices))
        or not np.all(np.isfinite(origin))
        or not np.all(np.isfinite(direction))
        or np.any(normalized_faces < 0)
        or (
            normalized_faces.size
            and np.any(normalized_faces >= len(normalized_vertices))
        )
    ):
        return None
    direction_length = float(np.linalg.norm(direction))
    if direction_length <= 1e-12 or len(normalized_faces) == 0:
        return None
    direction = direction / direction_length
    triangles = normalized_vertices[normalized_faces]
    first_edges = triangles[:, 1] - triangles[:, 0]
    second_edges = triangles[:, 2] - triangles[:, 0]
    cross_direction = np.cross(
        np.broadcast_to(direction, second_edges.shape),
        second_edges,
    )
    determinants = np.einsum("ij,ij->i", first_edges, cross_direction)
    usable = np.abs(determinants) > 1e-10
    inverse_determinants = np.zeros_like(determinants)
    inverse_determinants[usable] = 1.0 / determinants[usable]
    origin_offsets = origin[np.newaxis, :] - triangles[:, 0]
    first_coordinates = (
        np.einsum("ij,ij->i", origin_offsets, cross_direction)
        * inverse_determinants
    )
    offset_crosses = np.cross(origin_offsets, first_edges)
    second_coordinates = (
        np.einsum("j,ij->i", direction, offset_crosses)
        * inverse_determinants
    )
    distances = (
        np.einsum("ij,ij->i", second_edges, offset_crosses)
        * inverse_determinants
    )
    usable &= first_coordinates >= -1e-9
    usable &= second_coordinates >= -1e-9
    usable &= first_coordinates + second_coordinates <= 1.0 + 1e-9
    usable &= distances >= 0.0
    hit_indices = np.flatnonzero(usable)
    if hit_indices.size == 0:
        return None
    face_index = int(hit_indices[np.argmin(distances[hit_indices])])
    return face_index, float(distances[face_index])


def _select_face_indices_in_view_rectangle(
    view: SelectableGLViewWidget,
    geometry: Sequence[tuple[np.ndarray, np.ndarray]],
    rectangle: QRect,
    *,
    xray: bool,
) -> set[int]:
    """Select logical IDs owning pixels inside a current-camera rectangle."""

    captured = _capture_face_selection_raster_input(
        view,
        geometry,
        rectangle,
    )
    if captured is None:
        return set()
    projected_geometry, rectangle_values = captured
    return _rasterize_face_selection(
        projected_geometry,
        QRect(*rectangle_values),
        xray=bool(xray),
    )


def _capture_face_selection_raster_input(
    view: SelectableGLViewWidget,
    geometry: Sequence[tuple[np.ndarray, np.ndarray]],
    rectangle: QRect,
) -> tuple[
    tuple[tuple[np.ndarray, np.ndarray], ...],
    tuple[int, int, int, int],
] | None:
    """Capture camera projection and own read-only arrays on the GUI thread."""

    viewport_width = max(int(view.width()), 1)
    viewport_height = max(int(view.height()), 1)
    clipped = rectangle.normalized().intersected(
        QRect(0, 0, viewport_width, viewport_height)
    )
    if clipped.isEmpty():
        return None
    viewport = (0, 0, viewport_width, viewport_height)
    view_projection = (
        view.projectionMatrix(viewport, viewport) * view.viewMatrix()
    )
    projected_geometry: list[tuple[np.ndarray, np.ndarray]] = []
    for vertices, faces in geometry:
        projected_vertices = np.ascontiguousarray(
            _project_vertices_to_view(
                vertices,
                view_projection,
                viewport_width,
                viewport_height,
            ),
            dtype=float,
        )
        owned_faces = np.ascontiguousarray(
            np.asarray(faces, dtype=np.int64)
        ).copy()
        projected_vertices.setflags(write=False)
        owned_faces.setflags(write=False)
        projected_geometry.append((projected_vertices, owned_faces))
    return (
        tuple(projected_geometry),
        (
            int(clipped.x()),
            int(clipped.y()),
            int(clipped.width()),
            int(clipped.height()),
        ),
    )


def _project_vertices_to_view(
    vertices: np.ndarray,
    view_projection: object,
    viewport_width: int,
    viewport_height: int,
) -> np.ndarray:
    """Project every XYZ vertex with one captured Qt matrix."""

    normalized_vertices = np.asarray(vertices, dtype=np.float32)
    projected = np.full((len(normalized_vertices), 4), np.nan, dtype=float)
    if len(normalized_vertices) == 0:
        return projected

    matrix_data = np.asarray(view_projection.data(), dtype=np.float32)
    if matrix_data.shape != (16,):
        raise ValueError("Face selection requires a 4x4 view-projection matrix.")
    matrix = matrix_data.reshape((4, 4), order="F")
    clip_vertices = np.empty((len(normalized_vertices), 4), dtype=np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        for component_index, row in enumerate(matrix):
            clip_vertices[:, component_index] = (
                row[0] * normalized_vertices[:, 0]
                + row[1] * normalized_vertices[:, 1]
                + row[2] * normalized_vertices[:, 2]
                + row[3]
            )

    clip_w = clip_vertices[:, 3]
    usable = np.isfinite(clip_w) & (clip_w > 1e-10)
    normalized_clip = np.full(
        (len(normalized_vertices), 3),
        np.nan,
        dtype=float,
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(
            np.asarray(clip_vertices[:, :3], dtype=float),
            np.asarray(clip_w[:, np.newaxis], dtype=float),
            out=normalized_clip,
            where=usable[:, np.newaxis],
        )
    usable &= np.all(np.isfinite(normalized_clip), axis=1)
    projected[usable, 0] = (
        (normalized_clip[usable, 0] + 1.0) * 0.5 * viewport_width
    )
    projected[usable, 1] = (
        (1.0 - normalized_clip[usable, 1]) * 0.5 * viewport_height
    )
    projected[usable, 2] = normalized_clip[usable, 2]
    projected[usable, 3] = 1.0
    return projected


def _get_nearest_projected_line_hit(
    point: np.ndarray,
    projected_line_positions: np.ndarray,
) -> tuple[float, float] | None:
    """Return screen distance and depth for paired projected line segments."""

    normalized_point = np.asarray(point, dtype=float)
    projected = np.asarray(projected_line_positions, dtype=float)
    if (
        normalized_point.shape != (2,)
        or not np.all(np.isfinite(normalized_point))
        or projected.ndim != 2
        or projected.shape[1:] != (4,)
        or len(projected) % 2
    ):
        return None
    nearest: tuple[float, float] | None = None
    for first, second in projected.reshape((-1, 2, 4)):
        if not (
            _is_usable_projected_point(first)
            and _is_usable_projected_point(second)
        ):
            continue
        segment = second[:2] - first[:2]
        segment_length_squared = float(np.dot(segment, segment))
        if segment_length_squared <= 1e-12:
            parameter = 0.0
        else:
            parameter = float(
                np.clip(
                    np.dot(normalized_point - first[:2], segment)
                    / segment_length_squared,
                    0.0,
                    1.0,
                )
            )
        nearest_point = first[:2] + segment * parameter
        distance = float(np.linalg.norm(normalized_point - nearest_point))
        depth = float(first[2] + (second[2] - first[2]) * parameter)
        candidate = (distance, depth)
        if nearest is None or candidate < nearest:
            nearest = candidate
    return nearest


def _is_usable_projected_point(point: np.ndarray) -> bool:
    normalized = np.asarray(point, dtype=float)
    return bool(
        normalized.shape == (4,)
        and np.all(np.isfinite(normalized))
        and normalized[3] > 0.0
        and -1.0 <= normalized[2] <= 1.0
    )


def _rasterize_face_selection(
    projected_geometry: Sequence[tuple[np.ndarray, np.ndarray]],
    rectangle: QRect,
    *,
    xray: bool,
    cancel_event: threading.Event | None = None,
) -> set[int]:
    """Depth-test current-camera faces in a bounded logical-ID buffer.

    Candidate filtering and the capped buffer keep detached 4K views
    responsive. Screen-space barycentric weights use the same pixel centers
    for triangle coverage and interpolated depth, so hidden faces remain
    excluded without rounding the projected triangle boundary.
    """

    left = float(rectangle.left())
    top = float(rectangle.top())
    width = max(int(rectangle.width()), 1)
    height = max(int(rectangle.height()), 1)
    right = left + width
    bottom = top + height
    candidate_triangles: list[np.ndarray] = []
    candidate_face_indices: list[np.ndarray] = []

    for projected_vertices, faces in projected_geometry:
        if cancel_event is not None and cancel_event.is_set():
            return set()
        normalized_faces = np.asarray(faces, dtype=np.int64)
        if len(normalized_faces) == 0:
            continue
        triangles = np.asarray(
            projected_vertices[normalized_faces],
            dtype=float,
        )
        finite = np.all(np.isfinite(triangles), axis=(1, 2))
        minimum = np.min(triangles[:, :, :2], axis=1)
        maximum = np.max(triangles[:, :, :2], axis=1)
        overlaps_bounds = (
            (maximum[:, 0] >= left)
            & (minimum[:, 0] <= right)
            & (maximum[:, 1] >= top)
            & (minimum[:, 1] <= bottom)
        )
        depth = triangles[:, :, 2]
        overlaps_depth = (
            (np.max(depth, axis=1) >= -1.0 - 1e-6)
            & (np.min(depth, axis=1) <= 1.0 + 1e-6)
        )
        candidates = finite & overlaps_bounds & overlaps_depth
        if not np.any(candidates):
            continue
        candidate_triangles.append(triangles[candidates])
        candidate_face_indices.append(
            np.flatnonzero(candidates).astype(np.int64)
        )

    if not candidate_triangles:
        return set()
    triangles = np.concatenate(candidate_triangles, axis=0)
    logical_face_indices = np.concatenate(candidate_face_indices, axis=0)
    intersects = _triangles_intersect_screen_rectangle(
        triangles[:, :, :2],
        left=left,
        top=top,
        right=right,
        bottom=bottom,
    )
    triangles = triangles[intersects]
    logical_face_indices = logical_face_indices[intersects]
    if not len(triangles):
        return set()
    if cancel_event is not None and cancel_event.is_set():
        return set()
    if xray:
        return {int(index) for index in logical_face_indices}

    raster_scale = min(
        1.0,
        FACE_SELECTION_MAX_RASTER_DIMENSION / max(width, height),
    )
    raster_width = max(int(math.ceil(width * raster_scale)), 1)
    raster_height = max(int(math.ceil(height * raster_scale)), 1)
    depth_buffer = np.full(
        (raster_height, raster_width),
        np.inf,
        dtype=np.float32,
    )
    face_buffer = np.full((raster_height, raster_width), -1, dtype=np.int32)
    sample_x = np.arange(raster_width, dtype=np.float32) + 0.5
    sample_y = np.arange(raster_height, dtype=np.float32) + 0.5
    local_triangles = triangles[:, :, :2].copy()
    local_triangles[:, :, 0] = (
        local_triangles[:, :, 0] - left
    ) * raster_scale
    local_triangles[:, :, 1] = (
        local_triangles[:, :, 1] - top
    ) * raster_scale
    for candidate_index, triangle in enumerate(local_triangles):
        if cancel_event is not None and cancel_event.is_set():
            return set()
        triangle_depth = triangles[candidate_index, :, 2]
        first, second, third = triangle
        denominator = (
            (first[0] - third[0]) * (second[1] - third[1])
            - (second[0] - third[0]) * (first[1] - third[1])
        )
        if abs(float(denominator)) <= 1e-12:
            continue
        minimum_x = max(int(math.floor(np.min(triangle[:, 0]))), 0)
        maximum_x = min(
            int(math.ceil(np.max(triangle[:, 0]))),
            raster_width - 1,
        )
        minimum_y = max(int(math.floor(np.min(triangle[:, 1]))), 0)
        maximum_y = min(
            int(math.ceil(np.max(triangle[:, 1]))),
            raster_height - 1,
        )
        if minimum_x > maximum_x or minimum_y > maximum_y:
            continue
        local_y = slice(minimum_y, maximum_y + 1)
        local_x = slice(minimum_x, maximum_x + 1)
        sample_region_x = sample_x[local_x][np.newaxis, :]
        sample_region_y = sample_y[local_y][:, np.newaxis]
        first_weight = (
            (second[1] - third[1]) * (sample_region_x - third[0])
            + (third[0] - second[0]) * (sample_region_y - third[1])
        ) / denominator
        second_weight = (
            (third[1] - first[1]) * (sample_region_x - third[0])
            + (first[0] - third[0]) * (sample_region_y - third[1])
        ) / denominator
        third_weight = 1.0 - first_weight - second_weight
        inside = (
            (first_weight >= -1e-7)
            & (second_weight >= -1e-7)
            & (third_weight >= -1e-7)
        )
        interpolated_depth = (
            first_weight * triangle_depth[0]
            + second_weight * triangle_depth[1]
            + third_weight * triangle_depth[2]
        )
        inside &= (
            (interpolated_depth >= -1.0 - 1e-6)
            & (interpolated_depth <= 1.0 + 1e-6)
        )
        current_depth = depth_buffer[local_y, local_x]
        nearer = inside & (interpolated_depth < current_depth)
        if not np.any(nearer):
            continue
        current_depth[nearer] = interpolated_depth[nearer]
        face_buffer[local_y, local_x][nearer] = int(
            logical_face_indices[candidate_index]
        )
    return {
        int(face_index)
        for face_index in np.unique(face_buffer)
        if face_index >= 0
    }


def _run_face_rectangle_selection_task(
    task: _FaceRectangleSelectionTask,
    cancel_event: threading.Event,
    viewer_reference: weakref.ReferenceType[GlbViewerWidget],
) -> None:
    """Rasterize immutable input and queue its result back to the GUI."""

    if cancel_event.is_set():
        return
    try:
        selected = _rasterize_face_selection(
            task.projected_geometry,
            QRect(*task.rectangle),
            xray=task.xray,
            cancel_event=cancel_event,
        )
    except Exception:
        return
    if cancel_event.is_set():
        return
    viewer = viewer_reference()
    if viewer is None:
        return
    result = _FaceRectangleSelectionResult(
        request_revision=task.request_revision,
        geometry_revision=task.geometry_revision,
        face_indices=frozenset(selected),
    )
    try:
        viewer._face_rectangle_selection_completed.emit(result)
    except RuntimeError:
        # The Qt wrapper may disappear between weak-reference lookup and emit.
        return


def _triangles_intersect_screen_rectangle(
    triangles: np.ndarray,
    *,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> np.ndarray:
    """Use the triangle/axis-aligned-rectangle separating-axis test."""

    separated = np.zeros(len(triangles), dtype=bool)
    rectangle_corners = np.asarray(
        ((left, top), (right, top), (right, bottom), (left, bottom)),
        dtype=float,
    )
    for edge_index in range(3):
        edge = (
            triangles[:, (edge_index + 1) % 3]
            - triangles[:, edge_index]
        )
        axes = np.column_stack((-edge[:, 1], edge[:, 0]))
        axis_lengths = np.linalg.norm(axes, axis=1)
        usable = axis_lengths > 1e-12
        triangle_projection = np.einsum(
            "fvc,fc->fv",
            triangles,
            axes,
        )
        rectangle_projection = np.einsum(
            "vc,fc->fv",
            rectangle_corners,
            axes,
        )
        separated |= usable & (
            (np.max(triangle_projection, axis=1)
             < np.min(rectangle_projection, axis=1) - 1e-7)
            | (np.min(triangle_projection, axis=1)
               > np.max(rectangle_projection, axis=1) + 1e-7)
        )
    return ~separated


# ### Doorway preview helpers ###
def _normalize_doorway_preview_outline_positions(
    positions: object,
) -> np.ndarray:
    """Own finite paired XYZ positions for one doorway wireframe outline."""

    try:
        raw_positions = np.asarray(positions, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "A doorway preview outline requires numeric XYZ positions."
        ) from error
    if raw_positions.ndim != 2 or raw_positions.shape[1] != 3:
        raise ValueError(
            "A doorway preview outline requires paired XYZ positions shaped "
            "(N, 3)."
        )
    position_count = raw_positions.shape[0]
    if position_count < 2 or position_count % 2 != 0:
        raise ValueError(
            "A doorway preview outline requires an even number of at least "
            "two paired XYZ positions."
        )
    if not np.all(np.isfinite(raw_positions)):
        raise ValueError(
            "Doorway preview outline positions must all be finite."
        )
    return np.ascontiguousarray(raw_positions, dtype=np.float32).copy()


# ### Window editor helpers ###
def _world_point_tuple(point: object) -> tuple[float, float, float]:
    """Return one finite world point as a plain immutable tuple."""

    raw_point = np.asarray(point, dtype=float)
    if raw_point.shape != (3,) or not np.isfinite(raw_point).all():
        raise ValueError("A window pointer hit must contain finite XYZ coordinates.")
    return tuple(float(value) for value in raw_point)


def _get_nearest_fixed_surface_ray_hit(
    surfaces: tuple[FixedSurface, ...],
    ray_origin: object,
    ray_direction: object,
) -> tuple[FixedSurface, np.ndarray, float] | None:
    """Return the closest double-sided triangle hit without optional ray indexes."""

    origin = np.asarray(ray_origin, dtype=float)
    direction = np.asarray(ray_direction, dtype=float)
    if (
        origin.shape != (3,)
        or direction.shape != (3,)
        or not np.isfinite(origin).all()
        or not np.isfinite(direction).all()
    ):
        return None
    direction_length = float(np.linalg.norm(direction))
    if direction_length <= 1e-12:
        return None
    direction = direction / direction_length

    nearest: tuple[FixedSurface, np.ndarray, float] | None = None
    for surface in surfaces:
        hit = _get_nearest_triangle_ray_hit(
            surface.mesh,
            origin,
            direction,
        )
        if hit is None:
            continue
        hit_point, hit_distance = hit
        if nearest is None or hit_distance < nearest[2] - 1e-9:
            nearest = (surface, hit_point, hit_distance)
    return nearest


def _get_nearest_triangle_ray_hit(
    mesh: object,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """Intersect one ray with all mesh triangles using Moller-Trumbore."""

    vertices = np.asarray(getattr(mesh, "vertices", ()), dtype=float)
    faces = np.asarray(getattr(mesh, "faces", ()), dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or len(vertices) == 0
        or len(faces) == 0
    ):
        return None
    triangles = vertices[faces]
    first_edges = triangles[:, 1] - triangles[:, 0]
    second_edges = triangles[:, 2] - triangles[:, 0]
    repeated_direction = np.broadcast_to(ray_direction, second_edges.shape)
    cross_direction = np.cross(repeated_direction, second_edges)
    determinants = np.einsum("ij,ij->i", first_edges, cross_direction)
    usable = np.abs(determinants) > 1e-10
    if not np.any(usable):
        return None

    inverse_determinants = np.zeros_like(determinants)
    inverse_determinants[usable] = 1.0 / determinants[usable]
    origin_offsets = ray_origin[np.newaxis, :] - triangles[:, 0]
    first_coordinates = (
        np.einsum("ij,ij->i", origin_offsets, cross_direction)
        * inverse_determinants
    )
    offset_crosses = np.cross(origin_offsets, first_edges)
    second_coordinates = (
        np.einsum("j,ij->i", ray_direction, offset_crosses)
        * inverse_determinants
    )
    distances = (
        np.einsum("ij,ij->i", second_edges, offset_crosses)
        * inverse_determinants
    )
    usable &= first_coordinates >= -1e-9
    usable &= second_coordinates >= -1e-9
    usable &= first_coordinates + second_coordinates <= 1.0 + 1e-9
    usable &= distances >= 0.0
    hit_indices = np.flatnonzero(usable)
    if hit_indices.size == 0:
        return None
    hit_index = int(hit_indices[np.argmin(distances[hit_indices])])
    distance = float(distances[hit_index])
    return ray_origin + ray_direction * distance, distance


def _get_fixed_surface_wall_frame(
    surface: FixedSurface,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return wall origin, horizontal tangent, and plane normal."""

    if surface.wall_start_world is None or surface.wall_end_world is None:
        raise ValueError("The selected wall has no stable placement frame.")
    wall_start = np.asarray(surface.wall_start_world, dtype=float)
    wall_end = np.asarray(surface.wall_end_world, dtype=float)
    if (
        wall_start.shape != (3,)
        or wall_end.shape != (3,)
        or not np.isfinite(wall_start).all()
        or not np.isfinite(wall_end).all()
    ):
        raise ValueError("The selected wall has an invalid placement frame.")
    wall_axis = wall_end - wall_start
    wall_axis[2] = 0.0
    wall_width = float(np.linalg.norm(wall_axis))
    if wall_width <= 1e-10:
        raise ValueError("The selected wall is too narrow for a window.")
    tangent = wall_axis / wall_width
    plane_normal = np.cross(tangent, np.asarray((0.0, 0.0, 1.0)))
    normal_length = float(np.linalg.norm(plane_normal))
    if normal_length <= 1e-10:
        raise ValueError("The selected wall has an invalid placement plane.")
    return wall_start, tangent, plane_normal / normal_length


def _get_fixed_surface_plane_normal(surface: FixedSurface) -> np.ndarray:
    """Return a deterministic unit normal for a semantic wall."""

    try:
        return _get_fixed_surface_wall_frame(surface)[2]
    except ValueError:
        vertices = np.asarray(surface.mesh.vertices, dtype=float)
        faces = np.asarray(surface.mesh.faces, dtype=np.int64)
        if len(faces) == 0:
            return np.asarray((0.0, 1.0, 0.0), dtype=float)
        triangle = vertices[faces[0]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = float(np.linalg.norm(normal))
        if length <= 1e-10:
            return np.asarray((0.0, 1.0, 0.0), dtype=float)
        return normal / length


def _intersect_ray_with_fixed_surface_plane(
    surface: FixedSurface,
    ray_origin: object,
    ray_direction: object,
) -> tuple[float, float, float] | None:
    """Intersect a camera ray with the infinite plane of one selected wall."""

    try:
        plane_point, _tangent, plane_normal = _get_fixed_surface_wall_frame(
            surface
        )
    except ValueError:
        return None
    origin = np.asarray(ray_origin, dtype=float)
    direction = np.asarray(ray_direction, dtype=float)
    if origin.shape != (3,) or direction.shape != (3,):
        return None
    denominator = float(np.dot(plane_normal, direction))
    if abs(denominator) <= 1e-10:
        return None
    distance = float(np.dot(plane_point - origin, plane_normal) / denominator)
    if not math.isfinite(distance) or distance < 0.0:
        return None
    return _world_point_tuple(origin + direction * distance)


def _build_wall_rectangle_corners(
    surface: FixedSurface,
    first_world: object,
    second_world: object,
) -> tuple[tuple[float, float, float], ...] | None:
    """Build raw drag corners for valid and invalid live feedback."""

    try:
        _origin, tangent, _normal = _get_fixed_surface_wall_frame(surface)
        first = np.asarray(_world_point_tuple(first_world), dtype=float)
        second = np.asarray(_world_point_tuple(second_world), dtype=float)
    except ValueError:
        return None
    horizontal_delta = tangent * float(np.dot(second - first, tangent))
    vertical_delta = np.asarray((0.0, 0.0, second[2] - first[2]), dtype=float)
    corners = (
        first,
        first + horizontal_delta,
        first + horizontal_delta + vertical_delta,
        first + vertical_delta,
    )
    return tuple(_world_point_tuple(corner) for corner in corners)


def _build_validated_wall_window_placement(
    surface: FixedSurface,
    first_world: object,
    second_world: object,
) -> WallWindowPlacement:
    """Keep the geometry validation call isolated for UI tests."""

    return build_wall_window_placement(surface, first_world, second_world)


def _get_validated_wall_window_world_corners(
    surface: FixedSurface,
    placement: object,
) -> tuple[tuple[float, float, float], ...]:
    """Resolve validated immutable placement bounds into preview corners."""

    if not isinstance(placement, WallWindowPlacement):
        raise TypeError("A Canvas window preview requires a wall placement.")
    return get_wall_window_world_corners(surface, placement)


def _build_fixed_surface_boundary_line_positions(
    surface: FixedSurface,
) -> np.ndarray | None:
    """Return paired boundary vertices for a visible semantic selection."""

    mesh = surface.mesh.copy()
    try:
        mesh.merge_vertices()
    except (AttributeError, TypeError, ValueError):
        pass
    vertices = np.asarray(mesh.vertices, dtype=float)
    unique_edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    inverse_edges = np.asarray(mesh.edges_unique_inverse, dtype=np.int64)
    if len(vertices) == 0 or len(unique_edges) == 0:
        return None
    edge_counts = np.bincount(inverse_edges, minlength=len(unique_edges))
    boundary_edges = unique_edges[edge_counts == 1]
    if len(boundary_edges) == 0:
        boundary_edges = unique_edges
    return np.asarray(vertices[boundary_edges].reshape(-1, 3), dtype=float)


def _offset_points_toward_camera(
    points: object,
    plane_normal: object,
    camera_position: object,
    distance: float,
) -> np.ndarray:
    """Offset coplanar feedback toward the current camera to avoid flicker."""

    raw_points = np.asarray(points, dtype=float)
    normal = np.asarray(plane_normal, dtype=float)
    if isinstance(camera_position, QVector3D):
        camera = np.asarray(
            (camera_position.x(), camera_position.y(), camera_position.z()),
            dtype=float,
        )
    else:
        camera = np.asarray(camera_position, dtype=float)
    if (
        raw_points.ndim != 2
        or raw_points.shape[1:] != (3,)
        or normal.shape != (3,)
        or camera.shape != (3,)
    ):
        return raw_points
    center = np.mean(raw_points, axis=0)
    sign = 1.0 if float(np.dot(camera - center, normal)) >= 0.0 else -1.0
    return raw_points + normal[np.newaxis, :] * sign * abs(float(distance))


# ### Transform helpers ###
def _numpy_transform_to_qt(transform: object) -> Transform3D:
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Placed-object transforms must be finite 4 by 4 matrices.")
    return Transform3D(matrix.tolist())


def _transform_point(transform: object, point: object) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float)
    coordinate = np.asarray(point, dtype=float)
    if matrix.shape != (4, 4) or coordinate.shape != (3,):
        raise ValueError("A transformed point requires a 4 by 4 matrix and XYZ.")
    transformed = matrix @ np.append(coordinate, 1.0)
    if abs(float(transformed[3])) <= 1e-12:
        raise ValueError("A transformed point cannot have a zero homogeneous W.")
    return np.asarray(transformed[:3] / transformed[3], dtype=float)


def _get_render_group_world_pivot(
    group: _PlacedObjectRenderGroup,
) -> np.ndarray:
    local_pivot = _transform_point(
        np.linalg.inv(group.preview.placement_transform),
        group.preview.world_position,
    )
    return _transform_point(group.current_transform, local_pivot)


def _build_pivoted_world_transform(
    world_position: object,
    world_rotation: object,
    local_pivot: object,
) -> np.ndarray:
    position = np.asarray(world_position, dtype=float)
    rotation = np.asarray(world_rotation, dtype=float)
    pivot = np.asarray(local_pivot, dtype=float)
    if position.shape != (3,) or rotation.shape != (3, 3) or pivot.shape != (3,):
        raise ValueError("A pivoted object transform requires XYZ and 3D rotation.")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = position - rotation @ pivot
    return transform


def _rotation_matrix_to_degrees(
    rotation: object,
) -> tuple[float, float, float]:
    matrix = np.eye(4, dtype=float)
    raw_rotation = np.asarray(rotation, dtype=float)
    if raw_rotation.shape != (3, 3) or not np.all(np.isfinite(raw_rotation)):
        raise ValueError("Placed-object rotations must be finite 3 by 3 matrices.")
    matrix[:3, :3] = raw_rotation
    angles = trimesh.transformations.euler_from_matrix(matrix, axes="sxyz")
    return tuple(float(math.degrees(angle)) for angle in angles)


def _build_local_symmetric_preview_meshes(
    preview: PreviewPlacedObject,
) -> tuple[object, ...]:
    orientation = preview.symmetric_preview_orientation
    plane_coordinate = preview.symmetric_preview_plane_coordinate
    if orientation is None or plane_coordinate is None:
        return ()
    axis = SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION[orientation]
    mirrored_meshes: list[object] = []
    for retained_mesh in preview.meshes:
        mirrored_mesh = retained_mesh.copy()
        vertices = np.asarray(mirrored_mesh.vertices, dtype=float).copy()
        vertices[:, axis] = float(plane_coordinate) * 2.0 - vertices[:, axis]
        mirrored_mesh.vertices = vertices
        mirrored_mesh.faces = np.asarray(
            mirrored_mesh.faces,
            dtype=np.int64,
        )[:, (0, 2, 1)]
        mirrored_meshes.append(mirrored_mesh)
    return tuple(mirrored_meshes)


def _get_combined_mesh_bounds(
    meshes: object,
) -> np.ndarray | None:
    vertices = [
        raw_vertices
        for mesh in meshes
        if (
            (raw_vertices := np.asarray(
                getattr(mesh, "vertices", ()),
                dtype=float,
            )).ndim
            == 2
            and raw_vertices.shape[1:] == (3,)
            and len(raw_vertices)
            and np.all(np.isfinite(raw_vertices))
        )
    ]
    if not vertices:
        return None
    combined = np.vstack(vertices)
    return np.asarray((np.min(combined, axis=0), np.max(combined, axis=0)))


def _build_bounds_line_positions(bounds: object) -> np.ndarray:
    raw_bounds = np.asarray(bounds, dtype=float)
    if raw_bounds.shape != (2, 3) or not np.all(np.isfinite(raw_bounds)):
        raise ValueError("Object-selection bounds must contain finite XYZ limits.")
    minimum, maximum = raw_bounds
    corners = np.asarray(
        [
            (x, y, z)
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=float,
    )
    edges = (
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    )
    return np.asarray(
        [corners[index] for edge in edges for index in edge],
        dtype=float,
    )


def _build_rotation_ring_positions(
    pivot: object,
    axis_index: int,
    radius: float,
) -> np.ndarray:
    center = np.asarray(pivot, dtype=float)
    if center.shape != (3,) or axis_index not in {0, 1, 2}:
        raise ValueError("Rotation rings require an XYZ center and axis.")
    first_axis = (axis_index + 1) % 3
    second_axis = (axis_index + 2) % 3
    angles = np.linspace(
        0.0,
        2.0 * math.pi,
        TRANSFORM_GIZMO_RING_POINT_COUNT + 1,
    )
    positions = np.tile(center, (len(angles), 1))
    positions[:, first_axis] += np.cos(angles) * float(radius)
    positions[:, second_axis] += np.sin(angles) * float(radius)
    return positions


def _normalize_vector(vector: object) -> np.ndarray | None:
    raw_vector = np.asarray(vector, dtype=float)
    if raw_vector.shape != (3,) or not np.all(np.isfinite(raw_vector)):
        return None
    length = float(np.linalg.norm(raw_vector))
    if length <= 1e-12:
        return None
    return raw_vector / length


def _normalize_ray(
    ray_origin: object,
    ray_direction: object,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    origin = np.asarray(ray_origin, dtype=float)
    direction = _normalize_vector(ray_direction)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        return None, None
    return origin, direction


def _intersect_ray_with_plane(
    ray_origin: object,
    ray_direction: object,
    plane_point: object,
    plane_normal: object,
) -> np.ndarray | None:
    """Return the forward ray hit on one infinite plane."""

    origin, direction = _normalize_ray(ray_origin, ray_direction)
    point = np.asarray(plane_point, dtype=float)
    normal = _normalize_vector(plane_normal)
    if (
        origin is None
        or direction is None
        or point.shape != (3,)
        or not np.all(np.isfinite(point))
        or normal is None
    ):
        return None
    denominator = float(np.dot(normal, direction))
    if abs(denominator) <= 1e-10:
        return None
    distance = float(np.dot(point - origin, normal) / denominator)
    if not math.isfinite(distance) or distance < 0.0:
        return None
    return origin + direction * distance


def _build_axis_drag_plane_normal(
    axis: object,
    camera_direction: object,
) -> np.ndarray:
    """Build a stable camera-facing plane that still contains an axis."""

    normalized_axis = _normalize_vector(axis)
    normalized_direction = _normalize_vector(camera_direction)
    if normalized_axis is None or normalized_direction is None:
        raise ValueError("Axis dragging requires finite non-zero directions.")
    projected = normalized_direction - normalized_axis * float(
        np.dot(normalized_direction, normalized_axis)
    )
    normal = _normalize_vector(projected)
    if normal is not None:
        return normal
    fallback_axis = np.eye(3, dtype=float)[
        int(np.argmin(np.abs(normalized_axis)))
    ]
    fallback = fallback_axis - normalized_axis * float(
        np.dot(fallback_axis, normalized_axis)
    )
    normal = _normalize_vector(fallback)
    if normal is None:
        raise ValueError("A stable axis drag plane could not be built.")
    return normal


def _get_signed_rotation_degrees(
    axis: object,
    first_vector: object,
    second_vector: object,
) -> float:
    normalized_axis = _normalize_vector(axis)
    first = _normalize_vector(first_vector)
    second = _normalize_vector(second_vector)
    if normalized_axis is None or first is None or second is None:
        raise ValueError("Rotation angles require finite non-zero vectors.")
    sine = float(np.dot(normalized_axis, np.cross(first, second)))
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return math.degrees(math.atan2(sine, cosine))


def _get_ray_segment_distance(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> float | None:
    segment = segment_end - segment_start
    segment_length_squared = float(np.dot(segment, segment))
    if segment_length_squared <= 1e-12:
        return None
    offset = ray_origin - segment_start
    ray_segment_dot = float(np.dot(ray_direction, segment))
    ray_offset_dot = float(np.dot(ray_direction, offset))
    segment_offset_dot = float(np.dot(segment, offset))
    denominator = segment_length_squared - ray_segment_dot**2
    if abs(denominator) <= 1e-12:
        segment_parameter = float(
            np.clip(segment_offset_dot / segment_length_squared, 0.0, 1.0)
        )
    else:
        segment_parameter = float(
            np.clip(
                (segment_offset_dot - ray_segment_dot * ray_offset_dot)
                / denominator,
                0.0,
                1.0,
            )
        )
    ray_parameter = max(
        0.0,
        ray_segment_dot * segment_parameter - ray_offset_dot,
    )
    segment_parameter = float(
        np.clip(
            (
                segment_offset_dot
                + ray_segment_dot * ray_parameter
            )
            / segment_length_squared,
            0.0,
            1.0,
        )
    )
    ray_point = ray_origin + ray_direction * ray_parameter
    segment_point = segment_start + segment * segment_parameter
    return float(np.linalg.norm(ray_point - segment_point))


def _get_nearest_preview_placed_object_ray_hit(
    targets: tuple[PreviewPlacedObject, ...],
    ray_origin: object,
    ray_direction: object,
) -> tuple[PreviewPlacedObject, np.ndarray, float] | None:
    """Pick retained or fading-half geometry for the nearest placed object."""

    origin, direction = _normalize_ray(ray_origin, ray_direction)
    if origin is None or direction is None:
        return None
    nearest: tuple[PreviewPlacedObject, np.ndarray, float] | None = None
    for target in targets:
        if not isinstance(target, PreviewPlacedObject):
            continue
        meshes = (
            *target.meshes,
            *_build_local_symmetric_preview_meshes(target),
        )
        for local_mesh in meshes:
            world_mesh = local_mesh.copy()
            world_mesh.apply_transform(target.placement_transform)
            hit = _get_nearest_triangle_ray_hit(
                world_mesh,
                origin,
                direction,
            )
            if hit is None:
                continue
            hit_point, hit_distance = hit
            if nearest is None or hit_distance < nearest[2] - 1e-9:
                nearest = (target, hit_point, hit_distance)
    return nearest


def _build_textured_wall_transform(
    textured_wall: PreviewTexturedWall,
    offset_sign: float,
) -> Transform3D:
    texture_width = max(1.0, float(textured_wall.texture_rgba.shape[0]))
    texture_height = max(1.0, float(textured_wall.texture_rgba.shape[1]))
    start_point = np.asarray(textured_wall.start_point, dtype=float)
    end_point = np.asarray(textured_wall.end_point, dtype=float)
    wall_vector = end_point - start_point
    wall_length = float(np.linalg.norm(wall_vector[:2]))
    if wall_length <= 1e-6:
        return Transform3D()

    wall_normal = np.array(
        [-wall_vector[1] / wall_length, wall_vector[0] / wall_length, 0.0],
        dtype=float,
    )
    if offset_sign >= 0.0:
        origin = start_point + wall_normal * TEXTURE_PREVIEW_OFFSET_METERS
        z_axis = wall_normal
    else:
        origin = start_point - wall_normal * TEXTURE_PREVIEW_OFFSET_METERS
        z_axis = -wall_normal

    x_axis = wall_vector / texture_width
    y_axis = np.array(
        [0.0, 0.0, float(textured_wall.height_meters) / texture_height],
        dtype=float,
    )

    return Transform3D(
        [
            [x_axis[0], y_axis[0], z_axis[0], origin[0]],
            [x_axis[1], y_axis[1], z_axis[1], origin[1]],
            [x_axis[2], y_axis[2], z_axis[2], origin[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _get_point_distance(first_point: QPointF, second_point: QPointF) -> float:
    delta = first_point - second_point
    return float((delta.x() ** 2 + delta.y() ** 2) ** 0.5)


# ### Navigation helpers ###
def _normalize_navigation_mode(value: object) -> str:
    """Validate the persisted/user-facing navigation mode vocabulary."""

    if not isinstance(value, str):
        raise ValueError("3D navigation mode must be a string.")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in VIEWER_NAVIGATION_MODES:
        supported = ", ".join(sorted(VIEWER_NAVIGATION_MODES))
        raise ValueError(
            f"Unknown 3D navigation mode {value!r}; expected one of: {supported}."
        )
    return normalized


def _first_person_movement_keys() -> frozenset[int]:
    return frozenset(
        (
            Qt.Key.Key_Z,
            Qt.Key.Key_Q,
            Qt.Key.Key_S,
            Qt.Key.Key_D,
            Qt.Key.Key_R,
            Qt.Key.Key_F,
        )
    )


def _point_is_near(delta: QPointF, tolerance: float) -> bool:
    return abs(delta.x()) <= tolerance and abs(delta.y()) <= tolerance


# ### Color helpers ###
def _get_mesh_face_colors(
    mesh,
    faces: np.ndarray,
) -> np.ndarray:
    """Return procedural mesh colors while preserving the house fallback."""

    visual_kind = getattr(getattr(mesh, "visual", None), "kind", None)
    if visual_kind == "face":
        raw_colors = np.asarray(mesh.visual.face_colors)
    elif visual_kind == "vertex":
        vertex_colors = np.asarray(mesh.visual.vertex_colors, dtype=float)
        raw_colors = vertex_colors[faces].mean(axis=1)
    elif visual_kind == "texture":
        try:
            baked_colors = mesh.visual.to_color()
            raw_colors = np.asarray(baked_colors.vertex_colors, dtype=float)[
                faces
            ].mean(axis=1)
        except Exception:
            return np.tile(FACE_COLOR, (faces.shape[0], 1))
    else:
        return np.tile(FACE_COLOR, (faces.shape[0], 1))

    if raw_colors.ndim != 2 or raw_colors.shape[0] != faces.shape[0]:
        return np.tile(FACE_COLOR, (faces.shape[0], 1))
    if raw_colors.shape[1] == 3:
        alpha_column = np.full((raw_colors.shape[0], 1), 255.0)
        raw_colors = np.hstack((raw_colors, alpha_column))
    if raw_colors.shape[1] != 4:
        return np.tile(FACE_COLOR, (faces.shape[0], 1))

    normalized = np.asarray(raw_colors, dtype=float)
    if normalized.size and float(np.nanmax(normalized)) > 1.0:
        normalized /= 255.0
    if not np.all(np.isfinite(normalized)):
        return np.tile(FACE_COLOR, (faces.shape[0], 1))
    return np.clip(normalized, 0.0, 1.0)


# ### Texture helpers ###
def _mirror_preview_vertices(
    vertices: np.ndarray,
    orientation: str,
    plane_coordinate: float,
) -> np.ndarray:
    """Reflect copied Z-up vertices around the persisted global cut plane."""

    mirrored = np.ascontiguousarray(vertices, dtype=np.float32).copy()
    axis = SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION[orientation]
    mirrored[:, axis] = float(plane_coordinate) * 2.0 - mirrored[:, axis]
    return mirrored


def _mirror_texture_mesh_data(
    texture_mesh_data: TextureMeshData,
    orientation: str,
    plane_coordinate: float,
) -> TextureMeshData:
    """Reflect face-expanded textured geometry and reverse its winding."""

    vertices = _mirror_preview_vertices(
        texture_mesh_data.vertices,
        orientation,
        plane_coordinate,
    ).reshape((-1, 3, 3))
    normals = np.ascontiguousarray(
        texture_mesh_data.normals,
        dtype=np.float32,
    ).copy()
    axis = SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION[orientation]
    normals[:, axis] *= -1.0
    normals = normals.reshape((-1, 3, 3))
    texture_coordinates = np.ascontiguousarray(
        texture_mesh_data.texture_coordinates,
        dtype=np.float32,
    ).reshape((-1, 3, 2))
    reverse_winding = (0, 2, 1)
    return TextureMeshData(
        vertices=np.ascontiguousarray(
            vertices[:, reverse_winding, :].reshape((-1, 3))
        ),
        normals=np.ascontiguousarray(
            normals[:, reverse_winding, :].reshape((-1, 3))
        ),
        texture_coordinates=np.ascontiguousarray(
            texture_coordinates[:, reverse_winding, :].reshape((-1, 2))
        ),
        texture_rgba=texture_mesh_data.texture_rgba,
    )


def _build_texture_mesh_data(mesh) -> TextureMeshData | None:
    """Build face-expanded UV geometry for an embedded base-color texture."""

    visual = getattr(mesh, "visual", None)
    if getattr(visual, "kind", None) != "texture":
        return None

    texture_rgba = _get_base_color_texture_rgba(visual)
    texture_coordinates = getattr(visual, "uv", None)
    faces = np.asarray(getattr(mesh, "faces", ()), dtype=np.int64)
    vertices = np.asarray(getattr(mesh, "vertices", ()), dtype=np.float32)
    if (
        texture_rgba is None
        or texture_coordinates is None
        or faces.ndim != 2
        or faces.shape[1] != 3
        or vertices.ndim != 2
        or vertices.shape[1] != 3
    ):
        return None

    texture_coordinates = np.asarray(texture_coordinates, dtype=np.float32)
    if (
        texture_coordinates.ndim != 2
        or texture_coordinates.shape[1] != 2
        or len(texture_coordinates) != len(vertices)
        or faces.size == 0
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        return None

    try:
        face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
    except Exception:
        return None
    if face_normals.shape != faces.shape:
        return None

    face_vertex_indices = faces.reshape(-1)
    return TextureMeshData(
        vertices=np.ascontiguousarray(vertices[face_vertex_indices]),
        normals=np.ascontiguousarray(np.repeat(face_normals, 3, axis=0)),
        texture_coordinates=np.ascontiguousarray(
            texture_coordinates[face_vertex_indices]
        ),
        texture_rgba=_limit_texture_preview_size(texture_rgba),
    )


def _get_base_color_texture_rgba(visual) -> np.ndarray | None:
    material = getattr(visual, "material", None)
    texture = getattr(material, "baseColorTexture", None)
    if texture is None:
        texture = getattr(material, "image", None)
    if texture is None:
        return None

    try:
        if hasattr(texture, "convert"):
            rgba = np.asarray(texture.convert("RGBA"), dtype=np.uint8)
        else:
            raw_texture = np.asarray(texture)
            if raw_texture.ndim == 1:
                with Image.open(BytesIO(raw_texture.tobytes())) as image:
                    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            else:
                rgba = np.asarray(raw_texture, dtype=np.uint8)
    except Exception:
        return None

    if rgba.ndim == 2:
        rgba = np.repeat(rgba[:, :, np.newaxis], 3, axis=2)
    if rgba.ndim != 3 or rgba.shape[2] not in {3, 4}:
        return None
    if rgba.shape[2] == 3:
        alpha = np.full(rgba.shape[:2] + (1,), 255, dtype=np.uint8)
        rgba = np.concatenate((rgba, alpha), axis=2)
    if rgba.shape[0] <= 0 or rgba.shape[1] <= 0:
        return None
    return np.ascontiguousarray(rgba, dtype=np.uint8)


def _limit_texture_preview_size(texture_rgba: np.ndarray) -> np.ndarray:
    height, width = texture_rgba.shape[:2]
    largest_dimension = max(height, width)
    if largest_dimension <= MAX_TEXTURE_PREVIEW_DIMENSION:
        return texture_rgba

    scale = MAX_TEXTURE_PREVIEW_DIMENSION / float(largest_dimension)
    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    resized = Image.fromarray(texture_rgba, mode="RGBA").resize(
        resized_size,
        Image.Resampling.LANCZOS,
    )
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


# ### Lighting helpers ###
def _build_ambient_shader(ambient_light_intensity: float):
    """Create an isolated shader so settings do not leak between viewers."""

    shader_name = "housemaker_ambient_lit"
    shader = gl_shaders.ShaderProgram(
        shader_name,
        [
            gl_shaders.VertexShader(AMBIENT_LIT_VERTEX_SHADER),
            gl_shaders.FragmentShader(AMBIENT_LIT_FRAGMENT_SHADER),
        ],
        uniforms={
            "u_ambient_light": [
                _normalize_ambient_light_intensity(ambient_light_intensity)
            ]
        },
    )
    gl_shaders.ShaderProgram.names.pop(shader_name, None)
    return shader


def _normalize_ambient_light_intensity(intensity: float) -> float:
    normalized_intensity = float(intensity)
    if not np.isfinite(normalized_intensity):
        raise ValueError("Ambient light intensity must be finite.")
    return min(
        max(normalized_intensity, MIN_AMBIENT_LIGHT_INTENSITY),
        MAX_AMBIENT_LIGHT_INTENSITY,
    )


def _normalize_preview_opacity(opacity: float) -> float:
    normalized_opacity = float(opacity)
    if not math.isfinite(normalized_opacity):
        raise ValueError("Preview opacity must be finite.")
    return min(max(normalized_opacity, 0.0), 1.0)


# ### OpenGL helpers ###
def _upload_array_buffer(values: np.ndarray) -> int:
    buffer_id = int(GL.glGenBuffers(1))
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
    GL.glBufferData(
        GL.GL_ARRAY_BUFFER,
        values.nbytes,
        values,
        GL.GL_STATIC_DRAW,
    )
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    return buffer_id


def _upload_texture(texture_rgba: np.ndarray, *, repeat: bool = False) -> int:
    texture_id = int(GL.glGenTextures(1))
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_WRAP_S,
        GL.GL_REPEAT if repeat else GL.GL_CLAMP_TO_EDGE,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_WRAP_T,
        GL.GL_REPEAT if repeat else GL.GL_CLAMP_TO_EDGE,
    )
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGBA,
        texture_rgba.shape[1],
        texture_rgba.shape[0],
        0,
        GL.GL_RGBA,
        GL.GL_UNSIGNED_BYTE,
        np.ascontiguousarray(np.flipud(texture_rgba)),
    )
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return texture_id


def _upload_mask_texture(mask: np.ndarray) -> int:
    """Upload a one-channel UV edit mask for shader-side highlighting."""

    texture_id = int(GL.glGenTextures(1))
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_WRAP_S,
        GL.GL_CLAMP_TO_EDGE,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_WRAP_T,
        GL.GL_CLAMP_TO_EDGE,
    )
    _replace_uploaded_mask_texture(texture_id, mask)
    return texture_id


def _replace_uploaded_mask_texture(
    texture_id: int,
    mask: np.ndarray,
) -> None:
    """Replace one mask texture while its OpenGL context is current."""

    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, int(texture_id))
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_LUMINANCE,
        mask.shape[1],
        mask.shape[0],
        0,
        GL.GL_LUMINANCE,
        GL.GL_UNSIGNED_BYTE,
        np.ascontiguousarray(np.flipud(mask)),
    )
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glActiveTexture(GL.GL_TEXTURE0)


def _bind_float_attribute(
    shader_program: int,
    name: str,
    buffer_id: int | None,
    component_count: int,
    enabled_locations: list[int],
) -> None:
    if buffer_id is None:
        return
    location = int(GL.glGetAttribLocation(shader_program, name))
    if location < 0:
        return
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
    GL.glVertexAttribPointer(
        location,
        component_count,
        GL.GL_FLOAT,
        False,
        0,
        None,
    )
    GL.glEnableVertexAttribArray(location)
    enabled_locations.append(location)


def _set_matrix_uniform(
    shader_program: int,
    name: str,
    values: np.ndarray,
    dimension: int,
) -> None:
    location = int(GL.glGetUniformLocation(shader_program, name))
    if location < 0:
        return
    if dimension == 4:
        GL.glUniformMatrix4fv(location, 1, False, values)
    else:
        GL.glUniformMatrix3fv(location, 1, False, values)


def _set_float_uniform(shader_program: int, name: str, value: float) -> None:
    location = int(GL.glGetUniformLocation(shader_program, name))
    if location >= 0:
        GL.glUniform1f(location, float(value))


def _set_integer_uniform(shader_program: int, name: str, value: int) -> None:
    location = int(GL.glGetUniformLocation(shader_program, name))
    if location >= 0:
        GL.glUniform1i(location, int(value))
