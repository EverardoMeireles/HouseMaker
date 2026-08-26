# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph.opengl as gl
import trimesh
from PySide6.QtCore import QPointF, QTimer, Qt, Signal
from PySide6.QtGui import QCursor, QKeyEvent, QMouseEvent, QVector3D
from PySide6.QtWidgets import QLabel, QStackedLayout, QWidget
from PIL import Image

from housemaker.camera_models import CameraPose, InitialFirstPersonCamera
from housemaker.glb import GeneratedModel, PreviewTexturedSurface
from housemaker.generation_state import (
    MASK_MODE_ERASE,
    MASK_MODE_PAINT,
    MaskPoint,
    MaskStroke,
)
from housemaker.models import LevelData
from housemaker.surface_geometry import (
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    FixedSurface,
    build_fixed_surfaces,
    get_combined_surface_area,
)
from housemaker.viewer import (
    DEFAULT_AMBIENT_LIGHT_INTENSITY as CANVAS_AMBIENT_LIGHT_INTENSITY,
    EDGE_COLOR,
    FACE_COLOR,
    TextureMeshData,
    TexturedMeshItem,
    _WireframeOverlayMeshItem,
    _build_ambient_shader,
    _build_texture_mesh_data,
    _build_textured_wall_transform,
    _get_mesh_face_colors,
    _limit_texture_preview_size,
)


# ### Constants ###
DEFAULT_FIRST_PERSON_HEIGHT_METERS = 1.65
DEFAULT_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND = 2.5
DEFAULT_MOUSE_LOOK_SENSITIVITY_DEGREES = 0.16
DEFAULT_TEXTURE_WORLD_SIZE_METERS = 2.0
DEFAULT_AMBIENT_LIGHT_INTENSITY = CANVAS_AMBIENT_LIGHT_INTENSITY
DEFAULT_TEXTURE_INPAINT_BRUSH_RADIUS_PIXELS = 24
FIRST_PERSON_UPDATE_INTERVAL_MILLISECONDS = 16
FIRST_PERSON_LOOK_DISTANCE_METERS = 1.0
MAX_FIRST_PERSON_PITCH_DEGREES = 89.0
PICK_REGION_SIZE_PIXELS = 7
RAY_INTERSECTION_EPSILON = 1e-7
SURFACE_EDGE_COLOR = (0.14, 0.16, 0.20, 0.8)
SELECTED_SURFACE_EDGE_COLOR = (1.0, 0.58, 0.08, 1.0)
SELECTED_SURFACE_OUTLINE_RADIUS_METERS = 0.012
SELECTED_SURFACE_OUTLINE_SECTIONS = 8
OUTLINE_EDGE_ROUNDING_DECIMALS = 8
SURFACE_FACE_COLORS = {
    SURFACE_TYPE_WALL: np.array((0.72, 0.75, 0.80, 1.0), dtype=float),
    SURFACE_TYPE_FLOOR: np.array((0.44, 0.48, 0.54, 1.0), dtype=float),
    SURFACE_TYPE_CEILING: np.array((0.82, 0.83, 0.86, 1.0), dtype=float),
}
TEXTURE_MASK_OVERLAY_RGBA = np.array((255, 126, 24, 255), dtype=np.uint8)
TEXTURE_MASK_OVERLAY_STRENGTH = 0.68


# ### Render models ###
@dataclass
class SurfaceRenderItems:
    face_item: gl.GLMeshItem
    outline_item: gl.GLMeshItem | None = None
    texture_item: RepeatingTexturedMeshItem | None = None
    additional_texture_items: tuple[RepeatingTexturedMeshItem, ...] = ()


@dataclass
class CanvasSceneRenderItems:
    """Canvas-compatible background items hosted by the semantic viewer."""

    grid_item: gl.GLGridItem | None = None
    mesh_item: gl.GLMeshItem | None = None
    textured_mesh_item: TexturedMeshItem | None = None
    legacy_wall_items: tuple[gl.GLImageItem, ...] = ()


@dataclass(frozen=True)
class SurfaceTextureHit:
    """Nearest semantic surface hit and its repeating texture-space point."""

    surface_id: str
    world_position: tuple[float, float, float]
    texture_point: MaskPoint


@dataclass(frozen=True)
class _MeshRayHit:
    distance: float
    face_index: int
    is_back_facing: bool


def _group_preview_surfaces_by_id(
    preview_surfaces: Iterable[PreviewTexturedSurface],
) -> dict[str, tuple[PreviewTexturedSurface, ...]]:
    grouped: dict[str, list[PreviewTexturedSurface]] = {}
    for preview_surface in preview_surfaces:
        grouped.setdefault(preview_surface.surface_id, []).append(
            preview_surface
        )
    return {
        surface_id: tuple(surface_values)
        for surface_id, surface_values in grouped.items()
    }


# ### Textured render item ###
class RepeatingTexturedMeshItem(TexturedMeshItem):
    """Texture renderer whose world-scale UVs repeat beyond the first tile."""

    def __init__(
        self,
        texture_mesh_data: TextureMeshData,
        ambient_light_intensity: float,
        *,
        double_sided: bool = False,
    ) -> None:
        # Keep the complete shared texture renderer, including its edit-mask
        # sampler.  The old custom resource upload omitted that sampler, so the
        # textured draw failed after the fallback surface face was hidden.
        super().__init__(
            texture_mesh_data,
            ambient_light_intensity,
            texture_repeat=True,
            double_sided=double_sided,
        )


# ### First-person view ###
class SurfaceFirstPersonViewWidget(gl.GLViewWidget):
    """Captured-mouse ZQSD plus R/F first-person view with item picking."""

    items_clicked = Signal(object, object)
    surface_pick_requested = Signal(object, object)
    camera_pose_changed = Signal(object)
    first_person_active_changed = Signal(bool)
    inpaint_pointer_pressed = Signal(object)
    inpaint_pointer_moved = Signal(object)
    inpaint_pointer_released = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._camera_pose = CameraPose(z=DEFAULT_FIRST_PERSON_HEIGHT_METERS)
        self._first_person_active = False
        self._inpaint_enabled = False
        self._pressed_movement_keys: set[int] = set()
        self._ignore_center_mouse_move = False
        self._mouse_look_sensitivity = DEFAULT_MOUSE_LOOK_SENSITIVITY_DEGREES
        self._move_speed = DEFAULT_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND
        self._movement_timer = QTimer(self)
        self._movement_timer.setInterval(FIRST_PERSON_UPDATE_INTERVAL_MILLISECONDS)
        self._movement_timer.timeout.connect(self._advance_from_pressed_keys)
        self.set_camera_pose(self._camera_pose, emit_signal=False)

    @property
    def is_first_person_active(self) -> bool:
        return self._first_person_active

    @property
    def is_inpaint_enabled(self) -> bool:
        return self._inpaint_enabled

    def set_inpaint_enabled(self, enabled: bool) -> None:
        self._inpaint_enabled = bool(enabled)
        if self._inpaint_enabled:
            self.exit_first_person_mode()
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()

    def get_camera_pose(self) -> CameraPose:
        return self._camera_pose

    def set_camera_pose(
        self,
        pose: CameraPose,
        *,
        emit_signal: bool = True,
    ) -> None:
        if not isinstance(pose, CameraPose):
            raise TypeError("The first-person camera pose must be a CameraPose.")
        self._camera_pose = CameraPose(
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
        self._sync_view_to_camera_pose()
        if emit_signal:
            self.camera_pose_changed.emit(self._camera_pose)

    def enter_first_person_mode(self) -> None:
        if self._first_person_active or self._inpaint_enabled:
            return
        self._first_person_active = True
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.setCursor(Qt.CursorShape.BlankCursor)
        if self.isVisible():
            self.grabMouse()
            self._center_pointer()
        self._movement_timer.start()
        self.first_person_active_changed.emit(True)

    def exit_first_person_mode(self) -> None:
        if not self._first_person_active:
            return
        self._first_person_active = False
        self._pressed_movement_keys.clear()
        self._movement_timer.stop()
        if self.isVisible():
            self.releaseMouse()
        self.unsetCursor()
        self.first_person_active_changed.emit(False)

    def step_movement(self, elapsed_seconds: float) -> None:
        """Advance held ZQSD and R/F keys; public for deterministic tests."""

        elapsed = max(0.0, float(elapsed_seconds))
        if elapsed <= 0.0 or not self._pressed_movement_keys:
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
        yaw_radians = math.radians(self._camera_pose.yaw_degrees)
        forward_x = math.cos(yaw_radians)
        forward_y = math.sin(yaw_radians)
        # In this Z-up coordinate system, camera-right is forward cross up.
        # The previous inverse vector made French-layout Q and D feel swapped.
        right_x = math.sin(yaw_radians)
        right_y = -math.cos(yaw_radians)
        distance = self._move_speed * elapsed
        self.set_camera_pose(
            CameraPose(
                x=self._camera_pose.x
                + (forward_x * forward_amount + right_x * right_amount)
                * distance,
                y=self._camera_pose.y
                + (forward_y * forward_amount + right_y * right_amount)
                * distance,
                z=self._camera_pose.z + vertical_amount * distance,
                yaw_degrees=self._camera_pose.yaw_degrees,
                pitch_degrees=self._camera_pose.pitch_degrees,
                roll_degrees=self._camera_pose.roll_degrees,
                fov_degrees=self._camera_pose.fov_degrees,
            )
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._inpaint_enabled and event.button() == Qt.MouseButton.LeftButton:
            self.inpaint_pointer_pressed.emit(event.position())
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            if self._first_person_active:
                self.exit_first_person_mode()
                event.accept()
                return
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if not self._first_person_active:
                self.enter_first_person_mode()
            self.surface_pick_requested.emit(
                event.position(),
                event.modifiers(),
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._inpaint_enabled:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.inpaint_pointer_moved.emit(event.position())
            event.accept()
            return
        if not self._first_person_active:
            super().mouseMoveEvent(event)
            return
        center = QPointF(self.rect().center())
        delta = event.position() - center
        if self._ignore_center_mouse_move and _point_is_near(delta, 0.5):
            self._ignore_center_mouse_move = False
            event.accept()
            return
        if not _point_is_near(delta, 0.0):
            self.set_camera_pose(
                CameraPose(
                    x=self._camera_pose.x,
                    y=self._camera_pose.y,
                    z=self._camera_pose.z,
                    yaw_degrees=(
                        self._camera_pose.yaw_degrees
                        - delta.x() * self._mouse_look_sensitivity
                    ),
                    pitch_degrees=(
                        self._camera_pose.pitch_degrees
                        - delta.y() * self._mouse_look_sensitivity
                    ),
                    roll_degrees=self._camera_pose.roll_degrees,
                    fov_degrees=self._camera_pose.fov_degrees,
                )
            )
            if self.isVisible():
                self._center_pointer()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._inpaint_enabled and event.button() == Qt.MouseButton.LeftButton:
            self.inpaint_pointer_released.emit(event.position())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if self._first_person_active and event.key() in _movement_keys():
            self._pressed_movement_keys.add(event.key())
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() in _movement_keys():
            self._pressed_movement_keys.discard(event.key())
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event) -> None:  # type: ignore[override]
        self.exit_first_person_mode()
        super().focusOutEvent(event)

    def _advance_from_pressed_keys(self) -> None:
        self.step_movement(FIRST_PERSON_UPDATE_INTERVAL_MILLISECONDS / 1000.0)

    def _sync_view_to_camera_pose(self) -> None:
        pose = self._camera_pose
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

    def _center_pointer(self) -> None:
        self._ignore_center_mouse_move = True
        QCursor.setPos(self.mapToGlobal(self.rect().center()))

    def _get_clicked_items(self, position: QPointF) -> list[object]:
        half_size = PICK_REGION_SIZE_PIXELS // 2
        region = (
            int(position.x()) - half_size,
            int(position.y()) - half_size,
            PICK_REGION_SIZE_PIXELS,
            PICK_REGION_SIZE_PIXELS,
        )
        try:
            return list(self.itemsAt(region))
        except Exception:
            return []


# ### Surface texture viewer ###
class SurfaceTextureViewer(QWidget):
    """Selectable semantic house surfaces rendered from a first-person camera."""

    selection_changed = Signal(object)
    camera_pose_changed = Signal(object)
    first_person_active_changed = Signal(bool)
    texture_mask_strokes_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._surfaces: list[FixedSurface] = []
        self._surface_by_id: dict[str, FixedSurface] = {}
        self._render_items_by_surface_id: dict[str, SurfaceRenderItems] = {}
        self._surface_id_by_item_id: dict[int, str] = {}
        self._selected_surface_ids: list[str] = []
        self._surface_textures: dict[str, np.ndarray] = {}
        self._texture_mask_strokes: dict[str, list[MaskStroke]] = {}
        self._scene_model: GeneratedModel | None = None
        self._canvas_scene_render_items = CanvasSceneRenderItems()
        self._preview_surfaces_by_id: dict[
            str,
            tuple[PreviewTexturedSurface, ...],
        ] = {}
        self._texture_world_size_meters = DEFAULT_TEXTURE_WORLD_SIZE_METERS
        self._ambient_light_intensity = DEFAULT_AMBIENT_LIGHT_INTENSITY
        self._ambient_shader = _build_ambient_shader(
            self._ambient_light_intensity
        )
        self._inpaint_brush_mode = MASK_MODE_PAINT
        self._inpaint_brush_radius_pixels = (
            DEFAULT_TEXTURE_INPAINT_BRUSH_RADIUS_PIXELS
        )
        self._active_inpaint_surface_id: str | None = None
        self._active_inpaint_points: list[MaskPoint] = []
        self._last_inpaint_surface_id: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QStackedLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self.view = SurfaceFirstPersonViewWidget()
        self.view.setBackgroundColor((24, 24, 28))
        self.view.surface_pick_requested.connect(
            self._handle_surface_pick_requested
        )
        self.view.camera_pose_changed.connect(self.camera_pose_changed.emit)
        self.view.first_person_active_changed.connect(
            self.first_person_active_changed.emit
        )
        self.view.inpaint_pointer_pressed.connect(
            self._handle_inpaint_pointer_pressed
        )
        self.view.inpaint_pointer_moved.connect(
            self._handle_inpaint_pointer_moved
        )
        self.view.inpaint_pointer_released.connect(
            self._handle_inpaint_pointer_released
        )
        layout.addWidget(self.view)

        self.crosshair_label = QLabel("+")
        self.crosshair_label.setObjectName("surface_crosshair_label")
        self.crosshair_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.crosshair_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
        )
        self.crosshair_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.crosshair_label.setStyleSheet(
            "color: rgba(255, 255, 255, 220); font-size: 22px; "
            "font-weight: 600; background: transparent;"
        )
        self.crosshair_label.hide()
        self.view.first_person_active_changed.connect(
            self.crosshair_label.setVisible
        )
        layout.addWidget(self.crosshair_label)

    def set_levels(
        self,
        levels: Sequence[LevelData],
        initial_camera: InitialFirstPersonCamera | CameraPose | None = None,
    ) -> None:
        self.set_surfaces(build_fixed_surfaces(levels))
        pose = _get_initial_pose(initial_camera)
        if pose is None:
            pose = _build_default_camera_pose(self._surfaces)
        self.set_camera_pose(pose)

    def set_scene_model(self, model: GeneratedModel | None) -> None:
        """Render the exact Canvas model behind semantic interaction geometry."""

        if model is not None and not isinstance(model, GeneratedModel):
            raise TypeError("The Surface preview model must be a GeneratedModel.")
        self._scene_model = model
        self._populate_scene()

    def get_scene_model(self) -> GeneratedModel | None:
        return self._scene_model

    def set_surfaces(self, surfaces: Sequence[FixedSurface]) -> None:
        normalized = list(surfaces)
        if not all(isinstance(surface, FixedSurface) for surface in normalized):
            raise TypeError("Surface views require FixedSurface values.")
        surface_ids = [surface.surface_id for surface in normalized]
        if len(surface_ids) != len(set(surface_ids)):
            raise ValueError("Fixed surface IDs must be unique.")
        retained_textures = {
            surface_id: texture
            for surface_id, texture in self._surface_textures.items()
            if surface_id in surface_ids
        }
        retained_texture_mask_strokes = {
            surface_id: list(strokes)
            for surface_id, strokes in self._texture_mask_strokes.items()
            if surface_id in surface_ids
        }
        self._surfaces = normalized
        self._surface_by_id = {
            surface.surface_id: surface for surface in self._surfaces
        }
        self._surface_textures = retained_textures
        self._texture_mask_strokes = retained_texture_mask_strokes
        self._selected_surface_ids = [
            surface_id
            for surface_id in self._selected_surface_ids
            if surface_id in self._surface_by_id
        ]
        self._populate_scene()

    def get_surfaces(self) -> list[FixedSurface]:
        return list(self._surfaces)

    def get_surface(self, surface_id: str) -> FixedSurface | None:
        return self._surface_by_id.get(str(surface_id))

    def get_selected_surface_ids(self) -> tuple[str, ...]:
        return tuple(self._selected_surface_ids)

    def get_selected_surface_type(self) -> str | None:
        if not self._selected_surface_ids:
            return None
        surface = self._surface_by_id.get(self._selected_surface_ids[0])
        return None if surface is None else surface.surface_type

    def get_combined_selected_area(self) -> float:
        return get_combined_surface_area(
            self._surfaces,
            self._selected_surface_ids,
        )

    def get_combined_surface_area(self, surface_ids: Iterable[str]) -> float:
        return get_combined_surface_area(self._surfaces, surface_ids)

    def set_selected_surface_ids(self, surface_ids: Iterable[str]) -> None:
        normalized_ids = _deduplicate_strings(surface_ids)
        surfaces = [
            self._surface_by_id[surface_id]
            for surface_id in normalized_ids
            if surface_id in self._surface_by_id
        ]
        if surfaces and len({surface.surface_type for surface in surfaces}) != 1:
            raise ValueError("A surface selection can contain only one surface type.")
        self._selected_surface_ids = [
            surface.surface_id for surface in surfaces
        ]
        self._sync_selection_rendering()
        self.selection_changed.emit(self.get_selected_surface_ids())

    def select_surface(self, surface_id: str, *, shift_pressed: bool = False) -> bool:
        surface = self._surface_by_id.get(str(surface_id))
        if surface is None:
            return False
        if not shift_pressed:
            next_ids = [surface.surface_id]
        else:
            selected_type = self.get_selected_surface_type()
            if selected_type is not None and selected_type != surface.surface_type:
                return False
            next_ids = list(self._selected_surface_ids)
            if surface.surface_id in next_ids:
                next_ids.remove(surface.surface_id)
            else:
                next_ids.append(surface.surface_id)
        if next_ids == self._selected_surface_ids:
            return False
        self._selected_surface_ids = next_ids
        self._sync_selection_rendering()
        self.selection_changed.emit(self.get_selected_surface_ids())
        return True

    def clear_selection(self) -> None:
        if not self._selected_surface_ids:
            return
        self._selected_surface_ids = []
        self._sync_selection_rendering()
        self.selection_changed.emit(())

    def set_surface_texture(
        self,
        surface_ids: Iterable[str],
        texture: bytes | bytearray | memoryview | str | Path | np.ndarray | Image.Image,
    ) -> None:
        texture_rgba = _load_texture_rgba(texture)
        changed_ids: list[str] = []
        for surface_id in _deduplicate_strings(surface_ids):
            if surface_id not in self._surface_by_id:
                continue
            self._surface_textures[surface_id] = texture_rgba.copy()
            changed_ids.append(surface_id)
        for surface_id in changed_ids:
            self._rebuild_surface_texture_item(surface_id)

    def has_surface_texture(self, surface_id: str) -> bool:
        return str(surface_id) in self._surface_textures

    def get_surface_texture_rgba(self, surface_id: str) -> np.ndarray | None:
        texture = self._surface_textures.get(str(surface_id))
        return None if texture is None else texture.copy()

    def get_texture_mask_strokes(self) -> dict[str, list[MaskStroke]]:
        return {
            surface_id: list(strokes)
            for surface_id, strokes in self._texture_mask_strokes.items()
        }

    def set_texture_mask_strokes(
        self,
        strokes_by_surface_id: Mapping[str, Sequence[MaskStroke]],
        *,
        emit_signal: bool = False,
    ) -> None:
        if not isinstance(strokes_by_surface_id, Mapping):
            raise TypeError("3D texture masks must contain a surface mapping.")
        normalized: dict[str, list[MaskStroke]] = {}
        for raw_surface_id, raw_strokes in strokes_by_surface_id.items():
            surface_id = str(raw_surface_id)
            if surface_id not in self._surface_by_id:
                continue
            strokes = list(raw_strokes)
            if not all(isinstance(stroke, MaskStroke) for stroke in strokes):
                raise ValueError("3D texture masks must contain MaskStroke values.")
            if strokes:
                normalized[surface_id] = strokes
        changed_ids = set(self._texture_mask_strokes) | set(normalized)
        self._texture_mask_strokes = normalized
        for surface_id in changed_ids:
            if surface_id in self._surface_textures:
                self._rebuild_surface_texture_item(surface_id)
        if emit_signal:
            self.texture_mask_strokes_changed.emit(
                self.get_texture_mask_strokes()
            )

    def add_texture_mask_stroke(
        self,
        surface_id: str,
        stroke: MaskStroke,
    ) -> None:
        normalized_id = str(surface_id)
        if normalized_id not in self._selected_surface_ids:
            raise ValueError("3D inpainting is limited to selected surfaces.")
        if normalized_id not in self._surface_textures:
            raise ValueError("Apply a texture before inpainting this surface.")
        if not isinstance(stroke, MaskStroke):
            raise TypeError("A 3D texture mask stroke must be a MaskStroke.")
        self._texture_mask_strokes.setdefault(normalized_id, []).append(stroke)
        self._last_inpaint_surface_id = normalized_id
        self._rebuild_surface_texture_item(normalized_id)
        self.texture_mask_strokes_changed.emit(self.get_texture_mask_strokes())

    def clear_texture_mask(self, surface_ids: Iterable[str] | None = None) -> None:
        target_ids = (
            list(self._selected_surface_ids)
            if surface_ids is None
            else _deduplicate_strings(surface_ids)
        )
        changed_ids = [
            surface_id
            for surface_id in target_ids
            if self._texture_mask_strokes.pop(surface_id, None) is not None
        ]
        if not changed_ids:
            return
        for surface_id in changed_ids:
            if surface_id in self._surface_textures:
                self._rebuild_surface_texture_item(surface_id)
        self.texture_mask_strokes_changed.emit(self.get_texture_mask_strokes())

    def undo_last_texture_mask_stroke(self) -> None:
        candidate_ids = list(reversed(self._selected_surface_ids))
        if self._last_inpaint_surface_id in candidate_ids:
            candidate_ids.remove(self._last_inpaint_surface_id)
            candidate_ids.insert(0, self._last_inpaint_surface_id)
        for surface_id in candidate_ids:
            strokes = self._texture_mask_strokes.get(surface_id)
            if not strokes:
                continue
            strokes.pop()
            if not strokes:
                self._texture_mask_strokes.pop(surface_id, None)
            self._rebuild_surface_texture_item(surface_id)
            self.texture_mask_strokes_changed.emit(
                self.get_texture_mask_strokes()
            )
            return

    def has_selected_texture_mask(self) -> bool:
        return bool(self.get_masked_selected_surface_ids())

    def get_masked_selected_surface_ids(self) -> tuple[str, ...]:
        return tuple(
            surface_id
            for surface_id in self._selected_surface_ids
            if self._texture_mask_strokes.get(surface_id)
        )

    def get_combined_masked_selected_area(self) -> float:
        return get_combined_surface_area(
            self._surfaces,
            self.get_masked_selected_surface_ids(),
        )

    def get_selected_texture_edit_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return identical base texture and editable-white union mask."""

        masked_ids = self.get_masked_selected_surface_ids()
        if not masked_ids:
            return None
        textures: list[np.ndarray] = []
        for surface_id in masked_ids:
            texture = self._surface_textures.get(surface_id)
            if texture is None:
                raise ValueError(
                    "Every selected 3D surface needs an applied texture."
                )
            textures.append(texture)
        base_texture = textures[0]
        if any(
            texture.shape != base_texture.shape
            or not np.array_equal(texture, base_texture)
            for texture in textures[1:]
        ):
            raise ValueError(
                "Selected surfaces must share the same texture for partial inpainting."
            )
        mask = np.zeros(base_texture.shape[:2], dtype=np.uint8)
        for surface_id in masked_ids:
            strokes = self._texture_mask_strokes.get(surface_id, [])
            if strokes:
                mask = np.maximum(
                    mask,
                    rasterize_texture_mask_strokes(
                        (base_texture.shape[1], base_texture.shape[0]),
                        strokes,
                    ),
                )
        if not np.any(mask):
            return None
        return base_texture.copy(), mask

    def get_selected_texture_edit_masks(self) -> dict[str, np.ndarray]:
        """Return editable-white masks for only selected, painted textures."""

        masked_ids = self.get_masked_selected_surface_ids()
        edit_data = self.get_selected_texture_edit_data()
        if edit_data is None:
            return {}
        base_texture, _union_mask = edit_data
        return {
            surface_id: rasterize_texture_mask_strokes(
                (base_texture.shape[1], base_texture.shape[0]),
                self._texture_mask_strokes[surface_id],
            )
            for surface_id in masked_ids
        }

    def set_surface_textures(
        self,
        textures: Mapping[str, object],
    ) -> None:
        self.clear_surface_textures()
        for surface_id, texture in textures.items():
            self.set_surface_texture((str(surface_id),), texture)  # type: ignore[arg-type]

    def set_textures(self, textures: Mapping[str, object]) -> None:
        self.set_surface_textures(textures)

    def set_assignments(self, assignments: Iterable[object]) -> None:
        """Apply persisted assignment-like values without coupling to state models."""

        self.clear_surface_textures()
        for assignment in assignments:
            surface_ids = _get_assignment_value(
                assignment,
                "surface_ids",
                "selected_surface_ids",
            )
            texture = _get_assignment_value(
                assignment,
                "texture_png",
                "png_bytes",
                "texture_path",
                "asset_path",
                "image_path",
            )
            if surface_ids is None or texture is None:
                continue
            try:
                self.set_surface_texture(surface_ids, texture)
            except (OSError, TypeError, ValueError):
                continue

    def clear_surface_textures(self) -> None:
        self._surface_textures = {}
        for surface_id in list(self._render_items_by_surface_id):
            self._remove_surface_texture_item(surface_id)
        self._sync_selection_rendering()

    def set_texture_world_size_meters(self, size_meters: float) -> None:
        size = float(size_meters)
        if not math.isfinite(size) or size <= 0.0:
            raise ValueError("Texture world size must be finite and positive.")
        self._texture_world_size_meters = size
        for surface_id in list(self._surface_textures):
            self._rebuild_surface_texture_item(surface_id)

    def get_texture_world_size_meters(self) -> float:
        return self._texture_world_size_meters

    def set_camera_pose(self, pose: CameraPose) -> None:
        self.view.set_camera_pose(pose)

    def get_camera_pose(self) -> CameraPose:
        return self.view.get_camera_pose()

    def enter_first_person_mode(self) -> None:
        self.view.enter_first_person_mode()

    def exit_first_person_mode(self) -> None:
        self.view.exit_first_person_mode()

    @property
    def is_inpaint_enabled(self) -> bool:
        return self.view.is_inpaint_enabled

    def can_inpaint_selection(self) -> bool:
        return any(
            surface_id in self._surface_textures
            for surface_id in self._selected_surface_ids
        )

    def set_inpaint_enabled(self, enabled: bool) -> None:
        should_enable = bool(enabled)
        if should_enable and not self.can_inpaint_selection():
            raise ValueError(
                "Select at least one surface with an applied texture first."
            )
        self._commit_active_inpaint_stroke()
        self.view.set_inpaint_enabled(should_enable)

    def set_inpaint_brush_mode(self, mode: str) -> None:
        if mode not in (MASK_MODE_PAINT, MASK_MODE_ERASE):
            raise ValueError(f"Unknown 3D inpaint brush mode: {mode!r}.")
        self._inpaint_brush_mode = mode

    def set_inpaint_brush_radius_pixels(self, radius_pixels: int) -> None:
        self._inpaint_brush_radius_pixels = max(1, int(radius_pixels))

    def pick_surface_at_view_position(
        self,
        position: QPointF | tuple[float, float],
    ) -> str | None:
        """Return the nearest semantic surface under one viewport position."""

        if isinstance(position, QPointF):
            position_x, position_y = position.x(), position.y()
        else:
            try:
                position_x, position_y = position
            except (TypeError, ValueError) as error:
                raise ValueError("A pick position must contain X and Y.") from error
        ray_origin, ray_direction = _build_camera_ray(
            self.get_camera_pose(),
            float(position_x),
            float(position_y),
            self.view.width(),
            self.view.height(),
        )
        return self.pick_surface_from_ray(ray_origin, ray_direction)

    def pick_surface_from_ray(
        self,
        ray_origin: Sequence[float],
        ray_direction: Sequence[float],
    ) -> str | None:
        """Return the nearest ray-hit surface without requiring OpenGL picking."""

        origin = _normalize_vector3(ray_origin, "Ray origin", normalize=False)
        direction = _normalize_vector3(
            ray_direction,
            "Ray direction",
            normalize=True,
        )
        candidates: list[tuple[float, int, str]] = []
        for surface in self._surfaces:
            hit = _get_nearest_mesh_ray_hit(surface.mesh, origin, direction)
            if hit is None:
                continue
            distance, is_back_facing = hit
            candidates.append(
                (distance, int(is_back_facing), surface.surface_id)
            )
        if not candidates:
            return None
        nearest_distance = min(candidate[0] for candidate in candidates)
        distance_tolerance = max(
            RAY_INTERSECTION_EPSILON,
            nearest_distance * 1e-7,
        )
        nearest_candidates = [
            candidate
            for candidate in candidates
            if candidate[0] <= nearest_distance + distance_tolerance
        ]
        return min(nearest_candidates, key=lambda candidate: candidate[1:])[2]

    def pick_surface_texture_hit_at_view_position(
        self,
        position: QPointF | tuple[float, float],
    ) -> SurfaceTextureHit | None:
        if isinstance(position, QPointF):
            position_x, position_y = position.x(), position.y()
        else:
            try:
                position_x, position_y = position
            except (TypeError, ValueError) as error:
                raise ValueError("A pick position must contain X and Y.") from error
        ray_origin, ray_direction = _build_camera_ray(
            self.get_camera_pose(),
            float(position_x),
            float(position_y),
            self.view.width(),
            self.view.height(),
        )
        return self.pick_surface_texture_hit_from_ray(ray_origin, ray_direction)

    def pick_surface_texture_hit_from_ray(
        self,
        ray_origin: Sequence[float],
        ray_direction: Sequence[float],
    ) -> SurfaceTextureHit | None:
        origin = _normalize_vector3(ray_origin, "Ray origin", normalize=False)
        direction = _normalize_vector3(
            ray_direction,
            "Ray direction",
            normalize=True,
        )
        candidates: list[tuple[float, int, str, _MeshRayHit]] = []
        for surface in self._surfaces:
            hit = _get_nearest_mesh_ray_hit_details(
                surface.mesh,
                origin,
                direction,
            )
            if hit is None:
                continue
            candidates.append(
                (
                    hit.distance,
                    int(hit.is_back_facing),
                    surface.surface_id,
                    hit,
                )
            )
        if not candidates:
            return None
        distance, _back_facing, surface_id, hit = min(
            candidates,
            key=lambda candidate: candidate[:3],
        )
        world_position = origin + direction * distance
        surface = self._surface_by_id[surface_id]
        face_normal = np.asarray(
            surface.mesh.face_normals[hit.face_index],
            dtype=float,
        )
        texture_point = _world_position_to_texture_point(
            surface.surface_type,
            world_position,
            face_normal,
            self._texture_world_size_meters,
        )
        return SurfaceTextureHit(
            surface_id=surface_id,
            world_position=tuple(float(value) for value in world_position),
            texture_point=texture_point,
        )

    def _populate_scene(self) -> None:
        self.view.clear()
        self._render_items_by_surface_id = {}
        self._surface_id_by_item_id = {}
        self._canvas_scene_render_items = CanvasSceneRenderItems()
        self._preview_surfaces_by_id = _group_preview_surfaces_by_id(
            ()
            if self._scene_model is None
            else self._scene_model.preview_textured_surfaces
        )
        if self._scene_model is not None:
            self._populate_canvas_scene_background(self._scene_model)
        for surface in self._surfaces:
            vertices = np.asarray(surface.mesh.vertices, dtype=np.float32)
            faces = np.asarray(surface.mesh.faces, dtype=np.int32)
            if vertices.size == 0 or faces.size == 0:
                continue
            face_color = SURFACE_FACE_COLORS[surface.surface_type]
            face_item = gl.GLMeshItem(
                vertexes=vertices,
                faces=faces,
                faceColors=np.tile(face_color, (len(faces), 1)),
                smooth=False,
                drawFaces=self._scene_model is None,
                drawEdges=False,
                edgeColor=SURFACE_EDGE_COLOR,
                glOptions="opaque",
            )
            self.view.addItem(face_item)
            outline_item = _build_surface_outline_item(surface.mesh)
            if outline_item is not None:
                outline_item.setVisible(False)
                self.view.addItem(outline_item)
            self._render_items_by_surface_id[surface.surface_id] = (
                SurfaceRenderItems(
                    face_item=face_item,
                    outline_item=outline_item,
                )
            )
            self._surface_id_by_item_id[id(face_item)] = surface.surface_id
            if surface.surface_id in self._surface_textures:
                self._rebuild_surface_texture_item(surface.surface_id)
        self._sync_selection_rendering()
        self.view.update()

    def _populate_canvas_scene_background(self, model: GeneratedModel) -> None:
        """Add the same base geometry, wireframe, grid, and legacy walls as Canvas."""

        grid_item = gl.GLGridItem()
        grid_item.setSize(x=20.0, y=20.0)
        grid_item.setSpacing(x=1.0, y=1.0)
        self.view.addItem(grid_item)

        display_mesh = (
            model.preview_untextured_mesh
            if model.preview_textured_surfaces
            and model.preview_untextured_mesh is not None
            else model.mesh
        )
        vertices = np.asarray(display_mesh.vertices, dtype=np.float32)
        faces = np.asarray(display_mesh.faces, dtype=np.int32)
        if vertices.size == 0 or faces.size == 0:
            self._canvas_scene_render_items = CanvasSceneRenderItems(
                grid_item=grid_item
            )
            return

        texture_mesh_data = _build_texture_mesh_data(display_mesh)
        textured_mesh_item: TexturedMeshItem | None = None
        if texture_mesh_data is not None:
            textured_mesh_item = TexturedMeshItem(
                texture_mesh_data,
                self._ambient_light_intensity,
            )
            self.view.addItem(textured_mesh_item)
        face_colors = (
            np.tile(FACE_COLOR, (len(faces), 1))
            if texture_mesh_data is not None
            else _get_mesh_face_colors(display_mesh, faces)
        )
        mesh_item = _WireframeOverlayMeshItem(
            vertexes=vertices,
            faces=faces,
            faceColors=face_colors,
            smooth=False,
            drawFaces=texture_mesh_data is None,
            drawEdges=True,
            edgeColor=EDGE_COLOR,
            shader=self._ambient_shader,
            cull_back_faces=bool(model.preview_textured_surfaces),
        )
        self.view.addItem(mesh_item)
        legacy_wall_items = self._add_canvas_legacy_wall_items(model)
        self._canvas_scene_render_items = CanvasSceneRenderItems(
            grid_item=grid_item,
            mesh_item=mesh_item,
            textured_mesh_item=textured_mesh_item,
            legacy_wall_items=legacy_wall_items,
        )

    def _add_canvas_legacy_wall_items(
        self,
        model: GeneratedModel,
    ) -> tuple[gl.GLImageItem, ...]:
        assigned_wall_keys = {
            (
                surface.level_index,
                surface.room_index,
                surface.wall_key,
            )
            for surface in model.preview_textured_surfaces
            if surface.surface_type == SURFACE_TYPE_WALL
            and surface.level_index is not None
            and surface.room_index is not None
            and surface.wall_key is not None
        }
        items: list[gl.GLImageItem] = []
        for textured_wall in model.preview_textured_walls:
            if (
                textured_wall.level_index,
                textured_wall.room_index,
                textured_wall.wall_key,
            ) in assigned_wall_keys:
                continue
            texture_rgba = np.asarray(textured_wall.texture_rgba, dtype=np.ubyte)
            if texture_rgba.ndim != 3 or texture_rgba.shape[2] != 4:
                continue
            image_item = gl.GLImageItem(
                texture_rgba,
                smooth=True,
                glOptions="opaque",
            )
            image_item.setTransform(
                _build_textured_wall_transform(
                    textured_wall=textured_wall,
                    offset_sign=-1.0,
                )
            )
            self.view.addItem(image_item)
            items.append(image_item)
        return tuple(items)

    def _rebuild_surface_texture_item(self, surface_id: str) -> None:
        self._remove_surface_texture_item(surface_id)
        surface = self._surface_by_id.get(surface_id)
        render_items = self._render_items_by_surface_id.get(surface_id)
        texture_rgba = self._surface_textures.get(surface_id)
        if surface is None or render_items is None or texture_rgba is None:
            return
        preview_texture_rgba = _build_texture_mask_preview(
            texture_rgba,
            self._texture_mask_strokes.get(surface_id, []),
        )
        texture_data_values = self._build_surface_texture_data_values(
            surface,
            preview_texture_rgba,
        )
        texture_items = tuple(
            RepeatingTexturedMeshItem(
                texture_data,
                self._ambient_light_intensity,
                double_sided=double_sided,
            )
            for texture_data, double_sided in texture_data_values
        )
        if not texture_items:
            return
        for texture_item in texture_items:
            self.view.addItem(texture_item)
            self._surface_id_by_item_id[id(texture_item)] = surface_id
        render_items.texture_item = texture_items[0]
        render_items.additional_texture_items = texture_items[1:]
        self._sync_surface_rendering(surface_id)

    def _build_surface_texture_data_values(
        self,
        surface: FixedSurface,
        texture_rgba: np.ndarray,
    ) -> tuple[tuple[TextureMeshData, bool], ...]:
        previews = self._preview_surfaces_by_id.get(surface.surface_id, ())
        preview_data = tuple(
            (texture_data, preview.double_sided)
            for preview in previews
            if (texture_data := _build_texture_mesh_data(preview.mesh))
            is not None
        )
        if preview_data:
            return tuple(
                (
                    TextureMeshData(
                        vertices=texture_data.vertices,
                        normals=texture_data.normals,
                        texture_coordinates=texture_data.texture_coordinates,
                        texture_rgba=_limit_texture_preview_size(texture_rgba),
                    ),
                    double_sided,
                )
                for texture_data, double_sided in preview_data
            )

        if self._scene_model is not None:
            # The shared Canvas model is authoritative.  A missing preview
            # means its transactional refresh has not arrived yet; drawing a
            # second coplanar mesh here would reintroduce z-fighting.
            return ()

        texture_data = _build_surface_texture_mesh_data(
            surface,
            texture_rgba,
            self._texture_world_size_meters,
        )
        double_sided = (
            surface.surface_type == SURFACE_TYPE_WALL
            and surface.room_index is None
        )
        return ((texture_data, double_sided),)

    def _remove_surface_texture_item(self, surface_id: str) -> None:
        render_items = self._render_items_by_surface_id.get(surface_id)
        if render_items is None or render_items.texture_item is None:
            return
        texture_items = (
            render_items.texture_item,
            *render_items.additional_texture_items,
        )
        for texture_item in texture_items:
            self._surface_id_by_item_id.pop(id(texture_item), None)
            try:
                self.view.removeItem(texture_item)
            except ValueError:
                pass
        render_items.texture_item = None
        render_items.additional_texture_items = ()

    def _handle_items_clicked(
        self,
        clicked_items: Sequence[object],
        modifiers: Qt.KeyboardModifier,
    ) -> None:
        for clicked_item in clicked_items:
            surface_id = self._surface_id_by_item_id.get(id(clicked_item))
            if surface_id is None:
                continue
            self.select_surface(
                surface_id,
                shift_pressed=bool(
                    modifiers & Qt.KeyboardModifier.ShiftModifier
                ),
            )
            return

    def _handle_surface_pick_requested(
        self,
        position: QPointF,
        modifiers: Qt.KeyboardModifier,
    ) -> None:
        surface_id = self.pick_surface_at_view_position(position)
        if surface_id is None:
            return
        self.select_surface(
            surface_id,
            shift_pressed=bool(
                modifiers & Qt.KeyboardModifier.ShiftModifier
            ),
        )

    def _handle_inpaint_pointer_pressed(self, position: QPointF) -> None:
        self._commit_active_inpaint_stroke()
        self._append_inpaint_point_at_view_position(position)

    def _handle_inpaint_pointer_moved(self, position: QPointF) -> None:
        self._append_inpaint_point_at_view_position(position)

    def _handle_inpaint_pointer_released(self, position: QPointF) -> None:
        self._append_inpaint_point_at_view_position(position)
        self._commit_active_inpaint_stroke()

    def _append_inpaint_point_at_view_position(self, position: QPointF) -> None:
        hit = self.pick_surface_texture_hit_at_view_position(position)
        if (
            hit is None
            or hit.surface_id not in self._selected_surface_ids
            or hit.surface_id not in self._surface_textures
        ):
            self._commit_active_inpaint_stroke()
            return
        if self._active_inpaint_surface_id != hit.surface_id:
            self._commit_active_inpaint_stroke()
            self._active_inpaint_surface_id = hit.surface_id
        if self._active_inpaint_points and not _texture_points_are_distinct(
            self._active_inpaint_points[-1],
            hit.texture_point,
            self._surface_textures[hit.surface_id].shape,
        ):
            return
        self._active_inpaint_points.append(hit.texture_point)

    def _commit_active_inpaint_stroke(self) -> None:
        surface_id = self._active_inpaint_surface_id
        points = tuple(self._active_inpaint_points)
        self._active_inpaint_surface_id = None
        self._active_inpaint_points = []
        if surface_id is None or not points:
            return
        texture = self._surface_textures.get(surface_id)
        if texture is None:
            return
        shortest_side = max(1, min(texture.shape[:2]))
        stroke = MaskStroke(
            mode=self._inpaint_brush_mode,
            radius_normalized=min(
                float(self._inpaint_brush_radius_pixels) / shortest_side,
                1.0,
            ),
            points=points,
        )
        self.add_texture_mask_stroke(surface_id, stroke)

    def _sync_selection_rendering(self) -> None:
        for surface_id in self._render_items_by_surface_id:
            self._sync_surface_rendering(surface_id)
        self.view.update()

    def _sync_surface_rendering(self, surface_id: str) -> None:
        render_items = self._render_items_by_surface_id.get(surface_id)
        if render_items is None:
            return
        is_selected = surface_id in self._selected_surface_ids
        has_texture = render_items.texture_item is not None
        render_items.face_item.opts["drawFaces"] = (
            self._scene_model is None and not has_texture
        )
        render_items.face_item.opts["drawEdges"] = False
        render_items.face_item.meshDataChanged()
        if render_items.outline_item is not None:
            render_items.outline_item.setVisible(is_selected)
        if render_items.texture_item is not None:
            render_items.texture_item.setVisible(True)
        for texture_item in render_items.additional_texture_items:
            texture_item.setVisible(True)


# ### Selection outline geometry ###
def _build_surface_outline_item(
    mesh: trimesh.Trimesh,
) -> gl.GLMeshItem | None:
    outline_mesh = _build_surface_outline_mesh(mesh)
    if outline_mesh is None:
        return None
    return gl.GLMeshItem(
        vertexes=np.asarray(outline_mesh.vertices, dtype=np.float32),
        faces=np.asarray(outline_mesh.faces, dtype=np.int32),
        color=SELECTED_SURFACE_EDGE_COLOR,
        smooth=False,
        computeNormals=False,
        drawFaces=True,
        drawEdges=False,
        shader=None,
        glOptions="opaque",
    )


def _build_surface_outline_mesh(
    mesh: trimesh.Trimesh,
) -> trimesh.Trimesh | None:
    """Build solid tubes around geometric boundary edges.

    Solid geometry remains thick in OpenGL core profiles where line widths
    greater than one pixel are unsupported. Coordinate-based edge counting
    also removes triangulation diagonals when mesh vertices are duplicated.
    """

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
    ):
        return None
    edge_records: dict[
        tuple[tuple[float, float, float], tuple[float, float, float]],
        tuple[int, np.ndarray, np.ndarray],
    ] = {}
    for face in faces:
        for start_index, end_index in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            start = vertices[start_index]
            end = vertices[end_index]
            if np.linalg.norm(end - start) <= SURFACE_GEOMETRY_MINIMUM:
                continue
            start_key = _build_outline_point_key(start)
            end_key = _build_outline_point_key(end)
            edge_key = tuple(sorted((start_key, end_key)))
            previous = edge_records.get(edge_key)
            if previous is None:
                edge_records[edge_key] = (1, start, end)
            else:
                edge_records[edge_key] = (
                    previous[0] + 1,
                    previous[1],
                    previous[2],
                )

    outline_segments: list[trimesh.Trimesh] = []
    radius = SELECTED_SURFACE_OUTLINE_RADIUS_METERS
    for edge_count, start, end in edge_records.values():
        if edge_count != 1:
            continue
        edge_vector = end - start
        edge_length = float(np.linalg.norm(edge_vector))
        if edge_length <= SURFACE_GEOMETRY_MINIMUM:
            continue
        overlap = edge_vector / edge_length * radius
        outline_segments.append(
            trimesh.creation.cylinder(
                radius=radius,
                sections=SELECTED_SURFACE_OUTLINE_SECTIONS,
                segment=np.asarray((start - overlap, end + overlap)),
            )
        )
    if not outline_segments:
        return None
    return trimesh.util.concatenate(outline_segments)


def _build_outline_point_key(
    point: np.ndarray,
) -> tuple[float, float, float]:
    rounded = np.round(
        np.asarray(point, dtype=float),
        OUTLINE_EDGE_ROUNDING_DECIMALS,
    )
    return float(rounded[0]), float(rounded[1]), float(rounded[2])


# ### Texture coordinate helpers ###
def _build_surface_texture_mesh_data(
    surface: FixedSurface,
    texture_rgba: np.ndarray,
    texture_world_size_meters: float,
) -> TextureMeshData:
    mesh = surface.mesh
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    face_vertices = vertices[faces]
    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
    texture_coordinates = np.empty((len(faces), 3, 2), dtype=np.float32)
    tile_size = max(float(texture_world_size_meters), SURFACE_GEOMETRY_MINIMUM)
    for face_index, triangle_vertices in enumerate(face_vertices):
        normal = face_normals[face_index]
        if surface.surface_type == SURFACE_TYPE_WALL and abs(normal[2]) < 0.7:
            tangent = np.array((-normal[1], normal[0], 0.0), dtype=np.float32)
            tangent_length = float(np.linalg.norm(tangent))
            if tangent_length > SURFACE_GEOMETRY_MINIMUM:
                tangent /= tangent_length
            texture_coordinates[face_index, :, 0] = (
                triangle_vertices @ tangent
            ) / tile_size
            texture_coordinates[face_index, :, 1] = (
                triangle_vertices[:, 2] / tile_size
            )
        else:
            texture_coordinates[face_index, :, 0] = (
                triangle_vertices[:, 0] / tile_size
            )
            texture_coordinates[face_index, :, 1] = (
                triangle_vertices[:, 1] / tile_size
            )
    return TextureMeshData(
        vertices=np.ascontiguousarray(face_vertices.reshape(-1, 3)),
        normals=np.ascontiguousarray(np.repeat(face_normals, 3, axis=0)),
        texture_coordinates=np.ascontiguousarray(
            texture_coordinates.reshape(-1, 2)
        ),
        texture_rgba=_limit_texture_preview_size(texture_rgba),
    )


def rasterize_texture_mask_strokes(
    texture_size: tuple[int, int],
    strokes: Sequence[MaskStroke],
) -> np.ndarray:
    """Rasterize repeating normalized UV strokes as editable-white pixels."""

    width = max(0, int(texture_size[0]))
    height = max(0, int(texture_size[1]))
    if width <= 0 or height <= 0:
        return np.empty((0, 0), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    shortest_side = max(1, min(width, height))
    for stroke in strokes:
        radius = max(1, int(round(stroke.radius_normalized * shortest_side)))
        value = 0 if stroke.mode == MASK_MODE_ERASE else 255
        points = [
            (
                int(round(point.x * width)) % width,
                int(round((1.0 - point.y) * height)) % height,
            )
            for point in stroke.points
        ]
        _draw_periodic_circle(mask, points[0], radius, value)
        for start_point, end_point in zip(points, points[1:]):
            end_point = (
                start_point[0]
                + _shortest_periodic_delta(
                    end_point[0] - start_point[0],
                    width,
                ),
                start_point[1]
                + _shortest_periodic_delta(
                    end_point[1] - start_point[1],
                    height,
                ),
            )
            _draw_periodic_segment(
                mask,
                start_point,
                end_point,
                radius,
                value,
            )
            _draw_periodic_circle(mask, end_point, radius, value)
    return mask


def _build_texture_mask_preview(
    texture_rgba: np.ndarray,
    strokes: Sequence[MaskStroke],
) -> np.ndarray:
    if not strokes:
        return texture_rgba
    mask = rasterize_texture_mask_strokes(
        (texture_rgba.shape[1], texture_rgba.shape[0]),
        strokes,
    )
    if not np.any(mask):
        return texture_rgba
    preview = texture_rgba.copy()
    selected = mask > 0
    preview[selected, :3] = np.clip(
        preview[selected, :3].astype(float)
        * (1.0 - TEXTURE_MASK_OVERLAY_STRENGTH)
        + TEXTURE_MASK_OVERLAY_RGBA[:3].astype(float)
        * TEXTURE_MASK_OVERLAY_STRENGTH,
        0.0,
        255.0,
    ).astype(np.uint8)
    return preview


SURFACE_GEOMETRY_MINIMUM = 1e-8


def _load_texture_rgba(
    texture: bytes | bytearray | memoryview | str | Path | np.ndarray | Image.Image,
) -> np.ndarray:
    if isinstance(texture, Image.Image):
        image = texture.convert("RGBA")
    elif isinstance(texture, (str, Path)):
        with Image.open(Path(texture)) as loaded_image:
            image = loaded_image.convert("RGBA")
    elif isinstance(texture, (bytes, bytearray, memoryview)):
        with Image.open(BytesIO(bytes(texture))) as loaded_image:
            image = loaded_image.convert("RGBA")
    elif isinstance(texture, np.ndarray):
        array = np.asarray(texture, dtype=np.uint8)
        if array.ndim == 2:
            array = np.repeat(array[:, :, np.newaxis], 3, axis=2)
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ValueError("Texture arrays must contain RGB or RGBA pixels.")
        if array.shape[2] == 3:
            alpha = np.full(array.shape[:2] + (1,), 255, dtype=np.uint8)
            array = np.concatenate((array, alpha), axis=2)
        return np.ascontiguousarray(array)
    else:
        raise TypeError("Unsupported surface texture source.")
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


# ### CPU ray-picking helpers ###
def _build_camera_ray(
    pose: CameraPose,
    position_x: float,
    position_y: float,
    viewport_width: int,
    viewport_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    width = max(1.0, float(viewport_width))
    height = max(1.0, float(viewport_height))
    normalized_x = 2.0 * float(position_x) / width - 1.0
    normalized_y = 1.0 - 2.0 * float(position_y) / height
    yaw = math.radians(pose.yaw_degrees)
    pitch = math.radians(pose.pitch_degrees)
    forward = np.asarray(
        (
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ),
        dtype=float,
    )
    right = np.asarray((-math.sin(yaw), math.cos(yaw), 0.0), dtype=float)
    up = np.cross(forward, right)
    half_horizontal_span = math.tan(math.radians(pose.fov_degrees) / 2.0)
    direction = (
        forward
        + right * normalized_x * half_horizontal_span
        + up * normalized_y * half_horizontal_span * (height / width)
    )
    return (
        np.asarray((pose.x, pose.y, pose.z), dtype=float),
        _normalize_vector3(direction, "Camera ray", normalize=True),
    )


def _get_nearest_mesh_ray_hit(
    mesh: trimesh.Trimesh,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
) -> tuple[float, bool] | None:
    hit = _get_nearest_mesh_ray_hit_details(mesh, ray_origin, ray_direction)
    if hit is None:
        return None
    return hit.distance, hit.is_back_facing


def _get_nearest_mesh_ray_hit_details(
    mesh: trimesh.Trimesh,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
) -> _MeshRayHit | None:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(faces)
    ):
        return None
    triangles = vertices[faces]
    first_edges = triangles[:, 1] - triangles[:, 0]
    second_edges = triangles[:, 2] - triangles[:, 0]
    perpendicular = np.cross(ray_direction, second_edges)
    determinants = np.einsum("ij,ij->i", first_edges, perpendicular)
    valid = np.abs(determinants) > RAY_INTERSECTION_EPSILON
    safe_determinants = np.where(valid, determinants, 1.0)
    inverse_determinants = 1.0 / safe_determinants
    origin_deltas = ray_origin - triangles[:, 0]
    first_coordinates = (
        np.einsum("ij,ij->i", origin_deltas, perpendicular)
        * inverse_determinants
    )
    valid &= first_coordinates >= -RAY_INTERSECTION_EPSILON
    valid &= first_coordinates <= 1.0 + RAY_INTERSECTION_EPSILON
    cross_deltas = np.cross(origin_deltas, first_edges)
    second_coordinates = (
        np.einsum("j,ij->i", ray_direction, cross_deltas)
        * inverse_determinants
    )
    valid &= second_coordinates >= -RAY_INTERSECTION_EPSILON
    valid &= (
        first_coordinates + second_coordinates
        <= 1.0 + RAY_INTERSECTION_EPSILON
    )
    distances = (
        np.einsum("ij,ij->i", second_edges, cross_deltas)
        * inverse_determinants
    )
    valid &= distances > RAY_INTERSECTION_EPSILON
    if not np.any(valid):
        return None
    valid_indices = np.flatnonzero(valid)
    face_index = int(valid_indices[np.argmin(distances[valid_indices])])
    face_normal = np.asarray(mesh.face_normals[face_index], dtype=float)
    is_back_facing = float(np.dot(face_normal, ray_direction)) >= 0.0
    return _MeshRayHit(
        distance=float(distances[face_index]),
        face_index=face_index,
        is_back_facing=is_back_facing,
    )


def _world_position_to_texture_point(
    surface_type: str,
    world_position: np.ndarray,
    face_normal: np.ndarray,
    texture_world_size_meters: float,
) -> MaskPoint:
    tile_size = max(float(texture_world_size_meters), SURFACE_GEOMETRY_MINIMUM)
    if surface_type == SURFACE_TYPE_WALL and abs(face_normal[2]) < 0.7:
        tangent = np.asarray((-face_normal[1], face_normal[0], 0.0), dtype=float)
        tangent_length = float(np.linalg.norm(tangent))
        if tangent_length <= SURFACE_GEOMETRY_MINIMUM:
            tangent = np.asarray((1.0, 0.0, 0.0), dtype=float)
        else:
            tangent /= tangent_length
        coordinate_u = float(np.dot(world_position, tangent)) / tile_size
        coordinate_v = float(world_position[2]) / tile_size
    else:
        coordinate_u = float(world_position[0]) / tile_size
        coordinate_v = float(world_position[1]) / tile_size
    return MaskPoint(x=coordinate_u % 1.0, y=coordinate_v % 1.0)


def _texture_points_are_distinct(
    first: MaskPoint,
    second: MaskPoint,
    texture_shape: Sequence[int],
) -> bool:
    height = max(1, int(texture_shape[0]))
    width = max(1, int(texture_shape[1]))
    return math.hypot(
        _shortest_periodic_delta_float(first.x - second.x) * width,
        _shortest_periodic_delta_float(first.y - second.y) * height,
    ) >= 0.75


def _shortest_periodic_delta(delta: int, period: int) -> int:
    if period <= 0:
        return int(delta)
    return int((int(delta) + period // 2) % period - period // 2)


def _shortest_periodic_delta_float(delta: float) -> float:
    return (float(delta) + 0.5) % 1.0 - 0.5


def _draw_periodic_circle(
    mask: np.ndarray,
    point: tuple[int, int],
    radius: int,
    value: int,
) -> None:
    height, width = mask.shape
    for offset_x in (-width, 0, width):
        for offset_y in (-height, 0, height):
            cv2.circle(
                mask,
                (point[0] + offset_x, point[1] + offset_y),
                radius,
                value,
                -1,
                cv2.LINE_8,
            )


def _draw_periodic_segment(
    mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    radius: int,
    value: int,
) -> None:
    height, width = mask.shape
    for offset_x in (-width, 0, width):
        for offset_y in (-height, 0, height):
            offset = (offset_x, offset_y)
            cv2.line(
                mask,
                (start[0] + offset[0], start[1] + offset[1]),
                (end[0] + offset[0], end[1] + offset[1]),
                value,
                radius * 2,
                cv2.LINE_8,
            )


def _normalize_vector3(
    value: Sequence[float],
    label: str,
    *,
    normalize: bool,
) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain three finite values.")
    if not normalize:
        return vector
    length = float(np.linalg.norm(vector))
    if length <= RAY_INTERSECTION_EPSILON:
        raise ValueError(f"{label} cannot be a zero vector.")
    return vector / length


# ### Camera initialization helpers ###
def _get_initial_pose(
    camera: InitialFirstPersonCamera | CameraPose | None,
) -> CameraPose | None:
    if isinstance(camera, InitialFirstPersonCamera):
        return camera.pose
    if isinstance(camera, CameraPose):
        return camera
    if camera is not None:
        raise TypeError("Initial camera must be a CameraPose or project camera.")
    return None


def _build_default_camera_pose(surfaces: Sequence[FixedSurface]) -> CameraPose:
    if not surfaces:
        return CameraPose(z=DEFAULT_FIRST_PERSON_HEIGHT_METERS)
    vertices = np.vstack(
        [np.asarray(surface.mesh.vertices, dtype=float) for surface in surfaces]
    )
    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    center = (minimum + maximum) / 2.0
    return CameraPose(
        x=float(center[0]),
        y=float(center[1]),
        z=float(minimum[2] + DEFAULT_FIRST_PERSON_HEIGHT_METERS),
    )


def _movement_keys() -> frozenset[int]:
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


# ### Assignment and collection helpers ###
def _get_assignment_value(assignment: object, *names: str) -> object | None:
    if isinstance(assignment, Mapping):
        for name in names:
            if name in assignment:
                return assignment[name]
        return None
    for name in names:
        if hasattr(assignment, name):
            return getattr(assignment, name)
    return None


def _deduplicate_strings(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _point_is_near(delta: QPointF, tolerance: float) -> bool:
    return abs(delta.x()) <= tolerance and abs(delta.y()) <= tolerance
