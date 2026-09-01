# ### Imports ###
from __future__ import annotations

import copy
import sys
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

from PIL import Image
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPointF,
    QSignalBlocker,
    QSize,
    QTimer,
    Qt,
)
from PySide6.QtGui import QIcon, QKeySequence, QPixmap, QShortcut, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QListView,
    QDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.external_viewer_host import ExternalFullscreenViewerHost
from housemaker.generation_state import (
    GeneratedObjectPlacement,
    GeneratedObjectRecord,
    GenerationData,
)
from housemaker.generation_jobs import GenerationJobManager, JobsWindow
from housemaker.generation_workspace import (
    FACE_EDIT_TEXTURE_STALE_PIPELINE_KEY,
    GenerationWorkspace,
)
from housemaker.object_placement_dialog import ObjectPlacementDialog
from housemaker.surface_texture_state import (
    SURFACE_TYPE_WALL,
    SurfaceTextureAssignment,
    SurfaceTextureData,
)
from housemaker.surface_texture_workspace import (
    SurfaceTextureGenerationWorkspace,
)
from housemaker.surface_geometry import (
    WallWindowPlacement,
    add_wall_window,
    build_fixed_surfaces,
)
from housemaker.glb import (
    DEFAULT_WALL_HEIGHT_METERS,
    GeneratedModel,
    PlacedGeneratedModel,
    build_stair_meshes,
    build_texture_preview_plane_model,
    compose_placed_generated_models,
    compose_placed_generated_models_preview,
    convert_to_glb,
    convert_to_preview_model,
    export_glb_file,
    export_room_texture_pngs,
    import_generated_glb,
)
from housemaker.models import (
    DEFAULT_DOORWAY_ARCH_AMOUNT,
    DEFAULT_FLOOR_THICKNESS_METERS,
    DEFAULT_LEVEL_OFFSET_METERS,
    DEFAULT_LEVEL_SCALE,
    DEFAULT_ROOM_HEIGHT_METERS,
    DEFAULT_STAIR_STYLE,
    DEFAULT_UV_MAP_HEIGHT,
    DEFAULT_UV_MAP_WIDTH,
    DEFAULT_WALL_UV_ROTATION_DEGREES,
    DEFAULT_WALL_UV_SCALE,
    DOORWAY_SHAPE_ARCH,
    DOORWAY_SHAPE_RECTANGULAR,
    DoorwayData,
    DoorwayPreset,
    GROUND_LEVEL_INDEX,
    LevelData,
    MAX_DOORWAY_ARCH_AMOUNT,
    MAX_FLOOR_THICKNESS_METERS,
    MAX_LEVEL_OFFSET_METERS,
    MAX_LEVEL_SCALE,
    MIN_DOORWAY_ARCH_AMOUNT,
    MIN_FLOOR_THICKNESS_METERS,
    MIN_LEVEL_OFFSET_METERS,
    MIN_LEVEL_SCALE,
    RoomData,
    STAIR_STYLE_FLOATING,
    STAIR_STYLE_FLOATING_WITH_RISER,
    STAIR_STYLE_SUPPORTED,
    StairData,
    StairSectionData,
    WindowData,
    create_default_doorway_presets,
    create_default_levels,
)
from housemaker.level_coordinates import (
    build_doorway_world_outline_positions,
    build_level_base_z_lookup,
    level_image_to_world_xy,
    level_world_to_image_xy,
)
from housemaker.project_io import ProjectData, load_project, save_project
from housemaker.settings_widget import (
    DEFAULT_DOORWAY_MESH_UPDATE_DELAY_SECONDS,
    SettingsWidget,
    resolve_fullscreen_3d_viewer_screen,
)
from housemaker.texture_creator_canvas import TextureCreatorCanvas
from housemaker.texture_atlas_state import (
    ATLAS_PACKING_MODE_FULL,
    ATLAS_PACKING_MODE_SYMMETRIC_HALF,
    ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
    ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
    ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
    ATLAS_SLOT_HALF_LEFT,
    OBJECT_TEXTURE_RESOLUTIONS,
    TextureAtlasData,
)
from housemaker.texture_atlas_workspace import (
    AtlasObjectTextureSource,
    TextureAtlasWorkspace,
    build_atlas_wall_texture_source_id,
    choose_atlas_texture_resolution,
    get_atlas_wall_texture_assignment_id,
    is_atlas_wall_texture_source_id,
    load_atlas_object_texture_source,
)
from housemaker.uv_canvas import UvCanvas
from housemaker.uv_layout import (
    UvOptimizationResult,
    UvWallPlacement,
    build_uv_wall_layout,
    calculate_unoccupied_uv_pixels,
    optimize_room_wall_uvs,
    rebuild_room_subdivision_uvs,
)
from housemaker.viewer import (
    NAVIGATION_MODE_FIRST_PERSON,
    GlbViewerWidget,
)
from housemaker.manual_stitching import (
    VIDEO_FILE_FILTER,
    ManualVideoStitchDialog,
    build_unique_stitched_output_path,
    save_stitched_image,
)

# ### Constants ###
TEXTURE_CREATOR_DETAIL_SIZES = (512, 1024, 2048)
LAST_PROJECT_PATH_SETTING_KEY = "last_project_path"
PROJECT_LOAD_FAILURES = (
    AttributeError,
    KeyError,
    OSError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)

# ### Widgets ###
class PowerOfTwoSpinBox(QSpinBox):
    def setValue(self, value: int) -> None:  # type: ignore[override]
        super().setValue(_nearest_power_of_two_value(value))

    def stepBy(self, steps: int) -> None:  # type: ignore[override]
        self.setValue(_step_power_of_two_value(self.value(), steps))

    def valueFromText(self, text: str) -> int:  # type: ignore[override]
        try:
            return _nearest_power_of_two_value(int(text or self.minimum()))
        except ValueError:
            return self.value()

    def textFromValue(self, value: int) -> str:  # type: ignore[override]
        return str(_nearest_power_of_two_value(value))


class DegreeSpinBox(QSpinBox):
    def setValue(self, value: int) -> None:  # type: ignore[override]
        super().setValue(_normalize_degree_value(value))

    def stepBy(self, steps: int) -> None:  # type: ignore[override]
        self.setValue(self.value() + steps)

    def valueFromText(self, text: str) -> int:  # type: ignore[override]
        try:
            return _normalize_degree_value(int(text or self.minimum()))
        except ValueError:
            return self.value()

    def textFromValue(self, value: int) -> str:  # type: ignore[override]
        return str(_normalize_degree_value(value))


# ### Event filters ###
class RightPanelSpinBoxWheelFilter(QObject):
    """Scrolls a containing panel when its value inputs receive wheel events."""

    def __init__(self, scroll_area: QScrollArea) -> None:
        super().__init__(scroll_area)
        self._scroll_area = scroll_area

    def eventFilter(
        self,
        watched: QObject,
        event: QEvent,
    ) -> bool:  # type: ignore[override]
        if event.type() != QEvent.Type.Wheel or not isinstance(event, QWheelEvent):
            return super().eventFilter(watched, event)

        self._forward_wheel_event_to_scroll_area(event)
        event.accept()
        return True

    def _forward_wheel_event_to_scroll_area(self, event: QWheelEvent) -> None:
        viewport = self._scroll_area.viewport()
        viewport_position = viewport.mapFromGlobal(
            event.globalPosition().toPoint()
        )
        forwarded_event = QWheelEvent(
            QPointF(viewport_position),
            event.globalPosition(),
            event.pixelDelta(),
            event.angleDelta(),
            event.buttons(),
            event.modifiers(),
            event.phase(),
            event.inverted(),
            event.source(),
        )
        QApplication.sendEvent(viewport, forwarded_event)


class BlueprintWorkspace(QWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        application_settings: ApplicationSettingsStore | None = None,
    ) -> None:
        super().__init__(parent)
        self._application_settings = (
            application_settings
            if application_settings is not None
            else ApplicationSettingsStore()
        )
        self.current_project_path: str | None = None
        self.levels: list[LevelData] = create_default_levels()
        self.image_library_paths: list[str] = []
        self.doorway_presets: list[DoorwayPreset] = (
            create_default_doorway_presets()
        )
        self.stairs: list[StairData] = []
        self.current_level_index = GROUND_LEVEL_INDEX
        self._is_syncing_level_controls = False
        self._is_syncing_image_library_controls = False
        self._is_syncing_texture_controls = False
        self._is_syncing_uv_controls = False
        self._is_viewer_refresh_scheduled = False
        self._scheduled_viewer_refresh_preserve_camera = True
        self._viewer_preview_revision = 0
        self._viewer_preview_model_revision = -1
        self._canvas_viewer_preview_revision = -1
        self._surface_viewer_preview_revision = -1
        self._viewer_preview_model: GeneratedModel | None = None
        self._viewer_preview_dependency_signature: tuple[object, ...] | None = (
            None
        )
        self._viewer_preview_dependency_signature_revision = -1
        # Doorway dimensions remain live in project data while this separate
        # snapshot controls which openings have reached the expensive 3D mesh.
        self._viewer_doorways_by_level_index: dict[
            int,
            tuple[DoorwayData, ...],
        ] = {}
        self._reset_viewer_doorway_snapshots()
        self._doorway_mesh_update_delay_seconds = (
            DEFAULT_DOORWAY_MESH_UPDATE_DELAY_SECONDS
        )
        self._is_doorway_dimension_drag_active = False
        self._pending_doorway_mesh_level_index: int | None = None
        self._doorway_outline_commit_revision: int | None = None
        self._doorway_mesh_update_timer = QTimer(self)
        self._doorway_mesh_update_timer.setSingleShot(True)
        self._doorway_mesh_update_timer.setInterval(
            round(self._doorway_mesh_update_delay_seconds * 1000.0)
        )
        self._doorway_mesh_update_timer.timeout.connect(
            self._commit_pending_doorway_mesh_update
        )
        self._canvas_3d_viewer_is_external = False
        self._atlas_generation_signature: tuple[tuple[object, ...], ...] | None = None
        self._atlas_source_content_paths: (
            dict[str, tuple[tuple[object, ...], ...]] | None
        ) = None
        self._atlas_source_content_revisions: (
            dict[str, tuple[tuple[object, ...], ...]] | None
        ) = None
        self._atlas_pending_source_content_refresh_ids: set[str] = set()
        self._atlas_wall_texture_source_ids: set[str] = set()
        self._atlas_preview_variant_key: tuple[object, ...] | None = None
        self._image_preview_source_key: tuple[object, ...] | None = None
        self._image_preview_source_pixmap: QPixmap | None = None
        self._image_preview_scaled_key: tuple[object, ...] | None = None
        self._image_thumbnail_source_keys: dict[
            str,
            tuple[object, ...],
        ] = {}
        self._level_blueprint_image_revisions: dict[
            int,
            tuple[object, ...],
        ] = {}
        self._canvas_window_undo_ids: list[str] = []
        self._object_placement_dialog: ObjectPlacementDialog | None = None
        self._object_placement_operation_id: str | None = None
        self._is_shutdown = False
        self.texture_creator_level_index: int | None = None
        self.texture_creator_room_index: int | None = None
        self.texture_creator_wall_key: str | None = None
        self._build_ui()

    @property
    def vertex_data(self):
        return self.current_level.vertex_data

    @property
    def current_level(self) -> LevelData:
        return self.levels[self.current_level_index]

    def shutdown(self) -> None:
        """Release detached viewers and background work exactly once."""

        if self._is_shutdown:
            return
        self._is_shutdown = True
        self._cancel_pending_doorway_mesh_update(clear_outline=True)
        try:
            self.settings_widget.settings_changed.disconnect(
                self._handle_generation_settings_changed
            )
        except (RuntimeError, TypeError):
            pass
        self.settings_widget.dispose()
        self._close_object_placement_dialog()
        self._external_viewer_host.dispose()
        self.surface_texture_generation.shutdown()
        self.generation.shutdown()
        self.jobs_window.dispose()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.shutdown()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(self.workspace_splitter, 1)

        self.workspace_tabs = QTabWidget()
        # Hidden pages contain wide tool rows whose size hints must not push
        # the visible Canvas panel beyond the actual window bounds.
        workspace_tabs_policy = self.workspace_tabs.sizePolicy()
        workspace_tabs_policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self.workspace_tabs.setSizePolicy(workspace_tabs_policy)
        self.canvas = BlueprintCanvas()
        self.viewer = GlbViewerWidget(window_editing_enabled=True)
        self.canvas_3d_navigation_shortcut = QShortcut(self.viewer)
        self.canvas_3d_navigation_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.canvas_3d_navigation_shortcut.activated.connect(
            self._toggle_canvas_3d_navigation_mode
        )
        self.canvas_viewer_workspace = self._build_canvas_viewer_workspace()
        self._external_viewer_host = ExternalFullscreenViewerHost(self)
        self._external_viewer_host.close_requested.connect(
            self._handle_external_viewer_window_closed
        )
        self.job_manager = GenerationJobManager(self)
        self.jobs_window = JobsWindow(self.job_manager, self)
        self.generation = GenerationWorkspace(
            asset_directory=(
                self._application_settings.path.parent / "generated"
            ),
            job_manager=self.job_manager,
        )
        self.generation.set_texture_resolution_change_handler(
            self._handle_generation_texture_resolution_change_requested
        )
        self.generation.set_object_packing_change_handler(
            self._handle_generation_object_packing_change_requested
        )
        self.texture_atlas_workspace = TextureAtlasWorkspace(
            asset_directory=(
                self._application_settings.path.parent / "texture_atlases"
            )
        )
        self.atlas_object_preview_viewer = GlbViewerWidget(
            self.texture_atlas_workspace
        )
        self.atlas_object_preview_viewer.setObjectName(
            "texture_atlas_object_preview_viewer"
        )
        self.atlas_object_preview_viewer.set_ambient_light_intensity(1.0)
        self.atlas_object_preview_viewer.hide()
        self.atlas_object_preview_viewer.delete_requested.connect(
            self.texture_atlas_workspace.remove_selected_texture_from_atlas
        )
        self.texture_atlas_workspace.object_preview_requested.connect(
            self._handle_atlas_object_preview_requested
        )
        self.texture_atlas_workspace.object_preview_clear_requested.connect(
            self._clear_atlas_object_preview
        )
        self.texture_atlas_workspace.object_texture_resolution_changed.connect(
            self._handle_atlas_object_texture_resolution_changed
        )
        self.surface_texture_generation = SurfaceTextureGenerationWorkspace(
            asset_directory=(
                self._application_settings.path.parent / "surface_textures"
            ),
            application_settings=self._application_settings,
            job_manager=self.job_manager,
        )
        self.surface_texture_generation.set_texture_resolution_change_handler(
            self._handle_surface_texture_resolution_change_requested
        )
        self.settings_widget = SettingsWidget(
            application_settings=self._application_settings
        )
        generation_settings = self.settings_widget.get_settings()
        self._set_doorway_mesh_update_delay_seconds(
            generation_settings.doorway_mesh_update_delay_seconds
        )
        self._set_canvas_3d_navigation_shortcut(
            generation_settings.canvas_3d_navigation_toggle_hotkey
        )
        self.generation.set_runtime_settings(generation_settings)
        self.surface_texture_generation.set_runtime_settings(
            generation_settings
        )
        self.surface_texture_generation.generation_completed.connect(
            self._handle_surface_texture_generation_completed
        )
        self.surface_texture_generation.data_changed.connect(
            self._handle_surface_texture_data_changed_for_atlases
        )
        self.surface_texture_generation.assignments_removed.connect(
            self._handle_surface_texture_assignments_removed_for_atlases
        )
        self.surface_texture_generation.surface_content_changed.connect(
            self._handle_surface_texture_content_changed
        )
        self.generation.data_changed.connect(
            self._handle_generation_data_changed_for_atlases
        )
        self.generation.generated_object_changed.connect(
            self._handle_generated_object_changed_for_atlases
        )
        self.generation.generated_object_deleted.connect(
            self._handle_generated_object_deleted_for_atlases
        )
        self.generation.generation_completed.connect(
            self._handle_generated_object_completed_for_canvas
        )
        self.generation.generated_object_changed.connect(
            self._handle_generated_object_changed_for_canvas
        )
        self.generation.generated_object_placement_changed.connect(
            self._handle_generated_object_placement_changed_for_canvas
        )
        self.generation.generated_object_deleted.connect(
            self._handle_generated_object_deleted_for_canvas
        )
        self.generation.placement_requested.connect(
            self._handle_object_placement_requested
        )
        self.generation.operation_finished.connect(
            self._handle_object_placement_operation_finished
        )
        self.generation.placement_request_finished.connect(
            self._handle_object_placement_operation_finished
        )
        self.surface_texture_generation.set_levels(self.levels)
        self.workspace_tabs.addTab(self.canvas_viewer_workspace, "Canvas")
        self.workspace_tabs.addTab(self.texture_atlas_workspace, "Atlas")
        self.workspace_tabs.addTab(
            self.surface_texture_generation,
            "Surface texture generation",
        )
        self.workspace_tabs.addTab(self.generation, "Object generation")
        self.workspace_tabs.addTab(self.settings_widget, "Settings")
        self.workspace_tabs.currentChanged.connect(
            self._handle_workspace_tab_changed
        )
        self.viewer.wall_selected.connect(self._handle_viewer_wall_selected)
        self.viewer.window_placement_requested.connect(
            self._handle_canvas_window_placement_requested
        )
        self.viewer.window_undo_requested.connect(
            self._handle_canvas_window_undo_requested
        )
        self.viewer.placed_object_transform_changed.connect(
            self._handle_placed_object_transform_changed
        )
        self.viewer.navigation_mode_changed.connect(
            self._handle_canvas_3d_navigation_mode_changed
        )
        self.settings_widget.settings_changed.connect(
            self._handle_generation_settings_changed
        )
        self.workspace_splitter.addWidget(self.workspace_tabs)

        self.side_panel = QWidget()
        # The side-tab hint includes its widest hidden page. Let the splitter
        # use the available width instead of treating that hint as a minimum.
        side_panel_policy = self.side_panel.sizePolicy()
        side_panel_policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self.side_panel.setSizePolicy(side_panel_policy)
        side_layout = QVBoxLayout(self.side_panel)
        side_layout.setContentsMargins(16, 16, 16, 16)
        side_layout.setSpacing(12)

        self.side_tabs = QTabWidget()
        side_layout.addWidget(self.side_tabs, 1)

        generals_tab = QScrollArea()
        generals_tab.setWidgetResizable(True)
        generals_tab.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        generals_content = QWidget()
        generals_layout = QVBoxLayout(generals_content)
        generals_layout.setContentsMargins(10, 12, 10, 10)
        generals_layout.setSpacing(12)
        generals_tab.setWidget(generals_content)
        self.side_tabs.addTab(generals_tab, "Generals")
        side_layout = generals_layout

        height_label = QLabel("Height level")
        height_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(height_label)

        self.height_level_spinbox = QDoubleSpinBox()
        self.height_level_spinbox.setRange(0.1, 100.0)
        self.height_level_spinbox.setDecimals(2)
        self.height_level_spinbox.setSingleStep(0.1)
        self.height_level_spinbox.setValue(DEFAULT_WALL_HEIGHT_METERS)
        self.height_level_spinbox.setSuffix(" m")
        self.height_level_spinbox.setMinimumHeight(40)
        self.height_level_spinbox.valueChanged.connect(self._handle_height_level_changed)
        side_layout.addWidget(self.height_level_spinbox)

        floor_thickness_label = QLabel("Floor thickness")
        floor_thickness_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(floor_thickness_label)

        self.floor_thickness_spinbox = QDoubleSpinBox()
        self.floor_thickness_spinbox.setRange(
            MIN_FLOOR_THICKNESS_METERS,
            MAX_FLOOR_THICKNESS_METERS,
        )
        self.floor_thickness_spinbox.setDecimals(2)
        self.floor_thickness_spinbox.setSingleStep(0.05)
        self.floor_thickness_spinbox.setValue(DEFAULT_FLOOR_THICKNESS_METERS)
        self.floor_thickness_spinbox.setSuffix(" m")
        self.floor_thickness_spinbox.setMinimumHeight(40)
        self.floor_thickness_spinbox.valueChanged.connect(
            self._handle_floor_thickness_changed
        )
        side_layout.addWidget(self.floor_thickness_spinbox)

        level_scale_label = QLabel("Level scale")
        level_scale_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(level_scale_label)

        self.level_scale_spinbox = QDoubleSpinBox()
        self.level_scale_spinbox.setRange(MIN_LEVEL_SCALE, MAX_LEVEL_SCALE)
        self.level_scale_spinbox.setDecimals(3)
        self.level_scale_spinbox.setSingleStep(0.05)
        self.level_scale_spinbox.setValue(DEFAULT_LEVEL_SCALE)
        self.level_scale_spinbox.setSuffix(" x")
        self.level_scale_spinbox.setMinimumHeight(40)
        self.level_scale_spinbox.valueChanged.connect(
            self._handle_level_scale_changed
        )
        side_layout.addWidget(self.level_scale_spinbox)

        level_x_offset_label = QLabel("X offset")
        level_x_offset_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(level_x_offset_label)

        self.level_x_offset_spinbox = QDoubleSpinBox()
        self.level_x_offset_spinbox.setRange(
            MIN_LEVEL_OFFSET_METERS,
            MAX_LEVEL_OFFSET_METERS,
        )
        self.level_x_offset_spinbox.setDecimals(2)
        self.level_x_offset_spinbox.setSingleStep(0.1)
        self.level_x_offset_spinbox.setValue(DEFAULT_LEVEL_OFFSET_METERS)
        self.level_x_offset_spinbox.setSuffix(" m")
        self.level_x_offset_spinbox.setMinimumHeight(40)
        self.level_x_offset_spinbox.valueChanged.connect(
            self._handle_level_x_offset_changed
        )
        side_layout.addWidget(self.level_x_offset_spinbox)

        level_y_offset_label = QLabel("Y offset")
        level_y_offset_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(level_y_offset_label)

        self.level_y_offset_spinbox = QDoubleSpinBox()
        self.level_y_offset_spinbox.setRange(
            MIN_LEVEL_OFFSET_METERS,
            MAX_LEVEL_OFFSET_METERS,
        )
        self.level_y_offset_spinbox.setDecimals(2)
        self.level_y_offset_spinbox.setSingleStep(0.1)
        self.level_y_offset_spinbox.setValue(DEFAULT_LEVEL_OFFSET_METERS)
        self.level_y_offset_spinbox.setSuffix(" m")
        self.level_y_offset_spinbox.setMinimumHeight(40)
        self.level_y_offset_spinbox.valueChanged.connect(
            self._handle_level_y_offset_changed
        )
        side_layout.addWidget(self.level_y_offset_spinbox)

        self.floor_contour_status_label = QLabel("Floor contour: Not set")
        side_layout.addWidget(self.floor_contour_status_label)

        floor_contour_buttons_layout = QHBoxLayout()
        floor_contour_buttons_layout.setSpacing(10)

        self.set_floor_contour_button = QPushButton("Set floor contour")
        self.set_floor_contour_button.setMinimumHeight(40)
        self.set_floor_contour_button.clicked.connect(
            self._handle_set_floor_contour_clicked
        )
        floor_contour_buttons_layout.addWidget(self.set_floor_contour_button)

        self.clear_floor_contour_button = QPushButton("Clear floor contour")
        self.clear_floor_contour_button.setMinimumHeight(40)
        self.clear_floor_contour_button.clicked.connect(
            self._handle_clear_floor_contour_clicked
        )
        floor_contour_buttons_layout.addWidget(self.clear_floor_contour_button)
        side_layout.addLayout(floor_contour_buttons_layout)

        stairs_label = QLabel("Stairs")
        stairs_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(stairs_label)

        self.stair_style_combo = QComboBox()
        self.stair_style_combo.addItem("Supported", STAIR_STYLE_SUPPORTED)
        self.stair_style_combo.addItem("Floating", STAIR_STYLE_FLOATING)
        self.stair_style_combo.addItem(
            "Floating with riser",
            STAIR_STYLE_FLOATING_WITH_RISER,
        )
        self.stair_style_combo.setCurrentIndex(0)
        self.stair_style_combo.setMinimumHeight(34)
        stair_style_layout = QFormLayout()
        stair_style_layout.setContentsMargins(0, 0, 0, 0)
        stair_style_layout.addRow("Stair type", self.stair_style_combo)
        side_layout.addLayout(stair_style_layout)

        self.stair_status_label = QLabel("Stairs: none")
        self.stair_status_label.setWordWrap(True)
        side_layout.addWidget(self.stair_status_label)

        self.add_stairs_button = QPushButton("Add stairs")
        self.add_stairs_button.setMinimumHeight(40)
        self.add_stairs_button.clicked.connect(self._handle_add_stairs_clicked)
        side_layout.addWidget(self.add_stairs_button)

        doorway_label = QLabel("Doorways")
        doorway_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(doorway_label)

        self.selected_doorway_arch_checkbox = QCheckBox(
            "Arch selected doorway"
        )
        self.selected_doorway_arch_checkbox.setEnabled(False)
        self.selected_doorway_arch_checkbox.setToolTip(
            "Select a placed doorway on the Canvas, then enable this to "
            "replace its flat top with an arch."
        )
        self.selected_doorway_arch_checkbox.toggled.connect(
            self._handle_selected_doorway_arch_toggled
        )
        side_layout.addWidget(self.selected_doorway_arch_checkbox)

        self.selected_doorway_arch_amount_spinbox = QDoubleSpinBox()
        self.selected_doorway_arch_amount_spinbox.setRange(
            MIN_DOORWAY_ARCH_AMOUNT * 100.0,
            MAX_DOORWAY_ARCH_AMOUNT * 100.0,
        )
        self.selected_doorway_arch_amount_spinbox.setDecimals(1)
        self.selected_doorway_arch_amount_spinbox.setSingleStep(1.0)
        self.selected_doorway_arch_amount_spinbox.setKeyboardTracking(False)
        self.selected_doorway_arch_amount_spinbox.setSuffix(" %")
        self.selected_doorway_arch_amount_spinbox.setValue(
            DEFAULT_DOORWAY_ARCH_AMOUNT * 100.0
        )
        self.selected_doorway_arch_amount_spinbox.setMinimumHeight(34)
        self.selected_doorway_arch_amount_spinbox.setEnabled(False)
        self.selected_doorway_arch_amount_spinbox.setToolTip(
            "Control how far the selected doorway's top rises into an arch."
        )
        self.selected_doorway_arch_amount_spinbox.valueChanged.connect(
            self._handle_selected_doorway_arch_amount_changed
        )
        doorway_arch_form = QFormLayout()
        doorway_arch_form.setContentsMargins(0, 0, 0, 0)
        doorway_arch_form.addRow(
            "Arch amount",
            self.selected_doorway_arch_amount_spinbox,
        )
        side_layout.addLayout(doorway_arch_form)

        self.doorway_preset_list = QListWidget()
        self.doorway_preset_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.doorway_preset_list.setMinimumHeight(112)
        self.doorway_preset_list.currentRowChanged.connect(
            self._handle_doorway_preset_selection_changed
        )
        side_layout.addWidget(self.doorway_preset_list)

        self.save_doorway_template_button = QPushButton(
            "Save doorway template"
        )
        self.save_doorway_template_button.setMinimumHeight(40)
        self.save_doorway_template_button.setEnabled(False)
        self.save_doorway_template_button.clicked.connect(
            self._handle_save_doorway_template_clicked
        )
        side_layout.addWidget(self.save_doorway_template_button)

        doorway_buttons_layout = QHBoxLayout()
        doorway_buttons_layout.setSpacing(10)

        self.remove_doorway_preset_button = QPushButton("Remove selected preset")
        self.remove_doorway_preset_button.setMinimumHeight(40)
        self.remove_doorway_preset_button.clicked.connect(
            self._handle_remove_doorway_preset_clicked
        )
        doorway_buttons_layout.addWidget(self.remove_doorway_preset_button)

        self.place_doorway_button = QPushButton("Place selected doorway")
        self.place_doorway_button.setMinimumHeight(40)
        self.place_doorway_button.clicked.connect(
            self._handle_place_selected_doorway_clicked
        )
        doorway_buttons_layout.addWidget(self.place_doorway_button)
        side_layout.addLayout(doorway_buttons_layout)

        self.load_image_button = QPushButton("Load image")
        self.load_image_button.setMinimumHeight(44)
        self.load_image_button.clicked.connect(self._handle_load_image_clicked)
        side_layout.addWidget(self.load_image_button)

        snap_label = QLabel("Snap")
        snap_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(snap_label)

        self.snap_middle_equal_angle_radio = QRadioButton(
            "Snap to middle equal angle only"
        )
        self.snap_middle_equal_angle_radio.setAutoExclusive(False)
        self.snap_middle_equal_angle_radio.setChecked(True)
        self.snap_middle_equal_angle_radio.toggled.connect(
            self._handle_snap_middle_equal_angle_toggled
        )
        side_layout.addWidget(self.snap_middle_equal_angle_radio)

        self.blueprint_name_label = QLabel("Image: none for this level")
        self.blueprint_name_label.setWordWrap(True)
        side_layout.addWidget(self.blueprint_name_label)

        levels_label = QLabel("Levels")
        levels_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(levels_label)

        self.levels_list = QListWidget()
        self.levels_list.currentRowChanged.connect(self._handle_level_selection_changed)
        side_layout.addWidget(self.levels_list, 1)

        level_options_layout = QFormLayout()
        level_options_layout.setContentsMargins(0, 0, 0, 0)
        level_options_layout.setSpacing(8)
        include_widget = QWidget()
        include_layout = QHBoxLayout(include_widget)
        include_layout.setContentsMargins(0, 0, 0, 0)
        include_layout.setSpacing(12)

        self.include_button_group = QButtonGroup(self)
        self.include_yes_radio = QRadioButton("Yes")
        self.include_no_radio = QRadioButton("No")
        self.include_button_group.addButton(self.include_yes_radio)
        self.include_button_group.addButton(self.include_no_radio)
        self.include_yes_radio.toggled.connect(self._handle_include_toggled)
        self.include_no_radio.toggled.connect(self._handle_include_toggled)
        include_layout.addWidget(self.include_yes_radio)
        include_layout.addWidget(self.include_no_radio)
        include_layout.addStretch(1)
        level_options_layout.addRow("Include", include_widget)
        side_layout.addLayout(level_options_layout)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        self.save_button = QPushButton("Save")
        self.save_button.setMinimumHeight(56)
        self.save_button.clicked.connect(self._handle_save_clicked)
        buttons_layout.addWidget(self.save_button)

        self.load_button = QPushButton("Load")
        self.load_button.setMinimumHeight(56)
        self.load_button.clicked.connect(self._handle_load_clicked)
        buttons_layout.addWidget(self.load_button)

        self.export_button = QPushButton("GLB")
        self.export_button.setMinimumHeight(56)
        self.export_button.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.export_button.clicked.connect(self._handle_glb_export_clicked)
        buttons_layout.addWidget(self.export_button, 1)

        self.png_export_button = QPushButton("PNG")
        self.png_export_button.setMinimumHeight(56)
        self.png_export_button.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.png_export_button.clicked.connect(self._handle_png_export_clicked)
        buttons_layout.addWidget(self.png_export_button, 1)
        side_layout.addLayout(buttons_layout)

        self._refresh_doorway_preset_list(selected_index=0)

        self._generals_spinbox_wheel_filter = RightPanelSpinBoxWheelFilter(
            generals_tab
        )
        for spinbox in generals_content.findChildren(QAbstractSpinBox):
            spinbox.installEventFilter(self._generals_spinbox_wheel_filter)
            spinbox.lineEdit().installEventFilter(
                self._generals_spinbox_wheel_filter
            )

        self.uvs_tab = self._build_uvs_tab()
        self.side_tabs.addTab(self.uvs_tab, "UVs")
        self.images_tab = self._build_images_tab()
        self.side_tabs.addTab(self.images_tab, "Images")
        self.side_tabs.addTab(self._build_texture_creator_tab(), "Texture creator")
        self.side_tabs.currentChanged.connect(self._handle_side_tab_changed)

        self.workspace_splitter.addWidget(self.side_panel)
        self.workspace_splitter.setStretchFactor(0, 9)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes([1160, 440])

        self.canvas.rooms_changed.connect(
            self._refresh_room_dependent_controls
        )
        self.canvas.rooms_changed.connect(self._schedule_viewer_preview_refresh)
        self.canvas.floor_contour_changed.connect(
            self._handle_floor_contour_changed
        )
        self.canvas.doorways_changed.connect(self._handle_doorways_changed)
        self.canvas.doorway_dimension_preview_changed.connect(
            self._handle_doorway_dimension_preview_changed
        )
        self.canvas.doorway_dimension_drag_started.connect(
            self._handle_doorway_dimension_drag_started
        )
        self.canvas.doorway_dimension_drag_finished.connect(
            self._handle_doorway_dimension_drag_finished
        )
        self.canvas.selected_doorway_changed.connect(
            self._handle_canvas_doorway_selection_changed
        )
        self.canvas.stair_start_placed.connect(
            self._handle_stair_start_placed
        )
        self.canvas.stair_placement_ready.connect(
            self._handle_stair_placement_ready
        )
        self.canvas.stair_placement_completed.connect(
            self._handle_stair_placement_completed
        )
        self.canvas.stair_placement_cancelled.connect(
            self._handle_stair_placement_cancelled
        )
        self.canvas.stair_placement_invalid_endpoint.connect(
            self._handle_stair_placement_invalid_endpoint
        )
        self.canvas.stair_delete_requested.connect(
            self._handle_stair_delete_requested
        )
        self._refresh_levels_list()
        self._update_stair_button_state()
        self._refresh_room_dependent_controls()
        self._sync_level_controls()
        self._sync_canvas_to_current_level()
        self._sync_texture_creator_tab()
        self._schedule_viewer_preview_refresh(preserve_camera=False)
        self._apply_fullscreen_3d_viewer_screen(
            generation_settings.fullscreen_3d_viewer_screen_id
        )
        self._apply_jobs_window_screen(
            generation_settings.jobs_window_screen_id
        )

    def _build_canvas_viewer_workspace(self) -> QWidget:
        """Place the editable blueprint and its 3D preview in local tabs."""

        workspace = QWidget()
        workspace.setObjectName("canvas-viewer-workspace")
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas_viewer_tabs = QTabWidget()
        self.canvas_viewer_tabs.setObjectName("canvas-viewer-tabs")
        self.canvas_2d_view_tab_index = self.canvas_viewer_tabs.addTab(
            self.canvas,
            "2D view",
        )
        self.canvas_3d_view_tab_index = self.canvas_viewer_tabs.addTab(
            self.viewer,
            "3D view",
        )
        self.canvas_viewer_tabs.currentChanged.connect(
            self._handle_canvas_viewer_subtab_changed
        )
        workspace_layout.addWidget(self.canvas_viewer_tabs)
        return workspace

    def set_canvas_3d_viewer_external_display_active(
        self,
        is_active: bool,
    ) -> None:
        """Keep the Canvas workspace focused on 2D while 3D is external."""

        is_active = bool(is_active)
        if is_active == self._canvas_3d_viewer_is_external:
            return
        self._canvas_3d_viewer_is_external = is_active
        if is_active:
            self.canvas_viewer_tabs.setCurrentIndex(
                self.canvas_2d_view_tab_index
            )
            self.canvas_viewer_tabs.setTabEnabled(
                self.canvas_3d_view_tab_index,
                False,
            )
            self.canvas_viewer_tabs.tabBar().setVisible(False)
            return

        self.canvas_viewer_tabs.setTabEnabled(
            self.canvas_3d_view_tab_index,
            True,
        )
        self.canvas_viewer_tabs.tabBar().setVisible(True)

    def _handle_canvas_viewer_subtab_changed(self, tab_index: int) -> None:
        """Give the standard viewer keyboard focus when its tab opens."""

        if tab_index == self.canvas_3d_view_tab_index:
            self.viewer.focus_navigation()
            self._ensure_viewer_preview_current(preserve_camera=False)

    def _set_canvas_3d_navigation_shortcut(self, hotkey: str) -> None:
        """Apply the persisted Canvas-only navigation shortcut."""

        self.canvas_3d_navigation_shortcut.setKey(
            QKeySequence(hotkey, QKeySequence.SequenceFormat.PortableText)
        )

    def _toggle_canvas_3d_navigation_mode(self) -> None:
        """Toggle the active Canvas viewer between orbit and first person."""

        if self.workspace_tabs.currentWidget() is not self.canvas_viewer_workspace:
            return
        if (
            self._external_viewer_host.is_active
            and self._external_viewer_host.viewer is not self.viewer
        ):
            return

        self.viewer.toggle_navigation_mode()

    def _handle_canvas_3d_navigation_mode_changed(self, mode: str) -> None:
        """Reflect the standard viewer's navigation mode in its local tab."""

        tab_label = "3D view"
        if mode == NAVIGATION_MODE_FIRST_PERSON:
            tab_label = "3D view (first person)"
        self.canvas_viewer_tabs.setTabText(
            self.canvas_3d_view_tab_index,
            tab_label,
        )

    def _handle_canvas_window_placement_requested(
        self,
        raw_placement: object,
    ) -> None:
        """Commit one validated Canvas rectangle as a real wall opening."""

        if not isinstance(raw_placement, WallWindowPlacement):
            self.viewer.set_window_tools_status(
                "The requested window placement is invalid."
            )
            return

        try:
            window = add_wall_window(self.levels, raw_placement)
        except (TypeError, ValueError) as error:
            self.viewer.set_window_tools_status(f"Window not added: {error}")
            return

        try:
            validated_build = self._build_model_with_stable_dependencies(
                lambda: self._build_viewer_preview_model(None)
            )
        except Exception as error:
            self._rollback_canvas_window(window.window_id)
            self.viewer.set_window_tools_status(f"Window not added: {error}")
            return
        if validated_build is None:
            self._rollback_canvas_window(window.window_id)
            self.viewer.set_window_tools_status(
                "Window not added because the updated model could not be built."
            )
            return
        generated_model, dependency_signature = validated_build

        try:
            self._apply_canvas_window_preview(
                generated_model,
                dependency_signature=dependency_signature,
            )
        except Exception as error:
            self._rollback_canvas_window(window.window_id)
            self._restore_canvas_window_preview_after_rollback()
            self.viewer.set_window_tools_status(f"Window not added: {error}")
            return

        self._canvas_window_undo_ids.append(window.window_id)
        self._sync_canvas_window_undo_availability()
        self.viewer.set_window_tools_status("Window added.")

    def _handle_canvas_window_undo_requested(self) -> None:
        """Undo the latest successfully committed Canvas window transaction."""

        self._sync_canvas_window_undo_availability()
        if not self._canvas_window_undo_ids:
            self.viewer.set_window_tools_status("No added window to undo.")
            return

        window_id = self._canvas_window_undo_ids[-1]
        removed = self._remove_canvas_window(window_id)
        if removed is None:
            self._sync_canvas_window_undo_availability()
            self.viewer.set_window_tools_status("No added window to undo.")
            return
        level, window_index, window = removed

        try:
            validated_build = self._build_model_with_stable_dependencies(
                lambda: self._build_viewer_preview_model(None)
            )
        except Exception as error:
            level.windows.insert(window_index, window)
            self._sync_canvas_window_undo_availability()
            self.viewer.set_window_tools_status(
                f"Window could not be undone: {error}"
            )
            return
        if validated_build is None:
            level.windows.insert(window_index, window)
            self._sync_canvas_window_undo_availability()
            self.viewer.set_window_tools_status(
                "Window could not be undone because the model could not be built."
            )
            return
        generated_model, dependency_signature = validated_build

        try:
            self._apply_canvas_window_preview(
                generated_model,
                dependency_signature=dependency_signature,
            )
        except Exception as error:
            level.windows.insert(window_index, window)
            self._restore_canvas_window_preview_after_rollback()
            self._sync_canvas_window_undo_availability()
            self.viewer.set_window_tools_status(
                f"Window could not be undone: {error}"
            )
            return

        self._canvas_window_undo_ids.pop()
        self._sync_canvas_window_undo_availability()
        self.viewer.set_window_tools_status("Window undone.")

    def _apply_canvas_window_preview(
        self,
        generated_model: GeneratedModel,
        *,
        dependency_signature: tuple[object, ...],
    ) -> bool:
        """Commit one validated window model to the active Canvas consumer."""

        wall_targets = tuple(
            build_fixed_surfaces(self._build_viewer_preview_levels())
        )
        self.viewer.set_wall_targets(wall_targets)
        self.viewer.set_model(generated_model, preserve_camera=True)
        self._mark_viewer_preview_dirty(preserve_camera=True)
        if self._remember_current_canvas_preview_model(
            generated_model,
            validated_dependency_signature=dependency_signature,
        ):
            return True
        self._queue_viewer_preview_refresh()
        return False

    def _restore_canvas_window_preview_after_rollback(self) -> None:
        """Best-effort repair after a display refresh failed mid-transaction."""

        try:
            validated_build = self._build_model_with_stable_dependencies(
                lambda: self._build_viewer_preview_model(None)
            )
            if validated_build is not None:
                generated_model, dependency_signature = validated_build
                self._apply_canvas_window_preview(
                    generated_model,
                    dependency_signature=dependency_signature,
                )
        except Exception:
            return

    def _sync_canvas_window_undo_availability(self) -> None:
        """Discard stale history IDs and synchronize the Canvas undo button."""

        while self._canvas_window_undo_ids:
            window_id = self._canvas_window_undo_ids[-1]
            if self._find_canvas_window(window_id) is not None:
                break
            self._canvas_window_undo_ids.pop()
        self.viewer.set_window_undo_available(
            bool(self._canvas_window_undo_ids)
        )

    def _find_canvas_window(
        self,
        window_id: str,
    ) -> tuple[LevelData, int, WindowData] | None:
        """Locate one exact committed window without relying on active level."""

        normalized_id = str(window_id)
        for level in self.levels:
            for index, window in enumerate(level.windows):
                if window.window_id == normalized_id:
                    return level, index, window
        return None

    def _remove_canvas_window(
        self,
        window_id: str,
    ) -> tuple[LevelData, int, WindowData] | None:
        """Remove and return one window so a failed undo can restore its index."""

        found = self._find_canvas_window(window_id)
        if found is None:
            return None
        level, index, window = found
        del level.windows[index]
        return level, index, window

    def _rollback_canvas_window(self, window_id: str) -> None:
        """Remove only the just-created window after a pre-display failure."""

        self._remove_canvas_window(window_id)

    def _build_uvs_tab(self) -> QWidget:
        uvs_tab = QWidget()
        uvs_layout = QVBoxLayout(uvs_tab)
        uvs_layout.setContentsMargins(10, 12, 10, 10)
        uvs_layout.setSpacing(12)

        uv_map_label = QLabel("UV Map")
        uv_map_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        uvs_layout.addWidget(uv_map_label)

        self.uv_canvas = UvCanvas()
        self.uv_canvas.selected_wall_changed.connect(self._handle_uv_wall_selected)
        self.uv_canvas.uv_values_changed.connect(self._handle_uv_values_changed)
        uvs_layout.addWidget(self.uv_canvas, 2)

        uv_controls_layout = QFormLayout()
        uv_controls_layout.setContentsMargins(0, 0, 0, 0)
        uv_controls_layout.setSpacing(8)

        self.unoccupied_uv_pixels_label = QLabel("Unoccupied pixels: 0")
        self.unoccupied_uv_pixels_label.setMinimumHeight(24)
        uv_controls_layout.addRow("", self.unoccupied_uv_pixels_label)

        self.uv_aspect_ratio_label = QLabel("Aspect ratio: none")
        self.uv_aspect_ratio_label.setMinimumHeight(24)
        uv_controls_layout.addRow("", self.uv_aspect_ratio_label)

        optimize_layout = QHBoxLayout()
        optimize_layout.setContentsMargins(0, 0, 0, 0)
        optimize_layout.setSpacing(8)

        self.optimize_uv_button = QPushButton("Optimize")
        self.optimize_uv_button.setMinimumHeight(34)
        self.optimize_uv_button.clicked.connect(self._handle_optimize_uv_clicked)
        optimize_layout.addWidget(self.optimize_uv_button, 1)

        self.optimize_all_uv_button = QPushButton("Optimize all")
        self.optimize_all_uv_button.setMinimumHeight(34)
        self.optimize_all_uv_button.clicked.connect(
            self._handle_optimize_all_uv_clicked
        )
        optimize_layout.addWidget(self.optimize_all_uv_button, 1)

        self.uv_optimization_mode_group = QButtonGroup(self)
        self.uv_optimization_mode_group.setExclusive(True)
        self.basic_optimization_radio = QRadioButton("Basic")
        self.free_placement_radio = QRadioButton("Free placement")
        self.subdivision_optimization_radio = QRadioButton("Subdivision")
        self.uv_optimization_mode_group.addButton(self.basic_optimization_radio)
        self.uv_optimization_mode_group.addButton(self.free_placement_radio)
        self.uv_optimization_mode_group.addButton(
            self.subdivision_optimization_radio
        )
        self.basic_optimization_radio.toggled.connect(
            self._handle_optimization_mode_toggled
        )
        self.free_placement_radio.toggled.connect(
            self._handle_optimization_mode_toggled
        )
        self.subdivision_optimization_radio.toggled.connect(
            self._handle_optimization_mode_toggled
        )
        self.subdivision_optimization_radio.setChecked(True)
        optimize_layout.addWidget(self.basic_optimization_radio)
        optimize_layout.addWidget(self.free_placement_radio)
        optimize_layout.addWidget(self.subdivision_optimization_radio)
        uv_controls_layout.addRow("", optimize_layout)

        self.complex_optimization_passes_spinbox = QSpinBox()
        self.complex_optimization_passes_spinbox.setRange(1, 100)
        self.complex_optimization_passes_spinbox.setValue(3)
        self.complex_optimization_passes_spinbox.setMinimumHeight(34)
        uv_controls_layout.addRow(
            "Free placement passes",
            self.complex_optimization_passes_spinbox,
        )

        self.reset_uv_defaults_button = QPushButton("Reset defaults")
        self.reset_uv_defaults_button.setMinimumHeight(34)
        self.reset_uv_defaults_button.clicked.connect(
            self._handle_reset_uv_defaults_clicked
        )
        uv_controls_layout.addRow("", self.reset_uv_defaults_button)

        self.uv_map_width_spinbox = PowerOfTwoSpinBox()
        self.uv_map_width_spinbox.setRange(64, 8192)
        self.uv_map_width_spinbox.valueChanged.connect(
            self._handle_uv_map_width_changed
        )
        uv_controls_layout.addRow("Map X", self.uv_map_width_spinbox)

        self.uv_map_height_spinbox = PowerOfTwoSpinBox()
        self.uv_map_height_spinbox.setRange(64, 8192)
        self.uv_map_height_spinbox.valueChanged.connect(
            self._handle_uv_map_height_changed
        )
        uv_controls_layout.addRow("Map Y", self.uv_map_height_spinbox)

        self.uv_wall_scale_spinbox = QDoubleSpinBox()
        self.uv_wall_scale_spinbox.setRange(0.01, 100.0)
        self.uv_wall_scale_spinbox.setDecimals(3)
        self.uv_wall_scale_spinbox.setSingleStep(0.05)
        self.uv_wall_scale_spinbox.valueChanged.connect(
            self._handle_uv_wall_scale_changed
        )
        uv_controls_layout.addRow("Wall scale", self.uv_wall_scale_spinbox)

        self.uv_wall_rotation_spinbox = DegreeSpinBox()
        self.uv_wall_rotation_spinbox.setRange(0, 359)
        self.uv_wall_rotation_spinbox.setSingleStep(1)
        self.uv_wall_rotation_spinbox.setWrapping(True)
        self.uv_wall_rotation_spinbox.setSuffix(" deg")
        self.uv_wall_rotation_spinbox.valueChanged.connect(
            self._handle_uv_wall_rotation_changed
        )
        uv_controls_layout.addRow("Wall rotation", self.uv_wall_rotation_spinbox)
        uvs_layout.addLayout(uv_controls_layout)

        uv_rooms_label = QLabel("Rooms")
        uv_rooms_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        uvs_layout.addWidget(uv_rooms_label)

        self.uv_rooms_list = QListWidget()
        self.uv_rooms_list.currentRowChanged.connect(
            self._handle_uv_room_selection_changed
        )
        uvs_layout.addWidget(self.uv_rooms_list, 1)
        return uvs_tab

    def _build_images_tab(self) -> QWidget:
        images_tab = QWidget()
        images_layout = QVBoxLayout(images_tab)
        images_layout.setContentsMargins(10, 12, 10, 10)
        images_layout.setSpacing(12)

        images_label = QLabel("Selected image")
        images_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        images_layout.addWidget(images_label)

        self.image_preview_label = QLabel("No image loaded")
        self.image_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview_label.setMinimumHeight(260)
        self.image_preview_label.setStyleSheet(
            "border: 1px solid #4b5563; background: #1f242b; color: #f5f7fa;"
        )
        images_layout.addWidget(self.image_preview_label, 1)

        self.image_path_label = QLabel("No image selected")
        self.image_path_label.setWordWrap(True)
        images_layout.addWidget(self.image_path_label)

        loaded_images_label = QLabel("Loaded images")
        loaded_images_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        images_layout.addWidget(loaded_images_label)

        self.image_thumbnail_list = QListWidget()
        self.image_thumbnail_list.setViewMode(QListView.ViewMode.IconMode)
        self.image_thumbnail_list.setFlow(QListView.Flow.LeftToRight)
        self.image_thumbnail_list.setWrapping(False)
        self.image_thumbnail_list.setMovement(QListView.Movement.Static)
        self.image_thumbnail_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.image_thumbnail_list.setIconSize(QSize(72, 72))
        self.image_thumbnail_list.setGridSize(QSize(104, 104))
        self.image_thumbnail_list.setMinimumHeight(120)
        self.image_thumbnail_list.setMaximumHeight(132)
        self.image_thumbnail_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.image_thumbnail_list.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.image_thumbnail_list.currentRowChanged.connect(
            self._handle_image_thumbnail_selection_changed
        )
        images_layout.addWidget(self.image_thumbnail_list)

        image_buttons_layout = QHBoxLayout()
        image_buttons_layout.setContentsMargins(0, 0, 0, 0)
        image_buttons_layout.setSpacing(10)

        self.images_load_button = QPushButton("Load image")
        self.images_load_button.setMinimumHeight(44)
        self.images_load_button.clicked.connect(
            self._handle_load_library_image_clicked
        )
        image_buttons_layout.addWidget(self.images_load_button)

        self.images_convert_video_button = QPushButton("Convert video to image")
        self.images_convert_video_button.setMinimumHeight(44)
        self.images_convert_video_button.clicked.connect(
            self._handle_convert_video_to_image_clicked
        )
        image_buttons_layout.addWidget(self.images_convert_video_button)

        self.images_save_png_button = QPushButton("Save png")
        self.images_save_png_button.setMinimumHeight(44)
        self.images_save_png_button.setEnabled(False)
        self.images_save_png_button.clicked.connect(
            self._handle_save_selected_image_clicked
        )
        image_buttons_layout.addWidget(self.images_save_png_button)

        self.images_delete_button = QPushButton("Delete image")
        self.images_delete_button.setMinimumHeight(44)
        self.images_delete_button.clicked.connect(
            self._handle_delete_image_clicked
        )
        image_buttons_layout.addWidget(self.images_delete_button)

        images_layout.addLayout(image_buttons_layout)
        return images_tab

    def _build_texture_creator_tab(self) -> QWidget:
        self.texture_creator_tab = QWidget()
        texture_layout = QVBoxLayout(self.texture_creator_tab)
        texture_layout.setContentsMargins(10, 12, 10, 10)
        texture_layout.setSpacing(12)

        texture_label = QLabel("Wall texture")
        texture_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        texture_layout.addWidget(texture_label)

        self.texture_creator_wall_label = QLabel("Select a wall in Viewer")
        self.texture_creator_wall_label.setWordWrap(True)
        texture_layout.addWidget(self.texture_creator_wall_label)

        self.texture_creator_aspect_ratio_label = QLabel("Aspect ratio: none")
        self.texture_creator_aspect_ratio_label.setWordWrap(True)
        texture_layout.addWidget(self.texture_creator_aspect_ratio_label)

        self.texture_creator_resolution_label = QLabel("Resolutions: none")
        self.texture_creator_resolution_label.setWordWrap(True)
        texture_layout.addWidget(self.texture_creator_resolution_label)

        texture_form_layout = QFormLayout()
        texture_form_layout.setContentsMargins(0, 0, 0, 0)
        texture_form_layout.setSpacing(8)

        self.texture_image_combo = QComboBox()
        self.texture_image_combo.setMinimumHeight(34)
        self.texture_image_combo.currentIndexChanged.connect(
            self._handle_texture_image_selection_changed
        )
        texture_form_layout.addRow("Image", self.texture_image_combo)
        texture_layout.addLayout(texture_form_layout)

        self.texture_creator_canvas = TextureCreatorCanvas()
        self.texture_creator_canvas.texture_changed.connect(
            self._handle_texture_creator_texture_changed
        )
        texture_layout.addWidget(self.texture_creator_canvas, 1)
        return self.texture_creator_tab

    def load_blueprint(self, file_path: str) -> None:
        self._set_current_level_image(file_path)

    def _handle_glb_export_clicked(self) -> None:
        default_path = Path.cwd() / "housemaker_export.glb"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export GLB",
            str(default_path),
            "GLB Files (*.glb)",
        )
        if not file_path:
            return

        # Export is an explicit completion boundary, so include any doorway
        # dimensions that were still waiting for their debounce interval.
        self._commit_pending_doorway_mesh_update()

        export_path = Path(file_path)
        if export_path.suffix.lower() != ".glb":
            export_path = export_path.with_suffix(".glb")

        try:
            validated_build = self._build_model_with_stable_dependencies(
                lambda: self._build_generated_model("Export failed")
            )
        except RuntimeError as error:
            QMessageBox.warning(self, "Export blocked", str(error))
            return
        if validated_build is None:
            return
        generated_model, dependency_signature = validated_build

        try:
            exported_path = export_glb_file(generated_model, export_path)
        except OSError as error:
            QMessageBox.critical(self, "Export failed", str(error))
            return

        self.workspace_tabs.setCurrentWidget(self.canvas_viewer_workspace)
        self.viewer.set_wall_targets(
            tuple(build_fixed_surfaces(self._build_viewer_preview_levels()))
        )
        self.viewer.set_model(generated_model)
        if not self._remember_current_canvas_preview_model(
            generated_model,
            validated_dependency_signature=dependency_signature,
        ):
            self._mark_viewer_preview_dirty(preserve_camera=False)
            self._queue_viewer_preview_refresh()
        QMessageBox.information(
            self,
            "GLB exported",
            f"Saved GLB to:\n{exported_path}",
        )

    def _handle_png_export_clicked(self) -> None:
        directory_path = QFileDialog.getExistingDirectory(
            self,
            "Export PNG textures",
            str(Path.cwd()),
        )
        if not directory_path:
            return

        try:
            exported_paths = export_room_texture_pngs(
                levels=self.levels,
                directory=directory_path,
            )
        except OSError as error:
            QMessageBox.critical(self, "PNG export failed", str(error))
            return

        if not exported_paths:
            QMessageBox.warning(
                self,
                "PNG export skipped",
                "No room textures are available to export.",
            )
            return

        QMessageBox.information(
            self,
            "PNG textures exported",
            f"Saved {len(exported_paths)} PNG texture(s) to:\n{directory_path}",
        )

    # ### Workspace tab activation ###
    def _handle_side_tab_changed(self, tab_index: int) -> None:
        """Refresh only file-backed content owned by the visible side tab."""

        self._sync_visible_side_tab(self.side_tabs.widget(tab_index))

    def _sync_visible_side_tab(self, selected_widget: QWidget | None) -> None:
        """Recheck cheap revisions without rebuilding unrelated side tabs."""

        if selected_widget is self.images_tab:
            self._refresh_stale_image_thumbnails()
            self._sync_images_tab()
        elif selected_widget is self.texture_creator_tab:
            self._sync_texture_creator_canvas()

    def _refresh_blueprint_file_dependencies(
        self,
        *,
        include_exported_levels: bool,
    ) -> None:
        """Refresh changed blueprint pixels and geometry dimensions by revision."""

        geometry_dimensions_changed = False
        current_level = (
            self.levels[self.current_level_index]
            if 0 <= self.current_level_index < len(self.levels)
            else None
        )
        if current_level is not None:
            current_image_changed = (
                self.canvas.refresh_blueprint_image_if_stale()
            )
            current_revision = self.canvas.get_blueprint_image_revision()
            if current_image_changed:
                refreshed_image_size = self.canvas.get_image_size_pixels()
                if (
                    refreshed_image_size is not None
                    and refreshed_image_size
                    != current_level.image_size_pixels
                ):
                    current_level.image_size_pixels = refreshed_image_size
                    geometry_dimensions_changed = True
                self._update_blueprint_name_label()
            if current_revision is not None:
                self._level_blueprint_image_revisions[
                    current_level.index
                ] = current_revision

        if include_exported_levels:
            exported_level_indices = {
                level.index for level in self.levels if level.include_in_export
            }
            self._level_blueprint_image_revisions = {
                level_index: revision
                for level_index, revision
                in self._level_blueprint_image_revisions.items()
                if level_index in exported_level_indices
            }
            for level in self.levels:
                if (
                    not level.include_in_export
                    or (
                        current_level is not None
                        and level.index == current_level.index
                    )
                    or level.image_path is None
                ):
                    continue
                revision_before = _build_local_file_revision(level.image_path)
                if (
                    self._level_blueprint_image_revisions.get(level.index)
                    == revision_before
                ):
                    continue
                if not _local_file_revision_has_file(revision_before):
                    self._level_blueprint_image_revisions[
                        level.index
                    ] = revision_before
                    continue
                try:
                    with Image.open(level.image_path) as blueprint_image:
                        image_size = (
                            float(blueprint_image.width),
                            float(blueprint_image.height),
                        )
                except (OSError, TypeError, ValueError):
                    continue
                revision_after = _build_local_file_revision(level.image_path)
                if revision_before != revision_after:
                    continue
                self._level_blueprint_image_revisions[
                    level.index
                ] = revision_after
                if image_size != level.image_size_pixels:
                    level.image_size_pixels = image_size
                    geometry_dimensions_changed = True

        if geometry_dimensions_changed:
            self._mark_viewer_preview_dirty(preserve_camera=False)

    def _handle_workspace_tab_changed(self, tab_index: int) -> None:
        selected_widget = self.workspace_tabs.widget(tab_index)
        is_full_width_workspace = selected_widget in (
            self.texture_atlas_workspace,
            self.surface_texture_generation,
            self.generation,
            self.settings_widget,
        )
        self.side_panel.setVisible(not is_full_width_workspace)
        if selected_widget is self.surface_texture_generation:
            self.surface_texture_generation.refresh_file_backed_previews()
            self._ensure_viewer_preview_current(preserve_camera=True)
        elif selected_widget is self.texture_atlas_workspace:
            self._sync_atlas_object_texture_sources()
        elif selected_widget is self.generation:
            self.generation.refresh_file_backed_previews()

        self._apply_fullscreen_3d_viewer_screen(
            self.settings_widget.get_fullscreen_3d_viewer_screen_id()
        )
        if is_full_width_workspace:
            return

        if selected_widget is not self.canvas_viewer_workspace:
            return

        if not self._viewer_preview_is_active():
            self._refresh_blueprint_file_dependencies(
                include_exported_levels=False
            )
        self._sync_visible_side_tab(self.side_tabs.currentWidget())
        self._ensure_viewer_preview_current(preserve_camera=False)

    def _handle_generation_data_changed_for_atlases(
        self,
        _generation_data: object,
    ) -> None:
        """Refresh Atlas object choices after generation, deletion, or selection."""

        self._sync_atlas_object_texture_sources()
        self._request_hosted_atlas_object_preview()

    # ### Generated-object placement ###
    def _handle_object_placement_requested(
        self,
        operation_id: str,
    ) -> None:
        """Open exactly one modeless Canvas picker for an operation token."""

        exact_operation_id = str(operation_id)
        if self._is_shutdown or not exact_operation_id.strip():
            return

        self._close_object_placement_dialog()
        dialog = ObjectPlacementDialog(self.levels, self)
        self._object_placement_dialog = dialog
        self._object_placement_operation_id = exact_operation_id
        dialog.placement_selected.connect(
            partial(
                self._handle_object_placement_selected,
                dialog,
                exact_operation_id,
            )
        )
        dialog.finished.connect(
            partial(
                self._handle_object_placement_dialog_finished,
                dialog,
            )
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _handle_object_placement_selected(
        self,
        dialog: ObjectPlacementDialog,
        operation_id: str,
        raw_placement: object,
    ) -> None:
        """Commit a click only when both its dialog and operation are current."""

        if (
            dialog is not self._object_placement_dialog
            or operation_id != self._object_placement_operation_id
            or not isinstance(raw_placement, GeneratedObjectPlacement)
        ):
            return
        if not self.generation.set_active_object_placement(
            operation_id,
            raw_placement,
        ):
            self._close_object_placement_dialog()

    def _handle_object_placement_dialog_finished(
        self,
        dialog: ObjectPlacementDialog,
        _result: int,
    ) -> None:
        """Release the current picker without touching a replacement dialog."""

        if dialog is not self._object_placement_dialog:
            return
        operation_id = self._object_placement_operation_id
        self._object_placement_dialog = None
        self._object_placement_operation_id = None
        if operation_id is not None:
            self.generation.cancel_object_placement_request(operation_id)
        dialog.deleteLater()

    def _handle_object_placement_operation_finished(
        self,
        operation_id: str,
    ) -> None:
        """Close only the picker owned by the completed operation token."""

        if str(operation_id) != self._object_placement_operation_id:
            return
        self._close_object_placement_dialog()

    def _close_object_placement_dialog(self) -> None:
        """Close and forget the one modeless object-placement picker."""

        dialog = self._object_placement_dialog
        operation_id = self._object_placement_operation_id
        self._object_placement_dialog = None
        self._object_placement_operation_id = None
        if operation_id is not None:
            self.generation.cancel_object_placement_request(operation_id)
        if dialog is None:
            return
        dialog.close()
        dialog.deleteLater()

    def _handle_generated_object_completed_for_canvas(
        self,
        raw_record: object,
        _generated_model: object,
    ) -> None:
        """Reveal a newly placed object and refit the Canvas 3D view."""

        if (
            isinstance(raw_record, GeneratedObjectRecord)
            and raw_record.placement is not None
        ):
            self._schedule_viewer_preview_refresh(preserve_camera=False)

    def _handle_generated_object_changed_for_canvas(
        self,
        raw_record: object,
        _generated_model: object,
    ) -> None:
        """Refresh the Canvas model after one placed object revision changes."""

        if (
            isinstance(raw_record, GeneratedObjectRecord)
            and raw_record.placement is not None
        ):
            self._schedule_viewer_preview_refresh(preserve_camera=True)

    def _handle_generated_object_placement_changed_for_canvas(
        self,
        raw_record: object,
    ) -> None:
        """Move a completed object in the Canvas preview immediately."""

        if (
            isinstance(raw_record, GeneratedObjectRecord)
            and raw_record.placement is not None
        ):
            self._schedule_viewer_preview_refresh(preserve_camera=True)

    def _handle_placed_object_transform_changed(
        self,
        object_id: str,
        world_position: object,
        rotation_degrees: object,
    ) -> None:
        """Convert one world-space gizmo commit back to level-relative state."""

        normalized_object_id = str(object_id).strip()
        existing_placement = self.generation.get_generated_object_placement(
            normalized_object_id
        )
        if existing_placement is None:
            self._schedule_viewer_preview_refresh(preserve_camera=True)
            return
        level = next(
            (
                candidate
                for candidate in self.levels
                if candidate.index == existing_placement.level_index
            ),
            None,
        )
        base_z = build_level_base_z_lookup(self.levels).get(
            existing_placement.level_index
        )
        if level is None or base_z is None:
            self._schedule_viewer_preview_refresh(preserve_camera=True)
            return
        try:
            world_x, world_y, world_z = tuple(world_position)
            image_x, image_y = level_world_to_image_xy(
                level,
                float(world_x),
                float(world_y),
            )
            placement = GeneratedObjectPlacement(
                level_index=existing_placement.level_index,
                image_x=image_x,
                image_y=image_y,
                height_offset_meters=float(world_z) - float(base_z),
                rotation_degrees=rotation_degrees,
            )
        except (TypeError, ValueError, OverflowError):
            self._schedule_viewer_preview_refresh(preserve_camera=True)
            return
        # The gizmo already updated its independent preview root. Emitting the
        # ordinary placement signals here would serialize and rebuild both 3D
        # previews, even though only this retained transform changed.
        canvas_was_current = (
            self._canvas_viewer_preview_revision
            == self._viewer_preview_revision
        )
        dependency_signature_before: tuple[object, ...] | None = None
        if canvas_was_current:
            current_dependency_signature = (
                self._build_viewer_preview_dependency_signature()
            )
            canvas_was_current = bool(
                self._viewer_preview_dependency_signature_revision
                == self._viewer_preview_revision
                and current_dependency_signature
                == self._viewer_preview_dependency_signature
            )
            if canvas_was_current:
                dependency_signature_before = current_dependency_signature
        was_updated = self.generation.update_generated_object_placement(
            normalized_object_id,
            placement,
            emit_change_signals=False,
        )
        if not was_updated:
            self._schedule_viewer_preview_refresh(preserve_camera=True)
            return

        revision = self._mark_viewer_preview_dirty(preserve_camera=True)
        dependency_signature_after = (
            self._build_viewer_preview_dependency_signature()
            if canvas_was_current
            else None
        )
        if (
            dependency_signature_before is not None
            and dependency_signature_after is not None
            and self._dependency_change_is_only_target_placement(
                dependency_signature_before,
                dependency_signature_after,
                normalized_object_id,
                placement,
            )
        ):
            self._canvas_viewer_preview_revision = revision
            self._viewer_preview_dependency_signature = (
                dependency_signature_after
            )
            self._viewer_preview_dependency_signature_revision = revision
            return
        self._queue_viewer_preview_refresh()

    def _handle_generated_object_deleted_for_canvas(
        self,
        _object_id: str,
    ) -> None:
        """Remove any deleted placed object from the Canvas preview."""

        self._schedule_viewer_preview_refresh(preserve_camera=True)

    def _refresh_placed_object_texture_if_needed(
        self,
        object_id: str,
    ) -> None:
        """Show a newly selected texture on a placed Canvas object."""

        normalized_object_id = str(object_id).strip()
        if not normalized_object_id:
            return
        if any(
            record.object_id == normalized_object_id
            and record.placement is not None
            for record in self.generation.get_data().generated_objects
        ):
            self._schedule_viewer_preview_refresh(preserve_camera=True)

    def _handle_surface_texture_data_changed_for_atlases(
        self,
        _surface_texture_data: object,
    ) -> None:
        """Refresh Atlas wall choices without reloading unchanged thumbnails."""

        self._sync_atlas_object_texture_sources()

    def _handle_surface_texture_assignments_removed_for_atlases(
        self,
        raw_assignment_ids: object,
    ) -> None:
        """Remove fully replaced wall textures from every packed atlas."""

        if not isinstance(raw_assignment_ids, tuple | list):
            return
        assignment_ids = tuple(
            assignment_id
            for assignment_id in (
                str(value).strip() for value in raw_assignment_ids
            )
            if assignment_id
        )
        if not assignment_ids:
            return
        removable_assignment_ids = tuple(
            assignment_id
            for assignment_id in assignment_ids
            if build_atlas_wall_texture_source_id(assignment_id)
            in self._atlas_wall_texture_source_ids
        )
        if not removable_assignment_ids:
            return
        self.texture_atlas_workspace.remove_deleted_wall_texture_assignments(
            removable_assignment_ids
        )
        for assignment_id in removable_assignment_ids:
            self._atlas_wall_texture_source_ids.discard(
                build_atlas_wall_texture_source_id(assignment_id)
            )
        self._atlas_generation_signature = None
        self._sync_atlas_object_texture_sources()

    def _handle_generated_object_changed_for_atlases(
        self,
        raw_record: object,
        _generated_model: object,
    ) -> None:
        """Move pinned Atlas placements to the object's latest exact PNGs."""

        object_id = getattr(raw_record, "object_id", None)
        if not isinstance(object_id, str) or not object_id:
            return
        self._sync_atlas_object_texture_sources()
        self.texture_atlas_workspace.refresh_regenerated_object_texture(
            object_id
        )

    def _handle_generated_object_deleted_for_atlases(
        self,
        object_id: str,
    ) -> None:
        """Remove an explicitly deleted object's pixels from every atlas."""

        if (
            self._atlas_preview_variant_key is not None
            and self._atlas_preview_variant_key[0] == object_id
        ):
            self._clear_atlas_object_preview()
        self.texture_atlas_workspace.remove_deleted_object(object_id)

    def _handle_atlas_object_preview_requested(
        self,
        object_id: str,
        texture_resolution: int,
    ) -> None:
        """Display the exact selected Atlas texture variant in 3D."""

        if is_atlas_wall_texture_source_id(object_id):
            self._show_atlas_surface_texture_preview(
                object_id,
                texture_resolution,
            )
            return
        variant = self.generation.get_texture_variant(
            object_id,
            texture_resolution,
        )
        if variant is None:
            self._clear_atlas_object_preview()
            self._append_atlas_preview_status(
                "The selected object's exact 3D texture variant is missing."
            )
            return

        try:
            asset_path = Path(variant.glb_asset_path)
            symmetry = self.generation.get_object_symmetric_division(object_id)
            variant_key = (
                str(variant.object_id),
                int(variant.resolution),
                _build_local_file_revision(asset_path),
                None if symmetry is None else symmetry.orientation,
                None if symmetry is None else symmetry.plane_coordinate,
                None if symmetry is None else getattr(symmetry, "version", None),
                (
                    None
                    if symmetry is None
                    else getattr(symmetry, "packing_mode", None)
                ),
                (
                    None
                    if symmetry is None
                    else getattr(symmetry, "texture_content_half", None)
                ),
                (
                    None
                    if symmetry is None
                    else getattr(symmetry, "texture_content_quadrant", None)
                ),
            )
            if (
                self._atlas_preview_variant_key == variant_key
                and self.atlas_object_preview_viewer.model is not None
            ):
                return
            generated_model = import_generated_glb(asset_path.read_bytes())
        except Exception as error:
            self._clear_atlas_object_preview()
            self._append_atlas_preview_status(
                "The selected object's 3D preview could not be loaded: "
                f"{error}"
            )
            return

        preserve_camera = (
            self._atlas_preview_variant_key is not None
            and self._atlas_preview_variant_key[0] == object_id
        )
        self.atlas_object_preview_viewer.set_model(
            generated_model,
            preserve_camera=preserve_camera,
        )
        self.atlas_object_preview_viewer.set_symmetric_division_preview(
            None if symmetry is None else symmetry.orientation,
            None if symmetry is None else symmetry.plane_coordinate,
        )
        self._atlas_preview_variant_key = variant_key

    def _show_atlas_surface_texture_preview(
        self,
        source_id: str,
        texture_resolution: int,
    ) -> None:
        """Display one exact Atlas wall texture on an upright square plane."""

        assignment_id = get_atlas_wall_texture_assignment_id(source_id)
        try:
            normalized_resolution = int(texture_resolution)
        except (TypeError, ValueError, OverflowError):
            normalized_resolution = -1
        assignment = (
            None
            if assignment_id is None
            else self.surface_texture_generation.get_assignment(
                assignment_id
            )
        )
        requested_resolution = (
            normalized_resolution
            if assignment is not None and assignment.texture_variants
            else None
        )
        asset_path = (
            None
            if assignment is None
            else self.surface_texture_generation.get_assignment_asset_path(
                assignment.assignment_id,
                requested_resolution,
            )
        )
        if (
            assignment is None
            or asset_path is None
            or normalized_resolution <= 0
        ):
            self._clear_atlas_object_preview()
            self._append_atlas_preview_status(
                "The selected surface texture preview is unavailable."
            )
            return

        try:
            preview_key = (
                source_id,
                normalized_resolution,
                _build_local_file_revision(asset_path),
                "surface_texture_plane",
            )
            if (
                self._atlas_preview_variant_key == preview_key
                and self.atlas_object_preview_viewer.model is not None
            ):
                return
            source = self._build_atlas_wall_texture_source(
                assignment,
                requested_resolution,
            )
            if source is None:
                raise ValueError("The surface texture source is unavailable.")
            model = build_texture_preview_plane_model(
                source.load_texture_rgba()
            )
        except (OSError, TypeError, ValueError) as error:
            self._clear_atlas_object_preview()
            self._append_atlas_preview_status(
                "The selected surface texture preview could not be loaded: "
                f"{error}"
            )
            return

        preserve_camera = (
            self._atlas_preview_variant_key is not None
            and self._atlas_preview_variant_key[0] == source_id
        )
        self.atlas_object_preview_viewer.set_model(
            model,
            preserve_camera=preserve_camera,
        )
        self._atlas_preview_variant_key = preview_key

    def _handle_atlas_object_texture_resolution_changed(
        self,
        object_id: str,
        texture_resolution: int,
    ) -> None:
        """Make an accepted Atlas size the source's globally active variant."""

        assignment_id = get_atlas_wall_texture_assignment_id(object_id)
        if assignment_id is not None:
            if self.surface_texture_generation.select_assignment_texture_resolution(
                assignment_id,
                texture_resolution,
            ):
                return
            self._append_atlas_preview_status(
                "The atlas was resized, but its exact surface texture variant "
                "could not be assigned globally."
            )
            return

        if self.generation.select_object_texture_resolution(
            object_id,
            texture_resolution,
        ):
            self._refresh_placed_object_texture_if_needed(object_id)
            return
        self._append_atlas_preview_status(
            "The atlas was resized, but its exact 3D texture variant could "
            "not be assigned to the generated object."
        )

    def _handle_surface_texture_resolution_change_requested(
        self,
        assignment_id: str,
        texture_resolution: int,
    ) -> bool:
        """Commit a Surface-tab choice together with every Atlas placement."""

        assignment = self.surface_texture_generation.get_assignment(
            assignment_id
        )
        if assignment is None:
            return False
        try:
            target_resolution = int(texture_resolution)
        except (TypeError, ValueError, OverflowError):
            return False
        if assignment.selected_texture_resolution == target_resolution:
            return True
        if not (
            self.surface_texture_generation
            .can_select_assignment_texture_resolution(
                assignment.assignment_id,
                target_resolution,
            )
        ):
            return False

        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        if assignment.surface_type != SURFACE_TYPE_WALL:
            return (
                self.surface_texture_generation
                .select_assignment_texture_resolution(
                    assignment.assignment_id,
                    target_resolution,
                )
            )

        self._sync_atlas_object_texture_sources()
        if source_id not in self._atlas_wall_texture_source_ids:
            return (
                self.surface_texture_generation
                .select_assignment_texture_resolution(
                    assignment.assignment_id,
                    target_resolution,
                )
            )

        return self.texture_atlas_workspace.set_object_texture_resolution(
            source_id,
            target_resolution,
            commit_callback=lambda: (
                self.surface_texture_generation
                .select_assignment_texture_resolution(
                    assignment.assignment_id,
                    target_resolution,
                )
            ),
        )

    def _handle_generation_texture_resolution_change_requested(
        self,
        object_id: str,
        texture_resolution: int,
    ) -> bool:
        """Commit an Object-tab resolution together with every Atlas slot."""

        normalized_id = str(object_id).strip()
        try:
            target_resolution = int(texture_resolution)
        except (TypeError, ValueError, OverflowError):
            return False
        if not normalized_id or target_resolution not in OBJECT_TEXTURE_RESOLUTIONS:
            return False
        current_variant = self.generation.get_active_texture_variant(normalized_id)
        if current_variant is None:
            return False
        if current_variant.resolution == target_resolution:
            return True
        if self.generation.get_texture_variant(
            normalized_id,
            target_resolution,
        ) is None:
            return False

        self._sync_atlas_object_texture_sources()
        resolution_changed = (
            self.texture_atlas_workspace.set_object_texture_resolution(
                normalized_id,
                target_resolution,
                commit_callback=lambda: (
                    self.generation.select_object_texture_resolution(
                        normalized_id,
                        target_resolution,
                    )
                ),
            )
        )
        if resolution_changed:
            self._refresh_placed_object_texture_if_needed(normalized_id)
        return resolution_changed

    def _handle_generation_object_packing_change_requested(
        self,
        old_record: GeneratedObjectRecord,
        replacement_record: GeneratedObjectRecord,
        _preview_model: GeneratedModel,
        commit_callback: Callable[[], bool],
    ) -> bool:
        """Commit one prepared full or symmetric object and all Atlas PNGs."""

        if (
            not isinstance(old_record, GeneratedObjectRecord)
            or not isinstance(replacement_record, GeneratedObjectRecord)
            or old_record.object_id != replacement_record.object_id
            or not callable(commit_callback)
        ):
            return False
        symmetry = self.generation.resolve_symmetric_division_for_record(
            replacement_record
        )
        resolve_variant = (
            self.generation.resolve_atlas_texture_image_variant_for_record
        )
        candidate_sources: list[AtlasObjectTextureSource] = []
        for resolution in sorted(OBJECT_TEXTURE_RESOLUTIONS):
            variant = resolve_variant(
                replacement_record,
                resolution,
            )
            if variant is None:
                continue
            source = self._build_atlas_object_texture_source(
                variant,
                symmetry,
            )
            if source is None:
                return False
            candidate_sources.append(source)
        if not candidate_sources:
            if (
                replacement_record.pipeline.get(
                    FACE_EDIT_TEXTURE_STALE_PIPELINE_KEY
                )
                is True
            ):
                return False
            return bool(commit_callback())
        return self.texture_atlas_workspace.transition_object_packing(
            replacement_record.object_id,
            candidate_sources,
            commit_callback=commit_callback,
        )

    def _append_atlas_preview_status(self, message: str) -> None:
        """Report preview errors without hiding an Atlas resize result."""

        normalized_message = str(message).strip()
        existing_message = self.texture_atlas_workspace.status_label.text().strip()
        if normalized_message in existing_message:
            return
        self.texture_atlas_workspace.status_label.setText(
            (
                f"{existing_message} {normalized_message}"
                if existing_message
                else normalized_message
            )
        )

    def _clear_atlas_object_preview(self) -> None:
        """Drop stale Atlas preview content and its cached selection key."""

        if (
            self._atlas_preview_variant_key is None
            and self.atlas_object_preview_viewer.model is None
        ):
            return
        self._atlas_preview_variant_key = None
        self.atlas_object_preview_viewer.clear_model()

    def _sync_atlas_object_texture_sources(self) -> None:
        """Expose generated object and wall texture sources to Atlas."""

        active_variants: list[tuple[object, object | None]] = []
        signature_items: list[tuple[object, ...]] = []
        source_content_paths: dict[
            str,
            tuple[tuple[object, ...], ...],
        ] = {}
        source_content_revisions: dict[
            str,
            tuple[tuple[object, ...], ...],
        ] = {}
        generated_object_ids = self.generation.get_generated_object_ids()
        generated_object_id_lookup = set(generated_object_ids)
        for object_id in generated_object_ids:
            variant = self.generation.get_active_texture_variant(
                object_id
            )
            symmetry = self.generation.get_object_symmetric_division(
                object_id
            )
            texture_variant_signature = (
                self.generation.get_texture_variant_dependency_signature(
                    object_id
                )
            )
            texture_source_signature = tuple(
                (
                    item[0],
                    item[3],
                    item[4],
                    item[5] if len(item) > 5 else (),
                )
                for item in texture_variant_signature
            )
            source_content_paths[object_id] = tuple(
                (
                    item[0],
                    item[3],
                    tuple(
                        (map_item[0], map_item[1])
                        for map_item in (item[5] if len(item) > 5 else ())
                    ),
                )
                for item in texture_variant_signature
            )
            source_content_revisions[object_id] = tuple(
                (
                    item[0],
                    item[4],
                    tuple(
                        (map_item[0], map_item[2])
                        for map_item in (item[5] if len(item) > 5 else ())
                    ),
                )
                for item in texture_variant_signature
            )
            if variant is None:
                signature_items.append(
                    (
                        "object",
                        object_id,
                        0,
                        "",
                        texture_source_signature,
                        None if symmetry is None else symmetry.orientation,
                        None if symmetry is None else symmetry.plane_coordinate,
                        (
                            None
                            if symmetry is None
                            else getattr(symmetry, "version", None)
                        ),
                        (
                            None
                            if symmetry is None
                            else getattr(symmetry, "packing_mode", None)
                        ),
                        (
                            None
                            if symmetry is None
                            else getattr(
                                symmetry,
                                "texture_content_half",
                                None,
                            )
                        ),
                        (
                            None
                            if symmetry is None
                            else getattr(
                                symmetry,
                                "texture_content_quadrant",
                                None,
                            )
                        ),
                    )
                )
                continue
            active_variants.append((variant, symmetry))
            signature_items.append(
                (
                    "object",
                    object_id,
                    int(getattr(variant, "resolution")),
                    str(getattr(variant, "texture_asset_relative_path")),
                    texture_source_signature,
                    None if symmetry is None else symmetry.orientation,
                    None if symmetry is None else symmetry.kept_side,
                    None if symmetry is None else symmetry.plane_coordinate,
                    (
                        None
                        if symmetry is None
                        else getattr(symmetry, "texture_content_half", None)
                    ),
                    None if symmetry is None else getattr(symmetry, "version", None),
                    (
                        None
                        if symmetry is None
                        else getattr(symmetry, "packing_mode", None)
                    ),
                    (
                        None
                        if symmetry is None
                        else getattr(
                            symmetry,
                            "texture_content_quadrant",
                            None,
                        )
                    ),
                )
            )

        wall_assignments = list(
            self.surface_texture_generation.get_wall_assignments()
        )
        for assignment in wall_assignments:
            variant_signature: list[tuple[object, ...]] = []
            candidate_variants = (
                tuple(assignment.texture_variants)
                if assignment.texture_variants
                else (None,)
            )
            for texture_variant in candidate_variants:
                resolution = (
                    None
                    if texture_variant is None
                    else texture_variant.resolution
                )
                logical_path = (
                    assignment.asset_path
                    if texture_variant is None
                    else texture_variant.asset_path
                )
                physical_path = (
                    self.surface_texture_generation.get_assignment_asset_path(
                        assignment.assignment_id,
                        resolution,
                    )
                )
                variant_signature.append(
                    (
                        resolution,
                        str(logical_path),
                        _build_local_file_revision(physical_path),
                    )
                )
            signature_items.append(
                (
                    "wall",
                    assignment.assignment_id,
                    assignment.asset_path,
                    assignment.selected_texture_resolution,
                    assignment.texture_width,
                    assignment.texture_height,
                    assignment.surface_ids,
                    tuple(variant_signature),
                )
            )
            wall_source_id = build_atlas_wall_texture_source_id(
                assignment.assignment_id
            )
            if wall_source_id not in generated_object_id_lookup:
                source_content_paths[wall_source_id] = tuple(
                    (resolution, logical_path)
                    for resolution, logical_path, _revision in variant_signature
                )
                source_content_revisions[wall_source_id] = tuple(
                    (resolution, revision)
                    for resolution, _logical_path, revision in variant_signature
                )
        signature = tuple(signature_items)
        if signature == self._atlas_generation_signature:
            self._request_hosted_atlas_object_preview()
            return
        changed_source_ids: list[str] = []
        if (
            self._atlas_source_content_paths is not None
            and self._atlas_source_content_revisions is not None
        ):
            for source_id, content_revision in source_content_revisions.items():
                if source_id not in self._atlas_source_content_revisions:
                    continue
                previous_paths = self._atlas_source_content_paths.get(source_id)
                current_paths = source_content_paths.get(source_id)
                if (
                    _build_atlas_source_base_path_signature(previous_paths)
                    != _build_atlas_source_base_path_signature(current_paths)
                ):
                    continue
                if (
                    previous_paths != current_paths
                    or self._atlas_source_content_revisions.get(source_id)
                    != content_revision
                ):
                    changed_source_ids.append(source_id)

        active_sources: list[AtlasObjectTextureSource] = []
        available_source_ids: set[str] = set()
        failed_source_ids: set[str] = set()
        source_build_failed = False
        for variant, symmetry in active_variants:
            source = self._build_atlas_object_texture_source(
                variant,
                symmetry,
            )
            if source is None:
                source_build_failed = True
                failed_source_ids.add(str(getattr(variant, "object_id")))
                continue
            active_sources.append(source)
            available_source_ids.add(str(getattr(variant, "object_id")))
        wall_sources: dict[str, AtlasObjectTextureSource] = {}
        wall_assignments_by_source_id: dict[str, SurfaceTextureAssignment] = {}
        colliding_wall_texture_count = 0
        for assignment in wall_assignments:
            wall_source_id = build_atlas_wall_texture_source_id(
                assignment.assignment_id
            )
            if wall_source_id in generated_object_id_lookup:
                colliding_wall_texture_count += 1
                continue
            active_resolution = (
                assignment.selected_texture_resolution
                if assignment.texture_variants
                else None
            )
            if (
                self.surface_texture_generation.get_assignment_asset_path(
                    assignment.assignment_id,
                    active_resolution,
                )
                is None
            ):
                continue
            source = self._build_atlas_wall_texture_source(assignment)
            if source is None:
                source_build_failed = True
                failed_source_ids.add(wall_source_id)
                continue
            active_sources.append(source)
            available_source_ids.add(wall_source_id)
            wall_sources[source.object_id] = source
            wall_assignments_by_source_id[source.object_id] = assignment

        self._atlas_pending_source_content_refresh_ids.update(
            failed_source_ids
        )
        refreshable_source_ids = tuple(
            source_id
            for source_id in dict.fromkeys(
                (
                    *changed_source_ids,
                    *self._atlas_pending_source_content_refresh_ids,
                )
            )
            if source_id in available_source_ids
        )

        def resolve_variant(
            object_id: str,
            resolution: int,
        ) -> AtlasObjectTextureSource | None:
            wall_assignment = wall_assignments_by_source_id.get(object_id)
            if wall_assignment is not None:
                return self._build_atlas_wall_texture_source(
                    wall_assignment,
                    resolution,
                )
            return self._build_atlas_object_texture_source(
                self.generation.get_atlas_texture_image_variant(
                    object_id,
                    resolution,
                ),
                self.generation.get_object_symmetric_division(object_id),
            )

        def is_variant_selectable(
            object_id: str,
            resolution: int,
        ) -> bool:
            wall_assignment = wall_assignments_by_source_id.get(object_id)
            if wall_assignment is not None:
                return (
                    self.surface_texture_generation
                    .can_select_assignment_texture_resolution(
                        wall_assignment.assignment_id,
                        resolution,
                    )
                )
            if self.generation.has_active_object_job(object_id):
                return False
            variant = self.generation.get_texture_variant(
                object_id,
                resolution,
            )
            if variant is None:
                return False
            try:
                import_generated_glb(variant.glb_asset_path.read_bytes())
            except Exception:
                return False
            return True

        self.texture_atlas_workspace.set_object_texture_sources(
            active_sources,
            variant_resolver=resolve_variant,
            selectability_resolver=is_variant_selectable,
        )
        if not self.texture_atlas_workspace.refresh_texture_source_content(
            refreshable_source_ids
        ):
            self._atlas_pending_source_content_refresh_ids.update(
                refreshable_source_ids
            )
            self._atlas_generation_signature = None
            return
        self._atlas_pending_source_content_refresh_ids.difference_update(
            refreshable_source_ids
        )
        self._atlas_wall_texture_source_ids.update(wall_sources)
        self._atlas_generation_signature = (
            None if source_build_failed else signature
        )
        self._atlas_source_content_paths = source_content_paths
        self._atlas_source_content_revisions = source_content_revisions
        self._request_hosted_atlas_object_preview()
        if colliding_wall_texture_count:
            self._append_atlas_preview_status(
                f"Skipped {colliding_wall_texture_count} wall texture source"
                f"{'s' if colliding_wall_texture_count != 1 else ''} because "
                "a generated object uses the same reserved Atlas ID."
            )

    @staticmethod
    def _build_atlas_object_texture_source(
        variant: object,
        symmetry: object | None = None,
    ) -> AtlasObjectTextureSource | None:
        """Adapt one public Generation variant while tolerating missing assets."""

        if variant is None:
            return None
        try:
            packing_mode = ATLAS_PACKING_MODE_FULL
            if symmetry is not None:
                symmetry_version = getattr(symmetry, "version", None)
                is_legacy_pair = (
                    isinstance(symmetry_version, int)
                    and not isinstance(symmetry_version, bool)
                    and symmetry_version == 3
                    and getattr(symmetry, "packing_mode", None)
                    == ATLAS_PACKING_MODE_SYMMETRIC_PAIR
                    and getattr(symmetry, "texture_content_half", None)
                    == ATLAS_SLOT_HALF_LEFT
                )
                is_square_pair = (
                    isinstance(symmetry_version, int)
                    and not isinstance(symmetry_version, bool)
                    and symmetry_version == 4
                    and getattr(symmetry, "packing_mode", None)
                    == ATLAS_PACKING_MODE_SYMMETRIC_PAIR
                    and getattr(symmetry, "texture_content_half", None)
                    == ATLAS_SLOT_HALF_LEFT
                )
                is_quarter = (
                    symmetry_version == 2
                    and getattr(symmetry, "packing_mode", None)
                    == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER
                    and getattr(
                        symmetry,
                        "texture_content_quadrant",
                        None,
                    )
                    == "top_left"
                )
                if is_square_pair:
                    packing_mode = ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR
                elif is_legacy_pair:
                    packing_mode = ATLAS_PACKING_MODE_SYMMETRIC_PAIR
                elif is_quarter:
                    packing_mode = ATLAS_PACKING_MODE_SYMMETRIC_QUARTER
                elif symmetry_version == 1:
                    packing_mode = ATLAS_PACKING_MODE_SYMMETRIC_HALF
                else:
                    raise ValueError("Unknown symmetric texture packing metadata.")
            return load_atlas_object_texture_source(
                object_id=str(getattr(variant, "object_id")),
                object_name=str(getattr(variant, "object_name")),
                texture_path=str(
                    getattr(variant, "texture_asset_relative_path")
                ),
                texture_resolution=int(getattr(variant, "resolution")),
                physical_texture_path=getattr(
                    variant,
                    "texture_asset_path",
                ),
                map_texture_paths=getattr(
                    variant,
                    "map_texture_asset_relative_paths",
                    {},
                ),
                physical_map_texture_paths=getattr(
                    variant,
                    "map_texture_asset_paths",
                    {},
                ),
                packing_mode=packing_mode,
                symmetric_preview_orientation=(
                    None
                    if symmetry is None
                    else str(getattr(symmetry, "orientation"))
                ),
                symmetric_preview_plane_coordinate=(
                    None
                    if symmetry is None
                    else float(getattr(symmetry, "plane_coordinate"))
                ),
            )
        except (AttributeError, OSError, TypeError, ValueError):
            return None

    def _build_atlas_wall_texture_source(
        self,
        assignment: SurfaceTextureAssignment,
        resolution: int | None = None,
    ) -> AtlasObjectTextureSource | None:
        """Adapt one generated wall texture and its exact active variant."""

        if assignment.surface_type != SURFACE_TYPE_WALL:
            return None
        supports_resolution_changes = bool(assignment.texture_variants)
        requested_resolution = resolution
        if supports_resolution_changes:
            requested_resolution = (
                assignment.selected_texture_resolution
                if requested_resolution is None
                else int(requested_resolution)
            )
            if requested_resolution is None:
                return None
        physical_path = self.surface_texture_generation.get_assignment_asset_path(
            assignment.assignment_id,
            requested_resolution if supports_resolution_changes else None,
        )
        if physical_path is None:
            return None
        try:
            if supports_resolution_changes:
                texture_resolution = int(requested_resolution)
                fit_to_square = False
                variant = assignment.texture_variant_for_resolution(
                    texture_resolution
                )
                if variant is None:
                    return None
                asset_path = variant.asset_path
            else:
                with Image.open(physical_path) as image:
                    natural_resolution = choose_atlas_texture_resolution(
                        image.width, image.height
                    )
                texture_resolution = (
                    natural_resolution
                    if resolution is None
                    else int(resolution)
                )
                fit_to_square = True
                asset_path = assignment.asset_path
            surface_count = len(assignment.surface_ids)
            return load_atlas_object_texture_source(
                object_id=build_atlas_wall_texture_source_id(
                    assignment.assignment_id
                ),
                object_name=(
                    assignment.display_name
                    or (
                        f"Wall texture Â· {surface_count} surface"
                        f"{'s' if surface_count != 1 else ''}"
                    )
                ),
                texture_path=f"surface_textures/{asset_path}",
                texture_resolution=texture_resolution,
                physical_texture_path=physical_path,
                fit_to_square=fit_to_square,
                supports_resolution_changes=supports_resolution_changes,
            )
        except (AttributeError, OSError, TypeError, ValueError):
            return None

    def _handle_external_viewer_window_closed(self) -> None:
        """Keep Settings in sync when the detached viewer window is closed."""

        combo = self.settings_widget.fullscreen_3d_viewer_screen_combo
        if combo.currentIndex() != 0:
            combo.setCurrentIndex(0)
            return
        self._handle_generation_settings_changed()

    def _apply_fullscreen_3d_viewer_screen(
        self,
        screen_id: str | None,
    ) -> None:
        """Show the active workspace's 3D view on its chosen display."""

        if screen_id is None and not self._external_viewer_host.is_active:
            return
        screen = resolve_fullscreen_3d_viewer_screen(screen_id)
        viewer = self._active_workspace_3d_viewer()
        if screen is None or viewer is None:
            if not self._external_viewer_host.is_active:
                return
            self._external_viewer_host.restore()
            self._sync_external_3d_workspace_presentations()
            return
        self._external_viewer_host.show_on_screen(viewer, screen)
        self._sync_external_3d_workspace_presentations()
        self._request_hosted_atlas_object_preview()
        if (
            viewer is self.viewer
            or viewer is self.surface_texture_generation.surface_view
        ):
            self._ensure_viewer_preview_current(preserve_camera=True)

    def _apply_jobs_window_screen(self, screen_id: str | None) -> None:
        """Move the persistent Jobs window to its selected display."""

        screen = resolve_fullscreen_3d_viewer_screen(screen_id)
        self.jobs_window.set_target_screen(screen)

    def _sync_external_3d_workspace_presentations(self) -> None:
        """Show each workspace's local replacement for its detached 3D view."""

        hosted_viewer = self._external_viewer_host.viewer
        self.set_canvas_3d_viewer_external_display_active(
            hosted_viewer is self.viewer
        )
        self.surface_texture_generation.set_external_3d_viewer_active(
            hosted_viewer is self.surface_texture_generation.surface_view
        )
        self.generation.set_external_3d_viewer_active(
            hosted_viewer is self.generation.object_3d_panel
        )

    def _active_workspace_3d_viewer(self) -> QWidget | None:
        """Return the 3D widget belonging to the selected top-level tab."""

        selected_widget = self.workspace_tabs.currentWidget()
        if selected_widget is self.canvas_viewer_workspace:
            return self.viewer
        if selected_widget is self.surface_texture_generation:
            return self.surface_texture_generation.surface_view
        if selected_widget is self.generation:
            return self.generation.object_3d_panel
        if selected_widget is self.texture_atlas_workspace:
            return self.atlas_object_preview_viewer
        return None

    def _request_hosted_atlas_object_preview(self) -> None:
        """Refresh the selected Atlas object after a detached-view handoff."""

        if (
            self.workspace_tabs.currentWidget()
            is self.texture_atlas_workspace
            and self._external_viewer_host.is_active
            and self._external_viewer_host.viewer
            is self.atlas_object_preview_viewer
        ):
            self.texture_atlas_workspace.request_selected_object_preview()

    # ### Shared 3D preview cache ###
    def _canvas_viewer_preview_is_active(self) -> bool:
        return bool(
            (
                self.workspace_tabs.currentWidget()
                is self.canvas_viewer_workspace
                and self.canvas_viewer_tabs.currentIndex()
                == self.canvas_3d_view_tab_index
            )
            or (
                self._external_viewer_host.is_active
                and self._external_viewer_host.viewer is self.viewer
            )
        )

    def _surface_viewer_preview_is_active(self) -> bool:
        return bool(
            self.workspace_tabs.currentWidget()
            is self.surface_texture_generation
            or (
                self._external_viewer_host.is_active
                and self._external_viewer_host.viewer
                is self.surface_texture_generation.surface_view
            )
        )

    def _viewer_preview_is_active(self) -> bool:
        return bool(
            self._canvas_viewer_preview_is_active()
            or self._surface_viewer_preview_is_active()
        )

    def _active_viewer_preview_needs_refresh(self) -> bool:
        revision = self._viewer_preview_revision
        return bool(
            (
                self._canvas_viewer_preview_is_active()
                and self._canvas_viewer_preview_revision != revision
            )
            or (
                self._surface_viewer_preview_is_active()
                and self._surface_viewer_preview_revision != revision
            )
        )

    def _remember_current_canvas_preview_model(
        self,
        generated_model: GeneratedModel,
        *,
        validated_dependency_signature: tuple[object, ...],
    ) -> bool:
        """Cache an installed Canvas model only for validated dependencies."""

        if not isinstance(generated_model, GeneratedModel):
            raise TypeError("Canvas previews require a GeneratedModel.")
        current_dependency_signature = (
            self._build_viewer_preview_dependency_signature()
        )
        if current_dependency_signature != validated_dependency_signature:
            return False
        revision = self._viewer_preview_revision
        self._viewer_preview_model = generated_model
        self._viewer_preview_model_revision = revision
        self._viewer_preview_dependency_signature = current_dependency_signature
        self._viewer_preview_dependency_signature_revision = revision
        self._canvas_viewer_preview_revision = revision
        self._scheduled_viewer_refresh_preserve_camera = True
        self._clear_committed_doorway_outline_if_displayed()
        return True

    def _build_viewer_preview_dependency_signature(
        self,
    ) -> tuple[object, ...]:
        """Snapshot file-backed inputs that can change without a Qt signal."""

        room_texture_signature = tuple(
            (
                level.index,
                room_index,
                wall_key,
                texture_data.image_path,
                float(texture_data.source_x).hex(),
                float(texture_data.source_y).hex(),
                float(texture_data.source_width).hex(),
                float(texture_data.source_height).hex(),
                _build_local_file_revision(texture_data.image_path),
            )
            for level in self.levels
            for room_index, room in enumerate(level.rooms)
            for wall_key, texture_data in sorted(room.wall_textures.items())
        )
        return (
            room_texture_signature,
            self.generation.get_placed_preview_dependency_signature(),
            self.surface_texture_generation.get_preview_dependency_signature(),
        )

    @staticmethod
    def _dependency_change_is_only_target_placement(
        signature_before: tuple[object, ...],
        signature_after: tuple[object, ...],
        object_id: str,
        placement: GeneratedObjectPlacement,
    ) -> bool:
        """Accept a gizmo fast path only when no unrelated input changed."""

        if signature_before == signature_after:
            return True
        if len(signature_before) != 3 or len(signature_after) != 3:
            return False
        if (
            signature_before[0] != signature_after[0]
            or signature_before[2] != signature_after[2]
        ):
            return False
        placed_before = signature_before[1]
        placed_after = signature_after[1]
        if not isinstance(placed_before, tuple) or not isinstance(
            placed_after,
            tuple,
        ):
            return False
        if len(placed_before) != len(placed_after):
            return False

        target_count = 0
        for item_before, item_after in zip(
            placed_before,
            placed_after,
            strict=True,
        ):
            if (
                not isinstance(item_before, tuple)
                or not isinstance(item_after, tuple)
                or len(item_before) < 2
                or len(item_before) != len(item_after)
                or item_before[0] != item_after[0]
            ):
                return False
            if item_before[0] != object_id:
                if item_before != item_after:
                    return False
                continue
            target_count += 1
            if item_after[1] != placement:
                return False
            if (
                item_before[:1] + item_before[2:]
                != item_after[:1] + item_after[2:]
            ):
                return False
        return target_count == 1

    def _build_model_with_stable_dependencies(
        self,
        builder: Callable[[], GeneratedModel | None],
    ) -> tuple[GeneratedModel, tuple[object, ...]] | None:
        """Build once and reject a model assembled across file revisions."""

        dependency_signature_before = (
            self._build_viewer_preview_dependency_signature()
        )
        generated_model = builder()
        if generated_model is None:
            return None
        dependency_signature_after = (
            self._build_viewer_preview_dependency_signature()
        )
        if dependency_signature_before != dependency_signature_after:
            raise RuntimeError(
                "Preview inputs changed while the model was being built. "
                "Try the operation again."
            )
        return generated_model, dependency_signature_after

    def _invalidate_viewer_preview_for_dependency_changes(
        self,
        preserve_camera: bool,
    ) -> None:
        """Advance the preview revision after an out-of-band asset change."""

        if (
            self._viewer_preview_dependency_signature_revision
            != self._viewer_preview_revision
        ):
            return
        dependency_signature = (
            self._build_viewer_preview_dependency_signature()
        )
        if dependency_signature == self._viewer_preview_dependency_signature:
            return
        self._mark_viewer_preview_dirty(preserve_camera=preserve_camera)

    # ### Debounced doorway mesh previews ###
    @staticmethod
    def _copy_doorways(
        doorways: Sequence[DoorwayData],
    ) -> tuple[DoorwayData, ...]:
        """Own a doorway snapshot that cannot follow live Canvas mutations."""

        return tuple(copy.deepcopy(tuple(doorways)))

    def _reset_viewer_doorway_snapshots(self) -> None:
        """Make every rendered doorway snapshot match the loaded project."""

        self._viewer_doorways_by_level_index = {
            level.index: self._copy_doorways(level.doorways)
            for level in self.levels
        }

    def _build_viewer_preview_levels(self) -> list[LevelData]:
        """Copy levels while substituting only committed doorway dimensions."""

        preview_levels: list[LevelData] = []
        for level in self.levels:
            doorway_snapshot = self._viewer_doorways_by_level_index.get(
                level.index
            )
            if doorway_snapshot is None:
                doorway_snapshot = self._copy_doorways(level.doorways)
                self._viewer_doorways_by_level_index[level.index] = (
                    doorway_snapshot
                )
            preview_level = copy.copy(level)
            preview_level.doorways = list(copy.deepcopy(doorway_snapshot))
            preview_levels.append(preview_level)
        return preview_levels

    def _set_doorway_mesh_update_delay_seconds(
        self,
        delay_seconds: float,
    ) -> None:
        """Apply the setting and restart a live debounce only when it changed."""

        normalized_delay = float(delay_seconds)
        if normalized_delay <= 0.0:
            raise ValueError("Doorway mesh update delay must be positive.")
        if normalized_delay == self._doorway_mesh_update_delay_seconds:
            return

        timer_was_active = self._doorway_mesh_update_timer.isActive()
        self._doorway_mesh_update_delay_seconds = normalized_delay
        self._doorway_mesh_update_timer.setInterval(
            max(1, round(normalized_delay * 1000.0))
        )
        if timer_was_active:
            self._doorway_mesh_update_timer.start()

    def _cancel_pending_doorway_mesh_update(
        self,
        clear_outline: bool = True,
    ) -> None:
        """Cancel transient doorway work and optionally remove its outline."""

        self._doorway_mesh_update_timer.stop()
        self._pending_doorway_mesh_level_index = None
        if not clear_outline:
            return
        self._doorway_outline_commit_revision = None
        self.viewer.set_doorway_preview_outline(None)

    def _clear_committed_doorway_outline_if_displayed(self) -> None:
        """Clear an outline only after Canvas displays its committed revision."""

        target_revision = self._doorway_outline_commit_revision
        if (
            target_revision is None
            or self._pending_doorway_mesh_level_index is not None
            or self._canvas_viewer_preview_revision < target_revision
        ):
            return
        self._doorway_outline_commit_revision = None
        self.viewer.set_doorway_preview_outline(None)

    def _commit_pending_doorway_mesh_update(self) -> None:
        """Commit the latest stable doorway dimensions to the 3D mesh cache."""

        self._doorway_mesh_update_timer.stop()
        level_index = self._pending_doorway_mesh_level_index
        self._pending_doorway_mesh_level_index = None
        if level_index is None:
            return

        level = next(
            (
                candidate
                for candidate in self.levels
                if candidate.index == level_index
            ),
            None,
        )
        if level is None:
            self._doorway_outline_commit_revision = None
            self.viewer.set_doorway_preview_outline(None)
            return

        next_snapshot = self._copy_doorways(level.doorways)
        if (
            self._viewer_doorways_by_level_index.get(level.index)
            == next_snapshot
        ):
            self._clear_committed_doorway_outline_if_displayed()
            if self._doorway_outline_commit_revision is None:
                self.viewer.set_doorway_preview_outline(None)
            return

        self._viewer_doorways_by_level_index[level.index] = next_snapshot
        self._schedule_viewer_preview_refresh(preserve_camera=True)
        self._doorway_outline_commit_revision = self._viewer_preview_revision

    def _build_generated_model(
        self,
        failure_title: str | None,
    ) -> GeneratedModel | None:
        try:
            base_model = convert_to_glb(
                self.levels,
                stairs=self.stairs,
                surface_materials=(
                    self.surface_texture_generation.get_surface_material_sources()
                ),
            )
            placed_models = self._build_placed_generated_models()
            if not placed_models:
                return base_model
            return compose_placed_generated_models(
                base_model,
                placed_models,
            )
        except (TypeError, ValueError) as error:
            if failure_title is not None:
                QMessageBox.warning(self, failure_title, str(error))
            return None

    def _build_viewer_preview_model(
        self,
        failure_title: str | None,
    ) -> GeneratedModel | None:
        """Build render data without paying the GLB serialization cost."""

        try:
            base_model = convert_to_preview_model(
                self._build_viewer_preview_levels(),
                stairs=self.stairs,
                surface_materials=(
                    self.surface_texture_generation.get_surface_material_sources()
                ),
            )
            placed_models = self._build_placed_generated_models()
            if not placed_models:
                return base_model
            return compose_placed_generated_models_preview(
                base_model,
                placed_models,
            )
        except (TypeError, ValueError) as error:
            if failure_title is not None:
                QMessageBox.warning(self, failure_title, str(error))
            return None

    def _build_placed_generated_models(
        self,
    ) -> tuple[PlacedGeneratedModel, ...]:
        """Resolve persisted Canvas clicks into current world positions."""

        visible_level_by_index = {
            level.index: level
            for level in self.levels
            if level.include_in_export
        }
        if not visible_level_by_index:
            return ()
        base_z_by_level_index = build_level_base_z_lookup(self.levels)
        placed_models: list[PlacedGeneratedModel] = []
        for record in self.generation.get_data().generated_objects:
            placement = record.placement
            if placement is None:
                continue
            level = visible_level_by_index.get(placement.level_index)
            base_z = base_z_by_level_index.get(placement.level_index)
            if level is None or base_z is None:
                continue
            generated_model = self.generation.get_generated_object_model(
                record.object_id
            )
            if generated_model is None:
                if not self.generation.is_generated_object_asset_available(
                    record.object_id
                ):
                    continue
                raise ValueError(
                    f"Placed object '{getattr(record, 'object_name', record.object_id)}' "
                    "is temporarily "
                    "unavailable."
                )
            symmetry = self.generation.resolve_symmetric_division_for_record(
                record
            )
            world_x, world_y = level_image_to_world_xy(
                level,
                placement.image_x,
                placement.image_y,
            )
            placed_models.append(
                PlacedGeneratedModel(
                    object_id=record.object_id,
                    model=generated_model,
                    world_position=(
                        world_x,
                        world_y,
                        base_z + placement.height_offset_meters,
                    ),
                    symmetric_preview_orientation=(
                        None if symmetry is None else symmetry.orientation
                    ),
                    symmetric_preview_plane_coordinate=(
                        None if symmetry is None else symmetry.plane_coordinate
                    ),
                    rotation_degrees=placement.rotation_degrees,
                )
            )
        return tuple(placed_models)

    def _handle_surface_texture_generation_completed(
        self,
        _assignment: object,
    ) -> None:
        self._schedule_viewer_preview_refresh(preserve_camera=True)

    def _handle_surface_texture_content_changed(self) -> None:
        self._schedule_viewer_preview_refresh(preserve_camera=True)

    def _refresh_viewer_preview(self, preserve_camera: bool = False) -> None:
        revision = self._viewer_preview_revision
        canvas_is_stale = bool(
            self._canvas_viewer_preview_is_active()
            and self._canvas_viewer_preview_revision != revision
        )
        surface_is_stale = bool(
            self._surface_viewer_preview_is_active()
            and self._surface_viewer_preview_revision != revision
        )
        if not canvas_is_stale and not surface_is_stale:
            return

        preview_levels = self._build_viewer_preview_levels()
        if self._viewer_preview_model_revision != revision:
            dependency_signature_before = (
                self._build_viewer_preview_dependency_signature()
            )
            next_model = self._build_viewer_preview_model(None)
            if next_model is None:
                return
            dependency_signature_after = (
                self._build_viewer_preview_dependency_signature()
            )
            if dependency_signature_before != dependency_signature_after:
                return
            self._viewer_preview_model = next_model
            self._viewer_preview_model_revision = revision
            self._viewer_preview_dependency_signature = (
                dependency_signature_after
            )
            self._viewer_preview_dependency_signature_revision = revision
        generated_model = self._viewer_preview_model

        if canvas_is_stale:
            if generated_model is None:
                self.viewer.set_wall_targets(())
                self.viewer.clear_model()
            else:
                self.viewer.set_wall_targets(
                    tuple(build_fixed_surfaces(preview_levels))
                )
                self.viewer.set_model(
                    generated_model,
                    preserve_camera=preserve_camera,
                )
            self._canvas_viewer_preview_revision = revision
            self._scheduled_viewer_refresh_preserve_camera = True
            self._sync_canvas_window_undo_availability()
            self._clear_committed_doorway_outline_if_displayed()

        if surface_is_stale:
            self.surface_texture_generation.set_preview_context(
                preview_levels,
                generated_model,
            )
            self._surface_viewer_preview_revision = revision

    def _mark_viewer_preview_dirty(
        self,
        preserve_camera: bool = True,
    ) -> int:
        """Invalidate shared preview data and retain future camera intent."""

        self._viewer_preview_revision += 1
        self._scheduled_viewer_refresh_preserve_camera = bool(
            self._scheduled_viewer_refresh_preserve_camera
            and preserve_camera
        )
        return self._viewer_preview_revision

    def _queue_viewer_preview_refresh(self) -> None:
        if not self._active_viewer_preview_needs_refresh():
            return
        if self._is_viewer_refresh_scheduled:
            return
        self._is_viewer_refresh_scheduled = True
        QTimer.singleShot(0, self._run_scheduled_viewer_preview_refresh)

    def _schedule_viewer_preview_refresh(self, preserve_camera: bool = True) -> None:
        self._mark_viewer_preview_dirty(preserve_camera=preserve_camera)
        if not self._viewer_preview_is_active():
            return
        self._queue_viewer_preview_refresh()

    def _ensure_viewer_preview_current(
        self,
        preserve_camera: bool = True,
    ) -> None:
        """Display the current revision without treating a tab click as a change."""

        if not self._viewer_preview_is_active():
            return
        self._refresh_blueprint_file_dependencies(
            include_exported_levels=True
        )
        self._invalidate_viewer_preview_for_dependency_changes(
            preserve_camera=preserve_camera
        )
        if (
            self._canvas_viewer_preview_is_active()
            and self._canvas_viewer_preview_revision
            != self._viewer_preview_revision
        ):
            self._scheduled_viewer_refresh_preserve_camera = bool(
                self._scheduled_viewer_refresh_preserve_camera
                and preserve_camera
            )
        self._queue_viewer_preview_refresh()

    def _run_scheduled_viewer_preview_refresh(self) -> None:
        self._is_viewer_refresh_scheduled = False
        if not self._active_viewer_preview_needs_refresh():
            return

        preserve_camera = self._scheduled_viewer_refresh_preserve_camera
        self._refresh_viewer_preview(preserve_camera=preserve_camera)

    def _handle_save_clicked(self) -> None:
        default_path = (
            Path(self.current_project_path)
            if self.current_project_path is not None
            else Path.cwd() / "housemaker_project.json"
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "save project",
            str(default_path),
            "JSON Files (*.json)",
        )
        if not file_path:
            return

        try:
            save_project(
                path=file_path,
                current_level_index=self.current_level_index,
                levels=self.levels,
                image_library_paths=self.image_library_paths,
                doorway_presets=self.doorway_presets,
                generation=self.generation.get_data(),
                surface_texture_generation=(
                    self.surface_texture_generation.get_data()
                ),
                texture_atlases=self.texture_atlas_workspace.get_data(),
                stairs=self.stairs,
            )
        except ValueError as error:
            QMessageBox.critical(self, "Save failed", str(error))
            return

        self._remember_project_path(file_path)
        QMessageBox.information(self, "Project saved", f"Saved project to:\n{file_path}")

    def _handle_load_clicked(self) -> None:
        if (
            self.generation.is_generating
            or self.surface_texture_generation.is_generating
        ):
            QMessageBox.critical(
                self,
                "Project load failed",
                "Wait for the current generation request to finish before "
                "loading another project.",
            )
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "load project",
            str(Path.cwd()),
            "JSON Files (*.json)",
        )
        if not file_path:
            return

        try:
            self._load_project_path(file_path)
        except PROJECT_LOAD_FAILURES as error:
            QMessageBox.critical(self, "Project load failed", str(error))

    def restore_last_project(self) -> bool:
        """Load the most recently opened or saved project without blocking startup."""

        if self._application_settings is None:
            return False

        stored_path = self._application_settings.get(
            LAST_PROJECT_PATH_SETTING_KEY,
            "",
        )
        if not isinstance(stored_path, str) or not stored_path.strip():
            self._application_settings.remove(LAST_PROJECT_PATH_SETTING_KEY)
            return False

        try:
            self._load_project_path(stored_path)
        except PROJECT_LOAD_FAILURES:
            self.current_project_path = None
            self._application_settings.remove(LAST_PROJECT_PATH_SETTING_KEY)
            return False
        return True

    def _load_project_path(self, file_path: str | Path) -> None:
        project_data = load_project(file_path)
        self._apply_loaded_project(project_data)
        self._remember_project_path(file_path)

    def _remember_project_path(self, file_path: str | Path) -> None:
        normalized_path = str(Path(file_path).expanduser().resolve())
        self.current_project_path = normalized_path
        if self._application_settings is not None:
            self._application_settings.set(
                LAST_PROJECT_PATH_SETTING_KEY,
                normalized_path,
            )

    def _handle_load_image_clicked(self) -> None:
        file_path = self._get_image_file_path()
        if not file_path:
            return

        try:
            self._set_current_level_image(file_path)
        except ValueError as error:
            QMessageBox.critical(self, "Image load failed", str(error))

    def _handle_load_library_image_clicked(self) -> None:
        file_paths = self._get_image_file_paths()
        if not file_paths:
            return

        loaded_image_paths: list[str] = []
        skipped_image_paths: list[str] = []
        for file_path in file_paths:
            normalized_path = str(Path(file_path).resolve())
            if QPixmap(normalized_path).isNull():
                skipped_image_paths.append(normalized_path)
                continue

            self._add_image_to_library(normalized_path)
            loaded_image_paths.append(normalized_path)

        if not loaded_image_paths:
            QMessageBox.critical(
                self,
                "Image load failed",
                "Unable to load the selected image files.",
            )
            return

        self._refresh_image_thumbnail_list(
            selected_image_path=loaded_image_paths[-1]
        )
        if skipped_image_paths:
            QMessageBox.warning(
                self,
                "Some images skipped",
                f"Unable to load {len(skipped_image_paths)} selected image(s).",
            )

    def _handle_convert_video_to_image_clicked(self) -> None:
        file_path = self._get_video_file_path()
        if not file_path:
            return

        try:
            manual_dialog = ManualVideoStitchDialog(file_path, self)
        except ValueError as error:
            QMessageBox.critical(self, "Video conversion failed", str(error))
            return

        if manual_dialog.exec() != QDialog.DialogCode.Accepted:
            return

        stitched_image = manual_dialog.get_stitched_image()
        if stitched_image is None:
            return

        try:
            output_path = save_stitched_image(
                stitched_image,
                build_unique_stitched_output_path(file_path),
            )
        except OSError as error:
            QMessageBox.critical(self, "Video conversion failed", str(error))
            return

        normalized_path = str(output_path.resolve())
        self._add_image_to_library(normalized_path)
        self._refresh_image_thumbnail_list(selected_image_path=normalized_path)
        QMessageBox.information(
            self,
            "Video converted",
            "Saved stitched image to:\n"
            f"{normalized_path}\n\n"
            f"Stitched frames: {manual_dialog.stitched_frame_count}",
        )

    def _handle_image_thumbnail_selection_changed(self, row: int) -> None:
        if self._is_syncing_image_library_controls:
            return

        self._sync_images_tab()

    def _handle_save_selected_image_clicked(self) -> None:
        selected_image_path = self._get_selected_image_library_path()
        if selected_image_path is None:
            return

        selected_pixmap = QPixmap(selected_image_path)
        if selected_pixmap.isNull():
            QMessageBox.critical(
                self,
                "Image save failed",
                f"Unable to load selected image:\n{selected_image_path}",
            )
            return

        output_path = self._get_png_save_file_path(Path(selected_image_path).stem)
        if not output_path:
            return

        normalized_output_path = _ensure_png_file_suffix(output_path)
        if not selected_pixmap.save(normalized_output_path, "PNG"):
            QMessageBox.critical(
                self,
                "Image save failed",
                f"Unable to save PNG file:\n{normalized_output_path}",
            )
            return

        QMessageBox.information(
            self,
            "Image saved",
            f"Saved PNG image to:\n{normalized_output_path}",
        )

    def _handle_delete_image_clicked(self) -> None:
        selected_image_path = self._get_selected_image_library_path()
        if selected_image_path is None:
            return

        normalized_path = str(Path(selected_image_path).resolve())
        self.image_library_paths = [
            library_path
            for library_path in self.image_library_paths
            if library_path != normalized_path
        ]
        did_clear_textures = self._clear_wall_textures_using_image(normalized_path)
        self._refresh_image_thumbnail_list()
        if did_clear_textures:
            self._handle_texture_creator_texture_changed()

    def _get_image_file_path(self) -> str:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "load image",
            str(Path.home()),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        return file_path

    def _get_image_file_paths(self) -> list[str]:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "load images",
            str(Path.home()),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        return file_paths

    def _get_video_file_path(self) -> str:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "convert video to image",
            str(Path.home()),
            VIDEO_FILE_FILTER,
        )
        return file_path

    def _get_png_save_file_path(self, default_image_name: str) -> str:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "save png",
            str(Path.home() / f"{default_image_name}.png"),
            "PNG Files (*.png)",
        )
        return file_path

    def _refresh_levels_list(self) -> None:
        self._is_syncing_level_controls = True
        self.levels_list.clear()
        for level in self.levels:
            self.levels_list.addItem(self._build_level_item(level))
        self.levels_list.setCurrentRow(self.current_level_index)
        self._is_syncing_level_controls = False

    def _build_level_item(self, level: LevelData) -> QListWidgetItem:
        return QListWidgetItem(level.display_name)

    def _refresh_doorway_preset_list(
        self,
        selected_index: int | None = None,
    ) -> None:
        if selected_index is None:
            selected_index = self.doorway_preset_list.currentRow()

        self.doorway_preset_list.blockSignals(True)
        self.doorway_preset_list.clear()
        for preset in self.doorway_presets:
            self.doorway_preset_list.addItem(
                _format_doorway_preset_label(preset)
            )

        if 0 <= selected_index < self.doorway_preset_list.count():
            self.doorway_preset_list.setCurrentRow(selected_index)
        self.doorway_preset_list.blockSignals(False)
        self._update_doorway_preset_button_state()

    def _update_stair_button_state(self) -> None:
        placement_active = self.canvas.is_stair_placement_active()
        has_complete_endpoints = (
            self.canvas.get_stair_placement_draft() is not None
        )
        self.add_stairs_button.setText(
            "Confirm stairs" if has_complete_endpoints else "Add stairs"
        )
        self.add_stairs_button.setEnabled(
            not placement_active or has_complete_endpoints
        )
        self.stair_style_combo.setEnabled(not placement_active)

    def _get_selected_doorway_preset(self) -> DoorwayPreset | None:
        selected_index = self.doorway_preset_list.currentRow()
        if selected_index < 0 or selected_index >= len(self.doorway_presets):
            return None

        return self.doorway_presets[selected_index]

    def _get_selected_placed_doorway(self) -> DoorwayData | None:
        selected_index = self.canvas.selected_doorway_index
        if (
            selected_index is None
            or selected_index < 0
            or selected_index >= len(self.canvas.doorways)
        ):
            return None

        return self.canvas.doorways[selected_index]

    def _update_doorway_preset_button_state(self) -> None:
        has_selected_preset = self._get_selected_doorway_preset() is not None
        self.remove_doorway_preset_button.setEnabled(
            has_selected_preset and len(self.doorway_presets) > 1
        )
        self.place_doorway_button.setEnabled(has_selected_preset)
        self.save_doorway_template_button.setEnabled(
            self._get_selected_placed_doorway() is not None
        )

    def _refresh_room_dependent_controls(self) -> None:
        """Refresh features that still consume persisted room geometry."""

        self._refresh_uv_rooms_list()
        self._sync_uv_controls()
        self._sync_texture_creator_tab()

    def _refresh_uv_rooms_list(self) -> None:
        selected_room_index = self._get_selected_uv_room_index()
        self.uv_rooms_list.clear()
        for room_index, room in enumerate(self.current_level.rooms):
            room_item = self._build_room_item(room_index, room)
            self.uv_rooms_list.addItem(room_item)

        if self.uv_rooms_list.count() == 0:
            self._sync_uv_controls()
            return

        if selected_room_index is None:
            self.uv_rooms_list.setCurrentRow(0)
            return

        if selected_room_index < self.uv_rooms_list.count():
            self.uv_rooms_list.setCurrentRow(selected_room_index)
            return

        self.uv_rooms_list.setCurrentRow(self.uv_rooms_list.count() - 1)

    def _build_room_item(self, room_index: int, room: RoomData) -> QListWidgetItem:
        room_name = room.name or "Room"
        room_item = QListWidgetItem(f"{room_name} ({room.height_meters:.2f} m)")
        room_item.setData(Qt.ItemDataRole.UserRole, room_index)
        return room_item

    def _get_selected_uv_room_index(self) -> int | None:
        selected_item = self.uv_rooms_list.currentItem()
        if selected_item is None:
            return None

        room_index = selected_item.data(Qt.ItemDataRole.UserRole)
        if room_index is None:
            return None

        return int(room_index)

    def _get_selected_uv_room(self) -> RoomData | None:
        room_index = self._get_selected_uv_room_index()
        if room_index is None or room_index >= len(self.current_level.rooms):
            return None

        return self.current_level.rooms[room_index]

    def _get_selected_uv_wall_placement(self) -> UvWallPlacement | None:
        selected_room = self._get_selected_uv_room()
        selected_wall_key = self.uv_canvas.get_selected_wall_key()
        if selected_room is None or selected_wall_key is None:
            return None

        layout = build_uv_wall_layout(
            room=selected_room,
            vertex_data=self.current_level.vertex_data,
            wall_height_meters=selected_room.height_meters,
        )
        for placement in layout.placements:
            if placement.wall.key == selected_wall_key:
                return placement

        return None

    def _refresh_image_thumbnail_list(
        self,
        selected_image_path: str | None = None,
    ) -> None:
        if not hasattr(self, "image_thumbnail_list"):
            return

        if selected_image_path is None:
            selected_image_path = self._get_selected_image_library_path()

        self._is_syncing_image_library_controls = True
        self.image_thumbnail_list.clear()
        self._image_thumbnail_source_keys.clear()
        for image_path in self.image_library_paths:
            self.image_thumbnail_list.addItem(
                self._build_image_thumbnail_item(image_path)
            )
        self._select_image_thumbnail_path(selected_image_path)
        self._is_syncing_image_library_controls = False
        self._sync_images_tab()
        self._refresh_texture_image_combo()

    def _build_image_thumbnail_item(self, image_path: str) -> QListWidgetItem:
        image_name = Path(image_path).name
        thumbnail_item = QListWidgetItem(image_name)
        thumbnail_item.setData(Qt.ItemDataRole.UserRole, image_path)
        thumbnail_item.setToolTip(image_path)

        self._sync_image_thumbnail_item(thumbnail_item, image_path)
        return thumbnail_item

    def _refresh_stale_image_thumbnails(self) -> None:
        """Reload only library icons whose file revisions changed."""

        if not hasattr(self, "image_thumbnail_list"):
            return
        active_paths: set[str] = set()
        for row_index in range(self.image_thumbnail_list.count()):
            thumbnail_item = self.image_thumbnail_list.item(row_index)
            image_path = str(
                thumbnail_item.data(Qt.ItemDataRole.UserRole) or ""
            )
            if not image_path:
                continue
            active_paths.add(image_path)
            self._sync_image_thumbnail_item(thumbnail_item, image_path)
        self._image_thumbnail_source_keys = {
            image_path: revision
            for image_path, revision in self._image_thumbnail_source_keys.items()
            if image_path in active_paths
        }

    def _sync_image_thumbnail_item(
        self,
        thumbnail_item: QListWidgetItem,
        image_path: str,
    ) -> None:
        """Install one thumbnail only after its current revision decodes."""

        source_key = _build_local_file_revision(image_path)
        if self._image_thumbnail_source_keys.get(image_path) == source_key:
            return

        if not Path(image_path).is_file():
            thumbnail_item.setIcon(QIcon())
            thumbnail_item.setText(f"{Path(image_path).name}\nmissing")
            self._image_thumbnail_source_keys[image_path] = source_key
            return

        thumbnail_pixmap = _load_image_pixmap(image_path)
        if thumbnail_pixmap.isNull():
            thumbnail_item.setIcon(QIcon())
            thumbnail_item.setText(f"{Path(image_path).name}\nmissing")
            self._image_thumbnail_source_keys.pop(image_path, None)
            return

        thumbnail_item.setText(Path(image_path).name)
        thumbnail_item.setIcon(
            QIcon(
                thumbnail_pixmap.scaled(
                    72,
                    72,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        )
        self._image_thumbnail_source_keys[image_path] = source_key

    def _select_image_thumbnail_path(self, selected_image_path: str | None) -> None:
        self.image_thumbnail_list.clearSelection()
        self.image_thumbnail_list.setCurrentRow(-1)
        if selected_image_path is None:
            return

        normalized_path = str(Path(selected_image_path).resolve())
        for row_index in range(self.image_thumbnail_list.count()):
            thumbnail_item = self.image_thumbnail_list.item(row_index)
            image_path = thumbnail_item.data(Qt.ItemDataRole.UserRole)
            if image_path != normalized_path:
                continue

            self.image_thumbnail_list.setCurrentRow(row_index)
            return

    def _get_selected_image_library_path(self) -> str | None:
        if not hasattr(self, "image_thumbnail_list"):
            return None

        selected_item = self.image_thumbnail_list.currentItem()
        if selected_item is None:
            return None

        image_path = selected_item.data(Qt.ItemDataRole.UserRole)
        if image_path is None:
            return None

        return str(image_path)

    def _add_image_to_library(self, image_path: str) -> None:
        normalized_path = str(Path(image_path).resolve())
        if normalized_path in self.image_library_paths:
            return

        self.image_library_paths.append(normalized_path)

    def _clear_wall_textures_using_image(self, image_path: str) -> bool:
        normalized_path = str(Path(image_path).resolve())
        did_clear_textures = False
        for level in self.levels:
            for room in level.rooms:
                original_texture_count = len(room.wall_textures)
                room.wall_textures = {
                    wall_key: texture_data
                    for wall_key, texture_data in room.wall_textures.items()
                    if texture_data.image_path != normalized_path
                }
                did_clear_textures = (
                    did_clear_textures
                    or len(room.wall_textures) != original_texture_count
                )

        return did_clear_textures

    @staticmethod
    def _normalize_image_library_paths(image_paths: list[str]) -> list[str]:
        normalized_paths: list[str] = []
        for image_path in image_paths:
            normalized_path = str(Path(image_path).resolve())
            if normalized_path in normalized_paths:
                continue

            normalized_paths.append(normalized_path)

        return normalized_paths

    def _refresh_texture_image_combo(
        self,
        selected_image_path: str | None = None,
    ) -> None:
        if not hasattr(self, "texture_image_combo"):
            return

        if selected_image_path is None:
            selected_image_path = self._get_selected_texture_image_path()
        if selected_image_path is None:
            selected_image_path = self._get_texture_creator_saved_image_path()

        has_wall = self._get_texture_creator_wall_placement() is not None
        self.texture_image_combo.setEnabled(has_wall and bool(self.image_library_paths))
        self._is_syncing_texture_controls = True
        self.texture_image_combo.clear()
        self.texture_image_combo.addItem("Select image", None)
        for image_path in self.image_library_paths:
            self.texture_image_combo.addItem(Path(image_path).name, image_path)

        if selected_image_path is not None:
            normalized_path = str(Path(selected_image_path).resolve())
            for item_index in range(self.texture_image_combo.count()):
                if self.texture_image_combo.itemData(item_index) != normalized_path:
                    continue

                self.texture_image_combo.setCurrentIndex(item_index)
                break

        self._is_syncing_texture_controls = False
        self._sync_texture_creator_canvas()

    def _get_selected_texture_image_path(self) -> str | None:
        if not hasattr(self, "texture_image_combo"):
            return None

        image_path = self.texture_image_combo.currentData()
        if image_path is None:
            return None

        return str(image_path)

    def _get_texture_creator_saved_image_path(self) -> str | None:
        selected_room = self._get_texture_creator_room()
        if selected_room is None or self.texture_creator_wall_key is None:
            return None

        texture_data = selected_room.wall_textures.get(self.texture_creator_wall_key)
        if texture_data is None:
            return None

        return texture_data.image_path

    def _handle_viewer_wall_selected(
        self,
        level_index: int,
        room_index: int,
        wall_key: str,
    ) -> None:
        level_position = self._get_level_position_for_level_index(level_index)
        if level_position is None:
            return
        if room_index < 0 or room_index >= len(self.levels[level_position].rooms):
            return

        self.texture_creator_level_index = level_position
        self.texture_creator_room_index = room_index
        self.texture_creator_wall_key = wall_key
        if self.current_level_index != level_position:
            self.current_level_index = level_position
            self._sync_level_controls()
            self._sync_canvas_to_current_level()
            self._refresh_room_dependent_controls()

        if self.uv_rooms_list.currentRow() != room_index:
            self.uv_rooms_list.setCurrentRow(room_index)
        self.uv_canvas.set_selected_wall_key(wall_key)
        self._sync_uv_controls()
        self._sync_texture_creator_tab()
        self.side_tabs.setCurrentWidget(self.texture_creator_tab)

    def _handle_texture_image_selection_changed(self, item_index: int) -> None:
        if self._is_syncing_texture_controls:
            return

        self._sync_texture_creator_canvas()

    def _handle_texture_creator_texture_changed(self) -> None:
        self.uv_canvas.update()
        self._schedule_viewer_preview_refresh()

    def _sync_texture_creator_tab(self) -> None:
        if not hasattr(self, "texture_creator_canvas"):
            return

        selected_room = self._get_texture_creator_room()
        selected_placement = self._get_texture_creator_wall_placement()
        has_wall = selected_room is not None and selected_placement is not None
        self.texture_image_combo.setEnabled(has_wall and bool(self.image_library_paths))
        if not has_wall:
            self.texture_creator_wall_label.setText("Select a wall in Viewer")
            self._update_texture_creator_aspect_ratio_label(None)
            self._refresh_texture_image_combo(selected_image_path=None)
            return

        self.texture_creator_wall_label.setText(
            f"{selected_room.name or 'Room'} - "
            f"{selected_placement.wall.projection_direction} wall"
        )
        self._update_texture_creator_aspect_ratio_label(selected_placement)
        self._refresh_texture_image_combo()

    def _sync_texture_creator_canvas(self) -> None:
        if not hasattr(self, "texture_creator_canvas"):
            return

        selected_room = self._get_texture_creator_room()
        selected_placement = self._get_texture_creator_wall_placement()
        selected_image_path = self._get_selected_texture_image_path()
        if selected_room is None or selected_placement is None:
            self.texture_creator_canvas.set_context(None, None, None, None)
            return

        self.texture_creator_canvas.set_context(
            room=selected_room,
            wall_key=selected_placement.wall.key,
            wall_size=_get_logical_wall_size(selected_placement),
            image_path=selected_image_path,
            segment_count=selected_placement.segment_count,
        )

    def _get_texture_creator_room(self) -> RoomData | None:
        if self.texture_creator_level_index is None:
            return None
        if self.texture_creator_room_index is None:
            return None
        if self.texture_creator_level_index >= len(self.levels):
            return None

        level = self.levels[self.texture_creator_level_index]
        if self.texture_creator_room_index >= len(level.rooms):
            return None

        return level.rooms[self.texture_creator_room_index]

    def _get_texture_creator_wall_placement(self) -> UvWallPlacement | None:
        selected_room = self._get_texture_creator_room()
        if selected_room is None or self.texture_creator_wall_key is None:
            return None
        if self.texture_creator_level_index is None:
            return None

        selected_level = self.levels[self.texture_creator_level_index]
        layout = build_uv_wall_layout(
            room=selected_room,
            vertex_data=selected_level.vertex_data,
            wall_height_meters=selected_room.height_meters,
        )
        for placement in layout.placements:
            if placement.wall.key == self.texture_creator_wall_key:
                return placement

        return None

    def _get_level_position_for_level_index(self, level_index: int) -> int | None:
        for level_position, level in enumerate(self.levels):
            if level.index == level_index:
                return level_position

        return None

    def _sync_level_controls(self) -> None:
        self._is_syncing_level_controls = True
        self.height_level_spinbox.setValue(self.current_level.height_meters)
        self.level_scale_spinbox.setValue(self.current_level.scale)
        self.level_x_offset_spinbox.setValue(
            self.current_level.offset_x_meters
        )
        self.level_y_offset_spinbox.setValue(
            self.current_level.offset_y_meters
        )
        self.floor_thickness_spinbox.setValue(
            self.current_level.floor_thickness_meters
        )
        self._update_floor_contour_status_label()
        self.include_yes_radio.setChecked(self.current_level.include_in_export)
        self.include_no_radio.setChecked(not self.current_level.include_in_export)
        if self.levels_list.currentRow() != self.current_level_index:
            self.levels_list.setCurrentRow(self.current_level_index)
        self._update_blueprint_name_label()
        self._is_syncing_level_controls = False

    def _sync_uv_controls(self) -> None:
        self._is_syncing_uv_controls = True
        selected_room = self._get_selected_uv_room()
        has_room = selected_room is not None
        self.uv_map_width_spinbox.setEnabled(has_room)
        self.uv_map_height_spinbox.setEnabled(has_room)
        self.optimize_uv_button.setEnabled(has_room)
        self.optimize_all_uv_button.setEnabled(has_room)
        self.reset_uv_defaults_button.setEnabled(has_room)
        self.basic_optimization_radio.setEnabled(has_room)
        self.free_placement_radio.setEnabled(has_room)
        self.subdivision_optimization_radio.setEnabled(has_room)
        self.complex_optimization_passes_spinbox.setEnabled(
            has_room and self.free_placement_radio.isChecked()
        )
        self.unoccupied_uv_pixels_label.setEnabled(has_room)
        self.uv_aspect_ratio_label.setEnabled(has_room)

        if selected_room is None:
            self.uv_canvas.set_room_context(None, None, DEFAULT_ROOM_HEIGHT_METERS)
            self.uv_map_width_spinbox.setValue(DEFAULT_UV_MAP_WIDTH)
            self.uv_map_height_spinbox.setValue(DEFAULT_UV_MAP_HEIGHT)
            self.uv_wall_scale_spinbox.setValue(DEFAULT_WALL_UV_SCALE)
            self.uv_wall_scale_spinbox.setEnabled(False)
            self.uv_wall_rotation_spinbox.setValue(DEFAULT_WALL_UV_ROTATION_DEGREES)
            self.uv_wall_rotation_spinbox.setEnabled(False)
            self.unoccupied_uv_pixels_label.setText("Unoccupied pixels: 0")
            self._update_uv_aspect_ratio_label(None)
            self._is_syncing_uv_controls = False
            return

        self.uv_canvas.set_room_context(
            selected_room,
            self.current_level.vertex_data,
            selected_room.height_meters,
        )
        self.uv_map_width_spinbox.setValue(selected_room.uv_map_width)
        self.uv_map_height_spinbox.setValue(selected_room.uv_map_height)

        selected_wall_key = self.uv_canvas.get_selected_wall_key()
        selected_placement = self._get_selected_uv_wall_placement()
        has_selected_wall = selected_wall_key is not None
        self.uv_wall_scale_spinbox.setEnabled(has_selected_wall)
        self.uv_wall_rotation_spinbox.setEnabled(has_selected_wall)
        if selected_wall_key is None:
            self.uv_wall_scale_spinbox.setValue(DEFAULT_WALL_UV_SCALE)
            self.uv_wall_rotation_spinbox.setValue(DEFAULT_WALL_UV_ROTATION_DEGREES)
        else:
            self.uv_wall_scale_spinbox.setValue(
                selected_room.wall_uv_scales.get(
                    selected_wall_key,
                    DEFAULT_WALL_UV_SCALE,
                )
            )
            self.uv_wall_rotation_spinbox.setValue(
                selected_room.wall_uv_rotations.get(
                    selected_wall_key,
                    DEFAULT_WALL_UV_ROTATION_DEGREES,
                )
            )

        self._update_unoccupied_uv_pixels_label()
        self._update_uv_aspect_ratio_label(selected_placement)
        self._is_syncing_uv_controls = False

    def _update_unoccupied_uv_pixels_label(self) -> None:
        selected_room = self._get_selected_uv_room()
        if selected_room is None:
            self.unoccupied_uv_pixels_label.setText("Unoccupied pixels: 0")
            return

        unoccupied_pixels = calculate_unoccupied_uv_pixels(
            room=selected_room,
            vertex_data=self.current_level.vertex_data,
            wall_height_meters=selected_room.height_meters,
        )
        self.unoccupied_uv_pixels_label.setText(
            f"Unoccupied pixels: {unoccupied_pixels:,}"
        )

    def _update_uv_aspect_ratio_label(
        self,
        placement: UvWallPlacement | None,
    ) -> None:
        self.uv_aspect_ratio_label.setText(
            _build_wall_aspect_ratio_text(placement)
        )

    def _update_texture_creator_aspect_ratio_label(
        self,
        placement: UvWallPlacement | None,
    ) -> None:
        self.texture_creator_aspect_ratio_label.setText(
            _build_wall_aspect_ratio_text(placement)
        )
        self.texture_creator_resolution_label.setText(
            _build_wall_resolution_text(placement)
        )

    def _handle_level_selection_changed(self, level_index: int) -> None:
        if (
            self._is_syncing_level_controls
            or level_index < 0
            or level_index >= len(self.levels)
        ):
            return

        if level_index != self.current_level_index:
            self._commit_pending_doorway_mesh_update()
        self.current_level_index = level_index
        self._sync_level_controls()
        self._sync_canvas_to_current_level()
        self._update_pending_stair_level_status()
        self._refresh_room_dependent_controls()
        self._ensure_viewer_preview_current()

    def _handle_height_level_changed(self, value: float) -> None:
        if self._is_syncing_level_controls:
            return

        self.current_level.height_meters = value
        self._schedule_viewer_preview_refresh()

    def _handle_level_scale_changed(self, value: float) -> None:
        if self._is_syncing_level_controls:
            return

        self.current_level.scale = float(value)
        self.canvas.update()
        self._schedule_viewer_preview_refresh()

    def _handle_level_x_offset_changed(self, value: float) -> None:
        if self._is_syncing_level_controls:
            return

        self.current_level.offset_x_meters = float(value)
        self.canvas.update()
        self._schedule_viewer_preview_refresh()

    def _handle_level_y_offset_changed(self, value: float) -> None:
        if self._is_syncing_level_controls:
            return

        self.current_level.offset_y_meters = float(value)
        self.canvas.update()
        self._schedule_viewer_preview_refresh()

    def _handle_floor_thickness_changed(self, value: float) -> None:
        if self._is_syncing_level_controls:
            return

        self.current_level.floor_thickness_meters = float(value)
        self._schedule_viewer_preview_refresh()

    def _handle_doorway_preset_selection_changed(self, _row: int) -> None:
        self._update_doorway_preset_button_state()

    def _handle_save_doorway_template_clicked(self) -> None:
        doorway = self._get_selected_placed_doorway()
        if doorway is None:
            self._update_doorway_preset_button_state()
            return

        self.doorway_presets.append(
            DoorwayPreset(
                width_meters=doorway.width_meters,
                height_meters=doorway.height_meters,
                shape=doorway.shape,
                arch_amount=doorway.arch_amount,
            )
        )
        self._refresh_doorway_preset_list(
            selected_index=len(self.doorway_presets) - 1
        )

    def _handle_remove_doorway_preset_clicked(self) -> None:
        selected_index = self.doorway_preset_list.currentRow()
        if (
            len(self.doorway_presets) <= 1
            or selected_index < 0
            or selected_index >= len(self.doorway_presets)
        ):
            return

        del self.doorway_presets[selected_index]
        next_selected_index = min(selected_index, len(self.doorway_presets) - 1)
        self._refresh_doorway_preset_list(
            selected_index=next_selected_index
        )

    def _handle_place_selected_doorway_clicked(self) -> None:
        doorway_preset = self._get_selected_doorway_preset()
        if doorway_preset is None:
            return

        self.canvas.start_doorway_placement(doorway_preset)
        self.workspace_tabs.setCurrentWidget(self.canvas_viewer_workspace)

    def _handle_canvas_doorway_selection_changed(
        self,
        doorway_index: int,
    ) -> None:
        """Synchronize the arch control with the selected placed doorway."""

        doorway = (
            self.canvas.doorways[doorway_index]
            if 0 <= doorway_index < len(self.canvas.doorways)
            else None
        )
        blocker = QSignalBlocker(self.selected_doorway_arch_checkbox)
        amount_blocker = QSignalBlocker(
            self.selected_doorway_arch_amount_spinbox
        )
        arch_is_selected = bool(
            doorway is not None and doorway.shape == DOORWAY_SHAPE_ARCH
        )
        self.selected_doorway_arch_checkbox.setEnabled(doorway is not None)
        self.selected_doorway_arch_checkbox.setChecked(arch_is_selected)
        self.selected_doorway_arch_amount_spinbox.setEnabled(arch_is_selected)
        self.selected_doorway_arch_amount_spinbox.setValue(
            (
                doorway.arch_amount
                if doorway is not None
                else DEFAULT_DOORWAY_ARCH_AMOUNT
            )
            * 100.0
        )
        self._update_doorway_preset_button_state()
        del amount_blocker
        del blocker

    def _handle_selected_doorway_arch_toggled(self, enabled: bool) -> None:
        """Turn the selected doorway arch profile on or off."""

        shape = (
            DOORWAY_SHAPE_ARCH
            if enabled
            else DOORWAY_SHAPE_RECTANGULAR
        )
        self.canvas.set_selected_doorway_shape(shape)
        self._handle_canvas_doorway_selection_changed(
            -1
            if self.canvas.selected_doorway_index is None
            else self.canvas.selected_doorway_index
        )

    def _handle_selected_doorway_arch_amount_changed(
        self,
        arch_amount_percent: float,
    ) -> None:
        """Preview a normalized arch amount for the selected doorway."""

        if self.canvas.set_selected_doorway_arch_amount(
            arch_amount_percent / 100.0
        ):
            return
        self._handle_canvas_doorway_selection_changed(
            -1
            if self.canvas.selected_doorway_index is None
            else self.canvas.selected_doorway_index
        )

    def _handle_doorways_changed(self) -> None:
        """Commit structural doorway changes without a debounce delay."""

        self._is_doorway_dimension_drag_active = False
        self.current_level.doorways = self.canvas.doorways
        next_snapshot = self._copy_doorways(self.current_level.doorways)
        snapshot_changed = bool(
            self._viewer_doorways_by_level_index.get(
                self.current_level.index
            )
            != next_snapshot
        )
        self._cancel_pending_doorway_mesh_update(clear_outline=True)
        if not snapshot_changed:
            return
        self._viewer_doorways_by_level_index[self.current_level.index] = (
            next_snapshot
        )
        self._schedule_viewer_preview_refresh()

    def _handle_doorway_dimension_drag_started(self) -> None:
        """Pause doorway stabilization while a resize handle remains held."""

        self._is_doorway_dimension_drag_active = True
        self._doorway_mesh_update_timer.stop()

    def _handle_doorway_dimension_drag_finished(self) -> None:
        """Begin a fresh stabilization interval after the handle is released."""

        self._is_doorway_dimension_drag_active = False
        if self._pending_doorway_mesh_level_index is not None:
            self._doorway_mesh_update_timer.start()
            return
        self._handle_doorway_dimension_preview_changed()

    def _handle_doorway_dimension_preview_changed(self) -> None:
        """Show live dimensions as an outline and debounce the wall rebuild."""

        self.current_level.doorways = self.canvas.doorways
        level = self.current_level
        committed_doorways = self._viewer_doorways_by_level_index.get(
            level.index
        )
        if committed_doorways is None:
            committed_doorways = self._copy_doorways(level.doorways)
            self._viewer_doorways_by_level_index[level.index] = (
                committed_doorways
            )

        if self._copy_doorways(level.doorways) == committed_doorways:
            self._doorway_mesh_update_timer.stop()
            self._pending_doorway_mesh_level_index = None
            self._clear_committed_doorway_outline_if_displayed()
            if self._doorway_outline_commit_revision is None:
                self.viewer.set_doorway_preview_outline(None)
            return

        selected_index = self.canvas.selected_doorway_index
        if (
            selected_index is None
            or selected_index < 0
            or selected_index >= len(level.doorways)
        ):
            self._cancel_pending_doorway_mesh_update(clear_outline=True)
            return

        pending_level_index = self._pending_doorway_mesh_level_index
        if (
            pending_level_index is not None
            and pending_level_index != level.index
        ):
            self._commit_pending_doorway_mesh_update()

        doorway = level.doorways[selected_index]
        try:
            outline_positions = build_doorway_world_outline_positions(
                self.levels,
                level,
                doorway,
            )
        except (TypeError, ValueError):
            self.viewer.set_doorway_preview_outline(None)
        else:
            self.viewer.set_doorway_preview_outline(outline_positions)

        self._pending_doorway_mesh_level_index = level.index
        if self._is_doorway_dimension_drag_active:
            self._doorway_mesh_update_timer.stop()
        else:
            self._doorway_mesh_update_timer.start()

    def _handle_add_stairs_clicked(self) -> None:
        draft = self.canvas.get_stair_placement_draft()
        if draft is not None:
            if self.canvas.is_stair_ready_for_confirmation():
                try:
                    stair = _build_stair_data_from_placement(draft)
                    build_stair_meshes(self.levels, [stair])
                except (TypeError, ValueError) as error:
                    self.stair_status_label.setText(
                        f"Stair not added: {error}"
                    )
                    return
            self.canvas.confirm_stair_placement()
            return
        if self.canvas.is_stair_placement_active():
            return

        if self.canvas.blueprint_image is None:
            QMessageBox.information(
                self,
                "Blueprint required",
                "Load a blueprint image for this level before placing stairs.",
            )
            return

        style = self.stair_style_combo.currentData()
        self.workspace_tabs.setCurrentWidget(self.canvas_viewer_workspace)
        self.canvas_viewer_tabs.setCurrentWidget(self.canvas)
        self.canvas.start_stair_placement(
            DEFAULT_STAIR_STYLE if style is None else str(style)
        )
        self._update_stair_button_state()
        self.stair_status_label.setText(
            "Click two points to define the stair opening on this level."
        )
        QMessageBox.information(
            self,
            "Add stairs",
            "1. Click two points for the stair opening on this level.\n"
            "2. Select a different level.\n"
            "3. Click two points for the stair opening on that level.\n"
            "4. Optionally add two-point curve guides, in order from the "
            "stair start toward its end.\n"
            "5. Click Confirm stairs. Backspace removes the latest guide.\n\n"
            "Points may be placed freely or snapped to existing Canvas "
            "geometry. Each opening remains attached to its own level when "
            "that level is scaled or moved. Right-click or Escape cancels "
            "the entire draft.",
        )

    def _handle_stair_start_placed(self, placement: object) -> None:
        start_level_index = _get_stair_placement_value(
            placement,
            "start_level_index",
        )
        self.stair_status_label.setText(
            "Stair opening set on "
            f"{_format_level_name(self.levels, start_level_index)}. "
            "Select a different level, then click two points for its opening."
        )
        self._update_stair_button_state()

    def _handle_stair_placement_ready(self, placement: object) -> None:
        intermediate_count = len(
            _get_stair_intermediate_section_payloads(placement)
        )
        guide_text = (
            "No curve guides added yet."
            if intermediate_count == 0
            else (
                f"{intermediate_count} curve guide"
                f"{'s' if intermediate_count != 1 else ''} added."
            )
        )
        self.stair_status_label.setText(
            f"Stair endpoints are ready. {guide_text} Add another two-point "
            "guide or click Confirm stairs. Backspace removes the latest guide."
        )
        self._update_stair_button_state()

    def _handle_stair_placement_completed(self, placement: object) -> None:
        try:
            stair = _build_stair_data_from_placement(placement)
        except (TypeError, ValueError) as error:
            self.stair_status_label.setText(f"Stair not added: {error}")
            self._update_stair_button_state()
            return

        try:
            build_stair_meshes(self.levels, [stair])
        except ValueError as error:
            self.stair_status_label.setText(f"Stair not added: {error}")
            self._update_stair_button_state()
            return

        self.stairs.append(stair)
        self.canvas.set_stair_context(self.stairs, self.current_level)
        self.stair_status_label.setText(
            "Added "
            f"{_format_stair_style_label(stair.style).lower()} stairs from "
            f"{_format_level_name(self.levels, stair.start_level_index)} to "
            f"{_format_level_name(self.levels, stair.end_level_index)}."
        )
        self._update_stair_button_state()
        # A stair can extend beyond the previously framed house bounds. Refit
        # the Canvas 3D view so a successful placement is visible immediately.
        self._schedule_viewer_preview_refresh(preserve_camera=False)

    def _handle_stair_placement_cancelled(self) -> None:
        self.stair_status_label.setText("Stair placement cancelled.")
        self._update_stair_button_state()

    def _handle_stair_placement_invalid_endpoint(self, message: str) -> None:
        self.stair_status_label.setText(str(message))

    def _update_pending_stair_level_status(self) -> None:
        pending = self.canvas.get_pending_stair_placement()
        draft = self.canvas.get_stair_placement_draft()
        pending_point = self.canvas.get_pending_stair_point()
        if pending is None and draft is None and pending_point is None:
            return

        if draft is not None:
            if pending_point is not None:
                owner_level_index = _get_stair_placement_value(
                    pending_point,
                    "level_index",
                )
                if self.current_level.index != owner_level_index:
                    self.stair_status_label.setText(
                        "Return to "
                        f"{_format_level_name(self.levels, owner_level_index)} "
                        "and place the second point of this curve guide."
                    )
                else:
                    self.stair_status_label.setText(
                        "Click the second point of this curve guide."
                    )
            else:
                self._handle_stair_placement_ready(draft)
            return

        if pending_point is not None:
            owner_level_index = _get_stair_placement_value(
                pending_point,
                "level_index",
            )
            if self.current_level.index != owner_level_index:
                self.stair_status_label.setText(
                    "Return to "
                    f"{_format_level_name(self.levels, owner_level_index)} "
                    "and place the second point of this opening."
                )
            else:
                opening_name = "first" if pending is None else "second"
                self.stair_status_label.setText(
                    f"Click the second point of the {opening_name} opening."
                )
            return

        start_level_index = _get_stair_placement_value(
            pending,
            "start_level_index",
        )
        if self.current_level.index == start_level_index:
            self.stair_status_label.setText(
                "The first opening is complete. Select a different level."
            )
            return

        if self.canvas.blueprint_image is None:
            self.stair_status_label.setText(
                "Load a blueprint image on this level before placing the "
                "stair end."
            )
            return

        self.stair_status_label.setText(
            "Click two points for the stair opening on "
            f"{self.current_level.display_name}."
        )

    def _handle_stair_delete_requested(self, stair_index: int) -> None:
        self._delete_stair_at_index(stair_index)

    def _delete_stair_at_index(self, stair_index: int) -> None:
        if not 0 <= stair_index < len(self.stairs):
            return

        del self.stairs[stair_index]
        self.canvas.set_stair_context(self.stairs, self.current_level)
        self._update_stair_button_state()
        self.stair_status_label.setText("Stair deleted.")
        self._schedule_viewer_preview_refresh()

    def _handle_generation_settings_changed(self) -> None:
        settings = self.settings_widget.get_settings()
        self._set_doorway_mesh_update_delay_seconds(
            settings.doorway_mesh_update_delay_seconds
        )
        self._set_canvas_3d_navigation_shortcut(
            settings.canvas_3d_navigation_toggle_hotkey
        )
        self.generation.set_runtime_settings(settings)
        self.surface_texture_generation.set_runtime_settings(settings)
        self._apply_fullscreen_3d_viewer_screen(
            settings.fullscreen_3d_viewer_screen_id
        )
        self._apply_jobs_window_screen(settings.jobs_window_screen_id)

    def _handle_set_floor_contour_clicked(self) -> None:
        self.canvas.start_floor_contour_designation()
        self.workspace_tabs.setCurrentWidget(self.canvas_viewer_workspace)

    def _handle_clear_floor_contour_clicked(self) -> None:
        self.canvas.clear_floor_contour()

    def _handle_floor_contour_changed(self, vertex_ids: object) -> None:
        if not isinstance(vertex_ids, list | tuple):
            return

        self.current_level.floor_contour_vertex_ids = tuple(
            int(vertex_id) for vertex_id in vertex_ids
        )
        self._update_floor_contour_status_label()
        self._schedule_viewer_preview_refresh()

    def _handle_include_toggled(self, checked: bool) -> None:
        if self._is_syncing_level_controls or not checked:
            return

        self.current_level.include_in_export = self.include_yes_radio.isChecked()
        self._schedule_viewer_preview_refresh()

    def _handle_snap_middle_equal_angle_toggled(self, checked: bool) -> None:
        self.canvas.set_snap_middle_equal_angle_only(checked)

    def _handle_uv_room_selection_changed(self, room_index: int) -> None:
        self._sync_uv_controls()

    def _handle_optimization_mode_toggled(self, checked: bool) -> None:
        if not hasattr(self, "uv_rooms_list"):
            return

        selected_room = self._get_selected_uv_room()
        self.complex_optimization_passes_spinbox.setEnabled(
            selected_room is not None
            and self.free_placement_radio.isChecked()
        )

    def _handle_uv_map_width_changed(self, value: int) -> None:
        if self._is_syncing_uv_controls:
            return

        selected_room = self._get_selected_uv_room()
        if selected_room is None:
            return

        selected_room.uv_map_width = int(value)
        self._refresh_room_subdivision_layout(selected_room)
        self.uv_canvas.update()
        self._update_unoccupied_uv_pixels_label()
        self._schedule_viewer_preview_refresh()

    def _handle_uv_map_height_changed(self, value: int) -> None:
        if self._is_syncing_uv_controls:
            return

        selected_room = self._get_selected_uv_room()
        if selected_room is None:
            return

        selected_room.uv_map_height = int(value)
        self._refresh_room_subdivision_layout(selected_room)
        self.uv_canvas.update()
        self._update_unoccupied_uv_pixels_label()
        self._schedule_viewer_preview_refresh()

    def _handle_optimize_uv_clicked(self) -> None:
        selected_room = self._get_selected_uv_room()
        if selected_room is None:
            return

        optimized_result = self._optimize_uv_room(selected_room)
        if not optimized_result.wall_uv_rotations:
            QMessageBox.warning(
                self,
                "Optimization skipped",
                "No layout can show every wall with the current map size.",
            )
            return

        self._apply_uv_optimization_result(selected_room, optimized_result)
        self.uv_canvas.update()
        self._sync_uv_controls()
        self._schedule_viewer_preview_refresh()

    def _handle_optimize_all_uv_clicked(self) -> None:
        skipped_room_names: list[str] = []
        optimized_count = 0
        for room in self.current_level.rooms:
            optimized_result = self._optimize_uv_room(room)
            if not optimized_result.wall_uv_rotations:
                skipped_room_names.append(room.name or "Room")
                continue

            self._apply_uv_optimization_result(room, optimized_result)
            optimized_count += 1

        if optimized_count == 0:
            QMessageBox.warning(
                self,
                "Optimization skipped",
                "No layout can show every wall for any room with the current map sizes.",
            )
            return

        self.uv_canvas.update()
        self._sync_uv_controls()
        self._schedule_viewer_preview_refresh()
        if skipped_room_names:
            QMessageBox.warning(
                self,
                "Some rooms skipped",
                "No layout can show every wall for: "
                + ", ".join(skipped_room_names),
            )

    def _handle_reset_uv_defaults_clicked(self) -> None:
        selected_room = self._get_selected_uv_room()
        if selected_room is None:
            return

        selected_room.uv_map_width = DEFAULT_UV_MAP_WIDTH
        selected_room.uv_map_height = DEFAULT_UV_MAP_HEIGHT
        selected_room.wall_uv_scales.clear()
        selected_room.wall_uv_rotations.clear()
        selected_room.wall_uv_positions.clear()
        selected_room.wall_subdivisions.clear()
        selected_room.wall_subdivision_positions.clear()
        selected_room.wall_subdivision_source_ranges.clear()
        self.uv_canvas.update()
        self._sync_uv_controls()
        self._schedule_viewer_preview_refresh()

    def _optimize_uv_room(self, room: RoomData) -> UvOptimizationResult:
        return optimize_room_wall_uvs(
            room=room,
            vertex_data=self.current_level.vertex_data,
            wall_height_meters=room.height_meters,
            use_complex_optimization=(
                self.free_placement_radio.isChecked()
            ),
            use_subdivision_optimization=(
                self.subdivision_optimization_radio.isChecked()
            ),
            complex_optimization_passes=(
                self.complex_optimization_passes_spinbox.value()
            ),
        )

    @staticmethod
    def _apply_uv_optimization_result(
        selected_room: RoomData,
        optimized_result: UvOptimizationResult,
    ) -> None:
        selected_room.wall_uv_rotations = dict(optimized_result.wall_uv_rotations)
        selected_room.wall_uv_scales = dict(optimized_result.wall_uv_scales)
        selected_room.wall_uv_positions = dict(optimized_result.wall_uv_positions)
        selected_room.wall_subdivisions = dict(optimized_result.wall_subdivisions)
        selected_room.wall_subdivision_positions = dict(
            optimized_result.wall_subdivision_positions
        )
        selected_room.wall_subdivision_source_ranges = dict(
            optimized_result.wall_subdivision_source_ranges
        )

    def _refresh_room_subdivision_layout(self, room: RoomData) -> None:
        optimized_result = rebuild_room_subdivision_uvs(
            room=room,
            vertex_data=self.current_level.vertex_data,
            wall_height_meters=room.height_meters,
        )
        if optimized_result is None:
            return

        self._apply_uv_optimization_result(room, optimized_result)

    def _handle_uv_wall_selected(self, wall_key: str) -> None:
        if self.uv_canvas.get_selected_wall_key() != wall_key:
            self.uv_canvas.set_selected_wall_key(wall_key)

        selected_room_index = self._get_selected_uv_room_index()
        if selected_room_index is not None:
            self.texture_creator_level_index = self.current_level_index
            self.texture_creator_room_index = selected_room_index
            self.texture_creator_wall_key = wall_key
            self._sync_texture_creator_tab()
        self._sync_uv_controls()

    def _handle_uv_values_changed(self) -> None:
        self._sync_uv_controls()
        self._schedule_viewer_preview_refresh()

    def _handle_uv_wall_scale_changed(self, value: float) -> None:
        if self._is_syncing_uv_controls:
            return

        selected_room = self._get_selected_uv_room()
        selected_wall_key = self.uv_canvas.get_selected_wall_key()
        if selected_room is None or selected_wall_key is None:
            return

        selected_room.wall_uv_scales[selected_wall_key] = float(value)
        self._refresh_room_subdivision_layout(selected_room)
        self.uv_canvas.update()
        self._update_unoccupied_uv_pixels_label()
        self._schedule_viewer_preview_refresh()

    def _handle_uv_wall_rotation_changed(self, value: int) -> None:
        if self._is_syncing_uv_controls:
            return

        selected_room = self._get_selected_uv_room()
        selected_wall_key = self.uv_canvas.get_selected_wall_key()
        if selected_room is None or selected_wall_key is None:
            return

        selected_room.wall_uv_rotations[selected_wall_key] = _normalize_degree_value(
            value
        )
        self._refresh_room_subdivision_layout(selected_room)
        self.uv_canvas.update()
        self._update_unoccupied_uv_pixels_label()
        self._schedule_viewer_preview_refresh()

    def _apply_loaded_project(self, project_data: ProjectData) -> None:
        self._apply_project_state(
            levels=project_data.levels,
            current_level_index=project_data.current_level_index,
            image_library_paths=project_data.image_library_paths,
            doorway_presets=project_data.doorway_presets,
            generation=project_data.generation,
            surface_texture_generation=(
                project_data.surface_texture_generation
            ),
            texture_atlases=project_data.texture_atlases,
            stairs=project_data.stairs,
        )

    def _apply_project_state(
        self,
        levels: list[LevelData],
        current_level_index: int,
        image_library_paths: list[str] | None = None,
        doorway_presets: list[DoorwayPreset] | None = None,
        generation: GenerationData | None = None,
        surface_texture_generation: SurfaceTextureData | None = None,
        texture_atlases: TextureAtlasData | None = None,
        stairs: list[StairData] | None = None,
    ) -> None:
        if (
            self.generation.is_generating
            or self.surface_texture_generation.is_generating
        ):
            raise RuntimeError(
                "Wait for the current generation request to finish before "
                "loading another project."
            )

        self._is_doorway_dimension_drag_active = False
        self._cancel_pending_doorway_mesh_update(clear_outline=True)
        self.canvas.cancel_stair_placement()
        self._canvas_window_undo_ids.clear()
        self.viewer.set_window_undo_available(False)
        self.levels = levels
        self._reset_viewer_doorway_snapshots()
        self._level_blueprint_image_revisions.clear()
        self.stairs = list(stairs or [])
        self.image_library_paths = self._normalize_image_library_paths(
            image_library_paths or []
        )
        if doorway_presets is not None:
            self.doorway_presets = (
                list(doorway_presets) or create_default_doorway_presets()
            )
        self.current_level_index = min(max(current_level_index, 0), len(self.levels) - 1)
        self.texture_creator_level_index = None
        self.texture_creator_room_index = None
        self.texture_creator_wall_key = None

        self._refresh_image_thumbnail_list()
        self._refresh_doorway_preset_list(
            selected_index=0 if self.doorway_presets else -1
        )
        self._refresh_levels_list()
        self._update_stair_button_state()
        self.stair_status_label.setText(
            "Stairs: none"
            if not self.stairs
            else f"Stairs: {len(self.stairs)} loaded."
        )
        self._sync_level_controls()
        self._sync_canvas_to_current_level()
        self._refresh_room_dependent_controls()
        self._atlas_generation_signature = None
        self._atlas_source_content_paths = None
        self._atlas_source_content_revisions = None
        self._atlas_pending_source_content_refresh_ids.clear()
        self._atlas_wall_texture_source_ids.clear()
        self._clear_atlas_object_preview()
        self.generation.set_data(generation)
        self.texture_atlas_workspace.set_data(texture_atlases)
        self.surface_texture_generation.set_levels(self.levels)
        self.surface_texture_generation.set_data(
            surface_texture_generation
        )
        self._atlas_generation_signature = None
        self._atlas_source_content_paths = None
        self._atlas_source_content_revisions = None
        self._atlas_pending_source_content_refresh_ids.clear()
        self._atlas_wall_texture_source_ids.clear()
        self._sync_atlas_object_texture_sources()
        self.texture_atlas_workspace.materialize_missing_atlases()
        self._schedule_viewer_preview_refresh()

    def _set_current_level_image(self, file_path: str) -> None:
        normalized_path = str(Path(file_path).resolve())
        self.canvas.load_blueprint(
            file_path=normalized_path,
            vertex_data=self.current_level.vertex_data,
            rooms=self.current_level.rooms,
            doorways=self.current_level.doorways,
            floor_contour_vertex_ids=(
                self.current_level.floor_contour_vertex_ids
            ),
        )
        self.current_level.image_path = normalized_path
        self.current_level.image_size_pixels = self.canvas.get_image_size_pixels()
        self.canvas.set_stair_context(self.stairs, self.current_level)
        self.workspace_tabs.setCurrentWidget(self.canvas_viewer_workspace)
        self._update_blueprint_name_label()
        self._schedule_viewer_preview_refresh()

    def _sync_canvas_to_current_level(self) -> None:
        self._viewer_doorways_by_level_index.setdefault(
            self.current_level.index,
            self._copy_doorways(self.current_level.doorways),
        )
        self.canvas.set_level_data(
            vertex_data=self.current_level.vertex_data,
            rooms=self.current_level.rooms,
            doorways=self.current_level.doorways,
            floor_contour_vertex_ids=(
                self.current_level.floor_contour_vertex_ids
            ),
            image_path=self.current_level.image_path,
        )
        if self.canvas.blueprint_image is not None:
            self.current_level.image_size_pixels = self.canvas.get_image_size_pixels()
        self.canvas.set_stair_context(self.stairs, self.current_level)
        self._update_blueprint_name_label()

    def _update_blueprint_name_label(self) -> None:
        image_path = self.current_level.image_path
        if image_path is None:
            label_text = "Image: none for this level"
        elif self.canvas.blueprint_image is None:
            label_text = f"Image missing: {image_path}"
        else:
            label_text = f"Image: {Path(image_path).name}"

        self.blueprint_name_label.setText(label_text)
        self._sync_images_tab()

    def _update_floor_contour_status_label(self) -> None:
        vertex_count = len(self.current_level.floor_contour_vertex_ids)
        if vertex_count == 0:
            self.floor_contour_status_label.setText("Floor contour: Not set")
            return

        self.floor_contour_status_label.setText(
            f"Floor contour: {vertex_count} vertices"
        )

    def _sync_images_tab(self) -> None:
        if not hasattr(self, "image_preview_label"):
            return

        selected_image_path = self._get_selected_image_library_path()
        self.images_delete_button.setEnabled(selected_image_path is not None)
        self.images_save_png_button.setEnabled(selected_image_path is not None)

        if selected_image_path is None:
            self._image_preview_source_key = None
            self._image_preview_source_pixmap = None
            self._image_preview_scaled_key = None
            self.image_path_label.setText("No image selected")
            self.image_preview_label.setPixmap(QPixmap())
            self.image_preview_label.setText("No image loaded")
            return

        source_key = _build_local_file_revision(selected_image_path)
        if source_key != self._image_preview_source_key:
            if not Path(selected_image_path).is_file():
                self._image_preview_source_key = source_key
                self._image_preview_source_pixmap = None
                self._image_preview_scaled_key = None
            else:
                next_pixmap = _load_image_pixmap(selected_image_path)
                if next_pixmap.isNull():
                    self._image_preview_source_key = None
                    self._image_preview_source_pixmap = None
                    self._image_preview_scaled_key = None
                else:
                    self._image_preview_source_key = source_key
                    self._image_preview_source_pixmap = next_pixmap
                    self._image_preview_scaled_key = None
        preview_pixmap = self._image_preview_source_pixmap
        if preview_pixmap is None or preview_pixmap.isNull():
            self.image_path_label.setText(f"Image missing: {selected_image_path}")
            self.image_preview_label.setPixmap(QPixmap())
            self.image_preview_label.setText("Image missing")
            self.images_save_png_button.setEnabled(False)
            return

        self.image_path_label.setText(f"Image: {Path(selected_image_path).name}")
        target_width = max(320, self.image_preview_label.width() - 16)
        target_height = max(220, self.image_preview_label.height() - 16)
        scaled_key = (*source_key, target_width, target_height)
        if scaled_key == self._image_preview_scaled_key:
            return
        self.image_preview_label.setText("")
        self.image_preview_label.setPixmap(
            preview_pixmap.scaled(
                target_width,
                target_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._image_preview_scaled_key = scaled_key


class MainWindow(QMainWindow):
    def __init__(
        self,
        application_settings: ApplicationSettingsStore | None = None,
    ) -> None:
        super().__init__()
        self.application_settings = (
            application_settings
            if application_settings is not None
            else ApplicationSettingsStore()
        )
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("HouseMaker")
        self.resize(1600, 900)

        self.blueprint_workspace = BlueprintWorkspace(
            application_settings=self.application_settings
        )
        self.setCentralWidget(self.blueprint_workspace)
        self.blueprint_workspace.restore_last_project()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.blueprint_workspace.shutdown()
        super().closeEvent(event)


# ### Numeric helpers ###
def _nearest_power_of_two_value(value: int) -> int:
    clamped_value = min(max(int(value), 64), 8192)
    lower_power = 64
    while lower_power * 2 <= clamped_value:
        lower_power *= 2

    upper_power = min(lower_power * 2, 8192)
    if clamped_value - lower_power <= upper_power - clamped_value:
        return lower_power

    return upper_power


def _step_power_of_two_value(value: int, steps: int) -> int:
    power_value = _nearest_power_of_two_value(value)
    for _ in range(abs(steps)):
        if steps > 0:
            power_value = min(power_value * 2, 8192)
        else:
            power_value = max(power_value // 2, 64)

    return power_value


def _normalize_degree_value(value: int) -> int:
    return int(value) % 360


# ### Text helpers ###
def _format_stair_style_label(style: str) -> str:
    """Return the human-readable name for one persisted stair style."""

    if style == STAIR_STYLE_FLOATING_WITH_RISER:
        return "Floating with riser"
    if style == STAIR_STYLE_FLOATING:
        return "Floating"
    return "Supported"


def _format_level_name(
    levels: list[LevelData],
    level_index: object,
) -> str:
    for level in levels:
        if level.index == level_index:
            return level.display_name
    return f"L{level_index}"


def _get_stair_placement_value(placement: object, name: str) -> object:
    """Read one Canvas stair-payload field without coupling its model type."""

    value = getattr(placement, name, None)
    if value is None:
        raise ValueError(f"Stair placement is missing {name}.")
    return value


def _get_optional_stair_vertex_id(
    placement: object,
    name: str,
) -> int | None:
    """Read an optional Canvas vertex binding from a stair payload."""

    value = getattr(placement, name, None)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Stair placement has an invalid {name}.")
    return int(value)


def _get_stair_intermediate_section_payloads(
    placement: object,
) -> tuple[object, ...]:
    """Return the Canvas route controls while accepting straight stairs."""

    value = getattr(placement, "intermediate_sections", ())
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise ValueError("Stair intermediate sections must be a sequence.")
    try:
        return tuple(value)
    except TypeError as error:
        raise ValueError(
            "Stair intermediate sections must be a sequence."
        ) from error


def _build_stair_section_data(section: object) -> StairSectionData:
    """Convert one Canvas curve-guide payload into persisted stair data."""

    return StairSectionData(
        level_index=int(_get_stair_placement_value(section, "level_index")),
        a_x=float(_get_stair_placement_value(section, "a_x")),
        a_y=float(_get_stair_placement_value(section, "a_y")),
        b_x=float(_get_stair_placement_value(section, "b_x")),
        b_y=float(_get_stair_placement_value(section, "b_y")),
        a_vertex_id=_get_optional_stair_vertex_id(section, "a_vertex_id"),
        b_vertex_id=_get_optional_stair_vertex_id(section, "b_vertex_id"),
    )


def _build_stair_data_from_placement(placement: object) -> StairData:
    """Convert a complete Canvas draft into the persistent stair model."""

    return StairData(
        start_level_index=int(
            _get_stair_placement_value(placement, "start_level_index")
        ),
        end_level_index=int(
            _get_stair_placement_value(placement, "end_level_index")
        ),
        start_a_x=float(_get_stair_placement_value(placement, "start_a_x")),
        start_a_y=float(_get_stair_placement_value(placement, "start_a_y")),
        start_b_x=float(_get_stair_placement_value(placement, "start_b_x")),
        start_b_y=float(_get_stair_placement_value(placement, "start_b_y")),
        end_a_x=float(_get_stair_placement_value(placement, "end_a_x")),
        end_a_y=float(_get_stair_placement_value(placement, "end_a_y")),
        end_b_x=float(_get_stair_placement_value(placement, "end_b_x")),
        end_b_y=float(_get_stair_placement_value(placement, "end_b_y")),
        style=str(_get_stair_placement_value(placement, "style")),
        start_a_vertex_id=_get_optional_stair_vertex_id(
            placement,
            "start_a_vertex_id",
        ),
        start_b_vertex_id=_get_optional_stair_vertex_id(
            placement,
            "start_b_vertex_id",
        ),
        end_a_vertex_id=_get_optional_stair_vertex_id(
            placement,
            "end_a_vertex_id",
        ),
        end_b_vertex_id=_get_optional_stair_vertex_id(
            placement,
            "end_b_vertex_id",
        ),
        intermediate_sections=tuple(
            _build_stair_section_data(section)
            for section in _get_stair_intermediate_section_payloads(placement)
        ),
    )


def _format_doorway_preset_label(doorway_preset: DoorwayPreset) -> str:
    dimension_text = (
        f"{doorway_preset.width_meters:.2f} m × "
        f"{doorway_preset.height_meters:.2f} m"
    )
    if doorway_preset.shape != DOORWAY_SHAPE_ARCH:
        return dimension_text

    arch_amount_percent = round(doorway_preset.arch_amount * 100.0, 1)
    return f"{dimension_text} — Arch {arch_amount_percent:g}%"


def _build_wall_aspect_ratio_text(placement: UvWallPlacement | None) -> str:
    if placement is None:
        return "Aspect ratio: none"

    wall_width, wall_height = _get_logical_wall_size(placement)
    aspect_ratio = float(wall_width) / max(1.0, float(wall_height))
    return f"Aspect ratio: {aspect_ratio:.3f}:1"


def _build_wall_resolution_text(placement: UvWallPlacement | None) -> str:
    if placement is None:
        return "Resolutions: none"

    wall_width, wall_height = _get_logical_wall_size(placement)
    aspect_ratio = float(wall_width) / max(1.0, float(wall_height))
    resolution_lines = ["Suggested resolutions:"]
    for detail_size in TEXTURE_CREATOR_DETAIL_SIZES:
        resolution_width, resolution_height = _calculate_aspect_resolution(
            aspect_ratio=aspect_ratio,
            target_square_size=detail_size,
        )
        resolution_lines.append(
            f"{detail_size} detail: {resolution_width} x {resolution_height}"
        )

    return "\n".join(resolution_lines)


def _calculate_aspect_resolution(
    aspect_ratio: float,
    target_square_size: int,
) -> tuple[int, int]:
    target_area = float(target_square_size * target_square_size)
    safe_aspect_ratio = max(0.01, float(aspect_ratio))
    resolution_width = max(1, int(round((target_area * safe_aspect_ratio) ** 0.5)))
    resolution_height = max(1, int(round((target_area / safe_aspect_ratio) ** 0.5)))
    return resolution_width, resolution_height


def _get_logical_wall_size(placement: UvWallPlacement) -> tuple[float, float]:
    wall_width, wall_height = placement.natural_size
    source_span = max(
        0.001,
        placement.source_end_ratio - placement.source_start_ratio,
    )
    return wall_width / source_span, wall_height


# ### Path helpers ###
def _build_atlas_source_base_path_signature(
    source_paths: tuple[tuple[object, ...], ...] | None,
) -> tuple[tuple[object, object], ...]:
    """Return resolution and base path while ignoring optional PBR paths."""

    if source_paths is None:
        return ()
    return tuple(
        (source_path[0], source_path[1])
        for source_path in source_paths
        if len(source_path) >= 2
    )


def _build_local_file_revision(raw_path: object) -> tuple[object, ...]:
    """Return a cheap replacement-aware revision for one local file."""

    normalized_path = str(raw_path or "").strip()
    if not normalized_path:
        return ("", None, None, None)
    try:
        resolved_path = Path(normalized_path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return (normalized_path, None, None, None)
    try:
        path_stat = resolved_path.stat()
    except OSError:
        return (str(resolved_path), None, None, None)
    if not resolved_path.is_file():
        return (str(resolved_path), None, None, None)
    return (
        str(resolved_path),
        path_stat.st_size,
        path_stat.st_mtime_ns,
        path_stat.st_ctime_ns,
    )


def _local_file_revision_has_file(revision: tuple[object, ...]) -> bool:
    """Return whether a local-file revision represents an existing file."""

    return len(revision) == 4 and all(value is not None for value in revision[1:])


def _load_image_pixmap(image_path: str) -> QPixmap:
    """Decode one library image after its revision cache misses."""

    return QPixmap(image_path)


def _ensure_png_file_suffix(file_path: str) -> str:
    output_path = Path(file_path)
    if output_path.suffix.lower() == ".png":
        return str(output_path)

    return f"{output_path}.png"


# ### Entrypoint helpers ###
def _show_window_on_primary_screen(window: QMainWindow) -> None:
    screen = QApplication.primaryScreen()
    if screen is None:
        window.show()
        return

    available_geometry = screen.availableGeometry()
    window_width = min(window.width(), available_geometry.width())
    window_height = min(window.height(), available_geometry.height())
    if window_width != window.width() or window_height != window.height():
        window.resize(window_width, window_height)

    window_x = available_geometry.x() + max(
        0,
        (available_geometry.width() - window.width()) // 2,
    )
    window_y = available_geometry.y() + max(
        0,
        (available_geometry.height() - window.height()) // 2,
    )
    window.move(window_x, window_y)
    window.show()
    window.raise_()
    window.activateWindow()


# ### Entrypoint ###
def main() -> int:
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    app.setApplicationName("HouseMaker")
    app.setStyle("Fusion")

    window = MainWindow()
    _show_window_on_primary_screen(window)
    return app.exec()


# ### Direct execution ###
if __name__ == "__main__":
    raise SystemExit(main())
