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
    QTimer,
    Qt,
)
from PySide6.QtGui import QKeySequence, QShortcut, QWheelEvent
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
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.atlas_export import apply_texture_atlases_to_export
from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.canvas_openings import (
    CANVAS_OPENING_DOORWAY,
    CanvasOpeningEdit,
    CanvasOpeningReference,
    CanvasOpeningTarget,
    apply_canvas_opening_edit,
    build_canvas_opening_targets,
)
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
from housemaker.pbr_maps import (
    ATLAS_MAP_BASE_COLOR,
    ATLAS_MAP_TYPES,
    PBR_MAP_ROUGHNESS,
)
from housemaker.surface_materials import LEGACY_SURFACE_ROUGHNESS_FACTOR
from housemaker.surface_texture_state import (
    SurfaceTextureAssignment,
    SurfaceTextureData,
)
from housemaker.surface_texture_workspace import (
    SurfaceTextureGenerationWorkspace,
)
from housemaker.surface_geometry import (
    FixedSurface,
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
    import_generated_glb,
)
from housemaker.models import (
    DEFAULT_DOORWAY_ARCH_AMOUNT,
    DEFAULT_FLOOR_THICKNESS_METERS,
    DEFAULT_LEVEL_OFFSET_METERS,
    DEFAULT_LEVEL_SCALE,
    DEFAULT_STAIR_STYLE,
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
    DEFAULT_MESH_EDIT_UPDATE_DELAY_SECONDS,
    SettingsWidget,
    resolve_fullscreen_3d_viewer_screen,
)
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
from housemaker.viewer import (
    NAVIGATION_MODE_FIRST_PERSON,
    GlbViewerWidget,
)

# ### Constants ###
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
SURFACE_ATLAS_ROUGHNESS_BYTE = round(
    LEGACY_SURFACE_ROUGHNESS_FACTOR * 255.0
)

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
        # Opening edits remain live in project data while these separate
        # snapshots control which edits have reached the expensive 3D mesh.
        self._viewer_doorways_by_level_index: dict[
            int,
            tuple[DoorwayData, ...],
        ] = {}
        self._viewer_windows_by_level_index: dict[
            int,
            tuple[WindowData, ...],
        ] = {}
        self._reset_viewer_doorway_snapshots()
        self._doorway_mesh_update_delay_seconds = (
            DEFAULT_MESH_EDIT_UPDATE_DELAY_SECONDS
        )
        self._is_doorway_move_drag_active = False
        self._is_canvas_opening_drag_active = False
        self._active_canvas_opening_reference: (
            CanvasOpeningReference | None
        ) = None
        self._active_canvas_opening_start_edit: CanvasOpeningEdit | None = None
        self._canvas_opening_targets_by_key: dict[
            str,
            CanvasOpeningTarget,
        ] = {}
        self._pending_canvas_opening_key: str | None = None
        self._staged_canvas_opening_mesh_update = False
        self._staged_doorway_mesh_update = False
        self._pending_doorway_mesh_level_index: int | None = None
        self._pending_window_mesh_level_index: int | None = None
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
        self._atlas_available_source_ids: set[str] = set()
        self._is_automatically_assigning_atlas_textures = False
        self._last_automatic_atlas_assignment_key: tuple[object, ...] | None = (
            None
        )
        self._atlas_preview_variant_key: tuple[object, ...] | None = None
        self._level_blueprint_image_revisions: dict[
            int,
            tuple[object, ...],
        ] = {}
        self._canvas_window_undo_ids: list[str] = []
        self._object_placement_dialog: ObjectPlacementDialog | None = None
        self._object_placement_operation_id: str | None = None
        self._is_shutdown = False
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
        self.texture_atlas_workspace.selected_atlas_changed.connect(
            self._handle_selected_atlas_changed_for_automatic_assignment
        )
        generation_settings = self.settings_widget.get_settings()
        self._set_doorway_mesh_update_delay_seconds(
            generation_settings.mesh_edit_update_delay_seconds
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
        self.viewer.window_placement_requested.connect(
            self._handle_canvas_window_placement_requested
        )
        self.viewer.window_undo_requested.connect(
            self._handle_canvas_window_undo_requested
        )
        self.viewer.canvas_opening_selection_changed.connect(
            self._handle_canvas_opening_selection_changed
        )
        self.viewer.canvas_opening_edit_started.connect(
            self._handle_canvas_opening_edit_started
        )
        self.viewer.canvas_opening_edit_preview_changed.connect(
            self._handle_canvas_opening_edit_preview_changed
        )
        self.viewer.canvas_opening_edit_finished.connect(
            self._handle_canvas_opening_edit_finished
        )
        self.viewer.canvas_opening_edit_cancelled.connect(
            self._handle_canvas_opening_edit_cancelled
        )
        self.viewer.placed_object_transform_changed.connect(
            self._handle_placed_object_transform_changed
        )
        self.viewer.placed_object_removal_requested.connect(
            self._handle_placed_object_removal_requested
        )
        self.viewer.navigation_mode_changed.connect(
            self._handle_canvas_3d_navigation_mode_changed
        )
        self.viewer.first_person_camera_pose_changed.connect(
            self.canvas.set_camera_indicator_pose
        )
        self.canvas.set_camera_indicator_pose(
            self.viewer.get_first_person_camera_pose()
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

        self.workspace_splitter.addWidget(self.side_panel)
        self.workspace_splitter.setStretchFactor(0, 9)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes([1160, 440])

        self.canvas.rooms_changed.connect(self._schedule_viewer_preview_refresh)
        self.canvas.floor_contour_changed.connect(
            self._handle_floor_contour_changed
        )
        self.canvas.doorways_changed.connect(self._handle_doorways_changed)
        self.canvas.doorway_dimension_preview_changed.connect(
            self._handle_doorway_dimension_preview_changed
        )
        self.canvas.doorway_move_drag_started.connect(
            self._handle_doorway_move_drag_started
        )
        self.canvas.doorway_move_drag_finished.connect(
            self._handle_doorway_move_drag_finished
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
        self._sync_level_controls()
        self._sync_canvas_to_current_level()
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
            self.canvas.set_camera_indicator_pose(
                self.viewer.get_first_person_camera_pose()
            )
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
        added_window = self._find_canvas_window(window.window_id)
        if added_window is None:
            self._rollback_canvas_window(window.window_id)
            self._reset_viewer_window_snapshots()
            self.canvas.update()
            self.viewer.set_window_tools_status(
                "Window not added because its owning level could not be found."
            )
            return
        window_level, _, _ = added_window
        self._sync_viewer_window_snapshot(window_level)
        self._refresh_canvas_windows_for_level(window_level)

        try:
            validated_build = self._build_model_with_stable_dependencies(
                lambda: self._build_viewer_preview_model(None)
            )
        except Exception as error:
            self._rollback_canvas_window(window.window_id)
            self._sync_viewer_window_snapshot(window_level)
            self._refresh_canvas_windows_for_level(window_level)
            self.viewer.set_window_tools_status(f"Window not added: {error}")
            return
        if validated_build is None:
            self._rollback_canvas_window(window.window_id)
            self._sync_viewer_window_snapshot(window_level)
            self._refresh_canvas_windows_for_level(window_level)
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
            self._sync_viewer_window_snapshot(window_level)
            self._refresh_canvas_windows_for_level(window_level)
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
        self._sync_viewer_window_snapshot(level)
        self._refresh_canvas_windows_for_level(level)

        try:
            validated_build = self._build_model_with_stable_dependencies(
                lambda: self._build_viewer_preview_model(None)
            )
        except Exception as error:
            level.windows.insert(window_index, window)
            self._sync_viewer_window_snapshot(level)
            self._refresh_canvas_windows_for_level(level)
            self._sync_canvas_window_undo_availability()
            self.viewer.set_window_tools_status(
                f"Window could not be undone: {error}"
            )
            return
        if validated_build is None:
            level.windows.insert(window_index, window)
            self._sync_viewer_window_snapshot(level)
            self._refresh_canvas_windows_for_level(level)
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
            self._sync_viewer_window_snapshot(level)
            self._refresh_canvas_windows_for_level(level)
            self._restore_canvas_window_preview_after_rollback()
            self._sync_canvas_window_undo_availability()
            self.viewer.set_window_tools_status(
                f"Window could not be undone: {error}"
            )
            return

        self._canvas_window_undo_ids.pop()
        self._sync_canvas_window_undo_availability()
        self.viewer.set_window_tools_status("Window undone.")

    # ### Canvas opening gizmo edits ###
    def _handle_canvas_opening_selection_changed(
        self,
        raw_reference: object,
    ) -> None:
        """Keep doorway controls aligned with selection in the 3D viewer."""

        reference = (
            raw_reference
            if isinstance(raw_reference, CanvasOpeningReference)
            else None
        )
        doorway_index: int | None = None
        if (
            reference is not None
            and reference.kind == CANVAS_OPENING_DOORWAY
            and reference.level_index == self.current_level.index
        ):
            doorway_index = reference.item_index
        self.canvas._set_selected_doorway_index(doorway_index)
        self.canvas.update()

    def _handle_canvas_opening_edit_started(
        self,
        raw_edit: object,
    ) -> None:
        """Freeze the committed wall mesh for the duration of one drag."""

        if not isinstance(raw_edit, CanvasOpeningEdit):
            return
        pending_key = self._pending_canvas_opening_key
        if pending_key is not None and pending_key != raw_edit.reference.key:
            self._stage_pending_canvas_opening_snapshots()

        self._is_canvas_opening_drag_active = True
        self._active_canvas_opening_reference = raw_edit.reference
        self._active_canvas_opening_start_edit = raw_edit
        self._doorway_mesh_update_timer.stop()

    def _handle_canvas_opening_edit_preview_changed(
        self,
        raw_edit: object,
    ) -> None:
        """Apply a lightweight opening rectangle while retaining the old mesh."""

        if not isinstance(raw_edit, CanvasOpeningEdit):
            return
        target = self._canvas_opening_targets_by_key.get(
            raw_edit.reference.key
        )
        if target is None:
            self.viewer.set_window_tools_status(
                "Opening edit stopped because its wall is no longer available."
            )
            self.viewer.select_canvas_opening(None)
            return

        try:
            applied = apply_canvas_opening_edit(
                self.levels,
                target,
                raw_edit,
            )
        except (TypeError, ValueError) as error:
            self.viewer.set_window_tools_status(
                f"Opening could not be resized: {error}"
            )
            return

        self._canvas_opening_targets_by_key[target.key] = target.with_bounds(
            raw_edit.bounds
        )
        self._sync_live_canvas_opening(applied.reference, applied.level)
        self._refresh_pending_canvas_opening_state(applied.reference)
        if self._is_canvas_opening_drag_active:
            self._doorway_mesh_update_timer.stop()

    def _handle_canvas_opening_edit_finished(
        self,
        raw_edit: object,
        _changed: bool,
    ) -> None:
        """Start the complete configured delay only after mouse release."""

        if not isinstance(raw_edit, CanvasOpeningEdit):
            return
        if (
            self._active_canvas_opening_reference is not None
            and raw_edit.reference != self._active_canvas_opening_reference
        ):
            return
        self._finish_canvas_opening_drag()

    def _handle_canvas_opening_edit_cancelled(
        self,
        raw_start_edit: object,
    ) -> None:
        """Restore the exact drag-start rectangle after viewer cancellation."""

        start_edit = (
            raw_start_edit
            if isinstance(raw_start_edit, CanvasOpeningEdit)
            else self._active_canvas_opening_start_edit
        )
        if start_edit is not None:
            self._handle_canvas_opening_edit_preview_changed(start_edit)
        self._finish_canvas_opening_drag()

    def _finish_canvas_opening_drag(self) -> None:
        """End pointer ownership and resume a pending opening debounce."""

        self._is_canvas_opening_drag_active = False
        self._active_canvas_opening_reference = None
        self._active_canvas_opening_start_edit = None
        if (
            self._staged_canvas_opening_mesh_update
            or self._pending_doorway_mesh_level_index is not None
            or self._pending_window_mesh_level_index is not None
        ):
            self._doorway_mesh_update_timer.start()

    def _sync_live_canvas_opening(
        self,
        reference: CanvasOpeningReference,
        level: LevelData,
    ) -> None:
        """Repaint the active 2D Canvas from the same edited project object."""

        if level.index != self.current_level.index:
            return
        if reference.kind == CANVAS_OPENING_DOORWAY:
            self.canvas.doorways = level.doorways
            self.canvas._set_selected_doorway_index(reference.item_index)
        else:
            self.canvas.windows = level.windows
        self.canvas.update()

    def _refresh_pending_canvas_opening_state(
        self,
        reference: CanvasOpeningReference,
    ) -> None:
        """Compare live data with the rendered snapshot without rebuilding."""

        level = next(
            (
                candidate
                for candidate in self.levels
                if candidate.index == reference.level_index
            ),
            None,
        )
        if level is None:
            return
        if reference.kind == CANVAS_OPENING_DOORWAY:
            committed = self._viewer_doorways_by_level_index.get(level.index)
            is_pending = self._copy_doorways(level.doorways) != committed
            self._pending_doorway_mesh_level_index = (
                level.index if is_pending else None
            )
        else:
            committed = self._viewer_windows_by_level_index.get(level.index)
            is_pending = self._copy_windows(level.windows) != committed
            self._pending_window_mesh_level_index = (
                level.index if is_pending else None
            )
        if is_pending:
            self._pending_canvas_opening_key = reference.key
        elif self._pending_canvas_opening_key == reference.key:
            self._pending_canvas_opening_key = None

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
        self._set_canvas_viewer_targets(wall_targets)
        self.viewer.set_model(generated_model, preserve_camera=True)
        self._mark_viewer_preview_dirty(preserve_camera=True)
        if self._remember_current_canvas_preview_model(
            generated_model,
            validated_dependency_signature=dependency_signature,
        ):
            return True
        self._queue_viewer_preview_refresh()
        return False

    def _set_canvas_viewer_targets(
        self,
        surfaces: Sequence[FixedSurface],
    ) -> None:
        """Install wall targets and their explicit selectable hole overlays."""

        wall_targets = tuple(
            surface
            for surface in surfaces
            if isinstance(surface, FixedSurface)
        )
        self.viewer.set_wall_targets(wall_targets)
        try:
            opening_targets = build_canvas_opening_targets(
                self.levels,
                wall_targets,
            )
        except (TypeError, ValueError):
            opening_targets = ()
        drag_start = self._active_canvas_opening_start_edit
        if drag_start is not None:
            opening_targets = tuple(
                (
                    target.with_bounds(drag_start.bounds)
                    if target.key == drag_start.reference.key
                    else target
                )
                for target in opening_targets
            )
        self._canvas_opening_targets_by_key = {
            target.key: target for target in opening_targets
        }
        self.viewer.set_canvas_opening_targets(opening_targets)

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

    def _refresh_canvas_windows_for_level(self, level: LevelData) -> None:
        """Repaint 2D windows after one structural transaction."""

        if not 0 <= self.current_level_index < len(self.levels):
            return
        if level.index != self.levels[self.current_level_index].index:
            return
        self.canvas.windows = level.windows
        self.canvas.update()

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

    def load_blueprint(self, file_path: str) -> None:
        self._set_current_level_image(file_path)

    def _handle_glb_export_clicked(self) -> None:
        self._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )
        if self._show_unpacked_scene_texture_export_error():
            return

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
        self._set_canvas_viewer_targets(
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
        self._ensure_viewer_preview_current(preserve_camera=False)

    def _handle_generation_data_changed_for_atlases(
        self,
        _generation_data: object,
    ) -> None:
        """Refresh Atlas object choices after generation, deletion, or selection."""

        if self._is_automatically_assigning_atlas_textures:
            self._atlas_generation_signature = None
            return
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
        """Move or remove a completed object in the Canvas preview."""

        if isinstance(raw_record, GeneratedObjectRecord):
            self._schedule_viewer_preview_refresh(preserve_camera=True)

    def _handle_placed_object_removal_requested(self, object_id: str) -> None:
        """Remove a Canvas placement and unassign its texture from Atlases."""

        normalized_object_id = str(object_id).strip()
        if self.generation.remove_generated_object_placement(
            normalized_object_id
        ):
            self.texture_atlas_workspace.remove_scene_texture_from_atlases(
                normalized_object_id
            )
            return
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

    # ### Texture Atlas synchronization ###
    def _handle_surface_texture_data_changed_for_atlases(
        self,
        _surface_texture_data: object,
    ) -> None:
        """Refresh Atlas surface choices without reloading unchanged thumbnails."""

        if self._is_automatically_assigning_atlas_textures:
            self._atlas_generation_signature = None
            return
        self._sync_atlas_object_texture_sources()

    def _handle_selected_atlas_changed_for_automatic_assignment(
        self,
        _selected_atlas: object,
    ) -> None:
        """Retry pending scene textures when the user chooses an Atlas."""

        self._refresh_scene_atlas_texture_requirements()

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

    def _refresh_scene_atlas_texture_requirements(
        self,
        *,
        automatically_assign: bool = True,
    ) -> None:
        """Mark exported scene textures and optionally pack missing ones."""

        required_source_ids = self._build_required_scene_atlas_source_ids()
        self.texture_atlas_workspace.set_scene_texture_source_ids(
            required_source_ids
        )
        if (
            not automatically_assign
            or self._is_automatically_assigning_atlas_textures
        ):
            return
        self._automatically_assign_scene_textures()

    def _build_required_scene_atlas_source_ids(self) -> tuple[str, ...]:
        """Return available texture sources used by included scene content."""

        included_level_indices = {
            level.index for level in self.levels if level.include_in_export
        }
        required_ids: list[str] = []
        for object_id in self.generation.get_generated_object_ids():
            placement = self.generation.get_generated_object_placement(
                object_id
            )
            if (
                placement is not None
                and placement.level_index in included_level_indices
                and object_id in self._atlas_available_source_ids
            ):
                required_ids.append(object_id)
        exported_surface_ids = {
            surface.surface_id for surface in build_fixed_surfaces(self.levels)
        }
        required_ids.extend(
            source_id
            for surface_id, source_id in (
                self._build_atlas_surface_source_ids().items()
            )
            if (
                surface_id in exported_surface_ids
                and source_id in self._atlas_available_source_ids
            )
        )
        return tuple(dict.fromkeys(required_ids))

    def _automatically_assign_scene_textures(self) -> None:
        """Pack all currently unassigned scene textures in one Atlas update."""

        if self._is_automatically_assigning_atlas_textures:
            return
        settings = self.settings_widget.get_settings()
        target_resolution = settings.automatic_atlas_texture_resolution
        sort_by_pbr = settings.automatic_atlas_texture_sort_by_pbr
        attempt_key = self._build_automatic_atlas_assignment_key(
            target_resolution,
            sort_by_pbr,
        )
        if attempt_key == self._last_automatic_atlas_assignment_key:
            return
        self._last_automatic_atlas_assignment_key = attempt_key
        self._is_automatically_assigning_atlas_textures = True
        try:
            assigned_source_ids = (
                self.texture_atlas_workspace.auto_assign_scene_texture_sources(
                    target_resolution,
                    commit_callback=lambda source_ids: (
                        self._commit_automatic_atlas_texture_resolutions(
                            source_ids,
                            target_resolution,
                        )
                    ),
                    sort_by_pbr=sort_by_pbr,
                )
            )
            if not assigned_source_ids:
                return
            self._atlas_generation_signature = None
            self._sync_atlas_object_texture_sources()
            for source_id in assigned_source_ids:
                if not is_atlas_wall_texture_source_id(source_id):
                    self._refresh_placed_object_texture_if_needed(source_id)
        finally:
            self._is_automatically_assigning_atlas_textures = False
            self._last_automatic_atlas_assignment_key = (
                self._build_automatic_atlas_assignment_key(
                    target_resolution,
                    sort_by_pbr,
                )
            )

    def _build_automatic_atlas_assignment_key(
        self,
        target_resolution: int,
        sort_by_pbr: bool,
    ) -> tuple[object, ...]:
        """Describe inputs whose changes make a failed auto-pack worth retrying."""

        atlas_data = self.texture_atlas_workspace.get_data()
        atlas_signature = tuple(
            (
                atlas.atlas_id,
                atlas.name,
                atlas.resolution,
                tuple(
                    (
                        placement.object_id,
                        placement.texture_resolution,
                        placement.x,
                        placement.y,
                        placement.size,
                        placement.packing_mode,
                        placement.slot_half,
                        placement.slot_quadrant,
                    )
                    for placement in atlas.placements
                ),
            )
            for atlas in atlas_data.atlases
        )
        return (
            atlas_data.selected_atlas_id,
            int(target_resolution),
            bool(sort_by_pbr),
            self.texture_atlas_workspace.get_unpacked_scene_texture_source_ids(),
            atlas_signature,
            self._atlas_generation_signature,
        )

    def _commit_automatic_atlas_texture_resolutions(
        self,
        source_ids: tuple[str, ...],
        target_resolution: int,
    ) -> bool:
        """Select exact owning-workspace variants after Atlas materialization."""

        applied_changes: list[tuple[str, str, int]] = []
        for source_id in source_ids:
            assignment_id = get_atlas_wall_texture_assignment_id(source_id)
            if assignment_id is not None:
                assignment = self.surface_texture_generation.get_assignment(
                    assignment_id
                )
                if assignment is None:
                    self._rollback_automatic_atlas_texture_resolutions(
                        applied_changes
                    )
                    return False
                if not assignment.texture_variants:
                    continue
                previous_resolution = assignment.selected_texture_resolution
                if previous_resolution is None:
                    self._rollback_automatic_atlas_texture_resolutions(
                        applied_changes
                    )
                    return False
                if previous_resolution == target_resolution:
                    continue
                if not self.surface_texture_generation.select_assignment_texture_resolution(
                    assignment_id,
                    target_resolution,
                ):
                    self._rollback_automatic_atlas_texture_resolutions(
                        applied_changes
                    )
                    return False
                applied_changes.append(
                    ("surface", assignment_id, previous_resolution)
                )
                continue

            active_variant = self.generation.get_active_texture_variant(
                source_id
            )
            if active_variant is None:
                self._rollback_automatic_atlas_texture_resolutions(
                    applied_changes
                )
                return False
            previous_resolution = active_variant.resolution
            if previous_resolution == target_resolution:
                continue
            if not self.generation.select_object_texture_resolution(
                source_id,
                target_resolution,
            ):
                self._rollback_automatic_atlas_texture_resolutions(
                    applied_changes
                )
                return False
            applied_changes.append(
                ("object", source_id, previous_resolution)
            )
        return True

    def _rollback_automatic_atlas_texture_resolutions(
        self,
        applied_changes: list[tuple[str, str, int]],
    ) -> None:
        """Best-effort restore owning workspaces after a rejected batch."""

        for source_kind, source_id, previous_resolution in reversed(
            applied_changes
        ):
            if source_kind == "surface":
                self.surface_texture_generation.select_assignment_texture_resolution(
                    source_id,
                    previous_resolution,
                )
            else:
                self.generation.select_object_texture_resolution(
                    source_id,
                    previous_resolution,
                )

    def _show_unpacked_scene_texture_export_error(self) -> bool:
        """Block GLB export while any required source remains outside Atlases."""

        unpacked_source_ids = (
            self.texture_atlas_workspace.get_unpacked_scene_texture_source_ids()
        )
        if not unpacked_source_ids:
            return False
        display_names = [
            self._atlas_texture_source_display_name(source_id)
            for source_id in unpacked_source_ids
        ]
        visible_names = display_names[:10]
        remaining_count = len(display_names) - len(visible_names)
        detail_lines = [f"- {name}" for name in visible_names]
        if remaining_count:
            detail_lines.append(f"- and {remaining_count} more")
        QMessageBox.warning(
            self,
            "Export blocked",
            "Every texture used by the exported scene must be assigned to "
            "an Atlas before GLB export. The unassigned textures are shown "
            "in red in the Atlas tab:\n\n"
            + "\n".join(detail_lines),
        )
        return True

    def _atlas_texture_source_display_name(self, source_id: str) -> str:
        """Resolve a human-readable name for one export-blocking source."""

        assignment_id = get_atlas_wall_texture_assignment_id(source_id)
        if assignment_id is not None:
            assignment = self.surface_texture_generation.get_assignment(
                assignment_id
            )
            if assignment is not None:
                return assignment.display_name or (
                    f"{assignment.surface_type.title()} texture"
                )
        for record in self.generation.get_data().generated_objects:
            if record.object_id == source_id:
                return record.object_name
        return source_id

    def _sync_atlas_object_texture_sources(
        self,
        *,
        automatically_assign_scene_textures: bool = True,
    ) -> None:
        """Expose generated object and architectural-surface textures to Atlas."""

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

        surface_assignments = list(
            self.surface_texture_generation.get_assignments()
        )
        for assignment in surface_assignments:
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
                physical_map_paths = (
                    self.surface_texture_generation
                    .get_assignment_map_asset_paths(
                        assignment.assignment_id,
                        resolution,
                    )
                )
                physical_path = physical_map_paths.get(
                    ATLAS_MAP_BASE_COLOR
                )
                logical_map_paths = (
                    {ATLAS_MAP_BASE_COLOR: logical_path}
                    if texture_variant is None
                    else texture_variant.map_asset_paths
                )
                map_signature = tuple(
                    (
                        map_type,
                        map_asset_path,
                        _build_local_file_revision(
                            physical_map_paths.get(map_type)
                        ),
                    )
                    for map_type, map_asset_path in (
                        logical_map_paths.items()
                    )
                )
                variant_signature.append(
                    (
                        resolution,
                        str(logical_path),
                        _build_local_file_revision(physical_path),
                        map_signature,
                    )
                )
            signature_items.append(
                (
                    "surface",
                    assignment.assignment_id,
                    assignment.asset_path,
                    assignment.selected_texture_resolution,
                    assignment.texture_width,
                    assignment.texture_height,
                    assignment.surface_ids,
                    tuple(variant_signature),
                )
            )
            surface_source_id = build_atlas_wall_texture_source_id(
                assignment.assignment_id
            )
            if surface_source_id not in generated_object_id_lookup:
                source_content_paths[surface_source_id] = tuple(
                    (
                        item[0],
                        item[1],
                        tuple(
                            (map_item[0], map_item[1])
                            for map_item in item[3]
                        ),
                    )
                    for item in variant_signature
                )
                source_content_revisions[surface_source_id] = tuple(
                    (
                        item[0],
                        item[2],
                        tuple(
                            (map_item[0], map_item[2])
                            for map_item in item[3]
                        ),
                    )
                    for item in variant_signature
                )
        signature = tuple(signature_items)
        if signature == self._atlas_generation_signature:
            self._refresh_scene_atlas_texture_requirements(
                automatically_assign=automatically_assign_scene_textures
            )
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
        surface_sources: dict[str, AtlasObjectTextureSource] = {}
        surface_assignments_by_source_id: dict[
            str,
            SurfaceTextureAssignment,
        ] = {}
        colliding_surface_texture_count = 0
        for assignment in surface_assignments:
            surface_source_id = build_atlas_wall_texture_source_id(
                assignment.assignment_id
            )
            if surface_source_id in generated_object_id_lookup:
                colliding_surface_texture_count += 1
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
                failed_source_ids.add(surface_source_id)
                continue
            active_sources.append(source)
            available_source_ids.add(surface_source_id)
            surface_sources[source.object_id] = source
            surface_assignments_by_source_id[source.object_id] = assignment

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
            surface_assignment = surface_assignments_by_source_id.get(object_id)
            if surface_assignment is not None:
                return self._build_atlas_wall_texture_source(
                    surface_assignment,
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
            surface_assignment = surface_assignments_by_source_id.get(object_id)
            if surface_assignment is not None:
                return (
                    self.surface_texture_generation
                    .can_select_assignment_texture_resolution(
                        surface_assignment.assignment_id,
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
        # The backing field and source-ID prefix retain their legacy "wall"
        # names so existing project files and integrations remain compatible.
        self._atlas_wall_texture_source_ids.update(surface_sources)
        self._atlas_available_source_ids = set(available_source_ids)
        self._atlas_generation_signature = (
            None if source_build_failed else signature
        )
        self._atlas_source_content_paths = source_content_paths
        self._atlas_source_content_revisions = source_content_revisions
        self._refresh_scene_atlas_texture_requirements(
            automatically_assign=automatically_assign_scene_textures
        )
        self._request_hosted_atlas_object_preview()
        if colliding_surface_texture_count:
            self._append_atlas_preview_status(
                f"Skipped {colliding_surface_texture_count} surface texture source"
                f"{'s' if colliding_surface_texture_count != 1 else ''} because "
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
        """Adapt one generated surface texture and its exact active variant."""

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
        physical_map_paths = (
            self.surface_texture_generation.get_assignment_map_asset_paths(
                assignment.assignment_id,
                requested_resolution if supports_resolution_changes else None,
            )
        )
        physical_path = physical_map_paths.get(ATLAS_MAP_BASE_COLOR)
        if physical_path is None:
            return None
        legacy_logical_map_paths = {
            ATLAS_MAP_BASE_COLOR: assignment.asset_path
        }
        logical_map_paths = legacy_logical_map_paths
        if supports_resolution_changes:
            selected_variant = assignment.texture_variant_for_resolution(
                int(requested_resolution)
            )
            if selected_variant is None:
                return None
            logical_map_paths = selected_variant.map_asset_paths
        live_map_types = tuple(
            map_type
            for map_type in ATLAS_MAP_TYPES
            if map_type in logical_map_paths
            and map_type in physical_map_paths
        )
        atlas_logical_map_paths = {
            map_type: f"surface_textures/{logical_map_paths[map_type]}"
            for map_type in live_map_types
        }
        atlas_physical_map_paths = {
            map_type: physical_map_paths[map_type]
            for map_type in live_map_types
        }
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
                        f"{assignment.surface_type.title()} texture · "
                        f"{surface_count} surface"
                        f"{'s' if surface_count != 1 else ''}"
                    )
                ),
                texture_path=f"surface_textures/{asset_path}",
                texture_resolution=texture_resolution,
                physical_texture_path=physical_path,
                map_texture_paths=atlas_logical_map_paths,
                physical_map_texture_paths=atlas_physical_map_paths,
                fallback_map_rgba={
                    PBR_MAP_ROUGHNESS: (
                        SURFACE_ATLAS_ROUGHNESS_BYTE,
                        SURFACE_ATLAS_ROUGHNESS_BYTE,
                        SURFACE_ATLAS_ROUGHNESS_BYTE,
                        255,
                    )
                },
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

    # ### Debounced opening mesh previews ###
    @staticmethod
    def _copy_doorways(
        doorways: Sequence[DoorwayData],
    ) -> tuple[DoorwayData, ...]:
        """Own a doorway snapshot that cannot follow live Canvas mutations."""

        return tuple(copy.deepcopy(tuple(doorways)))

    @staticmethod
    def _copy_windows(
        windows: Sequence[WindowData],
    ) -> tuple[WindowData, ...]:
        """Own a window snapshot that cannot follow live Canvas mutations."""

        return tuple(copy.deepcopy(tuple(windows)))

    def _reset_viewer_doorway_snapshots(self) -> None:
        """Make every rendered opening snapshot match the loaded project."""

        self._viewer_doorways_by_level_index = {
            level.index: self._copy_doorways(level.doorways)
            for level in self.levels
        }
        self._reset_viewer_window_snapshots()

    def _reset_viewer_window_snapshots(self) -> None:
        """Make every rendered window snapshot match the loaded project."""

        self._viewer_windows_by_level_index = {
            level.index: self._copy_windows(level.windows)
            for level in self.levels
        }

    def _sync_viewer_window_snapshot(self, level: LevelData) -> None:
        """Commit one structural window list change before its model build."""

        self._viewer_windows_by_level_index[level.index] = self._copy_windows(
            level.windows
        )
        if self._pending_window_mesh_level_index != level.index:
            return
        self._pending_window_mesh_level_index = None
        if (
            self._pending_canvas_opening_key is not None
            and self._pending_canvas_opening_key.startswith("window:")
        ):
            self._pending_canvas_opening_key = None
        if (
            self._pending_doorway_mesh_level_index is None
            and not self._staged_canvas_opening_mesh_update
        ):
            self._doorway_mesh_update_timer.stop()

    def _build_viewer_preview_levels(self) -> list[LevelData]:
        """Copy levels while substituting only committed opening dimensions."""

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
            window_snapshot = self._viewer_windows_by_level_index.get(
                level.index
            )
            if window_snapshot is None:
                window_snapshot = self._copy_windows(level.windows)
                self._viewer_windows_by_level_index[level.index] = (
                    window_snapshot
                )
            preview_level = copy.copy(level)
            preview_level.doorways = list(copy.deepcopy(doorway_snapshot))
            preview_level.windows = list(copy.deepcopy(window_snapshot))
            preview_levels.append(preview_level)
        return preview_levels

    def _set_doorway_mesh_update_delay_seconds(
        self,
        delay_seconds: float,
    ) -> None:
        """Apply the setting and restart a live debounce only when it changed."""

        normalized_delay = float(delay_seconds)
        if normalized_delay <= 0.0:
            raise ValueError("Mesh edit update delay must be positive.")
        if normalized_delay == self._doorway_mesh_update_delay_seconds:
            return

        timer_was_active = self._doorway_mesh_update_timer.isActive()
        self._doorway_mesh_update_delay_seconds = normalized_delay
        self._doorway_mesh_update_timer.setInterval(
            max(1, round(normalized_delay * 1000.0))
        )
        if timer_was_active:
            self._doorway_mesh_update_timer.start()

    def _stage_pending_canvas_opening_snapshots(self) -> None:
        """Stage all stable live openings without refreshing during a drag."""

        doorway_level_index = self._pending_doorway_mesh_level_index
        window_level_index = self._pending_window_mesh_level_index
        self._pending_doorway_mesh_level_index = None
        self._pending_window_mesh_level_index = None
        self._pending_canvas_opening_key = None

        for level in self.levels:
            if level.index == doorway_level_index:
                next_doorways = self._copy_doorways(level.doorways)
                if (
                    self._viewer_doorways_by_level_index.get(level.index)
                    != next_doorways
                ):
                    self._viewer_doorways_by_level_index[level.index] = (
                        next_doorways
                    )
                    self._staged_canvas_opening_mesh_update = True
                    self._staged_doorway_mesh_update = True
            if level.index == window_level_index:
                next_windows = self._copy_windows(level.windows)
                if (
                    self._viewer_windows_by_level_index.get(level.index)
                    != next_windows
                ):
                    self._viewer_windows_by_level_index[level.index] = (
                        next_windows
                    )
                    self._staged_canvas_opening_mesh_update = True

    def _cancel_pending_doorway_mesh_update(
        self,
        clear_outline: bool = True,
    ) -> None:
        """Cancel transient mesh-edit work and optionally remove its outline."""

        self._doorway_mesh_update_timer.stop()
        self._pending_doorway_mesh_level_index = None
        self._pending_window_mesh_level_index = None
        self._pending_canvas_opening_key = None
        self._staged_canvas_opening_mesh_update = False
        self._staged_doorway_mesh_update = False
        self._is_canvas_opening_drag_active = False
        self._active_canvas_opening_reference = None
        self._active_canvas_opening_start_edit = None
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
        """Commit the latest stable doorway/window edit to the 3D mesh cache."""

        self._doorway_mesh_update_timer.stop()
        self._stage_pending_canvas_opening_snapshots()
        snapshot_changed = self._staged_canvas_opening_mesh_update
        doorway_snapshot_changed = self._staged_doorway_mesh_update
        self._staged_canvas_opening_mesh_update = False
        self._staged_doorway_mesh_update = False
        if not snapshot_changed:
            self._clear_committed_doorway_outline_if_displayed()
            if self._doorway_outline_commit_revision is None:
                self.viewer.set_doorway_preview_outline(None)
            return
        self._schedule_viewer_preview_refresh(preserve_camera=True)
        if doorway_snapshot_changed:
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
                export_untextured_surfaces=False,
            )
            placed_models = self._build_placed_generated_models()
            generated_model = (
                base_model
                if not placed_models
                else compose_placed_generated_models(
                    base_model,
                    placed_models,
                )
            )
            surface_source_ids = self._build_atlas_surface_source_ids()
            required_source_ids = tuple(
                dict.fromkeys(
                    (
                        *(placement.object_id for placement in placed_models),
                        *surface_source_ids.values(),
                    )
                )
            )
            materialized_atlases = (
                self.texture_atlas_workspace.prepare_export_atlases(
                    required_source_ids
                )
            )
            if not materialized_atlases:
                return generated_model
            return apply_texture_atlases_to_export(
                generated_model,
                materialized_atlases,
                surface_source_ids=surface_source_ids,
            )
        except (OSError, TypeError, ValueError) as error:
            if failure_title is not None:
                QMessageBox.warning(self, failure_title, str(error))
            return None

    def _build_atlas_surface_source_ids(self) -> dict[str, str]:
        """Map each assigned architectural surface to its Atlas source ID."""

        source_ids: dict[str, str] = {}
        generated_object_ids = set(self.generation.get_generated_object_ids())
        for assignment in self.surface_texture_generation.get_assignments():
            source_id = build_atlas_wall_texture_source_id(
                assignment.assignment_id
            )
            if source_id in generated_object_ids:
                continue
            for surface_id in assignment.surface_ids:
                source_ids[surface_id] = source_id
        return source_ids

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
                self._set_canvas_viewer_targets(())
                self.viewer.clear_model()
            else:
                self._set_canvas_viewer_targets(
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

    def _get_image_file_path(self) -> str:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "load image",
            str(Path.home()),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)",
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

    @staticmethod
    def _normalize_image_library_paths(image_paths: list[str]) -> list[str]:
        normalized_paths: list[str] = []
        for image_path in image_paths:
            normalized_path = str(Path(image_path).resolve())
            if normalized_path in normalized_paths:
                continue

            normalized_paths.append(normalized_path)

        return normalized_paths

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

        self._is_doorway_move_drag_active = False
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

    def _handle_doorway_move_drag_started(self) -> None:
        """Hold the old wall mesh while a doorway is moving in the Canvas."""

        self._is_doorway_move_drag_active = True
        self._doorway_mesh_update_timer.stop()

    def _handle_doorway_move_drag_finished(self, changed: bool) -> None:
        """Commit a moved doorway immediately when the mouse is released."""

        self._is_doorway_move_drag_active = False
        if changed:
            if self._pending_doorway_mesh_level_index is None:
                self._handle_doorway_dimension_preview_changed()
            self._commit_pending_doorway_mesh_update()
            return
        if self._pending_doorway_mesh_level_index is not None:
            self._doorway_mesh_update_timer.start()

    def _handle_doorway_dimension_preview_changed(self) -> None:
        """Show a live doorway outline and debounce an arch mesh rebuild."""

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
            doorway_key_prefix = f"doorway:{level.index}:"
            if (
                self._pending_canvas_opening_key is not None
                and self._pending_canvas_opening_key.startswith(
                    doorway_key_prefix
                )
            ):
                self._pending_canvas_opening_key = None
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
        self._pending_canvas_opening_key = (
            f"doorway:{level.index}:{selected_index}"
        )
        if self._is_doorway_move_drag_active:
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
            settings.mesh_edit_update_delay_seconds
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
        self._refresh_scene_atlas_texture_requirements()

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
        self._refresh_scene_atlas_texture_requirements()
        self._schedule_viewer_preview_refresh()

    def _handle_snap_middle_equal_angle_toggled(self, checked: bool) -> None:
        self.canvas.set_snap_middle_equal_angle_only(checked)

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

        self._is_doorway_move_drag_active = False
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
        self._atlas_generation_signature = None
        self._atlas_source_content_paths = None
        self._atlas_source_content_revisions = None
        self._atlas_pending_source_content_refresh_ids.clear()
        self._atlas_wall_texture_source_ids.clear()
        self._atlas_available_source_ids.clear()
        self._last_automatic_atlas_assignment_key = None
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
        self._atlas_available_source_ids.clear()
        self._last_automatic_atlas_assignment_key = None
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
            windows=self.current_level.windows,
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
        self._viewer_windows_by_level_index.setdefault(
            self.current_level.index,
            self._copy_windows(self.current_level.windows),
        )
        self.canvas.set_level_data(
            vertex_data=self.current_level.vertex_data,
            rooms=self.current_level.rooms,
            doorways=self.current_level.doorways,
            windows=self.current_level.windows,
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

    def _update_floor_contour_status_label(self) -> None:
        vertex_count = len(self.current_level.floor_contour_vertex_ids)
        if vertex_count == 0:
            self.floor_contour_status_label.setText("Floor contour: Not set")
            return

        self.floor_contour_status_label.setText(
            f"Floor contour: {vertex_count} vertices"
        )

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
