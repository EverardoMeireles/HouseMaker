# ### Imports ###
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QPointF, QSize, QTimer, Qt
from PySide6.QtGui import QIcon, QKeySequence, QPixmap, QShortcut, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QAbstractSpinBox,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QListView,
    QDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.external_viewer_host import ExternalFullscreenViewerHost
from housemaker.camera_models import (
    DEFAULT_FIRST_PERSON_LIGHT_INTENSITY,
    InitialFirstPersonCamera,
    MAX_FIRST_PERSON_LIGHT_INTENSITY,
    MIN_FIRST_PERSON_LIGHT_INTENSITY,
)
from housemaker.generation_state import GenerationData
from housemaker.generation_workspace import GenerationWorkspace
from housemaker.surface_texture_state import SurfaceTextureData
from housemaker.surface_texture_workspace import (
    SurfaceTextureGenerationWorkspace,
)
from housemaker.glb import (
    DEFAULT_WALL_HEIGHT_METERS,
    GeneratedModel,
    build_stair_meshes,
    convert_to_glb,
    export_glb_file,
    export_room_texture_pngs,
)
from housemaker.models import (
    DEFAULT_DOORWAY_HEIGHT_METERS,
    DEFAULT_DOORWAY_WIDTH_METERS,
    DEFAULT_FLOOR_THICKNESS_METERS,
    DEFAULT_IMAGE_OFFSET,
    DEFAULT_IMAGE_SCALE,
    DEFAULT_LEVEL_OFFSET_METERS,
    DEFAULT_LEVEL_SCALE,
    DEFAULT_ROOM_HEIGHT_METERS,
    DEFAULT_STAIR_STYLE,
    DEFAULT_UV_MAP_HEIGHT,
    DEFAULT_UV_MAP_WIDTH,
    DEFAULT_WALL_UV_ROTATION_DEGREES,
    DEFAULT_WALL_UV_SCALE,
    GROUND_LEVEL_INDEX,
    DoorwayPreset,
    LevelData,
    MAX_DOORWAY_HEIGHT_METERS,
    MAX_DOORWAY_WIDTH_METERS,
    MAX_FLOOR_THICKNESS_METERS,
    MAX_LEVEL_OFFSET_METERS,
    MAX_LEVEL_SCALE,
    MIN_DOORWAY_HEIGHT_METERS,
    MIN_DOORWAY_WIDTH_METERS,
    MIN_FLOOR_THICKNESS_METERS,
    MIN_LEVEL_OFFSET_METERS,
    MIN_LEVEL_SCALE,
    RoomData,
    STAIR_STYLE_FLOATING,
    STAIR_STYLE_FLOATING_WITH_RISER,
    STAIR_STYLE_SUPPORTED,
    StairData,
    StairSectionData,
    create_default_doorway_presets,
    create_default_levels,
)
from housemaker.level_coordinates import build_level_base_z_lookup
from housemaker.project_io import ProjectData, load_project, save_project
from housemaker.settings_widget import (
    SettingsWidget,
    resolve_fullscreen_3d_viewer_screen,
)
from housemaker.texture_creator_canvas import TextureCreatorCanvas
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
DEFAULT_FIRST_PERSON_CAMERA_HEIGHT_METERS = 1.65
FIRST_PERSON_LIGHT_PERCENT_SCALE = 100
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
        self.initial_first_person_camera: InitialFirstPersonCamera | None = None
        self.current_level_index = GROUND_LEVEL_INDEX
        self._is_syncing_level_controls = False
        self._is_syncing_first_person_camera_controls = False
        self._is_syncing_room_controls = False
        self._is_syncing_image_library_controls = False
        self._is_syncing_texture_controls = False
        self._is_syncing_uv_controls = False
        self._is_viewer_refresh_scheduled = False
        self._scheduled_viewer_refresh_preserve_camera = True
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
        self._external_viewer_host.dispose()
        self.surface_texture_generation.shutdown()
        self.generation.shutdown()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.shutdown()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, 1)

        self.workspace_tabs = QTabWidget()
        self.canvas = BlueprintCanvas()
        self.viewer = GlbViewerWidget()
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
        self.generation = GenerationWorkspace(
            asset_directory=(
                self._application_settings.path.parent / "generated"
            )
        )
        self.surface_texture_generation = SurfaceTextureGenerationWorkspace(
            asset_directory=(
                self._application_settings.path.parent / "surface_textures"
            ),
            application_settings=self._application_settings,
        )
        self.settings_widget = SettingsWidget(
            application_settings=self._application_settings
        )
        generation_settings = self.settings_widget.get_settings()
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
        self.surface_texture_generation.set_levels(
            self.levels,
            self.initial_first_person_camera,
        )
        self.workspace_tabs.addTab(self.canvas_viewer_workspace, "Canvas")
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
        self.viewer.navigation_mode_changed.connect(
            self._handle_canvas_3d_navigation_mode_changed
        )
        self.settings_widget.settings_changed.connect(
            self._handle_generation_settings_changed
        )
        splitter.addWidget(self.workspace_tabs)

        self.side_panel = QWidget()
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

        self.stairs_list = QListWidget()
        self.stairs_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.stairs_list.setMinimumHeight(86)
        self.stairs_list.currentRowChanged.connect(
            self._update_stair_button_state
        )
        side_layout.addWidget(self.stairs_list)

        self.delete_selected_stair_button = QPushButton("Delete selected stair")
        self.delete_selected_stair_button.setMinimumHeight(36)
        self.delete_selected_stair_button.clicked.connect(
            self._handle_delete_selected_stair_clicked
        )
        side_layout.addWidget(self.delete_selected_stair_button)

        first_person_camera_label = QLabel("Initial first person camera")
        first_person_camera_label.setStyleSheet(
            "font-size: 18px; font-weight: 600;"
        )
        side_layout.addWidget(first_person_camera_label)

        self.first_person_camera_status_label = QLabel("Camera: Not set")
        self.first_person_camera_status_label.setWordWrap(True)
        side_layout.addWidget(self.first_person_camera_status_label)

        self.first_person_camera_z_spinbox = QDoubleSpinBox()
        self.first_person_camera_z_spinbox.setRange(-10000.0, 10000.0)
        self.first_person_camera_z_spinbox.setDecimals(3)
        self.first_person_camera_z_spinbox.setSingleStep(0.05)
        self.first_person_camera_z_spinbox.setSuffix(" m")
        self.first_person_camera_z_spinbox.setKeyboardTracking(False)
        self.first_person_camera_z_spinbox.setEnabled(False)
        self.first_person_camera_z_spinbox.valueChanged.connect(
            self._handle_first_person_camera_z_changed
        )
        camera_z_layout = QFormLayout()
        camera_z_layout.setContentsMargins(0, 0, 0, 0)
        camera_z_layout.addRow("Camera Z", self.first_person_camera_z_spinbox)
        side_layout.addLayout(camera_z_layout)

        self.first_person_camera_light_slider = QSlider(Qt.Orientation.Horizontal)
        self.first_person_camera_light_slider.setRange(
            _light_intensity_to_percent(MIN_FIRST_PERSON_LIGHT_INTENSITY),
            _light_intensity_to_percent(MAX_FIRST_PERSON_LIGHT_INTENSITY),
        )
        self.first_person_camera_light_slider.setSingleStep(1)
        self.first_person_camera_light_slider.setPageStep(10)
        self.first_person_camera_light_slider.setToolTip(
            "Camera-mounted headlight intensity"
        )
        self.first_person_camera_light_slider.setValue(
            _light_intensity_to_percent(DEFAULT_FIRST_PERSON_LIGHT_INTENSITY)
        )
        self.first_person_camera_light_slider.setEnabled(False)
        self.first_person_camera_light_slider.sliderPressed.connect(
            self.canvas.begin_first_person_camera_light_intensity_adjustment
        )
        self.first_person_camera_light_slider.sliderReleased.connect(
            self.canvas.end_first_person_camera_light_intensity_adjustment
        )
        self.first_person_camera_light_slider.valueChanged.connect(
            self._handle_first_person_camera_light_intensity_changed
        )
        self.first_person_camera_light_value_label = QLabel(
            _format_light_intensity_percent(
                _light_intensity_to_percent(DEFAULT_FIRST_PERSON_LIGHT_INTENSITY)
            )
        )
        camera_light_control = QWidget()
        camera_light_layout = QHBoxLayout(camera_light_control)
        camera_light_layout.setContentsMargins(0, 0, 0, 0)
        camera_light_layout.setSpacing(8)
        camera_light_layout.addWidget(self.first_person_camera_light_slider, 1)
        camera_light_layout.addWidget(self.first_person_camera_light_value_label)
        camera_light_form = QFormLayout()
        camera_light_form.setContentsMargins(0, 0, 0, 0)
        camera_light_form.addRow("Light intensity", camera_light_control)
        side_layout.addLayout(camera_light_form)

        first_person_camera_buttons_layout = QHBoxLayout()
        first_person_camera_buttons_layout.setSpacing(10)
        self.set_first_person_camera_button = QPushButton(
            "Set first person camera"
        )
        self.set_first_person_camera_button.setMinimumHeight(40)
        self.set_first_person_camera_button.clicked.connect(
            self._handle_set_first_person_camera_clicked
        )
        first_person_camera_buttons_layout.addWidget(
            self.set_first_person_camera_button
        )

        self.clear_first_person_camera_button = QPushButton("Clear camera")
        self.clear_first_person_camera_button.setMinimumHeight(40)
        self.clear_first_person_camera_button.setEnabled(False)
        self.clear_first_person_camera_button.clicked.connect(
            self.canvas.clear_first_person_camera
        )
        first_person_camera_buttons_layout.addWidget(
            self.clear_first_person_camera_button
        )
        side_layout.addLayout(first_person_camera_buttons_layout)

        doorway_label = QLabel("Doorways")
        doorway_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(doorway_label)

        doorway_dimensions_layout = QFormLayout()
        doorway_dimensions_layout.setContentsMargins(0, 0, 0, 0)
        doorway_dimensions_layout.setSpacing(8)

        self.doorway_width_spinbox = QDoubleSpinBox()
        self.doorway_width_spinbox.setRange(
            MIN_DOORWAY_WIDTH_METERS,
            MAX_DOORWAY_WIDTH_METERS,
        )
        self.doorway_width_spinbox.setDecimals(2)
        self.doorway_width_spinbox.setSingleStep(0.05)
        self.doorway_width_spinbox.setValue(DEFAULT_DOORWAY_WIDTH_METERS)
        self.doorway_width_spinbox.setSuffix(" m")
        self.doorway_width_spinbox.setMinimumHeight(34)
        doorway_dimensions_layout.addRow("Width", self.doorway_width_spinbox)

        self.doorway_height_spinbox = QDoubleSpinBox()
        self.doorway_height_spinbox.setRange(
            MIN_DOORWAY_HEIGHT_METERS,
            MAX_DOORWAY_HEIGHT_METERS,
        )
        self.doorway_height_spinbox.setDecimals(2)
        self.doorway_height_spinbox.setSingleStep(0.05)
        self.doorway_height_spinbox.setValue(DEFAULT_DOORWAY_HEIGHT_METERS)
        self.doorway_height_spinbox.setSuffix(" m")
        self.doorway_height_spinbox.setMinimumHeight(34)
        doorway_dimensions_layout.addRow("Height", self.doorway_height_spinbox)
        side_layout.addLayout(doorway_dimensions_layout)

        self.add_doorway_preset_button = QPushButton("Add preset")
        self.add_doorway_preset_button.setMinimumHeight(40)
        self.add_doorway_preset_button.clicked.connect(
            self._handle_add_doorway_preset_clicked
        )
        side_layout.addWidget(self.add_doorway_preset_button)

        self.doorway_preset_list = QListWidget()
        self.doorway_preset_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.doorway_preset_list.setMinimumHeight(112)
        self.doorway_preset_list.currentRowChanged.connect(
            self._handle_doorway_preset_selection_changed
        )
        side_layout.addWidget(self.doorway_preset_list)

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

        image_transform_layout = QFormLayout()
        image_transform_layout.setContentsMargins(0, 0, 0, 0)
        image_transform_layout.setSpacing(8)

        self.image_scale_spinbox = self._build_image_transform_spinbox(
            value=DEFAULT_IMAGE_SCALE,
            minimum=0.01,
            maximum=20.0,
            decimals=3,
            single_step=0.05,
        )
        self.image_scale_spinbox.valueChanged.connect(self._handle_image_scale_changed)
        image_transform_layout.addRow("Blueprint scale", self.image_scale_spinbox)

        self.image_x_offset_spinbox = self._build_image_transform_spinbox(
            value=DEFAULT_IMAGE_OFFSET,
            minimum=-100000.0,
            maximum=100000.0,
            decimals=1,
            single_step=1.0,
            suffix=" px",
        )
        self.image_x_offset_spinbox.valueChanged.connect(
            self._handle_image_x_offset_changed
        )
        image_transform_layout.addRow(
            "Blueprint X offset",
            self.image_x_offset_spinbox,
        )

        self.image_y_offset_spinbox = self._build_image_transform_spinbox(
            value=DEFAULT_IMAGE_OFFSET,
            minimum=-100000.0,
            maximum=100000.0,
            decimals=1,
            single_step=1.0,
            suffix=" px",
        )
        self.image_y_offset_spinbox.valueChanged.connect(
            self._handle_image_y_offset_changed
        )
        image_transform_layout.addRow(
            "Blueprint Y offset",
            self.image_y_offset_spinbox,
        )

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
        image_transform_layout.addRow("Include", include_widget)
        side_layout.addLayout(image_transform_layout)

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

        rooms_label = QLabel("Rooms")
        rooms_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(rooms_label)

        self.rooms_list = QListWidget()
        self.rooms_list.currentRowChanged.connect(self._handle_room_selection_changed)
        side_layout.addWidget(self.rooms_list, 1)

        self.delete_room_shortcut = QShortcut(
            QKeySequence.StandardKey.Delete,
            self.rooms_list,
        )
        self.delete_room_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.delete_room_shortcut.activated.connect(self._delete_selected_room)

        room_layout = QFormLayout()
        room_layout.setContentsMargins(0, 0, 0, 0)
        room_layout.setSpacing(8)

        self.room_name_field = QLineEdit()
        self.room_name_field.setPlaceholderText("Room name")
        self.room_name_field.setMinimumHeight(34)
        room_layout.addRow("Room name", self.room_name_field)

        self.room_height_spinbox = QDoubleSpinBox()
        self.room_height_spinbox.setRange(0.1, 100.0)
        self.room_height_spinbox.setDecimals(2)
        self.room_height_spinbox.setSingleStep(0.1)
        self.room_height_spinbox.setValue(DEFAULT_ROOM_HEIGHT_METERS)
        self.room_height_spinbox.setSuffix(" m")
        self.room_height_spinbox.setMinimumHeight(34)
        self.room_height_spinbox.valueChanged.connect(
            self._handle_room_height_changed
        )
        room_layout.addRow("Room height", self.room_height_spinbox)
        side_layout.addLayout(room_layout)

        self.designate_room_button = QPushButton("Designate room")
        self.designate_room_button.setMinimumHeight(44)
        self.designate_room_button.clicked.connect(self._handle_designate_room_clicked)
        side_layout.addWidget(self.designate_room_button)

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

        self.side_tabs.addTab(self._build_uvs_tab(), "UVs")
        self.side_tabs.addTab(self._build_images_tab(), "Images")
        self.side_tabs.addTab(self._build_texture_creator_tab(), "Texture creator")

        splitter.addWidget(self.side_panel)
        splitter.setStretchFactor(0, 9)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1160, 440])

        self.canvas.rooms_changed.connect(self._refresh_room_lists)
        self.canvas.rooms_changed.connect(self._schedule_viewer_preview_refresh)
        self.canvas.floor_contour_changed.connect(
            self._handle_floor_contour_changed
        )
        self.canvas.doorways_changed.connect(self._handle_doorways_changed)
        self.canvas.first_person_camera_changed.connect(
            self._handle_canvas_first_person_camera_changed
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
        self.canvas.stair_selected.connect(self._handle_stair_selected)
        self.canvas.stair_delete_requested.connect(
            self._handle_stair_delete_requested
        )
        self._refresh_levels_list()
        self._refresh_stairs_list()
        self._refresh_room_lists()
        self._sync_level_controls()
        self._sync_canvas_to_current_level()
        self._sync_texture_creator_tab()
        self._schedule_viewer_preview_refresh(preserve_camera=False)
        self._apply_fullscreen_3d_viewer_screen(
            generation_settings.fullscreen_3d_viewer_screen_id
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

        if self.viewer.get_navigation_mode() != NAVIGATION_MODE_FIRST_PERSON:
            self._sync_canvas_viewer_first_person_camera()
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

    @staticmethod
    def _build_image_transform_spinbox(
        value: float,
        minimum: float,
        maximum: float,
        decimals: int,
        single_step: float,
        suffix: str = "",
    ) -> QDoubleSpinBox:
        spinbox = QDoubleSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setDecimals(decimals)
        spinbox.setSingleStep(single_step)
        spinbox.setValue(value)
        spinbox.setSuffix(suffix)
        spinbox.setMinimumHeight(34)
        return spinbox

    def load_blueprint(self, file_path: str) -> None:
        self._set_current_level_image(file_path)

    def _handle_glb_export_clicked(self) -> None:
        generated_model = self._build_generated_model("Export failed")
        if generated_model is None:
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

        export_path = Path(file_path)
        if export_path.suffix.lower() != ".glb":
            export_path = export_path.with_suffix(".glb")

        try:
            exported_path = export_glb_file(generated_model, export_path)
        except OSError as error:
            QMessageBox.critical(self, "Export failed", str(error))
            return

        self.workspace_tabs.setCurrentWidget(self.canvas_viewer_workspace)
        self.viewer.set_model(generated_model)
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

    def _handle_workspace_tab_changed(self, tab_index: int) -> None:
        selected_widget = self.workspace_tabs.widget(tab_index)
        is_full_width_workspace = selected_widget in (
            self.surface_texture_generation,
            self.generation,
            self.settings_widget,
        )
        self.side_panel.setVisible(not is_full_width_workspace)
        if selected_widget is self.surface_texture_generation:
            self.surface_texture_generation.set_levels(
                self.levels,
                self.initial_first_person_camera,
            )

        self._apply_fullscreen_3d_viewer_screen(
            self.settings_widget.get_settings().fullscreen_3d_viewer_screen_id
        )
        if is_full_width_workspace:
            return

        if selected_widget is not self.canvas_viewer_workspace:
            return

        self._schedule_viewer_preview_refresh(preserve_camera=False)

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

        screen = resolve_fullscreen_3d_viewer_screen(screen_id)
        viewer = self._active_workspace_3d_viewer()
        if screen is None or viewer is None:
            self._external_viewer_host.restore()
            self._sync_external_3d_workspace_presentations()
            return
        self._external_viewer_host.show_on_screen(viewer, screen)
        self._sync_external_3d_workspace_presentations()

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
        return None

    def _viewer_preview_is_active(self) -> bool:
        return (
            self.workspace_tabs.currentWidget()
            is self.canvas_viewer_workspace
            or (
                self._external_viewer_host.is_active
                and self._external_viewer_host.viewer is self.viewer
            )
        )

    def _build_generated_model(
        self,
        failure_title: str | None,
    ) -> GeneratedModel | None:
        try:
            return convert_to_glb(
                self.levels,
                stairs=self.stairs,
                surface_materials=(
                    self.surface_texture_generation.get_surface_material_sources()
                ),
            )
        except ValueError as error:
            if failure_title is not None:
                QMessageBox.warning(self, failure_title, str(error))
            return None

    def _handle_surface_texture_generation_completed(
        self,
        _assignment: object,
    ) -> None:
        self._schedule_viewer_preview_refresh(preserve_camera=True)

    def _refresh_viewer_preview(self, preserve_camera: bool = False) -> None:
        generated_model = self._build_generated_model(None)
        if generated_model is None:
            self.viewer.clear_model()
            return

        self.viewer.set_model(generated_model, preserve_camera=preserve_camera)

    def _schedule_viewer_preview_refresh(self, preserve_camera: bool = True) -> None:
        if not self._viewer_preview_is_active():
            return
        if self._is_viewer_refresh_scheduled:
            self._scheduled_viewer_refresh_preserve_camera = (
                self._scheduled_viewer_refresh_preserve_camera and preserve_camera
            )
            return

        self._is_viewer_refresh_scheduled = True
        self._scheduled_viewer_refresh_preserve_camera = preserve_camera
        QTimer.singleShot(0, self._run_scheduled_viewer_preview_refresh)

    def _run_scheduled_viewer_preview_refresh(self) -> None:
        self._is_viewer_refresh_scheduled = False
        if not self._viewer_preview_is_active():
            return

        preserve_camera = self._scheduled_viewer_refresh_preserve_camera
        self._scheduled_viewer_refresh_preserve_camera = True
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
                initial_first_person_camera=self.initial_first_person_camera,
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

    def _refresh_stairs_list(self) -> None:
        """Refresh the cross-level stair records without changing placement."""

        selected_index = self.stairs_list.currentRow()
        self.stairs_list.blockSignals(True)
        self.stairs_list.clear()
        for stair in self.stairs:
            self.stairs_list.addItem(_format_stair_label(stair))
        if 0 <= selected_index < self.stairs_list.count():
            self.stairs_list.setCurrentRow(selected_index)
        self.stairs_list.blockSignals(False)
        self._update_stair_button_state()

    def _update_stair_button_state(self, _row: int = -1) -> None:
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
        selected_index = self.stairs_list.currentRow()
        self.delete_selected_stair_button.setEnabled(
            not placement_active
            and 0 <= selected_index < len(self.stairs)
        )

    def _get_selected_doorway_preset(self) -> DoorwayPreset | None:
        selected_index = self.doorway_preset_list.currentRow()
        if selected_index < 0 or selected_index >= len(self.doorway_presets):
            return None

        return self.doorway_presets[selected_index]

    def _update_doorway_preset_button_state(self) -> None:
        has_selected_preset = self._get_selected_doorway_preset() is not None
        self.remove_doorway_preset_button.setEnabled(has_selected_preset)
        self.place_doorway_button.setEnabled(has_selected_preset)

    def _refresh_rooms_list(self) -> None:
        selected_room_index = self._get_selected_room_index()
        self._is_syncing_room_controls = True
        self.rooms_list.clear()
        for room_index, room in enumerate(self.current_level.rooms):
            room_item = self._build_room_item(room_index, room)
            self.rooms_list.addItem(room_item)

        if (
            selected_room_index is not None
            and selected_room_index < self.rooms_list.count()
        ):
            self.rooms_list.setCurrentRow(selected_room_index)
        self._is_syncing_room_controls = False
        self._sync_room_controls()

    def _refresh_room_lists(self) -> None:
        self._refresh_rooms_list()
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

    def _get_selected_room_index(self) -> int | None:
        selected_item = self.rooms_list.currentItem()
        if selected_item is None:
            return None

        room_index = selected_item.data(Qt.ItemDataRole.UserRole)
        if room_index is None:
            return None

        return int(room_index)

    def _get_selected_room(self) -> RoomData | None:
        room_index = self._get_selected_room_index()
        if room_index is None or room_index >= len(self.current_level.rooms):
            return None

        return self.current_level.rooms[room_index]

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

        thumbnail_pixmap = QPixmap(image_path)
        if thumbnail_pixmap.isNull():
            thumbnail_item.setText(f"{image_name}\nmissing")
            return thumbnail_item

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
        return thumbnail_item

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
            self._refresh_room_lists()

        if self.uv_rooms_list.currentRow() != room_index:
            self.uv_rooms_list.setCurrentRow(room_index)
        if self.rooms_list.currentRow() != room_index:
            self.rooms_list.setCurrentRow(room_index)
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
        self.image_scale_spinbox.setValue(self.current_level.image_scale)
        self.image_x_offset_spinbox.setValue(self.current_level.image_offset_x)
        self.image_y_offset_spinbox.setValue(self.current_level.image_offset_y)
        self.include_yes_radio.setChecked(self.current_level.include_in_export)
        self.include_no_radio.setChecked(not self.current_level.include_in_export)
        if self.levels_list.currentRow() != self.current_level_index:
            self.levels_list.setCurrentRow(self.current_level_index)
        self._update_blueprint_name_label()
        self._is_syncing_level_controls = False

    def _sync_room_controls(self) -> None:
        selected_room = self._get_selected_room()
        self._is_syncing_room_controls = True
        if selected_room is None:
            self.room_height_spinbox.setValue(DEFAULT_ROOM_HEIGHT_METERS)
        else:
            self.room_height_spinbox.setValue(selected_room.height_meters)
        self._is_syncing_room_controls = False

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

        self.current_level_index = level_index
        self._sync_level_controls()
        self._sync_canvas_to_current_level()
        self._update_pending_stair_level_status()
        self._refresh_room_lists()
        self._schedule_viewer_preview_refresh()

    def _handle_room_selection_changed(self, _room_index: int) -> None:
        if self._is_syncing_room_controls:
            return

        self._sync_room_controls()

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

    def _handle_add_doorway_preset_clicked(self) -> None:
        doorway_preset = DoorwayPreset(
            width_meters=float(self.doorway_width_spinbox.value()),
            height_meters=float(self.doorway_height_spinbox.value()),
        )
        self.doorway_presets.append(doorway_preset)
        self._refresh_doorway_preset_list(
            selected_index=len(self.doorway_presets) - 1
        )

    def _handle_doorway_preset_selection_changed(self, _row: int) -> None:
        self._update_doorway_preset_button_state()

    def _handle_remove_doorway_preset_clicked(self) -> None:
        selected_index = self.doorway_preset_list.currentRow()
        if selected_index < 0 or selected_index >= len(self.doorway_presets):
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

    def _handle_doorways_changed(self) -> None:
        self.current_level.doorways = self.canvas.doorways
        self._schedule_viewer_preview_refresh()

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
        self._refresh_stairs_list()
        self.stairs_list.setCurrentRow(len(self.stairs) - 1)
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

    def _handle_stair_selected(self, stair_index: int) -> None:
        if 0 <= stair_index < self.stairs_list.count():
            self.stairs_list.setCurrentRow(stair_index)

    def _handle_stair_delete_requested(self, stair_index: int) -> None:
        self._delete_stair_at_index(stair_index)

    def _handle_delete_selected_stair_clicked(self) -> None:
        self._delete_stair_at_index(self.stairs_list.currentRow())

    def _delete_stair_at_index(self, stair_index: int) -> None:
        if not 0 <= stair_index < len(self.stairs):
            return

        del self.stairs[stair_index]
        self.canvas.set_stair_context(self.stairs, self.current_level)
        self._refresh_stairs_list()
        self.stair_status_label.setText("Stair deleted.")
        self._schedule_viewer_preview_refresh()

    def _handle_set_first_person_camera_clicked(self) -> None:
        if self.canvas.blueprint_image is None:
            QMessageBox.information(
                self,
                "Blueprint required",
                "Load a blueprint image for this level before placing the camera.",
            )
            return

        camera = self.initial_first_person_camera
        if camera is not None and camera.level_index == self.current_level.index:
            z_meters = camera.pose.z
        else:
            level_base_z = build_level_base_z_lookup(self.levels).get(
                self.current_level.index,
                0.0,
            )
            z_meters = (
                level_base_z + DEFAULT_FIRST_PERSON_CAMERA_HEIGHT_METERS
            )

        self.workspace_tabs.setCurrentWidget(self.canvas_viewer_workspace)
        self.canvas.start_first_person_camera_placement(z_meters)
        self.first_person_camera_status_label.setText(
            "Camera placement active: click the blueprint to place it."
        )

    def _handle_canvas_first_person_camera_changed(self, raw_camera: object) -> None:
        if raw_camera is not None and not isinstance(
            raw_camera,
            InitialFirstPersonCamera,
        ):
            return
        self.initial_first_person_camera = raw_camera
        self._sync_canvas_viewer_first_person_camera()
        self._sync_first_person_camera_controls()
        self.surface_texture_generation.set_levels(
            self.levels,
            self.initial_first_person_camera,
        )

    def _sync_canvas_viewer_first_person_camera(self) -> None:
        """Use the Canvas camera as the standard viewer's FP entry pose."""

        camera = self.initial_first_person_camera
        if camera is not None:
            self.viewer.set_first_person_camera_pose(camera.pose)

    def _handle_generation_settings_changed(self) -> None:
        settings = self.settings_widget.get_settings()
        self._set_canvas_3d_navigation_shortcut(
            settings.canvas_3d_navigation_toggle_hotkey
        )
        self.generation.set_runtime_settings(settings)
        self.surface_texture_generation.set_runtime_settings(settings)
        self._apply_fullscreen_3d_viewer_screen(
            settings.fullscreen_3d_viewer_screen_id
        )

    def _handle_first_person_camera_z_changed(self, value: float) -> None:
        if self._is_syncing_first_person_camera_controls:
            return
        self.canvas.update_first_person_camera_z(float(value))

    def _handle_first_person_camera_light_intensity_changed(
        self,
        value: int,
    ) -> None:
        self.first_person_camera_light_value_label.setText(
            _format_light_intensity_percent(value)
        )
        if self._is_syncing_first_person_camera_controls:
            return
        self.canvas.update_first_person_camera_light_intensity(
            _percent_to_light_intensity(value)
        )

    def _sync_first_person_camera_controls(self) -> None:
        camera = self.initial_first_person_camera
        self._is_syncing_first_person_camera_controls = True
        self.first_person_camera_z_spinbox.setEnabled(camera is not None)
        self.first_person_camera_light_slider.setEnabled(camera is not None)
        self.clear_first_person_camera_button.setEnabled(camera is not None)
        if camera is None:
            self.first_person_camera_status_label.setText("Camera: Not set")
            light_percent = _light_intensity_to_percent(
                DEFAULT_FIRST_PERSON_LIGHT_INTENSITY
            )
            self.first_person_camera_light_slider.setValue(light_percent)
            self.first_person_camera_light_value_label.setText(
                _format_light_intensity_percent(light_percent)
            )
        else:
            self.first_person_camera_z_spinbox.setValue(camera.pose.z)
            light_percent = _light_intensity_to_percent(camera.light_intensity)
            self.first_person_camera_light_slider.setValue(light_percent)
            self.first_person_camera_light_value_label.setText(
                _format_light_intensity_percent(light_percent)
            )
            owner_level = next(
                (
                    level
                    for level in self.levels
                    if level.index == camera.level_index
                ),
                None,
            )
            owner_name = (
                f"L{camera.level_index}"
                if owner_level is None
                else owner_level.display_name
            )
            self.first_person_camera_status_label.setText(
                f"Camera: {owner_name} | "
                f"X {camera.pose.x:.2f} m | Y {camera.pose.y:.2f} m | "
                f"Z {camera.pose.z:.2f} m | Yaw {camera.pose.yaw_degrees:.1f} deg | "
                f"Light {light_percent}%"
            )
        self._is_syncing_first_person_camera_controls = False

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

    def _handle_room_height_changed(self, value: float) -> None:
        if self._is_syncing_room_controls:
            return

        selected_room = self._get_selected_room()
        if selected_room is None:
            return

        selected_room.height_meters = float(value)
        self._refresh_room_subdivision_layout(selected_room)
        self._refresh_room_lists()
        self.uv_canvas.update()
        self._schedule_viewer_preview_refresh()

    def _handle_image_scale_changed(self, value: float) -> None:
        if self._is_syncing_level_controls:
            return

        self.current_level.image_scale = value
        self._sync_canvas_image_transform()

    def _handle_image_x_offset_changed(self, value: float) -> None:
        if self._is_syncing_level_controls:
            return

        self.current_level.image_offset_x = value
        self._sync_canvas_image_transform()

    def _handle_image_y_offset_changed(self, value: float) -> None:
        if self._is_syncing_level_controls:
            return

        self.current_level.image_offset_y = value
        self._sync_canvas_image_transform()

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

    def _handle_designate_room_clicked(self) -> None:
        room_name = self.room_name_field.text().strip()
        if not room_name:
            QMessageBox.warning(self, "Room name required", "Enter a room name first.")
            return

        selected_vertex_ids = self.canvas.get_selected_vertex_ids()
        if len(selected_vertex_ids) < 3:
            QMessageBox.warning(
                self,
                "Select room vertices",
                "Shift-click at least three vertices before designating a room.",
            )
            return

        self.canvas.start_room_designation(
            room_name,
            selected_vertex_ids,
            self.room_height_spinbox.value(),
        )
        QMessageBox.information(
            self,
            "Set room center",
            "Click the vertex that marks the center of the room.",
        )

    def _delete_selected_room(self) -> None:
        room_index = self._get_selected_room_index()
        if room_index is None:
            return

        self.canvas.delete_room_at_index(room_index)

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
            initial_first_person_camera=(
                project_data.initial_first_person_camera
            ),
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
        initial_first_person_camera: InitialFirstPersonCamera | None = None,
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

        self.canvas.cancel_stair_placement()
        self.levels = levels
        self.stairs = list(stairs or [])
        self.initial_first_person_camera = initial_first_person_camera
        self._sync_canvas_viewer_first_person_camera()
        self.image_library_paths = self._normalize_image_library_paths(
            image_library_paths or []
        )
        if doorway_presets is not None:
            self.doorway_presets = list(doorway_presets)
        self.current_level_index = min(max(current_level_index, 0), len(self.levels) - 1)
        self.texture_creator_level_index = None
        self.texture_creator_room_index = None
        self.texture_creator_wall_key = None

        self._refresh_image_thumbnail_list()
        self._refresh_doorway_preset_list(
            selected_index=0 if self.doorway_presets else -1
        )
        self._refresh_levels_list()
        self._refresh_stairs_list()
        self.stair_status_label.setText(
            "Stairs: none"
            if not self.stairs
            else f"Stairs: {len(self.stairs)} loaded."
        )
        self._sync_level_controls()
        self._sync_canvas_to_current_level()
        self._refresh_room_lists()
        self.generation.set_data(generation)
        self.surface_texture_generation.set_levels(
            self.levels,
            self.initial_first_person_camera,
        )
        self.surface_texture_generation.set_data(
            surface_texture_generation
        )
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
            image_scale=self.current_level.image_scale,
            image_offset_x=self.current_level.image_offset_x,
            image_offset_y=self.current_level.image_offset_y,
        )
        self.current_level.image_path = normalized_path
        self.current_level.image_size_pixels = self.canvas.get_image_size_pixels()
        self.canvas.set_initial_first_person_camera_context(
            self.initial_first_person_camera,
            self.current_level,
        )
        self.canvas.set_stair_context(self.stairs, self.current_level)
        self.workspace_tabs.setCurrentWidget(self.canvas_viewer_workspace)
        self._update_blueprint_name_label()
        self._schedule_viewer_preview_refresh()

    def _sync_canvas_to_current_level(self) -> None:
        self.canvas.set_level_data(
            vertex_data=self.current_level.vertex_data,
            rooms=self.current_level.rooms,
            doorways=self.current_level.doorways,
            floor_contour_vertex_ids=(
                self.current_level.floor_contour_vertex_ids
            ),
            image_path=self.current_level.image_path,
            image_scale=self.current_level.image_scale,
            image_offset_x=self.current_level.image_offset_x,
            image_offset_y=self.current_level.image_offset_y,
        )
        if self.canvas.blueprint_image is not None:
            self.current_level.image_size_pixels = self.canvas.get_image_size_pixels()
        self.canvas.set_initial_first_person_camera_context(
            self.initial_first_person_camera,
            self.current_level,
        )
        self.canvas.set_stair_context(self.stairs, self.current_level)
        self._update_blueprint_name_label()
        self._sync_first_person_camera_controls()

    def _sync_canvas_image_transform(self) -> None:
        self.canvas.set_image_transform(
            image_scale=self.current_level.image_scale,
            image_offset_x=self.current_level.image_offset_x,
            image_offset_y=self.current_level.image_offset_y,
        )

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
            self.image_path_label.setText("No image selected")
            self.image_preview_label.setPixmap(QPixmap())
            self.image_preview_label.setText("No image loaded")
            return

        preview_pixmap = QPixmap(selected_image_path)
        if preview_pixmap.isNull():
            self.image_path_label.setText(f"Image missing: {selected_image_path}")
            self.image_preview_label.setPixmap(QPixmap())
            self.image_preview_label.setText("Image missing")
            self.images_save_png_button.setEnabled(False)
            return

        self.image_path_label.setText(f"Image: {Path(selected_image_path).name}")
        target_width = max(320, self.image_preview_label.width() - 16)
        target_height = max(220, self.image_preview_label.height() - 16)
        self.image_preview_label.setText("")
        self.image_preview_label.setPixmap(
            preview_pixmap.scaled(
                target_width,
                target_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
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


def _light_intensity_to_percent(intensity: float) -> int:
    return int(round(float(intensity) * FIRST_PERSON_LIGHT_PERCENT_SCALE))


def _percent_to_light_intensity(percent: int) -> float:
    return float(percent) / FIRST_PERSON_LIGHT_PERCENT_SCALE


def _format_light_intensity_percent(percent: int) -> str:
    return f"{int(percent)}%"


# ### Text helpers ###
def _format_stair_label(stair: StairData) -> str:
    style_label = _format_stair_style_label(stair.style)
    guide_count = len(stair.intermediate_sections)
    route_label = (
        "straight"
        if guide_count == 0
        else f"curved, {guide_count} guide{'s' if guide_count != 1 else ''}"
    )
    return (
        f"{style_label}: L{stair.start_level_index} "
        f"to L{stair.end_level_index} ({route_label})"
    )


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
    return (
        f"{doorway_preset.width_meters:.2f} m × "
        f"{doorway_preset.height_meters:.2f} m"
    )


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
