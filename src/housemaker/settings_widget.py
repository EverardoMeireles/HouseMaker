# ### Imports ###
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence, QScreen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from housemaker.app_settings import ApplicationSettingsStore


# ### Constants ###
MESHY_API_KEY_ENVIRONMENT_VARIABLE = "MESHY_API_KEY"
MESHY_API_KEY_SETTING_KEY = "generation/meshy_api_key"
OPENAI_API_KEY_ENVIRONMENT_VARIABLE = "OPENAI_API_KEY"
OPENAI_API_KEY_SETTING_KEY = "generation/openai_api_key"
SURFACE_TEXTURE_PROVIDER_SETTING_KEY = "generation/surface_texture_provider"
FULLSCREEN_3D_VIEWER_SCREEN_SETTING_KEY = (
    "display/fullscreen_3d_viewer_screen_id"
)
JOBS_WINDOW_SCREEN_SETTING_KEY = "display/jobs_window_screen_id"
AUTOMATIC_ATLAS_TEXTURE_SORT_BY_PBR_SETTING_KEY = (
    "atlas/automatic_texture_sort_by_pbr"
)
AUTOMATIC_ATLAS_TEXTURE_RESOLUTION_SETTING_KEY = (
    "atlas/automatic_texture_resolution"
)
CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY_SETTING_KEY = (
    "navigation/canvas_3d_navigation_toggle_hotkey"
)
UNUSED_FACE_REMOVAL_SETTING_KEY = "generation/unused_face_removal"
USE_UV_RAYCAST_FOR_OBJECT_GENERATION_SETTING_KEY = (
    "generation/use_uv_raycast_for_object_generation"
)
MINIMUM_FACE_VISIBILITY_PERCENTAGE_SETTING_KEY = (
    "generation/minimum_face_visibility_percentage"
)
MESH_EDIT_UPDATE_DELAY_SECONDS_SETTING_KEY = (
    # Keep the original persisted key so existing preferences remain valid.
    "canvas/doorway_mesh_update_delay_seconds"
)
DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY = "N"
DEFAULT_UNUSED_FACE_REMOVAL = False
DEFAULT_USE_UV_RAYCAST_FOR_OBJECT_GENERATION = False
DEFAULT_MINIMUM_FACE_VISIBILITY_PERCENTAGE = 5
AUTOMATIC_ATLAS_TEXTURE_RESOLUTIONS = (512, 1024)
DEFAULT_AUTOMATIC_ATLAS_TEXTURE_SORT_BY_PBR = False
DEFAULT_AUTOMATIC_ATLAS_TEXTURE_RESOLUTION = 512
MINIMUM_FACE_VISIBILITY_PERCENTAGE = 0
MAXIMUM_FACE_VISIBILITY_PERCENTAGE = 100
DEFAULT_MESH_EDIT_UPDATE_DELAY_SECONDS = 1.0
MIN_MESH_EDIT_UPDATE_DELAY_SECONDS = 0.1
MAX_MESH_EDIT_UPDATE_DELAY_SECONDS = 10.0
MESH_EDIT_UPDATE_DELAY_STEP_SECONDS = 0.1
MESHY_SMART_TOPOLOGY_MIN_TARGET_POLYCOUNT = 100
MESHY_SMART_TOPOLOGY_MAX_TARGET_POLYCOUNT = 15_000
DEFAULT_MESHY_TARGET_POLYCOUNT = 2_000
SURFACE_TEXTURE_PROVIDER_MESHY = "meshy"
SURFACE_TEXTURE_PROVIDER_GPT_4O_MINI = "gpt-4o-mini"
SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA = "gpt-5.6-luna"
SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA = "gpt-5.6-terra"
SURFACE_TEXTURE_PROVIDER_OPTIONS = (
    ("Meshy", SURFACE_TEXTURE_PROVIDER_MESHY),
    ("GPT-4o-mini", SURFACE_TEXTURE_PROVIDER_GPT_4O_MINI),
    ("GPT-5.6 Luna", SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA),
    ("GPT-5.6 Terra", SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA),
)
SURFACE_TEXTURE_PROVIDERS = frozenset(
    provider_id for _label, provider_id in SURFACE_TEXTURE_PROVIDER_OPTIONS
)
_MODIFIER_ONLY_SHORTCUT_KEYS = frozenset(
    {
        Qt.Key.Key_Alt,
        Qt.Key.Key_AltGr,
        Qt.Key.Key_Control,
        Qt.Key.Key_Hyper_L,
        Qt.Key.Key_Hyper_R,
        Qt.Key.Key_Meta,
        Qt.Key.Key_Shift,
        Qt.Key.Key_Super_L,
        Qt.Key.Key_Super_R,
    }
)
_FIRST_PERSON_MOVEMENT_SHORTCUT_KEYS = frozenset(
    {
        Qt.Key.Key_D,
        Qt.Key.Key_F,
        Qt.Key.Key_Q,
        Qt.Key.Key_R,
        Qt.Key.Key_S,
        Qt.Key.Key_Z,
    }
)


# ### Data models ###
@dataclass(frozen=True)
class Fullscreen3DViewerScreenOption:
    """A currently connected display that can host the fullscreen 3D view."""

    screen_id: str
    label: str


@dataclass(frozen=True)
class GenerationServiceSettings:
    meshy_api_key: str = field(default="", repr=False)
    meshy_target_polycount: int = DEFAULT_MESHY_TARGET_POLYCOUNT
    openai_api_key: str = field(default="", repr=False)
    surface_texture_provider: str = SURFACE_TEXTURE_PROVIDER_MESHY
    fullscreen_3d_viewer_screen_id: str | None = None
    canvas_3d_navigation_toggle_hotkey: str = (
        DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY
    )
    unused_face_removal: bool = DEFAULT_UNUSED_FACE_REMOVAL
    jobs_window_screen_id: str | None = None
    mesh_edit_update_delay_seconds: float = (
        DEFAULT_MESH_EDIT_UPDATE_DELAY_SECONDS
    )
    use_uv_raycast_for_object_generation: bool = (
        DEFAULT_USE_UV_RAYCAST_FOR_OBJECT_GENERATION
    )
    minimum_face_visibility_percentage: int = (
        DEFAULT_MINIMUM_FACE_VISIBILITY_PERCENTAGE
    )
    automatic_atlas_texture_resolution: int = (
        DEFAULT_AUTOMATIC_ATLAS_TEXTURE_RESOLUTION
    )
    automatic_atlas_texture_sort_by_pbr: bool = (
        DEFAULT_AUTOMATIC_ATLAS_TEXTURE_SORT_BY_PBR
    )

    def __post_init__(self) -> None:
        if not isinstance(self.automatic_atlas_texture_sort_by_pbr, bool):
            raise ValueError(
                "Automatic Atlas PBR sorting must be enabled or disabled."
            )
        if (
            isinstance(self.automatic_atlas_texture_resolution, bool)
            or not isinstance(self.automatic_atlas_texture_resolution, int)
            or self.automatic_atlas_texture_resolution
            not in AUTOMATIC_ATLAS_TEXTURE_RESOLUTIONS
        ):
            raise ValueError(
                "Automatic Atlas texture resolution must be 512 or 1024."
            )
        if not isinstance(self.unused_face_removal, bool):
            raise ValueError("Unused face removal must be enabled or disabled.")
        if not isinstance(self.use_uv_raycast_for_object_generation, bool):
            raise ValueError(
                "Weighted camera projection for object generation must be "
                "enabled or disabled."
            )
        if (
            isinstance(self.minimum_face_visibility_percentage, bool)
            or not isinstance(self.minimum_face_visibility_percentage, int)
            or not (
                MINIMUM_FACE_VISIBILITY_PERCENTAGE
                <= self.minimum_face_visibility_percentage
                <= MAXIMUM_FACE_VISIBILITY_PERCENTAGE
            )
        ):
            raise ValueError(
                "Minimum face visibility percentage must be between "
                f"{MINIMUM_FACE_VISIBILITY_PERCENTAGE}% and "
                f"{MAXIMUM_FACE_VISIBILITY_PERCENTAGE}%."
            )
        if (
            isinstance(self.meshy_target_polycount, bool)
            or not isinstance(self.meshy_target_polycount, int)
            or not (
                MESHY_SMART_TOPOLOGY_MIN_TARGET_POLYCOUNT
                <= self.meshy_target_polycount
                <= MESHY_SMART_TOPOLOGY_MAX_TARGET_POLYCOUNT
            )
        ):
            raise ValueError(
                "Meshy Smart Topology target polycount must be between "
                f"{MESHY_SMART_TOPOLOGY_MIN_TARGET_POLYCOUNT} and "
                f"{MESHY_SMART_TOPOLOGY_MAX_TARGET_POLYCOUNT}."
            )
        if self.surface_texture_provider not in SURFACE_TEXTURE_PROVIDERS:
            raise ValueError(
                "Unknown surface texture provider: "
                f"{self.surface_texture_provider!r}."
            )
        if (
            self.fullscreen_3d_viewer_screen_id is not None
            and not isinstance(self.fullscreen_3d_viewer_screen_id, str)
        ):
            raise ValueError(
                "Fullscreen 3D viewer screen ID must be a string or None."
            )
        object.__setattr__(
            self,
            "fullscreen_3d_viewer_screen_id",
            _normalize_fullscreen_3d_viewer_screen_id(
                self.fullscreen_3d_viewer_screen_id
            ),
        )
        if (
            self.jobs_window_screen_id is not None
            and not isinstance(self.jobs_window_screen_id, str)
        ):
            raise ValueError("Jobs window screen ID must be a string or None.")
        object.__setattr__(
            self,
            "jobs_window_screen_id",
            _normalize_fullscreen_3d_viewer_screen_id(
                self.jobs_window_screen_id
            ),
        )
        normalized_hotkey = _normalize_canvas_3d_navigation_toggle_hotkey(
            self.canvas_3d_navigation_toggle_hotkey
        )
        if normalized_hotkey is None:
            raise ValueError(
                "Canvas 3D navigation toggle hotkey must be one valid "
                "key combination."
            )
        object.__setattr__(
            self,
            "canvas_3d_navigation_toggle_hotkey",
            normalized_hotkey,
        )
        normalized_mesh_edit_delay = (
            _normalize_mesh_edit_update_delay_seconds(
                self.mesh_edit_update_delay_seconds
            )
        )
        if normalized_mesh_edit_delay is None:
            raise ValueError(
                "Mesh edit update delay must be between "
                f"{MIN_MESH_EDIT_UPDATE_DELAY_SECONDS} and "
                f"{MAX_MESH_EDIT_UPDATE_DELAY_SECONDS} seconds."
            )
        object.__setattr__(
            self,
            "mesh_edit_update_delay_seconds",
            normalized_mesh_edit_delay,
        )

    @property
    def has_meshy_api_key(self) -> bool:
        return bool(self.meshy_api_key)

    @property
    def has_openai_api_key(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def surface_texture_api_key(self) -> str:
        if self.surface_texture_provider == SURFACE_TEXTURE_PROVIDER_MESHY:
            return self.meshy_api_key
        return self.openai_api_key


# ### Settings widget ###
class SettingsWidget(QWidget):
    """Edits generation preferences in the application JSON settings file."""

    settings_changed = Signal()

    def __init__(
        self,
        application_settings: ApplicationSettingsStore | None = None,
        environment: Mapping[str, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._application_settings = (
            application_settings
            if application_settings is not None
            else ApplicationSettingsStore()
        )
        self._is_loading_settings = False
        self._is_disposed = False
        self._screen_signals_connected = False
        self._screen_application = _get_gui_application()
        environment_values = os.environ if environment is None else environment
        self._environment_meshy_api_key = str(
            environment_values.get(
                MESHY_API_KEY_ENVIRONMENT_VARIABLE,
                "",
            )
        ).strip()
        self._environment_openai_api_key = str(
            environment_values.get(
                OPENAI_API_KEY_ENVIRONMENT_VARIABLE,
                "",
            )
        ).strip()

        self._build_ui()
        self._load_settings()
        self._sync_key_status_labels()

    def get_settings(self) -> GenerationServiceSettings:
        return GenerationServiceSettings(
            meshy_api_key=(
                self.meshy_api_key_edit.text().strip()
                or self._environment_meshy_api_key
            ),
            openai_api_key=(
                self.openai_api_key_edit.text().strip()
                or self._environment_openai_api_key
            ),
            surface_texture_provider=read_surface_texture_provider(
                self._application_settings
            ),
            fullscreen_3d_viewer_screen_id=(
                self._selected_fullscreen_3d_viewer_screen_id()
            ),
            jobs_window_screen_id=self._selected_jobs_window_screen_id(),
            automatic_atlas_texture_sort_by_pbr=(
                self.automatic_atlas_texture_sort_by_pbr_checkbox.isChecked()
            ),
            automatic_atlas_texture_resolution=int(
                self.automatic_atlas_texture_resolution_combo.currentData()
            ),
            canvas_3d_navigation_toggle_hotkey=(
                self._selected_canvas_3d_navigation_toggle_hotkey()
            ),
            unused_face_removal=self.unused_face_removal_checkbox.isChecked(),
            use_uv_raycast_for_object_generation=(
                self.use_uv_raycast_for_object_generation_checkbox.isChecked()
            ),
            minimum_face_visibility_percentage=(
                self.minimum_face_visibility_percentage_spinbox.value()
            ),
            mesh_edit_update_delay_seconds=(
                self.mesh_edit_update_delay_spinbox.value()
            ),
        )

    def get_fullscreen_3d_viewer_screen_id(self) -> str | None:
        """Return the selected display without rereading the settings file."""

        return self._selected_fullscreen_3d_viewer_screen_id()

    def get_jobs_window_screen_id(self) -> str | None:
        """Return the selected Jobs-window display without rereading disk."""

        return self._selected_jobs_window_screen_id()

    def clear_session_keys(self) -> None:
        """Clear the temporary plaintext key values from settings.json."""

        self.meshy_api_key_edit.clear()
        self.openai_api_key_edit.clear()

    def dispose(self) -> None:
        """Disconnect application-wide display signals exactly once."""

        if self._is_disposed:
            return
        self._is_disposed = True
        if (
            self._screen_application is None
            or not self._screen_signals_connected
        ):
            return
        for signal in (
            self._screen_application.screenAdded,
            self._screen_application.screenRemoved,
            self._screen_application.primaryScreenChanged,
        ):
            try:
                signal.disconnect(self._handle_connected_screens_changed)
            except (RuntimeError, TypeError):
                pass
        self._screen_signals_connected = False

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(14)

        title_label = QLabel("Generation settings")
        title_label.setStyleSheet("font-size: 20px; font-weight: 600;")
        root_layout.addWidget(title_label)

        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        self.meshy_api_key_edit = self._build_secret_input(
            "Meshy AI API key"
        )
        self.meshy_api_key_edit.setObjectName("meshy_api_key_edit")
        self.meshy_api_key_edit.textChanged.connect(
            self._handle_secret_text_changed
        )
        form_layout.addRow("Meshy AI API key", self.meshy_api_key_edit)

        self.meshy_key_status_label = QLabel()
        self.meshy_key_status_label.setObjectName("meshy_key_status_label")
        form_layout.addRow("", self.meshy_key_status_label)

        self.openai_api_key_edit = self._build_secret_input(
            "OpenAI API key"
        )
        self.openai_api_key_edit.setObjectName("openai_api_key_edit")
        self.openai_api_key_edit.textChanged.connect(
            self._handle_secret_text_changed
        )
        form_layout.addRow("OpenAI API key", self.openai_api_key_edit)

        self.openai_key_status_label = QLabel()
        self.openai_key_status_label.setObjectName("openai_key_status_label")
        form_layout.addRow("", self.openai_key_status_label)

        self.fullscreen_3d_viewer_screen_combo = QComboBox()
        self.fullscreen_3d_viewer_screen_combo.setObjectName(
            "fullscreen_3d_viewer_screen_combo"
        )
        self.fullscreen_3d_viewer_screen_combo.currentIndexChanged.connect(
            self._handle_fullscreen_3d_viewer_screen_changed
        )
        form_layout.addRow(
            "Fullscreen 3D viewer display",
            self.fullscreen_3d_viewer_screen_combo,
        )

        self.jobs_window_screen_combo = QComboBox()
        self.jobs_window_screen_combo.setObjectName(
            "jobs_window_screen_combo"
        )
        self.jobs_window_screen_combo.setToolTip(
            "Choose the display where the detached Jobs window opens."
        )
        self.jobs_window_screen_combo.currentIndexChanged.connect(
            self._handle_jobs_window_screen_changed
        )
        form_layout.addRow(
            "Jobs window display",
            self.jobs_window_screen_combo,
        )

        self.automatic_atlas_texture_sort_by_pbr_checkbox = QCheckBox()
        self.automatic_atlas_texture_sort_by_pbr_checkbox.setObjectName(
            "automatic_atlas_texture_sort_by_pbr_checkbox"
        )
        self.automatic_atlas_texture_sort_by_pbr_checkbox.setToolTip(
            "Keep automatically assigned base-color-only textures out of "
            "Atlases containing generated PBR maps."
        )
        self.automatic_atlas_texture_sort_by_pbr_checkbox.toggled.connect(
            self._handle_automatic_atlas_texture_sort_by_pbr_changed
        )
        form_layout.addRow(
            "Automatic Atlas texture sort by PBR",
            self.automatic_atlas_texture_sort_by_pbr_checkbox,
        )

        self.automatic_atlas_texture_resolution_combo = QComboBox()
        self.automatic_atlas_texture_resolution_combo.setObjectName(
            "automatic_atlas_texture_resolution_combo"
        )
        self.automatic_atlas_texture_resolution_combo.setToolTip(
            "Choose the resolution used when a newly placed object or "
            "textured surface is first added to the selected Atlas. Existing "
            "Atlas placements are not resized."
        )
        for resolution in AUTOMATIC_ATLAS_TEXTURE_RESOLUTIONS:
            self.automatic_atlas_texture_resolution_combo.addItem(
                f"{resolution} x {resolution}",
                resolution,
            )
        self.automatic_atlas_texture_resolution_combo.currentIndexChanged.connect(
            self._handle_automatic_atlas_texture_resolution_changed
        )
        form_layout.addRow(
            "Automatic Atlas texture resolution",
            self.automatic_atlas_texture_resolution_combo,
        )

        self.canvas_3d_navigation_toggle_hotkey_edit = QKeySequenceEdit()
        self.canvas_3d_navigation_toggle_hotkey_edit.setObjectName(
            "canvas_3d_navigation_toggle_hotkey_edit"
        )
        self.canvas_3d_navigation_toggle_hotkey_edit.setMaximumSequenceLength(1)
        self.canvas_3d_navigation_toggle_hotkey_edit.setClearButtonEnabled(True)
        self.canvas_3d_navigation_toggle_hotkey_edit.setToolTip(
            "Press one key combination to switch the Canvas 3D view between "
            "top-down orbit and first-person navigation. Bare Z, Q, S, D, R, "
            "and F are reserved for first-person movement."
        )
        self.canvas_3d_navigation_toggle_hotkey_edit.keySequenceChanged.connect(
            self._handle_canvas_3d_navigation_toggle_hotkey_changed
        )
        form_layout.addRow(
            "Canvas 3D navigation hotkey",
            self.canvas_3d_navigation_toggle_hotkey_edit,
        )

        self.mesh_edit_update_delay_spinbox = QDoubleSpinBox()
        self.mesh_edit_update_delay_spinbox.setObjectName(
            "mesh_edit_update_delay_spinbox"
        )
        self.mesh_edit_update_delay_spinbox.setRange(
            MIN_MESH_EDIT_UPDATE_DELAY_SECONDS,
            MAX_MESH_EDIT_UPDATE_DELAY_SECONDS,
        )
        self.mesh_edit_update_delay_spinbox.setDecimals(1)
        self.mesh_edit_update_delay_spinbox.setSingleStep(
            MESH_EDIT_UPDATE_DELAY_STEP_SECONDS
        )
        self.mesh_edit_update_delay_spinbox.setSuffix(" s")
        self.mesh_edit_update_delay_spinbox.setKeyboardTracking(False)
        self.mesh_edit_update_delay_spinbox.setToolTip(
            "Wait this long after releasing a doorway or window edit before "
            "rebuilding the Canvas 3D wall mesh."
        )
        self.mesh_edit_update_delay_spinbox.valueChanged.connect(
            self._handle_mesh_edit_update_delay_changed
        )
        form_layout.addRow(
            "Mesh edit update delay",
            self.mesh_edit_update_delay_spinbox,
        )

        self.unused_face_removal_checkbox = QCheckBox()
        self.unused_face_removal_checkbox.setObjectName(
            "unused_face_removal_checkbox"
        )
        self.unused_face_removal_checkbox.setToolTip(
            "Generate geometry first, remove faces that do not meet the "
            "configured minimum visibility in the six Object-generation "
            "cameras, then submit the edited GLB to Meshy Retexture. This "
            "uses two Meshy tasks."
        )
        self.unused_face_removal_checkbox.toggled.connect(
            self._handle_unused_face_removal_changed
        )
        form_layout.addRow(
            "Unused face removal",
            self.unused_face_removal_checkbox,
        )

        self.use_uv_raycast_for_object_generation_checkbox = QCheckBox()
        self.use_uv_raycast_for_object_generation_checkbox.setObjectName(
            "use_uv_raycast_for_object_generation_checkbox"
        )
        self.use_uv_raycast_for_object_generation_checkbox.setToolTip(
            "After Meshy textures the model, rebuild its UVs from the six "
            "weighted Object-generation cameras and copy the existing "
            "texture continuously into the new layout. This first version "
            "packs without island spacing."
        )
        self.use_uv_raycast_for_object_generation_checkbox.toggled.connect(
            self._handle_use_uv_raycast_for_object_generation_changed
        )
        form_layout.addRow(
            "Use weighted camera projection",
            self.use_uv_raycast_for_object_generation_checkbox,
        )

        self.minimum_face_visibility_percentage_spinbox = QSpinBox()
        self.minimum_face_visibility_percentage_spinbox.setObjectName(
            "minimum_face_visibility_percentage_spinbox"
        )
        self.minimum_face_visibility_percentage_spinbox.setRange(
            MINIMUM_FACE_VISIBILITY_PERCENTAGE,
            MAXIMUM_FACE_VISIBILITY_PERCENTAGE,
        )
        self.minimum_face_visibility_percentage_spinbox.setSingleStep(1)
        self.minimum_face_visibility_percentage_spinbox.setSuffix("%")
        self.minimum_face_visibility_percentage_spinbox.setKeyboardTracking(
            False
        )
        self.minimum_face_visibility_percentage_spinbox.setToolTip(
            "During unused-face removal, retain a face from a camera only "
            "when at least this percentage of the face is visible. Higher "
            "values reject tiny glimpses through gaps."
        )
        self.minimum_face_visibility_percentage_spinbox.valueChanged.connect(
            self._handle_minimum_face_visibility_percentage_changed
        )
        form_layout.addRow(
            "Minimum percentage of face visible",
            self.minimum_face_visibility_percentage_spinbox,
        )

        root_layout.addLayout(form_layout)

        security_note = QLabel(
            "Testing mode: API keys are stored as plaintext in the local "
            "HouseMaker settings.json file, but never in project JSON files. "
            "Leave either field blank to use MESHY_API_KEY or OPENAI_API_KEY "
            "from the environment. "
            "Remove the saved value before sharing this PC."
        )
        security_note.setObjectName("api_key_security_note")
        security_note.setWordWrap(True)
        security_note.setStyleSheet("color: #666;")
        root_layout.addWidget(security_note)

        meshy_note = QLabel(
            "Object generation uses Meshy Image-to-3D. Surface texture "
            "generation can use Meshy or an OpenAI vision model; choose its "
            "provider in the Surface texture generation tab. GPT-4o-mini "
            "first analyzes the references, then GPT Image 2 renders the "
            "texture, so that choice makes two OpenAI requests. Provider "
            "requests may consume account credits."
        )
        meshy_note.setObjectName("meshy_availability_note")
        meshy_note.setWordWrap(True)
        meshy_note.setStyleSheet("color: #666;")
        root_layout.addWidget(meshy_note)
        root_layout.addStretch(1)

        self._connect_screen_change_signals()

    @staticmethod
    def _build_secret_input(placeholder_text: str) -> QLineEdit:
        line_edit = QLineEdit()
        line_edit.setEchoMode(QLineEdit.EchoMode.Password)
        line_edit.setClearButtonEnabled(True)
        line_edit.setPlaceholderText(placeholder_text)
        return line_edit

    def _load_settings(self) -> None:
        self._is_loading_settings = True
        self.meshy_api_key_edit.setText(
            str(
                self._application_settings.get(
                    MESHY_API_KEY_SETTING_KEY,
                    "",
                )
                or ""
            )
        )
        self.openai_api_key_edit.setText(
            str(
                self._application_settings.get(
                    OPENAI_API_KEY_SETTING_KEY,
                    "",
                )
                or ""
            )
        )
        self._refresh_fullscreen_3d_viewer_screen_options()
        self._refresh_jobs_window_screen_options()
        self.automatic_atlas_texture_sort_by_pbr_checkbox.setChecked(
            read_automatic_atlas_texture_sort_by_pbr(
                self._application_settings
            )
        )
        automatic_atlas_resolution = read_automatic_atlas_texture_resolution(
            self._application_settings
        )
        automatic_atlas_index = (
            self.automatic_atlas_texture_resolution_combo.findData(
                automatic_atlas_resolution
            )
        )
        self.automatic_atlas_texture_resolution_combo.setCurrentIndex(
            max(0, automatic_atlas_index)
        )
        self.canvas_3d_navigation_toggle_hotkey_edit.setKeySequence(
            QKeySequence(
                read_canvas_3d_navigation_toggle_hotkey(
                    self._application_settings
                ),
                QKeySequence.SequenceFormat.PortableText,
            )
        )
        self.mesh_edit_update_delay_spinbox.setValue(
            read_mesh_edit_update_delay_seconds(
                self._application_settings
            )
        )
        self.unused_face_removal_checkbox.setChecked(
            read_unused_face_removal(self._application_settings)
        )
        self.use_uv_raycast_for_object_generation_checkbox.setChecked(
            read_use_uv_raycast_for_object_generation(
                self._application_settings
            )
        )
        self.minimum_face_visibility_percentage_spinbox.setValue(
            read_minimum_face_visibility_percentage(
                self._application_settings
            )
        )
        self._is_loading_settings = False

    def _handle_secret_text_changed(self, _text: str) -> None:
        if self._is_loading_settings:
            return
        self._application_settings.set(
            MESHY_API_KEY_SETTING_KEY,
            self.meshy_api_key_edit.text().strip(),
        )
        self._application_settings.set(
            OPENAI_API_KEY_SETTING_KEY,
            self.openai_api_key_edit.text().strip(),
        )
        self._sync_key_status_labels()
        self.settings_changed.emit()

    def _connect_screen_change_signals(self) -> None:
        if (
            self._screen_application is None
            or self._screen_signals_connected
        ):
            return
        self._screen_application.screenAdded.connect(
            self._handle_connected_screens_changed
        )
        self._screen_application.screenRemoved.connect(
            self._handle_connected_screens_changed
        )
        self._screen_application.primaryScreenChanged.connect(
            self._handle_connected_screens_changed
        )
        self._screen_signals_connected = True

    def _handle_connected_screens_changed(
        self,
        _screen: QScreen | None,
    ) -> None:
        if self._is_disposed:
            return
        self._refresh_fullscreen_3d_viewer_screen_options()
        self._refresh_jobs_window_screen_options()
        if not self._is_loading_settings:
            # Reapply even unchanged "Primary display" selections because the
            # primary screen itself may have changed or been disconnected.
            self.settings_changed.emit()

    def _refresh_fullscreen_3d_viewer_screen_options(self) -> None:
        selected_screen_id = read_fullscreen_3d_viewer_screen_id(
            self._application_settings
        )
        blocker = QSignalBlocker(self.fullscreen_3d_viewer_screen_combo)
        self.fullscreen_3d_viewer_screen_combo.clear()
        self.fullscreen_3d_viewer_screen_combo.addItem("None", None)
        for option in connected_fullscreen_3d_viewer_display_options():
            self.fullscreen_3d_viewer_screen_combo.addItem(
                option.label,
                option.screen_id,
            )
        selected_index = self.fullscreen_3d_viewer_screen_combo.findData(
            selected_screen_id
        )
        self.fullscreen_3d_viewer_screen_combo.setCurrentIndex(
            selected_index if selected_index >= 0 else 0
        )
        del blocker

    def _selected_fullscreen_3d_viewer_screen_id(self) -> str | None:
        return _normalize_fullscreen_3d_viewer_screen_id(
            self.fullscreen_3d_viewer_screen_combo.currentData()
        )

    def _refresh_jobs_window_screen_options(self) -> None:
        selected_screen_id = read_jobs_window_screen_id(
            self._application_settings
        )
        blocker = QSignalBlocker(self.jobs_window_screen_combo)
        self.jobs_window_screen_combo.clear()
        self.jobs_window_screen_combo.addItem("Primary display", None)
        for option in connected_fullscreen_3d_viewer_display_options():
            self.jobs_window_screen_combo.addItem(
                option.label,
                option.screen_id,
            )
        selected_index = self.jobs_window_screen_combo.findData(
            selected_screen_id
        )
        self.jobs_window_screen_combo.setCurrentIndex(
            selected_index if selected_index >= 0 else 0
        )
        del blocker

    def _selected_jobs_window_screen_id(self) -> str | None:
        return _normalize_fullscreen_3d_viewer_screen_id(
            self.jobs_window_screen_combo.currentData()
        )

    def _handle_fullscreen_3d_viewer_screen_changed(
        self,
        _index: int,
    ) -> None:
        if self._is_loading_settings:
            return
        self._application_settings.set(
            FULLSCREEN_3D_VIEWER_SCREEN_SETTING_KEY,
            self._selected_fullscreen_3d_viewer_screen_id(),
        )
        self.settings_changed.emit()

    def _handle_jobs_window_screen_changed(self, _index: int) -> None:
        if self._is_loading_settings:
            return
        self._application_settings.set(
            JOBS_WINDOW_SCREEN_SETTING_KEY,
            self._selected_jobs_window_screen_id(),
        )
        self.settings_changed.emit()

    def _handle_automatic_atlas_texture_resolution_changed(
        self,
        _index: int,
    ) -> None:
        """Persist the resolution used for future automatic Atlas entries."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            AUTOMATIC_ATLAS_TEXTURE_RESOLUTION_SETTING_KEY,
            int(self.automatic_atlas_texture_resolution_combo.currentData()),
        )
        self.settings_changed.emit()

    def _handle_automatic_atlas_texture_sort_by_pbr_changed(
        self,
        checked: bool,
    ) -> None:
        """Persist whether automatic Atlas placement separates non-PBR maps."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            AUTOMATIC_ATLAS_TEXTURE_SORT_BY_PBR_SETTING_KEY,
            bool(checked),
        )
        self.settings_changed.emit()

    def _selected_canvas_3d_navigation_toggle_hotkey(self) -> str:
        hotkey = _hotkey_from_key_sequence(
            self.canvas_3d_navigation_toggle_hotkey_edit.keySequence()
        )
        return hotkey or DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY

    def _handle_canvas_3d_navigation_toggle_hotkey_changed(
        self,
        key_sequence: QKeySequence,
    ) -> None:
        if self._is_loading_settings:
            return
        hotkey = _hotkey_from_key_sequence(key_sequence)
        if hotkey is None:
            hotkey = DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY
            blocker = QSignalBlocker(
                self.canvas_3d_navigation_toggle_hotkey_edit
            )
            self.canvas_3d_navigation_toggle_hotkey_edit.setKeySequence(
                QKeySequence(
                    hotkey,
                    QKeySequence.SequenceFormat.PortableText,
                )
            )
            del blocker
        self._application_settings.set(
            CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY_SETTING_KEY,
            hotkey,
        )
        self.settings_changed.emit()

    def _handle_unused_face_removal_changed(self, enabled: bool) -> None:
        if self._is_loading_settings:
            return
        self._application_settings.set(
            UNUSED_FACE_REMOVAL_SETTING_KEY,
            bool(enabled),
        )
        self.settings_changed.emit()

    def _handle_use_uv_raycast_for_object_generation_changed(
        self,
        enabled: bool,
    ) -> None:
        if self._is_loading_settings:
            return
        self._application_settings.set(
            USE_UV_RAYCAST_FOR_OBJECT_GENERATION_SETTING_KEY,
            bool(enabled),
        )
        self.settings_changed.emit()

    def _handle_minimum_face_visibility_percentage_changed(
        self,
        value: int,
    ) -> None:
        if self._is_loading_settings:
            return
        self._application_settings.set(
            MINIMUM_FACE_VISIBILITY_PERCENTAGE_SETTING_KEY,
            int(value),
        )
        self.settings_changed.emit()

    def _handle_mesh_edit_update_delay_changed(self, value: float) -> None:
        if self._is_loading_settings:
            return
        self._application_settings.set(
            MESH_EDIT_UPDATE_DELAY_SECONDS_SETTING_KEY,
            float(value),
        )
        self.settings_changed.emit()

    def _sync_key_status_labels(self) -> None:
        self.meshy_key_status_label.setText(
            self._build_key_status_text(
                bool(self.meshy_api_key_edit.text().strip()),
                bool(self._environment_meshy_api_key),
                MESHY_API_KEY_ENVIRONMENT_VARIABLE,
            )
        )
        self.openai_key_status_label.setText(
            self._build_key_status_text(
                bool(self.openai_api_key_edit.text().strip()),
                bool(self._environment_openai_api_key),
                OPENAI_API_KEY_ENVIRONMENT_VARIABLE,
            )
        )

    @staticmethod
    def _build_key_status_text(
        has_session_value: bool,
        has_environment_value: bool,
        environment_variable: str,
    ) -> str:
        if has_session_value:
            return "Using the key saved in settings.json"
        if has_environment_value:
            return f"Using {environment_variable} from the environment"
        return "No key configured"


# ### Fullscreen display helpers ###
def connected_fullscreen_3d_viewer_display_options(
) -> tuple[Fullscreen3DViewerScreenOption, ...]:
    """Return the currently connected displays that can host the 3D view."""

    return _fullscreen_3d_viewer_display_options(_connected_screens())


def resolve_fullscreen_3d_viewer_screen(
    screen_id: str | None,
) -> QScreen | None:
    """Resolve a persisted display identity to a currently connected screen."""

    normalized_screen_id = _normalize_fullscreen_3d_viewer_screen_id(screen_id)
    if normalized_screen_id is None:
        return None
    screens = _connected_screens()
    options = _fullscreen_3d_viewer_display_options(screens)
    for screen, option in zip(screens, options, strict=True):
        if option.screen_id == normalized_screen_id:
            return screen
    return None


def fullscreen_3d_viewer_screen_id(screen: QScreen) -> str | None:
    """Return a screen identifier stable across ordinary geometry changes."""

    serial_number = _read_screen_text(screen, "serialNumber")
    if serial_number and serial_number.casefold() not in {"0", "unknown", "n/a"}:
        manufacturer = _read_screen_text(screen, "manufacturer")
        model = _read_screen_text(screen, "model")
        return "monitor:" + "|".join(
            (manufacturer or "-", model or "-", serial_number)
        )

    screen_name = _read_screen_text(screen, "name")
    if screen_name:
        return f"screen:{screen_name}"
    return None


def read_fullscreen_3d_viewer_screen_id(
    application_settings: ApplicationSettingsStore,
) -> str | None:
    """Read a saved display identity while safely ignoring malformed values."""

    return _normalize_fullscreen_3d_viewer_screen_id(
        application_settings.get(FULLSCREEN_3D_VIEWER_SCREEN_SETTING_KEY)
    )


def read_jobs_window_screen_id(
    application_settings: ApplicationSettingsStore,
) -> str | None:
    """Read the persisted detached Jobs-window display identity."""

    return _normalize_fullscreen_3d_viewer_screen_id(
        application_settings.get(JOBS_WINDOW_SCREEN_SETTING_KEY)
    )


# ### Atlas setting helpers ###
def read_automatic_atlas_texture_sort_by_pbr(
    application_settings: ApplicationSettingsStore,
) -> bool:
    """Read the persisted PBR-sorting policy with a safe default."""

    value = application_settings.get(
        AUTOMATIC_ATLAS_TEXTURE_SORT_BY_PBR_SETTING_KEY,
        DEFAULT_AUTOMATIC_ATLAS_TEXTURE_SORT_BY_PBR,
    )
    if isinstance(value, bool):
        return value
    return DEFAULT_AUTOMATIC_ATLAS_TEXTURE_SORT_BY_PBR


def read_automatic_atlas_texture_resolution(
    application_settings: ApplicationSettingsStore,
) -> int:
    """Read the automatic Atlas resolution with a safe supported default."""

    value = application_settings.get(
        AUTOMATIC_ATLAS_TEXTURE_RESOLUTION_SETTING_KEY,
        DEFAULT_AUTOMATIC_ATLAS_TEXTURE_RESOLUTION,
    )
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in AUTOMATIC_ATLAS_TEXTURE_RESOLUTIONS
    ):
        return DEFAULT_AUTOMATIC_ATLAS_TEXTURE_RESOLUTION
    return int(value)


# ### Object post-processing setting helpers ###
def read_unused_face_removal(
    application_settings: ApplicationSettingsStore,
) -> bool:
    """Read the persisted staged-generation option with a safe default."""

    value = application_settings.get(
        UNUSED_FACE_REMOVAL_SETTING_KEY,
        DEFAULT_UNUSED_FACE_REMOVAL,
    )
    return value if isinstance(value, bool) else DEFAULT_UNUSED_FACE_REMOVAL


def read_use_uv_raycast_for_object_generation(
    application_settings: ApplicationSettingsStore,
) -> bool:
    """Read the persisted weighted-projection option safely."""

    value = application_settings.get(
        USE_UV_RAYCAST_FOR_OBJECT_GENERATION_SETTING_KEY,
        DEFAULT_USE_UV_RAYCAST_FOR_OBJECT_GENERATION,
    )
    if isinstance(value, bool):
        return value
    return DEFAULT_USE_UV_RAYCAST_FOR_OBJECT_GENERATION


def read_minimum_face_visibility_percentage(
    application_settings: ApplicationSettingsStore,
) -> int:
    """Read the camera face-visibility threshold with a safe default."""

    value = application_settings.get(
        MINIMUM_FACE_VISIBILITY_PERCENTAGE_SETTING_KEY,
        DEFAULT_MINIMUM_FACE_VISIBILITY_PERCENTAGE,
    )
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not (
            MINIMUM_FACE_VISIBILITY_PERCENTAGE
            <= value
            <= MAXIMUM_FACE_VISIBILITY_PERCENTAGE
        )
    ):
        return DEFAULT_MINIMUM_FACE_VISIBILITY_PERCENTAGE
    return int(value)


# ### Mesh edit preview setting helpers ###
def read_mesh_edit_update_delay_seconds(
    application_settings: ApplicationSettingsStore,
) -> float:
    """Read the Canvas mesh-edit debounce delay with a safe default."""

    normalized_delay = _normalize_mesh_edit_update_delay_seconds(
        application_settings.get(
            MESH_EDIT_UPDATE_DELAY_SECONDS_SETTING_KEY,
            DEFAULT_MESH_EDIT_UPDATE_DELAY_SECONDS,
        )
    )
    if normalized_delay is None:
        return DEFAULT_MESH_EDIT_UPDATE_DELAY_SECONDS
    return normalized_delay


def _normalize_mesh_edit_update_delay_seconds(
    value: object,
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    normalized_value = float(value)
    if not math.isfinite(normalized_value):
        return None
    if not (
        MIN_MESH_EDIT_UPDATE_DELAY_SECONDS
        <= normalized_value
        <= MAX_MESH_EDIT_UPDATE_DELAY_SECONDS
    ):
        return None
    return normalized_value


# ### Canvas navigation hotkey helpers ###
def read_canvas_3d_navigation_toggle_hotkey(
    application_settings: ApplicationSettingsStore,
) -> str:
    """Read a valid Canvas navigation shortcut, with a safe default."""

    hotkey = _normalize_canvas_3d_navigation_toggle_hotkey(
        application_settings.get(CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY_SETTING_KEY)
    )
    return hotkey or DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY


def _get_gui_application() -> QGuiApplication | None:
    application = QGuiApplication.instance()
    if isinstance(application, QGuiApplication):
        return application
    return None


def _connected_screens() -> tuple[QScreen, ...]:
    application = _get_gui_application()
    if application is None:
        return ()
    try:
        return tuple(application.screens())
    except RuntimeError:
        return ()


def _fullscreen_3d_viewer_display_options(
    screens: tuple[QScreen, ...],
) -> tuple[Fullscreen3DViewerScreenOption, ...]:
    options: list[Fullscreen3DViewerScreenOption] = []
    used_screen_ids: set[str] = set()
    used_labels: set[str] = set()
    for index, screen in enumerate(screens, start=1):
        screen_id = fullscreen_3d_viewer_screen_id(screen)
        if screen_id is None or screen_id in used_screen_ids:
            screen_name = _read_screen_text(screen, "name")
            screen_id = f"screen:{screen_name or f'connected-display-{index}'}"
        if screen_id in used_screen_ids:
            screen_id = f"{screen_id}|{index}"
        label = _fullscreen_3d_viewer_screen_label(screen, index)
        if label in used_labels:
            label = f"{label} ({index})"
        options.append(
            Fullscreen3DViewerScreenOption(
                screen_id=screen_id,
                label=label,
            )
        )
        used_screen_ids.add(screen_id)
        used_labels.add(label)
    return tuple(options)


def _fullscreen_3d_viewer_screen_label(screen: QScreen, index: int) -> str:
    screen_name = _read_screen_text(screen, "name") or f"Display {index}"
    try:
        geometry = screen.geometry()
        width = int(geometry.width())
        height = int(geometry.height())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return screen_name
    if width <= 0 or height <= 0:
        return screen_name
    return f"{screen_name} ({width} × {height})"


def _read_screen_text(screen: QScreen, method_name: str) -> str:
    try:
        value = getattr(screen, method_name)()
    except (AttributeError, RuntimeError, TypeError):
        return ""
    return str(value or "").strip()


def _normalize_fullscreen_3d_viewer_screen_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized_value = value.strip()
    return normalized_value or None


def _normalize_canvas_3d_navigation_toggle_hotkey(
    value: object,
) -> str | None:
    if not isinstance(value, str):
        return None
    return _hotkey_from_key_sequence(
        QKeySequence(value.strip(), QKeySequence.SequenceFormat.PortableText)
    )


def _hotkey_from_key_sequence(key_sequence: QKeySequence) -> str | None:
    if key_sequence.count() != 1:
        return None
    key_combination = key_sequence[0]
    if key_combination.key() in _MODIFIER_ONLY_SHORTCUT_KEYS:
        return None
    if (
        key_combination.key() in _FIRST_PERSON_MOVEMENT_SHORTCUT_KEYS
        and key_combination.keyboardModifiers()
        == Qt.KeyboardModifier.NoModifier
    ):
        return None
    hotkey = key_sequence.toString(QKeySequence.SequenceFormat.PortableText)
    return hotkey.strip() or None


# ### Provider setting helpers ###
def read_surface_texture_provider(
    application_settings: ApplicationSettingsStore,
) -> str:
    """Return the persisted provider, falling back safely for stale values."""

    provider = str(
        application_settings.get(
            SURFACE_TEXTURE_PROVIDER_SETTING_KEY,
            SURFACE_TEXTURE_PROVIDER_MESHY,
        )
        or SURFACE_TEXTURE_PROVIDER_MESHY
    )
    if provider not in SURFACE_TEXTURE_PROVIDERS:
        return SURFACE_TEXTURE_PROVIDER_MESHY
    return provider
