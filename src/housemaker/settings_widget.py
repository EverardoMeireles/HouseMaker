# ### Imports ###
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence, QScreen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
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
CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY_SETTING_KEY = (
    "navigation/canvas_3d_navigation_toggle_hotkey"
)
UNUSED_FACE_REMOVAL_SETTING_KEY = "generation/unused_face_removal"
PROJECT_UVS_FROM_CAMERA_VIEWS_SETTING_KEY = (
    "generation/project_uvs_from_camera_views"
)
DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY = "N"
DEFAULT_UNUSED_FACE_REMOVAL = False
DEFAULT_PROJECT_UVS_FROM_CAMERA_VIEWS = False
MESHY_SMART_TOPOLOGY_MIN_TARGET_POLYCOUNT = 100
MESHY_SMART_TOPOLOGY_MAX_TARGET_POLYCOUNT = 15_000
DEFAULT_MESHY_TARGET_POLYCOUNT = 4_000
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
    project_uvs_from_camera_views: bool = (
        DEFAULT_PROJECT_UVS_FROM_CAMERA_VIEWS
    )

    def __post_init__(self) -> None:
        if not isinstance(self.unused_face_removal, bool):
            raise ValueError("Unused face removal must be enabled or disabled.")
        if not isinstance(self.project_uvs_from_camera_views, bool):
            raise ValueError(
                "Project UVs from camera views must be enabled or disabled."
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
            canvas_3d_navigation_toggle_hotkey=(
                self._selected_canvas_3d_navigation_toggle_hotkey()
            ),
            unused_face_removal=self.unused_face_removal_checkbox.isChecked(),
            project_uvs_from_camera_views=(
                self.project_uvs_from_camera_views_checkbox.isChecked()
            ),
        )

    def clear_session_keys(self) -> None:
        """Clear the temporary plaintext key values from settings.json."""

        self.meshy_api_key_edit.clear()
        self.openai_api_key_edit.clear()

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

        self.unused_face_removal_checkbox = QCheckBox()
        self.unused_face_removal_checkbox.setObjectName(
            "unused_face_removal_checkbox"
        )
        self.unused_face_removal_checkbox.setToolTip(
            "Generate geometry first, remove faces invisible from the enabled "
            "Object-generation cameras, then submit the edited GLB to Meshy "
            "Retexture. This uses two Meshy tasks."
        )
        self.unused_face_removal_checkbox.toggled.connect(
            self._handle_unused_face_removal_changed
        )
        form_layout.addRow(
            "Unused face removal",
            self.unused_face_removal_checkbox,
        )

        self.project_uvs_from_camera_views_checkbox = QCheckBox()
        self.project_uvs_from_camera_views_checkbox.setObjectName(
            "project_uvs_from_camera_views_checkbox"
        )
        self.project_uvs_from_camera_views_checkbox.setToolTip(
            "Generate geometry first, project the model's UVs using all six "
            "fixed camera views, pack the projected UV islands compactly, "
            "and place faces missed by every view in the bottom-left corner "
            "of the UV map before asking Meshy Retexture to preserve those "
            "UVs. This option uses all six cameras independently of the "
            "unused-face camera checkboxes and requires staged geometry and "
            "texturing tasks."
        )
        self.project_uvs_from_camera_views_checkbox.toggled.connect(
            self._handle_project_uvs_from_camera_views_changed
        )
        form_layout.addRow(
            "Project UVs from camera views",
            self.project_uvs_from_camera_views_checkbox,
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
        self.canvas_3d_navigation_toggle_hotkey_edit.setKeySequence(
            QKeySequence(
                read_canvas_3d_navigation_toggle_hotkey(
                    self._application_settings
                ),
                QKeySequence.SequenceFormat.PortableText,
            )
        )
        self.unused_face_removal_checkbox.setChecked(
            read_unused_face_removal(self._application_settings)
        )
        self.project_uvs_from_camera_views_checkbox.setChecked(
            read_project_uvs_from_camera_views(self._application_settings)
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
        if self._screen_application is None:
            return
        self._screen_application.screenAdded.connect(
            self._handle_connected_screens_changed
        )
        self._screen_application.screenRemoved.connect(
            self._handle_connected_screens_changed
        )

    def _handle_connected_screens_changed(self, _screen: QScreen) -> None:
        previous_screen_id = self._selected_fullscreen_3d_viewer_screen_id()
        self._refresh_fullscreen_3d_viewer_screen_options()
        if (
            not self._is_loading_settings
            and previous_screen_id
            != self._selected_fullscreen_3d_viewer_screen_id()
        ):
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

    def _handle_project_uvs_from_camera_views_changed(
        self,
        enabled: bool,
    ) -> None:
        if self._is_loading_settings:
            return
        self._application_settings.set(
            PROJECT_UVS_FROM_CAMERA_VIEWS_SETTING_KEY,
            bool(enabled),
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
    for screen in _connected_screens():
        if fullscreen_3d_viewer_screen_id(screen) == normalized_screen_id:
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


def read_project_uvs_from_camera_views(
    application_settings: ApplicationSettingsStore,
) -> bool:
    """Read the camera-projected UV option with a safe default."""

    value = application_settings.get(
        PROJECT_UVS_FROM_CAMERA_VIEWS_SETTING_KEY,
        DEFAULT_PROJECT_UVS_FROM_CAMERA_VIEWS,
    )
    if not isinstance(value, bool):
        return DEFAULT_PROJECT_UVS_FROM_CAMERA_VIEWS
    return value


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
