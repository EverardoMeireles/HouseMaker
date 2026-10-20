# ### Imports ###
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from uuid import uuid4

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QGuiApplication,
    QHideEvent,
    QKeySequence,
    QScreen,
    QShowEvent,
)
from PySide6.QtNetwork import (
    QNetworkAccessManager,
    QNetworkReply,
    QNetworkRequest,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.ceiling_height_estimation import (
    CEILING_HEIGHT_ESTIMATION_MODEL_SETTING_KEY,
    CEILING_HEIGHT_INFERENCE_MODEL,
    CEILING_HEIGHT_MODEL_OPTIONS,
)
from housemaker.first_person_navigation import (
    DEFAULT_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND,
    DEFAULT_FIRST_PERSON_NAVIGATION_MODE,
    FIRST_PERSON_MOVE_SPEED_STEP_METERS_PER_SECOND,
    FIRST_PERSON_NAVIGATION_MODE_OPTIONS,
    MAX_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND,
    MIN_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND,
    normalize_first_person_navigation_mode,
    normalize_first_person_speed_meters_per_second,
)
from housemaker.openai_model_pricing import (
    OPENAI_PRICING_MARKDOWN_URL,
    OPENAI_PRICING_RESPONSE_LIMIT_BYTES,
    OPENAI_PRICING_TIMEOUT_MILLISECONDS,
    ModelTokenPricing,
    format_plan_correction_model_label,
    parse_openai_pricing_markdown,
)
from housemaker.plan_correction_models import (
    DEFAULT_PLAN_CORRECTION_MODEL,
    PLAN_CORRECTION_MODEL_OPTIONS,
    PLAN_CORRECTION_MODELS,
    plan_correction_model_label,
)
from housemaker.reference_generation_models import (
    DEFAULT_REFERENCE_GENERATION_MODEL,
    REFERENCE_GENERATION_MODEL_OPTIONS,
    REFERENCE_GENERATION_MODEL_SETTINGS_KEY,
    REFERENCE_GENERATION_MODELS,
)
from housemaker.reference_symmetry_inference import (
    DEFAULT_SYMMETRY_INFERENCE_MODEL,
    SYMMETRY_INFERENCE_MODEL_OPTIONS,
)
from housemaker.reference_view_presets import (
    DEFAULT_REFERENCE_VIEW_PRESETS,
    ReferenceViewPreset,
    read_reference_view_presets,
    save_reference_view_presets,
)
from housemaker.wall_llm_verification_models import (
    DEFAULT_WALL_VERIFICATION_MODEL,
    WALL_VERIFICATION_MODEL_OPTIONS,
    WALL_VERIFICATION_MODELS,
)

# ### Constants ###
MESHY_API_KEY_ENVIRONMENT_VARIABLE = "MESHY_API_KEY"
MESHY_API_KEY_SETTING_KEY = "generation/meshy_api_key"
OPENAI_API_KEY_ENVIRONMENT_VARIABLE = "OPENAI_API_KEY"
OPENAI_API_KEY_SETTING_KEY = "generation/openai_api_key"
PLAN_CORRECTION_MODEL_SETTING_KEY = "canvas/plan_correction_model"
MAKE_WALLS_CONTINUOUS_SETTING_KEY = "canvas/make_walls_continuous"
SURFACE_TEXTURE_PROVIDER_SETTING_KEY = "generation/surface_texture_provider"
FULLSCREEN_3D_VIEWER_SCREEN_SETTING_KEY = (
    "display/fullscreen_3d_viewer_screen_id"
)
SCENE_3D_DISPLAY_SCREEN_SETTING_KEY = "display/scene_3d_screen_id"
GENERATION_DISPLAY_SCREEN_SETTING_KEY = "display/generation_screen_id"
JOBS_WINDOW_SCREEN_SETTING_KEY = "display/jobs_window_screen_id"
ATLAS_DISPLAY_SCREEN_SETTING_KEY = "display/atlas_screen_id"
AUTOMATIC_ATLAS_CREATION_SETTING_KEY = "atlas/automatic_creation"
AUTOMATIC_ATLAS_TEXTURE_SORT_BY_PBR_SETTING_KEY = (
    "atlas/automatic_texture_sort_by_pbr"
)
USE_HALF_MESH_TEXTURE_PREFIX_SETTING_KEY = (
    "atlas/use_half_mesh_texture_prefix"
)
AUTOMATIC_ATLAS_TEXTURE_RESOLUTION_SETTING_KEY = (
    "atlas/automatic_texture_resolution"
)
CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY_SETTING_KEY = (
    "navigation/canvas_3d_navigation_toggle_hotkey"
)
CLEAR_MASK_HOTKEY_SETTING_KEY = "generation/clear_mask_hotkey"
ENABLE_OBJECT_SYMMETRY_INFERENCE_SETTING_KEY = (
    "generation/enable_object_symmetry_inference"
)
SYMMETRY_INFERENCE_MODEL_SETTING_KEY = "generation/symmetry_inference_model"
FIRST_PERSON_NAVIGATION_MODE_SETTING_KEY = (
    "navigation/first_person_navigation_mode"
)
FIRST_PERSON_SPEED_SETTING_KEY = "navigation/first_person_speed"
IGNORE_TOP_DOWN_CEILING_SETTING_KEY = "navigation/ignore_top_down_ceiling"
MAKE_FLOORS_DOUBLE_SIDED_SETTING_KEY = "canvas/make_floors_double_sided"
MAKE_CEILINGS_DOUBLE_SIDED_SETTING_KEY = (
    "canvas/make_ceilings_double_sided"
)
HIDE_STAIR_MESH_WHEN_PREVIEWING_SETTING_KEY = (
    "canvas/hide_stair_mesh_when_previewing"
)
UNUSED_FACE_REMOVAL_SETTING_KEY = "generation/unused_face_removal"
USE_UV_RAYCAST_FOR_OBJECT_GENERATION_SETTING_KEY = (
    "generation/use_uv_raycast_for_object_generation"
)
MINIMUM_FACE_VISIBILITY_PERCENTAGE_SETTING_KEY = (
    "generation/minimum_face_visibility_percentage"
)
DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY_SETTING_KEY = (
    "preview/doors_and_windows_point_light_intensity"
)
MESH_EDIT_UPDATE_DELAY_SECONDS_SETTING_KEY = (
    # Keep the original persisted key so existing preferences remain valid.
    "canvas/doorway_mesh_update_delay_seconds"
)
SNAP_MIDDLE_EQUAL_ANGLE_ONLY_SETTING_KEY = (
    "canvas/snap_middle_equal_angle_only"
)
CUT_CONNECTION_ON_DOUBLE_VERTEX_DELETION_SETTING_KEY = (
    "canvas/cut_connection_on_double_vertex_deletion"
)
INCLUDE_LLM_WALL_VERIFICATION_SETTING_KEY = (
    "canvas/include_llm_wall_verification"
)
WALL_VERIFICATION_MODEL_SETTING_KEY = "canvas/wall_verification_model"
CEILING_HEIGHT_ESTIMATION_MODELS = frozenset(
    model_id for _label, model_id in CEILING_HEIGHT_MODEL_OPTIONS
)
DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY = "N"
DEFAULT_CLEAR_MASK_HOTKEY = "Ctrl+Shift+M"
CLEAR_MASK_HOTKEY_OPTIONS = (
    ("Off", ""),
    ("Ctrl+Shift+M", "Ctrl+Shift+M"),
    ("Ctrl+Shift+C", "Ctrl+Shift+C"),
    ("Ctrl+Alt+M", "Ctrl+Alt+M"),
    ("Ctrl+Alt+C", "Ctrl+Alt+C"),
    ("Alt+M", "Alt+M"),
    ("Alt+C", "Alt+C"),
)
CLEAR_MASK_HOTKEY_VALUES = frozenset(
    hotkey for _label, hotkey in CLEAR_MASK_HOTKEY_OPTIONS
)
DEFAULT_UNUSED_FACE_REMOVAL = False
DEFAULT_ENABLE_OBJECT_SYMMETRY_INFERENCE = False
SYMMETRY_INFERENCE_MODELS = frozenset(
    model_id for _label, model_id in SYMMETRY_INFERENCE_MODEL_OPTIONS
)
DEFAULT_USE_UV_RAYCAST_FOR_OBJECT_GENERATION = False
DEFAULT_MINIMUM_FACE_VISIBILITY_PERCENTAGE = 5
DEFAULT_DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY = 1.0
MIN_DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY = 0.0
MAX_DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY = 10.0
DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY_STEP = 0.1
AUTOMATIC_ATLAS_TEXTURE_RESOLUTIONS = (512, 1024)
DEFAULT_AUTOMATIC_ATLAS_CREATION = False
DEFAULT_AUTOMATIC_ATLAS_TEXTURE_SORT_BY_PBR = False
DEFAULT_USE_HALF_MESH_TEXTURE_PREFIX = False
DEFAULT_AUTOMATIC_ATLAS_TEXTURE_RESOLUTION = 512
MINIMUM_FACE_VISIBILITY_PERCENTAGE = 0
MAXIMUM_FACE_VISIBILITY_PERCENTAGE = 100
DEFAULT_MESH_EDIT_UPDATE_DELAY_SECONDS = 1.0
DEFAULT_SNAP_MIDDLE_EQUAL_ANGLE_ONLY = True
DEFAULT_CUT_CONNECTION_ON_DOUBLE_VERTEX_DELETION = False
DEFAULT_INCLUDE_LLM_WALL_VERIFICATION = False
DEFAULT_IGNORE_TOP_DOWN_CEILING = True
DEFAULT_MAKE_FLOORS_DOUBLE_SIDED = False
DEFAULT_MAKE_CEILINGS_DOUBLE_SIDED = False
DEFAULT_HIDE_STAIR_MESH_WHEN_PREVIEWING = True
DEFAULT_MAKE_WALLS_CONTINUOUS = False
MIN_MESH_EDIT_UPDATE_DELAY_SECONDS = 0.1
MAX_MESH_EDIT_UPDATE_DELAY_SECONDS = 10.0
MESH_EDIT_UPDATE_DELAY_STEP_SECONDS = 0.1
MESHY_SMART_TOPOLOGY_MIN_TARGET_POLYCOUNT = 100
MESHY_SMART_TOPOLOGY_MAX_TARGET_POLYCOUNT = 30_000
DEFAULT_MESHY_TARGET_POLYCOUNT = 2_000
SURFACE_TEXTURE_PROVIDER_MESHY = "meshy"
SURFACE_TEXTURE_PROVIDER_GPT_4O_MINI = "gpt-4o-mini"
SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA = "gpt-5.6-luna"
SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA_BATCH = "gpt-5.6-luna-batch"
SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA = "gpt-5.6-terra"
SURFACE_TEXTURE_PROVIDER_OPTIONS = (
    ("Meshy t2.5", SURFACE_TEXTURE_PROVIDER_MESHY),
    ("GPT-4o-mini", SURFACE_TEXTURE_PROVIDER_GPT_4O_MINI),
    ("GPT-5.6 Luna", SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA),
    ("GPT-5.6 Luna (Batch API)", SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA_BATCH),
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
_FIRST_PERSON_RESERVED_SHORTCUT_KEYS = frozenset(
    {
        Qt.Key.Key_A,
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
    scene_3d_display_screen_id: str | None = None
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
    use_half_mesh_texture_prefix: bool = (
        DEFAULT_USE_HALF_MESH_TEXTURE_PREFIX
    )
    atlas_display_screen_id: str | None = None
    snap_middle_equal_angle_only: bool = (
        DEFAULT_SNAP_MIDDLE_EQUAL_ANGLE_ONLY
    )
    first_person_navigation_mode: str = (
        DEFAULT_FIRST_PERSON_NAVIGATION_MODE
    )
    # Keep new fields at the end to preserve legacy positional construction.
    generation_display_screen_id: str | None = None
    automatic_atlas_creation: bool = DEFAULT_AUTOMATIC_ATLAS_CREATION
    ignore_top_down_ceiling: bool = DEFAULT_IGNORE_TOP_DOWN_CEILING
    hide_stair_mesh_when_previewing: bool = (
        DEFAULT_HIDE_STAIR_MESH_WHEN_PREVIEWING
    )
    clear_mask_hotkey: str = DEFAULT_CLEAR_MASK_HOTKEY
    plan_correction_model: str = DEFAULT_PLAN_CORRECTION_MODEL
    make_walls_continuous: bool = DEFAULT_MAKE_WALLS_CONTINUOUS
    doors_and_windows_point_light_intensity: float = (
        DEFAULT_DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY
    )
    make_floors_double_sided: bool = DEFAULT_MAKE_FLOORS_DOUBLE_SIDED
    make_ceilings_double_sided: bool = DEFAULT_MAKE_CEILINGS_DOUBLE_SIDED
    first_person_speed_meters_per_second: float = (
        DEFAULT_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND
    )
    cut_connection_on_double_vertex_deletion: bool = (
        DEFAULT_CUT_CONNECTION_ON_DOUBLE_VERTEX_DELETION
    )
    include_llm_wall_verification: bool = (
        DEFAULT_INCLUDE_LLM_WALL_VERIFICATION
    )
    wall_verification_model: str = DEFAULT_WALL_VERIFICATION_MODEL
    reference_generation_model: str = DEFAULT_REFERENCE_GENERATION_MODEL
    ceiling_height_estimation_model: str = CEILING_HEIGHT_INFERENCE_MODEL
    enable_object_symmetry_inference: bool = (
        DEFAULT_ENABLE_OBJECT_SYMMETRY_INFERENCE
    )
    symmetry_inference_model: str = DEFAULT_SYMMETRY_INFERENCE_MODEL

    def __post_init__(self) -> None:
        try:
            normalized_first_person_navigation_mode = (
                normalize_first_person_navigation_mode(
                    self.first_person_navigation_mode
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "First-person navigation mode must be Gravity or Noclip."
            ) from error
        object.__setattr__(
            self,
            "first_person_navigation_mode",
            normalized_first_person_navigation_mode,
        )
        try:
            normalized_first_person_speed = (
                normalize_first_person_speed_meters_per_second(
                    self.first_person_speed_meters_per_second
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "First-person speed must be within the supported range."
            ) from error
        object.__setattr__(
            self,
            "first_person_speed_meters_per_second",
            normalized_first_person_speed,
        )
        if not isinstance(self.snap_middle_equal_angle_only, bool):
            raise ValueError(
                "Snap-to-middle equal-angle filtering must be enabled or "
                "disabled."
            )
        if not isinstance(self.cut_connection_on_double_vertex_deletion, bool):
            raise ValueError(
                "Cut connection on double vertex deletion must be enabled "
                "or disabled."
            )
        if not isinstance(self.include_llm_wall_verification, bool):
            raise ValueError(
                "LLM wall verification must be enabled or disabled."
            )
        if (
            not isinstance(self.wall_verification_model, str)
            or self.wall_verification_model not in WALL_VERIFICATION_MODELS
        ):
            raise ValueError(
                "Unknown LLM wall verification model: "
                f"{self.wall_verification_model!r}."
            )
        if not isinstance(self.ignore_top_down_ceiling, bool):
            raise ValueError(
                "Top-down ceiling suppression must be enabled or disabled."
            )
        if not isinstance(self.make_floors_double_sided, bool):
            raise ValueError(
                "Floor double-sided rendering must be enabled or disabled."
            )
        if not isinstance(self.make_ceilings_double_sided, bool):
            raise ValueError(
                "Ceiling double-sided rendering must be enabled or disabled."
            )
        if not isinstance(self.hide_stair_mesh_when_previewing, bool):
            raise ValueError(
                "Hide stair mesh when previewing must be enabled or disabled."
            )
        if not isinstance(self.automatic_atlas_creation, bool):
            raise ValueError(
                "Automatic atlas creation must be enabled or disabled."
            )
        if not isinstance(self.use_half_mesh_texture_prefix, bool):
            raise ValueError(
                "Half-mesh texture prefix must be enabled or disabled."
            )
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
            not isinstance(self.plan_correction_model, str)
            or self.plan_correction_model not in PLAN_CORRECTION_MODELS
        ):
            raise ValueError(
                "Unknown plan correction model: "
                f"{self.plan_correction_model!r}."
            )
        if not isinstance(self.make_walls_continuous, bool):
            raise ValueError(
                "Make walls continuous must be enabled or disabled."
            )
        if (
            not isinstance(self.reference_generation_model, str)
            or self.reference_generation_model not in REFERENCE_GENERATION_MODELS
        ):
            raise ValueError(
                "Unknown reference generation model: "
                f"{self.reference_generation_model!r}."
            )
        if not isinstance(self.enable_object_symmetry_inference, bool):
            raise ValueError(
                "Object symmetry inference must be enabled or disabled."
            )
        if (
            not isinstance(self.symmetry_inference_model, str)
            or self.symmetry_inference_model not in SYMMETRY_INFERENCE_MODELS
        ):
            raise ValueError(
                "Unknown symmetric division inference model: "
                f"{self.symmetry_inference_model!r}."
            )
        if (
            not isinstance(self.ceiling_height_estimation_model, str)
            or self.ceiling_height_estimation_model
            not in CEILING_HEIGHT_ESTIMATION_MODELS
        ):
            raise ValueError(
                "Unknown ceiling height estimation model: "
                f"{self.ceiling_height_estimation_model!r}."
            )
        normalized_point_light_intensity = (
            _normalize_doors_and_windows_point_light_intensity(
                self.doors_and_windows_point_light_intensity
            )
        )
        if normalized_point_light_intensity is None:
            raise ValueError(
                "Doors and Windows point light intensity must be between "
                f"{MIN_DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY} and "
                f"{MAX_DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY}."
            )
        object.__setattr__(
            self,
            "doors_and_windows_point_light_intensity",
            normalized_point_light_intensity,
        )
        if (
            self.scene_3d_display_screen_id is not None
            and not isinstance(self.scene_3d_display_screen_id, str)
        ):
            raise ValueError(
                "3D scene display screen ID must be a string or None."
            )
        object.__setattr__(
            self,
            "scene_3d_display_screen_id",
            _normalize_fullscreen_3d_viewer_screen_id(
                self.scene_3d_display_screen_id
            ),
        )
        if (
            self.generation_display_screen_id is not None
            and not isinstance(self.generation_display_screen_id, str)
        ):
            raise ValueError(
                "Generation display screen ID must be a string or None."
            )
        object.__setattr__(
            self,
            "generation_display_screen_id",
            _normalize_fullscreen_3d_viewer_screen_id(
                self.generation_display_screen_id
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
        if (
            self.atlas_display_screen_id is not None
            and not isinstance(self.atlas_display_screen_id, str)
        ):
            raise ValueError("Atlas display screen ID must be a string or None.")
        object.__setattr__(
            self,
            "atlas_display_screen_id",
            _normalize_fullscreen_3d_viewer_screen_id(
                self.atlas_display_screen_id
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
        if (
            not isinstance(self.clear_mask_hotkey, str)
            or self.clear_mask_hotkey not in CLEAR_MASK_HOTKEY_VALUES
        ):
            raise ValueError("Clear mask hotkey must be a listed keymapping.")
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
    reference_view_presets_changed = Signal(object)

    def __init__(
        self,
        application_settings: ApplicationSettingsStore | None = None,
        environment: Mapping[str, str] | None = None,
        parent: QWidget | None = None,
        *,
        plan_pricing_network_manager: QNetworkAccessManager | None = None,
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
        self._plan_pricing_network_manager = plan_pricing_network_manager
        self._plan_pricing_reply: QNetworkReply | None = None
        self._plan_pricing_response = bytearray()
        self._plan_pricing_request_failed = False
        self._plan_pricing_lookup_started = False
        self._reference_view_presets: tuple[ReferenceViewPreset, ...] = ()
        self._reference_view_edit_id: str | None = None
        self._reference_view_is_loading = False
        self._reference_view_save_timer = QTimer(self)
        self._reference_view_save_timer.setSingleShot(True)
        self._reference_view_save_timer.setInterval(300)
        self._reference_view_save_timer.timeout.connect(
            self._save_reference_view_edits
        )
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
            scene_3d_display_screen_id=(
                self._selected_scene_3d_display_screen_id()
            ),
            generation_display_screen_id=(
                self._selected_generation_display_screen_id()
            ),
            jobs_window_screen_id=self._selected_jobs_window_screen_id(),
            atlas_display_screen_id=self._selected_atlas_display_screen_id(),
            automatic_atlas_creation=(
                self.automatic_atlas_creation_checkbox.isChecked()
            ),
            automatic_atlas_texture_sort_by_pbr=(
                self.automatic_atlas_texture_sort_by_pbr_checkbox.isChecked()
            ),
            use_half_mesh_texture_prefix=(
                self.use_half_mesh_texture_prefix_checkbox.isChecked()
            ),
            automatic_atlas_texture_resolution=int(
                self.automatic_atlas_texture_resolution_combo.currentData()
            ),
            canvas_3d_navigation_toggle_hotkey=(
                self._selected_canvas_3d_navigation_toggle_hotkey()
            ),
            clear_mask_hotkey=str(self.clear_mask_hotkey_combo.currentData()),
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
            snap_middle_equal_angle_only=(
                self.snap_middle_equal_angle_only_checkbox.isChecked()
            ),
            cut_connection_on_double_vertex_deletion=(
                self.cut_connection_on_double_vertex_deletion_checkbox.isChecked()
            ),
            include_llm_wall_verification=(
                self.include_llm_wall_verification_checkbox.isChecked()
            ),
            wall_verification_model=str(
                self.wall_verification_model_combo.currentData()
            ),
            first_person_navigation_mode=str(
                self.first_person_navigation_combo.currentData()
            ),
            first_person_speed_meters_per_second=(
                self._first_person_speed_from_slider()
            ),
            ignore_top_down_ceiling=(
                self.ignore_top_down_ceiling_checkbox.isChecked()
            ),
            make_floors_double_sided=(
                self.make_floors_double_sided_checkbox.isChecked()
            ),
            make_ceilings_double_sided=(
                self.make_ceilings_double_sided_checkbox.isChecked()
            ),
            hide_stair_mesh_when_previewing=(
                self.hide_stair_mesh_when_previewing_checkbox.isChecked()
            ),
            plan_correction_model=str(
                self.plan_correction_model_combo.currentData()
            ),
            reference_generation_model=str(
                self.reference_generation_model_combo.currentData()
            ),
            enable_object_symmetry_inference=(
                self.enable_object_symmetry_inference_checkbox.isChecked()
            ),
            symmetry_inference_model=str(
                self.symmetry_inference_model_combo.currentData()
            ),
            ceiling_height_estimation_model=str(
                self.ceiling_height_estimation_model_combo.currentData()
            ),
            make_walls_continuous=(
                self.make_walls_continuous_checkbox.isChecked()
            ),
            doors_and_windows_point_light_intensity=(
                self.doors_and_windows_point_light_intensity_spinbox.value()
            ),
        )

    def get_scene_3d_display_screen_id(self) -> str | None:
        """Return the selected 3D-scene display without rereading disk."""

        return self._selected_scene_3d_display_screen_id()

    def get_reference_view_presets(self) -> tuple[ReferenceViewPreset, ...]:
        """Return the last successfully saved reference views."""

        return self._reference_view_presets

    def get_generation_display_screen_id(self) -> str | None:
        """Return the selected Generation display without rereading disk."""

        return self._selected_generation_display_screen_id()

    def get_jobs_window_screen_id(self) -> str | None:
        """Return the selected Jobs-window display without rereading disk."""

        return self._selected_jobs_window_screen_id()

    def get_atlas_display_screen_id(self) -> str | None:
        """Return the selected Atlas display without rereading disk."""

        return self._selected_atlas_display_screen_id()

    def clear_session_keys(self) -> None:
        """Clear the temporary plaintext key values from settings.json."""

        self.meshy_api_key_edit.clear()
        self.openai_api_key_edit.clear()

    # ### QWidget lifecycle ###
    def showEvent(self, event: QShowEvent) -> None:
        """Load current model prices when Settings first becomes visible."""

        super().showEvent(event)
        self._start_plan_correction_pricing_lookup()

    def hideEvent(self, event: QHideEvent) -> None:
        """Flush valid view edits before switching back to Generation."""

        self._save_reference_view_edits()
        super().hideEvent(event)

    def dispose(self) -> None:
        """Disconnect application-wide display signals exactly once."""

        if self._is_disposed:
            return
        self._reference_view_save_timer.stop()
        self._save_reference_view_edits()
        self._is_disposed = True
        self._cancel_plan_correction_pricing_lookup()
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

        title_label = QLabel("Settings")
        title_label.setStyleSheet("font-size: 20px; font-weight: 600;")
        root_layout.addWidget(title_label)

        self.settings_scroll_area = QScrollArea()
        self.settings_scroll_area.setObjectName("settings_scroll_area")
        self.settings_scroll_area.setWidgetResizable(True)
        self.settings_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.settings_content = QWidget()
        self.settings_content.setObjectName("settings_content")
        sections_layout = QVBoxLayout(self.settings_content)
        sections_layout.setContentsMargins(0, 0, 0, 0)
        sections_layout.setSpacing(12)

        (
            self.api_credentials_group,
            api_credentials_form,
        ) = self._build_form_group(
            "API credentials",
            "api_credentials_settings_group",
        )
        self.display_settings_group, display_form = self._build_form_group(
            "Displays",
            "display_settings_group",
        )
        self.canvas_settings_group, canvas_form = self._build_form_group(
            "Canvas",
            "canvas_settings_group",
        )
        self.room_analysis_settings_group, room_analysis_form = (
            self._build_form_group(
                "Room analysis",
                "room_analysis_settings_group",
            )
        )
        self.generation_shortcuts_group, generation_shortcuts_form = (
            self._build_form_group(
                "Generation shortcuts",
                "generation_shortcuts_group",
            )
        )
        (
            self.object_generation_settings_group,
            object_generation_form,
        ) = self._build_form_group(
            "Object generation",
            "object_generation_settings_group",
        )
        (
            self.atlas_automation_settings_group,
            atlas_automation_form,
        ) = self._build_form_group(
            "Atlas automation",
            "atlas_automation_settings_group",
        )
        sections_layout.addWidget(self.api_credentials_group)
        sections_layout.addWidget(self.display_settings_group)
        sections_layout.addWidget(self.canvas_settings_group)
        sections_layout.addWidget(self.room_analysis_settings_group)
        sections_layout.addWidget(self.generation_shortcuts_group)
        sections_layout.addWidget(self.object_generation_settings_group)
        self._build_reference_view_presets_ui(sections_layout)
        sections_layout.addWidget(self.atlas_automation_settings_group)
        sections_layout.addStretch(1)
        self.settings_scroll_area.setWidget(self.settings_content)
        root_layout.addWidget(self.settings_scroll_area, 1)

        self.meshy_api_key_edit = self._build_secret_input(
            "Meshy AI API key"
        )
        self.meshy_api_key_edit.setObjectName("meshy_api_key_edit")
        self.meshy_api_key_edit.textChanged.connect(
            self._handle_secret_text_changed
        )
        api_credentials_form.addRow(
            "Meshy AI API key",
            self.meshy_api_key_edit,
        )

        self.meshy_key_status_label = QLabel()
        self.meshy_key_status_label.setObjectName("meshy_key_status_label")
        api_credentials_form.addRow("", self.meshy_key_status_label)

        self.openai_api_key_edit = self._build_secret_input(
            "OpenAI API key"
        )
        self.openai_api_key_edit.setObjectName("openai_api_key_edit")
        self.openai_api_key_edit.textChanged.connect(
            self._handle_secret_text_changed
        )
        api_credentials_form.addRow(
            "OpenAI API key",
            self.openai_api_key_edit,
        )

        self.openai_key_status_label = QLabel()
        self.openai_key_status_label.setObjectName("openai_key_status_label")
        api_credentials_form.addRow("", self.openai_key_status_label)

        self.plan_correction_model_combo = QComboBox()
        self.plan_correction_model_combo.setObjectName(
            "plan_correction_model_combo"
        )
        self.plan_correction_model_combo.setToolTip(
            "Choose the image model used to straighten and clean imported "
            "architectural plan photographs. GPT Image 2, GPT-5.6 Luna, and "
            "GPT-5.6 Terra use the configured OpenAI API key. The GPT-5.6 "
            "token prices shown here exclude the additional GPT Image 2 tool "
            "charges incurred when producing the corrected image. Displayed "
            "prices are Standard API rates per 1M tokens, except the explicit "
            "Luna (Batch API) option which shows Batch rates. Luna and Terra use "
            "their short-context rates. Batch jobs can take up to 24 hours, "
            "not an interactive-speed request. Qwen runs locally, "
            "but its Research License limits use to non-commercial research "
            "or evaluation unless you obtain a separate commercial license."
        )
        for label, model_id in PLAN_CORRECTION_MODEL_OPTIONS:
            display_label = format_plan_correction_model_label(
                label,
                model_id,
                None,
            )
            self.plan_correction_model_combo.addItem(display_label, model_id)
        self.plan_correction_model_combo.currentIndexChanged.connect(
            self._handle_plan_correction_model_changed
        )
        canvas_form.addRow(
            "Plan correction model",
            self.plan_correction_model_combo,
        )

        self.make_walls_continuous_checkbox = QCheckBox()
        self.make_walls_continuous_checkbox.setObjectName(
            "make_walls_continuous_checkbox"
        )
        self.make_walls_continuous_checkbox.setToolTip(
            "After AI correction, use local image processing to bridge paired "
            "wall boundaries interrupted by doorway openings."
        )
        self.make_walls_continuous_checkbox.toggled.connect(
            self._handle_make_walls_continuous_changed
        )
        canvas_form.addRow(
            "Make walls continuous",
            self.make_walls_continuous_checkbox,
        )

        self.ceiling_height_estimation_model_combo = QComboBox()
        self.ceiling_height_estimation_model_combo.setObjectName(
            "ceiling_height_estimation_model_combo"
        )
        self.ceiling_height_estimation_model_combo.setToolTip(
            "Choose the model used to estimate ceiling heights from marked "
            "video ranges. GPT models use the configured OpenAI API key; "
            "Qwen runs locally."
        )
        for label, model_id in CEILING_HEIGHT_MODEL_OPTIONS:
            self.ceiling_height_estimation_model_combo.addItem(label, model_id)
        self.ceiling_height_estimation_model_combo.currentIndexChanged.connect(
            self._handle_ceiling_height_estimation_model_changed
        )
        room_analysis_form.addRow(
            "Ceiling height estimation model",
            self.ceiling_height_estimation_model_combo,
        )

        self.scene_3d_display_screen_combo = QComboBox()
        self.scene_3d_display_screen_combo.setObjectName(
            "scene_3d_display_screen_combo"
        )
        self.scene_3d_display_screen_combo.setToolTip(
            "Choose a display for the shared 3D scene, or None to keep its "
            "tab in the main window."
        )
        self.scene_3d_display_screen_combo.currentIndexChanged.connect(
            self._handle_scene_3d_display_screen_changed
        )
        display_form.addRow(
            "3D scene display",
            self.scene_3d_display_screen_combo,
        )

        self.generation_display_screen_combo = QComboBox()
        self.generation_display_screen_combo.setObjectName(
            "generation_display_screen_combo"
        )
        self.generation_display_screen_combo.setToolTip(
            "Choose a display for the Generation workspace, or None to keep "
            "its tab in the main window."
        )
        self.generation_display_screen_combo.currentIndexChanged.connect(
            self._handle_generation_display_screen_changed
        )
        display_form.addRow(
            "Generation display",
            self.generation_display_screen_combo,
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
        display_form.addRow(
            "Jobs window display",
            self.jobs_window_screen_combo,
        )

        self.atlas_display_screen_combo = QComboBox()
        self.atlas_display_screen_combo.setObjectName(
            "atlas_display_screen_combo"
        )
        self.atlas_display_screen_combo.setToolTip(
            "Choose a display for a detached, maximized Atlas window, or "
            "None to keep the Atlas embedded."
        )
        self.atlas_display_screen_combo.currentIndexChanged.connect(
            self._handle_atlas_display_screen_changed
        )
        display_form.addRow(
            "Atlas display",
            self.atlas_display_screen_combo,
        )

        self.automatic_atlas_creation_checkbox = QCheckBox()
        self.automatic_atlas_creation_checkbox.setObjectName(
            "automatic_atlas_creation_checkbox"
        )
        self.automatic_atlas_creation_checkbox.setToolTip(
            "Create another Atlas automatically when scene textures do not "
            "fit in the currently selected Atlas."
        )
        self.automatic_atlas_creation_checkbox.toggled.connect(
            self._handle_automatic_atlas_creation_changed
        )
        atlas_automation_form.addRow(
            "Automatic atlas creation",
            self.automatic_atlas_creation_checkbox,
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
        atlas_automation_form.addRow(
            "Automatic Atlas texture sort by PBR",
            self.automatic_atlas_texture_sort_by_pbr_checkbox,
        )

        self.use_half_mesh_texture_prefix_checkbox = QCheckBox()
        self.use_half_mesh_texture_prefix_checkbox.setObjectName(
            "use_half_mesh_texture_prefix_checkbox"
        )
        self.use_half_mesh_texture_prefix_checkbox.setToolTip(
            "Automatically place symmetric half-mesh textures in Atlases "
            "whose names start with [HALF]."
        )
        self.use_half_mesh_texture_prefix_checkbox.toggled.connect(
            self._handle_use_half_mesh_texture_prefix_changed
        )
        atlas_automation_form.addRow(
            "Use [HALF] half-mesh texture prefix",
            self.use_half_mesh_texture_prefix_checkbox,
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
        atlas_automation_form.addRow(
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
            "top-down orbit and first-person navigation. Bare A is reserved "
            "for the Generation-frame overlay; Z, Q, S, D, R, and F are "
            "reserved for first-person movement; Ctrl+Z is reserved for "
            "Canvas undo."
        )
        self.canvas_3d_navigation_toggle_hotkey_edit.keySequenceChanged.connect(
            self._handle_canvas_3d_navigation_toggle_hotkey_changed
        )
        canvas_form.addRow(
            "Canvas 3D navigation hotkey",
            self.canvas_3d_navigation_toggle_hotkey_edit,
        )

        self.clear_mask_hotkey_combo = QComboBox()
        self.clear_mask_hotkey_combo.setObjectName("clear_mask_hotkey_combo")
        self.clear_mask_hotkey_combo.setToolTip(
            "Choose a shortcut for clearing the shared Generation mask. "
            "The shortcut works only while Generation is active."
        )
        for label, hotkey in CLEAR_MASK_HOTKEY_OPTIONS:
            self.clear_mask_hotkey_combo.addItem(label, hotkey)
        self.clear_mask_hotkey_combo.currentIndexChanged.connect(
            self._handle_clear_mask_hotkey_changed
        )
        generation_shortcuts_form.addRow(
            "Clear mask",
            self.clear_mask_hotkey_combo,
        )

        self.first_person_navigation_combo = QComboBox()
        self.first_person_navigation_combo.setObjectName(
            "first_person_navigation_combo"
        )
        self.first_person_navigation_combo.setToolTip(
            "Use Gravity for the current level movement, or Noclip to fly "
            "freely in the viewing direction."
        )
        for label, mode in FIRST_PERSON_NAVIGATION_MODE_OPTIONS:
            self.first_person_navigation_combo.addItem(label, mode)
        self.first_person_navigation_combo.currentIndexChanged.connect(
            self._handle_first_person_navigation_mode_changed
        )
        canvas_form.addRow(
            "First person navigation",
            self.first_person_navigation_combo,
        )

        self.first_person_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.first_person_speed_slider.setObjectName(
            "first_person_speed_slider"
        )
        self.first_person_speed_slider.setRange(
            round(
                MIN_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND
                / FIRST_PERSON_MOVE_SPEED_STEP_METERS_PER_SECOND
            ),
            round(
                MAX_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND
                / FIRST_PERSON_MOVE_SPEED_STEP_METERS_PER_SECOND
            ),
        )
        self.first_person_speed_slider.setSingleStep(1)
        self.first_person_speed_slider.setPageStep(10)
        self.first_person_speed_slider.setToolTip(
            "Set first-person movement speed in meters per second."
        )
        self.first_person_speed_value_label = QLabel()
        self.first_person_speed_value_label.setObjectName(
            "first_person_speed_value_label"
        )
        self.first_person_speed_control = QWidget()
        self.first_person_speed_control.setObjectName(
            "first_person_speed_control"
        )
        first_person_speed_layout = QHBoxLayout(
            self.first_person_speed_control
        )
        first_person_speed_layout.setContentsMargins(0, 0, 0, 0)
        first_person_speed_layout.setSpacing(8)
        first_person_speed_layout.addWidget(self.first_person_speed_slider, 1)
        first_person_speed_layout.addWidget(self.first_person_speed_value_label)
        self.first_person_speed_slider.valueChanged.connect(
            self._handle_first_person_speed_changed
        )
        canvas_form.addRow(
            "First person speed",
            self.first_person_speed_control,
        )

        self.ignore_top_down_ceiling_checkbox = QCheckBox()
        self.ignore_top_down_ceiling_checkbox.setObjectName(
            "ignore_top_down_ceiling_checkbox"
        )
        self.ignore_top_down_ceiling_checkbox.setToolTip(
            "Hide level ceilings and exclude them from selection while the "
            "shared 3D Scene uses top-down orbit navigation."
        )
        self.ignore_top_down_ceiling_checkbox.toggled.connect(
            self._handle_ignore_top_down_ceiling_changed
        )
        canvas_form.addRow(
            "Ignore top-down ceiling",
            self.ignore_top_down_ceiling_checkbox,
        )

        self.make_floors_double_sided_checkbox = QCheckBox()
        self.make_floors_double_sided_checkbox.setObjectName(
            "make_floors_double_sided_checkbox"
        )
        self.make_floors_double_sided_checkbox.setToolTip(
            "Render and export floor surfaces from both sides."
        )
        self.make_floors_double_sided_checkbox.toggled.connect(
            self._handle_make_floors_double_sided_changed
        )
        canvas_form.addRow(
            "Make floors double sided",
            self.make_floors_double_sided_checkbox,
        )

        self.make_ceilings_double_sided_checkbox = QCheckBox()
        self.make_ceilings_double_sided_checkbox.setObjectName(
            "make_ceilings_double_sided_checkbox"
        )
        self.make_ceilings_double_sided_checkbox.setToolTip(
            "Render and export ceiling surfaces from both sides."
        )
        self.make_ceilings_double_sided_checkbox.toggled.connect(
            self._handle_make_ceilings_double_sided_changed
        )
        canvas_form.addRow(
            "Make ceilings double sided",
            self.make_ceilings_double_sided_checkbox,
        )

        self.hide_stair_mesh_when_previewing_checkbox = QCheckBox()
        self.hide_stair_mesh_when_previewing_checkbox.setObjectName(
            "hide_stair_mesh_when_previewing_checkbox"
        )
        self.hide_stair_mesh_when_previewing_checkbox.setToolTip(
            "Hide the selected stair's existing mesh while staged stair "
            "changes are previewed. Restore it when the preview ends."
        )
        self.hide_stair_mesh_when_previewing_checkbox.toggled.connect(
            self._handle_hide_stair_mesh_when_previewing_changed
        )
        canvas_form.addRow(
            "Hide stair's mesh when previewing",
            self.hide_stair_mesh_when_previewing_checkbox,
        )

        self.snap_middle_equal_angle_only_checkbox = QCheckBox()
        self.snap_middle_equal_angle_only_checkbox.setObjectName(
            "snap_middle_equal_angle_only_checkbox"
        )
        self.snap_middle_equal_angle_only_checkbox.setToolTip(
            "Only snap a preview point to a wall midpoint when the two "
            "resulting corner angles are equal."
        )
        self.snap_middle_equal_angle_only_checkbox.toggled.connect(
            self._handle_snap_middle_equal_angle_only_changed
        )
        canvas_form.addRow(
            "Snap to middle equal angle only",
            self.snap_middle_equal_angle_only_checkbox,
        )

        self.cut_connection_on_double_vertex_deletion_checkbox = QCheckBox()
        self.cut_connection_on_double_vertex_deletion_checkbox.setObjectName(
            "cut_connection_on_double_vertex_deletion_checkbox"
        )
        self.cut_connection_on_double_vertex_deletion_checkbox.setToolTip(
            "When exactly two connected wall vertices are selected, Delete "
            "cuts their connection first and keeps both vertices selected. "
            "Press Delete again to remove the vertices."
        )
        self.cut_connection_on_double_vertex_deletion_checkbox.toggled.connect(
            self._handle_cut_connection_on_double_vertex_deletion_changed
        )
        canvas_form.addRow(
            "Cut connection on double vertex deletion",
            self.cut_connection_on_double_vertex_deletion_checkbox,
        )

        # ### Optional wall-generation review controls ###
        self.wall_llm_verification_control = QWidget()
        wall_verification_layout = QHBoxLayout(
            self.wall_llm_verification_control
        )
        wall_verification_layout.setContentsMargins(0, 0, 0, 0)
        self.include_llm_wall_verification_checkbox = QCheckBox()
        self.include_llm_wall_verification_checkbox.setObjectName(
            "include_llm_wall_verification_checkbox"
        )
        self.include_llm_wall_verification_checkbox.setToolTip(
            "After Confirm walls, review the plan and existing walls with "
            "the selected model. Preview suggested additions in red, then "
            "accept them or right-click to dismiss them. Existing walls "
            "are not moved or deleted."
        )
        self.wall_verification_model_label = QLabel("LLM verification model")
        self.wall_verification_model_combo = QComboBox()
        self.wall_verification_model_combo.setObjectName(
            "wall_verification_model_combo"
        )
        self.wall_verification_model_combo.setToolTip(
            "GPT models use the configured OpenAI API key and may incur "
            "API charges. Local Qwen uses the installed Qwen3-VL model."
        )
        for model_label, model_id in WALL_VERIFICATION_MODEL_OPTIONS:
            self.wall_verification_model_combo.addItem(model_label, model_id)
        self.wall_verification_model_label.setBuddy(
            self.wall_verification_model_combo
        )
        self.wall_verification_model_label.setEnabled(False)
        self.wall_verification_model_combo.setEnabled(False)
        wall_verification_layout.addWidget(
            self.include_llm_wall_verification_checkbox
        )
        wall_verification_layout.addWidget(self.wall_verification_model_label)
        wall_verification_layout.addWidget(
            self.wall_verification_model_combo, 1
        )
        self.include_llm_wall_verification_checkbox.toggled.connect(
            self._handle_include_llm_wall_verification_changed
        )
        self.wall_verification_model_combo.currentIndexChanged.connect(
            self._handle_wall_verification_model_changed
        )
        canvas_form.addRow(
            "Include LLM verification for wall generation",
            self.wall_llm_verification_control,
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
            "Wait this long after releasing a doorway, window, wall, "
            "open-space, or other structural edit before rebuilding the "
            "Canvas 3D mesh."
        )
        self.mesh_edit_update_delay_spinbox.valueChanged.connect(
            self._handle_mesh_edit_update_delay_changed
        )
        canvas_form.addRow(
            "Mesh edit update delay",
            self.mesh_edit_update_delay_spinbox,
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
        object_generation_form.addRow(
            "Use weighted camera projection",
            self.use_uv_raycast_for_object_generation_checkbox,
        )

        self.unused_face_removal_checkbox = QCheckBox()
        self.unused_face_removal_checkbox.setObjectName(
            "unused_face_removal_checkbox"
        )
        self.unused_face_removal_checkbox.setToolTip(
            "Generate geometry first, remove faces that do not meet the "
            "configured minimum visibility in 14 cleanup cameras (six "
            "axis-aligned and eight hidden diagonals), then submit the "
            "edited GLB to Meshy Retexture. This uses two Meshy tasks."
        )
        self.unused_face_removal_checkbox.toggled.connect(
            self._handle_unused_face_removal_changed
        )
        object_generation_form.addRow(
            "Unused face removal",
            self.unused_face_removal_checkbox,
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
            "During unused-face removal, retain a meaningfully projected "
            "face only when at least this percentage is visible in one "
            "cleanup view. Higher values remove more faces and reject tiny "
            "glimpses through gaps. Setting 0% does not disable removal: "
            "fully unseen or unsampled faces and stacked layers can still "
            "be removed."
        )
        self.minimum_face_visibility_percentage_spinbox.valueChanged.connect(
            self._handle_minimum_face_visibility_percentage_changed
        )
        object_generation_form.addRow(
            "Minimum percentage of face visible to retain(Higher means more "
            "faces will be removed)",
            self.minimum_face_visibility_percentage_spinbox,
        )

        self.doors_and_windows_point_light_intensity_spinbox = QDoubleSpinBox()
        self.doors_and_windows_point_light_intensity_spinbox.setObjectName(
            "doors_and_windows_point_light_intensity_spinbox"
        )
        self.doors_and_windows_point_light_intensity_spinbox.setRange(
            MIN_DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY,
            MAX_DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY,
        )
        self.doors_and_windows_point_light_intensity_spinbox.setDecimals(1)
        self.doors_and_windows_point_light_intensity_spinbox.setSingleStep(
            DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY_STEP
        )
        self.doors_and_windows_point_light_intensity_spinbox.setKeyboardTracking(
            False
        )
        self.doors_and_windows_point_light_intensity_spinbox.setToolTip(
            "Control the point light that follows the pointer over door and "
            "window meshes in their dedicated 3D preview."
        )
        self.doors_and_windows_point_light_intensity_spinbox.valueChanged.connect(
            self._handle_doors_and_windows_point_light_intensity_changed
        )
        object_generation_form.addRow(
            "Doors and Windows point light intensity",
            self.doors_and_windows_point_light_intensity_spinbox,
        )

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
        api_credentials_form.addRow(security_note)

        meshy_note = QLabel(
            "Object generation uses Meshy Image-to-3D. Surface texture "
            "generation can use Meshy or an OpenAI vision model; choose its "
            "provider in the Surface controls of the Generation tab. GPT-4o-mini "
            "first analyzes the references, then GPT Image 2 renders the "
            "texture, so that choice makes two OpenAI requests. Provider "
            "requests may consume account credits."
        )
        meshy_note.setObjectName("meshy_availability_note")
        meshy_note.setWordWrap(True)
        meshy_note.setStyleSheet("color: #666;")
        api_credentials_form.addRow(meshy_note)

        self._connect_screen_change_signals()

    # ### Reference generation view editor ###
    def _build_reference_view_presets_ui(
        self,
        sections_layout: QVBoxLayout,
    ) -> None:
        self.reference_view_presets_group = QGroupBox(
            "Reference generation views"
        )
        self.reference_view_presets_group.setObjectName(
            "reference_view_presets_group"
        )
        layout = QVBoxLayout(self.reference_view_presets_group)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        sections_layout.addWidget(self.reference_view_presets_group)

        model_form = QFormLayout()
        self.reference_generation_model_combo = QComboBox()
        self.reference_generation_model_combo.setObjectName(
            "reference_generation_model_combo"
        )
        self.reference_generation_model_combo.setToolTip(
            "Choose the model used to generate object reference images. "
            "OpenAI models use the configured API key; Qwen and Stable "
            "Diffusion run locally. Luna (Batch API) queues requests with "
            "a 24-hour completion window and discounted token rates. Stable "
            "Diffusion downloads its model "
            "(about 7 GB) on first use. Changes apply to the next request."
        )
        for label, model in REFERENCE_GENERATION_MODEL_OPTIONS:
            self.reference_generation_model_combo.addItem(label, model)
        self.reference_generation_model_combo.currentIndexChanged.connect(
            self._handle_reference_generation_model_changed
        )
        model_form.addRow(
            "Reference generation model", self.reference_generation_model_combo
        )
        self.enable_object_symmetry_inference_checkbox = QCheckBox(
            "Enable object symmetry inference"
        )
        self.enable_object_symmetry_inference_checkbox.setObjectName(
            "enable_object_symmetry_inference_checkbox"
        )
        self.enable_object_symmetry_inference_checkbox.setToolTip(
            "After generating an object reference, infer whether its geometry "
            "can use Symmetric division. Uncheck to leave the manual "
            "Symmetric division choice unchanged."
        )
        self.enable_object_symmetry_inference_checkbox.toggled.connect(
            self._handle_enable_object_symmetry_inference_changed
        )
        model_form.addRow(self.enable_object_symmetry_inference_checkbox)
        self.symmetry_inference_model_combo = QComboBox()
        self.symmetry_inference_model_combo.setObjectName(
            "symmetry_inference_model_combo"
        )
        self.symmetry_inference_model_combo.setToolTip(
            "Choose the model used to infer Symmetric division after reference "
            "generation. Local Qwen and Reflect3D run on this PC; GPT-5.6 "
            "Luna uses the configured OpenAI API key. Changes apply to the "
            "next generation job."
        )
        for label, model in SYMMETRY_INFERENCE_MODEL_OPTIONS:
            self.symmetry_inference_model_combo.addItem(label, model)
        self.symmetry_inference_model_combo.currentIndexChanged.connect(
            self._handle_symmetry_inference_model_changed
        )
        self.symmetry_inference_model_label = QLabel(
            "Symmetric division inference model"
        )
        model_form.addRow(
            self.symmetry_inference_model_label,
            self.symmetry_inference_model_combo,
        )
        layout.addLayout(model_form)

        self.reference_view_preset_list = QListWidget()
        self.reference_view_preset_list.setObjectName(
            "reference_view_preset_list"
        )
        self.reference_view_preset_list.setMaximumHeight(170)
        self.reference_view_preset_list.currentRowChanged.connect(
            self._handle_reference_view_selection_changed
        )
        layout.addWidget(self.reference_view_preset_list)

        form = QFormLayout()
        self.reference_view_name_edit = QLineEdit()
        self.reference_view_name_edit.setObjectName("reference_view_name_edit")
        self.reference_view_name_edit.setMaxLength(80)
        self.reference_view_name_edit.textChanged.connect(
            self._handle_reference_view_edit_changed
        )
        form.addRow("View name", self.reference_view_name_edit)
        self.reference_view_prompt_edit = QPlainTextEdit()
        self.reference_view_prompt_edit.setObjectName(
            "reference_view_prompt_edit"
        )
        self.reference_view_prompt_edit.setMinimumHeight(115)
        self.reference_view_prompt_edit.setMaximumHeight(200)
        self.reference_view_prompt_edit.textChanged.connect(
            self._handle_reference_view_edit_changed
        )
        form.addRow("Prompt", self.reference_view_prompt_edit)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.add_reference_view_button = QPushButton("Add view")
        self.add_reference_view_button.setObjectName("add_reference_view_button")
        self.add_reference_view_button.clicked.connect(
            self._handle_add_reference_view_clicked
        )
        self.remove_reference_view_button = QPushButton("Remove view")
        self.remove_reference_view_button.setObjectName(
            "remove_reference_view_button"
        )
        self.remove_reference_view_button.clicked.connect(
            self._handle_remove_reference_view_clicked
        )
        self.save_reference_view_button = QPushButton("Save view")
        self.save_reference_view_button.setObjectName("save_reference_view_button")
        self.save_reference_view_button.clicked.connect(
            self._save_reference_view_edits
        )
        buttons.addWidget(self.add_reference_view_button)
        buttons.addWidget(self.remove_reference_view_button)
        buttons.addWidget(self.save_reference_view_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.reference_view_preset_status_label = QLabel()
        self.reference_view_preset_status_label.setObjectName(
            "reference_view_preset_status_label"
        )
        self.reference_view_preset_status_label.setWordWrap(True)
        layout.addWidget(self.reference_view_preset_status_label)

    def _refresh_reference_view_preset_list(
        self,
        selected_id: str | None = None,
    ) -> None:
        blocker = QSignalBlocker(self.reference_view_preset_list)
        self.reference_view_preset_list.clear()
        selected_row = -1
        for row, preset in enumerate(self._reference_view_presets):
            item = QListWidgetItem(preset.name)
            item.setData(Qt.ItemDataRole.UserRole, preset.preset_id)
            self.reference_view_preset_list.addItem(item)
            if preset.preset_id == selected_id:
                selected_row = row
        if selected_row < 0 and self._reference_view_presets:
            selected_row = 0
        self.reference_view_preset_list.setCurrentRow(selected_row)
        del blocker
        self._load_reference_view_editor(selected_row)

    def _load_reference_view_editor(self, row: int) -> None:
        self._reference_view_is_loading = True
        selected = (
            self._reference_view_presets[row]
            if 0 <= row < len(self._reference_view_presets)
            else None
        )
        self._reference_view_edit_id = (
            selected.preset_id if selected is not None else None
        )
        self.reference_view_name_edit.setText(
            selected.name if selected is not None else ""
        )
        self.reference_view_prompt_edit.setPlainText(
            selected.prompt if selected is not None else ""
        )
        self.reference_view_name_edit.setEnabled(selected is not None)
        self.reference_view_prompt_edit.setEnabled(selected is not None)
        self.remove_reference_view_button.setEnabled(selected is not None)
        self.save_reference_view_button.setEnabled(False)
        self.reference_view_preset_status_label.setText(
            "Valid edits save automatically."
            if selected is not None
            else "Add a view, or use (Write prompt) in Generation."
        )
        self._reference_view_is_loading = False

    def _handle_reference_view_selection_changed(self, row: int) -> None:
        self._reference_view_save_timer.stop()
        if not self._flush_valid_reference_view_edits():
            previous_row = next(
                (
                    index
                    for index, preset in enumerate(self._reference_view_presets)
                    if preset.preset_id == self._reference_view_edit_id
                ),
                -1,
            )
            blocker = QSignalBlocker(self.reference_view_preset_list)
            self.reference_view_preset_list.setCurrentRow(previous_row)
            del blocker
            return
        self._load_reference_view_editor(row)

    def _reference_view_edited_presets(
        self,
    ) -> tuple[ReferenceViewPreset, ...] | None:
        if self._reference_view_edit_id is None:
            return None
        name = self.reference_view_name_edit.text().strip()
        prompt = self.reference_view_prompt_edit.toPlainText().strip()
        error = ""
        if not name or not prompt:
            error = "A view name and prompt are required; saved values are unchanged."
        elif name.casefold() == "(write prompt)":
            error = "(Write prompt) is reserved for custom prompts; choose another view name."
        elif len(name) > 80 or len(prompt) > 4000:
            error = "Use at most 80 characters for the name and 4,000 for the prompt."
        elif any(
            preset.preset_id != self._reference_view_edit_id
            and preset.name.casefold() == name.casefold()
            for preset in self._reference_view_presets
        ):
            error = "Choose a unique view name; saved values are unchanged."
        if error:
            self.reference_view_preset_status_label.setText(error)
            return None
        return tuple(
            ReferenceViewPreset(preset.preset_id, name, prompt)
            if preset.preset_id == self._reference_view_edit_id
            else preset
            for preset in self._reference_view_presets
        )

    def _handle_reference_view_edit_changed(self, *_args: object) -> None:
        if self._reference_view_is_loading or self._is_loading_settings:
            return
        self._reference_view_save_timer.stop()
        edited_presets = self._reference_view_edited_presets()
        changed = (
            edited_presets is not None
            and edited_presets != self._reference_view_presets
        )
        self.save_reference_view_button.setEnabled(changed)
        if changed:
            self.reference_view_preset_status_label.setText("Saving view…")
            self._reference_view_save_timer.start()

    def _persist_reference_view_presets(
        self,
        presets: tuple[ReferenceViewPreset, ...],
    ) -> bool:
        try:
            saved = save_reference_view_presets(
                self._application_settings,
                presets,
            )
        except (TypeError, ValueError):
            saved = False
        if not saved:
            self.reference_view_preset_status_label.setText(
                "Could not save reference views. Check the settings file and retry."
            )
            return False
        self._reference_view_presets = presets
        self.reference_view_presets_changed.emit(presets)
        return True

    def _save_reference_view_edits(self) -> bool:
        self._reference_view_save_timer.stop()
        edited_presets = self._reference_view_edited_presets()
        if edited_presets is None:
            self.save_reference_view_button.setEnabled(False)
            return False
        if edited_presets == self._reference_view_presets:
            self.save_reference_view_button.setEnabled(False)
            return True
        if not self._persist_reference_view_presets(edited_presets):
            return False
        for row, preset in enumerate(edited_presets):
            if preset.preset_id == self._reference_view_edit_id:
                self.reference_view_preset_list.item(row).setText(preset.name)
                break
        self.save_reference_view_button.setEnabled(False)
        self.reference_view_preset_status_label.setText("View saved.")
        return True

    def _flush_valid_reference_view_edits(self) -> bool:
        """Keep a valid unsaved draft selected if its settings write fails."""

        self._reference_view_save_timer.stop()
        if self._reference_view_edited_presets() is None:
            return True
        return self._save_reference_view_edits()

    def _handle_add_reference_view_clicked(self) -> None:
        if not self._flush_valid_reference_view_edits():
            return
        names = {preset.name.casefold() for preset in self._reference_view_presets}
        number = 1
        while f"new view {number}" in names:
            number += 1
        preset = ReferenceViewPreset(
            uuid4().hex,
            f"New view {number}",
            next(
                preset.prompt
                for preset in DEFAULT_REFERENCE_VIEW_PRESETS
                if preset.preset_id == "front"
            ),
        )
        if self._persist_reference_view_presets(
            self._reference_view_presets + (preset,)
        ):
            self._refresh_reference_view_preset_list(preset.preset_id)
            self.reference_view_name_edit.setFocus()
            self.reference_view_name_edit.selectAll()

    def _handle_remove_reference_view_clicked(self) -> None:
        self._reference_view_save_timer.stop()
        row = self.reference_view_preset_list.currentRow()
        if row < 0:
            return
        presets = tuple(
            preset
            for preset in self._reference_view_presets
            if preset.preset_id != self._reference_view_edit_id
        )
        if self._persist_reference_view_presets(presets):
            selected_id = (
                presets[min(row, len(presets) - 1)].preset_id
                if presets
                else None
            )
            self._refresh_reference_view_preset_list(selected_id)

    # ### General settings controls ###
    @staticmethod
    def _build_form_group(
        title: str,
        object_name: str,
    ) -> tuple[QGroupBox, QFormLayout]:
        """Build one consistently spaced Settings section."""

        group = QGroupBox(title)
        group.setObjectName(object_name)
        form_layout = QFormLayout(group)
        form_layout.setContentsMargins(12, 12, 12, 12)
        form_layout.setSpacing(10)
        return group, form_layout

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
        plan_correction_model = read_plan_correction_model(
            self._application_settings
        )
        self.plan_correction_model_combo.setCurrentIndex(
            max(
                0,
                self.plan_correction_model_combo.findData(
                    plan_correction_model
                ),
            )
        )
        self.reference_generation_model_combo.setCurrentIndex(
            self.reference_generation_model_combo.findData(
                read_reference_generation_model(self._application_settings)
            )
        )
        self.symmetry_inference_model_combo.setCurrentIndex(
            self.symmetry_inference_model_combo.findData(
                read_symmetry_inference_model(self._application_settings)
            )
        )
        symmetry_inference_enabled = read_enable_object_symmetry_inference(
            self._application_settings
        )
        self.enable_object_symmetry_inference_checkbox.setChecked(
            symmetry_inference_enabled
        )
        self.symmetry_inference_model_combo.setEnabled(symmetry_inference_enabled)
        self.symmetry_inference_model_label.setEnabled(symmetry_inference_enabled)
        self.ceiling_height_estimation_model_combo.setCurrentIndex(
            self.ceiling_height_estimation_model_combo.findData(
                read_ceiling_height_estimation_model(self._application_settings)
            )
        )
        self.make_walls_continuous_checkbox.setChecked(
            read_make_walls_continuous(self._application_settings)
        )
        self._refresh_scene_3d_display_screen_options()
        self._refresh_generation_display_screen_options()
        self._refresh_jobs_window_screen_options()
        self._refresh_atlas_display_screen_options()
        self.automatic_atlas_creation_checkbox.setChecked(
            read_automatic_atlas_creation(self._application_settings)
        )
        self.automatic_atlas_texture_sort_by_pbr_checkbox.setChecked(
            read_automatic_atlas_texture_sort_by_pbr(
                self._application_settings
            )
        )
        self.use_half_mesh_texture_prefix_checkbox.setChecked(
            read_use_half_mesh_texture_prefix(self._application_settings)
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
        self.clear_mask_hotkey_combo.setCurrentIndex(
            max(
                0,
                self.clear_mask_hotkey_combo.findData(
                    read_clear_mask_hotkey(self._application_settings)
                ),
            )
        )
        first_person_navigation_mode = read_first_person_navigation_mode(
            self._application_settings
        )
        first_person_navigation_index = (
            self.first_person_navigation_combo.findData(
                first_person_navigation_mode
            )
        )
        self.first_person_navigation_combo.setCurrentIndex(
            max(0, first_person_navigation_index)
        )
        self.first_person_speed_slider.setValue(
            round(
                read_first_person_speed(self._application_settings)
                / FIRST_PERSON_MOVE_SPEED_STEP_METERS_PER_SECOND
            )
        )
        self.ignore_top_down_ceiling_checkbox.setChecked(
            read_ignore_top_down_ceiling(self._application_settings)
        )
        self.make_floors_double_sided_checkbox.setChecked(
            read_make_floors_double_sided(self._application_settings)
        )
        self.make_ceilings_double_sided_checkbox.setChecked(
            read_make_ceilings_double_sided(self._application_settings)
        )
        self.hide_stair_mesh_when_previewing_checkbox.setChecked(
            read_hide_stair_mesh_when_previewing(self._application_settings)
        )
        self.snap_middle_equal_angle_only_checkbox.setChecked(
            read_snap_middle_equal_angle_only(self._application_settings)
        )
        self.cut_connection_on_double_vertex_deletion_checkbox.setChecked(
            read_cut_connection_on_double_vertex_deletion(
                self._application_settings
            )
        )
        self.wall_verification_model_combo.setCurrentIndex(
            max(
                0,
                self.wall_verification_model_combo.findData(
                    read_wall_verification_model(self._application_settings)
                ),
            )
        )
        wall_verification_enabled = read_include_llm_wall_verification(
            self._application_settings
        )
        self.include_llm_wall_verification_checkbox.setChecked(
            wall_verification_enabled
        )
        self.wall_verification_model_label.setEnabled(
            wall_verification_enabled
        )
        self.wall_verification_model_combo.setEnabled(
            wall_verification_enabled
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
        self.doors_and_windows_point_light_intensity_spinbox.setValue(
            read_doors_and_windows_point_light_intensity(
                self._application_settings
            )
        )
        self._reference_view_presets = read_reference_view_presets(
            self._application_settings
        )
        self._refresh_reference_view_preset_list()
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

    def _handle_plan_correction_model_changed(self, _index: int) -> None:
        """Persist the model used for architectural-plan correction."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            PLAN_CORRECTION_MODEL_SETTING_KEY,
            str(self.plan_correction_model_combo.currentData()),
        )
        self.settings_changed.emit()

    def _handle_reference_generation_model_changed(self, _index: int) -> None:
        """Persist the reference image provider for future generation jobs."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            REFERENCE_GENERATION_MODEL_SETTINGS_KEY,
            str(self.reference_generation_model_combo.currentData()),
        )
        self.settings_changed.emit()

    def _handle_ceiling_height_estimation_model_changed(self, _index: int) -> None:
        """Persist the ceiling-height model without starting an analysis."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            CEILING_HEIGHT_ESTIMATION_MODEL_SETTING_KEY,
            str(self.ceiling_height_estimation_model_combo.currentData()),
        )
        self.settings_changed.emit()

    def _handle_enable_object_symmetry_inference_changed(
        self,
        enabled: bool,
    ) -> None:
        """Enable automatic symmetry checks without losing the chosen model."""

        self.symmetry_inference_model_combo.setEnabled(enabled)
        self.symmetry_inference_model_label.setEnabled(enabled)
        if self._is_loading_settings:
            return
        self._application_settings.set(
            ENABLE_OBJECT_SYMMETRY_INFERENCE_SETTING_KEY,
            bool(enabled),
        )
        self.settings_changed.emit()

    def _handle_symmetry_inference_model_changed(self, _index: int) -> None:
        """Persist the symmetry backend for future reference generation jobs."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            SYMMETRY_INFERENCE_MODEL_SETTING_KEY,
            str(self.symmetry_inference_model_combo.currentData()),
        )
        self.settings_changed.emit()

    def _handle_make_walls_continuous_changed(
        self,
        enabled: bool,
    ) -> None:
        """Persist optional doorway-gap wall continuation."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            MAKE_WALLS_CONTINUOUS_SETTING_KEY,
            bool(enabled),
        )
        self.settings_changed.emit()

    # ### Plan correction pricing ###
    def _start_plan_correction_pricing_lookup(self) -> None:
        """Fetch current official token rates without blocking the Settings UI."""

        if self._is_disposed or self._plan_pricing_lookup_started:
            return
        self._plan_pricing_lookup_started = True
        request = QNetworkRequest(QUrl(OPENAI_PRICING_MARKDOWN_URL))
        request.setTransferTimeout(OPENAI_PRICING_TIMEOUT_MILLISECONDS)
        request.setAttribute(
            QNetworkRequest.Attribute.RedirectPolicyAttribute,
            QNetworkRequest.RedirectPolicy.NoLessSafeRedirectPolicy,
        )
        request.setRawHeader(b"User-Agent", b"HouseMaker/1.0")
        network_manager = self._plan_pricing_network_manager
        if network_manager is None:
            try:
                network_manager = QNetworkAccessManager(self)
            except RuntimeError:
                self._apply_plan_correction_model_pricing({})
                return
            self._plan_pricing_network_manager = network_manager
        try:
            reply = network_manager.get(request)
        except (RuntimeError, TypeError, ValueError):
            self._apply_plan_correction_model_pricing({})
            return

        self._plan_pricing_reply = reply
        self._plan_pricing_response.clear()
        self._plan_pricing_request_failed = False
        reply.readyRead.connect(self._read_plan_correction_pricing_response)
        reply.downloadProgress.connect(
            self._handle_plan_correction_pricing_download_progress
        )
        reply.finished.connect(self._finish_plan_correction_pricing_lookup)

    def _read_plan_correction_pricing_response(self) -> None:
        """Buffer one bounded response chunk from the official pricing page."""

        reply = self._plan_pricing_reply
        if reply is None or self._is_disposed:
            return
        try:
            chunk = bytes(reply.readAll())
        except RuntimeError:
            self._plan_pricing_request_failed = True
            return
        if (
            len(self._plan_pricing_response) + len(chunk)
            > OPENAI_PRICING_RESPONSE_LIMIT_BYTES
        ):
            self._plan_pricing_request_failed = True
            self._plan_pricing_response.clear()
            try:
                reply.abort()
            except RuntimeError:
                pass
            return
        self._plan_pricing_response.extend(chunk)

    def _handle_plan_correction_pricing_download_progress(
        self,
        bytes_received: int,
        bytes_total: int,
    ) -> None:
        """Abort an oversized pricing document before it can grow unbounded."""

        if self._is_disposed or self._plan_pricing_reply is None:
            return
        if bytes_received <= OPENAI_PRICING_RESPONSE_LIMIT_BYTES and (
            bytes_total < 0
            or bytes_total <= OPENAI_PRICING_RESPONSE_LIMIT_BYTES
        ):
            return
        self._plan_pricing_request_failed = True
        self._plan_pricing_response.clear()
        try:
            self._plan_pricing_reply.abort()
        except RuntimeError:
            pass

    def _finish_plan_correction_pricing_lookup(self) -> None:
        """Parse a successful official response and refresh only combo labels."""

        reply = self._plan_pricing_reply
        if reply is None:
            return
        self._read_plan_correction_pricing_response()
        self._plan_pricing_reply = None
        pricing: dict[str, ModelTokenPricing] = {}
        if not self._is_disposed and self._pricing_reply_is_usable(reply):
            try:
                markdown = bytes(self._plan_pricing_response).decode("utf-8")
            except UnicodeDecodeError:
                markdown = ""
            pricing = parse_openai_pricing_markdown(markdown)
        self._plan_pricing_response.clear()
        try:
            reply.deleteLater()
        except RuntimeError:
            pass
        if not self._is_disposed:
            self._apply_plan_correction_model_pricing(pricing)

    def _pricing_reply_is_usable(self, reply: QNetworkReply) -> bool:
        """Return whether a reply is an intact 200 Markdown response."""

        if self._plan_pricing_request_failed:
            return False
        try:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                return False
            status = reply.attribute(
                QNetworkRequest.Attribute.HttpStatusCodeAttribute
            )
            if int(status) != 200:
                return False
            response_url = reply.url()
            if (
                response_url.scheme().lower() != "https"
                or response_url.host().lower() != "developers.openai.com"
                or response_url.path() != "/api/docs/pricing.md"
                or response_url.hasQuery()
                or bool(response_url.userInfo())
                or response_url.port(-1) != -1
            ):
                return False
            content_type = str(
                reply.header(QNetworkRequest.KnownHeaders.ContentTypeHeader)
                or ""
            ).lower()
        except (RuntimeError, TypeError, ValueError):
            return False
        return content_type.split(";", 1)[0].strip() == "text/markdown"

    def _apply_plan_correction_model_pricing(
        self,
        pricing: Mapping[str, ModelTokenPricing],
    ) -> None:
        """Relabel models without changing their IDs, selection, or settings."""

        if self._is_disposed:
            return
        blocker = QSignalBlocker(self.plan_correction_model_combo)
        for index in range(self.plan_correction_model_combo.count()):
            model_id = str(self.plan_correction_model_combo.itemData(index))
            self.plan_correction_model_combo.setItemText(
                index,
                format_plan_correction_model_label(
                    plan_correction_model_label(model_id),
                    model_id,
                    pricing.get(model_id),
                ),
            )
        del blocker

    def _cancel_plan_correction_pricing_lookup(self) -> None:
        """Abort and detach the pending lookup during Settings disposal."""

        reply = self._plan_pricing_reply
        self._plan_pricing_reply = None
        self._plan_pricing_response.clear()
        self._plan_pricing_request_failed = True
        if reply is None:
            return
        for signal, callback in (
            (reply.readyRead, self._read_plan_correction_pricing_response),
            (
                reply.downloadProgress,
                self._handle_plan_correction_pricing_download_progress,
            ),
            (reply.finished, self._finish_plan_correction_pricing_lookup),
        ):
            try:
                signal.disconnect(callback)
            except (RuntimeError, TypeError):
                pass
        try:
            reply.abort()
            reply.deleteLater()
        except RuntimeError:
            pass

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
        self._refresh_scene_3d_display_screen_options()
        self._refresh_generation_display_screen_options()
        self._refresh_jobs_window_screen_options()
        self._refresh_atlas_display_screen_options()
        if not self._is_loading_settings:
            # Reapply even unchanged "Primary display" selections because the
            # primary screen itself may have changed or been disconnected.
            self.settings_changed.emit()

    def _refresh_scene_3d_display_screen_options(self) -> None:
        selected_screen_id = read_scene_3d_display_screen_id(
            self._application_settings
        )
        blocker = QSignalBlocker(self.scene_3d_display_screen_combo)
        self.scene_3d_display_screen_combo.clear()
        self.scene_3d_display_screen_combo.addItem("None", None)
        for option in connected_fullscreen_3d_viewer_display_options():
            self.scene_3d_display_screen_combo.addItem(
                option.label,
                option.screen_id,
            )
        selected_index = self.scene_3d_display_screen_combo.findData(
            selected_screen_id
        )
        self.scene_3d_display_screen_combo.setCurrentIndex(
            selected_index if selected_index >= 0 else 0
        )
        del blocker

    def _selected_scene_3d_display_screen_id(self) -> str | None:
        return _normalize_fullscreen_3d_viewer_screen_id(
            self.scene_3d_display_screen_combo.currentData()
        )

    def _refresh_generation_display_screen_options(self) -> None:
        selected_screen_id = read_generation_display_screen_id(
            self._application_settings
        )
        blocker = QSignalBlocker(self.generation_display_screen_combo)
        self.generation_display_screen_combo.clear()
        self.generation_display_screen_combo.addItem("None", None)
        for option in connected_fullscreen_3d_viewer_display_options():
            self.generation_display_screen_combo.addItem(
                option.label,
                option.screen_id,
            )
        selected_index = self.generation_display_screen_combo.findData(
            selected_screen_id
        )
        self.generation_display_screen_combo.setCurrentIndex(
            selected_index if selected_index >= 0 else 0
        )
        del blocker

    def _selected_generation_display_screen_id(self) -> str | None:
        return _normalize_fullscreen_3d_viewer_screen_id(
            self.generation_display_screen_combo.currentData()
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

    def _refresh_atlas_display_screen_options(self) -> None:
        selected_screen_id = read_atlas_display_screen_id(
            self._application_settings
        )
        blocker = QSignalBlocker(self.atlas_display_screen_combo)
        self.atlas_display_screen_combo.clear()
        self.atlas_display_screen_combo.addItem("None", None)
        for option in connected_fullscreen_3d_viewer_display_options():
            self.atlas_display_screen_combo.addItem(
                option.label,
                option.screen_id,
            )
        selected_index = self.atlas_display_screen_combo.findData(
            selected_screen_id
        )
        self.atlas_display_screen_combo.setCurrentIndex(
            selected_index if selected_index >= 0 else 0
        )
        del blocker

    def _selected_atlas_display_screen_id(self) -> str | None:
        return _normalize_fullscreen_3d_viewer_screen_id(
            self.atlas_display_screen_combo.currentData()
        )

    def _handle_scene_3d_display_screen_changed(
        self,
        _index: int,
    ) -> None:
        if self._is_loading_settings:
            return
        self._application_settings.set(
            SCENE_3D_DISPLAY_SCREEN_SETTING_KEY,
            self._selected_scene_3d_display_screen_id(),
        )
        self.settings_changed.emit()

    def _handle_generation_display_screen_changed(
        self,
        _index: int,
    ) -> None:
        if self._is_loading_settings:
            return
        self._application_settings.set(
            GENERATION_DISPLAY_SCREEN_SETTING_KEY,
            self._selected_generation_display_screen_id(),
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

    def _handle_atlas_display_screen_changed(self, _index: int) -> None:
        if self._is_loading_settings:
            return
        self._application_settings.set(
            ATLAS_DISPLAY_SCREEN_SETTING_KEY,
            self._selected_atlas_display_screen_id(),
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

    def _handle_automatic_atlas_creation_changed(self, checked: bool) -> None:
        """Persist whether full automatic Atlas destinations may overflow."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            AUTOMATIC_ATLAS_CREATION_SETTING_KEY,
            bool(checked),
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

    def _handle_use_half_mesh_texture_prefix_changed(
        self,
        checked: bool,
    ) -> None:
        """Persist whether half meshes use dedicated prefixed Atlases."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            USE_HALF_MESH_TEXTURE_PREFIX_SETTING_KEY,
            bool(checked),
        )
        self.settings_changed.emit()

    def _selected_canvas_3d_navigation_toggle_hotkey(self) -> str:
        hotkey = _hotkey_from_key_sequence(
            self.canvas_3d_navigation_toggle_hotkey_edit.keySequence()
        )
        return hotkey or DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY

    def _handle_clear_mask_hotkey_changed(self, _index: int) -> None:
        if self._is_loading_settings:
            return
        self._application_settings.set(
            CLEAR_MASK_HOTKEY_SETTING_KEY,
            str(self.clear_mask_hotkey_combo.currentData()),
        )
        self.settings_changed.emit()

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

    def _handle_first_person_navigation_mode_changed(
        self,
        _index: int,
    ) -> None:
        """Persist the movement model used by first-person viewers."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            FIRST_PERSON_NAVIGATION_MODE_SETTING_KEY,
            str(self.first_person_navigation_combo.currentData()),
        )
        self.settings_changed.emit()

    def _handle_first_person_speed_changed(self, _value: int) -> None:
        """Persist and display the first-person movement speed."""

        speed = self._first_person_speed_from_slider()
        self.first_person_speed_value_label.setText(f"{speed:.1f} m/s")
        if self._is_loading_settings:
            return
        self._application_settings.set(
            FIRST_PERSON_SPEED_SETTING_KEY,
            speed,
        )
        self.settings_changed.emit()

    def _first_person_speed_from_slider(self) -> float:
        """Return the slider's integer-tenths value in meters per second."""

        return round(
            self.first_person_speed_slider.value()
            * FIRST_PERSON_MOVE_SPEED_STEP_METERS_PER_SECOND,
            1,
        )

    def _handle_ignore_top_down_ceiling_changed(self, enabled: bool) -> None:
        """Persist whether orbit navigation suppresses Canvas ceilings."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            IGNORE_TOP_DOWN_CEILING_SETTING_KEY,
            bool(enabled),
        )
        self.settings_changed.emit()

    def _handle_make_floors_double_sided_changed(self, enabled: bool) -> None:
        """Persist whether floor surfaces render from both sides."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            MAKE_FLOORS_DOUBLE_SIDED_SETTING_KEY,
            bool(enabled),
        )
        self.settings_changed.emit()

    def _handle_make_ceilings_double_sided_changed(
        self,
        enabled: bool,
    ) -> None:
        """Persist whether ceiling surfaces render from both sides."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            MAKE_CEILINGS_DOUBLE_SIDED_SETTING_KEY,
            bool(enabled),
        )
        self.settings_changed.emit()

    def _handle_hide_stair_mesh_when_previewing_changed(
        self, enabled: bool
    ) -> None:
        """Persist the visibility of a stair while its edit is previewed."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            HIDE_STAIR_MESH_WHEN_PREVIEWING_SETTING_KEY,
            bool(enabled),
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

    def _handle_snap_middle_equal_angle_only_changed(
        self,
        enabled: bool,
    ) -> None:
        """Persist the Canvas midpoint angle-filtering preference."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            SNAP_MIDDLE_EQUAL_ANGLE_ONLY_SETTING_KEY,
            bool(enabled),
        )
        self.settings_changed.emit()

    def _handle_cut_connection_on_double_vertex_deletion_changed(
        self,
        enabled: bool,
    ) -> None:
        """Persist whether two selected vertices are disconnected first."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            CUT_CONNECTION_ON_DOUBLE_VERTEX_DELETION_SETTING_KEY,
            bool(enabled),
        )
        self.settings_changed.emit()

    # ### Wall-generation review settings ###
    def _handle_include_llm_wall_verification_changed(
        self,
        enabled: bool,
    ) -> None:
        """Enable model selection and persist the optional review pass."""

        self.wall_verification_model_label.setEnabled(enabled)
        self.wall_verification_model_combo.setEnabled(enabled)
        if self._is_loading_settings:
            return
        self._application_settings.set(
            INCLUDE_LLM_WALL_VERIFICATION_SETTING_KEY,
            bool(enabled),
        )
        self.settings_changed.emit()

    def _handle_wall_verification_model_changed(self, _index: int) -> None:
        """Remember the semantic model without changing image correction."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            WALL_VERIFICATION_MODEL_SETTING_KEY,
            str(self.wall_verification_model_combo.currentData()),
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

    def _handle_doors_and_windows_point_light_intensity_changed(
        self,
        value: float,
    ) -> None:
        """Persist the hover-light strength used by the opening preview."""

        if self._is_loading_settings:
            return
        self._application_settings.set(
            DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY_SETTING_KEY,
            float(value),
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
    """Read the legacy shared-viewer display identity."""

    return _normalize_fullscreen_3d_viewer_screen_id(
        application_settings.get(FULLSCREEN_3D_VIEWER_SCREEN_SETTING_KEY)
    )


def read_scene_3d_display_screen_id(
    application_settings: ApplicationSettingsStore,
) -> str | None:
    """Read the 3D-scene display, falling back to the legacy setting."""

    missing_value = object()
    saved_value = application_settings.get(
        SCENE_3D_DISPLAY_SCREEN_SETTING_KEY,
        missing_value,
    )
    if saved_value is missing_value:
        saved_value = application_settings.get(
            FULLSCREEN_3D_VIEWER_SCREEN_SETTING_KEY
        )
    return _normalize_fullscreen_3d_viewer_screen_id(saved_value)


def read_generation_display_screen_id(
    application_settings: ApplicationSettingsStore,
) -> str | None:
    """Read the persisted detached Generation display identity."""

    return _normalize_fullscreen_3d_viewer_screen_id(
        application_settings.get(GENERATION_DISPLAY_SCREEN_SETTING_KEY)
    )


def read_jobs_window_screen_id(
    application_settings: ApplicationSettingsStore,
) -> str | None:
    """Read the persisted detached Jobs-window display identity."""

    return _normalize_fullscreen_3d_viewer_screen_id(
        application_settings.get(JOBS_WINDOW_SCREEN_SETTING_KEY)
    )


def read_atlas_display_screen_id(
    application_settings: ApplicationSettingsStore,
) -> str | None:
    """Read the persisted detached Atlas display identity."""

    return _normalize_fullscreen_3d_viewer_screen_id(
        application_settings.get(ATLAS_DISPLAY_SCREEN_SETTING_KEY)
    )


# ### Atlas setting helpers ###
def read_automatic_atlas_creation(
    application_settings: ApplicationSettingsStore,
) -> bool:
    """Read the persisted automatic Atlas-creation policy safely."""

    value = application_settings.get(
        AUTOMATIC_ATLAS_CREATION_SETTING_KEY,
        DEFAULT_AUTOMATIC_ATLAS_CREATION,
    )
    if isinstance(value, bool):
        return value
    return DEFAULT_AUTOMATIC_ATLAS_CREATION


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


def read_use_half_mesh_texture_prefix(
    application_settings: ApplicationSettingsStore,
) -> bool:
    """Read the persisted half-mesh Atlas routing policy safely."""

    value = application_settings.get(
        USE_HALF_MESH_TEXTURE_PREFIX_SETTING_KEY,
        DEFAULT_USE_HALF_MESH_TEXTURE_PREFIX,
    )
    if isinstance(value, bool):
        return value
    return DEFAULT_USE_HALF_MESH_TEXTURE_PREFIX


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


# ### Preview lighting setting helpers ###
def read_doors_and_windows_point_light_intensity(
    application_settings: ApplicationSettingsStore,
) -> float:
    """Read the Doors and Windows hover-light strength safely."""

    normalized = _normalize_doors_and_windows_point_light_intensity(
        application_settings.get(
            DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY_SETTING_KEY,
            DEFAULT_DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY,
        )
    )
    if normalized is None:
        return DEFAULT_DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY
    return normalized


def _normalize_doors_and_windows_point_light_intensity(
    value: object,
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or not (
        MIN_DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY
        <= normalized
        <= MAX_DOORS_AND_WINDOWS_POINT_LIGHT_INTENSITY
    ):
        return None
    return normalized


# ### Canvas snapping setting helpers ###
def read_snap_middle_equal_angle_only(
    application_settings: ApplicationSettingsStore,
) -> bool:
    """Read the persisted Canvas midpoint angle filter safely."""

    value = application_settings.get(
        SNAP_MIDDLE_EQUAL_ANGLE_ONLY_SETTING_KEY,
        DEFAULT_SNAP_MIDDLE_EQUAL_ANGLE_ONLY,
    )
    if isinstance(value, bool):
        return value
    return DEFAULT_SNAP_MIDDLE_EQUAL_ANGLE_ONLY


# ### Canvas deletion setting helpers ###
def read_cut_connection_on_double_vertex_deletion(
    application_settings: ApplicationSettingsStore,
) -> bool:
    """Read the two-vertex disconnect preference with a safe default."""

    value = application_settings.get(
        CUT_CONNECTION_ON_DOUBLE_VERTEX_DELETION_SETTING_KEY,
        DEFAULT_CUT_CONNECTION_ON_DOUBLE_VERTEX_DELETION,
    )
    if isinstance(value, bool):
        return value
    return DEFAULT_CUT_CONNECTION_ON_DOUBLE_VERTEX_DELETION


# ### Wall-generation review setting helpers ###
def read_include_llm_wall_verification(
    application_settings: ApplicationSettingsStore,
) -> bool:
    """Treat malformed stored review preferences as disabled."""

    value = application_settings.get(
        INCLUDE_LLM_WALL_VERIFICATION_SETTING_KEY,
        DEFAULT_INCLUDE_LLM_WALL_VERIFICATION,
    )
    if isinstance(value, bool):
        return value
    return DEFAULT_INCLUDE_LLM_WALL_VERIFICATION


def read_wall_verification_model(
    application_settings: ApplicationSettingsStore,
) -> str:
    """Restore a supported semantic model, with the default as fallback."""

    value = application_settings.get(
        WALL_VERIFICATION_MODEL_SETTING_KEY,
        DEFAULT_WALL_VERIFICATION_MODEL,
    )
    if isinstance(value, str) and value in WALL_VERIFICATION_MODELS:
        return value
    return DEFAULT_WALL_VERIFICATION_MODEL


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


# ### Canvas navigation setting helpers ###
def read_first_person_navigation_mode(
    application_settings: ApplicationSettingsStore,
) -> str:
    """Read the first-person movement model with a safe default."""

    value = application_settings.get(
        FIRST_PERSON_NAVIGATION_MODE_SETTING_KEY,
        DEFAULT_FIRST_PERSON_NAVIGATION_MODE,
    )
    try:
        return normalize_first_person_navigation_mode(value)
    except ValueError:
        return DEFAULT_FIRST_PERSON_NAVIGATION_MODE


def read_first_person_speed(
    application_settings: ApplicationSettingsStore,
) -> float:
    """Read the first-person movement speed with a safe default."""

    value = application_settings.get(
        FIRST_PERSON_SPEED_SETTING_KEY,
        DEFAULT_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND,
    )
    try:
        return normalize_first_person_speed_meters_per_second(value)
    except (TypeError, ValueError):
        return DEFAULT_FIRST_PERSON_MOVE_SPEED_METERS_PER_SECOND


def read_ignore_top_down_ceiling(
    application_settings: ApplicationSettingsStore,
) -> bool:
    """Read the persisted orbit-mode ceiling suppression safely."""

    value = application_settings.get(
        IGNORE_TOP_DOWN_CEILING_SETTING_KEY,
        DEFAULT_IGNORE_TOP_DOWN_CEILING,
    )
    if isinstance(value, bool):
        return value
    return DEFAULT_IGNORE_TOP_DOWN_CEILING


def read_make_floors_double_sided(
    application_settings: ApplicationSettingsStore,
) -> bool:
    """Read the persisted floor-sidedness preference safely."""

    value = application_settings.get(
        MAKE_FLOORS_DOUBLE_SIDED_SETTING_KEY,
        DEFAULT_MAKE_FLOORS_DOUBLE_SIDED,
    )
    if isinstance(value, bool):
        return value
    return DEFAULT_MAKE_FLOORS_DOUBLE_SIDED


def read_make_ceilings_double_sided(
    application_settings: ApplicationSettingsStore,
) -> bool:
    """Read the persisted ceiling-sidedness preference safely."""

    value = application_settings.get(
        MAKE_CEILINGS_DOUBLE_SIDED_SETTING_KEY,
        DEFAULT_MAKE_CEILINGS_DOUBLE_SIDED,
    )
    if isinstance(value, bool):
        return value
    return DEFAULT_MAKE_CEILINGS_DOUBLE_SIDED


def read_hide_stair_mesh_when_previewing(
    application_settings: ApplicationSettingsStore,
) -> bool:
    """Read the stair-preview visibility preference with a safe default."""

    value = application_settings.get(
        HIDE_STAIR_MESH_WHEN_PREVIEWING_SETTING_KEY,
        DEFAULT_HIDE_STAIR_MESH_WHEN_PREVIEWING,
    )
    if isinstance(value, bool):
        return value
    return DEFAULT_HIDE_STAIR_MESH_WHEN_PREVIEWING


def read_canvas_3d_navigation_toggle_hotkey(
    application_settings: ApplicationSettingsStore,
) -> str:
    """Read a valid Canvas navigation shortcut, with a safe default."""

    hotkey = _normalize_canvas_3d_navigation_toggle_hotkey(
        application_settings.get(CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY_SETTING_KEY)
    )
    return hotkey or DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY


def read_clear_mask_hotkey(
    application_settings: ApplicationSettingsStore,
) -> str:
    """Read a listed Generation mask shortcut, falling back safely."""

    hotkey = application_settings.get(
        CLEAR_MASK_HOTKEY_SETTING_KEY,
        DEFAULT_CLEAR_MASK_HOTKEY,
    )
    return (
        hotkey
        if isinstance(hotkey, str) and hotkey in CLEAR_MASK_HOTKEY_VALUES
        else DEFAULT_CLEAR_MASK_HOTKEY
    )


# ### Room analysis setting helpers ###
def read_ceiling_height_estimation_model(
    application_settings: ApplicationSettingsStore | None,
) -> str:
    """Read a valid model, retaining preferences saved by the old dialog."""

    if application_settings is None:
        return CEILING_HEIGHT_INFERENCE_MODEL
    model = application_settings.get(
        CEILING_HEIGHT_ESTIMATION_MODEL_SETTING_KEY,
        CEILING_HEIGHT_INFERENCE_MODEL,
    )
    return (
        model
        if isinstance(model, str) and model in CEILING_HEIGHT_ESTIMATION_MODELS
        else CEILING_HEIGHT_INFERENCE_MODEL
    )


# ### Reference generation setting helpers ###
def read_reference_generation_model(
    application_settings: ApplicationSettingsStore | None,
) -> str:
    """Read the reference provider while preserving the existing saved key."""

    if application_settings is None:
        return DEFAULT_REFERENCE_GENERATION_MODEL
    model = application_settings.get(
        REFERENCE_GENERATION_MODEL_SETTINGS_KEY,
        DEFAULT_REFERENCE_GENERATION_MODEL,
    )
    return (
        model
        if isinstance(model, str) and model in REFERENCE_GENERATION_MODELS
        else DEFAULT_REFERENCE_GENERATION_MODEL
    )


def read_enable_object_symmetry_inference(
    application_settings: ApplicationSettingsStore | None,
) -> bool:
    """Read automatic object symmetry checking, defaulting to opt-in."""

    if application_settings is None:
        return DEFAULT_ENABLE_OBJECT_SYMMETRY_INFERENCE
    enabled = application_settings.get(
        ENABLE_OBJECT_SYMMETRY_INFERENCE_SETTING_KEY,
        DEFAULT_ENABLE_OBJECT_SYMMETRY_INFERENCE,
    )
    return (
        enabled
        if isinstance(enabled, bool)
        else DEFAULT_ENABLE_OBJECT_SYMMETRY_INFERENCE
    )


def read_symmetry_inference_model(
    application_settings: ApplicationSettingsStore | None,
) -> str:
    """Read a supported reference-symmetry model with a safe local default."""

    if application_settings is None:
        return DEFAULT_SYMMETRY_INFERENCE_MODEL
    model = application_settings.get(
        SYMMETRY_INFERENCE_MODEL_SETTING_KEY,
        DEFAULT_SYMMETRY_INFERENCE_MODEL,
    )
    return (
        model
        if isinstance(model, str) and model in SYMMETRY_INFERENCE_MODELS
        else DEFAULT_SYMMETRY_INFERENCE_MODEL
    )


# ### Plan correction setting helpers ###
def read_plan_correction_model(
    application_settings: ApplicationSettingsStore,
) -> str:
    """Read the architectural-plan correction model with a safe default."""

    model = application_settings.get(
        PLAN_CORRECTION_MODEL_SETTING_KEY,
        DEFAULT_PLAN_CORRECTION_MODEL,
    )
    if not isinstance(model, str) or model not in PLAN_CORRECTION_MODELS:
        return DEFAULT_PLAN_CORRECTION_MODEL
    return model


def read_make_walls_continuous(
    application_settings: ApplicationSettingsStore,
) -> bool:
    """Read optional doorway-gap wall continuation with a safe default."""

    enabled = application_settings.get(
        MAKE_WALLS_CONTINUOUS_SETTING_KEY,
        DEFAULT_MAKE_WALLS_CONTINUOUS,
    )
    if isinstance(enabled, bool):
        return enabled
    return DEFAULT_MAKE_WALLS_CONTINUOUS


# ### Qt application helpers ###
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
    if any(
        key_sequence.matches(undo_binding)
        == QKeySequence.SequenceMatch.ExactMatch
        for undo_binding in QKeySequence.keyBindings(
            QKeySequence.StandardKey.Undo
        )
    ):
        return None
    key_combination = key_sequence[0]
    if key_combination.key() in _MODIFIER_ONLY_SHORTCUT_KEYS:
        return None
    if (
        key_combination.key() in _FIRST_PERSON_RESERVED_SHORTCUT_KEYS
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
