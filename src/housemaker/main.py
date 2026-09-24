# ### Imports ###
from __future__ import annotations

import copy
import math
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import trimesh
from PIL import Image
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPointF,
    QSignalBlocker,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QKeySequence, QPalette, QShortcut, QWheelEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
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
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.architectural_surface_edits import (
    SurfaceFaceExtrusionRequest,
    SurfaceTopologyEditResult,
    SurfaceVertexInsertionRequest,
    build_surface_drawing_overlay,
    delete_directly_drawn_surface_faces,
    extrude_surface_faces,
    place_surface_vertex,
)
from housemaker.atlas_export import (
    AtlasDrawCallEstimate,
    SurfaceAmbientOcclusionAtlasContext,
    SurfaceAmbientOcclusionBakeResult,
    apply_texture_atlases_to_export,
    bake_surface_ambient_occlusion_for_atlas,
    estimate_texture_atlas_draw_calls,
    prepare_surface_ambient_occlusion_preview_for_atlas,
)
from housemaker.automatic_wall_detection import (
    PlanWallAnalysis,
    PlanWallDetectionOptions,
    PlanWallDetectionResult,
    analyze_plan_wall_image,
    reconstruct_plan_walls,
)
from housemaker.blueprint_canvas import (
    CANVAS_SNAPSHOT_ACTION_GENERATED_WALLS,
    CANVAS_SNAPSHOT_ACTION_OPEN_SPACE,
    CANVAS_SNAPSHOT_ACTION_VERTEX_DELETION,
    BlueprintCanvas,
    CanvasSnapshot,
    PlanImageEraseCommit,
)
from housemaker.canvas_openings import (
    CANVAS_OPENING_DOORWAY,
    CANVAS_OPENING_WINDOW,
    CanvasOpeningEdit,
    CanvasOpeningReference,
    CanvasOpeningTarget,
    apply_canvas_opening_edit,
    build_canvas_opening_targets,
)
from housemaker.canvas_surface_edits import (
    CANVAS_SURFACE_EDIT_FLOOR_THICKNESS,
    CANVAS_SURFACE_EDIT_WALL_KINDS,
    CANVAS_SURFACE_EDIT_WALL_TRANSLATION,
    AppliedCanvasSurfaceEdit,
    CanvasSurfaceEdit,
    CanvasSurfaceEditHandleTarget,
    apply_canvas_surface_edit,
    apply_canvas_wall_edit_batch,
    build_canvas_surface_edit_targets,
    canvas_surface_edit_targets_are_at_baseline,
    rebase_canvas_floor_edit_target,
    rebase_canvas_wall_edit_targets_batch,
    restore_canvas_surface_edit,
    restore_canvas_wall_edit_batch,
    validate_canvas_surface_edit_geometry,
    validate_canvas_wall_edit_batch_geometry,
)
from housemaker.external_viewer_host import ExternalFullscreenViewerHost
from housemaker.generation_jobs import GenerationJobManager, JobsWindow
from housemaker.generation_state import (
    GeneratedObjectPlacement,
    GeneratedObjectRecord,
    GenerationData,
)
from housemaker.generation_workspace import (
    FACE_EDIT_TEXTURE_STALE_PIPELINE_KEY,
    GenerationWorkspace,
)
from housemaker.glb import (
    DEFAULT_WALL_HEIGHT_METERS,
    STAIR_PART_TREADS,
    GeneratedModel,
    PlacedGeneratedModel,
    PreviewStairPart,
    build_canvas_stair_part_targets,
    build_stair_meshes,
    build_texture_preview_plane_model,
    compose_placed_generated_models,
    compose_placed_generated_models_preview,
    convert_to_export_scene_model,
    convert_to_glb,
    convert_to_preview_model,
    export_glb_file,
    import_generated_glb,
)
from housemaker.level_coordinates import (
    build_doorway_world_outline_positions,
    build_level_base_z_lookup,
    get_level_world_pivot,
    level_image_to_world_xy,
    level_world_to_image_xy,
)
from housemaker.merged_generation_workspace import MergedGenerationWorkspace
from housemaker.models import (
    DEFAULT_CANVAS_LEVEL_SCALE,
    DEFAULT_CANVAS_OFFSET_PIXELS,
    DEFAULT_DOORWAY_ARCH_AMOUNT,
    DEFAULT_FLOOR_THICKNESS_METERS,
    DEFAULT_LEVEL_OFFSET_METERS,
    DEFAULT_LEVEL_SCALE,
    DEFAULT_STAIR_NOSING_PLACEMENTS,
    DEFAULT_STAIR_STARTING_STEP,
    DEFAULT_STAIR_STARTING_STEP_EDGE_POINTS,
    DEFAULT_STAIR_STARTING_STEP_EDGE_RADIUS_METERS,
    DEFAULT_STAIR_STRINGER_PLACEMENT,
    DEFAULT_STAIR_TARGET_RISE_METERS,
    DEFAULT_STAIR_TREAD_EDGE_PROFILE,
    DEFAULT_STAIR_TREAD_EDGE_RADIUS_METERS,
    DEFAULT_STAIR_TREAD_OVERHANG_METERS,
    DEFAULT_STAIR_TREAD_THICKNESS_METERS,
    DEFAULT_STAIR_TYPE,
    DOORWAY_SHAPE_ARCH,
    DOORWAY_SHAPE_RECTANGULAR,
    GROUND_LEVEL_INDEX,
    MAX_CANVAS_LEVEL_SCALE,
    MAX_DOORWAY_ARCH_AMOUNT,
    MAX_FLOOR_THICKNESS_METERS,
    MAX_LEVEL_SCALE,
    MAX_STAIR_STARTING_STEP_EDGE_POINTS,
    MAX_STAIR_STARTING_STEP_EDGE_RADIUS_METERS,
    MAX_STAIR_TREAD_EDGE_RADIUS_METERS,
    MIN_CANVAS_LEVEL_SCALE,
    MIN_DOORWAY_ARCH_AMOUNT,
    MIN_FLOOR_THICKNESS_METERS,
    MIN_LEVEL_SCALE,
    MIN_STAIR_STARTING_STEP_EDGE_POINTS,
    MIN_STAIR_STARTING_STEP_EDGE_RADIUS_METERS,
    MIN_STAIR_TREAD_EDGE_RADIUS_METERS,
    STAIR_NOSING_FRONT,
    STAIR_NOSING_LEFT,
    STAIR_NOSING_RIGHT,
    STAIR_STARTING_STEP_BULLNOSE,
    STAIR_STARTING_STEP_CURTAIL,
    STAIR_STARTING_STEP_NONE,
    STAIR_STRINGER_BOTH,
    STAIR_STRINGER_LEFT,
    STAIR_STRINGER_NONE,
    STAIR_STRINGER_RIGHT,
    STAIR_STYLE_FLOATING,
    STAIR_STYLE_FLOATING_WITH_RISER,
    STAIR_TREAD_EDGE_ROUNDED,
    STAIR_TREAD_EDGE_STRAIGHT,
    STAIR_TYPE_FLOATING,
    STAIR_TYPE_SUPPORTED,
    DoorwayData,
    DoorwayPreset,
    EditableSurfaceMeshData,
    LevelData,
    StairData,
    StairSectionData,
    VertexData,
    WindowData,
    calculate_stair_step_layout,
    create_default_doorway_presets,
    create_default_levels,
    create_fallback_doorway_preset,
)
from housemaker.pbr_maps import (
    ATLAS_MAP_BASE_COLOR,
    ATLAS_MAP_TYPES,
    PBR_MAP_ROUGHNESS,
)
from housemaker.plan_correction_models import (
    OPENAI_PLAN_CORRECTION_MODELS,
    PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
    PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
    plan_correction_model_label,
)
from housemaker.plan_image_correction import (
    CORRECTION_METHOD_OPENAI,
    CORRECTION_METHOD_QWEN,
    PlanCorrectionProgress,
    PlanImageCorrectionResult,
    correct_plan_image,
)
from housemaker.project_io import ProjectData, load_project, save_project
from housemaker.qwen_plan_image_correction import (
    QwenPlanCorrectionError,
    create_default_qwen_plan_image_editor,
)
from housemaker.settings_widget import (
    DEFAULT_MESH_EDIT_UPDATE_DELAY_SECONDS,
    DEFAULT_WALL_VERTEX_UPDATE_DELAY_SECONDS,
    SettingsWidget,
    resolve_fullscreen_3d_viewer_screen,
)
from housemaker.surface_geometry import (
    SURFACE_TYPE_WALL,
    FixedSurface,
    WallWindowPlacement,
    add_wall_window,
    build_fixed_surfaces,
)
from housemaker.surface_materials import (
    LEGACY_SURFACE_ROUGHNESS_FACTOR,
    SurfaceMaterialSourceSpec,
)
from housemaker.surface_orientation_edits import (
    flip_surface_orientation,
    remap_flipped_surface_ids_with_lineage,
)
from housemaker.surface_texture_state import (
    SURFACE_TILING_MODE_EDGE_VARIANTS,
    SurfaceTextureAssignment,
    SurfaceTextureData,
)
from housemaker.surface_texture_tiling_dialog import (
    SurfaceTextureTilingPreviewDialog,
)
from housemaker.surface_texture_workspace import (
    PreparedSurfaceTextureTilingRepair,
    SurfaceTextureGenerationWorkspace,
    SurfaceTextureTilingPreparationSnapshot,
    SurfaceTextureTilingRevision,
    prepare_surface_texture_tiling_repair,
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
    TextureAtlasPlacement,
    TextureAtlasRecord,
)
from housemaker.texture_atlas_workspace import (
    AtlasObjectTextureSource,
    AtlasSurfaceTextureEntry,
    TextureAtlasWorkspace,
    build_atlas_wall_texture_source_id,
    choose_atlas_texture_resolution,
    get_atlas_wall_texture_assignment_id,
    is_atlas_wall_texture_source_id,
    load_atlas_object_texture_source,
)
from housemaker.viewer import (
    GlbViewerWidget,
    SceneObjectPlacementCandidate,
    build_scene_object_placement_group_candidate,
)
from housemaker.wall_mirroring import (
    WallMirrorTopologyResult,
    WallMirrorVertexLink,
    find_next_wall_mirror_target_level_index,
    get_wall_mirror_vertex_ids,
    mirror_wall_vertex_group,
    reconcile_wall_mirror_topology,
    remove_wall_vertex_mirrors,
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
SURFACE_ATLAS_ROUGHNESS_BYTE = round(LEGACY_SURFACE_ROUGHNESS_FACTOR * 255.0)
DELAYED_CANVAS_SURFACE_EDIT_KINDS = frozenset(
    (
        *CANVAS_SURFACE_EDIT_WALL_KINDS,
        CANVAS_SURFACE_EDIT_FLOOR_THICKNESS,
    )
)
LEVEL_POSITION_ITEM_ROLE = int(Qt.ItemDataRole.UserRole)
GROUND_LEVEL_BACKGROUND_BLEND = 0.12
LEVEL_SCALE_SLIDER_FACTOR = 1000
LEVEL_OFFSET_SLIDER_FACTOR = 100
LEVEL_OFFSET_SLIDER_MIN_METERS = -100.0
LEVEL_OFFSET_SLIDER_MAX_METERS = 100.0
CANVAS_LEVEL_SCALE_SLIDER_FACTOR = 100
CANVAS_OFFSET_SLIDER_FACTOR = 100
CANVAS_OFFSET_SLIDER_MIN_PIXELS = -2000.0
CANVAS_OFFSET_SLIDER_MAX_PIXELS = 2000.0
SURFACE_AO_SHUTDOWN_WAIT_MILLISECONDS = 100
SURFACE_AO_PREVIEW_REFRESH_DELAY_MILLISECONDS = 150
ATLAS_DRAW_CALL_ESTIMATE_REFRESH_DELAY_MILLISECONDS = 200
STAIR_PREVIEW_UPDATE_DELAY_MILLISECONDS = 35
PLAN_IMAGE_CORRECTION_SHUTDOWN_WAIT_MILLISECONDS = 100
PLAN_WALL_DETECTION_SHUTDOWN_WAIT_MILLISECONDS = 100
PLAN_WALL_PREVIEW_REFRESH_DELAY_MILLISECONDS = 75


# ### Plan-image correction jobs ###
class _PlanImageCorrectionThread(QThread):
    """Correct one immutable plan photograph outside the GUI thread."""

    progress = Signal(object)

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        api_key: str,
        model: str = PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
        parent: QObject | None = None,
        *,
        make_walls_continuous: bool = False,
    ) -> None:
        super().__init__(parent)
        self._input_path = input_path
        self._output_path = output_path
        self._api_key = api_key
        self._model = model
        self._make_walls_continuous = bool(make_walls_continuous)
        self.result: PlanImageCorrectionResult | None = None
        self.error_message: str | None = None
        self.was_cancelled = False

    @property
    def model(self) -> str:
        """Return the immutable model selected for this job."""

        return self._model

    def run(self) -> None:  # type: ignore[override]
        try:
            result = correct_plan_image(
                self._input_path,
                self._output_path,
                api_key=self._api_key,
                model=self._model,
                make_walls_continuous=self._make_walls_continuous,
                progress_callback=self.progress.emit,
                cancellation_check=self.isInterruptionRequested,
            )
        except Exception as error:  # noqa: BLE001 - worker failures cross Qt safely.
            if self.isInterruptionRequested():
                self.was_cancelled = True
            else:
                self.error_message = str(error) or type(error).__name__
            return
        if self.isInterruptionRequested():
            self.was_cancelled = True
            return
        self.result = result


@dataclass
class _PlanImageCorrectionRuntime:
    """GUI-owned lifecycle and stale-input guard for one correction job."""

    source_path: Path
    source_revision: tuple[object, ...]
    output_path: Path
    job_id: str
    thread: _PlanImageCorrectionThread
    cancel_requested: bool = False


# ### Automatic wall-generation jobs ###
class _PlanWallDetectionThread(QThread):
    """Analyze one immutable plan image outside the GUI thread."""

    def __init__(
        self,
        image_path: Path,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._image_path = Path(image_path)
        self.analysis: PlanWallAnalysis | None = None
        self.error_message: str | None = None
        self.was_cancelled = False

    def run(self) -> None:  # type: ignore[override]
        try:
            analysis = analyze_plan_wall_image(
                self._image_path,
                cancellation_check=self.isInterruptionRequested,
            )
        except Exception as error:  # noqa: BLE001 - worker failures cross Qt safely.
            if self.isInterruptionRequested():
                self.was_cancelled = True
            else:
                self.error_message = str(error) or type(error).__name__
            return
        if self.isInterruptionRequested():
            self.was_cancelled = True
            return
        self.analysis = analysis


@dataclass
class _PlanWallDetectionRuntime:
    """GUI-owned lifecycle and stale-input guard for wall analysis."""

    source_path: Path
    source_revision: tuple[object, ...]
    topology_signature: tuple[object, ...]
    baseline_vertex_data: VertexData
    job_id: str
    thread: _PlanWallDetectionThread
    cancel_requested: bool = False


@dataclass(frozen=True)
class _PlanWallPreviewSession:
    """Reusable wall evidence and immutable inputs for live reconstruction."""

    level_index: int
    source_revision: tuple[object, ...]
    topology_signature: tuple[object, ...]
    baseline_vertex_data: VertexData
    existing_edge_keys: frozenset[tuple[int, int]]
    analysis: PlanWallAnalysis


# ### Atlas ambient-occlusion jobs ###
@dataclass(frozen=True)
class _PreAtlasExportScene:
    """The shared scene and source bindings before Atlas UV remapping."""

    model: GeneratedModel
    placed_models: tuple[PlacedGeneratedModel, ...]
    surface_source_ids: dict[str, str]

    @property
    def required_source_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *(placement.object_id for placement in self.placed_models),
                    *self.surface_source_ids.values(),
                )
            )
        )


@dataclass(frozen=True)
class _PlacedGeneratedModelFileSnapshot:
    """One placed object whose GLB is loaded only by the AO worker."""

    object_id: str
    object_name: str
    asset_path: Path
    asset_revision: tuple[str, int, int, int]
    world_position: tuple[float, float, float]
    rotation_degrees: tuple[float, float, float]
    scale: float
    symmetric_preview_orientation: str | None = None
    symmetric_preview_plane_coordinate: float | None = None


@dataclass(frozen=True)
class _SurfaceAmbientOcclusionSceneSnapshot:
    """GUI-captured plain data needed to build the pre-Atlas AO scene."""

    levels: tuple[LevelData, ...]
    stairs: tuple[StairData, ...]
    surface_materials: tuple[tuple[str, object], ...]
    placed_models: tuple[_PlacedGeneratedModelFileSnapshot, ...]
    surface_source_ids: tuple[tuple[str, str], ...]

    @property
    def required_source_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *(placement.object_id for placement in self.placed_models),
                    *(source_id for _surface_id, source_id in self.surface_source_ids),
                )
            )
        )


@dataclass(frozen=True)
class _SurfaceAmbientOcclusionBakeSnapshot:
    """Inputs which must remain current before an asynchronous bake commits."""

    viewer_revision: int
    dependency_signature: tuple[object, ...]
    atlas_context_signature: tuple[tuple[object, ...], ...]
    required_source_ids: tuple[str, ...]
    surface_source_signature: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _AmbientOcclusionBakePreparation:
    """One GUI-thread snapshot shared by one or many Atlas AO jobs."""

    scene_snapshot: _SurfaceAmbientOcclusionSceneSnapshot
    atlas_context: tuple[SurfaceAmbientOcclusionAtlasContext, ...]
    snapshot: _SurfaceAmbientOcclusionBakeSnapshot


class _SurfaceAmbientOcclusionBakeThread(QThread):
    """Bake one immutable Atlas/scene snapshot outside the GUI thread."""

    progress = Signal(str)

    def __init__(
        self,
        scene_snapshot: _SurfaceAmbientOcclusionSceneSnapshot,
        atlas: SurfaceAmbientOcclusionAtlasContext,
        atlas_context: Sequence[SurfaceAmbientOcclusionAtlasContext],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._scene_snapshot = scene_snapshot
        self._atlas = atlas
        self._atlas_context = tuple(atlas_context)
        self.result: SurfaceAmbientOcclusionBakeResult | None = None
        self.error_message: str | None = None
        self.was_cancelled = False

    def run(self) -> None:  # type: ignore[override]
        try:
            self.progress.emit("Building ambient-occlusion scene (2%)")
            pre_atlas_scene = _build_surface_ao_pre_atlas_scene(
                self._scene_snapshot,
                self.isInterruptionRequested,
                allow_empty_base=True,
            )
            result = bake_surface_ambient_occlusion_for_atlas(
                pre_atlas_scene.model,
                self._atlas,
                atlas_context=self._atlas_context,
                surface_source_ids=pre_atlas_scene.surface_source_ids,
                cancellation_check=self.isInterruptionRequested,
                progress_callback=lambda message: self.progress.emit(str(message)),
            )
        except Exception as error:  # noqa: BLE001 - worker failures cross Qt safely.
            if self.isInterruptionRequested():
                self.was_cancelled = True
            else:
                self.error_message = str(error) or type(error).__name__
            return
        if self.isInterruptionRequested():
            self.was_cancelled = True
            return
        self.result = result


class _SurfaceAmbientOcclusionPreviewThread(QThread):
    """Prepare one immutable cached-AO preview outside the GUI thread."""

    progress = Signal(str)

    def __init__(
        self,
        scene_snapshot: _SurfaceAmbientOcclusionSceneSnapshot,
        atlas: SurfaceAmbientOcclusionAtlasContext,
        atlas_context: Sequence[SurfaceAmbientOcclusionAtlasContext],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._scene_snapshot = scene_snapshot
        self._atlas = atlas
        self._atlas_context = tuple(atlas_context)
        self.result: GeneratedModel | None = None
        self.error_message: str | None = None
        self.was_cancelled = False

    def run(self) -> None:  # type: ignore[override]
        try:
            self.progress.emit("Building ambient-occlusion preview scene")
            pre_atlas_scene = _build_surface_ao_pre_atlas_scene(
                self._scene_snapshot,
                self.isInterruptionRequested,
                allow_empty_base=True,
            )
            result = prepare_surface_ambient_occlusion_preview_for_atlas(
                pre_atlas_scene.model,
                self._atlas,
                atlas_context=self._atlas_context,
                surface_source_ids=pre_atlas_scene.surface_source_ids,
                cancellation_check=self.isInterruptionRequested,
                progress_callback=lambda message: self.progress.emit(str(message)),
            )
        except Exception as error:  # noqa: BLE001 - worker failures cross Qt safely.
            if self.isInterruptionRequested():
                self.was_cancelled = True
            else:
                self.error_message = str(error) or type(error).__name__
            return
        if self.isInterruptionRequested():
            self.was_cancelled = True
            return
        self.result = result


# ### Surface texture tiling preparation ###
class _SurfaceTextureTilingPreparationThread(QThread):
    """Prepare a texture-tiling comparison without blocking the Qt event loop."""

    def __init__(
        self,
        snapshot: SurfaceTextureTilingPreparationSnapshot,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._snapshot = snapshot
        self.result: PreparedSurfaceTextureTilingRepair | None = None
        self.error_message: str | None = None
        self.was_cancelled = False

    def run(self) -> None:  # type: ignore[override]
        try:
            result = prepare_surface_texture_tiling_repair(
                self._snapshot,
                cancellation_check=self.isInterruptionRequested,
            )
        except Exception as error:  # noqa: BLE001 - worker failures cross Qt safely.
            if self.isInterruptionRequested():
                self.was_cancelled = True
            else:
                self.error_message = str(error) or type(error).__name__
            return
        if self.isInterruptionRequested():
            self.was_cancelled = True
            return
        self.result = result


# ### Atlas draw-call estimation ###
class _AtlasDrawCallEstimateThread(QThread):
    """Estimate exact exported primitives from an immutable scene snapshot."""

    def __init__(
        self,
        scene_snapshot: _SurfaceAmbientOcclusionSceneSnapshot,
        atlases: Sequence[TextureAtlasRecord],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._scene_snapshot = scene_snapshot
        self._atlases = tuple(copy.deepcopy(tuple(atlases)))
        self.result: AtlasDrawCallEstimate | None = None
        self.error_message: str | None = None
        self.was_cancelled = False

    def run(self) -> None:  # type: ignore[override]
        try:
            pre_atlas_scene = _build_surface_ao_pre_atlas_scene(
                self._scene_snapshot,
                self.isInterruptionRequested,
                allow_empty_base=True,
            )
            result = estimate_texture_atlas_draw_calls(
                pre_atlas_scene.model,
                self._atlases,
                surface_source_ids=pre_atlas_scene.surface_source_ids,
            )
        except Exception as error:  # noqa: BLE001 - worker failures cross Qt safely.
            if self.isInterruptionRequested():
                self.was_cancelled = True
            else:
                self.error_message = str(error) or type(error).__name__
            return
        if self.isInterruptionRequested():
            self.was_cancelled = True
            return
        self.result = result


def _build_surface_ao_pre_atlas_scene(
    snapshot: _SurfaceAmbientOcclusionSceneSnapshot,
    cancellation_check: Callable[[], bool],
    *,
    allow_empty_base: bool = False,
) -> _PreAtlasExportScene:
    """Build AO geometry and load placed GLBs without touching Qt widgets."""

    if cancellation_check():
        raise RuntimeError("Ambient-occlusion bake cancelled.")
    surface_materials: dict[str, object] = {}
    for surface_id, source in snapshot.surface_materials:
        if isinstance(source, SurfaceMaterialSourceSpec):
            surface_materials[surface_id] = source
        elif isinstance(source, tuple):
            surface_materials[surface_id] = dict(source)
        else:
            surface_materials[surface_id] = source
    try:
        base_model = convert_to_export_scene_model(
            snapshot.levels,
            stairs=snapshot.stairs,
            surface_materials=surface_materials,
        )
    except ValueError as error:
        if not allow_empty_base or "does not contain usable edges" not in str(error):
            raise
        empty_mesh = trimesh.Trimesh(process=False)
        base_model = GeneratedModel(
            mesh=empty_mesh,
            scene=trimesh.Scene(),
            glb_bytes=b"",
        )
    placements: list[PlacedGeneratedModel] = []
    for placed in snapshot.placed_models:
        if cancellation_check():
            raise RuntimeError("Ambient-occlusion bake cancelled.")
        try:
            revision_before = _build_surface_ao_file_revision(placed.asset_path)
            if revision_before != placed.asset_revision:
                raise OSError("The placed GLB changed before it could be loaded.")
            payload = placed.asset_path.read_bytes()
            if _build_surface_ao_file_revision(placed.asset_path) != revision_before:
                raise OSError("The placed GLB changed while it was being loaded.")
            model = import_generated_glb(payload)
        except Exception as error:
            raise ValueError(
                f"Placed object {placed.object_name!r} is temporarily unavailable."
            ) from error
        placements.append(
            PlacedGeneratedModel(
                object_id=placed.object_id,
                object_name=placed.object_name,
                model=model,
                world_position=placed.world_position,
                symmetric_preview_orientation=(placed.symmetric_preview_orientation),
                symmetric_preview_plane_coordinate=(
                    placed.symmetric_preview_plane_coordinate
                ),
                rotation_degrees=placed.rotation_degrees,
                scale=placed.scale,
            )
        )
    if cancellation_check():
        raise RuntimeError("Ambient-occlusion bake cancelled.")
    generated_model = (
        base_model
        if not placements
        else compose_placed_generated_models_preview(base_model, placements)
    )
    return _PreAtlasExportScene(
        model=generated_model,
        placed_models=tuple(placements),
        surface_source_ids=dict(snapshot.surface_source_ids),
    )


def _build_surface_ao_file_revision(path: Path) -> tuple[str, int, int, int]:
    """Return the same stable local-file identity captured by Generation."""

    resolved_path = Path(path).resolve()
    file_stat = resolved_path.stat()
    return (
        str(resolved_path),
        int(file_stat.st_size),
        int(file_stat.st_mtime_ns),
        int(file_stat.st_ctime_ns),
    )


@dataclass
class _SurfaceAmbientOcclusionBakeRuntime:
    """GUI-owned lifecycle for one independently cancellable Atlas bake."""

    atlas_id: str
    job_id: str
    thread: _SurfaceAmbientOcclusionBakeThread
    snapshot: _SurfaceAmbientOcclusionBakeSnapshot
    cancel_requested: bool = False


@dataclass(frozen=True)
class _SurfaceAmbientOcclusionPreviewCacheEntry:
    """One AO-only model tied to exact scene, Atlas, and PNG revisions."""

    snapshot: _SurfaceAmbientOcclusionBakeSnapshot
    geometry_signature: str
    image_revision: tuple[str, int, int, int]
    model: GeneratedModel


@dataclass
class _SurfaceAmbientOcclusionPreviewRuntime:
    """GUI-owned lifecycle for one cancellable AO-only preview build."""

    request_id: int
    atlas_id: str
    thread: _SurfaceAmbientOcclusionPreviewThread
    snapshot: _SurfaceAmbientOcclusionBakeSnapshot
    image_revision: tuple[str, int, int, int]
    cancel_requested: bool = False


# ### Atlas draw-call estimator lifecycle ###
@dataclass(frozen=True)
class _AtlasDrawCallEstimateSnapshot:
    """Inputs that must remain current before an estimate is displayed."""

    scene_revision: int
    dependency_signature: tuple[object, ...]
    atlas_layout_signature: tuple[tuple[object, ...], ...]
    required_source_ids: tuple[str, ...]
    surface_source_signature: tuple[tuple[str, str], ...]


@dataclass
class _AtlasDrawCallEstimateRuntime:
    """GUI-owned lifecycle for one cancellable estimate worker."""

    request_id: int
    thread: _AtlasDrawCallEstimateThread
    snapshot: _AtlasDrawCallEstimateSnapshot
    cancel_requested: bool = False


# ### Canvas undo models ###
@dataclass(frozen=True)
class _CanvasBlueprintUndoState:
    """One pre-edit snapshot produced by the embedded 2D Canvas."""

    level_index: int
    snapshot: CanvasSnapshot
    assignments: tuple[SurfaceTextureAssignment, ...] = ()
    assignment_targets_after: tuple[SurfaceTextureAssignment, ...] | None = ()
    atlas_placements: tuple[tuple[str, TextureAtlasPlacement], ...] = ()
    wall_mirror_links: tuple[WallMirrorVertexLink, ...] = ()
    other_level_vertex_data: tuple[tuple[int, VertexData], ...] = ()
    other_level_doorways: tuple[tuple[int, tuple[DoorwayData, ...]], ...] = ()
    level_offsets_meters: tuple[float, float] | None = None


@dataclass(frozen=True)
class _CanvasPlanImageUndoState:
    """One immutable plan-image revision created by a marquee erase."""

    level_index: int
    commit: PlanImageEraseCommit


@dataclass(frozen=True)
class _CanvasTopologyUndoState:
    """Persistent topology, texture, and selection state before one 3D edit."""

    editable_surfaces_by_level: tuple[
        tuple[int, tuple[EditableSurfaceMeshData, ...]],
        ...,
    ]
    flipped_surface_ids_by_level: tuple[tuple[int, tuple[str, ...]], ...]
    assignments: tuple[SurfaceTextureAssignment, ...]
    assignment_targets_after: tuple[SurfaceTextureAssignment, ...]
    atlas_placements: tuple[tuple[str, TextureAtlasPlacement], ...]
    selected_surface_ids: tuple[str, ...]
    assignment_target_ids: tuple[str, ...]
    selected_object_id: str | None
    active_vertex_id: str | None
    selected_object_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _CanvasSurfaceEditUndoState:
    """Immutable structural gizmo baselines for one committed edit."""

    targets: tuple[CanvasSurfaceEditHandleTarget, ...]


@dataclass(frozen=True)
class _CanvasLevelPropertiesUndoState:
    """Transform and export values for one Canvas level before one edit."""

    level_index: int
    height_meters: float
    scale: float
    canvas_level_scale: float
    canvas_offset_x_pixels: float
    canvas_offset_y_pixels: float
    offset_x_meters: float
    offset_y_meters: float
    include_in_export: bool


@dataclass(frozen=True)
class _CanvasWallMirrorUndoState:
    """Cross-level wall topology before one mirror side-panel action."""

    vertex_data_by_level: tuple[tuple[int, VertexData], ...]
    doorways_by_level: tuple[tuple[int, tuple[DoorwayData, ...]], ...]
    wall_mirror_links: tuple[WallMirrorVertexLink, ...]
    selected_level_index: int
    selected_vertex_ids: tuple[int, ...]


@dataclass(frozen=True)
class _PendingLevelTransform:
    """One staged level transform waiting for the shared mesh-edit delay."""

    baseline: _CanvasLevelPropertiesUndoState
    scale: float
    offset_x_meters: float
    offset_y_meters: float

    @property
    def level_index(self) -> int:
        return self.baseline.level_index


@dataclass(frozen=True)
class _CanvasOpeningEditUndoState:
    """One doorway or window rectangle before a committed gizmo edit."""

    start_edit: CanvasOpeningEdit


@dataclass(frozen=True)
class _CanvasWindowAdditionUndoState:
    """Stable identity of one newly added Canvas wall window."""

    window_id: str


@dataclass(frozen=True)
class _CanvasPlacedObjectUndoState:
    """One object's prior Canvas state plus any Atlas allocations it owned."""

    object_id: str
    placement: GeneratedObjectPlacement | None
    atlas_placements: tuple[tuple[str, TextureAtlasPlacement], ...] = ()
    restore_atlas_bindings: bool = False
    selected_object_ids: tuple[str, ...] | None = None
    active_object_id: str | None = None


@dataclass(frozen=True)
class _CanvasPlacedObjectGroupUndoState:
    """One direct placement click that moved an ordered object batch."""

    members: tuple[_CanvasPlacedObjectUndoState, ...]


@dataclass(frozen=True)
class _DirectObjectPlacementSession:
    """Bind one shared-scene picker to existing requests or a generated batch."""

    request_id: str
    placeable_ids: tuple[str, ...] = ()
    generation_request_token: str | None = None
    accepts_next_generation_batch: bool = False


@dataclass(frozen=True)
class _StairEditorParameters:
    """Editable geometry settings shared by new and selected stairs."""

    stair_type: str = STAIR_TYPE_SUPPORTED
    target_rise_meters: float = DEFAULT_STAIR_TARGET_RISE_METERS
    tread_thickness_meters: float = DEFAULT_STAIR_TREAD_THICKNESS_METERS
    tread_overhang_meters: float = DEFAULT_STAIR_TREAD_OVERHANG_METERS
    nosing_placements: tuple[str, ...] = DEFAULT_STAIR_NOSING_PLACEMENTS
    tread_edge_profile: str = STAIR_TREAD_EDGE_STRAIGHT
    tread_edge_radius_meters: float = DEFAULT_STAIR_TREAD_EDGE_RADIUS_METERS
    starting_step: str = DEFAULT_STAIR_STARTING_STEP
    starting_step_edge_radius_meters: float = (
        DEFAULT_STAIR_STARTING_STEP_EDGE_RADIUS_METERS
    )
    starting_step_edge_points: int = DEFAULT_STAIR_STARTING_STEP_EDGE_POINTS
    stringer_placement: str = DEFAULT_STAIR_STRINGER_PLACEMENT


@dataclass(frozen=True)
class _CanvasStairsUndoState:
    """Stairs and conservatively restorable texture bindings before an edit."""

    stairs: tuple[StairData, ...]
    assignments: tuple[SurfaceTextureAssignment, ...] = ()
    assignment_targets_after: tuple[SurfaceTextureAssignment, ...] = ()
    atlas_placements: tuple[tuple[str, TextureAtlasPlacement], ...] = ()
    selected_stair_part_ids: tuple[str, ...] = ()
    assignment_target_ids: tuple[str, ...] = ()
    editing_stair_id: str | None = None


@dataclass(frozen=True)
class _SurfaceTextureTilingUndoState:
    """One accepted Surface tiling revision and its prior assets."""

    revision: SurfaceTextureTilingRevision


_CanvasUndoState = (
    _CanvasBlueprintUndoState
    | _CanvasPlanImageUndoState
    | _CanvasTopologyUndoState
    | _CanvasSurfaceEditUndoState
    | _CanvasLevelPropertiesUndoState
    | _CanvasWallMirrorUndoState
    | _CanvasOpeningEditUndoState
    | _CanvasWindowAdditionUndoState
    | _CanvasPlacedObjectUndoState
    | _CanvasPlacedObjectGroupUndoState
    | _CanvasStairsUndoState
    | _SurfaceTextureTilingUndoState
)


# ### Canvas undo helpers ###
def _surface_assignment_target_signature(
    assignment: SurfaceTextureAssignment | None,
) -> tuple[object, ...] | None:
    """Return only assignment fields changed by surface topology lineage."""

    if assignment is None:
        return None
    return (
        assignment.surface_type,
        assignment.surface_ids,
        float(assignment.combined_area_m2),
        assignment.area_description,
    )


# ### Event filters ###
class RightPanelValueInputWheelFilter(QObject):
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
        viewport_position = viewport.mapFromGlobal(event.globalPosition().toPoint())
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


class ViewportWidthRowFilter(QObject):
    """Keep composite form rows within, and expanded across, a viewport."""

    def __init__(
        self,
        viewport: QWidget,
        rows: tuple[QWidget, ...],
        *,
        horizontal_inset: int,
        on_width_changed: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(viewport)
        self._viewport = viewport
        self._rows = rows
        self._horizontal_inset = horizontal_inset
        self._on_width_changed = on_width_changed

    def sync_widths(self) -> None:
        """Use the complete visible form width without causing overflow."""

        maximum_width = max(0, self._viewport.width() - self._horizontal_inset)
        for row in self._rows:
            row.setMaximumWidth(maximum_width)
        if self._on_width_changed is not None:
            self._on_width_changed(self._viewport.width())

    def eventFilter(
        self,
        watched: QObject,
        event: QEvent,
    ) -> bool:  # type: ignore[override]
        if watched is self._viewport and event.type() == QEvent.Type.Resize:
            self.sync_widths()
        return super().eventFilter(watched, event)


# ### Responsive editor size observers ###
class StairEditorHeightFilter(QObject):
    """Refresh a horizontal-only scroll area's height after layout settles."""

    def __init__(
        self,
        editor: QWidget,
        layout_widgets: tuple[QWidget, ...],
        sync_height: Callable[[], None],
    ) -> None:
        super().__init__(editor)
        self._layout_widgets = layout_widgets
        self._sync_height = sync_height
        self._pending = False

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if watched in self._layout_widgets and event.type() in (
            QEvent.Type.LayoutRequest,
            QEvent.Type.Resize,
        ):
            if not self._pending:
                self._pending = True
                QTimer.singleShot(0, self._run_sync)
        return super().eventFilter(watched, event)

    def _run_sync(self) -> None:
        self._pending = False
        self._sync_height()


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
        self.wall_mirror_links: tuple[WallMirrorVertexLink, ...] = ()
        self.image_library_paths: list[str] = []
        self.doorway_presets: list[DoorwayPreset] = create_default_doorway_presets()
        self.stairs: list[StairData] = []
        self._is_syncing_stair_controls = False
        self._new_stair_parameters = _StairEditorParameters()
        self._editing_stair_index: int | None = None
        self._staged_stair: StairData | None = None
        self._pending_stair_parameters: _StairEditorParameters | None = None
        self._stair_preview_update_timer = QTimer(self)
        self._stair_preview_update_timer.setSingleShot(True)
        self._stair_preview_update_timer.setInterval(
            STAIR_PREVIEW_UPDATE_DELAY_MILLISECONDS
        )
        self._stair_preview_update_timer.timeout.connect(
            self._rebuild_staged_stair_preview
        )
        self._pending_stair_point_mesh_update = False
        self._pending_stair_point_undo_state: _CanvasStairsUndoState | None = None
        self._pending_stair_point_id: str | None = None
        self._stair_point_mesh_update_timer = QTimer(self)
        self._stair_point_mesh_update_timer.setSingleShot(True)
        self._desired_canvas_stair_part_ids: tuple[str, ...] = ()
        self._canvas_stair_part_targets_by_id: dict[str, PreviewStairPart] = {}
        self._canvas_stair_semantic_surfaces_by_id: dict[str, FixedSurface] = {}
        self.current_level_index = GROUND_LEVEL_INDEX
        self._is_syncing_level_controls = False
        self._level_transform_drag_active = False
        self._pending_level_transform: _PendingLevelTransform | None = None
        self._level_transform_outline_commit_revision: int | None = None
        self._canvas_transform_drag_active = False
        self._canvas_transform_drag_undo_state: (
            _CanvasLevelPropertiesUndoState | None
        ) = None
        self._is_viewer_refresh_scheduled = False
        self._scheduled_viewer_refresh_preserve_camera = True
        self._viewer_preview_revision = 0
        self._viewer_preview_model_revision = -1
        self._canvas_viewer_preview_revision = -1
        self._viewer_preview_model: GeneratedModel | None = None
        self._viewer_preview_dependency_signature: tuple[object, ...] | None = None
        self._viewer_preview_dependency_signature_revision = -1
        # Structural edits remain live in project data while these separate
        # snapshots control which edits have reached the expensive 3D mesh.
        self._viewer_doorways_by_level_index: dict[
            int,
            tuple[DoorwayData, ...],
        ] = {}
        self._viewer_windows_by_level_index: dict[
            int,
            tuple[WindowData, ...],
        ] = {}
        self._viewer_floor_thickness_by_level_index: dict[int, float] = {}
        self._reset_viewer_doorway_snapshots()
        self._mesh_edit_update_delay_seconds = DEFAULT_MESH_EDIT_UPDATE_DELAY_SECONDS
        self._wall_vertex_update_delay_seconds = (
            DEFAULT_WALL_VERTEX_UPDATE_DELAY_SECONDS
        )
        self._pending_wall_vertex_mesh_update = False
        self._pending_wall_vertex_doorway_level_indices: set[int] = set()
        self._is_canvas_wall_vertex_interaction_active = False
        self._is_doorway_move_drag_active = False
        self._is_doorway_resize_drag_active = False
        self._is_canvas_opening_drag_active = False
        self._active_canvas_opening_reference: CanvasOpeningReference | None = None
        self._active_canvas_opening_start_edit: CanvasOpeningEdit | None = None
        self._canvas_opening_targets_by_key: dict[
            str,
            CanvasOpeningTarget,
        ] = {}
        self._canvas_surface_targets_by_id: dict[str, FixedSurface] = {}
        self._canvas_surface_edit_targets_by_key: dict[
            tuple[str, str],
            CanvasSurfaceEditHandleTarget,
        ] = {}
        self._active_canvas_surface_edit_target: (
            CanvasSurfaceEditHandleTarget | None
        ) = None
        self._active_canvas_surface_edit_targets: tuple[
            CanvasSurfaceEditHandleTarget,
            ...,
        ] = ()
        self._pending_canvas_surface_mesh_update = False
        self._pending_canvas_surface_mesh_baseline: (
            CanvasSurfaceEditHandleTarget | None
        ) = None
        self._pending_canvas_surface_mesh_baselines: tuple[
            CanvasSurfaceEditHandleTarget,
            ...,
        ] = ()
        self._pending_canvas_wall_surface_ids: tuple[str, ...] = ()
        self._pending_floor_thickness_level_index: int | None = None
        self._pending_canvas_opening_key: str | None = None
        self._staged_canvas_opening_mesh_update = False
        self._staged_doorway_mesh_update = False
        self._pending_doorway_mesh_level_index: int | None = None
        self._pending_window_mesh_level_index: int | None = None
        self._doorway_outline_commit_revision: int | None = None
        self._doorway_mesh_update_timer = QTimer(self)
        self._doorway_mesh_update_timer.setSingleShot(True)
        self._doorway_mesh_update_timer.setInterval(
            round(self._mesh_edit_update_delay_seconds * 1000.0)
        )
        self._doorway_mesh_update_timer.timeout.connect(
            self._commit_pending_doorway_mesh_update
        )
        self._canvas_surface_mesh_update_timer = QTimer(self)
        self._canvas_surface_mesh_update_timer.setSingleShot(True)
        self._canvas_surface_mesh_update_timer.setInterval(
            round(self._mesh_edit_update_delay_seconds * 1000.0)
        )
        self._canvas_surface_mesh_update_timer.timeout.connect(
            self._commit_pending_canvas_surface_mesh_update
        )
        self._stair_point_mesh_update_timer.setInterval(
            round(self._mesh_edit_update_delay_seconds * 1000.0)
        )
        self._stair_point_mesh_update_timer.timeout.connect(
            self._commit_pending_stair_point_mesh_update
        )
        self._level_transform_mesh_update_timer = QTimer(self)
        self._level_transform_mesh_update_timer.setSingleShot(True)
        self._level_transform_mesh_update_timer.setInterval(
            round(self._mesh_edit_update_delay_seconds * 1000.0)
        )
        self._level_transform_mesh_update_timer.timeout.connect(
            self._commit_pending_level_transform_update
        )
        self._wall_vertex_update_timer = QTimer(self)
        self._wall_vertex_update_timer.setSingleShot(True)
        self._wall_vertex_update_timer.setInterval(
            round(self._wall_vertex_update_delay_seconds * 1000.0)
        )
        self._wall_vertex_update_timer.timeout.connect(
            self._commit_pending_wall_vertex_update
        )
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
        self._is_assigning_surface_texture_from_atlas = False
        self._atlas_surface_assignment_target_ids: tuple[str, ...] = ()
        self._selected_atlas_surface_source_id: str | None = None
        self._is_syncing_canvas_scene_selection = False
        self._desired_canvas_object_id: str | None = None
        self._desired_canvas_object_ids: tuple[str, ...] = ()
        self._desired_canvas_surface_ids: tuple[str, ...] = ()
        self._active_canvas_surface_drawing_vertex_id: str | None = None
        self._last_automatic_atlas_assignment_key: tuple[object, ...] | None = None
        self._atlas_preview_variant_key: tuple[object, ...] | None = None
        self._level_blueprint_image_revisions: dict[
            int,
            tuple[object, ...],
        ] = {}
        self._canvas_window_undo_ids: list[str] = []
        self._canvas_undo_stack: list[_CanvasUndoState] = []
        self._is_restoring_canvas_undo = False
        self._direct_object_placement_session: (
            _DirectObjectPlacementSession | None
        ) = None
        self._pending_generation_placement_anchor: (
            SceneObjectPlacementCandidate | None
        ) = None
        self._plan_image_correction_runtimes: dict[
            int,
            _PlanImageCorrectionRuntime,
        ] = {}
        self._plan_wall_detection_runtimes: dict[
            int,
            _PlanWallDetectionRuntime,
        ] = {}
        self._plan_wall_preview_session: _PlanWallPreviewSession | None = None
        self._plan_wall_preview_is_valid = False
        self._is_syncing_plan_wall_controls = False
        self._plan_wall_preview_refresh_timer = QTimer(self)
        self._plan_wall_preview_refresh_timer.setSingleShot(True)
        self._plan_wall_preview_refresh_timer.setInterval(
            PLAN_WALL_PREVIEW_REFRESH_DELAY_MILLISECONDS
        )
        self._plan_wall_preview_refresh_timer.timeout.connect(
            self._refresh_plan_wall_preview
        )
        self._surface_ao_bake_runtimes: dict[
            str,
            _SurfaceAmbientOcclusionBakeRuntime,
        ] = {}
        self._surface_texture_tiling_threads: dict[
            str,
            _SurfaceTextureTilingPreparationThread,
        ] = {}
        self._surface_ao_preview_runtimes: dict[
            int,
            _SurfaceAmbientOcclusionPreviewRuntime,
        ] = {}
        self._surface_ao_preview_cache: dict[
            str,
            _SurfaceAmbientOcclusionPreviewCacheEntry,
        ] = {}
        self._surface_ao_preview_request_id = 0
        self._surface_ao_preview_refresh_timer = QTimer(self)
        self._surface_ao_preview_refresh_timer.setSingleShot(True)
        self._surface_ao_preview_refresh_timer.setInterval(
            SURFACE_AO_PREVIEW_REFRESH_DELAY_MILLISECONDS
        )
        self._surface_ao_preview_refresh_timer.timeout.connect(
            self._refresh_surface_ambient_occlusion_preview
        )
        self._atlas_draw_call_estimate_runtimes: dict[
            int,
            _AtlasDrawCallEstimateRuntime,
        ] = {}
        self._atlas_draw_call_estimate_request_id = 0
        self._atlas_draw_call_scene_revision = 0
        self._atlas_draw_call_estimate_cache_signature: (
            _AtlasDrawCallEstimateSnapshot | None
        ) = None
        self._atlas_draw_call_estimate_cache_result: AtlasDrawCallEstimate | None = None
        self._atlas_draw_call_estimate_refresh_timer = QTimer(self)
        self._atlas_draw_call_estimate_refresh_timer.setSingleShot(True)
        self._atlas_draw_call_estimate_refresh_timer.setInterval(
            ATLAS_DRAW_CALL_ESTIMATE_REFRESH_DELAY_MILLISECONDS
        )
        self._atlas_draw_call_estimate_refresh_timer.timeout.connect(
            self._refresh_atlas_draw_call_estimate
        )
        self._atlas_preview_display_state: (
            tuple[bool, bool, bool, dict[str, bool]] | None
        ) = None
        self._canvas_ao_preview_display_state: (
            tuple[bool, bool, bool, dict[str, bool], float] | None
        ) = None
        self._surface_ao_canvas_preview_key: tuple[object, ...] | None = None
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
        self._is_viewer_refresh_scheduled = False
        self._surface_ao_preview_refresh_timer.stop()
        self._atlas_draw_call_estimate_refresh_timer.stop()
        self._cancel_pending_level_transform(
            sync_controls=False,
            restore_canvas_tools=False,
        )
        self._cancel_active_canvas_surface_edit()
        self._cancel_pending_canvas_surface_mesh_update()
        self._cancel_pending_wall_vertex_update()
        self._cancel_pending_doorway_mesh_update(clear_outline=True)
        self._cancel_and_join_plan_image_corrections()
        self._cancel_and_join_plan_wall_detections()
        self._cancel_and_join_atlas_draw_call_estimates()
        self._cancel_and_join_surface_ambient_occlusion_previews()
        self._cancel_and_join_surface_ambient_occlusion_bakes()
        self._cancel_and_join_surface_texture_tiling_preparations()
        try:
            self.settings_widget.settings_changed.disconnect(
                self._handle_generation_settings_changed
            )
        except (RuntimeError, TypeError):
            pass
        self.settings_widget.dispose()
        self._cancel_direct_object_placement()
        self._external_atlas_host.dispose()
        self._external_generation_host.dispose()
        self._external_scene_3d_host.dispose()
        self.surface_texture_generation.shutdown()
        self.generation.shutdown()
        self.jobs_window.dispose()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.shutdown()
        super().closeEvent(event)

    # ### Level transform control builders ###
    @staticmethod
    def _build_transform_slider_field(
        *,
        minimum: int,
        maximum: int,
        value: int,
        single_step: int,
        page_step: int,
        tick_interval: int,
        value_text: str,
        tooltip: str,
        tracking: bool,
    ) -> tuple[QWidget, QSlider, QLabel]:
        """Build one horizontal transform bar with a numeric readout."""

        field = QWidget()
        field_layout = QHBoxLayout(field)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(10)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setSingleStep(single_step)
        slider.setPageStep(page_step)
        slider.setTickInterval(tick_interval)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTracking(tracking)
        slider.setValue(value)
        slider.setProperty("transformBaseMinimum", minimum)
        slider.setProperty("transformBaseMaximum", maximum)
        slider.setMinimumHeight(40)
        slider.setToolTip(tooltip)
        field_layout.addWidget(slider, 1)

        value_label = QLabel(value_text)
        value_label.setMinimumWidth(68)
        value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        field_layout.addWidget(value_label)
        return field, slider, value_label

    @staticmethod
    def _build_plan_wall_slider_field(
        *,
        minimum: int,
        maximum: int,
        value: int,
        value_text: str,
        tooltip: str,
    ) -> tuple[QWidget, QSlider, QLabel]:
        """Build one live automatic-wall reconstruction control."""

        field = QWidget()
        layout = QHBoxLayout(field)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setSingleStep(1)
        slider.setPageStep(max(1, (maximum - minimum) // 10))
        slider.setTracking(True)
        slider.setValue(max(minimum, min(maximum, int(value))))
        slider.setMinimumHeight(30)
        slider.setToolTip(tooltip)
        layout.addWidget(slider, 1)

        value_label = QLabel(value_text)
        value_label.setMinimumWidth(54)
        value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        field.setToolTip(tooltip)
        value_label.setToolTip(tooltip)
        layout.addWidget(value_label)
        return field, slider, value_label

    @staticmethod
    def _fit_slider_range_to_value(slider: QSlider, value: int) -> None:
        """Use the practical base range plus any selected-level outlier."""

        base_minimum = int(slider.property("transformBaseMinimum"))
        base_maximum = int(slider.property("transformBaseMaximum"))
        slider.setRange(
            min(base_minimum, value),
            max(base_maximum, value),
        )

    def _add_transform_nudge_buttons(
        self,
        field: QWidget,
        slider: QSlider,
        control_name: str,
        pressed_handler: Callable[[QSlider, int], None],
        released_handler: Callable[[], None],
    ) -> tuple[QPushButton, QPushButton]:
        """Add one-shot controls with a press-and-hold level comparison."""

        layout = field.layout()
        if not isinstance(layout, QHBoxLayout):
            raise TypeError("Transform slider fields require a horizontal layout.")
        decrease_button = QPushButton("\N{MINUS SIGN}")
        increase_button = QPushButton("+")
        for button in (decrease_button, increase_button):
            button.setFixedWidth(32)
            button.setMinimumHeight(32)
            button.setAutoRepeat(False)
        decrease_button.setToolTip(
            f"Decrease {control_name} once. Hold to compare levels."
        )
        increase_button.setToolTip(
            f"Increase {control_name} once. Hold to compare levels."
        )
        decrease_button.pressed.connect(partial(pressed_handler, slider, -1))
        increase_button.pressed.connect(partial(pressed_handler, slider, 1))
        decrease_button.released.connect(released_handler)
        increase_button.released.connect(released_handler)
        layout.insertWidget(0, decrease_button)
        layout.insertWidget(2, increase_button)
        return decrease_button, increase_button

    def _nudge_canvas_offset(
        self,
        axis_name: str,
        delta_pixels: float,
    ) -> None:
        """Move one Canvas offset by an exact model-space pixel."""

        if self._is_syncing_level_controls:
            return
        is_active_gesture = self._canvas_transform_drag_active
        if not is_active_gesture:
            self._finish_level_transform_drag()
            self._commit_pending_level_transform_update()
        level = self.current_level
        normalized_axis = str(axis_name).strip().lower()
        if normalized_axis == "x":
            next_x = float(level.canvas_offset_x_pixels) + float(delta_pixels)
            next_y = float(level.canvas_offset_y_pixels)
            slider = self.canvas_x_offset_slider
            value_label = self.canvas_x_offset_value_label
            next_value = next_x
        elif normalized_axis == "y":
            next_x = float(level.canvas_offset_x_pixels)
            next_y = float(level.canvas_offset_y_pixels) + float(delta_pixels)
            slider = self.canvas_y_offset_slider
            value_label = self.canvas_y_offset_value_label
            next_value = next_y
        else:
            raise ValueError("A Canvas offset nudge requires the X or Y axis.")

        if not is_active_gesture:
            self._record_canvas_undo_state(
                self._capture_canvas_level_properties_undo_state(level)
            )
        level.canvas_offset_x_pixels = next_x
        level.canvas_offset_y_pixels = next_y
        slider_value = round(next_value * CANVAS_OFFSET_SLIDER_FACTOR)
        self._fit_slider_range_to_value(slider, slider_value)
        was_blocked = slider.blockSignals(True)
        slider.setValue(slider_value)
        slider.blockSignals(was_blocked)
        value_label.setText(self._format_canvas_offset_pixels(next_value))
        self.canvas.set_canvas_level_offsets(next_x, next_y)

    # ### Main workspace UI ###
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
        self.canvas.undo_snapshot_created.connect(
            self._handle_blueprint_undo_snapshot_created
        )
        self.canvas.undo_snapshot_discarded.connect(
            self._handle_blueprint_undo_snapshot_discarded
        )
        self.canvas.undo_requested.connect(self._handle_canvas_undo_requested)
        self.canvas.set_external_undo_history_enabled(True)
        self.viewer.undo_requested.connect(self._handle_canvas_undo_requested)
        self.canvas_3d_navigation_shortcut = QShortcut(self.viewer)
        self.canvas_3d_navigation_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.canvas_3d_navigation_shortcut.activated.connect(
            self._toggle_canvas_3d_navigation_mode
        )
        self.canvas_viewer_workspace = self._build_canvas_viewer_workspace()
        self.scene_3d_workspace = self._build_scene_3d_workspace()
        self._external_scene_3d_host = ExternalFullscreenViewerHost(
            self,
            window_title="HouseMaker 3D Scene",
            start_maximized=True,
        )
        self._external_scene_3d_host.close_requested.connect(
            self._handle_external_scene_3d_window_closed
        )
        self._external_scene_3d_host.viewer_restored.connect(
            self._handle_external_scene_3d_workspace_restored
        )
        self.job_manager = GenerationJobManager(self)
        self.jobs_window = JobsWindow(self.job_manager, self)
        self.generation = GenerationWorkspace(
            asset_directory=(self._application_settings.path.parent / "generated"),
            job_manager=self.job_manager,
        )
        self.generation.set_object_packing_change_handler(
            self._handle_generation_object_packing_change_requested
        )
        self.texture_atlas_workspace = TextureAtlasWorkspace(
            asset_directory=(self._application_settings.path.parent / "texture_atlases")
        )
        self.texture_atlas_undo_shortcut = QShortcut(
            QKeySequence.StandardKey.Undo,
            self.texture_atlas_workspace,
        )
        self.texture_atlas_undo_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.texture_atlas_undo_shortcut.activated.connect(
            self._handle_canvas_undo_requested
        )
        self._external_atlas_host = ExternalFullscreenViewerHost(
            self,
            window_title="HouseMaker Atlas",
            start_maximized=True,
        )
        self._external_atlas_host.close_requested.connect(
            self._handle_external_atlas_window_closed
        )
        self._external_atlas_host.viewer_restored.connect(
            self._handle_external_atlas_viewer_restored
        )
        self.atlas_object_preview_viewer = GlbViewerWidget(self.texture_atlas_workspace)
        self.atlas_object_preview_viewer.setObjectName(
            "texture_atlas_object_preview_viewer"
        )
        self.atlas_object_preview_viewer.set_ambient_light_intensity(1.0)
        self.texture_atlas_workspace.set_object_preview_widget(
            self.atlas_object_preview_viewer
        )
        self.atlas_object_preview_viewer.delete_requested.connect(
            self.texture_atlas_workspace.remove_selected_texture_from_atlas
        )
        self.texture_atlas_workspace.object_preview_requested.connect(
            self._handle_atlas_object_preview_requested
        )
        self.texture_atlas_workspace.placeable_object_preview_requested.connect(
            self._handle_atlas_placeable_object_preview_requested
        )
        self.texture_atlas_workspace.object_preview_clear_requested.connect(
            self._clear_atlas_object_preview
        )
        self.texture_atlas_workspace.object_texture_resolution_changed.connect(
            self._handle_atlas_object_texture_resolution_changed
        )
        self.texture_atlas_workspace.object_textures_selected.connect(
            self._handle_atlas_object_textures_selected
        )
        self.texture_atlas_workspace.surface_textures_selected.connect(
            self._handle_atlas_surface_textures_selected
        )
        self.texture_atlas_workspace.surface_texture_repeat_size_changed.connect(
            self._handle_atlas_surface_texture_repeat_size_changed
        )
        self.texture_atlas_workspace.object_place_requested.connect(
            self._handle_atlas_object_place_requested
        )
        self.texture_atlas_workspace.object_delete_requested.connect(
            self._handle_atlas_object_delete_requested
        )
        self.texture_atlas_workspace.surface_assign_requested.connect(
            self._handle_atlas_surface_assign_requested
        )
        self.texture_atlas_workspace.source_remove_requested.connect(
            self._handle_atlas_source_remove_requested
        )
        self.texture_atlas_workspace.surface_texture_delete_requested.connect(
            self._handle_atlas_surface_texture_delete_requested
        )
        self.texture_atlas_workspace.surface_texture_fix_tiling_requested.connect(
            self._handle_atlas_surface_texture_fix_tiling_requested
        )
        self.texture_atlas_workspace.ambient_occlusion_bake_requested.connect(
            self._handle_ambient_occlusion_bake_requested
        )
        self.texture_atlas_workspace.ambient_occlusion_bake_all_requested.connect(
            self._handle_all_ambient_occlusion_bakes_requested
        )
        self.texture_atlas_workspace.active_preview_map_changed.connect(
            self._handle_atlas_preview_map_changed
        )
        self.texture_atlas_workspace.face_orientation_mode_changed.connect(
            self.viewer.set_canvas_face_orientation_visible
        )
        self.texture_atlas_workspace.data_changed.connect(
            self._handle_texture_atlas_data_changed_for_ao_preview
        )
        self.surface_texture_generation = SurfaceTextureGenerationWorkspace(
            asset_directory=(
                self._application_settings.path.parent / "surface_textures"
            ),
            application_settings=self._application_settings,
            job_manager=self.job_manager,
            shared_controls=self.generation.get_shared_controls(),
        )
        self.merged_generation_workspace = MergedGenerationWorkspace(
            self.surface_texture_generation,
            self.generation,
            self.job_manager,
        )
        self.first_person_generation_frame_shortcut = QShortcut(
            QKeySequence("A", QKeySequence.SequenceFormat.PortableText),
            self.viewer,
        )
        self.first_person_generation_frame_shortcut.setObjectName(
            "first_person_generation_frame_shortcut"
        )
        self.first_person_generation_frame_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.first_person_generation_frame_shortcut.setAutoRepeat(False)
        self.first_person_generation_frame_shortcut.activated.connect(
            self._toggle_first_person_generation_frame_overlay
        )
        self.merged_generation_workspace.current_video_frame_changed.connect(
            self._sync_first_person_generation_frame_overlay
        )
        for checkbox in self.merged_generation_workspace.pbr_map_checkboxes.values():
            checkbox.toggled.connect(
                self._handle_generation_scene_pbr_maps_changed
            )
        self._external_generation_host = ExternalFullscreenViewerHost(
            self,
            window_title="HouseMaker Generation",
            start_maximized=True,
        )
        self._external_generation_host.close_requested.connect(
            self._handle_external_generation_window_closed
        )
        self._external_generation_host.viewer_restored.connect(
            self._handle_external_generation_workspace_restored
        )
        self.settings_widget = SettingsWidget(
            application_settings=self._application_settings
        )
        self.texture_atlas_workspace.selected_atlas_changed.connect(
            self._handle_selected_atlas_changed_for_automatic_assignment
        )
        generation_settings = self.settings_widget.get_settings()
        self._generation_settings = generation_settings
        self._set_mesh_edit_update_delay_seconds(
            generation_settings.mesh_edit_update_delay_seconds
        )
        self._set_wall_vertex_update_delay_seconds(
            generation_settings.wall_vertex_update_delay_seconds
        )
        self._set_canvas_3d_navigation_shortcut(
            generation_settings.canvas_3d_navigation_toggle_hotkey
        )
        self.viewer.set_first_person_movement_mode(
            generation_settings.first_person_navigation_mode
        )
        self.viewer.set_ignore_top_down_ceiling(
            generation_settings.ignore_top_down_ceiling
        )
        self.viewer.set_hide_stair_mesh_when_previewing(
            generation_settings.hide_stair_mesh_when_previewing
        )
        self.canvas.set_snap_middle_equal_angle_only(
            generation_settings.snap_middle_equal_angle_only
        )
        self.generation.set_runtime_settings(generation_settings)
        self.surface_texture_generation.set_runtime_settings(generation_settings)
        self.merged_generation_workspace.set_clear_mask_hotkey(
            generation_settings.clear_mask_hotkey
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
        self.generation.placeable_objects_changed.connect(
            self._handle_placeable_objects_changed_for_atlases
        )
        self.generation.generated_object_changed.connect(
            self._handle_generated_object_changed_for_atlases
        )
        self.generation.generated_object_deleted.connect(
            self._handle_generated_object_deleted_for_atlases
        )
        self.generation.generation_completed.connect(
            self._handle_generated_object_generated_for_atlases
        )
        self.generation.texture_regeneration_completed.connect(
            self._handle_generated_object_generated_for_atlases
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
        self.generation.generation_batch_started.connect(
            self._handle_generation_batch_started_for_placement
        )
        self.generation.new_object_placement_requested.connect(
            self._handle_generation_object_place_requested
        )
        self.viewer.object_placement_selected.connect(
            self._handle_direct_object_placement_selected
        )
        self.viewer.object_placement_cancelled.connect(
            self._handle_direct_object_placement_cancelled
        )
        self.generation.operation_finished.connect(
            self._handle_object_placement_operation_finished
        )
        self.generation.placement_request_finished.connect(
            self._handle_object_placement_operation_finished
        )
        self.surface_texture_generation.set_levels(self.levels)
        self.canvas_workspace_tab_index = self.workspace_tabs.addTab(
            self.canvas_viewer_workspace,
            "Canvas",
        )
        self.scene_3d_workspace_tab_index = self.workspace_tabs.addTab(
            self.scene_3d_workspace,
            "3D scene",
        )
        self.atlas_workspace_tab_index = self.workspace_tabs.addTab(
            self.texture_atlas_workspace,
            "Atlas",
        )
        self.generation_workspace_tab_index = self.workspace_tabs.addTab(
            self.merged_generation_workspace,
            "Generation",
        )
        self.settings_workspace_tab_index = self.workspace_tabs.addTab(
            self.settings_widget,
            "Settings",
        )
        self.workspace_tabs.currentChanged.connect(self._handle_workspace_tab_changed)
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
        self.viewer.placed_object_scales_changed.connect(
            self._handle_placed_object_scales_changed
        )
        self.viewer.placed_object_selection_changed.connect(
            self._handle_canvas_placed_object_selection_changed
        )
        self.viewer.placed_object_selection_set_changed.connect(
            self._handle_canvas_placed_object_selection_set_changed
        )
        self.viewer.canvas_surface_selection_changed.connect(
            self._handle_canvas_surface_selection_changed
        )
        self.viewer.canvas_stair_part_selection_changed.connect(
            self._handle_canvas_stair_part_selection_changed
        )
        self.viewer.canvas_stair_deletion_requested.connect(
            self._handle_canvas_stair_deletion_requested
        )
        self.viewer.canvas_stair_preview_cancelled.connect(
            self._handle_canvas_stair_preview_cancelled
        )
        self.viewer.view.escape_requested.connect(
            self._handle_canvas_stair_escape_requested
        )
        self.viewer.canvas_surface_orientation_flip_requested.connect(
            self._handle_canvas_surface_orientation_flip_requested
        )
        self.viewer.canvas_surface_vertex_insertion_requested.connect(
            self._handle_canvas_surface_vertex_insertion_requested
        )
        self.viewer.canvas_surface_vertex_chain_reset_requested.connect(
            self._handle_canvas_surface_vertex_chain_reset_requested
        )
        self.viewer.canvas_surface_face_extrusion_requested.connect(
            self._handle_canvas_surface_face_extrusion_requested
        )
        self.viewer.canvas_surface_face_deletion_requested.connect(
            self._handle_canvas_surface_face_deletion_requested
        )
        self.viewer.canvas_surface_edit_started.connect(
            self._handle_canvas_surface_edit_started
        )
        self.viewer.canvas_surface_edit_preview_changed.connect(
            self._handle_canvas_surface_edit_preview_changed
        )
        self.viewer.canvas_surface_edit_finished.connect(
            self._handle_canvas_surface_edit_finished
        )
        self.viewer.canvas_surface_edit_cancelled.connect(
            self._handle_canvas_surface_edit_cancelled
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
        self.canvas_side_panel_undo_shortcut = QShortcut(
            QKeySequence.StandardKey.Undo,
            self.side_panel,
        )
        self.canvas_side_panel_undo_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.canvas_side_panel_undo_shortcut.activated.connect(
            self._handle_canvas_undo_requested
        )
        side_layout = QVBoxLayout(self.side_panel)
        side_layout.setContentsMargins(16, 16, 16, 16)
        side_layout.setSpacing(12)

        self.side_tabs = QTabWidget()
        side_layout.addWidget(self.side_tabs, 1)

        generals_tab = QScrollArea()
        generals_tab.setWidgetResizable(True)
        generals_tab.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        generals_content = QWidget()
        generals_layout = QVBoxLayout(generals_content)
        generals_layout.setContentsMargins(10, 12, 10, 10)
        generals_layout.setSpacing(12)
        generals_tab.setWidget(generals_content)
        self.side_tabs.addTab(generals_tab, "Generals")
        side_layout = generals_layout

        self.level_dimensions_group = QGroupBox("Level dimensions")
        level_dimensions_layout = QFormLayout(self.level_dimensions_group)
        level_dimensions_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self.height_level_spinbox = QDoubleSpinBox()
        self.height_level_spinbox.setRange(0.1, 100.0)
        self.height_level_spinbox.setDecimals(2)
        self.height_level_spinbox.setSingleStep(0.1)
        self.height_level_spinbox.setValue(DEFAULT_WALL_HEIGHT_METERS)
        self.height_level_spinbox.setSuffix(" m")
        self.height_level_spinbox.setMinimumHeight(40)
        self.height_level_spinbox.valueChanged.connect(
            self._handle_height_level_changed
        )
        level_dimensions_layout.addRow(
            "Height level",
            self.height_level_spinbox,
        )

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
        level_dimensions_layout.addRow(
            "Floor thickness",
            self.floor_thickness_spinbox,
        )
        side_layout.addWidget(self.level_dimensions_group)

        self.open_spaces_group = QGroupBox("Open spaces")
        open_spaces_layout = QVBoxLayout(self.open_spaces_group)

        self.open_space_status_label = QLabel("Open spaces: none")
        self.open_space_status_label.setWordWrap(True)
        open_spaces_layout.addWidget(self.open_space_status_label)

        self.add_open_space_button = QPushButton("Add open space")
        self.add_open_space_button.setCheckable(True)
        self.add_open_space_button.setMinimumHeight(40)
        self.add_open_space_button.setToolTip(
            "Drag a rectangle on the 2D Canvas to remove this level's floor "
            "and the level below's ceiling."
        )
        self.add_open_space_button.clicked.connect(self._handle_add_open_space_clicked)
        open_spaces_layout.addWidget(self.add_open_space_button)
        side_layout.addWidget(self.open_spaces_group)

        self.level_transform_group = QGroupBox("Level transform")
        level_transform_layout = QFormLayout(self.level_transform_group)
        level_transform_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        (
            level_scale_field,
            self.level_scale_slider,
            self.level_scale_value_label,
        ) = self._build_transform_slider_field(
            minimum=round(MIN_LEVEL_SCALE * LEVEL_SCALE_SLIDER_FACTOR),
            maximum=round(MAX_LEVEL_SCALE * LEVEL_SCALE_SLIDER_FACTOR),
            value=round(DEFAULT_LEVEL_SCALE * LEVEL_SCALE_SLIDER_FACTOR),
            single_step=1,
            page_step=50,
            tick_interval=500,
            value_text="1.000 x",
            tooltip=(
                "Previews this level's 3D scale in yellow, then applies it "
                "after the Mesh edit update delay."
            ),
            tracking=False,
        )
        (
            self.level_scale_decrease_button,
            self.level_scale_increase_button,
        ) = self._add_transform_nudge_buttons(
            level_scale_field,
            self.level_scale_slider,
            "Level scale",
            self._handle_level_transform_button_pressed,
            self._handle_level_transform_button_released,
        )
        level_transform_layout.addRow("Level scale", level_scale_field)

        (
            level_x_offset_field,
            self.level_x_offset_slider,
            self.level_x_offset_value_label,
        ) = self._build_transform_slider_field(
            minimum=round(LEVEL_OFFSET_SLIDER_MIN_METERS * LEVEL_OFFSET_SLIDER_FACTOR),
            maximum=round(LEVEL_OFFSET_SLIDER_MAX_METERS * LEVEL_OFFSET_SLIDER_FACTOR),
            value=round(DEFAULT_LEVEL_OFFSET_METERS * LEVEL_OFFSET_SLIDER_FACTOR),
            single_step=1,
            page_step=10,
            tick_interval=1000,
            value_text="0.00 m",
            tooltip=(
                "Previews this level's X position in yellow, then applies it "
                "after the Mesh edit update delay."
            ),
            tracking=False,
        )
        (
            self.level_x_offset_decrease_button,
            self.level_x_offset_increase_button,
        ) = self._add_transform_nudge_buttons(
            level_x_offset_field,
            self.level_x_offset_slider,
            "X offset",
            self._handle_level_transform_button_pressed,
            self._handle_level_transform_button_released,
        )
        level_transform_layout.addRow("X offset", level_x_offset_field)

        (
            level_y_offset_field,
            self.level_y_offset_slider,
            self.level_y_offset_value_label,
        ) = self._build_transform_slider_field(
            minimum=round(LEVEL_OFFSET_SLIDER_MIN_METERS * LEVEL_OFFSET_SLIDER_FACTOR),
            maximum=round(LEVEL_OFFSET_SLIDER_MAX_METERS * LEVEL_OFFSET_SLIDER_FACTOR),
            value=round(DEFAULT_LEVEL_OFFSET_METERS * LEVEL_OFFSET_SLIDER_FACTOR),
            single_step=1,
            page_step=10,
            tick_interval=1000,
            value_text="0.00 m",
            tooltip=(
                "Previews this level's Y position in yellow, then applies it "
                "after the Mesh edit update delay."
            ),
            tracking=False,
        )
        (
            self.level_y_offset_decrease_button,
            self.level_y_offset_increase_button,
        ) = self._add_transform_nudge_buttons(
            level_y_offset_field,
            self.level_y_offset_slider,
            "Y offset",
            self._handle_level_transform_button_pressed,
            self._handle_level_transform_button_released,
        )
        level_transform_layout.addRow("Y offset", level_y_offset_field)
        side_layout.addWidget(self.level_transform_group)

        for slider in (
            self.level_scale_slider,
            self.level_x_offset_slider,
            self.level_y_offset_slider,
        ):
            slider.sliderPressed.connect(self._handle_level_transform_drag_started)
            slider.sliderReleased.connect(self._handle_level_transform_drag_finished)
        self.level_scale_slider.sliderMoved.connect(
            self._preview_level_scale_slider_value
        )
        self.level_scale_slider.valueChanged.connect(
            self._handle_level_scale_slider_changed
        )
        self.level_x_offset_slider.sliderMoved.connect(
            self._preview_level_x_offset_slider_value
        )
        self.level_x_offset_slider.valueChanged.connect(
            self._handle_level_x_offset_slider_changed
        )
        self.level_y_offset_slider.sliderMoved.connect(
            self._preview_level_y_offset_slider_value
        )
        self.level_y_offset_slider.valueChanged.connect(
            self._handle_level_y_offset_slider_changed
        )

        self.canvas_transform_group = QGroupBox("Canvas transform")
        canvas_transform_layout = QFormLayout(self.canvas_transform_group)
        canvas_transform_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        (
            canvas_level_scale_field,
            self.canvas_level_scale_slider,
            self.canvas_level_scale_value_label,
        ) = self._build_transform_slider_field(
            minimum=round(MIN_CANVAS_LEVEL_SCALE * CANVAS_LEVEL_SCALE_SLIDER_FACTOR),
            maximum=round(MAX_CANVAS_LEVEL_SCALE * CANVAS_LEVEL_SCALE_SLIDER_FACTOR),
            value=round(DEFAULT_CANVAS_LEVEL_SCALE * CANVAS_LEVEL_SCALE_SLIDER_FACTOR),
            single_step=1,
            page_step=10,
            tick_interval=50,
            value_text="1.00 x",
            tooltip=(
                "Scales only this level's 2D Canvas. Hold and drag to compare "
                "against the adjacent level."
            ),
            tracking=True,
        )
        (
            self.canvas_level_scale_decrease_button,
            self.canvas_level_scale_increase_button,
        ) = self._add_transform_nudge_buttons(
            canvas_level_scale_field,
            self.canvas_level_scale_slider,
            "Canvas level scale",
            self._handle_canvas_transform_button_pressed,
            self._handle_canvas_transform_button_released,
        )
        canvas_transform_layout.addRow(
            "Canvas level scale",
            canvas_level_scale_field,
        )

        (
            canvas_x_offset_field,
            self.canvas_x_offset_slider,
            self.canvas_x_offset_value_label,
        ) = self._build_transform_slider_field(
            minimum=round(
                CANVAS_OFFSET_SLIDER_MIN_PIXELS * CANVAS_OFFSET_SLIDER_FACTOR
            ),
            maximum=round(
                CANVAS_OFFSET_SLIDER_MAX_PIXELS * CANVAS_OFFSET_SLIDER_FACTOR
            ),
            value=round(DEFAULT_CANVAS_OFFSET_PIXELS * CANVAS_OFFSET_SLIDER_FACTOR),
            single_step=CANVAS_OFFSET_SLIDER_FACTOR,
            page_step=10 * CANVAS_OFFSET_SLIDER_FACTOR,
            tick_interval=100 * CANVAS_OFFSET_SLIDER_FACTOR,
            value_text="0 px",
            tooltip=(
                "Moves only this level's 2D Canvas along the horizontal axis. "
                "Use the side buttons for exact one-pixel steps."
            ),
            tracking=True,
        )
        (
            self.canvas_x_offset_decrease_button,
            self.canvas_x_offset_increase_button,
        ) = self._add_transform_nudge_buttons(
            canvas_x_offset_field,
            self.canvas_x_offset_slider,
            "Canvas X offset",
            partial(self._handle_canvas_offset_button_pressed, "X"),
            self._handle_canvas_transform_button_released,
        )
        canvas_transform_layout.addRow(
            "Canvas X offset",
            canvas_x_offset_field,
        )

        (
            canvas_y_offset_field,
            self.canvas_y_offset_slider,
            self.canvas_y_offset_value_label,
        ) = self._build_transform_slider_field(
            minimum=round(
                CANVAS_OFFSET_SLIDER_MIN_PIXELS * CANVAS_OFFSET_SLIDER_FACTOR
            ),
            maximum=round(
                CANVAS_OFFSET_SLIDER_MAX_PIXELS * CANVAS_OFFSET_SLIDER_FACTOR
            ),
            value=round(DEFAULT_CANVAS_OFFSET_PIXELS * CANVAS_OFFSET_SLIDER_FACTOR),
            single_step=CANVAS_OFFSET_SLIDER_FACTOR,
            page_step=10 * CANVAS_OFFSET_SLIDER_FACTOR,
            tick_interval=100 * CANVAS_OFFSET_SLIDER_FACTOR,
            value_text="0 px",
            tooltip=(
                "Moves only this level's 2D Canvas along the vertical axis. "
                "Use the side buttons for exact one-pixel steps."
            ),
            tracking=True,
        )
        (
            self.canvas_y_offset_decrease_button,
            self.canvas_y_offset_increase_button,
        ) = self._add_transform_nudge_buttons(
            canvas_y_offset_field,
            self.canvas_y_offset_slider,
            "Canvas Y offset",
            partial(self._handle_canvas_offset_button_pressed, "Y"),
            self._handle_canvas_transform_button_released,
        )
        canvas_transform_layout.addRow(
            "Canvas Y offset",
            canvas_y_offset_field,
        )
        side_layout.addWidget(self.canvas_transform_group)

        for slider in (
            self.canvas_level_scale_slider,
            self.canvas_x_offset_slider,
            self.canvas_y_offset_slider,
        ):
            slider.sliderPressed.connect(self._handle_canvas_transform_drag_started)
            slider.sliderReleased.connect(self._handle_canvas_transform_drag_finished)
        self.canvas_level_scale_slider.valueChanged.connect(
            self._handle_canvas_level_scale_changed
        )
        self.canvas_x_offset_slider.valueChanged.connect(
            self._handle_canvas_x_offset_changed
        )
        self.canvas_y_offset_slider.valueChanged.connect(
            self._handle_canvas_y_offset_changed
        )

        self.stairs_group = QGroupBox("Stairs")
        stairs_layout = QVBoxLayout(self.stairs_group)
        stairs_layout.setContentsMargins(5, 18, 5, 6)

        stair_parameters_layout = QFormLayout()
        stair_parameters_layout.setContentsMargins(0, 0, 0, 0)
        stair_parameters_layout.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.DontWrapRows
        )
        stair_parameters_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self.stair_type_combo = QComboBox()
        self.stair_type_combo.setObjectName("stair_type_combo")
        self.stair_type_combo.addItem("Supported", STAIR_TYPE_SUPPORTED)
        self.stair_type_combo.addItem("Floating", STAIR_TYPE_FLOATING)
        self.stair_type_combo.setMinimumHeight(34)
        self.stair_type_combo.currentIndexChanged.connect(
            self._handle_stair_parameter_changed
        )
        stair_parameters_layout.addRow("Type", self.stair_type_combo)

        self.stair_step_rise_target_spinbox = QDoubleSpinBox()
        self.stair_step_rise_target_spinbox.setObjectName(
            "stair_step_rise_target_spinbox"
        )
        self.stair_step_rise_target_spinbox.setRange(5.0, 50.0)
        self.stair_step_rise_target_spinbox.setDecimals(1)
        self.stair_step_rise_target_spinbox.setSingleStep(0.5)
        self.stair_step_rise_target_spinbox.setSuffix(" cm")
        self.stair_step_rise_target_spinbox.setValue(
            DEFAULT_STAIR_TARGET_RISE_METERS * 100.0
        )
        stair_step_rise_tooltip = (
            "Desired vertical height per step. The step count is rounded to a "
            "whole number and the actual rise is adjusted so the stair reaches "
            "the next level exactly."
        )
        self.stair_step_rise_target_spinbox.setToolTip(stair_step_rise_tooltip)
        self.stair_step_rise_target_spinbox.valueChanged.connect(
            self._handle_stair_parameter_changed
        )
        self.stair_step_rise_target_label = QLabel("Step rise target")
        self.stair_step_rise_target_label.setToolTip(stair_step_rise_tooltip)
        stair_parameters_layout.addRow(
            self.stair_step_rise_target_label,
            self.stair_step_rise_target_spinbox,
        )

        self.stair_tread_thickness_spinbox = QDoubleSpinBox()
        self.stair_tread_thickness_spinbox.setObjectName(
            "stair_tread_thickness_spinbox"
        )
        self.stair_tread_thickness_spinbox.setRange(0.5, 100.0)
        self.stair_tread_thickness_spinbox.setDecimals(1)
        self.stair_tread_thickness_spinbox.setSingleStep(0.5)
        self.stair_tread_thickness_spinbox.setSuffix(" cm")
        self.stair_tread_thickness_spinbox.setValue(
            DEFAULT_STAIR_TREAD_THICKNESS_METERS * 100.0
        )
        self.stair_tread_thickness_spinbox.valueChanged.connect(
            self._handle_stair_parameter_changed
        )
        stair_parameters_layout.addRow(
            "Tread thickness",
            self.stair_tread_thickness_spinbox,
        )

        self.stair_nosing_overhang_spinbox = QDoubleSpinBox()
        self.stair_nosing_overhang_spinbox.setObjectName(
            "stair_nosing_overhang_spinbox"
        )
        self.stair_nosing_overhang_spinbox.setRange(0.0, 50.0)
        self.stair_nosing_overhang_spinbox.setDecimals(1)
        self.stair_nosing_overhang_spinbox.setSingleStep(0.5)
        self.stair_nosing_overhang_spinbox.setSuffix(" cm")
        self.stair_nosing_overhang_spinbox.setValue(
            DEFAULT_STAIR_TREAD_OVERHANG_METERS * 100.0
        )
        self.stair_nosing_overhang_spinbox.valueChanged.connect(
            self._handle_stair_parameter_changed
        )

        self.stair_nosing_row_widget = QWidget()
        self.stair_nosing_row_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        nosing_row_layout = QGridLayout(self.stair_nosing_row_widget)
        nosing_row_layout.setContentsMargins(0, 0, 0, 0)
        nosing_row_layout.setSpacing(4)
        self.stair_nosing_overhang_label = QLabel("Nosing overhang")
        self.stair_nosing_overhang_label.setWordWrap(False)
        nosing_row_layout.addWidget(self.stair_nosing_overhang_label, 0, 0)
        self.stair_nosing_overhang_spinbox.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        nosing_row_layout.addWidget(self.stair_nosing_overhang_spinbox, 0, 1)
        self.stair_nosing_left_checkbox = QCheckBox("Left")
        self.stair_nosing_left_checkbox.setObjectName(
            "stair_nosing_left_checkbox"
        )
        self.stair_nosing_right_checkbox = QCheckBox("Right")
        self.stair_nosing_right_checkbox.setObjectName(
            "stair_nosing_right_checkbox"
        )
        self.stair_nosing_front_checkbox = QCheckBox("Front")
        self.stair_nosing_front_checkbox.setObjectName(
            "stair_nosing_front_checkbox"
        )
        self.stair_nosing_front_checkbox.setChecked(True)
        nosing_placement_widget = QWidget()
        nosing_placement_layout = QHBoxLayout(nosing_placement_widget)
        nosing_placement_layout.setContentsMargins(0, 0, 0, 0)
        nosing_placement_layout.setSpacing(1)
        for checkbox in (
            self.stair_nosing_left_checkbox,
            self.stair_nosing_right_checkbox,
            self.stair_nosing_front_checkbox,
        ):
            checkbox.toggled.connect(self._handle_stair_parameter_changed)
            checkbox.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            nosing_placement_layout.addWidget(checkbox, 1)
        nosing_placement_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.stair_nosing_placement_label = QLabel("Nosing placement")
        self.stair_nosing_placement_label.setWordWrap(False)
        nosing_row_layout.addWidget(self.stair_nosing_placement_label, 1, 0)
        nosing_row_layout.addWidget(nosing_placement_widget, 1, 1)
        nosing_row_layout.setColumnStretch(1, 1)
        stair_parameters_layout.addRow(self.stair_nosing_row_widget)

        self.stair_tread_edge_combo = QComboBox()
        self.stair_tread_edge_combo.setObjectName("stair_tread_edge_combo")
        self.stair_tread_edge_combo.addItem(
            "Straight",
            STAIR_TREAD_EDGE_STRAIGHT,
        )
        self.stair_tread_edge_combo.addItem(
            "Rounded",
            STAIR_TREAD_EDGE_ROUNDED,
        )
        self.stair_tread_edge_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.stair_tread_edge_combo.currentIndexChanged.connect(
            self._sync_stair_tread_edge_radius_enabled
        )
        self.stair_tread_edge_combo.currentIndexChanged.connect(
            self._handle_stair_parameter_changed
        )

        self.stair_tread_edge_radius_spinbox = QDoubleSpinBox()
        self.stair_tread_edge_radius_spinbox.setObjectName(
            "stair_tread_edge_radius_spinbox"
        )
        self.stair_tread_edge_radius_spinbox.setRange(
            MIN_STAIR_TREAD_EDGE_RADIUS_METERS * 100.0,
            MAX_STAIR_TREAD_EDGE_RADIUS_METERS * 100.0,
        )
        self.stair_tread_edge_radius_spinbox.setDecimals(1)
        self.stair_tread_edge_radius_spinbox.setSingleStep(0.5)
        self.stair_tread_edge_radius_spinbox.setSuffix(" cm")
        self.stair_tread_edge_radius_spinbox.setValue(
            DEFAULT_STAIR_TREAD_EDGE_RADIUS_METERS * 100.0
        )
        stair_tread_edge_radius_tooltip = (
            "Rounds each tread's exposed front and checked left/right edges. "
            "The effective radius is clamped when the tread does not have "
            "enough depth."
        )
        self.stair_tread_edge_radius_spinbox.setToolTip(
            stair_tread_edge_radius_tooltip
        )
        self.stair_tread_edge_radius_spinbox.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.stair_tread_edge_radius_spinbox.valueChanged.connect(
            self._handle_stair_parameter_changed
        )

        self.stair_tread_edge_row_widget = QWidget()
        self.stair_tread_edge_row_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        tread_edge_row_layout = QHBoxLayout(self.stair_tread_edge_row_widget)
        tread_edge_row_layout.setContentsMargins(0, 0, 0, 0)
        tread_edge_row_layout.setSpacing(6)
        self.stair_tread_edge_label = QLabel("Tread edge")
        self.stair_tread_edge_label.setWordWrap(False)
        self.stair_tread_edge_radius_label = QLabel("Edge radius")
        self.stair_tread_edge_radius_label.setWordWrap(False)
        self.stair_tread_edge_radius_label.setToolTip(
            stair_tread_edge_radius_tooltip
        )
        self.stair_tread_edge_field_widget = QWidget()
        self.stair_tread_edge_radius_field_widget = QWidget()
        tread_edge_fields = (
            (
                self.stair_tread_edge_field_widget,
                self.stair_tread_edge_label,
                self.stair_tread_edge_combo,
            ),
            (
                self.stair_tread_edge_radius_field_widget,
                self.stair_tread_edge_radius_label,
                self.stair_tread_edge_radius_spinbox,
            ),
        )
        for field_widget, field_label, field_control in tread_edge_fields:
            field_widget.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Preferred,
            )
            field_label.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Preferred,
            )
            field_layout = QVBoxLayout(field_widget)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.setSpacing(2)
            field_layout.addWidget(field_label)
            field_layout.addWidget(field_control)
            tread_edge_row_layout.addWidget(field_widget, 1)
        stair_parameters_layout.addRow(self.stair_tread_edge_row_widget)
        self._sync_stair_tread_edge_radius_enabled()

        self.stair_starting_step_combo = QComboBox()
        self.stair_starting_step_combo.setObjectName(
            "stair_starting_step_combo"
        )
        self.stair_starting_step_combo.addItem(
            "None",
            STAIR_STARTING_STEP_NONE,
        )
        self.stair_starting_step_combo.addItem(
            "Bullnose",
            STAIR_STARTING_STEP_BULLNOSE,
        )
        self.stair_starting_step_combo.addItem(
            "Curtail",
            STAIR_STARTING_STEP_CURTAIL,
        )
        self.stair_starting_step_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        self.stair_starting_step_combo.currentIndexChanged.connect(
            self._sync_stair_starting_step_edge_radius_enabled
        )
        self.stair_starting_step_combo.currentIndexChanged.connect(
            self._handle_stair_parameter_changed
        )

        self.stair_starting_step_edge_radius_spinbox = QDoubleSpinBox()
        self.stair_starting_step_edge_radius_spinbox.setObjectName(
            "stair_starting_step_edge_radius_spinbox"
        )
        self.stair_starting_step_edge_radius_spinbox.setRange(
            MIN_STAIR_STARTING_STEP_EDGE_RADIUS_METERS * 100.0,
            MAX_STAIR_STARTING_STEP_EDGE_RADIUS_METERS * 100.0,
        )
        self.stair_starting_step_edge_radius_spinbox.setDecimals(1)
        self.stair_starting_step_edge_radius_spinbox.setSingleStep(1.0)
        self.stair_starting_step_edge_radius_spinbox.setSuffix(" cm")
        self.stair_starting_step_edge_radius_spinbox.setValue(
            DEFAULT_STAIR_STARTING_STEP_EDGE_RADIUS_METERS * 100.0
        )
        stair_starting_step_radius_tooltip = (
            "Controls the Bullnose or Curtail curve in plan view. The effective "
            "radius is clamped to the available tread width and depth."
        )
        self.stair_starting_step_edge_radius_spinbox.setToolTip(
            stair_starting_step_radius_tooltip
        )
        self.stair_starting_step_edge_radius_spinbox.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        self.stair_starting_step_edge_radius_spinbox.valueChanged.connect(
            self._handle_stair_parameter_changed
        )

        self.stair_starting_step_edge_points_spinbox = QSpinBox()
        self.stair_starting_step_edge_points_spinbox.setObjectName(
            "stair_starting_step_edge_points_spinbox"
        )
        self.stair_starting_step_edge_points_spinbox.setRange(
            MIN_STAIR_STARTING_STEP_EDGE_POINTS,
            MAX_STAIR_STARTING_STEP_EDGE_POINTS,
        )
        self.stair_starting_step_edge_points_spinbox.setSingleStep(1)
        self.stair_starting_step_edge_points_spinbox.setValue(
            DEFAULT_STAIR_STARTING_STEP_EDGE_POINTS
        )
        stair_starting_step_points_tooltip = (
            "Controls the starting-step arch detail. One uses one midpoint; "
            "each higher value adds one matching point on each side for a "
            "rounder result."
        )
        self.stair_starting_step_edge_points_spinbox.setToolTip(
            stair_starting_step_points_tooltip
        )
        self.stair_starting_step_edge_points_spinbox.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        self.stair_starting_step_edge_points_spinbox.valueChanged.connect(
            self._handle_stair_parameter_changed
        )

        self.stair_starting_step_row_widget = QWidget()
        self.stair_starting_step_row_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        starting_step_row_layout = QVBoxLayout(
            self.stair_starting_step_row_widget
        )
        starting_step_row_layout.setContentsMargins(0, 0, 0, 0)
        starting_step_row_layout.setSpacing(6)
        self.stair_starting_step_label = QLabel("Starting step")
        self.stair_starting_step_label.setWordWrap(False)
        self.stair_starting_step_edge_radius_label = QLabel("Edge radius")
        self.stair_starting_step_edge_radius_label.setWordWrap(False)
        self.stair_starting_step_edge_radius_label.setToolTip(
            stair_starting_step_radius_tooltip
        )
        self.stair_starting_step_edge_points_label = QLabel("Points")
        self.stair_starting_step_edge_points_label.setToolTip(
            stair_starting_step_points_tooltip
        )
        self.stair_starting_step_field_widget = QWidget()
        self.stair_starting_step_edge_radius_field_widget = QWidget()
        self.stair_starting_step_edge_points_field_widget = QWidget()
        self.stair_starting_step_field_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        starting_step_field_layout = QHBoxLayout(
            self.stair_starting_step_field_widget
        )
        starting_step_field_layout.setContentsMargins(0, 0, 0, 0)
        starting_step_field_layout.setSpacing(6)
        starting_step_field_layout.addWidget(self.stair_starting_step_label)
        starting_step_field_layout.addWidget(self.stair_starting_step_combo, 1)
        starting_step_row_layout.addWidget(self.stair_starting_step_field_widget)

        self.stair_starting_step_detail_row_widget = QWidget()
        detail_row_layout = QHBoxLayout(
            self.stair_starting_step_detail_row_widget
        )
        detail_row_layout.setContentsMargins(0, 0, 0, 0)
        detail_row_layout.setSpacing(6)
        starting_step_detail_fields = (
            (
                self.stair_starting_step_edge_radius_field_widget,
                self.stair_starting_step_edge_radius_label,
                self.stair_starting_step_edge_radius_spinbox,
            ),
            (
                self.stair_starting_step_edge_points_field_widget,
                self.stair_starting_step_edge_points_label,
                self.stair_starting_step_edge_points_spinbox,
            ),
        )
        for field_widget, field_label, field_control in starting_step_detail_fields:
            field_widget.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Preferred,
            )
            field_label.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Preferred,
            )
            field_control.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            field_layout = QVBoxLayout(field_widget)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.setSpacing(2)
            field_layout.addWidget(field_label)
            field_layout.addWidget(field_control)
            detail_row_layout.addWidget(field_widget, 1)
        starting_step_row_layout.addWidget(
            self.stair_starting_step_detail_row_widget
        )
        stair_parameters_layout.addRow(self.stair_starting_step_row_widget)
        self._sync_stair_starting_step_edge_radius_enabled()

        self.stair_stringer_placement_combo = QComboBox()
        self.stair_stringer_placement_combo.setObjectName(
            "stair_stringer_placement_combo"
        )
        self.stair_stringer_placement_combo.addItem("None", STAIR_STRINGER_NONE)
        self.stair_stringer_placement_combo.addItem("Left", STAIR_STRINGER_LEFT)
        self.stair_stringer_placement_combo.addItem("Right", STAIR_STRINGER_RIGHT)
        self.stair_stringer_placement_combo.addItem("Both", STAIR_STRINGER_BOTH)
        self.stair_stringer_placement_combo.setCurrentIndex(
            self.stair_stringer_placement_combo.findData(
                DEFAULT_STAIR_STRINGER_PLACEMENT
            )
        )
        self.stair_stringer_placement_combo.currentIndexChanged.connect(
            self._handle_stair_parameter_changed
        )
        stair_parameters_layout.addRow(
            "Stringer placement",
            self.stair_stringer_placement_combo,
        )

        self.stair_calculated_step_count_label = QLabel("—")
        self.stair_calculated_step_count_label.setObjectName(
            "stair_calculated_step_count_label"
        )
        stair_parameters_layout.addRow(
            "Calculated step count",
            self.stair_calculated_step_count_label,
        )

        self.stair_actual_rise_label = QLabel("—")
        self.stair_actual_rise_label.setObjectName("stair_actual_rise_label")
        stair_parameters_layout.addRow(
            "Actual rise",
            self.stair_actual_rise_label,
        )
        stairs_layout.addLayout(stair_parameters_layout)

        # ### Responsive Stairs row layout ###
        nosing_inline_state = (True, True)
        placement_checkbox_width = max(
            checkbox.minimumSizeHint().width()
            for checkbox in (
                self.stair_nosing_left_checkbox,
                self.stair_nosing_right_checkbox,
                self.stair_nosing_front_checkbox,
            )
        )
        for checkbox in (
            self.stair_nosing_left_checkbox,
            self.stair_nosing_right_checkbox,
            self.stair_nosing_front_checkbox,
        ):
            checkbox.setMinimumWidth(placement_checkbox_width)

        def sync_stair_rows_to_viewport(viewport_width: int) -> None:
            nonlocal nosing_inline_state

            # The group border and its layout margins consume 28 px.
            usable_form_width = max(0, viewport_width - 28)
            overhang_inline = usable_form_width >= (
                self.stair_nosing_overhang_label.sizeHint().width()
                + nosing_row_layout.horizontalSpacing()
                + self.stair_nosing_overhang_spinbox.minimumSizeHint().width()
            )
            placement_inline = usable_form_width >= (
                self.stair_nosing_placement_label.sizeHint().width()
                + nosing_row_layout.horizontalSpacing()
                + 3 * placement_checkbox_width
                + 2 * nosing_placement_layout.spacing()
            )
            next_inline_state = (overhang_inline, placement_inline)
            layout_changed = next_inline_state != nosing_inline_state
            if layout_changed:
                for widget in (
                    self.stair_nosing_overhang_label,
                    self.stair_nosing_overhang_spinbox,
                    self.stair_nosing_placement_label,
                    nosing_placement_widget,
                ):
                    nosing_row_layout.removeWidget(widget)
                if overhang_inline:
                    nosing_row_layout.addWidget(self.stair_nosing_overhang_label, 0, 0)
                    nosing_row_layout.addWidget(
                        self.stair_nosing_overhang_spinbox, 0, 1
                    )
                    placement_row = 1
                else:
                    nosing_row_layout.addWidget(
                        self.stair_nosing_overhang_label, 0, 0, 1, 2
                    )
                    nosing_row_layout.addWidget(
                        self.stair_nosing_overhang_spinbox, 1, 0, 1, 2
                    )
                    placement_row = 2
                if placement_inline:
                    nosing_row_layout.addWidget(
                        self.stair_nosing_placement_label, placement_row, 0
                    )
                    nosing_row_layout.addWidget(
                        nosing_placement_widget, placement_row, 1
                    )
                else:
                    nosing_row_layout.addWidget(
                        self.stair_nosing_placement_label,
                        placement_row,
                        0,
                        1,
                        2,
                    )
                    nosing_row_layout.addWidget(
                        nosing_placement_widget,
                        placement_row + 1,
                        0,
                        1,
                        2,
                    )
                nosing_inline_state = next_inline_state

            single_field_controls = (
                self.stair_type_combo,
                self.stair_step_rise_target_spinbox,
                self.stair_tread_thickness_spinbox,
                self.stair_stringer_placement_combo,
                self.stair_calculated_step_count_label,
                self.stair_actual_rise_label,
            )
            widest_label = max(
                stair_parameters_layout.labelForField(control).sizeHint().width()
                for control in single_field_controls
            )
            widest_control = max(
                control.minimumSizeHint().width()
                for control in single_field_controls
            )
            row_wrap_policy = (
                QFormLayout.RowWrapPolicy.DontWrapRows
                if usable_form_width
                >= widest_label
                + stair_parameters_layout.horizontalSpacing()
                + widest_control
                else QFormLayout.RowWrapPolicy.WrapLongRows
            )
            if stair_parameters_layout.rowWrapPolicy() != row_wrap_policy:
                stair_parameters_layout.setRowWrapPolicy(row_wrap_policy)
                layout_changed = True

            if layout_changed:
                stair_parameters_layout.invalidate()
                stairs_layout.invalidate()
                stairs_layout.activate()
            sync_stairs_scroll_height()
            QTimer.singleShot(0, sync_stairs_scroll_height)

        self.stair_status_label = QLabel("Stairs: none")
        self.stair_status_label.setWordWrap(True)
        stairs_layout.addWidget(self.stair_status_label)

        self.add_stairs_button = QPushButton("Add stairs")
        self.add_stairs_button.setMinimumHeight(40)
        self.add_stairs_button.clicked.connect(self._handle_add_stairs_clicked)
        stairs_layout.addWidget(self.add_stairs_button)
        # The local scroll area protects readable field widths if the entire
        # application is resized narrower than the editor can reflow.
        minimum_stair_form_width = max(
            self.stair_starting_step_label.sizeHint().width()
            + starting_step_field_layout.spacing()
            + self.stair_starting_step_combo.minimumSizeHint().width(),
            2
            * max(
                self.stair_tread_edge_label.sizeHint().width(),
                self.stair_tread_edge_combo.minimumSizeHint().width(),
                self.stair_tread_edge_radius_label.sizeHint().width(),
                self.stair_tread_edge_radius_spinbox.minimumSizeHint().width(),
            )
            + tread_edge_row_layout.spacing(),
            3 * placement_checkbox_width
            + 2 * nosing_placement_layout.spacing(),
        )
        self.stairs_group.setMinimumWidth(minimum_stair_form_width + 16)
        self.stairs_scroll_area = QScrollArea()
        self.stairs_scroll_area.setWidgetResizable(True)
        self.stairs_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.stairs_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.stairs_scroll_area.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.stairs_scroll_area.setWidget(self.stairs_group)
        side_layout.addWidget(self.stairs_scroll_area)

        def sync_stairs_scroll_height() -> None:
            # Only the outer Generals page should scroll vertically. Refresh
            # once more after Qt settles a responsive layout change.
            editor_height = self.stairs_group.sizeHint().height()
            if self.stairs_group.maximumHeight() != editor_height:
                self.stairs_group.setMaximumHeight(editor_height)
            scroll_height = (
                editor_height
                + 2
                + self.stairs_scroll_area.horizontalScrollBar().sizeHint().height()
            )
            if self.stairs_scroll_area.minimumHeight() != scroll_height:
                self.stairs_scroll_area.setMinimumHeight(scroll_height)

        self._stair_editor_height_filter = StairEditorHeightFilter(
            self.stairs_group,
            (self.stairs_group, self.stair_nosing_row_widget),
            sync_stairs_scroll_height,
        )
        self.stairs_group.installEventFilter(self._stair_editor_height_filter)
        self.stair_nosing_row_widget.installEventFilter(
            self._stair_editor_height_filter
        )

        self._stairs_scroll_width_filter = ViewportWidthRowFilter(
            generals_tab.viewport(),
            (self.stairs_scroll_area,),
            horizontal_inset=10,
            on_width_changed=sync_stair_rows_to_viewport,
        )
        generals_tab.viewport().installEventFilter(self._stairs_scroll_width_filter)
        self._stairs_scroll_width_filter.sync_widths()

        self.doorways_group = QGroupBox("Doorways")
        doorways_layout = QVBoxLayout(self.doorways_group)

        self.selected_doorway_arch_checkbox = QCheckBox("Arch selected doorway")
        self.selected_doorway_arch_checkbox.setEnabled(False)
        self.selected_doorway_arch_checkbox.setToolTip(
            "Select a placed doorway on the Canvas, then enable this to "
            "replace its flat top with an arch."
        )
        self.selected_doorway_arch_checkbox.toggled.connect(
            self._handle_selected_doorway_arch_toggled
        )
        doorways_layout.addWidget(self.selected_doorway_arch_checkbox)

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
        doorways_layout.addLayout(doorway_arch_form)

        self.doorway_preset_list = QListWidget()
        self.doorway_preset_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.doorway_preset_list.setMinimumHeight(112)
        self.doorway_preset_list.currentRowChanged.connect(
            self._handle_doorway_preset_selection_changed
        )
        doorways_layout.addWidget(self.doorway_preset_list)

        self.save_doorway_template_button = QPushButton("Save doorway template")
        self.save_doorway_template_button.setMinimumHeight(40)
        self.save_doorway_template_button.setEnabled(False)
        self.save_doorway_template_button.clicked.connect(
            self._handle_save_doorway_template_clicked
        )
        doorways_layout.addWidget(self.save_doorway_template_button)

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
        doorways_layout.addLayout(doorway_buttons_layout)
        side_layout.addWidget(self.doorways_group)

        self.wall_mirrors_group = QGroupBox("Wall mirrors")
        wall_mirrors_layout = QHBoxLayout(self.wall_mirrors_group)
        wall_mirrors_layout.setSpacing(10)

        self.wall_mirror_up_button = QPushButton("Up")
        self.wall_mirror_up_button.setObjectName("canvas-wall-mirror-up-button")
        self.wall_mirror_up_button.setMinimumHeight(40)
        self.wall_mirror_up_button.setToolTip(
            "Copy the selected wall vertices and their shared edges to the "
            "next level above."
        )
        self.wall_mirror_up_button.clicked.connect(self._handle_wall_mirror_up_clicked)
        wall_mirrors_layout.addWidget(self.wall_mirror_up_button)

        self.wall_mirror_undo_button = QPushButton("Undo")
        self.wall_mirror_undo_button.setObjectName("canvas-wall-mirror-undo-button")
        self.wall_mirror_undo_button.setMinimumHeight(40)
        self.wall_mirror_undo_button.setToolTip(
            "Remove mirrors owned by or represented by the selected vertices."
        )
        self.wall_mirror_undo_button.clicked.connect(
            self._handle_wall_mirror_undo_clicked
        )
        wall_mirrors_layout.addWidget(self.wall_mirror_undo_button)

        self.wall_mirror_down_button = QPushButton("Down")
        self.wall_mirror_down_button.setObjectName("canvas-wall-mirror-down-button")
        self.wall_mirror_down_button.setMinimumHeight(40)
        self.wall_mirror_down_button.setToolTip(
            "Copy the selected wall vertices and their shared edges to the "
            "next level below."
        )
        self.wall_mirror_down_button.clicked.connect(
            self._handle_wall_mirror_down_clicked
        )
        wall_mirrors_layout.addWidget(self.wall_mirror_down_button)
        side_layout.addWidget(self.wall_mirrors_group)

        self.levels_group = QGroupBox("Levels")
        levels_layout = QVBoxLayout(self.levels_group)

        plan_image_buttons_layout = QHBoxLayout()
        plan_image_buttons_layout.setContentsMargins(0, 0, 0, 0)
        plan_image_buttons_layout.setSpacing(10)

        self.load_image_button = QPushButton("Load plan image")
        self.load_image_button.setMinimumHeight(44)
        self.load_image_button.clicked.connect(self._handle_load_image_clicked)
        plan_image_buttons_layout.addWidget(self.load_image_button)

        self.erase_plan_image_button = QPushButton("Erase")
        self.erase_plan_image_button.setCheckable(True)
        self.erase_plan_image_button.setMinimumHeight(44)
        self.erase_plan_image_button.setToolTip(
            "Drag a rectangle around unwanted plan-image content to erase it. "
            "Right-click or press Escape to exit."
        )
        self.erase_plan_image_button.clicked.connect(
            self._handle_erase_plan_image_clicked
        )
        plan_image_buttons_layout.addWidget(self.erase_plan_image_button)

        self.generate_walls_button = QPushButton("Generate walls")
        self.generate_walls_button.setMinimumHeight(44)
        self.generate_walls_button.setToolTip(
            "Analyze the current plan without changing the level. Generated "
            "wall faces appear as a live Canvas preview until confirmed."
        )
        self.generate_walls_button.clicked.connect(
            self._handle_generate_walls_clicked
        )
        plan_image_buttons_layout.addWidget(self.generate_walls_button)

        self.image_correction_button = QPushButton("Image correction")
        self.image_correction_button.setMinimumHeight(44)
        self.image_correction_button.setToolTip(
            "Use the plan correction model selected in Settings to flatten and "
            "clean the loaded photograph into a black-and-white plan. The original "
            "photo is retained. OpenAI models may incur usage and cost; the "
            "local Qwen model is limited to non-commercial research or evaluation "
            "unless separately licensed under the Qwen Research License."
        )
        self.image_correction_button.clicked.connect(
            self._handle_image_correction_clicked
        )
        plan_image_buttons_layout.addWidget(self.image_correction_button)
        levels_layout.addLayout(plan_image_buttons_layout)

        default_wall_options = PlanWallDetectionOptions()
        self.plan_wall_controls_group = QGroupBox("Automatic wall placement")
        plan_wall_controls_layout = QFormLayout(self.plan_wall_controls_group)
        plan_wall_controls_layout.setContentsMargins(8, 8, 8, 8)
        plan_wall_controls_layout.setSpacing(6)
        plan_wall_controls_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        (
            minimum_wall_separation_field,
            self.minimum_wall_separation_slider,
            self.minimum_wall_separation_value_label,
        ) = self._build_plan_wall_slider_field(
            minimum=1,
            maximum=80,
            value=round(default_wall_options.minimum_wall_separation_pixels),
            value_text=(
                f"{round(default_wall_options.minimum_wall_separation_pixels)} px"
            ),
            tooltip=(
                "Smallest perpendicular distance, in plan-image pixels, allowed "
                "between two parallel lines that form the faces of one wall. "
                "Increase it to reject thin annotation-line pairs; values that "
                "are too high can discard genuinely thin walls."
            ),
        )
        plan_wall_controls_layout.addRow(
            "Minimum wall thickness",
            minimum_wall_separation_field,
        )

        (
            maximum_wall_separation_field,
            self.maximum_wall_separation_slider,
            self.maximum_wall_separation_value_label,
        ) = self._build_plan_wall_slider_field(
            minimum=2,
            maximum=200,
            value=round(default_wall_options.maximum_wall_separation_pixels),
            value_text=(
                f"{round(default_wall_options.maximum_wall_separation_pixels)} px"
            ),
            tooltip=(
                "Largest perpendicular distance, in plan-image pixels, allowed "
                "between two parallel lines that form the faces of one wall. "
                "Increase it to detect thicker walls; values that are too high "
                "can pair unrelated parallel lines."
            ),
        )
        plan_wall_controls_layout.addRow(
            "Maximum wall thickness",
            maximum_wall_separation_field,
        )

        (
            parallel_angle_field,
            self.parallel_wall_angle_slider,
            self.parallel_wall_angle_value_label,
        ) = self._build_plan_wall_slider_field(
            minimum=1,
            maximum=30,
            value=round(default_wall_options.parallel_angle_tolerance_degrees),
            value_text=(
                f"{round(default_wall_options.parallel_angle_tolerance_degrees)}"
                "\N{DEGREE SIGN}"
            ),
            tooltip=(
                "Largest direction difference allowed when treating two lines "
                "as parallel wall faces. Increase it for skewed photographs or "
                "imperfect drawings; values that are too high can pair lines "
                "from different walls."
            ),
        )
        plan_wall_controls_layout.addRow(
            "Parallel angle tolerance",
            parallel_angle_field,
        )

        (
            parallel_overlap_field,
            self.parallel_wall_overlap_slider,
            self.parallel_wall_overlap_value_label,
        ) = self._build_plan_wall_slider_field(
            minimum=5,
            maximum=100,
            value=round(default_wall_options.minimum_parallel_overlap_ratio * 100),
            value_text=(
                f"{round(default_wall_options.minimum_parallel_overlap_ratio * 100)}%"
            ),
            tooltip=(
                "Minimum shared projected length of two candidate wall faces, "
                "measured against the longer line. Higher values reject short "
                "or staggered matches; lower values recover partially obscured "
                "walls but can accept unrelated lines."
            ),
        )
        plan_wall_controls_layout.addRow(
            "Minimum parallel overlap",
            parallel_overlap_field,
        )

        (
            gap_bridge_field,
            self.wall_gap_bridge_slider,
            self.wall_gap_bridge_value_label,
        ) = self._build_plan_wall_slider_field(
            minimum=0,
            maximum=250,
            value=round(default_wall_options.maximum_gap_bridge_pixels),
            value_text=f"{round(default_wall_options.maximum_gap_bridge_pixels)} px",
            tooltip=(
                "Largest along-line gap, in plan-image pixels, bridged between "
                "collinear wall fragments. Increase it to span door openings or "
                "broken scan ink; values that are too high can join separate "
                "walls that happen to align."
            ),
        )
        plan_wall_controls_layout.addRow("Gap bridge distance", gap_bridge_field)

        (
            endpoint_snap_field,
            self.wall_endpoint_snap_slider,
            self.wall_endpoint_snap_value_label,
        ) = self._build_plan_wall_slider_field(
            minimum=0,
            maximum=100,
            value=round(default_wall_options.endpoint_snap_distance_pixels),
            value_text=(
                f"{round(default_wall_options.endpoint_snap_distance_pixels)} px"
            ),
            tooltip=(
                "Maximum distance, in plan-image pixels, used to extend nearby "
                "nonparallel wall lines to their theoretical corner or T-junction. "
                "It also lets generated walls reuse nearby endpoints of existing "
                "walls; values that are too high can create false junctions."
            ),
        )
        plan_wall_controls_layout.addRow("Endpoint snap distance", endpoint_snap_field)

        (
            maximum_vertex_distance_field,
            self.maximum_vertex_distance_slider,
            self.maximum_vertex_distance_value_label,
        ) = self._build_plan_wall_slider_field(
            minimum=0,
            maximum=50,
            value=round(default_wall_options.maximum_vertex_distance_pixels),
            value_text=(
                f"{round(default_wall_options.maximum_vertex_distance_pixels)} px"
            ),
            tooltip=(
                "Generated vertex candidates at or closer than this distance are "
                "consolidated, reusing an existing wall vertex when possible. "
                "Opposite parallel wall faces are protected. Increase it to "
                "remove near-duplicate vertices; 0 disables this extra merge. "
                "Values that are too high can merge distinct nearby corners or "
                "erase very short wall details."
            ),
        )
        plan_wall_controls_layout.addRow(
            "Maximum vertex distance",
            maximum_vertex_distance_field,
        )

        (
            minimum_wall_length_field,
            self.minimum_wall_length_slider,
            self.minimum_wall_length_value_label,
        ) = self._build_plan_wall_slider_field(
            minimum=5,
            maximum=500,
            value=round(default_wall_options.minimum_wall_length_pixels),
            value_text=f"{round(default_wall_options.minimum_wall_length_pixels)} px",
            tooltip=(
                "Shortest detected line, in plan-image pixels, that may remain as "
                "a wall face. Increase it to filter text and dimension marks; "
                "values that are too high can remove short real walls."
            ),
        )
        plan_wall_controls_layout.addRow(
            "Minimum wall length",
            minimum_wall_length_field,
        )

        (
            confidence_field,
            self.wall_detection_confidence_slider,
            self.wall_detection_confidence_value_label,
        ) = self._build_plan_wall_slider_field(
            minimum=0,
            maximum=100,
            value=round(default_wall_options.confidence_threshold * 100),
            value_text=f"{round(default_wall_options.confidence_threshold * 100)}%",
            tooltip=(
                "Minimum geometric confidence required for a detected wall pair. "
                "Confidence combines line strength, length, overlap, angle, "
                "separation, existing-wall alignment, and supporting outlines. "
                "Higher values are cleaner but may omit uncertain walls."
            ),
        )
        plan_wall_controls_layout.addRow("Detection confidence", confidence_field)

        self.plan_wall_generation_status_label = QLabel(
            "Generate walls to analyze the current plan."
        )
        self.plan_wall_generation_status_label.setWordWrap(True)
        plan_wall_controls_layout.addRow(self.plan_wall_generation_status_label)
        self.plan_wall_controls_group.setEnabled(False)
        levels_layout.addWidget(self.plan_wall_controls_group)

        for slider in self._get_plan_wall_generation_sliders():
            slider.valueChanged.connect(self._handle_plan_wall_slider_changed)

        self.blueprint_name_label = QLabel("Image: none for this level")
        self.blueprint_name_label.setWordWrap(True)
        levels_layout.addWidget(self.blueprint_name_label)

        self.levels_list = QListWidget()
        self.levels_list.currentRowChanged.connect(self._handle_level_list_row_changed)
        levels_layout.addWidget(self.levels_list, 1)

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
        levels_layout.addLayout(level_options_layout)
        side_layout.addWidget(self.levels_group, 1)

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
        self._update_image_correction_button_state()
        self._update_plan_wall_generation_controls_state()
        self._update_plan_image_erase_button_state()

        self._generals_value_input_wheel_filter = RightPanelValueInputWheelFilter(
            generals_tab
        )
        for spinbox in generals_content.findChildren(QAbstractSpinBox):
            spinbox.installEventFilter(self._generals_value_input_wheel_filter)
            spinbox.lineEdit().installEventFilter(
                self._generals_value_input_wheel_filter
            )
        for combo in (
            self.stair_type_combo,
            self.stair_tread_edge_combo,
            self.stair_starting_step_combo,
            self.stair_stringer_placement_combo,
        ):
            combo.installEventFilter(self._generals_value_input_wheel_filter)
        for slider in (
            self.level_scale_slider,
            self.level_x_offset_slider,
            self.level_y_offset_slider,
            self.canvas_level_scale_slider,
            self.canvas_x_offset_slider,
            self.canvas_y_offset_slider,
        ):
            slider.installEventFilter(self._generals_value_input_wheel_filter)

        self.workspace_splitter.addWidget(self.side_panel)
        self.workspace_splitter.setStretchFactor(0, 9)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes([1060, 540])

        self.canvas.geometry_changed.connect(
            self._handle_canvas_surface_geometry_changed
        )
        self.canvas.wall_vertex_added.connect(self._handle_canvas_wall_vertex_added)
        self.canvas.wall_vertex_interaction_changed.connect(
            self._handle_canvas_wall_vertex_interaction_changed
        )
        self.canvas.selected_vertices_changed.connect(
            self._handle_canvas_selected_vertex_changed
        )
        self.canvas.doorways_changed.connect(self._handle_doorways_changed)
        self.canvas.open_spaces_changed.connect(self._handle_open_spaces_changed)
        self.canvas.open_space_placement_changed.connect(
            self._handle_open_space_placement_changed
        )
        self.canvas.plan_image_erase_mode_changed.connect(
            self._handle_plan_image_erase_mode_changed
        )
        self.canvas.plan_image_erase_committed.connect(
            self._handle_plan_image_erase_committed
        )
        self.canvas.plan_image_erase_failed.connect(
            self._handle_plan_image_erase_failed
        )
        self.canvas.doorway_dimension_preview_changed.connect(
            self._handle_doorway_dimension_preview_changed
        )
        self.canvas.doorway_move_drag_started.connect(
            self._handle_doorway_move_drag_started
        )
        self.canvas.doorway_move_drag_finished.connect(
            self._handle_doorway_move_drag_finished
        )
        self.canvas.doorway_resize_drag_started.connect(
            self._handle_doorway_resize_drag_started
        )
        self.canvas.doorway_resize_drag_finished.connect(
            self._handle_doorway_resize_drag_finished
        )
        self.canvas.selected_doorway_changed.connect(
            self._handle_canvas_doorway_selection_changed
        )
        self.canvas.stair_start_placed.connect(self._handle_stair_start_placed)
        self.canvas.stair_placement_ready.connect(self._handle_stair_placement_ready)
        self.canvas.stair_placement_completed.connect(
            self._handle_stair_placement_completed
        )
        self.canvas.stair_placement_cancelled.connect(
            self._handle_stair_placement_cancelled
        )
        self.canvas.stair_placement_invalid_endpoint.connect(
            self._handle_stair_placement_invalid_endpoint
        )
        self.canvas.stair_delete_requested.connect(self._handle_stair_delete_requested)
        self.canvas.stair_point_drag_finished.connect(
            self._handle_stair_point_drag_finished
        )
        self._update_wall_mirror_button_state()
        self._refresh_levels_list()
        self._update_stair_button_state()
        self._update_open_space_controls()
        self._sync_level_controls()
        self._sync_canvas_to_current_level()
        self._schedule_viewer_preview_refresh(preserve_camera=False)
        self._apply_scene_3d_display_screen(
            generation_settings.scene_3d_display_screen_id
        )
        self._apply_generation_display_screen(
            generation_settings.generation_display_screen_id
        )
        self._apply_jobs_window_screen(generation_settings.jobs_window_screen_id)
        self._apply_atlas_display_screen(generation_settings.atlas_display_screen_id)

    def _build_canvas_viewer_workspace(self) -> QWidget:
        """Place only the editable 2D blueprint in the Canvas workspace."""

        workspace = QWidget()
        workspace.setObjectName("canvas-viewer-workspace")
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.addWidget(self.canvas)
        return workspace

    def _build_scene_3d_workspace(self) -> QWidget:
        """Wrap the shared scene viewer as an independently hosted workspace."""

        workspace = QWidget()
        workspace.setObjectName("scene-3d-workspace")
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.addWidget(self.viewer)
        return workspace

    def _set_canvas_3d_navigation_shortcut(self, hotkey: str) -> None:
        """Apply the persisted Canvas-only navigation shortcut."""

        self.canvas_3d_navigation_shortcut.setKey(
            QKeySequence(hotkey, QKeySequence.SequenceFormat.PortableText)
        )

    def _toggle_canvas_3d_navigation_mode(self) -> None:
        """Toggle the active Canvas viewer between orbit and first person."""

        scene_is_local = (
            self.workspace_tabs.currentWidget() is self.scene_3d_workspace
        )
        if not scene_is_local and not self._external_scene_3d_host.is_active:
            return

        self.viewer.toggle_navigation_mode()

    def _toggle_first_person_generation_frame_overlay(self) -> None:
        """Toggle the current Generation frame above the first-person scene."""

        if not self.viewer.is_first_person_active:
            return
        if self.viewer.is_first_person_frame_overlay_visible:
            self.viewer.clear_first_person_frame_overlay()
            return
        self.viewer.set_first_person_frame_overlay(
            self.merged_generation_workspace.get_current_video_frame_bgr()
        )

    def _sync_first_person_generation_frame_overlay(self) -> None:
        """Keep an active scene overlay synchronized with Generation seeking."""

        if not self.viewer.is_first_person_frame_overlay_visible:
            return
        self.viewer.set_first_person_frame_overlay(
            self.merged_generation_workspace.get_current_video_frame_bgr()
        )

    def _handle_canvas_3d_navigation_mode_changed(self, mode: str) -> None:
        """Keep the 2D camera indicator synchronized with scene navigation."""

        del mode
        self.canvas.set_camera_indicator_pose(
            self.viewer.get_first_person_camera_pose()
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
        self._record_canvas_undo_state(
            _CanvasWindowAdditionUndoState(window_id=window.window_id)
        )
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
            self.viewer.set_window_tools_status(f"Window could not be undone: {error}")
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
            self.viewer.set_window_tools_status(f"Window could not be undone: {error}")
            return

        self._canvas_window_undo_ids.pop()
        if not self._is_restoring_canvas_undo:
            self._discard_removed_window_undo_states(window_id)
        self._sync_canvas_window_undo_availability()
        self.viewer.set_window_tools_status("Window undone.")

    def _discard_removed_window_undo_states(self, window_id: str) -> None:
        """Remove history entries whose stable window target was removed."""

        normalized_window_id = str(window_id).strip()
        self._canvas_undo_stack = [
            state
            for state in self._canvas_undo_stack
            if not (
                (
                    isinstance(state, _CanvasWindowAdditionUndoState)
                    and state.window_id == normalized_window_id
                )
                or (
                    isinstance(state, _CanvasOpeningEditUndoState)
                    and state.start_edit.reference.kind == CANVAS_OPENING_WINDOW
                    and state.start_edit.reference.stable_id == normalized_window_id
                )
            )
        ]

    # ### Canvas opening gizmo edits ###
    def _handle_canvas_opening_selection_changed(
        self,
        raw_reference: object,
    ) -> None:
        """Keep doorway controls aligned with selection in the 3D viewer."""

        reference = (
            raw_reference if isinstance(raw_reference, CanvasOpeningReference) else None
        )
        if reference is not None:
            self._discard_staged_stair_edit(clear_selection=True)
            self._desired_canvas_object_id = None
            self._desired_canvas_object_ids = ()
            self._desired_canvas_surface_ids = ()
            self._atlas_surface_assignment_target_ids = ()
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
        self._commit_pending_canvas_surface_mesh_update()
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
        target = self._canvas_opening_targets_by_key.get(raw_edit.reference.key)
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
        changed: bool,
    ) -> None:
        """Start the complete configured delay only after mouse release."""

        if not isinstance(raw_edit, CanvasOpeningEdit):
            return
        if (
            self._active_canvas_opening_reference is not None
            and raw_edit.reference != self._active_canvas_opening_reference
        ):
            return
        start_edit = self._active_canvas_opening_start_edit
        if changed and start_edit is not None and raw_edit.bounds != start_edit.bounds:
            self._record_canvas_undo_state(
                _CanvasOpeningEditUndoState(start_edit=start_edit),
                commit_pending_surface_edit=False,
            )
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
            self._pending_doorway_mesh_level_index = level.index if is_pending else None
        else:
            committed = self._viewer_windows_by_level_index.get(level.index)
            is_pending = self._copy_windows(level.windows) != committed
            self._pending_window_mesh_level_index = level.index if is_pending else None
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

        wall_targets = tuple(build_fixed_surfaces(self._build_viewer_preview_levels()))
        if not self.texture_atlas_workspace.is_ambient_occlusion_preview_active:
            self._set_canvas_viewer_targets(wall_targets)
            self._is_syncing_canvas_scene_selection = True
            try:
                self.viewer.set_model(generated_model, preserve_camera=True)
                self._restore_desired_canvas_scene_selection()
            finally:
                self._is_syncing_canvas_scene_selection = False
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
        """Install semantic surfaces and their selectable opening overlays."""

        surface_targets = tuple(
            surface for surface in surfaces if isinstance(surface, FixedSurface)
        )
        installed_surface_ids = {surface.surface_id for surface in surface_targets}
        self._canvas_surface_targets_by_id = {
            surface.surface_id: surface for surface in surface_targets
        }
        self._desired_canvas_surface_ids = tuple(
            surface_id
            for surface_id in self._desired_canvas_surface_ids
            if surface_id in installed_surface_ids
        )
        self._is_syncing_canvas_scene_selection = True
        try:
            self.viewer.set_wall_targets(surface_targets)
            try:
                edit_targets = build_canvas_surface_edit_targets(
                    self.levels,
                    surface_targets,
                )
            except (TypeError, ValueError):
                edit_targets = ()
            self._canvas_surface_edit_targets_by_key = {
                (target.surface_id, target.reference.key): target
                for target in edit_targets
            }
            self.viewer.set_canvas_surface_edit_targets(edit_targets)
            self._sync_canvas_surface_drawing_overlay()
            if self._desired_canvas_surface_ids:
                self.viewer.set_selected_canvas_surface_ids(
                    self._desired_canvas_surface_ids
                )
        finally:
            self._is_syncing_canvas_scene_selection = False
        try:
            opening_targets = build_canvas_opening_targets(
                self.levels,
                surface_targets,
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
        stair_part_targets = self._sync_canvas_stair_semantic_targets(
            self._build_viewer_preview_levels()
        )
        assignable_surface_ids = (
            installed_surface_ids
            | self._canvas_stair_semantic_surfaces_by_id.keys()
        )
        self._atlas_surface_assignment_target_ids = tuple(
            surface_id
            for surface_id in self._atlas_surface_assignment_target_ids
            if surface_id in assignable_surface_ids
        )
        self._desired_canvas_stair_part_ids = tuple(
            semantic_id
            for semantic_id in self._desired_canvas_stair_part_ids
            if semantic_id in self._canvas_stair_part_targets_by_id
        )
        self._is_syncing_canvas_scene_selection = True
        try:
            self.viewer.set_canvas_stair_part_targets(stair_part_targets)
            if self._desired_canvas_stair_part_ids:
                self.viewer.set_selected_canvas_stair_part_ids(
                    self._desired_canvas_stair_part_ids
                )
        finally:
            self._is_syncing_canvas_scene_selection = False
        self._sync_surface_generation_selection(
            self._desired_canvas_stair_part_ids
            or self._desired_canvas_surface_ids
        )
        selected_surface_source_ids = (
            self.texture_atlas_workspace.selected_surface_texture_ids
        )
        if selected_surface_source_ids:
            self._handle_atlas_surface_textures_selected(selected_surface_source_ids)
        elif self._selected_atlas_surface_source_id is not None:
            self._handle_atlas_surface_texture_selected(
                self._selected_atlas_surface_source_id
            )
        else:
            self._set_atlas_canvas_surface_highlights(())
            self._sync_atlas_green_outline_to_canvas_highlight(None)
        self._sync_selected_canvas_wall_highlight(
            self.viewer.get_active_canvas_surface_id()
        )

    # ### Canvas stair editor ###
    def _sync_canvas_stair_semantic_targets(
        self,
        levels: Sequence[LevelData],
    ) -> tuple[PreviewStairPart, ...]:
        """Publish authoritative stair groups to selection and texture state."""

        stair_part_targets_list: list[PreviewStairPart] = []
        for stair_index, stair in enumerate(self.stairs):
            try:
                stair_part_targets_list.extend(
                    build_canvas_stair_part_targets(
                        levels,
                        (stair,),
                        stair_indices=(stair_index,),
                    )
                )
            except (TypeError, ValueError):
                continue
        stair_part_targets = tuple(stair_part_targets_list)
        self._canvas_stair_part_targets_by_id = {
            target.semantic_id: target for target in stair_part_targets
        }
        stair_semantic_surfaces = tuple(
            FixedSurface(
                surface_id=target.semantic_id,
                surface_type=target.surface_type,
                level_index=target.level_indices[0],
                room_index=None,
                mesh=target.mesh,
                area_square_meters=float(target.mesh.area),
            )
            for target in stair_part_targets
            if target.level_indices and float(target.mesh.area) > 0.0
        )
        self._canvas_stair_semantic_surfaces_by_id = {
            surface.surface_id: surface for surface in stair_semantic_surfaces
        }
        self.surface_texture_generation.set_external_semantic_surfaces(
            stair_semantic_surfaces
        )
        return stair_part_targets

    def _reconcile_surface_assignments_with_scene(
        self,
        *,
        emit_signals: bool = True,
    ) -> bool:
        """Rebuild procedural targets before reconciling shared assignments."""

        if hasattr(self, "stairs"):
            BlueprintWorkspace._sync_canvas_stair_semantic_targets(
                self,
                self.levels,
            )
        if hasattr(self, "_stair_preview_update_timer"):
            BlueprintWorkspace._refresh_stair_editor_geometry(self)
        if emit_signals:
            return self.surface_texture_generation.reconcile_assignments_with_levels(
                self.levels
            )
        return self.surface_texture_generation.reconcile_assignments_with_levels(
            self.levels,
            emit_signals=False,
        )

    def _retarget_canvas_stair_selection_after_edit(
        self,
        edited_stair_id: str,
    ) -> None:
        """Keep an edited stair selected when its previous part disappears."""

        retained_ids = [
            semantic_id
            for semantic_id in self._desired_canvas_stair_part_ids
            if semantic_id in self._canvas_stair_part_targets_by_id
        ]
        retains_edited_stair = any(
            self._canvas_stair_part_targets_by_id[semantic_id].stair_id
            == edited_stair_id
            for semantic_id in retained_ids
        )
        if not retains_edited_stair:
            fallback = next(
                (
                    target.semantic_id
                    for target in self._canvas_stair_part_targets_by_id.values()
                    if target.stair_id == edited_stair_id
                    and target.part_kind == STAIR_PART_TREADS
                ),
                None,
            )
            if fallback is not None:
                retained_ids.append(fallback)
        self._desired_canvas_stair_part_ids = tuple(dict.fromkeys(retained_ids))
        self._atlas_surface_assignment_target_ids = (
            self._desired_canvas_stair_part_ids
        )
        self._sync_surface_generation_selection(
            self._desired_canvas_stair_part_ids
        )
        self._sync_atlas_texture_selection_from_canvas_scene()

    def _handle_canvas_stair_part_selection_changed(
        self,
        raw_semantic_ids: object,
    ) -> None:
        """Load the owner of any selected stair part into the editor."""

        if self._is_syncing_canvas_scene_selection:
            return
        try:
            semantic_ids = tuple(
                dict.fromkeys(
                    str(value)
                    for value in raw_semantic_ids  # type: ignore[union-attr]
                )
            )
        except TypeError:
            return
        semantic_ids = tuple(
            semantic_id
            for semantic_id in semantic_ids
            if semantic_id in self._canvas_stair_part_targets_by_id
        )
        self._desired_canvas_stair_part_ids = semantic_ids
        target = (
            self._canvas_stair_part_targets_by_id.get(semantic_ids[-1])
            if semantic_ids
            else None
        )
        stair_index = None if target is None else int(target.stair_index)
        if stair_index is None or not 0 <= stair_index < len(self.stairs):
            self._discard_staged_stair_edit(clear_selection=True)
            self._atlas_surface_assignment_target_ids = ()
            self._sync_surface_generation_selection(())
            self._sync_atlas_texture_selection_from_canvas_scene()
            return
        self._desired_canvas_object_id = None
        self._desired_canvas_object_ids = ()
        self._desired_canvas_surface_ids = ()
        self._atlas_surface_assignment_target_ids = semantic_ids
        self._sync_surface_generation_selection(semantic_ids)
        self._sync_atlas_texture_selection_from_canvas_scene()
        if self._editing_stair_index == stair_index:
            return

        if self._editing_stair_index is None:
            self._new_stair_parameters = self._read_stair_editor_parameters()
        self._discard_staged_stair_edit(clear_selection=False)
        self._editing_stair_index = stair_index
        self._load_stair_editor_from_stair(self.stairs[stair_index])
        self._update_stair_button_state()

    def _handle_canvas_stair_preview_cancelled(self) -> None:
        """Restore persisted settings after Escape or transient undo."""

        self._discard_staged_stair_edit(clear_selection=False)

    def _handle_canvas_stair_escape_requested(self) -> None:
        """Discard a field change even before its preview debounce expires."""

        if self._pending_stair_parameters is None:
            return
        self._discard_staged_stair_edit(clear_selection=False)

    # ### Canvas surface selection synchronization ###
    def _get_canvas_wall_surface_ids(
        self,
        surface_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Return selected root-wall IDs in stable selection order."""

        return tuple(
            surface_id
            for surface_id in dict.fromkeys(surface_ids)
            if (
                (surface := self._canvas_surface_targets_by_id.get(surface_id))
                is not None
                and surface.surface_type == SURFACE_TYPE_WALL
                and surface.source_surface_id is None
            )
        )

    def _handle_canvas_surface_selection_changed(
        self,
        raw_surface_ids: object,
    ) -> None:
        """Remember manual Canvas surface choices across preview rebuilds."""

        if self._is_syncing_canvas_scene_selection:
            return
        try:
            normalized_surface_ids = (
                str(value)
                for value in raw_surface_ids  # type: ignore[arg-type]
            )
            surface_ids = tuple(dict.fromkeys(normalized_surface_ids))
        except TypeError:
            return
        self._desired_canvas_surface_ids = surface_ids
        self._atlas_surface_assignment_target_ids = surface_ids
        if surface_ids:
            self._discard_staged_stair_edit(clear_selection=True)
            self._desired_canvas_object_id = None
            self._desired_canvas_object_ids = ()
        self._sync_surface_generation_selection(surface_ids)
        self._sync_atlas_texture_selection_from_canvas_scene()
        active_surface_id = surface_ids[-1] if surface_ids else None
        self._sync_selected_canvas_wall_highlight(active_surface_id)
        self._commit_pending_wall_vertex_update()
        pending_baseline = self._pending_canvas_surface_mesh_baseline
        if self._pending_canvas_surface_mesh_update:
            selected_wall_ids = self._get_canvas_wall_surface_ids(surface_ids)
            wall_group_changed = bool(
                self._pending_canvas_wall_surface_ids
                and selected_wall_ids != self._pending_canvas_wall_surface_ids
            )
            active_owner_changed = bool(
                not self._pending_canvas_wall_surface_ids
                and (
                    pending_baseline is None
                    or active_surface_id != pending_baseline.surface_id
                )
            )
            if wall_group_changed or active_owner_changed:
                self._commit_pending_canvas_surface_mesh_update()

    def _sync_surface_generation_selection(
        self,
        surface_ids: Sequence[str],
    ) -> None:
        """Give Surface generation the shared scene's semantic selection."""

        selected_surfaces = tuple(
            surface
            for surface_id in dict.fromkeys(str(value) for value in surface_ids)
            if (
                surface := (
                    self._canvas_surface_targets_by_id.get(surface_id)
                    or self._canvas_stair_semantic_surfaces_by_id.get(surface_id)
                )
            )
            is not None
        )
        self.surface_texture_generation.set_scene_surface_selection(
            selected_surfaces
        )

    def _handle_canvas_surface_orientation_flip_requested(
        self,
        surface_id: str,
    ) -> None:
        """Persist one clicked surface winding change through normal history."""

        self._apply_canvas_surface_topology_edit(
            lambda: flip_surface_orientation(self.levels, surface_id),
            success_message=(
                "Surface orientation flipped. Click it again to restore the "
                "automatic orientation."
            ),
        )

    def _sync_selected_canvas_wall_highlight(
        self,
        surface_id: str | None,
    ) -> None:
        """Mirror the active 3D wall selection onto the current 2D Canvas."""

        surface = self._canvas_surface_targets_by_id.get(surface_id or "")
        current_level = (
            self.levels[self.current_level_index]
            if 0 <= self.current_level_index < len(self.levels)
            else None
        )
        selected_wall_id = (
            surface.surface_id
            if surface is not None
            and surface.surface_type == SURFACE_TYPE_WALL
            and surface.source_surface_id is None
            and current_level is not None
            and surface.level_index == current_level.index
            else None
        )
        self.canvas.set_selected_wall_surface_id(selected_wall_id)

    # ### Canvas persistent surface topology edits ###
    def _record_canvas_undo_state(
        self,
        state: _CanvasUndoState,
        *,
        commit_pending_surface_edit: bool = True,
    ) -> None:
        """Append one action after committing every chronologically older edit."""

        if self._is_restoring_canvas_undo:
            return
        self._commit_pending_level_transform_update()
        if commit_pending_surface_edit:
            self._commit_pending_canvas_surface_mesh_update()
        self._canvas_undo_stack.append(state)

    def _clear_canvas_undo_history(self) -> None:
        """Start a new history branch after replacing Canvas coordinates."""

        self._stair_point_mesh_update_timer.stop()
        self._pending_stair_point_mesh_update = False
        self._pending_stair_point_undo_state = None
        self._pending_stair_point_id = None
        self._canvas_undo_stack.clear()
        self.canvas.undo_stack.clear()

    def _handle_blueprint_undo_snapshot_created(
        self,
        raw_snapshot: object,
    ) -> None:
        """Add one existing 2D Canvas transaction to shared Canvas history."""

        if self._is_restoring_canvas_undo or not isinstance(
            raw_snapshot,
            CanvasSnapshot,
        ):
            return
        tracks_surface_bindings = raw_snapshot.action_kind in {
            CANVAS_SNAPSHOT_ACTION_OPEN_SPACE,
            CANVAS_SNAPSHOT_ACTION_VERTEX_DELETION,
        }
        assignments = (
            self.surface_texture_generation.snapshot_assignments()
            if tracks_surface_bindings
            else ()
        )
        assignment_source_ids = {
            build_atlas_wall_texture_source_id(assignment.assignment_id)
            for assignment in assignments
        }
        self._record_canvas_undo_state(
            _CanvasBlueprintUndoState(
                level_index=self.current_level.index,
                snapshot=raw_snapshot,
                assignments=assignments,
                assignment_targets_after=(None if tracks_surface_bindings else ()),
                atlas_placements=tuple(
                    (atlas.atlas_id, placement)
                    for atlas in self.texture_atlas_workspace.get_data().atlases
                    for placement in atlas.placements
                    if placement.object_id in assignment_source_ids
                ),
                wall_mirror_links=self.wall_mirror_links,
                other_level_vertex_data=tuple(
                    (level.index, level.vertex_data.clone())
                    for level in self.levels
                    if level.index != self.current_level.index
                ),
                other_level_doorways=tuple(
                    (level.index, self._copy_doorways(level.doorways))
                    for level in self.levels
                    if level.index != self.current_level.index
                ),
                level_offsets_meters=(
                    (
                        float(self.current_level.offset_x_meters),
                        float(self.current_level.offset_y_meters),
                    )
                    if raw_snapshot.action_kind
                    == CANVAS_SNAPSHOT_ACTION_GENERATED_WALLS
                    else None
                ),
            )
        )

    def _finalize_blueprint_surface_binding_undo_state(self) -> None:
        """Record only bindings changed by the newest topology edit."""

        if self._is_restoring_canvas_undo or not self._canvas_undo_stack:
            return
        state = self._canvas_undo_stack[-1]
        if not (
            isinstance(state, _CanvasBlueprintUndoState)
            and state.assignment_targets_after is None
        ):
            return
        current_by_id = {
            assignment.assignment_id: assignment
            for assignment in self.surface_texture_generation.snapshot_assignments()
        }
        affected_assignments = tuple(
            assignment
            for assignment in state.assignments
            if _surface_assignment_target_signature(assignment)
            != _surface_assignment_target_signature(
                current_by_id.get(assignment.assignment_id)
            )
        )
        affected_source_ids = {
            build_atlas_wall_texture_source_id(assignment.assignment_id)
            for assignment in affected_assignments
        }
        self._canvas_undo_stack[-1] = replace(
            state,
            assignments=affected_assignments,
            assignment_targets_after=tuple(
                current_by_id[assignment.assignment_id]
                for assignment in affected_assignments
                if assignment.assignment_id in current_by_id
            ),
            atlas_placements=tuple(
                (atlas_id, placement)
                for atlas_id, placement in state.atlas_placements
                if placement.object_id in affected_source_ids
            ),
        )

    def _handle_blueprint_undo_snapshot_discarded(
        self,
        raw_snapshot: object,
    ) -> None:
        """Retract a provisional 2D snapshot when its drag ends unchanged."""

        if self._is_restoring_canvas_undo or not isinstance(
            raw_snapshot,
            CanvasSnapshot,
        ):
            return
        for index in range(len(self._canvas_undo_stack) - 1, -1, -1):
            state = self._canvas_undo_stack[index]
            if (
                isinstance(state, _CanvasBlueprintUndoState)
                and state.snapshot is raw_snapshot
            ):
                del self._canvas_undo_stack[index]
                return

    def _capture_canvas_level_properties_undo_state(
        self,
        level: LevelData,
    ) -> _CanvasLevelPropertiesUndoState:
        """Capture the level fields edited directly by Canvas side controls."""

        return _CanvasLevelPropertiesUndoState(
            level_index=level.index,
            height_meters=float(level.height_meters),
            scale=float(level.scale),
            canvas_level_scale=float(level.canvas_level_scale),
            canvas_offset_x_pixels=float(level.canvas_offset_x_pixels),
            canvas_offset_y_pixels=float(level.canvas_offset_y_pixels),
            offset_x_meters=float(level.offset_x_meters),
            offset_y_meters=float(level.offset_y_meters),
            include_in_export=bool(level.include_in_export),
        )

    def _capture_canvas_topology_undo_state(self) -> _CanvasTopologyUndoState:
        """Capture structurally shared geometry plus assignment provenance."""

        assignments = self.surface_texture_generation.snapshot_assignments()
        assignment_source_ids = {
            build_atlas_wall_texture_source_id(assignment.assignment_id)
            for assignment in assignments
        }
        atlas_placements = tuple(
            (atlas.atlas_id, placement)
            for atlas in self.texture_atlas_workspace.get_data().atlases
            for placement in atlas.placements
            if placement.object_id in assignment_source_ids
        )
        return _CanvasTopologyUndoState(
            editable_surfaces_by_level=tuple(
                (level.index, tuple(level.editable_surfaces)) for level in self.levels
            ),
            flipped_surface_ids_by_level=tuple(
                (level.index, tuple(sorted(level.flipped_surface_ids)))
                for level in self.levels
            ),
            assignments=assignments,
            assignment_targets_after=(),
            atlas_placements=atlas_placements,
            selected_surface_ids=self._desired_canvas_surface_ids,
            assignment_target_ids=self._atlas_surface_assignment_target_ids,
            selected_object_id=self._desired_canvas_object_id,
            active_vertex_id=self._active_canvas_surface_drawing_vertex_id,
            selected_object_ids=getattr(
                self,
                "_desired_canvas_object_ids",
                (
                    (self._desired_canvas_object_id,)
                    if self._desired_canvas_object_id is not None
                    else ()
                ),
            ),
        )

    def _finalize_canvas_topology_undo_state(
        self,
        state: _CanvasTopologyUndoState,
    ) -> _CanvasTopologyUndoState:
        """Limit texture history to assignments changed by this topology edit."""

        current_by_id = {
            assignment.assignment_id: assignment
            for assignment in self.surface_texture_generation.snapshot_assignments()
        }
        affected_assignments = tuple(
            assignment
            for assignment in state.assignments
            if _surface_assignment_target_signature(assignment)
            != _surface_assignment_target_signature(
                current_by_id.get(assignment.assignment_id)
            )
        )
        affected_source_ids = {
            build_atlas_wall_texture_source_id(assignment.assignment_id)
            for assignment in affected_assignments
        }
        return replace(
            state,
            assignments=affected_assignments,
            assignment_targets_after=tuple(
                current_by_id[assignment.assignment_id]
                for assignment in affected_assignments
                if assignment.assignment_id in current_by_id
            ),
            atlas_placements=tuple(
                (atlas_id, placement)
                for atlas_id, placement in state.atlas_placements
                if placement.object_id in affected_source_ids
            ),
        )

    def _capture_canvas_stairs_undo_state(self) -> _CanvasStairsUndoState:
        """Capture stairs plus texture bindings that a stair edit may change."""

        editing_stair_id = (
            self.stairs[self._editing_stair_index].stair_id
            if self._editing_stair_index is not None
            and 0 <= self._editing_stair_index < len(self.stairs)
            else None
        )
        assignments = self.surface_texture_generation.snapshot_assignments()
        assignment_source_ids = {
            build_atlas_wall_texture_source_id(assignment.assignment_id)
            for assignment in assignments
        }
        atlas_placements = tuple(
            (atlas.atlas_id, placement)
            for atlas in self.texture_atlas_workspace.get_data().atlases
            for placement in atlas.placements
            if placement.object_id in assignment_source_ids
        )
        return _CanvasStairsUndoState(
            stairs=tuple(self.stairs),
            assignments=assignments,
            atlas_placements=atlas_placements,
            selected_stair_part_ids=self._desired_canvas_stair_part_ids,
            assignment_target_ids=self._atlas_surface_assignment_target_ids,
            editing_stair_id=editing_stair_id,
        )

    def _finalize_canvas_stairs_undo_state(
        self,
        state: _CanvasStairsUndoState,
    ) -> None:
        """Limit the newest stair history entry to bindings it changed."""

        current_by_id = {
            assignment.assignment_id: assignment
            for assignment in self.surface_texture_generation.snapshot_assignments()
        }
        affected_assignments = tuple(
            assignment
            for assignment in state.assignments
            if _surface_assignment_target_signature(assignment)
            != _surface_assignment_target_signature(
                current_by_id.get(assignment.assignment_id)
            )
        )
        affected_source_ids = {
            build_atlas_wall_texture_source_id(assignment.assignment_id)
            for assignment in affected_assignments
        }
        finalized_state = replace(
            state,
            assignments=affected_assignments,
            assignment_targets_after=tuple(
                current_by_id[assignment.assignment_id]
                for assignment in affected_assignments
                if assignment.assignment_id in current_by_id
            ),
            atlas_placements=tuple(
                (atlas_id, placement)
                for atlas_id, placement in state.atlas_placements
                if placement.object_id in affected_source_ids
            ),
        )
        for index, undo_entry in enumerate(self._canvas_undo_stack):
            if undo_entry is state:
                self._canvas_undo_stack[index] = finalized_state
                break

    def _handle_canvas_undo_requested(self) -> None:
        """Undo the latest committed Canvas action from either Canvas view."""

        if (
            self._pending_stair_parameters is not None
            or self._staged_stair is not None
        ):
            self._discard_staged_stair_edit(clear_selection=False)
            self.viewer.set_surface_tools_status(
                "Current stair changes discarded."
            )
            return
        if self.canvas.cancel_open_space_placement():
            self.viewer.set_surface_tools_status(
                "Current open-space placement cancelled."
            )
            return
        if self.canvas.is_stair_placement_active():
            self.canvas.cancel_stair_placement()
            self.viewer.set_surface_tools_status("Current stair placement cancelled.")
            return
        if self.viewer.cancel_uncommitted_canvas_interaction_for_undo():
            self.viewer.set_surface_tools_status(
                "Current Canvas interaction cancelled."
            )
            return
        if self.viewer.cancel_canvas_opening_edit():
            self.viewer.set_surface_tools_status("Current opening drag cancelled.")
            return
        if self._cancel_active_canvas_surface_edit():
            self.viewer.set_surface_tools_status("Current Canvas drag cancelled.")
            return
        if self._pending_level_transform is not None:
            self._level_transform_drag_active = False
            self._cancel_pending_level_transform()
            self.viewer.set_surface_tools_status("Level transform edit undone.")
            return
        if self._undo_pending_canvas_surface_mesh_update():
            self.viewer.set_surface_tools_status("Canvas surface edit undone.")
            return
        if self._pending_stair_point_mesh_update:
            self._stair_point_mesh_update_timer.stop()
            self._pending_stair_point_mesh_update = False
            self._pending_stair_point_undo_state = None
            self._pending_stair_point_id = None
        if not self._canvas_undo_stack:
            self.viewer.set_surface_tools_status("No Canvas action to undo.")
            return
        state = self._canvas_undo_stack[-1]
        skipped_texture_bindings = 0
        self._is_restoring_canvas_undo = True
        try:
            self._cancel_pending_canvas_surface_mesh_update()
            self._cancel_pending_wall_vertex_update()
            self._cancel_pending_doorway_mesh_update(clear_outline=True)
            self._stair_point_mesh_update_timer.stop()
            self._pending_stair_point_mesh_update = False
            self._pending_stair_point_undo_state = None
            self._pending_stair_point_id = None
            if isinstance(state, _CanvasPlanImageUndoState):
                self._restore_canvas_plan_image_undo_state(state)
            elif isinstance(state, _CanvasTopologyUndoState):
                skipped_texture_bindings = self._restore_canvas_topology_undo_state(
                    state
                )
            elif isinstance(state, _CanvasSurfaceEditUndoState):
                self._restore_canvas_surface_edit_undo_state(state)
            elif isinstance(state, _CanvasLevelPropertiesUndoState):
                self._restore_canvas_level_properties_undo_state(state)
            elif isinstance(state, _CanvasWallMirrorUndoState):
                self._restore_canvas_wall_mirror_undo_state(state)
            elif isinstance(state, _CanvasOpeningEditUndoState):
                self._restore_canvas_opening_edit_undo_state(state)
            elif isinstance(state, _CanvasWindowAdditionUndoState):
                self._restore_canvas_window_addition_undo_state(state)
            elif isinstance(state, _CanvasPlacedObjectUndoState):
                skipped_texture_bindings = (
                    self._restore_canvas_placed_object_undo_state(state)
                )
            elif isinstance(state, _CanvasPlacedObjectGroupUndoState):
                skipped_texture_bindings = sum(
                    self._restore_canvas_placed_object_undo_state(member)
                    for member in reversed(state.members)
                )
            elif isinstance(state, _CanvasStairsUndoState):
                skipped_texture_bindings = (
                    self._restore_canvas_stairs_undo_state(state)
                )
            elif isinstance(state, _SurfaceTextureTilingUndoState):
                self._restore_surface_texture_tiling_undo_state(state)
            else:
                skipped_texture_bindings = self._restore_blueprint_undo_state(state)
        except (RuntimeError, TypeError, ValueError) as error:
            self.viewer.set_surface_tools_status(f"Canvas undo stopped: {error}")
            return
        finally:
            self._is_restoring_canvas_undo = False

        self._canvas_undo_stack.pop()
        if skipped_texture_bindings:
            self.viewer.set_surface_tools_status(
                "Canvas action undone; newer texture bindings were kept or "
                "some Atlas placements could not be restored."
            )
        else:
            self.viewer.set_surface_tools_status("Canvas action undone.")

    def _restore_surface_texture_tiling_undo_state(
        self,
        state: _SurfaceTextureTilingUndoState,
    ) -> None:
        """Restore one Surface tiling revision and any changed Atlas paths."""

        revision = state.revision
        source_id = build_atlas_wall_texture_source_id(
            revision.previous_assignment.assignment_id
        )
        previous_was_activated = False
        try:
            previous_was_activated = (
                self.surface_texture_generation
                .activate_assignment_tiling_revision(
                    revision,
                    repaired=False,
                    emit_signals=False,
                )
            )
            previous_assignment = self.surface_texture_generation.get_assignment(
                revision.previous_assignment.assignment_id
            )
            if previous_assignment != revision.previous_assignment:
                raise RuntimeError(
                    "The original Surface texture revision is unavailable."
                )
            if revision.created_asset_paths:
                previous_sources = (
                    self._build_atlas_wall_texture_sources_for_assignment(
                        previous_assignment,
                        source_id,
                    )
                )
                if not self.texture_atlas_workspace.transition_object_packing(
                    source_id,
                    previous_sources,
                    commit_callback=lambda: (
                        self.surface_texture_generation.get_assignment(
                            previous_assignment.assignment_id
                        )
                        == previous_assignment
                    ),
                ):
                    raise RuntimeError(
                        self.texture_atlas_workspace.status_label.text().strip()
                        or "The Atlas could not restore the original texture."
                    )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            if previous_was_activated:
                try:
                    self.surface_texture_generation.activate_assignment_tiling_revision(
                        revision,
                        repaired=True,
                        emit_signals=False,
                    )
                except (OSError, RuntimeError, TypeError, ValueError):
                    pass
            raise RuntimeError(
                str(error) or "The original Surface texture could not be restored."
            ) from error

        self._atlas_generation_signature = None
        self.surface_texture_generation.publish_assignment_tiling_change(
            "Original Surface texture tiling restored."
        )
        cleanup_failure_count = (
            self.surface_texture_generation.discard_assignment_tiling_revision(
                revision
            )
        )
        if cleanup_failure_count:
            self.texture_atlas_workspace.status_label.setText(
                "Original texture restored, but some unused repaired files "
                "could not be removed."
            )

    def _restore_canvas_surface_edit_undo_state(
        self,
        state: _CanvasSurfaceEditUndoState,
    ) -> None:
        """Restore wall, floor, or ceiling values from immutable gizmo targets."""

        if not state.targets:
            raise ValueError("Canvas surface undo has no structural baseline.")
        applied_edits = self._restore_canvas_surface_edit_targets(state.targets)
        self._sync_live_canvas_surface_edits(applied_edits)
        self._canvas_surface_edit_targets_by_key = {}
        self.viewer.set_canvas_surface_edit_targets(())
        self.viewer.clear_canvas_surface_edit_pending_outline()
        self._reconcile_canvas_surface_edit_and_refresh()

    def _restore_canvas_level_properties_undo_state(
        self,
        state: _CanvasLevelPropertiesUndoState,
    ) -> None:
        """Restore one direct Canvas level-control change."""

        self._cancel_pending_level_transform(sync_controls=False)
        level = self._get_level_by_index(state.level_index)
        if level is None:
            raise ValueError("The Canvas level in this undo step no longer exists.")
        level.height_meters = state.height_meters
        level.scale = state.scale
        level.canvas_level_scale = state.canvas_level_scale
        level.canvas_offset_x_pixels = state.canvas_offset_x_pixels
        level.canvas_offset_y_pixels = state.canvas_offset_y_pixels
        level.offset_x_meters = state.offset_x_meters
        level.offset_y_meters = state.offset_y_meters
        level.include_in_export = state.include_in_export
        if level is self.current_level:
            self._sync_level_controls()
            self.canvas.set_canvas_level_scale(level.canvas_level_scale)
            self.canvas.set_canvas_level_offsets(
                level.canvas_offset_x_pixels,
                level.canvas_offset_y_pixels,
            )
            self.canvas.update()
        self._sync_canvas_wall_mirror_state()
        self._reconcile_surface_assignments_with_scene()
        self._refresh_scene_atlas_texture_requirements()
        self._schedule_viewer_preview_refresh(preserve_camera=True)

    def _restore_canvas_wall_mirror_undo_state(
        self,
        state: _CanvasWallMirrorUndoState,
    ) -> None:
        """Restore every level touched by one atomic wall-mirror action."""

        restored_level_indices: set[int] = set()
        for level_index, vertex_data in state.vertex_data_by_level:
            level = self._get_level_by_index(level_index)
            if level is None:
                raise ValueError(
                    "A wall-mirror level in this undo step no longer exists."
                )
            level.vertex_data.copy_from(vertex_data)
            restored_level_indices.add(level_index)
        for level_index, doorways in state.doorways_by_level:
            level = self._get_level_by_index(level_index)
            if level is None:
                raise ValueError(
                    "A wall-mirror doorway level in this undo step no longer exists."
                )
            level.doorways[:] = copy.deepcopy(doorways)
            self._viewer_doorways_by_level_index[level_index] = (
                self._copy_doorways(level.doorways)
            )

        self.wall_mirror_links = state.wall_mirror_links
        if self.current_level.index == state.selected_level_index:
            self.canvas.set_selected_vertex_ids(state.selected_vertex_ids)
        self._sync_canvas_wall_mirror_state()
        self._update_wall_mirror_button_state()
        if restored_level_indices:
            self._reconcile_canvas_surface_edit_and_refresh()

    def _restore_canvas_opening_edit_undo_state(
        self,
        state: _CanvasOpeningEditUndoState,
    ) -> None:
        """Restore one doorway or window edit and resume its mesh debounce."""

        target = self._canvas_opening_targets_by_key.get(state.start_edit.reference.key)
        if target is None:
            raise ValueError("The Canvas opening in this undo step no longer exists.")
        applied = apply_canvas_opening_edit(
            self.levels,
            target,
            state.start_edit,
        )
        restored_target = target.with_bounds(state.start_edit.bounds)
        self._canvas_opening_targets_by_key[target.key] = restored_target
        self._sync_live_canvas_opening(applied.reference, applied.level)
        self._refresh_pending_canvas_opening_state(applied.reference)
        if (
            self._pending_doorway_mesh_level_index is not None
            or self._pending_window_mesh_level_index is not None
        ):
            self._doorway_mesh_update_timer.start()

    def _restore_canvas_window_addition_undo_state(
        self,
        state: _CanvasWindowAdditionUndoState,
    ) -> None:
        """Remove one added window through its existing transactional undo."""

        if (
            not self._canvas_window_undo_ids
            or self._canvas_window_undo_ids[-1] != state.window_id
        ):
            raise ValueError("The added Canvas window can no longer be undone.")
        previous_count = len(self._canvas_window_undo_ids)
        self._handle_canvas_window_undo_requested()
        if len(self._canvas_window_undo_ids) != previous_count - 1:
            raise RuntimeError("The added Canvas window could not be undone.")

    def _restore_canvas_placed_object_undo_state(
        self,
        state: _CanvasPlacedObjectUndoState,
    ) -> int:
        """Restore one object's prior placement and eligible Atlas bindings."""

        restored = self.generation.restore_placeable_object_placement(
            state.object_id,
            state.placement,
            emit_change_signals=False,
        )
        if not restored:
            raise ValueError("The placed object in this undo step no longer exists.")
        if state.restore_atlas_bindings and state.placement is None:
            self.texture_atlas_workspace.remove_scene_texture_from_atlases(
                state.object_id
            )
        skipped_atlas_placements = (
            self._restore_canvas_atlas_placements(state.atlas_placements)
            if state.restore_atlas_bindings
            else 0
        )
        self._atlas_generation_signature = None
        self._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )
        if state.restore_atlas_bindings and state.atlas_placements:
            self.texture_atlas_workspace.refresh_texture_source_content(
                tuple(
                    dict.fromkeys(
                        placement.object_id
                        for _atlas_id, placement in state.atlas_placements
                    )
                )
            )
        if state.selected_object_ids is None:
            selected_object_ids = (
                (state.object_id,) if state.placement is not None else ()
            )
            active_object_id = state.object_id if selected_object_ids else None
        else:
            selected_object_ids = tuple(
                dict.fromkeys(
                    object_id
                    for object_id in state.selected_object_ids
                    if self.generation.get_generated_object_placement(object_id)
                    is not None
                )
            )
            active_object_id = (
                state.active_object_id
                if state.active_object_id in selected_object_ids
                else (
                    selected_object_ids[-1] if selected_object_ids else None
                )
            )
        self._desired_canvas_object_ids = selected_object_ids
        self._desired_canvas_object_id = active_object_id
        self._desired_canvas_surface_ids = ()
        self._schedule_viewer_preview_refresh(preserve_camera=True)
        return skipped_atlas_placements

    def _restore_canvas_stairs_undo_state(
        self,
        state: _CanvasStairsUndoState,
    ) -> int:
        """Restore one stair transaction and unchanged texture bindings."""

        self.stairs = list(state.stairs)
        self.canvas.set_stair_context(self.stairs, self.current_level)
        self._staged_stair = None
        self.viewer.clear_canvas_stair_preview()
        self._sync_canvas_stair_semantic_targets(self.levels)

        expected_by_id = {
            assignment.assignment_id: assignment
            for assignment in state.assignment_targets_after
        }
        current_by_id = {
            assignment.assignment_id: assignment
            for assignment in self.surface_texture_generation.snapshot_assignments()
        }
        restorable_assignments = tuple(
            assignment
            for assignment in state.assignments
            if (
                assignment.assignment_id in expected_by_id
                and _surface_assignment_target_signature(
                    current_by_id.get(assignment.assignment_id)
                )
                == _surface_assignment_target_signature(
                    expected_by_id[assignment.assignment_id]
                )
            )
        )
        self.surface_texture_generation.restore_assignment_target_snapshot(
            restorable_assignments,
            emit_signals=False,
        )
        self.surface_texture_generation.reconcile_assignments_with_levels(
            self.levels,
            emit_signals=False,
        )
        restorable_assignment_ids = {
            assignment.assignment_id for assignment in restorable_assignments
        }
        restorable_source_ids = {
            build_atlas_wall_texture_source_id(assignment_id)
            for assignment_id in restorable_assignment_ids
        }
        restorable_atlas_placements = tuple(
            (atlas_id, placement)
            for atlas_id, placement in state.atlas_placements
            if placement.object_id in restorable_source_ids
        )

        self._atlas_generation_signature = None
        self._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )
        skipped_atlas_placements = self._restore_canvas_atlas_placements(
            restorable_atlas_placements
        )
        self.texture_atlas_workspace.refresh_texture_source_content(
            tuple(
                dict.fromkeys(
                    placement.object_id
                    for _atlas_id, placement in restorable_atlas_placements
                )
            )
        )

        self._desired_canvas_stair_part_ids = tuple(
            semantic_id
            for semantic_id in state.selected_stair_part_ids
            if semantic_id in self._canvas_stair_part_targets_by_id
        )
        assignable_surface_ids = (
            self._canvas_surface_targets_by_id.keys()
            | self._canvas_stair_semantic_surfaces_by_id.keys()
        )
        self._atlas_surface_assignment_target_ids = tuple(
            surface_id
            for surface_id in state.assignment_target_ids
            if surface_id in assignable_surface_ids
        )
        self._editing_stair_index = next(
            (
                stair_index
                for stair_index, stair in enumerate(self.stairs)
                if stair.stair_id == state.editing_stair_id
            ),
            None,
        )
        if (
            self._editing_stair_index is not None
            and 0 <= self._editing_stair_index < len(self.stairs)
        ):
            self._load_stair_editor_from_stair(
                self.stairs[self._editing_stair_index]
            )
        else:
            self._editing_stair_index = None
            self._desired_canvas_stair_part_ids = ()
            self._set_stair_editor_parameters(self._new_stair_parameters)
            self._sync_stair_calculated_values(None)
        self._sync_surface_generation_selection(
            self._desired_canvas_stair_part_ids
        )
        self._update_stair_button_state()
        self._schedule_viewer_preview_refresh(preserve_camera=True)
        return (
            len(state.assignments)
            - len(restorable_assignments)
            + skipped_atlas_placements
        )

    def _restore_canvas_plan_image_undo_state(
        self,
        state: _CanvasPlanImageUndoState,
    ) -> None:
        """Restore the immutable plan image preceding one marquee erase."""

        level = self._get_level_by_index(state.level_index)
        if level is None:
            raise ValueError("The plan-image level in this undo step no longer exists.")
        commit = state.commit
        current_path = (
            None
            if level.image_path is None
            else str(Path(level.image_path).resolve())
        )
        replacement_path = str(Path(commit.replacement_path).resolve())
        previous_path = str(Path(commit.previous_path).resolve())
        if current_path != replacement_path:
            raise RuntimeError(
                "The plan image changed after this erase; it was not overwritten."
            )
        if (
            _build_local_file_revision(replacement_path)
            != commit.replacement_revision
        ):
            raise RuntimeError(
                "The erased plan-image file changed outside HouseMaker."
            )
        previous_revision = _build_local_file_revision(previous_path)
        if (
            not _local_file_revision_has_file(previous_revision)
            or previous_revision != commit.previous_revision
        ):
            raise RuntimeError(
                "The previous plan-image revision is no longer available."
            )

        if (
            level is self.current_level
            and not self.canvas.load_blueprint_image_preserving_view(previous_path)
        ):
            raise RuntimeError("The previous plan image could not be restored.")
        level.image_path = previous_path
        level.image_size_pixels = tuple(commit.image_size_pixels)
        self._level_blueprint_image_revisions[level.index] = previous_revision
        if level is self.current_level:
            self._clear_plan_wall_preview(update_controls=False)
            self._update_blueprint_name_label()
            self._update_plan_wall_generation_controls_state()
            self._update_image_correction_button_state()
            if self.canvas.is_plan_image_erasing():
                self.canvas.set_plan_image_erase_output_path(
                    self._new_plan_image_erase_output_path(level)
                )

    def _restore_blueprint_undo_state(
        self,
        state: _CanvasBlueprintUndoState,
    ) -> int:
        """Restore one 2D snapshot plus conservatively affected bindings."""

        level = self._get_level_by_index(state.level_index)
        if level is None:
            raise ValueError("The Canvas level in this undo step no longer exists.")
        if state.level_offsets_meters is not None:
            (
                level.offset_x_meters,
                level.offset_y_meters,
            ) = state.level_offsets_meters
        if level is self.current_level:
            self.canvas.discard_undo_snapshot(state.snapshot)
            self.canvas.restore_snapshot(state.snapshot)
            if state.level_offsets_meters is not None:
                self._sync_level_controls()
        else:
            level.vertex_data.copy_from(state.snapshot.vertex_data)
            level.rooms.clear()
            level.rooms.extend(copy.deepcopy(state.snapshot.rooms))
            level.doorways.clear()
            level.doorways.extend(copy.deepcopy(state.snapshot.doorways))
            level.open_spaces.clear()
            level.open_spaces.extend(state.snapshot.open_spaces)
            self._reset_viewer_doorway_snapshots()
            self._reconcile_surface_assignments_with_scene()
            self._schedule_viewer_preview_refresh(preserve_camera=True)
        for other_level_index, vertex_data in state.other_level_vertex_data:
            other_level = self._get_level_by_index(other_level_index)
            if other_level is None:
                raise ValueError(
                    "A wall-mirror level in this undo step no longer exists."
                )
            other_level.vertex_data.copy_from(vertex_data)
        other_doorways_changed = False
        for other_level_index, doorways in state.other_level_doorways:
            other_level = self._get_level_by_index(other_level_index)
            if other_level is None:
                raise ValueError(
                    "A wall-mirror doorway level in this undo step no longer exists."
                )
            if self._copy_doorways(other_level.doorways) != doorways:
                other_level.doorways[:] = copy.deepcopy(doorways)
                self._viewer_doorways_by_level_index[other_level_index] = (
                    self._copy_doorways(other_level.doorways)
                )
                other_doorways_changed = True
        self.wall_mirror_links = state.wall_mirror_links
        self._sync_canvas_wall_mirror_state()
        if other_doorways_changed:
            self._schedule_viewer_preview_refresh(preserve_camera=True)
        return self._restore_blueprint_surface_bindings(state)

    def _restore_blueprint_surface_bindings(
        self,
        state: _CanvasBlueprintUndoState,
    ) -> int:
        """Restore hole-affected targets without replacing newer texture data."""

        expected_after = state.assignment_targets_after
        if expected_after is None:
            return len(state.assignments)
        if not state.assignments:
            return 0
        expected_by_id = {
            assignment.assignment_id: assignment for assignment in expected_after
        }
        current_by_id = {
            assignment.assignment_id: assignment
            for assignment in self.surface_texture_generation.snapshot_assignments()
        }
        restorable_assignments = tuple(
            assignment
            for assignment in state.assignments
            if (
                assignment.assignment_id in expected_by_id
                and _surface_assignment_target_signature(
                    current_by_id.get(assignment.assignment_id)
                )
                == _surface_assignment_target_signature(
                    expected_by_id[assignment.assignment_id]
                )
            )
        )
        self.surface_texture_generation.restore_assignment_target_snapshot(
            restorable_assignments,
            emit_signals=False,
        )
        restorable_assignment_ids = {
            assignment.assignment_id for assignment in restorable_assignments
        }
        restorable_source_ids = {
            build_atlas_wall_texture_source_id(assignment_id)
            for assignment_id in restorable_assignment_ids
        }
        restorable_atlas_placements = tuple(
            (atlas_id, placement)
            for atlas_id, placement in state.atlas_placements
            if placement.object_id in restorable_source_ids
        )

        self._atlas_generation_signature = None
        self._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )
        restored_assignments_by_id = {
            assignment.assignment_id: assignment
            for assignment in self.surface_texture_generation.snapshot_assignments()
            if assignment.assignment_id in restorable_assignment_ids
        }
        live_sources: dict[str, AtlasObjectTextureSource] = {}
        for assignment_id, assignment in restored_assignments_by_id.items():
            source = self._build_atlas_wall_texture_source(assignment)
            if source is not None:
                live_sources[build_atlas_wall_texture_source_id(assignment_id)] = source
        unresolved_placement_source_ids = {
            placement.object_id
            for _atlas_id, placement in restorable_atlas_placements
            if placement.object_id not in live_sources
        }
        live_atlas_placements = tuple(
            (atlas_id, placement)
            for atlas_id, placement in restorable_atlas_placements
            if placement.object_id in live_sources
        )
        skipped_atlas_placements = self._restore_canvas_atlas_placements(
            live_atlas_placements,
            source_overrides=live_sources,
        )
        affected_source_ids = tuple(
            dict.fromkeys(
                placement.object_id for _atlas_id, placement in live_atlas_placements
            )
        )
        self.texture_atlas_workspace.refresh_texture_source_content(affected_source_ids)
        self._sync_canvas_surface_drawing_overlay()
        self._schedule_viewer_preview_refresh(preserve_camera=True)
        return (
            len(state.assignments)
            - len(restorable_assignments)
            + skipped_atlas_placements
            + len(unresolved_placement_source_ids)
        )

    def _restore_canvas_topology_undo_state(
        self,
        state: _CanvasTopologyUndoState,
    ) -> int:
        """Restore one Canvas topology transaction and its texture bindings."""

        levels_by_index = {level.index: level for level in self.levels}
        restoration_targets: list[
            tuple[LevelData, tuple[EditableSurfaceMeshData, ...]]
        ] = []
        for level_index, editable_surfaces in state.editable_surfaces_by_level:
            level = levels_by_index.get(level_index)
            if level is None:
                raise ValueError(
                    "The Canvas level in this undo step no longer exists."
                )
            restoration_targets.append((level, editable_surfaces))
        for level, editable_surfaces in restoration_targets:
            level.editable_surfaces = list(editable_surfaces)
        for level_index, flipped_surface_ids in state.flipped_surface_ids_by_level:
            level = levels_by_index.get(level_index)
            if level is None:
                raise ValueError(
                    "The Canvas level in this undo step no longer exists."
                )
            level.flipped_surface_ids = set(flipped_surface_ids)
        self._desired_canvas_surface_ids = state.selected_surface_ids
        self._atlas_surface_assignment_target_ids = state.assignment_target_ids
        self._desired_canvas_object_id = state.selected_object_id
        self._desired_canvas_object_ids = (
            state.selected_object_ids
            or (
                (state.selected_object_id,)
                if state.selected_object_id is not None
                else ()
            )
        )
        self._active_canvas_surface_drawing_vertex_id = state.active_vertex_id
        expected_by_id = {
            assignment.assignment_id: assignment
            for assignment in state.assignment_targets_after
        }
        current_by_id = {
            assignment.assignment_id: assignment
            for assignment in self.surface_texture_generation.snapshot_assignments()
        }
        restorable_assignments = tuple(
            assignment
            for assignment in state.assignments
            if (
                assignment.assignment_id in expected_by_id
                and _surface_assignment_target_signature(
                    current_by_id.get(assignment.assignment_id)
                )
                == _surface_assignment_target_signature(
                    expected_by_id[assignment.assignment_id]
                )
            )
        )
        restorable_assignment_ids = {
            assignment.assignment_id for assignment in restorable_assignments
        }
        self.surface_texture_generation.restore_assignment_target_snapshot(
            restorable_assignments,
            emit_signals=False,
        )
        restorable_source_ids = {
            build_atlas_wall_texture_source_id(assignment_id)
            for assignment_id in restorable_assignment_ids
        }
        restorable_atlas_placements = tuple(
            (atlas_id, placement)
            for atlas_id, placement in state.atlas_placements
            if placement.object_id in restorable_source_ids
        )
        skipped_atlas_placements = self._restore_canvas_atlas_placements(
            restorable_atlas_placements
        )
        self._atlas_generation_signature = None
        self._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )
        affected_source_ids = tuple(
            dict.fromkeys(
                placement.object_id
                for _atlas_id, placement in restorable_atlas_placements
            )
        )
        self.texture_atlas_workspace.refresh_texture_source_content(affected_source_ids)
        self._sync_canvas_surface_drawing_overlay()
        self._schedule_viewer_preview_refresh(preserve_camera=True)
        return (
            len(state.assignments)
            - len(restorable_assignments)
            + skipped_atlas_placements
        )

    def _restore_canvas_atlas_placements(
        self,
        placements: Sequence[tuple[str, TextureAtlasPlacement]],
        *,
        source_overrides: dict[str, AtlasObjectTextureSource] | None = None,
    ) -> int:
        """Restore each source once without overwriting newer Atlas work."""

        live_sources = source_overrides or {}
        atlas_data = self.texture_atlas_workspace.get_data()
        requested_by_source_id: dict[
            str,
            tuple[str, TextureAtlasPlacement],
        ] = {}
        for atlas_id, placement in placements:
            requested_by_source_id.setdefault(
                placement.object_id,
                (atlas_id, placement),
            )
        current_by_source_id = {
            placement.object_id: (atlas.atlas_id, placement)
            for atlas in atlas_data.atlases
            for placement in atlas.placements
        }
        changed = False
        skipped = 0
        for source_id, (atlas_id, placement) in requested_by_source_id.items():
            current = current_by_source_id.get(source_id)
            live_source = live_sources.get(source_id)
            if current is not None:
                if current[0] != atlas_id:
                    skipped += 1
                    continue
                if live_source is None:
                    if current != (atlas_id, placement):
                        skipped += 1
                    continue
                try:
                    restored_placement = atlas_data.assign_object(
                        atlas_id,
                        live_source.object_id,
                        live_source.texture_path,
                        live_source.texture_resolution,
                        live_source.packing_mode,
                    )
                except (OSError, TypeError, ValueError):
                    skipped += 1
                    continue
                current_by_source_id[source_id] = (
                    atlas_id,
                    restored_placement,
                )
                changed = bool(restored_placement != current[1] or changed)
                continue
            atlas = atlas_data.atlas_by_id(atlas_id)
            if atlas is None:
                skipped += 1
                continue
            try:
                restored_placement = atlas_data.assign_object(
                    atlas_id,
                    (
                        placement.object_id
                        if live_source is None
                        else live_source.object_id
                    ),
                    (
                        placement.texture_path
                        if live_source is None
                        else live_source.texture_path
                    ),
                    (
                        placement.texture_resolution
                        if live_source is None
                        else live_source.texture_resolution
                    ),
                    (
                        placement.packing_mode
                        if live_source is None
                        else live_source.packing_mode
                    ),
                )
            except (OSError, TypeError, ValueError):
                skipped += 1
                continue
            current_by_source_id[source_id] = (
                atlas_id,
                restored_placement,
            )
            changed = True
        if changed:
            self.texture_atlas_workspace.set_data(atlas_data)
        return skipped

    def _sync_canvas_surface_drawing_overlay(self) -> None:
        """Keep persistent drawn vertices and edges aligned with preview levels."""

        overlay = build_surface_drawing_overlay(self._build_viewer_preview_levels())
        known_vertex_ids = {vertex.vertex_id for vertex in overlay.vertices}
        if self._active_canvas_surface_drawing_vertex_id not in known_vertex_ids:
            self._active_canvas_surface_drawing_vertex_id = None
        self.viewer.set_canvas_surface_drawing_overlay(
            overlay,
            active_vertex_id=(self._active_canvas_surface_drawing_vertex_id),
        )

    def _handle_canvas_surface_vertex_chain_reset_requested(self) -> None:
        """Forget the transient edge-chain endpoint when drawing is cancelled."""

        self._active_canvas_surface_drawing_vertex_id = None
        self._sync_canvas_surface_drawing_overlay()

    def _handle_canvas_surface_vertex_insertion_requested(
        self,
        raw_request: object,
    ) -> None:
        """Insert one persistent vertex and preserve any parent assignment."""

        if not isinstance(raw_request, SurfaceVertexInsertionRequest):
            return
        self._apply_canvas_surface_topology_edit(
            lambda: place_surface_vertex(
                self.levels,
                raw_request.surface_id,
                raw_request.world_point,
                raw_request.active_vertex_id,
            ),
            success_message=(
                "Surface drawing updated. Edge-to-edge paths split the "
                "surface; nearby 45-degree alignments snap automatically."
            ),
        )

    def _handle_canvas_surface_face_extrusion_requested(
        self,
        raw_request: object,
    ) -> None:
        """Extrude the selected connected faces after the gizmo is released."""

        if not isinstance(raw_request, SurfaceFaceExtrusionRequest):
            return
        self._apply_canvas_surface_topology_edit(
            lambda: extrude_surface_faces(
                self.levels,
                raw_request.surface_ids,
                raw_request.delta_meters,
            ),
            success_message=(
                "Face extrusion updated. The cap remains selected; any side "
                "faces can be selected and textured separately."
            ),
        )

    def _handle_canvas_surface_face_deletion_requested(
        self,
        raw_surface_ids: object,
    ) -> None:
        """Delete selected Add vertices faces through the topology transaction."""

        try:
            surface_ids = tuple(
                dict.fromkeys(str(value) for value in raw_surface_ids)  # type: ignore[arg-type]
            )
        except TypeError:
            return
        if not surface_ids:
            return
        deleted = self._apply_canvas_surface_topology_edit(
            lambda: delete_directly_drawn_surface_faces(
                self.levels,
                surface_ids,
            ),
            success_message=("Selected face deleted. Press Ctrl+Z to restore it."),
        )
        if not deleted:
            return
        self._is_syncing_canvas_scene_selection = True
        try:
            self.viewer.set_selected_canvas_surface_ids(())
        finally:
            self._is_syncing_canvas_scene_selection = False

    def _apply_canvas_surface_topology_edit(
        self,
        operation: Callable[[], SurfaceTopologyEditResult],
        *,
        success_message: str,
    ) -> bool:
        """Apply geometry and texture-lineage changes as one UI transaction."""

        self._commit_pending_canvas_surface_mesh_update()
        self._commit_pending_wall_vertex_update()
        self._commit_pending_doorway_mesh_update()
        undo_state = BlueprintWorkspace._capture_canvas_topology_undo_state(self)
        previous_edits = [
            (level, copy.deepcopy(level.editable_surfaces)) for level in self.levels
        ]
        previous_flipped_surface_ids = [
            (level, set(level.flipped_surface_ids)) for level in self.levels
        ]
        previous_surface_ids = self._desired_canvas_surface_ids
        previous_assignment_target_ids = self._atlas_surface_assignment_target_ids
        previous_object_id = self._desired_canvas_object_id
        previous_object_ids = getattr(
            self,
            "_desired_canvas_object_ids",
            ((previous_object_id,) if previous_object_id is not None else ()),
        )
        previous_active_vertex_id = self._active_canvas_surface_drawing_vertex_id
        result: SurfaceTopologyEditResult | None = None
        try:
            result = operation()
            replacements = result.replacements
            remap_flipped_surface_ids_with_lineage(
                self.levels,
                replacements,
            )
            selected_surface_ids = result.selected_surface_ids
            self._desired_canvas_surface_ids = selected_surface_ids
            self._atlas_surface_assignment_target_ids = selected_surface_ids
            if selected_surface_ids:
                self._desired_canvas_object_id = None
                self._desired_canvas_object_ids = ()
            self._active_canvas_surface_drawing_vertex_id = result.active_vertex_id
            if result.requires_mesh_refresh:
                self.surface_texture_generation.remap_assignments_with_surface_lineage(
                    self.levels,
                    replacements,
                )
        except (RuntimeError, TypeError, ValueError) as error:
            for level, editable_surfaces in previous_edits:
                level.editable_surfaces = editable_surfaces
            for level, flipped_surface_ids in previous_flipped_surface_ids:
                level.flipped_surface_ids = flipped_surface_ids
            self._desired_canvas_surface_ids = previous_surface_ids
            self._atlas_surface_assignment_target_ids = previous_assignment_target_ids
            self._desired_canvas_object_id = previous_object_id
            self._desired_canvas_object_ids = previous_object_ids
            self._active_canvas_surface_drawing_vertex_id = previous_active_vertex_id
            self._sync_canvas_surface_drawing_overlay()
            self.viewer.set_surface_tools_status(f"Surface edit stopped: {error}")
            if result is not None and result.requires_mesh_refresh:
                self._schedule_viewer_preview_refresh(preserve_camera=True)
            return False

        if result.requires_mesh_refresh:
            BlueprintWorkspace._reconcile_surface_assignments_with_scene(self)
        if result.state_changed:
            self._record_canvas_undo_state(
                BlueprintWorkspace._finalize_canvas_topology_undo_state(
                    self,
                    undo_state,
                ),
                commit_pending_surface_edit=False,
            )
        self._sync_canvas_surface_drawing_overlay()
        self.viewer.set_surface_tools_status(success_message)
        if result.requires_mesh_refresh:
            self._schedule_viewer_preview_refresh(preserve_camera=True)
        return True

    # ### Canvas structural surface edits ###
    def _resolve_canvas_surface_edit_targets(
        self,
        primary: CanvasSurfaceEditHandleTarget,
    ) -> tuple[CanvasSurfaceEditHandleTarget, ...]:
        """Resolve every selected wall target controlled by one handle."""

        if primary.reference.kind not in CANVAS_SURFACE_EDIT_WALL_KINDS:
            return (primary,)
        selected_wall_ids = set(
            self._get_canvas_wall_surface_ids(
                self.viewer.get_selected_canvas_surface_ids()
            )
        )
        selected_wall_ids.add(primary.surface_id)
        if primary.reference.kind == CANVAS_SURFACE_EDIT_WALL_TRANSLATION:
            related = tuple(
                target
                for target in self._canvas_surface_edit_targets_by_key.values()
                if (
                    target.surface_id in selected_wall_ids
                    and target.reference.kind == CANVAS_SURFACE_EDIT_WALL_TRANSLATION
                    and target.reference.axis_index == primary.reference.axis_index
                )
            )
        else:
            related = tuple(
                target
                for target in self._canvas_surface_edit_targets_by_key.values()
                if (
                    target.surface_id in selected_wall_ids
                    and target.reference == primary.reference
                )
            )
        return tuple(dict.fromkeys((primary, *related)))

    def _get_selected_canvas_wall_baselines(
        self,
        primary: CanvasSurfaceEditHandleTarget,
    ) -> tuple[CanvasSurfaceEditHandleTarget, ...]:
        """Snapshot one complete-chain baseline for each selected wall."""

        selected_wall_ids = self._get_canvas_wall_surface_ids(
            self.viewer.get_selected_canvas_surface_ids()
        )
        if primary.surface_id not in selected_wall_ids:
            selected_wall_ids = (*selected_wall_ids, primary.surface_id)
        baselines: list[CanvasSurfaceEditHandleTarget] = []
        for surface_id in selected_wall_ids:
            candidate = next(
                (
                    target
                    for target in self._canvas_surface_edit_targets_by_key.values()
                    if (
                        target.surface_id == surface_id
                        and target.reference.kind
                        == CANVAS_SURFACE_EDIT_WALL_TRANSLATION
                        and target.reference.axis_index == primary.reference.axis_index
                    )
                ),
                None,
            )
            if candidate is not None:
                baselines.append(candidate)
        return tuple(baselines) or (primary,)

    def _restore_canvas_surface_edit_targets(
        self,
        targets: tuple[CanvasSurfaceEditHandleTarget, ...],
    ) -> tuple[AppliedCanvasSurfaceEdit, ...]:
        """Restore one structural target or one atomic wall-target group."""

        if not targets:
            return ()
        if len(targets) == 1:
            return (
                restore_canvas_surface_edit(
                    self.levels,
                    targets[0],
                    validate_project_geometry=False,
                ),
            )
        return restore_canvas_wall_edit_batch(
            self.levels,
            targets,
            validate_project_geometry=False,
        )

    def _sync_live_canvas_surface_edits(
        self,
        applied_edits: Sequence[AppliedCanvasSurfaceEdit],
    ) -> None:
        """Repaint each affected level once after an atomic wall edit."""

        synced_level_ids: set[int] = set()
        for applied in applied_edits:
            level_identity = id(applied.level)
            if level_identity in synced_level_ids:
                continue
            synced_level_ids.add(level_identity)
            self._sync_live_canvas_surface_edit(applied)
        if synced_level_ids:
            self._sync_canvas_wall_mirror_state()

    def _handle_canvas_surface_edit_started(self, raw_edit: object) -> None:
        """Remember the immutable baseline for one live structural drag."""

        if self._pending_wall_vertex_mesh_update:
            self._commit_pending_wall_vertex_update()
            self.viewer.cancel_canvas_surface_edit()
            return
        if not isinstance(raw_edit, CanvasSurfaceEdit):
            return
        target = self._canvas_surface_edit_targets_by_key.get(
            (raw_edit.surface_id, raw_edit.reference.key)
        )
        if (
            target is None
            or target.reference != raw_edit.reference
            or target.surface_id != raw_edit.surface_id
        ):
            return

        previous_target = self._active_canvas_surface_edit_target
        if previous_target is not None and previous_target != target:
            try:
                applied_edits = self._restore_canvas_surface_edit_targets(
                    self._active_canvas_surface_edit_targets or (previous_target,)
                )
            except (TypeError, ValueError):
                pass
            else:
                self._sync_live_canvas_surface_edits(applied_edits)
        if target.reference.kind in DELAYED_CANVAS_SURFACE_EDIT_KINDS:
            self._canvas_surface_mesh_update_timer.stop()
        self._active_canvas_surface_edit_target = target
        self._active_canvas_surface_edit_targets = (
            self._resolve_canvas_surface_edit_targets(target)
        )

    def _handle_canvas_surface_edit_preview_changed(
        self,
        raw_edit: object,
    ) -> None:
        """Apply absolute drag deltas without rebuilding the complete 3D mesh."""

        self._apply_active_canvas_surface_edit(
            raw_edit,
            validate_project_geometry=False,
        )

    def _handle_canvas_surface_edit_finished(
        self,
        raw_edit: object,
        changed: bool,
    ) -> None:
        """Commit one valid drag and schedule its one required mesh rebuild."""

        target = self._active_canvas_surface_edit_target
        if target is None or not isinstance(raw_edit, CanvasSurfaceEdit):
            return
        active_targets = self._active_canvas_surface_edit_targets or (target,)
        is_floor_edit = target.reference.kind == CANVAS_SURFACE_EDIT_FLOOR_THICKNESS
        uses_mesh_delay = target.reference.kind in DELAYED_CANVAS_SURFACE_EDIT_KINDS
        if changed and not self._apply_active_canvas_surface_edit(
            raw_edit,
            validate_project_geometry=not uses_mesh_delay,
        ):
            return
        if not changed:
            try:
                applied_edits = self._restore_canvas_surface_edit_targets(
                    active_targets
                )
            except (TypeError, ValueError):
                pass
            else:
                self._sync_live_canvas_surface_edits(applied_edits)

        self._active_canvas_surface_edit_target = None
        self._active_canvas_surface_edit_targets = ()
        if not changed:
            if self._pending_canvas_surface_mesh_update and uses_mesh_delay:
                self._canvas_surface_mesh_update_timer.start()
            else:
                self._queue_viewer_preview_refresh()
            return

        if uses_mesh_delay:
            if self._pending_canvas_surface_mesh_baseline is None:
                self._pending_canvas_surface_mesh_baseline = target
                self._pending_canvas_surface_mesh_baselines = (
                    (target,)
                    if is_floor_edit
                    else self._get_selected_canvas_wall_baselines(target)
                )
                self._pending_canvas_wall_surface_ids = (
                    ()
                    if is_floor_edit
                    else self._get_canvas_wall_surface_ids(
                        self.viewer.get_selected_canvas_surface_ids()
                    )
                )
            self._pending_canvas_surface_mesh_update = True
            if is_floor_edit:
                self._pending_floor_thickness_level_index = target.reference.level_index
                current_targets = (target,)
            else:
                selected_wall_ids = set(
                    self._get_canvas_wall_surface_ids(
                        self.viewer.get_selected_canvas_surface_ids()
                    )
                )
                selected_wall_ids.add(target.surface_id)
                current_targets = tuple(
                    candidate
                    for candidate in (self._canvas_surface_edit_targets_by_key.values())
                    if candidate.surface_id in selected_wall_ids
                    and candidate.reference.kind in CANVAS_SURFACE_EDIT_WALL_KINDS
                )
            try:
                rebased_targets = (
                    (
                        rebase_canvas_floor_edit_target(
                            self.levels,
                            target,
                        ),
                    )
                    if is_floor_edit
                    else rebase_canvas_wall_edit_targets_batch(
                        self.levels,
                        current_targets,
                    )
                )
            except (TypeError, ValueError) as error:
                self._reject_pending_canvas_surface_mesh_update(error)
                return
            self._canvas_surface_edit_targets_by_key = {
                (candidate.surface_id, candidate.reference.key): candidate
                for candidate in rebased_targets
            }
            self.viewer.set_canvas_surface_edit_targets(
                rebased_targets,
                preserve_pending_outline=True,
            )
            self._canvas_surface_mesh_update_timer.start()
            return

        # Height edits rebuild immediately, so their old immutable baselines
        # are retired until the refreshed scene installs new ones.
        self._record_canvas_undo_state(
            _CanvasSurfaceEditUndoState(targets=tuple(active_targets))
        )
        self._canvas_surface_edit_targets_by_key = {}
        self.viewer.set_canvas_surface_edit_targets(())
        self._reconcile_canvas_surface_edit_and_refresh()

    def _reconcile_canvas_surface_edit_and_refresh(self) -> None:
        """Reconcile semantic assignments before one structural mesh rebuild."""

        assignments_changed = (
            self._reconcile_surface_assignments_with_scene()
        )
        self._finalize_blueprint_surface_binding_undo_state()
        if not assignments_changed:
            self._schedule_viewer_preview_refresh(preserve_camera=True)

    def _commit_pending_canvas_surface_mesh_update(self) -> None:
        """Release one validated surface edit after its quiet period."""

        self._canvas_surface_mesh_update_timer.stop()
        if not self._pending_canvas_surface_mesh_update:
            return
        baseline = self._pending_canvas_surface_mesh_baseline
        baselines = self._pending_canvas_surface_mesh_baselines
        if not baselines and baseline is not None:
            baselines = (baseline,)
        if baselines:
            try:
                if len(baselines) == 1:
                    validate_canvas_surface_edit_geometry(
                        self.levels,
                        baselines[0],
                    )
                else:
                    validate_canvas_wall_edit_batch_geometry(
                        self.levels,
                        baselines,
                    )
            except (TypeError, ValueError) as error:
                self._reject_pending_canvas_surface_mesh_update(error)
                return

        if baselines and not canvas_surface_edit_targets_are_at_baseline(
            self.levels,
            baselines,
        ):
            self._record_canvas_undo_state(
                _CanvasSurfaceEditUndoState(targets=tuple(baselines)),
                commit_pending_surface_edit=False,
            )
        self._commit_viewer_floor_thickness_snapshot()
        self._pending_canvas_surface_mesh_update = False
        self._pending_canvas_surface_mesh_baseline = None
        self._pending_canvas_surface_mesh_baselines = ()
        self._pending_canvas_wall_surface_ids = ()
        self._canvas_surface_edit_targets_by_key = {}
        self.viewer.set_canvas_surface_edit_targets(
            (),
            preserve_pending_outline=True,
        )
        self._reconcile_canvas_surface_edit_and_refresh()

    def _cancel_pending_canvas_surface_mesh_update(self) -> None:
        """Discard delayed surface-mesh work without changing project data."""

        self._canvas_surface_mesh_update_timer.stop()
        self._pending_canvas_surface_mesh_update = False
        self._pending_canvas_surface_mesh_baseline = None
        self._pending_canvas_surface_mesh_baselines = ()
        self._pending_canvas_wall_surface_ids = ()
        self._pending_floor_thickness_level_index = None
        self.viewer.clear_canvas_surface_edit_pending_outline()

    def _undo_pending_canvas_surface_mesh_update(self) -> bool:
        """Restore a delayed structural edit before it enters undo history."""

        if not self._pending_canvas_surface_mesh_update:
            return False
        baseline = self._pending_canvas_surface_mesh_baseline
        baselines = self._pending_canvas_surface_mesh_baselines
        if not baselines and baseline is not None:
            baselines = (baseline,)
        if not baselines:
            return False
        applied_edits = self._restore_canvas_surface_edit_targets(baselines)
        self._cancel_pending_canvas_surface_mesh_update()
        self._canvas_surface_edit_targets_by_key = {}
        self.viewer.set_canvas_surface_edit_targets(())
        self._sync_live_canvas_surface_edits(applied_edits)
        self._reconcile_canvas_surface_edit_and_refresh()
        return True

    def _reject_pending_canvas_surface_mesh_update(
        self,
        error: Exception,
    ) -> None:
        """Roll a rejected adjustment burst back to its first baseline."""

        baseline = self._pending_canvas_surface_mesh_baseline
        baselines = self._pending_canvas_surface_mesh_baselines
        if not baselines and baseline is not None:
            baselines = (baseline,)
        self._canvas_surface_mesh_update_timer.stop()
        self._pending_canvas_surface_mesh_update = False
        self._pending_canvas_surface_mesh_baseline = None
        self._pending_canvas_surface_mesh_baselines = ()
        self._pending_canvas_wall_surface_ids = ()
        self._pending_floor_thickness_level_index = None
        if baselines:
            try:
                restored_edits = self._restore_canvas_surface_edit_targets(baselines)
            except (TypeError, ValueError):
                pass
            else:
                self._sync_live_canvas_surface_edits(restored_edits)
        self.viewer.set_window_tools_status(f"Surface edit stopped: {error}")
        self.viewer.clear_canvas_surface_edit_pending_outline()
        self._restore_canvas_surface_edit_targets_after_rejection()
        self._queue_viewer_preview_refresh()

    def _restore_canvas_surface_edit_targets_after_rejection(self) -> None:
        """Re-arm unchanged viewer handles after delayed validation rolls back."""

        try:
            edit_targets = build_canvas_surface_edit_targets(
                self.levels,
                tuple(self._canvas_surface_targets_by_id.values()),
            )
        except (TypeError, ValueError):
            edit_targets = ()
        self._canvas_surface_edit_targets_by_key = {
            (target.surface_id, target.reference.key): target for target in edit_targets
        }
        self.viewer.set_canvas_surface_edit_targets(edit_targets)

    def _handle_canvas_surface_edit_cancelled(self, raw_edit: object) -> None:
        """Restore the exact structural baseline after Escape or navigation."""

        target = self._active_canvas_surface_edit_target
        if target is None:
            return
        active_targets = self._active_canvas_surface_edit_targets or (target,)
        if isinstance(raw_edit, CanvasSurfaceEdit) and (
            raw_edit.reference != target.reference
            or raw_edit.surface_id != target.surface_id
        ):
            return
        try:
            applied_edits = self._restore_canvas_surface_edit_targets(active_targets)
        except (TypeError, ValueError):
            pass
        else:
            self._sync_live_canvas_surface_edits(applied_edits)
        self._active_canvas_surface_edit_target = None
        self._active_canvas_surface_edit_targets = ()
        if (
            self._pending_canvas_surface_mesh_update
            and target.reference.kind in DELAYED_CANVAS_SURFACE_EDIT_KINDS
        ):
            self._canvas_surface_mesh_update_timer.start()
        else:
            self._queue_viewer_preview_refresh()

    def _cancel_active_canvas_surface_edit(self) -> bool:
        """Cancel viewer ownership or restore a stranded Main transaction."""

        viewer_cancelled = self.viewer.cancel_canvas_surface_edit()
        if self._active_canvas_surface_edit_target is None:
            return viewer_cancelled
        self._handle_canvas_surface_edit_cancelled(None)
        return True

    def _apply_active_canvas_surface_edit(
        self,
        raw_edit: object,
        *,
        validate_project_geometry: bool,
    ) -> bool:
        """Validate and apply one absolute edit against its drag-start target."""

        target = self._active_canvas_surface_edit_target
        if not isinstance(raw_edit, CanvasSurfaceEdit) or target is None:
            return False
        active_targets = self._active_canvas_surface_edit_targets or (target,)
        if (
            raw_edit.reference != target.reference
            or raw_edit.surface_id != target.surface_id
        ):
            return False
        try:
            if len(active_targets) == 1:
                applied_edits = (
                    apply_canvas_surface_edit(
                        self.levels,
                        target,
                        raw_edit,
                        validate_project_geometry=validate_project_geometry,
                    ),
                )
            else:
                applied_edits = apply_canvas_wall_edit_batch(
                    self.levels,
                    active_targets,
                    raw_edit,
                    validate_project_geometry=validate_project_geometry,
                )
        except (TypeError, ValueError) as error:
            resume_pending_surface_update = bool(
                self._pending_canvas_surface_mesh_update
                and target.reference.kind in DELAYED_CANVAS_SURFACE_EDIT_KINDS
            )
            try:
                restored_edits = self._restore_canvas_surface_edit_targets(
                    active_targets
                )
            except (TypeError, ValueError):
                pass
            else:
                self._sync_live_canvas_surface_edits(restored_edits)
            self._active_canvas_surface_edit_target = None
            self._active_canvas_surface_edit_targets = ()
            self.viewer.set_window_tools_status(f"Surface edit stopped: {error}")
            if not resume_pending_surface_update:
                self.viewer.clear_canvas_surface_edit_pending_outline()
            self.viewer.cancel_canvas_surface_edit()
            if resume_pending_surface_update:
                self._canvas_surface_mesh_update_timer.start()
            else:
                self._queue_viewer_preview_refresh()
            return False
        self._sync_live_canvas_surface_edits(applied_edits)
        return True

    def _sync_live_canvas_surface_edit(
        self,
        applied: AppliedCanvasSurfaceEdit,
    ) -> None:
        """Repaint shared 2D data and reflect edited level-wide values."""

        level = applied.level
        if level.index != self.current_level.index:
            return
        self.canvas.update()
        self._is_syncing_level_controls = True
        try:
            self.height_level_spinbox.setValue(level.height_meters)
            self.floor_thickness_spinbox.setValue(level.floor_thickness_meters)
            level_x_slider_value = round(
                level.offset_x_meters * LEVEL_OFFSET_SLIDER_FACTOR
            )
            level_y_slider_value = round(
                level.offset_y_meters * LEVEL_OFFSET_SLIDER_FACTOR
            )
            self._fit_slider_range_to_value(
                self.level_x_offset_slider,
                level_x_slider_value,
            )
            self._fit_slider_range_to_value(
                self.level_y_offset_slider,
                level_y_slider_value,
            )
            self.level_x_offset_slider.setValue(level_x_slider_value)
            self.level_y_offset_slider.setValue(level_y_slider_value)
            self._update_level_x_offset_value_label(level.offset_x_meters)
            self._update_level_y_offset_value_label(level.offset_y_meters)
        finally:
            self._is_syncing_level_controls = False

    def _get_canvas_floor_edit_target(
        self,
        level_index: int,
    ) -> CanvasSurfaceEditHandleTarget | None:
        """Return the active floor handle, or another handle for its level."""

        candidates = tuple(
            target
            for target in self._canvas_surface_edit_targets_by_key.values()
            if (
                target.reference.kind == CANVAS_SURFACE_EDIT_FLOOR_THICKNESS
                and target.reference.level_index == level_index
            )
        )
        active_surface_id = self.viewer.get_active_canvas_surface_id()
        return next(
            (target for target in candidates if target.surface_id == active_surface_id),
            candidates[0] if candidates else None,
        )

    def _stage_floor_thickness_mesh_update(
        self,
        level: LevelData,
        thickness_meters: float,
    ) -> None:
        """Stage a live floor value behind the shared mesh-edit debounce."""

        if self._pending_canvas_surface_mesh_update and (
            self._pending_floor_thickness_level_index != level.index
        ):
            self._commit_pending_canvas_surface_mesh_update()
        self._commit_pending_wall_vertex_update()

        previous_thickness = float(level.floor_thickness_meters)
        next_thickness = float(thickness_meters)
        if next_thickness == previous_thickness:
            return
        self._viewer_floor_thickness_by_level_index.setdefault(
            level.index,
            previous_thickness,
        )

        target = self._get_canvas_floor_edit_target(level.index)
        if target is not None and (target.baseline_value_meters != previous_thickness):
            try:
                target = rebase_canvas_floor_edit_target(
                    self.levels,
                    target,
                )
            except (TypeError, ValueError):
                target = None
        if self._pending_canvas_surface_mesh_baseline is None and target is not None:
            self._pending_canvas_surface_mesh_baseline = target
            self._pending_canvas_surface_mesh_baselines = (target,)
            self._pending_canvas_wall_surface_ids = ()

        level.floor_thickness_meters = next_thickness
        self._pending_floor_thickness_level_index = level.index
        self._pending_canvas_surface_mesh_update = True
        self._is_syncing_level_controls = True
        try:
            self.floor_thickness_spinbox.setValue(next_thickness)
        finally:
            self._is_syncing_level_controls = False

        if target is not None:
            try:
                rebased_target = rebase_canvas_floor_edit_target(
                    self.levels,
                    target,
                )
            except (TypeError, ValueError) as error:
                self._reject_pending_canvas_surface_mesh_update(error)
                return
            self._canvas_surface_edit_targets_by_key = {
                (
                    rebased_target.surface_id,
                    rebased_target.reference.key,
                ): rebased_target
            }
            self.viewer.set_canvas_surface_edit_targets(
                (rebased_target,),
                preserve_pending_outline=True,
            )
        self._canvas_surface_mesh_update_timer.start()

    # ### Canvas scene selection synchronization ###
    def _handle_canvas_placed_object_selection_changed(
        self,
        raw_object_id: object,
    ) -> None:
        """Remember the active object while retaining a multi-object selection."""

        if self._is_syncing_canvas_scene_selection:
            return
        object_id = (
            None if raw_object_id is None else str(raw_object_id).strip() or None
        )
        try:
            viewer_object_ids = self.viewer.get_selected_placed_object_ids()
        except (AttributeError, TypeError):
            viewer_object_ids = (object_id,) if object_id is not None else ()
        if not isinstance(viewer_object_ids, tuple | list):
            viewer_object_ids = (object_id,) if object_id is not None else ()
        self._remember_desired_canvas_object_selection(
            viewer_object_ids,
            active_object_id=object_id,
        )

    def _handle_canvas_placed_object_selection_set_changed(
        self,
        raw_object_ids: object,
    ) -> None:
        """Remember every selected Canvas object across preview rebuilds."""

        if self._is_syncing_canvas_scene_selection:
            return
        try:
            object_ids = (
                (raw_object_ids,)
                if isinstance(raw_object_ids, str)
                else tuple(raw_object_ids)  # type: ignore[arg-type]
            )
        except TypeError:
            return
        try:
            active_object_id = self.viewer.get_selected_placed_object_id()
        except (AttributeError, TypeError):
            active_object_id = self._desired_canvas_object_id
        self._remember_desired_canvas_object_selection(
            object_ids,
            active_object_id=active_object_id,
        )

    def _remember_desired_canvas_object_selection(
        self,
        object_ids: Sequence[object],
        *,
        active_object_id: object | None,
    ) -> None:
        """Normalize and persist one semantic Canvas object selection set."""

        normalized_object_ids = tuple(
            dict.fromkeys(
                object_id
                for object_id in (str(value).strip() for value in object_ids)
                if object_id
            )
        )
        normalized_active_id = (
            None
            if active_object_id is None
            else str(active_object_id).strip() or None
        )
        if normalized_active_id not in normalized_object_ids:
            normalized_active_id = (
                normalized_object_ids[-1] if normalized_object_ids else None
            )
        self._desired_canvas_object_ids = normalized_object_ids
        self._desired_canvas_object_id = normalized_active_id
        if normalized_object_ids:
            self._discard_staged_stair_edit(clear_selection=True)
            self._desired_canvas_surface_ids = ()
            self._atlas_surface_assignment_target_ids = ()
            self._sync_surface_generation_selection(())
        if normalized_active_id is not None:
            self.generation.select_generated_object(normalized_active_id)
        self._sync_atlas_texture_selection_from_canvas_scene()

    def _sync_atlas_texture_selection_from_canvas_scene(self) -> None:
        """Select textures assigned to the current semantic 3D selection."""

        object_ids = self._desired_canvas_object_ids
        if object_ids:
            textured_object_ids = tuple(
                object_id
                for object_id in object_ids
                if self.generation.get_active_texture_variant(object_id) is not None
            )
            active_object_id = (
                self._desired_canvas_object_id
                if self._desired_canvas_object_id in textured_object_ids
                else None
            )
            self.texture_atlas_workspace.select_source_ids(
                textured_object_ids,
                active_source_id=active_object_id,
            )
            self._selected_atlas_surface_source_id = None
            self._set_atlas_canvas_surface_highlights(())
            self._sync_atlas_green_outline_to_canvas_highlight(None)
            return

        surface_ids = tuple(
            dict.fromkeys(
                (
                    *self._desired_canvas_surface_ids,
                    *self._desired_canvas_stair_part_ids,
                )
            )
        )
        source_id_by_surface_id = self._build_atlas_surface_source_ids()
        source_ids = tuple(
            dict.fromkeys(
                source_id_by_surface_id[surface_id]
                for surface_id in surface_ids
                if surface_id in source_id_by_surface_id
            )
        )
        active_surface_id = surface_ids[-1] if surface_ids else None
        active_source_id = (
            None
            if active_surface_id is None
            else source_id_by_surface_id.get(active_surface_id)
        )
        selected_source_ids = self.texture_atlas_workspace.select_source_ids(
            source_ids,
            active_source_id=active_source_id,
        )
        self._handle_atlas_surface_textures_selected(selected_source_ids)

    def _discard_desired_canvas_object(self, object_id: str) -> None:
        """Forget one vanished object without dropping other selected objects."""

        normalized_object_id = str(object_id).strip()
        selected_object_ids = getattr(
            self,
            "_desired_canvas_object_ids",
            (
                (self._desired_canvas_object_id,)
                if self._desired_canvas_object_id is not None
                else ()
            ),
        )
        remaining_object_ids = tuple(
            candidate
            for candidate in selected_object_ids
            if candidate != normalized_object_id
        )
        active_object_id = self._desired_canvas_object_id
        if active_object_id == normalized_object_id:
            active_object_id = (
                remaining_object_ids[-1] if remaining_object_ids else None
            )
        self._remember_desired_canvas_object_selection(
            remaining_object_ids,
            active_object_id=active_object_id,
        )

    def _restore_desired_canvas_scene_selection(self) -> None:
        """Reapply the last semantic Canvas selection after a model refresh."""

        if self._desired_canvas_stair_part_ids:
            was_syncing_selection = self._is_syncing_canvas_scene_selection
            self._is_syncing_canvas_scene_selection = True
            try:
                self.viewer.set_selected_placed_object_ids(())
                self.viewer.select_canvas_opening(None)
                self.viewer.set_selected_canvas_surface_ids(())
                self.viewer.set_selected_canvas_stair_part_ids(
                    self._desired_canvas_stair_part_ids
                )
            finally:
                self._is_syncing_canvas_scene_selection = was_syncing_selection
            self._sync_selected_canvas_wall_highlight(None)
            return
        desired_object_ids = getattr(
            self,
            "_desired_canvas_object_ids",
            (
                (self._desired_canvas_object_id,)
                if self._desired_canvas_object_id is not None
                else ()
            ),
        )
        if desired_object_ids:
            was_syncing_selection = self._is_syncing_canvas_scene_selection
            self._is_syncing_canvas_scene_selection = True
            try:
                self.viewer.set_selected_placed_object_ids(
                    desired_object_ids,
                    active_object_id=self._desired_canvas_object_id,
                )
                restored_object_ids = self.viewer.get_selected_placed_object_ids()
                restored_active_id = self.viewer.get_selected_placed_object_id()
            finally:
                self._is_syncing_canvas_scene_selection = was_syncing_selection
            self._remember_desired_canvas_object_selection(
                restored_object_ids,
                active_object_id=restored_active_id,
            )
        if self._desired_canvas_object_ids:
            self._sync_selected_canvas_wall_highlight(None)
            return
        if self._desired_canvas_surface_ids:
            self.viewer.set_selected_placed_object_ids(())
            self.viewer.select_canvas_opening(None)
            selected_surface = (
                self._canvas_surface_targets_by_id.get(
                    self._desired_canvas_surface_ids[0]
                )
                if len(self._desired_canvas_surface_ids) == 1
                else None
            )
            if (
                selected_surface is not None
                and selected_surface.surface_type == SURFACE_TYPE_WALL
            ):
                self.viewer.select_wall_target(selected_surface.surface_id)
            else:
                self.viewer.select_wall_target(None)
                self.viewer.set_selected_canvas_surface_ids(
                    self._desired_canvas_surface_ids
                )
        self._sync_selected_canvas_wall_highlight(
            self.viewer.get_active_canvas_surface_id()
        )

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
        self.viewer.set_window_undo_available(bool(self._canvas_window_undo_ids))

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

    # ### Atlas draw-call estimation ###
    def _atlas_workspace_is_active(self) -> bool:
        """Return whether the embedded or detached Atlas workspace is visible."""

        external_host = getattr(self, "_external_atlas_host", None)
        return bool(
            self.workspace_tabs.currentWidget() is self.texture_atlas_workspace
            or (external_host is not None and external_host.is_active)
        )

    @staticmethod
    def _build_atlas_draw_call_layout_signature(
        atlases: Sequence[TextureAtlasRecord],
        required_source_ids: Sequence[str],
    ) -> tuple[tuple[object, ...], ...]:
        """Identify Atlas bindings that can change the exported primitive count."""

        required_ids = {
            str(source_id).strip()
            for source_id in required_source_ids
            if str(source_id).strip()
        }
        return tuple(
            (
                atlas.atlas_id,
                tuple(
                    placement.object_id
                    for placement in atlas.placements
                    if placement.object_id in required_ids
                ),
            )
            for atlas in atlases
            if any(
                placement.object_id in required_ids for placement in atlas.placements
            )
        )

    def _build_atlas_draw_call_dependency_signature(
        self,
    ) -> tuple[tuple[object, ...], ...]:
        """Track placed assets and level membership while ignoring transforms."""

        signature: list[tuple[object, ...]] = []
        for item in self.generation.get_placed_preview_dependency_signature():
            if not isinstance(item, tuple) or len(item) < 5:
                signature.append(tuple(item) if isinstance(item, tuple) else (item,))
                continue
            placement = item[1]
            signature.append(
                (
                    item[0],
                    (
                        placement.level_index
                        if isinstance(placement, GeneratedObjectPlacement)
                        else None
                    ),
                    item[2],
                    item[4],
                )
            )
        return tuple(signature)

    def _capture_atlas_draw_call_estimate_request(
        self,
    ) -> tuple[
        _SurfaceAmbientOcclusionSceneSnapshot,
        tuple[TextureAtlasRecord, ...],
        _AtlasDrawCallEstimateSnapshot,
    ]:
        """Capture one stable, serializable estimator request on the GUI thread."""

        scene_revision = self._atlas_draw_call_scene_revision
        dependency_signature_before = self._build_viewer_preview_dependency_signature()
        draw_call_dependency_before = self._build_atlas_draw_call_dependency_signature()
        scene_snapshot = self._capture_surface_ao_scene_snapshot(
            dependency_signature_before
        )
        atlases = tuple(self.texture_atlas_workspace.get_data().atlases)
        atlas_layout_signature = self._build_atlas_draw_call_layout_signature(
            atlases,
            scene_snapshot.required_source_ids,
        )
        dependency_signature_after = self._build_viewer_preview_dependency_signature()
        draw_call_dependency_after = self._build_atlas_draw_call_dependency_signature()
        if (
            scene_revision != self._atlas_draw_call_scene_revision
            or dependency_signature_before != dependency_signature_after
            or draw_call_dependency_before != draw_call_dependency_after
            or atlas_layout_signature
            != self._build_atlas_draw_call_layout_signature(
                self.texture_atlas_workspace.get_data().atlases,
                scene_snapshot.required_source_ids,
            )
            or scene_snapshot.surface_source_ids
            != tuple(sorted(self._build_atlas_surface_source_ids().items()))
        ):
            raise RuntimeError(
                "Scene inputs changed while the draw-call estimate was prepared."
            )
        snapshot = _AtlasDrawCallEstimateSnapshot(
            scene_revision=scene_revision,
            dependency_signature=draw_call_dependency_after,
            atlas_layout_signature=atlas_layout_signature,
            required_source_ids=scene_snapshot.required_source_ids,
            surface_source_signature=scene_snapshot.surface_source_ids,
        )
        return scene_snapshot, atlases, snapshot

    def _atlas_draw_call_estimate_snapshot_is_current(
        self,
        snapshot: _AtlasDrawCallEstimateSnapshot,
    ) -> bool:
        """Reject an estimate built from superseded geometry or bindings."""

        if self._atlas_draw_call_scene_revision != snapshot.scene_revision:
            return False
        if (
            self._build_atlas_draw_call_dependency_signature()
            != snapshot.dependency_signature
        ):
            return False
        if (
            tuple(sorted(self._build_atlas_surface_source_ids().items()))
            != snapshot.surface_source_signature
        ):
            return False
        return (
            self._build_atlas_draw_call_layout_signature(
                self.texture_atlas_workspace.get_data().atlases,
                snapshot.required_source_ids,
            )
            == snapshot.atlas_layout_signature
        )

    def _schedule_atlas_draw_call_estimate(self) -> None:
        """Debounce scene mutations and cancel any superseded estimator."""

        if self._is_shutdown:
            return
        self._cancel_atlas_draw_call_estimate_preparations()
        self.texture_atlas_workspace.set_draw_call_estimate_pending()
        if self._atlas_workspace_is_active():
            self._atlas_draw_call_estimate_refresh_timer.start()

    def _refresh_atlas_draw_call_estimate(self) -> None:
        """Start a non-blocking exact estimate for the current export scene."""

        self._atlas_draw_call_estimate_refresh_timer.stop()
        if self._is_shutdown or not self._atlas_workspace_is_active():
            return
        if any(
            runtime.thread.isRunning()
            for runtime in self._atlas_draw_call_estimate_runtimes.values()
        ):
            self._atlas_draw_call_estimate_refresh_timer.start()
            return
        try:
            scene_snapshot, atlases, snapshot = (
                self._capture_atlas_draw_call_estimate_request()
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self.texture_atlas_workspace.set_draw_call_estimate_unavailable(str(error))
            return
        if (
            snapshot == self._atlas_draw_call_estimate_cache_signature
            and self._atlas_draw_call_estimate_cache_result is not None
        ):
            self.texture_atlas_workspace.set_draw_call_estimate(
                self._atlas_draw_call_estimate_cache_result
            )
            return

        request_id = self._cancel_atlas_draw_call_estimate_preparations()
        thread = _AtlasDrawCallEstimateThread(scene_snapshot, atlases, parent=self)
        runtime = _AtlasDrawCallEstimateRuntime(
            request_id=request_id,
            thread=thread,
            snapshot=snapshot,
        )
        self._atlas_draw_call_estimate_runtimes[request_id] = runtime
        thread.finished.connect(
            partial(
                self._handle_atlas_draw_call_estimate_finished,
                request_id,
                thread,
            )
        )
        self.texture_atlas_workspace.set_draw_call_estimate_pending()
        thread.start()

    def _handle_atlas_draw_call_estimate_finished(
        self,
        request_id: int,
        thread: _AtlasDrawCallEstimateThread,
    ) -> None:
        """Commit one estimator result only while every input remains current."""

        runtime = self._atlas_draw_call_estimate_runtimes.pop(request_id, None)
        try:
            if runtime is None or runtime.thread is not thread:
                return
            if (
                self._is_shutdown
                or runtime.cancel_requested
                or thread.was_cancelled
                or request_id != self._atlas_draw_call_estimate_request_id
            ):
                return
            if not self._atlas_draw_call_estimate_snapshot_is_current(runtime.snapshot):
                self._schedule_atlas_draw_call_estimate()
                return
            if thread.error_message is not None or thread.result is None:
                self.texture_atlas_workspace.set_draw_call_estimate_unavailable(
                    thread.error_message
                    or "The estimator finished without returning a result."
                )
                return
            self._atlas_draw_call_estimate_cache_signature = runtime.snapshot
            self._atlas_draw_call_estimate_cache_result = thread.result
            self.texture_atlas_workspace.set_draw_call_estimate(thread.result)
        finally:
            thread.deleteLater()

    def _cancel_atlas_draw_call_estimate_preparations(self) -> int:
        """Invalidate current estimator work without blocking the GUI."""

        self._atlas_draw_call_estimate_request_id += 1
        for runtime in self._atlas_draw_call_estimate_runtimes.values():
            runtime.cancel_requested = True
            runtime.thread.requestInterruption()
        return self._atlas_draw_call_estimate_request_id

    def _cancel_and_join_atlas_draw_call_estimates(self) -> None:
        """Join every estimator worker before replacing its project state."""

        self._atlas_draw_call_estimate_refresh_timer.stop()
        self._cancel_atlas_draw_call_estimate_preparations()
        runtimes = tuple(self._atlas_draw_call_estimate_runtimes.values())
        for runtime in runtimes:
            while runtime.thread.isRunning():
                runtime.thread.wait(SURFACE_AO_SHUTDOWN_WAIT_MILLISECONDS)
            runtime.thread.deleteLater()
        self._atlas_draw_call_estimate_runtimes.clear()
        self._atlas_draw_call_estimate_cache_signature = None
        self._atlas_draw_call_estimate_cache_result = None
        if hasattr(self, "texture_atlas_workspace"):
            self.texture_atlas_workspace.set_draw_call_estimate(None)

    # ### Atlas ambient-occlusion jobs ###
    def _handle_all_ambient_occlusion_bakes_requested(self) -> None:
        """Start one isolated AO job for every Atlas used by the scene."""

        if self._is_shutdown:
            return
        self._commit_pending_ambient_occlusion_scene_edits()
        try:
            preparation = self._capture_ambient_occlusion_bake_preparation()
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            message = str(error) or type(error).__name__
            self.texture_atlas_workspace.status_label.setText(
                f"Ambient-occlusion bake failed: {message}"
            )
            QMessageBox.warning(self, "Ambient-occlusion bake failed", message)
            return

        required_source_ids = set(preparation.scene_snapshot.required_source_ids)
        claimed_source_ids: set[str] = set()
        targets: list[tuple[str, float]] = []
        for context in preparation.atlas_context:
            atlas = context.atlas
            placement_source_ids = {
                placement.object_id for placement in atlas.placements
            }
            if any(
                source_id in required_source_ids
                and source_id not in claimed_source_ids
                for source_id in placement_source_ids
            ):
                targets.append((atlas.atlas_id, atlas.surface_ao_intensity))
            claimed_source_ids.update(placement_source_ids)
        if not targets:
            self.texture_atlas_workspace.status_label.setText(
                "No Atlas contains geometry used by the current exported scene."
            )
            return
        for atlas_id, intensity in targets:
            self._start_ambient_occlusion_bake(
                atlas_id,
                intensity,
                preparation=preparation,
            )

    def _capture_surface_ao_scene_snapshot(
        self,
        dependency_signature: tuple[object, ...],
    ) -> _SurfaceAmbientOcclusionSceneSnapshot:
        """Copy only plain scene state and placed-asset paths on the GUI thread."""

        if len(dependency_signature) != 3 or not isinstance(
            dependency_signature[1],
            tuple,
        ):
            raise RuntimeError("The surface AO scene dependencies are invalid.")
        levels = tuple(copy.deepcopy(self.levels))
        stairs = tuple(copy.deepcopy(self.stairs))
        visible_level_by_index = {
            level.index: level for level in levels if level.include_in_export
        }
        base_z_by_level_index = build_level_base_z_lookup(levels)
        object_names = self.generation.get_generated_object_names_by_id()
        placed_models: list[_PlacedGeneratedModelFileSnapshot] = []
        for raw_item in dependency_signature[1]:
            if not isinstance(raw_item, tuple) or len(raw_item) < 5:
                raise RuntimeError("A placed-object AO dependency is invalid.")
            object_id = str(raw_item[0])
            placement = raw_item[1]
            asset_revision = raw_item[2]
            symmetry = raw_item[4]
            if not isinstance(placement, GeneratedObjectPlacement):
                raise RuntimeError("A placed-object AO transform is invalid.")
            level = visible_level_by_index.get(placement.level_index)
            base_z = base_z_by_level_index.get(placement.level_index)
            if level is None or base_z is None:
                continue
            if (
                not isinstance(asset_revision, tuple)
                or len(asset_revision) != 4
                or any(value is None for value in asset_revision)
            ):
                continue
            world_x, world_y = level_image_to_world_xy(
                level,
                placement.image_x,
                placement.image_y,
            )
            raw_orientation = getattr(symmetry, "orientation", None)
            raw_plane_coordinate = getattr(
                symmetry,
                "plane_coordinate",
                None,
            )
            placed_models.append(
                _PlacedGeneratedModelFileSnapshot(
                    object_id=object_id,
                    object_name=object_names.get(object_id, object_id),
                    asset_path=Path(str(asset_revision[0])),
                    asset_revision=(
                        str(asset_revision[0]),
                        int(asset_revision[1]),
                        int(asset_revision[2]),
                        int(asset_revision[3]),
                    ),
                    world_position=(
                        world_x,
                        world_y,
                        base_z + placement.height_offset_meters,
                    ),
                    rotation_degrees=placement.rotation_degrees,
                    scale=placement.scale,
                    symmetric_preview_orientation=(
                        None if raw_orientation is None else str(raw_orientation)
                    ),
                    symmetric_preview_plane_coordinate=(
                        None
                        if raw_plane_coordinate is None
                        else float(raw_plane_coordinate)
                    ),
                )
            )

        material_items: list[tuple[str, object]] = []
        for surface_id, source in sorted(
            self.surface_texture_generation.get_surface_material_sources().items()
        ):
            if isinstance(source, SurfaceMaterialSourceSpec):
                frozen_source: object = SurfaceMaterialSourceSpec(
                    map_sources={
                        str(map_type): Path(map_path)
                        for map_type, map_path in source.map_sources.items()
                    },
                    tiling_mode=source.tiling_mode,
                    tiling_seed=source.tiling_seed,
                    texture_repeat_size_m=source.texture_repeat_size_m,
                )
            elif isinstance(source, Mapping):
                frozen_source = tuple(
                    sorted(
                        (str(map_type), Path(map_path))
                        for map_type, map_path in source.items()
                    )
                )
            else:
                frozen_source = Path(source)
            material_items.append((str(surface_id), frozen_source))
        return _SurfaceAmbientOcclusionSceneSnapshot(
            levels=levels,
            stairs=stairs,
            surface_materials=tuple(material_items),
            placed_models=tuple(placed_models),
            surface_source_ids=tuple(
                sorted(self._build_atlas_surface_source_ids().items())
            ),
        )

    def _handle_atlas_preview_map_changed(
        self,
        _map_type: str,
        is_ambient_occlusion: bool,
    ) -> None:
        """Enter or leave the Atlas viewer's AO-only material mode."""

        if self._is_shutdown:
            return
        if is_ambient_occlusion:
            self._surface_ao_preview_refresh_timer.stop()
            if self._atlas_preview_display_state is None:
                self._atlas_preview_display_state = (
                    self.atlas_object_preview_viewer.get_textures_enabled(),
                    self.atlas_object_preview_viewer.get_wireframe_enabled(),
                    self.atlas_object_preview_viewer.get_wireframe_only(),
                    self.atlas_object_preview_viewer.get_pbr_maps_enabled(),
                )
            if self._canvas_ao_preview_display_state is None:
                self._canvas_ao_preview_display_state = (
                    self.viewer.get_textures_enabled(),
                    self.viewer.get_wireframe_enabled(),
                    self.viewer.get_wireframe_only(),
                    self.viewer.get_pbr_maps_enabled(),
                    self.viewer.get_ambient_light_intensity(),
                )
            self.atlas_object_preview_viewer.set_textures_enabled(True)
            self.atlas_object_preview_viewer.set_wireframe_enabled(False)
            self.atlas_object_preview_viewer.set_wireframe_only(False)
            self.atlas_object_preview_viewer.set_pbr_maps_enabled(())
            self._set_canvas_ambient_occlusion_preview_pending()
            self._refresh_surface_ambient_occlusion_preview()
            return

        previous_atlas_state = self._atlas_preview_display_state
        previous_canvas_state = self._canvas_ao_preview_display_state
        if previous_atlas_state is None and previous_canvas_state is None:
            return
        self._atlas_preview_display_state = None
        self._canvas_ao_preview_display_state = None
        self._surface_ao_canvas_preview_key = None
        self._surface_ao_preview_refresh_timer.stop()
        self._cancel_surface_ambient_occlusion_preview_preparations()
        if previous_atlas_state is not None:
            (
                textures_enabled,
                wireframe_enabled,
                wireframe_only,
                pbr_maps_enabled,
            ) = previous_atlas_state
            self.atlas_object_preview_viewer.set_textures_enabled(textures_enabled)
            self.atlas_object_preview_viewer.set_wireframe_enabled(wireframe_enabled)
            self.atlas_object_preview_viewer.set_wireframe_only(wireframe_only)
            self.atlas_object_preview_viewer.set_pbr_maps_enabled(pbr_maps_enabled)
        if previous_canvas_state is not None:
            (
                textures_enabled,
                wireframe_enabled,
                wireframe_only,
                pbr_maps_enabled,
                ambient_light_intensity,
            ) = previous_canvas_state
            self.viewer.set_textures_enabled(textures_enabled)
            self.viewer.set_wireframe_enabled(wireframe_enabled)
            self.viewer.set_wireframe_only(wireframe_only)
            self.viewer.set_pbr_maps_enabled(pbr_maps_enabled)
            self.viewer.set_ambient_light_intensity(ambient_light_intensity)
        self._clear_atlas_object_preview()
        self.texture_atlas_workspace.request_selected_object_preview()
        self._restore_canvas_preview_after_ambient_occlusion()

    def _set_canvas_ambient_occlusion_preview_pending(self) -> None:
        """Keep the Canvas scene visible but hide stale textures during prep."""

        self._surface_ao_canvas_preview_key = None
        self._canvas_viewer_preview_revision = -1
        self.viewer.set_textures_enabled(False)
        self.viewer.set_wireframe_enabled(False)
        self.viewer.set_wireframe_only(False)
        self.viewer.set_pbr_maps_enabled(())
        self.viewer.set_ambient_light_intensity(1.0)

    def _canvas_interaction_blocks_ambient_occlusion_preview(self) -> bool:
        """Avoid replacing the Canvas model during an edit or pointer drag."""

        return bool(
            self._active_canvas_surface_edit_target is not None
            or self._pending_canvas_surface_mesh_update
            or self._pending_wall_vertex_mesh_update
            or self._is_canvas_wall_vertex_interaction_active
            or self._is_canvas_opening_drag_active
            or self._is_doorway_move_drag_active
            or self._is_doorway_resize_drag_active
            or self._pending_doorway_mesh_level_index is not None
            or self._pending_window_mesh_level_index is not None
            or self._level_transform_drag_active
            or self._pending_level_transform is not None
            or self.viewer.view.is_primary_pointer_drag_reserved
            or self.viewer.view.is_middle_navigation_active
            or self.viewer.view.is_face_selection_gesture_active
            or self.viewer.is_window_placement_active()
            or self.viewer.is_surface_vertex_placement_active()
        )

    def _restore_canvas_preview_after_ambient_occlusion(self) -> None:
        """Reinstall the cached normal scene without disturbing its camera."""

        self._canvas_viewer_preview_revision = -1
        revision = self._viewer_preview_revision
        cached_model = self._viewer_preview_model
        cache_is_current = bool(
            cached_model is not None
            and self._viewer_preview_model_revision == revision
            and self._viewer_preview_dependency_signature_revision == revision
            and self._viewer_preview_dependency_signature
            == self._build_viewer_preview_dependency_signature()
        )
        if cache_is_current:
            assert cached_model is not None
            self._set_canvas_viewer_targets(
                tuple(build_fixed_surfaces(self._build_viewer_preview_levels()))
            )
            self._is_syncing_canvas_scene_selection = True
            try:
                self.viewer.set_model(cached_model, preserve_camera=True)
                self._restore_desired_canvas_scene_selection()
            finally:
                self._is_syncing_canvas_scene_selection = False
            self._canvas_viewer_preview_revision = revision
            return
        self._ensure_viewer_preview_current(preserve_camera=True)

    def _handle_texture_atlas_data_changed_for_ao_preview(
        self,
        _data: object,
    ) -> None:
        """Debounce Atlas-derived views after membership changes."""

        self._schedule_surface_ambient_occlusion_preview_refresh()
        self._schedule_atlas_draw_call_estimate()

    def _schedule_surface_ambient_occlusion_preview_refresh(self) -> None:
        """Coalesce scene and Atlas mutations into one AO-only rebuild."""

        if (
            self._is_shutdown
            or not self.texture_atlas_workspace.is_ambient_occlusion_preview_active
        ):
            return
        self._surface_ao_preview_refresh_timer.start()

    def _refresh_surface_ambient_occlusion_preview(self) -> None:
        """Show or asynchronously prepare the selected Atlas's AO receivers."""

        self._surface_ao_preview_refresh_timer.stop()
        if (
            self._is_shutdown
            or not self.texture_atlas_workspace.is_ambient_occlusion_preview_active
        ):
            return
        if self._canvas_interaction_blocks_ambient_occlusion_preview():
            self._cancel_surface_ambient_occlusion_preview_preparations()
            self._surface_ao_preview_refresh_timer.start()
            self.texture_atlas_workspace.status_label.setText(
                "Ambient-occlusion preview is waiting for the active Canvas "
                "edit to finish."
            )
            return
        self._set_canvas_ambient_occlusion_preview_pending()
        atlas = self.texture_atlas_workspace.selected_atlas
        if atlas is None:
            self._cancel_surface_ambient_occlusion_preview_preparations()
            self._clear_atlas_object_preview()
            self.texture_atlas_workspace.status_label.setText(
                "Select an Atlas to preview ambient occlusion."
            )
            return
        atlas_id = atlas.atlas_id
        if (
            atlas.surface_ao_image_path is None
            or atlas.surface_ao_geometry_signature is None
        ):
            self._surface_ao_preview_cache.pop(atlas_id, None)
            self._cancel_surface_ambient_occlusion_preview_preparations()
            self._clear_atlas_object_preview()
            self.texture_atlas_workspace.status_label.setText(
                "No ambient-occlusion bake exists for this Atlas. Use "
                "Bake ambient occlusion first."
            )
            return

        cached = self._surface_ao_preview_cache.get(atlas_id)
        if cached is not None and self._surface_ao_preview_cache_is_current(
            atlas_id,
            cached,
        ):
            self._cancel_surface_ambient_occlusion_preview_preparations()
            self._show_surface_ambient_occlusion_preview(
                atlas_id,
                atlas.name,
                cached,
            )
            return
        self._surface_ao_preview_cache.pop(atlas_id, None)

        request_id = self._cancel_surface_ambient_occlusion_preview_preparations()
        try:
            (
                scene_snapshot,
                target_context,
                atlas_context,
                snapshot,
                image_revision,
            ) = self._capture_surface_ambient_occlusion_preview_request(atlas_id)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._clear_atlas_object_preview()
            self.texture_atlas_workspace.status_label.setText(
                "Ambient-occlusion preview unavailable: "
                f"{str(error) or type(error).__name__}"
            )
            return

        thread = _SurfaceAmbientOcclusionPreviewThread(
            scene_snapshot,
            target_context,
            atlas_context,
            self,
        )
        runtime = _SurfaceAmbientOcclusionPreviewRuntime(
            request_id=request_id,
            atlas_id=atlas_id,
            thread=thread,
            snapshot=snapshot,
            image_revision=image_revision,
        )
        self._surface_ao_preview_runtimes[request_id] = runtime
        thread.progress.connect(
            partial(
                self._handle_surface_ambient_occlusion_preview_progress,
                request_id,
                thread,
            )
        )
        thread.finished.connect(
            partial(
                self._handle_surface_ambient_occlusion_preview_finished,
                request_id,
                thread,
            )
        )
        self._clear_atlas_object_preview()
        self.texture_atlas_workspace.status_label.setText(
            f"Preparing ambient-occlusion-only preview for {atlas.name}."
        )
        thread.start()

    def _capture_surface_ambient_occlusion_preview_request(
        self,
        atlas_id: str,
    ) -> tuple[
        _SurfaceAmbientOcclusionSceneSnapshot,
        SurfaceAmbientOcclusionAtlasContext,
        tuple[SurfaceAmbientOcclusionAtlasContext, ...],
        _SurfaceAmbientOcclusionBakeSnapshot,
        tuple[str, int, int, int],
    ]:
        """Capture one stable cached-AO preview request on the GUI thread."""

        viewer_revision = self._viewer_preview_revision
        dependency_signature_before = self._build_viewer_preview_dependency_signature()
        scene_snapshot = self._capture_surface_ao_scene_snapshot(
            dependency_signature_before
        )
        required_source_ids = scene_snapshot.required_source_ids
        atlas_context_signature_before = self._build_surface_ao_atlas_context_signature(
            required_source_ids
        )
        atlas_context = (
            self.texture_atlas_workspace.prepare_surface_ao_preview_atlas_context(
                required_source_ids,
                atlas_id,
            )
        )
        dependency_signature_after = self._build_viewer_preview_dependency_signature()
        atlas_context_signature_after = self._build_surface_ao_atlas_context_signature(
            required_source_ids
        )
        if (
            viewer_revision != self._viewer_preview_revision
            or dependency_signature_before != dependency_signature_after
            or atlas_context_signature_before != atlas_context_signature_after
        ):
            raise RuntimeError(
                "Scene inputs changed while the AO preview was prepared."
            )
        target_context = next(
            (item for item in atlas_context if item.atlas.atlas_id == atlas_id),
            None,
        )
        if (
            target_context is None
            or target_context.surface_ao_image_path is None
            or target_context.surface_ao_geometry_signature is None
        ):
            raise ValueError("The selected Atlas has no ambient-occlusion bake.")
        image_revision = _build_surface_ao_file_revision(
            target_context.surface_ao_image_path
        )
        snapshot = _SurfaceAmbientOcclusionBakeSnapshot(
            viewer_revision=viewer_revision,
            dependency_signature=dependency_signature_after,
            atlas_context_signature=atlas_context_signature_after,
            required_source_ids=required_source_ids,
            surface_source_signature=scene_snapshot.surface_source_ids,
        )
        return (
            scene_snapshot,
            target_context,
            atlas_context,
            snapshot,
            image_revision,
        )

    def _surface_ao_preview_cache_is_current(
        self,
        atlas_id: str,
        cached: _SurfaceAmbientOcclusionPreviewCacheEntry,
    ) -> bool:
        """Validate cached geometry and AO pixels without rebuilding UV1."""

        atlas = self.texture_atlas_workspace.selected_atlas
        if (
            atlas is None
            or atlas.atlas_id != atlas_id
            or atlas.surface_ao_geometry_signature != cached.geometry_signature
            or not self._surface_ambient_occlusion_snapshot_is_current(
                cached.snapshot,
                atlas_id,
            )
        ):
            return False
        try:
            return (
                _build_surface_ao_file_revision(Path(cached.image_revision[0]))
                == cached.image_revision
            )
        except OSError:
            return False

    def _show_surface_ambient_occlusion_preview(
        self,
        atlas_id: str,
        atlas_name: str,
        cached: _SurfaceAmbientOcclusionPreviewCacheEntry,
    ) -> None:
        """Display raw grayscale AO with no lighting or wireframe modulation."""

        if self._canvas_interaction_blocks_ambient_occlusion_preview():
            self._surface_ao_preview_refresh_timer.start()
            self.texture_atlas_workspace.status_label.setText(
                "Ambient-occlusion preview is waiting for the active Canvas "
                "edit to finish."
            )
            return

        preview_key = (
            "surface_ao_only",
            atlas_id,
            cached.geometry_signature,
            cached.image_revision,
            cached.snapshot.viewer_revision,
        )
        atlas_preview_is_current = bool(
            self._atlas_preview_variant_key == preview_key
            and self.atlas_object_preview_viewer.model is cached.model
        )
        canvas_preview_is_current = bool(
            self._surface_ao_canvas_preview_key == preview_key
            and self.viewer.model is cached.model
        )
        if atlas_preview_is_current and canvas_preview_is_current:
            return
        if not atlas_preview_is_current:
            preserve_camera = self.atlas_object_preview_viewer.model is not None
            self.atlas_object_preview_viewer.set_model(
                cached.model,
                preserve_camera=preserve_camera,
            )
        if not canvas_preview_is_current:
            preserve_camera = self.viewer.model is not None
            self._is_syncing_canvas_scene_selection = True
            try:
                self.viewer.set_model(
                    cached.model,
                    preserve_camera=preserve_camera,
                )
                self.viewer.set_wall_targets(())
                self.viewer.set_canvas_surface_edit_targets(())
                self.viewer.set_canvas_opening_targets(())
            finally:
                self._is_syncing_canvas_scene_selection = False
        self.atlas_object_preview_viewer.set_ambient_light_intensity(1.0)
        self.atlas_object_preview_viewer.set_textures_enabled(True)
        self.atlas_object_preview_viewer.set_wireframe_enabled(False)
        self.atlas_object_preview_viewer.set_wireframe_only(False)
        self.atlas_object_preview_viewer.set_pbr_maps_enabled(())
        self.viewer.set_ambient_light_intensity(1.0)
        self.viewer.set_textures_enabled(True)
        self.viewer.set_wireframe_enabled(False)
        self.viewer.set_wireframe_only(False)
        self.viewer.set_pbr_maps_enabled(())
        self._atlas_preview_variant_key = preview_key
        self._surface_ao_canvas_preview_key = preview_key
        self._canvas_viewer_preview_revision = -1
        self.texture_atlas_workspace.status_label.setText(
            f"Ambient-occlusion-only preview: {atlas_name}."
        )

    def _handle_surface_ambient_occlusion_preview_progress(
        self,
        request_id: int,
        thread: _SurfaceAmbientOcclusionPreviewThread,
        message: str,
    ) -> None:
        """Publish progress only for the currently requested AO-only view."""

        runtime = self._surface_ao_preview_runtimes.get(request_id)
        stage = str(message).strip()
        if (
            self._is_shutdown
            or runtime is None
            or runtime.thread is not thread
            or runtime.cancel_requested
            or request_id != self._surface_ao_preview_request_id
            or not self.texture_atlas_workspace.is_ambient_occlusion_preview_active
            or not stage
        ):
            return
        self.texture_atlas_workspace.status_label.setText(stage)

    def _handle_surface_ambient_occlusion_preview_finished(
        self,
        request_id: int,
        thread: _SurfaceAmbientOcclusionPreviewThread,
    ) -> None:
        """Accept only the latest still-current AO-only preview result."""

        runtime = self._surface_ao_preview_runtimes.pop(request_id, None)
        try:
            if runtime is None or runtime.thread is not thread:
                return
            if (
                self._is_shutdown
                or runtime.cancel_requested
                or thread.was_cancelled
                or request_id != self._surface_ao_preview_request_id
                or not self.texture_atlas_workspace.is_ambient_occlusion_preview_active
            ):
                return
            selected_atlas = self.texture_atlas_workspace.selected_atlas
            if selected_atlas is None or selected_atlas.atlas_id != runtime.atlas_id:
                return
            if thread.error_message is not None:
                self._clear_atlas_object_preview()
                self.texture_atlas_workspace.status_label.setText(
                    f"Ambient-occlusion preview unavailable: {thread.error_message}"
                )
                return
            result = thread.result
            if result is None:
                self._clear_atlas_object_preview()
                self.texture_atlas_workspace.status_label.setText(
                    "Ambient-occlusion preview finished without a model."
                )
                return
            if not self._surface_ambient_occlusion_snapshot_is_current(
                runtime.snapshot,
                runtime.atlas_id,
            ):
                self._refresh_surface_ambient_occlusion_preview()
                return
            try:
                if (
                    _build_surface_ao_file_revision(Path(runtime.image_revision[0]))
                    != runtime.image_revision
                ):
                    self._refresh_surface_ambient_occlusion_preview()
                    return
            except OSError:
                self._refresh_surface_ambient_occlusion_preview()
                return
            atlas = self.texture_atlas_workspace.selected_atlas
            geometry_signature = (
                None
                if atlas is None or atlas.atlas_id != runtime.atlas_id
                else atlas.surface_ao_geometry_signature
            )
            if geometry_signature is None:
                self._refresh_surface_ambient_occlusion_preview()
                return
            cached = _SurfaceAmbientOcclusionPreviewCacheEntry(
                snapshot=runtime.snapshot,
                geometry_signature=geometry_signature,
                image_revision=runtime.image_revision,
                model=result,
            )
            self._surface_ao_preview_cache[runtime.atlas_id] = cached
            self._show_surface_ambient_occlusion_preview(
                runtime.atlas_id,
                selected_atlas.name,
                cached,
            )
        finally:
            thread.deleteLater()

    def _cancel_surface_ambient_occlusion_preview_preparations(self) -> int:
        """Invalidate current AO preview work without blocking the GUI."""

        self._surface_ao_preview_request_id += 1
        for runtime in self._surface_ao_preview_runtimes.values():
            runtime.cancel_requested = True
            runtime.thread.requestInterruption()
        return self._surface_ao_preview_request_id

    def _cancel_and_join_surface_ambient_occlusion_previews(self) -> None:
        """Join every AO preview worker before replacing its project state."""

        self._cancel_surface_ambient_occlusion_preview_preparations()
        runtimes = tuple(self._surface_ao_preview_runtimes.values())
        for runtime in runtimes:
            while runtime.thread.isRunning():
                runtime.thread.wait(SURFACE_AO_SHUTDOWN_WAIT_MILLISECONDS)
            runtime.thread.deleteLater()
        self._surface_ao_preview_runtimes.clear()
        self._surface_ao_preview_cache.clear()

    def _commit_pending_ambient_occlusion_scene_edits(self) -> None:
        """Finish preview edits before an AO snapshot reads scene geometry."""

        self._cancel_active_canvas_surface_edit()
        self._commit_pending_canvas_surface_mesh_update()
        self._commit_pending_wall_vertex_update()
        self._finish_level_transform_drag()
        self._commit_pending_level_transform_update()
        self._commit_pending_doorway_mesh_update()

    def _capture_ambient_occlusion_bake_preparation(
        self,
    ) -> _AmbientOcclusionBakePreparation:
        """Capture one immutable AO input set on the GUI thread."""

        viewer_revision = self._viewer_preview_revision
        dependency_signature_before = self._build_viewer_preview_dependency_signature()
        scene_snapshot = self._capture_surface_ao_scene_snapshot(
            dependency_signature_before
        )
        required_source_ids = scene_snapshot.required_source_ids
        atlas_context_signature_before = (
            self._build_surface_ao_atlas_context_signature(required_source_ids)
        )
        atlas_context = self.texture_atlas_workspace.prepare_surface_ao_atlas_context(
            required_source_ids
        )
        dependency_signature_after = self._build_viewer_preview_dependency_signature()
        atlas_context_signature_after = (
            self._build_surface_ao_atlas_context_signature(required_source_ids)
        )
        if (
            viewer_revision != self._viewer_preview_revision
            or dependency_signature_before != dependency_signature_after
            or atlas_context_signature_before != atlas_context_signature_after
        ):
            raise RuntimeError(
                "Scene inputs changed while the AO snapshot was being built. "
                "Try the operation again."
            )
        snapshot = _SurfaceAmbientOcclusionBakeSnapshot(
            viewer_revision=viewer_revision,
            dependency_signature=dependency_signature_after,
            atlas_context_signature=atlas_context_signature_after,
            required_source_ids=required_source_ids,
            surface_source_signature=scene_snapshot.surface_source_ids,
        )
        return _AmbientOcclusionBakePreparation(
            scene_snapshot=scene_snapshot,
            atlas_context=atlas_context,
            snapshot=snapshot,
        )

    def _handle_ambient_occlusion_bake_requested(
        self,
        atlas_id: str,
        intensity: float,
    ) -> None:
        """Prepare one stable scene snapshot and start its Atlas AO bake."""

        normalized_atlas_id = str(atlas_id).strip()
        if self._is_shutdown or not normalized_atlas_id:
            return
        if normalized_atlas_id in self._surface_ao_bake_runtimes:
            self.texture_atlas_workspace.status_label.setText(
                "Ambient occlusion is already being baked for this Atlas."
            )
            return
        self._commit_pending_ambient_occlusion_scene_edits()
        self._start_ambient_occlusion_bake(normalized_atlas_id, intensity)

    def _start_ambient_occlusion_bake(
        self,
        atlas_id: str,
        intensity: float,
        *,
        preparation: _AmbientOcclusionBakePreparation | None = None,
    ) -> None:
        """Launch one Atlas AO worker from fresh or shared immutable inputs."""

        normalized_atlas_id = str(atlas_id).strip()
        if self._is_shutdown or not normalized_atlas_id:
            return
        if normalized_atlas_id in self._surface_ao_bake_runtimes:
            self.texture_atlas_workspace.status_label.setText(
                "Ambient occlusion is already being baked for this Atlas."
            )
            return

        atlas_record = self.texture_atlas_workspace.get_data().atlas_by_id(
            normalized_atlas_id
        )
        atlas_name = normalized_atlas_id if atlas_record is None else atlas_record.name
        job = self.job_manager.create_job(
            kind="Ambient occlusion",
            requested_name="",
            default_name=f"{atlas_name} ambient occlusion",
            stage="Preparing scene snapshot (1%)",
        )
        try:
            requested_intensity = float(intensity)
            if (
                not math.isfinite(requested_intensity)
                or not 0.0 <= requested_intensity <= 1.0
            ):
                raise ValueError("AO intensity must be between 0 and 1.")
            if atlas_record is None:
                raise ValueError(
                    "The ambient-occlusion target Atlas no longer exists."
                )
            if preparation is None:
                preparation = self._capture_ambient_occlusion_bake_preparation()
            scene_snapshot = preparation.scene_snapshot
            atlas_context = preparation.atlas_context
            snapshot = preparation.snapshot
            target_atlas_context = next(
                (
                    item
                    for item in atlas_context
                    if item.atlas.atlas_id == normalized_atlas_id
                ),
                None,
            )
            if target_atlas_context is None:
                raise ValueError(
                    "The ambient-occlusion target Atlas has no source used by "
                    "the current scene."
                )
            target_atlas_context = replace(
                target_atlas_context,
                surface_ao_intensity=requested_intensity,
            )
            atlas_context = tuple(
                target_atlas_context
                if item.atlas.atlas_id == normalized_atlas_id
                else item
                for item in atlas_context
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            message = str(error) or type(error).__name__
            self.job_manager.fail_job(job.job_id, f"Failed: {message}")
            self.texture_atlas_workspace.status_label.setText(
                f"Ambient-occlusion bake failed: {message}"
            )
            QMessageBox.warning(self, "Ambient-occlusion bake failed", message)
            return

        thread = _SurfaceAmbientOcclusionBakeThread(
            scene_snapshot,
            target_atlas_context,
            atlas_context,
            self,
        )
        runtime = _SurfaceAmbientOcclusionBakeRuntime(
            atlas_id=normalized_atlas_id,
            job_id=job.job_id,
            thread=thread,
            snapshot=snapshot,
        )
        self._surface_ao_bake_runtimes[normalized_atlas_id] = runtime
        self.job_manager.set_cancel_callback(
            job.job_id,
            lambda target_atlas_id=normalized_atlas_id: (
                self._cancel_surface_ambient_occlusion_bake(target_atlas_id)
            ),
        )
        thread.progress.connect(
            partial(
                self._handle_surface_ambient_occlusion_bake_progress,
                normalized_atlas_id,
                job.job_id,
                thread,
            )
        )
        thread.finished.connect(
            partial(
                self._handle_surface_ambient_occlusion_bake_finished,
                normalized_atlas_id,
                job.job_id,
                thread,
            )
        )
        self.job_manager.update_job(
            job.job_id,
            stage="Preparing ambient occlusion (1%)",
        )
        self.texture_atlas_workspace.status_label.setText(
            f"Preparing ambient occlusion for {atlas_name}."
        )
        thread.start()

    @staticmethod
    def _build_surface_ao_atlas_layout_signature(
        atlas: TextureAtlasRecord,
    ) -> tuple[object, ...]:
        """Identify only Atlas fields which affect a surface AO layout."""

        return (
            atlas.atlas_id,
            int(atlas.resolution),
            tuple(
                (
                    placement.object_id,
                    placement.texture_path,
                    int(placement.texture_resolution),
                    int(placement.x),
                    int(placement.y),
                    int(placement.size),
                    placement.packing_mode,
                    placement.slot_half,
                    placement.slot_quadrant,
                )
                for placement in atlas.placements
            ),
        )

    def _build_surface_ao_atlas_context_signature(
        self,
        required_source_ids: Sequence[str],
    ) -> tuple[tuple[object, ...], ...]:
        """Identify stable export-binding order and relevant Atlas layouts."""

        required_ids = {
            str(source_id).strip()
            for source_id in required_source_ids
            if str(source_id).strip()
        }
        return tuple(
            self._build_surface_ao_atlas_layout_signature(atlas)
            for atlas in self.texture_atlas_workspace.get_data().atlases
            if atlas.placements
            and any(
                placement.object_id in required_ids for placement in atlas.placements
            )
        )

    def _surface_ambient_occlusion_snapshot_is_current(
        self,
        snapshot: _SurfaceAmbientOcclusionBakeSnapshot,
        atlas_id: str,
    ) -> bool:
        """Reject results built from a superseded scene or Atlas layout."""

        if self._viewer_preview_revision != snapshot.viewer_revision:
            return False
        if (
            self._build_viewer_preview_dependency_signature()
            != snapshot.dependency_signature
        ):
            return False
        if (
            tuple(sorted(self._build_atlas_surface_source_ids().items()))
            != snapshot.surface_source_signature
        ):
            return False
        current_atlas_context = self._build_surface_ao_atlas_context_signature(
            snapshot.required_source_ids
        )
        return bool(
            any(
                signature and signature[0] == atlas_id
                for signature in current_atlas_context
            )
            and current_atlas_context == snapshot.atlas_context_signature
        )

    def _handle_surface_ambient_occlusion_bake_progress(
        self,
        atlas_id: str,
        job_id: str,
        thread: _SurfaceAmbientOcclusionBakeThread,
        message: str,
    ) -> None:
        """Forward one current worker update into the shared Jobs window."""

        runtime = self._surface_ao_bake_runtimes.get(atlas_id)
        stage = str(message).strip()
        if (
            self._is_shutdown
            or runtime is None
            or runtime.job_id != job_id
            or runtime.thread is not thread
            or runtime.cancel_requested
            or not stage
        ):
            return
        self.job_manager.update_job(job_id, stage=stage)
        self.texture_atlas_workspace.status_label.setText(stage)

    def _handle_surface_ambient_occlusion_bake_finished(
        self,
        atlas_id: str,
        job_id: str,
        thread: _SurfaceAmbientOcclusionBakeThread,
    ) -> None:
        """Commit one successful current snapshot and finalize its job row."""

        runtime = self._surface_ao_bake_runtimes.get(atlas_id)
        if runtime is None or runtime.job_id != job_id or runtime.thread is not thread:
            thread.deleteLater()
            return
        self._surface_ao_bake_runtimes.pop(atlas_id, None)
        self.job_manager.set_cancel_callback(job_id, None)
        try:
            if self._is_shutdown or runtime.cancel_requested or thread.was_cancelled:
                self.job_manager.mark_cancelled(job_id)
                if not self._is_shutdown:
                    self.texture_atlas_workspace.status_label.setText(
                        "Ambient-occlusion bake cancelled."
                    )
                return
            if thread.error_message is not None:
                self._report_surface_ambient_occlusion_bake_failure(
                    job_id,
                    thread.error_message,
                )
                return
            result = thread.result
            if result is None:
                self._report_surface_ambient_occlusion_bake_failure(
                    job_id,
                    "The AO worker finished without a result.",
                )
                return
            if result.atlas_id != atlas_id:
                self._report_surface_ambient_occlusion_bake_failure(
                    job_id,
                    "The AO worker returned a result for a different Atlas.",
                )
                return
            if not self._surface_ambient_occlusion_snapshot_is_current(
                runtime.snapshot,
                atlas_id,
            ):
                self._report_surface_ambient_occlusion_bake_failure(
                    job_id,
                    "The scene or Atlas changed during the bake; the stale "
                    "result was discarded.",
                )
                return
            self.job_manager.update_job(
                job_id,
                stage="Saving ambient occlusion (99%)",
            )
            try:
                image_path = (
                    self.texture_atlas_workspace.commit_surface_ambient_occlusion_bake(
                        atlas_id,
                        result.ambient_occlusion,
                        result.geometry_signature,
                    )
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                self._report_surface_ambient_occlusion_bake_failure(
                    job_id,
                    str(error) or type(error).__name__,
                )
                return
            if result.preview_model is not None:
                try:
                    image_revision = _build_surface_ao_file_revision(image_path)
                except OSError:
                    self._surface_ao_preview_cache.pop(atlas_id, None)
                else:
                    cached = _SurfaceAmbientOcclusionPreviewCacheEntry(
                        snapshot=runtime.snapshot,
                        geometry_signature=result.geometry_signature,
                        image_revision=image_revision,
                        model=result.preview_model,
                    )
                    self._surface_ao_preview_cache[atlas_id] = cached
                    selected_atlas = self.texture_atlas_workspace.selected_atlas
                    if (
                        self.texture_atlas_workspace.is_ambient_occlusion_preview_active
                        and selected_atlas is not None
                        and selected_atlas.atlas_id == atlas_id
                    ):
                        self._cancel_surface_ambient_occlusion_preview_preparations()
                        self._show_surface_ambient_occlusion_preview(
                            atlas_id,
                            selected_atlas.name,
                            cached,
                        )
            self.job_manager.complete_job(job_id, "Ambient occlusion baked")
        finally:
            thread.deleteLater()

    def _report_surface_ambient_occlusion_bake_failure(
        self,
        job_id: str,
        message: str,
    ) -> None:
        """Show one asynchronous AO failure and preserve existing bake data."""

        normalized_message = str(message).strip() or "Unknown AO bake error."
        self.job_manager.fail_job(job_id, f"Failed: {normalized_message}")
        self.texture_atlas_workspace.status_label.setText(
            f"Ambient-occlusion bake failed: {normalized_message}"
        )
        QMessageBox.warning(
            self,
            "Ambient-occlusion bake failed",
            normalized_message,
        )

    def _cancel_surface_ambient_occlusion_bake(self, atlas_id: str) -> bool:
        """Request cancellation of one Atlas bake without touching siblings."""

        runtime = self._surface_ao_bake_runtimes.get(str(atlas_id))
        if runtime is None or runtime.cancel_requested:
            return runtime is not None
        runtime.cancel_requested = True
        runtime.thread.requestInterruption()
        self.texture_atlas_workspace.status_label.setText(
            "Cancelling ambient-occlusion bake..."
        )
        return True

    def _cancel_and_join_surface_ambient_occlusion_bakes(self) -> None:
        """Interrupt and join every AO thread before retiring its project."""

        runtimes = tuple(self._surface_ao_bake_runtimes.values())
        for runtime in runtimes:
            runtime.cancel_requested = True
            self.job_manager.mark_cancelled(runtime.job_id)
            runtime.thread.requestInterruption()
        for runtime in runtimes:
            while runtime.thread.isRunning():
                runtime.thread.wait(SURFACE_AO_SHUTDOWN_WAIT_MILLISECONDS)
        self._surface_ao_bake_runtimes.clear()

    def _cancel_and_join_surface_texture_tiling_preparations(self) -> None:
        """Join local tiling workers before their Surface state is replaced."""

        threads = tuple(self._surface_texture_tiling_threads.values())
        for thread in threads:
            thread.requestInterruption()
        for thread in threads:
            while thread.isRunning():
                thread.wait(SURFACE_AO_SHUTDOWN_WAIT_MILLISECONDS)
        self._surface_texture_tiling_threads.clear()

    # ### GLB export ###
    def _handle_glb_export_clicked(self) -> None:
        self._cancel_active_canvas_surface_edit()
        self._commit_pending_canvas_surface_mesh_update()
        self._commit_pending_wall_vertex_update()
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
        self._finish_level_transform_drag()
        self._commit_pending_level_transform_update()
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
        generated_model, _dependency_signature = validated_build

        try:
            exported_path = export_glb_file(generated_model, export_path)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Export failed", str(error))
            return

        self.workspace_tabs.setCurrentWidget(self.canvas_viewer_workspace)
        if not self.texture_atlas_workspace.is_ambient_occlusion_preview_active:
            # The export model intentionally excludes levels whose Include
            # option is off. Keep that model out of the interactive viewer,
            # whose independent level list controls visibility instead.
            self._ensure_viewer_preview_current(preserve_camera=True)
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
            current_image_changed = self.canvas.refresh_blueprint_image_if_stale()
            current_revision = self.canvas.get_blueprint_image_revision()
            if current_image_changed:
                refreshed_image_size = self.canvas.get_image_size_pixels()
                if (
                    refreshed_image_size is not None
                    and refreshed_image_size != current_level.image_size_pixels
                ):
                    current_level.image_size_pixels = refreshed_image_size
                    geometry_dimensions_changed = True
                self._update_blueprint_name_label()
            if current_revision is not None:
                self._level_blueprint_image_revisions[current_level.index] = (
                    current_revision
                )

        if include_exported_levels:
            known_level_indices = {level.index for level in self.levels}
            self._level_blueprint_image_revisions = {
                level_index: revision
                for level_index, revision in (
                    self._level_blueprint_image_revisions.items()
                )
                if level_index in known_level_indices
            }
            for level in self.levels:
                if (
                    (
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
                    self._level_blueprint_image_revisions[level.index] = revision_before
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
                self._level_blueprint_image_revisions[level.index] = revision_after
                if image_size != level.image_size_pixels:
                    level.image_size_pixels = image_size
                    geometry_dimensions_changed = True

        if geometry_dimensions_changed:
            self._cancel_pending_canvas_surface_mesh_update()
            self._cancel_pending_wall_vertex_update()
            self._cancel_pending_doorway_mesh_update(clear_outline=True)
            self._clear_canvas_undo_history()
            self._sync_canvas_wall_mirror_state()
            self._mark_viewer_preview_dirty(preserve_camera=False)

    def _handle_workspace_tab_changed(self, tab_index: int) -> None:
        selected_widget = self.workspace_tabs.widget(tab_index)
        is_atlas_workspace = selected_widget is self.texture_atlas_workspace
        is_full_width_workspace = bool(
            is_atlas_workspace
            or selected_widget
            in (
                self.merged_generation_workspace,
                self.settings_widget,
            )
        )
        self.side_panel.setVisible(not is_full_width_workspace)
        if selected_widget is self.merged_generation_workspace:
            self.merged_generation_workspace.refresh_file_backed_previews()
        elif selected_widget is self.scene_3d_workspace:
            self.viewer.focus_navigation()
        elif is_atlas_workspace:
            self._sync_atlas_object_texture_sources()
            self._schedule_atlas_draw_call_estimate()
        if is_full_width_workspace:
            return

        if selected_widget not in (
            self.canvas_viewer_workspace,
            self.scene_3d_workspace,
        ):
            return

        if not self._viewer_preview_is_active():
            self._refresh_blueprint_file_dependencies(include_exported_levels=False)
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

    def _handle_placeable_objects_changed_for_atlases(
        self,
        _placeable_objects: object,
    ) -> None:
        """Refresh active placement choices and clear vanished selections."""

        selected_source_id = self.texture_atlas_workspace.selected_object_id
        self._handle_generation_data_changed_for_atlases(_placeable_objects)
        if (
            selected_source_id is not None
            and self.texture_atlas_workspace.selected_object_id != selected_source_id
        ):
            self._sync_canvas_selection_to_current_atlas_source()

    # ### Generated-object placement ###
    def _handle_object_placement_requested(
        self,
        operation_id: str,
    ) -> None:
        """Arm the shared 3D scene for one Generation-owned request token."""

        exact_operation_id = str(operation_id)
        if self._is_shutdown or not exact_operation_id.strip():
            return
        previous_state = self.generation.get_existing_object_placement_request_state(
            exact_operation_id
        )
        preview_model = (
            None
            if previous_state is None
            else self.generation.get_generated_object_model(previous_state[0])
        )
        self._begin_direct_object_placement(
            _DirectObjectPlacementSession(
                request_id=exact_operation_id,
                generation_request_token=exact_operation_id,
            ),
            preview_model=preview_model,
        )

    def _handle_generation_object_place_requested(self) -> None:
        """Place the newest generated batch or stage the next batch's anchor."""

        placeable_ids = self.generation.get_latest_generation_batch_placeable_ids()
        latest_batch_is_active = (
            self.generation.latest_generation_batch_has_active_members()
        )
        if not latest_batch_is_active and (
            self.generation.has_unsubmitted_generation_mask()
            or self.generation.latest_generation_batch_is_fully_placed()
        ):
            placeable_ids = ()
        self._pending_generation_placement_anchor = None
        request_id = f"generation-placement-{uuid.uuid4().hex}"
        self._begin_direct_object_placement(
            _DirectObjectPlacementSession(
                request_id=request_id,
                placeable_ids=placeable_ids,
                accepts_next_generation_batch=not bool(placeable_ids),
            ),
            preview_count=max(1, len(placeable_ids)),
        )

    def _begin_direct_object_placement(
        self,
        session: _DirectObjectPlacementSession,
        *,
        preview_model: GeneratedModel | None = None,
        preview_count: int = 1,
    ) -> None:
        """Activate the existing local or detached scene and arm its floor picker."""

        self._cancel_direct_object_placement()
        self._direct_object_placement_session = session
        if self._external_scene_3d_host.is_active:
            scene_window = self._external_scene_3d_host.window
            scene_window.show()
            scene_window.raise_()
            scene_window.activateWindow()
        else:
            scene_index = self.workspace_tabs.indexOf(self.scene_3d_workspace)
            if scene_index >= 0:
                self.workspace_tabs.setCurrentIndex(scene_index)
        self._ensure_viewer_preview_current(preserve_camera=True)
        preview_meshes = (
            None
            if preview_model is None
            else (preview_model.mesh,)
        )
        if self.viewer.begin_object_placement(
            session.request_id,
            preview_meshes=preview_meshes,
            preview_count=preview_count,
        ):
            self.viewer.focus_navigation()
            return
        self._direct_object_placement_session = None
        if session.generation_request_token is not None:
            self.generation.cancel_object_placement_request(
                session.generation_request_token
            )

    def _handle_direct_object_placement_selected(
        self,
        request_id: str,
        raw_candidate: object,
    ) -> None:
        """Commit one floor hover to an existing object, a batch, or the next batch."""

        session = self._direct_object_placement_session
        if (
            session is None
            or session.request_id != str(request_id)
            or not isinstance(raw_candidate, SceneObjectPlacementCandidate)
        ):
            return
        self._direct_object_placement_session = None
        if session.accepts_next_generation_batch and not session.placeable_ids:
            anchor = raw_candidate.world_positions[0]
            self._pending_generation_placement_anchor = (
                SceneObjectPlacementCandidate(
                    level_index=raw_candidate.level_index,
                    world_positions=(anchor,),
                )
            )
            self.generation.status_label.setText(
                "Placement selected. The next generated object batch will appear there."
            )
            return

        if session.generation_request_token is not None:
            placement = self._build_generated_object_placement_from_world(
                raw_candidate.level_index,
                raw_candidate.world_positions[0],
            )
            if placement is not None:
                self._commit_generation_owned_placement_request(
                    session.generation_request_token,
                    placement,
                )
            return

        self._commit_placeable_object_group(
            session.placeable_ids,
            raw_candidate,
        )

    def _commit_generation_owned_placement_request(
        self,
        request_id: str,
        placement: GeneratedObjectPlacement,
    ) -> bool:
        """Preserve the historical undo semantics for one Atlas Place request."""

        previous_state = self.generation.get_existing_object_placement_request_state(
            request_id
        )
        previous_atlas_placements = (
            self._capture_canvas_atlas_placements(previous_state[0])
            if previous_state is not None
            else ()
        )
        if not self.generation.set_active_object_placement(request_id, placement):
            self.generation.cancel_object_placement_request(request_id)
            return False
        if previous_state is None or self._is_restoring_canvas_undo:
            return True
        object_id, previous_placement = previous_state
        current_placement = self.generation.get_generated_object_placement(object_id)
        if current_placement != previous_placement:
            self._record_canvas_undo_state(
                _CanvasPlacedObjectUndoState(
                    object_id=object_id,
                    placement=previous_placement,
                    atlas_placements=previous_atlas_placements,
                    restore_atlas_bindings=True,
                )
            )
        return True

    def _commit_placeable_object_group(
        self,
        placeable_ids: Sequence[str],
        candidate: SceneObjectPlacementCandidate,
    ) -> bool:
        """Apply ordered preview positions to active jobs or completed objects."""

        normalized_ids = tuple(str(value).strip() for value in placeable_ids)
        if (
            not normalized_ids
            or len(normalized_ids) != len(candidate.world_positions)
            or any(not value for value in normalized_ids)
        ):
            return False
        placements = tuple(
            self._build_generated_object_placement_from_world(
                candidate.level_index,
                position,
            )
            for position in candidate.world_positions
        )
        if any(placement is None for placement in placements):
            return False

        completed_object_ids = set(self.generation.get_generated_object_ids())
        previous_states: list[
            tuple[
                str,
                GeneratedObjectPlacement | None,
                tuple[tuple[str, TextureAtlasPlacement], ...],
            ]
        ] = []
        next_placements: list[GeneratedObjectPlacement] = []
        for placeable_id, raw_placement in zip(
            normalized_ids,
            placements,
            strict=True,
        ):
            assert raw_placement is not None
            previous_state = self.generation.get_placeable_object_placement_state(
                placeable_id
            )
            previous_placement = (
                None if previous_state is None else previous_state[1]
            )
            placement = raw_placement
            if previous_placement is not None:
                placement = replace(
                    placement,
                    rotation_degrees=previous_placement.rotation_degrees,
                    scale=previous_placement.scale,
                )
            previous_states.append(
                (
                    placeable_id,
                    previous_placement,
                    (
                        self._capture_canvas_atlas_placements(placeable_id)
                        if placeable_id in completed_object_ids
                        else ()
                    ),
                )
            )
            next_placements.append(placement)

        changed_ids: list[str] = []
        for (placeable_id, _previous_placement, _atlas), placement in zip(
            previous_states,
            next_placements,
            strict=True,
        ):
            if not self.generation.restore_placeable_object_placement(
                placeable_id,
                placement,
                emit_change_signals=False,
            ):
                for (
                    changed_id,
                    changed_previous_placement,
                    _changed_atlas,
                ) in reversed(previous_states[: len(changed_ids)]):
                    self.generation.restore_placeable_object_placement(
                        changed_id,
                        changed_previous_placement,
                        emit_change_signals=False,
                    )
                self.generation.status_label.setText(
                    "The object group could not be placed; its previous "
                    "positions were restored."
                )
                return False
            changed_ids.append(placeable_id)

        self.generation.publish_placeable_object_placement_changes(changed_ids)
        undo_states: list[_CanvasPlacedObjectUndoState] = []
        for (
            placeable_id,
            previous_placement,
            previous_atlas_placements,
        ), placement in zip(
            previous_states,
            next_placements,
            strict=True,
        ):
            if placement != previous_placement and not self._is_restoring_canvas_undo:
                undo_states.append(
                    _CanvasPlacedObjectUndoState(
                        object_id=placeable_id,
                        placement=previous_placement,
                        atlas_placements=previous_atlas_placements,
                        restore_atlas_bindings=True,
                    )
                )
        if undo_states:
            self._record_canvas_undo_state(
                undo_states[0]
                if len(undo_states) == 1
                else _CanvasPlacedObjectGroupUndoState(tuple(undo_states))
            )
        changed_count = len(changed_ids)
        self.generation.status_label.setText(
            f"Placed {changed_count} generated object"
            + ("s." if changed_count != 1 else ".")
        )
        return True

    def _build_generated_object_placement_from_world(
        self,
        level_index: int,
        world_position: Sequence[float],
    ) -> GeneratedObjectPlacement | None:
        """Convert one exact world floor hit to persistent level-relative state."""

        level = next(
            (candidate for candidate in self.levels if candidate.index == level_index),
            None,
        )
        base_z = build_level_base_z_lookup(self.levels).get(level_index)
        if level is None or base_z is None:
            return None
        try:
            world_x, world_y, world_z = tuple(float(value) for value in world_position)
            image_x, image_y = level_world_to_image_xy(level, world_x, world_y)
            return GeneratedObjectPlacement(
                level_index=level_index,
                image_x=image_x,
                image_y=image_y,
                height_offset_meters=world_z - float(base_z),
            )
        except (TypeError, ValueError, OverflowError):
            return None

    def _handle_generation_batch_started_for_placement(
        self,
        raw_placeable_ids: object,
    ) -> None:
        """Bind an armed or already-picked future placement to a new batch."""

        if not isinstance(raw_placeable_ids, tuple):
            return
        placeable_ids = tuple(str(value).strip() for value in raw_placeable_ids)
        if not placeable_ids or any(not value for value in placeable_ids):
            return
        pending_anchor = self._pending_generation_placement_anchor
        if pending_anchor is not None:
            self._pending_generation_placement_anchor = None
            candidate = build_scene_object_placement_group_candidate(
                pending_anchor.level_index,
                pending_anchor.world_positions[0],
                len(placeable_ids),
            )
            self._commit_placeable_object_group(placeable_ids, candidate)
            return

        session = self._direct_object_placement_session
        if session is None or not session.accepts_next_generation_batch:
            return
        self._direct_object_placement_session = replace(
            session,
            placeable_ids=placeable_ids,
            accepts_next_generation_batch=False,
        )
        self.viewer.update_object_placement_preview_count(len(placeable_ids))

    def _handle_direct_object_placement_cancelled(self, request_id: str) -> None:
        """Forget only the current picker and its Generation-owned request."""

        session = self._direct_object_placement_session
        if session is None or session.request_id != str(request_id):
            return
        self._direct_object_placement_session = None
        if session.generation_request_token is not None:
            self.generation.cancel_object_placement_request(
                session.generation_request_token
            )

    def _handle_object_placement_operation_finished(
        self,
        operation_id: str,
    ) -> None:
        """Retarget a batch picker, or close one exact finished request."""

        self._discard_canvas_placement_undo_object_id(operation_id)
        session = self._direct_object_placement_session
        if session is None:
            return
        normalized_operation_id = str(operation_id)
        if session.generation_request_token == normalized_operation_id:
            self._cancel_direct_object_placement()
            return
        if normalized_operation_id not in session.placeable_ids:
            return
        next_ids = self.generation.get_latest_generation_batch_placeable_ids()
        if not next_ids:
            self._cancel_direct_object_placement()
            return
        if len(next_ids) != len(session.placeable_ids):
            self.viewer.update_object_placement_preview_count(len(next_ids))
        self._direct_object_placement_session = replace(
            session,
            placeable_ids=next_ids,
        )

    def _cancel_direct_object_placement(self) -> None:
        """Disarm the scene picker without allowing stale request callbacks."""

        session = self._direct_object_placement_session
        self._direct_object_placement_session = None
        self.viewer.cancel_object_placement(notify=False)
        if session is not None and session.generation_request_token is not None:
            self.generation.cancel_object_placement_request(
                session.generation_request_token
            )

    def _handle_generated_object_completed_for_canvas(
        self,
        raw_record: object,
        _generated_model: object,
    ) -> None:
        """Reveal a newly placed object and refit the Canvas 3D view."""

        if not isinstance(raw_record, GeneratedObjectRecord):
            return
        operation_id = self.generation.get_generation_operation_id_for_object(
            raw_record.object_id
        )
        placement_undo_was_migrated = bool(
            operation_id is not None
            and self._migrate_canvas_placement_undo_operation(
                operation_id,
                raw_record.object_id,
            )
        )
        if raw_record.placement is None:
            return
        if not placement_undo_was_migrated:
            self._record_canvas_undo_state(
                _CanvasPlacedObjectUndoState(
                    object_id=raw_record.object_id,
                    placement=None,
                    restore_atlas_bindings=True,
                )
            )
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

    def _capture_canvas_atlas_placements(
        self,
        source_id: str,
    ) -> tuple[tuple[str, TextureAtlasPlacement], ...]:
        """Capture every current Atlas occurrence of one source ID."""

        normalized_source_id = str(source_id).strip()
        return tuple(
            (atlas.atlas_id, placement)
            for atlas in self.texture_atlas_workspace.get_data().atlases
            for placement in atlas.placements
            if placement.object_id == normalized_source_id
        )

    def _discard_canvas_placement_undo_object_id(
        self,
        object_id: str,
    ) -> None:
        """Remove one retired object from single and grouped placement history."""

        normalized_object_id = str(object_id).strip()
        retained_states: list[_CanvasUndoState] = []
        for state in self._canvas_undo_stack:
            if isinstance(state, _CanvasPlacedObjectUndoState):
                if state.object_id != normalized_object_id:
                    retained_states.append(state)
                continue
            if isinstance(state, _CanvasPlacedObjectGroupUndoState):
                members = tuple(
                    member
                    for member in state.members
                    if member.object_id != normalized_object_id
                )
                if members:
                    retained_states.append(
                        members[0]
                        if len(members) == 1
                        else replace(state, members=members)
                    )
                continue
            retained_states.append(state)
        self._canvas_undo_stack = retained_states

    def _migrate_canvas_placement_undo_operation(
        self,
        operation_id: str,
        object_id: str,
    ) -> bool:
        """Move in-flight placement history onto its committed object ID."""

        normalized_operation_id = str(operation_id).strip()
        normalized_object_id = str(object_id).strip()
        migrated = False
        for index, state in enumerate(self._canvas_undo_stack):
            if isinstance(state, _CanvasPlacedObjectUndoState):
                if state.object_id != normalized_operation_id:
                    continue
                self._canvas_undo_stack[index] = (
                    self._migrate_canvas_placed_object_undo_state(
                        state,
                        normalized_object_id,
                    )
                )
                migrated = True
                continue
            if isinstance(state, _CanvasPlacedObjectGroupUndoState):
                members = tuple(
                    self._migrate_canvas_placed_object_undo_state(
                        member,
                        normalized_object_id,
                    )
                    if member.object_id == normalized_operation_id
                    else member
                    for member in state.members
                )
                if members != state.members:
                    self._canvas_undo_stack[index] = replace(
                        state,
                        members=members,
                    )
                    migrated = True
        return migrated

    @staticmethod
    def _migrate_canvas_placed_object_undo_state(
        state: _CanvasPlacedObjectUndoState,
        object_id: str,
    ) -> _CanvasPlacedObjectUndoState:
        """Retarget one pending-operation undo member to its stable object ID."""

        return replace(
            state,
            object_id=object_id,
            atlas_placements=tuple(
                (
                    atlas_id,
                    replace(placement, object_id=object_id),
                )
                for atlas_id, placement in state.atlas_placements
            ),
        )

    def _handle_placed_object_removal_requested(self, object_id: str) -> None:
        """Remove a Canvas placement and unassign its texture from Atlases."""

        normalized_object_id = str(object_id).strip()
        existing_placement = self.generation.get_generated_object_placement(
            normalized_object_id
        )
        atlas_placements = self._capture_canvas_atlas_placements(normalized_object_id)
        if self.generation.remove_generated_object_placement(normalized_object_id):
            if existing_placement is not None:
                self._record_canvas_undo_state(
                    _CanvasPlacedObjectUndoState(
                        object_id=normalized_object_id,
                        placement=existing_placement,
                        atlas_placements=atlas_placements,
                        restore_atlas_bindings=True,
                    )
                )
            self._discard_desired_canvas_object(normalized_object_id)
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
                scale=existing_placement.scale,
            )
        except (TypeError, ValueError, OverflowError):
            self._schedule_viewer_preview_refresh(preserve_camera=True)
            return
        # The gizmo already updated its independent preview root. Emitting the
        # ordinary placement signals here would serialize and rebuild both 3D
        # previews, even though only this retained transform changed.
        canvas_was_current = (
            self._canvas_viewer_preview_revision == self._viewer_preview_revision
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
        if placement != existing_placement:
            self._record_canvas_undo_state(
                _CanvasPlacedObjectUndoState(
                    object_id=normalized_object_id,
                    placement=existing_placement,
                    selected_object_ids=self._desired_canvas_object_ids,
                    active_object_id=self._desired_canvas_object_id,
                )
            )

        revision = self._mark_viewer_preview_dirty(
            preserve_camera=True,
            affects_draw_call_estimate=False,
        )
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
            self._viewer_preview_dependency_signature = dependency_signature_after
            self._viewer_preview_dependency_signature_revision = revision
            return
        self._queue_viewer_preview_refresh()

    def _handle_placed_object_scales_changed(self, raw_updates: object) -> None:
        """Persist one wheel gesture for every selected placed object."""

        try:
            updates = tuple(raw_updates)  # type: ignore[arg-type]
        except TypeError:
            self._schedule_viewer_preview_refresh(preserve_camera=True)
            return
        normalized_updates: list[
            tuple[str, GeneratedObjectPlacement, GeneratedObjectPlacement]
        ] = []
        seen_object_ids: set[str] = set()
        try:
            for raw_update in updates:
                object_id, raw_scale = tuple(raw_update)
                normalized_object_id = str(object_id).strip()
                if (
                    not normalized_object_id
                    or normalized_object_id in seen_object_ids
                ):
                    raise ValueError("Placed-object scale IDs must be unique.")
                existing = self.generation.get_generated_object_placement(
                    normalized_object_id
                )
                if existing is None:
                    raise ValueError("A scaled object is no longer placed.")
                replacement = replace(existing, scale=raw_scale)
                seen_object_ids.add(normalized_object_id)
                if replacement != existing:
                    normalized_updates.append(
                        (normalized_object_id, existing, replacement)
                    )
        except (TypeError, ValueError, OverflowError):
            self._schedule_viewer_preview_refresh(preserve_camera=True)
            return
        if not normalized_updates:
            return

        canvas_was_current = (
            self._canvas_viewer_preview_revision == self._viewer_preview_revision
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

        committed_updates: list[
            tuple[str, GeneratedObjectPlacement, GeneratedObjectPlacement]
        ] = []
        for object_id, existing, replacement in normalized_updates:
            if not self.generation.update_generated_object_placement(
                object_id,
                replacement,
                emit_change_signals=False,
            ):
                for committed_id, previous, _next in reversed(committed_updates):
                    self.generation.update_generated_object_placement(
                        committed_id,
                        previous,
                        emit_change_signals=False,
                    )
                self._schedule_viewer_preview_refresh(preserve_camera=True)
                return
            committed_updates.append((object_id, existing, replacement))

        if not self._is_restoring_canvas_undo:
            undo_members = tuple(
                _CanvasPlacedObjectUndoState(
                    object_id=object_id,
                    placement=existing,
                    selected_object_ids=self._desired_canvas_object_ids,
                    active_object_id=self._desired_canvas_object_id,
                )
                for object_id, existing, _replacement in committed_updates
            )
            self._record_canvas_undo_state(
                undo_members[0]
                if len(undo_members) == 1
                else _CanvasPlacedObjectGroupUndoState(undo_members)
            )

        revision = self._mark_viewer_preview_dirty(
            preserve_camera=True,
            affects_draw_call_estimate=False,
        )
        dependency_signature_after = (
            self._build_viewer_preview_dependency_signature()
            if canvas_was_current
            else None
        )
        expected_placements = {
            object_id: replacement
            for object_id, _existing, replacement in committed_updates
        }
        if (
            dependency_signature_before is not None
            and dependency_signature_after is not None
            and self._dependency_change_is_only_target_placements(
                dependency_signature_before,
                dependency_signature_after,
                expected_placements,
            )
        ):
            self._canvas_viewer_preview_revision = revision
            self._viewer_preview_dependency_signature = dependency_signature_after
            self._viewer_preview_dependency_signature_revision = revision
            return
        self._queue_viewer_preview_refresh()

    def _handle_generated_object_deleted_for_canvas(
        self,
        object_id: str,
    ) -> None:
        """Remove any deleted placed object from the Canvas preview."""

        normalized_object_id = str(object_id).strip()
        self._discard_canvas_placement_undo_object_id(normalized_object_id)
        self._discard_desired_canvas_object(normalized_object_id)
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
            record.object_id == normalized_object_id and record.placement is not None
            for record in self.generation.get_data().generated_objects
        ):
            self._schedule_viewer_preview_refresh(preserve_camera=True)

    # ### Texture Atlas synchronization ###
    def _is_atlas_surface_texture_source_id(self, source_id: object) -> bool:
        """Classify reserved IDs without shadowing a real generated object."""

        normalized_id = str(source_id).strip()
        return bool(
            is_atlas_wall_texture_source_id(normalized_id)
            and normalized_id not in self.generation.get_placeable_object_names_by_id()
        )

    def _handle_surface_texture_data_changed_for_atlases(
        self,
        _surface_texture_data: object,
    ) -> None:
        """Refresh Atlas surface choices without reloading unchanged thumbnails."""

        if (
            self._is_automatically_assigning_atlas_textures
            or self._is_assigning_surface_texture_from_atlas
        ):
            self._atlas_generation_signature = None
            return
        self._sync_atlas_object_texture_sources()

    def _handle_selected_atlas_changed_for_automatic_assignment(
        self,
        _selected_atlas: object,
    ) -> None:
        """Retry pending textures and refresh an active AO-only preview."""

        self._refresh_scene_atlas_texture_requirements()
        self._schedule_surface_ambient_occlusion_preview_refresh()

    def _handle_surface_texture_assignments_removed_for_atlases(
        self,
        raw_assignment_ids: object,
    ) -> None:
        """Remove explicitly deleted surface textures from every Atlas."""

        if not isinstance(raw_assignment_ids, tuple | list):
            return
        assignment_ids = tuple(
            assignment_id
            for assignment_id in (str(value).strip() for value in raw_assignment_ids)
            if assignment_id
        )
        if not assignment_ids:
            return
        removed_source_ids: set[str] = set()
        removable_assignment_ids: list[str] = []
        for assignment_id in assignment_ids:
            source_id = build_atlas_wall_texture_source_id(assignment_id)
            if not self._is_atlas_surface_texture_source_id(source_id):
                continue
            removed_source_ids.add(source_id)
            removable_assignment_ids.append(assignment_id)
        selected_source_was_removed = (
            any(
                source_id in removed_source_ids
                for source_id in (
                    self.texture_atlas_workspace.selected_surface_texture_ids
                )
            )
        )
        if removable_assignment_ids:
            self.texture_atlas_workspace.remove_deleted_wall_texture_assignments(
                tuple(removable_assignment_ids)
            )
        if self._selected_atlas_surface_source_id in removed_source_ids:
            self._selected_atlas_surface_source_id = None
            self._set_atlas_canvas_surface_highlights(())
            self._sync_atlas_green_outline_to_canvas_highlight(None)
        for assignment_id in removable_assignment_ids:
            self._atlas_wall_texture_source_ids.discard(
                build_atlas_wall_texture_source_id(assignment_id)
            )
        self._atlas_generation_signature = None
        self._sync_atlas_object_texture_sources()
        if selected_source_was_removed:
            self._sync_canvas_selection_to_current_atlas_source()

    def _sync_canvas_selection_to_current_atlas_source(self) -> None:
        """Reconcile Canvas selection after an Atlas source disappears."""

        surface_source_ids = self.texture_atlas_workspace.selected_surface_texture_ids
        if surface_source_ids:
            self._handle_atlas_surface_textures_selected(surface_source_ids)
            return
        object_ids = self.texture_atlas_workspace.selected_object_texture_ids
        if object_ids:
            self._handle_atlas_object_textures_selected(object_ids)
            return
        self._selected_atlas_surface_source_id = None
        self._desired_canvas_object_id = None
        self._desired_canvas_object_ids = ()
        self._set_atlas_canvas_surface_highlights(())
        self._sync_atlas_green_outline_to_canvas_highlight(None)
        self._is_syncing_canvas_scene_selection = True
        try:
            self.viewer.select_placed_object(None)
        finally:
            self._is_syncing_canvas_scene_selection = False

    def _handle_atlas_object_texture_selected(self, object_id: str) -> None:
        """Select the matching placed Canvas object without changing tabs."""

        normalized_id = str(object_id).strip()
        if not normalized_id:
            return
        self._selected_atlas_surface_source_id = None
        self._atlas_surface_assignment_target_ids = ()
        self._desired_canvas_object_id = normalized_id
        self._desired_canvas_object_ids = (normalized_id,)
        self._desired_canvas_surface_ids = ()
        self._discard_staged_stair_edit(clear_selection=True)
        self._sync_surface_generation_selection(())
        self.generation.select_generated_object(normalized_id)
        self._set_atlas_canvas_surface_highlights(())
        self._sync_atlas_green_outline_to_canvas_highlight(None)
        self._is_syncing_canvas_scene_selection = True
        try:
            self.viewer.select_placed_object(None)
            self.viewer.select_canvas_opening(None)
            self.viewer.select_wall_target(None)
            self.viewer.set_selected_canvas_stair_part_ids(())
            self.viewer.select_placed_object(normalized_id)
        finally:
            self._is_syncing_canvas_scene_selection = False

    def _handle_atlas_object_textures_selected(self, object_ids: object) -> None:
        """Mirror Atlas multi-selection onto placed Canvas objects."""

        normalized_ids = tuple(
            dict.fromkeys(
                normalized
                for value in object_ids
                if (normalized := str(value).strip())
            )
        )
        if not normalized_ids:
            if self.texture_atlas_workspace.selected_surface_texture_ids:
                return
            self._desired_canvas_object_id = None
            self._desired_canvas_object_ids = ()
            self._is_syncing_canvas_scene_selection = True
            try:
                self.viewer.select_placed_object(None)
            finally:
                self._is_syncing_canvas_scene_selection = False
            return
        active_id = self.texture_atlas_workspace.selected_object_texture_id
        if active_id not in normalized_ids:
            active_id = normalized_ids[-1]
        assert active_id is not None
        if len(normalized_ids) == 1:
            self._handle_atlas_object_texture_selected(active_id)
            return
        self._selected_atlas_surface_source_id = None
        self._atlas_surface_assignment_target_ids = ()
        self._desired_canvas_object_id = active_id
        self._desired_canvas_object_ids = normalized_ids
        self._desired_canvas_surface_ids = ()
        self._discard_staged_stair_edit(clear_selection=True)
        self._sync_surface_generation_selection(())
        self.generation.select_generated_object(active_id)
        self._set_atlas_canvas_surface_highlights(())
        self._sync_atlas_green_outline_to_canvas_highlight(None)
        self._is_syncing_canvas_scene_selection = True
        try:
            self.viewer.select_canvas_opening(None)
            self.viewer.select_wall_target(None)
            self.viewer.set_selected_canvas_stair_part_ids(())
            self.viewer.set_selected_placed_object_ids(
                normalized_ids,
                active_object_id=active_id,
            )
        finally:
            self._is_syncing_canvas_scene_selection = False

    def _handle_atlas_surface_texture_selected(self, source_id: str) -> None:
        """Highlight every Canvas surface using the selected texture family."""

        assignment_id = get_atlas_wall_texture_assignment_id(source_id)
        if assignment_id is None:
            self._selected_atlas_surface_source_id = None
            self._set_atlas_canvas_surface_highlights(())
            self._sync_atlas_green_outline_to_canvas_highlight(None)
            return
        assignment = self.surface_texture_generation.get_assignment(assignment_id)
        if assignment is None:
            self._selected_atlas_surface_source_id = None
            self._set_atlas_canvas_surface_highlights(())
            self._sync_atlas_green_outline_to_canvas_highlight(None)
            return
        self._selected_atlas_surface_source_id = source_id
        self._set_atlas_canvas_surface_highlights(assignment.surface_ids)
        self._sync_atlas_green_outline_to_canvas_highlight(source_id)

    def _handle_atlas_surface_textures_selected(self, source_ids: object) -> None:
        """Highlight the union of surfaces using selected Atlas textures."""

        normalized_ids = tuple(
            dict.fromkeys(
                normalized
                for value in source_ids
                if (normalized := str(value).strip())
            )
        )
        if len(normalized_ids) == 1:
            self._handle_atlas_surface_texture_selected(normalized_ids[0])
            return
        if not normalized_ids:
            if self.texture_atlas_workspace.selected_object_texture_ids:
                return
            self._selected_atlas_surface_source_id = None
            self._set_atlas_canvas_surface_highlights(())
            self.texture_atlas_workspace.set_green_outline_source_ids(())
            return
        assignments = {
            source_id: self.surface_texture_generation.get_assignment(assignment_id)
            for source_id in normalized_ids
            if (assignment_id := get_atlas_wall_texture_assignment_id(source_id))
            is not None
        }
        surface_ids = tuple(
            dict.fromkeys(
                surface_id
                for assignment in assignments.values()
                if assignment is not None
                for surface_id in assignment.surface_ids
            )
        )
        self._selected_atlas_surface_source_id = (
            self.texture_atlas_workspace.selected_surface_texture_id
        )
        self._set_atlas_canvas_surface_highlights(surface_ids)
        highlighted_ids = set(self.viewer.get_highlighted_canvas_surface_ids())
        highlighted_ids.update(self.viewer.get_highlighted_canvas_stair_part_ids())
        self.texture_atlas_workspace.set_green_outline_source_ids(
            tuple(
                source_id
                for source_id, assignment in assignments.items()
                if assignment is not None
                and any(
                    surface_id in highlighted_ids
                    for surface_id in assignment.surface_ids
                )
            )
        )

    def _handle_atlas_surface_texture_repeat_size_changed(
        self,
        source_id: str,
        repeat_size_m: float,
    ) -> None:
        """Apply one metres-per-repeat value to the selected Surface family."""

        assignment_id = get_atlas_wall_texture_assignment_id(source_id)
        if assignment_id is None:
            return
        try:
            changed = (
                self.surface_texture_generation.set_assignment_texture_repeat_size(
                    assignment_id,
                    repeat_size_m,
                )
            )
        except (TypeError, ValueError):
            self.texture_atlas_workspace.status_label.setText(
                "Texture repeat size must be a positive number."
            )
            return
        if changed:
            self.texture_atlas_workspace.status_label.setText(
                f"Texture repeat size: {repeat_size_m:g} m."
            )

    def _set_atlas_canvas_surface_highlights(
        self,
        surface_ids: Sequence[str],
    ) -> None:
        """Split Atlas highlighting between architecture and stair parts."""

        normalized_ids = tuple(dict.fromkeys(str(value) for value in surface_ids))
        self.viewer.set_highlighted_canvas_surface_ids(
            tuple(
                surface_id
                for surface_id in normalized_ids
                if surface_id in self._canvas_surface_targets_by_id
            )
        )
        self.viewer.set_highlighted_canvas_stair_part_ids(
            tuple(
                surface_id
                for surface_id in normalized_ids
                if surface_id in self._canvas_stair_part_targets_by_id
            )
        )

    def _sync_atlas_green_outline_to_canvas_highlight(
        self,
        source_id: str | None,
    ) -> None:
        """Outline only the texture whose Canvas surfaces glow green."""

        normalized_source_id = (
            None if source_id is None else str(source_id).strip() or None
        )
        outlined_source_ids = (
            (normalized_source_id,)
            if normalized_source_id is not None
            and (
                self.viewer.get_highlighted_canvas_surface_ids()
                or self.viewer.get_highlighted_canvas_stair_part_ids()
            )
            else ()
        )
        self.texture_atlas_workspace.set_green_outline_source_ids(outlined_source_ids)

    def _handle_atlas_object_place_requested(self, object_id: str) -> None:
        """Arm the shared 3D scene picker for one Atlas object."""

        if self.generation.request_placeable_object_placement(object_id):
            return
        self.texture_atlas_workspace.status_label.setText(
            "The selected object is not currently available for placement."
        )

    def _handle_atlas_object_delete_requested(
        self, object_ids: str | tuple[str, ...]
    ) -> None:
        """Confirm once, then delete the snapshotted generated objects."""

        requested_ids = (object_ids,) if isinstance(object_ids, str) else object_ids
        requested_ids = tuple(
            dict.fromkeys(
                object_id.strip()
                for object_id in requested_ids
                if isinstance(object_id, str) and object_id.strip()
            )
        )
        records_by_id = {
            record.object_id: record
            for record in self.generation.get_data().generated_objects
        }
        records = tuple(
            records_by_id[object_id]
            for object_id in requested_ids
            if object_id in records_by_id
        )
        if not records:
            self.texture_atlas_workspace.status_label.setText(
                "The selected generated objects are no longer available."
            )
            return
        dialog_parent = (
            self._external_atlas_host.window
            if self._external_atlas_host.is_active
            else self.texture_atlas_workspace
        )
        if len(records) == 1:
            confirmation_text = (
                f'Permanently delete "{records[0].object_name}", its embedded '
                "textures, and its unreferenced local GLB revisions?"
            )
        else:
            confirmation_text = (
                f"Permanently delete these {len(records)} generated objects, "
                "their embedded textures, and their unreferenced local GLB "
                "revisions?"
            )
        answer = QMessageBox.question(
            dialog_parent,
            "Delete generated objects" if len(records) > 1 else "Delete generated object",
            confirmation_text,
            (
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel
            ),
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        deleted_names: list[str] = []
        failed_names: list[str] = []
        for record in records:
            if self.generation.delete_generated_object(record.object_id):
                deleted_names.append(record.object_name)
            else:
                failed_names.append(record.object_name)
        if len(records) == 1:
            if deleted_names:
                status = f"Deleted generated object: {deleted_names[0]}."
            else:
                generation_status = self.generation.status_label.text().strip()
                detail = (
                    f" {generation_status}"
                    if generation_status
                    else " The object may be busy or unavailable."
                )
                status = "The selected generated object could not be deleted." + detail
        else:
            status = (
                f"Deleted {len(deleted_names)} of {len(records)} selected "
                "generated objects."
            )
            if failed_names:
                status += f" Could not delete: {', '.join(failed_names)}."
        self.texture_atlas_workspace.status_label.setText(status)

    def _handle_atlas_source_remove_requested(
        self,
        source_kind: str,
        source_id: str,
    ) -> None:
        """Remove one selected source from its Atlas and Canvas bindings."""

        normalized_source_id = str(source_id).strip()
        if not normalized_source_id:
            return
        if source_kind == "object":
            previous_state = self.generation.get_placeable_object_placement_state(
                normalized_source_id
            )
            previous_atlas_placements = self._capture_canvas_atlas_placements(
                normalized_source_id
            )
            removed_from_canvas = self.generation.remove_placeable_object_placement(
                normalized_source_id
            )
            removed_from_atlases = (
                self.texture_atlas_workspace.remove_scene_texture_from_atlases(
                    normalized_source_id
                )
            )
            if removed_from_canvas and previous_state is not None:
                stable_id, previous_placement = previous_state
                self._record_canvas_undo_state(
                    _CanvasPlacedObjectUndoState(
                        object_id=stable_id,
                        placement=previous_placement,
                        atlas_placements=previous_atlas_placements,
                        restore_atlas_bindings=True,
                    )
                )
            self._discard_desired_canvas_object(normalized_source_id)
            self._atlas_generation_signature = None
            self._sync_atlas_object_texture_sources()
            if removed_from_canvas or removed_from_atlases:
                self.texture_atlas_workspace.status_label.setText(
                    "Removed the selected object from Canvas and its texture "
                    "from the Atlas."
                )
            else:
                self.texture_atlas_workspace.status_label.setText(
                    "The selected object was not placed or packed."
                )
            return
        if source_kind != "surface":
            return

        assignment_id = get_atlas_wall_texture_assignment_id(normalized_source_id)
        if assignment_id is None:
            return
        self._is_assigning_surface_texture_from_atlas = True
        try:
            removed_from_canvas = (
                self.surface_texture_generation.clear_assignment_surfaces(assignment_id)
            )
        finally:
            self._is_assigning_surface_texture_from_atlas = False
        if not removed_from_canvas:
            self.texture_atlas_workspace.status_label.setText(
                "The selected surface texture could not be removed."
            )
            return
        self.texture_atlas_workspace.remove_scene_texture_from_atlases(
            normalized_source_id
        )
        if self._selected_atlas_surface_source_id == normalized_source_id:
            self._set_atlas_canvas_surface_highlights(())
            self._sync_atlas_green_outline_to_canvas_highlight(None)
        self._atlas_generation_signature = None
        self._sync_atlas_object_texture_sources()
        self.texture_atlas_workspace.status_label.setText(
            "Removed the selected texture from its surfaces and the Atlas."
        )

    def _handle_atlas_surface_texture_delete_requested(
        self,
        source_id: str,
    ) -> None:
        """Confirm and permanently delete one selected Surface texture family."""

        assignment_id = get_atlas_wall_texture_assignment_id(source_id)
        if assignment_id is None:
            self.texture_atlas_workspace.status_label.setText(
                "Select a Surface texture to delete."
            )
            return
        assignment = self.surface_texture_generation.get_assignment(assignment_id)
        if assignment is None:
            self.texture_atlas_workspace.status_label.setText(
                "The selected Surface texture no longer exists."
            )
            return
        answer = QMessageBox.question(
            self,
            "Delete surface texture",
            (
                f"Delete this {assignment.surface_type} texture from all "
                f"{len(assignment.surface_ids)} assigned surface(s)?\n\n"
                f"Texture ID: {assignment.assignment_id}\n\n"
                "Every generated resolution in this texture family will be "
                "deleted."
            ),
            (
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel
            ),
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.surface_texture_generation.delete_assignment_texture(
            assignment.assignment_id
        ):
            self.texture_atlas_workspace.status_label.setText(
                self.surface_texture_generation.status_label.text()
            )
            return
        self.texture_atlas_workspace.status_label.setText(
            "The selected Surface texture is busy or could not be deleted."
        )

    def _handle_atlas_surface_texture_fix_tiling_requested(
        self,
        source_id: str,
    ) -> None:
        """Preview edge-compatible rotated variants of the selected texture."""

        self._start_surface_texture_tiling_repair(source_id)

    def _start_surface_texture_tiling_repair(
        self,
        source_id: str,
    ) -> None:
        """Prepare edge-compatible tiling outside the GUI thread."""

        normalized_source_id = str(source_id).strip()
        assignment_id = get_atlas_wall_texture_assignment_id(
            normalized_source_id
        )
        if assignment_id is None:
            self.texture_atlas_workspace.status_label.setText(
                "Select a loaded Surface texture to fix its tiling."
            )
            return
        if normalized_source_id in self._surface_texture_tiling_threads:
            self.texture_atlas_workspace.status_label.setText(
                "This Surface texture tiling preview is already being prepared."
            )
            return
        try:
            preparation_snapshot = (
                self.surface_texture_generation
                .snapshot_assignment_tiling_repair(
                    assignment_id,
                    method=SURFACE_TILING_MODE_EDGE_VARIANTS,
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self.texture_atlas_workspace.status_label.setText(
                "Texture tiling repair could not start: "
                + (str(error) or type(error).__name__)
            )
            return

        thread = _SurfaceTextureTilingPreparationThread(
            preparation_snapshot,
            self,
        )
        self._surface_texture_tiling_threads[normalized_source_id] = thread
        thread.finished.connect(
            partial(
                self._handle_surface_texture_tiling_prepared,
                normalized_source_id,
                thread,
            )
        )
        self.texture_atlas_workspace.status_label.setText(
            "Preparing a 3 x 3 edge-compatible variant comparison..."
        )
        thread.start()

    def _handle_surface_texture_tiling_prepared(
        self,
        source_id: str,
        thread: _SurfaceTextureTilingPreparationThread,
    ) -> None:
        """Show the completed comparison and commit it only after acceptance."""

        current_thread = self._surface_texture_tiling_threads.get(source_id)
        if current_thread is not thread:
            thread.deleteLater()
            return
        self._surface_texture_tiling_threads.pop(source_id, None)
        candidate = thread.result
        error_message = thread.error_message
        was_cancelled = thread.was_cancelled
        thread.deleteLater()
        if self._is_shutdown:
            return
        if candidate is None:
            if was_cancelled:
                self.texture_atlas_workspace.status_label.setText(
                    "Texture tiling repair cancelled; the original was kept."
                )
            else:
                self.texture_atlas_workspace.status_label.setText(
                    "Texture tiling repair failed: "
                    + (error_message or "the preview could not be prepared.")
                )
            return
        if (
            self.surface_texture_generation.get_assignment(
                candidate.assignment.assignment_id
            )
            != candidate.assignment
        ):
            self.texture_atlas_workspace.status_label.setText(
                "Texture tiling preview discarded because the Surface "
                "texture changed while it was being prepared."
            )
            return
        if not candidate.has_changes:
            self.texture_atlas_workspace.status_label.setText(
                "No different tiling preview could be created. The original was kept."
            )
            return

        dialog_parent = (
            self._external_atlas_host.window
            if self._external_atlas_host.is_active
            else self.texture_atlas_workspace
        )
        try:
            dialog = SurfaceTextureTilingPreviewDialog(
                candidate.before_preview_png,
                candidate.after_preview_png,
                parent=dialog_parent,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            self.texture_atlas_workspace.status_label.setText(
                f"Texture tiling preview failed: {error}"
            )
            return
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.texture_atlas_workspace.status_label.setText(
                "Texture tiling repair cancelled; the original was kept."
            )
            return
        self._commit_surface_texture_tiling_repair(source_id, candidate)

    def _commit_surface_texture_tiling_repair(
        self,
        source_id: str,
        candidate: PreparedSurfaceTextureTilingRepair,
    ) -> None:
        """Commit a tiling revision and update Atlas pixels only if they changed."""

        revision: SurfaceTextureTilingRevision | None = None
        repaired_was_activated = False
        try:
            revision = (
                self.surface_texture_generation.stage_assignment_tiling_repair(
                    candidate
                )
            )
            repaired_was_activated = (
                self.surface_texture_generation
                .activate_assignment_tiling_revision(
                    revision,
                    repaired=True,
                    emit_signals=False,
                )
            )
            repaired_assignment = (
                self.surface_texture_generation.get_assignment(
                    revision.repaired_assignment.assignment_id
                )
            )
            if repaired_assignment != revision.repaired_assignment:
                raise RuntimeError(
                    "The repaired Surface texture could not be activated."
                )
            if revision.created_asset_paths:
                candidate_sources = (
                    self._build_atlas_wall_texture_sources_for_assignment(
                        repaired_assignment,
                        source_id,
                    )
                )
                atlas_updated = self.texture_atlas_workspace.transition_object_packing(
                    source_id,
                    candidate_sources,
                    commit_callback=lambda: (
                        self.surface_texture_generation.get_assignment(
                            repaired_assignment.assignment_id
                        )
                        == repaired_assignment
                    ),
                )
                if not atlas_updated:
                    atlas_status = (
                        self.texture_atlas_workspace.status_label.text().strip()
                    )
                    raise RuntimeError(
                        atlas_status
                        or "The Atlas could not accept the repaired texture."
                    )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            if revision is not None and repaired_was_activated:
                try:
                    self.surface_texture_generation.activate_assignment_tiling_revision(
                        revision,
                        repaired=False,
                        emit_signals=False,
                    )
                except (OSError, RuntimeError, TypeError, ValueError):
                    pass
            if revision is not None:
                self.surface_texture_generation.discard_assignment_tiling_revision(
                    revision
                )
            self.texture_atlas_workspace.status_label.setText(
                "Texture tiling repair failed; the original texture and Atlas "
                f"were kept: {error}"
            )
            return

        assert revision is not None
        self._record_canvas_undo_state(
            _SurfaceTextureTilingUndoState(revision=revision)
        )
        self._atlas_generation_signature = None
        self.surface_texture_generation.publish_assignment_tiling_change(
            "Surface texture tiling repaired."
        )
        self.texture_atlas_workspace.status_label.setText(
            "Fixed the selected Surface texture tiling. Press Ctrl+Z to "
            "restore the original revision."
        )

    def _build_atlas_wall_texture_sources_for_assignment(
        self,
        assignment: SurfaceTextureAssignment,
        source_id: str,
    ) -> tuple[AtlasObjectTextureSource, ...]:
        """Resolve every exact source needed to transition existing placements."""

        if assignment.texture_variants:
            requested_resolutions: tuple[int | None, ...] = tuple(
                variant.resolution for variant in assignment.texture_variants
            )
        else:
            packed_resolutions = tuple(
                dict.fromkeys(
                    placement.texture_resolution
                    for atlas in self.texture_atlas_workspace.get_data().atlases
                    for placement in atlas.placements
                    if placement.object_id == source_id
                )
            )
            requested_resolutions = packed_resolutions or (None,)
        sources: list[AtlasObjectTextureSource] = []
        for resolution in requested_resolutions:
            source = self._build_atlas_wall_texture_source(
                assignment,
                resolution,
            )
            if source is None:
                resolution_label = (
                    "active"
                    if resolution is None
                    else f"{resolution} x {resolution}"
                )
                raise ValueError(
                    "The repaired Surface texture is missing its "
                    f"{resolution_label} Atlas source."
                )
            sources.append(source)
        return tuple(sources)

    def _handle_atlas_surface_assign_requested(self, source_id: str) -> None:
        """Apply and pack one surface texture as a single user transaction."""

        assignment_id = get_atlas_wall_texture_assignment_id(source_id)
        if assignment_id is None:
            return
        assignment = self.surface_texture_generation.get_assignment(assignment_id)
        if assignment is None:
            return
        target_surface_ids = self._atlas_surface_assignment_targets(assignment)
        if not target_surface_ids:
            self.texture_atlas_workspace.status_label.setText(
                "Select at least one Canvas surface before assigning a texture."
            )
            return
        if not self._atlas_surface_targets_match_assignment(
            assignment,
            target_surface_ids,
        ):
            self.texture_atlas_workspace.status_label.setText(
                "The selected texture can only be assigned to surfaces of the "
                f"same type ({assignment.surface_type})."
            )
            return
        source_ids_to_remove = self._atlas_surface_sources_displaced_by(
            assignment.assignment_id,
            target_surface_ids,
        )

        def commit_surface_assignment() -> bool:
            self._is_assigning_surface_texture_from_atlas = True
            try:
                return self.surface_texture_generation.apply_assignment_texture(
                    assignment.assignment_id,
                    target_surface_ids,
                )
            finally:
                self._is_assigning_surface_texture_from_atlas = False

        if self.texture_atlas_workspace.is_source_assigned_to_any_atlas(source_id):
            assigned = commit_surface_assignment()
        elif self.texture_atlas_workspace.can_assign_source_to_selected_atlas(
            source_id,
            source_ids_to_remove=source_ids_to_remove,
        ):
            assigned = self.texture_atlas_workspace.assign_source_to_selected_atlas(
                source_id,
                commit_callback=commit_surface_assignment,
                source_ids_to_remove=source_ids_to_remove,
            )
        else:
            if self.texture_atlas_workspace.selected_atlas is None:
                self.texture_atlas_workspace.status_label.setText(
                    "Select or create an Atlas before assigning the texture."
                )
                return
            dialog_parent = (
                self._external_atlas_host.window
                if self._external_atlas_host.is_active
                else self.texture_atlas_workspace
            )
            answer = QMessageBox.question(
                dialog_parent,
                "Atlas full",
                "not space in atlas for the texture, create a new atlas?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            assigned = self.texture_atlas_workspace.create_atlas_and_assign_source(
                source_id,
                commit_callback=commit_surface_assignment,
                source_ids_to_remove=source_ids_to_remove,
            )

        if not assigned:
            self.texture_atlas_workspace.status_label.setText(
                "The texture could not be assigned to the selected surfaces."
            )
            return
        self._atlas_surface_assignment_target_ids = ()
        self._atlas_generation_signature = None
        self._sync_atlas_object_texture_sources()

    def _atlas_surface_sources_displaced_by(
        self,
        selected_assignment_id: str,
        target_surface_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Return packed textures that will have no surfaces after assignment."""

        target_ids = set(target_surface_ids)
        displaced_source_ids: list[str] = []
        for assignment in self.surface_texture_generation.get_assignments():
            if (
                assignment.assignment_id == selected_assignment_id
                or not assignment.surface_ids
                or not set(assignment.surface_ids).issubset(target_ids)
            ):
                continue
            source_id = build_atlas_wall_texture_source_id(assignment.assignment_id)
            if self._is_atlas_surface_texture_source_id(source_id):
                displaced_source_ids.append(source_id)
        return tuple(displaced_source_ids)

    def _atlas_surface_assignment_targets(
        self,
        _assignment: SurfaceTextureAssignment,
    ) -> tuple[str, ...]:
        """Resolve the explicit or currently visible Canvas surface choice."""

        if self._atlas_surface_assignment_target_ids:
            return self._atlas_surface_assignment_target_ids
        selected_surface_ids = self.viewer.get_selected_canvas_surface_ids()
        if selected_surface_ids:
            return selected_surface_ids
        return self.viewer.get_selected_canvas_stair_part_ids()

    def _atlas_surface_targets_match_assignment(
        self,
        assignment: SurfaceTextureAssignment,
        target_surface_ids: Sequence[str],
    ) -> bool:
        """Reject stale or mixed-type targets before changing Atlas PNGs."""

        return self.surface_texture_generation.assignment_targets_are_valid(
            assignment.assignment_id,
            target_surface_ids,
        )

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
        self.texture_atlas_workspace.refresh_regenerated_object_texture(object_id)

    def _handle_generated_object_generated_for_atlases(
        self,
        raw_record: object,
        _generated_model: object,
    ) -> None:
        """Pulse one newly generated object or texture until it is clicked."""

        object_id = getattr(raw_record, "object_id", None)
        if not isinstance(object_id, str) or not object_id.strip():
            return
        self.texture_atlas_workspace.mark_sources_new((object_id,))

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

        if self.texture_atlas_workspace.is_ambient_occlusion_preview_active:
            return
        if self._is_atlas_surface_texture_source_id(object_id):
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
                (None if symmetry is None else getattr(symmetry, "packing_mode", None)),
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
                f"The selected object's 3D preview could not be loaded: {error}"
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

    def _handle_atlas_placeable_object_preview_requested(
        self,
        object_id: str,
    ) -> None:
        """Preview generated geometry that does not yet have a texture."""

        if self.texture_atlas_workspace.is_ambient_occlusion_preview_active:
            return
        generated_model = self.generation.get_generated_object_model(object_id)
        if generated_model is None:
            self._clear_atlas_object_preview()
            self._append_atlas_preview_status(
                "The selected object's 3D preview is unavailable."
            )
            return
        symmetry = self.generation.get_object_symmetric_division(object_id)
        preview_key = (
            str(object_id),
            "geometry_only",
            id(generated_model),
            None if symmetry is None else symmetry.orientation,
            None if symmetry is None else symmetry.plane_coordinate,
        )
        if (
            self._atlas_preview_variant_key == preview_key
            and self.atlas_object_preview_viewer.model is not None
        ):
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
        self._atlas_preview_variant_key = preview_key

    def _show_atlas_surface_texture_preview(
        self,
        source_id: str,
        texture_resolution: int,
    ) -> None:
        """Display one exact surface texture on an upright square plane."""

        assignment_id = get_atlas_wall_texture_assignment_id(source_id)
        try:
            normalized_resolution = int(texture_resolution)
        except (TypeError, ValueError, OverflowError):
            normalized_resolution = -1
        assignment = (
            None
            if assignment_id is None
            else self.surface_texture_generation.get_assignment(assignment_id)
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
        if assignment is None or asset_path is None or normalized_resolution <= 0:
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
            model = build_texture_preview_plane_model(source.load_texture_rgba())
        except (OSError, TypeError, ValueError) as error:
            self._clear_atlas_object_preview()
            self._append_atlas_preview_status(
                f"The selected surface texture preview could not be loaded: {error}"
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

        assignment_id = (
            get_atlas_wall_texture_assignment_id(object_id)
            if self._is_atlas_surface_texture_source_id(object_id)
            else None
        )
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
        resolve_variant = self.generation.resolve_atlas_texture_image_variant_for_record
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
                replacement_record.pipeline.get(FACE_EDIT_TEXTURE_STALE_PIPELINE_KEY)
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
        self.texture_atlas_workspace.set_scene_texture_source_ids(required_source_ids)
        if not automatically_assign or self._is_automatically_assigning_atlas_textures:
            return
        self._automatically_assign_scene_textures()

    def _build_required_scene_atlas_source_ids(self) -> tuple[str, ...]:
        """Return texture source IDs used by included scene content."""

        included_level_indices = {
            level.index for level in self.levels if level.include_in_export
        }
        required_ids: list[str] = []
        for object_id in self.generation.get_generated_object_ids():
            placement = self.generation.get_generated_object_placement(object_id)
            if (
                placement is not None
                and placement.level_index in included_level_indices
                and (
                    object_id in self._atlas_available_source_ids
                    or self.generation.has_generated_object_texture_variants(object_id)
                )
            ):
                required_ids.append(object_id)
        exported_surface_ids = {
            surface.surface_id for surface in build_fixed_surfaces(self.levels)
        }
        exported_surface_ids.update(
            semantic_id
            for semantic_id, target in self._canvas_stair_part_targets_by_id.items()
            if target.level_indices
            and all(
                level_index in included_level_indices
                for level_index in target.level_indices
            )
        )
        required_ids.extend(
            source_id
            for surface_id, source_id in (
                self._build_atlas_surface_source_ids().items()
            )
            if surface_id in exported_surface_ids
        )
        return tuple(dict.fromkeys(required_ids))

    def _automatically_assign_scene_textures(self) -> None:
        """Pack all currently unassigned scene textures in one Atlas update."""

        if self._is_automatically_assigning_atlas_textures:
            return
        settings = self._generation_settings
        target_resolution = settings.automatic_atlas_texture_resolution
        allow_atlas_creation = settings.automatic_atlas_creation
        sort_by_pbr = settings.automatic_atlas_texture_sort_by_pbr
        use_half_mesh_texture_prefix = settings.use_half_mesh_texture_prefix
        attempt_key = self._build_automatic_atlas_assignment_key(
            target_resolution,
            allow_atlas_creation,
            sort_by_pbr,
            use_half_mesh_texture_prefix,
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
                    use_half_mesh_texture_prefix=(use_half_mesh_texture_prefix),
                    allow_atlas_creation=allow_atlas_creation,
                )
            )
            if not assigned_source_ids:
                return
            self._atlas_generation_signature = None
            self._sync_atlas_object_texture_sources()
            for source_id in assigned_source_ids:
                if not self._is_atlas_surface_texture_source_id(source_id):
                    self._refresh_placed_object_texture_if_needed(source_id)
        finally:
            self._is_automatically_assigning_atlas_textures = False
            self._last_automatic_atlas_assignment_key = (
                self._build_automatic_atlas_assignment_key(
                    target_resolution,
                    allow_atlas_creation,
                    sort_by_pbr,
                    use_half_mesh_texture_prefix,
                )
            )

    def _build_automatic_atlas_assignment_key(
        self,
        target_resolution: int,
        allow_atlas_creation: bool,
        sort_by_pbr: bool,
        use_half_mesh_texture_prefix: bool,
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
            bool(allow_atlas_creation),
            bool(sort_by_pbr),
            bool(use_half_mesh_texture_prefix),
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
            assignment_id = (
                get_atlas_wall_texture_assignment_id(source_id)
                if self._is_atlas_surface_texture_source_id(source_id)
                else None
            )
            if assignment_id is not None:
                assignment = self.surface_texture_generation.get_assignment(
                    assignment_id
                )
                if assignment is None:
                    self._rollback_automatic_atlas_texture_resolutions(applied_changes)
                    return False
                if not assignment.texture_variants:
                    continue
                previous_resolution = assignment.selected_texture_resolution
                if previous_resolution is None:
                    self._rollback_automatic_atlas_texture_resolutions(applied_changes)
                    return False
                if previous_resolution == target_resolution:
                    continue
                select_resolution = (
                    self.surface_texture_generation.select_assignment_texture_resolution
                )
                if not select_resolution(
                    assignment_id,
                    target_resolution,
                ):
                    self._rollback_automatic_atlas_texture_resolutions(applied_changes)
                    return False
                applied_changes.append(("surface", assignment_id, previous_resolution))
                continue

            active_variant = self.generation.get_active_texture_variant(source_id)
            if active_variant is None:
                self._rollback_automatic_atlas_texture_resolutions(applied_changes)
                return False
            previous_resolution = active_variant.resolution
            if previous_resolution == target_resolution:
                continue
            if not self.generation.select_object_texture_resolution(
                source_id,
                target_resolution,
            ):
                self._rollback_automatic_atlas_texture_resolutions(applied_changes)
                return False
            applied_changes.append(("object", source_id, previous_resolution))
        return True

    def _rollback_automatic_atlas_texture_resolutions(
        self,
        applied_changes: list[tuple[str, str, int]],
    ) -> None:
        """Best-effort restore owning workspaces after a rejected batch."""

        for source_kind, source_id, previous_resolution in reversed(applied_changes):
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
            "an Atlas before GLB export. Review these unpacked textures in "
            "the Atlas tab:\n\n" + "\n".join(detail_lines),
        )
        return True

    def _atlas_texture_source_display_name(self, source_id: str) -> str:
        """Resolve a human-readable name for one export-blocking source."""

        for record in self.generation.get_data().generated_objects:
            if record.object_id == source_id:
                return record.object_name
        assignment_id = (
            get_atlas_wall_texture_assignment_id(source_id)
            if self._is_atlas_surface_texture_source_id(source_id)
            else None
        )
        if assignment_id is not None:
            assignment = self.surface_texture_generation.get_assignment(assignment_id)
            if assignment is not None:
                return assignment.display_name or (
                    f"{assignment.surface_type.title()} texture"
                )
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
        placeable_object_names_by_id = (
            self.generation.get_placeable_object_names_by_id()
        )
        generated_object_ids = self.generation.get_generated_object_ids()
        generated_object_id_lookup = set(generated_object_ids)
        scene_bound_source_ids = list(
            self.generation.get_scene_bound_placeable_object_ids()
        )
        signature_items.append(
            (
                "placeable_objects",
                tuple(placeable_object_names_by_id.items()),
            )
        )
        signature_items.append(
            ("deletable_objects", tuple(generated_object_ids))
        )
        for object_id in generated_object_ids:
            variant = self.generation.get_active_texture_variant(object_id)
            symmetry = self.generation.get_object_symmetric_division(object_id)
            texture_variant_signature = (
                self.generation.get_texture_variant_dependency_signature(object_id)
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

        surface_assignments = list(self.surface_texture_generation.get_assignments())
        surface_texture_entries: list[AtlasSurfaceTextureEntry] = []
        for assignment in surface_assignments:
            variant_signature: list[tuple[object, ...]] = []
            candidate_variants = (
                tuple(assignment.texture_variants)
                if assignment.texture_variants
                else (None,)
            )
            for texture_variant in candidate_variants:
                resolution = (
                    None if texture_variant is None else texture_variant.resolution
                )
                logical_path = (
                    assignment.asset_path
                    if texture_variant is None
                    else texture_variant.asset_path
                )
                physical_map_paths = (
                    self.surface_texture_generation.get_assignment_map_asset_paths(
                        assignment.assignment_id,
                        resolution,
                    )
                )
                physical_path = physical_map_paths.get(ATLAS_MAP_BASE_COLOR)
                logical_map_paths = (
                    {ATLAS_MAP_BASE_COLOR: logical_path}
                    if texture_variant is None
                    else texture_variant.map_asset_paths
                )
                map_signature = tuple(
                    (
                        map_type,
                        map_asset_path,
                        _build_local_file_revision(physical_map_paths.get(map_type)),
                    )
                    for map_type, map_asset_path in (logical_map_paths.items())
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
                    assignment.surface_type,
                    assignment.display_name,
                    assignment.asset_path,
                    assignment.selected_texture_resolution,
                    assignment.texture_width,
                    assignment.texture_height,
                    assignment.surface_ids,
                    assignment.texture_repeat_size_m,
                    assignment.tiling_fix_needed,
                    tuple(variant_signature),
                )
            )
            surface_source_id = build_atlas_wall_texture_source_id(
                assignment.assignment_id
            )
            if surface_source_id not in generated_object_id_lookup:
                surface_texture_entries.append(
                    AtlasSurfaceTextureEntry(
                        source_id=surface_source_id,
                        display_name=(
                            assignment.display_name
                            or f"{assignment.surface_type.title()} texture"
                        ),
                        surface_usage_count=len(assignment.surface_ids),
                        surface_type=assignment.surface_type,
                        texture_repeat_size_m=assignment.texture_repeat_size_m,
                        tiling_fix_needed=assignment.tiling_fix_needed,
                    )
                )
                if assignment.surface_ids:
                    scene_bound_source_ids.append(surface_source_id)
                source_content_paths[surface_source_id] = tuple(
                    (
                        item[0],
                        item[1],
                        tuple((map_item[0], map_item[1]) for map_item in item[3]),
                    )
                    for item in variant_signature
                )
                source_content_revisions[surface_source_id] = tuple(
                    (
                        item[0],
                        item[2],
                        tuple((map_item[0], map_item[2]) for map_item in item[3]),
                    )
                    for item in variant_signature
                )
        normalized_scene_bound_source_ids = tuple(dict.fromkeys(scene_bound_source_ids))
        signature_items.append(
            ("scene_bound_sources", normalized_scene_bound_source_ids)
        )
        signature = tuple(signature_items)
        if signature == self._atlas_generation_signature:
            self.texture_atlas_workspace.set_scene_bound_source_ids(
                normalized_scene_bound_source_ids
            )
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
                if _build_atlas_source_base_path_signature(
                    previous_paths
                ) != _build_atlas_source_base_path_signature(current_paths):
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

        self._atlas_pending_source_content_refresh_ids.update(failed_source_ids)
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
            placeable_objects=placeable_object_names_by_id,
            deletable_object_ids=generated_object_ids,
            surface_texture_entries=surface_texture_entries,
            variant_resolver=resolve_variant,
            selectability_resolver=is_variant_selectable,
        )
        self.texture_atlas_workspace.set_scene_bound_source_ids(
            normalized_scene_bound_source_ids
        )
        # The backing field and source-ID prefix retain their legacy "wall"
        # names so existing project files and integrations remain compatible.
        self._atlas_wall_texture_source_ids = set(surface_sources)
        selected_surface_source_ids = (
            self.texture_atlas_workspace.selected_surface_texture_ids
        )
        if (
            selected_surface_source_ids
            and self._selected_atlas_surface_source_id
            in selected_surface_source_ids
        ):
            self._handle_atlas_surface_textures_selected(selected_surface_source_ids)
        zero_usage_cleanup_failed = False
        for assignment in surface_assignments:
            if assignment.surface_ids:
                continue
            source_id = build_atlas_wall_texture_source_id(assignment.assignment_id)
            if not self._is_atlas_surface_texture_source_id(source_id):
                continue
            if self.texture_atlas_workspace.is_source_assigned_to_any_atlas(source_id):
                removed_count = (
                    self.texture_atlas_workspace.remove_scene_texture_from_atlases(
                        source_id
                    )
                )
                zero_usage_cleanup_failed = bool(
                    zero_usage_cleanup_failed or removed_count <= 0
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
        self._atlas_available_source_ids = set(available_source_ids)
        self._atlas_generation_signature = (
            None if source_build_failed or zero_usage_cleanup_failed else signature
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
                texture_path=str(getattr(variant, "texture_asset_relative_path")),
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
                    None if symmetry is None else str(getattr(symmetry, "orientation"))
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
        legacy_logical_map_paths = {ATLAS_MAP_BASE_COLOR: assignment.asset_path}
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
            if map_type in logical_map_paths and map_type in physical_map_paths
        )
        atlas_logical_map_paths = {
            map_type: f"surface_textures/{logical_map_paths[map_type]}"
            for map_type in live_map_types
        }
        atlas_physical_map_paths = {
            map_type: physical_map_paths[map_type] for map_type in live_map_types
        }
        try:
            if supports_resolution_changes:
                texture_resolution = int(requested_resolution)
                fit_to_square = False
                variant = assignment.texture_variant_for_resolution(texture_resolution)
                if variant is None:
                    return None
                asset_path = variant.asset_path
            else:
                with Image.open(physical_path) as image:
                    natural_resolution = choose_atlas_texture_resolution(
                        image.width, image.height
                    )
                texture_resolution = (
                    natural_resolution if resolution is None else int(resolution)
                )
                fit_to_square = True
                asset_path = assignment.asset_path
            surface_count = len(assignment.surface_ids)
            return load_atlas_object_texture_source(
                object_id=build_atlas_wall_texture_source_id(assignment.assignment_id),
                object_name=(
                    assignment.display_name
                    or f"{assignment.surface_type.title()} texture"
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
                surface_usage_count=surface_count,
            )
        except (AttributeError, OSError, TypeError, ValueError):
            return None

    def _handle_external_scene_3d_window_closed(self) -> None:
        """Return the shared scene to its tab and clear its display setting."""

        combo = self.settings_widget.scene_3d_display_screen_combo
        if combo.currentIndex() != 0:
            combo.setCurrentIndex(0)
            return
        self._handle_generation_settings_changed()

    def _handle_external_generation_window_closed(self) -> None:
        """Return Generation to its tab and clear its display setting."""

        combo = self.settings_widget.generation_display_screen_combo
        if combo.currentIndex() != 0:
            combo.setCurrentIndex(0)
            return
        self._handle_generation_settings_changed()

    def _handle_external_atlas_window_closed(self) -> None:
        """Return Atlas to its tab and synchronize the display setting."""

        combo = self.settings_widget.atlas_display_screen_combo
        if combo.currentIndex() != 0:
            combo.setCurrentIndex(0)
            return
        self._handle_generation_settings_changed()

    def _handle_external_atlas_viewer_restored(
        self,
        restored_viewer: object,
    ) -> None:
        """Restore the Atlas tab whenever its external host releases it."""

        if restored_viewer is self.texture_atlas_workspace and not self._is_shutdown:
            self._restore_atlas_workspace_tab()

    def _handle_external_scene_3d_workspace_restored(
        self,
        restored_viewer: object,
    ) -> None:
        """Restore the 3D scene tab whenever its external host releases it."""

        if restored_viewer is self.scene_3d_workspace and not self._is_shutdown:
            self._restore_workspace_tab(self.scene_3d_workspace)

    def _handle_external_generation_workspace_restored(
        self,
        restored_viewer: object,
    ) -> None:
        """Restore Generation whenever its external host releases it."""

        if (
            restored_viewer is self.merged_generation_workspace
            and not self._is_shutdown
        ):
            self._restore_workspace_tab(self.merged_generation_workspace)

    def _apply_scene_3d_display_screen(
        self,
        screen_id: str | None,
    ) -> None:
        """Keep the shared scene maximized on its independently chosen display."""

        if screen_id is None:
            self._external_scene_3d_host.restore()
            self._restore_workspace_tab(self.scene_3d_workspace)
            return
        screen = resolve_fullscreen_3d_viewer_screen(screen_id)
        if screen is None:
            self._external_scene_3d_host.restore()
            self._restore_workspace_tab(self.scene_3d_workspace)
            return
        tab_index = self._prepare_workspace_for_detachment(
            self.scene_3d_workspace
        )
        self._external_scene_3d_host.show_on_screen(
            self.scene_3d_workspace,
            screen,
        )
        self._remove_detached_workspace_tab(
            self.scene_3d_workspace,
            tab_index,
        )
        self.viewer.focus_navigation()
        self._ensure_viewer_preview_current(preserve_camera=True)

    def _apply_generation_display_screen(
        self,
        screen_id: str | None,
    ) -> None:
        """Keep Generation maximized on its independently chosen display."""

        if screen_id is None:
            self._external_generation_host.restore()
            self._restore_workspace_tab(self.merged_generation_workspace)
            return
        screen = resolve_fullscreen_3d_viewer_screen(screen_id)
        if screen is None:
            self._external_generation_host.restore()
            self._restore_workspace_tab(self.merged_generation_workspace)
            return
        tab_index = self._prepare_workspace_for_detachment(
            self.merged_generation_workspace
        )
        self._external_generation_host.show_on_screen(
            self.merged_generation_workspace,
            screen,
        )
        self._remove_detached_workspace_tab(
            self.merged_generation_workspace,
            tab_index,
        )
        self.merged_generation_workspace.refresh_file_backed_previews()

    def _apply_jobs_window_screen(self, screen_id: str | None) -> None:
        """Move the persistent Jobs window to its selected display."""

        screen = resolve_fullscreen_3d_viewer_screen(screen_id)
        self.jobs_window.set_target_screen(screen)

    def _apply_atlas_display_screen(self, screen_id: str | None) -> None:
        """Keep the complete Atlas workspace maximized on its chosen display."""

        if screen_id is None:
            self._external_atlas_host.restore()
            self._restore_atlas_workspace_tab()
            return
        screen = resolve_fullscreen_3d_viewer_screen(screen_id)
        if screen is None:
            self._external_atlas_host.restore()
            self._restore_atlas_workspace_tab()
            return
        atlas_tab_index = self._prepare_workspace_for_detachment(
            self.texture_atlas_workspace
        )
        self._external_atlas_host.show_on_screen(
            self.texture_atlas_workspace,
            screen,
        )
        self._remove_detached_workspace_tab(
            self.texture_atlas_workspace,
            atlas_tab_index,
        )
        self._sync_atlas_object_texture_sources()
        self._schedule_atlas_draw_call_estimate()

    def _prepare_workspace_for_detachment(self, workspace: QWidget) -> int:
        """Return one tab slot and move local focus away before detaching it."""

        tab_index = self.workspace_tabs.indexOf(workspace)
        if (
            tab_index >= 0
            and self.workspace_tabs.currentWidget() is workspace
        ):
            self.workspace_tabs.setCurrentWidget(self.canvas_viewer_workspace)
        return tab_index

    def _remove_detached_workspace_tab(
        self,
        workspace: QWidget,
        tab_index: int,
    ) -> None:
        """Remove the placeholder tab left by a successful workspace handoff."""

        if tab_index < 0:
            self._refresh_workspace_tab_indices()
            return
        self.workspace_tabs.removeTab(tab_index)
        self._refresh_workspace_tab_indices()

    def _restore_atlas_workspace_tab(self) -> None:
        """Restore Atlas at its canonical top-level position."""

        self._restore_workspace_tab(self.texture_atlas_workspace)

    def _restore_workspace_tab(self, workspace: QWidget) -> None:
        """Insert a detached workspace in canonical order without focus theft."""

        existing_index = self.workspace_tabs.indexOf(workspace)
        if (
            0 <= existing_index < self.workspace_tabs.count()
            and self.workspace_tabs.widget(existing_index)
            is workspace
        ):
            self._refresh_workspace_tab_indices()
            return

        current_widget = self.workspace_tabs.currentWidget()
        canonical_entries = self._canonical_workspace_tabs()
        target_position = next(
            (
                index
                for index, (candidate, _label) in enumerate(canonical_entries)
                if candidate is workspace
            ),
            len(canonical_entries),
        )
        insertion_index = sum(
            1
            for candidate, _label in canonical_entries[:target_position]
            if self.workspace_tabs.indexOf(candidate) >= 0
        )
        label = next(
            (
                candidate_label
                for candidate, candidate_label in canonical_entries
                if candidate is workspace
            ),
            "Workspace",
        )
        self.workspace_tabs.insertTab(
            insertion_index,
            workspace,
            label,
        )
        self._refresh_workspace_tab_indices()
        if (
            current_widget is not None
            and self.workspace_tabs.indexOf(current_widget) >= 0
            and self.workspace_tabs.currentWidget() is not current_widget
        ):
            self.workspace_tabs.setCurrentWidget(current_widget)

    def _canonical_workspace_tabs(self) -> tuple[tuple[QWidget, str], ...]:
        """Return the stable tab order shared by detach and restore paths."""

        return (
            (self.canvas_viewer_workspace, "Canvas"),
            (self.scene_3d_workspace, "3D scene"),
            (self.texture_atlas_workspace, "Atlas"),
            (self.merged_generation_workspace, "Generation"),
            (self.settings_widget, "Settings"),
        )

    def _set_workspace_tab_index(self, workspace: QWidget, index: int) -> None:
        """Keep compatibility index attributes synchronized after tab moves."""

        if workspace is self.canvas_viewer_workspace:
            self.canvas_workspace_tab_index = index
        elif workspace is self.scene_3d_workspace:
            self.scene_3d_workspace_tab_index = index
        elif workspace is self.texture_atlas_workspace:
            self.atlas_workspace_tab_index = index
        elif workspace is self.merged_generation_workspace:
            self.generation_workspace_tab_index = index
        elif workspace is self.settings_widget:
            self.settings_workspace_tab_index = index

    def _refresh_workspace_tab_indices(self) -> None:
        """Recompute every public tab index after one workspace moves."""

        for workspace, _label in self._canonical_workspace_tabs():
            self._set_workspace_tab_index(
                workspace,
                self.workspace_tabs.indexOf(workspace),
            )

    def _request_hosted_atlas_object_preview(self) -> None:
        """Refresh the selected object in Atlas's embedded 3D preview."""

        if (
            self._external_atlas_host.is_active
            or self.workspace_tabs.currentWidget() is self.texture_atlas_workspace
        ):
            self.texture_atlas_workspace.request_selected_object_preview()

    # ### Shared 3D preview cache ###
    def _canvas_viewer_preview_is_active(self) -> bool:
        return bool(
            self.workspace_tabs.currentWidget() is self.scene_3d_workspace
            or (
                self._external_scene_3d_host.is_active
                and self._external_scene_3d_host.viewer
                is self.scene_3d_workspace
            )
        )

    def _viewer_preview_is_active(self) -> bool:
        return self._canvas_viewer_preview_is_active()

    def _active_viewer_preview_needs_refresh(self) -> bool:
        revision = self._viewer_preview_revision
        ambient_occlusion_override_active = (
            self.texture_atlas_workspace.is_ambient_occlusion_preview_active
        )
        return bool(
            self._canvas_viewer_preview_is_active()
            and not ambient_occlusion_override_active
            and self._canvas_viewer_preview_revision != revision
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
        current_dependency_signature = self._build_viewer_preview_dependency_signature()
        if current_dependency_signature != validated_dependency_signature:
            return False
        revision = self._viewer_preview_revision
        self._viewer_preview_model = generated_model
        self._viewer_preview_model_revision = revision
        self._viewer_preview_dependency_signature = current_dependency_signature
        self._viewer_preview_dependency_signature_revision = revision
        self._canvas_viewer_preview_revision = (
            -1
            if self.texture_atlas_workspace.is_ambient_occlusion_preview_active
            else revision
        )
        self._scheduled_viewer_refresh_preserve_camera = True
        if self._pending_level_transform is not None:
            self._refresh_pending_level_transform_outline()
        self._clear_committed_doorway_outline_if_displayed()
        self._clear_committed_level_transform_outline_if_displayed()
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

        return BlueprintWorkspace._dependency_change_is_only_target_placements(
            signature_before,
            signature_after,
            {object_id: placement},
        )

    @staticmethod
    def _dependency_change_is_only_target_placements(
        signature_before: tuple[object, ...],
        signature_after: tuple[object, ...],
        placements_by_object_id: Mapping[str, GeneratedObjectPlacement],
    ) -> bool:
        """Accept a retained-preview fast path for one exact object group."""

        if signature_before == signature_after:
            return True
        expected_placements = dict(placements_by_object_id)
        if not expected_placements:
            return False
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

        matched_object_ids: set[str] = set()
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
            object_id = str(item_before[0])
            expected_placement = expected_placements.get(object_id)
            if expected_placement is None:
                if item_before != item_after:
                    return False
                continue
            matched_object_ids.add(object_id)
            if item_after[1] != expected_placement:
                return False
            if item_before[:1] + item_before[2:] != item_after[:1] + item_after[2:]:
                return False
        return matched_object_ids == expected_placements.keys()

    def _build_model_with_stable_dependencies(
        self,
        builder: Callable[[], GeneratedModel | None],
    ) -> tuple[GeneratedModel, tuple[object, ...]] | None:
        """Build once and reject a model assembled across file revisions."""

        dependency_signature_before = self._build_viewer_preview_dependency_signature()
        generated_model = builder()
        if generated_model is None:
            return None
        dependency_signature_after = self._build_viewer_preview_dependency_signature()
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
        dependency_signature = self._build_viewer_preview_dependency_signature()
        if dependency_signature == self._viewer_preview_dependency_signature:
            return
        self._mark_viewer_preview_dirty(preserve_camera=preserve_camera)

    # ### Debounced Canvas mesh previews ###
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
        """Make every rendered structural snapshot match the project."""

        self._viewer_doorways_by_level_index = {
            level.index: self._copy_doorways(level.doorways) for level in self.levels
        }
        self._reset_viewer_window_snapshots()
        self._reset_viewer_floor_thickness_snapshots()

    def _reset_viewer_window_snapshots(self) -> None:
        """Make every rendered window snapshot match the loaded project."""

        self._viewer_windows_by_level_index = {
            level.index: self._copy_windows(level.windows) for level in self.levels
        }

    def _reset_viewer_floor_thickness_snapshots(self) -> None:
        """Make rendered floor thicknesses match authoritative level data."""

        self._viewer_floor_thickness_by_level_index = {
            level.index: float(level.floor_thickness_meters) for level in self.levels
        }

    def _commit_viewer_floor_thickness_snapshot(self) -> None:
        """Publish one delayed floor value to future preview builds."""

        level_index = self._pending_floor_thickness_level_index
        self._pending_floor_thickness_level_index = None
        if level_index is None:
            return
        level = next(
            (candidate for candidate in self.levels if candidate.index == level_index),
            None,
        )
        if level is None:
            return
        self._viewer_floor_thickness_by_level_index[level_index] = float(
            level.floor_thickness_meters
        )

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
        """Copy every level for display while substituting committed values."""

        preview_levels: list[LevelData] = []
        for level in self.levels:
            doorway_snapshot = self._viewer_doorways_by_level_index.get(level.index)
            if doorway_snapshot is None:
                doorway_snapshot = self._copy_doorways(level.doorways)
                self._viewer_doorways_by_level_index[level.index] = doorway_snapshot
            window_snapshot = self._viewer_windows_by_level_index.get(level.index)
            if window_snapshot is None:
                window_snapshot = self._copy_windows(level.windows)
                self._viewer_windows_by_level_index[level.index] = window_snapshot
            floor_thickness = self._viewer_floor_thickness_by_level_index.get(
                level.index
            )
            if floor_thickness is None:
                floor_thickness = float(level.floor_thickness_meters)
                self._viewer_floor_thickness_by_level_index[level.index] = (
                    floor_thickness
                )
            preview_level = copy.copy(level)
            # Canvas Include controls govern GLB export only. The scene's own
            # level list independently controls interactive visibility.
            preview_level.include_in_export = True
            preview_level.doorways = list(copy.deepcopy(doorway_snapshot))
            preview_level.windows = list(copy.deepcopy(window_snapshot))
            preview_level.floor_thickness_meters = floor_thickness
            preview_levels.append(preview_level)
        return preview_levels

    def _sync_viewer_scene_levels(
        self,
        *,
        reset_visibility: bool = False,
    ) -> None:
        """Publish independently visible scene levels and object ownership."""

        level_items = tuple(
            (level.index, level.display_name)
            for level in sorted(
                self.levels,
                key=lambda candidate: candidate.index,
                reverse=True,
            )
            if level.vertex_data.vertices
        )
        placed_object_levels = {
            record.object_id: record.placement.level_index
            for record in self.generation.get_data().generated_objects
            if record.placement is not None
        }
        self.viewer.set_canvas_scene_levels(
            level_items,
            placed_object_levels=placed_object_levels,
            reset_visibility=reset_visibility,
        )

    def _set_mesh_edit_update_delay_seconds(
        self,
        delay_seconds: float,
    ) -> None:
        """Apply the shared setting and restart active mesh-edit debounces."""

        normalized_delay = float(delay_seconds)
        if normalized_delay <= 0.0:
            raise ValueError("Mesh edit update delay must be positive.")
        if normalized_delay == self._mesh_edit_update_delay_seconds:
            return

        interval_milliseconds = max(1, round(normalized_delay * 1000.0))
        timers = (
            self._doorway_mesh_update_timer,
            self._canvas_surface_mesh_update_timer,
            self._level_transform_mesh_update_timer,
            self._stair_point_mesh_update_timer,
        )
        active_timers = tuple(timer.isActive() for timer in timers)
        self._mesh_edit_update_delay_seconds = normalized_delay
        for timer, was_active in zip(timers, active_timers, strict=True):
            timer.setInterval(interval_milliseconds)
            if was_active:
                timer.start()

    def _set_wall_vertex_update_delay_seconds(
        self,
        delay_seconds: float,
    ) -> None:
        """Apply the independent wall-vertex rebuild debounce interval."""

        normalized_delay = float(delay_seconds)
        if normalized_delay <= 0.0:
            raise ValueError("Wall vertex update delay must be positive.")
        if normalized_delay == self._wall_vertex_update_delay_seconds:
            return

        was_active = self._wall_vertex_update_timer.isActive()
        self._wall_vertex_update_delay_seconds = normalized_delay
        self._wall_vertex_update_timer.setInterval(
            max(1, round(normalized_delay * 1000.0))
        )
        if was_active:
            self._wall_vertex_update_timer.start()

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
                    self._viewer_doorways_by_level_index[level.index] = next_doorways
                    self._staged_canvas_opening_mesh_update = True
                    self._staged_doorway_mesh_update = True
            if level.index == window_level_index:
                next_windows = self._copy_windows(level.windows)
                if self._viewer_windows_by_level_index.get(level.index) != next_windows:
                    self._viewer_windows_by_level_index[level.index] = next_windows
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
            pre_atlas_scene = self._build_pre_atlas_export_scene()
            materialized_atlases = self.texture_atlas_workspace.prepare_export_atlases(
                pre_atlas_scene.required_source_ids
            )
            if not materialized_atlases:
                return pre_atlas_scene.model
            return apply_texture_atlases_to_export(
                pre_atlas_scene.model,
                materialized_atlases,
                surface_source_ids=pre_atlas_scene.surface_source_ids,
            )
        except (OSError, TypeError, ValueError) as error:
            if failure_title is not None:
                QMessageBox.warning(self, failure_title, str(error))
            return None

    def _build_pre_atlas_export_scene(self) -> _PreAtlasExportScene:
        """Build the exact shared geometry snapshot used by Atlas export."""

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
        return _PreAtlasExportScene(
            model=generated_model,
            placed_models=placed_models,
            surface_source_ids=self._build_atlas_surface_source_ids(),
        )

    def _build_atlas_surface_source_ids(self) -> dict[str, str]:
        """Map each assigned architectural surface to its Atlas source ID."""

        source_ids: dict[str, str] = {}
        generated_object_ids = set(self.generation.get_generated_object_ids())
        for assignment in self.surface_texture_generation.get_assignments():
            source_id = build_atlas_wall_texture_source_id(assignment.assignment_id)
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
            placed_models = self._build_placed_generated_models(
                include_excluded_levels=True
            )
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
        *,
        include_excluded_levels: bool = False,
    ) -> tuple[PlacedGeneratedModel, ...]:
        """Resolve persisted Canvas clicks into current world positions."""

        visible_level_by_index = {
            level.index: level
            for level in self.levels
            if include_excluded_levels or level.include_in_export
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
                object_name = getattr(
                    record,
                    "object_name",
                    record.object_id,
                )
                raise ValueError(
                    f"Placed object '{object_name}' is temporarily unavailable."
                )
            symmetry = self.generation.resolve_symmetric_division_for_record(record)
            world_x, world_y = level_image_to_world_xy(
                level,
                placement.image_x,
                placement.image_y,
            )
            placed_models.append(
                PlacedGeneratedModel(
                    object_id=record.object_id,
                    object_name=record.object_name,
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
                    scale=placement.scale,
                )
            )
        return tuple(placed_models)

    def _handle_surface_texture_generation_completed(
        self,
        assignment: object,
    ) -> None:
        assignment_id = getattr(assignment, "assignment_id", None)
        if isinstance(assignment_id, str) and assignment_id.strip():
            self.texture_atlas_workspace.mark_sources_new(
                (build_atlas_wall_texture_source_id(assignment_id),)
            )
        self._schedule_viewer_preview_refresh(preserve_camera=True)

    def _handle_surface_texture_content_changed(self) -> None:
        self._schedule_viewer_preview_refresh(preserve_camera=True)

    def _handle_generation_scene_pbr_maps_changed(self, _checked: bool) -> None:
        """Apply the shared Generation map toggles to the shared 3D scene."""

        if self.texture_atlas_workspace.is_ambient_occlusion_preview_active:
            return
        enabled_maps = tuple(
            map_type
            for map_type, checkbox in (
                self.merged_generation_workspace.pbr_map_checkboxes.items()
            )
            if checkbox.isChecked()
        )
        self.viewer.set_pbr_maps_enabled(enabled_maps)

    def _refresh_viewer_preview(self, preserve_camera: bool = False) -> None:
        if (
            self._active_canvas_surface_edit_target is not None
            or self._pending_canvas_surface_mesh_update
            or self._pending_wall_vertex_mesh_update
            or self._pending_stair_point_mesh_update
        ):
            return
        revision = self._viewer_preview_revision
        ambient_occlusion_override_active = (
            self.texture_atlas_workspace.is_ambient_occlusion_preview_active
        )
        canvas_is_stale = bool(
            self._canvas_viewer_preview_is_active()
            and not ambient_occlusion_override_active
            and self._canvas_viewer_preview_revision != revision
        )
        if not canvas_is_stale:
            return

        preview_levels = self._build_viewer_preview_levels()
        self._sync_viewer_scene_levels()
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
            self._viewer_preview_dependency_signature = dependency_signature_after
            self._viewer_preview_dependency_signature_revision = revision
        generated_model = self._viewer_preview_model

        if canvas_is_stale:
            if generated_model is None:
                self._set_canvas_viewer_targets(())
                self._is_syncing_canvas_scene_selection = True
                try:
                    self.viewer.clear_model()
                finally:
                    self._is_syncing_canvas_scene_selection = False
            else:
                self._set_canvas_viewer_targets(
                    tuple(build_fixed_surfaces(preview_levels))
                )
                self._is_syncing_canvas_scene_selection = True
                try:
                    self.viewer.set_model(
                        generated_model,
                        preserve_camera=preserve_camera,
                    )
                    self._restore_desired_canvas_scene_selection()
                finally:
                    self._is_syncing_canvas_scene_selection = False
            self._canvas_viewer_preview_revision = revision
            self._scheduled_viewer_refresh_preserve_camera = True
            self._sync_canvas_window_undo_availability()
            if self._pending_level_transform is not None:
                self._refresh_pending_level_transform_outline()
            self._clear_committed_doorway_outline_if_displayed()
            self._clear_committed_level_transform_outline_if_displayed()

    def _mark_viewer_preview_dirty(
        self,
        preserve_camera: bool = True,
        *,
        affects_draw_call_estimate: bool = True,
    ) -> int:
        """Invalidate shared preview data and retain future camera intent."""

        self._viewer_preview_revision += 1
        self._scheduled_viewer_refresh_preserve_camera = bool(
            self._scheduled_viewer_refresh_preserve_camera and preserve_camera
        )
        if hasattr(self, "texture_atlas_workspace"):
            self._schedule_surface_ambient_occlusion_preview_refresh()
            if affects_draw_call_estimate:
                self._atlas_draw_call_scene_revision += 1
                self._schedule_atlas_draw_call_estimate()
        return self._viewer_preview_revision

    def _queue_viewer_preview_refresh(self) -> None:
        if self._is_shutdown:
            return
        if not self._active_viewer_preview_needs_refresh():
            return
        if (
            self._active_canvas_surface_edit_target is not None
            or self._pending_canvas_surface_mesh_update
            or self._pending_wall_vertex_mesh_update
            or self._pending_stair_point_mesh_update
        ):
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
        self._refresh_blueprint_file_dependencies(include_exported_levels=True)
        self._invalidate_viewer_preview_for_dependency_changes(
            preserve_camera=preserve_camera
        )
        if (
            self._canvas_viewer_preview_is_active()
            and self._canvas_viewer_preview_revision != self._viewer_preview_revision
        ):
            self._scheduled_viewer_refresh_preserve_camera = bool(
                self._scheduled_viewer_refresh_preserve_camera and preserve_camera
            )
        self._queue_viewer_preview_refresh()

    def _run_scheduled_viewer_preview_refresh(self) -> None:
        self._is_viewer_refresh_scheduled = False
        if self._is_shutdown:
            return
        if not self._active_viewer_preview_needs_refresh():
            return
        if (
            self._active_canvas_surface_edit_target is not None
            or self._pending_canvas_surface_mesh_update
            or self._pending_wall_vertex_mesh_update
            or self._pending_stair_point_mesh_update
        ):
            return

        preserve_camera = self._scheduled_viewer_refresh_preserve_camera
        self._refresh_viewer_preview(preserve_camera=preserve_camera)

    def _handle_save_clicked(self) -> None:
        self._cancel_active_canvas_surface_edit()
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

        self._finish_level_transform_drag()
        self._commit_pending_level_transform_update()
        self._commit_pending_canvas_surface_mesh_update()
        self._commit_pending_wall_vertex_update()
        self._commit_pending_stair_point_mesh_update()

        try:
            save_project(
                path=file_path,
                current_level_index=self.current_level_index,
                levels=self.levels,
                image_library_paths=self.image_library_paths,
                doorway_presets=self.doorway_presets,
                generation=self.generation.get_data(),
                surface_texture_generation=(self.surface_texture_generation.get_data()),
                texture_atlases=self.texture_atlas_workspace.get_data(),
                stairs=self.stairs,
                wall_mirror_links=self.wall_mirror_links,
            )
        except ValueError as error:
            QMessageBox.critical(self, "Save failed", str(error))
            return

        self._remember_project_path(file_path)
        QMessageBox.information(
            self,
            "Project saved",
            f"Saved project to:\n{file_path}",
        )

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

        self._commit_pending_canvas_surface_mesh_update()
        self._commit_pending_wall_vertex_update()

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

    # ### Plan images and automatic wall generation ###
    def _get_plan_wall_generation_sliders(self) -> tuple[QSlider, ...]:
        """Return every control that reconstructs the cached wall evidence."""

        return (
            self.minimum_wall_separation_slider,
            self.maximum_wall_separation_slider,
            self.parallel_wall_angle_slider,
            self.parallel_wall_overlap_slider,
            self.wall_gap_bridge_slider,
            self.wall_endpoint_snap_slider,
            self.maximum_vertex_distance_slider,
            self.minimum_wall_length_slider,
            self.wall_detection_confidence_slider,
        )

    def _build_plan_wall_detection_options(self) -> PlanWallDetectionOptions:
        """Read one validated immutable reconstruction configuration."""

        return PlanWallDetectionOptions(
            minimum_wall_separation_pixels=float(
                self.minimum_wall_separation_slider.value()
            ),
            maximum_wall_separation_pixels=float(
                self.maximum_wall_separation_slider.value()
            ),
            parallel_angle_tolerance_degrees=float(
                self.parallel_wall_angle_slider.value()
            ),
            minimum_parallel_overlap_ratio=(
                self.parallel_wall_overlap_slider.value() / 100.0
            ),
            maximum_gap_bridge_pixels=float(self.wall_gap_bridge_slider.value()),
            endpoint_snap_distance_pixels=float(
                self.wall_endpoint_snap_slider.value()
            ),
            maximum_vertex_distance_pixels=float(
                self.maximum_vertex_distance_slider.value()
            ),
            minimum_wall_length_pixels=float(self.minimum_wall_length_slider.value()),
            confidence_threshold=(
                self.wall_detection_confidence_slider.value() / 100.0
            ),
        )

    def _handle_plan_wall_slider_changed(self, _value: int) -> None:
        """Refresh value labels and debounce one live Canvas reconstruction."""

        if self._is_syncing_plan_wall_controls:
            return
        self._is_syncing_plan_wall_controls = True
        try:
            minimum_slider = self.minimum_wall_separation_slider
            maximum_slider = self.maximum_wall_separation_slider
            if minimum_slider.value() > maximum_slider.value():
                if self.sender() is minimum_slider:
                    maximum_slider.setValue(minimum_slider.value())
                else:
                    minimum_slider.setValue(maximum_slider.value())
            self.minimum_wall_separation_value_label.setText(
                f"{minimum_slider.value()} px"
            )
            self.maximum_wall_separation_value_label.setText(
                f"{maximum_slider.value()} px"
            )
            self.parallel_wall_angle_value_label.setText(
                f"{self.parallel_wall_angle_slider.value()}\N{DEGREE SIGN}"
            )
            self.parallel_wall_overlap_value_label.setText(
                f"{self.parallel_wall_overlap_slider.value()}%"
            )
            self.wall_gap_bridge_value_label.setText(
                f"{self.wall_gap_bridge_slider.value()} px"
            )
            self.wall_endpoint_snap_value_label.setText(
                f"{self.wall_endpoint_snap_slider.value()} px"
            )
            self.maximum_vertex_distance_value_label.setText(
                f"{self.maximum_vertex_distance_slider.value()} px"
            )
            self.minimum_wall_length_value_label.setText(
                f"{self.minimum_wall_length_slider.value()} px"
            )
            self.wall_detection_confidence_value_label.setText(
                f"{self.wall_detection_confidence_slider.value()}%"
            )
        finally:
            self._is_syncing_plan_wall_controls = False

        session = self._plan_wall_preview_session
        if session is None or session.level_index != self.current_level.index:
            return
        self._plan_wall_preview_is_valid = False
        self.plan_wall_generation_status_label.setText("Updating wall preview...")
        self._update_plan_wall_generation_controls_state()
        if not self._plan_wall_preview_refresh_timer.isActive():
            self._plan_wall_preview_refresh_timer.start()

    def _handle_generate_walls_clicked(self) -> None:
        """Start image analysis or confirm the current generated-wall preview."""

        session = self._plan_wall_preview_session
        if session is not None and session.level_index == self.current_level.index:
            self._confirm_generated_walls()
            return
        self._start_plan_wall_detection()

    def _start_plan_wall_detection(self) -> None:
        """Start one non-blocking wall analysis for the active level image."""

        if self._is_shutdown:
            return
        self.canvas.stop_plan_image_erasing()
        self._refresh_blueprint_file_dependencies(include_exported_levels=False)
        level = self.current_level
        if level.index in self._plan_wall_detection_runtimes:
            return
        if level.index in self._plan_image_correction_runtimes:
            return
        raw_source_path = str(level.image_path or "").strip()
        if not raw_source_path:
            QMessageBox.warning(
                self,
                "Wall generation unavailable",
                "Load or correct a plan image before generating walls.",
            )
            return
        try:
            source_path = Path(raw_source_path).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            source_path = Path(raw_source_path)
        source_revision = _build_local_file_revision(source_path)
        if not _local_file_revision_has_file(source_revision):
            QMessageBox.warning(
                self,
                "Wall generation unavailable",
                "The current plan image cannot be read.",
            )
            return
        if (
            self.canvas.blueprint_image is None
            or self.canvas.get_blueprint_image_revision() != source_revision
        ):
            QMessageBox.warning(
                self,
                "Wall generation unavailable",
                "The current plan image could not be refreshed safely.",
            )
            return

        self._clear_plan_wall_preview(update_controls=False)
        baseline_vertex_data = level.vertex_data.clone()
        topology_signature = _build_vertex_data_signature(baseline_vertex_data)
        thread = _PlanWallDetectionThread(source_path, parent=self)
        job = self.job_manager.create_job(
            kind="Wall generation",
            requested_name="",
            default_name=f"Detect {level.display_name} walls",
            stage="Analyzing plan wall evidence (5%)",
        )
        runtime = _PlanWallDetectionRuntime(
            source_path=source_path,
            source_revision=source_revision,
            topology_signature=topology_signature,
            baseline_vertex_data=baseline_vertex_data,
            job_id=job.job_id,
            thread=thread,
        )
        self._plan_wall_detection_runtimes[level.index] = runtime
        self.job_manager.set_cancel_callback(
            job.job_id,
            partial(self._cancel_plan_wall_detection_for_level, level.index),
        )
        thread.finished.connect(
            partial(
                self._handle_plan_wall_detection_finished,
                level.index,
                job.job_id,
                thread,
            )
        )
        self.plan_wall_generation_status_label.setText("Analyzing plan walls...")
        self._update_plan_wall_generation_controls_state()
        self._update_image_correction_button_state()
        thread.start()

    def _handle_plan_wall_detection_finished(
        self,
        level_index: int,
        job_id: str,
        thread: _PlanWallDetectionThread,
    ) -> None:
        """Create a preview only while image and anchored topology still match."""

        runtime = self._plan_wall_detection_runtimes.get(level_index)
        try:
            if (
                runtime is None
                or runtime.job_id != job_id
                or runtime.thread is not thread
            ):
                return
            self._plan_wall_detection_runtimes.pop(level_index, None)
            self.job_manager.set_cancel_callback(job_id, None)
            if self._is_shutdown or runtime.cancel_requested or thread.was_cancelled:
                self.job_manager.mark_cancelled(job_id)
                return
            if thread.error_message is not None or thread.analysis is None:
                message = thread.error_message or (
                    "Wall analysis finished without returning usable evidence."
                )
                self.job_manager.fail_job(job_id, f"Failed: {message}")
                QMessageBox.critical(self, "Wall generation failed", message)
                return

            level = self._get_level_by_index(level_index)
            if level is None or not self._plan_wall_detection_context_is_current(
                level,
                runtime,
            ):
                message = (
                    "The plan image or its existing walls changed during analysis."
                )
                self.job_manager.fail_job(job_id, f"Not applied: {message}")
                if level is self.current_level:
                    QMessageBox.warning(self, "Wall generation not applied", message)
                return

            session = _PlanWallPreviewSession(
                level_index=level_index,
                source_revision=runtime.source_revision,
                topology_signature=runtime.topology_signature,
                baseline_vertex_data=runtime.baseline_vertex_data,
                existing_edge_keys=_get_vertex_data_edge_keys(
                    runtime.baseline_vertex_data
                ),
                analysis=thread.analysis,
            )
            if level is not self.current_level:
                self.job_manager.complete_job(job_id, "Analysis completed")
                return
            self._plan_wall_preview_session = session
            if not self._refresh_plan_wall_preview():
                message = (
                    "Wall evidence was found, but no valid preview could be built."
                )
                self.job_manager.fail_job(job_id, f"Failed: {message}")
                return
            self.job_manager.complete_job(job_id, "Wall preview ready")
        finally:
            thread.deleteLater()
            self._update_plan_wall_generation_controls_state()
            self._update_image_correction_button_state()

    def _plan_wall_detection_context_is_current(
        self,
        level: LevelData,
        runtime: _PlanWallDetectionRuntime,
    ) -> bool:
        """Return whether a wall analysis can still target its captured level."""

        if (
            _build_vertex_data_signature(level.vertex_data)
            != runtime.topology_signature
        ):
            return False
        return (
            _build_local_file_revision(level.image_path)
            == runtime.source_revision
        )

    def _refresh_plan_wall_preview(self) -> bool:
        """Reconstruct the cached evidence and publish one lightweight overlay."""

        self._plan_wall_preview_refresh_timer.stop()
        session = self._plan_wall_preview_session
        if session is None or session.level_index != self.current_level.index:
            self._plan_wall_preview_is_valid = False
            self._update_plan_wall_generation_controls_state()
            return False
        if (
            _build_local_file_revision(self.current_level.image_path)
            != session.source_revision
            or _build_vertex_data_signature(self.current_level.vertex_data)
            != session.topology_signature
        ):
            self._clear_plan_wall_preview()
            return False

        try:
            result = reconstruct_plan_walls(
                session.analysis,
                self._build_plan_wall_detection_options(),
                existing_vertex_data=session.baseline_vertex_data,
            )
            added_count = _validate_plan_wall_detection_result(session, result)
            self.canvas.set_generated_wall_preview(
                result.vertex_data,
                session.existing_edge_keys,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            self._plan_wall_preview_is_valid = False
            self.canvas.clear_generated_wall_preview()
            self.plan_wall_generation_status_label.setText(
                f"Preview unavailable: {error}"
            )
            self._update_plan_wall_generation_controls_state()
            return False
        self._plan_wall_preview_is_valid = added_count > 0
        if self._plan_wall_preview_is_valid:
            self.plan_wall_generation_status_label.setText(
                f"{added_count} generated wall faces. Adjust the sliders or confirm."
            )
        else:
            self.plan_wall_generation_status_label.setText(
                "No wall pairs match the current controls. Adjust the sliders."
            )
        self._update_plan_wall_generation_controls_state()
        return True

    def _confirm_generated_walls(self) -> None:
        """Commit the latest preview and build its 3D walls exactly once."""

        self._finish_level_transform_drag()
        self._commit_pending_level_transform_update()
        if (
            not self._refresh_plan_wall_preview()
            or not self._plan_wall_preview_is_valid
        ):
            return
        session = self._plan_wall_preview_session
        if session is None:
            return
        level = self.current_level
        preview = self.canvas.get_generated_wall_preview()
        offset_compensation = (0.0, 0.0)
        if preview is not None and session.baseline_vertex_data.vertices:
            previous_pivot_x, previous_pivot_y = get_level_world_pivot(level)
            preview_level = copy.copy(level)
            preview_level.vertex_data = preview.to_vertex_data()
            next_pivot_x, next_pivot_y = get_level_world_pivot(preview_level)
            scale = float(level.scale)
            offset_compensation = (
                (1.0 - scale) * (previous_pivot_x - next_pivot_x),
                (1.0 - scale) * (previous_pivot_y - next_pivot_y),
            )
        self._commit_pending_canvas_surface_mesh_update()
        self._commit_pending_wall_vertex_update()
        self._commit_pending_doorway_mesh_update()
        self._plan_wall_preview_session = None
        self._plan_wall_preview_is_valid = False
        if not self.canvas.commit_generated_wall_preview():
            self._update_plan_wall_generation_controls_state()
            return
        level.offset_x_meters += offset_compensation[0]
        level.offset_y_meters += offset_compensation[1]
        self._sync_level_controls()
        self.plan_wall_generation_status_label.setText(
            "Generated walls confirmed."
        )
        self.viewer.set_surface_tools_status(
            "Generated walls confirmed. Press Ctrl+Z to undo."
        )
        self._schedule_viewer_preview_refresh(preserve_camera=True)
        self._update_plan_wall_generation_controls_state()
        self._update_image_correction_button_state()

    def _clear_plan_wall_preview(self, *, update_controls: bool = True) -> None:
        """Discard cached wall evidence and its non-destructive Canvas overlay."""

        self._plan_wall_preview_refresh_timer.stop()
        self._plan_wall_preview_session = None
        self._plan_wall_preview_is_valid = False
        if hasattr(self, "canvas"):
            self.canvas.clear_generated_wall_preview()
        if hasattr(self, "plan_wall_generation_status_label"):
            self.plan_wall_generation_status_label.setText(
                "Generate walls to analyze the current plan."
            )
        if update_controls:
            self._update_plan_wall_generation_controls_state()
            self._update_image_correction_button_state()

    def _cancel_plan_wall_detection_for_level(self, level_index: int) -> bool:
        runtime = self._plan_wall_detection_runtimes.get(int(level_index))
        if runtime is None:
            return False
        runtime.cancel_requested = True
        if runtime.thread.isRunning():
            runtime.thread.requestInterruption()
        self._update_plan_wall_generation_controls_state()
        return True

    def _cancel_and_join_plan_wall_detections(self) -> None:
        """Retire every wall-analysis worker before its project context changes."""

        self._clear_plan_wall_preview(update_controls=False)
        runtimes = tuple(self._plan_wall_detection_runtimes.values())
        for runtime in runtimes:
            runtime.cancel_requested = True
            runtime.thread.requestInterruption()
            self.job_manager.mark_cancelled(runtime.job_id)
        for runtime in runtimes:
            while runtime.thread.isRunning():
                runtime.thread.wait(PLAN_WALL_DETECTION_SHUTDOWN_WAIT_MILLISECONDS)
            runtime.thread.deleteLater()
        self._plan_wall_detection_runtimes.clear()
        if hasattr(self, "generate_walls_button"):
            self._update_plan_wall_generation_controls_state()

    def _update_plan_wall_generation_controls_state(self) -> None:
        """Synchronize the Generate/Confirm wall action and live controls."""

        if not hasattr(self, "generate_walls_button"):
            return
        try:
            level = self.current_level
        except IndexError:
            self.generate_walls_button.setEnabled(False)
            self.generate_walls_button.setText("Generate walls")
            self.plan_wall_controls_group.setEnabled(False)
            self._update_plan_image_erase_button_state()
            return
        runtime = self._plan_wall_detection_runtimes.get(level.index)
        correction_runtime = self._plan_image_correction_runtimes.get(level.index)
        session = self._plan_wall_preview_session
        has_preview = session is not None and session.level_index == level.index
        source_revision = _build_local_file_revision(level.image_path)
        has_source = _local_file_revision_has_file(source_revision)
        if runtime is not None:
            self.generate_walls_button.setEnabled(False)
            self.generate_walls_button.setText("Generating walls...")
            self.plan_wall_controls_group.setEnabled(False)
            self._update_plan_image_erase_button_state()
            return
        if correction_runtime is not None:
            self.generate_walls_button.setEnabled(False)
            self.generate_walls_button.setText("Generate walls")
            self.plan_wall_controls_group.setEnabled(False)
            self._update_plan_image_erase_button_state()
            return
        self.generate_walls_button.setText(
            "Confirm walls" if has_preview else "Generate walls"
        )
        self.generate_walls_button.setEnabled(
            not self._is_shutdown
            and has_source
            and (self._plan_wall_preview_is_valid if has_preview else True)
        )
        self.plan_wall_controls_group.setEnabled(
            not self._is_shutdown and has_preview
        )
        self._update_plan_image_erase_button_state()

    def _handle_load_image_clicked(self) -> None:
        file_path = self._get_image_file_path()
        if not file_path:
            return

        try:
            self._set_current_level_image(file_path)
        except ValueError as error:
            QMessageBox.critical(self, "Image load failed", str(error))

    # ### Plan-image erasing ###
    def _new_plan_image_erase_output_path(self, level: LevelData) -> Path:
        """Return one application-owned immutable PNG path for a marquee erase."""

        output_directory = self._application_settings.path.parent / "edited_plans"
        return (
            output_directory
            / f"edited-plan-L{level.index}-{uuid.uuid4().hex}.png"
        )

    def _handle_erase_plan_image_clicked(self, checked: bool) -> None:
        """Enter or leave the Canvas plan-image Erase mode."""

        if not checked:
            self.canvas.stop_plan_image_erasing()
            self._update_plan_image_erase_button_state()
            return

        level = self.current_level
        if (
            self._is_shutdown
            or self.canvas.blueprint_image is None
            or level.image_path is None
            or level.index in self._plan_image_correction_runtimes
            or level.index in self._plan_wall_detection_runtimes
        ):
            was_blocked = self.erase_plan_image_button.blockSignals(True)
            self.erase_plan_image_button.setChecked(False)
            self.erase_plan_image_button.blockSignals(was_blocked)
            self._update_plan_image_erase_button_state()
            return

        output_path = self._new_plan_image_erase_output_path(level)
        if not self.canvas.start_plan_image_erasing(output_path):
            was_blocked = self.erase_plan_image_button.blockSignals(True)
            self.erase_plan_image_button.setChecked(False)
            self.erase_plan_image_button.blockSignals(was_blocked)
            return
        self._clear_plan_wall_preview()
        self.viewer.set_surface_tools_status(
            "Plan-image Erase active. Drag a rectangle on the Canvas; "
            "right-click or press Escape to exit."
        )

    def _handle_plan_image_erase_mode_changed(self, active: bool) -> None:
        """Keep the checkable Erase button synchronized with the Canvas mode."""

        if not hasattr(self, "erase_plan_image_button"):
            return
        was_blocked = self.erase_plan_image_button.blockSignals(True)
        self.erase_plan_image_button.setChecked(bool(active))
        self.erase_plan_image_button.blockSignals(was_blocked)
        self._update_plan_image_erase_button_state()

    def _handle_plan_image_erase_committed(self, raw_commit: object) -> None:
        """Publish one atomically saved marquee erase to the active level."""

        if not isinstance(raw_commit, PlanImageEraseCommit):
            return
        level = self.current_level
        previous_path = str(Path(raw_commit.previous_path).resolve())
        replacement_path = str(Path(raw_commit.replacement_path).resolve())
        current_path = (
            None
            if level.image_path is None
            else str(Path(level.image_path).resolve())
        )
        if current_path != previous_path:
            if current_path is not None:
                self.canvas.load_blueprint_image_preserving_view(current_path)
            if self.canvas.is_plan_image_erasing():
                self.canvas.set_plan_image_erase_output_path(
                    self._new_plan_image_erase_output_path(level)
                )
            QMessageBox.critical(
                self,
                "Plan erase not applied",
                "The active level image changed before the erase could be "
                "committed.",
            )
            return

        self._clear_plan_wall_preview(update_controls=False)
        self._record_canvas_undo_state(
            _CanvasPlanImageUndoState(
                level_index=level.index,
                commit=raw_commit,
            )
        )
        if level.original_image_path is None:
            level.original_image_path = previous_path
        level.image_path = replacement_path
        level.image_size_pixels = tuple(raw_commit.image_size_pixels)
        self._level_blueprint_image_revisions[level.index] = (
            raw_commit.replacement_revision
        )
        self.canvas.set_plan_image_erase_output_path(
            self._new_plan_image_erase_output_path(level)
        )
        self._update_blueprint_name_label()
        self._update_plan_wall_generation_controls_state()
        self._update_plan_image_erase_button_state()
        self.viewer.set_surface_tools_status(
            "Plan-image selection erased. Press Ctrl+Z to undo."
        )

    def _handle_plan_image_erase_failed(self, message: str) -> None:
        """Report an atomic-save failure after the Canvas restored its pixels."""

        QMessageBox.critical(
            self,
            "Plan erase failed",
            str(message).strip() or "The edited plan image could not be saved.",
        )
        if self.canvas.is_plan_image_erasing():
            self.canvas.set_plan_image_erase_output_path(
                self._new_plan_image_erase_output_path(self.current_level)
            )
        self._update_plan_image_erase_button_state()

    def _update_plan_image_erase_button_state(self) -> None:
        """Enable Erase only while the active plan can be edited safely."""

        if not hasattr(self, "erase_plan_image_button"):
            return
        try:
            level = self.current_level
        except IndexError:
            self.erase_plan_image_button.setEnabled(False)
            self.erase_plan_image_button.setChecked(False)
            return
        has_source = (
            level.image_path is not None
            and self.canvas.blueprint_image is not None
            and _local_file_revision_has_file(
                _build_local_file_revision(level.image_path)
            )
        )
        self.erase_plan_image_button.setEnabled(
            not self._is_shutdown
            and has_source
            and level.index not in self._plan_image_correction_runtimes
            and level.index not in self._plan_wall_detection_runtimes
        )
        was_blocked = self.erase_plan_image_button.blockSignals(True)
        self.erase_plan_image_button.setChecked(
            self.canvas.is_plan_image_erasing()
        )
        self.erase_plan_image_button.blockSignals(was_blocked)

    def _handle_image_correction_clicked(self) -> None:
        """Start one non-blocking correction for the current level's source photo."""

        if self._is_shutdown:
            return
        self.canvas.stop_plan_image_erasing()
        level = self.current_level
        if level.index in self._plan_image_correction_runtimes:
            return
        source_path = self._get_plan_correction_source_path(level)
        source_revision = _build_local_file_revision(source_path)
        if source_path is None or not _local_file_revision_has_file(source_revision):
            QMessageBox.warning(
                self,
                "Image correction unavailable",
                "Load a valid plan image before correcting it.",
            )
            return
        if self._level_has_plan_dependent_geometry(level):
            QMessageBox.warning(
                self,
                "Image correction unavailable",
                "Correct the plan image before adding walls, openings, stairs, "
                "open spaces, or editable surfaces to this level.",
            )
            return

        service_settings = self.settings_widget.get_settings()
        correction_model = service_settings.plan_correction_model
        api_key = service_settings.openai_api_key.strip()
        if correction_model in OPENAI_PLAN_CORRECTION_MODELS and not api_key:
            QMessageBox.warning(
                self,
                "OpenAI API key required",
                "Add an OpenAI API key in Settings before using Image correction.",
            )
            return
        if correction_model == PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1:
            try:
                create_default_qwen_plan_image_editor()
            except QwenPlanCorrectionError as error:
                QMessageBox.warning(
                    self,
                    "Local Qwen model unavailable",
                    str(error),
                )
                return

        source_path = Path(str(source_revision[0]))
        correction_directory = (
            self._application_settings.path.parent / "corrected_plans"
        )
        output_path = (
            correction_directory
            / f"corrected-plan-L{level.index}-{uuid.uuid4().hex}.png"
        )
        thread = _PlanImageCorrectionThread(
            source_path,
            output_path,
            api_key,
            correction_model,
            parent=self,
            make_walls_continuous=service_settings.make_walls_continuous,
        )
        job = self.job_manager.create_job(
            kind="Image correction",
            requested_name="",
            default_name=f"Correct {level.display_name} plan",
            stage="Preparing plan image (0%)",
        )
        runtime = _PlanImageCorrectionRuntime(
            source_path=source_path,
            source_revision=source_revision,
            output_path=output_path,
            job_id=job.job_id,
            thread=thread,
        )
        self._plan_image_correction_runtimes[level.index] = runtime
        self.job_manager.set_cancel_callback(
            job.job_id,
            partial(self._cancel_plan_image_correction_for_level, level.index),
        )
        thread.progress.connect(
            partial(
                self._handle_plan_image_correction_progress,
                level.index,
                job.job_id,
            )
        )
        thread.finished.connect(
            partial(
                self._handle_plan_image_correction_finished,
                level.index,
                job.job_id,
                thread,
            )
        )
        self._update_image_correction_button_state()
        self._update_plan_wall_generation_controls_state()
        thread.start()

    @staticmethod
    def _get_plan_correction_source_path(level: LevelData) -> Path | None:
        raw_path = level.original_image_path or level.image_path
        if raw_path is None or not str(raw_path).strip():
            return None
        try:
            return Path(raw_path).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            return None

    def _level_has_plan_dependent_geometry(self, level: LevelData) -> bool:
        """Return whether changing this level's image coordinates is unsafe."""

        return bool(
            level.vertex_data.vertices
            or level.vertex_data.edges
            or level.rooms
            or level.doorways
            or level.windows
            or level.open_spaces
            or level.editable_surfaces
            or any(
                stair.start_level_index == level.index
                or stair.end_level_index == level.index
                for stair in self.stairs
            )
        )

    def _handle_plan_image_correction_progress(
        self,
        level_index: int,
        job_id: str,
        raw_progress: object,
    ) -> None:
        runtime = self._plan_image_correction_runtimes.get(level_index)
        if (
            runtime is None
            or runtime.job_id != job_id
            or not isinstance(raw_progress, PlanCorrectionProgress)
        ):
            return
        message = raw_progress.message.strip() or raw_progress.stage.strip()
        stage = f"{message} ({raw_progress.percent}%)"
        self.job_manager.update_job(
            job_id,
            stage=stage,
            progress=raw_progress.percent,
        )

    def _handle_plan_image_correction_finished(
        self,
        level_index: int,
        job_id: str,
        thread: _PlanImageCorrectionThread,
    ) -> None:
        """Commit a correction only when its source and level are unchanged."""

        runtime = self._plan_image_correction_runtimes.get(level_index)
        try:
            if (
                runtime is None
                or runtime.job_id != job_id
                or runtime.thread is not thread
            ):
                return
            self._plan_image_correction_runtimes.pop(level_index, None)
            self.job_manager.set_cancel_callback(job_id, None)
            if (
                self._is_shutdown
                or runtime.cancel_requested
                or thread.was_cancelled
            ):
                self.job_manager.mark_cancelled(job_id)
                self._discard_plan_correction_output(runtime.output_path)
                return
            if thread.error_message is not None or thread.result is None:
                message = thread.error_message or (
                    "Image correction finished without returning an image."
                )
                self.job_manager.fail_job(job_id, f"Failed: {message}")
                self._discard_plan_correction_output(runtime.output_path)
                QMessageBox.critical(self, "Image correction failed", message)
                return

            level = self._get_level_by_index(level_index)
            if not self._plan_image_correction_context_is_current(level, runtime):
                message = (
                    "The plan or level changed while it was being corrected. "
                    "The generated image was not applied."
                )
                self.job_manager.fail_job(job_id, f"Failed: {message}")
                self._discard_plan_correction_output(runtime.output_path)
                QMessageBox.warning(self, "Image correction not applied", message)
                return

            assert level is not None
            self._apply_plan_image_correction_result(level, runtime, thread.result)
            if thread.result.method in {
                CORRECTION_METHOD_OPENAI,
                CORRECTION_METHOD_QWEN,
            }:
                model_label = plan_correction_model_label(thread.model)
                completion_stage = f"{model_label} plan correction completed"
            else:
                completion_stage = "Plan correction completed"
            self.job_manager.complete_job(job_id, completion_stage)
        finally:
            self._update_image_correction_button_state()
            self._update_plan_wall_generation_controls_state()
            thread.deleteLater()

    def _plan_image_correction_context_is_current(
        self,
        level: LevelData | None,
        runtime: _PlanImageCorrectionRuntime,
    ) -> bool:
        if level is None or self._level_has_plan_dependent_geometry(level):
            return False
        source_path = self._get_plan_correction_source_path(level)
        if source_path != runtime.source_path:
            return False
        return _build_local_file_revision(source_path) == runtime.source_revision

    def _apply_plan_image_correction_result(
        self,
        level: LevelData,
        runtime: _PlanImageCorrectionRuntime,
        result: PlanImageCorrectionResult,
    ) -> None:
        corrected_path = str(Path(result.output_path).resolve())
        source_path = str(runtime.source_path)
        if level is self.current_level:
            self._set_current_level_image(
                corrected_path,
                original_image_path=source_path,
            )
            return
        level.image_path = corrected_path
        level.original_image_path = source_path
        level.image_size_pixels = tuple(float(value) for value in result.output_size)

    def _cancel_plan_image_correction_for_level(self, level_index: int) -> bool:
        runtime = self._plan_image_correction_runtimes.get(int(level_index))
        if runtime is None or not runtime.thread.isRunning():
            return False
        runtime.cancel_requested = True
        runtime.thread.requestInterruption()
        self._update_image_correction_button_state()
        return True

    def _cancel_and_join_plan_image_corrections(self) -> None:
        runtimes = tuple(self._plan_image_correction_runtimes.values())
        for runtime in runtimes:
            runtime.cancel_requested = True
            runtime.thread.requestInterruption()
            self.job_manager.mark_cancelled(runtime.job_id)
        for runtime in runtimes:
            while runtime.thread.isRunning():
                runtime.thread.wait(PLAN_IMAGE_CORRECTION_SHUTDOWN_WAIT_MILLISECONDS)
            self._discard_plan_correction_output(runtime.output_path)
            runtime.thread.deleteLater()
        self._plan_image_correction_runtimes.clear()
        if hasattr(self, "image_correction_button"):
            self._update_image_correction_button_state()

    @staticmethod
    def _discard_plan_correction_output(output_path: Path) -> None:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _update_image_correction_button_state(self) -> None:
        if not hasattr(self, "image_correction_button"):
            return
        try:
            level = self.current_level
        except IndexError:
            self.image_correction_button.setEnabled(False)
            self.image_correction_button.setText("Image correction")
            self._update_plan_image_erase_button_state()
            return
        runtime = self._plan_image_correction_runtimes.get(level.index)
        wall_runtime = self._plan_wall_detection_runtimes.get(level.index)
        wall_preview = self._plan_wall_preview_session
        has_wall_preview = (
            wall_preview is not None and wall_preview.level_index == level.index
        )
        source_path = self._get_plan_correction_source_path(level)
        has_source = source_path is not None and _local_file_revision_has_file(
            _build_local_file_revision(source_path)
        )
        self.image_correction_button.setEnabled(
            not self._is_shutdown
            and has_source
            and runtime is None
            and wall_runtime is None
            and not has_wall_preview
        )
        self.image_correction_button.setText(
            "Correcting..." if runtime is not None else "Image correction"
        )
        self._update_plan_image_erase_button_state()

    def _get_image_file_path(self) -> str:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load plan image",
            str(Path.home()),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        return file_path

    def _refresh_levels_list(self) -> None:
        self._is_syncing_level_controls = True
        self.levels_list.clear()
        ordered_positions = sorted(
            range(len(self.levels)),
            key=lambda position: self.levels[position].index,
            reverse=True,
        )
        for level_position in ordered_positions:
            level = self.levels[level_position]
            self.levels_list.addItem(self._build_level_item(level, level_position))
        self.levels_list.setCurrentRow(
            self._level_list_row_for_position(self.current_level_index)
        )
        self._is_syncing_level_controls = False

    def _build_level_item(
        self,
        level: LevelData,
        level_position: int,
    ) -> QListWidgetItem:
        item = QListWidgetItem(level.display_name)
        item.setData(LEVEL_POSITION_ITEM_ROLE, level_position)
        if level.index == GROUND_LEVEL_INDEX:
            item.setBackground(
                _build_ground_level_background_color(self.levels_list.palette())
            )
        return item

    def _level_list_row_for_position(self, level_position: int) -> int:
        for row in range(self.levels_list.count()):
            item = self.levels_list.item(row)
            if item.data(LEVEL_POSITION_ITEM_ROLE) == level_position:
                return row
        return -1

    def _refresh_doorway_preset_list(
        self,
        selected_index: int | None = None,
    ) -> None:
        if not self.doorway_presets:
            self.doorway_presets.append(create_fallback_doorway_preset())
            selected_index = 0
        if selected_index is None:
            selected_index = self.doorway_preset_list.currentRow()

        self.doorway_preset_list.blockSignals(True)
        self.doorway_preset_list.clear()
        for preset in self.doorway_presets:
            self.doorway_preset_list.addItem(_format_doorway_preset_label(preset))

        if 0 <= selected_index < self.doorway_preset_list.count():
            self.doorway_preset_list.setCurrentRow(selected_index)
        self.doorway_preset_list.blockSignals(False)
        self._update_doorway_preset_button_state()

    # ### Stair control helpers ###
    def _sync_stair_tread_edge_radius_enabled(
        self,
        _index: int | None = None,
    ) -> None:
        """Enable edge-radius editing only for a rounded modern tread."""

        has_rounded_edge = (
            self.stair_tread_edge_combo.currentData()
            == STAIR_TREAD_EDGE_ROUNDED
        )
        placement_active = self.canvas.is_stair_placement_active()
        self.stair_tread_edge_radius_label.setEnabled(has_rounded_edge)
        self.stair_tread_edge_radius_spinbox.setEnabled(
            has_rounded_edge and not placement_active
        )

    def _sync_stair_starting_step_edge_radius_enabled(
        self,
        _index: int | None = None,
    ) -> None:
        """Enable starting-curve editing only when a profile is selected."""

        has_starting_step = (
            self.stair_starting_step_combo.currentData()
            != STAIR_STARTING_STEP_NONE
        )
        placement_active = self.canvas.is_stair_placement_active()
        self.stair_starting_step_edge_radius_label.setEnabled(has_starting_step)
        self.stair_starting_step_edge_radius_spinbox.setEnabled(
            has_starting_step and not placement_active
        )
        self.stair_starting_step_edge_points_label.setEnabled(has_starting_step)
        self.stair_starting_step_edge_points_spinbox.setEnabled(
            has_starting_step and not placement_active
        )

    def _stair_parameter_controls(self) -> tuple[QWidget, ...]:
        """Return every control locked while a placement draft is active."""

        return (
            self.stair_type_combo,
            self.stair_step_rise_target_spinbox,
            self.stair_tread_thickness_spinbox,
            self.stair_nosing_overhang_spinbox,
            self.stair_nosing_left_checkbox,
            self.stair_nosing_right_checkbox,
            self.stair_nosing_front_checkbox,
            self.stair_tread_edge_combo,
            self.stair_tread_edge_radius_spinbox,
            self.stair_starting_step_combo,
            self.stair_starting_step_edge_radius_spinbox,
            self.stair_starting_step_edge_points_spinbox,
            self.stair_stringer_placement_combo,
        )

    def _read_stair_editor_parameters(self) -> _StairEditorParameters:
        """Capture one normalized parameter set from the Canvas controls."""

        return _StairEditorParameters(
            stair_type=str(
                self.stair_type_combo.currentData() or DEFAULT_STAIR_TYPE
            ),
            target_rise_meters=float(
                self.stair_step_rise_target_spinbox.value()
            ) / 100.0,
            tread_thickness_meters=float(
                self.stair_tread_thickness_spinbox.value()
            ) / 100.0,
            tread_overhang_meters=float(
                self.stair_nosing_overhang_spinbox.value()
            ) / 100.0,
            nosing_placements=tuple(
                placement
                for placement, checkbox in (
                    (STAIR_NOSING_LEFT, self.stair_nosing_left_checkbox),
                    (STAIR_NOSING_RIGHT, self.stair_nosing_right_checkbox),
                    (STAIR_NOSING_FRONT, self.stair_nosing_front_checkbox),
                )
                if checkbox.isChecked()
            ),
            tread_edge_profile=str(
                self.stair_tread_edge_combo.currentData()
                or DEFAULT_STAIR_TREAD_EDGE_PROFILE
            ),
            tread_edge_radius_meters=float(
                self.stair_tread_edge_radius_spinbox.value()
            )
            / 100.0,
            starting_step=str(
                self.stair_starting_step_combo.currentData()
                or DEFAULT_STAIR_STARTING_STEP
            ),
            starting_step_edge_radius_meters=float(
                self.stair_starting_step_edge_radius_spinbox.value()
            )
            / 100.0,
            starting_step_edge_points=int(
                self.stair_starting_step_edge_points_spinbox.value()
            ),
            stringer_placement=str(
                self.stair_stringer_placement_combo.currentData()
                or DEFAULT_STAIR_STRINGER_PLACEMENT
            ),
        )

    @staticmethod
    def _stair_editor_parameters_for_stair(
        stair: StairData,
    ) -> _StairEditorParameters:
        """Expose one persisted stair through the supported editor choices."""

        stair_type = str(stair.stair_type)
        if stair_type not in {STAIR_TYPE_SUPPORTED, STAIR_TYPE_FLOATING}:
            stair_type = (
                STAIR_TYPE_FLOATING
                if stair.style in {
                    STAIR_STYLE_FLOATING,
                    STAIR_STYLE_FLOATING_WITH_RISER,
                }
                else STAIR_TYPE_SUPPORTED
            )
        tread_edge_profile = str(stair.tread_edge_profile)
        if tread_edge_profile not in {
            STAIR_TREAD_EDGE_STRAIGHT,
            STAIR_TREAD_EDGE_ROUNDED,
        }:
            tread_edge_profile = DEFAULT_STAIR_TREAD_EDGE_PROFILE
        starting_step = str(stair.starting_step)
        if starting_step not in {
            STAIR_STARTING_STEP_NONE,
            STAIR_STARTING_STEP_BULLNOSE,
            STAIR_STARTING_STEP_CURTAIL,
        }:
            starting_step = DEFAULT_STAIR_STARTING_STEP
        stringer_placement = str(stair.stringer_placement)
        if stringer_placement not in {
            STAIR_STRINGER_NONE,
            STAIR_STRINGER_LEFT,
            STAIR_STRINGER_RIGHT,
            STAIR_STRINGER_BOTH,
        }:
            stringer_placement = DEFAULT_STAIR_STRINGER_PLACEMENT
        return _StairEditorParameters(
            stair_type=stair_type,
            target_rise_meters=float(stair.target_rise_meters),
            tread_thickness_meters=float(stair.tread_thickness_meters),
            tread_overhang_meters=float(stair.tread_overhang_meters),
            nosing_placements=tuple(stair.nosing_placements),
            tread_edge_profile=tread_edge_profile,
            tread_edge_radius_meters=float(stair.tread_edge_radius_meters),
            starting_step=starting_step,
            starting_step_edge_radius_meters=float(
                stair.starting_step_edge_radius_meters
            ),
            starting_step_edge_points=int(stair.starting_step_edge_points),
            stringer_placement=stringer_placement,
        )

    def _set_stair_editor_parameters(
        self,
        parameters: _StairEditorParameters,
    ) -> None:
        """Load controls without recursively staging a stair preview."""

        self._is_syncing_stair_controls = True
        try:
            for combo, value in (
                (self.stair_type_combo, parameters.stair_type),
                (self.stair_tread_edge_combo, parameters.tread_edge_profile),
                (self.stair_starting_step_combo, parameters.starting_step),
                (
                    self.stair_stringer_placement_combo,
                    parameters.stringer_placement,
                ),
            ):
                index = combo.findData(value)
                combo.setCurrentIndex(max(0, index))
            self.stair_step_rise_target_spinbox.setValue(
                parameters.target_rise_meters * 100.0
            )
            self.stair_tread_thickness_spinbox.setValue(
                parameters.tread_thickness_meters * 100.0
            )
            self.stair_nosing_overhang_spinbox.setValue(
                parameters.tread_overhang_meters * 100.0
            )
            self.stair_tread_edge_radius_spinbox.setValue(
                parameters.tread_edge_radius_meters * 100.0
            )
            self.stair_starting_step_edge_radius_spinbox.setValue(
                parameters.starting_step_edge_radius_meters * 100.0
            )
            self.stair_starting_step_edge_points_spinbox.setValue(
                parameters.starting_step_edge_points
            )
            selected_nosing_placements = set(parameters.nosing_placements)
            for checkbox, placement in (
                (self.stair_nosing_left_checkbox, STAIR_NOSING_LEFT),
                (self.stair_nosing_right_checkbox, STAIR_NOSING_RIGHT),
                (self.stair_nosing_front_checkbox, STAIR_NOSING_FRONT),
            ):
                checkbox.setChecked(placement in selected_nosing_placements)
        finally:
            self._is_syncing_stair_controls = False
        self._sync_stair_tread_edge_radius_enabled()
        self._sync_stair_starting_step_edge_radius_enabled()

    def _load_stair_editor_from_stair(self, stair: StairData) -> None:
        """Load persisted settings and their derived step measurements."""

        self._set_stair_editor_parameters(
            self._stair_editor_parameters_for_stair(stair)
        )
        self._sync_stair_calculated_values(stair)

    def _sync_stair_calculated_values(self, stair: StairData | None) -> None:
        """Show exact step count and rise for a complete stair route."""

        if stair is None:
            self.stair_calculated_step_count_label.setText("—")
            self.stair_actual_rise_label.setText("—")
            return
        try:
            base_z_by_level = build_level_base_z_lookup(self.levels)
            total_rise = abs(
                float(base_z_by_level[stair.end_level_index])
                - float(base_z_by_level[stair.start_level_index])
            )
            step_count, actual_rise = calculate_stair_step_layout(
                total_rise,
                stair.target_rise_meters,
            )
        except (KeyError, TypeError, ValueError):
            self.stair_calculated_step_count_label.setText("—")
            self.stair_actual_rise_label.setText("—")
            return
        self.stair_calculated_step_count_label.setText(str(step_count))
        self.stair_actual_rise_label.setText(f"{actual_rise * 100.0:.1f} cm")

    def _refresh_stair_editor_geometry(self) -> None:
        """Rebuild selected-stair preview data after scene geometry changes."""

        stair_index = self._editing_stair_index
        if stair_index is not None and 0 <= stair_index < len(self.stairs):
            if self._pending_stair_parameters is not None:
                self._stair_preview_update_timer.stop()
                self._rebuild_staged_stair_preview()
                return
            if self._staged_stair is not None:
                self._pending_stair_parameters = (
                    self._stair_editor_parameters_for_stair(self._staged_stair)
                )
                self._rebuild_staged_stair_preview()
                return
            self._sync_stair_calculated_values(self.stairs[stair_index])
            return

        draft = self.canvas.get_stair_placement_draft()
        if draft is None:
            self._sync_stair_calculated_values(None)
            return
        try:
            stair = _build_stair_data_from_placement(
                draft,
                self._read_stair_editor_parameters(),
            )
        except (TypeError, ValueError):
            self._sync_stair_calculated_values(None)
        else:
            self._sync_stair_calculated_values(stair)

    @staticmethod
    def _apply_stair_editor_parameters(
        stair: StairData,
        parameters: _StairEditorParameters,
    ) -> StairData:
        """Return a stair with only its editable geometry settings changed."""

        if parameters == BlueprintWorkspace._stair_editor_parameters_for_stair(
            stair
        ):
            return stair
        return replace(
            stair,
            stair_type=parameters.stair_type,
            target_rise_meters=parameters.target_rise_meters,
            tread_thickness_meters=parameters.tread_thickness_meters,
            tread_overhang_meters=parameters.tread_overhang_meters,
            nosing_placements=parameters.nosing_placements,
            tread_edge_profile=parameters.tread_edge_profile,
            tread_edge_radius_meters=parameters.tread_edge_radius_meters,
            starting_step=parameters.starting_step,
            starting_step_edge_radius_meters=(
                parameters.starting_step_edge_radius_meters
            ),
            starting_step_edge_points=parameters.starting_step_edge_points,
            stringer_placement=parameters.stringer_placement,
            legacy_part_layout=False,
        )

    def _handle_stair_parameter_changed(self, _value: object = None) -> None:
        """Stage selected-stair geometry or remember settings for a new one."""

        if self._is_syncing_stair_controls:
            return
        parameters = self._read_stair_editor_parameters()
        stair_index = self._editing_stair_index
        if stair_index is None or not 0 <= stair_index < len(self.stairs):
            self._new_stair_parameters = parameters
            draft = self.canvas.get_stair_placement_draft()
            if draft is None:
                self._sync_stair_calculated_values(None)
                return
            try:
                stair = _build_stair_data_from_placement(draft, parameters)
            except (TypeError, ValueError):
                self._sync_stair_calculated_values(None)
            else:
                self._sync_stair_calculated_values(stair)
            return

        self._pending_stair_parameters = parameters
        self._stair_preview_update_timer.start()
        self._update_stair_button_state()

    def _rebuild_staged_stair_preview(self) -> None:
        """Build one coalesced geometry overlay after stair fields settle."""

        parameters = self._pending_stair_parameters
        self._pending_stair_parameters = None
        stair_index = self._editing_stair_index
        if (
            parameters is None
            or stair_index is None
            or not 0 <= stair_index < len(self.stairs)
        ):
            self._update_stair_button_state()
            return
        persisted_stair = self.stairs[stair_index]
        try:
            candidate = self._apply_stair_editor_parameters(
                persisted_stair,
                parameters,
            )
            if candidate == persisted_stair:
                self._staged_stair = None
                self.viewer.clear_canvas_stair_preview()
                self._sync_stair_calculated_values(candidate)
                self._update_stair_button_state()
                return
            preview_parts = build_canvas_stair_part_targets(
                self.levels,
                (candidate,),
                stair_indices=(stair_index,),
            )
        except (TypeError, ValueError) as error:
            self._staged_stair = None
            self.viewer.clear_canvas_stair_preview()
            self.stair_status_label.setText(f"Stair preview unavailable: {error}")
            self._sync_stair_calculated_values(None)
            self._update_stair_button_state()
            return

        self._sync_stair_calculated_values(candidate)
        self._staged_stair = candidate
        self.viewer.set_canvas_stair_preview(stair_index, preview_parts)
        self.stair_status_label.setText(
            "Previewing stair changes. Apply them or press Escape to discard."
        )
        self._update_stair_button_state()

    def _discard_staged_stair_edit(self, *, clear_selection: bool) -> None:
        """Discard the overlay and restore either persisted or creation values."""

        self._stair_preview_update_timer.stop()
        self._pending_stair_parameters = None
        self._staged_stair = None
        self.viewer.clear_canvas_stair_preview()
        stair_index = self._editing_stair_index
        if clear_selection:
            self._editing_stair_index = None
            self._desired_canvas_stair_part_ids = ()
            self._set_stair_editor_parameters(self._new_stair_parameters)
            self._sync_stair_calculated_values(None)
        elif stair_index is not None and 0 <= stair_index < len(self.stairs):
            self._load_stair_editor_from_stair(self.stairs[stair_index])
        self._update_stair_button_state()

    def _apply_staged_stair_edit(self) -> bool:
        """Commit one complete preview as a single Ctrl+Z undo action."""

        if self._pending_stair_parameters is not None:
            self._stair_preview_update_timer.stop()
            self._rebuild_staged_stair_preview()
        stair_index = self._editing_stair_index
        stair = self._staged_stair
        if (
            stair is None
            or stair_index is None
            or not 0 <= stair_index < len(self.stairs)
        ):
            return False
        try:
            build_stair_meshes(self.levels, (stair,))
        except (TypeError, ValueError) as error:
            self.stair_status_label.setText(f"Stair changes not applied: {error}")
            return False

        undo_state = self._capture_canvas_stairs_undo_state()
        self._record_canvas_undo_state(undo_state)
        self.stairs[stair_index] = stair
        self._staged_stair = None
        self.viewer.clear_canvas_stair_preview()
        self.canvas.set_stair_context(self.stairs, self.current_level)
        self._load_stair_editor_from_stair(stair)
        self._update_stair_button_state()
        self.stair_status_label.setText("Applied changes to stair.")
        assignments_changed = self._reconcile_surface_assignments_with_scene()
        self._retarget_canvas_stair_selection_after_edit(stair.stair_id)
        self._finalize_canvas_stairs_undo_state(undo_state)
        if not assignments_changed:
            self._schedule_viewer_preview_refresh(preserve_camera=True)
        return True

    def _update_stair_button_state(self) -> None:
        placement_active = self.canvas.is_stair_placement_active()
        has_complete_endpoints = self.canvas.get_stair_placement_draft() is not None
        editing_stair = (
            self._editing_stair_index is not None
            and 0 <= self._editing_stair_index < len(self.stairs)
        )
        has_staged_changes = (
            self._staged_stair is not None
            or self._pending_stair_parameters is not None
        )
        if editing_stair:
            button_text = "Apply changes to stair"
            button_enabled = has_staged_changes
        elif has_complete_endpoints:
            button_text = "Confirm stairs"
            button_enabled = True
        else:
            button_text = "Add stairs"
            button_enabled = not placement_active
        self.add_stairs_button.setText(button_text)
        self.add_stairs_button.setEnabled(button_enabled)
        for control in self._stair_parameter_controls():
            control.setEnabled(not placement_active)
        self._sync_stair_tread_edge_radius_enabled()
        self._sync_stair_starting_step_edge_radius_enabled()

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
        self.remove_doorway_preset_button.setEnabled(has_selected_preset)
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
        level_scale, level_offset_x, level_offset_y = (
            self._get_displayed_level_transform(self.current_level)
        )
        self._is_syncing_level_controls = True
        self.height_level_spinbox.setValue(self.current_level.height_meters)
        self.floor_thickness_spinbox.setValue(self.current_level.floor_thickness_meters)
        self.level_scale_slider.setValue(round(level_scale * LEVEL_SCALE_SLIDER_FACTOR))
        self._update_level_scale_value_label(level_scale)
        self.canvas_level_scale_slider.setValue(
            round(
                self.current_level.canvas_level_scale * CANVAS_LEVEL_SCALE_SLIDER_FACTOR
            )
        )
        self._update_canvas_level_scale_value_label(
            self.current_level.canvas_level_scale
        )
        level_x_slider_value = round(level_offset_x * LEVEL_OFFSET_SLIDER_FACTOR)
        self._fit_slider_range_to_value(
            self.level_x_offset_slider,
            level_x_slider_value,
        )
        self.level_x_offset_slider.setValue(level_x_slider_value)
        self._update_level_x_offset_value_label(level_offset_x)
        level_y_slider_value = round(level_offset_y * LEVEL_OFFSET_SLIDER_FACTOR)
        self._fit_slider_range_to_value(
            self.level_y_offset_slider,
            level_y_slider_value,
        )
        self.level_y_offset_slider.setValue(level_y_slider_value)
        self._update_level_y_offset_value_label(level_offset_y)
        canvas_x_slider_value = round(
            self.current_level.canvas_offset_x_pixels * CANVAS_OFFSET_SLIDER_FACTOR
        )
        self._fit_slider_range_to_value(
            self.canvas_x_offset_slider,
            canvas_x_slider_value,
        )
        self.canvas_x_offset_slider.setValue(canvas_x_slider_value)
        self._update_canvas_x_offset_value_label(
            self.current_level.canvas_offset_x_pixels
        )
        canvas_y_slider_value = round(
            self.current_level.canvas_offset_y_pixels * CANVAS_OFFSET_SLIDER_FACTOR
        )
        self._fit_slider_range_to_value(
            self.canvas_y_offset_slider,
            canvas_y_slider_value,
        )
        self.canvas_y_offset_slider.setValue(canvas_y_slider_value)
        self._update_canvas_y_offset_value_label(
            self.current_level.canvas_offset_y_pixels
        )
        self.include_yes_radio.setChecked(self.current_level.include_in_export)
        self.include_no_radio.setChecked(not self.current_level.include_in_export)
        current_level_row = self._level_list_row_for_position(self.current_level_index)
        if self.levels_list.currentRow() != current_level_row:
            self.levels_list.setCurrentRow(current_level_row)
        self._update_blueprint_name_label()
        self._update_image_correction_button_state()
        self._update_plan_wall_generation_controls_state()
        self._is_syncing_level_controls = False

    def _handle_level_list_row_changed(self, level_row: int) -> None:
        if level_row < 0:
            return
        item = self.levels_list.item(level_row)
        level_position = item.data(LEVEL_POSITION_ITEM_ROLE)
        if isinstance(level_position, bool) or not isinstance(
            level_position,
            int,
        ):
            return
        self._handle_level_selection_changed(level_position)

    def _handle_level_selection_changed(self, level_index: int) -> None:
        if (
            self._is_syncing_level_controls
            or level_index < 0
            or level_index >= len(self.levels)
        ):
            return

        if level_index != self.current_level_index:
            previous_level = self.current_level
            active_wall_detection = self._plan_wall_detection_runtimes.get(
                previous_level.index
            )
            if active_wall_detection is not None:
                self.job_manager.cancel_job(active_wall_detection.job_id)
            self._clear_plan_wall_preview()
            self.canvas.cancel_open_space_placement()
            self._finish_level_transform_drag()
            self._commit_pending_level_transform_update()
            self._finish_canvas_transform_drag()
            self._cancel_active_canvas_surface_edit()
            self._commit_pending_canvas_surface_mesh_update()
            self._commit_pending_wall_vertex_update()
            self._commit_pending_doorway_mesh_update()
        self.current_level_index = level_index
        self._sync_level_controls()
        self._sync_canvas_to_current_level()
        self._update_pending_stair_level_status()
        self._ensure_viewer_preview_current()

    def _handle_height_level_changed(self, value: float) -> None:
        if self._is_syncing_level_controls:
            return

        next_value = float(value)
        self._finish_level_transform_drag()
        self._commit_pending_level_transform_update()
        if next_value == self.current_level.height_meters:
            return
        self._record_canvas_undo_state(
            self._capture_canvas_level_properties_undo_state(self.current_level)
        )
        self.current_level.height_meters = next_value
        assignments_changed = self._reconcile_surface_assignments_with_scene()
        if not assignments_changed:
            self._schedule_viewer_preview_refresh()

    def _handle_floor_thickness_changed(self, value: float) -> None:
        """Stage a floor thickness change for one delayed mesh rebuild."""

        if self._is_syncing_level_controls:
            return
        self._finish_level_transform_drag()
        self._commit_pending_level_transform_update()
        self._stage_floor_thickness_mesh_update(
            self.current_level,
            float(value),
        )

    # ### Open-space controls ###
    def _handle_add_open_space_clicked(self) -> None:
        """Toggle the one-shot rectangular floor-opening tool."""

        if self.canvas.is_open_space_placement_active():
            self.canvas.cancel_open_space_placement()
            return
        if self.canvas.blueprint_image is None:
            self._update_open_space_controls()
            return

        self._finish_level_transform_drag()
        self._finish_canvas_transform_drag()
        self._cancel_active_canvas_surface_edit()
        self._commit_pending_canvas_surface_mesh_update()
        self._commit_pending_wall_vertex_update()
        self._commit_pending_doorway_mesh_update()
        self.workspace_tabs.setCurrentWidget(self.canvas_viewer_workspace)
        self.canvas.start_open_space_placement()

    def _handle_open_spaces_changed(self) -> None:
        """Keep hole UI current and retire unrelated delayed mesh work."""

        if self._pending_wall_vertex_mesh_update:
            self._commit_pending_wall_vertex_update()
        if self._pending_doorway_mesh_level_index is not None:
            self._commit_pending_doorway_mesh_update()
        self._update_open_space_controls()

    def _handle_open_space_placement_changed(self, active: bool) -> None:
        """Keep the side-panel tool state synchronized with the Canvas."""

        self._update_open_space_controls(bool(active))

    def _update_open_space_controls(
        self,
        placement_active: bool | None = None,
    ) -> None:
        """Describe persisted rectangles and the current drag mode."""

        active = (
            self.canvas.is_open_space_placement_active()
            if placement_active is None
            else bool(placement_active)
        )
        self.add_open_space_button.setChecked(active)
        self.add_open_space_button.setText(
            "Cancel open space" if active else "Add open space"
        )
        self.add_open_space_button.setEnabled(
            active or self.canvas.blueprint_image is not None
        )
        if active:
            self.open_space_status_label.setText(
                "Drag a rectangle on the 2D Canvas. Escape cancels."
            )
            return

        open_space_count = len(self.current_level.open_spaces)
        if open_space_count == 0:
            status = "Open spaces: none"
        elif open_space_count == 1:
            status = "Open spaces: 1 area"
        else:
            status = f"Open spaces: {open_space_count} areas"
        self.open_space_status_label.setText(status)

    # ### 3D level transform bars ###
    def _get_displayed_level_transform(
        self,
        level: LevelData,
    ) -> tuple[float, float, float]:
        """Return pending values for the edited level and committed ones otherwise."""

        pending = self._pending_level_transform
        if pending is not None and pending.level_index == level.index:
            return (
                pending.scale,
                pending.offset_x_meters,
                pending.offset_y_meters,
            )
        return (
            float(level.scale),
            float(level.offset_x_meters),
            float(level.offset_y_meters),
        )

    def _preview_level_scale_slider_value(self, slider_value: int) -> None:
        next_value = float(slider_value) / LEVEL_SCALE_SLIDER_FACTOR
        self._update_level_scale_value_label(next_value)
        self._handle_level_scale_changed(next_value)

    def _preview_level_x_offset_slider_value(self, slider_value: int) -> None:
        next_value = float(slider_value) / LEVEL_OFFSET_SLIDER_FACTOR
        self._update_level_x_offset_value_label(next_value)
        self._handle_level_x_offset_changed(next_value)

    def _preview_level_y_offset_slider_value(self, slider_value: int) -> None:
        next_value = float(slider_value) / LEVEL_OFFSET_SLIDER_FACTOR
        self._update_level_y_offset_value_label(next_value)
        self._handle_level_y_offset_changed(next_value)

    def _handle_level_scale_slider_changed(self, slider_value: int) -> None:
        next_value = float(slider_value) / LEVEL_SCALE_SLIDER_FACTOR
        self._update_level_scale_value_label(next_value)
        self._handle_level_scale_changed(next_value)

    def _handle_level_x_offset_slider_changed(self, slider_value: int) -> None:
        next_value = float(slider_value) / LEVEL_OFFSET_SLIDER_FACTOR
        self._update_level_x_offset_value_label(next_value)
        self._handle_level_x_offset_changed(next_value)

    def _handle_level_y_offset_slider_changed(self, slider_value: int) -> None:
        next_value = float(slider_value) / LEVEL_OFFSET_SLIDER_FACTOR
        self._update_level_y_offset_value_label(next_value)
        self._handle_level_y_offset_changed(next_value)

    def _handle_level_transform_drag_started(self) -> None:
        """Pause the mesh debounce while a 3D transform bar is held."""

        if self._is_syncing_level_controls or self._level_transform_drag_active:
            return
        self._commit_pending_canvas_surface_mesh_update()
        self._level_transform_mesh_update_timer.stop()
        self._level_transform_drag_active = True

    def _handle_level_transform_button_pressed(
        self,
        slider: QSlider,
        direction: int,
    ) -> None:
        """Nudge one 3D transform and retain its comparison while held."""

        if self._is_syncing_level_controls:
            return
        needs_comparison = not self._level_transform_drag_active
        self._handle_level_transform_drag_started()
        slider.setValue(slider.value() + direction * slider.singleStep())
        if needs_comparison or self.canvas.get_level_comparison_overlay() is None:
            self.canvas.set_level_comparison_overlay(
                self._get_canvas_transform_comparison_level()
            )

    def _handle_level_transform_button_released(self) -> None:
        """End a 3D transform-button gesture and begin its delayed update."""

        self.canvas.clear_level_comparison_overlay()
        self._handle_level_transform_drag_finished()

    def _handle_level_transform_drag_finished(self) -> None:
        self._finish_level_transform_drag()

    def _finish_level_transform_drag(self) -> None:
        """Start the full mesh-edit delay after a transform bar is released."""

        if not self._level_transform_drag_active:
            return
        self._level_transform_drag_active = False
        if self._pending_level_transform is not None:
            self._level_transform_mesh_update_timer.start()

    def _handle_level_scale_changed(self, value: float) -> None:
        next_value = float(value)
        self._update_level_scale_value_label(next_value)
        if self._is_syncing_level_controls:
            return
        self._stage_pending_level_transform(scale=next_value)

    def _handle_level_x_offset_changed(self, value: float) -> None:
        next_value = float(value)
        self._update_level_x_offset_value_label(next_value)
        if self._is_syncing_level_controls:
            return
        self._stage_pending_level_transform(offset_x_meters=next_value)

    def _handle_level_y_offset_changed(self, value: float) -> None:
        next_value = float(value)
        self._update_level_y_offset_value_label(next_value)
        if self._is_syncing_level_controls:
            return
        self._stage_pending_level_transform(offset_y_meters=next_value)

    def _stage_pending_level_transform(
        self,
        *,
        scale: float | None = None,
        offset_x_meters: float | None = None,
        offset_y_meters: float | None = None,
    ) -> None:
        """Coalesce level transforms while retaining one immutable baseline."""

        level = self.current_level
        pending = self._pending_level_transform
        if pending is not None and pending.level_index != level.index:
            self._commit_pending_level_transform_update()
            pending = None
        if pending is None:
            self._finish_canvas_transform_drag()
            self._commit_pending_canvas_surface_mesh_update()
            self._commit_pending_wall_vertex_update()
            self._commit_pending_doorway_mesh_update()
            pending = _PendingLevelTransform(
                baseline=self._capture_canvas_level_properties_undo_state(level),
                scale=float(level.scale),
                offset_x_meters=float(level.offset_x_meters),
                offset_y_meters=float(level.offset_y_meters),
            )

        next_pending = replace(
            pending,
            scale=pending.scale if scale is None else float(scale),
            offset_x_meters=(
                pending.offset_x_meters
                if offset_x_meters is None
                else float(offset_x_meters)
            ),
            offset_y_meters=(
                pending.offset_y_meters
                if offset_y_meters is None
                else float(offset_y_meters)
            ),
        )
        baseline = next_pending.baseline
        if (
            math.isclose(next_pending.scale, baseline.scale)
            and math.isclose(
                next_pending.offset_x_meters,
                baseline.offset_x_meters,
            )
            and math.isclose(
                next_pending.offset_y_meters,
                baseline.offset_y_meters,
            )
        ):
            self._cancel_pending_level_transform(sync_controls=False)
            return

        self._pending_level_transform = next_pending
        self._refresh_pending_level_transform_outline()
        if self._level_transform_drag_active:
            self._level_transform_mesh_update_timer.stop()
        else:
            self._level_transform_mesh_update_timer.start()

    def _refresh_pending_level_transform_outline(self) -> None:
        """Transform the rendered level boundary without rebuilding its mesh."""

        pending = self._pending_level_transform
        if pending is None:
            self.viewer.clear_level_transform_preview()
            return
        level = next(
            (
                candidate
                for candidate in self.levels
                if candidate.index == pending.level_index
            ),
            None,
        )
        if level is None:
            self._cancel_pending_level_transform(sync_controls=False)
            return

        committed_scale = float(level.scale)
        scale_ratio = pending.scale / committed_scale
        pivot_x, pivot_y = get_level_world_pivot(level)
        translation_x = (
            pivot_x * (1.0 - scale_ratio)
            + pending.offset_x_meters
            - scale_ratio * float(level.offset_x_meters)
        )
        translation_y = (
            pivot_y * (1.0 - scale_ratio)
            + pending.offset_y_meters
            - scale_ratio * float(level.offset_y_meters)
        )
        self.viewer.set_level_transform_preview(
            pending.level_index,
            (
                (scale_ratio, 0.0, 0.0, translation_x),
                (0.0, scale_ratio, 0.0, translation_y),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )

    def _cancel_pending_level_transform(
        self,
        *,
        sync_controls: bool = True,
        restore_canvas_tools: bool = True,
    ) -> bool:
        """Discard a staged transform and its yellow preview."""

        had_pending = self._pending_level_transform is not None
        self._level_transform_mesh_update_timer.stop()
        self._pending_level_transform = None
        self._level_transform_outline_commit_revision = None
        self.viewer.clear_level_transform_preview(
            restore_canvas_tools=restore_canvas_tools
        )
        if sync_controls and hasattr(self, "level_scale_slider"):
            self._sync_level_controls()
        return had_pending

    def _commit_pending_level_transform_update(self) -> bool:
        """Apply one delayed scale/offset transaction and schedule its mesh."""

        self._level_transform_mesh_update_timer.stop()
        pending = self._pending_level_transform
        if pending is None:
            return False
        self._level_transform_drag_active = False
        level = next(
            (
                candidate
                for candidate in self.levels
                if candidate.index == pending.level_index
            ),
            None,
        )
        if level is None:
            self._cancel_pending_level_transform(sync_controls=False)
            return False

        self._pending_level_transform = None
        changed = (
            not math.isclose(float(level.scale), pending.scale)
            or not math.isclose(
                float(level.offset_x_meters),
                pending.offset_x_meters,
            )
            or not math.isclose(
                float(level.offset_y_meters),
                pending.offset_y_meters,
            )
        )
        if not changed:
            self.viewer.clear_level_transform_preview()
            return False

        level.scale = pending.scale
        level.offset_x_meters = pending.offset_x_meters
        level.offset_y_meters = pending.offset_y_meters
        self._record_canvas_undo_state(
            pending.baseline,
            commit_pending_surface_edit=False,
        )
        if level is self.current_level:
            self.canvas.update()
        self._sync_canvas_wall_mirror_state()
        self._schedule_viewer_preview_refresh(preserve_camera=True)
        self._level_transform_outline_commit_revision = self._viewer_preview_revision
        return True

    def _clear_committed_level_transform_outline_if_displayed(self) -> None:
        """Retire the yellow outline after Canvas shows the committed mesh."""

        target_revision = self._level_transform_outline_commit_revision
        if (
            target_revision is None
            or self._pending_level_transform is not None
            or self._canvas_viewer_preview_revision < target_revision
        ):
            return
        self._level_transform_outline_commit_revision = None
        self.viewer.clear_level_transform_preview()

    def _update_level_scale_value_label(self, value: float) -> None:
        self.level_scale_value_label.setText(f"{float(value):.3f} x")

    def _update_level_x_offset_value_label(self, value: float) -> None:
        self.level_x_offset_value_label.setText(f"{float(value):.2f} m")

    def _update_level_y_offset_value_label(self, value: float) -> None:
        self.level_y_offset_value_label.setText(f"{float(value):.2f} m")

    # ### 2D Canvas transform bars ###
    def _handle_canvas_transform_drag_started(self) -> None:
        """Start one live Canvas transform and reveal its comparison level."""

        if self._is_syncing_level_controls or self._canvas_transform_drag_active:
            return
        self._finish_level_transform_drag()
        self._commit_pending_level_transform_update()
        self._commit_pending_canvas_surface_mesh_update()
        self._canvas_transform_drag_active = True
        self._canvas_transform_drag_undo_state = (
            self._capture_canvas_level_properties_undo_state(self.current_level)
        )
        self.canvas.set_level_comparison_overlay(
            self._get_canvas_transform_comparison_level()
        )

    def _handle_canvas_transform_button_pressed(
        self,
        slider: QSlider,
        direction: int,
    ) -> None:
        """Nudge one Canvas transform and keep its comparison visible."""

        if self._is_syncing_level_controls:
            return
        self._handle_canvas_transform_drag_started()
        slider.setValue(slider.value() + direction * slider.singleStep())

    def _handle_canvas_offset_button_pressed(
        self,
        axis_name: str,
        _slider: QSlider,
        direction: int,
    ) -> None:
        """Nudge one Canvas offset without discarding persisted subpixels."""

        if self._is_syncing_level_controls:
            return
        self._handle_canvas_transform_drag_started()
        self._nudge_canvas_offset(axis_name, float(direction))

    def _handle_canvas_transform_button_released(self) -> None:
        """Finish a Canvas transform-button gesture and hide its comparison."""

        self._handle_canvas_transform_drag_finished()

    def _handle_canvas_level_scale_changed(self, slider_value: int) -> None:
        """Apply one Canvas scale slider step without changing 3D geometry."""

        next_value = float(slider_value) / CANVAS_LEVEL_SCALE_SLIDER_FACTOR
        self._update_canvas_level_scale_value_label(next_value)
        if self._is_syncing_level_controls:
            return
        self._finish_level_transform_drag()
        self._commit_pending_level_transform_update()
        if math.isclose(next_value, self.current_level.canvas_level_scale):
            return
        if not self._canvas_transform_drag_active:
            self._record_canvas_undo_state(
                self._capture_canvas_level_properties_undo_state(self.current_level)
            )
        self.current_level.canvas_level_scale = next_value
        self.canvas.set_canvas_level_scale(next_value)

    def _handle_canvas_x_offset_changed(self, slider_value: int) -> None:
        """Apply one horizontal Canvas translation step immediately."""

        next_value = float(slider_value) / CANVAS_OFFSET_SLIDER_FACTOR
        self._update_canvas_x_offset_value_label(next_value)
        if self._is_syncing_level_controls:
            return
        self._finish_level_transform_drag()
        self._commit_pending_level_transform_update()
        if math.isclose(next_value, self.current_level.canvas_offset_x_pixels):
            return
        if not self._canvas_transform_drag_active:
            self._record_canvas_undo_state(
                self._capture_canvas_level_properties_undo_state(self.current_level)
            )
        self.current_level.canvas_offset_x_pixels = next_value
        self.canvas.set_canvas_level_offsets(
            next_value,
            self.current_level.canvas_offset_y_pixels,
        )

    def _handle_canvas_y_offset_changed(self, slider_value: int) -> None:
        """Apply one vertical Canvas translation step immediately."""

        next_value = float(slider_value) / CANVAS_OFFSET_SLIDER_FACTOR
        self._update_canvas_y_offset_value_label(next_value)
        if self._is_syncing_level_controls:
            return
        self._finish_level_transform_drag()
        self._commit_pending_level_transform_update()
        if math.isclose(next_value, self.current_level.canvas_offset_y_pixels):
            return
        if not self._canvas_transform_drag_active:
            self._record_canvas_undo_state(
                self._capture_canvas_level_properties_undo_state(self.current_level)
            )
        self.current_level.canvas_offset_y_pixels = next_value
        self.canvas.set_canvas_level_offsets(
            self.current_level.canvas_offset_x_pixels,
            next_value,
        )

    def _handle_canvas_transform_drag_finished(self) -> None:
        """Commit one Canvas transform drag and remove its comparison."""

        self._finish_canvas_transform_drag()

    def _finish_canvas_transform_drag(self) -> None:
        """Finalize any active Canvas scale or translation gesture."""

        if not self._canvas_transform_drag_active:
            self.canvas.clear_level_comparison_overlay()
            return
        state = self._canvas_transform_drag_undo_state
        self._canvas_transform_drag_active = False
        self._canvas_transform_drag_undo_state = None
        self.canvas.clear_level_comparison_overlay()
        if state is None:
            return
        changed = (
            not math.isclose(
                state.canvas_level_scale,
                self.current_level.canvas_level_scale,
            )
            or not math.isclose(
                state.canvas_offset_x_pixels,
                self.current_level.canvas_offset_x_pixels,
            )
            or not math.isclose(
                state.canvas_offset_y_pixels,
                self.current_level.canvas_offset_y_pixels,
            )
        )
        if changed:
            self._record_canvas_undo_state(
                state,
                commit_pending_surface_edit=False,
            )

    def _get_canvas_transform_comparison_level(self) -> LevelData | None:
        """Return the level below, except underground levels compare upward."""

        current_index = self.current_level.index
        comparison_index = (
            current_index + 1
            if current_index < GROUND_LEVEL_INDEX
            else current_index - 1
        )
        return next(
            (level for level in self.levels if level.index == comparison_index),
            None,
        )

    def _update_canvas_level_scale_value_label(self, value: float) -> None:
        """Keep the numeric readout adjacent to the scale bar."""

        self.canvas_level_scale_value_label.setText(f"{float(value):.2f} x")

    def _update_canvas_x_offset_value_label(self, value: float) -> None:
        self.canvas_x_offset_value_label.setText(
            self._format_canvas_offset_pixels(value)
        )

    def _update_canvas_y_offset_value_label(self, value: float) -> None:
        self.canvas_y_offset_value_label.setText(
            self._format_canvas_offset_pixels(value)
        )

    @staticmethod
    def _format_canvas_offset_pixels(value: float) -> str:
        """Show persisted subpixels without cluttering integer offsets."""

        rounded_value = round(float(value), 6)
        if rounded_value == 0.0:
            rounded_value = 0.0
        value_text = f"{rounded_value:.6f}".rstrip("0").rstrip(".")
        return f"{value_text} px"

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
        self._refresh_doorway_preset_list(selected_index=len(self.doorway_presets) - 1)

    def _handle_remove_doorway_preset_clicked(self) -> None:
        selected_index = self.doorway_preset_list.currentRow()
        if (
            selected_index < 0
            or selected_index >= len(self.doorway_presets)
        ):
            return

        del self.doorway_presets[selected_index]
        if not self.doorway_presets:
            self.doorway_presets.append(create_fallback_doorway_preset())
        next_selected_index = min(selected_index, len(self.doorway_presets) - 1)
        self._refresh_doorway_preset_list(selected_index=next_selected_index)

    def _handle_place_selected_doorway_clicked(self) -> None:
        doorway_preset = self._get_selected_doorway_preset()
        if doorway_preset is None:
            return

        self.canvas.start_doorway_placement(doorway_preset)
        self.workspace_tabs.setCurrentWidget(self.canvas_viewer_workspace)

    # ### Wall level mirror controls ###
    def _handle_wall_mirror_up_clicked(self) -> None:
        """Mirror the selected wall-vertex group onto the next upper level."""

        self._mirror_selected_wall_vertices(1)

    def _handle_wall_mirror_down_clicked(self) -> None:
        """Mirror the selected wall-vertex group onto the next lower level."""

        self._mirror_selected_wall_vertices(-1)

    def _handle_wall_mirror_undo_clicked(self) -> None:
        """Remove mirrors owned by, or materialized as, selected vertices."""

        selected_vertex_ids = self._get_selected_canvas_wall_vertex_ids()
        if not self._selected_vertices_have_wall_mirrors(selected_vertex_ids):
            self._update_wall_mirror_button_state()
            return

        self._commit_pending_wall_mirror_prerequisites()
        undo_state = self._capture_canvas_wall_mirror_undo_state()
        result = remove_wall_vertex_mirrors(
            self.levels,
            self.wall_mirror_links,
            self.current_level.index,
            selected_vertex_ids,
        )
        if not result.changed:
            self._update_wall_mirror_button_state()
            return
        self._record_canvas_undo_state(
            undo_state,
            commit_pending_surface_edit=False,
        )
        self.wall_mirror_links = result.links
        self._finalize_wall_mirror_topology_change(result)

    def _mirror_selected_wall_vertices(self, direction: int) -> None:
        """Copy one selected group and its internal edges to another level."""

        selected_vertex_ids = self._get_selected_canvas_wall_vertex_ids()
        if not selected_vertex_ids:
            self._update_wall_mirror_button_state()
            return
        target_level_index = find_next_wall_mirror_target_level_index(
            self.levels,
            self.wall_mirror_links,
            self.current_level.index,
            selected_vertex_ids,
            direction,
        )
        if target_level_index is None:
            self._update_wall_mirror_button_state()
            return

        self._commit_pending_wall_mirror_prerequisites()
        undo_state = self._capture_canvas_wall_mirror_undo_state()
        result = mirror_wall_vertex_group(
            self.levels,
            self.wall_mirror_links,
            self.current_level.index,
            selected_vertex_ids,
            target_level_index,
        )
        if not result.changed:
            self._update_wall_mirror_button_state()
            return
        self._record_canvas_undo_state(
            undo_state,
            commit_pending_surface_edit=False,
        )
        self.wall_mirror_links = result.links
        self._finalize_wall_mirror_topology_change(result)

    def _commit_pending_wall_mirror_prerequisites(self) -> None:
        """Commit older delayed edits before a cross-level topology action."""

        self._finish_level_transform_drag()
        self._commit_pending_level_transform_update()
        self._commit_pending_canvas_surface_mesh_update()
        self._commit_pending_wall_vertex_update()

    def _capture_canvas_wall_mirror_undo_state(
        self,
    ) -> _CanvasWallMirrorUndoState:
        """Snapshot cross-level wall topology and its current selection."""

        return _CanvasWallMirrorUndoState(
            vertex_data_by_level=tuple(
                (level.index, level.vertex_data.clone()) for level in self.levels
            ),
            doorways_by_level=tuple(
                (level.index, self._copy_doorways(level.doorways))
                for level in self.levels
            ),
            wall_mirror_links=self.wall_mirror_links,
            selected_level_index=self.current_level.index,
            selected_vertex_ids=self.canvas.selected_vertex_ids,
        )

    def _finalize_wall_mirror_topology_change(
        self,
        result: WallMirrorTopologyResult,
    ) -> None:
        """Refresh mirror markers and debounce one resulting mesh rebuild."""

        self._sync_canvas_wall_mirror_state()
        self._update_wall_mirror_button_state()
        if result.changed_level_indices:
            self._pending_wall_vertex_doorway_level_indices.update(
                result.changed_level_indices
            )
            self._pending_wall_vertex_mesh_update = True
            self._restart_pending_wall_vertex_update_if_idle()

    def _handle_canvas_selected_vertex_changed(
        self,
        _vertex_ids: object,
    ) -> None:
        """Keep Wall mirrors availability aligned with Canvas selection."""

        self._update_wall_mirror_button_state()

    def _update_wall_mirror_button_state(self) -> None:
        """Enable only mirror operations valid for the selected vertex group."""

        if not hasattr(self, "wall_mirror_up_button"):
            return
        selected_vertex_ids = self._get_selected_canvas_wall_vertex_ids()
        has_selection = bool(selected_vertex_ids)
        upper_target_index = (
            find_next_wall_mirror_target_level_index(
                self.levels,
                self.wall_mirror_links,
                self.current_level.index,
                selected_vertex_ids,
                1,
            )
            if has_selection
            else None
        )
        lower_target_index = (
            find_next_wall_mirror_target_level_index(
                self.levels,
                self.wall_mirror_links,
                self.current_level.index,
                selected_vertex_ids,
                -1,
            )
            if has_selection
            else None
        )
        self.wall_mirror_up_button.setEnabled(upper_target_index is not None)
        self.wall_mirror_undo_button.setEnabled(
            self._selected_vertices_have_wall_mirrors(selected_vertex_ids)
        )
        self.wall_mirror_down_button.setEnabled(lower_target_index is not None)

    def _get_selected_canvas_wall_vertex_ids(self) -> tuple[int, ...]:
        """Return valid selected vertices, excluding room-center helpers."""

        vertex_data = self.current_level.vertex_data
        room_center_vertex_ids = {
            room.center_vertex_id for room in self.current_level.rooms
        }
        wall_vertex_ids = {
            vertex_id
            for edge in vertex_data.edges
            for vertex_id in (edge.start_vertex_id, edge.end_vertex_id)
        }
        wall_vertex_ids.update(
            get_wall_mirror_vertex_ids(
                self.wall_mirror_links,
                self.current_level.index,
            )
        )
        return tuple(
            vertex_id
            for vertex_id in self.canvas.selected_vertex_ids
            if vertex_data.get_vertex(vertex_id) is not None
            and vertex_id not in room_center_vertex_ids
            and vertex_id in wall_vertex_ids
        )

    def _selected_vertices_have_wall_mirrors(
        self,
        selected_vertex_ids: Sequence[int],
    ) -> bool:
        """Return whether selected vertices own or are owned mirror copies."""

        selected_ids = set(selected_vertex_ids)
        level_index = self.current_level.index
        return any(
            (
                link.source_level_index == level_index
                and link.source_vertex_id in selected_ids
            )
            or (
                link.target_level_index == level_index
                and link.target_vertex_id in selected_ids
            )
            for link in self.wall_mirror_links
        )

    def _get_level_by_index(self, level_index: int) -> LevelData | None:
        """Resolve a persistent level by elevation index, never list row."""

        return next(
            (level for level in self.levels if level.index == level_index),
            None,
        )

    def _sync_canvas_wall_mirror_state(self) -> WallMirrorTopologyResult:
        """Reconcile owned copies and mark local mirror endpoints in green."""

        result = reconcile_wall_mirror_topology(
            self.levels,
            self.wall_mirror_links,
        )
        self._pending_wall_vertex_doorway_level_indices.update(
            result.changed_level_indices
        )
        self.wall_mirror_links = result.links
        self.canvas.set_wall_mirror_vertex_ids(
            get_wall_mirror_vertex_ids(
                self.wall_mirror_links,
                self.current_level.index,
            )
        )
        self.canvas.prune_missing_vertex_references()
        self.canvas.update()
        return result

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
        amount_blocker = QSignalBlocker(self.selected_doorway_arch_amount_spinbox)
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

        shape = DOORWAY_SHAPE_ARCH if enabled else DOORWAY_SHAPE_RECTANGULAR
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

        if self.canvas.set_selected_doorway_arch_amount(arch_amount_percent / 100.0):
            return
        self._handle_canvas_doorway_selection_changed(
            -1
            if self.canvas.selected_doorway_index is None
            else self.canvas.selected_doorway_index
        )

    def _handle_doorways_changed(self) -> None:
        """Commit structural doorway changes without a debounce delay."""

        self._is_doorway_move_drag_active = False
        self._is_doorway_resize_drag_active = False
        self.current_level.doorways = self.canvas.doorways
        next_snapshot = self._copy_doorways(self.current_level.doorways)
        snapshot_changed = bool(
            self._viewer_doorways_by_level_index.get(self.current_level.index)
            != next_snapshot
        )
        self._cancel_pending_doorway_mesh_update(clear_outline=True)
        if not snapshot_changed:
            return
        self._viewer_doorways_by_level_index[self.current_level.index] = next_snapshot
        self._schedule_viewer_preview_refresh()

    def _handle_doorway_move_drag_started(self) -> None:
        """Hold the old wall mesh while a doorway is moving in the Canvas."""

        self._is_doorway_move_drag_active = True
        self._doorway_mesh_update_timer.stop()

    def _handle_doorway_move_drag_finished(self, changed: bool) -> None:
        """Start the shared mesh-edit delay after a 2D move is released."""

        self._is_doorway_move_drag_active = False
        if changed and self._pending_doorway_mesh_level_index is None:
            self._handle_doorway_dimension_preview_changed()
        if self._pending_doorway_mesh_level_index is not None:
            self._doorway_mesh_update_timer.start()

    def _handle_doorway_resize_drag_started(self) -> None:
        """Pause doorway mesh rebuilding while a 2D side handle is held."""

        self._is_doorway_resize_drag_active = True
        self._doorway_mesh_update_timer.stop()

    def _handle_doorway_resize_drag_finished(self, changed: bool) -> None:
        """Start the shared mesh-edit delay after a 2D resize is released."""

        self._is_doorway_resize_drag_active = False
        if changed and self._pending_doorway_mesh_level_index is None:
            self._handle_doorway_dimension_preview_changed()
        if self._pending_doorway_mesh_level_index is not None:
            self._doorway_mesh_update_timer.start()

    def _handle_doorway_dimension_preview_changed(self) -> None:
        """Show a live doorway outline and debounce the wall mesh rebuild."""

        self.current_level.doorways = self.canvas.doorways
        level = self.current_level
        committed_doorways = self._viewer_doorways_by_level_index.get(level.index)
        if committed_doorways is None:
            committed_doorways = self._copy_doorways(level.doorways)
            self._viewer_doorways_by_level_index[level.index] = committed_doorways

        if self._copy_doorways(level.doorways) == committed_doorways:
            self._doorway_mesh_update_timer.stop()
            self._pending_doorway_mesh_level_index = None
            doorway_key_prefix = f"doorway:{level.index}:"
            if (
                self._pending_canvas_opening_key is not None
                and self._pending_canvas_opening_key.startswith(doorway_key_prefix)
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
        if pending_level_index is not None and pending_level_index != level.index:
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
        self._pending_canvas_opening_key = f"doorway:{level.index}:{selected_index}"
        if (
            self._is_doorway_move_drag_active
            or self._is_doorway_resize_drag_active
        ):
            self._doorway_mesh_update_timer.stop()
        else:
            self._doorway_mesh_update_timer.start()

    def _handle_add_stairs_clicked(self) -> None:
        if (
            self._staged_stair is not None
            or self._pending_stair_parameters is not None
        ):
            self._apply_staged_stair_edit()
            return
        draft = self.canvas.get_stair_placement_draft()
        if draft is not None:
            if self.canvas.is_stair_ready_for_confirmation():
                try:
                    stair = _build_stair_data_from_placement(
                        draft,
                        self._read_stair_editor_parameters(),
                    )
                    build_stair_meshes(self.levels, [stair])
                except (TypeError, ValueError) as error:
                    self.stair_status_label.setText(f"Stair not added: {error}")
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

        if self._editing_stair_index is not None:
            self.viewer.set_selected_canvas_stair_part_ids(())
            self._discard_staged_stair_edit(clear_selection=True)
        parameters = self._read_stair_editor_parameters()
        self._new_stair_parameters = parameters
        self.workspace_tabs.setCurrentWidget(self.canvas_viewer_workspace)
        self.canvas.start_stair_placement(
            parameters.stair_type
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
        self._sync_stair_calculated_values(None)
        self._update_stair_button_state()

    def _handle_stair_placement_ready(self, placement: object) -> None:
        try:
            preview_stair = _build_stair_data_from_placement(
                placement,
                self._read_stair_editor_parameters(),
            )
        except (TypeError, ValueError):
            self._sync_stair_calculated_values(None)
        else:
            self._sync_stair_calculated_values(preview_stair)
        intermediate_count = len(_get_stair_intermediate_section_payloads(placement))
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
            stair = _build_stair_data_from_placement(
                placement,
                self._read_stair_editor_parameters(),
            )
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

        undo_state = self._capture_canvas_stairs_undo_state()
        self._record_canvas_undo_state(undo_state)
        self.stairs.append(stair)
        self.canvas.set_stair_context(self.stairs, self.current_level)
        self.stair_status_label.setText(
            "Added "
            f"{_format_stair_style_label(stair.style).lower()} stairs from "
            f"{_format_level_name(self.levels, stair.start_level_index)} to "
            f"{_format_level_name(self.levels, stair.end_level_index)}."
        )
        self._sync_stair_calculated_values(stair)
        self._update_stair_button_state()
        assignments_changed = self._reconcile_surface_assignments_with_scene()
        self._finalize_canvas_stairs_undo_state(undo_state)
        # A stair can extend beyond the previously framed house bounds. Refit
        # the Canvas 3D view so a successful placement is visible immediately.
        if not assignments_changed:
            self._schedule_viewer_preview_refresh(preserve_camera=False)

    def _handle_stair_placement_cancelled(self) -> None:
        self.stair_status_label.setText("Stair placement cancelled.")
        self._sync_stair_calculated_values(None)
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
                "Load a blueprint image on this level before placing the stair end."
            )
            return

        self.stair_status_label.setText(
            "Click two points for the stair opening on "
            f"{self.current_level.display_name}."
        )

    # ### Canvas stair point edits ###
    def _handle_stair_point_drag_finished(
        self,
        stair_index: int,
        endpoint_name: str,
        image_x: float,
        image_y: float,
        changed: bool,
    ) -> None:
        """Save a released 2D point edit and debounce its 3D mesh rebuild."""

        if not changed or not 0 <= stair_index < len(self.stairs):
            return
        if not math.isfinite(image_x) or not math.isfinite(image_y):
            return
        stair = self.stairs[stair_index]
        section_name, separator, side = endpoint_name.rpartition("_")
        if not separator or side not in ("a", "b"):
            return
        coordinates = {
            f"{side}_x": image_x,
            f"{side}_y": image_y,
            f"{side}_vertex_id": None,
        }
        try:
            if section_name in ("start", "end"):
                candidate = replace(
                    stair,
                    **{
                        f"{section_name}_{key}": value
                        for key, value in coordinates.items()
                    },
                )
            elif section_name.startswith("intermediate_"):
                section_index = int(section_name.removeprefix("intermediate_"))
                sections = list(stair.intermediate_sections)
                if not 0 <= section_index < len(sections):
                    return
                sections[section_index] = replace(
                    sections[section_index], **coordinates
                )
                candidate = replace(stair, intermediate_sections=tuple(sections))
            else:
                return
            if candidate == stair:
                return
            build_stair_meshes(self.levels, (candidate,))
        except (TypeError, ValueError) as error:
            self.stair_status_label.setText(f"Stair point not moved: {error}")
            return

        undo_state = self._capture_canvas_stairs_undo_state()
        self._record_canvas_undo_state(undo_state)
        self.stairs[stair_index] = candidate
        self.canvas.set_stair_context(self.stairs, self.current_level)
        if self._editing_stair_index == stair_index:
            self._sync_stair_calculated_values(candidate)
        self._pending_stair_point_mesh_update = True
        self._pending_stair_point_undo_state = undo_state
        self._pending_stair_point_id = candidate.stair_id
        self._stair_point_mesh_update_timer.start()
        self.stair_status_label.setText("Stair point moved; waiting to update mesh.")

    def _commit_pending_stair_point_mesh_update(self) -> None:
        """Rebuild a moved stair only after the shared mesh-edit delay."""

        self._stair_point_mesh_update_timer.stop()
        if not self._pending_stair_point_mesh_update:
            return
        self._pending_stair_point_mesh_update = False
        undo_state = self._pending_stair_point_undo_state
        stair_id = self._pending_stair_point_id
        self._pending_stair_point_undo_state = None
        self._pending_stair_point_id = None
        assignments_changed = self._reconcile_surface_assignments_with_scene()
        if stair_id is not None:
            self._retarget_canvas_stair_selection_after_edit(stair_id)
        if undo_state is not None:
            self._finalize_canvas_stairs_undo_state(undo_state)
        if not assignments_changed:
            self._schedule_viewer_preview_refresh(preserve_camera=True)
        self.stair_status_label.setText("Stair mesh updated.")

    def _handle_stair_delete_requested(self, stair_index: int) -> None:
        self._delete_stair_at_index(stair_index)

    def _handle_canvas_stair_deletion_requested(
        self,
        raw_stair_ids: object,
    ) -> None:
        """Delete the owner of each selected 3D stair part once."""

        try:
            stair_ids = {
                str(value).strip()
                for value in raw_stair_ids  # type: ignore[arg-type]
            }
        except TypeError:
            return
        stair_indices = sorted(
            (
                index
                for index, stair in enumerate(self.stairs)
                if stair.stair_id in stair_ids
            ),
            reverse=True,
        )
        for stair_index in stair_indices:
            self._delete_stair_at_index(stair_index)

    def _delete_stair_at_index(self, stair_index: int) -> None:
        if not 0 <= stair_index < len(self.stairs):
            return

        self._stair_point_mesh_update_timer.stop()
        self._pending_stair_point_mesh_update = False
        self._pending_stair_point_undo_state = None
        self._pending_stair_point_id = None
        undo_state = self._capture_canvas_stairs_undo_state()
        self._record_canvas_undo_state(undo_state)
        editing_index = self._editing_stair_index
        if editing_index == stair_index:
            self.viewer.set_selected_canvas_stair_part_ids(())
            self._discard_staged_stair_edit(clear_selection=True)
        elif editing_index is not None and editing_index > stair_index:
            self._editing_stair_index = editing_index - 1
        del self.stairs[stair_index]
        self.canvas.set_stair_context(self.stairs, self.current_level)
        self._update_stair_button_state()
        self.stair_status_label.setText("Stair deleted.")
        assignments_changed = self._reconcile_surface_assignments_with_scene()
        self._finalize_canvas_stairs_undo_state(undo_state)
        if not assignments_changed:
            self._schedule_viewer_preview_refresh()

    def _handle_generation_settings_changed(self) -> None:
        settings = self.settings_widget.get_settings()
        self._generation_settings = settings
        self._set_mesh_edit_update_delay_seconds(
            settings.mesh_edit_update_delay_seconds
        )
        self._set_wall_vertex_update_delay_seconds(
            settings.wall_vertex_update_delay_seconds
        )
        self._set_canvas_3d_navigation_shortcut(
            settings.canvas_3d_navigation_toggle_hotkey
        )
        self.viewer.set_first_person_movement_mode(
            settings.first_person_navigation_mode
        )
        self.viewer.set_ignore_top_down_ceiling(
            settings.ignore_top_down_ceiling
        )
        self.viewer.set_hide_stair_mesh_when_previewing(
            settings.hide_stair_mesh_when_previewing
        )
        self.canvas.set_snap_middle_equal_angle_only(
            settings.snap_middle_equal_angle_only
        )
        self.generation.set_runtime_settings(settings)
        self.surface_texture_generation.set_runtime_settings(settings)
        self.merged_generation_workspace.set_clear_mask_hotkey(
            settings.clear_mask_hotkey
        )
        self._apply_scene_3d_display_screen(settings.scene_3d_display_screen_id)
        self._apply_generation_display_screen(
            settings.generation_display_screen_id
        )
        self._apply_jobs_window_screen(settings.jobs_window_screen_id)
        self._apply_atlas_display_screen(settings.atlas_display_screen_id)
        self._refresh_scene_atlas_texture_requirements()

    # ### Canvas wall drawing updates ###
    def _handle_canvas_surface_geometry_changed(self) -> None:
        """Delay active wall edits and reconcile other Canvas geometry."""

        session = self._plan_wall_preview_session
        if (
            session is not None
            and session.level_index == self.current_level.index
            and _build_vertex_data_signature(self.current_level.vertex_data)
            != session.topology_signature
        ):
            self._clear_plan_wall_preview()
        self._sync_canvas_wall_mirror_state()
        if self._is_canvas_wall_vertex_interaction_active:
            self._pending_wall_vertex_mesh_update = True
            self._restart_pending_wall_vertex_update_if_idle()
            return
        if self._pending_wall_vertex_mesh_update:
            self._restart_pending_wall_vertex_update_if_idle()
            return
        assignments_changed = (
            self._reconcile_surface_assignments_with_scene()
        )
        self._finalize_blueprint_surface_binding_undo_state()
        if not assignments_changed:
            self._schedule_viewer_preview_refresh()

    def _handle_canvas_wall_vertex_added(self) -> None:
        """Debounce mesh work while new Canvas wall vertices are added."""

        self._pending_wall_vertex_mesh_update = True
        self._restart_pending_wall_vertex_update_if_idle()

    def _handle_canvas_wall_vertex_interaction_changed(
        self,
        active: bool,
    ) -> None:
        """Pause the wall rebuild countdown while a vertex is held."""

        self._is_canvas_wall_vertex_interaction_active = bool(active)
        self._restart_pending_wall_vertex_update_if_idle()

    def _restart_pending_wall_vertex_update_if_idle(self) -> None:
        """Run the wall debounce only after the pointer interaction ends."""

        if (
            not self._pending_wall_vertex_mesh_update
            or self._is_canvas_wall_vertex_interaction_active
        ):
            self._wall_vertex_update_timer.stop()
            return
        self._wall_vertex_update_timer.start()

    def _commit_pending_wall_vertex_update(self) -> None:
        """Reconcile and rebuild once after a wall-vertex editing burst."""

        self._wall_vertex_update_timer.stop()
        if not self._pending_wall_vertex_mesh_update:
            return
        self._pending_wall_vertex_mesh_update = False
        doorway_level_indices = self._pending_wall_vertex_doorway_level_indices
        self._pending_wall_vertex_doorway_level_indices = set()
        for level_index in doorway_level_indices:
            level = self._get_level_by_index(level_index)
            if level is not None:
                self._viewer_doorways_by_level_index[level_index] = (
                    self._copy_doorways(level.doorways)
                )
        self._reconcile_canvas_surface_edit_and_refresh()

    def _cancel_pending_wall_vertex_update(self) -> None:
        """Discard a wall-vertex timer when its project context is retired."""

        self._wall_vertex_update_timer.stop()
        self._pending_wall_vertex_mesh_update = False
        self._pending_wall_vertex_doorway_level_indices.clear()
        self._is_canvas_wall_vertex_interaction_active = False

    def _handle_include_toggled(self, checked: bool) -> None:
        if self._is_syncing_level_controls or not checked:
            return

        next_value = self.include_yes_radio.isChecked()
        self._finish_level_transform_drag()
        self._commit_pending_level_transform_update()
        if next_value == self.current_level.include_in_export:
            return
        self._record_canvas_undo_state(
            self._capture_canvas_level_properties_undo_state(self.current_level)
        )
        self.current_level.include_in_export = next_value
        self._refresh_scene_atlas_texture_requirements()

    def _apply_loaded_project(self, project_data: ProjectData) -> None:
        self._apply_project_state(
            levels=project_data.levels,
            current_level_index=project_data.current_level_index,
            image_library_paths=project_data.image_library_paths,
            doorway_presets=project_data.doorway_presets,
            generation=project_data.generation,
            surface_texture_generation=(project_data.surface_texture_generation),
            texture_atlases=project_data.texture_atlases,
            stairs=project_data.stairs,
            wall_mirror_links=project_data.wall_mirror_links,
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
        wall_mirror_links: Sequence[WallMirrorVertexLink] | None = None,
    ) -> None:
        if (
            self.generation.is_generating
            or self.surface_texture_generation.is_generating
        ):
            raise RuntimeError(
                "Wait for the current generation request to finish before "
                "loading another project."
            )

        # Manual loading and startup restoration both reach this shared state
        # boundary. Finish retiring workers while the old scene and Atlas data
        # are still intact, so a queued completion cannot commit into the
        # incoming project.
        self._cancel_and_join_plan_image_corrections()
        self._cancel_and_join_plan_wall_detections()
        self._cancel_and_join_atlas_draw_call_estimates()
        self._cancel_and_join_surface_ambient_occlusion_previews()
        self._cancel_and_join_surface_ambient_occlusion_bakes()
        self._cancel_and_join_surface_texture_tiling_preparations()
        self._pending_generation_placement_anchor = None
        self._cancel_direct_object_placement()

        self._is_doorway_move_drag_active = False
        self._is_doorway_resize_drag_active = False
        self._level_transform_drag_active = False
        self._cancel_pending_level_transform(
            sync_controls=False,
            restore_canvas_tools=False,
        )
        self._canvas_transform_drag_active = False
        self._canvas_transform_drag_undo_state = None
        self.canvas.clear_level_comparison_overlay()
        self._cancel_active_canvas_surface_edit()
        self._cancel_pending_canvas_surface_mesh_update()
        self._cancel_pending_wall_vertex_update()
        self._cancel_pending_doorway_mesh_update(clear_outline=True)
        self._stair_point_mesh_update_timer.stop()
        self._pending_stair_point_mesh_update = False
        self._pending_stair_point_undo_state = None
        self._pending_stair_point_id = None
        self._stair_preview_update_timer.stop()
        self._pending_stair_parameters = None
        self.canvas.cancel_open_space_placement()
        self.canvas.cancel_stair_placement()
        self._desired_canvas_object_id = None
        self._desired_canvas_object_ids = ()
        self._desired_canvas_surface_ids = ()
        self._active_canvas_surface_drawing_vertex_id = None
        self._atlas_surface_assignment_target_ids = ()
        self._selected_atlas_surface_source_id = None
        self._set_atlas_canvas_surface_highlights(())
        self.texture_atlas_workspace.set_green_outline_source_ids(())
        self._canvas_window_undo_ids.clear()
        self._clear_canvas_undo_history()
        self.viewer.set_window_undo_available(False)
        self.levels = levels
        wall_mirror_result = reconcile_wall_mirror_topology(
            self.levels,
            wall_mirror_links or (),
        )
        self.wall_mirror_links = wall_mirror_result.links
        self._reset_viewer_doorway_snapshots()
        self._level_blueprint_image_revisions.clear()
        self.stairs = list(stairs or [])
        self._editing_stair_index = None
        self._staged_stair = None
        self._desired_canvas_stair_part_ids = ()
        self._canvas_stair_part_targets_by_id = {}
        self._canvas_stair_semantic_surfaces_by_id = {}
        self.surface_texture_generation.set_external_semantic_surfaces(())
        self.viewer.clear_canvas_stair_preview()
        self._set_stair_editor_parameters(self._new_stair_parameters)
        self._sync_stair_calculated_values(None)
        self.image_library_paths = self._normalize_image_library_paths(
            image_library_paths or []
        )
        if doorway_presets is not None:
            self.doorway_presets = list(doorway_presets)
            if not self.doorway_presets:
                self.doorway_presets.append(create_fallback_doorway_preset())
        self.current_level_index = min(
            max(current_level_index, 0),
            len(self.levels) - 1,
        )
        self._refresh_doorway_preset_list(
            selected_index=0 if self.doorway_presets else -1
        )
        self._refresh_levels_list()
        self._update_stair_button_state()
        self.stair_status_label.setText(
            "Stairs: none" if not self.stairs else f"Stairs: {len(self.stairs)} loaded."
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
        generation, surface_texture_generation = (
            self.merged_generation_workspace.merge_project_video_state(
                generation,
                surface_texture_generation,
            )
        )
        self.generation.set_data(generation)
        self._sync_viewer_scene_levels(reset_visibility=True)
        self.texture_atlas_workspace.set_data(texture_atlases)
        self.surface_texture_generation.set_levels(self.levels)
        self._sync_canvas_stair_semantic_targets(self.levels)
        self.surface_texture_generation.set_data(surface_texture_generation)
        restored_surface_ids = tuple(
            surface_texture_generation.selected_surface_ids
        )
        self._desired_canvas_stair_part_ids = tuple(
            surface_id
            for surface_id in restored_surface_ids
            if surface_id in self._canvas_stair_part_targets_by_id
        )
        self._desired_canvas_surface_ids = tuple(
            surface_id
            for surface_id in restored_surface_ids
            if surface_id not in self._canvas_stair_part_targets_by_id
        )
        self._atlas_surface_assignment_target_ids = restored_surface_ids
        selected_stair_target = (
            self._canvas_stair_part_targets_by_id.get(
                self._desired_canvas_stair_part_ids[-1]
            )
            if self._desired_canvas_stair_part_ids
            else None
        )
        if (
            selected_stair_target is not None
            and 0 <= selected_stair_target.stair_index < len(self.stairs)
        ):
            self._editing_stair_index = selected_stair_target.stair_index
            self._load_stair_editor_from_stair(
                self.stairs[self._editing_stair_index]
            )
        self._update_stair_button_state()
        self.merged_generation_workspace.sync_shared_controls()
        self._reconcile_surface_assignments_with_scene(
            emit_signals=False,
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
        if self.texture_atlas_workspace.is_ambient_occlusion_preview_active:
            self._refresh_surface_ambient_occlusion_preview()

    def _set_current_level_image(
        self,
        file_path: str,
        *,
        original_image_path: str | None = None,
    ) -> None:
        active_correction = self._plan_image_correction_runtimes.get(
            self.current_level.index
        )
        if active_correction is not None:
            self.job_manager.cancel_job(active_correction.job_id)
        active_wall_detection = self._plan_wall_detection_runtimes.get(
            self.current_level.index
        )
        if active_wall_detection is not None:
            self.job_manager.cancel_job(active_wall_detection.job_id)
        self._clear_plan_wall_preview(update_controls=False)
        self._finish_level_transform_drag()
        self._commit_pending_level_transform_update()
        self._cancel_active_canvas_surface_edit()
        self._commit_pending_canvas_surface_mesh_update()
        self._commit_pending_wall_vertex_update()
        self._commit_pending_doorway_mesh_update()
        normalized_path = str(Path(file_path).resolve())
        self.canvas.load_blueprint(
            file_path=normalized_path,
            vertex_data=self.current_level.vertex_data,
            rooms=self.current_level.rooms,
            doorways=self.current_level.doorways,
            windows=self.current_level.windows,
            open_spaces=self.current_level.open_spaces,
            canvas_level_scale=self.current_level.canvas_level_scale,
            canvas_offset_x_pixels=(self.current_level.canvas_offset_x_pixels),
            canvas_offset_y_pixels=(self.current_level.canvas_offset_y_pixels),
        )
        self._clear_canvas_undo_history()
        self.current_level.image_path = normalized_path
        self.current_level.original_image_path = str(
            Path(original_image_path or normalized_path).resolve()
        )
        self.current_level.image_size_pixels = self.canvas.get_image_size_pixels()
        self.canvas.set_stair_context(self.stairs, self.current_level)
        self._sync_canvas_wall_mirror_state()
        self.workspace_tabs.setCurrentWidget(self.canvas_viewer_workspace)
        self._update_blueprint_name_label()
        self._update_open_space_controls()
        self._schedule_viewer_preview_refresh()
        self._update_plan_wall_generation_controls_state()

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
            open_spaces=self.current_level.open_spaces,
            image_path=self.current_level.image_path,
            canvas_level_scale=self.current_level.canvas_level_scale,
            canvas_offset_x_pixels=(self.current_level.canvas_offset_x_pixels),
            canvas_offset_y_pixels=(self.current_level.canvas_offset_y_pixels),
        )
        if self.canvas.blueprint_image is not None:
            self.current_level.image_size_pixels = self.canvas.get_image_size_pixels()
        self.canvas.set_stair_context(self.stairs, self.current_level)
        self._sync_canvas_wall_mirror_state()
        self._update_wall_mirror_button_state()
        self._sync_selected_canvas_wall_highlight(
            self.viewer.get_active_canvas_surface_id()
        )
        self._update_blueprint_name_label()
        self._update_open_space_controls()
        self._update_plan_wall_generation_controls_state()

    def _update_blueprint_name_label(self) -> None:
        image_path = self.current_level.image_path
        if image_path is None:
            label_text = "Image: none for this level"
        elif self.canvas.blueprint_image is None:
            label_text = f"Image missing: {image_path}"
        else:
            label_text = f"Image: {Path(image_path).name}"
            original_path = self.current_level.original_image_path
            if original_path and Path(original_path) != Path(image_path):
                label_text += f" (corrected from {Path(original_path).name})"

        self.blueprint_name_label.setText(label_text)
        self._update_image_correction_button_state()


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


# ### Palette helpers ###
def _build_ground_level_background_color(palette: QPalette) -> QColor:
    """Return a subtle Ground tint that stays visible on light and dark themes."""

    base = palette.color(QPalette.ColorRole.Base)
    target = (
        QColor(255, 255, 255)
        if base.lightnessF() < 0.75
        else palette.color(QPalette.ColorRole.Highlight)
    )
    blend = GROUND_LEVEL_BACKGROUND_BLEND
    color = QColor(
        round(base.red() * (1.0 - blend) + target.red() * blend),
        round(base.green() * (1.0 - blend) + target.green() * blend),
        round(base.blue() * (1.0 - blend) + target.blue() * blend),
    )
    if color.rgb() == base.rgb():
        fallback = palette.color(QPalette.ColorRole.Text)
        color = QColor(
            round(base.red() * 0.94 + fallback.red() * 0.06),
            round(base.green() * 0.94 + fallback.green() * 0.06),
            round(base.blue() * 0.94 + fallback.blue() * 0.06),
        )
    return color


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
        raise ValueError("Stair intermediate sections must be a sequence.") from error


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


def _build_stair_data_from_placement(
    placement: object,
    parameters: _StairEditorParameters | None = None,
) -> StairData:
    """Convert a complete Canvas draft into the persistent stair model."""

    stair = StairData(
        start_level_index=int(
            _get_stair_placement_value(placement, "start_level_index")
        ),
        end_level_index=int(_get_stair_placement_value(placement, "end_level_index")),
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
    if parameters is None:
        return stair
    return BlueprintWorkspace._apply_stair_editor_parameters(stair, parameters)


def _format_doorway_preset_label(doorway_preset: DoorwayPreset) -> str:
    dimension_text = (
        f"{doorway_preset.width_meters:.2f} m × {doorway_preset.height_meters:.2f} m"
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


def _build_vertex_data_signature(vertex_data: VertexData) -> tuple[object, ...]:
    """Return stable topology and coordinate evidence for stale-preview guards."""

    vertices = tuple(
        sorted(
            (
                int(vertex.id),
                float(vertex.x),
                float(vertex.y),
            )
            for vertex in vertex_data.vertices
        )
    )
    edges = tuple(
        sorted(
            tuple(
                sorted(
                    (
                        int(edge.start_vertex_id),
                        int(edge.end_vertex_id),
                    )
                )
            )
            for edge in vertex_data.edges
        )
    )
    return vertices, edges


def _get_vertex_data_edge_keys(
    vertex_data: VertexData,
) -> frozenset[tuple[int, int]]:
    """Return direction-independent keys for every existing wall edge."""

    return frozenset(
        tuple(sorted((int(edge.start_vertex_id), int(edge.end_vertex_id))))
        for edge in vertex_data.edges
    )


def _validate_plan_wall_detection_result(
    session: _PlanWallPreviewSession,
    result: object,
) -> int:
    """Validate that a detector candidate can only extend its baseline graph."""

    if not isinstance(result, PlanWallDetectionResult):
        raise TypeError("The wall detector returned an invalid result.")
    candidate = result.vertex_data
    if not isinstance(candidate, VertexData):
        raise TypeError("The wall detector returned an invalid wall graph.")

    baseline = session.baseline_vertex_data
    baseline_vertex_count = len(baseline.vertices)
    baseline_edge_count = len(baseline.edges)
    if candidate.vertices[:baseline_vertex_count] != baseline.vertices:
        raise ValueError("Generated walls cannot replace existing vertices.")
    if candidate.edges[:baseline_edge_count] != baseline.edges:
        raise ValueError("Generated walls cannot replace existing wall edges.")

    raw_added_edge_count = result.added_edge_count
    if isinstance(raw_added_edge_count, bool) or not isinstance(
        raw_added_edge_count,
        int,
    ):
        raise TypeError("The generated wall count must be an integer.")
    expected_added_edge_count = len(candidate.edges) - baseline_edge_count
    if (
        expected_added_edge_count < 0
        or raw_added_edge_count != expected_added_edge_count
    ):
        raise ValueError("The generated wall count does not match the wall graph.")
    return expected_added_edge_count


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
