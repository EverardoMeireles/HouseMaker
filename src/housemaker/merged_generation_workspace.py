# ### Imports ###
from __future__ import annotations

import copy
import hashlib
import inspect
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QBuffer,
    QByteArray,
    QIODevice,
    QMimeData,
    QSignalBlocker,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QImage, QImageReader, QKeySequence, QShortcut, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.ceiling_height_dialog import CeilingHeightAnalysisDialog
from housemaker.ceiling_height_estimation import (
    CEILING_HEIGHT_JOB_KIND,
    CEILING_HEIGHT_MODEL_OPTIONS,
    CeilingHeightAnalysisResult,
    CeilingHeightEstimate,
    CeilingHeightEstimationCancelled,
    CeilingHeightEstimationError,
    CeilingHeightEstimator,
    aggregate_marked_level_ceiling_height_estimates,
    build_ceiling_height_plan_image_data_uri,
    ceiling_height_model_requires_api_key,
    estimate_ceiling_height_api_cost_usd,
    format_ceiling_height_api_cost,
    open_ceiling_height_estimator,
)
from housemaker.ceiling_height_ranges import (
    CeilingHeightVideoRange,
    sample_marked_video_frames,
)
from housemaker.door_state import is_door_body_slot
from housemaker.generation_jobs import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_RUNNING,
    GenerationJobManager,
)
from housemaker.generation_state import (
    MAX_MASK_STROKES_PER_FRAME,
    GenerationData,
    MaskStroke,
)
from housemaker.generation_workspace import (
    GenerationWorkspace,
)
from housemaker.geometry_generation_models import (
    GEOMETRY_MODEL_7_1,
    GEOMETRY_MODEL_OPTIONS,
    GEOMETRY_RESOLUTION_OPTIONS,
    maximum_geometry_target_polycount,
)
from housemaker.models import LevelData
from housemaker.openai_batch import LUNA_BATCH_MODEL
from housemaker.plan_correction_models import PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1
from housemaker.plan_metric_state import (
    PlanMetricCalibration,
    build_ceiling_height_plan_context,
)
from housemaker.reference_generation import (
    MAX_REFERENCE_PROMPT_CHARACTERS,
    REMOVE_OBJECTS_FROM_FURNITURE_PROMPT,
)
from housemaker.reference_generation_batch import (
    APIReferenceBatchThread as _APIReferenceBatchThread,
)
from housemaker.reference_generation_batch import (
    LocalReferenceBatchItem,
)
from housemaker.reference_generation_batch import (
    LocalReferenceBatchThread as _LocalReferenceBatchThread,
)
from housemaker.reference_generation_batch import (
    OpenAIReferenceBatchThread as _OpenAIReferenceBatchThread,
)
from housemaker.reference_generation_models import (
    DEFAULT_REFERENCE_NEGATIVE_PROMPT,
    REFERENCE_GENERATION_MODEL_MESHY,
    REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
    REFERENCE_GENERATION_MODELS,
)
from housemaker.reference_symmetry_inference import (
    SYMMETRY_INFERENCE_MODEL_OPTIONS,
)
from housemaker.reference_view_presets import (
    DEFAULT_REFERENCE_VIEW_PRESETS,
    ReferenceViewPreset,
    read_reference_view_presets,
)
from housemaker.settings_widget import (
    DEFAULT_CLEAR_MASK_HOTKEY,
    read_ceiling_height_estimation_model,
    read_enable_object_symmetry_inference,
    read_reference_generation_model,
    read_symmetry_inference_model,
)
from housemaker.surface_texture_state import SurfaceTextureData
from housemaker.surface_texture_workspace import SurfaceTextureGenerationWorkspace
from housemaker.video_source import (
    VIDEO_FILE_FILTER,
    VideoFrameSource,
    VideoMetadata,
    probe_video,
)

# ### Constants ###
WORKFLOW_SECTION_SPACING = 10
GENERATION_CONTROLS_PADDING = 10
GENERATION_CONTROLS_MINIMUM_HEIGHT = 96
GENERATION_ROWS_HANDLE_WIDTH = 8
CEILING_HEIGHT_SHUTDOWN_WAIT_MILLISECONDS = 100
CEILING_HEIGHT_NOT_ESTIMATED_TEXT = "Ceiling height: Not estimated"
MAX_CONSECUTIVE_CEILING_HEIGHT_FRAME_FAILURES = 3
DOOR_SLOT_GENERATION_BLINK_INTERVAL_MILLISECONDS = 500
DOOR_SLOT_GENERATION_ACTIVE_STYLE = "color: #ff3030; font-weight: 700;"
DOOR_SLOT_GENERATION_HIDDEN_STYLE = (
    "color: rgba(255, 48, 48, 0); font-weight: 700;"
)
REFERENCE_IMAGE_FILE_FILTER = (
    "Image files (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;"
    "All files (*.*)"
)
GENERATION_ACTION_GENERATE = "generate"
GENERATION_ACTION_GEOMETRY = "generate_geometry"
GENERATION_ACTION_OBJECT_TEXTURE = "generate_object_texture"
GENERATION_ACTION_SURFACE_TEXTURE = "generate_surface_texture"
REFERENCE_GENERATION_JOB_KIND = "Reference generation"
REFERENCE_GENERATION_PROMPT_SETTINGS_KEY = "generation/reference_prompt"
REMOVE_OBJECTS_FROM_FURNITURE_SETTINGS_KEY = "generation/remove_objects_from_furniture"
REFERENCE_GENERATION_VIEW_SETTINGS_KEY = "generation/reference_generation_view"
REFERENCE_GENERATION_CUSTOM_VIEW_ID = ""
REFERENCE_GENERATION_NEGATIVE_PROMPT_SETTINGS_KEY = (
    "generation/reference_generation_negative_prompt"
)
REFERENCE_GENERATION_STRENGTH_SETTINGS_KEY = "generation/reference_generation_strength"


# ### Ceiling-height plan references ###
@dataclass(frozen=True)
class _CeilingHeightPlanSource:
    """One corrected plan and its matching metric annotations."""

    level_index: int
    image_path: str
    calibration: PlanMetricCalibration


# ### Ceiling-height inference jobs ###
class _CeilingHeightInferenceThread(QThread):
    """Analyze sampled video frames with an independent background reader."""

    progress_changed = Signal(int, int, int)
    status_changed = Signal(str)

    def __init__(
        self,
        video_path: str,
        frame_samples: tuple[tuple[int, int], ...],
        api_key: str,
        model: str,
        estimator: CeilingHeightEstimator | None,
        level_options: tuple[tuple[int, str], ...],
        level_plan_contexts: tuple[tuple[int, str], ...],
        marked_ranges: tuple[CeilingHeightVideoRange, ...],
        level_plan_sources: tuple[_CeilingHeightPlanSource, ...] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._video_path = str(video_path)
        self._frame_samples = tuple(frame_samples)
        self._api_key = str(api_key)
        self._model = str(model)
        self._estimator = estimator
        self._level_options = tuple(level_options)
        self._level_plan_contexts = tuple(level_plan_contexts)
        self._level_plan_sources = tuple(level_plan_sources)
        self._marked_ranges = tuple(marked_ranges)
        self.result: CeilingHeightAnalysisResult | None = None
        self.error_message: str | None = None
        self.was_cancelled = False

    def run(self) -> None:  # type: ignore[override]
        try:
            try:
                from housemaker.openai_batch import LUNA_BATCH_MODEL

                if self._model == LUNA_BATCH_MODEL and self._estimator is None:
                    self.result = self._run_batch_analysis()
                    return
                frame_estimates: list[tuple[int, int, CeilingHeightEstimate]] = []
                previous_frame: np.ndarray | None = None
                previous_range_index: int | None = None
                consecutive_failures = 0
                plan_context_by_level = dict(self._level_plan_contexts)
                plan_image_by_level: dict[int, str] = {}
                for source in self._level_plan_sources:
                    if self.isInterruptionRequested():
                        raise CeilingHeightEstimationCancelled("Cancelled")
                    try:
                        plan_image_by_level[source.level_index] = (
                            build_ceiling_height_plan_image_data_uri(
                                source.image_path,
                                source.calibration,
                            )
                        )
                    except CeilingHeightEstimationError:
                        continue
                with open_ceiling_height_estimator(
                    model=self._model,
                    api_key=self._api_key,
                    estimator_override=self._estimator,
                    startup_cancellation_check=self.isInterruptionRequested,
                ) as estimator, VideoFrameSource(
                    self._video_path
                ) as video_source:
                    total = len(self._frame_samples)
                    for completed, (frame_index, level_index) in enumerate(
                        self._frame_samples,
                        start=1,
                    ):
                        if self.isInterruptionRequested():
                            raise CeilingHeightEstimationCancelled("Cancelled")
                        current_frame = video_source.get_frame(frame_index)
                        range_index = next(
                            index
                            for index, marked_range in enumerate(self._marked_ranges)
                            if marked_range.level_index == level_index
                            and marked_range.start_frame
                            <= frame_index
                            <= marked_range.end_frame
                        )
                        if range_index != previous_range_index:
                            previous_frame = None
                        try:
                            plan_image_data_uri = plan_image_by_level.get(level_index)
                            plan_context = (
                                plan_context_by_level.get(level_index, "")
                                if plan_image_data_uri is not None
                                else ""
                            )
                            estimate = _invoke_bound_ceiling_height_estimator(
                                estimator,
                                current_frame,
                                previous_frame=previous_frame,
                                plan_context=plan_context,
                                plan_image_data_uri=plan_image_data_uri,
                                cancellation_check=self.isInterruptionRequested,
                            )
                        except CeilingHeightEstimationCancelled:
                            raise
                        except CeilingHeightEstimationError:
                            consecutive_failures += 1
                            if (
                                consecutive_failures
                                >= MAX_CONSECUTIVE_CEILING_HEIGHT_FRAME_FAILURES
                            ):
                                raise
                            estimate = CeilingHeightEstimate(
                                status="insufficient_evidence",
                                estimate_m=None,
                                minimum_m=None,
                                maximum_m=None,
                                confidence="low",
                                basis="This sampled frame could not be analysed.",
                                level_movement="uncertain",
                                movement_confidence="low",
                            )
                        else:
                            consecutive_failures = 0
                        frame_estimates.append(
                            (
                                level_index,
                                frame_index,
                                estimate,
                            )
                        )
                        previous_frame = current_frame
                        previous_range_index = range_index
                        self.progress_changed.emit(
                            completed,
                            total,
                            frame_index,
                        )
                result = aggregate_marked_level_ceiling_height_estimates(
                    frame_estimates,
                    levels=self._level_options,
                )
            except CeilingHeightEstimationCancelled:
                self.was_cancelled = True
                return
            except CeilingHeightEstimationError as error:
                if self.isInterruptionRequested():
                    self.was_cancelled = True
                else:
                    self.error_message = str(error)
                return
            except Exception:  # noqa: BLE001 - redact unexpected worker failures.
                if self.isInterruptionRequested():
                    self.was_cancelled = True
                else:
                    self.error_message = "Ceiling height inference failed."
                return
            if self.isInterruptionRequested():
                self.was_cancelled = True
                return
            self.result = result
        finally:
            self._api_key = ""
            self._video_path = ""

    def _run_batch_analysis(self) -> CeilingHeightAnalysisResult:
        """Queue every explicit frame pair together, not one 24-hour job per pair."""

        from housemaker.ceiling_height_estimation import (
            build_ceiling_height_batch_request,
            infer_ceiling_height_batch,
        )

        contexts = dict(self._level_plan_contexts)
        plan_images: dict[int, str] = {}
        for source in self._level_plan_sources:
            if self.isInterruptionRequested():
                raise CeilingHeightEstimationCancelled("Cancelled")
            try:
                plan_images[source.level_index] = build_ceiling_height_plan_image_data_uri(
                    source.image_path, source.calibration
                )
            except CeilingHeightEstimationError:
                continue
        requests = {}
        previous_frame: np.ndarray | None = None
        previous_range: int | None = None
        total = len(self._frame_samples)
        with VideoFrameSource(self._video_path) as source:
            for prepared, (frame_index, level_index) in enumerate(self._frame_samples, 1):
                if self.isInterruptionRequested():
                    raise CeilingHeightEstimationCancelled("Cancelled")
                current_frame = source.get_frame(frame_index)
                range_index = next(
                    index for index, interval in enumerate(self._marked_ranges)
                    if interval.level_index == level_index
                    and interval.start_frame <= frame_index <= interval.end_frame
                )
                if range_index != previous_range:
                    previous_frame = None
                plan_image = plan_images.get(level_index)
                requests[str(prepared)] = build_ceiling_height_batch_request(
                    current_frame,
                    previous_frame_bgr=previous_frame,
                    plan_context=contexts.get(level_index, "") if plan_image else "",
                    plan_image_data_uri=plan_image,
                )
                previous_frame = current_frame
                previous_range = range_index
                self.status_changed.emit(
                    f"Preparing GPT-5.6 Luna Batch API frame intervals {prepared} of {total}"
                )
        results = infer_ceiling_height_batch(
            requests,
            api_key=self._api_key,
            cancellation_check=self.isInterruptionRequested,
            progress_callback=self.status_changed.emit,
        )
        frame_estimates = []
        consecutive_failures = 0
        for completed, (frame_index, level_index) in enumerate(self._frame_samples, 1):
            if self.isInterruptionRequested():
                raise CeilingHeightEstimationCancelled("Cancelled")
            estimate = results.get(str(completed))
            if isinstance(estimate, CeilingHeightEstimationError):
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_CEILING_HEIGHT_FRAME_FAILURES:
                    raise estimate
                estimate = CeilingHeightEstimate(
                    status="insufficient_evidence", estimate_m=None,
                    minimum_m=None, maximum_m=None, confidence="low",
                    basis=f"This sampled frame could not be analysed: {estimate}",
                    level_movement="uncertain", movement_confidence="low",
                )
            elif estimate is None:
                raise CeilingHeightEstimationError(
                    "GPT-5.6 Luna Batch API omitted a sampled frame interval."
                )
            else:
                consecutive_failures = 0
            frame_estimates.append((level_index, frame_index, estimate))
            self.progress_changed.emit(completed, total, frame_index)
        return aggregate_marked_level_ceiling_height_estimates(
            frame_estimates, levels=self._level_options
        )


@dataclass
class _CeilingHeightInferenceRuntime:
    """GUI-owned lifecycle for one source-bound video analysis."""

    video_path: str
    frame_indices: tuple[int, ...]
    level_signature: tuple[tuple[int, str], ...]
    plan_context_signature: tuple[tuple[int, str], ...]
    plan_source_signature: tuple[_CeilingHeightPlanSource, ...]
    marked_ranges: tuple[CeilingHeightVideoRange, ...]
    context_revision: int
    job_id: str
    thread: _CeilingHeightInferenceThread
    cancel_requested: bool = False


# ### Ceiling-height plan-context helpers ###
def _invoke_bound_ceiling_height_estimator(
    estimator: object,
    frame_bgr: np.ndarray,
    *,
    previous_frame: np.ndarray | None,
    plan_context: str,
    plan_image_data_uri: str | None,
    cancellation_check: Callable[[], bool],
) -> CeilingHeightEstimate:
    """Pass metric context when supported by an injected estimator."""

    supports_context = True
    supports_plan_image = True
    try:
        parameters = inspect.signature(estimator).parameters
        accepts_keywords = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        supports_context = bool(
            "plan_context" in parameters
            or accepts_keywords
        )
        supports_plan_image = bool(
            "plan_image_data_uri" in parameters or accepts_keywords
        )
    except (TypeError, ValueError):
        pass
    arguments = {
        "previous_frame_bgr": previous_frame,
        "cancellation_check": cancellation_check,
    }
    if supports_context:
        arguments["plan_context"] = plan_context
    if supports_plan_image:
        arguments["plan_image_data_uri"] = plan_image_data_uri
    return estimator(frame_bgr, **arguments)  # type: ignore[operator]


def _largest_ceiling_height_plan_reference_size(
    sources: Iterable[_CeilingHeightPlanSource],
) -> tuple[int, int] | None:
    """Return a conservative plan-image size for paid cost estimation."""

    return max(
        (source.calibration.normalized_image_size for source in sources),
        key=lambda size: size[0] * size[1],
        default=None,
    )


# ### Ceiling-height presentation helpers ###
def _format_cost_confirmation_suffix(cost_usd: float | None) -> str:
    formatted = format_ceiling_height_api_cost(cost_usd)
    return f" ({formatted})" if formatted else ""


def _format_unassigned_frame_suffix(
    result: CeilingHeightAnalysisResult,
) -> str:
    count = len(result.unassigned_frame_indices)
    if count == 0:
        return ""
    return (
        f"; {count} transition or uncertain frame"
        f"{'s were' if count != 1 else ' was'} not assigned to a level"
    )


def _format_level_height_result(
    result: CeilingHeightAnalysisResult,
) -> str:
    lines = ["Detected ceiling heights:"]
    for level_estimate in result.level_estimates:
        analysis = level_estimate.analysis
        if analysis.average_height_m is None:
            lines.append(
                f"{level_estimate.level_name}: no reliable height estimate"
            )
            continue
        line = (
            f"{level_estimate.level_name}: {analysis.average_height_m:.2f} m "
            f"from {analysis.retained_count} consistent estimate"
            f"{'s' if analysis.retained_count != 1 else ''}"
        )
        if analysis.outlier_count:
            line += f"; {analysis.outlier_count} outlier(s) discarded"
        lines.append(line)
    unassigned_suffix = _format_unassigned_frame_suffix(result)
    if unassigned_suffix:
        lines.append(unassigned_suffix.removeprefix("; ").capitalize())
    return "\n".join(lines)


# ### Temporary generation references ###
class _ReferenceGenerationThread(QThread):
    """Keep image-model imports and expensive inference off the GUI thread."""

    progress_changed = Signal(str, int)

    def __init__(
        self,
        source_bgra: np.ndarray,
        prompt: str,
        model: str,
        *,
        api_key: str,
        strength: float,
        negative_prompt: str = DEFAULT_REFERENCE_NEGATIVE_PROMPT,
        symmetry_api_key: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._source_bgra = np.ascontiguousarray(source_bgra).copy()
        self._prompt = str(prompt)
        self._model = str(model)
        self._api_key = str(api_key)
        self._symmetry_api_key = (
            self._api_key if symmetry_api_key is None else str(symmetry_api_key)
        )
        self._strength = float(strength)
        self._negative_prompt = str(negative_prompt)
        self.result: np.ndarray | None = None
        self.error_message: str | None = None
        self.symmetry_model: str | None = None
        self.symmetry_result = None
        self.symmetry_error: str | None = None
        self.was_cancelled = False

    def run(self) -> None:  # type: ignore[override]
        try:
            from housemaker.reference_generation import generate_object_reference

            self.result = generate_object_reference(
                self._source_bgra,
                self._prompt,
                self._model,
                api_key=self._api_key,
                strength=self._strength,
                negative_prompt=self._negative_prompt,
                progress_callback=self.progress_changed.emit,
                cancellation_check=self.isInterruptionRequested,
            )
            self.was_cancelled = self.isInterruptionRequested()
            if not self.was_cancelled:
                self._infer_generated_symmetry()
        except Exception as error:  # noqa: BLE001 - report provider failures on the GUI thread.
            self.was_cancelled = self.isInterruptionRequested()
            if not self.was_cancelled:
                self.error_message = self._analysis_error(error)
        finally:
            self._api_key = ""
            self._symmetry_api_key = ""

    # ### Post-generation symmetry analysis ###
    def _analysis_error(self, error) -> str:
        message = str(error)
        for secret in sorted({self._api_key, self._symmetry_api_key}, key=len, reverse=True):
            if secret:
                message = message.replace(secret, "[redacted]")
        return message

    def _infer_generated_symmetry(self) -> None:
        if self.symmetry_model is None or self.was_cancelled:
            return
        from housemaker.reference_symmetry_inference import (
            SymmetryInferenceCancelled,
            infer_reference_symmetry,
        )

        self.progress_changed.emit("Inferring symmetric division from the generated reference...", 95)
        try:
            arguments = {
                "cancellation_check": self.isInterruptionRequested,
                "progress_callback": lambda message: self.progress_changed.emit(message, 95),
            }
            self.symmetry_result = infer_reference_symmetry(
                self.result, self.symmetry_model, api_key=self._symmetry_api_key, **arguments,
            )
        except SymmetryInferenceCancelled:
            self.was_cancelled = True
        except Exception as error:  # noqa: BLE001 - Keep the usable image on analysis failure.
            if self.isInterruptionRequested():
                self.was_cancelled = True
            else:
                self.symmetry_error = self._analysis_error(error)
        self.was_cancelled = self.was_cancelled or self.isInterruptionRequested()


@dataclass
class _ReferenceGenerationRuntime:
    """Tie one cancellable request to the exact input it was launched from."""

    thread: _ReferenceGenerationThread
    job_id: str
    context_signature: tuple[object, ...]
    source_bgra: np.ndarray
    reference_view: tuple[str, str] | None = None
    cancel_requested: bool = False


@dataclass
class _ReferenceGenerationRetrySource:
    """Retry from the original image, never from the previous model output."""

    source_bgra: np.ndarray
    context_signature: tuple[object, ...]


@dataclass
class _BatchReferenceGenerationRuntime:
    """One queued reference job, owned by a blob rather than the current frame."""

    thread: _ReferenceGenerationThread | _LocalReferenceBatchThread
    job_id: str
    inpaint_key: tuple[int, str]
    generation_data: object
    video_source: object
    source_bgra: np.ndarray
    input_signature: tuple[object, ...]
    started: bool = False
    reference_applied: bool = False
    reference_view: tuple[str, str] | None = None
    cancel_requested: bool = False


@dataclass
class _BatchReferenceRetrySource:
    """Reuse the original input only while its accepted output is unchanged."""

    generation_data: object
    video_source: object
    source_bgra: np.ndarray
    output_signature: tuple[object, ...]


def _reference_image_signature(image: np.ndarray) -> tuple[object, ...]:
    return image.shape, hashlib.blake2b(
        np.ascontiguousarray(image).tobytes(), digest_size=16,
    ).digest()


@dataclass
class _TemporaryObjectReferenceSession:
    """Keep one external reference bound to its selected inpaint object."""

    active_bgra: np.ndarray
    source_label: str
    generation_data_identity: int
    video_source: object | None
    frame_index: int | None
    inpaint_key: tuple[object, ...] | None = None


# ### Merged generation workspace ###
class MergedGenerationWorkspace(QWidget):
    """Present Surface and Object generation around shared reference media."""

    current_video_frame_changed = Signal()
    ceiling_height_estimate_changed = Signal(object)

    def __init__(
        self,
        surface_workspace: SurfaceTextureGenerationWorkspace,
        object_workspace: GenerationWorkspace,
        job_manager: GenerationJobManager,
        parent: QWidget | None = None,
        *,
        application_settings: ApplicationSettingsStore | None = None,
        ceiling_height_estimator: CeilingHeightEstimator | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("merged_generation_workspace")
        self.surface_workspace = surface_workspace
        self.object_workspace = object_workspace
        self.job_manager = job_manager
        self._application_settings = application_settings
        self._ceiling_height_estimator = ceiling_height_estimator
        self._ceiling_height_runtime: _CeilingHeightInferenceRuntime | None = None
        self._ceiling_height_dialog: CeilingHeightAnalysisDialog | None = None
        self._ceiling_height_video_path = ""
        self._ceiling_height_context_revision = 0
        self._ceiling_height_level_images: tuple[tuple[int, str], ...] = ()
        self._ceiling_height_level_options: tuple[tuple[int, str], ...] = ()
        self._ceiling_height_level_plan_contexts: tuple[tuple[int, str], ...] = ()
        self._ceiling_height_level_plan_sources: tuple[
            _CeilingHeightPlanSource, ...
        ] = ()
        self._ceiling_height_current_level_index: int | None = None
        self._temporary_object_reference_session: (
            _TemporaryObjectReferenceSession | None
        ) = None
        self._door_slot_generation_name = ""
        self._door_slot_generation_blink_visible = True
        self._generation_action_job_ids: dict[str, tuple[str, ...]] = {}
        self._generation_action_buttons: dict[str, QPushButton] = {}
        self._generation_action_default_text: dict[str, str] = {}
        self._generation_action_default_tooltips: dict[str, str] = {}
        self._is_shutdown = False
        self._reference_generation_runtime: (
            _ReferenceGenerationRuntime | None
        ) = None
        self._retired_reference_generation_runtimes: dict[
            str, _ReferenceGenerationRuntime | _BatchReferenceGenerationRuntime
        ] = {}
        self._batch_reference_generation_runtimes: dict[str, _BatchReferenceGenerationRuntime] = {}
        self._local_reference_batch_thread: _LocalReferenceBatchThread | None = None
        self._batch_reference_generation_job_ids: tuple[str, ...] = ()
        self._batch_reference_retry_sources: dict[tuple[int, str], _BatchReferenceRetrySource] = {}
        self._reference_batch_scope: tuple[object, object] | None = None
        self._cancelling_all_reference_jobs = False
        self._reference_generation_retry_source: (
            _ReferenceGenerationRetrySource | None
        ) = None
        self._reference_generation_input_revision = 0
        self._standalone_reference_tried_views: set[tuple[str, str]] = set()
        self._reference_generation_model = read_reference_generation_model(
            application_settings
        )
        self._enable_object_symmetry_inference = read_enable_object_symmetry_inference(application_settings)
        self._symmetry_inference_model = read_symmetry_inference_model(application_settings)
        self._ceiling_height_estimation_model = read_ceiling_height_estimation_model(
            application_settings
        )
        self._reference_view_presets = read_reference_view_presets(application_settings)
        default_view = self._reference_view_presets[0].preset_id if self._reference_view_presets else ""
        self._default_inpaint_reference_view_preferences = (
            self._read_initial_reference_preferences(default_view)
        )
        self._default_inpaint_remove_objects_from_furniture = (
            self._reference_generation_preference(
                REMOVE_OBJECTS_FROM_FURNITURE_SETTINGS_KEY, False,
            ) is True
        )
        self._displayed_reference_view_scope = None
        self._reference_preferences_dirty = False
        self._reference_generate_can_start = False
        self._reference_generate_all_can_start = False

        self.surface_workspace.setParent(self)
        self.object_workspace.setParent(self)
        self.surface_workspace.hide()
        self.object_workspace.hide()
        self.object_workspace.set_shared_control_state_managed_externally(True)
        self._adopt_shared_controls()
        self._build_ui()
        self._door_slot_generation_blink_timer = QTimer(self)
        self._door_slot_generation_blink_timer.setInterval(
            DOOR_SLOT_GENERATION_BLINK_INTERVAL_MILLISECONDS
        )
        self._door_slot_generation_blink_timer.timeout.connect(
            self._toggle_door_slot_generation_blink
        )
        self._connect_shared_controls()
        self.clear_mask_shortcut = QShortcut(self)
        self.clear_mask_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.clear_mask_shortcut.activated.connect(self._clear_mask_from_shortcut)
        self.set_clear_mask_hotkey(DEFAULT_CLEAR_MASK_HOTKEY)
        self.sync_shared_controls()

    def shutdown(self) -> None:
        """Cancel and join Generation-tab analysis workers exactly once."""

        if self._is_shutdown:
            return
        if self._reference_preferences_dirty:
            try:
                self.flush_reference_generation_preferences()
            except OSError:
                # A preference-store failure must not prevent worker cleanup.
                pass
        self._is_shutdown = True
        self._invalidate_reference_generation_input()
        self._cancel_all_reference_generations()
        local_batch = self._local_reference_batch_thread
        if local_batch is not None:
            local_batch.requestInterruption()
            while local_batch.isRunning():
                local_batch.wait(CEILING_HEIGHT_SHUTDOWN_WAIT_MILLISECONDS)
            local_batch.deleteLater()
            self._local_reference_batch_thread = None
        for retired_runtime in tuple(
            self._retired_reference_generation_runtimes.values()
        ):
            retired_runtime.thread.requestInterruption()
            while retired_runtime.thread.isRunning():
                retired_runtime.thread.wait(
                    CEILING_HEIGHT_SHUTDOWN_WAIT_MILLISECONDS
                )
            retired_runtime.thread.deleteLater()
        self._retired_reference_generation_runtimes.clear()
        self._door_slot_generation_blink_timer.stop()
        try:
            QApplication.clipboard().dataChanged.disconnect(
                self.sync_shared_controls
            )
        except (RuntimeError, TypeError):
            pass
        runtime = self._ceiling_height_runtime
        if runtime is not None:
            runtime.cancel_requested = True
            runtime.thread.requestInterruption()
            self.job_manager.set_cancel_callback(runtime.job_id, None)
            self.job_manager.mark_cancelled(
                runtime.job_id,
                stage="Cancelled during shutdown",
            )
            while runtime.thread.isRunning():
                runtime.thread.wait(CEILING_HEIGHT_SHUTDOWN_WAIT_MILLISECONDS)
            runtime.thread.deleteLater()
        self._ceiling_height_runtime = None
        if self._ceiling_height_dialog is not None:
            self._ceiling_height_dialog.dispose()
            self._ceiling_height_dialog = None

    def set_clear_mask_hotkey(self, hotkey: str) -> None:
        """Apply the selected Generation-only Clear mask keymapping."""

        self.clear_mask_shortcut.setKey(
            QKeySequence(hotkey, QKeySequence.SequenceFormat.PortableText)
        )
        self.clear_mask_shortcut.setEnabled(bool(hotkey))

    @Slot()
    def _clear_mask_from_shortcut(self) -> None:
        """Use the same enabled-state guard as the shared Clear mask button."""

        if self.clear_mask_button.isEnabled():
            self.clear_mask_button.click()

    def refresh_file_backed_previews(self) -> None:
        """Refresh the generated-object preview at the tab cache boundary."""

        self.object_workspace.refresh_file_backed_previews()

    def focus_navigation(self) -> None:
        """Forward navigation focus to the embedded generated-object view."""

        self.object_workspace.object_3d_panel.focus_navigation()

    def get_current_video_frame_bgr(self) -> np.ndarray | None:
        """Return a defensive copy of the frame displayed by Generation."""

        return self.video_view.get_frame_bgr()

    @staticmethod
    def merge_project_video_state(
        generation: GenerationData | None,
        surface: SurfaceTextureData | None,
    ) -> tuple[GenerationData, SurfaceTextureData]:
        """Migrate two legacy media states into one deterministic state."""

        object_data = GenerationData() if generation is None else generation.clone()
        surface_data = SurfaceTextureData() if surface is None else surface.clone()
        source = _select_richest_video_state(object_data, surface_data)
        if source is None:
            return object_data, surface_data

        metadata, frame_index, primary_frame_strokes = source
        frame_strokes = primary_frame_strokes
        if object_data.video_metadata == surface_data.video_metadata:
            frame_strokes = _merge_legacy_frame_strokes(
                primary_frame_strokes,
                object_data.frame_strokes,
                surface_data.frame_strokes,
            )
        object_data.video_metadata = copy.deepcopy(metadata)
        object_data.current_frame_index = int(frame_index)
        object_data.frame_strokes = copy.deepcopy(frame_strokes)
        surface_data.video_metadata = copy.deepcopy(metadata)
        surface_data.current_frame_index = int(frame_index)
        surface_data.frame_strokes = copy.deepcopy(frame_strokes)
        return object_data, surface_data

    def _adopt_shared_controls(self) -> None:
        """Install the Object-owned controls and migrate legacy duplicates."""

        surface = self.surface_workspace
        objects = self.object_workspace

        self.video_view = objects.video_view
        self.seekbar = objects.seekbar
        self.load_video_button = objects.load_video_button
        self.paint_mask_button = objects.paint_mask_button
        self.erase_mask_button = objects.erase_mask_button
        self.mask_mode_control = objects.mask_mode_control
        self.brush_size_spinbox = objects.brush_size_spinbox
        self.clear_mask_button = objects.clear_mask_button
        self.pbr_map_control = objects.pbr_map_control
        self.pbr_map_checkboxes = objects.pbr_map_checkboxes
        self.ai_prompt_edit = objects.ai_prompt_edit
        objects.cancel_operation_button.hide()

        if surface.video_view is self.video_view:
            return

        retired_surface_widgets = (
            surface.video_view,
            surface.seekbar,
            surface.load_video_button,
            surface.mask_mode_control,
            surface.brush_size_spinbox,
            surface.clear_mask_button,
            surface.ai_prompt_edit,
            surface.pbr_map_control,
        )
        retired_button_group = surface._mask_mode_group

        surface.video_view = self.video_view
        surface.seekbar = self.seekbar
        surface.load_video_button = self.load_video_button
        surface.paint_mask_button = self.paint_mask_button
        surface.erase_mask_button = self.erase_mask_button
        surface.mask_mode_control = self.mask_mode_control
        surface.brush_size_spinbox = self.brush_size_spinbox
        surface.clear_mask_button = self.clear_mask_button
        surface.pbr_map_control = self.pbr_map_control
        surface.pbr_map_checkboxes = self.pbr_map_checkboxes
        surface.ai_prompt_edit = self.ai_prompt_edit
        surface._mask_mode_group = objects._mask_mode_button_group
        surface._shared_controls = objects.get_shared_controls()

        for retired_widget in _unique_widgets(retired_surface_widgets):
            retired_widget.hide()
            retired_widget.setParent(None)
            retired_widget.deleteLater()
        retired_button_group.deleteLater()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        self.rows_splitter = QSplitter(Qt.Orientation.Vertical)
        self.rows_splitter.setObjectName("merged_generation_rows_splitter")
        self.rows_splitter.setChildrenCollapsible(False)
        self.rows_splitter.setHandleWidth(GENERATION_ROWS_HANDLE_WIDTH)
        self.preview_row = QWidget()
        self.preview_row.setObjectName("merged_generation_preview_row")
        preview_layout = QVBoxLayout(self.preview_row)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)

        self.views_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.views_splitter.setObjectName("merged_generation_views_splitter")
        self.views_splitter.setChildrenCollapsible(False)
        self.views_splitter.addWidget(
            _build_labeled_view(
                "Video material references and object mask",
                self.video_view,
            )
        )
        self.views_splitter.addWidget(self.object_workspace.object_3d_page)
        self.views_splitter.setStretchFactor(0, 1)
        self.views_splitter.setStretchFactor(1, 1)
        self.views_splitter.setSizes([1_000, 1_000])
        preview_layout.addWidget(self.views_splitter, 1)
        preview_layout.addWidget(self.seekbar)
        self.rows_splitter.addWidget(self.preview_row)

        self.shared_controls = self._build_shared_controls()
        self.object_controls = self._build_object_controls()
        self.controls_row = QWidget()
        self.controls_row.setObjectName("merged_generation_controls_row")
        controls_row_layout = QHBoxLayout(self.controls_row)
        controls_row_layout.setContentsMargins(
            GENERATION_CONTROLS_PADDING,
            GENERATION_CONTROLS_PADDING,
            GENERATION_CONTROLS_PADDING,
            GENERATION_CONTROLS_PADDING,
        )
        controls_row_layout.setSpacing(WORKFLOW_SECTION_SPACING)
        controls_row_layout.addWidget(self.shared_controls, 2)
        controls_row_layout.addWidget(self.object_controls, 2)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setObjectName("merged_generation_controls_scroll")
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.controls_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.controls_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.controls_scroll.setWidget(self.controls_row)
        self.controls_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.controls_scroll.setMinimumHeight(
            GENERATION_CONTROLS_MINIMUM_HEIGHT
        )
        self.rows_splitter.addWidget(self.controls_scroll)
        self.rows_splitter.setStretchFactor(0, 1)
        self.rows_splitter.setStretchFactor(1, 0)
        handle = self.rows_splitter.handle(1)
        handle.setCursor(Qt.CursorShape.SizeVerCursor)
        handle.setToolTip("Drag to resize the previews and generation controls.")
        root_layout.addWidget(self.rows_splitter, 1)
        self._initial_rows_sizing_applied = False

    # ### Resizable preview and control rows ###
    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._initial_rows_sizing_applied:
            QTimer.singleShot(0, self._initialize_row_sizes)

    def _initialize_row_sizes(self) -> None:
        """Fit the controls once; later resizes keep the user's chosen divider."""

        if self._initial_rows_sizing_applied or not self.isVisible():
            return
        available = self.rows_splitter.height() - self.rows_splitter.handleWidth()
        preview_minimum = self.preview_row.minimumSizeHint().height()
        controls_height = (
            self.controls_row.sizeHint().height()
            + self.controls_scroll.horizontalScrollBar().sizeHint().height()
        )
        controls_height = max(
            GENERATION_CONTROLS_MINIMUM_HEIGHT,
            min(controls_height, available - preview_minimum),
        )
        self.rows_splitter.setSizes([
            max(preview_minimum, available - controls_height),
            controls_height,
        ])
        self._initial_rows_sizing_applied = True

    # ### Shared generation controls ###
    def _build_shared_controls(self) -> QWidget:
        section, layout = _build_controls_section("shared")

        shared_column, shared_layout = _build_controls_column(
            "merged_generation_shared_primary_column"
        )
        reference_media_actions = QWidget()
        reference_media_actions.setObjectName(
            "merged_generation_reference_media_actions"
        )
        reference_media_layout = QHBoxLayout(reference_media_actions)
        reference_media_layout.setContentsMargins(0, 0, 0, 0)
        reference_media_layout.setSpacing(6)
        self.load_video_button.setObjectName("load_video_button")
        self.copy_inpaint_button = QPushButton("Copy inpaint")
        self.copy_inpaint_button.setObjectName("copy_inpaint_button")
        self.copy_inpaint_button.setToolTip(
            "Copy the masked object or active temporary reference with "
            "transparency."
        )
        self.paste_inpaint_button = QPushButton("Paste")
        self.paste_inpaint_button.setObjectName("paste_inpaint_button")
        self.paste_inpaint_button.setToolTip(
            "Use a clipboard image as the next Object generation reference."
        )
        self.load_reference_image_button = QPushButton("Load image")
        self.load_reference_image_button.setObjectName(
            "load_reference_image_button"
        )
        self.load_reference_image_button.setToolTip(
            "Use an image as the next Object generation reference without "
            "replacing the loaded video."
        )
        reference_media_layout.addWidget(self.load_video_button)
        reference_media_layout.addWidget(self.copy_inpaint_button)
        reference_media_layout.addWidget(self.paste_inpaint_button)
        reference_media_layout.addWidget(self.load_reference_image_button)
        shared_layout.addWidget(reference_media_actions)

        self.reference_generation_section = (
            self._build_reference_generation_controls()
        )
        shared_layout.addWidget(self.reference_generation_section)

        self.ceiling_height_section, ceiling_height_layout = (
            _build_boxed_section(
                "Room analysis",
                "merged_generation_ceiling_height_section",
            )
        )
        self.infer_ceiling_height_button = QPushButton(
            "Infer ceiling height"
        )
        self.infer_ceiling_height_button.setObjectName(
            "infer_ceiling_height_button"
        )
        self.infer_ceiling_height_button.setToolTip(
            "Open multi-frame video analysis controls for a robust "
            "floor-to-ceiling height estimate."
        )
        ceiling_height_layout.addWidget(self.infer_ceiling_height_button)
        self.ceiling_height_result_label = QLabel(
            CEILING_HEIGHT_NOT_ESTIMATED_TEXT
        )
        self.ceiling_height_result_label.setObjectName(
            "ceiling_height_result_label"
        )
        self.ceiling_height_result_label.setWordWrap(True)
        self.ceiling_height_result_label.setTextFormat(
            Qt.TextFormat.PlainText
        )
        self.ceiling_height_result_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        ceiling_height_layout.addWidget(self.ceiling_height_result_label)
        shared_layout.addWidget(self.ceiling_height_section)

        self.clear_mask_button.setText("Clear mask")
        shared_layout.addWidget(self.mask_mode_control)

        self.shared_material_section, material_layout = _build_boxed_section(
            "Material generation",
            "merged_generation_shared_material_section",
        )
        material_layout.addWidget(self.pbr_map_control)

        self.ai_prompt_edit.setObjectName("generation_ai_prompt_edit")
        self.ai_prompt_edit.setPlaceholderText("Texture prompt (optional)")
        self.ai_prompt_edit.setMaxLength(800)
        self.ai_prompt_edit.setMinimumWidth(190)
        self.ai_prompt_edit.setToolTip(
            "Guide Surface material generation and direct new Object "
            "texturing. Staged unused-face removal and later Object "
            "Retexture jobs keep the painted image as their sole style source."
        )
        material_layout.addWidget(
            _build_labeled_inline_control("Texture prompt", self.ai_prompt_edit)
        )

        shared_layout.addWidget(self.shared_material_section)
        shared_layout.addStretch(1)

        layout.addWidget(shared_column, 0, 0)
        layout.setColumnStretch(0, 1)
        return section

    # ### Reference generation controls ###
    def _build_reference_generation_controls(self) -> QWidget:
        section, layout = _build_boxed_section(
            "Reference generation",
            "merged_generation_reference_generation_section",
        )
        self.reference_generation_view_combo = QComboBox()
        self.reference_generation_view_combo.setObjectName(
            "reference_generation_view_combo"
        )
        default_view = (
            self._reference_view_presets[0].preset_id
            if self._reference_view_presets
            else REFERENCE_GENERATION_CUSTOM_VIEW_ID
        )
        self._populate_reference_generation_views(
            str(self._reference_generation_preference(
                REFERENCE_GENERATION_VIEW_SETTINGS_KEY, default_view
            ))
        )
        layout.addWidget(
            _build_labeled_inline_control(
                "Reference view", self.reference_generation_view_combo
            )
        )
        self.remove_objects_from_furniture_checkbox = QCheckBox(
            "Remove objects from furniture"
        )
        self.remove_objects_from_furniture_checkbox.setObjectName(
            "remove_objects_from_furniture_checkbox"
        )
        self.remove_objects_from_furniture_checkbox.setChecked(
            self._default_inpaint_remove_objects_from_furniture
        )
        self.remove_objects_from_furniture_checkbox.setToolTip(
            REMOVE_OBJECTS_FROM_FURNITURE_PROMPT
        )
        layout.addWidget(self.remove_objects_from_furniture_checkbox)
        self.reference_generation_prompt_edit = QLineEdit()
        self.reference_generation_prompt_edit.setObjectName(
            "reference_generation_prompt_edit"
        )
        self.reference_generation_prompt_edit.setMaxLength(MAX_REFERENCE_PROMPT_CHARACTERS)
        self.reference_generation_prompt_edit.setPlaceholderText(
            "Choose a reference view, then edit its prompt for this object"
        )
        self.reference_generation_prompt_edit.setToolTip(
            "The prompt sent to reference image generation. Selecting a view "
            "fills its prompt; edits apply only to the current inpainted object, "
            "without changing the reusable view in Settings."
        )
        self.reference_generation_prompt_edit.setText(
            self._default_inpaint_reference_view_preferences[1]
        )
        self.reference_generation_prompt_control = _build_labeled_inline_control(
            "Reference prompt", self.reference_generation_prompt_edit
        )
        self.reference_generation_prompt_control.setObjectName(
            "reference_generation_prompt_control"
        )
        layout.addWidget(self.reference_generation_prompt_control)
        self.reference_generation_strength_spinbox = QDoubleSpinBox()
        self.reference_generation_strength_spinbox.setObjectName(
            "reference_generation_strength_spinbox"
        )
        self.reference_generation_strength_spinbox.setRange(0.05, 0.95)
        self.reference_generation_strength_spinbox.setSingleStep(0.05)
        self.reference_generation_strength_spinbox.setDecimals(2)
        saved_strength = self._reference_generation_preference(
            REFERENCE_GENERATION_STRENGTH_SETTINGS_KEY, 0.45
        )
        try:
            self.reference_generation_strength_spinbox.setValue(
                float(saved_strength)
            )
        except (TypeError, ValueError, OverflowError):
            self.reference_generation_strength_spinbox.setValue(0.45)
        self.reference_generation_strength_spinbox.setToolTip(
            "Stable Diffusion img2img strength: lower preserves the input; "
            "higher changes it more."
        )
        self.reference_generation_strength_control = _build_labeled_inline_control(
            "Image change strength", self.reference_generation_strength_spinbox
        )
        layout.addWidget(self.reference_generation_strength_control)
        self.reference_generation_negative_prompt_edit = QLineEdit()
        self.reference_generation_negative_prompt_edit.setObjectName(
            "reference_generation_negative_prompt_edit"
        )
        self.reference_generation_negative_prompt_edit.setMaxLength(4_000)
        self.reference_generation_negative_prompt_edit.setText(
            str(self._reference_generation_preference(
                REFERENCE_GENERATION_NEGATIVE_PROMPT_SETTINGS_KEY,
                DEFAULT_REFERENCE_NEGATIVE_PROMPT,
            ))
        )
        self.reference_generation_negative_prompt_edit.setToolTip(
            "Stable Diffusion only: things to avoid. The default favors a "
            "front-facing isolated object. Remove the view-related terms "
            "when requesting another angle. Clear this field to use no negative prompt."
        )
        self.reference_generation_negative_prompt_control = (
            _build_labeled_inline_control(
                "Negative prompt", self.reference_generation_negative_prompt_edit
            )
        )
        layout.addWidget(self.reference_generation_negative_prompt_control)
        self.generate_reference_button = QPushButton("Generate reference")
        self.generate_reference_button.setObjectName(
            "generate_reference_button"
        )
        self.generate_reference_button.setToolTip(
            "Generate a 1024 × 1024 object reference from the current mask, "
            "pasted image or loaded image. The result is automatically used "
            "by the next generation. Click again to retry the original input."
        )
        layout.addWidget(self.generate_reference_button)
        self.generate_all_references_button = QPushButton("Generate reference for all")
        self.generate_all_references_button.setObjectName("generate_all_references_button")
        self.generate_all_references_button.setToolTip(
            "Create one reference-generation job for each painted object across all video frames. "
            "Uses the selected model and view; results stay attached to their original objects."
        )
        layout.addWidget(self.generate_all_references_button)
        self.reference_generation_status_label = QLabel(
            "Paint a mask, paste or load an image."
        )
        self.reference_generation_status_label.setObjectName(
            "reference_generation_status_label"
        )
        self.reference_generation_status_label.setWordWrap(True)
        self.reference_generation_status_label.setTextFormat(
            Qt.TextFormat.PlainText
        )
        self.reference_generation_status_label.setText(
            self._reference_prompt_migration_warning
        )
        layout.addWidget(self.reference_generation_status_label)
        layout.addStretch(1)
        return section

    # ### Reference view presets ###
    def _populate_reference_generation_views(self, selected_id: str) -> None:
        """Keep views selected by ID without replacing the editable prompt."""

        combo = self.reference_generation_view_combo
        with QSignalBlocker(combo):
            combo.clear()
            for preset in self._reference_view_presets:
                combo.addItem(preset.name, preset.preset_id)
                combo.setItemData(
                    combo.count() - 1, preset.prompt, Qt.ItemDataRole.ToolTipRole
                )
            combo.addItem("(Write prompt)", REFERENCE_GENERATION_CUSTOM_VIEW_ID)
            index = combo.findData(selected_id)
            combo.setCurrentIndex(index if index >= 0 else combo.count() - 1)
        self._update_reference_generation_view_tooltip()

    def set_reference_view_presets(
        self, presets: Iterable[ReferenceViewPreset]
    ) -> None:
        """Apply Settings edits to future requests, leaving active jobs untouched."""

        selected_id = str(self.reference_generation_view_combo.currentData() or "")
        updated_presets = tuple(presets)
        if updated_presets == self._reference_view_presets:
            return
        previous_prompts = {
            preset.preset_id: preset.prompt for preset in self._reference_view_presets
        }
        updated_prompts = {preset.preset_id: preset.prompt for preset in updated_presets}

        def updated_preferences(preferences):
            preset_id, prompt = preferences
            if preset_id in updated_prompts:
                if prompt == previous_prompts.get(preset_id):
                    prompt = updated_prompts[preset_id]
            elif preset_id:
                preset_id = REFERENCE_GENERATION_CUSTOM_VIEW_ID
            return preset_id, prompt

        self._store_current_reference_preferences()
        self._default_inpaint_reference_view_preferences = updated_preferences(
            self._default_inpaint_reference_view_preferences
        )
        objects = self.object_workspace
        for key, preferences in tuple(objects._inpaint_reference_view_preferences.items()):
            objects.set_inpaint_reference_view_preferences(
                key, *updated_preferences(preferences)
            )
        self._reference_view_presets = updated_presets
        if (
            objects.get_landing_editing_target() is not None
            and previous_prompts.get(selected_id) != updated_prompts.get(selected_id)
        ):
            self._invalidate_landing_reference_view()
            if (
                selected_id in updated_prompts
                and self.reference_generation_prompt_edit.text()
                == previous_prompts.get(selected_id)
            ):
                with QSignalBlocker(self.reference_generation_prompt_edit):
                    self.reference_generation_prompt_edit.setText(updated_prompts[selected_id])
        self._populate_reference_generation_views(selected_id)
        self._displayed_reference_view_scope = None
        self._sync_reference_generation_controls()
        if objects.get_landing_editing_target() is not None:
            self._sync_object_reference_preview()
        if objects.current_inpaint_reference_key() is None:
            self._handle_reference_generation_preferences_changed()

    @Slot(int)
    def _handle_reference_generation_view_changed(self, _index: int) -> None:
        """Use a preset as an editable starting point for this object only."""

        landing_mode = self.object_workspace.get_landing_editing_target() is not None
        if landing_mode:
            self._invalidate_landing_reference_view()
        selected_id = self.reference_generation_view_combo.currentData()
        preset = next(
            (view for view in self._reference_view_presets if view.preset_id == selected_id),
            None,
        )
        if preset is not None:
            with QSignalBlocker(self.reference_generation_prompt_edit):
                self.reference_generation_prompt_edit.setText(preset.prompt)
        self._handle_reference_generation_preferences_changed()
        if landing_mode:
            self._handle_landing_references_changed()

    def _invalidate_landing_reference_view(self) -> None:
        """Never submit a reference for a different selected landing view."""

        objects = self.object_workspace
        objects.cancel_landing_generation()
        objects._landing_references = ()
        objects._landing_reference_signature = None
        objects.status_label.setText(
            "Landing reference view changed. Generate reference for the selected view."
        )
        objects._sync_controls()

    def _selected_reference_generation_prompt(self) -> str:
        """Use the visible prompt, with optional furniture-emptying instructions."""

        prompt = self.reference_generation_prompt_edit.text().strip()
        removal_prompt = (
            REMOVE_OBJECTS_FROM_FURNITURE_PROMPT
            if self.remove_objects_from_furniture_checkbox.isChecked()
            else ""
        )
        return ", ".join(
            part for part in (prompt, removal_prompt) if part
        )

    def _update_reference_generation_view_tooltip(self) -> None:
        selected_prompt = next(
            (
                preset.prompt
                for preset in self._reference_view_presets
                if preset.preset_id == self.reference_generation_view_combo.currentData()
            ),
            "Write a custom reference prompt. Edit reusable views in Settings.",
        )
        self.reference_generation_view_combo.setToolTip(selected_prompt)

    def _make_default_reference_negative_prompt_view_compatible(self) -> None:
        """Do not let the old frontal default contradict a different saved view."""

        if (
            self._reference_generation_model
            != REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION
            or self.reference_generation_negative_prompt_edit.text()
            != DEFAULT_REFERENCE_NEGATIVE_PROMPT
        ):
            return
        selected_view = next(
            (
                preset
                for preset in self._reference_view_presets
                if preset.preset_id == self.reference_generation_view_combo.currentData()
            ),
            None,
        )
        default_front_view = next(
            preset for preset in DEFAULT_REFERENCE_VIEW_PRESETS if preset.preset_id == "front"
        )
        if selected_view is None or selected_view == default_front_view:
            return
        with QSignalBlocker(self.reference_generation_negative_prompt_edit):
            self.reference_generation_negative_prompt_edit.setText(
                DEFAULT_REFERENCE_NEGATIVE_PROMPT.removeprefix(
                    "side view, three-quarter view, "
                )
            )

    # ### Reference generation preferences and state ###
    def _handle_geometry_options_changed(self, _index: int = 0) -> None:
        model = str(self.geometry_provider_combo.currentData())
        resolution = str(self.geometry_resolution_combo.currentData())
        self.object_workspace.set_geometry_generation_options(
            model, resolution,
        )
        self._sync_geometry_generation_controls()
        self.sync_shared_controls()

    def _sync_geometry_generation_controls(self) -> None:
        """Restore provider choices and show their supported sizing controls."""

        objects = self.object_workspace
        if not hasattr(self, "geometry_provider_combo"):
            return
        model, resolution = objects.geometry_generation_options()
        for control, value in (
            (self.geometry_provider_combo, model),
            (self.geometry_resolution_combo, resolution),
        ):
            with QSignalBlocker(control):
                control.setCurrentIndex(max(0, control.findData(value)))
        is_7_1 = model == GEOMETRY_MODEL_7_1
        if not is_7_1:
            tris = objects.meshy_target_polycount_spinbox
            tris.setMaximum(maximum_geometry_target_polycount(model))
            tris.setToolTip(
                f"Target triangle count for {self.geometry_provider_combo.currentText()} "
                f"({tris.minimum():,}-{tris.maximum():,}). Lower values make lighter geometry."
            )
        self.geometry_resolution_control.setVisible(is_7_1)
        objects.meshy_target_polycount_control.setVisible(not is_7_1)
        self.generate_all_references_button.setVisible(
            objects.get_landing_editing_target() is None
        )

    def _read_initial_reference_preferences(self, default_view: str) -> tuple[str, str]:
        """Migrate the former custom/extra fields into one authoritative prompt."""

        self._reference_prompt_migration_warning = ""
        preset_id = str(self._reference_generation_preference(
            REFERENCE_GENERATION_VIEW_SETTINGS_KEY, default_view,
        ))
        preset = next(
            (view for view in self._reference_view_presets if view.preset_id == preset_id),
            None,
        )
        if preset is None:
            preset_id = REFERENCE_GENERATION_CUSTOM_VIEW_ID
        prompt = self._reference_generation_preference(
            REFERENCE_GENERATION_PROMPT_SETTINGS_KEY, None,
        )
        if not isinstance(prompt, str):
            legacy_custom = self._reference_generation_preference(
                "generation/reference_generation_prompt", "",
            )
            legacy_extra = self._reference_generation_preference(
                "generation/reference_generation_ai_prompt", "",
            )
            base = preset.prompt if preset is not None else str(legacy_custom).strip()
            prompt = ", ".join(
                part for part in (base, str(legacy_extra).strip()) if part
            )
        if len(prompt) > MAX_REFERENCE_PROMPT_CHARACTERS:
            self._reference_prompt_migration_warning = (
                "Saved reference instructions are too long. Only the first "
                f"{MAX_REFERENCE_PROMPT_CHARACTERS:,} characters were loaded; "
                "the full saved text was left unchanged."
            )
            prompt = prompt[:MAX_REFERENCE_PROMPT_CHARACTERS]
        return preset_id, prompt

    def set_reference_generation_model(self, model: str) -> None:
        """Apply the Settings provider to future requests and local controls."""

        if not isinstance(model, str) or model not in REFERENCE_GENERATION_MODELS:
            raise ValueError(f"Unknown reference generation model: {model!r}.")
        if model == self._reference_generation_model:
            return
        self._reference_generation_model = model
        self._sync_reference_generation_controls()

    def set_object_symmetry_inference(self, enabled: bool, model: str) -> None:
        """Apply Settings to future reference jobs without changing running ones."""

        if type(enabled) is not bool:
            raise TypeError("Object symmetry inference must be enabled or disabled.")
        if not isinstance(model, str) or model not in dict(SYMMETRY_INFERENCE_MODEL_OPTIONS).values():
            raise ValueError(f"Unknown symmetry inference model: {model!r}.")
        self._enable_object_symmetry_inference = enabled
        self._symmetry_inference_model = model

    def _reference_symmetry_model(self) -> str | None:
        return self._symmetry_inference_model if self._enable_object_symmetry_inference else None

    def _reference_generation_preference(
        self, key: str, default: object
    ) -> object:
        if self._application_settings is None:
            return default
        return self._application_settings.get(key, default)

    def _reference_generation_api_key(self) -> str:
        """Keep image-provider credentials separate from OpenAI analysis keys."""

        settings = self.object_workspace.get_runtime_settings()
        return (
            settings.meshy_api_key
            if self._reference_generation_model == REFERENCE_GENERATION_MODEL_MESHY
            else settings.openai_api_key
        ).strip()

    def _reference_symmetry_credentials(self) -> dict[str, str]:
        """Override analysis credentials only when the image provider differs."""

        if self._reference_generation_model != REFERENCE_GENERATION_MODEL_MESHY:
            return {}
        return {
            "symmetry_api_key": self.object_workspace.get_runtime_settings().openai_api_key.strip()
        }

    @Slot()
    def _handle_reference_generation_text_changed(self, *_arguments: object) -> None:
        """Keep each keystroke local; committing text must not rebuild the scene."""

        if self._is_shutdown:
            return
        self._store_current_reference_preferences()
        self._reference_preferences_dirty = True
        self._sync_reference_prompt_action_availability()

    def flush_reference_generation_preferences(self) -> None:
        """Commit the latest visible draft without scene/Atlas notifications."""

        if self._is_shutdown:
            return
        self._store_current_reference_preferences()
        self._persist_reference_generation_preferences()
        self._reference_preferences_dirty = False

    def _sync_reference_prompt_action_availability(self) -> None:
        """Refresh prompt-dependent buttons using already-known input readiness."""

        has_prompt = bool(self._selected_reference_generation_prompt())
        landing_mode = self.object_workspace.get_landing_editing_target() is not None
        running = (
            self.object_workspace.is_landing_generating if landing_mode
            else self._reference_generation_runtime is not None
        )
        batch_running = bool(self._batch_reference_generation_runtimes) or self._local_reference_batch_thread is not None
        self.generate_reference_button.setEnabled(
            not self._is_shutdown and not batch_running
            and (running or (self._reference_generate_can_start and has_prompt))
        )
        self.generate_all_references_button.setEnabled(
            not self._is_shutdown
            and (batch_running or (self._reference_generate_all_can_start and has_prompt))
        )

    @Slot()
    def _handle_reference_generation_preferences_changed(
        self, *_arguments: object
    ) -> None:
        """Remember reference-only settings without changing Meshy's prompt."""

        if self._is_shutdown:
            return
        self._update_reference_generation_view_tooltip()
        self._make_default_reference_negative_prompt_view_compatible()
        self.flush_reference_generation_preferences()
        self._sync_reference_generation_controls()

    def _persist_reference_generation_preferences(self) -> None:
        """Write changed application defaults only at an explicit commit boundary."""

        if self._application_settings is not None:
            preferences = [
                (
                    REFERENCE_GENERATION_STRENGTH_SETTINGS_KEY,
                    self.reference_generation_strength_spinbox.value(),
                ),
                (
                    REFERENCE_GENERATION_NEGATIVE_PROMPT_SETTINGS_KEY,
                    self.reference_generation_negative_prompt_edit.text(),
                ),
            ]
            if self.object_workspace.current_inpaint_reference_key() is None:
                preferences.extend(
                    (
                        (
                            REFERENCE_GENERATION_PROMPT_SETTINGS_KEY,
                            self.reference_generation_prompt_edit.text(),
                        ),
                        (
                            REFERENCE_GENERATION_VIEW_SETTINGS_KEY,
                            self.reference_generation_view_combo.currentData(),
                        ),
                        (
                            REMOVE_OBJECTS_FROM_FURNITURE_SETTINGS_KEY,
                            self.remove_objects_from_furniture_checkbox.isChecked(),
                        ),
                    )
                )
            for key, value in preferences:
                if self._application_settings.get(key) != value:
                    self._application_settings.set(key, value)

    def _sync_reference_generation_controls(self) -> None:
        """Keep local-only options and the cancellable action coherent."""

        if self.object_workspace.get_landing_editing_target() is not None:
            running = self.object_workspace.is_landing_generating
            self.generate_reference_button.setText(
                "Cancel" if running else "Generate reference"
            )
            self._reference_generate_can_start = self.object_workspace._has_current_object_generation_reference()
            self._reference_generate_all_can_start = False
            self._sync_reference_prompt_action_availability()
            self.generate_all_references_button.setVisible(False)
            self.reference_generation_view_combo.setEnabled(not running)
            self.reference_generation_prompt_edit.setEnabled(not running)
            self.reference_generation_strength_spinbox.setEnabled(not running)
            self.reference_generation_negative_prompt_edit.setEnabled(not running)
            self.remove_objects_from_furniture_checkbox.setEnabled(False)
            self.reference_generation_status_label.setText(
                self.object_workspace.status_label.text()
            )
            self.reference_generation_strength_control.setVisible(
                self._reference_generation_model
                == REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION
            )
            self.reference_generation_negative_prompt_control.setVisible(
                self._reference_generation_model
                == REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION
            )
            return
        self._sync_geometry_generation_controls()
        self._sync_inpaint_reference_preferences()
        self._make_default_reference_negative_prompt_view_compatible()
        self._cancel_batch_references_if_source_changed()
        self._update_reference_view_tried_labels()
        running = self._reference_generation_runtime is not None
        batch_running = bool(self._batch_reference_generation_runtimes) or self._local_reference_batch_thread is not None
        self.generate_reference_button.setText(
            "Cancel" if running else "Generate reference"
        )
        self._reference_generate_can_start = (
            not self._is_shutdown
            and not batch_running
            and (
                self.object_workspace.has_temporary_object_reference()
                or self.video_view.has_selection()
            )
        )
        self.generate_all_references_button.setText(
            "Cancel all" if batch_running else "Generate reference for all"
        )
        self._reference_generate_all_can_start = (
            not self._is_shutdown
            and not running
            and not self.object_workspace.component_generation_active
            and bool(self.object_workspace.inpaint_queue())
        )
        self._sync_reference_prompt_action_availability()
        settings_available = not running and not batch_running and not self._is_shutdown
        self.reference_generation_view_combo.setEnabled(settings_available)
        self.remove_objects_from_furniture_checkbox.setEnabled(settings_available)
        self.reference_generation_prompt_edit.setEnabled(settings_available)
        self.reference_generation_strength_spinbox.setEnabled(settings_available)
        self.reference_generation_negative_prompt_edit.setEnabled(settings_available)
        is_stable_diffusion = (
            self._reference_generation_model
            == REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION
        )
        self.reference_generation_strength_control.setVisible(
            is_stable_diffusion
        )
        self.reference_generation_negative_prompt_control.setVisible(
            is_stable_diffusion
        )

    # ### Reference generation jobs ###
    def _reference_generation_context_signature(self) -> tuple[object, ...]:
        """Compare input bytes only at request boundaries, never every paint."""

        objects = self.object_workspace
        reference = objects.get_temporary_object_reference()
        temporary_signature = None
        if reference is not None:
            temporary_signature = (
                reference.shape,
                hashlib.blake2b(
                    np.ascontiguousarray(reference).tobytes(), digest_size=16
                ).digest(),
            )
        door_target = objects._door_slot_editing_target
        trim_target = objects._architectural_trim_editing_target
        if door_target is not None and not door_target.generation_ready:
            door_target = None
        if trim_target is not None and not trim_target.generation_ready:
            trim_target = None
        return (
            self._reference_generation_input_revision,
            id(objects._data),
            id(objects._video_source),
            objects._displayed_frame_index,
            objects.current_inpaint_reference_key(),
            tuple(self.video_view.get_strokes()),
            temporary_signature,
            (
                None
                if door_target is None
                else (door_target.door_id, door_target.slot_id)
            ),
            (
                None
                if trim_target is None
                else (trim_target.level_index, trim_target.trim_id)
            ),
        )

    def _cancel_reference_generation_if_context_changed(self) -> None:
        """Honor non-interactive frame refreshes and programmatic source changes."""

        runtime = self._reference_generation_runtime
        retry_source = self._reference_generation_retry_source
        if runtime is None and retry_source is None:
            return
        signature = self._reference_generation_context_signature()
        if (
            runtime is not None and runtime.context_signature != signature
        ) or (
            retry_source is not None and retry_source.context_signature != signature
        ):
            self._invalidate_reference_generation_input()

    @Slot()
    def _invalidate_reference_generation_input(self) -> None:
        """A new mask, source or frame cannot receive an older async result."""

        self._reference_generation_input_revision += 1
        self._reference_generation_retry_source = None
        self._standalone_reference_tried_views.clear()
        runtime = self._reference_generation_runtime
        if runtime is not None:
            self._cancel_reference_generation(runtime.job_id)

    @Slot()
    def _handle_generate_reference_clicked(self) -> None:
        self.flush_reference_generation_preferences()
        if self.object_workspace.get_landing_editing_target() is not None:
            if self.object_workspace.is_landing_generating:
                self.object_workspace.cancel_landing_generation()
            else:
                prompt = self._selected_reference_generation_prompt()
                if not prompt:
                    self.reference_generation_status_label.setText(
                        "Choose a reference view or write a custom reference prompt first."
                    )
                    return
                selected_id = self.reference_generation_view_combo.currentData()
                preset = next(
                    (view for view in self._reference_view_presets
                     if view.preset_id == selected_id), None
                )
                view = ReferenceViewPreset(
                    preset.preset_id if preset is not None else "landing-custom",
                    preset.name if preset is not None else "Custom prompt",
                    prompt,
                )
                self.object_workspace.generate_landing_references(
                    (view,),
                    model=self._reference_generation_model,
                    api_key=self._reference_generation_api_key(),
                    strength=self.reference_generation_strength_spinbox.value(),
                    negative_prompt=self.reference_generation_negative_prompt_edit.text(),
                )
            self._sync_reference_generation_controls()
            return
        runtime = self._reference_generation_runtime
        if runtime is not None:
            self.job_manager.cancel_job(runtime.job_id)
            return
        if self._is_shutdown or self._batch_reference_generation_runtimes or self._local_reference_batch_thread is not None:
            return
        prompt = self._selected_reference_generation_prompt()
        source = self.object_workspace.get_current_object_generation_reference()
        if source is None or not prompt:
            self.reference_generation_status_label.setText(
                "Paint a mask, paste or load an image, then choose a reference view "
                "or select (Write prompt) and enter a custom prompt."
            )
            self._sync_reference_generation_controls()
            return
        signature = self._reference_generation_context_signature()
        retry_source = self._reference_generation_retry_source
        if retry_source is not None and retry_source.context_signature == signature:
            source = retry_source.source_bgra.copy()
        else:
            batch_retry = self._batch_reference_retry_sources.get(
                self.object_workspace.current_inpaint_reference_key()
            )
            if self._batch_retry_source_matches(batch_retry, source):
                source = batch_retry.source_bgra.copy()
        model = self._reference_generation_model
        api_key = self._reference_generation_api_key()
        thread = _ReferenceGenerationThread(
            source,
            prompt,
            model,
            api_key=api_key,
            strength=self.reference_generation_strength_spinbox.value(),
            negative_prompt=self.reference_generation_negative_prompt_edit.text(),
            parent=self,
            **self._reference_symmetry_credentials(),
        )
        thread.symmetry_model = self._reference_symmetry_model()
        job = self.job_manager.create_job(
            kind=REFERENCE_GENERATION_JOB_KIND,
            requested_name="",
            default_name="Object reference",
            stage="Preparing reference image",
        )
        self._reference_generation_runtime = _ReferenceGenerationRuntime(
            thread=thread,
            job_id=job.job_id,
            context_signature=signature,
            source_bgra=np.ascontiguousarray(source).copy(),
            reference_view=self._selected_reference_view_token(),
        )
        self.job_manager.set_cancel_callback(
            job.job_id, partial(self._cancel_reference_generation, job.job_id)
        )
        thread.progress_changed.connect(
            partial(
                self._handle_reference_generation_progress, job.job_id, thread
            )
        )
        thread.finished.connect(
            partial(
                self._handle_reference_generation_finished, job.job_id, thread
            )
        )
        self.reference_generation_status_label.setText(
            "Generating reference..."
        )
        self._sync_reference_generation_controls()
        thread.start()

    def _cancel_reference_generation(self, job_id: str) -> bool:
        """Retire a worker without blocking the GUI or a new reference request."""

        runtime = self._reference_generation_runtime
        if runtime is None or runtime.job_id != str(job_id):
            return False
        runtime.cancel_requested = True
        runtime.thread.requestInterruption()
        self._reference_generation_runtime = None
        self._retired_reference_generation_runtimes[runtime.job_id] = runtime
        self.job_manager.set_cancel_callback(runtime.job_id, None)
        self.job_manager.mark_cancelled(runtime.job_id)
        self.reference_generation_status_label.setText(
            "Reference generation cancelled."
        )
        self._sync_reference_generation_controls()
        return True

    def _handle_reference_generation_progress(
        self,
        job_id: str,
        thread: _ReferenceGenerationThread,
        stage: str,
        progress: int,
    ) -> None:
        runtime = self._reference_generation_runtime
        if (
            runtime is None
            or runtime.job_id != job_id
            or runtime.thread is not thread
        ):
            return
        self.job_manager.update_job(job_id, stage=stage, progress=progress)
        self.reference_generation_status_label.setText(str(stage))

    def _handle_reference_generation_finished(
        self,
        job_id: str,
        thread: _ReferenceGenerationThread,
    ) -> None:
        """Automatically accept only results still belonging to the input."""

        retired = self._retired_reference_generation_runtimes.pop(job_id, None)
        if retired is not None:
            thread.deleteLater()
            return
        runtime = self._reference_generation_runtime
        if (
            runtime is None
            or runtime.job_id != job_id
            or runtime.thread is not thread
        ):
            return
        self._reference_generation_runtime = None
        self.job_manager.set_cancel_callback(job_id, None)
        try:
            if (
                self._is_shutdown
                or runtime.cancel_requested
                or thread.was_cancelled
                or runtime.context_signature
                != self._reference_generation_context_signature()
            ):
                self.job_manager.mark_cancelled(
                    job_id, stage="Reference input changed; result discarded"
                )
                self.reference_generation_status_label.setText(
                    "Reference input changed; the previous result was discarded."
                )
                return
            if thread.error_message is not None or thread.result is None:
                message = (
                    thread.error_message
                    or "The model did not return a usable image."
                )
                self.job_manager.fail_job(job_id, stage=message)
                self.reference_generation_status_label.setText(
                    f"Reference generation failed: {message}"
                )
                return
            self._activate_temporary_object_reference(
                thread.result,
                source_label="Generated reference",
                generated=True,
            )
            active_reference = (
                self.object_workspace.get_temporary_object_reference()
            )
            if active_reference is None or not np.array_equal(
                active_reference, thread.result
            ):
                self.job_manager.fail_job(
                    job_id, stage="The generated reference could not be applied."
                )
                return
            self._reference_generation_retry_source = (
                _ReferenceGenerationRetrySource(
                    source_bgra=runtime.source_bgra.copy(),
                    context_signature=(
                        self._reference_generation_context_signature()
                    ),
                )
            )
            self._mark_reference_view_tried(runtime.reference_view, self.object_workspace.current_inpaint_reference_key())
            symmetry_status = self._apply_generated_reference_symmetry(
                thread, self.object_workspace.current_inpaint_reference_key(),
            )
            if getattr(thread, "symmetry_error", None):
                self.job_manager.fail_job(job_id, stage=symmetry_status)
                self.reference_generation_status_label.setText(symmetry_status)
            else:
                self.job_manager.complete_job(job_id, stage=symmetry_status or "Completed")
                self.reference_generation_status_label.setText(
                    "Generated reference is active. Generate reference retries the original input; "
                    "move the video seekbar to return to the video. " + symmetry_status
                )
        finally:
            thread.deleteLater()
            self._sync_reference_generation_controls()

    def _apply_generated_reference_symmetry(self, thread, inpaint_key) -> str:
        """Only the matching reference's completed inference may change its flag."""

        error = getattr(thread, "symmetry_error", None)
        if error:
            return f"Reference generated; symmetry inference failed: {error}. The checkbox was kept unchanged."
        result = getattr(thread, "symmetry_result", None)
        if result is None:
            return ""
        return self._apply_reference_symmetry_result(result, inpaint_key)

    def _apply_reference_symmetry_result(self, result, inpaint_key) -> str:
        if not isinstance(result.symmetric, bool):
            raise TypeError("The symmetry model did not return a boolean decision.")
        if inpaint_key is not None:
            if not self.object_workspace.set_inpaint_symmetric_division(inpaint_key, result.symmetric):
                return "The painted object changed; its symmetry result was discarded."
        elif self.object_workspace.symmetric_division_checkbox.isEnabled():
            self.object_workspace.symmetric_division_checkbox.setChecked(result.symmetric)
        labels = {model: name for name, model in SYMMETRY_INFERENCE_MODEL_OPTIONS}
        label = labels.get(result.model, result.model)
        decision = "on" if result.symmetric else "off"
        confidence = "" if result.confidence is None else f", score {result.confidence:.2f}"
        return (f"Symmetric division {decision} ({label}, {result.elapsed_seconds:.1f}s{confidence}). "
                f"{result.reason}").strip()

    # ### Per-object reference preferences ###
    def _store_current_reference_preferences(self) -> None:
        preferences = (
            str(self.reference_generation_view_combo.currentData() or ""),
            self.reference_generation_prompt_edit.text(),
        )
        key = self.object_workspace.current_inpaint_reference_key()
        if key is None:
            self._default_inpaint_reference_view_preferences = preferences
            self._default_inpaint_remove_objects_from_furniture = self.remove_objects_from_furniture_checkbox.isChecked()
        else:
            self.object_workspace.set_inpaint_reference_view_preferences(key, *preferences)
            self.object_workspace.set_inpaint_remove_objects_from_furniture(
                key, self.remove_objects_from_furniture_checkbox.isChecked(),
            )

    def _inpaint_reference_view_preferences(self, key):
        preferences = self.object_workspace.get_inpaint_reference_view_preferences(key)
        preset_id, prompt = preferences or self._default_inpaint_reference_view_preferences
        if preset_id and not any(preset.preset_id == preset_id for preset in self._reference_view_presets):
            preset_id = REFERENCE_GENERATION_CUSTOM_VIEW_ID
        return preset_id, prompt

    def _inpaint_remove_objects_from_furniture(self, key) -> bool:
        enabled = self.object_workspace.get_inpaint_remove_objects_from_furniture(key)
        return self._default_inpaint_remove_objects_from_furniture if enabled is None else enabled

    def _sync_inpaint_reference_preferences(self) -> None:
        """Restore one blob's reference options without changing other blobs."""

        objects = self.object_workspace
        key = objects.current_inpaint_reference_key()
        scope = id(objects._data), id(objects._video_source), key
        if scope == self._displayed_reference_view_scope:
            return
        self._displayed_reference_view_scope = scope
        if key is None:
            preset_id, prompt = self._default_inpaint_reference_view_preferences
            if self.reference_generation_view_combo.findData(preset_id) < 0:
                preset_id = REFERENCE_GENERATION_CUSTOM_VIEW_ID
            remove_objects = self._default_inpaint_remove_objects_from_furniture
        else:
            preset_id, prompt = self._inpaint_reference_view_preferences(key)
            remove_objects = self._inpaint_remove_objects_from_furniture(key)
            objects.set_inpaint_reference_view_preferences(key, preset_id, prompt)
            objects.set_inpaint_remove_objects_from_furniture(key, remove_objects)
        combo = self.reference_generation_view_combo
        with (
            QSignalBlocker(combo), QSignalBlocker(self.reference_generation_prompt_edit),
            QSignalBlocker(self.remove_objects_from_furniture_checkbox),
        ):
            combo.setCurrentIndex(combo.findData(preset_id))
            self.reference_generation_prompt_edit.setText(prompt)
            self.remove_objects_from_furniture_checkbox.setChecked(remove_objects)
        self._update_reference_generation_view_tooltip()

    def _reference_options_for_inpaint(self, key):
        preset_id, reference_prompt = self._inpaint_reference_view_preferences(key)
        preset = next((view for view in self._reference_view_presets if view.preset_id == preset_id), None)
        removal_prompt = (
            REMOVE_OBJECTS_FROM_FURNITURE_PROMPT
            if self._inpaint_remove_objects_from_furniture(key)
            else ""
        )
        prompt = ", ".join(part for part in (reference_prompt.strip(), removal_prompt) if part)
        if not prompt:
            raise ValueError(f"The inpainted object on frame {key[0] + 1} needs a reference view or custom prompt.")
        token = (preset.preset_id, preset.prompt.strip()) if preset is not None else None
        return prompt, token

    # ### Tried reference views ###
    def _selected_reference_view_token(self) -> tuple[str, str] | None:
        selected_id = self.reference_generation_view_combo.currentData()
        return next(
            ((preset.preset_id, preset.prompt.strip()) for preset in self._reference_view_presets
             if preset.preset_id == selected_id),
            None,
        )

    def _mark_reference_view_tried(self, token, inpaint_key=None) -> None:
        """Record the view captured by an accepted job, never its current label."""

        if token is None:
            return
        if inpaint_key is None:
            self._standalone_reference_tried_views.add(token)
        else:
            self.object_workspace.mark_inpaint_reference_view_tried(inpaint_key, *token)
        self._update_reference_view_tried_labels()

    def _update_reference_view_tried_labels(self) -> None:
        """Decorate this object's menu without changing IDs or saved presets."""

        key = self.object_workspace.current_inpaint_reference_key()
        tried = (
            self._standalone_reference_tried_views if key is None
            else self.object_workspace.get_inpaint_tried_reference_views(key)
        )
        combo = self.reference_generation_view_combo
        with QSignalBlocker(combo):
            for preset in self._reference_view_presets:
                index = combo.findData(preset.preset_id)
                if index >= 0:
                    suffix = " (Already tried)" if (preset.preset_id, preset.prompt.strip()) in tried else ""
                    label = preset.name + suffix
                    if combo.itemText(index) != label:
                        combo.setItemText(index, label)

    # ### Reference generation for all painted objects ###
    def _batch_retry_source_matches(self, retry, current_source: np.ndarray) -> bool:
        objects = self.object_workspace
        return bool(
            retry is not None
            and retry.generation_data is objects._data
            and retry.video_source is objects._video_source
            and retry.output_signature == _reference_image_signature(current_source)
        )

    def _cancel_batch_references_if_source_changed(self) -> None:
        scope = self._reference_batch_scope
        if scope is None or self._cancelling_all_reference_jobs:
            return
        objects = self.object_workspace
        if scope[0] is not objects._data or scope[1] is not objects._video_source:
            self._reference_batch_scope = None
            self._batch_reference_retry_sources.clear()
            self._cancel_all_reference_generations()

    @Slot()
    def _handle_generate_all_references_clicked(self) -> None:
        if self.object_workspace.get_landing_editing_target() is not None:
            self._handle_generate_reference_clicked()
            return
        if self._batch_reference_generation_runtimes or self._local_reference_batch_thread is not None:
            self._cancel_all_reference_generations()
            return
        objects = self.object_workspace
        if (self._is_shutdown or self._reference_generation_runtime is not None
                or objects.component_generation_active):
            return
        self.flush_reference_generation_preferences()
        prompt = self._selected_reference_generation_prompt()
        if not prompt:
            self.reference_generation_status_label.setText("Choose a reference view or write a prompt first.")
            return
        try:
            inputs = objects.snapshot_inpaint_reference_inputs()
        except (OSError, RuntimeError, ValueError) as error:
            self.reference_generation_status_label.setText(f"Reference generation could not start: {error}")
            return
        if not inputs:
            self.reference_generation_status_label.setText("Paint at least one object before generating references.")
            return
        try:
            reference_options = {key: self._reference_options_for_inpaint(key) for key, _source in inputs}
        except ValueError as error:
            self.reference_generation_status_label.setText(str(error))
            return
        self._cancel_batch_references_if_source_changed()
        self._reference_batch_scope = objects._data, objects._video_source
        api_key = self._reference_generation_api_key()
        single_retry = self._reference_generation_retry_source
        single_signature = self._reference_generation_context_signature() if single_retry is not None else None
        if self._reference_generation_model in {
            PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1, REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
            LUNA_BATCH_MODEL,
        } or self._reference_symmetry_model() == LUNA_BATCH_MODEL:
            self._start_shared_reference_batch(inputs, prompt, api_key, single_retry, single_signature, reference_options)
            return
        for ordinal, (key, current_source) in enumerate(inputs, start=1):
            retry = self._batch_reference_retry_sources.get(key)
            use_batch_retry = self._batch_retry_source_matches(retry, current_source)
            source = retry.source_bgra if use_batch_retry else current_source
            if (not use_batch_retry and single_retry is not None
                    and single_retry.context_signature == single_signature
                    and key == objects.current_inpaint_reference_key()):
                source = single_retry.source_bgra
            thread = _ReferenceGenerationThread(
                source, reference_options[key][0], self._reference_generation_model,
                api_key=api_key,
                strength=self.reference_generation_strength_spinbox.value(),
                negative_prompt=self.reference_generation_negative_prompt_edit.text(),
                parent=self,
                **self._reference_symmetry_credentials(),
            )
            thread.symmetry_model = self._reference_symmetry_model()
            job = self.job_manager.create_job(
                kind=REFERENCE_GENERATION_JOB_KIND, requested_name="",
                default_name=f"Object reference {ordinal} (frame {key[0] + 1})",
                stage="Queued reference generation",
            )
            runtime = _BatchReferenceGenerationRuntime(
                thread=thread, job_id=job.job_id, inpaint_key=key,
                generation_data=objects._data, video_source=objects._video_source,
                source_bgra=np.ascontiguousarray(source).copy(),
                input_signature=_reference_image_signature(current_source),
                reference_view=reference_options[key][1],
            )
            self._batch_reference_generation_runtimes[job.job_id] = runtime
            self.job_manager.set_cancel_callback(job.job_id, partial(self._cancel_batch_reference_generation, job.job_id))
            thread.progress_changed.connect(partial(self._handle_batch_reference_generation_progress, job.job_id, thread))
            thread.finished.connect(partial(self._handle_batch_reference_generation_finished, job.job_id, thread))
        self._batch_reference_generation_job_ids = tuple(self._batch_reference_generation_runtimes)
        self.reference_generation_status_label.setText(f"Queued {len(inputs)} object reference jobs.")
        self._sync_reference_generation_controls()
        self._start_next_batch_reference_generation()

    def _batch_reference_input_matches(self, runtime: _BatchReferenceGenerationRuntime) -> bool:
        objects = self.object_workspace
        if (self._is_shutdown or runtime.generation_data is not objects._data
                or runtime.video_source is not objects._video_source):
            return False
        current = objects.get_inpaint_reference_input(runtime.inpaint_key)
        return current is not None and _reference_image_signature(current) == runtime.input_signature

    def _start_next_batch_reference_generation(self) -> None:
        # Create all job rows immediately, but avoid loading multiple local
        # image models concurrently or retaining a thread per waiting GPU job.
        if (self._is_shutdown or self._cancelling_all_reference_jobs
                or self._local_reference_batch_thread is not None
                or any(runtime.started for runtime in self._batch_reference_generation_runtimes.values())
                or any(isinstance(runtime, _BatchReferenceGenerationRuntime) and runtime.thread.isRunning()
                       for runtime in self._retired_reference_generation_runtimes.values())):
            return
        for job_id, runtime in tuple(self._batch_reference_generation_runtimes.items()):
            try:
                matches_input = self._batch_reference_input_matches(runtime)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                self._batch_reference_generation_runtimes.pop(job_id)
                self.job_manager.set_cancel_callback(job_id, None)
                self.job_manager.fail_job(job_id, stage=f"The object reference could not be read: {error}")
                runtime.thread.deleteLater()
                continue
            if not matches_input:
                self._batch_reference_generation_runtimes.pop(job_id)
                self.job_manager.set_cancel_callback(job_id, None)
                self.job_manager.mark_cancelled(job_id, stage="Object reference input changed")
                runtime.thread.deleteLater()
                continue
            runtime.started = True
            self.job_manager.update_job(job_id, stage="Generating object reference", progress=0)
            try:
                runtime.thread.start()
            except RuntimeError as error:
                self._batch_reference_generation_runtimes.pop(job_id)
                self.job_manager.set_cancel_callback(job_id, None)
                self.job_manager.fail_job(job_id, stage=f"The reference worker could not start: {error}")
                runtime.thread.deleteLater()
                continue
            break
        self._update_batch_reference_status()
        self._sync_reference_generation_controls()

    def _cancel_batch_reference_generation(self, job_id: str) -> bool:
        runtime = self._batch_reference_generation_runtimes.pop(str(job_id), None)
        if runtime is None:
            return False
        runtime.cancel_requested = True
        if runtime.thread is self._local_reference_batch_thread:
            runtime.thread.cancel_item(str(job_id))
            self.job_manager.set_cancel_callback(job_id, None)
            self.job_manager.mark_cancelled(job_id)
            if not self._cancelling_all_reference_jobs:
                self._update_batch_reference_status()
                self._sync_reference_generation_controls()
            return True
        runtime.thread.requestInterruption()
        self.job_manager.set_cancel_callback(job_id, None)
        self.job_manager.mark_cancelled(job_id)
        if runtime.started:
            self._retired_reference_generation_runtimes[job_id] = runtime
        else:
            runtime.thread.deleteLater()
        if not self._cancelling_all_reference_jobs:
            self._start_next_batch_reference_generation()
            self._update_batch_reference_status()
        return True

    def _cancel_all_reference_generations(self) -> None:
        self._cancelling_all_reference_jobs = True
        try:
            if self._local_reference_batch_thread is not None:
                self._local_reference_batch_thread.requestInterruption()
            for job_id in tuple(self._batch_reference_generation_runtimes):
                self._cancel_batch_reference_generation(job_id)
        finally:
            self._cancelling_all_reference_jobs = False
        self._update_batch_reference_status()
        self._sync_reference_generation_controls()

    def _update_batch_reference_status(self) -> None:
        counts = Counter(
            job.status for job_id in self._batch_reference_generation_job_ids
            if (job := self.job_manager.get_job(job_id)) is not None
        )
        self.reference_generation_status_label.setText(
            f"Reference jobs: {counts[JOB_STATUS_COMPLETED]} completed, "
            f"{counts[JOB_STATUS_FAILED]} failed, {counts[JOB_STATUS_CANCELLED]} cancelled; "
            f"{len(self._batch_reference_generation_runtimes)} remaining."
        )

    def _handle_batch_reference_generation_progress(self, job_id, thread, stage, progress) -> None:
        runtime = self._batch_reference_generation_runtimes.get(job_id)
        if runtime is None or runtime.thread is not thread:
            return
        self.job_manager.update_job(job_id, stage=stage, progress=progress)
        self.reference_generation_status_label.setText(f"{self.job_manager.get_job(job_id).name}: {stage}")

    def _handle_batch_reference_generation_finished(self, job_id, thread) -> None:
        retired = self._retired_reference_generation_runtimes.pop(job_id, None)
        if retired is not None:
            thread.deleteLater()
            self._start_next_batch_reference_generation()
            return
        runtime = self._batch_reference_generation_runtimes.get(job_id)
        if runtime is None or runtime.thread is not thread:
            return
        self._batch_reference_generation_runtimes.pop(job_id)
        self.job_manager.set_cancel_callback(job_id, None)
        try:
            if runtime.cancel_requested or thread.was_cancelled or not self._batch_reference_input_matches(runtime):
                self.job_manager.mark_cancelled(job_id, stage="Object reference input changed; result discarded")
            elif thread.error_message is not None or thread.result is None:
                self.job_manager.fail_job(job_id, stage=thread.error_message or "The model returned no usable image.")
            elif not self.object_workspace.set_inpaint_reference(runtime.inpaint_key, thread.result):
                self.job_manager.mark_cancelled(job_id, stage="The painted object no longer exists")
            else:
                self._batch_reference_retry_sources[runtime.inpaint_key] = _BatchReferenceRetrySource(
                    generation_data=runtime.generation_data, video_source=runtime.video_source,
                    source_bgra=runtime.source_bgra.copy(), output_signature=_reference_image_signature(thread.result),
                )
                self._mark_reference_view_tried(runtime.reference_view, runtime.inpaint_key)
                symmetry_status = self._apply_generated_reference_symmetry(thread, runtime.inpaint_key)
                if getattr(thread, "symmetry_error", None):
                    self.job_manager.fail_job(job_id, stage=symmetry_status)
                else:
                    self.job_manager.complete_job(job_id, stage=symmetry_status or "Completed")
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self.job_manager.fail_job(job_id, stage=f"Reference generation failed: {error}")
        finally:
            thread.deleteLater()
            self._start_next_batch_reference_generation()
            self._update_batch_reference_status()
            self._sync_reference_generation_controls()

    # ### Resident local and remote OpenAI batch integration ###
    def _start_shared_reference_batch(
        self, inputs, prompt, api_key, single_retry, single_signature, reference_options,
    ) -> None:
        """Snapshot all inputs for one shared local session or remote batch."""

        objects = self.object_workspace
        remote_batch = self._reference_generation_model == LUNA_BATCH_MODEL
        local_images = self._reference_generation_model in {
            PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1, REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
        }
        queued = []
        for ordinal, (key, current_source) in enumerate(inputs, start=1):
            retry = self._batch_reference_retry_sources.get(key)
            use_batch_retry = self._batch_retry_source_matches(retry, current_source)
            source = retry.source_bgra if use_batch_retry else current_source
            if (not use_batch_retry and single_retry is not None
                    and single_retry.context_signature == single_signature
                    and key == objects.current_inpaint_reference_key()):
                source = single_retry.source_bgra
            job = self.job_manager.create_job(
                kind=REFERENCE_GENERATION_JOB_KIND, requested_name="",
                default_name=f"Object reference {ordinal} (frame {key[0] + 1})",
                stage=(
                    "Queued OpenAI Batch reference generation (up to 24 hours)"
                    if remote_batch else "Queued reference generation"
                ),
            )
            queued.append((job.job_id, key, np.ascontiguousarray(source).copy(), current_source))
        worker_class = (
            _OpenAIReferenceBatchThread if remote_batch
            else _LocalReferenceBatchThread if local_images
            else _APIReferenceBatchThread
        )
        thread = worker_class(
            tuple(LocalReferenceBatchItem(job_id, source, reference_options[key][0])
                  for job_id, key, source, _ in queued),
            self._reference_generation_model, self._reference_symmetry_model(),
            api_key=api_key, prompt=prompt,
            strength=self.reference_generation_strength_spinbox.value(),
            negative_prompt=self.reference_generation_negative_prompt_edit.text(), parent=self,
            **self._reference_symmetry_credentials(),
        )
        self._local_reference_batch_thread = thread
        for job_id, key, source, current_source in queued:
            self._batch_reference_generation_runtimes[job_id] = _BatchReferenceGenerationRuntime(
                thread=thread, job_id=job_id, inpaint_key=key,
                generation_data=objects._data, video_source=objects._video_source,
                source_bgra=source, input_signature=_reference_image_signature(current_source),
                reference_view=reference_options[key][1],
            )
            self.job_manager.set_cancel_callback(job_id, partial(self._cancel_batch_reference_generation, job_id))
        self._batch_reference_generation_job_ids = tuple(self._batch_reference_generation_runtimes)
        thread.item_progress.connect(self._handle_local_reference_progress)
        thread.reference_ready.connect(self._handle_local_reference_ready)
        thread.item_finished.connect(self._handle_local_reference_item_finished)
        thread.finished.connect(self._handle_local_reference_worker_stopped)
        phases = (
            "image generation first, symmetry inference second"
            if thread.symmetry_model is not None else "symmetry inference disabled"
        )
        self.reference_generation_status_label.setText(
            (
                f"Queued {len(queued)} OpenAI Batch references (up to 24 hours); "
                "keep the app open. Cancelling one item ignores its result; "
                "submitted requests may still be billed. Cancel all requests remote cancellation."
                if remote_batch
                else (
                    f"Queued {len(queued)} references; symmetry uses OpenAI Batch "
                    "(up to 24 hours). Keep the app open."
                    if thread.symmetry_model == LUNA_BATCH_MODEL
                    else f"Queued {len(queued)} references; {phases}."
                )
            )
        )
        self._sync_reference_generation_controls()
        try:
            thread.start()
        except RuntimeError as error:
            thread.error_message = f"The reference batch worker could not start: {error}"
            self._handle_local_reference_batch_finished(thread)

    def _matching_local_reference_runtime(self, job_id):
        runtime = self._batch_reference_generation_runtimes.get(job_id)
        if runtime is None or runtime.thread is not self._local_reference_batch_thread:
            return None
        try:
            matches = self._batch_reference_input_matches(runtime)
        except (OSError, RuntimeError, TypeError, ValueError):
            matches = False
        if not matches:
            self._cancel_batch_reference_generation(job_id)
            return None
        return runtime

    @Slot(str, str, int)
    def _handle_local_reference_progress(self, job_id, stage, progress) -> None:
        runtime = self._matching_local_reference_runtime(job_id)
        if runtime is None:
            return
        runtime.started = True
        self.job_manager.update_job(job_id, stage=stage, progress=progress)
        self.reference_generation_status_label.setText(f"{self.job_manager.get_job(job_id).name}: {stage}")

    @Slot(str, object)
    def _handle_local_reference_ready(self, job_id, image) -> None:
        runtime = self._matching_local_reference_runtime(job_id)
        if runtime is None:
            return
        try:
            if not self.object_workspace.set_inpaint_reference(runtime.inpaint_key, image):
                self._cancel_batch_reference_generation(job_id)
                return
            runtime.input_signature = _reference_image_signature(image)
            runtime.reference_applied = True
            self._mark_reference_view_tried(runtime.reference_view, runtime.inpaint_key)
            self._batch_reference_retry_sources[runtime.inpaint_key] = _BatchReferenceRetrySource(
                generation_data=runtime.generation_data, video_source=runtime.video_source,
                source_bgra=runtime.source_bgra.copy(), output_signature=runtime.input_signature,
            )
            stage = "Reference ready; queued for symmetry inference" if runtime.thread.symmetry_model is not None else "Reference ready"
            self.job_manager.update_job(job_id, stage=stage, progress=85)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._cancel_batch_reference_generation(job_id)
            self.job_manager.fail_job(job_id, stage=f"The generated reference could not be applied: {error}")

    @Slot(str, object, str, bool)
    def _handle_local_reference_item_finished(self, job_id, result, error, cancelled) -> None:
        runtime = self._matching_local_reference_runtime(job_id)
        if runtime is None:
            return
        self._batch_reference_generation_runtimes.pop(job_id)
        self.job_manager.set_cancel_callback(job_id, None)
        try:
            if cancelled or runtime.cancel_requested:
                self.job_manager.mark_cancelled(job_id)
            elif not runtime.reference_applied:
                self.job_manager.fail_job(job_id, stage=error or "The model did not return a usable reference.")
            else:
                symmetry_status = self._apply_reference_symmetry_result(result, runtime.inpaint_key) if result is not None else ""
                if error:
                    self.job_manager.fail_job(job_id, stage=" ".join(part for part in (symmetry_status, error) if part))
                else:
                    self.job_manager.complete_job(job_id, stage=symmetry_status or "Completed")
        except (OSError, RuntimeError, TypeError, ValueError) as failure:
            self.job_manager.fail_job(job_id, stage=f"Reference processing failed: {failure}")
        self._update_batch_reference_status()
        self._sync_reference_generation_controls()

    @Slot()
    def _handle_local_reference_worker_stopped(self) -> None:
        self._handle_local_reference_batch_finished(self.sender())

    def _handle_local_reference_batch_finished(self, thread) -> None:
        if thread is not self._local_reference_batch_thread:
            return
        for job_id, runtime in tuple(self._batch_reference_generation_runtimes.items()):
            if runtime.thread is thread:
                self._batch_reference_generation_runtimes.pop(job_id)
                self.job_manager.set_cancel_callback(job_id, None)
                if self._is_shutdown:
                    self.job_manager.mark_cancelled(job_id)
                else:
                    self.job_manager.fail_job(job_id, stage=thread.error_message or "The local reference worker stopped unexpectedly.")
        self._local_reference_batch_thread = None
        thread.deleteLater()
        self._update_batch_reference_status()
        self._sync_reference_generation_controls()

    # ### Object generation controls ###
    def _build_object_controls(self) -> QWidget:
        objects = self.object_workspace
        section, layout = _build_controls_section("object")

        primary_column, primary_layout = _build_controls_column(
            "merged_generation_object_primary_column"
        )

        preview_options = QWidget()
        preview_options.setObjectName("merged_generation_object_preview_options")
        preview_options_layout = QHBoxLayout(preview_options)
        preview_options_layout.setContentsMargins(0, 0, 0, 0)
        preview_options_layout.setSpacing(6)
        preview_options_layout.addWidget(objects.textures_checkbox)
        preview_options_layout.addWidget(objects.wireframe_checkbox)
        preview_options_layout.addStretch(1)

        details_layout = objects.object_3d_panel.details_panel.layout()
        relocated_detail_widgets = (
            objects.delete_selected_faces_button,
            objects.convert_faces_to_glass_button,
            objects.model_statistics_label,
            objects.object_3d_panel.projection_camera_controls,
        )
        if details_layout is not None:
            for widget in relocated_detail_widgets:
                details_layout.removeWidget(widget)

        self.object_editing_section, editing_layout = _build_boxed_section(
            "Object editing",
            "merged_generation_object_editing_section",
        )
        editing_layout.addWidget(preview_options)

        face_actions = QWidget()
        face_actions.setObjectName("merged_generation_object_face_actions")
        face_actions_layout = QHBoxLayout(face_actions)
        face_actions_layout.setContentsMargins(0, 0, 0, 0)
        face_actions_layout.setSpacing(6)
        face_actions_layout.addWidget(objects.delete_selected_faces_button, 1)
        face_actions_layout.addWidget(objects.convert_faces_to_glass_button, 1)
        editing_layout.addWidget(face_actions)
        primary_layout.addWidget(self.object_editing_section)

        self.object_creation_section, creation_layout = _build_boxed_section(
            "",
            "merged_generation_object_creation_section",
        )
        generation_header = QWidget(self.object_creation_section)
        generation_header_layout = QHBoxLayout(generation_header)
        generation_header_layout.setContentsMargins(0, 0, 0, 0)
        generation_header_layout.setSpacing(5)
        self.object_generation_heading_label = QLabel("Generation")
        self.object_generation_heading_label.setObjectName(
            "object_generation_heading_label"
        )
        self.door_slot_generation_indicator = QLabel()
        self.door_slot_generation_indicator.setObjectName(
            "door_slot_generation_indicator"
        )
        self.door_slot_generation_indicator.setStyleSheet(
            DOOR_SLOT_GENERATION_ACTIVE_STYLE
        )
        self.door_slot_generation_indicator.hide()
        generation_header_layout.addWidget(self.object_generation_heading_label)
        generation_header_layout.addWidget(self.door_slot_generation_indicator)
        generation_header_layout.addStretch(1)
        creation_layout.addWidget(generation_header)

        self.inpaint_navigation_control = QWidget()
        self.inpaint_navigation_control.setObjectName("inpaint_navigation_control")
        navigation_layout = QHBoxLayout(self.inpaint_navigation_control)
        navigation_layout.setContentsMargins(0, 0, 0, 0)
        navigation_layout.setSpacing(6)
        self.previous_inpaint_button = QPushButton("Previous")
        self.previous_inpaint_button.setObjectName("previous_inpaint_button")
        self.next_inpaint_button = QPushButton("Next")
        self.next_inpaint_button.setObjectName("next_inpaint_button")
        self.inpaint_position_label = QLabel("No inpaints")
        self.inpaint_position_label.setObjectName("inpaint_position_label")
        self.previous_inpaint_button.setToolTip(
            "Select the previous painted object, including objects on other video frames."
        )
        self.next_inpaint_button.setToolTip(
            "Select the next painted object, including objects on other video frames."
        )
        navigation_layout.addWidget(self.previous_inpaint_button)
        navigation_layout.addWidget(self.next_inpaint_button)
        navigation_layout.addWidget(self.inpaint_position_label)
        navigation_layout.addStretch(1)
        creation_layout.addWidget(self.inpaint_navigation_control)

        self.landing_mode_label = QLabel(
            "Landing mode — generate the selected Reference view, then use Meshy "
            "to generate geometry or the complete textured model. This one "
            "reference guides both geometry and texturing."
        )
        self.landing_mode_label.setObjectName("landing_generation_mode_label")
        self.landing_mode_label.setWordWrap(True)
        self.landing_mode_label.setStyleSheet("color: #ffd166; font-weight: 600;")
        self.landing_mode_label.hide()
        creation_layout.addWidget(self.landing_mode_label)

        generation_settings = QWidget()
        generation_settings.setObjectName(
            "merged_generation_object_generation_settings"
        )
        generation_settings_layout = QHBoxLayout(generation_settings)
        generation_settings_layout.setContentsMargins(0, 0, 0, 0)
        generation_settings_layout.setSpacing(6)

        self.division_options_section, division_options_layout = (
            _build_boxed_section(
                "Division",
                "merged_generation_division_options_section",
            )
        )
        division_options_row = QWidget()
        division_options_row_layout = QHBoxLayout(division_options_row)
        division_options_row_layout.setContentsMargins(0, 0, 0, 0)
        division_options_row_layout.setSpacing(6)
        division_options_row_layout.addWidget(
            objects.symmetric_division_checkbox
        )
        division_options_row_layout.addWidget(
            objects.side_door_duplication_checkbox
        )
        division_options_layout.addWidget(division_options_row)
        generation_settings_layout.addWidget(self.division_options_section)

        self.opening_profile_options_section, profile_options_layout = (
            _build_boxed_section(
                "Door / window profile",
                "merged_generation_opening_profile_options_section",
            )
        )
        profile_options_row = QWidget()
        profile_options_row_layout = QHBoxLayout(profile_options_row)
        profile_options_row_layout.setContentsMargins(0, 0, 0, 0)
        profile_options_row_layout.setSpacing(6)
        profile_options_row_layout.addWidget(
            objects.generate_displacement_checkbox
        )
        profile_options_row_layout.addWidget(
            objects.preserve_door_profile_checkbox
        )
        profile_options_layout.addWidget(profile_options_row)
        generation_settings_layout.addWidget(
            self.opening_profile_options_section
        )
        generation_settings_layout.addStretch(1)
        creation_layout.addWidget(generation_settings)
        creation_layout.addWidget(objects.meshy_target_polycount_control)

        self.geometry_resolution_combo = QComboBox()
        self.geometry_resolution_combo.setObjectName("geometry_resolution_combo")
        for label, value in GEOMETRY_RESOLUTION_OPTIONS:
            self.geometry_resolution_combo.addItem(label, value)
        self.geometry_resolution_control = _build_labeled_inline_control(
            "Resolution", self.geometry_resolution_combo,
        )
        self.geometry_resolution_control.hide()
        creation_layout.addWidget(self.geometry_resolution_control)
        self.geometry_provider_combo = QComboBox()
        self.geometry_provider_combo.setObjectName("geometry_provider_combo")
        for label, value in GEOMETRY_MODEL_OPTIONS:
            self.geometry_provider_combo.addItem(label, value)
        self.geometry_provider_control = _build_labeled_inline_control(
            "Geometry provider", self.geometry_provider_combo,
        )
        creation_layout.addWidget(self.geometry_provider_control)

        self.texture_provider_control = _build_labeled_inline_control(
            "Surface texture provider",
            self.surface_workspace.surface_texture_provider_combo,
        )
        self.texture_provider_control.setObjectName(
            "merged_generation_texture_provider_control"
        )
        creation_layout.addWidget(self.texture_provider_control)

        primary_actions = QWidget()
        primary_actions.setObjectName("merged_generation_primary_object_actions")
        primary_actions_layout = QHBoxLayout(primary_actions)
        primary_actions_layout.setContentsMargins(0, 0, 0, 0)
        primary_actions_layout.setSpacing(6)
        primary_actions_layout.addWidget(objects.generate_geometry_button, 1)
        self.generate_texture_stack = QStackedWidget()
        self.generate_texture_stack.setObjectName(
            "merged_generation_texture_action_stack"
        )
        objects.generate_texture_button.setText("Generate texture")
        self.surface_workspace.generate_button.setText("Generate texture")
        self.generate_texture_stack.addWidget(objects.generate_texture_button)
        self.generate_texture_stack.addWidget(
            self.surface_workspace.generate_button
        )
        primary_actions_layout.addWidget(self.generate_texture_stack, 1)
        creation_layout.addWidget(primary_actions)

        generation_actions = QWidget()
        generation_actions.setObjectName("merged_generation_object_generation_actions")
        generation_actions_layout = QHBoxLayout(generation_actions)
        generation_actions_layout.setContentsMargins(0, 0, 0, 0)
        generation_actions_layout.setSpacing(6)
        generation_actions_layout.addWidget(objects.generate_button, 1)
        self.place_object_button = objects.place_button
        generation_actions_layout.addWidget(self.place_object_button, 1)
        creation_layout.addWidget(generation_actions)

        # Both rows share two equal-width columns, including the alternate
        # surface-texture action and each action's temporary Cancel state.
        action_buttons = (
            objects.generate_geometry_button,
            objects.generate_texture_button,
            self.surface_workspace.generate_button,
            objects.generate_button,
            self.place_object_button,
        )
        action_button_height = max(
            max(button.minimumHeight(), button.sizeHint().height())
            for button in action_buttons
        )
        for button in action_buttons:
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            button.setFixedHeight(action_button_height)
        self.generate_texture_stack.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self.generate_texture_stack.setFixedHeight(action_button_height)

        primary_layout.addWidget(self.object_creation_section)
        primary_layout.addWidget(objects.model_statistics_label)
        primary_layout.addStretch(1)

        projection_controls = objects.object_3d_panel.projection_camera_controls
        projection_controls.setMinimumWidth(0)
        projection_controls.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        layout.addWidget(primary_column, 0, 0)
        layout.addWidget(projection_controls, 0, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(1, 1)
        return section

    def _connect_shared_controls(self) -> None:
        surface = self.surface_workspace
        objects = self.object_workspace

        try:
            self.load_video_button.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.load_video_button.clicked.connect(self._handle_load_video_clicked)
        self.copy_inpaint_button.clicked.connect(
            self._handle_copy_inpaint_clicked
        )
        self.paste_inpaint_button.clicked.connect(
            self._handle_paste_inpaint_clicked
        )
        self.load_reference_image_button.clicked.connect(
            self._handle_load_reference_image_clicked
        )
        self.generate_reference_button.clicked.connect(self._handle_generate_reference_clicked)
        self.generate_all_references_button.clicked.connect(self._handle_generate_all_references_clicked)
        self.geometry_provider_combo.currentIndexChanged.connect(self._handle_geometry_options_changed)
        self.geometry_resolution_combo.currentIndexChanged.connect(self._handle_geometry_options_changed)
        self.previous_inpaint_button.clicked.connect(
            partial(self._navigate_inpaints, -1)
        )
        self.next_inpaint_button.clicked.connect(
            partial(self._navigate_inpaints, 1)
        )
        self.reference_generation_view_combo.currentIndexChanged.connect(
            self._handle_reference_generation_view_changed
        )
        self.remove_objects_from_furniture_checkbox.toggled.connect(
            self._handle_reference_generation_preferences_changed
        )
        self.reference_generation_prompt_edit.textChanged.connect(
            self._handle_reference_generation_text_changed
        )
        self.reference_generation_prompt_edit.editingFinished.connect(
            self._handle_reference_generation_preferences_changed
        )
        self.reference_generation_strength_spinbox.valueChanged.connect(
            self._handle_reference_generation_preferences_changed
        )
        self.reference_generation_negative_prompt_edit.textChanged.connect(
            self._handle_reference_generation_text_changed
        )
        self.reference_generation_negative_prompt_edit.editingFinished.connect(
            self._handle_reference_generation_preferences_changed
        )
        QApplication.clipboard().dataChanged.connect(
            self.sync_shared_controls
        )
        self.infer_ceiling_height_button.clicked.connect(
            self._show_ceiling_height_analysis_dialog
        )

        self.video_view.strokes_changed.connect(surface._handle_video_strokes_changed)
        self.video_view.strokes_changed.connect(self._invalidate_reference_generation_input)
        self.video_view.strokes_changed.connect(self.sync_shared_controls)
        self.video_view.frame_changed.connect(self._handle_video_frame_changed)
        objects.inpaint_selection_changed.connect(self._handle_inpaint_selection_changed)
        objects.door_slot_editing_changed.connect(
            self._handle_door_slot_editing_changed
        )
        objects.architectural_trim_editing_changed.connect(
            self._handle_architectural_trim_editing_changed
        )
        objects.landing_editing_changed.connect(self._handle_landing_editing_changed)
        objects.landing_references_changed.connect(self._handle_landing_references_changed)
        surface.data_changed.connect(
            self._handle_surface_generation_context_changed
        )
        surface.selection_changed.connect(
            self._handle_surface_generation_context_changed
        )
        objects.door_slot_generation_status_changed.connect(
            self._handle_door_slot_generation_status_changed
        )
        objects.controls_synced.connect(self._sync_generation_action_buttons)
        objects.controls_synced.connect(self._sync_geometry_generation_controls)
        objects.controls_synced.connect(self._sync_generation_texture_context)
        surface.controls_synced.connect(self._sync_generation_action_buttons)
        try:
            self.seekbar.valueChanged.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.seekbar.valueChanged.connect(self._handle_seekbar_changed)
        for checkbox in self.pbr_map_checkboxes.values():
            checkbox.toggled.connect(surface._handle_pbr_map_toggled)

        self._connect_generation_action_buttons()
        self.job_manager.job_added.connect(self._handle_job_changed)
        self.job_manager.job_updated.connect(self._handle_job_changed)
        self.job_manager.jobs_cleared.connect(self._sync_generation_action_buttons)

        objects._sync_controls()
        surface._sync_controls()
        self._sync_generation_texture_context()

    def _connect_generation_action_buttons(self) -> None:
        """Route each generation action through its own cancellable job group."""

        objects = self.object_workspace
        actions = {
            GENERATION_ACTION_GENERATE: objects.generate_button,
            GENERATION_ACTION_GEOMETRY: objects.generate_geometry_button,
            GENERATION_ACTION_OBJECT_TEXTURE: objects.generate_texture_button,
            GENERATION_ACTION_SURFACE_TEXTURE: self.surface_workspace.generate_button,
        }
        self._generation_action_buttons = actions
        self._generation_action_default_text = {
            action: button.text() for action, button in actions.items()
        }
        self._generation_action_default_tooltips = {
            action: button.toolTip() for action, button in actions.items()
        }
        for action, button in actions.items():
            try:
                button.clicked.disconnect()
            except (RuntimeError, TypeError):
                pass
            button.clicked.connect(
                partial(self._handle_generation_action_clicked, action)
            )

    @Slot(str, bool)
    def _handle_generation_action_clicked(
        self,
        action: str,
        _checked: bool = False,
    ) -> None:
        """Start one action or cancel only the jobs launched by its last click."""

        if self._generation_action_has_pending_jobs(action):
            self._cancel_generation_action(action)
            return

        previous_job_ids = {job.job_id for job in self.job_manager.jobs()}
        self._start_generation_action(action)
        created_job_ids = tuple(
            job.job_id
            for job in self.job_manager.jobs()
            if job.job_id not in previous_job_ids and not job.is_finished
        )
        if created_job_ids:
            self._generation_action_job_ids[action] = created_job_ids
        self._sync_generation_action_buttons()

    def _start_generation_action(self, action: str) -> None:
        """Invoke the current backend method for one visible action button."""

        if action == GENERATION_ACTION_GENERATE:
            self.object_workspace.generate()
            return
        if action == GENERATION_ACTION_GEOMETRY:
            self.object_workspace.generate_geometry()
            return
        if action == GENERATION_ACTION_OBJECT_TEXTURE:
            self.object_workspace.generate_selected_object_texture()
            return
        if action == GENERATION_ACTION_SURFACE_TEXTURE:
            self.surface_workspace.generate()
            return
        raise ValueError(f"Unknown generation action: {action}")

    def _generation_action_has_pending_jobs(self, action: str) -> bool:
        """Return whether the action still owns a running or cancelling job."""

        return any(
            job is not None and not job.is_finished
            for job_id in self._generation_action_job_ids.get(action, ())
            for job in (self.job_manager.get_job(job_id),)
        )

    def _cancel_generation_action(self, action: str) -> bool:
        """Cancel every still-running job produced by one action click."""

        accepted = False
        for job_id in self._generation_action_job_ids.get(action, ()):
            job = self.job_manager.get_job(job_id)
            if job is None or job.status != JOB_STATUS_RUNNING:
                continue
            accepted = self.job_manager.cancel_job(job_id) or accepted
        self._sync_generation_action_buttons()
        return accepted

    @Slot()
    def _sync_generation_action_buttons(self) -> None:
        """Keep each action's label and availability aligned with its own jobs."""

        for action, button in self._generation_action_buttons.items():
            job_ids = self._generation_action_job_ids.get(action, ())
            pending_jobs = tuple(
                job
                for job_id in job_ids
                if (
                    (job := self.job_manager.get_job(job_id)) is not None
                    and not job.is_finished
                )
            )
            if not pending_jobs:
                self._generation_action_job_ids.pop(action, None)
                button.setText(self._generation_action_default_text[action])
                button.setToolTip(
                    self._generation_action_default_tooltips[action]
                )
                continue

            button.setText("Cancel")
            button.setEnabled(
                any(job.status == JOB_STATUS_RUNNING for job in pending_jobs)
            )
            job_count = len(pending_jobs)
            button.setToolTip(
                "Cancel this generation job."
                if job_count == 1
                else f"Cancel all {job_count} jobs from this generation batch."
            )

    @Slot(bool)
    def _handle_door_slot_editing_changed(self, active: bool) -> None:
        """Refresh opening-only generation controls after a slot change."""

        del active
        self._cancel_reference_generation_if_context_changed()
        self._sync_opening_generation_sections()
        self.sync_shared_controls()

    def _sync_opening_generation_sections(self) -> None:
        """Ignore passive library selections until slot generation is opened."""

        target = self.object_workspace._door_slot_editing_target
        body_slot_is_active = bool(
            target is not None
            and target.generation_ready
            and is_door_body_slot(target.slot_id)
        )
        non_body_slot_is_active = bool(
            target is not None
            and target.generation_ready
            and not body_slot_is_active
        )
        self.division_options_section.setVisible(
            not non_body_slot_is_active
        )
        self.opening_profile_options_section.setVisible(body_slot_is_active)

    @Slot(bool)
    def _handle_architectural_trim_editing_changed(self, active: bool) -> None:
        """Refresh the texture route after trim selection changes."""

        del active
        self._cancel_reference_generation_if_context_changed()
        self._sync_generation_texture_context()

    @Slot(object)
    def _handle_surface_generation_context_changed(self, _data: object) -> None:
        """Follow shared-scene surface selection without duplicating its state."""

        self._sync_generation_texture_context()

    def _sync_generation_texture_context(self) -> None:
        """Show the texture action belonging to the active semantic target."""

        objects = self.object_workspace
        component_is_active = objects.component_generation_active
        surface_is_selected = bool(
            self.surface_workspace.get_selected_surface_ids()
        )
        surface_target_is_active = surface_is_selected and not component_is_active
        object_is_selected = (
            objects._selected_object_id in objects.get_placeable_object_names_by_id()
        )
        use_surface_texture_pipeline = surface_target_is_active or (
            not component_is_active and not object_is_selected
        )
        # Library-only Surface generation must not disable new Object meshes.
        objects.set_external_surface_generation_target_selected(
            surface_target_is_active
        )
        target_button = (
            self.surface_workspace.generate_button
            if use_surface_texture_pipeline
            else self.object_workspace.generate_texture_button
        )
        self.generate_texture_stack.setCurrentWidget(target_button)
        self._sync_generation_action_buttons()

    @Slot(bool)
    def _handle_landing_editing_changed(self, active: bool) -> None:
        """Keep normal reference workers out of the landing's separate context."""

        self._cancel_all_reference_generations()
        if active:
            self.surface_workspace.set_scene_surface_selection(())
        self._handle_landing_references_changed()

    @Slot()
    def _handle_landing_references_changed(self) -> None:
        self.landing_mode_label.setVisible(
            self.object_workspace.get_landing_editing_target() is not None
        )
        self._sync_object_reference_preview()
        self._sync_reference_generation_controls()
        self._sync_generation_texture_context()
        self._sync_generation_action_buttons()

    @Slot(str)
    def _handle_door_slot_generation_status_changed(
        self,
        slot_name: str,
    ) -> None:
        """Blink the exact in-flight opening slot beside Generation."""

        self._door_slot_generation_name = str(slot_name).strip()
        if not self._door_slot_generation_name:
            self._door_slot_generation_blink_timer.stop()
            self.door_slot_generation_indicator.hide()
            return
        asset_kind = self.object_workspace.active_opening_component_kind()
        asset_label = "window" if asset_kind == "window" else "door"
        self.door_slot_generation_indicator.setText(
            f"- Generating the {asset_label}'s {self._door_slot_generation_name}"
        )
        self._door_slot_generation_blink_visible = True
        self.door_slot_generation_indicator.setStyleSheet(
            DOOR_SLOT_GENERATION_ACTIVE_STYLE
        )
        self.door_slot_generation_indicator.show()
        self._door_slot_generation_blink_timer.start()

    @Slot()
    def _toggle_door_slot_generation_blink(self) -> None:
        """Alternate red and transparent text without shifting the layout."""

        if not self._door_slot_generation_name:
            return
        self._door_slot_generation_blink_visible = (
            not self._door_slot_generation_blink_visible
        )
        self.door_slot_generation_indicator.setStyleSheet(
            DOOR_SLOT_GENERATION_ACTIVE_STYLE
            if self._door_slot_generation_blink_visible
            else DOOR_SLOT_GENERATION_HIDDEN_STYLE
        )

    @Slot()
    def _handle_video_frame_changed(self) -> None:
        """Leave marked analysis independent of Generation's seek position."""

        self._restore_temporary_reference_after_same_frame_refresh()
        self._cancel_reference_generation_if_context_changed()
        self._sync_object_reference_preview()
        self._sync_ceiling_height_analysis_source()
        self.current_video_frame_changed.emit()
        self.sync_shared_controls()

    # ### Cross-frame manual object navigation ###
    def _navigate_inpaints(self, delta: int) -> None:
        """Keep the shared Surface editor on the selected object's frame too."""

        if not self.object_workspace.navigate_inpaints(delta):
            return
        objects, surface = self.object_workspace, self.surface_workspace
        surface._data.current_frame_index = objects._data.current_frame_index
        surface._displayed_frame_index = objects._displayed_frame_index
        if surface._displayed_frame_index is not None:
            surface._data.set_frame_strokes(
                surface._displayed_frame_index, self.video_view.get_strokes(),
            )
        surface._sync_painted_frames_label()
        self._handle_inpaint_selection_changed()

    @Slot()
    def _handle_inpaint_selection_changed(self) -> None:
        """Never reuse a different blob's image or late reference-generation result."""

        self._cancel_reference_generation_if_context_changed()
        session = self._temporary_object_reference_session
        if session is not None and not self._temporary_reference_session_matches_context(session):
            self._clear_temporary_object_reference_session()
        reference = self.object_workspace.get_temporary_object_reference()
        if reference is None:
            self.surface_workspace.clear_temporary_reference()
        else:
            self.surface_workspace.set_temporary_reference(reference)
        self.surface_workspace._handle_pbr_map_toggled(False)
        self._sync_object_reference_preview()
        self.sync_shared_controls()

    @Slot()
    def _show_ceiling_height_analysis_dialog(self) -> None:
        """Open the independent player, defaulting to Generation's video."""

        if self._is_shutdown:
            return
        self._sync_ceiling_height_analysis_source()
        dialog = self._ensure_ceiling_height_analysis_dialog()
        self._sync_ceiling_height_analysis_source()
        self._sync_ceiling_height_dialog_levels()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _ensure_ceiling_height_analysis_dialog(
        self,
    ) -> CeilingHeightAnalysisDialog:
        dialog = self._ceiling_height_dialog
        if dialog is not None:
            return dialog
        dialog = CeilingHeightAnalysisDialog(
            application_settings=self._application_settings,
            parent=self.infer_ceiling_height_button.window(),
        )
        dialog.analysis_requested.connect(
            self._start_ceiling_height_inference
        )
        dialog.cancellation_requested.connect(
            self._cancel_active_ceiling_height_inference
        )
        self._ceiling_height_dialog = dialog
        dialog.set_model(self._ceiling_height_estimation_model)
        dialog.source_changed.connect(self._handle_ceiling_height_analysis_input_changed)
        dialog.ranges_changed.connect(self._handle_ceiling_height_analysis_input_changed)
        self._sync_ceiling_height_dialog_levels()
        self._sync_ceiling_height_analysis_source()
        return dialog

    @Slot(str, int, object)
    def _start_ceiling_height_inference(
        self,
        model: str,
        frame_interval: int,
        marked_ranges: object,
    ) -> None:
        """Analyze only marked sections, keeping their explicit level labels."""

        if self._is_shutdown or self._ceiling_height_runtime is not None:
            return
        dialog = self._ensure_ceiling_height_analysis_dialog()
        if not dialog.video_path:
            dialog.set_result("Load a video before starting analysis.", is_error=True)
            return
        level_options = self._ceiling_height_level_options
        known_levels = dict(level_options)
        level_plan_contexts = self._ceiling_height_level_plan_contexts
        level_plan_sources = self._ceiling_height_level_plan_sources
        try:
            metadata = probe_video(dialog.video_path)
            ranges = tuple(marked_ranges)  # type: ignore[arg-type]
            frame_samples = sample_marked_video_frames(
                ranges,
                frame_interval,
                metadata.frame_count,
            )
            if any(level_index not in known_levels for _, level_index in frame_samples):
                raise ValueError("Mark only levels that have a loaded plan image.")
        except (OSError, TypeError, ValueError) as error:
            dialog.set_result(str(error), is_error=True)
            return
        if not frame_samples:
            dialog.set_result("Mark at least one video section before analysis.", is_error=True)
            return
        frame_indices = tuple(frame_index for frame_index, _ in frame_samples)
        marked_level_indices = {level_index for _, level_index in frame_samples}
        marked_plan_sources = tuple(
            source
            for source in level_plan_sources
            if source.level_index in marked_level_indices
        )
        plan_reference_size = _largest_ceiling_height_plan_reference_size(
            level_plan_sources
        )
        estimated_cost = estimate_ceiling_height_api_cost_usd(
            model=model,
            frame_count=metadata.frame_count,
            interval=frame_interval,
            frame_width=metadata.width,
            frame_height=metadata.height,
            plan_reference_size=plan_reference_size,
            sample_count=len(frame_samples),
        )
        if len(frame_indices) > 120:
            request_description = (
                "paid vision requests"
                if ceiling_height_model_requires_api_key(model)
                else "local vision analyses"
            )
            answer = QMessageBox.question(
                dialog,
                "Analyse many video frames?",
                (
                    f"The marked sections will run {len(frame_indices):,} "
                    f"{request_description}"
                    f"{_format_cost_confirmation_suffix(estimated_cost)}. "
                    "Continue?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        uses_local_model = not ceiling_height_model_requires_api_key(model)
        api_key = ""
        if not uses_local_model:
            api_key = (
                self.object_workspace.get_runtime_settings()
                .openai_api_key.strip()
            )
        if not uses_local_model and not api_key:
            dialog.set_result(
                "Add an OpenAI API key in Settings before starting analysis.",
                is_error=True,
            )
            return

        video_path = str(metadata.path)
        thread = _CeilingHeightInferenceThread(
            video_path,
            frame_samples,
            api_key,
            model,
            self._ceiling_height_estimator,
            level_options,
            level_plan_contexts,
            ranges,
            level_plan_sources=marked_plan_sources,
            parent=self,
        )
        initial_stage = (
            "Loading local Qwen model"
            if uses_local_model
            else (
                "Preparing OpenAI Batch analysis (up to 24 hours; keep the app open)"
                if model == LUNA_BATCH_MODEL
                else f"Analysing 0 of {len(frame_indices):,} frame intervals"
            )
        )
        job = self.job_manager.create_job(
            kind=CEILING_HEIGHT_JOB_KIND,
            requested_name="",
            default_name="Ceiling height",
            stage=initial_stage,
        )
        runtime = _CeilingHeightInferenceRuntime(
            video_path=video_path,
            frame_indices=frame_indices,
            level_signature=level_options,
            plan_context_signature=level_plan_contexts,
            plan_source_signature=level_plan_sources,
            marked_ranges=ranges,
            context_revision=self._ceiling_height_context_revision,
            job_id=job.job_id,
            thread=thread,
        )
        self._ceiling_height_runtime = runtime
        self.job_manager.set_cancel_callback(
            job.job_id,
            partial(self._cancel_ceiling_height_inference, job.job_id),
        )
        thread.finished.connect(
            partial(
                self._handle_ceiling_height_inference_finished,
                job.job_id,
                thread,
            )
        )
        thread.progress_changed.connect(
            partial(
                self._handle_ceiling_height_inference_progress,
                job.job_id,
                thread,
            )
        )
        status_signal = getattr(thread, "status_changed", None)
        if status_signal is not None:
            status_signal.connect(partial(
                self._handle_ceiling_height_inference_status,
                job.job_id, thread,
            ))
        self.ceiling_height_result_label.setText(
            "Ceiling height: Loading local Qwen model..."
            if uses_local_model
            else f"Ceiling height: Analysing {len(frame_indices)} frames..."
        )
        dialog.set_running(True)
        dialog.set_progress(0, len(frame_indices))
        self.sync_shared_controls()
        thread.start()

    @Slot()
    def _cancel_active_ceiling_height_inference(self) -> None:
        runtime = self._ceiling_height_runtime
        if runtime is not None:
            self.job_manager.cancel_job(runtime.job_id)

    def _cancel_ceiling_height_inference(self, job_id: str) -> bool:
        runtime = self._ceiling_height_runtime
        if runtime is None or runtime.job_id != str(job_id):
            return False
        runtime.cancel_requested = True
        runtime.thread.requestInterruption()
        self.sync_shared_controls()
        return True

    def _handle_ceiling_height_inference_progress(
        self,
        job_id: str,
        thread: _CeilingHeightInferenceThread,
        completed_frames: int,
        total_frames: int,
        current_frame: int,
    ) -> None:
        runtime = self._ceiling_height_runtime
        if (
            runtime is None
            or runtime.job_id != job_id
            or runtime.thread is not thread
        ):
            return
        progress = round(
            100.0 * int(completed_frames) / max(int(total_frames), 1)
        )
        self.job_manager.update_job(
            job_id,
            stage=(
                f"Analysed {int(completed_frames):,} of "
                f"{int(total_frames):,} frame intervals "
                f"(video frame {int(current_frame) + 1:,})"
            ),
            progress=progress,
        )
        if self._ceiling_height_dialog is not None:
            self._ceiling_height_dialog.set_progress(
                completed_frames,
                total_frames,
                current_frame=current_frame,
            )

    def _handle_ceiling_height_inference_status(
        self, job_id: str, thread: _CeilingHeightInferenceThread, stage: str,
    ) -> None:
        """Expose remote Batch queuing and progress without blocking the editor."""

        runtime = self._ceiling_height_runtime
        if runtime is None or runtime.job_id != job_id or runtime.thread is not thread:
            return
        self.job_manager.update_job(job_id, stage=stage)
        self.ceiling_height_result_label.setText(f"Ceiling height: {stage}")
        if self._ceiling_height_dialog is not None:
            self._ceiling_height_dialog.set_result(stage)

    def _handle_ceiling_height_inference_finished(
        self,
        job_id: str,
        thread: _CeilingHeightInferenceThread,
    ) -> None:
        """Publish per-level consensus results for the source-bound video."""

        runtime = self._ceiling_height_runtime
        try:
            if (
                runtime is None
                or runtime.job_id != job_id
                or runtime.thread is not thread
            ):
                return
            self.job_manager.set_cancel_callback(job_id, None)
            dialog = self._ceiling_height_dialog
            levels_are_stale = (
                self._ceiling_height_level_options
                != runtime.level_signature
                or self._ceiling_height_level_plan_contexts
                != runtime.plan_context_signature
                or self._ceiling_height_level_plan_sources
                != runtime.plan_source_signature
            )
            is_stale = bool(
                dialog is None
                or dialog.video_path != runtime.video_path
                or dialog.marked_ranges != runtime.marked_ranges
                or runtime.context_revision != self._ceiling_height_context_revision
                or levels_are_stale
            )
            if (
                self._is_shutdown
                or runtime.cancel_requested
                or thread.was_cancelled
                or is_stale
            ):
                stage = (
                    "Ignored after the video or project levels changed"
                    if is_stale
                    else "Cancelled"
                )
                self.job_manager.mark_cancelled(job_id, stage=stage)
                if not self._is_shutdown:
                    if is_stale and self._ceiling_height_dialog is not None:
                        self._ceiling_height_dialog.set_result(
                            "Analysis stopped because the video or project "
                            "levels changed."
                        )
                    elif not is_stale:
                        self.ceiling_height_result_label.setText(
                            "Ceiling height: Analysis cancelled."
                        )
                        if self._ceiling_height_dialog is not None:
                            self._ceiling_height_dialog.set_result(
                                "Ceiling height analysis cancelled."
                            )
                return
            if thread.error_message is not None or thread.result is None:
                message = thread.error_message or (
                    "Ceiling height analysis returned no result."
                )
                self.job_manager.fail_job(job_id, stage=f"Failed: {message}")
                self.ceiling_height_result_label.setText(
                    f"Ceiling height analysis failed: {message}"
                )
                if self._ceiling_height_dialog is not None:
                    self._ceiling_height_dialog.set_result(
                        message,
                        is_error=True,
                    )
                return

            analysis = thread.result
            if analysis.has_consensus:
                result_text = _format_level_height_result(analysis)
                self.ceiling_height_result_label.setText(result_text)
                if self._ceiling_height_dialog is not None:
                    self._ceiling_height_dialog.set_result(result_text)
                completion_stage = (
                    "Detected ceiling heights for "
                    f"{sum(item.has_consensus for item in analysis.level_estimates)} "
                    "level(s)"
                )
                self.job_manager.complete_job(job_id, stage=completion_stage)
                QTimer.singleShot(
                    0,
                    partial(
                        self._deliver_ceiling_height_result,
                        analysis,
                        runtime.context_revision,
                    ),
                )
            else:
                result_text = (
                    "No reliable ceiling-height result was found for any "
                    f"detected level. Analysed {analysis.analyzed_count} "
                    "frame intervals"
                    f"{_format_unassigned_frame_suffix(analysis)}."
                )
                self.ceiling_height_result_label.setText(result_text)
                if self._ceiling_height_dialog is not None:
                    self._ceiling_height_dialog.set_result(
                        result_text,
                        is_error=True,
                    )
                self.job_manager.complete_job(
                    job_id,
                    stage="No reliable ceiling-height consensus",
                )
        finally:
            if runtime is not None and runtime.thread is thread:
                self._ceiling_height_runtime = None
            if self._ceiling_height_dialog is not None:
                self._ceiling_height_dialog.set_running(False)
            self.sync_shared_controls()
            thread.deleteLater()

    # ### Ceiling-height analysis state ###
    def reset_ceiling_height_analysis(self) -> None:
        """Do not carry old-project marks or delayed results into a new project."""

        self._ceiling_height_context_revision += 1
        self._cancel_active_ceiling_height_inference()
        if self._ceiling_height_dialog is not None:
            self._ceiling_height_dialog.set_marked_ranges(())
        self.ceiling_height_result_label.setText(CEILING_HEIGHT_NOT_ESTIMATED_TEXT)

    def _deliver_ceiling_height_result(
        self, result: CeilingHeightAnalysisResult, context_revision: int
    ) -> None:
        if (
            not self._is_shutdown
            and context_revision == self._ceiling_height_context_revision
        ):
            self.ceiling_height_estimate_changed.emit(result)

    def set_ceiling_height_estimation_model(self, model: str) -> None:
        """Apply Settings to future analyses without changing a running job."""

        if model not in {value for _label, value in CEILING_HEIGHT_MODEL_OPTIONS}:
            raise ValueError(f"Unknown ceiling height estimation model: {model!r}.")
        self._ceiling_height_estimation_model = model
        if self._ceiling_height_dialog is not None:
            self._ceiling_height_dialog.set_model(model)

    def set_ceiling_height_levels(
        self,
        levels: Iterable[LevelData],
        current_level_index: int,
    ) -> None:
        """Publish only levels with readable plan images, not stale file paths."""

        previous_signature = (
            self._ceiling_height_level_images,
            self._ceiling_height_level_options,
            self._ceiling_height_level_plan_contexts,
            self._ceiling_height_level_plan_sources,
        )
        levels = tuple(
            level
            for level in levels
            if level.image_path and QImageReader(str(level.image_path)).canRead()
        )
        self._ceiling_height_level_images = tuple(
            sorted((int(level.index), str(level.image_path)) for level in levels)
        )
        options = tuple(
            sorted(
                (
                    (int(level.index), str(level.display_name))
                    for level in levels
                ),
                key=lambda item: item[0],
            )
        )
        self._ceiling_height_level_options = options
        self._ceiling_height_level_plan_contexts = tuple(
            sorted(
                (
                    (
                        int(level.index),
                        build_ceiling_height_plan_context(
                            level.display_name,
                            level.plan_metric_calibration,
                        ),
                    )
                    for level in levels
                ),
                key=lambda item: item[0],
            )
        )
        self._ceiling_height_level_plan_sources = tuple(
            _CeilingHeightPlanSource(
                level_index=int(level.index),
                image_path=str(level.image_path),
                calibration=level.plan_metric_calibration,
            )
            for level in levels
            if (
                level.plan_metric_calibration is not None
                and level.image_path is not None
            )
        )
        if previous_signature != (
            self._ceiling_height_level_images,
            self._ceiling_height_level_options,
            self._ceiling_height_level_plan_contexts,
            self._ceiling_height_level_plan_sources,
        ):
            self._ceiling_height_context_revision += 1
            self._cancel_active_ceiling_height_inference()
        self._ceiling_height_current_level_index = int(current_level_index)
        self._sync_ceiling_height_dialog_levels()

    def _current_video_source(self) -> VideoFrameSource | None:
        return (
            self.object_workspace._video_source
            or self.surface_workspace._video_source
        )

    def _current_video_frame_index(self) -> int | None:
        video_source = self._current_video_source()
        if video_source is None:
            return None
        displayed_index = self.object_workspace._displayed_frame_index
        if displayed_index is None:
            displayed_index = self.surface_workspace._displayed_frame_index
        if displayed_index is None:
            return None
        return min(
            max(int(displayed_index), 0),
            video_source.metadata.frame_count - 1,
        )

    def _sync_ceiling_height_analysis_source(self) -> None:
        """Adopt Generation's source without replacing a custom analysis video."""

        video_source = self._current_video_source()
        next_path = (
            "" if video_source is None else str(video_source.metadata.path)
        )
        previous_path = self._ceiling_height_video_path
        self._ceiling_height_video_path = next_path
        dialog = self._ceiling_height_dialog
        if dialog is None or not next_path:
            return
        if dialog.video_path and not (
            next_path != previous_path and dialog.video_path == previous_path
        ):
            return
        try:
            dialog.set_video_path(
                next_path,
                current_frame=self._current_video_frame_index() or 0,
            )
        except (OSError, ValueError) as error:
            dialog.set_result(str(error), is_error=True)

    def _handle_ceiling_height_analysis_input_changed(self, *_arguments: object) -> None:
        """Cancel stale work if its own source or explicit level ranges change."""

        dialog = self._ceiling_height_dialog
        runtime = self._ceiling_height_runtime
        if dialog is None:
            return
        self._ceiling_height_context_revision += 1
        self.ceiling_height_result_label.setText(CEILING_HEIGHT_NOT_ESTIMATED_TEXT)
        if runtime is not None and (
            dialog.video_path != runtime.video_path
            or dialog.marked_ranges != runtime.marked_ranges
        ):
            runtime.cancel_requested = True
            if not self.job_manager.cancel_job(runtime.job_id):
                runtime.thread.requestInterruption()

    def _sync_ceiling_height_dialog_levels(self) -> None:
        if self._ceiling_height_dialog is None:
            return
        self._ceiling_height_dialog.set_level_options(
            tuple(
                (level_name, level_index)
                for level_index, level_name in reversed(
                    self._ceiling_height_level_options
                )
            ),
            current_level_index=self._ceiling_height_current_level_index,
        )
        self._ceiling_height_dialog.set_plan_reference_size(
            _largest_ceiling_height_plan_reference_size(
                self._ceiling_height_level_plan_sources
            )
        )

    # ### Temporary generation references ###
    def _remember_temporary_object_reference(
        self,
        reference_bgra: np.ndarray,
        *,
        source_label: str,
    ) -> _TemporaryObjectReferenceSession:
        """Start an external-reference session bound to the current object."""

        reference = np.ascontiguousarray(reference_bgra).copy()
        objects = self.object_workspace
        session = _TemporaryObjectReferenceSession(
            active_bgra=reference,
            source_label=str(source_label).strip() or "Temporary image",
            generation_data_identity=id(objects._data),
            video_source=objects._video_source,
            frame_index=(
                None
                if objects._video_source is None
                else objects._displayed_frame_index
            ),
            inpaint_key=objects.current_inpaint_reference_key(),
        )
        self._temporary_object_reference_session = session
        return session

    def _clear_temporary_object_reference_session(self) -> None:
        """Forget retry input after the user leaves its owning frame or source."""

        self._temporary_object_reference_session = None

    def _temporary_reference_session_matches_context(
        self,
        session: _TemporaryObjectReferenceSession,
    ) -> bool:
        """Return whether a cached external image still belongs to this blob."""

        objects = self.object_workspace
        return bool(
            session.generation_data_identity == id(objects._data)
            and session.video_source is objects._video_source
            and session.inpaint_key == objects.current_inpaint_reference_key()
            and (
                session.video_source is None
                or session.frame_index == objects._displayed_frame_index
            )
        )

    def _adopt_untracked_temporary_reference(
        self,
    ) -> _TemporaryObjectReferenceSession | None:
        """Mirror an externally assigned temporary reference into shared state."""

        reference = self.object_workspace.get_temporary_object_reference()
        if reference is None:
            return None
        session = self._temporary_object_reference_session
        if (
            session is None
            or not self._temporary_reference_session_matches_context(session)
            or not np.array_equal(reference, session.active_bgra)
        ):
            session = self._remember_temporary_object_reference(
                reference,
                source_label="Temporary image",
            )
        return session

    def _restore_temporary_reference_after_same_frame_refresh(self) -> None:
        """Repair shared backend state after a non-navigating frame refresh."""

        session = self._temporary_object_reference_session
        if session is None:
            return
        if not self._temporary_reference_session_matches_context(session):
            self._clear_temporary_object_reference_session()
            return
        if not self.object_workspace.has_temporary_object_reference():
            self.object_workspace.set_temporary_object_reference(
                session.active_bgra
            )
        surface_reference = self.surface_workspace.get_temporary_reference()
        if (
            surface_reference is None
            or not np.array_equal(surface_reference, session.active_bgra)
        ):
            self.surface_workspace.set_temporary_reference(
                session.active_bgra
            )

    def _sync_object_reference_preview(self) -> None:
        """Render the authoritative override, or restore the ordinary frame."""

        if self.object_workspace.get_landing_editing_target() is not None:
            references = self.object_workspace._landing_references
            source = self.object_workspace.get_current_object_generation_reference()
            if references and source is not None:
                from housemaker.landing_generation import landing_reference_signature

                references_are_current = (
                    self.object_workspace._landing_reference_signature
                    == landing_reference_signature(source)
                )
            else:
                references_are_current = False
            if references_are_current:
                selected_id = self.reference_generation_view_combo.currentData()
                reference = next((item for item in references if item.view_id == selected_id), references[0])
                self.video_view.set_reference_preview_bgra(reference.image_bgra)
                return
        temporary_reference = (
            self.object_workspace.get_temporary_object_reference()
        )
        if temporary_reference is not None:
            self._adopt_untracked_temporary_reference()
            self.video_view.set_reference_preview_bgra(temporary_reference)
            return
        self.video_view.clear_reference_preview()

    @Slot()
    def _handle_load_video_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load generation reference video",
            str(Path.cwd()),
            VIDEO_FILE_FILTER,
        )
        if not file_path:
            return
        try:
            probe_video(file_path)
            self._invalidate_reference_generation_input()
            self._clear_temporary_object_reference_session()
            self.object_workspace.load_video(file_path)
            self.surface_workspace.load_video(file_path)
            self.sync_shared_controls()
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.critical(self, "Video load failed", str(error))

    @Slot()
    def _handle_copy_inpaint_clicked(self) -> None:
        """Copy the exact current Object reference with its alpha mask."""

        reference = self.object_workspace.get_current_object_generation_reference()
        if reference is None:
            QMessageBox.information(
                self,
                "Nothing to copy",
                "Paint an object mask or load a reference image first.",
            )
            return

        image = _bgra_array_to_qimage(reference)
        mime_data = QMimeData()
        mime_data.setImageData(image)
        encoded_png = _encode_qimage_png(image)
        if encoded_png:
            mime_data.setData("image/png", encoded_png)
        QApplication.clipboard().setMimeData(mime_data)
        self.object_workspace.status_label.setText(
            "Object reference copied to the clipboard."
        )
        self.sync_shared_controls()

    @Slot()
    def _handle_paste_inpaint_clicked(self) -> None:
        """Use the clipboard image as a shared generation reference."""

        image = _read_clipboard_image()
        if image is None:
            QMessageBox.warning(
                self,
                "Paste",
                "The clipboard does not contain a usable image.",
            )
            return
        self._activate_temporary_object_reference(
            _qimage_to_bgra_array(image),
            source_label="Clipboard image",
        )

    @Slot()
    def _handle_load_reference_image_clicked(self) -> None:
        """Load a shared temporary reference without replacing the video."""

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load generation reference image",
            str(Path.cwd()),
            REFERENCE_IMAGE_FILE_FILTER,
        )
        if not file_path:
            return
        image = QImage(file_path)
        if image.isNull():
            QMessageBox.critical(
                self,
                "Image load failed",
                "The selected file could not be decoded as an image.",
            )
            return
        self._activate_temporary_object_reference(
            _qimage_to_bgra_array(image),
            source_label="Loaded image",
        )

    def _activate_temporary_object_reference(
        self,
        reference_bgra: np.ndarray,
        *,
        source_label: str,
        generated: bool = False,
    ) -> None:
        """Display and bind one temporary reference to both generators."""

        if not generated:
            self._invalidate_reference_generation_input()
        try:
            objects = self.object_workspace
            objects.set_temporary_object_reference(reference_bgra)
            self.surface_workspace.set_temporary_reference(reference_bgra)
        except ValueError as error:
            self.object_workspace.clear_temporary_object_reference()
            self.surface_workspace.clear_temporary_reference()
            QMessageBox.critical(self, "Reference image failed", str(error))
            return
        active_reference = (
            self.object_workspace.get_temporary_object_reference()
        )
        if active_reference is None:
            return
        self._remember_temporary_object_reference(
            active_reference,
            source_label=source_label,
        )
        self._sync_object_reference_preview()
        message = (
            f"{source_label} is active for Object and Surface generation. Move the "
            "video seekbar to return to the video; this inpaint keeps its reference."
            if self.object_workspace._video_source is not None
            else f"{source_label} is active for Object and Surface generation."
        )
        self.object_workspace.status_label.setText(message)
        self.sync_shared_controls()

    @Slot(int)
    def _handle_seekbar_changed(self, frame_index: int) -> None:
        """Seek once without letting one backend overwrite the other's old frame."""

        surface = self.surface_workspace
        objects = self.object_workspace
        if objects._is_syncing_seekbar or surface._is_syncing_seekbar:
            return
        video_source = objects._video_source or surface._video_source
        if video_source is None:
            return

        current_strokes = self.video_view.get_strokes()
        for workspace in (objects, surface):
            displayed_index = workspace._displayed_frame_index
            if displayed_index is None or workspace._video_source is None:
                continue
            workspace._data.set_frame_strokes(displayed_index, current_strokes)

        safe_index = min(
            max(int(frame_index), 0),
            max(video_source.metadata.frame_count - 1, 0),
        )
        try:
            frame_bgr = video_source.get_frame(safe_index)
        except (IndexError, ValueError) as error:
            objects.status_label.setText(str(error))
            surface.status_label.setText(str(error))
            return
        target_strokes = objects._data.strokes_for_frame(safe_index)
        if not target_strokes:
            target_strokes = surface._data.strokes_for_frame(safe_index)
        current_object_frame_index = objects._displayed_frame_index
        object_frame_is_changing = (
            current_object_frame_index is None
            or int(current_object_frame_index) != safe_index
        )
        if object_frame_is_changing:
            self._invalidate_reference_generation_input()
            objects.clear_door_slot_editing_target()
            self._clear_temporary_object_reference_session()
            objects.clear_temporary_object_reference_for_video_navigation()
            surface.clear_temporary_reference()
        for workspace in (objects, surface):
            workspace._data.current_frame_index = safe_index
            workspace._displayed_frame_index = safe_index
        self.video_view.set_frame(frame_bgr, target_strokes)
        surface._sync_painted_frames_label()
        objects._sync_controls()
        surface._sync_controls()
        self.sync_shared_controls()

    @Slot(object)
    def _handle_job_changed(self, _job: object) -> None:
        self._sync_generation_action_buttons()

    @Slot()
    def sync_shared_controls(self) -> None:
        """Own availability for widgets consumed by both pipelines."""

        objects = self.object_workspace
        surface = self.surface_workspace
        has_video = bool(
            objects._video_source is not None or surface._video_source is not None
        )
        has_untracked_object_job = bool(
            objects._generation_thread is not None and not objects._object_job_runtimes
        )
        shared_reference_controls_are_available = not has_untracked_object_job
        video_editor_is_available = (
            has_video and shared_reference_controls_are_available
        )
        temporary_reference_is_available = (
            objects.has_temporary_object_reference()
        )
        inpainting_is_available = (
            video_editor_is_available
            and not temporary_reference_is_available
        )
        reference_media_changes_are_available = not has_untracked_object_job
        self.load_video_button.setEnabled(reference_media_changes_are_available)
        self.load_reference_image_button.setEnabled(
            reference_media_changes_are_available
        )
        self.paste_inpaint_button.setEnabled(
            reference_media_changes_are_available
            and _clipboard_has_usable_image()
        )
        self.copy_inpaint_button.setEnabled(
            objects.has_temporary_object_reference()
            or self.video_view.has_selection()
        )
        self.seekbar.setEnabled(video_editor_is_available)
        self.paint_mask_button.setEnabled(inpainting_is_available)
        self.erase_mask_button.setEnabled(inpainting_is_available)
        self.brush_size_spinbox.setEnabled(inpainting_is_available)
        self.clear_mask_button.setEnabled(
            video_editor_is_available and self.video_view.has_selection()
        )
        self.clear_mask_button.setToolTip(
            "Remove only the currently selected red inpaint object."
        )
        inpaint_index, inpaint_count = objects.current_inpaint_position()
        opening_target = objects._door_slot_editing_target
        trim_target = objects._architectural_trim_editing_target
        navigation_enabled = bool(
            video_editor_is_available
            and inpaint_count
            and not (opening_target is not None and opening_target.generation_ready)
            and not (trim_target is not None and trim_target.generation_ready)
        )
        self.previous_inpaint_button.setEnabled(navigation_enabled)
        self.next_inpaint_button.setEnabled(navigation_enabled)
        self.inpaint_position_label.setText(
            (f"Inpaint {inpaint_index + 1} of {inpaint_count}"
             if inpaint_index is not None else f"{inpaint_count} inpaints queued")
            if inpaint_count else "No inpaints"
        )
        self.pbr_map_control.setEnabled(not has_untracked_object_job)
        self.ai_prompt_edit.setEnabled(not has_untracked_object_job)
        self.video_view.set_interaction_enabled(inpainting_is_available)
        self.infer_ceiling_height_button.setEnabled(
            not self._is_shutdown
        )
        self.infer_ceiling_height_button.setText("Infer ceiling height")
        self._sync_ceiling_height_analysis_source()
        self._sync_opening_generation_sections()
        self._sync_generation_texture_context()
        self._sync_reference_generation_controls()


# ### Reference image helpers ###
def _bgra_array_to_qimage(image_bgra: np.ndarray) -> QImage:
    """Build an owned Qt image while preserving BGRA alpha exactly."""

    image = np.asarray(image_bgra)
    if image.ndim != 3 or image.shape[2] != 4 or image.size == 0:
        return QImage()
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    rgba = np.ascontiguousarray(image[:, :, (2, 1, 0, 3)])
    height, width = rgba.shape[:2]
    return QImage(
        rgba.data,
        width,
        height,
        int(rgba.strides[0]),
        QImage.Format.Format_RGBA8888,
    ).copy()


def _qimage_to_bgra_array(image: QImage) -> np.ndarray:
    """Normalize any decoded Qt image to an owned contiguous BGRA array."""

    if image.isNull() or image.width() <= 0 or image.height() <= 0:
        raise ValueError("The reference image is empty.")
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    height = converted.height()
    width = converted.width()
    bytes_per_line = converted.bytesPerLine()
    rgba_rows = np.frombuffer(
        converted.constBits(),
        dtype=np.uint8,
        count=height * bytes_per_line,
    ).reshape(height, bytes_per_line)
    rgba = rgba_rows[:, : width * 4].reshape(height, width, 4)
    return np.ascontiguousarray(rgba[:, :, (2, 1, 0, 3)])


def _encode_qimage_png(image: QImage) -> QByteArray:
    """Encode a clipboard image explicitly so external apps retain alpha."""

    encoded = QByteArray()
    buffer = QBuffer(encoded)
    if not buffer.open(QIODevice.OpenModeFlag.WriteOnly):
        return QByteArray()
    try:
        if not image.save(buffer, "PNG"):
            return QByteArray()
    finally:
        buffer.close()
    return encoded


def _read_clipboard_image() -> QImage | None:
    """Prefer lossless PNG clipboard data, then use Qt's image fallback."""

    clipboard = QApplication.clipboard()
    mime_data = clipboard.mimeData()
    if mime_data is not None and mime_data.hasFormat("image/png"):
        image = QImage.fromData(mime_data.data("image/png"), "PNG")
        if not image.isNull():
            return image
    image = clipboard.image()
    return None if image.isNull() else image


def _clipboard_has_usable_image() -> bool:
    """Return whether Paste can resolve an image right now."""

    clipboard = QApplication.clipboard()
    mime_data = clipboard.mimeData()
    return bool(
        (
            mime_data is not None
            and (
                mime_data.hasImage()
                or mime_data.hasFormat("image/png")
            )
        )
        or not clipboard.image().isNull()
    )


# ### Widget helpers ###
def _build_labeled_view(title: str, view: QWidget) -> QWidget:
    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    title_label = QLabel(title)
    title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title_label.setWordWrap(True)
    title_label.setStyleSheet("font-weight: 600;")
    layout.addWidget(title_label)
    layout.addWidget(view, 1)
    return wrapper


def _build_controls_section(kind: str) -> tuple[QWidget, QGridLayout]:
    section = QWidget()
    section.setObjectName(f"merged_generation_{kind}_controls")
    outer_layout = QVBoxLayout(section)
    outer_layout.setContentsMargins(0, 0, 0, 0)
    outer_layout.setSpacing(5)

    fields = QWidget()
    fields.setObjectName(f"merged_generation_{kind}_fields")
    fields_layout = QGridLayout(fields)
    fields_layout.setContentsMargins(0, 0, 0, 0)
    fields_layout.setHorizontalSpacing(8)
    fields_layout.setVerticalSpacing(5)
    outer_layout.addWidget(fields, 1)
    return section, fields_layout


def _build_controls_column(object_name: str) -> tuple[QWidget, QVBoxLayout]:
    column = QWidget()
    column.setObjectName(object_name)
    layout = QVBoxLayout(column)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    return column, layout


def _build_boxed_section(
    title: str,
    object_name: str,
) -> tuple[QGroupBox, QVBoxLayout]:
    section = QGroupBox(title)
    section.setObjectName(object_name)
    layout = QVBoxLayout(section)
    layout.setContentsMargins(7, 7, 7, 7)
    layout.setSpacing(5)
    return section, layout


def _build_labeled_inline_control(label: str, control: QWidget) -> QWidget:
    wrapper = QWidget()
    layout = QHBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    layout.addWidget(QLabel(label))
    layout.addWidget(control, 1)
    return wrapper


def _unique_widgets(widgets: Iterable[QWidget]) -> tuple[QWidget, ...]:
    unique: list[QWidget] = []
    seen_ids: set[int] = set()
    for widget in widgets:
        widget_id = id(widget)
        if widget_id in seen_ids:
            continue
        seen_ids.add(widget_id)
        unique.append(widget)
    return tuple(unique)


# ### Legacy project migration helpers ###
def _select_richest_video_state(
    object_data: GenerationData,
    surface_data: SurfaceTextureData,
) -> tuple[VideoMetadata, int, dict[int, list[MaskStroke]]] | None:
    candidates = []
    for preference, data in enumerate((surface_data, object_data)):
        metadata = data.video_metadata
        if metadata is None:
            continue
        stroke_count = sum(len(strokes) for strokes in data.frame_strokes.values())
        candidates.append(
            (
                len(data.frame_strokes),
                stroke_count,
                preference,
                metadata,
                data.current_frame_index,
                data.frame_strokes,
            )
        )
    if not candidates:
        return None
    _, _, _, metadata, frame_index, frame_strokes = max(
        candidates,
        key=lambda value: value[:3],
    )
    return metadata, int(frame_index), frame_strokes


def _merge_legacy_frame_strokes(
    primary: dict[int, list[MaskStroke]],
    *sources: dict[int, list[MaskStroke]],
) -> dict[int, list[MaskStroke]]:
    """Preserve disjoint legacy masks when both tabs used the same video."""

    merged: dict[int, list[MaskStroke]] = {}
    frame_indices = sorted(
        {frame_index for source in (primary, *sources) for frame_index in source}
    )
    for frame_index in frame_indices:
        strokes = [copy.deepcopy(stroke) for stroke in primary.get(frame_index, ())]
        retained_counts = Counter(strokes)
        for source in sources:
            source_counts: Counter[MaskStroke] = Counter()
            for stroke in source.get(frame_index, ()):
                source_counts[stroke] += 1
                if source_counts[stroke] <= retained_counts[stroke]:
                    continue
                retained_counts[stroke] += 1
                strokes.append(copy.deepcopy(stroke))
                if len(strokes) >= MAX_MASK_STROKES_PER_FRAME:
                    break
            if len(strokes) >= MAX_MASK_STROKES_PER_FRAME:
                break
        if strokes:
            merged[frame_index] = strokes
    return merged
