"""Generate landing references and Meshy geometry without object-library records."""

# ### Imports ###
from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal

from housemaker.glb import GeneratedModel
from housemaker.reference_generation_models import (
    DEFAULT_REFERENCE_NEGATIVE_PROMPT,
    REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
)
from housemaker.reference_view_presets import ReferenceViewPreset


# ### Generation result models ###
class LandingGenerationCancelled(RuntimeError):
    """A cancelled landing job must not apply its late result."""


@dataclass(frozen=True)
class LandingViewReference:
    view_id: str
    view_name: str
    image_bgra: np.ndarray


@dataclass(frozen=True)
class LandingMeshGenerationResult:
    """A provider Y-up source, applied only to its original structural landing."""

    model: GeneratedModel
    task_id: str
    geometry_only: bool = False
    texture_only: bool = False


# ### Pure input validation ###
def landing_reference_signature(image: np.ndarray) -> tuple[object, ...]:
    pixels = _reference_pixels(image)
    return pixels.shape, hashlib.sha256(pixels.tobytes()).hexdigest()


def _reference_pixels(image: np.ndarray) -> np.ndarray:
    pixels = np.asarray(image)
    if (
        pixels.dtype != np.uint8
        or pixels.ndim != 3
        or pixels.shape[2] != 4
        or not pixels.size
        or not np.any(pixels[:, :, 3])
    ):
        raise ValueError("A landing reference must be a visible BGRA image.")
    return np.ascontiguousarray(pixels)


# ### Reference rendering with resident local models ###
@contextmanager
def _reference_session(model, api_key, cancellation_check):
    from housemaker.plan_correction_models import PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1
    from housemaker.reference_generation import (
        generate_object_reference,
        open_object_reference_session,
    )

    if model in (
        PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
        REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
    ):
        with open_object_reference_session(
            model, cancellation_check=cancellation_check
        ) as generate:
            yield partial(generate, api_key=api_key)
    else:
        yield partial(generate_object_reference, model=model, api_key=api_key)


def _raise_if_cancelled(check):
    if check is not None and check():
        raise LandingGenerationCancelled("Landing generation cancelled.")


# ### Existing OpenAI Batch transport ###
def _generate_batch_references(thread) -> tuple[LandingViewReference, ...]:
    """Submit the selected view through the existing Batch transport."""

    from housemaker.openai_batch import run_responses_batch
    from housemaker.plan_image_correction import (
        _decode_openai_image_generation_result,
        build_openai_responses_image_edit_body,
    )
    from housemaker.reference_generation import (
        _decode_generated_reference,
        build_reference_generation_prompt,
        prepare_reference_generation_input,
    )

    prepared = prepare_reference_generation_input(thread.source_bgra)
    requests = {}
    for view in thread.views:
        _raise_if_cancelled(thread.isInterruptionRequested)
        prompt = view.prompt + (", " + thread.prompt if thread.prompt.strip() else "")
        requests[view.preset_id] = build_openai_responses_image_edit_body(
            prepared,
            model=thread.model,
            prompt=build_reference_generation_prompt(prompt, model=thread.model),
            action="generate",
            background="opaque",
        )
    outcomes = run_responses_batch(
        requests,
        api_key=thread._api_key,
        cancellation_check=thread.isInterruptionRequested,
        progress_callback=lambda stage: thread.progress.emit(stage, 25),
    )
    results = []
    for view in thread.views:
        _raise_if_cancelled(thread.isInterruptionRequested)
        outcome = outcomes.get(view.preset_id)
        if outcome is None or outcome.error or outcome.body is None:
            raise ValueError(
                "OpenAI Batch did not return the selected landing reference."
            )
        image = _decode_generated_reference(
            _decode_openai_image_generation_result(
                outcome.body, secrets=(thread._api_key,)
            )
        )
        results.append(
            LandingViewReference(
                view.preset_id, view.name, _reference_pixels(image).copy()
            )
        )
    return tuple(results)


# ### Cancellable background job ###
class LandingGenerationThread(QThread):
    """Own reference rendering or the shared Meshy workers away from the GUI."""

    progress = Signal(str, int)

    def __init__(
        self,
        landing,
        source_bgra,
        *,
        action,
        views=(),
        references=(),
        model="",
        api_key="",
        prompt="",
        strength=0.45,
        negative_prompt=DEFAULT_REFERENCE_NEGATIVE_PROMPT,
        mesh_request=None,
        planner=None,
        executor=None,
        texture_regenerator=None,
        reference_session=None,
        parent=None,
    ):
        super().__init__(parent)
        self.landing = landing
        self.source_bgra = _reference_pixels(source_bgra).copy()
        self.source_signature = landing_reference_signature(self.source_bgra)
        self.action = str(action)
        self.views = tuple(views)
        self.references = tuple(references)
        self.model = model
        self._api_key = api_key
        self.prompt = str(prompt)
        self.strength = strength
        self.negative_prompt = negative_prompt
        self.mesh_request = mesh_request
        self._planner = planner
        self._executor = executor
        self._texture_regenerator = texture_regenerator
        self._mesh_worker = None
        self._mesh_progress = 5
        self._reference_session = reference_session or _reference_session
        self.result = None
        self.error_message = None
        self.was_cancelled = False

    # ### Shared Meshy execution and cancellation ###
    def requestInterruption(self):
        super().requestInterruption()
        worker = self._mesh_worker
        if worker is not None:
            worker.cancel()

    def _forward_mesh_progress(self, stage):
        percentages = re.findall(r"(?<!\d)(100|[1-9]?\d)\s*%", stage)
        if percentages:
            self._mesh_progress = max(self._mesh_progress, int(percentages[-1]))
        self.progress.emit(stage, self._mesh_progress)

    def _generate_mesh(self):
        from housemaker.generation_workspace import (
            GenerationWorker,
            TextureRegenerationWorker,
        )

        request = self.mesh_request
        if request is None:
            raise ValueError("The landing Meshy request is missing.")
        if self.action == "texture":
            from housemaker.uv_integrity import UvIntegrityError, build_uv_fingerprint

            try:
                fingerprint = build_uv_fingerprint(request.model_glb)
            except UvIntegrityError:
                fingerprint = None
            if fingerprint is not None:
                request = replace(
                    request,
                    enable_original_uv=True,
                    submitted_uv_fingerprint=fingerprint,
                )
            worker = TextureRegenerationWorker(
                self._texture_regenerator, self._executor, request
            )
        else:
            worker = GenerationWorker(self._planner, self._executor, request)
        self._mesh_worker = worker
        outcomes = []
        failures = []
        worker.succeeded.connect(
            lambda payload, model: outcomes.append((payload, model)),
            Qt.ConnectionType.DirectConnection,
        )
        worker.failed.connect(failures.append, Qt.ConnectionType.DirectConnection)
        worker.progress.connect(
            self._forward_mesh_progress,
            Qt.ConnectionType.DirectConnection,
        )
        if self.isInterruptionRequested():
            worker.cancel()
        try:
            worker.run()
            _raise_if_cancelled(self.isInterruptionRequested)
            if failures:
                raise ValueError(failures[0])
            if not outcomes:
                raise ValueError("Meshy did not return a landing model.")
            payload, model = outcomes[0]
            result = getattr(payload, "result", payload)
            return LandingMeshGenerationResult(
                model=model,
                task_id=str(getattr(result, "task_id", "")),
                geometry_only=self.action == "geometry",
                texture_only=self.action == "texture",
            )
        finally:
            self._mesh_worker = None

    def run(self):
        try:
            _raise_if_cancelled(self.isInterruptionRequested)
            if self.action == "references":
                if len(self.views) != 1 or not isinstance(
                    self.views[0], ReferenceViewPreset
                ):
                    raise ValueError("Select exactly one landing reference view.")
                from housemaker.openai_batch import LUNA_BATCH_MODEL

                if self.model == LUNA_BATCH_MODEL:
                    self.result = _generate_batch_references(self)
                    _raise_if_cancelled(self.isInterruptionRequested)
                    return
                view = self.views[0]
                self.progress.emit(f"Landing reference: {view.name}", 0)
                with self._reference_session(
                    self.model, self._api_key, self.isInterruptionRequested
                ) as generate:
                    prompt = view.prompt + (
                        ", " + self.prompt if self.prompt.strip() else ""
                    )
                    image = generate(
                        self.source_bgra,
                        prompt,
                        strength=self.strength,
                        negative_prompt=self.negative_prompt,
                        cancellation_check=self.isInterruptionRequested,
                    )
                self.result = (
                    LandingViewReference(
                        view.preset_id, view.name, _reference_pixels(image).copy()
                    ),
                )
            elif self.action in ("geometry", "generate", "texture"):
                if not self.references:
                    raise ValueError("Generate the landing's selected reference first.")
                self.progress.emit("Generating landing with Meshy", 5)
                self.result = self._generate_mesh()
            else:
                raise ValueError("Unknown landing generation action.")
            _raise_if_cancelled(self.isInterruptionRequested)
        except Exception as error:  # noqa: BLE001 - Worker failures are presented safely in the UI.
            self.was_cancelled = self.isInterruptionRequested() or isinstance(
                error, LandingGenerationCancelled
            )
            message = str(error) or type(error).__name__
            self.error_message = (
                message.replace(self._api_key, "[redacted]")
                if self._api_key
                else message
            )
            self.result = None
        finally:
            self._api_key = ""
