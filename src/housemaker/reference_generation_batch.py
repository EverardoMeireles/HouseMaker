"""Generate grouped references with resident local models or OpenAI Batch."""

# ### Imports ###
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import numpy as np
from PySide6.QtCore import QThread, Signal

from housemaker.openai_batch import (
    LUNA_BATCH_MODEL,
    run_responses_batch,
)
from housemaker.reference_generation_models import DEFAULT_REFERENCE_NEGATIVE_PROMPT


# ### Batch inputs ###
@dataclass(frozen=True)
class LocalReferenceBatchItem:
    job_id: str
    source_bgra: np.ndarray
    prompt: str | None = None


# ### Phased local worker ###
class LocalReferenceBatchThread(QThread):
    """Own all resident processes on a worker thread, never on the GUI thread."""

    item_progress = Signal(str, str, int)
    reference_ready = Signal(str, object)
    item_finished = Signal(str, object, str, bool)

    def __init__(
        self, items, model, symmetry_model, *, api_key, prompt, strength,
        negative_prompt=DEFAULT_REFERENCE_NEGATIVE_PROMPT, parent=None,
    ) -> None:
        super().__init__(parent)
        self.items = tuple(items)
        self.model = model
        self.symmetry_model = symmetry_model
        self._api_key = api_key
        self._prompt = prompt
        self._strength = strength
        self._negative_prompt = negative_prompt
        self._cancelled = {item.job_id: threading.Event() for item in self.items}
        self._finished_items: set[str] = set()
        self.error_message: str | None = None

    def cancel_item(self, job_id: str) -> None:
        event = self._cancelled.get(job_id)
        if event is not None:
            event.set()

    def _is_cancelled(self, job_id: str) -> bool:
        return self.isInterruptionRequested() or self._cancelled[job_id].is_set()

    def _error_text(self, error: Exception) -> str:
        message = str(error) or type(error).__name__
        return message.replace(self._api_key, "[redacted]") if self._api_key else message

    def _finish(self, job_id, result=None, error="", cancelled=False) -> None:
        if job_id not in self._finished_items:
            self._finished_items.add(job_id)
            self.item_finished.emit(job_id, result, error, cancelled)

    def run(self) -> None:  # type: ignore[override]
        """Close the image model before opening any local symmetry model."""

        try:
            generated = self._render_references()
            if generated and not self.isInterruptionRequested():
                if self.symmetry_model is None:
                    for job_id in generated:
                        self._finish(job_id, cancelled=self._is_cancelled(job_id))
                else:
                    self._infer_symmetry(generated)
        except Exception as error:  # noqa: BLE001 - Report setup/cleanup failures per remaining job.
            self.error_message = self._error_text(error)
        finally:
            for item in self.items:
                self._finish(
                    item.job_id, error=self.error_message or "Local reference processing stopped.",
                    cancelled=self._is_cancelled(item.job_id),
                )
            self._api_key = ""

    # ### Image-generation phase ###
    def _reference_session(self):
        from housemaker.reference_generation import open_object_reference_session

        return open_object_reference_session(
            self.model, cancellation_check=self.isInterruptionRequested,
        )

    def _render_references(self) -> dict[str, np.ndarray]:
        generated = {}
        with self._reference_session() as generate:
            for item in self.items:
                job_id = item.job_id
                if self._is_cancelled(job_id):
                    self._finish(job_id, cancelled=True)
                    continue
                self.item_progress.emit(job_id, "Generating reference image", 0)
                try:
                    result = generate(
                        item.source_bgra, item.prompt if item.prompt is not None else self._prompt, strength=self._strength,
                        negative_prompt=self._negative_prompt,
                        cancellation_check=lambda key=job_id: self._is_cancelled(key),
                        progress_callback=lambda message, percent, key=job_id: self.item_progress.emit(
                            key, message, min(85, round(percent * 0.85)),
                        ),
                    )
                    if self._is_cancelled(job_id):
                        self._finish(job_id, cancelled=True)
                        continue
                    generated[job_id] = result
                    self.reference_ready.emit(job_id, result)
                    stage = "Reference ready; queued for symmetry inference" if self.symmetry_model is not None else "Reference ready"
                    self.item_progress.emit(job_id, stage, 85)
                except Exception as error:  # noqa: BLE001 - One failed image must not drop other objects.
                    self._finish(job_id, error=self._error_text(error), cancelled=self._is_cancelled(job_id))
        return generated

    # ### Symmetry-inference phase ###
    def _infer_symmetry(self, generated: dict[str, np.ndarray]) -> None:
        if self.symmetry_model == LUNA_BATCH_MODEL:
            self._infer_batch_symmetry(generated)
            return
        from housemaker.reference_symmetry_inference import (
            open_reference_symmetry_session,
        )

        with open_reference_symmetry_session(
            self.symmetry_model, api_key=self._api_key,
            cancellation_check=self.isInterruptionRequested,
        ) as session:
            for job_id in tuple(generated):
                image = generated.pop(job_id)
                if self._is_cancelled(job_id):
                    self._finish(job_id, cancelled=True)
                    continue
                self.item_progress.emit(job_id, "Inferring symmetry with the shared model session", 90)
                try:
                    result = session.infer(
                        image, cancellation_check=lambda key=job_id: self._is_cancelled(key),
                        progress_callback=lambda message, key=job_id: self.item_progress.emit(key, message, 95),
                    )
                    self._finish(job_id, result=result, cancelled=self._is_cancelled(job_id))
                except Exception as error:  # noqa: BLE001 - Keep accepted references when inference fails.
                    self._finish(
                        job_id, error=f"Symmetry inference failed: {self._error_text(error)}",
                        cancelled=self._is_cancelled(job_id),
                    )

    # ### Grouped remote symmetry inference ###
    def _infer_batch_symmetry(self, generated: dict[str, np.ndarray]) -> None:
        from housemaker.reference_symmetry_inference import (
            SymmetryInferenceError,
            infer_reference_symmetry_batch,
        )

        pending = {
            job_id: image for job_id, image in generated.items()
            if not self._is_cancelled(job_id)
        }
        generated.clear()
        if not pending:
            return

        def progress(stage: str) -> None:
            for job_id in pending:
                if not self._is_cancelled(job_id):
                    self.item_progress.emit(job_id, stage, 90)

        results = infer_reference_symmetry_batch(
            pending, api_key=self._api_key,
            cancellation_check=lambda: all(self._is_cancelled(key) for key in pending),
            progress_callback=progress,
        )
        for job_id in pending:
            result = results.get(job_id)
            error = ""
            if isinstance(result, SymmetryInferenceError):
                error = f"Symmetry inference failed: {self._error_text(result)}"
                result = None
            elif result is None:
                error = "Symmetry inference failed: No Batch result was returned."
            self._finish(
                job_id, result=result, error=error,
                cancelled=self._is_cancelled(job_id),
            )


# ### Standard API images followed by grouped remote symmetry ###
class APIReferenceBatchThread(LocalReferenceBatchThread):
    """Finish standard image calls before submitting one symmetry Batch job."""

    @contextmanager
    def _reference_session(self):
        from housemaker.reference_generation import generate_object_reference

        yield partial(
            generate_object_reference, model=self.model, api_key=self._api_key,
        )


# ### Grouped OpenAI image-reference generation ###
class OpenAIReferenceBatchThread(LocalReferenceBatchThread):
    """Submit all independent object images in one asynchronous OpenAI batch.

    An individual cancellation discards that object's result locally. OpenAI
    cancels only whole batches, so submitted individual requests may still be
    processed and billed. Cancel all also requests remote batch cancellation.
    """

    def _render_references(self) -> dict[str, np.ndarray]:
        from housemaker.plan_image_correction import (
            _decode_openai_image_generation_result,
            build_openai_responses_image_edit_body,
        )
        from housemaker.reference_generation import (
            _decode_generated_reference,
            build_reference_generation_prompt,
            prepare_reference_generation_input,
        )

        requests = {}
        for item in self.items:
            job_id = item.job_id
            if self._is_cancelled(job_id):
                self._finish(job_id, cancelled=True)
                continue
            try:
                prompt = build_reference_generation_prompt(
                    item.prompt if item.prompt is not None else self._prompt,
                    model=self.model,
                )
                requests[job_id] = build_openai_responses_image_edit_body(
                    prepare_reference_generation_input(item.source_bgra),
                    model=self.model, prompt=prompt,
                    action="generate", background="opaque",
                )
            except Exception as error:  # noqa: BLE001 - Isolate invalid object inputs.
                self._finish(job_id, error=self._error_text(error))
        if not requests:
            return {}

        def progress(stage: str) -> None:
            for job_id in requests:
                if not self._is_cancelled(job_id):
                    self.item_progress.emit(job_id, stage, 25)

        outcomes = run_responses_batch(
            requests, api_key=self._api_key,
            cancellation_check=lambda: all(self._is_cancelled(key) for key in requests),
            progress_callback=progress,
        )
        generated = {}
        for job_id in requests:
            if self._is_cancelled(job_id):
                self._finish(job_id, cancelled=True)
                continue
            outcome = outcomes.get(job_id)
            if outcome is None or outcome.error or outcome.body is None:
                self._finish(
                    job_id,
                    error=(
                        "OpenAI Batch reference generation failed: "
                        + ("No result was returned." if outcome is None else outcome.error or "Empty response.")
                    ),
                )
                continue
            try:
                image = _decode_generated_reference(
                    _decode_openai_image_generation_result(
                        outcome.body, secrets=(self._api_key,),
                    )
                )
                generated[job_id] = image
                self.reference_ready.emit(job_id, image)
                self.item_progress.emit(job_id, "Batch reference ready", 85)
            except Exception as error:  # noqa: BLE001 - Keep other successful images.
                self._finish(job_id, error=self._error_text(error))
        return generated
