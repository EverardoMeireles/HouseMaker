"""Render local references with one resident model, then infer their symmetry."""

# ### Imports ###
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QThread, Signal

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
    def _render_references(self) -> dict[str, np.ndarray]:
        from housemaker.reference_generation import open_object_reference_session

        generated = {}
        with open_object_reference_session(
            self.model, cancellation_check=self.isInterruptionRequested,
        ) as generate:
            for item in self.items:
                job_id = item.job_id
                if self._is_cancelled(job_id):
                    self._finish(job_id, cancelled=True)
                    continue
                self.item_progress.emit(job_id, "Rendering references with the resident local model", 0)
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
