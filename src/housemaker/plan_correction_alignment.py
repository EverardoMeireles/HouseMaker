# ### Imports ###
from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .canvas_plan_alignment import (
    CanvasPlanAlignmentCancelled,
    CanvasPlanAlignmentError,
    CanvasPlanAlignmentLevel,
    CanvasPlanAlignmentResult,
    align_canvas_plan_levels,
)

# ### Coordinator constants ###
_WAIT_POLL_SECONDS = 0.05


# ### Public data models ###
@dataclass(frozen=True, slots=True)
class PlanCorrectionAlignmentLevel:
    """One level's pre-correction Canvas transform."""

    level_index: int
    canvas_level_scale: float
    canvas_offset_x_pixels: float
    canvas_offset_y_pixels: float


@dataclass(frozen=True, slots=True)
class PlanCorrectionAlignmentOutcome:
    """Best-effort registration result shared by every correction worker."""

    result: CanvasPlanAlignmentResult | None
    warning: str = ""


@dataclass(frozen=True, slots=True)
class _PlanCorrectionAlignmentSubmission:
    """One final corrected raster waiting at the batch barrier."""

    level_index: int
    image_path: Path


# ### Public coordinator ###
class PlanCorrectionAlignmentCoordinator:
    """Run structural plan registration once every corrected PNG is saved.

    Correction workers meet at this barrier after their final image has been
    written.  The final worker performs the OpenCV registration, so expensive
    visual analysis never blocks the Qt GUI thread.
    """

    def __init__(
        self,
        levels: Sequence[PlanCorrectionAlignmentLevel],
        *,
        anchor_level_index: int,
        viewport_size: tuple[float, float],
    ) -> None:
        normalized_levels = tuple(levels)
        if len(normalized_levels) < 2:
            raise ValueError("Plan correction alignment requires at least two levels.")
        indices = tuple(level.level_index for level in normalized_levels)
        if len(set(indices)) != len(indices):
            raise ValueError("Plan correction alignment levels must be unique.")
        if int(anchor_level_index) not in indices:
            raise ValueError("The plan alignment anchor must be in the batch.")
        self._levels = {level.level_index: level for level in normalized_levels}
        self._level_order = indices
        self._anchor_level_index = int(anchor_level_index)
        self._viewport_size = (
            float(viewport_size[0]),
            float(viewport_size[1]),
        )
        self._condition = threading.Condition()
        self._submissions: dict[int, _PlanCorrectionAlignmentSubmission] = {}
        self._outcome: PlanCorrectionAlignmentOutcome | None = None
        self._cancelled = False

    @property
    def outcome(self) -> PlanCorrectionAlignmentOutcome | None:
        """Return the immutable result once every plan has been submitted."""

        with self._condition:
            return self._outcome

    def submit(
        self,
        level_index: int,
        image_path: str | Path,
    ) -> PlanCorrectionAlignmentOutcome | None:
        """Submit one saved raster; the final submitter performs registration."""

        normalized_index = int(level_index)
        submission = _PlanCorrectionAlignmentSubmission(
            level_index=normalized_index,
            image_path=Path(image_path).expanduser(),
        )
        alignment_levels: tuple[CanvasPlanAlignmentLevel, ...] | None = None
        with self._condition:
            if self._cancelled:
                raise CanvasPlanAlignmentCancelled(
                    "Canvas plan alignment was cancelled."
                )
            if self._outcome is not None:
                return self._outcome
            if normalized_index not in self._levels:
                raise ValueError(
                    f"Level {normalized_index} is not in the correction batch."
                )
            if normalized_index in self._submissions:
                raise ValueError(
                    f"Level {normalized_index} submitted alignment more than once."
                )
            self._submissions[normalized_index] = submission
            if len(self._submissions) == len(self._levels):
                alignment_levels = self._build_alignment_levels()

        if alignment_levels is not None:
            outcome = self._calculate_outcome(alignment_levels)
            with self._condition:
                if self._cancelled:
                    raise CanvasPlanAlignmentCancelled(
                        "Canvas plan alignment was cancelled."
                    )
                if self._outcome is None:
                    self._outcome = outcome
                self._condition.notify_all()
                return self._outcome
        return None

    def align(
        self,
        level_index: int,
        image_path: str | Path,
        *,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> PlanCorrectionAlignmentOutcome:
        """Submit one plan and wait for the shared best-effort result."""

        immediate = self.submit(
            level_index,
            image_path,
        )
        if immediate is not None:
            return immediate
        while True:
            with self._condition:
                if self._cancelled:
                    raise CanvasPlanAlignmentCancelled(
                        "Canvas plan alignment was cancelled."
                    )
                if self._outcome is not None:
                    return self._outcome
                self._condition.wait(_WAIT_POLL_SECONDS)
            if cancellation_check is not None and cancellation_check():
                self.cancel()

    def cancel(self) -> None:
        """Cancel registration and release every worker waiting at the barrier."""

        with self._condition:
            if self._outcome is not None or self._cancelled:
                return
            self._cancelled = True
            self._condition.notify_all()

    def _build_alignment_levels(self) -> tuple[CanvasPlanAlignmentLevel, ...]:
        levels: list[CanvasPlanAlignmentLevel] = []
        for level_index in self._level_order:
            baseline = self._levels[level_index]
            submission = self._submissions[level_index]
            levels.append(
                CanvasPlanAlignmentLevel(
                    level_index=level_index,
                    image_path=str(submission.image_path),
                    canvas_level_scale=baseline.canvas_level_scale,
                    canvas_offset_x_pixels=baseline.canvas_offset_x_pixels,
                    canvas_offset_y_pixels=baseline.canvas_offset_y_pixels,
                )
            )
        return tuple(levels)

    def _calculate_outcome(
        self,
        levels: tuple[CanvasPlanAlignmentLevel, ...],
    ) -> PlanCorrectionAlignmentOutcome:
        try:
            result = align_canvas_plan_levels(
                levels,
                anchor_level_index=self._anchor_level_index,
                viewport_size=self._viewport_size,
                cancellation_check=self._is_cancelled,
            )
        except CanvasPlanAlignmentCancelled:
            raise
        except CanvasPlanAlignmentError as error:
            return PlanCorrectionAlignmentOutcome(
                result=None,
                warning=str(error).strip()
                or "No corrected levels could be aligned reliably.",
            )
        except Exception:  # noqa: BLE001 - keep CV/provider internals private.
            return PlanCorrectionAlignmentOutcome(
                result=None,
                warning="The corrected plans could not be aligned from visual cues.",
            )

        skipped = tuple(result.skipped_levels)
        if not skipped:
            return PlanCorrectionAlignmentOutcome(result=result)
        skipped_text = "; ".join(
            f"L{level_index}: {reason}" for level_index, reason in skipped
        )
        return PlanCorrectionAlignmentOutcome(
            result=result,
            warning=(
                "Some corrected levels could not be structurally aligned: "
                f"{skipped_text}"
            ),
        )

    def _is_cancelled(self) -> bool:
        with self._condition:
            return self._cancelled
