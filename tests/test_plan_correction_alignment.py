# ### Imports ###
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from housemaker.canvas_plan_alignment import (
    CanvasPlanAlignmentCancelled,
    CanvasPlanAlignmentError,
    CanvasPlanAlignmentResult,
    CanvasPlanAlignmentTransform,
)
from housemaker.plan_correction_alignment import (
    PlanCorrectionAlignmentCoordinator,
    PlanCorrectionAlignmentLevel,
)


# ### Fixture helpers ###
def _levels() -> tuple[PlanCorrectionAlignmentLevel, ...]:
    return (
        PlanCorrectionAlignmentLevel(2, 1.25, 18.0, -9.0),
        PlanCorrectionAlignmentLevel(3, 0.75, -12.0, 21.0),
    )


def _alignment_result() -> CanvasPlanAlignmentResult:
    return CanvasPlanAlignmentResult(
        anchor_level_index=2,
        transforms=(
            CanvasPlanAlignmentTransform(
                level_index=2,
                canvas_offset_x_pixels=18.0,
                canvas_offset_y_pixels=-9.0,
                reference_level_index=2,
                score=1.0,
                wall_score=1.0,
                stair_score=0.0,
            ),
            CanvasPlanAlignmentTransform(
                level_index=3,
                canvas_offset_x_pixels=42.0,
                canvas_offset_y_pixels=-16.0,
                reference_level_index=2,
                score=0.86,
                wall_score=0.84,
                stair_score=0.25,
            ),
        ),
    )


# ### Barrier and transform tests ###
def test_final_submission_preserves_every_baseline_canvas_transform(
    tmp_path: Path,
) -> None:
    coordinator = PlanCorrectionAlignmentCoordinator(
        _levels(),
        anchor_level_index=2,
        viewport_size=(800.0, 600.0),
    )
    first_path = tmp_path / "ground.png"
    second_path = tmp_path / "story.png"

    with patch(
        "housemaker.plan_correction_alignment.align_canvas_plan_levels",
        return_value=_alignment_result(),
    ) as align:
        assert (
            coordinator.submit(
                2,
                first_path,
            )
            is None
        )
        outcome = coordinator.submit(
            3,
            second_path,
        )

    assert outcome is not None
    assert outcome.result == _alignment_result()
    assert coordinator.outcome == outcome
    align.assert_called_once()
    submitted_levels = align.call_args.args[0]
    assert tuple(level.image_path for level in submitted_levels) == (
        str(first_path),
        str(second_path),
    )
    assert tuple(level.canvas_level_scale for level in submitted_levels) == (
        1.25,
        0.75,
    )
    assert tuple(level.canvas_offset_x_pixels for level in submitted_levels) == (
        18.0,
        -12.0,
    )
    assert tuple(level.canvas_offset_y_pixels for level in submitted_levels) == (
        -9.0,
        21.0,
    )
    assert "lock_native_scale" not in align.call_args.kwargs


def test_every_submission_uses_the_same_captured_baseline_contract(
    tmp_path: Path,
) -> None:
    coordinator = PlanCorrectionAlignmentCoordinator(
        _levels(),
        anchor_level_index=2,
        viewport_size=(800.0, 600.0),
    )

    with patch(
        "housemaker.plan_correction_alignment.align_canvas_plan_levels",
        return_value=_alignment_result(),
    ) as align:
        coordinator.submit(
            2,
            tmp_path / "ground.png",
        )
        coordinator.submit(
            3,
            tmp_path / "story.png",
        )

    submitted_levels = align.call_args.args[0]
    assert submitted_levels[0].canvas_level_scale == 1.25
    assert submitted_levels[0].canvas_offset_x_pixels == 18.0
    assert submitted_levels[0].canvas_offset_y_pixels == -9.0
    assert submitted_levels[1].canvas_level_scale == 0.75
    assert submitted_levels[1].canvas_offset_x_pixels == -12.0
    assert submitted_levels[1].canvas_offset_y_pixels == 21.0


# ### Failure and cancellation tests ###
def test_unreliable_registration_keeps_correction_as_a_warning(
    tmp_path: Path,
) -> None:
    coordinator = PlanCorrectionAlignmentCoordinator(
        _levels(),
        anchor_level_index=2,
        viewport_size=(800.0, 600.0),
    )
    coordinator.submit(
        2,
        tmp_path / "ground.png",
    )

    with patch(
        "housemaker.plan_correction_alignment.align_canvas_plan_levels",
        side_effect=CanvasPlanAlignmentError("The wall evidence is ambiguous."),
    ):
        outcome = coordinator.submit(
            3,
            tmp_path / "story.png",
        )

    assert outcome is not None
    assert outcome.result is None
    assert outcome.warning == "The wall evidence is ambiguous."


def test_cancelled_coordinator_rejects_late_submissions(tmp_path: Path) -> None:
    coordinator = PlanCorrectionAlignmentCoordinator(
        _levels(),
        anchor_level_index=2,
        viewport_size=(800.0, 600.0),
    )
    coordinator.cancel()

    with pytest.raises(CanvasPlanAlignmentCancelled, match="cancelled"):
        coordinator.submit(
            2,
            tmp_path / "ground.png",
        )
