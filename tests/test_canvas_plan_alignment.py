# ### Imports ###
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from housemaker.automatic_wall_detection import PlanWallAnalysis, WallLineEvidence
from housemaker.canvas_plan_alignment import (
    CanvasPlanAlignmentError,
    CanvasPlanAlignmentLevel,
    _build_line_heat_map,
    _build_stair_cue_mask,
    _feature_map_from_analysis,
    _has_reliable_visual_alignment,
    _PairRegistration,
    _PlanFeatureMap,
    _register_feature_maps,
    _RegistrationCandidate,
    _WallAlignmentScore,
    align_canvas_plan_levels,
)

# ### Test helpers ###
_IMAGE_WIDTH = 300
_IMAGE_HEIGHT = 220
_VIEWPORT_SIZE = (900.0, 660.0)


def _level(level_index: int, image_path: str) -> CanvasPlanAlignmentLevel:
    return CanvasPlanAlignmentLevel(
        level_index=level_index,
        image_path=image_path,
        canvas_level_scale=1.0,
        canvas_offset_x_pixels=0.0,
        canvas_offset_y_pixels=0.0,
    )


def _evidence(
    segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[WallLineEvidence, ...]:
    return tuple(
        WallLineEvidence(start=start, end=end, confidence=0.95)
        for start, end in segments
    )


def _analysis(
    segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> PlanWallAnalysis:
    return PlanWallAnalysis(
        image_width=_IMAGE_WIDTH,
        image_height=_IMAGE_HEIGHT,
        line_evidence=_evidence(segments),
    )


def _transform_segments(
    segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    *,
    target_to_reference_scale: float,
    target_to_reference_translation: tuple[float, float],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    scale = target_to_reference_scale
    translation = np.asarray(target_to_reference_translation, dtype=np.float64)
    transformed = []
    for start, end in segments:
        target_start = (np.asarray(start, dtype=np.float64) - translation) / scale
        target_end = (np.asarray(end, dtype=np.float64) - translation) / scale
        transformed.append((tuple(target_start), tuple(target_end)))
    return tuple(transformed)


def _as_feature_map(
    level: CanvasPlanAlignmentLevel,
    analysis: PlanWallAnalysis,
) -> _PlanFeatureMap:
    return _feature_map_from_analysis(level, analysis)


def _cross_feature_map(level: CanvasPlanAlignmentLevel) -> _PlanFeatureMap:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 9] = 255
    mask[10, 5:15] = 255
    heat = _build_line_heat_map(mask)
    return _PlanFeatureMap(
        level=level,
        image_width=20,
        image_height=20,
        native_to_feature_scale=1.0,
        alignment_tolerance_pixels=2.25,
        wall_masks=(mask, mask, mask, mask),
        wall_heat_maps=(heat, heat, heat, heat),
        stair_mask=np.zeros_like(mask),
        stair_heat_map=np.zeros_like(mask, dtype=np.float32),
    )


def _base_segments() -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return (
        ((42.0, 34.0), (252.0, 34.0)),
        ((252.0, 34.0), (252.0, 185.0)),
        ((252.0, 185.0), (42.0, 185.0)),
        ((42.0, 185.0), (42.0, 34.0)),
        ((112.0, 34.0), (112.0, 142.0)),
        ((112.0, 142.0), (207.0, 142.0)),
        ((207.0, 82.0), (252.0, 82.0)),
        ((166.0, 34.0), (166.0, 93.0)),
    )


# ### Public alignment tests ###
def test_transformed_plans_keep_canvas_scales_and_recover_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_level = _level(2, "ground.png")
    target_scale = 1.20
    target_level = CanvasPlanAlignmentLevel(
        level_index=3,
        image_path="story.png",
        canvas_level_scale=target_scale,
        canvas_offset_x_pixels=0.0,
        canvas_offset_y_pixels=0.0,
    )
    target_translation = (-20.0, 10.0)
    analyses = {
        reference_level.image_path: _analysis(_base_segments()),
        target_level.image_path: _analysis(
            _transform_segments(
                _base_segments(),
                target_to_reference_scale=target_scale,
                target_to_reference_translation=target_translation,
            )
        ),
    }

    def build_features(
        level: CanvasPlanAlignmentLevel,
        _cancellation_check: object,
        tolerance: float,
    ) -> _PlanFeatureMap:
        return _feature_map_from_analysis(
            level,
            analyses[level.image_path],
            tolerance,
        )

    monkeypatch.setattr(
        "housemaker.canvas_plan_alignment._build_plan_feature_map",
        build_features,
    )

    result = align_canvas_plan_levels(
        (reference_level, target_level),
        anchor_level_index=reference_level.level_index,
        viewport_size=_VIEWPORT_SIZE,
    )

    transform = next(
        item
        for item in result.transforms
        if item.level_index == target_level.level_index
    )
    assert transform.canvas_offset_x_pixels == pytest.approx(30.0, abs=10.0)
    assert transform.canvas_offset_y_pixels == pytest.approx(96.0, abs=10.0)
    assert transform.wall_score >= 0.70
    assert reference_level.canvas_level_scale == 1.0
    assert target_level.canvas_level_scale == target_scale


def test_unrelated_plans_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    reference_level = _level(2, "ground.png")
    target_level = _level(3, "unrelated.png")
    unrelated_segments = (
        ((20.0, 25.0), (280.0, 196.0)),
        ((20.0, 196.0), (280.0, 25.0)),
        ((28.0, 110.0), (272.0, 110.0)),
        ((150.0, 16.0), (150.0, 204.0)),
    )
    analyses = {
        reference_level.image_path: _analysis(_base_segments()),
        target_level.image_path: _analysis(unrelated_segments),
    }

    monkeypatch.setattr(
        "housemaker.canvas_plan_alignment._build_plan_feature_map",
        lambda level, _cancel, tolerance: _feature_map_from_analysis(
            level,
            analyses[level.image_path],
            tolerance,
        ),
    )

    with pytest.raises(
        CanvasPlanAlignmentError,
        match="No other plan image could be aligned reliably",
    ):
        align_canvas_plan_levels(
            (reference_level, target_level),
            anchor_level_index=reference_level.level_index,
            viewport_size=_VIEWPORT_SIZE,
        )


def test_alignment_graph_can_use_an_intermediate_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    levels = (
        _level(1, "anchor.png"),
        _level(2, "bridge.png"),
        _level(3, "target.png"),
    )
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 9] = 255
    mask[10, 5:15] = 255
    heat = _build_line_heat_map(mask)

    def build_features(
        level: CanvasPlanAlignmentLevel,
        _cancellation_check: object,
        tolerance: float,
    ) -> _PlanFeatureMap:
        return _PlanFeatureMap(
            level=level,
            image_width=20,
            image_height=20,
            native_to_feature_scale=1.0,
            alignment_tolerance_pixels=tolerance,
            wall_masks=(mask, mask, mask, mask),
            wall_heat_maps=(heat, heat, heat, heat),
            stair_mask=np.zeros_like(mask),
            stair_heat_map=np.zeros_like(mask, dtype=np.float32),
        )

    attempted_pairs: list[tuple[int, int]] = []

    def register(
        reference: _PlanFeatureMap,
        target: _PlanFeatureMap,
        _viewport: tuple[float, float],
        _cancel: object,
        **_options: object,
    ) -> _PairRegistration:
        pair = (reference.level.level_index, target.level.level_index)
        attempted_pairs.append(pair)
        if pair == (1, 2):
            return _PairRegistration(1.0, (0.0, 0.0), 0.8, 0.8, 0.0)
        if pair == (2, 3):
            return _PairRegistration(1.0, (4.0, 0.0), 0.9, 0.9, 0.0)
        raise CanvasPlanAlignmentError("No shared evidence.")

    monkeypatch.setattr(
        "housemaker.canvas_plan_alignment._build_plan_feature_map",
        build_features,
    )
    monkeypatch.setattr(
        "housemaker.canvas_plan_alignment._register_feature_maps",
        register,
    )

    result = align_canvas_plan_levels(
        levels,
        anchor_level_index=1,
        viewport_size=(200.0, 200.0),
    )

    target = next(item for item in result.transforms if item.level_index == 3)
    assert target.reference_level_index == 2
    assert (1, 3) in attempted_pairs
    assert (2, 3) in attempted_pairs


def test_unusable_requested_anchor_falls_back_to_analyzable_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    levels = (
        _level(2, "ground.png"),
        _level(3, "story-1.png"),
        _level(4, "story-2.png"),
    )

    def build_features(
        level: CanvasPlanAlignmentLevel,
        _cancellation_check: object,
        _tolerance: float,
    ) -> _PlanFeatureMap:
        if level.level_index == 2:
            raise CanvasPlanAlignmentError("No usable wall lines were found.")
        return _cross_feature_map(level)

    monkeypatch.setattr(
        "housemaker.canvas_plan_alignment._build_plan_feature_map",
        build_features,
    )
    monkeypatch.setattr(
        "housemaker.canvas_plan_alignment._register_feature_maps",
        lambda *_args, **_kwargs: _PairRegistration(
            1.0,
            (0.0, 0.0),
            0.9,
            0.9,
            0.0,
        ),
    )

    result = align_canvas_plan_levels(
        levels,
        anchor_level_index=2,
        viewport_size=(200.0, 200.0),
    )

    assert result.anchor_level_index == 3
    assert {item.level_index for item in result.transforms} == {3, 4}
    assert result.skipped_levels == ((2, "No usable wall lines were found."),)


def test_alignment_rejects_scale_mismatch_instead_of_rescaling_canvas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_level = _level(2, "ground.png")
    target_level = CanvasPlanAlignmentLevel(
        level_index=3,
        image_path="story.png",
        canvas_level_scale=1.5,
        canvas_offset_x_pixels=0.0,
        canvas_offset_y_pixels=0.0,
    )
    translation = (25.0, -15.0)
    analyses = {
        reference_level.image_path: _analysis(_base_segments()),
        target_level.image_path: _analysis(
            _transform_segments(
                _base_segments(),
                target_to_reference_scale=1.0,
                target_to_reference_translation=translation,
            )
        ),
    }

    monkeypatch.setattr(
        "housemaker.canvas_plan_alignment._build_plan_feature_map",
        lambda level, _cancel, tolerance: _feature_map_from_analysis(
            level,
            analyses[level.image_path],
            tolerance,
        ),
    )

    with pytest.raises(
        CanvasPlanAlignmentError,
        match="No other plan image could be aligned reliably",
    ):
        align_canvas_plan_levels(
            (reference_level, target_level),
            anchor_level_index=reference_level.level_index,
            viewport_size=_VIEWPORT_SIZE,
        )

    assert target_level.canvas_level_scale == 1.5


def test_saved_corrected_rasters_align_from_inner_wall_lines(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "ground.png"
    target_path = tmp_path / "story.png"
    reference_image = np.full(
        (_IMAGE_HEIGHT, _IMAGE_WIDTH),
        255,
        dtype=np.uint8,
    )
    target_image = reference_image.copy()
    for start, end in _base_segments():
        integer_start = tuple(round(value) for value in start)
        integer_end = tuple(round(value) for value in end)
        cv2.line(reference_image, integer_start, integer_end, 0, 5)
        cv2.line(
            target_image,
            (integer_start[0] - 15, integer_start[1] + 8),
            (integer_end[0] - 15, integer_end[1] + 8),
            0,
            5,
        )
    assert cv2.imwrite(str(reference_path), reference_image)
    assert cv2.imwrite(str(target_path), target_image)

    result = align_canvas_plan_levels(
        (
            _level(2, str(reference_path)),
            _level(3, str(target_path)),
        ),
        anchor_level_index=2,
        viewport_size=_VIEWPORT_SIZE,
    )

    transform = next(item for item in result.transforms if item.level_index == 3)
    assert transform.canvas_offset_x_pixels == pytest.approx(45.0, abs=3.0)
    assert transform.canvas_offset_y_pixels == pytest.approx(-24.0, abs=3.0)
    assert transform.wall_score >= 0.8


def test_different_raster_sizes_align_without_changing_canvas_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_level = _level(2, "ground.png")
    target_level = CanvasPlanAlignmentLevel(
        level_index=3,
        image_path="larger-story.png",
        canvas_level_scale=2.0,
        canvas_offset_x_pixels=0.0,
        canvas_offset_y_pixels=0.0,
    )
    analyses = {
        reference_level.image_path: _analysis(_base_segments()),
        target_level.image_path: PlanWallAnalysis(
            image_width=_IMAGE_WIDTH * 2,
            image_height=_IMAGE_HEIGHT * 2,
            line_evidence=_evidence(_base_segments()),
        ),
    }
    monkeypatch.setattr(
        "housemaker.canvas_plan_alignment._build_plan_feature_map",
        lambda level, _cancel, tolerance: _feature_map_from_analysis(
            level,
            analyses[level.image_path],
            tolerance,
        ),
    )

    result = align_canvas_plan_levels(
        (reference_level, target_level),
        anchor_level_index=2,
        viewport_size=_VIEWPORT_SIZE,
    )

    transform = next(item for item in result.transforms if item.level_index == 3)
    assert transform.canvas_offset_x_pixels == pytest.approx(450.0, abs=3.0)
    assert transform.canvas_offset_y_pixels == pytest.approx(330.0, abs=3.0)
    assert reference_level.canvas_level_scale == 1.0
    assert target_level.canvas_level_scale == 2.0


# ### Stair cue tests ###
def test_stair_cue_detects_repeated_short_parallel_lines() -> None:
    repeated = tuple(
        WallLineEvidence(
            start=(80.0, float(y)),
            end=(125.0, float(y)),
            confidence=0.95,
        )
        for y in (70, 78, 86, 94, 102, 110)
    )
    isolated = WallLineEvidence(
        start=(205.0, 45.0),
        end=(245.0, 45.0),
        confidence=0.95,
    )

    mask = _build_stair_cue_mask(
        (*repeated, isolated),
        _IMAGE_WIDTH,
        _IMAGE_HEIGHT,
        1.0,
        min(_IMAGE_WIDTH, _IMAGE_HEIGHT),
    )

    assert np.count_nonzero(mask[65:115, 75:130]) > 100
    assert np.count_nonzero(mask[40:50, 200:250]) == 0


def test_stair_cue_cannot_authorize_weak_wall_evidence() -> None:
    weak_walls = _WallAlignmentScore(
        score=0.1,
        precision=0.1,
        reverse_coverage=0.1,
        matched_wall_pixels=500.0,
        target_wall_pixels=500.0,
        reference_roi_wall_pixels=500.0,
        matched_orientation_count=4,
    )
    candidate = _RegistrationCandidate(
        score=0.172,
        feature_scale=1.0,
        feature_translation=(0.0, 0.0),
        wall_alignment=weak_walls,
        stair_score=1.0,
    )

    assert not _has_reliable_visual_alignment(candidate)


def test_pair_registration_chooses_reliable_candidate_over_higher_weak_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _cross_feature_map(_level(2, "ground.png"))
    target = _cross_feature_map(_level(3, "story.png"))
    weak = _RegistrationCandidate(
        score=0.95,
        feature_scale=1.0,
        feature_translation=(70.0, 30.0),
        wall_alignment=_WallAlignmentScore(
            score=0.2,
            precision=0.2,
            reverse_coverage=0.2,
            matched_wall_pixels=200.0,
            target_wall_pixels=500.0,
            reference_roi_wall_pixels=500.0,
            matched_orientation_count=1,
        ),
        stair_score=1.0,
    )
    reliable = _RegistrationCandidate(
        score=0.8,
        feature_scale=1.0,
        feature_translation=(4.0, -3.0),
        wall_alignment=_WallAlignmentScore(
            score=0.8,
            precision=0.8,
            reverse_coverage=0.8,
            matched_wall_pixels=400.0,
            target_wall_pixels=500.0,
            reference_roi_wall_pixels=500.0,
            matched_orientation_count=2,
        ),
        stair_score=0.5,
    )

    def find_candidate(
        *_args: object,
        candidate_sink: list[_RegistrationCandidate],
        **_kwargs: object,
    ) -> _RegistrationCandidate:
        candidate_sink.extend((weak, reliable))
        return weak

    monkeypatch.setattr(
        "housemaker.canvas_plan_alignment._find_best_registration_candidate",
        find_candidate,
    )

    registration = _register_feature_maps(
        reference,
        target,
        (200.0, 200.0),
        None,
    )

    assert registration.native_scale == pytest.approx(1.0)
    assert registration.native_translation == pytest.approx((4.0, -3.0))
    assert registration.wall_score == pytest.approx(0.8)
