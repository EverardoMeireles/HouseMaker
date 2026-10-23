# ### Imports ###
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from housemaker.opening_proportions_geometry import measure_opening_proportions


# ### Test projection helpers ###
def _project_opening(
    *,
    width=1.2,
    height=2.0,
    rotation=(20.0, -18.0, 4.0),
    focal=950.0,
    image_size=(1600, 1200),
    arch_rise=0.0,
):
    points = np.array(
        [
            [-width / 2, -height / 2, 0],
            [width / 2, -height / 2, 0],
            [width / 2, height / 2, 0],
            [-width / 2, height / 2, 0],
            [0, -height / 2 - arch_rise, 0],
        ]
    )
    points = Rotation.from_euler("yxz", rotation, degrees=True).apply(points)
    points += (0.15, 0.08, 4.0)
    pixels = points[:, :2] / points[:, 2, None] * focal + np.asarray(image_size) / 2
    normalized = pixels / image_size
    return normalized[:4], normalized[4]


# ### Rectangular opening measurements ###
def test_front_view_uses_pixels_not_normalized_image_aspect():
    outline = ((0.3, 0.1), (0.5, 0.1), (0.5, 0.7), (0.3, 0.7))
    result = measure_opening_proportions(outline, (1600, 800))
    assert result.height_to_width == pytest.approx(1.5)
    assert result.arch_height_to_width == 0
    assert result.arch_amount == 0
    assert result.method == "front-view-ratio"


@pytest.mark.parametrize(
    "rotation", [(20, -18, 4), (-30, 25, -8), (32, 32, 15), (-18, -28, 0)]
)
@pytest.mark.parametrize("image_size", [(1600, 1200), (1200, 1600)])
def test_two_vanishing_directions_recover_perspective_aspect(rotation, image_size):
    outline, _ = _project_opening(rotation=rotation, image_size=image_size)
    result = measure_opening_proportions(outline, image_size)
    assert result.method == "perspective-rectified"
    assert result.height_to_width == pytest.approx(2.0 / 1.2, abs=1e-8)


def test_rolled_frontal_frame_retains_correct_aspect():
    outline, _ = _project_opening(rotation=(0, 0, 25))
    result = measure_opening_proportions(outline, (1600, 1200))
    assert result.height_to_width == pytest.approx(2.0 / 1.2)
    assert result.method == "front-view-ratio"


def test_one_vanishing_direction_uses_explicit_approximation_without_invented_fov():
    outline, _ = _project_opening(rotation=(40, 0, 0))
    result = measure_opening_proportions(outline, (1600, 1200))
    assert result.method == "apparent-image-ratio"
    assert "cannot be uniquely corrected" in result.warning
    assert result.height_to_width > 0


def test_arbitrary_convex_manual_quad_is_accepted_with_warning():
    result = measure_opening_proportions(
        ((0.2, 0.2), (0.7, 0.3), (0.65, 0.8), (0.15, 0.75)), (1000, 700)
    )
    assert math.isfinite(result.height_to_width)
    assert result.warning


def test_repeated_measurement_is_deterministic():
    outline, _ = _project_opening()
    assert measure_opening_proportions(
        outline, (1600, 1200)
    ) == measure_opening_proportions(outline, (1600, 1200))


# ### Arched doorway measurements ###
def test_perspective_arch_apex_recovers_total_height_and_model_arch_amount():
    outline, apex = _project_opening(height=1.8, width=1.2, arch_rise=0.4)
    result = measure_opening_proportions(
        outline, (1600, 1200), arch_apex=apex, is_doorway=True
    )
    assert result.height_to_width == pytest.approx(2.2 / 1.2)
    assert result.arch_height_to_width == pytest.approx(0.4 / 1.2)
    assert result.arch_amount == pytest.approx(0.4 / 0.6)


def test_apex_on_springline_means_rectangular_doorway():
    outline, apex = _project_opening(arch_rise=0)
    result = measure_opening_proportions(
        outline, (1600, 1200), arch_apex=apex, is_doorway=True
    )
    assert result.arch_height_to_width == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("apex", [(0.5, 0.5), (0.1, 0.1)])
def test_apex_must_be_above_middle_of_springline(apex):
    outline = ((0.3, 0.3), (0.7, 0.3), (0.7, 0.8), (0.3, 0.8))
    with pytest.raises(ValueError, match="above the middle"):
        measure_opening_proportions(
            outline, (1000, 1000), arch_apex=apex, is_doorway=True
        )


def test_arch_rise_must_fit_existing_doorway_profile_model():
    outline = ((0.4, 0.4), (0.6, 0.4), (0.6, 0.8), (0.4, 0.8))
    with pytest.raises(ValueError, match="half the opening width"):
        measure_opening_proportions(
            outline, (1000, 1000), arch_apex=(0.5, 0.1), is_doorway=True
        )


def test_window_cannot_receive_doorway_arch():
    with pytest.raises(ValueError, match="Only doorway"):
        measure_opening_proportions(
            ((0.3, 0.2), (0.7, 0.2), (0.7, 0.8), (0.3, 0.8)),
            (1000, 1000),
            arch_apex=(0.5, 0.1),
        )


# ### Invalid annotation handling ###
@pytest.mark.parametrize(
    "outline",
    [
        (),
        ((0.1, 0.1),) * 4,
        ((0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9)),
        ((0.1, 0.1), (0.9, 0.1), (0.5, 0.2), (0.1, 0.9)),
        ((-0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
        ((math.nan, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
    ],
)
def test_invalid_outlines_are_rejected(outline):
    with pytest.raises(ValueError):
        measure_opening_proportions(outline, (1000, 1000))


@pytest.mark.parametrize("size", [(0, 1000), (1000,), (True, 1000), (1.5, 1000)])
def test_invalid_image_size_is_rejected(size):
    with pytest.raises(ValueError, match="valid width and height"):
        measure_opening_proportions(
            ((0.3, 0.2), (0.7, 0.2), (0.7, 0.8), (0.3, 0.8)), size
        )
