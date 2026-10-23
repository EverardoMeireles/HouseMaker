"""Authored sunlight reaches each viewer's actual shader uniforms."""

# ### Environment and imports ###
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QMatrix4x4, QVector3D
from PySide6.QtWidgets import QApplication

import housemaker.viewer as viewer_module
from housemaker.directional_light_state import DirectionalLightData
from housemaker.global_scale import GlobalScale

# ### Fixtures and uniform helpers ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def viewer():
    widget = viewer_module.GlbViewerWidget(window_editing_enabled=True)
    try:
        yield widget
    finally:
        widget.close()
        widget.deleteLater()
        _application.processEvents()


def _light(*, light_id="sun", position=(0.0, 0.0, 3.0),
           target=(0.0, 0.0, 0.0), color="#ffffff", intensity=1.0):
    return DirectionalLightData(light_id, "Sun", position, target,
                                color=color, intensity=intensity)


def _direction(uniforms, index=0):
    return np.asarray([uniforms[f"u_sun_direction_{axis}"][index] for axis in "xyz"])


def _color(uniforms, index=0):
    return np.asarray([uniforms[f"u_sun_color_{channel}"][index] for channel in "rgb"])


def _texture_item():
    data = viewer_module.TextureMeshData(
        np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float32),
        np.asarray(((0, 0, 1),) * 3, dtype=np.float32),
        np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.float32),
        np.full((1, 1, 4), 255, dtype=np.uint8),
    )
    return viewer_module.TexturedMeshItem(data, ambient_light_intensity=0.2)


# ### World directions, intensity and no distance attenuation ###
def test_light_snapshot_uses_toward_sun_direction_and_authored_color(viewer):
    viewer.set_directional_lights((_light(
        position=(4, 0, 3), target=(0, 0, 0), color="#804020", intensity=2.0,
    ),))
    snapshot = viewer._get_sun_light_snapshot()
    assert len(snapshot) == 1
    direction, color, intensity = snapshot[0]
    np.testing.assert_allclose(direction, (0.8, 0.0, 0.6))
    np.testing.assert_allclose(color, np.asarray((128, 64, 32)) / 255, atol=2e-5)
    assert intensity == 2.0
    uniforms = viewer_module._build_sun_light_uniforms(snapshot)
    np.testing.assert_allclose(_color(uniforms), np.asarray(color) * 2.0)


def test_translating_sun_and_target_together_has_no_falloff(viewer):
    near = _light(position=(3, 4, 10), target=(0, 0, 0), intensity=2.5)
    viewer.set_directional_lights((near,))
    before = viewer_module._build_sun_light_uniforms(viewer._get_sun_light_snapshot())
    viewer.set_directional_lights((near.translated((10000, -30000, 20000)),))
    after = viewer_module._build_sun_light_uniforms(viewer._get_sun_light_snapshot())
    for key in before:
        np.testing.assert_allclose(after[key], before[key])
    viewer.set_directional_lights((_light(position=(300, 400, 1000), intensity=2.5),))
    far = viewer_module._build_sun_light_uniforms(viewer._get_sun_light_snapshot())
    np.testing.assert_allclose(_direction(far), _direction(before))
    np.testing.assert_allclose(_color(far), _color(before))


def test_live_rotation_gizmo_endpoints_change_lighting_before_commit(viewer):
    viewer.set_directional_lights((_light(),))
    original = viewer._get_sun_light_snapshot()
    viewer._directional_light_drag = SimpleNamespace(
        light_id="sun", preview_position=(0.0, 0.0, 3.0),
        preview_target=(-3.0, 0.0, 3.0),
    )
    try:
        preview = viewer._get_sun_light_snapshot()
        np.testing.assert_allclose(preview[0][0], (1.0, 0.0, 0.0))
        assert preview != original
        assert viewer._directional_lights[0].target == (0.0, 0.0, 0.0)
    finally:
        viewer._directional_light_drag = None
    assert viewer._get_sun_light_snapshot() == original


# ### View-space mapping, anisotropic scale and camera independence ###
def test_sun_direction_uses_view_linear_transform_not_translation_or_normal_matrix():
    snapshot = (((1.0, 1.0, 1.0), (0.2, 0.5, 1.0), 1.5),)
    matrix = QMatrix4x4()
    matrix.translate(10000.0, -30000.0, 4000.0)
    matrix.rotate(90.0, 0.0, 0.0, 1.0)
    matrix.scale(2.0, 1.0, 0.5)
    uniforms = viewer_module._build_sun_light_uniforms(snapshot, matrix)
    expected_vector = matrix.mapVector(QVector3D(1.0, 1.0, 1.0))
    expected = np.asarray((expected_vector.x(), expected_vector.y(), expected_vector.z()))
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(_direction(uniforms), expected, atol=1e-6)
    np.testing.assert_allclose(_color(uniforms), (0.3, 0.75, 1.5))


def test_textured_item_reads_owning_view_provider_and_current_camera_each_draw(viewer):
    viewer.set_directional_lights((_light(position=(2, 0, 0)),))
    item = _texture_item()
    viewer.view.addItem(item)
    identity = QMatrix4x4()
    rotated = QMatrix4x4()
    rotated.rotate(180.0, 0.0, 0.0, 1.0)
    with patch.object(viewer.view, "viewMatrix", return_value=identity):
        first = item._get_sun_light_uniforms()
    with patch.object(viewer.view, "viewMatrix", return_value=rotated):
        second = item._get_sun_light_uniforms()
    np.testing.assert_allclose(_direction(first), (1, 0, 0), atol=1e-6)
    np.testing.assert_allclose(_direction(second), (-1, 0, 0), atol=1e-6)
    assert item._get_sun_light_uniforms()["u_sun_count"] == [1.0]


def test_global_scale_changes_physical_direction_once_without_scaling_intensity(viewer):
    viewer.set_directional_lights((_light(position=(1, 1, 1), intensity=3.0),))
    snapshot = viewer._get_sun_light_snapshot()
    viewer.set_global_scale(GlobalScale(2.0, 1.0, 0.5, linked=False))
    item = _texture_item()
    viewer.view.addItem(item)
    matrix = QMatrix4x4()
    matrix.scale(2.0, 1.0, 0.5)
    with patch.object(viewer.view, "viewMatrix", return_value=matrix):
        actual = item._get_sun_light_uniforms()
    expected = np.asarray((2.0, 1.0, 0.5))
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(_direction(actual), expected, atol=1e-6)
    np.testing.assert_allclose(_color(actual), (3.0, 3.0, 3.0))
    np.testing.assert_allclose(viewer._get_sun_light_snapshot()[0][0], snapshot[0][0])


# ### Zero intensity, limits and isolated GL uniform state ###
def test_zero_authored_intensity_does_not_restore_default_headlight():
    empty = viewer_module._build_sun_light_uniforms(())
    off = viewer_module._build_sun_light_uniforms((((0, 0, 1), (1, 1, 1), 0.0),))
    assert empty["u_sun_count"] == [0.0]
    assert off["u_sun_count"] == [1.0]
    np.testing.assert_array_equal(_color(off), (0.0, 0.0, 0.0))


def test_shader_uniform_arrays_are_bounded_scalar_arrays_for_both_shaders():
    snapshot = tuple(((0, 0, 1), (1, 1, 1), float(index + 1)) for index in range(8))
    uniforms = viewer_module._build_sun_light_uniforms(snapshot)
    assert viewer_module.MAX_PREVIEW_SUN_LIGHTS == 4
    assert uniforms["u_sun_count"] == [4.0]
    for axis in "xyz":
        assert len(uniforms[f"u_sun_direction_{axis}"]) == 4
    for channel in "rgb":
        assert uniforms[f"u_sun_color_{channel}"] == [1.0, 2.0, 3.0, 4.0]
    for source in (viewer_module.AMBIENT_LIT_FRAGMENT_SHADER,
                   viewer_module.TEXTURED_AMBIENT_FRAGMENT_SHADER):
        for name in uniforms:
            assert name in source


def test_ambient_shader_uniform_callbacks_refresh_and_are_not_shared_between_viewers():
    red = (((1, 0, 0), (1, 0, 0), 2.0),)
    blue = (((0, 0, 1), (0, 0, 1), 3.0),)
    current = {"lights": red}
    first = viewer_module._build_ambient_shader(
        0.2, sun_uniforms_provider=lambda: viewer_module._build_sun_light_uniforms(current["lights"]),
    )
    second = viewer_module._build_ambient_shader(
        0.5, sun_uniforms_provider=lambda: viewer_module._build_sun_light_uniforms(blue),
    )
    first._refresh_sun_uniforms()
    second._refresh_sun_uniforms()
    np.testing.assert_allclose(_color(first.uniformData), (2, 0, 0))
    np.testing.assert_allclose(_color(second.uniformData), (0, 0, 3))
    current["lights"] = ()
    with patch.object(viewer_module.gl_shaders.ShaderProgram, "__enter__", return_value=None):
        first.__enter__()
    assert first.uniformData["u_sun_count"] == [0.0]
    np.testing.assert_allclose(_color(second.uniformData), (0, 0, 3))
    assert first.uniformData["u_ambient_light"] == [0.2]
    assert second.uniformData["u_ambient_light"] == [0.5]


def test_two_viewers_do_not_share_textured_sun_uniform_values(viewer):
    other = viewer_module.GlbViewerWidget()
    try:
        viewer.set_directional_lights((_light(color="#ff0000", intensity=2),))
        other.set_directional_lights((_light(color="#0000ff", intensity=3),))
        first, second = _texture_item(), _texture_item()
        viewer.view.addItem(first)
        other.view.addItem(second)
        np.testing.assert_allclose(_color(first._get_sun_light_uniforms()), (2, 0, 0))
        np.testing.assert_allclose(_color(second._get_sun_light_uniforms()), (0, 0, 3))
        viewer.set_directional_lights(())
        assert first._get_sun_light_uniforms()["u_sun_count"] == [0.0]
        np.testing.assert_allclose(_color(second._get_sun_light_uniforms()), (0, 0, 3))
    finally:
        other.close()
        other.deleteLater()
        _application.processEvents()


def test_textured_program_upload_replaces_all_uniform_arrays_including_empty_suns(viewer):
    viewer.set_directional_lights((_light(color="#ff0000", intensity=2),))
    item = _texture_item()
    viewer.view.addItem(item)
    locations, uploaded = {}, {}

    def location(_program, name):
        locations.setdefault(name, len(locations) + 1)
        return locations[name]

    def upload(uniform_location, length, values):
        name = next(name for name, value in locations.items() if value == uniform_location)
        assert length == len(values)
        uploaded[name] = list(values)

    item._shader_program = 17
    try:
        with (
            patch.object(viewer_module.GL, "glGetUniformLocation", side_effect=location) as get_location,
            patch.object(viewer_module.GL, "glUniform1fv", side_effect=upload) as set_uniform,
        ):
            item._upload_sun_light_uniforms()
            assert uploaded["u_sun_count"] == [1.0]
            assert uploaded["u_sun_color_r"][0] == 2.0
            first_uploads = set_uniform.call_count
            viewer.set_directional_lights(())
            item._upload_sun_light_uniforms()
            assert uploaded["u_sun_count"] == [0.0]
            assert all(value == 0.0 for value in uploaded["u_sun_color_r"])
            assert set_uniform.call_count == first_uploads * 2
            assert get_location.call_count == first_uploads
    finally:
        # No real OpenGL program was created; do not try to release a fake ID.
        item._shader_program = None
