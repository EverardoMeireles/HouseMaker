# ### Environment setup ###
from __future__ import annotations

import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import numpy as np
import pytest
import trimesh
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
from trimesh.visual.material import MultiMaterial, PBRMaterial

from housemaker.directional_light_projection import SUN_RAY_FALLBACK_CACHE_KEY
from housemaker.directional_light_state import DirectionalLightData, SunOrbitData
from housemaker.glass_material import build_housemaker_glass_material
from housemaker.glb import GeneratedModel
from housemaker.global_scale import GlobalScale
from housemaker.viewer import (
    TRANSFORM_GIZMO_TRANSLATE,
    GlbViewerWidget,
    _get_opaque_sun_collision_mesh,
    _TransformGizmoHandle,
)

# ### Fixtures ###
_application = QApplication.instance() or QApplication([])


def _light(light_id="sun-1"):
    return DirectionalLightData(
        light_id=light_id, name=light_id,
        position=(0.0, 0.0, 2.0), target=(1.0, 0.0, 1.0),
    )


@pytest.fixture
def viewer():
    widget = GlbViewerWidget(window_editing_enabled=True)
    widget.resize(900, 650)
    widget.show()
    _application.processEvents()
    widget.set_directional_lights((_light(), _light("sun-2")))
    yield widget
    widget.close()
    widget.deleteLater()
    _application.processEvents()


def _tick_at(viewer, now):
    with patch("housemaker.viewer.time.monotonic", return_value=now):
        viewer._tick_sun_orbit_animation()


# ### Preview-only monotonic orbit ###
def test_orbit_advances_first_sun_only_without_mutating_authored_state(viewer):
    original = viewer._directional_lights
    viewer.set_sun_orbit(SunOrbitData(enabled=True, speed_degrees_per_second=90.0))
    viewer._sun_orbit_last_tick_time = 10.0
    _tick_at(viewer, 11.0)
    assert viewer.get_sun_orbit_phase_degrees() == 90.0
    first_position, first_target = viewer._get_directional_light_display_endpoints(original[0])
    np.testing.assert_allclose(first_position, original[0].position)
    np.testing.assert_allclose(first_target, (0.0, 1.0, 1.0), atol=1e-8)
    assert viewer._get_directional_light_display_endpoints(original[1]) == (
        original[1].position, original[1].target
    )
    assert viewer._directional_lights == original
    _tick_at(viewer, 15.0)
    assert viewer.get_sun_orbit_phase_degrees() == 90.0


def test_orbit_preserves_physical_pitch_under_nonuniform_global_scale(viewer):
    viewer.set_global_scale(GlobalScale(2.0, 1.0, 0.5, linked=False))
    viewer.set_sun_orbit(SunOrbitData(enabled=True, speed_degrees_per_second=90.0))
    viewer._sun_orbit_last_tick_time = 10.0
    _tick_at(viewer, 11.0)
    position, target = viewer._get_directional_light_display_endpoints(_light())
    physical = (np.asarray(target) - position) * (2.0, 1.0, 0.5)
    np.testing.assert_allclose(physical, (0.0, 2.0, -0.5), atol=1e-8)


def test_shader_and_visible_ray_guide_share_the_same_effective_direction(viewer):
    mesh = trimesh.creation.box()
    viewer.set_model(GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b""))
    viewer.set_selected_directional_light_id("sun-1")
    viewer.set_sun_orbit(SunOrbitData(enabled=True, speed_degrees_per_second=90.0))
    viewer._sun_orbit_last_tick_time = 10.0
    with patch("housemaker.viewer.build_directional_light_projection_grid",
               return_value=np.empty((0, 3), np.float32)) as build_rays:
        _tick_at(viewer, 11.0)
    toward_sun = np.asarray(viewer._get_sun_light_snapshot()[0][0])
    np.testing.assert_allclose(build_rays.call_args.args[1], -toward_sun)


def test_orbit_throttles_raycasting_without_rebuilding_the_scene(viewer):
    viewer.set_sun_orbit(SunOrbitData(enabled=True, speed_degrees_per_second=5.0))
    viewer._sun_orbit_last_tick_time = 10.0
    with (
        patch.object(viewer, "_refresh_directional_light_projection_grid_item") as rays,
        patch.object(viewer, "_populate_scene") as populate,
        patch.object(viewer, "set_model") as set_model,
    ):
        for now in (10.05, 10.1, 10.24, 10.26):
            _tick_at(viewer, now)
    assert rays.call_count == 2
    assert viewer.get_sun_orbit_phase_degrees() == pytest.approx(1.3)
    populate.assert_not_called()
    set_model.assert_not_called()


def test_fallback_guides_have_an_adaptive_cpu_budget_and_one_second_minimum(viewer):
    mesh = trimesh.creation.box()
    viewer.set_model(GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b""))
    viewer.set_selected_directional_light_id("sun-1")
    collision_mesh = viewer._get_directional_light_projection_collision_mesh()
    collision_mesh._cache[SUN_RAY_FALLBACK_CACHE_KEY] = True
    with (
        patch("housemaker.viewer.build_directional_light_projection_grid",
              return_value=np.empty((0, 3), np.float32)),
        patch("housemaker.viewer.time.perf_counter", side_effect=(10.0, 10.04)),
    ):
        viewer._directional_light_projection_render_key = None
        viewer._refresh_directional_light_projection_grid_item()
    assert viewer._sun_orbit_ray_refresh_interval_seconds == 1.0
    with (
        patch("housemaker.viewer.build_directional_light_projection_grid",
              return_value=np.empty((0, 3), np.float32)),
        patch("housemaker.viewer.time.perf_counter", side_effect=(10.0, 10.15)),
    ):
        viewer._directional_light_projection_render_key = None
        viewer._refresh_directional_light_projection_grid_item()
    assert viewer._sun_orbit_ray_refresh_interval_seconds == pytest.approx(1.5)


# ### Initial-direction editing and lifecycle ###
def test_initial_direction_edit_resets_and_pauses_orbit_with_rotation_only_gizmos(viewer):
    viewer.set_selected_directional_light_id("sun-1")
    viewer.set_sun_orbit(SunOrbitData(enabled=True))
    viewer._sun_orbit_last_tick_time = 10.0
    _tick_at(viewer, 11.0)
    viewer.set_sun_direction_editing(True)
    assert viewer.get_sun_orbit_phase_degrees() == 0.0
    assert not viewer._sun_orbit_timer.isActive()
    assert len(viewer._directional_light_gizmo_items) == 3
    assert not viewer._begin_directional_light_gizmo_drag(
        _TransformGizmoHandle(TRANSFORM_GIZMO_TRANSLATE, 0), QPointF(100, 100)
    )
    assert "paused" in viewer.object_transform_status_label.text()
    _tick_at(viewer, 30.0)
    assert viewer.get_sun_orbit_phase_degrees() == 0.0
    viewer.set_sun_direction_editing(False)
    assert viewer._sun_orbit_timer.isActive()
    assert len(viewer._directional_light_gizmo_items) == 9
    assert viewer.get_sun_orbit_phase_degrees() == 0.0


def test_hidden_and_closed_viewers_stop_orbit_without_accumulating_hidden_time(viewer):
    viewer.set_sun_orbit(SunOrbitData(enabled=True, speed_degrees_per_second=5.0))
    viewer._sun_orbit_last_tick_time = 10.0
    _tick_at(viewer, 11.0)
    phase = viewer.get_sun_orbit_phase_degrees()
    viewer.hide()
    assert not viewer._sun_orbit_timer.isActive()
    _tick_at(viewer, 1000.0)
    assert viewer.get_sun_orbit_phase_degrees() == phase
    viewer.show()
    with patch("housemaker.viewer.time.monotonic", return_value=1000.0):
        viewer._sun_orbit_timer.stop()
        viewer._sync_sun_orbit_timer()
    _tick_at(viewer, 1000.1)
    assert viewer.get_sun_orbit_phase_degrees() == pytest.approx(phase + 0.5)
    viewer.close()
    _tick_at(viewer, 2000.0)
    assert not viewer._sun_orbit_timer.isActive()
    assert viewer.get_sun_orbit_phase_degrees() == pytest.approx(phase + 0.5)


def test_zero_speed_and_empty_sun_list_do_not_run_timers(viewer):
    viewer.set_sun_orbit(SunOrbitData(enabled=True, speed_degrees_per_second=0.0))
    assert not viewer._sun_orbit_timer.isActive()
    viewer.set_sun_orbit(SunOrbitData(enabled=True))
    viewer.set_directional_lights(())
    assert not viewer._sun_orbit_timer.isActive()


# ### Marker activation versus initial-direction editing ###
def test_normal_marker_selection_previews_orbit_at_the_configured_slow_speed(viewer):
    mesh = trimesh.creation.box()
    viewer.set_model(GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b""))
    viewer.set_sun_orbit(SunOrbitData(enabled=True, speed_degrees_per_second=0.5))
    authored = viewer._directional_lights
    requests = []
    changes = []
    viewer.directional_light_preview_requested.connect(requests.append)
    viewer.directional_light_transform_changed.connect(lambda *value: changes.append(value))
    marker = viewer._project_tour_world_point(authored[0].position)[0]
    viewer._handle_window_wall_pick_requested(marker)
    assert requests == [authored[0].light_id]
    assert viewer._sun_orbit_timer.isActive()
    before = viewer._directional_light_projection_grid_item.pos.copy()
    viewer._sun_orbit_last_tick_time = 10.0
    _tick_at(viewer, 12.0)
    after = viewer._directional_light_projection_grid_item.pos
    assert viewer.get_sun_orbit_phase_degrees() == 1.0
    assert before.shape != after.shape or not np.allclose(before, after)
    assert viewer._directional_lights == authored
    assert not changes


def test_marker_reactivation_requests_preview_but_programmatic_selection_does_not(viewer):
    requests = []
    viewer.directional_light_preview_requested.connect(requests.append)
    viewer.set_selected_directional_light_id("sun-1")
    viewer.set_sun_direction_editing(True)
    assert requests == []
    # This models the controller handling the dedicated marker-only signal.
    viewer.directional_light_preview_requested.connect(
        lambda _light_id: viewer.set_sun_direction_editing(False)
    )
    marker = viewer._project_tour_world_point(_light().position)[0]
    viewer._handle_window_wall_pick_requested(marker)
    assert requests == ["sun-1"]
    assert not viewer._sun_direction_editing
    assert viewer._sun_orbit_timer.isActive()
    assert viewer.get_sun_orbit_phase_degrees() == 0.0
    viewer._sun_orbit_last_tick_time = 10.0
    _tick_at(viewer, 12.0)
    assert viewer.get_sun_orbit_phase_degrees() == 1.0


@pytest.mark.parametrize("orientation_visible", (False, True))
def test_real_marker_click_exits_initial_edit_via_its_dedicated_request(
    viewer, orientation_visible
):
    mesh = trimesh.creation.box()
    viewer.set_model(GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b""))
    viewer.set_selected_directional_light_id("sun-1")
    viewer.set_canvas_face_orientation_visible(orientation_visible)
    viewer.set_sun_direction_editing(True)
    requests = []
    viewer.directional_light_preview_requested.connect(requests.append)
    viewer.directional_light_preview_requested.connect(
        lambda _light_id: viewer.set_sun_direction_editing(False)
    )
    marker = viewer._project_tour_world_point(_light().position)[0]
    for event_type, buttons in (
        (QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
    ):
        event = QMouseEvent(
            event_type, marker, marker, Qt.MouseButton.LeftButton, buttons,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(viewer.view, event)
    assert requests == ["sun-1"]
    assert not viewer._sun_direction_editing
    assert viewer._sun_orbit_timer.isActive()


# ### Opaque first-hit collision inputs ###
@pytest.mark.parametrize("material", (
    build_housemaker_glass_material(False), PBRMaterial(alphaMode="BLEND"),
    PBRMaterial(alphaMode="MASK"),
))
def test_ray_guides_ignore_transparent_or_cutout_surfaces(material):
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    assert _get_opaque_sun_collision_mesh(mesh) is None


def test_mixed_material_collision_keeps_only_opaque_faces():
    mesh = trimesh.creation.box()
    material = MultiMaterial([PBRMaterial(), build_housemaker_glass_material(False)])
    face_materials = np.zeros(len(mesh.faces), dtype=int)
    face_materials[:2] = 1
    mesh.visual = trimesh.visual.TextureVisuals(
        material=material, face_materials=face_materials
    )
    opaque = _get_opaque_sun_collision_mesh(mesh)
    assert len(opaque.faces) == len(mesh.faces) - 2
