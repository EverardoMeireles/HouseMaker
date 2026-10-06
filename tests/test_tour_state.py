# ### Imports ###
from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.project_io import load_project, save_project
from housemaker.scene_export import build_runtime_scene_manifest
from housemaker.tour_state import (
    DEFAULT_IDLE_CAMERA_CYCLE_DURATION_SECONDS,
    DEFAULT_IDLE_CAMERA_RADIUS_METERS,
    DEFAULT_SPEED_EASING,
    DEFAULT_SPEED_OVERRIDE,
    DEFAULT_TEXT_FADE_DELAY_MS,
    DEFAULT_TEXT_FADE_DURATION_MS,
    DEFAULT_TEXT_ROTATION_DEGREES,
    DEFAULT_TEXT_SIZE_POINTS,
    DEFAULT_TOOLTIP_HTML_BODY,
    DEFAULT_TOOLTIP_POSITION,
    DEFAULT_TOOLTIP_STYLE,
    DEFAULT_TOUR_DURATION_SECONDS,
    DEFAULT_TOUR_PROGRESS_SPEED,
    DEFAULT_TOUR_TRIGGER_AREA_SIZE_METERS,
    DEFAULT_WAIT_INPUT_TOKEN,
    TOUR_SPEED_EASING_OPTIONS,
    TOUR_TOOLTIP_POSITION_OPTIONS,
    TourCurveData,
    TourData,
    TourFloatingTooltipData,
    TourIdleCameraAnimationData,
    TourSpeedOverrideData,
    TourStepData,
    TourText3DData,
    TourWaitForKeyPressData,
    effective_step_easing,
    evaluate_catmull_rom_point,
    evaluate_tour_camera_look_direction,
    evaluate_tour_camera_target,
    tour_to_runtime_dict,
    tours_from_payload,
    tours_to_runtime_dicts,
)


# ### Fixture helpers ###
def _tour() -> TourData:
    text = TourText3DData(
        component_id="text-1",
        text="Welcome",
        position=(7.0, 8.0, 9.0),
        size_points=18.0,
        color="#AABBCC",
        fade_delay_ms=250.0,
        fade_duration_ms=1500.0,
        rotation_degrees=(20.0, -35.0, 70.0),
    )
    later_step = TourStepData(
        step_id="step-b",
        progress=0.75,
        camera_target=(0.0, 3.0, 0.0),
        actions=(
            text,
            TourSpeedOverrideData(
                component_id="speed-1",
                speed=0.75,
            ),
        ),
    )
    earlier_step = TourStepData(
        step_id="step-a",
        progress=0.25,
        camera_target=(2.0, 0.0, 0.0),
    )
    return TourData(
        tour_id="tour-1",
        name="Ground floor",
        trigger_point=(10.0, 20.0, 30.0),
        trigger_area_size=(4.0, 6.0),
        curve=TourCurveData(
            points=(
                (0.0, 0.0, 0.0),
                (2.0, 3.0, 4.0),
                (5.0, 6.0, 7.0),
            ),
            tension=0.35,
        ),
        steps=(later_step, earlier_step),
        duration_seconds=18.0,
        progress_speed=1.5,
    )


# ### Tour model tests ###
class TourModelTests(unittest.TestCase):
    def test_project_dictionary_round_trip_preserves_authored_state(self) -> None:
        tour = _tour()

        restored = TourData.from_dict(tour.to_dict())

        self.assertEqual(restored, tour)
        self.assertEqual(restored.curve_points, tour.curve.points)
        self.assertEqual(restored.duration_seconds, 18.0)
        self.assertEqual(restored.progress_speed, 1.5)
        self.assertEqual(restored.trigger_area_size, (4.0, 6.0))
        self.assertEqual(restored.steps[0].speed_override, 0.75)
        self.assertEqual(restored.steps[0].actions[0].color, "#aabbcc")
        serialized_step = tour.to_dict()["steps"][0]
        self.assertIn("camera_target", serialized_step)
        self.assertNotIn("camera_look_direction", serialized_step)
        serialized_action = serialized_step["actions"][0]
        self.assertNotIn("components", serialized_step)
        self.assertEqual(serialized_action["size_points"], 18.0)
        self.assertEqual(serialized_action["fade_delay_ms"], 250.0)
        self.assertEqual(serialized_action["fade_duration_ms"], 1500.0)
        self.assertEqual(serialized_action["rotation_degrees"], [20.0, -35.0, 70.0])
        self.assertNotIn("speed_override", serialized_step)
        self.assertEqual(serialized_step["actions"][1]["speed"], 0.75)
        self.assertEqual(serialized_step["actions"][1]["easing"], "smoothstep")
        self.assertEqual(tour.to_dict()["progress_speed"], 1.5)

    def test_legacy_point_trigger_receives_default_area_size(self) -> None:
        payload = _tour().to_dict()
        payload.pop("trigger_area_size")

        restored = TourData.from_dict(payload)

        self.assertEqual(
            restored.trigger_area_size,
            DEFAULT_TOUR_TRIGGER_AREA_SIZE_METERS,
        )

    def test_legacy_tour_without_duration_uses_safe_default(self) -> None:
        payload = _tour().to_dict()
        payload.pop("duration_seconds")

        restored = TourData.from_dict(payload)

        self.assertEqual(restored.duration_seconds, DEFAULT_TOUR_DURATION_SECONDS)

    def test_legacy_timing_fields_use_safe_defaults(self) -> None:
        payload = _tour().to_dict()
        payload.pop("progress_speed")
        for step in payload["steps"]:
            step["actions"] = [
                action
                for action in step["actions"]
                if action["type"] != "speed_override"
            ]

        restored = TourData.from_dict(payload)

        self.assertEqual(restored.progress_speed, DEFAULT_TOUR_PROGRESS_SPEED)
        self.assertTrue(all(step.speed_override is None for step in restored.steps))

    def test_progress_speed_and_step_override_require_positive_finite_values(
        self,
    ) -> None:
        tour = _tour()

        for invalid_value in (0.0, -1.0, math.nan, math.inf):
            with (
                self.subTest(progress_speed=invalid_value),
                self.assertRaisesRegex(ValueError, "greater than zero|finite"),
            ):
                replace(tour, progress_speed=invalid_value)
            with (
                self.subTest(speed_override=invalid_value),
                self.assertRaisesRegex(ValueError, "greater than zero|finite"),
            ):
                TourSpeedOverrideData(
                    component_id="invalid-speed",
                    speed=invalid_value,
                )

    def test_damaged_collection_entries_are_skipped_individually(self) -> None:
        duplicate = replace(_tour(), name="Duplicate")

        restored = tours_from_payload(
            [_tour().to_dict(), {"broken": True}, duplicate.to_dict()]
        )

        self.assertEqual(restored, (_tour(),))

    def test_incomplete_or_closed_curves_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            TourCurveData(points=((0.0, 0.0, 0.0),))
        with self.assertRaisesRegex(ValueError, "open curve"):
            TourCurveData(
                points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                closed=True,
            )

    def test_two_point_catmull_rom_curve_is_linear(self) -> None:
        curve = TourCurveData(
            points=((1.0, -2.0, 4.0), (5.0, 6.0, 8.0)),
            tension=0.5,
        )

        self.assertEqual(evaluate_catmull_rom_point(curve, 0.0), curve.points[0])
        self.assertEqual(evaluate_catmull_rom_point(curve, 1.0), curve.points[1])
        self.assertEqual(
            evaluate_catmull_rom_point(curve, 0.5),
            (3.0, 2.0, 6.0),
        )

    def test_camera_target_is_interpolated_in_world_space(self) -> None:
        tour = _tour()

        self.assertEqual(
            tour.steps[0].camera_target,
            (0.0, 3.0, 0.0),
        )
        self.assertEqual(
            tour.steps[1].camera_target,
            (2.0, 0.0, 0.0),
        )
        self.assertEqual(evaluate_tour_camera_target(tour, 0.5), (1.0, 1.5, 0.0))

    def test_camera_look_direction_is_derived_from_camera_and_target(self) -> None:
        tour = _tour()

        direction = evaluate_tour_camera_look_direction(tour, 0.5)

        self.assertIsNotNone(direction)
        length = math.sqrt(19.25)
        self.assertAlmostEqual(direction[0], -1.0 / length)
        self.assertAlmostEqual(direction[1], -1.5 / length)
        self.assertAlmostEqual(direction[2], -4.0 / length)
        self.assertEqual(
            evaluate_tour_camera_target(tour, -1.0),
            (2.0, 0.0, 0.0),
        )
        self.assertEqual(
            evaluate_tour_camera_target(tour, 2.0),
            (0.0, 3.0, 0.0),
        )

    def test_camera_target_is_absent_without_steps(self) -> None:
        self.assertIsNone(evaluate_tour_camera_target(replace(_tour(), steps=()), 0.5))

    def test_nonfinite_camera_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            TourStepData(
                step_id="bad-target",
                progress=0.5,
                camera_target=(0.0, math.nan, 0.0),
            )

    def test_direction_era_steps_migrate_to_world_targets(self) -> None:
        payload = {
            "tour_id": "legacy-tour",
            "name": "Legacy tour",
            "trigger_point": [0.0, 0.0, 0.0],
            "curve": {
                "points": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            },
            "steps": [
                {
                    "step_id": "direction-above",
                    "progress": 0.5,
                    "camera_look_direction": [0.0, 4.0, 0.0],
                    "components": [
                        {
                            "component_id": "legacy-action",
                            "text": "Legacy action",
                            "position": [5.0, 4.0, 0.0],
                            "size_meters": 0.5,
                        }
                    ],
                },
                {
                    "step_id": "direction-forward",
                    "progress": 1.0,
                    "camera_look_direction": [2.0, 0.0, 0.0],
                },
            ],
        }

        restored = TourData.from_dict(payload)

        self.assertEqual(
            restored.steps[0].camera_target,
            (5.0, 1.0, 0.0),
        )
        self.assertEqual(
            restored.steps[1].camera_target,
            (11.0, 0.0, 0.0),
        )
        self.assertEqual(restored.steps[0].actions[0].component_id, "legacy-action")
        self.assertEqual(restored.steps[0].actions[0].size_points, 24.0)

    def test_legacy_text_size_migrates_to_points_and_fade_defaults(self) -> None:
        component = TourText3DData.from_dict(
            {
                "component_id": "legacy-text",
                "text": "Legacy",
                "position": [0.0, 0.0, 0.0],
                "size_meters": 0.5,
            }
        )

        self.assertEqual(component.size_points, 24.0)
        self.assertEqual(component.fade_delay_ms, DEFAULT_TEXT_FADE_DELAY_MS)
        self.assertEqual(component.fade_duration_ms, DEFAULT_TEXT_FADE_DURATION_MS)
        self.assertEqual(component.rotation_degrees, DEFAULT_TEXT_ROTATION_DEGREES)

    def test_text_action_defaults_and_fade_validation(self) -> None:
        component = TourText3DData(
            component_id="default-text",
            text="Default",
            position=(0.0, 0.0, 0.0),
        )

        self.assertEqual(component.size_points, DEFAULT_TEXT_SIZE_POINTS)
        self.assertEqual(component.fade_delay_ms, DEFAULT_TEXT_FADE_DELAY_MS)
        self.assertEqual(component.fade_duration_ms, DEFAULT_TEXT_FADE_DURATION_MS)
        self.assertEqual(component.rotation_degrees, DEFAULT_TEXT_ROTATION_DEGREES)
        with self.assertRaisesRegex(ValueError, "delay cannot be negative"):
            replace(component, fade_delay_ms=-1.0)
        with self.assertRaisesRegex(ValueError, "duration cannot be negative"):
            replace(component, fade_duration_ms=-1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            replace(component, rotation_degrees=(0.0, math.nan, 0.0))
        self.assertEqual(replace(component, fade_duration_ms=0.0).fade_duration_ms, 0.0)

    def test_nonvisual_action_defaults_and_project_round_trip(self) -> None:
        wait_action = TourWaitForKeyPressData(component_id="wait")
        idle_action = TourIdleCameraAnimationData(
            component_id="idle",
            pivot_point=(1.0, 2.0, 3.0),
        )
        speed_action = TourSpeedOverrideData(component_id="speed")
        step = TourStepData(
            step_id="interactive",
            progress=0.4,
            camera_target=(4.0, 5.0, 6.0),
            actions=(wait_action, idle_action, speed_action),
        )

        restored = TourStepData.from_dict(step.to_dict())

        self.assertEqual(restored, step)
        self.assertEqual(wait_action.input_token, DEFAULT_WAIT_INPUT_TOKEN)
        self.assertEqual(wait_action.easing, DEFAULT_SPEED_EASING)
        self.assertEqual(
            idle_action.radius_meters,
            DEFAULT_IDLE_CAMERA_RADIUS_METERS,
        )
        self.assertEqual(
            idle_action.cycle_duration_seconds,
            DEFAULT_IDLE_CAMERA_CYCLE_DURATION_SECONDS,
        )
        self.assertEqual(speed_action.speed, DEFAULT_SPEED_OVERRIDE)
        self.assertEqual(speed_action.easing, DEFAULT_SPEED_EASING)

    def test_floating_tooltip_defaults_have_lorem_body_and_standard_css(
        self,
    ) -> None:
        visible_body = (
            DEFAULT_TOOLTIP_HTML_BODY.removeprefix("<p>")
            .removesuffix("</p>")
        )

        self.assertEqual(
            visible_body,
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
            "eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut",
        )
        self.assertEqual(len(visible_body.split()), 20)
        for declaration in (
            "box-sizing: border-box;",
            "max-width: 360px;",
            "background: rgba(24, 27, 34, 0.94);",
            "border-radius: 8px;",
            "font-family: sans-serif;",
            "font-size: 14px;",
            "line-height: 1.5;",
        ):
            with self.subTest(declaration=declaration):
                self.assertIn(declaration, DEFAULT_TOOLTIP_STYLE)

    def test_floating_tooltip_project_round_trip_preserves_html_and_style(
        self,
    ) -> None:
        tooltip = TourFloatingTooltipData(
            component_id="tooltip-1",
            anchor_point=(1.0, 2.0, 3.0),
            tooltip_position=" RIGHT ",
            html_body=(
                '<article><h2>Kitchen</h2><img src="kitchen.webp" '
                'alt="Kitchen"></article>'
            ),
            style="background: #222; color: white; width: 24rem;",
        )
        step = TourStepData(
            step_id="tooltip-step",
            progress=0.4,
            camera_target=(4.0, 5.0, 6.0),
            actions=(tooltip,),
        )

        restored = TourStepData.from_dict(step.to_dict())

        self.assertEqual(restored, step)
        self.assertEqual(tooltip.tooltip_position, "right")
        self.assertEqual(
            tooltip.to_dict(),
            {
                "component_id": "tooltip-1",
                "type": "floating_tooltip",
                "anchor_point": [1.0, 2.0, 3.0],
                "tooltip_position": "right",
                "html_body": (
                    '<article><h2>Kitchen</h2><img src="kitchen.webp" '
                    'alt="Kitchen"></article>'
                ),
                "style": "background: #222; color: white; width: 24rem;",
            },
        )

    def test_floating_tooltip_legacy_optional_fields_use_safe_defaults(self) -> None:
        tooltip = TourFloatingTooltipData.from_dict(
            {
                "component_id": "legacy-tooltip",
                "type": "floating_tooltip",
                "anchor_point": [1.0, 2.0, 3.0],
            }
        )

        self.assertEqual(tooltip.tooltip_position, DEFAULT_TOOLTIP_POSITION)
        self.assertEqual(tooltip.html_body, DEFAULT_TOOLTIP_HTML_BODY)
        self.assertEqual(tooltip.style, DEFAULT_TOOLTIP_STYLE)

    def test_floating_tooltip_values_are_validated(self) -> None:
        for tooltip_position in TOUR_TOOLTIP_POSITION_OPTIONS:
            with self.subTest(tooltip_position=tooltip_position):
                tooltip = TourFloatingTooltipData(
                    component_id=f"tooltip-{tooltip_position}",
                    anchor_point=(0.0, 0.0, 0.0),
                    tooltip_position=tooltip_position,
                )
                self.assertEqual(tooltip.tooltip_position, tooltip_position)

        with self.assertRaisesRegex(ValueError, "Unsupported.*position"):
            TourFloatingTooltipData(
                component_id="invalid-position",
                anchor_point=(0.0, 0.0, 0.0),
                tooltip_position="center",
            )
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            TourFloatingTooltipData(
                component_id="empty-html",
                anchor_point=(0.0, 0.0, 0.0),
                html_body="   ",
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            TourFloatingTooltipData(
                component_id="invalid-anchor",
                anchor_point=(0.0, math.nan, 0.0),
            )

    def test_idle_camera_requires_wait_and_special_actions_are_singletons(self) -> None:
        idle_action = TourIdleCameraAnimationData(
            component_id="idle",
            pivot_point=(0.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(ValueError, "requires a wait"):
            TourStepData(
                step_id="orphan-idle",
                progress=0.5,
                camera_target=(0.0, 1.0, 0.0),
                actions=(idle_action,),
            )

        duplicate_waits = (
            TourWaitForKeyPressData(component_id="wait-1"),
            TourWaitForKeyPressData(component_id="wait-2"),
        )
        with self.assertRaisesRegex(ValueError, "only one"):
            TourStepData(
                step_id="duplicate-wait",
                progress=0.5,
                camera_target=(0.0, 1.0, 0.0),
                actions=duplicate_waits,
            )

    def test_action_values_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported.*easing"):
            TourSpeedOverrideData(
                component_id="speed",
                easing="bounce",
            )
        with self.assertRaisesRegex(TypeError, "easing function must be a string"):
            TourSpeedOverrideData(
                component_id="speed",
                easing=None,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "Unsupported.*easing"):
            TourWaitForKeyPressData(
                component_id="wait",
                easing="bounce",
            )
        with self.assertRaisesRegex(TypeError, "easing function must be a string"):
            TourWaitForKeyPressData(
                component_id="wait",
                easing=None,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "requires an input"):
            TourWaitForKeyPressData(component_id="wait", input_token="  ")
        for invalid_token in (None, "keyboard:a", "mouse:wheel", "key:space\0"):
            with (
                self.subTest(input_token=invalid_token),
                self.assertRaisesRegex(
                    (TypeError, ValueError),
                    "input token|Unsupported",
                ),
            ):
                TourWaitForKeyPressData(
                    component_id="wait",
                    input_token=invalid_token,  # type: ignore[arg-type]
                )
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            TourIdleCameraAnimationData(
                component_id="idle",
                pivot_point=(0.0, 0.0, 0.0),
                radius_meters=0.0,
            )

    def test_wait_input_tokens_are_canonicalized_and_cover_detected_keys(self) -> None:
        for raw_token, expected in (
            (" ANY ", "any"),
            ("Mouse:Left", "mouse:left"),
            ("KEY:SPACE", "key:space"),
            ("key:?", "key:?"),
            ("key_code:16777264", "key_code:16777264"),
        ):
            with self.subTest(input_token=raw_token):
                action = TourWaitForKeyPressData(
                    component_id="wait",
                    input_token=raw_token,
                )

                self.assertEqual(action.input_token, expected)

    def test_wait_easing_uses_speed_options_and_legacy_default(self) -> None:
        for easing in TOUR_SPEED_EASING_OPTIONS:
            with self.subTest(easing=easing):
                action = TourWaitForKeyPressData(
                    component_id="wait",
                    easing=easing.upper(),
                )

                self.assertEqual(action.easing, easing)
                self.assertEqual(
                    TourWaitForKeyPressData.from_dict(action.to_dict()),
                    action,
                )

        legacy_payload = {
            "component_id": "legacy-wait",
            "type": "wait_for_key_press",
            "input_token": "key:space",
        }
        restored = TourWaitForKeyPressData.from_dict(legacy_payload)

        self.assertEqual(restored.easing, DEFAULT_SPEED_EASING)
        self.assertEqual(restored.to_dict()["easing"], DEFAULT_SPEED_EASING)

    def test_wait_easing_takes_precedence_over_colocated_speed_easing(self) -> None:
        step = TourStepData(
            step_id="wait-and-speed",
            progress=0.5,
            camera_target=(0.0, 1.0, 0.0),
            actions=(
                TourSpeedOverrideData(
                    component_id="speed",
                    speed=2.0,
                    easing="linear",
                ),
                TourWaitForKeyPressData(
                    component_id="wait",
                    easing="ease_in_out_cubic",
                ),
            ),
        )

        self.assertEqual(effective_step_easing(step), "ease_in_out_cubic")
        self.assertEqual(
            effective_step_easing(replace(step, actions=(step.actions[0],))),
            "linear",
        )
        self.assertIsNone(effective_step_easing(replace(step, actions=())))
        with self.assertRaisesRegex(TypeError, "TourStepData"):
            effective_step_easing(None)  # type: ignore[arg-type]

    def test_legacy_step_speed_field_migrates_once_to_an_action(self) -> None:
        payload = _tour().steps[1].to_dict()
        payload["speed_override"] = 2.5

        restored = TourStepData.from_dict(payload)

        self.assertEqual(restored.speed_override, 2.5)
        self.assertEqual(len(restored.actions), 1)
        self.assertIsInstance(restored.actions[0], TourSpeedOverrideData)
        self.assertEqual(restored.actions[0].easing, "none")
        self.assertNotIn("speed_override", restored.to_dict())

        payload_with_new_action = restored.to_dict()
        payload_with_new_action["speed_override"] = 9.0
        restored_again = TourStepData.from_dict(payload_with_new_action)
        self.assertEqual(len(restored_again.actions), 1)
        self.assertEqual(restored_again.speed_override, 2.5)


# ### Project persistence tests ###
class TourPersistenceTests(unittest.TestCase):
    def test_project_save_and_load_round_trip_tours(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "tour-project.json"
            save_project(
                project_path,
                current_level_index=GROUND_LEVEL_INDEX,
                levels=create_default_levels(),
                tours=(_tour(),),
            )

            restored = load_project(project_path)

        self.assertEqual(restored.tours, (_tour(),))

    def test_project_save_and_load_preserves_wait_easing(self) -> None:
        wait_step = replace(
            _tour().steps[1],
            actions=(
                TourWaitForKeyPressData(
                    component_id="wait",
                    input_token="key:enter",
                    easing="smootherstep",
                ),
            ),
        )
        tour = replace(_tour(), steps=(wait_step,))
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "wait-easing-project.json"
            save_project(
                project_path,
                current_level_index=GROUND_LEVEL_INDEX,
                levels=create_default_levels(),
                tours=(tour,),
            )

            restored = load_project(project_path)

        self.assertEqual(restored.tours, (tour,))

    def test_project_save_and_load_preserves_floating_tooltips(self) -> None:
        tooltip = TourFloatingTooltipData(
            component_id="tooltip-persisted",
            anchor_point=(-2.0, 3.5, 1.25),
            tooltip_position="left",
            html_body="<strong>Original fixture</strong>",
            style="border: 1px solid currentColor;",
        )
        tooltip_tour = replace(
            _tour(),
            steps=(replace(_tour().steps[1], actions=(tooltip,)),),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "tooltip-project.json"
            save_project(
                project_path,
                current_level_index=GROUND_LEVEL_INDEX,
                levels=create_default_levels(),
                tours=(tooltip_tour,),
            )

            restored = load_project(project_path)

        self.assertEqual(restored.tours, (tooltip_tour,))

    def test_project_without_tours_loads_an_empty_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "legacy-project.json"
            save_project(
                project_path,
                current_level_index=GROUND_LEVEL_INDEX,
                levels=create_default_levels(),
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload.pop("tours")
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_project(project_path)

        self.assertEqual(restored.tours, ())


# ### Runtime manifest tests ###
class TourRuntimeManifestTests(unittest.TestCase):
    def test_runtime_tours_preserve_project_order(self) -> None:
        first = replace(_tour(), tour_id="z-tour", name="First")
        second = replace(_tour(), tour_id="a-tour", name="Second")

        exported = tours_to_runtime_dicts((first, second))

        self.assertEqual([tour["name"] for tour in exported], ["First", "Second"])

    def test_runtime_timing_defaults_are_exported_explicitly(self) -> None:
        tour = replace(
            _tour(),
            progress_speed=DEFAULT_TOUR_PROGRESS_SPEED,
            steps=(_tour().steps[1],),
        )

        exported = tours_to_runtime_dicts((tour,))[0]

        self.assertEqual(exported["progressSpeed"], DEFAULT_TOUR_PROGRESS_SPEED)
        self.assertNotIn("speedOverride", exported["steps"][0])
        self.assertEqual(exported["steps"][0]["actions"], [])

    def test_runtime_equal_progress_steps_preserve_authored_order(self) -> None:
        tour = _tour()
        first = replace(
            tour.steps[0],
            step_id="z-first-authored",
            progress=0.5,
            actions=(
                TourSpeedOverrideData(
                    component_id="first-speed",
                    speed=2.0,
                ),
            ),
        )
        second = replace(
            tour.steps[1],
            step_id="a-second-authored",
            progress=0.5,
            actions=(
                TourSpeedOverrideData(
                    component_id="second-speed",
                    speed=0.5,
                ),
            ),
        )

        exported = tour_to_runtime_dict(replace(tour, steps=(first, second)))

        self.assertEqual(
            [step["id"] for step in exported["steps"]],
            ["z-first-authored", "a-second-authored"],
        )
        self.assertEqual(exported["steps"][-1]["actions"][0]["speed"], 0.5)

    def test_manifest_exports_tours_in_gltf_coordinates(self) -> None:
        payload = build_runtime_scene_manifest(
            glb_name="house.glb",
            glb_bytes=b"glb",
            source_placements={},
            instance_placements=(),
            tours=(_tour(),),
        )

        exported = payload["tours"][0]
        self.assertEqual(exported["id"], "tour-1")
        self.assertEqual(exported["durationSeconds"], 18.0)
        self.assertEqual(exported["progressSpeed"], 1.5)
        self.assertEqual(exported["triggerPoint"], [10.0, 30.0, -20.0])
        self.assertEqual(
            exported["triggerArea"],
            {
                "center": [10.0, 30.0, -20.0],
                "sizeMeters": [4.0, 6.0],
                "horizontalAxes": "xz",
            },
        )
        self.assertEqual(exported["curve"]["type"], "catmullRom")
        self.assertEqual(exported["curve"]["curveType"], "catmullrom")
        self.assertIs(exported["curve"]["closed"], False)
        self.assertEqual(
            exported["curve"]["points"][1],
            [2.0, 4.0, -3.0],
        )
        self.assertEqual(
            [step["id"] for step in exported["steps"]],
            ["step-a", "step-b"],
        )
        self.assertEqual(
            exported["steps"][0]["cameraTarget"],
            [2.0, 0.0, 0.0],
        )
        self.assertNotIn("speedOverride", exported["steps"][0])
        self.assertNotIn("speedOverride", exported["steps"][1])
        action = exported["steps"][1]["actions"][0]
        self.assertNotIn("components", exported["steps"][1])
        self.assertEqual(action["type"], "text3d")
        self.assertEqual(action["position"], [7.0, 9.0, -8.0])
        self.assertEqual(action["sizePoints"], 18.0)
        self.assertEqual(action["fadeDelayMs"], 250.0)
        self.assertEqual(action["fadeDurationMs"], 1500.0)
        self.assertEqual(action["rotationOrder"], "XYZ")
        expected_rotation = (77.60231024, 50.33154592, 63.96655464)
        for actual, expected in zip(
            action["rotationDegrees"],
            expected_rotation,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected, places=7)
        speed_action = exported["steps"][1]["actions"][1]
        self.assertEqual(
            speed_action,
            {
                "id": "speed-1",
                "type": "speedOverride",
                "speed": 0.75,
                "easing": "smoothstep",
            },
        )

    def test_runtime_exports_wait_and_idle_actions_in_gltf_coordinates(self) -> None:
        wait_action = TourWaitForKeyPressData(
            component_id="wait",
            input_token="Mouse:Left",
            easing="ease_in_out_sine",
        )
        idle_action = TourIdleCameraAnimationData(
            component_id="idle",
            pivot_point=(2.0, 3.0, 4.0),
            radius_meters=0.12,
            cycle_duration_seconds=8.0,
        )
        step = TourStepData(
            step_id="wait-step",
            progress=0.5,
            camera_target=(0.0, 1.0, 2.0),
            actions=(wait_action, idle_action),
        )
        tour = replace(_tour(), steps=(step,))

        actions = tour_to_runtime_dict(tour)["steps"][0]["actions"]

        self.assertEqual(
            actions,
            [
                {
                    "id": "wait",
                    "type": "waitForKeyPress",
                    "input": "mouse:left",
                    "easing": "ease_in_out_sine",
                },
                {
                    "id": "idle",
                    "type": "idleCameraAnimation",
                    "pivotPoint": [2.0, 4.0, -3.0],
                    "radiusMeters": 0.12,
                    "cycleDurationSeconds": 8.0,
                },
            ],
        )

    def test_runtime_exports_floating_tooltip_in_gltf_coordinates(self) -> None:
        tooltip = TourFloatingTooltipData(
            component_id="tooltip-runtime",
            anchor_point=(2.0, 3.0, 4.0),
            tooltip_position="opposite",
            html_body="<p>Hover content</p>",
            style="max-width: 320px;",
        )
        tour = replace(
            _tour(),
            steps=(replace(_tour().steps[1], actions=(tooltip,)),),
        )

        action = tour_to_runtime_dict(tour)["steps"][0]["actions"][0]

        self.assertEqual(
            action,
            {
                "id": "tooltip-runtime",
                "type": "floatingTooltip",
                "anchorPoint": [2.0, 4.0, -3.0],
                "tooltipPosition": "opposite",
                "htmlBody": "<p>Hover content</p>",
                "style": "max-width: 320px;",
            },
        )


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
