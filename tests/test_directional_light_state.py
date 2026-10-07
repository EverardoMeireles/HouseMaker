# ### Imports ###
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from housemaker.directional_light_state import (
    DirectionalLightData,
    create_directional_light,
    directional_lights_from_payload,
    directional_lights_to_runtime_dicts,
)
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.project_io import load_project, save_project


# ### Fixture helpers ###
def _directional_light(
    *,
    light_id: str = "directional-light-1",
    name: str = "Directional light 1",
    level_index: int | None = None,
) -> DirectionalLightData:
    return DirectionalLightData(
        light_id=light_id,
        name=name,
        position=(1.0, 2.0, 3.0),
        target=(4.0, 5.0, 6.0),
        color="#A0B1C2",
        intensity=2.5,
        cast_shadow=True,
        level_index=level_index,
    )


# ### Directional-light validation tests ###
class DirectionalLightValidationTests(unittest.TestCase):
    def test_record_normalizes_values_and_round_trips_project_payload(self) -> None:
        light = DirectionalLightData.from_dict(_directional_light().to_dict())

        self.assertEqual(light.light_id, "directional-light-1")
        self.assertEqual(light.name, "Directional light 1")
        self.assertEqual(light.position, (1.0, 2.0, 3.0))
        self.assertEqual(light.target, (4.0, 5.0, 6.0))
        self.assertEqual(light.color, "#a0b1c2")
        self.assertEqual(light.intensity, 2.5)
        self.assertIs(light.cast_shadow, True)
        self.assertIsNone(light.level_index)

    def test_record_rejects_an_invalid_direction_or_appearance(self) -> None:
        invalid_overrides = (
            {"target": (1.0, 2.0, 3.0)},
            {"position": (float("nan"), 0.0, 0.0)},
            {"color": "white"},
            {"intensity": -0.01},
            {"intensity": float("inf")},
            {"cast_shadow": 1},
            {"level_index": True},
            {"level_index": -1},
        )
        base_values = {
            "light_id": "directional-light-1",
            "name": "Directional light 1",
            "position": (1.0, 2.0, 3.0),
            "target": (4.0, 5.0, 6.0),
            "color": "#ffffff",
            "intensity": 1.0,
            "cast_shadow": False,
            "level_index": None,
        }

        for override in invalid_overrides:
            with self.subTest(override=override), self.assertRaises(
                (TypeError, ValueError)
            ):
                DirectionalLightData(**(base_values | override))

    def test_payload_loader_skips_malformed_and_duplicate_records(self) -> None:
        valid = _directional_light().to_dict()
        duplicate = dict(valid)
        duplicate["name"] = "Duplicate"

        loaded = directional_lights_from_payload(
            [
                valid,
                duplicate,
                {"light_id": "broken"},
                "not-a-light",
            ]
        )

        self.assertEqual(loaded, (_directional_light(),))
        self.assertEqual(directional_lights_from_payload({}), ())

    def test_optional_level_index_round_trips_only_when_present(self) -> None:
        unbound_payload = _directional_light().to_dict()
        bound = _directional_light(level_index=2)

        self.assertNotIn("level_index", unbound_payload)
        self.assertEqual(bound.to_dict()["level_index"], 2)
        self.assertEqual(
            DirectionalLightData.from_dict(bound.to_dict()),
            bound,
        )

    def test_immutable_edits_preserve_direction_and_original_record(self) -> None:
        light = _directional_light()

        translated = light.translated((5.0, -2.0, 1.0))
        moved = light.with_position((10.0, 20.0, 30.0))

        self.assertEqual(light.position, (1.0, 2.0, 3.0))
        self.assertEqual(translated.position, (6.0, 0.0, 4.0))
        self.assertEqual(translated.target, (9.0, 3.0, 7.0))
        self.assertEqual(moved.position, (10.0, 20.0, 30.0))
        self.assertEqual(moved.target, (13.0, 23.0, 33.0))
        self.assertEqual(
            tuple(
                moved.target[index] - moved.position[index]
                for index in range(3)
            ),
            (3.0, 3.0, 3.0),
        )

    def test_intensity_edits_use_raw_tenths_and_clamp_at_zero(self) -> None:
        light = _directional_light().with_intensity(1.0)

        self.assertEqual(light.adjust_intensity(1).intensity, 1.1)
        self.assertEqual(light.adjust_intensity(-20).intensity, 0.0)
        self.assertEqual(light.adjust_intensity(0.5).intensity, 1.05)
        self.assertEqual(light.intensity, 1.0)
        with self.assertRaises(TypeError):
            light.adjust_intensity(True)
        with self.assertRaises(ValueError):
            light.adjust_intensity(float("inf"))


# ### Directional-light factory tests ###
class DirectionalLightFactoryTests(unittest.TestCase):
    def test_factory_uses_the_first_available_incremental_id_and_name(self) -> None:
        existing = (
            _directional_light(
                light_id="directional-light-1",
                name="Directional light 1",
            ),
            _directional_light(
                light_id="directional-light-3",
                name="Directional light 3",
            ),
        )

        created = create_directional_light(
            existing,
            position=(10.0, 20.0, 30.0),
            target=(10.0, 21.0, 30.0),
        )

        self.assertEqual(created.light_id, "directional-light-2")
        self.assertEqual(created.name, "Directional light 2")
        self.assertEqual(created.position, (10.0, 20.0, 30.0))
        self.assertEqual(created.target, (10.0, 21.0, 30.0))
        self.assertEqual(created.color, "#ffffff")
        self.assertEqual(created.intensity, 1.0)
        self.assertIs(created.cast_shadow, False)

    def test_factory_defaults_to_a_straight_down_target(self) -> None:
        created = create_directional_light(
            (),
            position=(10.0, 20.0, 30.0),
            level_index=2,
        )

        self.assertEqual(created.position, (10.0, 20.0, 30.0))
        self.assertEqual(created.target, (10.0, 20.0, 29.0))
        self.assertEqual(created.level_index, 2)


# ### Runtime coordinate conversion tests ###
class DirectionalLightRuntimeTests(unittest.TestCase):
    def test_runtime_payload_converts_z_up_points_to_gltf_y_up(self) -> None:
        runtime = directional_lights_to_runtime_dicts(
            (_directional_light(level_index=2),)
        )

        self.assertEqual(
            runtime,
            [
                {
                    "id": "directional-light-1",
                    "name": "Directional light 1",
                    "position": [1.0, 3.0, -2.0],
                    "target": [4.0, 6.0, -5.0],
                    "color": "#a0b1c2",
                    "intensity": 2.5,
                    "castShadow": True,
                }
            ],
        )


# ### Project persistence tests ###
class DirectionalLightProjectPersistenceTests(unittest.TestCase):
    def test_project_save_and_load_preserve_directional_lights(self) -> None:
        light = _directional_light()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.json"
            save_project(
                path=project_path,
                current_level_index=GROUND_LEVEL_INDEX,
                levels=create_default_levels(),
                directional_lights=(light,),
            )

            raw_project = json.loads(project_path.read_text(encoding="utf-8"))
            loaded_project = load_project(project_path)

        self.assertEqual(raw_project["directional_lights"], [light.to_dict()])
        self.assertEqual(loaded_project.directional_lights, (light,))

    def test_project_round_trip_preserves_editor_level_association(self) -> None:
        light = _directional_light(level_index=2)
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.json"
            save_project(
                path=project_path,
                current_level_index=GROUND_LEVEL_INDEX,
                levels=create_default_levels(),
                directional_lights=(light,),
            )

            loaded_project = load_project(project_path)

        self.assertEqual(loaded_project.directional_lights, (light,))

    def test_project_without_directional_lights_loads_an_empty_collection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.json"
            save_project(
                path=project_path,
                current_level_index=GROUND_LEVEL_INDEX,
                levels=create_default_levels(),
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload.pop("directional_lights")
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded_project = load_project(project_path)

        self.assertEqual(loaded_project.directional_lights, ())


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
