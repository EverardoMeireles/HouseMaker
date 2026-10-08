# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.directional_light_state import DirectionalLightData
from housemaker.glb import GeneratedModel
from housemaker.main import BlueprintWorkspace
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.viewer import GlbViewerWidget

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _directional_light(
    *,
    light_id: str = "directional-light-1",
    name: str = "Directional light 1",
) -> DirectionalLightData:
    return DirectionalLightData(
        light_id=light_id,
        name=name,
        position=(1.0, 2.0, 3.0),
        target=(4.0, 5.0, 6.0),
    )


def _generated_box_model() -> GeneratedModel:
    mesh = trimesh.creation.box()
    return GeneratedModel(
        mesh=mesh,
        scene=trimesh.Scene(mesh),
        glb_bytes=b"",
    )


def _process_events() -> None:
    _qt_application.processEvents()


# ### Main-window integration tests ###
class DirectionalLightMainIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        self.workspace.resize(1400, 850)
        self.workspace.show()
        _process_events()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _process_events()
        self._temporary_directory.cleanup()

    def test_add_directional_light_button_is_immediately_below_doors(
        self,
    ) -> None:
        side_layout = self.workspace.doors_group.parentWidget().layout()
        self.assertIsNotNone(side_layout)
        assert side_layout is not None

        self.assertEqual(
            side_layout.indexOf(self.workspace.add_directional_light_button),
            side_layout.indexOf(self.workspace.doors_group) + 1,
        )
        self.assertEqual(
            self.workspace.add_directional_light_button.text(),
            "Add directional light",
        )

    def test_button_arms_point_placement_without_creating_a_light(self) -> None:
        with (
            patch.object(
                self.workspace.canvas,
                "start_directional_light_placement",
                return_value=True,
            ) as start_canvas_placement,
            patch.object(
                self.workspace.canvas,
                "is_directional_light_placement_active",
                return_value=True,
            ),
            patch.object(
                self.workspace.viewer,
                "begin_directional_light_placement",
                return_value=True,
            ) as start_viewer_placement,
        ):
            self.workspace.add_directional_light_button.click()

        self.assertTrue(self.workspace.add_directional_light_button.isChecked())
        self.assertEqual(self.workspace.directional_lights, ())
        start_canvas_placement.assert_called_once_with()
        start_viewer_placement.assert_called_once()

    def test_scene_drops_create_incremental_straight_down_lights(self) -> None:
        positions = ((1.0, 2.0, 3.0), (7.0, 8.0, 9.0))

        for position in positions:
            self.workspace._handle_viewer_directional_light_placement_requested(
                position
            )

        self.assertEqual(
            [light.light_id for light in self.workspace.directional_lights],
            ["directional-light-1", "directional-light-2"],
        )
        self.assertEqual(
            [light.name for light in self.workspace.directional_lights],
            ["Directional light 1", "Directional light 2"],
        )
        self.assertEqual(
            [light.position for light in self.workspace.directional_lights],
            list(positions),
        )
        self.assertEqual(
            [light.target for light in self.workspace.directional_lights],
            [(1.0, 2.0, 2.0), (7.0, 8.0, 8.0)],
        )
        self.assertEqual(
            [light.level_index for light in self.workspace.directional_lights],
            [self.workspace.current_level.index] * 2,
        )
        self.assertIn(
            "Current lights: 2",
            self.workspace.add_directional_light_button.toolTip(),
        )

    def test_delete_from_viewer_removes_only_the_selected_light(self) -> None:
        first_light = _directional_light()
        second_light = _directional_light(
            light_id="directional-light-2",
            name="Directional light 2",
        )
        self.workspace.directional_lights = (first_light, second_light)
        self.workspace._selected_directional_light_id = first_light.light_id
        self.workspace._sync_directional_light_views()

        self.workspace.viewer._handle_view_delete_requested()

        self.assertEqual(self.workspace.directional_lights, (second_light,))
        self.assertIsNone(self.workspace._selected_directional_light_id)
        self.assertIsNone(
            self.workspace.viewer.get_selected_directional_light_id()
        )
        self.assertIsNone(
            self.workspace.canvas.get_selected_directional_light_id()
        )

    def test_wheel_steps_use_raw_r3f_intensity_units(self) -> None:
        light = _directional_light()
        self.workspace.directional_lights = (light,)

        self.workspace._handle_directional_light_intensity_step_requested(
            light.light_id,
            2,
        )

        self.assertEqual(self.workspace.directional_lights[0].intensity, 1.2)

    def test_project_apply_and_save_preserve_directional_lights(self) -> None:
        light = _directional_light()

        with patch.object(
            self.workspace.viewer,
            "set_directional_lights",
        ) as set_viewer_lights:
            self.workspace._apply_project_state(
                levels=create_default_levels(),
                current_level_index=GROUND_LEVEL_INDEX,
                directional_lights=(light,),
            )

        self.assertEqual(self.workspace.directional_lights, (light,))
        set_viewer_lights.assert_called_once_with((light,))

        save_path = Path(self._temporary_directory.name) / "light-project.json"
        with (
            patch(
                "housemaker.main.QFileDialog.getSaveFileName",
                return_value=(str(save_path), "JSON Files (*.json)"),
            ),
            patch("housemaker.main.save_project") as save_project,
            patch("housemaker.main.QMessageBox.information"),
        ):
            self.workspace._handle_save_clicked()

        save_project.assert_called_once()
        self.assertEqual(
            save_project.call_args.kwargs["directional_lights"],
            (light,),
        )

    def test_export_passes_lights_to_the_runtime_scene_manifest(self) -> None:
        light = _directional_light()
        self.workspace.directional_lights = (light,)
        export_path = Path(self._temporary_directory.name) / "light-scene.glb"
        manifest_path = export_path.with_suffix(".json")
        generated_model = object()

        with (
            patch.object(
                self.workspace,
                "_sync_atlas_object_texture_sources",
            ),
            patch.object(
                self.workspace,
                "_show_unpacked_scene_texture_export_error",
                return_value=False,
            ),
            patch(
                "housemaker.main.QFileDialog.getSaveFileName",
                return_value=(str(export_path), "GLB Files (*.glb)"),
            ),
            patch.object(
                self.workspace,
                "_build_model_with_stable_dependencies",
                return_value=(generated_model, ("revision",)),
            ),
            patch.object(
                self.workspace,
                "_build_export_placed_models",
                return_value=((), ()),
            ),
            patch("housemaker.main.export_glb_file", return_value=export_path),
            patch(
                "housemaker.main.write_runtime_scene_manifest",
                return_value=manifest_path,
            ) as write_manifest,
            patch.object(self.workspace, "_ensure_viewer_preview_current"),
            patch("housemaker.main.QMessageBox.information"),
        ):
            self.workspace._handle_glb_export_clicked()

        write_manifest.assert_called_once_with(
            export_path,
            source_placements={},
            instance_placements=(),
            tours=(),
            door_body_reconstructions=(),
            directional_lights=(light,),
        )


# ### Viewer overlay tests ###
class DirectionalLightViewerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.viewer = GlbViewerWidget()

    def tearDown(self) -> None:
        self.viewer.close()
        self.viewer.deleteLater()
        _process_events()

    def test_marker_and_arrow_survive_a_model_rebuild(self) -> None:
        light = _directional_light()
        self.viewer.set_directional_lights((light,))

        initial_items = tuple(self.viewer._directional_light_overlay_items)
        self.assertEqual(len(initial_items), 2)
        self.assertTrue(
            all(
                item._housemaker_directional_light_id == light.light_id
                for item in initial_items
            )
        )
        np.testing.assert_allclose(initial_items[0].pos, (light.position,))

        self.viewer.set_model(_generated_box_model())

        rebuilt_items = tuple(self.viewer._directional_light_overlay_items)
        self.assertEqual(len(rebuilt_items), 2)
        self.assertNotEqual(rebuilt_items, initial_items)
        self.assertTrue(all(item in self.viewer.view.items for item in rebuilt_items))
        self.assertTrue(
            all(
                item._housemaker_directional_light_id == light.light_id
                for item in rebuilt_items
            )
        )
        np.testing.assert_allclose(rebuilt_items[0].pos, (light.position,))


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
