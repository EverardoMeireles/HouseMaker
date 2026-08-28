# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import trimesh
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid as is_valid_qt_object

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_state import (
    GeneratedObjectPlacement,
    GeneratedObjectRecord,
    GenerationData,
)
from housemaker.glb import GeneratedModel, PlacedGeneratedModel
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
)
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, VertexData


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _generated_model(
    extents: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> GeneratedModel:
    mesh = trimesh.creation.box(extents=extents)
    scene = trimesh.Scene(mesh.copy())
    return GeneratedModel(
        mesh=mesh,
        scene=scene,
        glb_bytes=scene.export(file_type="glb"),
    )


def _level(
    index: int,
    *,
    include_in_export: bool = True,
    height_meters: float = 3.0,
    scale: float = 1.0,
    offset_x_meters: float = 0.0,
    offset_y_meters: float = 0.0,
) -> LevelData:
    vertex_data = VertexData()
    first = vertex_data.add_vertex(0.0, 0.0)
    second = vertex_data.add_vertex(200.0, 100.0)
    vertex_data.add_edge(first.id, second.id)
    return LevelData(
        index=index,
        name=f"Level {index}",
        height_meters=height_meters,
        scale=scale,
        offset_x_meters=offset_x_meters,
        offset_y_meters=offset_y_meters,
        vertex_data=vertex_data,
        image_size_pixels=(200.0, 100.0),
        include_in_export=include_in_export,
    )


def _record(
    object_id: str,
    placement: GeneratedObjectPlacement | None,
    *,
    pipeline: dict[str, object] | None = None,
) -> GeneratedObjectRecord:
    return GeneratedObjectRecord(
        object_id=object_id,
        frame_index=0,
        object_name=f"Object {object_id}",
        pipeline={} if pipeline is None else pipeline,
        provider="meshy",
        provider_task_id=f"task-{object_id}",
        asset_path=f"{object_id}.glb",
        placement=placement,
    )


# ### Main-workspace placement tests ###
class GeneratedObjectPlacementMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        settings_path = Path(self.temporary_directory.name) / "settings.json"
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(settings_path)
        )
        self.workspace.resize(1200, 760)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def test_modeless_dialog_is_unique_and_bound_to_operation_token(self) -> None:
        self.workspace.generation.placement_requested.emit("operation-one")
        _qt_application.processEvents()
        first_dialog = self.workspace._object_placement_dialog
        self.assertIsNotNone(first_dialog)
        assert first_dialog is not None
        self.assertFalse(first_dialog.isModal())
        self.assertEqual(
            self.workspace._object_placement_operation_id,
            "operation-one",
        )

        self.workspace.generation.placement_requested.emit("operation-two")
        second_dialog = self.workspace._object_placement_dialog
        self.assertIsNotNone(second_dialog)
        self.assertIsNot(second_dialog, first_dialog)
        self.assertEqual(
            self.workspace._object_placement_operation_id,
            "operation-two",
        )
        if is_valid_qt_object(first_dialog):
            self.assertFalse(first_dialog.isVisible())

        self.workspace.generation.operation_finished.emit("operation-one")
        self.assertIs(
            self.workspace._object_placement_dialog,
            second_dialog,
        )

        self.workspace.generation.operation_finished.emit("operation-two")
        self.assertIsNone(self.workspace._object_placement_dialog)
        self.assertIsNone(self.workspace._object_placement_operation_id)
        if second_dialog is not None and is_valid_qt_object(second_dialog):
            self.assertFalse(second_dialog.isVisible())

    def test_stale_dialog_or_token_cannot_place_the_active_operation(self) -> None:
        self.workspace.generation.placement_requested.emit("old-token")
        old_dialog = self.workspace._object_placement_dialog
        assert old_dialog is not None
        self.workspace.generation.placement_requested.emit("current-token")
        current_dialog = self.workspace._object_placement_dialog
        assert current_dialog is not None
        placement = GeneratedObjectPlacement(2, 80.0, 40.0)

        with patch.object(
            self.workspace.generation,
            "set_active_object_placement",
            return_value=True,
        ) as setter:
            self.workspace._handle_object_placement_selected(
                old_dialog,
                "old-token",
                placement,
            )
            self.workspace._handle_object_placement_selected(
                current_dialog,
                "wrong-token",
                placement,
            )
            setter.assert_not_called()

            self.workspace._handle_object_placement_selected(
                current_dialog,
                "current-token",
                placement,
            )

        setter.assert_called_once_with("current-token", placement)

    def test_shutdown_closes_the_active_placement_dialog(self) -> None:
        self.workspace.generation.placement_requested.emit("operation")
        dialog = self.workspace._object_placement_dialog
        assert dialog is not None

        self.workspace.shutdown()

        self.assertIsNone(self.workspace._object_placement_dialog)
        self.assertIsNone(self.workspace._object_placement_operation_id)
        if is_valid_qt_object(dialog):
            self.assertFalse(dialog.isVisible())

    def test_completed_object_request_finish_closes_only_its_dialog(self) -> None:
        self.workspace.generation.placement_requested.emit("completed-token")
        dialog = self.workspace._object_placement_dialog
        assert dialog is not None

        self.workspace.generation.placement_request_finished.emit(
            "stale-token"
        )
        self.assertIs(self.workspace._object_placement_dialog, dialog)
        self.workspace.generation.placement_request_finished.emit(
            "completed-token"
        )

        self.assertIsNone(self.workspace._object_placement_dialog)
        self.assertIsNone(self.workspace._object_placement_operation_id)
        if is_valid_qt_object(dialog):
            self.assertFalse(dialog.isVisible())

    def test_closing_dialog_cancels_its_completed_object_request(self) -> None:
        self.workspace.generation.placement_requested.emit("completed-token")
        dialog = self.workspace._object_placement_dialog
        assert dialog is not None

        with patch.object(
            self.workspace.generation,
            "cancel_object_placement_request",
            return_value=True,
        ) as cancel_request:
            dialog.reject()
            _qt_application.processEvents()

        cancel_request.assert_called_once_with("completed-token")
        self.assertIsNone(self.workspace._object_placement_dialog)

    def test_completed_object_placement_change_refreshes_canvas_preview(
        self,
    ) -> None:
        record = _record(
            "chair",
            GeneratedObjectPlacement(2, 75.0, 30.0),
        )

        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ) as schedule_refresh:
            self.workspace.generation.generated_object_placement_changed.emit(
                record
            )

        schedule_refresh.assert_called_once_with(preserve_camera=True)

    def test_build_model_places_visible_record_using_current_level_transform(
        self,
    ) -> None:
        ground = _level(2, height_meters=4.25)
        target_level = _level(
            3,
            scale=1.75,
            offset_x_meters=2.5,
            offset_y_meters=-3.0,
        )
        hidden_level = _level(4, include_in_export=False)
        self.workspace.levels = [ground, target_level, hidden_level]
        visible_placement = GeneratedObjectPlacement(3, 150.0, 25.0)
        generation_data = GenerationData(
            generated_objects=[
                _record(
                    "visible",
                    visible_placement,
                    pipeline={
                        "symmetric_division": {
                            "version": 1,
                            "orientation": "vertical",
                            "kept_side": "left",
                            "plane_coordinate": 0.25,
                            "texture_content_half": "left",
                        }
                    },
                ),
                _record(
                    "hidden",
                    GeneratedObjectPlacement(4, 20.0, 30.0),
                ),
                _record(
                    "missing-level",
                    GeneratedObjectPlacement(7, 10.0, 15.0),
                ),
                _record("unplaced", None),
            ]
        )
        base_model = _generated_model((2.0, 2.0, 1.0))
        object_model = _generated_model((0.5, 0.75, 1.25))
        composed_model = _generated_model((3.0, 3.0, 3.0))

        with (
            patch(
                "housemaker.main.convert_to_glb",
                return_value=base_model,
            ),
            patch.object(
                self.workspace.generation,
                "get_data",
                return_value=generation_data,
            ),
            patch.object(
                self.workspace.generation,
                "get_generated_object_model",
                return_value=object_model,
            ) as model_getter,
            patch(
                "housemaker.main.compose_placed_generated_models",
                return_value=composed_model,
            ) as compose,
        ):
            result = self.workspace._build_generated_model(None)

        self.assertIs(result, composed_model)
        model_getter.assert_called_once_with("visible")
        compose.assert_called_once()
        self.assertIs(compose.call_args.args[0], base_model)
        placed_models = compose.call_args.args[1]
        self.assertEqual(len(placed_models), 1)
        placed_model = placed_models[0]
        self.assertIsInstance(placed_model, PlacedGeneratedModel)
        self.assertEqual(placed_model.object_id, "visible")
        self.assertIs(placed_model.model, object_model)
        expected_x, expected_y = level_image_to_world_xy(
            target_level,
            visible_placement.image_x,
            visible_placement.image_y,
        )
        expected_z = build_level_base_z_lookup(self.workspace.levels)[3]
        self.assertEqual(
            placed_model.world_position,
            (expected_x, expected_y, expected_z),
        )
        self.assertEqual(
            placed_model.symmetric_preview_orientation,
            "vertical",
        )
        self.assertEqual(
            placed_model.symmetric_preview_plane_coordinate,
            0.25,
        )

    def test_composed_scene_contains_the_placed_object(self) -> None:
        level = _level(2)
        self.workspace.levels = [level]
        placement = GeneratedObjectPlacement(2, 175.0, 50.0)
        data = GenerationData(
            generated_objects=[_record("chair", placement)]
        )
        base_model = _generated_model((1.0, 1.0, 1.0))
        object_model = _generated_model((0.5, 0.5, 0.5))
        target_x, target_y = level_image_to_world_xy(
            level,
            placement.image_x,
            placement.image_y,
        )

        with (
            patch(
                "housemaker.main.convert_to_glb",
                return_value=base_model,
            ),
            patch.object(
                self.workspace.generation,
                "get_data",
                return_value=data,
            ),
            patch.object(
                self.workspace.generation,
                "get_generated_object_model",
                return_value=object_model,
            ),
        ):
            result = self.workspace._build_generated_model(None)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result.scene.geometry), 2)
        self.assertGreaterEqual(result.mesh.bounds[1][0], target_x + 0.24)
        self.assertGreaterEqual(result.mesh.bounds[1][1], target_y + 0.24)

    def test_placed_object_events_refresh_canvas_preview(self) -> None:
        placed_record = _record(
            "placed",
            GeneratedObjectPlacement(2, 50.0, 60.0),
        )
        unplaced_record = _record("unplaced", None)
        model = _generated_model()

        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ) as refresh:
            self.workspace._handle_generated_object_completed_for_canvas(
                unplaced_record,
                model,
            )
            refresh.assert_not_called()

            self.workspace._handle_generated_object_completed_for_canvas(
                placed_record,
                model,
            )
            refresh.assert_called_once_with(preserve_camera=False)

            refresh.reset_mock()
            self.workspace._handle_generated_object_changed_for_canvas(
                placed_record,
                model,
            )
            refresh.assert_called_once_with(preserve_camera=True)

            refresh.reset_mock()
            self.workspace._handle_generated_object_deleted_for_canvas(
                "placed"
            )
            refresh.assert_called_once_with(preserve_camera=True)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
