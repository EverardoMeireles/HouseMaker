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
    level_world_to_image_xy,
)
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, VertexData
from housemaker.viewer import SceneObjectPlacementCandidate


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

    def test_direct_picker_is_unique_and_bound_to_operation_token(self) -> None:
        self.workspace.generation.placement_requested.emit("operation-one")
        _qt_application.processEvents()
        first_session = self.workspace._direct_object_placement_session
        self.assertIsNotNone(first_session)
        assert first_session is not None
        self.assertEqual(
            first_session.request_id,
            "operation-one",
        )
        self.assertTrue(self.workspace.viewer.is_object_placement_active)
        self.assertIs(
            self.workspace.workspace_tabs.currentWidget(),
            self.workspace.scene_3d_workspace,
        )

        self.workspace.generation.placement_requested.emit("operation-two")
        second_session = self.workspace._direct_object_placement_session
        self.assertIsNotNone(second_session)
        self.assertIsNot(second_session, first_session)
        assert second_session is not None
        self.assertEqual(
            second_session.request_id,
            "operation-two",
        )

        self.workspace.generation.operation_finished.emit("operation-one")
        self.assertIs(
            self.workspace._direct_object_placement_session,
            second_session,
        )

        self.workspace.generation.operation_finished.emit("operation-two")
        self.assertIsNone(self.workspace._direct_object_placement_session)
        self.assertFalse(self.workspace.viewer.is_object_placement_active)

    def test_completed_atlas_request_uses_actual_model_preview(self) -> None:
        model = _generated_model()
        self.workspace.generation.set_data(
            GenerationData(generated_objects=[_record("chair", None)])
        )
        with (
            patch.object(
                self.workspace.generation,
                "get_generated_object_model",
                return_value=model,
            ),
            patch.object(
                self.workspace.viewer,
                "begin_object_placement",
                return_value=True,
            ) as begin_placement,
        ):
            self.workspace._handle_atlas_object_place_requested("chair")

        begin_placement.assert_called_once()
        session = self.workspace._direct_object_placement_session
        assert session is not None
        self.assertEqual(begin_placement.call_args.args, (session.request_id,))
        self.assertIs(
            begin_placement.call_args.kwargs["preview_meshes"][0],
            model.mesh,
        )

    def test_stale_direct_candidate_cannot_place_the_active_request(self) -> None:
        level = _level(2)
        self.workspace.levels = [level]
        self.workspace.generation.placement_requested.emit("current-token")
        candidate = SceneObjectPlacementCandidate(
            level_index=2,
            world_positions=((1.0, 0.5, 0.25),),
        )

        with patch.object(
            self.workspace.generation,
            "set_active_object_placement",
            return_value=True,
        ) as setter:
            self.workspace._handle_direct_object_placement_selected(
                "wrong-token",
                candidate,
            )
            setter.assert_not_called()

            self.workspace._handle_direct_object_placement_selected(
                "current-token",
                candidate,
            )

        setter.assert_called_once()
        request_id, placement = setter.call_args.args
        self.assertEqual(request_id, "current-token")
        self.assertEqual(placement.level_index, level.index)
        self.assertAlmostEqual(
            placement.height_offset_meters,
            0.25 - build_level_base_z_lookup((level,))[level.index],
        )

    def test_shutdown_cancels_the_active_direct_picker(self) -> None:
        self.workspace.generation.placement_requested.emit("operation")
        self.assertTrue(self.workspace.viewer.is_object_placement_active)

        self.workspace.shutdown()

        self.assertIsNone(self.workspace._direct_object_placement_session)
        self.assertFalse(self.workspace.viewer.is_object_placement_active)

    def test_completed_object_request_finish_closes_only_its_picker(self) -> None:
        self.workspace.generation.placement_requested.emit("completed-token")
        session = self.workspace._direct_object_placement_session
        assert session is not None

        self.workspace.generation.placement_request_finished.emit(
            "stale-token"
        )
        self.assertIs(self.workspace._direct_object_placement_session, session)
        self.workspace.generation.placement_request_finished.emit(
            "completed-token"
        )

        self.assertIsNone(self.workspace._direct_object_placement_session)
        self.assertFalse(self.workspace.viewer.is_object_placement_active)

    def test_cancelling_picker_cancels_completed_object_request(self) -> None:
        self.workspace.generation.placement_requested.emit("completed-token")

        with patch.object(
            self.workspace.generation,
            "cancel_object_placement_request",
            return_value=True,
        ) as cancel_request:
            self.workspace.viewer.cancel_object_placement()

        cancel_request.assert_called_once_with("completed-token")
        self.assertIsNone(self.workspace._direct_object_placement_session)

    def test_generation_place_before_start_applies_anchor_to_whole_batch(self) -> None:
        level = _level(2)
        self.workspace.levels = [level]
        with patch.object(
            self.workspace.generation,
            "get_latest_generation_batch_placeable_ids",
            return_value=(),
        ):
            self.workspace._handle_generation_object_place_requested()
        session = self.workspace._direct_object_placement_session
        assert session is not None
        candidate = SceneObjectPlacementCandidate(
            level_index=2,
            world_positions=((0.0, 0.0, 0.1),),
        )
        self.workspace._handle_direct_object_placement_selected(
            session.request_id,
            candidate,
        )

        with patch.object(
            self.workspace.generation,
            "restore_placeable_object_placement",
            return_value=True,
        ) as restore:
            self.workspace._handle_generation_batch_started_for_placement(
                ("blob-a", "blob-b"),
            )

        self.assertEqual(restore.call_count, 2)
        first = restore.call_args_list[0].args[1]
        second = restore.call_args_list[1].args[1]
        self.assertNotEqual((first.image_x, first.image_y), (second.image_x, second.image_y))

    def test_generation_place_with_a_new_mask_stages_the_next_batch(self) -> None:
        stale_anchor = SceneObjectPlacementCandidate(
            level_index=2,
            world_positions=((9.0, 9.0, 0.0),),
        )
        self.workspace._pending_generation_placement_anchor = stale_anchor
        with (
            patch.object(
                self.workspace.generation,
                "get_latest_generation_batch_placeable_ids",
                return_value=("previous-object",),
            ),
            patch.object(
                self.workspace.generation,
                "has_unsubmitted_generation_mask",
                return_value=True,
            ),
            patch.object(
                self.workspace.viewer,
                "begin_object_placement",
                return_value=True,
            ),
        ):
            self.workspace._handle_generation_object_place_requested()

        session = self.workspace._direct_object_placement_session
        assert session is not None
        self.assertEqual(session.placeable_ids, ())
        self.assertTrue(session.accepts_next_generation_batch)
        self.assertIsNone(self.workspace._pending_generation_placement_anchor)

    def test_generation_place_prefers_the_active_batch_over_a_new_mask(self) -> None:
        with (
            patch.object(
                self.workspace.generation,
                "get_latest_generation_batch_placeable_ids",
                return_value=("active-a", "active-b"),
            ),
            patch.object(
                self.workspace.generation,
                "latest_generation_batch_has_active_members",
                return_value=True,
            ),
            patch.object(
                self.workspace.generation,
                "has_unsubmitted_generation_mask",
                return_value=True,
            ),
        ):
            self.workspace._handle_generation_object_place_requested()

        session = self.workspace._direct_object_placement_session
        assert session is not None
        self.assertEqual(session.placeable_ids, ("active-a", "active-b"))
        self.assertFalse(session.accepts_next_generation_batch)

    def test_generation_place_after_fully_placed_batch_stages_next(self) -> None:
        with (
            patch.object(
                self.workspace.generation,
                "get_latest_generation_batch_placeable_ids",
                return_value=("placed-a", "placed-b"),
            ),
            patch.object(
                self.workspace.generation,
                "latest_generation_batch_has_active_members",
                return_value=False,
            ),
            patch.object(
                self.workspace.generation,
                "has_unsubmitted_generation_mask",
                return_value=False,
            ),
            patch.object(
                self.workspace.generation,
                "latest_generation_batch_is_fully_placed",
                return_value=True,
            ),
        ):
            self.workspace._handle_generation_object_place_requested()

        session = self.workspace._direct_object_placement_session
        assert session is not None
        self.assertEqual(session.placeable_ids, ())
        self.assertTrue(session.accepts_next_generation_batch)

    def test_generation_place_after_start_targets_every_batch_member(self) -> None:
        level = _level(2)
        self.workspace.levels = [level]
        with patch.object(
            self.workspace.generation,
            "get_latest_generation_batch_placeable_ids",
            return_value=("blob-a", "blob-b"),
        ):
            self.workspace._handle_generation_object_place_requested()
        session = self.workspace._direct_object_placement_session
        assert session is not None
        self.assertEqual(session.placeable_ids, ("blob-a", "blob-b"))
        candidate = SceneObjectPlacementCandidate(
            level_index=2,
            world_positions=((-0.625, 0.0, 0.1), (0.625, 0.0, 0.1)),
        )
        with patch.object(
            self.workspace.generation,
            "restore_placeable_object_placement",
            return_value=True,
        ) as restore:
            self.workspace._handle_direct_object_placement_selected(
                session.request_id,
                candidate,
            )

        self.assertEqual(restore.call_count, 2)

    def test_failed_batch_member_reduces_the_armed_cube_preview(self) -> None:
        with patch.object(
            self.workspace.generation,
            "get_latest_generation_batch_placeable_ids",
            return_value=("blob-a", "blob-b"),
        ):
            self.workspace._handle_generation_object_place_requested()

        with (
            patch.object(
                self.workspace.generation,
                "get_latest_generation_batch_placeable_ids",
                return_value=("completed-b",),
            ),
            patch.object(
                self.workspace.viewer,
                "update_object_placement_preview_count",
                return_value=True,
            ) as update_preview,
        ):
            self.workspace._handle_object_placement_operation_finished("blob-a")

        session = self.workspace._direct_object_placement_session
        assert session is not None
        self.assertEqual(session.placeable_ids, ("completed-b",))
        update_preview.assert_called_once_with(1)

    def test_finished_batch_member_retargets_operation_id_to_object_id(self) -> None:
        with patch.object(
            self.workspace.generation,
            "get_latest_generation_batch_placeable_ids",
            return_value=("operation-a", "operation-b"),
        ):
            self.workspace._handle_generation_object_place_requested()

        with (
            patch.object(
                self.workspace.generation,
                "get_latest_generation_batch_placeable_ids",
                return_value=("object-a", "operation-b"),
            ),
            patch.object(
                self.workspace.viewer,
                "update_object_placement_preview_count",
            ) as update_preview,
        ):
            self.workspace._handle_object_placement_operation_finished(
                "operation-a"
            )

        session = self.workspace._direct_object_placement_session
        assert session is not None
        self.assertEqual(session.placeable_ids, ("object-a", "operation-b"))
        update_preview.assert_not_called()

    def test_failed_group_placement_rolls_back_every_changed_member(self) -> None:
        level = _level(2)
        self.workspace.levels = [level]
        original = GeneratedObjectPlacement(2, 10.0, 20.0)
        candidate = SceneObjectPlacementCandidate(
            level_index=2,
            world_positions=((-0.5, 0.0, 0.1), (0.5, 0.0, 0.1)),
        )
        with (
            patch.object(
                self.workspace.generation,
                "get_generated_object_ids",
                return_value=("object-a", "object-b"),
            ),
            patch.object(
                self.workspace.generation,
                "get_placeable_object_placement_state",
                side_effect=(("object-a", original), None),
            ),
            patch.object(
                self.workspace,
                "_capture_canvas_atlas_placements",
                side_effect=((), ()),
            ) as capture_atlas,
            patch.object(
                self.workspace.generation,
                "restore_placeable_object_placement",
                side_effect=(True, False, True),
            ) as restore,
            patch.object(
                self.workspace,
                "_record_canvas_undo_state",
            ) as record_undo,
        ):
            placed = self.workspace._commit_placeable_object_group(
                ("object-a", "object-b"),
                candidate,
            )

        self.assertFalse(placed)
        self.assertEqual(capture_atlas.call_count, 2)
        self.assertEqual(restore.call_count, 3)
        self.assertEqual(restore.call_args_list[-1].args, ("object-a", original))
        record_undo.assert_not_called()

    def test_group_placement_is_one_undoable_canvas_action(self) -> None:
        level = _level(2)
        self.workspace.levels = [level]
        self.workspace.generation.set_data(
            GenerationData(
                generated_objects=[
                    _record("object-a", None),
                    _record("object-b", None),
                ]
            )
        )
        candidate = SceneObjectPlacementCandidate(
            level_index=2,
            world_positions=((-0.5, 0.0, 0.1), (0.5, 0.0, 0.1)),
        )
        undo_count = len(self.workspace._canvas_undo_stack)

        self.assertTrue(
            self.workspace._commit_placeable_object_group(
                ("object-a", "object-b"),
                candidate,
            )
        )
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count + 1)
        self.assertTrue(
            all(
                self.workspace.generation.get_generated_object_placement(object_id)
                is not None
                for object_id in ("object-a", "object-b")
            )
        )

        self.workspace._handle_canvas_undo_requested()

        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count)
        self.assertTrue(
            all(
                self.workspace.generation.get_generated_object_placement(object_id)
                is None
                for object_id in ("object-a", "object-b")
            )
        )

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

    def test_canvas_removal_signal_routes_only_to_placement_controller(
        self,
    ) -> None:
        with (
            patch.object(
                self.workspace.generation,
                "remove_generated_object_placement",
                return_value=True,
            ) as remove_placement,
            patch.object(
                self.workspace.generation,
                "delete_generated_object",
            ) as delete_object,
        ):
            self.workspace.viewer.placed_object_removal_requested.emit("chair")

        remove_placement.assert_called_once_with("chair")
        delete_object.assert_not_called()

    def test_removed_placement_event_refreshes_canvas_preview(self) -> None:
        unplaced_record = _record("chair", None)

        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ) as schedule_refresh:
            self.workspace.generation.generated_object_placement_changed.emit(
                unplaced_record
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
        visible_placement = GeneratedObjectPlacement(
            3,
            150.0,
            25.0,
            height_offset_meters=1.75,
            rotation_degrees=(12.0, -25.0, 70.0),
        )
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
        self.assertEqual(placed_model.object_name, "Object visible")
        self.assertIs(placed_model.model, object_model)
        expected_x, expected_y = level_image_to_world_xy(
            target_level,
            visible_placement.image_x,
            visible_placement.image_y,
        )
        expected_z = (
            build_level_base_z_lookup(self.workspace.levels)[3]
            + visible_placement.height_offset_meters
        )
        self.assertEqual(
            placed_model.world_position,
            (expected_x, expected_y, expected_z),
        )
        self.assertEqual(
            placed_model.rotation_degrees,
            visible_placement.rotation_degrees,
        )
        self.assertEqual(
            placed_model.symmetric_preview_orientation,
            "vertical",
        )
        self.assertEqual(
            placed_model.symmetric_preview_plane_coordinate,
            0.25,
        )

    def test_gizmo_transform_converts_world_position_back_to_level_placement(
        self,
    ) -> None:
        ground = _level(2, height_meters=4.25)
        target_level = _level(
            3,
            scale=1.75,
            offset_x_meters=2.5,
            offset_y_meters=-3.0,
        )
        self.workspace.levels = [ground, target_level]
        original = GeneratedObjectPlacement(3, 40.0, 60.0)
        world_position = (8.25, -4.5, 6.0)
        rotation_degrees = (15.0, 35.0, -80.0)
        expected_image_x, expected_image_y = level_world_to_image_xy(
            target_level,
            world_position[0],
            world_position[1],
        )
        expected_height_offset = (
            world_position[2]
            - build_level_base_z_lookup(self.workspace.levels)[3]
        )

        with (
            patch.object(
                self.workspace.generation,
                "get_generated_object_placement",
                return_value=original,
            ),
            patch.object(
                self.workspace.generation,
                "update_generated_object_placement",
                return_value=True,
            ) as update_placement,
            patch.object(
                self.workspace,
                "_sync_atlas_object_texture_sources",
            ) as sync_atlas,
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
        ):
            self.workspace._handle_placed_object_transform_changed(
                "chair",
                world_position,
                rotation_degrees,
            )

        update_placement.assert_called_once()
        sync_atlas.assert_not_called()
        schedule_refresh.assert_not_called()
        object_id, placement = update_placement.call_args.args
        self.assertEqual(
            update_placement.call_args.kwargs,
            {"emit_change_signals": False},
        )
        self.assertEqual(object_id, "chair")
        self.assertIsInstance(placement, GeneratedObjectPlacement)
        self.assertEqual(placement.level_index, target_level.index)
        self.assertAlmostEqual(placement.image_x, expected_image_x)
        self.assertAlmostEqual(placement.image_y, expected_image_y)
        self.assertAlmostEqual(
            placement.height_offset_meters,
            expected_height_offset,
        )
        self.assertEqual(placement.rotation_degrees, rotation_degrees)

    def test_failed_gizmo_transform_restores_the_persisted_preview(self) -> None:
        level = _level(2)
        self.workspace.levels = [level]
        existing_placement = GeneratedObjectPlacement(2, 40.0, 60.0)

        with (
            patch.object(
                self.workspace.generation,
                "get_generated_object_placement",
                return_value=existing_placement,
            ),
            patch.object(
                self.workspace.generation,
                "update_generated_object_placement",
                return_value=False,
            ),
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
        ):
            self.workspace._handle_placed_object_transform_changed(
                "chair",
                (2.0, 3.0, 4.0),
                (10.0, 20.0, 30.0),
            )

        schedule_refresh.assert_called_once_with(preserve_camera=True)

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
