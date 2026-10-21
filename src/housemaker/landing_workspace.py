"""Landing authoring and shared stair/landing edit transactions."""

# ### Imports ###
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import trimesh
from PySide6.QtCore import QTimer
from shapely.geometry import Point, Polygon

from housemaker.glb import (
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
    GeneratedModel,
    build_stair_meshes,
)
from housemaker.landing_generation import LandingMeshGenerationResult
from housemaker.landing_geometry import (
    LANDING_IMAGE_TO_LOCAL_Z_UP_TRANSFORM,
    build_landing_mesh,
    build_landing_meshes,
    create_landing,
    landing_world_transform,
    prepare_generated_landing,
)
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_world_to_image_xy,
)
from housemaker.models import LandingAttachment, LandingData, StairData


# ### Undo snapshots ###
@dataclass(frozen=True)
class CanvasLandingUndoState:
    """Landings and their stair attachments before one authoring operation."""

    landings_by_level: tuple[tuple[int, tuple[LandingData, ...]], ...]
    stairs: tuple[StairData, ...]
    selected_id: str | None


# ### Workspace integration ###
class LandingWorkspaceMixin:
    """Keep landing controls outside the main window's general UI code."""

    def _init_landing_state(self) -> None:
        self._selected_landing_id: str | None = None
        self._active_landing_undo_state: CanvasLandingUndoState | None = None
        self._landing_edit_had_pending_mesh_update = False
        self._pending_landing_mesh_update = False
        self._is_syncing_landing_views = False
        self._viewer_landings_by_level_index: dict[int, tuple[LandingData, ...]] = {}
        self._landing_mesh_update_timer = QTimer(self)
        self._landing_mesh_update_timer.setSingleShot(True)
        self._landing_mesh_update_timer.setInterval(
            max(1, round(self._mesh_edit_update_delay_seconds * 1000))
        )
        self._landing_mesh_update_timer.timeout.connect(
            self._commit_landing_mesh_update
        )

    def _connect_landing_controls(self) -> None:
        self.add_landing_button.clicked.connect(self._handle_add_landing_clicked)
        self.canvas.landing_placement_requested.connect(self._place_canvas_landing)
        self.viewer.landing_placement_requested.connect(self._place_world_landing)
        for view in (self.canvas, self.viewer):
            view.landing_placement_cancelled.connect(self._cancel_landing_placement)
            view.landing_selection_requested.connect(self._select_landing)
            view.landing_edit_started.connect(self._begin_landing_edit)
            view.landing_transform_preview_changed.connect(self._preview_landing_edit)
            view.landing_edit_finished.connect(self._finish_landing_edit)
            view.landing_edit_cancelled.connect(self._cancel_landing_edit)
            view.landing_generation_requested.connect(self._open_landing_generation)
            view.landing_deletion_requested.connect(self._delete_landing)
        self.generation.landing_generation_completed.connect(
            self._apply_generated_landing
        )

    def _find_landing(self, landing_id: str):
        for level in self.levels:
            for index, landing in enumerate(level.landings):
                if landing.landing_id == landing_id:
                    return level, index, landing
        return None

    def _capture_landing_undo(self) -> CanvasLandingUndoState:
        return CanvasLandingUndoState(
            tuple((level.index, tuple(level.landings)) for level in self.levels),
            tuple(self.stairs),
            self._selected_landing_id,
        )

    def _reset_viewer_landing_snapshots(self) -> None:
        self._viewer_landings_by_level_index = {
            level.index: tuple(level.landings) for level in self.levels
        }

    def _sync_landing_views(self) -> None:
        """Publish lightweight proxies and footprints, never rebuild the house."""

        if (
            not hasattr(self, "viewer")
            or not hasattr(self, "canvas")
            or self._is_syncing_landing_views
        ):
            return
        if (
            self._selected_landing_id
            and self._find_landing(self._selected_landing_id) is None
        ):
            self._selected_landing_id = None
        self._is_syncing_landing_views = True
        try:
            self.viewer.set_landing_context(self.levels, self._selected_landing_id)
            selected = (
                self._find_landing(self._selected_landing_id)
                if self._selected_landing_id
                else None
            )
            self.canvas.select_landing(
                self._selected_landing_id
                if selected is not None and selected[0] is self.current_level
                else None
            )
            self.canvas.update()
        finally:
            self._is_syncing_landing_views = False

    # ### Placement and selection ###
    def _handle_add_landing_clicked(self) -> None:
        self._commit_landing_mesh_update()
        self.canvas.cancel_stair_placement()
        if self._editing_stair_index is not None:
            self.viewer.set_selected_canvas_stair_part_ids(())
            self._discard_staged_stair_edit(clear_selection=True)
        self.canvas.cancel_window_placement()
        self.canvas.begin_landing_placement()
        self.viewer.begin_landing_placement(
            fallback_plane_z=build_level_base_z_lookup(self.levels).get(
                self.current_level.index, 0.0
            )
        )
        self.stair_status_label.setText(
            "Click the plan or a floor in the 3D view to place a landing. "
            "Right-click or Escape cancels."
        )

    def _cancel_landing_placement(self) -> None:
        self.canvas.cancel_landing_placement()
        self.viewer.cancel_landing_placement()

    def _place_canvas_landing(self, image_x: float, image_y: float) -> None:
        self._place_landing(self.current_level, image_x, image_y)

    def _place_world_landing(self, point: object) -> None:
        level = self.current_level
        if isinstance(point, dict):
            level = self._get_level_by_index(point.get("level_index")) or level
            point = point["point"]
        elif len(point) >= 3:
            floor_heights = build_level_base_z_lookup(self.levels)
            matching = [
                candidate
                for candidate in self.levels
                if abs(floor_heights[candidate.index] - float(point[2])) <= 1e-5
            ]
            if matching:
                level = next(
                    (
                        candidate
                        for candidate in matching
                        if candidate.index == level.index
                    ),
                    matching[0],
                )
        image_x, image_y = level_world_to_image_xy(
            level, float(point[0]), float(point[1])
        )
        self._place_landing(level, image_x, image_y)

    def _place_landing(self, level, image_x: float, image_y: float) -> None:
        landing = create_landing(level, image_x, image_y)
        self._record_canvas_undo_state(self._capture_landing_undo())
        level.landings.append(landing)
        self.canvas.cancel_landing_placement()
        self.viewer.cancel_landing_placement()
        self._selected_landing_id = landing.landing_id
        self._queue_landing_mesh_update()
        self._sync_landing_views()
        self.stair_status_label.setText(
            "Landing added. Select its * control to generate its shape."
        )

    def _select_landing(self, landing_id: str | None) -> None:
        if landing_id and self._find_landing(landing_id) is None:
            return
        self._selected_landing_id = landing_id or None
        self._sync_landing_views()

    # ### Transform transactions ###
    def _begin_landing_edit(self, landing_id: str) -> None:
        if (
            self._find_landing(landing_id) is None
            or self._active_landing_undo_state is not None
        ):
            return
        # The viewer has already started its drag. Rebuilding here can replace
        # the model and clear that drag before the first pointer movement.
        self._landing_mesh_update_timer.stop()
        self._landing_edit_had_pending_mesh_update = self._pending_landing_mesh_update
        self._active_landing_undo_state = self._capture_landing_undo()
        # Also block scheduled house refreshes before this drag's first change.
        self._pending_landing_mesh_update = True

    def _preview_landing_edit(self, landing_id: str, values: object) -> None:
        found = self._find_landing(landing_id)
        if found is None or not isinstance(values, dict):
            return
        level, index, landing = found
        fields = {
            key: value
            for key, value in values.items()
            if key
            in {
                "center_x",
                "center_y",
                "elevation_meters",
                "rotation_degrees",
                "scale_x",
                "scale_y",
                "scale_z",
            }
        }
        try:
            updated = replace(landing, **fields)
        except (TypeError, ValueError) as error:
            self.stair_status_label.setText(f"Landing not changed: {error}")
            return
        level.landings[index] = updated
        self._pending_landing_mesh_update = True
        self._sync_landing_views()

    def _finish_landing_edit(self, landing_id: str, changed: bool) -> None:
        snapshot = self._active_landing_undo_state
        previous_update_pending = self._landing_edit_had_pending_mesh_update
        self._active_landing_undo_state = None
        self._landing_edit_had_pending_mesh_update = False
        if snapshot is None:
            return
        if (
            changed
            and self._capture_landing_undo().landings_by_level
            != snapshot.landings_by_level
        ):
            self._record_canvas_undo_state(snapshot, commit_pending_surface_edit=False)
            self._queue_landing_mesh_update()
        else:
            self._restore_landing_undo_state(snapshot, schedule=previous_update_pending)

    def _cancel_landing_edit(self, _landing_id: str) -> None:
        snapshot = self._active_landing_undo_state
        previous_update_pending = self._landing_edit_had_pending_mesh_update
        self._active_landing_undo_state = None
        self._landing_edit_had_pending_mesh_update = False
        if snapshot is not None:
            self._restore_landing_undo_state(snapshot, schedule=previous_update_pending)

    def _restore_landing_undo_state(
        self, snapshot: CanvasLandingUndoState, *, schedule: bool = True
    ) -> None:
        by_level = dict(snapshot.landings_by_level)
        for level in self.levels:
            if level.index in by_level:
                level.landings[:] = by_level[level.index]
        self.stairs = list(snapshot.stairs)
        self._selected_landing_id = snapshot.selected_id
        self.canvas.set_stair_context(self.stairs, self.current_level)
        self._pending_landing_mesh_update = False
        self._landing_mesh_update_timer.stop()
        self._sync_landing_views()
        self._refresh_landing_generation_preview()
        if schedule:
            self._queue_landing_mesh_update()

    def _queue_landing_mesh_update(self) -> None:
        self._pending_landing_mesh_update = True
        if self._active_landing_undo_state is None:
            self._landing_mesh_update_timer.start()

    def _commit_landing_mesh_update(self) -> None:
        self._landing_mesh_update_timer.stop()
        if (
            self._active_landing_undo_state is not None
            or not self._pending_landing_mesh_update
        ):
            return
        self._pending_landing_mesh_update = False
        self._reset_viewer_landing_snapshots()
        self.canvas.set_stair_context(self.stairs, self.current_level)
        self._reconcile_canvas_surface_edit_and_refresh()

    # ### Meshy generation and retained assets ###
    def _landing_generation_preview_model(self, level, landing) -> GeneratedModel:
        """Preview canonical geometry without baking the user's transforms."""

        local_transform = np.linalg.inv(
            landing_world_transform(level, landing, self.levels)
        )
        parts = build_landing_meshes(level, landing, self.levels)
        scene = trimesh.Scene()
        for index, mesh in enumerate(parts):
            mesh.apply_transform(local_transform)
            mesh.apply_transform(LANDING_IMAGE_TO_LOCAL_Z_UP_TRANSFORM)
            exported = mesh.copy()
            exported.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
            scene.add_geometry(exported, node_name=f"landing-preview:{index}")
        scene.metadata["housemaker_landing_preview"] = True
        return GeneratedModel(
            mesh=trimesh.util.concatenate(parts),
            scene=scene,
            glb_bytes=scene.export(file_type="glb"),
        )

    def _persist_generated_landing(
        self, landing: LandingData, result: LandingMeshGenerationResult
    ) -> LandingData:
        """Keep immutable revisions so shared undo can restore the old mesh."""

        asset_directory = Path(self.generation._asset_directory)
        asset_directory.mkdir(parents=True, exist_ok=True)
        output_path = (
            asset_directory / f"landing-{landing.landing_id}-{uuid4().hex}.glb"
        )
        with TemporaryDirectory(
            prefix="landing-source-", dir=asset_directory
        ) as folder:
            source_path = Path(folder) / "meshy.glb"
            source_path.write_bytes(result.model.glb_bytes)
            updated = prepare_generated_landing(
                landing,
                source_path,
                output_path=output_path,
                texture_only=result.texture_only,
            )
        return updated

    def _refresh_landing_generation_preview(self) -> None:
        target = self.generation.get_landing_editing_target()
        if target is None:
            return
        found = self._find_landing(target.landing_id)
        if found is None:
            self.generation.clear_landing_editing_target()
            return
        level, _index, landing = found
        self.generation.update_landing_editing_preview(
            landing, self._landing_generation_preview_model(level, landing)
        )

    def _open_landing_generation(self, landing_id: str) -> None:
        found = self._find_landing(landing_id)
        if found is None:
            return
        level, _index, landing = found
        model = self._landing_generation_preview_model(level, landing)
        self.generation.set_landing_editing_target(landing, model)
        self.workspace_tabs.setCurrentIndex(self.generation_workspace_tab_index)

    def _apply_generated_landing(self, landing_id: str, result: object) -> None:
        found = self._find_landing(landing_id)
        if found is None or not isinstance(result, LandingMeshGenerationResult):
            return
        level, index, landing = found
        try:
            updated = self._persist_generated_landing(landing, result)
            build_landing_mesh(level, updated, self.levels)
            updated_stairs = (
                list(self.stairs)
                if result.texture_only
                else self._retarget_landing_stair_anchors(updated)
            )
            staged_level = replace(
                level,
                landings=[
                    updated if item.landing_id == landing_id else item
                    for item in level.landings
                ],
            )
            affected_stairs = [
                stair
                for stair in updated_stairs
                if any(
                    anchor is not None and anchor.landing_id == landing_id
                    for section in stair.sections
                    for anchor in (section.a_landing, section.b_landing)
                )
            ]
            build_stair_meshes(
                [
                    staged_level if item.index == level.index else item
                    for item in self.levels
                ],
                affected_stairs,
            )
        except (OSError, TypeError, ValueError) as error:
            self.stair_status_label.setText(f"Landing generation rejected: {error}")
            return
        self._record_canvas_undo_state(self._capture_landing_undo())
        level.landings[index] = updated
        self.stairs = updated_stairs
        self._selected_landing_id = landing_id
        self._queue_landing_mesh_update()
        self._sync_landing_views()
        self._refresh_landing_generation_preview()
        self.stair_status_label.setText(
            "Landing generated; connected stairs will update after the mesh delay."
        )

    def _retarget_landing_stair_anchors(self, landing: LandingData) -> list[StairData]:
        boundary = Polygon(landing.footprint_meters).exterior

        def retarget(anchor):
            if anchor is None or anchor.landing_id != landing.landing_id:
                return anchor
            point = boundary.interpolate(
                boundary.project(Point(anchor.local_x_meters, anchor.local_y_meters))
            )
            return LandingAttachment(landing.landing_id, point.x, point.y)

        return [
            replace(
                stair,
                **{
                    f"{end}_{side}_landing": retarget(
                        getattr(stair, f"{end}_{side}_landing")
                    )
                    for end in ("start", "end")
                    for side in ("a", "b")
                },
                intermediate_sections=tuple(
                    replace(
                        section,
                        a_landing=retarget(section.a_landing),
                        b_landing=retarget(section.b_landing),
                    )
                    for section in stair.intermediate_sections
                ),
            )
            for stair in self.stairs
        ]

    # ### Removal ###
    def _delete_landing(self, landing_id: str) -> None:
        found = self._find_landing(landing_id)
        if found is None:
            return
        if any(
            anchor is not None and anchor.landing_id == landing_id
            for stair in self.stairs
            for section in stair.sections
            for anchor in (section.a_landing, section.b_landing)
        ):
            self.stair_status_label.setText(
                "Disconnect its stair points before deleting this landing."
            )
            return
        self._record_canvas_undo_state(self._capture_landing_undo())
        del found[0].landings[found[1]]
        self._selected_landing_id = None
        if self.generation.get_landing_editing_target() is not None:
            self.generation.clear_landing_editing_target()
        self._queue_landing_mesh_update()
        self._sync_landing_views()
