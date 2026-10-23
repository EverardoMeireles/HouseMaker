"""Room overlays, room Atlas publication and the temporary runtime export path."""

# ### Imports ###
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, QTimer
from PySide6.QtWidgets import QCheckBox

from housemaker.room_zones import build_room_zones, room_zone_image_rings
from housemaker.scene_surface_cache import build_scene_surface_geometry_signature
from housemaker.texture_atlas_state import TextureAtlasData, normalize_atlas_name

# ### Temporary export destination ###
RUNTIME_EXPORT_PUBLIC_DIRECTORY = Path(
    r"C:\Users\Everardo\Documents\everardomeireles\public"
)
RUNTIME_EXPORT_MODEL_NAME = "housemaker_export.glb"
ROOM_OVERLAY_REFRESH_MILLISECONDS = 250


# ### Room visualization and export integration ###
class RoomExportWorkspaceMixin:
    def _init_room_export_state(self) -> None:
        self._room_layout_cache = {}
        self._last_room_export_plan = None
        self._last_room_export_input_atlas_ids = ()
        self._last_room_export_layout = None
        self._last_room_export_placements = ()
        self._last_material_streaming_policy = None
        self._room_overlay_timer = QTimer(self)
        self._room_overlay_timer.setSingleShot(True)
        self._room_overlay_timer.setInterval(ROOM_OVERLAY_REFRESH_MILLISECONDS)
        self._room_overlay_timer.timeout.connect(self._refresh_room_overlays)

    def _add_show_rooms_control(self, layout) -> None:
        self.show_rooms_checkbox = QCheckBox("Show rooms")
        self.show_rooms_checkbox.setObjectName("show-rooms-checkbox")
        self.show_rooms_checkbox.setToolTip(
            "Color detected rooms on the plan. Doorways separate rooms; stairs "
            "connect them across levels. This does not change the walls or floors."
        )
        self.show_rooms_checkbox.toggled.connect(self._handle_show_rooms_toggled)
        layout.addWidget(self.show_rooms_checkbox)

    def _add_material_streaming_controls(self, layout) -> None:
        self.use_ktx2_basis_compression_checkbox = QCheckBox(
            "Use KTX2/Basis compression"
        )
        self.use_ktx2_basis_compression_checkbox.setObjectName(
            "use-ktx2-basis-compression-checkbox"
        )
        self.use_ktx2_basis_compression_checkbox.setToolTip(
            "Compress external preview/full material GLBs with the local Khronos "
            "KTX encoder. Portable embedded color previews remain JPEG/PNG."
        )
        self.use_ktx2_basis_compression_checkbox.toggled.connect(
            self._set_ktx2_export_enabled
        )
        layout.addWidget(self.use_ktx2_basis_compression_checkbox)
        self._sync_material_streaming_controls()

    def _set_ktx2_export_enabled(self, enabled: bool) -> None:
        self.runtime_loading = replace(
            self.runtime_loading, use_ktx2_basis_compression=enabled
        )

    def _sync_material_streaming_controls(self) -> None:
        checkbox = getattr(self, "use_ktx2_basis_compression_checkbox", None)
        if checkbox is not None:
            with QSignalBlocker(checkbox):
                checkbox.setChecked(self.runtime_loading.use_ktx2_basis_compression)

    def _get_room_layout(self, *, include_excluded: bool = False):
        signature = build_scene_surface_geometry_signature(self.levels, self.stairs)
        cached = self._room_layout_cache.get(include_excluded)
        if cached is not None and cached[0] == signature:
            return cached[1]
        layout = build_room_zones(
            self.levels, self.stairs, include_excluded=include_excluded
        )
        self._room_layout_cache[include_excluded] = (signature, layout)
        return layout

    # ### Debounced, non-editable Canvas room overlays ###
    def _handle_show_rooms_toggled(self, enabled: bool) -> None:
        self.canvas.set_room_overlays_visible(enabled)
        if enabled:
            self._refresh_room_overlays()
        else:
            self._room_overlay_timer.stop()

    def _schedule_room_overlay_refresh(self) -> None:
        checkbox = getattr(self, "show_rooms_checkbox", None)
        if checkbox is not None and checkbox.isChecked() and not self._is_shutdown:
            self._room_overlay_timer.start()

    def _refresh_room_overlays(self) -> None:
        from housemaker.blueprint_canvas import CanvasRoomProfile

        if self._is_shutdown or not self.show_rooms_checkbox.isChecked():
            return
        level = self.current_level
        layout = self._get_room_layout(include_excluded=True)
        profiles = []
        for room in layout.rooms_on_level(level.index):
            if room.is_fallback or room.polygon.is_empty:
                continue
            exterior, holes = room_zone_image_rings(room, level)
            profiles.append(CanvasRoomProfile(room.room_id, exterior, holes))
        self.canvas.set_room_overlays(profiles)

    # ### Shared room-aware export geometry and Atlas preparation ###
    def _prepare_room_export_atlases(self, materialized_atlases, *, placements=None):
        from housemaker.room_atlas_export import prepare_room_export_atlases

        if placements is None:
            authored, instances = self._build_export_placed_models()
            placements = (*authored, *instances)
        levels_by_object = {
            record.object_id: record.placement.level_index
            for record in self.generation.get_generated_object_records()
            if record.placement is not None
        }
        levels_by_object.update(
            (instance.instance_id, instance.placement.level_index)
            for instance in self.generation.get_generated_object_instances()
        )
        layout = self._get_room_layout()
        plan = prepare_room_export_atlases(
            materialized_atlases,
            layout,
            placements,
            output_directory=self.texture_atlas_workspace._asset_directory,
            placement_level_indices=levels_by_object,
        )
        self._last_room_export_plan = plan
        self._last_room_export_input_atlas_ids = tuple(
            item.atlas.atlas_id for item in materialized_atlases
        )
        self._last_room_export_layout = layout
        self._last_room_export_placements = tuple(placements)
        return plan.atlases, layout

    def _publish_room_export_atlases(self, *, refresh_visibility: bool = False) -> None:
        """Publish successful room packing, preserving unrelated library Atlases."""

        plan = self._last_room_export_plan
        if plan is None:
            return
        current = self.texture_atlas_workspace.get_data()
        replaced_ids = set(self._last_room_export_input_atlas_ids)
        records = [
            copy.deepcopy(atlas)
            for atlas in current.atlases
            if atlas.atlas_id not in replaced_ids
        ]
        planned_ids = {item.atlas.atlas_id for item in plan.atlases}
        records = [atlas for atlas in records if atlas.atlas_id not in planned_ids]
        records.extend(copy.deepcopy(item.atlas) for item in plan.atlases)
        # User library Atlases may already use an automatically chosen room
        # name. Keep every record, changing only the new duplicate label.
        used_names = set()
        for atlas in records:
            original_name = atlas.name
            suffix = 2
            while atlas.name.casefold() in used_names:
                atlas.name = normalize_atlas_name(
                    f"{original_name} ({suffix})", atlas.atlas_type
                )
                suffix += 1
            used_names.add(atlas.name.casefold())
        retained_ids = {atlas.atlas_id for atlas in records}
        selected = current.selected_atlas_id
        if selected not in retained_ids:
            previous = current.atlas_by_id(selected) if selected is not None else None
            previous_sources = (
                {placement.object_id for placement in previous.placements}
                if previous is not None
                else set()
            )
            selected = next(
                (
                    atlas.atlas_id
                    for atlas in records
                    if any(p.object_id in previous_sources for p in atlas.placements)
                ),
                records[0].atlas_id if records else None,
            )
        grouped = TextureAtlasData(records, selected)
        if grouped.to_dict() != current.to_dict():
            self.texture_atlas_workspace.set_data(grouped)
            self.texture_atlas_workspace._emit_data_changed()
            self._atlas_draw_call_scene_revision += 1
            self._schedule_atlas_draw_call_estimate()
        if plan.warnings:
            self.texture_atlas_workspace.status_label.setText("\n".join(plan.warnings))
        if refresh_visibility and self._pvs_visibility is not None:
            self._pvs_visibility = replace(
                self._pvs_visibility,
                authoring_signature=self._build_pvs_authoring_signature(),
            )
            self._refresh_pvs_status()

    def _build_room_export_metadata(self, material_export):
        from housemaker.room_export_metadata import build_room_asset_manifest

        plan = self._last_room_export_plan
        return build_room_asset_manifest(
            self._last_room_export_layout or self._get_room_layout(),
            material_export,
            () if plan is None else plan.atlases,
            {} if plan is None else plan.source_room_ids,
            global_scale=self.global_scale,
        )

    def _prepare_export_material_packs(self, model):
        from housemaker.lazy_loading_export import (
            build_material_streaming_policy,
            resolve_camera_bootstrap,
            select_bootstrap_material_indices,
        )
        from housemaker.material_pack_export import build_progressive_material_export

        layout = self._last_room_export_layout or self._get_room_layout()
        bootstrap = resolve_camera_bootstrap(
            self.levels,
            layout,
            self.runtime_loading,
            self.global_scale,
            self._generation_settings.first_person_eye_height_meters,
        )
        indices = select_bootstrap_material_indices(
            model,
            bootstrap,
            self._last_room_export_plan,
            self._last_room_export_placements,
            scale=self.global_scale,
        )
        material_export = build_progressive_material_export(
            model,
            bootstrap_material_indices=indices,
            use_ktx2_basis_compression=self.runtime_loading.use_ktx2_basis_compression,
        )
        metadata = self._build_room_export_metadata(material_export)
        self._last_material_streaming_policy = build_material_streaming_policy(
            metadata,
            material_export,
            bootstrap,
            bootstrap_indices=indices,
            settings=self.runtime_loading,
        )
        return material_export, metadata

    def _write_export_material_packs(self, material_export, material_directory) -> None:
        from housemaker.scene_export import _write_bytes_atomically

        for pack in getattr(material_export, "all_packs", material_export.packs):
            _write_bytes_atomically(material_directory / pack.file_name, pack.glb_bytes)

    # ### Fixed temporary output layout ###
    def _temporary_runtime_export_paths(self) -> tuple[Path, Path]:
        root = RUNTIME_EXPORT_PUBLIC_DIRECTORY
        model_directory = root / "models"
        material_directory = root / "materials"
        model_directory.mkdir(parents=True, exist_ok=True)
        material_directory.mkdir(parents=True, exist_ok=True)
        return model_directory / RUNTIME_EXPORT_MODEL_NAME, material_directory
