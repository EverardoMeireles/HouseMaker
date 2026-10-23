"""Background PVS baking and export integration for the authored house."""

# ### Imports ###
from __future__ import annotations

import copy
import hashlib
import json
import struct
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import trimesh
from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton

from housemaker.atlas_export import (
    MaterializedTextureAtlas,
    apply_texture_atlases_to_export,
)
from housemaker.glb import (
    GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    PlacedGeneratedModel,
    build_placed_generated_model_transform,
    compose_generated_model_instance_sources,
    compose_placed_generated_models,
    convert_to_glb,
)
from housemaker.global_scale import GlobalScale
from housemaker.landing_geometry import build_landing_mesh
from housemaker.models import LevelData, StairData
from housemaker.pvs_bake import (
    PvsAabb,
    PvsBakeCancelled,
    PvsRenderCluster,
    bake_pvs,
    build_camera_cells,
    build_occluder_patches,
)
from housemaker.pvs_geometry import (
    collect_pvs_occluder_triangles,
    partition_glb_for_pvs,
)
from housemaker.pvs_state import PvsBakeData, PvsCell, PvsCluster
from housemaker.room_atlas_export import prepare_room_export_atlases
from housemaker.room_zones import build_room_zones
from housemaker.scene_surface_cache import build_scene_surface_geometry_signature
from housemaker.surface_geometry import build_fixed_surfaces

# ### Constants and physical coordinate transforms ###
PVS_JOB_KIND = "Visibility (PVS)"
PVS_PREPARATION_DELAY_MILLISECONDS = 0
Z_UP_TO_GLTF_TRANSFORM = np.linalg.inv(GLTF_Y_UP_TO_Z_UP_TRANSFORM)


def pvs_world_transform(scale: GlobalScale) -> np.ndarray:
    """Match the runtime export's physical glTF Y-up building transform."""

    return np.diag((scale.x, scale.z, scale.y, 1.0))


def _transformed_bounds(minimum, maximum, transform):
    corners = np.asarray(tuple(product(*zip(minimum, maximum, strict=True))))
    points = trimesh.transform_points(corners, transform)
    return tuple(points.min(axis=0)), tuple(points.max(axis=0))


def _stable_value(value):
    """Produce process-independent fingerprints, including sets in geometry."""

    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _stable_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _stable_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted((_stable_value(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported PVS fingerprint input: {type(value).__name__}.")


def pvs_fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(
        _stable_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


# ### Immutable authored export snapshot ###
@dataclass(frozen=True)
class PvsSceneSnapshot:
    levels: tuple[LevelData, ...]
    stairs: tuple[StairData, ...]
    surface_materials: tuple[tuple[str, object], ...]
    authored_models: tuple[PlacedGeneratedModel, ...]
    instance_models: tuple[PlacedGeneratedModel, ...]
    materialized_atlases: tuple[MaterializedTextureAtlas, ...]
    surface_source_ids: tuple[tuple[str, str], ...]
    global_scale: GlobalScale
    eye_height_meters: float
    authoring_signature: str
    make_floors_double_sided: bool = False
    make_ceilings_double_sided: bool = False
    output_directory: Path | None = None
    placement_level_indices: tuple[tuple[str, int], ...] = ()


def _export_placement_groups(authored_models, instance_models):
    source_ids = {str(item.source_object_id) for item in instance_models}
    placed = tuple(item for item in authored_models if item.object_id not in source_ids)
    authored_by_id = {item.object_id: item for item in authored_models}
    sources = tuple(
        authored_by_id.get(source_id) or min(
            (item for item in instance_models if str(item.source_object_id) == source_id),
            key=lambda item: item.object_id,
        )
        for source_id in sorted(source_ids)
    )
    return placed, sources


def _build_snapshot_model(snapshot: PvsSceneSnapshot):
    """Use the same export operations as Main, entirely outside Qt widgets."""

    base = convert_to_glb(
        snapshot.levels, stairs=snapshot.stairs,
        surface_materials=dict(snapshot.surface_materials),
        export_untextured_surfaces=True,
        make_floors_double_sided=snapshot.make_floors_double_sided,
        make_ceilings_double_sided=snapshot.make_ceilings_double_sided,
    )
    placed, sources = _export_placement_groups(snapshot.authored_models, snapshot.instance_models)
    model = compose_placed_generated_models(base, placed) if placed else base
    if sources:
        model = compose_generated_model_instance_sources(model, sources)
    layout = build_room_zones(snapshot.levels, snapshot.stairs)
    # Old callers can bake without retaining generated Atlas assets. The real
    # workspace captures its owned asset directory, just like normal export.
    destination = (
        TemporaryDirectory(prefix="housemaker-pvs-atlases-")
        if snapshot.output_directory is None
        else nullcontext(snapshot.output_directory)
    )
    with destination as output_directory:
        plan = prepare_room_export_atlases(
            snapshot.materialized_atlases, layout,
            (*snapshot.authored_models, *snapshot.instance_models),
            output_directory=output_directory,
            placement_level_indices=dict(snapshot.placement_level_indices),
        )
        model = apply_texture_atlases_to_export(
            model, plan.atlases,
            surface_source_ids=dict(snapshot.surface_source_ids),
            room_layout=layout,
        )
    return base, model


def _has_static_instance_bounds(model):
    """Leave animated/reconstructed source geometry outside instance culling."""

    try:
        payload = model.glb_bytes
        if payload[:4] != b"glTF" or len(payload) < 20:
            return False
        length, kind = struct.unpack_from("<I4s", payload, 12)
        if kind != b"JSON":
            return False
        document = json.loads(payload[20:20 + length])
        if document.get("animations") or document.get("skins"):
            return False
        for record in (*document.get("nodes", ()), *document.get("meshes", ())):
            extras = record.get("extras", {})
            if any(key in extras for key in (
                "halfMesh", "housemaker_door_body", "housemakerDoorBody",
                "housemaker_door_component", "housemakerDoorComponent",
            )) or extras.get("housemaker_dynamic") is True:
                return False
            if (
                "skin" in record or "weights" in record
                or "EXT_mesh_gpu_instancing" in record.get("extensions", {})
            ):
                return False
            if any(primitive.get("targets") for primitive in record.get("primitives", ())):
                return False
        return True
    except (AttributeError, TypeError, ValueError, struct.error):
        return False


def _instance_clusters(authored_models, instance_models, scale):
    """Runtime instances retain source sharing but receive independent bounds."""

    source_ids = {str(item.source_object_id) for item in instance_models}
    placements = (*instance_models, *(item for item in authored_models if item.object_id in source_ids))
    clusters = []
    for item in sorted(placements, key=lambda item: item.object_id):
        # Runtime symmetry/reconstructed moving components can extend beyond
        # their source triangles; leave those unmapped and therefore visible.
        if (
            item.symmetric_preview_orientation is not None
            or item.object_id.startswith(("door-placement:", "window-placement:"))
            or str(item.source_object_id).startswith(("door-placement:", "window-placement:"))
            or not _has_static_instance_bounds(item.model)
        ):
            continue
        bounds = np.asarray(item.model.mesh.bounds, dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            continue
        transform = pvs_world_transform(scale) @ Z_UP_TO_GLTF_TRANSFORM @ build_placed_generated_model_transform(item)
        lower, upper = _transformed_bounds(*bounds, transform)
        cluster_id = "pvs_instance_" + hashlib.sha256(item.object_id.encode()).hexdigest()[:24]
        clusters.append(PvsCluster(cluster_id, (), lower, upper, (item.object_id,)))
    return tuple(clusters)


def prepare_pvs_geometry(model, authored_models, instance_models, scale, eye_height):
    """Return partitioned bytes and the exact immutable visibility dependencies."""

    world = pvs_world_transform(scale)
    payload, descriptors = partition_glb_for_pvs(model.glb_bytes, world_transform=world)
    clusters = tuple(PvsCluster(
        item.cluster_id, item.node_names, item.minimum, item.maximum,
    ) for item in descriptors) + _instance_clusters(authored_models, instance_models, scale)
    occluders = collect_pvs_occluder_triangles(model.glb_bytes, world_transform=world)
    # Ignore primitive traversal and cyclic vertex order, but preserve winding:
    # a single-sided wall is not an occluder when seen from its back.
    ordered = np.asarray(occluders, dtype="<f8").copy()
    if len(ordered):
        vertex_order = np.lexsort((ordered[:, :, 2], ordered[:, :, 1], ordered[:, :, 0]), axis=1)
        cyclic_order = (vertex_order[:, :1] + np.arange(3)) % 3
        ordered = np.take_along_axis(ordered, cyclic_order[:, :, None], axis=1).reshape(-1, 9)
        ordered = ordered[np.lexsort(ordered[:, ::-1].T)]
    geometry_hash = hashlib.sha256()
    geometry_hash.update(pvs_fingerprint((clusters, float(eye_height))).encode())
    geometry_hash.update(ordered.tobytes())
    return payload, clusters, occluders, geometry_hash.hexdigest()


def bake_scene_visibility(snapshot, *, progress=None, is_cancelled=None):
    """Build the final export geometry, camera cells and conservative PVS."""

    def stage(message):
        if is_cancelled is not None and is_cancelled():
            raise PvsBakeCancelled("Visibility bake cancelled.")
        if progress is not None:
            progress(message)

    stage("Building export geometry (5%)")
    base, model = _build_snapshot_model(snapshot)
    stage("Partitioning visibility groups (20%)")
    _payload, clusters, opaque_triangles, signature = prepare_pvs_geometry(
        model, snapshot.authored_models, snapshot.instance_models,
        snapshot.global_scale, snapshot.eye_height_meters,
    )
    if not clusters:
        raise ValueError("The scene has no supported static geometry to bake.")
    stage("Finding walkable camera cells (30%)")
    included_levels = {level.index for level in snapshot.levels if level.include_in_export}
    walkable = [
        np.asarray(surface.mesh.triangles, dtype=float)
        for surface in build_fixed_surfaces(snapshot.levels, stairs=snapshot.stairs)
        if surface.surface_type == "floor" and surface.level_index in included_levels
    ]
    walkable.extend(np.asarray(part.mesh.triangles, dtype=float)
                    for part in base.preview_stair_parts if part.part_kind == "treads")
    for level in snapshot.levels:
        if level.include_in_export:
            for landing in level.landings:
                triangles = np.asarray(build_landing_mesh(level, landing, snapshot.levels).triangles)
                normal_z = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])[:, 2]
                walkable.append(triangles[normal_z > 1e-9])
    if not walkable:
        raise ValueError("Create a floor or stairs before baking visibility.")
    physical_walkable = np.concatenate(walkable) * np.asarray(snapshot.global_scale.factors)
    cells = build_camera_cells(
        physical_walkable, eye_height=snapshot.eye_height_meters,
        is_cancelled=is_cancelled,
    )
    if not cells:
        raise ValueError("No walkable camera cells were found in included levels.")
    stage("Building opaque wall and slab occluders (40%)")
    physical_occluders = trimesh.transform_points(
        opaque_triangles.reshape(-1, 3), GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    ).reshape(-1, 3, 3)
    patches = build_occluder_patches(
        (physical_occluders,), one_sided=True, is_cancelled=is_cancelled,
    )
    core_clusters = tuple(PvsRenderCluster(
        item.cluster_id, PvsAabb(*_transformed_bounds(
            item.minimum, item.maximum, GLTF_Y_UP_TO_Z_UP_TRANSFORM,
        )),
    ) for item in clusters)
    stage("Baking potentially visible sets (45%)")
    result = bake_pvs(
        cells, core_clusters, patches, is_cancelled=is_cancelled,
        progress=lambda done, total: stage(
            f"Baking potentially visible sets ({45 + int(53 * done / max(1, total))}%)"
        ),
    )
    saved_cells = tuple(PvsCell(
        cell.cell_id, *_transformed_bounds(
            cell.bounds.minimum, cell.bounds.maximum, Z_UP_TO_GLTF_TRANSFORM,
        ), visible,
    ) for cell, visible in zip(result.cells, result.visible_cluster_ids_by_cell, strict=True))
    stage("Finalizing visibility data (99%)")
    return PvsBakeData(
        signature, clusters, saved_cells, snapshot.eye_height_meters,
        authoring_signature=snapshot.authoring_signature,
    )


# ### Cancellable Qt worker ###
class PvsBakeThread(QThread):
    progress = Signal(str)

    def __init__(self, snapshot, parent=None):
        super().__init__(parent)
        self.snapshot = snapshot
        self.result = None
        self.error_message = None
        self.was_cancelled = False

    def run(self):
        try:
            self.result = bake_scene_visibility(
                self.snapshot, progress=self.progress.emit,
                is_cancelled=self.isInterruptionRequested,
            )
        except Exception as error:  # noqa: BLE001 - worker errors cross Qt safely.
            if self.isInterruptionRequested() or isinstance(error, PvsBakeCancelled):
                self.was_cancelled = True
            else:
                self.error_message = str(error) or type(error).__name__


# ### Main workspace integration ###
class PvsWorkspaceMixin:
    def _init_pvs_state(self):
        self._pvs_visibility = None
        self._pvs_bake_thread = None
        self._pvs_job_id = None
        self._pvs_cancel_requested = False
        self._pvs_export_note = ""
        self._pvs_status_timer = QTimer(self)
        self._pvs_status_timer.setSingleShot(True)
        self._pvs_status_timer.setInterval(300)
        self._pvs_status_timer.timeout.connect(self._refresh_pvs_status)

    def _add_pvs_controls(self, layout):
        row = QHBoxLayout()
        self.bake_visibility_button = QPushButton("Bake visibility(PVS)")
        self.bake_visibility_button.setObjectName("bake-visibility-pvs-button")
        self.bake_visibility_button.setToolTip(
            "Bake conservative visibility for all included levels. The result is "
            "saved with the project and exported for r3f; no room definitions are needed."
        )
        self.bake_visibility_button.clicked.connect(self._handle_bake_visibility_clicked)
        row.addWidget(self.bake_visibility_button)
        self.pvs_status_label = QLabel("Visibility not baked")
        self.pvs_status_label.setObjectName("pvs-status-label")
        self.pvs_status_label.setWordWrap(True)
        row.addWidget(self.pvs_status_label, 1)
        layout.addLayout(row)

    def _build_pvs_authoring_signature(self):
        return pvs_fingerprint((
            build_scene_surface_geometry_signature(self.levels, self.stairs),
            tuple(level.name for level in self.levels), self.global_scale,
            self._generation_settings.first_person_eye_height_meters,
            self._generation_settings.make_floors_double_sided,
            self._generation_settings.make_ceilings_double_sided,
            self._build_viewer_preview_dependency_signature(),
            self.texture_atlas_workspace.get_data().to_dict(),
            tuple((record.object_id, record.object_name, record.asset_path)
                  for record in self.generation.get_generated_object_records()),
            self.doors_workspace.data().to_dict(),
        ))

    def _refresh_pvs_status(self):
        if self._is_shutdown or not hasattr(self, "pvs_status_label"):
            return
        running = self._pvs_job_id is not None
        self.bake_visibility_button.setEnabled(not running)
        if running:
            return
        data = self._pvs_visibility
        if data is None:
            self.pvs_status_label.setText("Visibility not baked")
        elif data.authoring_signature != self._build_pvs_authoring_signature():
            self.pvs_status_label.setText("Visibility outdated — rebake")
        else:
            self.pvs_status_label.setText(
                f"Visibility baked: {len(data.cells)} cells, {len(data.clusters)} groups"
            )

    def _capture_pvs_snapshot(self):
        # Import here to avoid a cycle with Main's established trim helpers.
        from housemaker.architectural_trim_generation import (
            remove_generated_architectural_trims_from_levels,
        )

        _models, generated_trim_keys = self._build_active_architectural_trim_component_models()
        levels = tuple(copy.deepcopy(remove_generated_architectural_trims_from_levels(
            self.levels, generated_trim_keys,
        )))
        authored, instances = self._build_export_placed_models()
        placed, sources = _export_placement_groups(authored, instances)
        surface_source_ids = self._build_atlas_surface_source_ids()
        required = tuple(dict.fromkeys((
            *(item.object_id for item in placed if not item.object_id.startswith("door-placement:")),
            *(str(item.source_object_id) for item in sources), *surface_source_ids.values(),
        )))
        atlases = self.texture_atlas_workspace.prepare_export_atlases(required)
        levels_by_object = {
            record.object_id: record.placement.level_index
            for record in self.generation.get_generated_object_records()
            if record.placement is not None
        }
        levels_by_object.update(
            (instance.instance_id, instance.placement.level_index)
            for instance in self.generation.get_generated_object_instances()
        )
        return PvsSceneSnapshot(
            levels, tuple(copy.deepcopy(self.stairs)),
            tuple(self.surface_texture_generation.get_surface_material_sources().items()),
            tuple(authored), tuple(instances), tuple(atlases),
            tuple(sorted(surface_source_ids.items())), self.global_scale,
            self._generation_settings.first_person_eye_height_meters,
            self._build_pvs_authoring_signature(),
            self._generation_settings.make_floors_double_sided,
            self._generation_settings.make_ceilings_double_sided,
            Path(self.texture_atlas_workspace._asset_directory),
            tuple(sorted(levels_by_object.items())),
        )

    def _handle_bake_visibility_clicked(self):
        if self._is_shutdown or self._pvs_job_id is not None:
            return
        self._handle_confirm_mesh_updates_clicked()
        self._commit_pending_doorway_mesh_update()
        self._pvs_cancel_requested = False
        self.bake_visibility_button.setEnabled(False)
        self.pvs_status_label.setText("Preparing visibility bake…")
        job = self.job_manager.create_job(
            kind=PVS_JOB_KIND, requested_name="House visibility", default_name="House visibility",
            stage="Preparing scene (0%)", cancel_callback=self._cancel_pvs_bake,
        )
        self._pvs_job_id = job.job_id
        # Publish the Jobs row before capturing UI-owned asset bindings. All
        # geometry construction, partitioning and visibility work runs in Qt's worker.
        QTimer.singleShot(PVS_PREPARATION_DELAY_MILLISECONDS, self._start_pvs_bake)

    def _start_pvs_bake(self):
        if self._is_shutdown or self._pvs_job_id is None or self._pvs_bake_thread is not None:
            return
        try:
            snapshot = self._capture_pvs_snapshot()
            thread = PvsBakeThread(snapshot, self)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self.job_manager.fail_job(self._pvs_job_id, str(error))
            self._pvs_job_id = None
            self.bake_visibility_button.setEnabled(True)
            self.pvs_status_label.setText(f"Visibility bake failed: {error}")
            return
        self._pvs_bake_thread = thread
        thread.progress.connect(
            lambda message, worker=thread: self._handle_pvs_progress(message, worker)
        )
        thread.finished.connect(lambda: self._handle_pvs_finished(thread))
        thread.start()

    def _handle_pvs_progress(self, message, thread=None):
        if thread is not None and thread is not self._pvs_bake_thread:
            return
        if self._pvs_job_id is not None and not self._is_shutdown:
            self.job_manager.update_job(self._pvs_job_id, stage=message)
            self.pvs_status_label.setText(message)

    def _handle_pvs_finished(self, thread):
        if self._pvs_bake_thread is not thread:
            return
        job_id = self._pvs_job_id
        self._pvs_bake_thread = None
        self._pvs_job_id = None
        if self._is_shutdown:
            thread.deleteLater()
            return
        if self._pvs_cancel_requested or thread.was_cancelled or thread.isInterruptionRequested():
            self.job_manager.mark_cancelled(job_id)
            self.pvs_status_label.setText("Visibility bake cancelled")
        elif thread.error_message is not None or thread.result is None:
            self.job_manager.fail_job(job_id, thread.error_message or "No visibility data was produced.")
            self.pvs_status_label.setText(f"Visibility bake failed: {thread.error_message}")
        elif thread.snapshot.authoring_signature != self._build_pvs_authoring_signature():
            self.job_manager.fail_job(job_id, "The scene changed during baking. Bake visibility again.")
            self.pvs_status_label.setText("Scene changed — bake visibility again")
        else:
            self._pvs_visibility = thread.result
            self.job_manager.complete_job(job_id, f"Baked {len(thread.result.cells)} visibility cells")
            self._refresh_pvs_status()
        self.bake_visibility_button.setEnabled(True)
        thread.deleteLater()

    def _cancel_pvs_bake(self):
        if self._pvs_job_id is None:
            return False
        self._pvs_cancel_requested = True
        if self._pvs_bake_thread is not None:
            self._pvs_bake_thread.requestInterruption()
        else:
            self.job_manager.mark_cancelled(self._pvs_job_id)
            self._pvs_job_id = None
            self.bake_visibility_button.setEnabled(True)
            self.pvs_status_label.setText("Visibility bake cancelled")
        return True

    def _cancel_and_join_pvs_bake(self):
        self._pvs_status_timer.stop()
        thread = self._pvs_bake_thread
        self._cancel_pvs_bake()
        if thread is not None:
            thread.requestInterruption()
            thread.wait()
            if self._pvs_job_id is not None:
                self.job_manager.mark_cancelled(self._pvs_job_id)
            self._pvs_bake_thread = None
            self._pvs_job_id = None
            thread.deleteLater()
        if hasattr(self, "bake_visibility_button"):
            self.bake_visibility_button.setEnabled(True)

    def _prepare_pvs_export_model(self, model, authored_models, instance_models):
        """Never attach old visibility to changed geometry or changed grouping."""

        self._pvs_export_note = ""
        if self._pvs_visibility is None:
            return model, None
        try:
            payload, clusters, _triangles, signature = prepare_pvs_geometry(
                model, authored_models, instance_models, self.global_scale,
                self._generation_settings.first_person_eye_height_meters,
            )
        except (TypeError, ValueError, OverflowError) as error:
            self._pvs_export_note = (
                f"Exported without PVS because visibility groups could not be "
                f"validated: {error}"
            )
            self.pvs_status_label.setText("Visibility unavailable — exported without PVS")
            return model, None
        if signature != self._pvs_visibility.source_signature or clusters != self._pvs_visibility.clusters:
            self._pvs_export_note = "Visibility bake is outdated; exported without PVS. Click Bake visibility(PVS) and export again."
            self.pvs_status_label.setText("Visibility outdated — exported without PVS")
            return model, None
        data = replace(self._pvs_visibility, authoring_signature=self._build_pvs_authoring_signature())
        self._pvs_visibility = data
        self._pvs_export_note = f"PVS included: {len(data.cells)} camera cells."
        return replace(model, glb_bytes=payload), data
