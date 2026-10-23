"""Deterministic startup selection and a declarative R3F streaming contract."""

# ### Imports ###
from __future__ import annotations

import math
from dataclasses import dataclass

from housemaker.global_scale import GlobalScale
from housemaker.lazy_loading_state import RuntimeLoadingData
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
)
from housemaker.scene_export import _parse_glb

# ### Conservative default budgets (runtime may lower them for the device) ###
BOOTSTRAP_RADIUS_METERS = 8.0
FALLBACK_MAX_SIZE = 128
PREVIEW_MAX_SIZE = 1024


# ### Resolved physical camera pose ###
@dataclass(frozen=True)
class CameraBootstrap:
    position: tuple[float, float, float]
    floor_position: tuple[float, float, float]
    direction: tuple[float, float, float]
    level_index: int | None
    room_ids: tuple[str, ...]
    source: str

    def to_dict(self):
        return {
            "position": list(self.position),
            "floorPosition": list(self.floor_position),
            "lookDirection": list(self.direction),
            "up": [0, 1, 0],
            "fovDegrees": 70,
            "levelIndex": self.level_index,
            "roomIds": list(self.room_ids),
            "source": self.source,
            "coordinateSystem": "gltf-y-up",
            "globalScaleAlreadyApplied": True,
        }


def _physical_point(point, scale):
    x, y, z = point
    return (x * scale.x, z * scale.z, -y * scale.y)


def resolve_camera_bootstrap(levels, layout, settings, scale, eye_height):
    """Add physical eye height AFTER global Z, never scale the operator's height."""

    included = {level.index: level for level in levels if level.include_in_export}
    floor_z = build_level_base_z_lookup(levels)
    point = settings.start_point
    if point is not None:
        level = included.get(point.level_index)
        if level is None:
            raise ValueError(
                "The FPS starting point belongs to a missing or excluded level. "
                "Set a starting point on an included level before exporting."
            )
        xy = level_image_to_world_xy(level, point.image_x, point.image_y)
        floor = (*xy, floor_z[level.index] + point.height_offset_meters)
        yaw, pitch = math.radians(point.yaw_degrees), math.radians(point.pitch_degrees)
        direction = (
            math.cos(pitch) * math.cos(yaw),
            math.sin(pitch),
            -math.cos(pitch) * math.sin(yaw),
        )
        source = "authored"
    else:
        rooms = sorted(
            layout.rooms,
            key=lambda room: (
                room.is_fallback,
                room.level_index != 2,
                room.level_index,
                room.room_id,
            ),
        )
        if rooms:
            room = rooms[0]
            xy = room.polygon.representative_point().coords[0]
            level = included[room.level_index]
        elif included:
            level = min(
                included.values(), key=lambda item: (item.index != 2, item.index)
            )
            size = level.image_size_pixels or (0, 0)
            xy = level_image_to_world_xy(level, size[0] / 2, size[1] / 2)
        else:
            return CameraBootstrap(
                (0, eye_height, 0), (0, 0, 0), (1, 0, 0), None, (), "automatic"
            )
        floor = (*xy, floor_z[level.index])
        direction, source = (1, 0, 0), "automatic"
    from shapely import Point

    owners = tuple(
        sorted(
            room.room_id
            for room in layout.rooms_on_level(level.index)
            if room.polygon.buffer(1e-6).covers(Point(floor[:2]))
        )
    )
    physical_floor = _physical_point(floor, scale)
    camera = (physical_floor[0], physical_floor[1] + eye_height, physical_floor[2])
    return CameraBootstrap(
        camera, physical_floor, direction, level.index, owners, source
    )


# ### Embedded startup textures retain the original Atlas layouts ###
def select_bootstrap_material_indices(
    model, bootstrap, plan=None, placements=(), *, scale=None
):
    scale = scale or GlobalScale()
    owners = set(bootstrap.room_ids)
    near_sources = set()
    for placement in placements:
        position = _physical_point(placement.world_position, scale)
        if math.dist(position, bootstrap.floor_position) <= BOOTSTRAP_RADIUS_METERS:
            near_sources.add(str(placement.atlas_source_object_id))
    startup_atlases = set()
    if plan is not None:
        for item in plan.atlases:
            atlas = item.atlas
            if (
                atlas.atlas_type == "surface"
                or atlas.room_id in owners
                or any(
                    p.object_id in near_sources
                    or owners.intersection(plan.source_room_ids.get(p.object_id, ()))
                    for p in atlas.placements
                )
            ):
                startup_atlases.add(atlas.atlas_id)
    document, _ = _parse_glb(model.glb_bytes, "The startup model is invalid.")
    selected = set()
    meshes = document.get("meshes", [])
    for node in document.get("nodes", []):
        index = node.get("mesh")
        if not isinstance(index, int) or not 0 <= index < len(meshes):
            continue
        mesh = meshes[index]
        extras = {**mesh.get("extras", {}), **node.get("extras", {})}
        if (
            extras.get("housemaker_atlas_id") in startup_atlases
            or extras.get("housemaker_atlas_type") == "surface"
            or owners.intersection(extras.get("housemaker_room_ids", ()))
        ):
            selected.update(
                primitive["material"]
                for primitive in mesh.get("primitives", [])
                if isinstance(primitive.get("material"), int)
            )
    return frozenset(selected)


# ### Versioned, machine-readable loading policy ###
def build_material_streaming_policy(
    room_assets,
    material_export,
    bootstrap,
    *,
    bootstrap_indices=(),
    settings=None,
):
    """Attach variants to roomAssets and return instructions for a generic loader.

    No R3F application files are changed here. A consumer must implement this
    schema rather than evaluating project-supplied JavaScript or prose.
    """

    settings = settings or RuntimeLoadingData()
    previews = {pack.material_id: pack for pack in material_export.preview_packs}
    full = {pack.material_id: pack for pack in material_export.packs}
    atlases = {atlas["atlasId"]: atlas for atlas in room_assets["atlases"]}
    bootstrap_indices = set(bootstrap_indices)
    for material in room_assets["materials"]:
        pack = full[material["materialId"]]
        preview = previews[pack.material_id]
        atlas = atlases.get(pack.atlas_id, {})
        shared = bool(atlas.get("shared", len(set(material.get("roomIds", ()))) != 1))
        startup = pack.material_index in bootstrap_indices
        material.update(
            {
                "shared": shared,
                "atlasType": atlas.get("type"),
                "variants": {
                    "embedded": {
                        "inMainGlb": True,
                        "maxSize": PREVIEW_MAX_SIZE if startup else FALLBACK_MAX_SIZE,
                        "qualityTier": "preview" if startup else "fallback",
                        "mapSet": "base-color-and-required-opacity-emission",
                        "compression": "jpeg-or-png",
                    },
                    "preview": _variant(preview),
                    "full": _variant(pack),
                },
                "bootstrap": startup,
                "binding": {
                    "materialIndex": pack.material_index,
                    "materialExtrasId": pack.material_id,
                },
            }
        )
    room_assets["previewMaxSize"] = PREVIEW_MAX_SIZE
    room_assets["fallbackMaxSize"] = FALLBACK_MAX_SIZE
    adjacent = sorted(
        {
            room
            for portal in room_assets["portals"]
            if set(portal["roomIds"]).intersection(bootstrap.room_ids)
            for room in portal["roomIds"]
            if room not in bootstrap.room_ids
        }
    )
    return {
        "format": "housemaker-material-streaming",
        "version": 1,
        "enabled": True,
        "materialTable": "roomAssets.materials",
        "initialCamera": bootstrap.to_dict(),
        "textureCompression": {
            "externalPacks": "ktx2-basis"
            if settings.use_ktx2_basis_compression
            else "original-images",
            "embeddedFallbacks": "jpeg-or-png",
            "requiredLoader": "KTX2Loader"
            if settings.use_ktx2_basis_compression
            else None,
            "decoder": {
                "type": "BasisUniversal",
                "initializeBeforeExternalPacks": True,
                "detectRendererSupportBeforeLoad": True,
                "attachTo": "GLTFLoader",
                "transcoderPathSource": "runtime-configured-basis-assets",
                "workerLimit": 2,
                "preloadAlongsideBootstrap": True,
            }
            if settings.use_ktx2_basis_compression
            else None,
        },
        "bootstrap": {
            "readyWhen": "geometry-and-embedded-materials-rendered",
            "minimumTutorialSeconds": 0,
            "roomIds": list(bootstrap.room_ids),
            "adjacentRoomIds": adjacent,
            "materialIds": sorted(
                m["materialId"] for m in room_assets["materials"] if m["bootstrap"]
            ),
            "radiusMeters": BOOTSTRAP_RADIUS_METERS,
            "keepEmbeddedFallbacks": True,
        },
        "tutorial": {
            "enabled": True,
            "showOnFirstVisit": True,
            "skippable": True,
            "showWhen": "bootstrap-rendered",
            "loadingContinuesWhileOpen": True,
            "waitForFullMaterials": False,
            "bindingsSource": "runtime-active-controls",
            "instructions": [
                {
                    "action": "move",
                    "label": "Move around",
                    "defaultKeys": ["W", "A", "S", "D"],
                },
                {
                    "action": "look",
                    "label": "Move the mouse to look around",
                    "defaultKeys": ["Mouse"],
                },
                {
                    "action": "toggleObjects",
                    "label": "Hide or show objects",
                    "requireRuntimeBinding": True,
                },
                {
                    "action": "releasePointer",
                    "label": "Release the cursor",
                    "defaultKeys": ["Escape"],
                },
            ],
            "omitActionsWithoutBinding": True,
            "replayAction": "showNavigationHelp",
        },
        "priorityOrder": [
            {
                "rank": 0,
                "variant": "preview",
                "scope": "visible-not-yet-preview-quality",
            },
            {"rank": 1, "variant": "full", "scope": "visible-in-current-room"},
            {
                "rank": 2,
                "variant": "full",
                "scope": "visible-outside-current-room",
                "priorityHints": ["portal", "pvs"],
            },
            {"rank": 3, "variant": "full", "scope": "remaining-current-room-nonshared"},
            {"rank": 4, "variant": "preview", "scope": "adjacent-rooms-heading-first"},
            {
                "rank": 5,
                "variant": "full",
                "scope": "adjacent-rooms-heading-first",
                "onlyWithinBudget": True,
            },
            {"rank": 6, "variant": "preview", "scope": "other-rooms", "idleOnly": True},
        ],
        "sharedMaterials": {
            "inferFullPriorityFromRoomDependencies": False,
            "fullPrioritySource": "actual-visible-primitives",
            "previewEarly": True,
            "onlyWhenVisibleOrPrefetched": True,
            "evictOnlyWithoutVisibleOrPrefetchedReferences": True,
        },
        "scheduler": {
            "maxConcurrentDownloads": 4,
            "maxConcurrentDecodes": 2,
            "maxGpuUploadsPerFrame": 1,
            "gpuWorkBudgetMilliseconds": 2,
            "reprioritizeOn": [
                "camera-move",
                "camera-turn",
                "room-change",
                "visibility-change",
            ],
            "preferVisibleBeforeDistance": True,
            "warmShadersBeforeSwap": True,
            "readyStages": ["downloaded", "decoded", "gpu-uploaded", "shader-ready"],
            "swapWhen": "shader-ready",
            "cancelQueuedObsoleteRequests": True,
        },
        "roomSelection": {
            "method": "floor-polygon-and-height",
            "boundsAreBroadPhaseOnly": True,
            "fallbackZonesUseDistancePriorities": True,
            "neverUseFallbackZonesToHideGeometry": True,
            "usePortalsAndPvsForPrefetchNotAsSoleVisibilityProof": True,
            "exitHysteresisMeters": 0.5,
            "exitHysteresisMilliseconds": 1500,
            "unknownRoomPolicy": "frustum-and-distance-fail-open",
        },
        "cache": {
            "key": "sha256",
            "deduplicateByMaterialIdAndVariant": True,
            "persistent": "cache-storage-if-available",
            "onQuotaFailure": "memory-only",
            "desktopGpuBudgetMiB": 512,
            "mobileGpuBudgetMiB": 128,
            "deviceMayReduceBudgets": True,
            "accountFor": ["all-maps", "mipmaps", "resident-variants"],
            "eviction": "least-recently-used-nonvisible-full-first",
            "retainPreviousRoomMilliseconds": 10000,
            "keepEmbeddedFallbacks": True,
            "disposeAfterLastReference": True,
        },
        "binding": {
            "measureEmbeddedQualityFromActualTextureDimensions": True,
            "sharedSourceMayReuseHigherResolutionEmbeddedImage": True,
            "matchBy": [
                "housemaker_material_id",
                "materialIndex-if-main-sha256-matches",
            ],
            "applyTo": [
                "main-scene",
                "detached-instance-prototypes",
                "instances",
                "mirrored-halves",
            ],
            "preserve": [
                "geometry",
                "uvs",
                "uv1-ao",
                "transforms",
                "object-ids",
                "visibility",
                "shadows",
            ],
            "replacement": "atomic-per-material",
            "addMaterialCubeToScene": False,
            "keepFallbackUntilReplacementReady": True,
        },
        "failurePolicy": {
            "maxAttempts": 3,
            "retryBackoffMilliseconds": [1000, 3000],
            "onFullFailure": "retain-preview",
            "onPreviewFailure": "retain-embedded",
            "onKtx2Unsupported": "retain-portable-embedded-and-report",
            "onHashMismatch": "discard-and-refetch",
            "neverHideGeometryForMissingMaterial": True,
        },
        "telemetry": {
            "measure": [
                "first-textured-frame-ms",
                "current-room-full-ready-ms",
                "bytes-downloaded",
                "gpu-resident-bytes",
                "fallback-visible-count",
                "material-errors",
                "upload-frame-stalls",
            ]
        },
    }


def _variant(pack):
    return {
        "file": f"../materials/{pack.file_name}",
        "sha256": pack.sha256,
        "byteSize": pack.byte_size,
        "maxResolution": pack.resolution,
        "mapTypes": list(pack.map_types),
        "compression": pack.compression,
        "container": "glb-material-cube",
        "materialIndex": 0,
    }
