"""Room/material identity and dependencies, without a runtime loading policy."""

# ### Imports ###
from __future__ import annotations

import hashlib
from collections import defaultdict

from housemaker.global_scale import GlobalScale
from housemaker.material_pack_export import MaterialPackExport
from housemaker.room_zones import RoomZoneLayout
from housemaker.scene_export import _parse_glb


# ### Physical runtime coordinates ###
def _runtime_point(x, y, z, scale):
    return [float(x * scale.x), float(z * scale.z), float(-y * scale.y)]


# ### Room and material dependency description ###
def build_room_asset_manifest(
    layout: RoomZoneLayout,
    material_export: MaterialPackExport,
    atlases=(),
    source_room_ids=None,
    *,
    global_scale: GlobalScale | None = None,
) -> dict[str, object]:
    """Describe exported assets; do not prescribe lazy-loading or hiding rules.

    Bounds and outlines already include global scaling and use glTF Y-up.
    Shared assets deliberately belong to every potentially dependent room.
    Material indices match the main GLB, not each one-cube material GLB.
    """

    scale = global_scale or GlobalScale()
    source_room_ids = source_room_ids or {}
    room_ids = {room.room_id for room in layout.rooms}
    rooms_by_material = defaultdict(set)
    nodes_by_room = defaultdict(set)
    materials_by_index = {pack.material_index: pack for pack in material_export.packs}
    document, _binary = _parse_glb(
        material_export.stripped_model.glb_bytes, "The room export GLB is invalid."
    )
    meshes = document.get("meshes", [])
    for node in document.get("nodes", []):
        mesh_index = node.get("mesh")
        if not isinstance(mesh_index, int) or not 0 <= mesh_index < len(meshes):
            continue
        mesh = meshes[mesh_index]
        extras = {**mesh.get("extras", {}), **node.get("extras", {})}
        owners = set(extras.get("housemaker_room_ids", ())) & room_ids
        single = extras.get("housemaker_room_id")
        if single in room_ids:
            owners.add(single)
        for owner in owners:
            if node.get("name"):
                nodes_by_room[owner].add(node["name"])
        for primitive in mesh.get("primitives", []):
            pack = materials_by_index.get(primitive.get("material"))
            if pack is not None:
                rooms_by_material[pack.material_id].update(owners)

    atlas_bindings = []
    for item in atlases:
        atlas = item.atlas
        owners = set()
        unknown = False
        for placement in atlas.placements:
            found = source_room_ids.get(placement.object_id, ())
            owners.update(found)
            unknown |= not bool(found)
        if atlas.atlas_type == "surface":
            # A shared Surface Atlas can also appear on unclassified exterior
            # geometry. Tagged interior uses alone cannot narrow ownership.
            owners = set(room_ids)
        elif unknown:
            owners = set(room_ids)
        if atlas.room_id in room_ids:
            owners.add(atlas.room_id)
        for pack in material_export.packs:
            if pack.atlas_id == atlas.atlas_id:
                rooms_by_material[pack.material_id].update(owners)
        atlas_bindings.append(
            {
                "atlasId": atlas.atlas_id,
                "name": atlas.name,
                "type": atlas.atlas_type,
                "resolution": atlas.resolution,
                "roomIds": sorted(owners),
                "shared": atlas.atlas_type == "surface" or atlas.room_id is None,
                "textureSourceIds": sorted(p.object_id for p in atlas.placements),
            }
        )

    material_bindings = []
    for pack in material_export.packs:
        owners = rooms_by_material[pack.material_id] or room_ids
        material_bindings.append(
            {
                **pack.to_dict(),
                "file": f"../materials/{pack.file_name}",
                "sha256": hashlib.sha256(pack.glb_bytes).hexdigest(),
                "roomIds": sorted(owners),
            }
        )

    rooms = []
    for room in layout.rooms:
        if room.polygon.is_empty:
            continue
        x0, y0, x1, y1 = room.polygon.bounds
        floor = room.floor_z_meters
        rooms.append(
            {
                "id": room.room_id,
                "name": room.room_name,
                "levelIndex": room.level_index,
                "fallback": room.is_fallback,
                "bounds": {
                    "min": _runtime_point(x0, y1, floor, scale),
                    "max": _runtime_point(x1, y0, room.top_z_meters, scale),
                },
                "floorOutline": [
                    _runtime_point(x, y, floor, scale)
                    for x, y in room.polygon.exterior.coords
                ],
                "floorHoles": [
                    [_runtime_point(x, y, floor, scale) for x, y in ring.coords]
                    for ring in room.polygon.interiors
                ],
                "surfaceNodeNames": sorted(nodes_by_room[room.room_id]),
                "materialIds": sorted(
                    binding["materialId"]
                    for binding in material_bindings
                    if room.room_id in binding["roomIds"]
                ),
            }
        )
    return {
        "version": 1,
        "coordinateSystem": "gltf-y-up",
        "previewMaxSize": 256,
        "rooms": rooms,
        "portals": [
            {
                "id": portal.portal_id,
                "kind": portal.kind,
                "roomIds": list(portal.room_ids),
                "levelIndices": list(portal.level_indices),
                "walkable": portal.walkable,
            }
            for portal in layout.portals
        ],
        "atlases": atlas_bindings,
        "materials": material_bindings,
    }
