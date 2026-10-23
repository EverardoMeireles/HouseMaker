# Room and material export

## Workflow

1. Confirm the walls, openings and stairs. In the Canvas tab, enable **Show rooms** below the angle/dead-end highlight option to inspect the detected rooms. Colors are an overlay only: they do not edit walls, floors or stair holes. Open or uncertain areas remain unassigned/fallback rather than becoming guessed closed rooms.
2. Place objects and assign their textures as usual. **Export** prepares room-based Object atlases and publishes their sections in the Atlas tab after a successful export. Unrequired library textures and unrelated atlases are retained.
3. Inspect **Surface atlases**, each room's section, and **Shared / unassigned object atlases**. Object atlas dimensions use the smallest supported size that fits the existing source texel density; larger rooms can need multiple pages. Sources used in several rooms, crossing room boundaries or lacking reliable ownership use shared atlases. Surface atlases remain shared.
4. Rebake **visibility (PVS)** when its status becomes outdated after geometry, placement or grouping changes. An invalid or stale bake is omitted rather than hiding geometry incorrectly.

Doorways provide room portals; stairs connect level zones. This classification does not change structural floor connectivity. Surface geometry is batched by room while sharing Surface materials. Triangles spanning rooms and uncertain structural geometry stay in conservative shared groups. Objects retain their numbered identities and instance relationships.

## Temporary output paths

Export currently writes to this fixed development destination without a save dialog:

```text
C:\Users\Everardo\Documents\everardomeireles\public\
  models\housemaker_export.glb
  models\housemaker_export.json
  models\housemaker_export.pvs.mjs
  materials\<material-name>__<material-id>.glb
```

The model files are replaced on subsequent exports. Material packs are written under stable generated filenames; obsolete files are not automatically deleted.

The main GLB keeps the house geometry, transforms, UVs, material indices and object IDs. Starting-area and shared Surface materials retain lit base-color previews up to 1024px; other materials keep tiny fallbacks up to 128px. Each textured material has an external 1024px preview GLB and a separate full-resolution GLB, each containing one unit cube with its material assigned. Full packs retain their available PBR maps; preview tiers omit normal/ORM/AO maps but retain required opacity/glass/emissive behavior. A pack is a material carrier, not another copy of the room or object geometry. Untextured materials do not require a separate pack.

The **Set FPS starting point** and **Use current FPS position** controls select the initial room and camera. **Use KTX2/Basis compression**, below the PVS bake button, optionally compresses external material packs; portable main-GLB fallbacks remain JPEG/PNG. See the [progressive loading workflow and JSON contract](lazy-material-loading.md) and [optional encoder setup](ktx2-compression.md).

## JSON contract

The companion JSON's additive `roomAssets` section describes:

- `rooms`: stable IDs, display names, level indices, fallback flags, bounds, floor outlines/holes, structural node names and material dependencies.
- `portals`: doorway/stair connections and their room and level IDs.
- `atlases`: atlas IDs, types, dimensions, shared/room ownership and texture-source IDs.
- `materials`: material IDs/names, main-GLB material indices, optional atlas IDs, room dependencies and embedded/preview/full variants. External variants include relative pack paths, SHA-256 hashes, byte sizes, resolutions, map types and compression. Pack paths are relative to the JSON file, such as `../materials/...preview.glb`.

Room coordinates are final physical **glTF Y-up** coordinates with global multipliers already applied. Do not apply those multipliers a second time. `materialIndex` identifies the material in the main GLB, not the material index inside its one-cube pack.

The companion `materialStreaming` section now defines a versioned loading policy: starting camera, skippable first-visit tutorial, visible/current/adjacent-room priorities, shared-material handling, scheduler and GPU budgets, cache/eviction rules, binding invariants and retry/fallback behavior. The existing `roomAssets` table remains the source of room/material dependencies.

HouseMaker exports this policy but does not implement streaming or the tutorial in the external R3F app. Until that app follows the policy and replaces the matching materials, it displays the embedded previews. The cube itself must not be added to the house scene. Geometry still loads as one main GLB; this feature progressively loads materials only.
