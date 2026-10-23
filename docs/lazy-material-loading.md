# Progressive material loading

## HouseMaker workflow

1. Build the house, confirm openings and place objects normally. Optionally enable **Show rooms** to inspect room ownership.
2. Under **Levels**, click **Set FPS starting point**, then click a point on the 2D plan or a floor in the dedicated 3D window. Right-click or Escape cancels placement. A cyan marker and direction line show the saved point.
3. For a specific viewing direction, navigate the FPS camera to the desired starting position and click **Use current FPS position** instead. The physical eye height from Settings is removed before storing the floor-relative point and added back once during export. **Clear** restores automatic starting-point selection.
4. Optionally check **Use KTX2/Basis compression**, directly below **Bake visibility(PVS)**. It is unchecked by default. The optional local encoder must already be available; see [KTX2 compression setup](ktx2-compression.md).
5. Save the project and click **Export**. The camera point and compression preference persist in the project. A point bound to an excluded or missing level must be cleared or replaced before export.

The starting point follows its level's plan transform. Its vertical offset follows level floor elevation; physical FPS eye height is not multiplied by Global Z. Clicking a raised surface can store an additional floor-relative offset, so choose the floor rather than furniture when placing the point.

Without an authored point, export picks a representative interior point from a detected room, preferring the ground level. If no rooms can be identified, it falls back to the center of an included plan. Inspect or explicitly set this position when room detection is uncertain.

This change exports assets and machine-readable instructions. The external R3F viewer still needs to implement the loader and tutorial; HouseMaker does not modify that application automatically.

## Exported texture tiers

The main GLB contains the complete geometry but not its original full-resolution texture payloads. It remains immediately renderable with small lit materials:

| Tier | Location | Images and maps |
| --- | --- | --- |
| Embedded startup preview | Main house GLB | Base color up to 1024px for the starting room/nearby object atlases and shared Surface atlases. |
| Embedded fallback | Main house GLB | Base color up to 128px elsewhere, retained even after upgrades. |
| External preview | One material cube GLB per material | Up to 1024px base color for early room prefetching. |
| External full | One material cube GLB per material | Original-resolution material with its original available PBR maps. Atlas size can be 512, 1024, 2048 or 4096; full does not force an upscale to 4096. |

Preview dimensions are upper bounds along the longest image axis, not a forced square or an upscale. Preview and full tiers use the same Atlas layout, UVs and material identity. Normal, roughness/metallic and AO maps are omitted from preview tiers; representative scalar PBR values keep them lit. Required opacity, glass/transmission and emissive behavior are retained rather than replacing the entire scene with unlit placeholders.

The starting-area selection includes the starting room's material dependencies and object sources within an 8-meter radius. Shared Surface atlases receive startup previews, but this does **not** mean all their full maps should download immediately.

With compression disabled, embedded and external previews use compressed JPEG/PNG and full packs keep their source image data. When enabled, external preview and full packs use KTX2/Basis; embedded images stay portable JPEG/PNG. First-time KTX2 encoding adds export work, and subsequent identical image encodes reuse the local content-addressed cache. Export does not install or download an encoder implicitly.

Output remains:

```text
public/models/housemaker_export.glb
public/models/housemaker_export.json
public/models/housemaker_export.pvs.mjs
public/materials/<material-name>__<material-id>.preview.glb
public/materials/<material-name>__<material-id>.glb
```

The fixed development `public` directory is `C:\Users\Everardo\Documents\everardomeireles\public`. Material URLs in the JSON are relative to that JSON, for example `../materials/...preview.glb`. Each material GLB contains one cube as a material carrier; do not add its cube to the house scene.

## JSON entry points

The existing `roomAssets` dependency table is supplemented with a declarative policy:

```json
{
  "materialStreaming": {
    "format": "housemaker-material-streaming",
    "version": 1,
    "enabled": true,
    "materialTable": "roomAssets.materials"
  }
}
```

The full generated policy contains these sections:

- `initialCamera`: physical glTF Y-up camera position, floor position, look direction, up vector, FOV, level and room IDs. `globalScaleAlreadyApplied` is true. Do not apply the global multipliers again. The same values are published in `firstPersonCamera`.
- `bootstrap`: ready condition, starting and adjacent room IDs, embedded startup material IDs and radius. Geometry and its embedded materials must render before reporting bootstrap readiness.
- `tutorial`: first-visit navigation help that can be skipped or replayed. Loading continues while open. Bindings come from the runtime's active control map, including its object-visibility hotkey; export does not invent that application's key binding. Missing actions are omitted.
- `priorityOrder`: ordered download priorities that the runtime reevaluates as visibility, room or camera heading changes.
- `sharedMaterials`: shared previews load early; full-resolution priority comes from visible primitives, not every room's broad dependency list.
- `scheduler`: download/decode/upload concurrency, upload work budget and shader readiness before replacement.
- `roomSelection`: polygon-and-height membership, transition hysteresis and fail-open behavior outside known rooms.
- `cache`: content-addressed deduplication, optional persistent browser caching, GPU residency budgets and eviction rules.
- `binding`: stable material IDs, fallback material indices, replacement scope and invariants.
- `failurePolicy`: retry limits, hash verification, portable fallbacks and failure reporting.
- `textureCompression`: external pack encoding and required decoder, separately from the always-portable embedded tier.
- `telemetry`: first textured frame, current-room full readiness, downloads, residency and material errors.

`roomAssets.materials` holds one entry per externally loadable material. Each includes `materialId`, main-house `materialIndex`, room IDs, Atlas ownership, `bootstrap`, and `variants`:

- `embedded`: identifies the material already present in the main GLB, its maximum image size and map set.
- `preview` and `full`: relative `file`, `sha256`, `byteSize`, `maxResolution`, `mapTypes`, `compression`, cube container type and material index **inside the pack** (zero).

Main-house material identity is stamped into material extras as `housemaker_material_id`. Prefer that ID over display names. Only fall back to a main-house material index when the loaded house matches `asset.sha256`. Reuse a single upgraded material for every matching main-scene mesh, detached instance prototype, instance and mirrored half; do not lose object-specific transforms, visibility or shadow flags.

## Runtime loading sequence

1. Load the JSON and main geometry GLB, resolve matching material IDs, then apply `initialCamera`.
2. Render the embedded lit materials immediately. Show the skippable tutorial once bootstrap has rendered; there is no enforced waiting period and no dependency on the user reading it.
3. Upgrade visible low-quality fallbacks to external previews first. Prioritize full materials visible in the current room, then other visible materials (using portals/PVS as hints), then the non-shared remainder of the current room.
4. Prefetch previews of adjacent rooms, favoring the current heading. Full adjacent-room materials are lower priority and budget-limited. Other rooms receive idle preview prefetching.
5. Reprioritize on camera moves/turns, room transitions and visibility changes. Cancel obsolete queued work; do not interrupt a still-needed shared asset.
6. Keep the existing material until the replacement is downloaded, decoded, uploaded and shader-ready. Swap atomically per material rather than briefly clearing its textures.

Default budgets are four concurrent downloads, two decodes and at most one GPU upload per frame with a 2ms work budget. Desktop/mobile GPU material budgets are 512/128 MiB and may be reduced by the runtime. Include all maps, mipmaps and resident variants in that budget. Retain the previous room for ten seconds, then evict least-recently-used nonvisible full tiers before previews. Dispose GPU resources only after their final consumer releases them.

Room bounds are a broad-phase check, not sufficient membership proof. Use the room floor polygon and holes plus vertical bounds. Doorway/stair portals and valid PVS guide prefetching; neither is the sole proof that a material is currently visible. Unknown room membership must fall back to frustum/distance prioritization rather than hiding geometry.

For KTX2 packs, initialize the runtime's compatible KTX2/Basis loader before loading those packs. If unsupported or decoding fails, keep the portable embedded tier and report the problem. The export does not include a second uncompressed copy of every full pack.

## Failure handling and scope

The policy permits three attempts with 1s and 3s retry delays. A failed full pack retains its preview; a failed preview retains the embedded material. A hash mismatch is discarded and refetched. Missing materials must never make geometry disappear.

This feature streams **materials**, not the geometry GLB. It does not make a large geometry payload lazy, force all downloads to finish within the tutorial, or implement a new culling scheme. Use telemetry to verify first-render latency, visible material upgrades and upload stalls in the eventual R3F integration.
