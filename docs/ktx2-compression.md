# Optional KTX2/Basis export compression

## Export behavior

The Canvas tab's **Use KTX2/Basis compression** checkbox is off by default. When
enabled, separately exported preview and full material cube GLBs use embedded
KTX2 images and require `KHR_texture_basisu`. The starting-scene fallback images
remain portable JPEG/PNG so they can render before the runtime's Basis decoder
has loaded. Geometry, UVs, object identities and material settings are unchanged.

Preview color images use compact ETC1S/BasisLZ quality 160. Full-resolution color
images and all non-color data maps use UASTC quality 2 and lossless Zstandard
level 9. Both profiles include complete mipmaps.
Base-color/emissive maps carry sRGB transfer metadata; normal/ORM/AO maps stay
linear. Normal RGB and ORM channel meanings are retained without swizzling.
For unusual image dimensions, the complete image is resampled up by at most
three pixels per axis to meet KTX2's multiples-of-four requirement; it is never
padded, so normalized UV placement is unchanged.
Encoding uses at most four threads per image. Content-addressed local image
caching avoids recompressing identical maps on subsequent exports.

KTX2 principally reduces GPU texture memory. Depending on image content, its
download size can be larger than an aggressively compressed JPEG, especially
for high-quality full materials. The checkbox deliberately preserves this choice
rather than assuming KTX2 always produces the smallest download.

## Install the optional encoder

HouseMaker uses the official [Khronos KTX Software tools](https://github.com/KhronosGroup/KTX-Software/releases/tag/v4.4.2).
Install KTX Software 4.4.2, restart HouseMaker, and enable the checkbox. Discovery
checks the application tool cache, PATH, and the standard Windows KTX-Software
installation directory. Alternatively set `HOUSEMAKER_TOKTX` to the full path of
`toktx.exe` (or the newer `ktx` executable).

Export never downloads software, installs global packages or silently ignores an
enabled compression option. If the encoder is unavailable, export stops with an
actionable message before writing the runtime bundle. Disable the checkbox to
export ordinary compressed JPEG/PNG instead.

## Explicit portable setup on Windows x64

If 7-Zip is already installed, the optional setup command downloads the official
6.4 MB Windows release, verifies its pinned SHA-256, and extracts its `bin` tools
to `%LOCALAPPDATA%/HouseMaker/tools/ktx/4.4.2`. It does **not** execute the
installer, request administrator access, change PATH, or modify the registry:

```powershell
.venv\Scripts\python.exe -m housemaker.ktx2_compression --setup --extractor "C:\Program Files\7-Zip\7z.exe"
```

Inspect the currently discovered encoder without downloading anything:

```powershell
.venv\Scripts\python.exe -m housemaker.ktx2_compression
```

## R3F integration requirement

The downstream viewer must configure Three.js `KTX2Loader` and its Basis
transcoder, call `detectSupport(renderer)`, then attach it with
`GLTFLoader.setKTX2Loader(...)` **before** loading compressed material cube GLBs.
Ordinary scene fallbacks remain available while the decoder and material packs
are loading. GPU texture formats are selected by the viewer for each device.

References: [Three.js KTX2Loader](https://threejs.org/docs/pages/KTX2Loader.html),
[Khronos KHR_texture_basisu requirements](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_texture_basisu).
