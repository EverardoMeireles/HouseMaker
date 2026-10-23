"""Optional, geometry-preserving KTX2/Basis material-pack compression."""

# ### Imports ###
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import os
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

from housemaker.scene_export import _parse_glb, _serialize_glb

# ### Constants ###
KTX_SOFTWARE_VERSION = "4.4.2"
KTX_WINDOWS_URL = (
    "https://github.com/KhronosGroup/KTX-Software/releases/download/"
    f"v{KTX_SOFTWARE_VERSION}/KTX-Software-{KTX_SOFTWARE_VERSION}-Windows-x64.exe"
)
KTX_WINDOWS_SHA256 = "1f323b0fec19794f5e6c0425a61d4b1da396872a10be862d105f4f4b2d2957fe"
KTX2_IDENTIFIER = b"\xabKTX 20\xbb\r\n\x1a\n"
KTX2_EXTENSION = "KHR_texture_basisu"
DEFAULT_ENCODING_THREADS = 4
ENCODING_PROFILE = "uastc2-zstd9-mips-rgba-transfer-primaries-ceil4-v3"
PREVIEW_ENCODING_PROFILE = "etc1s-q160-clevel2-mips-rgba-transfer-primaries-ceil4-v1"
_SRGB_TEXTURE_NAMES = frozenset(
    {
        "baseColorTexture",
        "emissiveTexture",
        "diffuseTexture",
        "diffuseTransmissionColorTexture",
        "specularGlossinessTexture",
        "sheenColorTexture",
        "specularColorTexture",
    }
)


# ### Encoder discovery and explicit portable setup ###
class Ktx2CompressionError(RuntimeError):
    """An explicit compression or optional-encoder setup failure."""


def default_ktx2_directory() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    return base / "HouseMaker" / "tools" / "ktx" / KTX_SOFTWARE_VERSION


def find_ktx2_encoder() -> Path | None:
    """Find a local encoder without installing tools or accessing the network."""

    override = os.environ.get("HOUSEMAKER_TOKTX")
    if override:
        candidate = Path(override).expanduser()
        return candidate.resolve() if candidate.is_file() else None
    executable = "toktx.exe" if os.name == "nt" else "toktx"
    candidates = [default_ktx2_directory() / "bin" / executable]
    discovered = shutil.which(executable) or shutil.which("ktx")
    if discovered:
        candidates.append(Path(discovered))
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        if directory := os.environ.get(variable):
            candidates.append(Path(directory) / "KTX-Software" / "bin" / executable)
    return next((path.resolve() for path in candidates if path.is_file()), None)


def require_ktx2_encoder(tool_path: Path | str | None = None) -> Path:
    candidate = (
        Path(tool_path).expanduser() if tool_path is not None else find_ktx2_encoder()
    )
    if candidate is None or not candidate.is_file():
        raise Ktx2CompressionError(
            "KTX2/Basis compression needs the official Khronos KTX Software encoder. "
            "Install KTX Software 4.4.2 and restart HouseMaker, or set HOUSEMAKER_TOKTX "
            "to its toktx executable. Uncheck 'Use KTX2/Basis compression' to export "
            "without it. See docs/ktx2-compression.md for portable setup."
        )
    return candidate.resolve()


def provision_windows_ktx2_encoder(
    *, extractor_path: Path | str, directory: Path | None = None
) -> Path:
    """Explicitly download and extract verified official tools; never run the installer.

    The caller supplies an existing 7-Zip executable. Export itself never invokes
    this network/setup function. No administrator rights, registry or PATH edits.
    """

    if os.name != "nt":
        raise Ktx2CompressionError(
            "Portable setup supports Windows x64; install KTX Software on this OS."
        )
    extractor = Path(extractor_path).resolve()
    if not extractor.is_file():
        raise Ktx2CompressionError(
            "Portable KTX setup requires an existing 7-Zip executable."
        )
    target = Path(directory) if directory is not None else default_ktx2_directory()
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="housemaker-ktx-setup-") as temporary:
        archive = Path(temporary) / "ktx-software.exe"
        request = Request(
            KTX_WINDOWS_URL, headers={"User-Agent": "HouseMaker-KTX2/1.0"}
        )
        try:
            with urlopen(request, timeout=60) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        except OSError as error:
            raise Ktx2CompressionError(
                f"The official KTX Software download failed: {error}"
            ) from error
        if hashlib.sha256(archive.read_bytes()).hexdigest() != KTX_WINDOWS_SHA256:
            raise Ktx2CompressionError(
                "The downloaded KTX Software failed its SHA-256 integrity check."
            )
        _run_encoder_command(
            [
                str(extractor),
                "x",
                str(archive),
                "-y",
                f"-o{target}",
                "bin/*",
                "LICENSE*",
            ],
            timeout=120,
        )
    encoder = target / "bin" / "toktx.exe"
    if not encoder.is_file():
        raise Ktx2CompressionError(
            "The verified KTX archive did not contain bin/toktx.exe."
        )
    _run_encoder_command([str(encoder), "--version"], timeout=30)
    return encoder


# ### Bounded, cached image encoding ###
def encode_ktx2_image(
    payload: bytes,
    *,
    color_space: str,
    tool_path: Path | str | None = None,
    cache_directory: Path | None = None,
    preview: bool = False,
) -> bytes:
    """Encode without swizzling ORM channels or converting RGB normal maps to RG."""

    if color_space not in {"srgb", "linear"}:
        raise ValueError("KTX2 color space must be srgb or linear.")
    if not isinstance(preview, bool):
        raise TypeError("The KTX2 preview profile must be a boolean.")
    tool = require_ktx2_encoder(tool_path)
    use_etc1s = preview and color_space == "srgb"
    profile = PREVIEW_ENCODING_PROFILE if use_etc1s else ENCODING_PROFILE
    identity = hashlib.sha256(
        profile.encode() + b"\0" + color_space.encode() + b"\0" + payload
    ).hexdigest()
    cache = (
        Path(cache_directory)
        if cache_directory is not None
        else default_ktx2_directory() / "encoded-images"
    )
    cached = cache / f"{identity}.ktx2"
    if cached.is_file():
        result = cached.read_bytes()
        if len(result) >= 80 and result.startswith(KTX2_IDENTIFIER):
            return result
    try:
        with Image.open(BytesIO(payload)) as source:
            image = source.convert("RGBA")
            # KHR_texture_basisu requires level-zero dimensions divisible by four.
            # Resample, never pad: padding would shift normalized UV content.
            dimensions = tuple((size + 3) // 4 * 4 for size in image.size)
            if image.size != dimensions:
                image = image.resize(dimensions, Image.Resampling.LANCZOS)
    except (OSError, UnidentifiedImageError) as error:
        raise Ktx2CompressionError(
            "A material image cannot be decoded for KTX2 compression."
        ) from error
    with tempfile.TemporaryDirectory(prefix="housemaker-ktx2-") as temporary:
        source_path, output_path = (
            Path(temporary) / "input.png",
            Path(temporary) / "output.ktx2",
        )
        image.save(source_path, format="PNG")
        threads = str(DEFAULT_ENCODING_THREADS)
        if tool.stem.lower() == "ktx":
            codec_options = (
                ["--encode", "basis-lz", "--qlevel", "160", "--clevel", "2"]
                if use_etc1s
                else ["--encode", "uastc", "--uastc-quality", "2", "--zstd", "9"]
            )
            command = [
                str(tool),
                "create",
                "--format",
                "R8G8B8A8_SRGB" if color_space == "srgb" else "R8G8B8A8_UNORM",
                *codec_options,
                "--generate-mipmap",
                "--assign-tf",
                color_space,
                "--assign-primaries",
                "bt709" if color_space == "srgb" else "none",
                "--threads",
                threads,
                str(source_path),
                str(output_path),
            ]
        else:
            codec_options = (
                ["--encode", "etc1s", "--qlevel", "160", "--clevel", "2"]
                if use_etc1s
                else ["--encode", "uastc", "--uastc_quality", "2", "--zcmp", "9"]
            )
            command = [
                str(tool),
                "--t2",
                *codec_options,
                "--genmipmap",
                "--assign_oetf",
                color_space,
                "--assign_primaries",
                "bt709" if color_space == "srgb" else "none",
                "--upper_left_maps_to_s0t0",
                "--threads",
                threads,
                str(output_path),
                str(source_path),
            ]
        _run_encoder_command(command)
        try:
            result = output_path.read_bytes()
        except OSError as error:
            raise Ktx2CompressionError(
                "The KTX encoder did not create its output texture."
            ) from error
    if not result.startswith(KTX2_IDENTIFIER):
        raise Ktx2CompressionError("The KTX encoder returned an invalid KTX2 texture.")
    cache.mkdir(parents=True, exist_ok=True)
    # Unique staging files avoid races between exports while replacing atomically.
    with tempfile.NamedTemporaryFile(
        dir=cache, prefix=identity + ".", suffix=".tmp", delete=False
    ) as output:
        staging = Path(output.name)
        output.write(result)
    try:
        staging.replace(cached)
    finally:
        staging.unlink(missing_ok=True)
    return result


def _run_encoder_command(command: list[str], *, timeout: int = 1800) -> None:
    environment = os.environ.copy()
    # A shell-wide override must not flip images or silently swizzle normal/ORM maps.
    environment.pop("TOKTX_OPTIONS", None)
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Ktx2CompressionError(
            f"The local KTX encoder could not run: {error}"
        ) from error
    if result.returncode:
        details = (result.stderr or result.stdout or "No diagnostic output.").strip()[
            -1800:
        ]
        raise Ktx2CompressionError(f"The local KTX encoder failed: {details}")


# ### Material texture references and embedded payloads ###
def _texture_bindings(value):
    if isinstance(value, dict):
        for name, child in value.items():
            if (
                name.endswith("Texture")
                and isinstance(child, dict)
                and isinstance(child.get("index"), int)
            ):
                yield child, "srgb" if name in _SRGB_TEXTURE_NAMES else "linear"
            else:
                yield from _texture_bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _texture_bindings(child)


def _image_bytes(document, binary, image):
    if "bufferView" in image:
        return _view_bytes(document, binary, image["bufferView"])
    uri = image.get("uri", "")
    if isinstance(uri, str) and uri.startswith("data:") and ";base64," in uri:
        try:
            return base64.b64decode(uri.split(",", 1)[1], validate=True)
        except ValueError as error:
            raise Ktx2CompressionError(
                "An embedded material image is invalid."
            ) from error
    raise Ktx2CompressionError(
        "KTX2 export requires embedded material images, not external paths."
    )


def _view_bytes(document, binary, index):
    views = document.get("bufferViews", [])
    if not isinstance(index, int) or not 0 <= index < len(views):
        raise Ktx2CompressionError("A material references an invalid buffer view.")
    view = views[index]
    offset, size = view.get("byteOffset", 0), view.get("byteLength", 0)
    if (
        not isinstance(offset, int)
        or not isinstance(size, int)
        or offset < 0
        or size < 0
        or offset + size > len(binary)
    ):
        raise Ktx2CompressionError("An embedded GLB buffer view has invalid bounds.")
    return binary[offset : offset + size]


# ### Geometry-preserving material GLB rewriting ###
def compress_material_glb(
    payload: bytes,
    *,
    tool_path: Path | str | None = None,
    encoder=None,
    preview: bool = False,
) -> bytes:
    """Replace material images only; keep accessor bytes, UV1 and scene identity.

    ``encoder`` is an injectable image function accepting bytes and the keyword
    ``color_space``; normal exports use the verified local Khronos CLI. A texture
    shared by color and data slots is duplicated to preserve both transfer functions.
    Preview color maps use ETC1S; full color and all data maps use high-quality UASTC.
    """

    if not isinstance(preview, bool):
        raise TypeError("The KTX2 preview profile must be a boolean.")
    document, binary = _parse_glb(payload, "The KTX2 material GLB is invalid.")
    buffers = document.get("buffers", [])
    if len(buffers) != 1 or buffers[0].get("uri") is not None:
        raise Ktx2CompressionError("KTX2 export requires a single embedded GLB buffer.")
    if buffers[0].get("byteLength", 0) > len(binary):
        raise Ktx2CompressionError("The embedded GLB buffer is truncated.")
    for view in document.get("bufferViews", []):
        if view.get("buffer", 0) != 0 or "EXT_meshopt_compression" in view.get(
            "extensions", {}
        ):
            raise Ktx2CompressionError(
                "KTX2 image rewriting does not support external or meshopt buffers."
            )
    textures = document.get("textures", [])
    if not textures:
        return payload
    if encoder is None:
        tool = require_ktx2_encoder(tool_path)

        def encoder(image, *, color_space):
            return encode_ktx2_image(
                image, color_space=color_space, tool_path=tool, preview=preview
            )

    # Duplicate texture records, not geometry, for unusual shared-color/data inputs.
    texture_roles = {}
    variants = {}
    for binding, role in _texture_bindings(document.get("materials", [])):
        index = binding["index"]
        if not 0 <= index < len(textures):
            raise Ktx2CompressionError("A material references an unavailable texture.")
        key = index, role
        if key not in variants:
            if index not in texture_roles:
                variants[key] = index
                texture_roles[index] = role
            else:
                variants[key] = len(textures)
                textures.append(copy.deepcopy(textures[index]))
                texture_roles[variants[key]] = role
        binding["index"] = variants[key]

    original_images = document.get("images", [])
    new_images, image_payloads, compressed = [], [], {}
    for index, texture in enumerate(textures):
        extensions = texture.setdefault("extensions", {})
        source = extensions.get(
            KTX2_EXTENSION, extensions.get("EXT_texture_webp", {})
        ).get("source", texture.get("source"))
        if not isinstance(source, int) or not 0 <= source < len(original_images):
            raise Ktx2CompressionError(
                "A texture references an unavailable embedded image."
            )
        role = texture_roles.get(index, "linear")
        key = source, role
        if key not in compressed:
            image = copy.deepcopy(original_images[source])
            original = _image_bytes(document, binary, image)
            result = (
                original
                if original.startswith(KTX2_IDENTIFIER)
                else encoder(original, color_space=role)
            )
            if not isinstance(result, bytes) or not result.startswith(KTX2_IDENTIFIER):
                raise Ktx2CompressionError(
                    "The material image encoder returned invalid KTX2 bytes."
                )
            image.pop("uri", None)
            image.pop("bufferView", None)
            image["mimeType"] = "image/ktx2"
            compressed[key] = len(new_images)
            new_images.append(image)
            image_payloads.append(result)
        texture.pop("source", None)
        extensions.pop("EXT_texture_webp", None)
        extensions[KTX2_EXTENSION] = {"source": compressed[key]}

    compact = _replace_image_views(
        document, binary, original_images, new_images, image_payloads
    )
    for field in ("extensionsUsed", "extensionsRequired"):
        names = set(document.get(field, ()))
        names.discard("EXT_texture_webp")
        names.add(KTX2_EXTENSION)
        document[field] = sorted(names)
    return _serialize_glb(document, compact)


def _replace_image_views(document, binary, old_images, new_images, image_payloads):
    old_image_views = {
        image["bufferView"] for image in old_images if "bufferView" in image
    }
    referenced = set()

    def collect(value):
        if isinstance(value, dict):
            if isinstance(value.get("bufferView"), int):
                referenced.add(value["bufferView"])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    without_images = {key: value for key, value in document.items() if key != "images"}
    collect(without_images)
    result, new_views, remap = bytearray(), [], {}
    for index, view in enumerate(document.get("bufferViews", [])):
        if index in old_image_views and index not in referenced:
            continue
        data = _view_bytes(document, binary, index)
        result.extend(b"\0" * (-len(result) % 4))
        remap[index] = len(new_views)
        updated = copy.deepcopy(view)
        updated.update(buffer=0, byteOffset=len(result), byteLength=len(data))
        new_views.append(updated)
        result.extend(data)

    def update(value):
        if isinstance(value, dict):
            if isinstance(value.get("bufferView"), int):
                old = value["bufferView"]
                if old not in remap:
                    raise Ktx2CompressionError(
                        "A GLB dependency references an unavailable buffer view."
                    )
                value["bufferView"] = remap[old]
            for child in value.values():
                update(child)
        elif isinstance(value, list):
            for child in value:
                update(child)

    update(without_images)
    for image, data in zip(new_images, image_payloads):
        result.extend(b"\0" * (-len(result) % 4))
        image["bufferView"] = len(new_views)
        new_views.append(
            {"buffer": 0, "byteOffset": len(result), "byteLength": len(data)}
        )
        result.extend(data)
    document["images"], document["bufferViews"] = new_images, new_views
    document["buffers"][0]["byteLength"] = len(result)
    return bytes(result)


# ### Explicit optional-tool command line ###
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Explicitly download verified portable Windows tools",
    )
    parser.add_argument(
        "--extractor",
        type=Path,
        help="Existing 7-Zip executable used to extract, not run, the installer",
    )
    options = parser.parse_args()
    if options.setup:
        if options.extractor is None:
            parser.error(
                "--setup requires --extractor pointing to an existing 7-Zip executable"
            )
        print(provision_windows_ktx2_encoder(extractor_path=options.extractor))
    else:
        print(require_ktx2_encoder())


if __name__ == "__main__":
    main()
