# ### Imports ###
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import trimesh


# ### Integrity constants ###
UV_FINGERPRINT_VERSION = "per-face-uv-triangle-sha256-v1"
UV_FINGERPRINT_QUANTIZATION_SCALE = 1_000_000
MAX_ABSOLUTE_UV_COORDINATE = 1_000_000.0


# ### Public data models ###
@dataclass(frozen=True)
class UvFingerprint:
    """One deterministic per-face UV-layout identity."""

    version: str
    sha256: str
    face_count: int


# ### Public exceptions ###
class UvIntegrityError(ValueError):
    """Raised when a model's required UV layout cannot be verified."""


# ### Public integrity API ###
def build_uv_fingerprint(glb_bytes: bytes) -> UvFingerprint:
    """Fingerprint ordered face UV triangles without using vertex indices.

    Each triangle's three quantized UV corners are sorted before hashing, and
    primitive fingerprints are sorted before their final combination. This
    preserves face-to-UV ownership inside each primitive while tolerating
    primitive reordering, winding changes, vertex reindexing, and redundant
    vertex welding.
    """

    scene = _load_glb_scene(glb_bytes)
    geometry_fingerprints: list[tuple[int, bytes]] = []
    face_count = 0
    for geometry in scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        faces = np.asarray(geometry.faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1:] != (3,):
            raise UvIntegrityError(
                "UV integrity requires triangle mesh faces."
            )
        if len(faces) == 0:
            continue
        uv = getattr(geometry.visual, "uv", None)
        if uv is None:
            raise UvIntegrityError(
                "The GLB is missing UV coordinates needed for integrity "
                "validation."
            )
        normalized_uv = np.asarray(uv, dtype=float)
        if normalized_uv.ndim != 2 or normalized_uv.shape[1:] != (2,):
            raise UvIntegrityError("The GLB contains malformed UV coordinates.")
        if len(faces) and (
            np.any(faces < 0) or np.any(faces >= len(normalized_uv))
        ):
            raise UvIntegrityError("The GLB contains invalid UV vertex indices.")
        if not np.all(np.isfinite(normalized_uv)):
            raise UvIntegrityError("The GLB contains non-finite UV coordinates.")
        if np.any(np.abs(normalized_uv) > MAX_ABSOLUTE_UV_COORDINATE):
            raise UvIntegrityError(
                "The GLB contains UV coordinates outside the supported "
                "integrity range."
            )

        geometry_hasher = hashlib.sha256()
        geometry_face_count = 0
        for face in faces:
            corners = sorted(
                _quantize_uv_corner(normalized_uv[vertex_index])
                for vertex_index in face
            )
            geometry_hasher.update(
                struct.pack(
                    "<6q",
                    *(coordinate for corner in corners for coordinate in corner),
                )
            )
            face_count += 1
            geometry_face_count += 1
        if geometry_face_count:
            geometry_fingerprints.append(
                (geometry_face_count, geometry_hasher.digest())
            )

    if face_count == 0:
        raise UvIntegrityError(
            "UV integrity requires at least one triangle face."
        )
    hasher = hashlib.sha256()
    hasher.update(UV_FINGERPRINT_VERSION.encode("ascii"))
    hasher.update(b"\0")
    for geometry_face_count, geometry_digest in sorted(
        geometry_fingerprints
    ):
        hasher.update(struct.pack("<Q", geometry_face_count))
        hasher.update(geometry_digest)
    return UvFingerprint(
        version=UV_FINGERPRINT_VERSION,
        sha256=hasher.hexdigest(),
        face_count=face_count,
    )


# ### GLB helpers ###
def _load_glb_scene(glb_bytes: bytes) -> trimesh.Scene:
    payload = bytes(glb_bytes)
    if not payload:
        raise UvIntegrityError("UV integrity cannot inspect an empty GLB.")
    try:
        loaded = trimesh.load(
            BytesIO(payload),
            file_type="glb",
            force="scene",
            process=False,
        )
    except Exception as error:
        raise UvIntegrityError(
            "UV integrity could not load the GLB."
        ) from error
    if isinstance(loaded, trimesh.Trimesh):
        return trimesh.Scene(loaded)
    if isinstance(loaded, trimesh.Scene):
        return loaded
    raise UvIntegrityError("UV integrity found no mesh scene in the GLB.")


# ### Fingerprint helpers ###
def _quantize_uv_corner(uv: np.ndarray) -> tuple[int, int]:
    return (
        int(round(float(uv[0]) * UV_FINGERPRINT_QUANTIZATION_SCALE)),
        int(round(float(uv[1]) * UV_FINGERPRINT_QUANTIZATION_SCALE)),
    )
