# ### Imports ###
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import trimesh


# ### Integrity constants ###
CAMERA_UV_PROJECTION_VERSION = "camera-view-uv-v3-strict"
CAMERA_UV_FINGERPRINT_VERSION = "per-face-uv-triangle-sha256-v1"
UV_FINGERPRINT_QUANTIZATION_SCALE = 1_000_000
MAX_ABSOLUTE_UV_COORDINATE = 1_000_000.0


# ### Public data models ###
@dataclass(frozen=True)
class CameraUvFingerprint:
    """One deterministic per-face UV-layout identity."""

    version: str
    sha256: str
    face_count: int


# ### Public exceptions ###
class CameraUvIntegrityError(ValueError):
    """Raised when Meshy Retexture does not preserve submitted camera UVs."""


# ### Public integrity API ###
def build_camera_uv_fingerprint(glb_bytes: bytes) -> CameraUvFingerprint:
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
            raise CameraUvIntegrityError(
                "Camera UV integrity requires triangle mesh faces."
            )
        if len(faces) == 0:
            continue
        uv = getattr(geometry.visual, "uv", None)
        if uv is None:
            raise CameraUvIntegrityError(
                "A staged GLB is missing the projected UV coordinates needed "
                "for integrity validation."
            )
        normalized_uv = np.asarray(uv, dtype=float)
        if normalized_uv.ndim != 2 or normalized_uv.shape[1:] != (2,):
            raise CameraUvIntegrityError(
                "A staged GLB contains malformed projected UV coordinates."
            )
        if len(faces) and (
            np.any(faces < 0) or np.any(faces >= len(normalized_uv))
        ):
            raise CameraUvIntegrityError(
                "A staged GLB contains invalid UV vertex indices."
            )
        if not np.all(np.isfinite(normalized_uv)):
            raise CameraUvIntegrityError(
                "A staged GLB contains non-finite projected UV coordinates."
            )
        if np.any(np.abs(normalized_uv) > MAX_ABSOLUTE_UV_COORDINATE):
            raise CameraUvIntegrityError(
                "A staged GLB contains projected UV coordinates outside the "
                "supported integrity range."
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
        raise CameraUvIntegrityError(
            "Camera UV integrity requires at least one triangle face."
        )
    hasher = hashlib.sha256()
    hasher.update(CAMERA_UV_FINGERPRINT_VERSION.encode("ascii"))
    hasher.update(b"\0")
    for geometry_face_count, geometry_digest in sorted(
        geometry_fingerprints
    ):
        hasher.update(struct.pack("<Q", geometry_face_count))
        hasher.update(geometry_digest)
    return CameraUvFingerprint(
        version=CAMERA_UV_FINGERPRINT_VERSION,
        sha256=hasher.hexdigest(),
        face_count=face_count,
    )


def validate_camera_uv_retexture_integrity(
    submitted: CameraUvFingerprint,
    returned: CameraUvFingerprint,
) -> None:
    """Reject a Meshy result that changed submitted per-face camera UVs."""

    if submitted.version != returned.version:
        raise CameraUvIntegrityError(
            "Camera UV integrity versions do not match. No texture variants "
            "were saved; retry generation with the current HouseMaker build."
        )
    if submitted.face_count != returned.face_count:
        raise CameraUvIntegrityError(
            "Meshy Retexture changed the projected model's face count from "
            f"{submitted.face_count} to {returned.face_count}. No texture "
            "variants were saved; retry or disable 'Project UVs from camera "
            "views' for this object."
        )
    if submitted.sha256 != returned.sha256:
        raise CameraUvIntegrityError(
            "Meshy Retexture changed the camera-projected UV layout. No "
            "texture variants were saved; retry or disable 'Project UVs from "
            "camera views' for this object."
        )


# ### GLB helpers ###
def _load_glb_scene(glb_bytes: bytes) -> trimesh.Scene:
    payload = bytes(glb_bytes)
    if not payload:
        raise CameraUvIntegrityError(
            "Camera UV integrity cannot inspect an empty GLB."
        )
    try:
        loaded = trimesh.load(
            BytesIO(payload),
            file_type="glb",
            force="scene",
            process=False,
        )
    except Exception as error:
        raise CameraUvIntegrityError(
            "Camera UV integrity could not load the staged GLB."
        ) from error
    if isinstance(loaded, trimesh.Trimesh):
        return trimesh.Scene(loaded)
    if isinstance(loaded, trimesh.Scene):
        return loaded
    raise CameraUvIntegrityError(
        "Camera UV integrity found no mesh scene in the staged GLB."
    )


# ### Fingerprint helpers ###
def _quantize_uv_corner(uv: np.ndarray) -> tuple[int, int]:
    return (
        int(round(float(uv[0]) * UV_FINGERPRINT_QUANTIZATION_SCALE)),
        int(round(float(uv[1]) * UV_FINGERPRINT_QUANTIZATION_SCALE)),
    )
