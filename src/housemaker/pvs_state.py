"""Immutable, validated baked visibility in final physical glTF coordinates."""

# ### Imports ###
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from housemaker.first_person_navigation import (
    DEFAULT_FIRST_PERSON_EYE_HEIGHT_METERS,
    normalize_first_person_eye_height_meters,
)

# ### Constants ###
PVS_STATE_VERSION = 1
PVS_ALGORITHM_VERSION = 1
PVS_COORDINATE_SYSTEM = "gltf-y-up"
MAX_PVS_CELLS = 250_000
MAX_PVS_CLUSTERS = 100_000
MAX_PVS_VISIBLE_REFERENCES = 10_000_000
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


# ### Validation helpers ###
def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"PVS {label} must be a string.")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 1_024
        or any(ord(char) < 32 for char in normalized)
    ):
        raise ValueError(f"PVS {label} is invalid.")
    return normalized


def _vector(value: object) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("PVS bounds require three finite coordinates.")
    if any(
        isinstance(part, bool) or not isinstance(part, (int, float)) for part in value
    ):
        raise TypeError("PVS bounds require numeric coordinates.")
    result = tuple(float(part) for part in value)
    if not all(math.isfinite(part) for part in result):
        raise ValueError("PVS bounds require finite coordinates.")
    return result


def _bounds(minimum: object, maximum: object, *, volume: bool = False):
    lower, upper = _vector(minimum), _vector(maximum)
    if any(low > high for low, high in zip(lower, upper)):
        raise ValueError("PVS minimum bounds cannot exceed maximum bounds.")
    if volume and any(low == high for low, high in zip(lower, upper)):
        raise ValueError("PVS camera cells must have positive volume.")
    return lower, upper


def _identifiers(values: object, label: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"PVS {label} must be a list.")
    result = tuple(_identifier(value, label) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"PVS {label} cannot contain duplicates.")
    return result


def _mapping(value: object, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"PVS {label} must be an object.")
    return value


def _version(value: object, expected: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"Unsupported PVS {label}: {value!r}.")
    return value


def _signature(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"PVS {label} must be a SHA256 fingerprint.")
    return value.lower()


# ### Render clusters and camera cells ###
@dataclass(frozen=True, slots=True)
class PvsCluster:
    cluster_id: str
    node_names: tuple[str, ...]
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    instance_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cluster_id", _identifier(self.cluster_id, "cluster ID")
        )
        names = _identifiers(self.node_names, "node names")
        instance_ids = _identifiers(self.instance_ids, "instance IDs")
        if not names and not instance_ids:
            raise ValueError("A PVS render cluster requires a node or instance ID.")
        object.__setattr__(self, "node_names", names)
        object.__setattr__(self, "instance_ids", instance_ids)
        lower, upper = _bounds(self.minimum, self.maximum)
        object.__setattr__(self, "minimum", lower)
        object.__setattr__(self, "maximum", upper)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.cluster_id,
            "nodeNames": list(self.node_names),
            "instanceIds": list(self.instance_ids),
            "min": list(self.minimum),
            "max": list(self.maximum),
        }

    @classmethod
    def from_dict(cls, value: object) -> PvsCluster:
        data = _mapping(value, "cluster")
        return cls(
            data.get("id"),
            data.get("nodeNames"),
            data.get("min"),
            data.get("max"),
            data.get("instanceIds", ()),
        )


@dataclass(frozen=True, slots=True)
class PvsCell:
    cell_id: str
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    visible_cluster_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_id", _identifier(self.cell_id, "cell ID"))
        lower, upper = _bounds(self.minimum, self.maximum, volume=True)
        object.__setattr__(self, "minimum", lower)
        object.__setattr__(self, "maximum", upper)
        object.__setattr__(
            self,
            "visible_cluster_ids",
            _identifiers(self.visible_cluster_ids, "visible cluster IDs"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.cell_id,
            "min": list(self.minimum),
            "max": list(self.maximum),
            "visibleClusterIds": list(self.visible_cluster_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> PvsCell:
        data = _mapping(value, "cell")
        return cls(
            data.get("id"),
            data.get("min"),
            data.get("max"),
            data.get("visibleClusterIds"),
        )


# ### Persisted bake and runtime metadata ###
@dataclass(frozen=True, slots=True)
class PvsBakeData:
    source_signature: str
    clusters: tuple[PvsCluster, ...]
    cells: tuple[PvsCell, ...]
    eye_height_meters: float = DEFAULT_FIRST_PERSON_EYE_HEIGHT_METERS
    algorithm_version: int = PVS_ALGORITHM_VERSION
    coordinate_system: str = PVS_COORDINATE_SYSTEM
    authoring_signature: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_signature",
            _signature(self.source_signature, "source signature"),
        )
        if self.authoring_signature is not None:
            object.__setattr__(
                self,
                "authoring_signature",
                _signature(self.authoring_signature, "authoring signature"),
            )
        object.__setattr__(
            self,
            "eye_height_meters",
            normalize_first_person_eye_height_meters(self.eye_height_meters),
        )
        _version(self.algorithm_version, PVS_ALGORITHM_VERSION, "algorithm version")
        if self.coordinate_system != PVS_COORDINATE_SYSTEM:
            raise ValueError("PVS data must use final physical glTF Y-up coordinates.")
        for name, values, expected, limit in (
            ("clusters", self.clusters, PvsCluster, MAX_PVS_CLUSTERS),
            ("cells", self.cells, PvsCell, MAX_PVS_CELLS),
        ):
            if (
                not isinstance(values, (tuple, list))
                or not values
                or len(values) > limit
            ):
                raise ValueError(
                    f"PVS {name} must contain between 1 and {limit} entries."
                )
            if not all(isinstance(value, expected) for value in values):
                raise TypeError(f"PVS {name} contain invalid records.")
            object.__setattr__(self, name, tuple(values))
        cluster_ids = {item.cluster_id for item in self.clusters}
        if len(cluster_ids) != len(self.clusters) or len(
            {item.cell_id for item in self.cells}
        ) != len(self.cells):
            raise ValueError("PVS cell and cluster IDs must be unique.")
        node_names = tuple(
            name for cluster in self.clusters for name in cluster.node_names
        )
        if len(set(node_names)) != len(node_names):
            raise ValueError("A render node cannot belong to multiple PVS clusters.")
        instance_ids = tuple(
            item for cluster in self.clusters for item in cluster.instance_ids
        )
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError("An instance cannot belong to multiple PVS clusters.")
        if (
            sum(len(cell.visible_cluster_ids) for cell in self.cells)
            > MAX_PVS_VISIBLE_REFERENCES
        ):
            raise ValueError("PVS contains too many visibility references.")
        if any(set(cell.visible_cluster_ids) - cluster_ids for cell in self.cells):
            raise ValueError("PVS camera cells reference unknown render clusters.")

    @property
    def minimum(self) -> tuple[float, float, float]:
        return tuple(
            min(item.minimum[axis] for item in (*self.clusters, *self.cells))
            for axis in range(3)
        )

    @property
    def maximum(self) -> tuple[float, float, float]:
        return tuple(
            max(item.maximum[axis] for item in (*self.clusters, *self.cells))
            for axis in range(3)
        )

    def matches_source(
        self, signature: str, *, eye_height_meters: float | None = None
    ) -> bool:
        return self.source_signature == signature and (
            eye_height_meters is None
            or self.eye_height_meters
            == normalize_first_person_eye_height_meters(eye_height_meters)
        )

    def to_dict(self) -> dict[str, object]:
        data = {
            "version": PVS_STATE_VERSION,
            "algorithmVersion": self.algorithm_version,
            "coordinateSystem": self.coordinate_system,
            "sourceSignature": self.source_signature,
            "eyeHeightMeters": self.eye_height_meters,
            "worldBounds": {"min": list(self.minimum), "max": list(self.maximum)},
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "cells": [cell.to_dict() for cell in self.cells],
            # Never hide unlisted/dynamic nodes or cameras outside baked cells.
            "fallback": "frustum-only",
            "unlistedNodes": "frustum-only",
        }
        if self.authoring_signature is not None:
            data["authoringSignature"] = self.authoring_signature
        return data

    def to_runtime_dict(self, *, glb_sha256: str) -> dict[str, object]:
        if (
            not isinstance(glb_sha256, str)
            or SHA256_PATTERN.fullmatch(glb_sha256) is None
        ):
            raise ValueError("PVS runtime metadata requires the exported GLB SHA256.")
        return {**self.to_dict(), "glbSha256": glb_sha256.lower()}

    @classmethod
    def from_dict(cls, value: object) -> PvsBakeData:
        data = _mapping(value, "bake")
        _version(data.get("version"), PVS_STATE_VERSION, "state version")
        clusters, cells = data.get("clusters"), data.get("cells")
        if not isinstance(clusters, (list, tuple)) or len(clusters) > MAX_PVS_CLUSTERS:
            raise ValueError("Invalid PVS cluster collection.")
        if not isinstance(cells, (list, tuple)) or len(cells) > MAX_PVS_CELLS:
            raise ValueError("Invalid PVS camera cell collection.")
        return cls(
            source_signature=data.get("sourceSignature"),
            clusters=tuple(PvsCluster.from_dict(item) for item in clusters),
            cells=tuple(PvsCell.from_dict(item) for item in cells),
            eye_height_meters=data.get("eyeHeightMeters"),
            algorithm_version=data.get("algorithmVersion"),
            coordinate_system=data.get("coordinateSystem"),
            authoring_signature=data.get("authoringSignature"),
        )
