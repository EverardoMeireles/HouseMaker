"""Persistent numbered export labels, independent of UUIDs and display names."""

# ### Imports ###
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

# ### Constants ###
OBJECT_EXPORT_ID_PATTERN = re.compile(r"Object ([1-9][0-9]{0,15})\Z")
OBJECT_EXPORT_NODE_NAME_PATTERN = re.compile(r"Object_([1-9][0-9]{0,15})\Z")
MAX_OBJECT_EXPORT_NUMBER = 9_007_199_254_740_991


# ### Validation and monotonic allocation ###
def _object_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("Object export mappings require string object IDs.")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 1_024
        or any(ord(char) < 32 for char in normalized)
    ):
        raise ValueError("Object export mappings contain an invalid object ID.")
    return normalized


def normalize_object_export_ids(payload: object) -> dict[str, str]:
    """Copy a unique UUID-to-Object-n mapping without guessing identities."""

    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError("Object export IDs must contain an object mapping.")
    normalized: dict[str, str] = {}
    labels: set[str] = set()
    for raw_id, label in payload.items():
        object_id = _object_id(raw_id)
        match = (
            OBJECT_EXPORT_ID_PATTERN.fullmatch(label)
            if isinstance(label, str)
            else None
        )
        if match is None or int(match.group(1)) > MAX_OBJECT_EXPORT_NUMBER:
            raise ValueError(
                "Object export IDs must use 'Object n' with a positive number."
            )
        if object_id in normalized or label in labels:
            raise ValueError(
                "Object export IDs and their numbered labels must be unique."
            )
        normalized[object_id] = label
        labels.add(label)
    return normalized


def allocate_object_export_ids(
    existing: Mapping[str, str] | None,
    object_ids: Iterable[str],
) -> dict[str, str]:
    """Retain old/deleted entries and append labels without recycling numbers."""

    normalized = normalize_object_export_ids(existing)
    if isinstance(object_ids, (str, bytes, bytearray)):
        raise TypeError("Allocate object export IDs from an iterable of object IDs.")
    next_number = 1 + max((int(label[7:]) for label in normalized.values()), default=0)
    for raw_id in object_ids:
        object_id = _object_id(raw_id)
        if object_id in normalized:
            continue
        if next_number > MAX_OBJECT_EXPORT_NUMBER:
            raise ValueError("The numbered object export ID range is exhausted.")
        normalized[object_id] = f"Object {next_number}"
        next_number += 1
    return normalized


# ### GLTFLoader-safe node names ###
def object_export_node_name(label: str) -> str:
    """Keep human labels separate from Three.js whitespace sanitization."""

    normalized = normalize_object_export_ids({"object": label})["object"]
    return normalized.replace(" ", "_", 1)


def object_export_label_from_node_name(node_name: str) -> str:
    match = (
        OBJECT_EXPORT_NODE_NAME_PATTERN.fullmatch(node_name)
        if isinstance(node_name, str)
        else None
    )
    if match is None:
        raise ValueError("Numbered object node names must use 'Object_n'.")
    return normalize_object_export_ids({"object": f"Object {match.group(1)}"})["object"]
