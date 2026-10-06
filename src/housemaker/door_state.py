# ### Imports ###
from __future__ import annotations

import copy
import math
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from housemaker.models import (
    DEFAULT_DOORWAY_ARCH_AMOUNT,
    DEFAULT_DOORWAY_SHAPE,
    DoorwayData,
    normalize_doorway_arch_amount,
    normalize_doorway_shape,
)

# ### Constants ###
DOOR_SLOT_BODY = "body"
DOOR_SLOT_HINGES = "hinges"
DOOR_SLOT_KNOB = "door_knob"
DOOR_SLOT_KINDS = (
    DOOR_SLOT_BODY,
    DOOR_SLOT_HINGES,
    DOOR_SLOT_KNOB,
)
DOOR_SLOT_DISPLAY_NAMES = {
    DOOR_SLOT_BODY: "Body",
    DOOR_SLOT_HINGES: "Hinges",
    DOOR_SLOT_KNOB: "Door knob",
}
DEFAULT_DOOR_BODY_THICKNESS_METERS = 0.04
MIN_DOOR_DIMENSION_METERS = 0.001
MAX_DOOR_DIMENSION_METERS = 20.0
MAX_DOOR_NAME_LENGTH = 256
MAX_DOOR_ID_LENGTH = 256
DOOR_NAME_PREFIX = "New door"
DOOR_SIDE_DUPLICATION_FRONT = "front"
DOOR_SIDE_DUPLICATION_BACK = "back"
DOOR_SIDE_DUPLICATION_SIDES = frozenset(
    {DOOR_SIDE_DUPLICATION_FRONT, DOOR_SIDE_DUPLICATION_BACK}
)
DOOR_SIDE_DUPLICATION_METADATA_VERSION = 1
DOOR_SIDE_DUPLICATION_UV_MODE = "reuse"
DOOR_LIBRARY_SCHEMA_VERSION = 2
_DOOR_NAME_PATTERN = re.compile(r"^New door\s+([1-9]\d*)$", re.IGNORECASE)


# ### Type aliases ###
Vector3 = tuple[float, float, float]


# ### Door slot models ###
@dataclass(frozen=True, slots=True)
class DoorSlotData:
    """One editable component slot in a reusable door definition.

    Positions use the door body's local, Z-up coordinates. The X axis spans
    the door width, Y spans its thickness, and Z points upward. Component
    positions refer to the component model's bottom-center anchor.
    """

    slot_id: str
    source_object_id: str
    position_meters: Vector3 = (0.0, 0.0, 0.0)
    rotation_degrees: Vector3 = (0.0, 0.0, 0.0)
    joined: bool = False

    def __post_init__(self) -> None:
        slot_id = str(self.slot_id).strip().lower()
        if slot_id not in DOOR_SLOT_KINDS:
            raise ValueError(f"Unknown door slot: {self.slot_id!r}.")
        source_object_id = _normalize_identifier(
            self.source_object_id,
            "Door slot source object",
        )
        if not isinstance(self.joined, bool):
            raise TypeError("Door slot joined state must be boolean.")
        if slot_id == DOOR_SLOT_BODY and not self.joined:
            raise ValueError("The door body slot must always be joined.")

        object.__setattr__(self, "slot_id", slot_id)
        object.__setattr__(self, "source_object_id", source_object_id)
        object.__setattr__(
            self,
            "position_meters",
            _normalize_vector3(self.position_meters, "Door slot position"),
        )
        object.__setattr__(
            self,
            "rotation_degrees",
            _normalize_vector3(self.rotation_degrees, "Door slot rotation"),
        )

    @property
    def display_name(self) -> str:
        return DOOR_SLOT_DISPLAY_NAMES[self.slot_id]

    def with_transform(
        self,
        *,
        position_meters: Sequence[float] | None = None,
        rotation_degrees: Sequence[float] | None = None,
        joined: bool | None = None,
    ) -> DoorSlotData:
        """Return this slot with a validated edited transform."""

        return replace(
            self,
            position_meters=(
                self.position_meters
                if position_meters is None
                else tuple(position_meters)
            ),
            rotation_degrees=(
                self.rotation_degrees
                if rotation_degrees is None
                else tuple(rotation_degrees)
            ),
            joined=self.joined if joined is None else joined,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "source_object_id": self.source_object_id,
            "position_meters": list(self.position_meters),
            "rotation_degrees": list(self.rotation_degrees),
            "joined": self.joined,
        }

    @classmethod
    def from_dict(cls, payload: object) -> DoorSlotData:
        raw = _require_mapping(payload, "Door slot")
        return cls(
            slot_id=str(raw["slot_id"]),
            source_object_id=str(raw["source_object_id"]),
            position_meters=raw.get("position_meters", (0.0, 0.0, 0.0)),
            rotation_degrees=raw.get("rotation_degrees", (0.0, 0.0, 0.0)),
            joined=raw.get("joined", False),
        )


# ### Door definition models ###
@dataclass(frozen=True, slots=True)
class DoorSideDuplication:
    """Reconstruct a door body's omitted depth half at runtime."""

    kept_side: str = DOOR_SIDE_DUPLICATION_FRONT

    def __post_init__(self) -> None:
        kept_side = str(self.kept_side).strip().lower()
        if kept_side not in DOOR_SIDE_DUPLICATION_SIDES:
            raise ValueError("Door side duplication must keep Front or Back.")
        object.__setattr__(self, "kept_side", kept_side)

    def to_dict(self) -> dict[str, object]:
        return {"kept_side": self.kept_side}

    @classmethod
    def from_dict(cls, payload: object) -> DoorSideDuplication:
        raw = _require_mapping(payload, "Door side duplication")
        return cls(kept_side=str(raw["kept_side"]))


@dataclass(frozen=True, slots=True)
class DoorSideDuplicationMetadata:
    """Persisted authored-half provenance produced for a door body."""

    kept_side: str
    plane_coordinate: float
    version: int = DOOR_SIDE_DUPLICATION_METADATA_VERSION
    uv_mode: str = DOOR_SIDE_DUPLICATION_UV_MODE

    def __post_init__(self) -> None:
        kept_side = DoorSideDuplication(self.kept_side).kept_side
        plane_coordinate = float(self.plane_coordinate)
        if not math.isfinite(plane_coordinate):
            raise ValueError("Door side-duplication plane must be finite.")
        if self.version != DOOR_SIDE_DUPLICATION_METADATA_VERSION:
            raise ValueError("Unsupported door side-duplication metadata version.")
        if self.uv_mode != DOOR_SIDE_DUPLICATION_UV_MODE:
            raise ValueError("Door side duplication requires UV reuse.")
        object.__setattr__(self, "kept_side", kept_side)
        object.__setattr__(self, "plane_coordinate", plane_coordinate)

    def to_pipeline_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "kept_side": self.kept_side,
            "plane_coordinate": self.plane_coordinate,
            "uv_mode": self.uv_mode,
        }

    @classmethod
    def from_pipeline_dict(
        cls,
        payload: object,
    ) -> DoorSideDuplicationMetadata:
        raw = _require_mapping(payload, "Door side-duplication metadata")
        return cls(
            version=raw.get("version", 0),
            kept_side=str(raw.get("kept_side", "")),
            plane_coordinate=raw.get("plane_coordinate"),
            uv_mode=str(raw.get("uv_mode", "")),
        )


@dataclass(frozen=True, slots=True)
class DoorDefinition:
    """One reusable door whose canonical body fits a doorway profile."""

    door_id: str
    name: str
    source_doorway_id: str
    width_meters: float
    height_meters: float
    thickness_meters: float = DEFAULT_DOOR_BODY_THICKNESS_METERS
    shape: str = DEFAULT_DOORWAY_SHAPE
    arch_amount: float = DEFAULT_DOORWAY_ARCH_AMOUNT
    side_duplication: DoorSideDuplication | None = None
    slots: tuple[DoorSlotData, ...] = ()

    def __post_init__(self) -> None:
        door_id = _normalize_identifier(self.door_id, "Door")
        source_doorway_id = _normalize_identifier(
            self.source_doorway_id,
            "Source doorway",
        )
        name = str(self.name).strip()
        if not name:
            raise ValueError("Door names cannot be empty.")
        if len(name) > MAX_DOOR_NAME_LENGTH:
            raise ValueError("Door name is too long.")
        slots = tuple(self.slots)
        if any(not isinstance(slot, DoorSlotData) for slot in slots):
            raise TypeError("Door definitions require DoorSlotData values.")
        slot_ids = tuple(slot.slot_id for slot in slots)
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("A door cannot contain duplicate component slots.")
        if DOOR_SLOT_BODY not in slot_ids:
            raise ValueError("A door definition requires a body slot.")
        if self.side_duplication is not None and not isinstance(
            self.side_duplication,
            DoorSideDuplication,
        ):
            raise TypeError(
                "Door side duplication must be DoorSideDuplication or None."
            )

        object.__setattr__(self, "door_id", door_id)
        object.__setattr__(self, "source_doorway_id", source_doorway_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "width_meters",
            _normalize_dimension(self.width_meters, "Door width"),
        )
        object.__setattr__(
            self,
            "height_meters",
            _normalize_dimension(self.height_meters, "Door height"),
        )
        object.__setattr__(
            self,
            "thickness_meters",
            _normalize_dimension(self.thickness_meters, "Door thickness"),
        )
        object.__setattr__(self, "shape", normalize_doorway_shape(self.shape))
        object.__setattr__(
            self,
            "arch_amount",
            normalize_doorway_arch_amount(self.arch_amount),
        )
        object.__setattr__(self, "slots", slots)

    def get_slot(self, slot_id: str) -> DoorSlotData | None:
        normalized_id = str(slot_id).strip().lower()
        return next(
            (slot for slot in self.slots if slot.slot_id == normalized_id),
            None,
        )

    def replace_slot(self, replacement: DoorSlotData) -> DoorDefinition:
        """Return the definition with one existing slot replaced."""

        if not isinstance(replacement, DoorSlotData):
            raise TypeError("Door slots can only be replaced by DoorSlotData.")
        if self.get_slot(replacement.slot_id) is None:
            raise ValueError(f"Door slot {replacement.slot_id!r} does not exist.")
        return replace(
            self,
            slots=tuple(
                replacement if slot.slot_id == replacement.slot_id else slot
                for slot in self.slots
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "door_id": self.door_id,
            "name": self.name,
            "source_doorway_id": self.source_doorway_id,
            "width_meters": self.width_meters,
            "height_meters": self.height_meters,
            "thickness_meters": self.thickness_meters,
            "shape": self.shape,
            "arch_amount": self.arch_amount,
            "side_duplication": (
                None
                if self.side_duplication is None
                else self.side_duplication.to_dict()
            ),
            "slots": [slot.to_dict() for slot in self.slots],
        }

    @classmethod
    def from_dict(cls, payload: object) -> DoorDefinition:
        raw = _require_mapping(payload, "Door definition")
        raw_slots = raw.get("slots", ())
        if isinstance(raw_slots, (str, bytes, bytearray)) or not isinstance(
            raw_slots,
            Sequence,
        ):
            raise TypeError("Door slots must contain a list.")
        return cls(
            door_id=str(raw["door_id"]),
            name=str(raw["name"]),
            source_doorway_id=str(raw["source_doorway_id"]),
            width_meters=raw["width_meters"],
            height_meters=raw["height_meters"],
            thickness_meters=raw.get(
                "thickness_meters",
                DEFAULT_DOOR_BODY_THICKNESS_METERS,
            ),
            shape=raw.get("shape", DEFAULT_DOORWAY_SHAPE),
            arch_amount=raw.get(
                "arch_amount",
                DEFAULT_DOORWAY_ARCH_AMOUNT,
            ),
            side_duplication=(
                None
                if raw.get("side_duplication") is None
                else DoorSideDuplication.from_dict(raw["side_duplication"])
            ),
            slots=tuple(DoorSlotData.from_dict(slot) for slot in raw_slots),
        )


# ### Door placement models ###
@dataclass(frozen=True, slots=True)
class DoorPlacement:
    """One door attached to a stable doorway identity on a level."""

    placement_id: str
    door_id: str
    level_index: int
    doorway_id: str
    mirrored_horizontally: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.level_index, bool)
            or not isinstance(self.level_index, int)
            or self.level_index < 0
        ):
            raise ValueError("Door placement level index must be non-negative.")
        if not isinstance(self.mirrored_horizontally, bool):
            raise TypeError("Door placement mirror state must be boolean.")
        object.__setattr__(
            self,
            "placement_id",
            _normalize_identifier(self.placement_id, "Door placement"),
        )
        object.__setattr__(
            self,
            "door_id",
            _normalize_identifier(self.door_id, "Door"),
        )
        object.__setattr__(
            self,
            "doorway_id",
            _normalize_identifier(self.doorway_id, "Doorway"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "placement_id": self.placement_id,
            "door_id": self.door_id,
            "level_index": self.level_index,
            "doorway_id": self.doorway_id,
            "mirrored_horizontally": self.mirrored_horizontally,
        }

    @classmethod
    def from_dict(cls, payload: object) -> DoorPlacement:
        raw = _require_mapping(payload, "Door placement")
        return cls(
            placement_id=str(raw["placement_id"]),
            door_id=str(raw["door_id"]),
            level_index=raw["level_index"],
            doorway_id=str(raw["doorway_id"]),
            mirrored_horizontally=raw.get("mirrored_horizontally", False),
        )


# ### Door library model ###
@dataclass
class DoorLibraryData:
    """Persistent reusable doors and their doorway-bound placements."""

    doors: list[DoorDefinition] = field(default_factory=list)
    placements: list[DoorPlacement] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.doors = list(self.doors)
        self.placements = list(self.placements)
        if any(not isinstance(door, DoorDefinition) for door in self.doors):
            raise TypeError("Door libraries require DoorDefinition values.")
        if any(
            not isinstance(placement, DoorPlacement)
            for placement in self.placements
        ):
            raise TypeError("Door libraries require DoorPlacement values.")
        _require_unique(
            (door.door_id for door in self.doors),
            "Door IDs",
        )
        _require_unique(
            (placement.placement_id for placement in self.placements),
            "Door placement IDs",
        )
        _require_unique(
            (
                f"{placement.level_index}:{placement.doorway_id}"
                for placement in self.placements
            ),
            "Doorway placement bindings",
        )
        door_ids = {door.door_id for door in self.doors}
        if any(
            placement.door_id not in door_ids for placement in self.placements
        ):
            raise ValueError("Every door placement must reference a known door.")

    def clone(self) -> DoorLibraryData:
        return copy.deepcopy(self)

    def get_door(self, door_id: str) -> DoorDefinition | None:
        normalized_id = str(door_id).strip()
        return next(
            (door for door in self.doors if door.door_id == normalized_id),
            None,
        )

    def replace_door(self, replacement: DoorDefinition) -> None:
        if not isinstance(replacement, DoorDefinition):
            raise TypeError("Doors can only be replaced by DoorDefinition values.")
        for index, door in enumerate(self.doors):
            if door.door_id == replacement.door_id:
                self.doors[index] = replacement
                return
        raise ValueError(f"Door {replacement.door_id!r} does not exist.")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": DOOR_LIBRARY_SCHEMA_VERSION,
            "doors": [door.to_dict() for door in self.doors],
            "placements": [placement.to_dict() for placement in self.placements],
        }

    @classmethod
    def from_dict(cls, payload: object) -> DoorLibraryData:
        if payload is None:
            return cls()
        raw = _require_mapping(payload, "Door library")
        raw_doors = _require_sequence(raw.get("doors", ()), "Door definitions")
        raw_placements = _require_sequence(
            raw.get("placements", ()),
            "Door placements",
        )

        doors: list[DoorDefinition] = []
        occupied_door_ids: set[str] = set()
        for raw_door in raw_doors:
            try:
                door = DoorDefinition.from_dict(raw_door)
            except (KeyError, TypeError, ValueError):
                continue
            if door.door_id in occupied_door_ids:
                continue
            doors.append(door)
            occupied_door_ids.add(door.door_id)

        placements: list[DoorPlacement] = []
        occupied_placement_ids: set[str] = set()
        occupied_doorway_bindings: set[tuple[int, str]] = set()
        for raw_placement in raw_placements:
            try:
                placement = DoorPlacement.from_dict(raw_placement)
            except (KeyError, TypeError, ValueError):
                continue
            doorway_binding = (placement.level_index, placement.doorway_id)
            if (
                placement.placement_id in occupied_placement_ids
                or placement.door_id not in occupied_door_ids
                or doorway_binding in occupied_doorway_bindings
            ):
                continue
            placements.append(placement)
            occupied_placement_ids.add(placement.placement_id)
            occupied_doorway_bindings.add(doorway_binding)
        return cls(doors=doors, placements=placements)


# ### Door construction helpers ###
def create_door_definition_for_doorway(
    doorway: DoorwayData,
    *,
    name: str,
    door_id: str | None = None,
    thickness_meters: float | None = None,
) -> DoorDefinition:
    """Create canonical body, hinge, and knob slots for one doorway."""

    if not isinstance(doorway, DoorwayData):
        raise TypeError("Doors can only be created from DoorwayData.")
    normalized_door_id = uuid.uuid4().hex if door_id is None else door_id
    thickness = (
        min(DEFAULT_DOOR_BODY_THICKNESS_METERS, doorway.depth_meters)
        if thickness_meters is None
        else thickness_meters
    )
    hinge_radius = min(0.012, doorway.width_meters * 0.025)
    knob_radius = min(0.035, doorway.width_meters * 0.075)
    slots = (
        DoorSlotData(
            slot_id=DOOR_SLOT_BODY,
            source_object_id=_slot_source_object_id(
                normalized_door_id,
                DOOR_SLOT_BODY,
            ),
            joined=True,
        ),
        DoorSlotData(
            slot_id=DOOR_SLOT_HINGES,
            source_object_id=_slot_source_object_id(
                normalized_door_id,
                DOOR_SLOT_HINGES,
            ),
            position_meters=(
                -doorway.width_meters / 2.0 + hinge_radius,
                0.0,
                doorway.height_meters * 0.18,
            ),
        ),
        DoorSlotData(
            slot_id=DOOR_SLOT_KNOB,
            source_object_id=_slot_source_object_id(
                normalized_door_id,
                DOOR_SLOT_KNOB,
            ),
            position_meters=(
                doorway.width_meters / 2.0 - max(0.08, knob_radius * 2.0),
                0.0,
                doorway.height_meters * 0.45 - knob_radius,
            ),
        ),
    )
    return DoorDefinition(
        door_id=normalized_door_id,
        name=name,
        source_doorway_id=doorway.doorway_id,
        width_meters=doorway.width_meters,
        height_meters=doorway.height_meters,
        thickness_meters=thickness,
        shape=doorway.shape,
        arch_amount=doorway.arch_amount,
        slots=slots,
    )


def next_door_name(doors: Iterable[DoorDefinition]) -> str:
    """Return the next monotonic ``New door n`` library name."""

    largest_index = 0
    for door in doors:
        if not isinstance(door, DoorDefinition):
            continue
        match = _DOOR_NAME_PATTERN.fullmatch(door.name.strip())
        if match is not None:
            largest_index = max(largest_index, int(match.group(1)))
    return f"{DOOR_NAME_PREFIX} {largest_index + 1}"


def door_fits_doorway(
    door: DoorDefinition,
    doorway: DoorwayData,
    *,
    tolerance_meters: float = 0.01,
) -> bool:
    """Return whether a reusable door matches a destination doorway profile."""

    tolerance = _normalize_non_negative_float(
        tolerance_meters,
        "Door fit tolerance",
    )
    return (
        abs(door.width_meters - doorway.width_meters) <= tolerance
        and abs(door.height_meters - doorway.height_meters) <= tolerance
        and door.shape == doorway.shape
        and (
            door.shape == DEFAULT_DOORWAY_SHAPE
            or math.isclose(
                door.arch_amount,
                doorway.arch_amount,
                abs_tol=1e-6,
            )
        )
    )


# ### Validation helpers ###
def _normalize_identifier(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} ID must be a string.")
    identifier = value.strip()
    if not identifier:
        raise ValueError(f"{label} ID cannot be empty.")
    if len(identifier) > MAX_DOOR_ID_LENGTH:
        raise ValueError(f"{label} ID is too long.")
    return identifier


def _normalize_dimension(value: object, label: str) -> float:
    dimension = _normalize_finite_float(value, label)
    if not MIN_DOOR_DIMENSION_METERS <= dimension <= MAX_DOOR_DIMENSION_METERS:
        raise ValueError(
            f"{label} must be between {MIN_DOOR_DIMENSION_METERS:g} and "
            f"{MAX_DOOR_DIMENSION_METERS:g} meters."
        )
    return dimension


def _normalize_non_negative_float(value: object, label: str) -> float:
    number = _normalize_finite_float(value, label)
    if number < 0.0:
        raise ValueError(f"{label} cannot be negative.")
    return number


def _normalize_finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{label} must be a number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _normalize_vector3(value: object, label: str) -> Vector3:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        Sequence,
    ):
        raise TypeError(f"{label} must contain three coordinates.")
    if len(value) != 3:
        raise ValueError(f"{label} must contain three coordinates.")
    coordinates = tuple(
        _normalize_finite_float(coordinate, label) for coordinate in value
    )
    return coordinates[0], coordinates[1], coordinates[2]


def _require_mapping(payload: object, label: str) -> Mapping[object, object]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} JSON must contain an object.")
    return payload


def _require_sequence(payload: object, label: str) -> Sequence[object]:
    if isinstance(payload, (str, bytes, bytearray)) or not isinstance(
        payload,
        Sequence,
    ):
        raise TypeError(f"{label} must contain a list.")
    return payload


def _require_unique(values: Iterable[str], label: str) -> None:
    normalized = tuple(values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be unique.")


def _slot_source_object_id(door_id: str, slot_id: str) -> str:
    return f"door:{door_id}:slot:{slot_id}"
