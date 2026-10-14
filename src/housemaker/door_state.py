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
    DOORWAY_SHAPE_ARCH,
    DoorwayData,
    normalize_doorway_arch_amount,
    normalize_doorway_shape,
)

# ### Constants ###
DOOR_SLOT_BODY = "body"
DOOR_SLOT_KNOB = "door_knob"
DOOR_SLOT_LEFT_BODY = "left_body"
DOOR_SLOT_LEFT_KNOB = "left_door_knob"
DOOR_SLOT_BACK_BODY = "back_body"
DOOR_SLOT_BACK_KNOB = "back_door_knob"
DOOR_SLOT_BACK_LEFT_BODY = "back_left_body"
DOOR_SLOT_BACK_LEFT_KNOB = "back_left_door_knob"
DOOR_SLOT_KINDS = (
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    DOOR_SLOT_LEFT_BODY,
    DOOR_SLOT_LEFT_KNOB,
    DOOR_SLOT_BACK_BODY,
    DOOR_SLOT_BACK_KNOB,
    DOOR_SLOT_BACK_LEFT_BODY,
    DOOR_SLOT_BACK_LEFT_KNOB,
)
DOOR_BODY_SLOT_IDS = frozenset(
    {
        DOOR_SLOT_BODY,
        DOOR_SLOT_LEFT_BODY,
        DOOR_SLOT_BACK_BODY,
        DOOR_SLOT_BACK_LEFT_BODY,
    }
)
DOOR_KNOB_SLOT_IDS = frozenset(
    {
        DOOR_SLOT_KNOB,
        DOOR_SLOT_LEFT_KNOB,
        DOOR_SLOT_BACK_KNOB,
        DOOR_SLOT_BACK_LEFT_KNOB,
    }
)
DOOR_SLOT_DISPLAY_NAMES = {
    DOOR_SLOT_BODY: "Body",
    DOOR_SLOT_KNOB: "Handle",
    DOOR_SLOT_LEFT_BODY: "Body left",
    DOOR_SLOT_LEFT_KNOB: "Handle left",
    DOOR_SLOT_BACK_BODY: "Body",
    DOOR_SLOT_BACK_KNOB: "Handle",
    DOOR_SLOT_BACK_LEFT_BODY: "Body left",
    DOOR_SLOT_BACK_LEFT_KNOB: "Handle left",
}
DOOR_LEAF_RIGHT = "right"
DOOR_LEAF_LEFT = "left"
DEFAULT_DOOR_BODY_THICKNESS_METERS = 0.04
MIN_DOOR_DIMENSION_METERS = 0.001
MAX_DOOR_DIMENSION_METERS = 20.0
MAX_DOOR_NAME_LENGTH = 256
MAX_DOOR_ID_LENGTH = 256
DOOR_NAME_PREFIX = "New door"
WINDOW_NAME_PREFIX = "New window"
GLASS_AREA_NAME_PREFIX = "Glass area"
MAX_GLASS_AREA_NAME_LENGTH = 256
MIN_GLASS_AREA_NORMALIZED_SPAN = 1e-6
DOOR_SIDE_DUPLICATION_FRONT = "front"
DOOR_SIDE_DUPLICATION_BACK = "back"
DOOR_SIDE_DUPLICATION_SIDES = frozenset(
    {DOOR_SIDE_DUPLICATION_FRONT, DOOR_SIDE_DUPLICATION_BACK}
)
DOOR_SIDE_DUPLICATION_METADATA_VERSION = 1
DOOR_SIDE_DUPLICATION_UV_MODE = "reuse"
DOOR_LIBRARY_SCHEMA_VERSION = 5
_DOOR_NAME_PATTERN = re.compile(r"^New door\s+([1-9]\d*)$", re.IGNORECASE)
_WINDOW_NAME_PATTERN = re.compile(r"^New window\s+([1-9]\d*)$", re.IGNORECASE)
_GLASS_AREA_NAME_PATTERN = re.compile(
    r"^Glass area\s+([1-9]\d*)$",
    re.IGNORECASE,
)
_LEGACY_DOOR_SLOT_HINGES = "hinges"


# ### Type aliases ###
Vector3 = tuple[float, float, float]


# ### Glass-area models ###
@dataclass(frozen=True, slots=True)
class GlassAreaData:
    """One stable through-body glass rectangle in normalized local X/Z space."""

    glass_area_id: str
    source_object_id: str
    name: str
    minimum_x_ratio: float
    maximum_x_ratio: float
    minimum_z_ratio: float
    maximum_z_ratio: float
    body_slot_id: str = DOOR_SLOT_BODY

    def __post_init__(self) -> None:
        glass_area_id = _normalize_identifier(self.glass_area_id, "Glass area")
        source_object_id = _normalize_identifier(
            self.source_object_id,
            "Glass area source object",
        )
        body_slot_id = str(self.body_slot_id).strip().lower()
        if not is_door_body_slot(body_slot_id):
            raise ValueError("Glass areas must reference a body slot.")
        name = str(self.name).strip()
        if not name:
            raise ValueError("Glass area names cannot be empty.")
        if len(name) > MAX_GLASS_AREA_NAME_LENGTH:
            raise ValueError("Glass area name is too long.")
        minimum_x, maximum_x = _normalize_ratio_bounds(
            self.minimum_x_ratio,
            self.maximum_x_ratio,
            "Glass area X",
        )
        minimum_z, maximum_z = _normalize_ratio_bounds(
            self.minimum_z_ratio,
            self.maximum_z_ratio,
            "Glass area Z",
        )
        object.__setattr__(self, "glass_area_id", glass_area_id)
        object.__setattr__(self, "source_object_id", source_object_id)
        object.__setattr__(self, "body_slot_id", body_slot_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "minimum_x_ratio", minimum_x)
        object.__setattr__(self, "maximum_x_ratio", maximum_x)
        object.__setattr__(self, "minimum_z_ratio", minimum_z)
        object.__setattr__(self, "maximum_z_ratio", maximum_z)

    def to_dict(self) -> dict[str, object]:
        return {
            "glass_area_id": self.glass_area_id,
            "source_object_id": self.source_object_id,
            "name": self.name,
            "minimum_x_ratio": self.minimum_x_ratio,
            "maximum_x_ratio": self.maximum_x_ratio,
            "minimum_z_ratio": self.minimum_z_ratio,
            "maximum_z_ratio": self.maximum_z_ratio,
            "body_slot_id": self.body_slot_id,
        }

    @classmethod
    def from_dict(cls, payload: object) -> GlassAreaData:
        raw = _require_mapping(payload, "Glass area")
        return cls(
            glass_area_id=str(raw["glass_area_id"]),
            source_object_id=str(raw["source_object_id"]),
            name=str(raw["name"]),
            minimum_x_ratio=raw["minimum_x_ratio"],
            maximum_x_ratio=raw["maximum_x_ratio"],
            minimum_z_ratio=raw["minimum_z_ratio"],
            maximum_z_ratio=raw["maximum_z_ratio"],
            body_slot_id=str(raw.get("body_slot_id", DOOR_SLOT_BODY)),
        )


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
    axis_scales: Vector3 = (1.0, 1.0, 1.0)
    joined: bool = False
    mirrored_horizontally: bool = False

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
        if not isinstance(self.mirrored_horizontally, bool):
            raise TypeError("Door slot horizontal mirror state must be boolean.")
        if is_door_body_slot(slot_id) and not self.joined:
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
        object.__setattr__(
            self,
            "axis_scales",
            _normalize_positive_vector3(
                self.axis_scales,
                "Door slot axis scales",
            ),
        )

    @property
    def display_name(self) -> str:
        return DOOR_SLOT_DISPLAY_NAMES[self.slot_id]

    def with_transform(
        self,
        *,
        position_meters: Sequence[float] | None = None,
        rotation_degrees: Sequence[float] | None = None,
        axis_scales: Sequence[float] | None = None,
        joined: bool | None = None,
        mirrored_horizontally: bool | None = None,
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
            axis_scales=(
                self.axis_scales if axis_scales is None else tuple(axis_scales)
            ),
            joined=self.joined if joined is None else joined,
            mirrored_horizontally=(
                self.mirrored_horizontally
                if mirrored_horizontally is None
                else mirrored_horizontally
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "source_object_id": self.source_object_id,
            "position_meters": list(self.position_meters),
            "rotation_degrees": list(self.rotation_degrees),
            "axis_scales": list(self.axis_scales),
            "joined": self.joined,
            "mirrored_horizontally": self.mirrored_horizontally,
        }

    @classmethod
    def from_dict(cls, payload: object) -> DoorSlotData:
        raw = _require_mapping(payload, "Door slot")
        return cls(
            slot_id=str(raw["slot_id"]),
            source_object_id=str(raw["source_object_id"]),
            position_meters=raw.get("position_meters", (0.0, 0.0, 0.0)),
            rotation_degrees=raw.get("rotation_degrees", (0.0, 0.0, 0.0)),
            axis_scales=raw.get("axis_scales", (1.0, 1.0, 1.0)),
            joined=raw.get("joined", False),
            mirrored_horizontally=raw.get("mirrored_horizontally", False),
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
    make_double: bool = False
    make_double_sided: bool = False
    generate_displacement: bool = True
    preserve_profile: bool = False
    side_duplication: DoorSideDuplication | None = None
    slots: tuple[DoorSlotData, ...] = ()
    glass_areas: tuple[GlassAreaData, ...] = ()

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
        if not isinstance(self.make_double_sided, bool):
            raise TypeError("Door double-sided state must be boolean.")
        if not isinstance(self.make_double, bool):
            raise TypeError("Door double-leaf state must be boolean.")
        if not isinstance(self.generate_displacement, bool):
            raise TypeError("Door displacement-generation state must be boolean.")
        if not isinstance(self.preserve_profile, bool):
            raise TypeError("Door profile-preservation state must be boolean.")
        required_slot_ids = {DOOR_SLOT_BODY, DOOR_SLOT_KNOB}
        left_slot_ids = {DOOR_SLOT_LEFT_BODY, DOOR_SLOT_LEFT_KNOB}
        back_slot_ids = {DOOR_SLOT_BACK_BODY, DOOR_SLOT_BACK_KNOB}
        back_left_slot_ids = {
            DOOR_SLOT_BACK_LEFT_BODY,
            DOOR_SLOT_BACK_LEFT_KNOB,
        }
        if self.make_double:
            required_slot_ids.update(left_slot_ids)
        if self.make_double_sided:
            required_slot_ids.update(back_slot_ids)
            if self.make_double:
                required_slot_ids.update(back_left_slot_ids)
        missing_slot_ids = required_slot_ids.difference(slot_ids)
        if missing_slot_ids:
            raise ValueError(
                "A door definition is missing required component slots: "
                + ", ".join(sorted(missing_slot_ids))
                + "."
            )
        unexpected_slot_ids = set(slot_ids).difference(required_slot_ids)
        if unexpected_slot_ids:
            raise ValueError(
                "A door definition contains inactive component slots: "
                + ", ".join(sorted(unexpected_slot_ids))
                + "."
            )
        if self.side_duplication is not None and not isinstance(
            self.side_duplication,
            DoorSideDuplication,
        ):
            raise TypeError(
                "Door side duplication must be DoorSideDuplication or None."
            )
        if self.make_double_sided and self.side_duplication is not None:
            raise ValueError(
                "A double-sided door cannot use side door duplication."
            )
        glass_areas = _normalize_glass_areas(
            self.glass_areas,
            "Door glass areas",
        )
        if any(area.body_slot_id not in slot_ids for area in glass_areas):
            raise ValueError(
                "Every door glass area must reference an active body slot."
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
        object.__setattr__(self, "glass_areas", glass_areas)

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

    def get_glass_area(self, glass_area_id: str) -> GlassAreaData | None:
        """Return one glass area by its stable identifier."""

        normalized_id = str(glass_area_id).strip()
        return next(
            (
                area
                for area in self.glass_areas
                if area.glass_area_id == normalized_id
            ),
            None,
        )

    def with_double_sided(self, enabled: bool) -> DoorDefinition:
        """Return this door with deterministic front/back component slots."""

        return set_door_double_sided(self, enabled)

    def with_double(self, enabled: bool) -> DoorDefinition:
        """Return this door with explicit left/right leaf component slots."""

        return set_door_double(self, enabled)

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
            "make_double": self.make_double,
            "make_double_sided": self.make_double_sided,
            "generate_displacement": self.generate_displacement,
            "preserve_profile": self.preserve_profile,
            "side_duplication": (
                None
                if self.side_duplication is None
                else self.side_duplication.to_dict()
            ),
            "slots": [slot.to_dict() for slot in self.slots],
            "glass_areas": [area.to_dict() for area in self.glass_areas],
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
        raw_glass_areas = raw.get("glass_areas", ())
        if (
            isinstance(raw_glass_areas, (str, bytes, bytearray))
            or not isinstance(raw_glass_areas, Sequence)
        ):
            raise TypeError("Door glass areas must contain a list.")
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
            make_double=raw.get("make_double", False),
            make_double_sided=raw.get("make_double_sided", False),
            generate_displacement=raw.get("generate_displacement", False),
            preserve_profile=raw.get("preserve_profile", False),
            side_duplication=(
                None
                if raw.get("side_duplication") is None
                else DoorSideDuplication.from_dict(raw["side_duplication"])
            ),
            slots=tuple(
                DoorSlotData.from_dict(slot)
                for slot in raw_slots
                if not _is_legacy_hinge_slot_payload(slot)
            ),
            glass_areas=tuple(
                GlassAreaData.from_dict(area) for area in raw_glass_areas
            ),
        )


# ### Window definition models ###
@dataclass(frozen=True, slots=True)
class WindowDefinition:
    """One reusable window whose canonical body fits a wall-window frame."""

    window_id: str
    name: str
    source_window_frame_id: str
    width_meters: float
    height_meters: float
    thickness_meters: float = DEFAULT_DOOR_BODY_THICKNESS_METERS
    make_double: bool = False
    make_double_sided: bool = False
    generate_displacement: bool = True
    preserve_profile: bool = False
    slots: tuple[DoorSlotData, ...] = ()
    glass_areas: tuple[GlassAreaData, ...] = ()

    def __post_init__(self) -> None:
        window_id = _normalize_identifier(self.window_id, "Window")
        source_frame_id = _normalize_identifier(
            self.source_window_frame_id,
            "Source window frame",
        )
        name = str(self.name).strip()
        if not name:
            raise ValueError("Window names cannot be empty.")
        if len(name) > MAX_DOOR_NAME_LENGTH:
            raise ValueError("Window name is too long.")
        if not isinstance(self.generate_displacement, bool):
            raise TypeError("Window displacement-generation state must be boolean.")
        if not isinstance(self.preserve_profile, bool):
            raise TypeError("Window profile-preservation state must be boolean.")
        if not isinstance(self.make_double, bool):
            raise TypeError("Window double-leaf state must be boolean.")
        if not isinstance(self.make_double_sided, bool):
            raise TypeError("Window double-sided state must be boolean.")

        slots = tuple(self.slots)
        if any(not isinstance(slot, DoorSlotData) for slot in slots):
            raise TypeError("Window definitions require DoorSlotData values.")
        slot_ids = tuple(slot.slot_id for slot in slots)
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("A window cannot contain duplicate component slots.")
        required_slot_ids = {DOOR_SLOT_BODY, DOOR_SLOT_KNOB}
        if self.make_double:
            required_slot_ids.update(
                {DOOR_SLOT_LEFT_BODY, DOOR_SLOT_LEFT_KNOB}
            )
        if self.make_double_sided:
            required_slot_ids.update(
                {DOOR_SLOT_BACK_BODY, DOOR_SLOT_BACK_KNOB}
            )
            if self.make_double:
                required_slot_ids.update(
                    {DOOR_SLOT_BACK_LEFT_BODY, DOOR_SLOT_BACK_LEFT_KNOB}
                )
        missing_slot_ids = required_slot_ids.difference(slot_ids)
        if missing_slot_ids:
            raise ValueError(
                "A window definition is missing required component slots: "
                + ", ".join(sorted(missing_slot_ids))
                + "."
            )
        unexpected_slot_ids = set(slot_ids).difference(required_slot_ids)
        if unexpected_slot_ids:
            raise ValueError(
                "A window definition contains inactive component slots: "
                + ", ".join(sorted(unexpected_slot_ids))
                + "."
            )

        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "source_window_frame_id", source_frame_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "width_meters",
            _normalize_dimension(self.width_meters, "Window width"),
        )
        object.__setattr__(
            self,
            "height_meters",
            _normalize_dimension(self.height_meters, "Window height"),
        )
        object.__setattr__(
            self,
            "thickness_meters",
            _normalize_dimension(self.thickness_meters, "Window thickness"),
        )
        object.__setattr__(self, "slots", slots)
        glass_areas = _normalize_glass_areas(
            self.glass_areas,
            "Window glass areas",
        )
        if any(area.body_slot_id not in slot_ids for area in glass_areas):
            raise ValueError(
                "Every window glass area must reference an active Body slot."
            )
        object.__setattr__(self, "glass_areas", glass_areas)

    def get_slot(self, slot_id: str) -> DoorSlotData | None:
        normalized_id = str(slot_id).strip().lower()
        return next(
            (slot for slot in self.slots if slot.slot_id == normalized_id),
            None,
        )

    def replace_slot(self, replacement: DoorSlotData) -> WindowDefinition:
        if not isinstance(replacement, DoorSlotData):
            raise TypeError("Window slots can only be replaced by DoorSlotData.")
        if self.get_slot(replacement.slot_id) is None:
            raise ValueError(
                f"Window slot {replacement.slot_id!r} does not exist."
            )
        return replace(
            self,
            slots=tuple(
                replacement if slot.slot_id == replacement.slot_id else slot
                for slot in self.slots
            ),
        )

    def get_glass_area(self, glass_area_id: str) -> GlassAreaData | None:
        normalized_id = str(glass_area_id).strip()
        return next(
            (
                area
                for area in self.glass_areas
                if area.glass_area_id == normalized_id
            ),
            None,
        )

    def with_double(self, enabled: bool) -> WindowDefinition:
        """Return this window with explicit left/right component slots."""

        return set_window_double(self, enabled)

    def with_double_sided(self, enabled: bool) -> WindowDefinition:
        """Return this window with deterministic front/back component slots."""

        return set_window_double_sided(self, enabled)

    def to_dict(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "name": self.name,
            "source_window_frame_id": self.source_window_frame_id,
            "width_meters": self.width_meters,
            "height_meters": self.height_meters,
            "thickness_meters": self.thickness_meters,
            "make_double": self.make_double,
            "make_double_sided": self.make_double_sided,
            "generate_displacement": self.generate_displacement,
            "preserve_profile": self.preserve_profile,
            "slots": [slot.to_dict() for slot in self.slots],
            "glass_areas": [area.to_dict() for area in self.glass_areas],
        }

    @classmethod
    def from_dict(cls, payload: object) -> WindowDefinition:
        raw = _require_mapping(payload, "Window definition")
        raw_slots = _require_sequence(raw.get("slots", ()), "Window slots")
        raw_glass_areas = _require_sequence(
            raw.get("glass_areas", ()),
            "Window glass areas",
        )
        return cls(
            window_id=str(raw["window_id"]),
            name=str(raw["name"]),
            source_window_frame_id=str(raw["source_window_frame_id"]),
            width_meters=raw["width_meters"],
            height_meters=raw["height_meters"],
            thickness_meters=raw.get(
                "thickness_meters",
                DEFAULT_DOOR_BODY_THICKNESS_METERS,
            ),
            make_double=raw.get("make_double", False),
            make_double_sided=raw.get("make_double_sided", False),
            generate_displacement=raw.get("generate_displacement", False),
            preserve_profile=raw.get("preserve_profile", False),
            slots=tuple(DoorSlotData.from_dict(slot) for slot in raw_slots),
            glass_areas=tuple(
                GlassAreaData.from_dict(area) for area in raw_glass_areas
            ),
        )


# ### Placement models ###
@dataclass(frozen=True, slots=True)
class DoorPlacement:
    """One door attached to a stable doorway identity on a level."""

    placement_id: str
    door_id: str
    level_index: int
    doorway_id: str
    mirrored_horizontally: bool = False
    mirrored_depthwise: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.level_index, bool)
            or not isinstance(self.level_index, int)
            or self.level_index < 0
        ):
            raise ValueError("Door placement level index must be non-negative.")
        if not isinstance(self.mirrored_horizontally, bool):
            raise TypeError("Door placement horizontal mirror state must be boolean.")
        if not isinstance(self.mirrored_depthwise, bool):
            raise TypeError("Door placement depth mirror state must be boolean.")
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
            "mirrored_depthwise": self.mirrored_depthwise,
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
            mirrored_depthwise=raw.get("mirrored_depthwise", False),
        )


@dataclass(frozen=True, slots=True)
class WindowPlacement:
    """One reusable window attached to a stable wall-window frame."""

    placement_id: str
    window_id: str
    level_index: int
    window_frame_id: str
    mirrored_horizontally: bool = False
    mirrored_depthwise: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.level_index, bool)
            or not isinstance(self.level_index, int)
            or self.level_index < 0
        ):
            raise ValueError("Window placement level index must be non-negative.")
        if not isinstance(self.mirrored_horizontally, bool):
            raise TypeError(
                "Window placement horizontal mirror state must be boolean."
            )
        if not isinstance(self.mirrored_depthwise, bool):
            raise TypeError("Window placement depth mirror state must be boolean.")
        object.__setattr__(
            self,
            "placement_id",
            _normalize_identifier(self.placement_id, "Window placement"),
        )
        object.__setattr__(
            self,
            "window_id",
            _normalize_identifier(self.window_id, "Window"),
        )
        object.__setattr__(
            self,
            "window_frame_id",
            _normalize_identifier(self.window_frame_id, "Window frame"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "placement_id": self.placement_id,
            "window_id": self.window_id,
            "level_index": self.level_index,
            "window_frame_id": self.window_frame_id,
            "mirrored_horizontally": self.mirrored_horizontally,
            "mirrored_depthwise": self.mirrored_depthwise,
        }

    @classmethod
    def from_dict(cls, payload: object) -> WindowPlacement:
        raw = _require_mapping(payload, "Window placement")
        return cls(
            placement_id=str(raw["placement_id"]),
            window_id=str(raw["window_id"]),
            level_index=raw["level_index"],
            window_frame_id=str(raw["window_frame_id"]),
            mirrored_horizontally=raw.get("mirrored_horizontally", False),
            mirrored_depthwise=raw.get("mirrored_depthwise", False),
        )


# ### Door and window library model ###
@dataclass
class DoorLibraryData:
    """Persistent reusable door/window assets and opening-bound placements."""

    doors: list[DoorDefinition] = field(default_factory=list)
    placements: list[DoorPlacement] = field(default_factory=list)
    windows: list[WindowDefinition] = field(default_factory=list)
    window_placements: list[WindowPlacement] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.doors = list(self.doors)
        self.placements = list(self.placements)
        self.windows = list(self.windows)
        self.window_placements = list(self.window_placements)
        if any(not isinstance(door, DoorDefinition) for door in self.doors):
            raise TypeError("Door libraries require DoorDefinition values.")
        if any(
            not isinstance(placement, DoorPlacement)
            for placement in self.placements
        ):
            raise TypeError("Door libraries require DoorPlacement values.")
        if any(
            not isinstance(window, WindowDefinition) for window in self.windows
        ):
            raise TypeError("Door libraries require WindowDefinition values.")
        if any(
            not isinstance(placement, WindowPlacement)
            for placement in self.window_placements
        ):
            raise TypeError("Door libraries require WindowPlacement values.")
        _require_unique(
            (door.door_id for door in self.doors),
            "Door IDs",
        )
        _require_unique(
            (window.window_id for window in self.windows),
            "Window IDs",
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
        _require_unique(
            (
                placement.placement_id
                for placement in self.window_placements
            ),
            "Window placement IDs",
        )
        _require_unique(
            (
                f"{placement.level_index}:{placement.window_frame_id}"
                for placement in self.window_placements
            ),
            "Window-frame placement bindings",
        )
        door_ids = {door.door_id for door in self.doors}
        if any(
            placement.door_id not in door_ids for placement in self.placements
        ):
            raise ValueError("Every door placement must reference a known door.")
        window_ids = {window.window_id for window in self.windows}
        if any(
            placement.window_id not in window_ids
            for placement in self.window_placements
        ):
            raise ValueError(
                "Every window placement must reference a known window."
            )

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

    def get_window(self, window_id: str) -> WindowDefinition | None:
        normalized_id = str(window_id).strip()
        return next(
            (
                window
                for window in self.windows
                if window.window_id == normalized_id
            ),
            None,
        )

    def replace_window(self, replacement: WindowDefinition) -> None:
        if not isinstance(replacement, WindowDefinition):
            raise TypeError(
                "Windows can only be replaced by WindowDefinition values."
            )
        for index, window in enumerate(self.windows):
            if window.window_id == replacement.window_id:
                self.windows[index] = replacement
                return
        raise ValueError(f"Window {replacement.window_id!r} does not exist.")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": DOOR_LIBRARY_SCHEMA_VERSION,
            "doors": [door.to_dict() for door in self.doors],
            "placements": [placement.to_dict() for placement in self.placements],
            "windows": [window.to_dict() for window in self.windows],
            "window_placements": [
                placement.to_dict() for placement in self.window_placements
            ],
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
        raw_windows = _require_sequence(
            raw.get("windows", ()),
            "Window definitions",
        )
        raw_window_placements = _require_sequence(
            raw.get("window_placements", ()),
            "Window placements",
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

        windows: list[WindowDefinition] = []
        occupied_window_ids: set[str] = set()
        for raw_window in raw_windows:
            try:
                window = WindowDefinition.from_dict(raw_window)
            except (KeyError, TypeError, ValueError):
                continue
            if window.window_id in occupied_window_ids:
                continue
            windows.append(window)
            occupied_window_ids.add(window.window_id)

        window_placements: list[WindowPlacement] = []
        occupied_window_placement_ids: set[str] = set()
        occupied_window_frame_bindings: set[tuple[int, str]] = set()
        for raw_placement in raw_window_placements:
            try:
                placement = WindowPlacement.from_dict(raw_placement)
            except (KeyError, TypeError, ValueError):
                continue
            frame_binding = (
                placement.level_index,
                placement.window_frame_id,
            )
            if (
                placement.placement_id in occupied_window_placement_ids
                or placement.window_id not in occupied_window_ids
                or frame_binding in occupied_window_frame_bindings
            ):
                continue
            window_placements.append(placement)
            occupied_window_placement_ids.add(placement.placement_id)
            occupied_window_frame_bindings.add(frame_binding)
        return cls(
            doors=doors,
            placements=placements,
            windows=windows,
            window_placements=window_placements,
        )


# ### Door construction helpers ###
def create_door_definition_for_doorway(
    doorway: DoorwayData,
    *,
    name: str,
    door_id: str | None = None,
    thickness_meters: float | None = None,
) -> DoorDefinition:
    """Create canonical body and knob/handle slots for one doorway."""

    if not isinstance(doorway, DoorwayData):
        raise TypeError("Doors can only be created from DoorwayData.")
    normalized_door_id = uuid.uuid4().hex if door_id is None else door_id
    thickness = (
        min(DEFAULT_DOOR_BODY_THICKNESS_METERS, doorway.depth_meters)
        if thickness_meters is None
        else thickness_meters
    )
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
        preserve_profile=doorway.shape == DOORWAY_SHAPE_ARCH,
        slots=slots,
    )


# ### Window construction helpers ###
def create_window_definition_for_frame(
    source_window_frame_id: str,
    width_meters: float,
    height_meters: float,
    *,
    name: str,
    window_id: str | None = None,
    thickness_meters: float = DEFAULT_DOOR_BODY_THICKNESS_METERS,
) -> WindowDefinition:
    """Create canonical Body and Handle slots for one wall-window frame."""

    normalized_window_id = uuid.uuid4().hex if window_id is None else window_id
    width = _normalize_dimension(width_meters, "Window width")
    height = _normalize_dimension(height_meters, "Window height")
    handle_radius = min(0.035, width * 0.075)
    return WindowDefinition(
        window_id=normalized_window_id,
        name=name,
        source_window_frame_id=source_window_frame_id,
        width_meters=width,
        height_meters=height,
        thickness_meters=thickness_meters,
        slots=(
            DoorSlotData(
                slot_id=DOOR_SLOT_BODY,
                source_object_id=_window_slot_source_object_id(
                    normalized_window_id,
                    DOOR_SLOT_BODY,
                ),
                joined=True,
            ),
            DoorSlotData(
                slot_id=DOOR_SLOT_KNOB,
                source_object_id=_window_slot_source_object_id(
                    normalized_window_id,
                    DOOR_SLOT_KNOB,
                ),
                position_meters=(
                    width / 2.0 - max(0.08, handle_radius * 2.0),
                    0.0,
                    height * 0.45 - handle_radius,
                ),
            ),
        ),
    )


# ### Glass-area construction helpers ###
def create_glass_area_data(
    owner_id: str,
    existing_areas: Iterable[GlassAreaData],
    *,
    minimum_x_ratio: float,
    maximum_x_ratio: float,
    minimum_z_ratio: float,
    maximum_z_ratio: float,
    glass_area_id: str | None = None,
    source_object_id: str | None = None,
    body_slot_id: str = DOOR_SLOT_BODY,
) -> GlassAreaData:
    """Create one stable glass-area modifier with a monotonic display name."""

    normalized_owner_id = _normalize_identifier(owner_id, "Glass area owner")
    normalized_existing = tuple(existing_areas)
    if any(not isinstance(area, GlassAreaData) for area in normalized_existing):
        raise TypeError("Existing glass areas must contain GlassAreaData values.")
    normalized_area_id = (
        uuid.uuid4().hex if glass_area_id is None else glass_area_id
    )
    normalized_source_id = (
        f"opening:{normalized_owner_id}:glass:{normalized_area_id}"
        if source_object_id is None
        else source_object_id
    )
    return GlassAreaData(
        glass_area_id=normalized_area_id,
        source_object_id=normalized_source_id,
        name=next_glass_area_name(normalized_existing),
        minimum_x_ratio=minimum_x_ratio,
        maximum_x_ratio=maximum_x_ratio,
        minimum_z_ratio=minimum_z_ratio,
        maximum_z_ratio=maximum_z_ratio,
        body_slot_id=body_slot_id,
    )


# ### Door structural helpers ###
def set_door_double(
    door: DoorDefinition,
    enabled: bool,
) -> DoorDefinition:
    """Add or remove independently editable left/right leaf slots."""

    if not isinstance(door, DoorDefinition):
        raise TypeError("Only DoorDefinition values can be made double.")
    return _set_opening_double(door, enabled)


def set_window_double(
    window: WindowDefinition,
    enabled: bool,
) -> WindowDefinition:
    """Add or remove independently editable window leaf slots."""

    if not isinstance(window, WindowDefinition):
        raise TypeError("Only WindowDefinition values can be made double.")
    return _set_opening_double(window, enabled)


def _set_opening_double(
    door: DoorDefinition | WindowDefinition,
    enabled: bool,
) -> DoorDefinition | WindowDefinition:
    """Apply one reversible left/right split to an opening asset."""

    if not isinstance(enabled, bool):
        raise TypeError("Opening double-leaf state must be boolean.")
    if enabled == door.make_double:
        return door

    if not enabled:
        retained_slots: list[DoorSlotData] = []
        for slot in door.slots:
            if door_slot_leaf(slot.slot_id) == DOOR_LEAF_LEFT:
                continue
            if is_door_body_slot(slot.slot_id):
                slot = slot.with_transform(
                    position_meters=(
                        0.0,
                        slot.position_meters[1],
                        slot.position_meters[2],
                    ),
                    axis_scales=(
                        slot.axis_scales[0] * 2.0,
                        slot.axis_scales[1],
                        slot.axis_scales[2],
                    ),
                    mirrored_horizontally=(
                        not slot.mirrored_horizontally
                    ),
                )
            elif is_door_knob_slot(slot.slot_id):
                slot = slot.with_transform(
                    position_meters=(
                        door.width_meters * 0.5
                        - slot.position_meters[0] * 2.0,
                        slot.position_meters[1],
                        slot.position_meters[2],
                    ),
                    mirrored_horizontally=(
                        not slot.mirrored_horizontally
                    ),
                )
            retained_slots.append(slot)
        retained_slot_ids = {slot.slot_id for slot in retained_slots}
        return replace(
            door,
            make_double=False,
            slots=tuple(retained_slots),
            glass_areas=tuple(
                area
                for area in door.glass_areas
                if area.body_slot_id in retained_slot_ids
            ),
        )

    slots_by_id = {slot.slot_id: slot for slot in door.slots}
    owner_id = (
        door.door_id if isinstance(door, DoorDefinition) else door.window_id
    )
    source_object_id = (
        _slot_source_object_id
        if isinstance(door, DoorDefinition)
        else _window_slot_source_object_id
    )
    updated_slots: list[DoorSlotData] = []
    for right_body_id, right_handle_id, left_body_id, left_handle_id in (
        (
            DOOR_SLOT_BODY,
            DOOR_SLOT_KNOB,
            DOOR_SLOT_LEFT_BODY,
            DOOR_SLOT_LEFT_KNOB,
        ),
        (
            DOOR_SLOT_BACK_BODY,
            DOOR_SLOT_BACK_KNOB,
            DOOR_SLOT_BACK_LEFT_BODY,
            DOOR_SLOT_BACK_LEFT_KNOB,
        ),
    ):
        right_body = slots_by_id.get(right_body_id)
        right_handle = slots_by_id.get(right_handle_id)
        if right_body is None or right_handle is None:
            continue
        source_body = right_body
        source_handle = right_handle
        right_body = source_body.with_transform(
            position_meters=(
                door.width_meters * 0.25,
                source_body.position_meters[1],
                source_body.position_meters[2],
            ),
            axis_scales=(
                source_body.axis_scales[0] * 0.5,
                source_body.axis_scales[1],
                source_body.axis_scales[2],
            ),
            mirrored_horizontally=(
                not source_body.mirrored_horizontally
            ),
        )
        left_body = replace(
            source_body,
            slot_id=left_body_id,
            source_object_id=source_object_id(
                owner_id,
                left_body_id,
            ),
            position_meters=(
                -door.width_meters * 0.25,
                source_body.position_meters[1],
                source_body.position_meters[2],
            ),
            axis_scales=(
                source_body.axis_scales[0] * 0.5,
                source_body.axis_scales[1],
                source_body.axis_scales[2],
            ),
        )
        right_handle_x = (
            door.width_meters * 0.25
            - source_handle.position_meters[0] * 0.5
        )
        right_handle = source_handle.with_transform(
            position_meters=(
                right_handle_x,
                source_handle.position_meters[1],
                source_handle.position_meters[2],
            ),
            mirrored_horizontally=(
                not source_handle.mirrored_horizontally
            ),
        )
        left_handle = replace(
            source_handle,
            slot_id=left_handle_id,
            source_object_id=source_object_id(
                owner_id,
                left_handle_id,
            ),
            position_meters=(
                -right_handle_x,
                source_handle.position_meters[1],
                source_handle.position_meters[2],
            ),
        )
        updated_slots.extend(
            (right_body, left_body, right_handle, left_handle)
        )
    return replace(
        door,
        make_double=True,
        slots=tuple(updated_slots),
    )


def set_door_double_sided(
    door: DoorDefinition,
    enabled: bool,
) -> DoorDefinition:
    """Add or remove deterministic back-side slots without changing the front."""

    if not isinstance(door, DoorDefinition):
        raise TypeError("Only DoorDefinition values can be made double-sided.")
    return _set_opening_double_sided(door, enabled)


def set_window_double_sided(
    window: WindowDefinition,
    enabled: bool,
) -> WindowDefinition:
    """Add or remove deterministic back-side window slots."""

    if not isinstance(window, WindowDefinition):
        raise TypeError("Only WindowDefinition values can be made double-sided.")
    return _set_opening_double_sided(window, enabled)


def _set_opening_double_sided(
    door: DoorDefinition | WindowDefinition,
    enabled: bool,
) -> DoorDefinition | WindowDefinition:
    """Apply one reversible front/back split to an opening asset."""

    if not isinstance(enabled, bool):
        raise TypeError("Opening double-sided state must be boolean.")
    if enabled == door.make_double_sided:
        return door
    if (
        enabled
        and isinstance(door, DoorDefinition)
        and door.side_duplication is not None
    ):
        raise ValueError(
            "A side-duplicated door must be regenerated before it can be "
            "made double-sided."
        )
    if not enabled:
        return replace(
            door,
            make_double_sided=False,
            slots=tuple(
                slot
                for slot in door.slots
                if slot.slot_id
                not in {
                    DOOR_SLOT_BACK_BODY,
                    DOOR_SLOT_BACK_KNOB,
                    DOOR_SLOT_BACK_LEFT_BODY,
                    DOOR_SLOT_BACK_LEFT_KNOB,
                }
            ),
            glass_areas=tuple(
                area
                for area in door.glass_areas
                if door_slot_side(area.body_slot_id)
                != DOOR_SIDE_DUPLICATION_BACK
            ),
        )

    owner_id = (
        door.door_id if isinstance(door, DoorDefinition) else door.window_id
    )
    source_object_id = (
        _slot_source_object_id
        if isinstance(door, DoorDefinition)
        else _window_slot_source_object_id
    )
    back_slots: list[DoorSlotData] = []
    for front_body_id, front_handle_id, back_body_id, back_handle_id in (
        (
            DOOR_SLOT_BODY,
            DOOR_SLOT_KNOB,
            DOOR_SLOT_BACK_BODY,
            DOOR_SLOT_BACK_KNOB,
        ),
        (
            DOOR_SLOT_LEFT_BODY,
            DOOR_SLOT_LEFT_KNOB,
            DOOR_SLOT_BACK_LEFT_BODY,
            DOOR_SLOT_BACK_LEFT_KNOB,
        ),
    ):
        front_body = door.get_slot(front_body_id)
        front_handle = door.get_slot(front_handle_id)
        if front_body is None or front_handle is None:
            continue
        back_slots.append(
            replace(
                front_body,
                slot_id=back_body_id,
                source_object_id=source_object_id(
                    owner_id,
                    back_body_id,
                ),
                position_meters=(
                    front_body.position_meters[0],
                    -front_body.position_meters[1],
                    front_body.position_meters[2],
                ),
                rotation_degrees=(
                    -front_body.rotation_degrees[0],
                    front_body.rotation_degrees[1],
                    -front_body.rotation_degrees[2],
                ),
            )
        )
        back_slots.append(
            replace(
                front_handle,
                slot_id=back_handle_id,
                source_object_id=source_object_id(
                    owner_id,
                    back_handle_id,
                ),
                position_meters=(
                    front_handle.position_meters[0],
                    -front_handle.position_meters[1],
                    front_handle.position_meters[2],
                ),
                rotation_degrees=(
                    -front_handle.rotation_degrees[0],
                    front_handle.rotation_degrees[1],
                    -front_handle.rotation_degrees[2],
                ),
                joined=False,
            )
        )
    return replace(
        door,
        make_double_sided=True,
        slots=(*door.slots, *back_slots),
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


def next_window_name(windows: Iterable[WindowDefinition]) -> str:
    """Return the next monotonic ``New window n`` library name."""

    largest_index = 0
    for window in windows:
        if not isinstance(window, WindowDefinition):
            continue
        match = _WINDOW_NAME_PATTERN.fullmatch(window.name.strip())
        if match is not None:
            largest_index = max(largest_index, int(match.group(1)))
    return f"{WINDOW_NAME_PREFIX} {largest_index + 1}"


def next_glass_area_name(areas: Iterable[GlassAreaData]) -> str:
    """Return a monotonic name without reusing deleted glass-area indices."""

    largest_index = 0
    for area in areas:
        if not isinstance(area, GlassAreaData):
            continue
        match = _GLASS_AREA_NAME_PATTERN.fullmatch(area.name.strip())
        if match is not None:
            largest_index = max(largest_index, int(match.group(1)))
    return f"{GLASS_AREA_NAME_PREFIX} {largest_index + 1}"


def door_fits_doorway(
    door: DoorDefinition,
    doorway: DoorwayData,
) -> bool:
    """Return whether the door can be conformed to the opening profile.

    Width, height, and arch amount are destination-specific fitting inputs.
    A rectangular body deliberately remains incompatible with an arched
    opening because changing that topology would alter the authored asset's
    design rather than resize the same kind of door.
    """

    return door.shape == doorway.shape


# ### Door slot helpers ###
def is_door_body_slot(slot_id: object) -> bool:
    """Return whether a slot identifies a front or back door body."""

    return str(slot_id).strip().lower() in DOOR_BODY_SLOT_IDS


def is_door_knob_slot(slot_id: object) -> bool:
    """Return whether a slot identifies a front or back knob/handle."""

    return str(slot_id).strip().lower() in DOOR_KNOB_SLOT_IDS


def door_slot_side(slot_id: object) -> str:
    """Return the stable front/back hierarchy side for an active slot."""

    normalized_slot_id = str(slot_id).strip().lower()
    if normalized_slot_id not in DOOR_SLOT_KINDS:
        raise ValueError(f"Unknown door slot: {slot_id!r}.")
    if normalized_slot_id in {
        DOOR_SLOT_BACK_BODY,
        DOOR_SLOT_BACK_KNOB,
        DOOR_SLOT_BACK_LEFT_BODY,
        DOOR_SLOT_BACK_LEFT_KNOB,
    }:
        return DOOR_SIDE_DUPLICATION_BACK
    return DOOR_SIDE_DUPLICATION_FRONT


def door_slot_component(slot_id: object) -> str:
    """Return the body or knob component represented by an active slot."""

    normalized_slot_id = str(slot_id).strip().lower()
    if normalized_slot_id in DOOR_BODY_SLOT_IDS:
        return DOOR_SLOT_BODY
    if normalized_slot_id in DOOR_KNOB_SLOT_IDS:
        return DOOR_SLOT_KNOB
    raise ValueError(f"Unknown door slot: {slot_id!r}.")


def door_slot_leaf(slot_id: object) -> str:
    """Return the stable left/right leaf represented by a door slot."""

    normalized_slot_id = str(slot_id).strip().lower()
    if normalized_slot_id not in DOOR_SLOT_KINDS:
        raise ValueError(f"Unknown door slot: {slot_id!r}.")
    if normalized_slot_id in {
        DOOR_SLOT_LEFT_BODY,
        DOOR_SLOT_LEFT_KNOB,
        DOOR_SLOT_BACK_LEFT_BODY,
        DOOR_SLOT_BACK_LEFT_KNOB,
    }:
        return DOOR_LEAF_LEFT
    return DOOR_LEAF_RIGHT


def door_slot_counterpart(slot_id: object) -> str:
    """Return the opposite left/right slot on the same face and component."""

    normalized_slot_id = str(slot_id).strip().lower()
    counterparts = {
        DOOR_SLOT_BODY: DOOR_SLOT_LEFT_BODY,
        DOOR_SLOT_LEFT_BODY: DOOR_SLOT_BODY,
        DOOR_SLOT_KNOB: DOOR_SLOT_LEFT_KNOB,
        DOOR_SLOT_LEFT_KNOB: DOOR_SLOT_KNOB,
        DOOR_SLOT_BACK_BODY: DOOR_SLOT_BACK_LEFT_BODY,
        DOOR_SLOT_BACK_LEFT_BODY: DOOR_SLOT_BACK_BODY,
        DOOR_SLOT_BACK_KNOB: DOOR_SLOT_BACK_LEFT_KNOB,
        DOOR_SLOT_BACK_LEFT_KNOB: DOOR_SLOT_BACK_KNOB,
    }
    try:
        return counterparts[normalized_slot_id]
    except KeyError as error:
        raise ValueError(f"Unknown door slot: {slot_id!r}.") from error


def copy_door_slot_to_counterpart(
    source: DoorSlotData,
    target: DoorSlotData,
    *,
    source_object_id: str,
) -> DoorSlotData:
    """Copy one leaf part onto its exact mirrored counterpart transform."""

    if not isinstance(source, DoorSlotData) or not isinstance(
        target,
        DoorSlotData,
    ):
        raise TypeError("Door counterpart copies require two door slots.")
    if door_slot_counterpart(source.slot_id) != target.slot_id:
        raise ValueError("Door slots are not left/right counterparts.")
    if door_slot_component(source.slot_id) != door_slot_component(
        target.slot_id
    ):
        raise ValueError("Door counterpart slots must contain the same part.")
    return replace(
        target,
        source_object_id=_normalize_identifier(
            source_object_id,
            "Door slot source object",
        ),
        position_meters=(
            -source.position_meters[0],
            source.position_meters[1],
            source.position_meters[2],
        ),
        rotation_degrees=(
            source.rotation_degrees[0],
            -source.rotation_degrees[1],
            -source.rotation_degrees[2],
        ),
        axis_scales=source.axis_scales,
        joined=source.joined,
        mirrored_horizontally=not source.mirrored_horizontally,
    )


def door_slot_display_name(slot_id: object, *, make_double: bool) -> str:
    """Return a contextual component name for the active door structure."""

    normalized_slot_id = str(slot_id).strip().lower()
    if normalized_slot_id not in DOOR_SLOT_KINDS:
        raise ValueError(f"Unknown door slot: {slot_id!r}.")
    component_name = (
        "Body" if normalized_slot_id in DOOR_BODY_SLOT_IDS else "Handle"
    )
    if not make_double:
        return component_name
    return f"{component_name} {door_slot_leaf(normalized_slot_id)}"


# ### Validation helpers ###
def _normalize_ratio_bounds(
    raw_minimum: object,
    raw_maximum: object,
    label: str,
) -> tuple[float, float]:
    minimum = _normalize_finite_float(raw_minimum, f"{label} minimum")
    maximum = _normalize_finite_float(raw_maximum, f"{label} maximum")
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"{label} bounds must stay between zero and one.")
    if maximum - minimum <= MIN_GLASS_AREA_NORMALIZED_SPAN:
        raise ValueError(f"{label} bounds must have a positive span.")
    return minimum, maximum


def _normalize_glass_areas(
    raw_areas: object,
    label: str,
) -> tuple[GlassAreaData, ...]:
    if isinstance(raw_areas, (str, bytes, bytearray)) or not isinstance(
        raw_areas,
        Sequence,
    ):
        raise TypeError(f"{label} must contain a sequence.")
    areas = tuple(raw_areas)
    if any(not isinstance(area, GlassAreaData) for area in areas):
        raise TypeError(f"{label} must contain GlassAreaData values.")
    _require_unique(
        (area.glass_area_id for area in areas),
        f"{label} IDs",
    )
    _require_unique(
        (area.source_object_id for area in areas),
        f"{label} source object IDs",
    )
    _require_unique(
        (area.name.casefold() for area in areas),
        f"{label} names",
    )
    for first_index, first in enumerate(areas):
        for second in areas[first_index + 1 :]:
            if _glass_areas_overlap(first, second):
                raise ValueError(
                    f"{label} cannot overlap: {first.name!r} and "
                    f"{second.name!r}."
                )
    return areas


def _glass_areas_overlap(
    first: GlassAreaData,
    second: GlassAreaData,
) -> bool:
    """Return whether two normalized pane rectangles share positive area."""

    if door_slot_leaf(first.body_slot_id) != door_slot_leaf(
        second.body_slot_id
    ):
        return False
    overlap_x = min(first.maximum_x_ratio, second.maximum_x_ratio) - max(
        first.minimum_x_ratio,
        second.minimum_x_ratio,
    )
    overlap_z = min(first.maximum_z_ratio, second.maximum_z_ratio) - max(
        first.minimum_z_ratio,
        second.minimum_z_ratio,
    )
    return overlap_x > MIN_GLASS_AREA_NORMALIZED_SPAN and (
        overlap_z > MIN_GLASS_AREA_NORMALIZED_SPAN
    )


def _is_legacy_hinge_slot_payload(payload: object) -> bool:
    """Return whether a serialized v1/v2 slot is the removed hinge slot."""

    if not isinstance(payload, Mapping):
        return False
    return (
        str(payload.get("slot_id", "")).strip().lower()
        == _LEGACY_DOOR_SLOT_HINGES
    )


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


def _normalize_positive_vector3(value: object, label: str) -> Vector3:
    """Return three finite values that are all strictly positive."""

    coordinates = _normalize_vector3(value, label)
    if any(coordinate <= 0.0 for coordinate in coordinates):
        raise ValueError(f"{label} must be greater than zero on every axis.")
    return coordinates


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


def _window_slot_source_object_id(window_id: str, slot_id: str) -> str:
    return f"window:{window_id}:slot:{slot_id}"
