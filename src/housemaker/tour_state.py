# ### Imports ###
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise

# ### Constants ###
DEFAULT_TOUR_TENSION = 0.5
DEFAULT_TOUR_DURATION_SECONDS = 10.0
DEFAULT_TOUR_PROGRESS_SPEED = 1.0
DEFAULT_TOUR_TRIGGER_AREA_SIZE_METERS = (1.5, 1.5)
DEFAULT_TEXT_SIZE_POINTS = 12.0
DEFAULT_TEXT_FADE_DELAY_MS = 0.0
DEFAULT_TEXT_FADE_DURATION_MS = 2000.0
DEFAULT_TEXT_COLOR = "#ffffff"
DEFAULT_TEXT_ROTATION_DEGREES: tuple[float, float, float] = (0.0, 0.0, 0.0)
DEFAULT_SPEED_OVERRIDE = 1.0
DEFAULT_SPEED_EASING = "smoothstep"
DEFAULT_WAIT_INPUT_TOKEN = "any"
DEFAULT_IDLE_CAMERA_RADIUS_METERS = 0.08
DEFAULT_IDLE_CAMERA_CYCLE_DURATION_SECONDS = 6.0
DEFAULT_TOOLTIP_POSITION = "opposite"
DEFAULT_TOOLTIP_HTML_BODY = (
    "<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
    "eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut</p>"
)
DEFAULT_TOOLTIP_STYLE = """\
box-sizing: border-box;
max-width: 360px;
padding: 14px 16px;
color: #f5f7fb;
background: rgba(24, 27, 34, 0.94);
border: 1px solid rgba(255, 255, 255, 0.24);
border-radius: 8px;
font-family: sans-serif;
font-size: 14px;
line-height: 1.5;
""".strip()
LEGACY_TEXT_POINTS_PER_METER = 48.0
MAX_TOUR_NAME_LENGTH = 256
MAX_TOUR_TEXT_LENGTH = 10_000
MAX_TOUR_TOOLTIP_HTML_LENGTH = 100_000
MAX_TOUR_TOOLTIP_STYLE_LENGTH = 50_000
MAX_TOUR_INPUT_TOKEN_LENGTH = 128
MAX_TOUR_CURVE_POINTS = 10_000
MAX_TOUR_STEPS = 10_000
MAX_TOUR_ACTIONS_PER_STEP = 10_000
TOUR_CURVE_TYPE = "catmull_rom"
TOUR_TEXT_ACTION_TYPE = "text_3d"
TOUR_SPEED_OVERRIDE_ACTION_TYPE = "speed_override"
TOUR_WAIT_FOR_KEY_PRESS_ACTION_TYPE = "wait_for_key_press"
TOUR_IDLE_CAMERA_ANIMATION_ACTION_TYPE = "idle_camera_animation"
TOUR_FLOATING_TOOLTIP_ACTION_TYPE = "floating_tooltip"
TOUR_TOOLTIP_POSITION_OPTIONS = ("left", "right", "opposite")
TOUR_SPEED_EASING_OPTIONS = (
    "none",
    "linear",
    "smoothstep",
    "smootherstep",
    "ease_in_quad",
    "ease_out_quad",
    "ease_in_out_sine",
    "ease_in_out_quad",
    "ease_in_out_cubic",
)
_HEX_COLOR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")
_WAIT_INPUT_TOKEN_PATTERN = re.compile(
    r"^(?:any|mouse:(?:left|middle|right)|"
    r"key:(?:space|enter|escape|tab|up|down|left|right|\S)|"
    r"key_code:[0-9]+)$"
)


# ### Type aliases ###
WorldPoint = tuple[float, float, float]
PlanarSize = tuple[float, float]


# ### Action models ###
@dataclass(frozen=True, slots=True)
class TourText3DData:
    """One world-space 3D text action attached to a tour step.

    ``rotation_degrees`` stores static/extrinsic XYZ Euler angles in
    HouseMaker's Z-up world coordinate system. Runtime export converts that
    orientation into the glTF Y-up coordinate system before decomposing it
    back into static XYZ Euler angles.
    """

    component_id: str
    text: str
    position: WorldPoint
    size_points: float = DEFAULT_TEXT_SIZE_POINTS
    color: str = DEFAULT_TEXT_COLOR
    fade_delay_ms: float = DEFAULT_TEXT_FADE_DELAY_MS
    fade_duration_ms: float = DEFAULT_TEXT_FADE_DURATION_MS
    rotation_degrees: WorldPoint = DEFAULT_TEXT_ROTATION_DEGREES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_id",
            _normalize_identifier(self.component_id, "Tour action"),
        )
        normalized_text = str(self.text).strip()
        if not normalized_text:
            raise ValueError("Tour 3D text cannot be empty.")
        if len(normalized_text) > MAX_TOUR_TEXT_LENGTH:
            raise ValueError("Tour 3D text is too long.")
        object.__setattr__(self, "text", normalized_text)
        object.__setattr__(
            self,
            "position",
            _normalize_world_point(self.position, "Tour 3D text position"),
        )
        size_points = _normalize_finite_float(
            self.size_points,
            "Tour 3D text size",
        )
        if size_points <= 0.0:
            raise ValueError("Tour 3D text size must be greater than zero.")
        object.__setattr__(self, "size_points", size_points)
        normalized_color = str(self.color).strip().lower()
        if _HEX_COLOR_PATTERN.fullmatch(normalized_color) is None:
            raise ValueError("Tour 3D text color must use #RRGGBB format.")
        object.__setattr__(self, "color", normalized_color)
        fade_delay_ms = _normalize_finite_float(
            self.fade_delay_ms,
            "Tour 3D text fade delay",
        )
        if fade_delay_ms < 0.0:
            raise ValueError("Tour 3D text fade delay cannot be negative.")
        object.__setattr__(self, "fade_delay_ms", fade_delay_ms)
        fade_duration_ms = _normalize_finite_float(
            self.fade_duration_ms,
            "Tour 3D text fade duration",
        )
        if fade_duration_ms < 0.0:
            raise ValueError("Tour 3D text fade duration cannot be negative.")
        object.__setattr__(self, "fade_duration_ms", fade_duration_ms)
        object.__setattr__(
            self,
            "rotation_degrees",
            _normalize_world_point(
                self.rotation_degrees,
                "Tour 3D text rotation",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "type": TOUR_TEXT_ACTION_TYPE,
            "text": self.text,
            "position": list(self.position),
            "size_points": self.size_points,
            "color": self.color,
            "fade_delay_ms": self.fade_delay_ms,
            "fade_duration_ms": self.fade_duration_ms,
            "rotation_degrees": list(self.rotation_degrees),
        }

    @classmethod
    def from_dict(cls, payload: object) -> TourText3DData:
        raw = _require_mapping(payload, "Tour 3D text")
        action_type = str(raw.get("type", TOUR_TEXT_ACTION_TYPE))
        if action_type != TOUR_TEXT_ACTION_TYPE:
            raise ValueError(f"Unsupported tour action type: {action_type!r}.")
        if "size_points" in raw:
            size_points = raw["size_points"]
        elif "size_meters" in raw:
            size_points = (
                _normalize_finite_float(
                    raw["size_meters"],
                    "Legacy tour 3D text size",
                )
                * LEGACY_TEXT_POINTS_PER_METER
            )
        else:
            size_points = DEFAULT_TEXT_SIZE_POINTS
        return cls(
            component_id=str(raw["component_id"]),
            text=str(raw["text"]),
            position=_normalize_world_point(
                raw["position"],
                "Tour 3D text position",
            ),
            size_points=size_points,
            color=str(raw.get("color", DEFAULT_TEXT_COLOR)),
            fade_delay_ms=raw.get(
                "fade_delay_ms",
                DEFAULT_TEXT_FADE_DELAY_MS,
            ),
            fade_duration_ms=raw.get(
                "fade_duration_ms",
                DEFAULT_TEXT_FADE_DURATION_MS,
            ),
            rotation_degrees=raw.get(
                "rotation_degrees",
                DEFAULT_TEXT_ROTATION_DEGREES,
            ),
        )


# ### Additional action models ###
@dataclass(frozen=True, slots=True)
class TourSpeedOverrideData:
    """A step action that replaces the active tour progress multiplier."""

    component_id: str
    speed: float = DEFAULT_SPEED_OVERRIDE
    easing: str = DEFAULT_SPEED_EASING

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_id",
            _normalize_identifier(self.component_id, "Tour action"),
        )
        object.__setattr__(
            self,
            "speed",
            _normalize_positive_multiplier(self.speed, "Tour speed override"),
        )
        object.__setattr__(
            self,
            "easing",
            _normalize_action_easing(self.easing, "Tour speed"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "type": TOUR_SPEED_OVERRIDE_ACTION_TYPE,
            "speed": self.speed,
            "easing": self.easing,
        }

    @classmethod
    def from_dict(cls, payload: object) -> TourSpeedOverrideData:
        raw = _require_action_mapping(payload, TOUR_SPEED_OVERRIDE_ACTION_TYPE)
        return cls(
            component_id=str(raw["component_id"]),
            speed=raw.get("speed", DEFAULT_SPEED_OVERRIDE),
            easing=raw.get("easing", DEFAULT_SPEED_EASING),
        )


@dataclass(frozen=True, slots=True)
class TourWaitForKeyPressData:
    """Pause until input, then ease progress through the following interval."""

    component_id: str
    input_token: str = DEFAULT_WAIT_INPUT_TOKEN
    easing: str = DEFAULT_SPEED_EASING

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_id",
            _normalize_identifier(self.component_id, "Tour action"),
        )
        if not isinstance(self.input_token, str):
            raise TypeError("A wait-for-key-press input token must be a string.")
        input_token = self.input_token.strip().casefold()
        if not input_token:
            raise ValueError("A wait-for-key-press action requires an input token.")
        if len(input_token) > MAX_TOUR_INPUT_TOKEN_LENGTH:
            raise ValueError("The wait-for-key-press input token is too long.")
        if (
            not all(character.isprintable() for character in input_token)
            or _WAIT_INPUT_TOKEN_PATTERN.fullmatch(input_token) is None
        ):
            raise ValueError(
                f"Unsupported wait-for-key-press input token: {input_token!r}."
            )
        object.__setattr__(self, "input_token", input_token)
        object.__setattr__(
            self,
            "easing",
            _normalize_action_easing(self.easing, "Wait-for-key-press"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "type": TOUR_WAIT_FOR_KEY_PRESS_ACTION_TYPE,
            "input_token": self.input_token,
            "easing": self.easing,
        }

    @classmethod
    def from_dict(cls, payload: object) -> TourWaitForKeyPressData:
        raw = _require_action_mapping(payload, TOUR_WAIT_FOR_KEY_PRESS_ACTION_TYPE)
        return cls(
            component_id=str(raw["component_id"]),
            input_token=raw.get("input_token", DEFAULT_WAIT_INPUT_TOKEN),
            easing=raw.get("easing", DEFAULT_SPEED_EASING),
        )


@dataclass(frozen=True, slots=True)
class TourIdleCameraAnimationData:
    """A subtle looping camera orbit used while a tour waits for input."""

    component_id: str
    pivot_point: WorldPoint
    radius_meters: float = DEFAULT_IDLE_CAMERA_RADIUS_METERS
    cycle_duration_seconds: float = DEFAULT_IDLE_CAMERA_CYCLE_DURATION_SECONDS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_id",
            _normalize_identifier(self.component_id, "Tour action"),
        )
        object.__setattr__(
            self,
            "pivot_point",
            _normalize_world_point(self.pivot_point, "Idle camera pivot point"),
        )
        object.__setattr__(
            self,
            "radius_meters",
            _normalize_positive_multiplier(
                self.radius_meters,
                "Idle camera animation radius",
            ),
        )
        object.__setattr__(
            self,
            "cycle_duration_seconds",
            _normalize_positive_multiplier(
                self.cycle_duration_seconds,
                "Idle camera animation cycle duration",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "type": TOUR_IDLE_CAMERA_ANIMATION_ACTION_TYPE,
            "pivot_point": list(self.pivot_point),
            "radius_meters": self.radius_meters,
            "cycle_duration_seconds": self.cycle_duration_seconds,
        }

    @classmethod
    def from_dict(cls, payload: object) -> TourIdleCameraAnimationData:
        raw = _require_action_mapping(payload, TOUR_IDLE_CAMERA_ANIMATION_ACTION_TYPE)
        return cls(
            component_id=str(raw["component_id"]),
            pivot_point=_normalize_world_point(
                raw["pivot_point"],
                "Idle camera pivot point",
            ),
            radius_meters=raw.get(
                "radius_meters",
                DEFAULT_IDLE_CAMERA_RADIUS_METERS,
            ),
            cycle_duration_seconds=raw.get(
                "cycle_duration_seconds",
                DEFAULT_IDLE_CAMERA_CYCLE_DURATION_SECONDS,
            ),
        )


@dataclass(frozen=True, slots=True)
class TourFloatingTooltipData:
    """A hoverable world anchor that reveals a screen-space HTML tooltip."""

    component_id: str
    anchor_point: WorldPoint
    tooltip_position: str = DEFAULT_TOOLTIP_POSITION
    html_body: str = DEFAULT_TOOLTIP_HTML_BODY
    style: str = DEFAULT_TOOLTIP_STYLE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_id",
            _normalize_identifier(self.component_id, "Tour action"),
        )
        object.__setattr__(
            self,
            "anchor_point",
            _normalize_world_point(
                self.anchor_point,
                "Floating tooltip anchor point",
            ),
        )
        if not isinstance(self.tooltip_position, str):
            raise TypeError("Floating tooltip position must be a string.")
        tooltip_position = self.tooltip_position.strip().casefold()
        if tooltip_position not in TOUR_TOOLTIP_POSITION_OPTIONS:
            raise ValueError(
                "Unsupported floating tooltip position: "
                f"{tooltip_position!r}."
            )
        object.__setattr__(self, "tooltip_position", tooltip_position)
        if not isinstance(self.html_body, str):
            raise TypeError("Floating tooltip HTML body must be a string.")
        if not self.html_body.strip():
            raise ValueError("Floating tooltip HTML body cannot be empty.")
        if len(self.html_body) > MAX_TOUR_TOOLTIP_HTML_LENGTH:
            raise ValueError("Floating tooltip HTML body is too long.")
        if not isinstance(self.style, str):
            raise TypeError("Floating tooltip style must be a string.")
        if len(self.style) > MAX_TOUR_TOOLTIP_STYLE_LENGTH:
            raise ValueError("Floating tooltip style is too long.")

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "type": TOUR_FLOATING_TOOLTIP_ACTION_TYPE,
            "anchor_point": list(self.anchor_point),
            "tooltip_position": self.tooltip_position,
            "html_body": self.html_body,
            "style": self.style,
        }

    @classmethod
    def from_dict(cls, payload: object) -> TourFloatingTooltipData:
        raw = _require_action_mapping(payload, TOUR_FLOATING_TOOLTIP_ACTION_TYPE)
        return cls(
            component_id=str(raw["component_id"]),
            anchor_point=_normalize_world_point(
                raw["anchor_point"],
                "Floating tooltip anchor point",
            ),
            tooltip_position=raw.get(
                "tooltip_position",
                DEFAULT_TOOLTIP_POSITION,
            ),
            html_body=raw.get("html_body", DEFAULT_TOOLTIP_HTML_BODY),
            style=raw.get("style", DEFAULT_TOOLTIP_STYLE),
        )


# ### Action helpers ###
TourActionData = (
    TourText3DData
    | TourSpeedOverrideData
    | TourWaitForKeyPressData
    | TourIdleCameraAnimationData
    | TourFloatingTooltipData
)


def tour_action_from_dict(payload: object) -> TourActionData:
    """Deserialize one supported action using its explicit project type."""

    raw = _require_mapping(payload, "Tour action")
    action_type = str(raw.get("type", TOUR_TEXT_ACTION_TYPE))
    action_types = {
        TOUR_TEXT_ACTION_TYPE: TourText3DData,
        TOUR_SPEED_OVERRIDE_ACTION_TYPE: TourSpeedOverrideData,
        TOUR_WAIT_FOR_KEY_PRESS_ACTION_TYPE: TourWaitForKeyPressData,
        TOUR_IDLE_CAMERA_ANIMATION_ACTION_TYPE: TourIdleCameraAnimationData,
        TOUR_FLOATING_TOOLTIP_ACTION_TYPE: TourFloatingTooltipData,
    }
    action_class = action_types.get(action_type)
    if action_class is None:
        raise ValueError(f"Unsupported tour action type: {action_type!r}.")
    return action_class.from_dict(raw)


# ### Step and curve models ###
@dataclass(frozen=True, slots=True)
class TourStepData:
    """One normalized position on a tour with its world target and actions."""

    step_id: str
    progress: float
    camera_target: WorldPoint
    actions: tuple[TourActionData, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "step_id",
            _normalize_identifier(self.step_id, "Tour step"),
        )
        progress = _normalize_finite_float(self.progress, "Tour step progress")
        if not 0.0 <= progress <= 1.0:
            raise ValueError("Tour step progress must be in [0, 1].")
        object.__setattr__(self, "progress", progress)
        object.__setattr__(
            self,
            "camera_target",
            _normalize_world_point(
                self.camera_target,
                "Tour camera target",
            ),
        )
        actions = tuple(self.actions)
        if len(actions) > MAX_TOUR_ACTIONS_PER_STEP:
            raise ValueError("A tour step contains too many actions.")
        supported_action_types = (
            TourText3DData,
            TourSpeedOverrideData,
            TourWaitForKeyPressData,
            TourIdleCameraAnimationData,
            TourFloatingTooltipData,
        )
        if not all(isinstance(value, supported_action_types) for value in actions):
            raise ValueError("Tour step contains an unsupported action value.")
        _require_unique_identifiers(
            (action.component_id for action in actions),
            "Tour action IDs",
        )
        _validate_singleton_step_actions(actions)
        has_idle_animation = any(
            isinstance(action, TourIdleCameraAnimationData) for action in actions
        )
        has_wait_action = any(
            isinstance(action, TourWaitForKeyPressData) for action in actions
        )
        if has_idle_animation and not has_wait_action:
            raise ValueError(
                "Idle camera animation requires a wait-for-key-press action."
            )
        object.__setattr__(self, "actions", actions)

    @property
    def components(self) -> tuple[TourActionData, ...]:
        """Return actions under the legacy attribute name for old callers."""

        return self.actions

    @property
    def speed_override(self) -> float | None:
        """Return the action speed for read-only compatibility with older callers."""

        speed_action = next(
            (
                action
                for action in self.actions
                if isinstance(action, TourSpeedOverrideData)
            ),
            None,
        )
        return speed_action.speed if speed_action is not None else None

    def to_dict(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "progress": self.progress,
            "camera_target": list(self.camera_target),
            "actions": [action.to_dict() for action in self.actions],
        }

    @classmethod
    def from_dict(
        cls,
        payload: object,
        *,
        curve: TourCurveData | None = None,
    ) -> TourStepData:
        raw = _require_mapping(payload, "Tour step")
        raw_actions = _require_sequence(
            raw.get("actions", raw.get("components", ())),
            "Tour step actions",
        )
        progress = _normalize_finite_float(raw["progress"], "Tour step progress")
        camera_target = _read_camera_target(
            raw,
            curve=curve,
            progress=progress,
        )
        step_id = str(raw["step_id"])
        actions = tuple(tour_action_from_dict(action) for action in raw_actions)
        legacy_speed_override = raw.get("speed_override")
        if legacy_speed_override is not None and not any(
            isinstance(action, TourSpeedOverrideData) for action in actions
        ):
            actions = (
                *actions,
                TourSpeedOverrideData(
                    component_id=_unique_legacy_speed_action_id(step_id, actions),
                    speed=legacy_speed_override,
                    easing="none",
                ),
            )
        return cls(
            step_id=step_id,
            progress=progress,
            camera_target=camera_target,
            actions=actions,
        )


# ### Step action semantics ###
def effective_step_easing(step: TourStepData) -> str | None:
    """Return the easing that controls progress after a step begins.

    A wait action owns the post-resume easing when it shares a step with a
    speed override. The speed override still supplies the interval's speed.
    """

    if not isinstance(step, TourStepData):
        raise TypeError("Effective easing requires TourStepData.")
    wait_action = next(
        (
            action
            for action in step.actions
            if isinstance(action, TourWaitForKeyPressData)
        ),
        None,
    )
    if wait_action is not None:
        return wait_action.easing
    speed_action = next(
        (
            action
            for action in step.actions
            if isinstance(action, TourSpeedOverrideData)
        ),
        None,
    )
    return speed_action.easing if speed_action is not None else None


# ### Curve models ###
@dataclass(frozen=True, slots=True)
class TourCurveData:
    """An open Catmull-Rom path stored in HouseMaker's Z-up world space."""

    points: tuple[WorldPoint, ...]
    tension: float = DEFAULT_TOUR_TENSION
    closed: bool = False

    def __post_init__(self) -> None:
        points = tuple(
            _normalize_world_point(point, "Tour curve point") for point in self.points
        )
        if len(points) < 2:
            raise ValueError("A tour curve requires at least two points.")
        if len(points) > MAX_TOUR_CURVE_POINTS:
            raise ValueError("A tour curve contains too many points.")
        if any(first == second for first, second in pairwise(points)):
            raise ValueError("Consecutive tour curve points must be distinct.")
        object.__setattr__(self, "points", points)
        tension = _normalize_finite_float(self.tension, "Tour curve tension")
        if not 0.0 <= tension <= 1.0:
            raise ValueError("Tour curve tension must be in [0, 1].")
        object.__setattr__(self, "tension", tension)
        if not isinstance(self.closed, bool):
            raise TypeError("Tour curve closed state must be a boolean.")
        if self.closed:
            raise ValueError("HouseMaker tours currently require an open curve.")

    def to_dict(self) -> dict[str, object]:
        return {
            "type": TOUR_CURVE_TYPE,
            "closed": self.closed,
            "tension": self.tension,
            "points": [list(point) for point in self.points],
        }

    @classmethod
    def from_dict(cls, payload: object) -> TourCurveData:
        raw = _require_mapping(payload, "Tour curve")
        curve_type = str(raw.get("type", TOUR_CURVE_TYPE))
        if curve_type != TOUR_CURVE_TYPE:
            raise ValueError(f"Unsupported tour curve type: {curve_type!r}.")
        raw_points = _require_sequence(raw.get("points"), "Tour curve points")
        return cls(
            points=tuple(
                _normalize_world_point(point, "Tour curve point")
                for point in raw_points
            ),
            tension=float(raw.get("tension", DEFAULT_TOUR_TENSION)),
            closed=raw.get("closed", False),
        )


# ### Tour models ###
@dataclass(frozen=True, slots=True)
class TourData:
    """One triggerable authored camera tour."""

    tour_id: str
    name: str
    trigger_point: WorldPoint
    curve: TourCurveData
    trigger_area_size: PlanarSize = DEFAULT_TOUR_TRIGGER_AREA_SIZE_METERS
    steps: tuple[TourStepData, ...] = ()
    duration_seconds: float = DEFAULT_TOUR_DURATION_SECONDS
    progress_speed: float = DEFAULT_TOUR_PROGRESS_SPEED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tour_id",
            _normalize_identifier(self.tour_id, "Tour"),
        )
        normalized_name = str(self.name).strip()
        if not normalized_name:
            raise ValueError("Tour names cannot be empty.")
        if len(normalized_name) > MAX_TOUR_NAME_LENGTH:
            raise ValueError("Tour name is too long.")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(
            self,
            "trigger_point",
            _normalize_world_point(self.trigger_point, "Tour trigger point"),
        )
        object.__setattr__(
            self,
            "trigger_area_size",
            _normalize_planar_size(
                self.trigger_area_size,
                "Tour trigger area size",
            ),
        )
        if not isinstance(self.curve, TourCurveData):
            raise TypeError("Tours require a TourCurveData value.")
        duration_seconds = _normalize_finite_float(
            self.duration_seconds,
            "Tour duration",
        )
        if duration_seconds <= 0.0:
            raise ValueError("Tour duration must be greater than zero.")
        object.__setattr__(self, "duration_seconds", duration_seconds)
        progress_speed = _normalize_positive_multiplier(
            self.progress_speed,
            "Tour progress speed",
        )
        object.__setattr__(self, "progress_speed", progress_speed)
        steps = tuple(self.steps)
        if len(steps) > MAX_TOUR_STEPS:
            raise ValueError("A tour contains too many steps.")
        if not all(isinstance(value, TourStepData) for value in steps):
            raise ValueError("Tour steps must be TourStepData values.")
        _require_unique_identifiers(
            (step.step_id for step in steps),
            "Tour step IDs",
        )
        all_component_ids = (
            action.component_id for step in steps for action in step.actions
        )
        _require_unique_identifiers(all_component_ids, "Tour action IDs")
        object.__setattr__(self, "steps", steps)

    @property
    def curve_points(self) -> tuple[WorldPoint, ...]:
        return self.curve.points

    @property
    def tension(self) -> float:
        return self.curve.tension

    @property
    def closed(self) -> bool:
        return self.curve.closed

    def to_dict(self) -> dict[str, object]:
        return {
            "tour_id": self.tour_id,
            "name": self.name,
            "trigger_point": list(self.trigger_point),
            "trigger_area_size": list(self.trigger_area_size),
            "curve": self.curve.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "duration_seconds": self.duration_seconds,
            "progress_speed": self.progress_speed,
        }

    @classmethod
    def from_dict(cls, payload: object) -> TourData:
        raw = _require_mapping(payload, "Tour")
        raw_steps = _require_sequence(raw.get("steps", ()), "Tour steps")
        curve = TourCurveData.from_dict(raw["curve"])
        return cls(
            tour_id=str(raw["tour_id"]),
            name=str(raw["name"]),
            trigger_point=_normalize_world_point(
                raw["trigger_point"],
                "Tour trigger point",
            ),
            curve=curve,
            trigger_area_size=_normalize_planar_size(
                raw.get(
                    "trigger_area_size",
                    DEFAULT_TOUR_TRIGGER_AREA_SIZE_METERS,
                ),
                "Tour trigger area size",
            ),
            steps=tuple(
                TourStepData.from_dict(step, curve=curve) for step in raw_steps
            ),
            duration_seconds=float(
                raw.get("duration_seconds", DEFAULT_TOUR_DURATION_SECONDS)
            ),
            progress_speed=raw.get(
                "progress_speed",
                DEFAULT_TOUR_PROGRESS_SPEED,
            ),
        )


# ### Collection helpers ###
def tours_to_dicts(tours: Iterable[TourData]) -> list[dict[str, object]]:
    """Serialize a validated, ordered collection of project-space tours."""

    normalized = _normalize_tours(tours)
    return [tour.to_dict() for tour in normalized]


def tours_from_payload(payload: object) -> tuple[TourData, ...]:
    """Load valid tours while isolating individual damaged project entries."""

    if not isinstance(payload, (list, tuple)):
        return ()
    tours: list[TourData] = []
    occupied_ids: set[str] = set()
    for raw_tour in payload:
        try:
            tour = TourData.from_dict(raw_tour)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if tour.tour_id in occupied_ids:
            continue
        occupied_ids.add(tour.tour_id)
        tours.append(tour)
    return tuple(tours)


# ### Runtime action helpers ###
def _tour_action_to_runtime_dict(action: TourActionData) -> dict[str, object]:
    if isinstance(action, TourText3DData):
        return {
            "id": action.component_id,
            "type": "text3d",
            "text": action.text,
            "position": _world_point_to_gltf(action.position),
            "sizePoints": action.size_points,
            "color": action.color,
            "fadeDelayMs": action.fade_delay_ms,
            "fadeDurationMs": action.fade_duration_ms,
            "rotationDegrees": _world_euler_degrees_to_gltf(action.rotation_degrees),
            "rotationOrder": "XYZ",
        }
    if isinstance(action, TourSpeedOverrideData):
        return {
            "id": action.component_id,
            "type": "speedOverride",
            "speed": action.speed,
            "easing": action.easing,
        }
    if isinstance(action, TourWaitForKeyPressData):
        return {
            "id": action.component_id,
            "type": "waitForKeyPress",
            "input": action.input_token,
            "easing": action.easing,
        }
    if isinstance(action, TourIdleCameraAnimationData):
        return {
            "id": action.component_id,
            "type": "idleCameraAnimation",
            "pivotPoint": _world_point_to_gltf(action.pivot_point),
            "radiusMeters": action.radius_meters,
            "cycleDurationSeconds": action.cycle_duration_seconds,
        }
    if isinstance(action, TourFloatingTooltipData):
        return {
            "id": action.component_id,
            "type": "floatingTooltip",
            "anchorPoint": _world_point_to_gltf(action.anchor_point),
            "tooltipPosition": action.tooltip_position,
            "htmlBody": action.html_body,
            "style": action.style,
        }
    raise TypeError("Unsupported tour action value.")


# ### Runtime tour helpers ###
def tour_to_runtime_dict(tour: TourData) -> dict[str, object]:
    """Serialize one tour in the exported glTF/Y-up coordinate system."""

    if not isinstance(tour, TourData):
        raise TypeError("Runtime tours must contain TourData values.")
    ordered_steps = tuple(
        step
        for _index, step in sorted(
            enumerate(tour.steps),
            key=lambda item: (item[1].progress, item[0]),
        )
    )
    return {
        "id": tour.tour_id,
        "name": tour.name,
        "durationSeconds": tour.duration_seconds,
        "progressSpeed": tour.progress_speed,
        "triggerPoint": _world_point_to_gltf(tour.trigger_point),
        "triggerArea": {
            "center": _world_point_to_gltf(tour.trigger_point),
            "sizeMeters": list(tour.trigger_area_size),
            "horizontalAxes": "xz",
        },
        "curve": {
            "type": "catmullRom",
            "curveType": "catmullrom",
            "closed": False,
            "tension": tour.curve.tension,
            "points": [_world_point_to_gltf(point) for point in tour.curve.points],
        },
        "steps": [
            {
                "id": step.step_id,
                "progress": step.progress,
                "cameraTarget": _world_point_to_gltf(step.camera_target),
                "actions": [
                    _tour_action_to_runtime_dict(action) for action in step.actions
                ],
            }
            for step in ordered_steps
        ],
    }


def tours_to_runtime_dicts(
    tours: Iterable[TourData],
) -> list[dict[str, object]]:
    """Serialize tours in their user-visible project order for the manifest."""

    normalized = _normalize_tours(tours)
    return [tour_to_runtime_dict(tour) for tour in normalized]


def evaluate_catmull_rom_point(
    curve: TourCurveData,
    progress: float,
) -> WorldPoint:
    """Evaluate an open uniform Catmull-Rom curve in project Z-up space.

    Endpoint control points are extrapolated in the same way as Three.js'
    open ``CatmullRomCurve3`` implementation.  Consequently a two-point tour
    remains a straight line and the preview agrees with R3F when it builds a
    curve using ``curveType='catmullrom'`` and the exported tension.
    """

    if not isinstance(curve, TourCurveData):
        raise TypeError("Catmull-Rom evaluation requires TourCurveData.")
    normalized_progress = _normalize_finite_float(
        progress,
        "Tour curve progress",
    )
    if not 0.0 <= normalized_progress <= 1.0:
        raise ValueError("Tour curve progress must be in [0, 1].")

    segment_count = len(curve.points) - 1
    scaled_progress = normalized_progress * segment_count
    if normalized_progress == 1.0:
        segment_index = segment_count - 1
        segment_progress = 1.0
    else:
        segment_index = math.floor(scaled_progress)
        segment_progress = scaled_progress - segment_index

    first = curve.points[segment_index]
    second = curve.points[segment_index + 1]
    before = (
        curve.points[segment_index - 1]
        if segment_index > 0
        else _subtract_points(_scale_point(first, 2.0), second)
    )
    after = (
        curve.points[segment_index + 2]
        if segment_index + 2 < len(curve.points)
        else _subtract_points(_scale_point(second, 2.0), first)
    )
    return tuple(
        _evaluate_catmull_rom_axis(
            before[axis],
            first[axis],
            second[axis],
            after[axis],
            segment_progress,
            curve.tension,
        )
        for axis in range(3)
    )


def evaluate_tour_camera_look_direction(
    tour: TourData,
    progress: float,
) -> WorldPoint | None:
    """Derive a normalized look direction from the interpolated camera target."""

    if not isinstance(tour, TourData):
        raise TypeError("Camera look-direction evaluation requires TourData.")
    target = evaluate_tour_camera_target(tour, progress)
    if target is None:
        return None
    normalized_progress = min(
        1.0,
        max(0.0, _normalize_finite_float(progress, "Tour progress")),
    )
    camera_position = evaluate_catmull_rom_point(tour.curve, normalized_progress)
    direction = _subtract_points(target, camera_position)
    if math.hypot(*direction) > 0.0:
        return _normalize_world_direction(direction, "Tour camera look direction")
    return _evaluate_forward_curve_direction(tour.curve, normalized_progress)


def evaluate_tour_camera_target(
    tour: TourData,
    progress: float,
) -> WorldPoint | None:
    """Interpolate authored world-space camera targets across timeline steps."""

    if not isinstance(tour, TourData):
        raise TypeError("Camera-target evaluation requires TourData.")
    if not tour.steps:
        return None
    normalized_progress = min(
        1.0,
        max(0.0, _normalize_finite_float(progress, "Tour progress")),
    )
    ordered_steps = sorted(
        tour.steps,
        key=lambda step: (step.progress, step.step_id),
    )
    first_step = ordered_steps[0]
    last_step = ordered_steps[-1]
    if normalized_progress <= first_step.progress:
        return first_step.camera_target
    if normalized_progress >= last_step.progress:
        return last_step.camera_target

    for first, second in pairwise(ordered_steps):
        if normalized_progress > second.progress:
            continue
        progress_span = second.progress - first.progress
        if math.isclose(progress_span, 0.0, abs_tol=1e-12):
            return second.camera_target
        interpolation = (normalized_progress - first.progress) / progress_span
        return tuple(
            first.camera_target[axis]
            + (second.camera_target[axis] - first.camera_target[axis]) * interpolation
            for axis in range(3)
        )
    return last_step.camera_target


# ### Step migration helpers ###
def _read_camera_target(
    raw: dict[str, object],
    *,
    curve: TourCurveData | None,
    progress: float,
) -> WorldPoint:
    if "camera_target" in raw:
        return _normalize_world_point(
            raw["camera_target"],
            "Tour camera target",
        )
    if "camera_look_direction" not in raw:
        raise KeyError("camera_target")
    if curve is None:
        raise ValueError(
            "Legacy tour camera directions require the tour curve for migration."
        )
    camera_position = evaluate_catmull_rom_point(curve, progress)
    legacy_direction = _normalize_world_direction(
        raw["camera_look_direction"],
        "Legacy tour camera look direction",
    )
    return tuple(camera_position[axis] + legacy_direction[axis] for axis in range(3))


def _unique_legacy_speed_action_id(
    step_id: str,
    actions: tuple[TourActionData, ...],
) -> str:
    step_digest = sha256(step_id.encode("utf-8")).hexdigest()[:16]
    base_identifier = f"legacy-speed-override-{step_digest}"
    occupied = {action.component_id for action in actions}
    if base_identifier not in occupied:
        return base_identifier
    suffix = 2
    while f"{base_identifier}-{suffix}" in occupied:
        suffix += 1
    return f"{base_identifier}-{suffix}"


def _evaluate_forward_curve_direction(
    curve: TourCurveData,
    progress: float,
) -> WorldPoint:
    sample_distance = 1e-5
    camera_position = evaluate_catmull_rom_point(curve, progress)
    if progress < 1.0:
        nearby_progress = min(1.0, progress + sample_distance)
        nearby_position = evaluate_catmull_rom_point(curve, nearby_progress)
        tangent = _subtract_points(nearby_position, camera_position)
    else:
        nearby_progress = max(0.0, progress - sample_distance)
        nearby_position = evaluate_catmull_rom_point(curve, nearby_progress)
        tangent = _subtract_points(camera_position, nearby_position)
    if math.hypot(*tangent) > 0.0:
        return _normalize_world_direction(tangent, "Tour curve tangent")

    segment_count = len(curve.points) - 1
    segment_index = min(
        segment_count - 1,
        math.floor(progress * segment_count),
    )
    control_point_direction = _subtract_points(
        curve.points[segment_index + 1],
        curve.points[segment_index],
    )
    return _normalize_world_direction(
        control_point_direction,
        "Tour curve tangent",
    )


# ### Validation helpers ###
def _normalize_tours(tours: Iterable[TourData]) -> tuple[TourData, ...]:
    if isinstance(tours, (str, bytes, bytearray)):
        raise TypeError("Tours must contain TourData values.")
    normalized = tuple(tours)
    if not all(isinstance(value, TourData) for value in normalized):
        raise TypeError("Tours must contain TourData values.")
    _require_unique_identifiers(
        (tour.tour_id for tour in normalized),
        "Tour IDs",
    )
    return normalized


def _validate_singleton_step_actions(actions: tuple[TourActionData, ...]) -> None:
    singleton_types = (
        TourSpeedOverrideData,
        TourWaitForKeyPressData,
        TourIdleCameraAnimationData,
    )
    for action_type in singleton_types:
        if sum(isinstance(action, action_type) for action in actions) > 1:
            raise ValueError(
                f"A tour step can contain only one {action_type.__name__} action."
            )


def _normalize_identifier(value: object, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} IDs cannot be empty.")
    if len(normalized) > MAX_TOUR_NAME_LENGTH:
        raise ValueError(f"{label} ID is too long.")
    return normalized


def _normalize_finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a finite number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be a finite number.")
    return normalized


def _normalize_positive_multiplier(value: object, label: str) -> float:
    normalized = _normalize_finite_float(value, label)
    if normalized <= 0.0:
        raise ValueError(f"{label} must be greater than zero.")
    return normalized


def _normalize_action_easing(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} easing function must be a string.")
    easing = value.strip().lower()
    if easing not in TOUR_SPEED_EASING_OPTIONS:
        raise ValueError(f"Unsupported {label.lower()} easing function: {easing!r}.")
    return easing


def _normalize_world_point(value: object, label: str) -> WorldPoint:
    if (
        isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, (list, tuple))
        or len(value) != 3
    ):
        raise ValueError(f"{label} must contain exactly three coordinates.")
    return (
        _normalize_finite_float(value[0], label),
        _normalize_finite_float(value[1], label),
        _normalize_finite_float(value[2], label),
    )


def _normalize_planar_size(value: object, label: str) -> PlanarSize:
    if (
        isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, (list, tuple))
        or len(value) != 2
    ):
        raise ValueError(f"{label} must contain exactly two dimensions.")
    width = _normalize_positive_multiplier(value[0], label)
    depth = _normalize_positive_multiplier(value[1], label)
    return (width, depth)


def _normalize_world_direction(value: object, label: str) -> WorldPoint:
    direction = _normalize_world_point(value, label)
    length = math.hypot(*direction)
    if length <= 0.0:
        raise ValueError(f"{label} must be nonzero.")
    return tuple(coordinate / length for coordinate in direction)


def _require_mapping(payload: object, label: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError(f"{label} data must contain an object.")
    return payload


def _require_action_mapping(
    payload: object,
    expected_type: str,
) -> dict[str, object]:
    raw = _require_mapping(payload, "Tour action")
    action_type = str(raw.get("type", expected_type))
    if action_type != expected_type:
        raise ValueError(f"Unsupported tour action type: {action_type!r}.")
    return raw


def _require_sequence(payload: object, label: str) -> list[object] | tuple[object, ...]:
    if not isinstance(payload, (list, tuple)):
        raise TypeError(f"{label} data must contain a list.")
    return payload


def _require_unique_identifiers(
    identifiers: Iterable[str],
    label: str,
) -> None:
    occupied: set[str] = set()
    for identifier in identifiers:
        if identifier in occupied:
            raise ValueError(f"{label} must be unique.")
        occupied.add(identifier)


# ### Curve evaluation helpers ###
def _evaluate_catmull_rom_axis(
    before: float,
    first: float,
    second: float,
    after: float,
    progress: float,
    tension: float,
) -> float:
    first_tangent = (second - before) * tension
    second_tangent = (after - first) * tension
    progress_squared = progress * progress
    progress_cubed = progress_squared * progress
    return (
        (2.0 * first - 2.0 * second + first_tangent + second_tangent) * progress_cubed
        + (-3.0 * first + 3.0 * second - 2.0 * first_tangent - second_tangent)
        * progress_squared
        + first_tangent * progress
        + first
    )


def _scale_point(point: WorldPoint, scalar: float) -> WorldPoint:
    return tuple(float(coordinate) * scalar for coordinate in point)


def _subtract_points(first: WorldPoint, second: WorldPoint) -> WorldPoint:
    return tuple(float(first[axis]) - float(second[axis]) for axis in range(3))


# ### Coordinate conversion helpers ###
def _world_point_to_gltf(point: WorldPoint) -> list[float]:
    """Convert HouseMaker Z-up coordinates to glTF's exported Y-up space."""

    x_value, y_value, z_value = point
    values = (x_value, z_value, -y_value)
    return [
        0.0 if math.isclose(value, 0.0, abs_tol=1e-12) else float(value)
        for value in values
    ]


def _world_euler_degrees_to_gltf(rotation_degrees: WorldPoint) -> list[float]:
    """Convert a Z-up static-XYZ Euler orientation to glTF Y-up XYZ Euler.

    Euler components cannot be safely permuted when more than one axis is
    rotated. The orientation matrix is therefore conjugated by HouseMaker's
    world-to-glTF basis transform before being decomposed again.
    """

    x_radians, y_radians, z_radians = (
        math.radians(value) for value in rotation_degrees
    )
    rotation = _multiply_matrix3(
        _rotation_z_matrix(z_radians),
        _multiply_matrix3(
            _rotation_y_matrix(y_radians),
            _rotation_x_matrix(x_radians),
        ),
    )
    world_to_gltf = (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, -1.0, 0.0),
    )
    gltf_to_world = _transpose_matrix3(world_to_gltf)
    gltf_rotation = _multiply_matrix3(
        world_to_gltf,
        _multiply_matrix3(rotation, gltf_to_world),
    )
    return [
        _clean_angle_degrees(math.degrees(value))
        for value in _static_xyz_euler_from_matrix(gltf_rotation)
    ]


def _static_xyz_euler_from_matrix(
    matrix: tuple[tuple[float, float, float], ...],
) -> tuple[float, float, float]:
    """Decompose a rotation matrix into static XYZ Euler radians."""

    sine_y = max(-1.0, min(1.0, -matrix[2][0]))
    y_radians = math.asin(sine_y)
    cosine_y = math.cos(y_radians)
    if abs(cosine_y) > 1e-10:
        x_radians = math.atan2(matrix[2][1], matrix[2][2])
        z_radians = math.atan2(matrix[1][0], matrix[0][0])
    elif sine_y > 0.0:
        x_radians = math.atan2(matrix[0][1], matrix[0][2])
        z_radians = 0.0
    else:
        x_radians = math.atan2(-matrix[1][2], matrix[1][1])
        z_radians = 0.0
    return (x_radians, y_radians, z_radians)


def _rotation_x_matrix(
    angle_radians: float,
) -> tuple[tuple[float, float, float], ...]:
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    return ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))


def _rotation_y_matrix(
    angle_radians: float,
) -> tuple[tuple[float, float, float], ...]:
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    return ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine))


def _rotation_z_matrix(
    angle_radians: float,
) -> tuple[tuple[float, float, float], ...]:
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    return ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))


def _multiply_matrix3(
    first: tuple[tuple[float, float, float], ...],
    second: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(
            sum(first[row][inner] * second[inner][column] for inner in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def _transpose_matrix3(
    matrix: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))


def _clean_angle_degrees(value: float) -> float:
    """Return stable signed degrees without negative zero or full turns."""

    normalized = (float(value) + 180.0) % 360.0 - 180.0
    if math.isclose(normalized, -180.0, abs_tol=1e-10) and value > 0.0:
        normalized = 180.0
    return 0.0 if math.isclose(normalized, 0.0, abs_tol=1e-10) else normalized
