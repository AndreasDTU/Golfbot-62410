"""Shared robot-domain data models for the detector stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from perception.vision.models import ParallaxConfig

ROBOT_MARKER_IDS = (4, 5)


class RobotCalibrationPhase(Enum):
    """Non-blocking robot origin calibration state."""

    STATE_NORMAL = "normal"
    STATE_CALIBRATING_SPIN = "calibrating_spin"
    STATE_CALIBRATING_FORWARD = "calibrating_forward"


class DriveControlState(Enum):
    """Master-controller dispatch state shown in the detector overlay."""

    DISABLED = "DISABLED"
    NO_POSE = "NO POSE"
    NO_ROUTE = "NO ROUTE"
    TRACKING = "TRACKING"
    PRECISE_MOVE = "PRECISE MOVE"
    PRE_UNLOAD_PIVOT = "PRE-UNLOAD PIVOT"
    POST_PICKUP_ESCAPE = "POST-PICKUP ESCAPE"
    POST_PICKUP_ALIGN = "POST-PICKUP ALIGN"
    CALIBRATING = "CALIBRATING"
    BLIND_APPROACH = "BLIND APPROACH"
    PICKUP = "PICKUP"
    REPLANNING = "REPLANNING -> MOTORS HALTED"
    STOPPED = "STOPPED"
    TANK_TURN = "TANK TURN"
    DISPATCH_ERROR = "DISPATCH ERROR"


@dataclass(frozen=True)
class RobotPose:
    """Robot origin and pickup tube pose in field coordinates with bottom-left cm origin."""

    x_cm: float
    y_cm: float
    heading_rad: float
    tube_x_cm: float
    tube_y_cm: float


@dataclass(frozen=True)
class RobotGeometry:
    """Live-tunable robot drawing and pickup geometry in centimeters.

    All distances are measured from the robot origin, which sits between the
    drive wheels (differential-drive axle center).

    Physical layout (heading = up):

                 ○          ← tube tip (4.5 cm diameter pipe)
                 │
                 │  tube_forward_cm (13.1, to pipe center)
                 │
        ┌────────┼────────┐  ← front_cm (3.8) from origin
        │        │        │
        │       (O)       │  ← origin (wheel axle center)
        │                 │
        │                 │
        │                 │  rear_cm (15.1) from origin
        └─────────────────┘
        ←    width_cm    →
              (19.5)

    The pickup tube is NOT retractable — it protrudes forward at all times
    (both raised and lowered) and must be included in collision geometry.

    During an in-place (tank) turn the full robot (body + tube) sweeps a
    circle.  The radius is determined by the farthest point from origin:

        rear corner:  sqrt(rear_cm² + (width_cm / 2)²)           ≈ 18.0 cm
        tube tip edge: sqrt(tube_forward_cm² + (pipe_radius)²)   ≈ 13.3 cm

        tank_turn_radius ≈ 18.0 cm  (dominated by rear corners)

    Ground-truth values live in data/robot_calibration.json and are loaded at
    startup; the defaults in config.py RobotGeometryConfig are fallbacks only.
    """

    width_cm: float  # full body width (left wheel edge to right wheel edge)
    front_cm: float  # origin to front edge of body
    rear_cm: float  # origin to rear edge of body
    tube_forward_cm: float  # origin to pickup tube tip (along heading)
    tube_right_cm: float  # lateral offset of tube tip (0 = centered)
    tube_width_cm: float = 6.0  # pickup tube opening diameter
    mouth_radius_cm: float = 2.0  # tolerance radius around tube tip
    unload_extension_cm: float = 30.0  # reverse distance when unloading
    pipe_diameter_cm: float = 4.5  # physical pickup pipe outer diameter


@dataclass(frozen=True)
class WheelCommand:
    """Bounded differential-drive wheel command in speed percent units."""

    left_pct: float
    right_pct: float


@dataclass
class DriveRuntime:
    """Live master-controller state shared by control, dispatch, and overlays."""

    enabled: bool
    commander: object | None = None
    state: DriveControlState = DriveControlState.DISABLED
    last_error: object | None = None
    last_command: WheelCommand = field(default_factory=lambda: WheelCommand(0.0, 0.0))
    last_message: str = ""
    suppress_dispatch_this_frame: bool = False
    active_route_identity: int | None = None
    route_progress_segment_index: int = 0

    def stop(self, state: DriveControlState, message: str = "") -> None:
        """Send a deterministic zero-speed command and update overlay state."""
        previous_command = self.last_command
        previous_state = self.state
        self.state = state
        self.last_message = message
        self.last_command = WheelCommand(0.0, 0.0)
        if self.commander is not None:
            force = (
                previous_state != state
                or abs(previous_command.left_pct) > 1e-6
                or abs(previous_command.right_pct) > 1e-6
            )
            self.commander.stop(force=force)


@dataclass(frozen=True)
class RobotMarkerObservation:
    """Detected robot marker after strict parallax projection to the ground plane."""

    marker_id: int
    center: np.ndarray
    ground_center: np.ndarray
    corners: np.ndarray
    ground_corners: np.ndarray
    yaw_rad: float


@dataclass
class RobotCalibrationRuntime:
    """Mutable robot calibration state used by the live detector loop."""

    phase: RobotCalibrationPhase = RobotCalibrationPhase.STATE_NORMAL
    calibration: dict[str, Any] | None = None
    collected_points: dict[int, list[tuple[float, float]]] = field(
        default_factory=lambda: {marker_id: [] for marker_id in ROBOT_MARKER_IDS}
    )
    fitted_centers: dict[int, tuple[float, float]] = field(default_factory=dict)
    ellipse_ratios: dict[int, float] = field(default_factory=dict)
    latest_observations: dict[int, RobotMarkerObservation] = field(default_factory=dict)
    latest_parallax_config: ParallaxConfig | None = None
    warning: str = ""
