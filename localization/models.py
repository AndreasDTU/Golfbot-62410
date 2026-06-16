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
    """Live-tunable robot drawing and pickup geometry in centimeters."""

    width_cm: float
    front_cm: float
    rear_cm: float
    tube_forward_cm: float
    tube_right_cm: float
    unload_extension_cm: float = 15.0


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
