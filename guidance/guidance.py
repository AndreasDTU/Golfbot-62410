"""Stage 2 Guidance layer — per-frame waypoint follower.

Computes the geometry: given a route and the live pose, produces
turn / drive / adjust commands each tick. Uses RobotCommander exclusively;
no direct wheel-speed or legacy steer() calls.

Architecture: non-blocking ticking. Guidance runs every frame, reads the
live pose, and calls one Control command per tick. Status flows upward as
RUNNING / ARRIVED / NO_POSE / NO_ROUTE / ERROR.
"""

from __future__ import annotations

import math
from enum import Enum

from control.commander import RobotCommander
from control.telemetry import log_event
from localization.localization import normalize_angle
from localization.models import RobotPose
from path.pathfinding.models import HybridPose
from perception.vision.config import DriveConfig


class GuidanceStatus(str, Enum):
    """Status returned by every guidance tick."""

    RUNNING = "RUNNING"
    ARRIVED = "ARRIVED"
    NO_POSE = "NO_POSE"
    NO_ROUTE = "NO_ROUTE"
    ERROR = "ERROR"


# Default waypoint arrival tolerance in cm (matches PlannerConfig.goal_tolerance_cm).
DEFAULT_WAYPOINT_ARRIVAL_CM = 4.0


class GuidanceController:
    """Per-frame waypoint follower.

    Each call to ``tick()`` reads the live pose, decides whether to
    ``turn``, ``drive`` + ``adjust``, or ``stop``, and returns a
    ``GuidanceStatus`` for the layer above.

    Sign / unit conventions (matching Control layer):
        * Heading: radians, 0 = +X, positive = CCW.  Normalized to [-pi, pi).
        * ``commander.turn()`` and ``commander.adjust()`` take **degrees**.
        * ``commander.drive()`` takes **cm**.
        * Guidance converts at the call site with ``math.degrees()``.
    """

    def __init__(
        self,
        commander: RobotCommander,
        config: DriveConfig | None = None,
        waypoint_arrival_cm: float = DEFAULT_WAYPOINT_ARRIVAL_CM,
    ) -> None:
        self._commander = commander
        self._config = config or DriveConfig()
        self._waypoint_arrival_cm = waypoint_arrival_cm

        self._waypoints: list[HybridPose] = []
        self._cursor: int = 0
        self._route_complete: bool = False

    # ------------------------------------------------------------------
    # Route management
    # ------------------------------------------------------------------

    def set_route(self, waypoints: list[HybridPose]) -> None:
        """Load a new route and reset the cursor to the first waypoint."""
        self._waypoints = list(waypoints)
        self._cursor = 0
        self._route_complete = False
        log_event("GUIDANCE", "route set", waypoints=len(self._waypoints))

    def clear_route(self) -> None:
        """Clear the active route and stop the robot."""
        self._waypoints = []
        self._cursor = 0
        self._route_complete = False
        self._commander.stop()
        log_event("GUIDANCE", "route cleared")

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def waypoint_count(self) -> int:
        return len(self._waypoints)

    # ------------------------------------------------------------------
    # Per-frame tick
    # ------------------------------------------------------------------

    def tick(self, pose: RobotPose | None, dt_s: float) -> GuidanceStatus:
        """Run one guidance frame.

        Parameters
        ----------
        pose : RobotPose or None
            Current robot pose from localization, or None if vision lost.
        dt_s : float
            Time elapsed since the previous tick (seconds).  Passed to
            ``commander.drive()`` for slew-rate limiting.

        Returns
        -------
        GuidanceStatus
            RUNNING while pursuing waypoints, ARRIVED when done,
            NO_POSE / NO_ROUTE / ERROR for exceptional states.
        """
        # 1. No pose — stop immediately.
        if pose is None:
            self._commander.stop()
            self._log("NO_POSE", cmd="stop")
            return GuidanceStatus.NO_POSE

        # 2. No route loaded or already complete.
        if not self._waypoints:
            return GuidanceStatus.NO_ROUTE
        if self._route_complete:
            return GuidanceStatus.ARRIVED

        # 3. Compute geometry to current target waypoint.
        target = self._waypoints[self._cursor]
        distance, heading_error = self._compute_geometry(pose, target)

        # 4. Check arrival at current waypoint.
        if distance < self._waypoint_arrival_cm:
            self._cursor += 1
            if self._cursor >= len(self._waypoints):
                self._commander.stop()
                self._route_complete = True
                self._log("ARRIVED", dist=distance)
                return GuidanceStatus.ARRIVED
            # Recalculate for the next waypoint.
            target = self._waypoints[self._cursor]
            distance, heading_error = self._compute_geometry(pose, target)

        # 5. Large heading error — rotate in place.
        if abs(heading_error) > self._config.max_heading_for_forward_rad:
            ok = self._commander.turn(math.degrees(heading_error))
            self._log(
                "TURNING", dist=distance,
                heading_err=math.degrees(heading_error), ok=ok,
            )
            return GuidanceStatus.RUNNING if ok else GuidanceStatus.ERROR

        # 6. Drive forward with arc correction.
        ok_drive = self._commander.drive(distance, dt_s)
        ok_adjust = self._commander.adjust(math.degrees(heading_error))
        ok = ok_drive and ok_adjust
        self._log(
            "DRIVING", dist=distance,
            heading_err=math.degrees(heading_error), ok=ok,
        )
        return GuidanceStatus.RUNNING if ok else GuidanceStatus.ERROR

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_geometry(
        pose: RobotPose, target: HybridPose,
    ) -> tuple[float, float]:
        """Return (distance_cm, heading_error_rad) from *pose* to *target*."""
        dx = target.x_cm - pose.x_cm
        dy = target.y_cm - pose.y_cm
        distance = math.hypot(dx, dy)
        heading_to_target = math.atan2(dy, dx)
        heading_error = normalize_angle(heading_to_target - pose.heading_rad)
        return distance, heading_error

    def _log(self, action: str, **kwargs) -> None:
        wp = f"{self._cursor}/{len(self._waypoints)}"
        log_event("GUIDANCE", action, wp=wp, **kwargs)
