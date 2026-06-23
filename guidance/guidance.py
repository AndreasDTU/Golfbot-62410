"""Stage 2 Guidance layer — per-frame waypoint follower.

Computes the geometry: given a route and the live pose, produces
turn / drive_adjusted commands each tick. Uses RobotCommander exclusively;
no direct wheel-speed or legacy steer() calls.

Architecture: non-blocking ticking. Guidance runs every frame, reads the
live pose, and calls one Control command per tick. Status flows upward as
RUNNING / ARRIVED / NO_POSE / NO_ROUTE / ERROR.
"""

from __future__ import annotations

import math
from enum import Enum

from config import DriveConfig
from control.commander import RobotCommander
from control.telemetry import log_event
from guidance.route_tracking import compute_route_tracking_error
from localization.localization import normalize_angle
from localization.models import RobotPose
from path.models import HybridPose, SafePickupZone
from config import DriveConfig
from path.pathfinding.models import HybridPose


class GuidanceStatus(str, Enum):
    """Status returned by every guidance tick."""

    RUNNING = "RUNNING"
    ARRIVED = "ARRIVED"
    NO_POSE = "NO_POSE"
    NO_ROUTE = "NO_ROUTE"
    ERROR = "ERROR"
    OFF_PATH = "OFF_PATH"


class GuidanceController:
    """Per-frame waypoint follower.

    Each call to ``tick()`` reads the live pose, decides whether to
    ``turn``, ``drive_adjusted``, or ``stop``, and returns a
    ``GuidanceStatus`` for the layer above.

    Sign / unit conventions (matching Control layer):
        * Heading: radians, 0 = +X, positive = CCW.  Normalized to [-pi, pi).
        * ``commander.turn()`` takes **degrees**.
        * ``commander.drive_adjusted()`` takes **cm** and **degrees**.
        * Guidance converts at the call site with ``math.degrees()``.
    """

    def __init__(
        self,
        commander: RobotCommander,
        config: DriveConfig | None = None,
    ) -> None:
        self._commander = commander
        self._config = config or DriveConfig()
        self._waypoint_arrival_cm = self._config.waypoint_arrival_cm
        self._ball_arrival_cm = self._config.ball_arrival_cm
        self._final_heading_tolerance_rad = self._config.final_heading_tolerance_rad

        self._waypoints: list[HybridPose] = []
        self._cursor: int = 0
        self._route_complete: bool = False
        self._aligning: bool = False
        # SAFE acceptance region for the final pickup, or None to home to the
        # exact final waypoint (constrained pickup / navigation / unload).
        self._pickup_zone: SafePickupZone | None = None

        self._driving: bool = False

    # ------------------------------------------------------------------
    # Route management
    # ------------------------------------------------------------------

    def set_route(
        self,
        waypoints: list[HybridPose],
        pickup_zone: SafePickupZone | None = None,
    ) -> None:
        """Load a new route and reset the cursor to the first waypoint.

        *pickup_zone*, when given, lets the route finish at *any* SAFE point on
        the ball's reach circle: the robot accepts wherever it lands in the
        zone (within the standoff tolerance) and turns to face the ball,
        instead of homing to the exact final waypoint.  Arrival and heading
        tolerances are the same as for an ordinary pickup -- only *which* point
        on the circle is free.
        """
        self._waypoints = list(waypoints)
        self._cursor = 0
        self._route_complete = False
        self._aligning = False
        self._pickup_zone = pickup_zone
        log_event("GUIDANCE", "route set", waypoints=len(self._waypoints))

    def clear_route(self) -> None:
        """Clear the active route and stop the robot."""
        self._waypoints = []
        self._cursor = 0
        self._route_complete = False
        self._aligning = False
        self._pickup_zone = None
        self._driving = False
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

        # 3. Final-waypoint heading alignment phase.
        if self._aligning:
            return self._tick_align(pose)

        # 4. Compute geometry to current target waypoint.
        target = self._waypoints[self._cursor]
        distance, heading_error = self._compute_geometry(pose, target)

        # 5. Check arrival at current waypoint (before driving init or turn check).
        is_last = self._cursor >= len(self._waypoints) - 1
        if is_last:
            if self._pickup_arrived(pose, distance):
                # In position (exact point, or anywhere on the SAFE arc) —
                # enter heading alignment / re-aim phase.
                self._aligning = True
                self._log("ALIGNING", dist=distance)
                return self._tick_align(pose)
        elif distance < self._waypoint_arrival_cm:
            # Intermediate waypoint — advance cursor.
            self._cursor += 1
            target = self._waypoints[self._cursor]
            distance, _ = self._compute_geometry(pose, target)
            self._driving = self._commander.start_drive(distance)
            self._log("DRIVING", dist=distance, ok=self._driving)
            return GuidanceStatus.RUNNING if self._driving else GuidanceStatus.ERROR

        # 6. Cross-track error guard — replan if too far off path.
        off_path = self._check_xte(pose)
        if off_path is not None:
            return off_path

        # 7. Target behind and close — reverse instead of turning.
        if abs(heading_error) > math.radians(150) and distance < 25.0:
            reverse_error = normalize_angle(heading_error - math.pi)
            ok = self._commander.drive_adjusted(
                -distance, -math.degrees(reverse_error),
            )
            self._log(
                "REVERSING", dist=distance,
                heading_err=math.degrees(heading_error), ok=ok,
            )
            return GuidanceStatus.RUNNING if ok else GuidanceStatus.ERROR

        max_heading_error = self._config.max_heading_for_tank_rad if self._commander._current_speed == 0 else self._config.max_heading_for_forward_rad
        # 8. Large heading error — rotate in place.
        if abs(heading_error) > max_heading_error:
            ok = self._commander.turn(math.degrees(heading_error))
            self._log(
                "TURNING",
                dist=distance,
                heading_err=math.degrees(heading_error),
                ok=ok,
            )
            return GuidanceStatus.RUNNING if ok else GuidanceStatus.ERROR

        # 9. Initialize driving or continue with heading correction.
        if not self._driving:
            self._driving = self._commander.start_drive(
                distance, math.degrees(heading_error)
            )
            self._log("DRIVING", dist=distance, ok=self._driving)
            return GuidanceStatus.RUNNING if self._driving else GuidanceStatus.ERROR

        # 10. Drive forward with arc correction (single combined LR command).
        ok = self._commander.drive_adjusted(distance, math.degrees(heading_error))
        self._log(
            "DRIVING ADJUSTED",
            dist=distance,
            heading_err=math.degrees(heading_error),
            ok=ok,
        )
        return GuidanceStatus.RUNNING if ok else GuidanceStatus.ERROR

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pickup_arrived(self, pose: RobotPose, distance: float) -> bool:
        """Whether the robot has reached the final pickup.

        With a SAFE pickup zone, arrival is *anywhere* on the ball's SAFE arc
        at reach distance -- the robot accepts where it landed instead of
        homing to one exact spot.  Reaching the exact planned point still
        counts (fallback), so behaviour is never worse than before.
        """
        if self._pickup_zone is not None and self._pickup_zone.accepts(
            pose.x_cm, pose.y_cm, self._ball_arrival_cm,
        ):
            return True
        return distance < self._ball_arrival_cm

    def _tick_align(self, pose: RobotPose) -> GuidanceStatus:
        """Rotate in place to face the ball, then finish.

        For a SAFE pickup zone the target heading is recomputed each tick as
        "turn towards the ball from wherever I landed"; otherwise it is the
        fixed final-waypoint heading.  The heading tolerance is the same tight
        default in both cases -- only *which* point on the circle is free.
        """
        if self._pickup_zone is not None:
            target_theta = self._pickup_zone.grab_heading(pose.x_cm, pose.y_cm)
        else:
            target_theta = self._waypoints[self._cursor].theta_rad
        heading_error = normalize_angle(target_theta - pose.heading_rad)

        tolerance = self._final_heading_tolerance_rad

        if abs(heading_error) <= tolerance:
            self._commander.stop()
            self._route_complete = True
            self._log("ARRIVED", heading_err=math.degrees(heading_error))
            return GuidanceStatus.ARRIVED

        ok = self._commander.turn(math.degrees(heading_error))
        self._log(
            "ALIGN_TURN",
            heading_err=math.degrees(heading_error),
            ok=ok,
        )
        return GuidanceStatus.RUNNING if ok else GuidanceStatus.ERROR

    def _check_xte(self, pose: RobotPose) -> GuidanceStatus | None:
        """Return OFF_PATH if the robot is too far from the current route.

        Computes the Euclidean distance to the nearest point on the
        current and future path segments.  If it exceeds
        ``max_cross_track_error_cm`` the robot is stopped and OFF_PATH
        is returned so the layer above can trigger a replan.

        Returns None when the robot is within tolerance.
        """
        tracking = compute_route_tracking_error(
            pose,
            self._waypoints,
            start_segment_index=max(0, self._cursor - 1),
        )
        if tracking is None:
            return None
        if tracking.xte_cm <= self._config.max_cross_track_error_cm:
            return None
        self._commander.stop()
        self._driving = False
        self._log(
            "OFF_PATH",
            xte_cm=f"{tracking.xte_cm:.1f}",
            threshold=f"{self._config.max_cross_track_error_cm:.1f}",
        )
        return GuidanceStatus.OFF_PATH

    @staticmethod
    def _compute_geometry(
        pose: RobotPose,
        target: HybridPose,
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
