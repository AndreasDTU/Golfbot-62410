"""Drive control and safety guards for route tracking."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from pathfinding.models import HybridPose, RouteTrackingError
from pathfinding.planner import RoutePlanningFacade
from robot.models import DriveControlState, DriveRuntime, RobotPose, WheelCommand
from vision.config import DriveConfig


class WheelCommandController:
    """Translate route tracking error into bounded differential-drive speeds."""

    def __init__(self, drive_config: DriveConfig | None = None) -> None:
        self.config = drive_config or DriveConfig()

    def compute(self, error: RouteTrackingError) -> WheelCommand:
        """Compute the wheel command used by the legacy controller."""
        heading_error = float(
            np.clip(
                error.heading_error_rad,
                -self.config.max_heading_for_forward_rad,
                self.config.max_heading_for_forward_rad,
            )
        )
        forward_scale = max(0.0, 1.0 - abs(heading_error) / self.config.max_heading_for_forward_rad)
        base_speed = self.config.base_speed_pct * forward_scale
        turn_speed = self.config.heading_kp * heading_error - self.config.xte_kp * error.signed_xte_cm
        left = float(np.clip(base_speed - turn_speed, -self.config.max_speed_pct, self.config.max_speed_pct))
        right = float(np.clip(base_speed + turn_speed, -self.config.max_speed_pct, self.config.max_speed_pct))
        return WheelCommand(left, right)


class DriveSafetyGuard:
    """Apply deterministic route-tracking safety decisions before motor dispatch."""

    def __init__(
        self,
        drive_config: DriveConfig | None = None,
        route_facade: RoutePlanningFacade | None = None,
        wheel_controller: WheelCommandController | None = None,
    ) -> None:
        self.config = drive_config or DriveConfig()
        self.route_facade = route_facade or RoutePlanningFacade()
        self.wheel_controller = wheel_controller or WheelCommandController(self.config)

    def enforce_xte_guard_before_replan(
        self,
        robot_pose: RobotPose | None,
        route: list[HybridPose] | None,
        drive_runtime: DriveRuntime | None,
        clear_route_cache: Callable[[], None] | None = None,
    ) -> None:
        """Stop on excessive XTE before route cache updates can hide the error."""
        if (
            drive_runtime is None
            or robot_pose is None
            or not route
            or len(route) < 2
        ):
            return

        tracking_error = self.route_facade.compute_route_tracking_error(robot_pose, route)
        if tracking_error is None or tracking_error.xte_cm <= self.config.max_cross_track_error_cm:
            return

        drive_runtime.last_error = tracking_error
        drive_runtime.stop(
            DriveControlState.REPLANNING,
            f"XTE {tracking_error.xte_cm:.1f}cm > {self.config.max_cross_track_error_cm:.1f}cm",
        )
        drive_runtime.suppress_dispatch_this_frame = True
        if clear_route_cache is not None:
            clear_route_cache()

    def update_drive_control(
        self,
        robot_pose: RobotPose | None,
        route: list[HybridPose] | None,
        drive_runtime: DriveRuntime | None,
        clear_route_cache: Callable[[], None] | None = None,
        replan: Callable[[], None] | None = None,
    ) -> None:
        """Run the master-controller step after perception and route-cache update."""
        if drive_runtime is None:
            return
        if drive_runtime.suppress_dispatch_this_frame:
            drive_runtime.suppress_dispatch_this_frame = False
            return
        if not drive_runtime.enabled:
            drive_runtime.stop(DriveControlState.DISABLED, "dry run")
            return
        if robot_pose is None:
            if clear_route_cache is not None:
                clear_route_cache()
            drive_runtime.last_error = None
            drive_runtime.stop(DriveControlState.NO_POSE, "robot marker missing")
            return
        if not route or len(route) < 2:
            drive_runtime.last_error = None
            drive_runtime.stop(DriveControlState.NO_ROUTE, "waiting for route")
            return

        tracking_error = self.route_facade.compute_route_tracking_error(robot_pose, route)
        drive_runtime.last_error = tracking_error
        if tracking_error is None:
            drive_runtime.stop(DriveControlState.NO_ROUTE, "route has no usable segment")
            return

        if tracking_error.xte_cm > self.config.max_cross_track_error_cm:
            drive_runtime.stop(
                DriveControlState.REPLANNING,
                f"XTE {tracking_error.xte_cm:.1f}cm > {self.config.max_cross_track_error_cm:.1f}cm",
            )
            if clear_route_cache is not None:
                clear_route_cache()
            if replan is not None:
                replan()
            return

        command = self.wheel_controller.compute(tracking_error)
        drive_runtime.last_command = command
        if drive_runtime.dispatcher is None:
            drive_runtime.state = DriveControlState.DISABLED
            drive_runtime.last_message = "no dispatcher"
            return

        dispatched = drive_runtime.dispatcher.send_wheel_speeds(command.left_pct, command.right_pct)
        if dispatched:
            drive_runtime.state = DriveControlState.TRACKING
            drive_runtime.last_message = ""
        else:
            drive_runtime.state = DriveControlState.DISPATCH_ERROR
            drive_runtime.last_message = drive_runtime.dispatcher.last_error
            drive_runtime.last_command = WheelCommand(0.0, 0.0)
