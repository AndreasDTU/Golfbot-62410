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
        self.previous_cross_track_error: float | None = None
        self.previous_heading_error: float | None = None
        self.previous_profile_speed_pct: float = 0.0

    def reset(self) -> None:
        """Clear derivative history after stops, replans, or route loss."""
        self.previous_cross_track_error = None
        self.previous_heading_error = None
        self.previous_profile_speed_pct = 0.0

    @staticmethod
    def target_speed_for_distance(distance_to_goal_cm: float, config: DriveConfig | None = None) -> float:
        """Return the profiled forward speed target for a goal distance."""
        drive_config = config or DriveConfig()
        if not math.isfinite(distance_to_goal_cm):
            return drive_config.base_speed_pct
        distance = max(0.0, float(distance_to_goal_cm))
        if distance <= drive_config.creep_distance_cm:
            return drive_config.creep_speed_pct
        if distance >= drive_config.cruise_distance_cm:
            return drive_config.base_speed_pct

        span = max(1e-6, drive_config.cruise_distance_cm - drive_config.creep_distance_cm)
        ratio = (distance - drive_config.creep_distance_cm) / span
        return drive_config.creep_speed_pct + ratio * (drive_config.base_speed_pct - drive_config.creep_speed_pct)

    def slew_limited_speed(self, desired_speed_pct: float, dt_s: float | None) -> float:
        """Apply deterministic acceleration/deceleration limits to forward speed."""
        desired = float(np.clip(desired_speed_pct, 0.0, self.config.max_speed_pct))
        if dt_s is None or dt_s <= 1e-6:
            self.previous_profile_speed_pct = desired
            return desired

        previous = self.previous_profile_speed_pct
        accel_step = max(0.0, self.config.acceleration_limit_pct_per_s) * dt_s
        decel_step = max(0.0, self.config.deceleration_limit_pct_per_s) * dt_s
        limited = float(np.clip(desired, previous - decel_step, previous + accel_step))
        self.previous_profile_speed_pct = limited
        return limited

    def compute(
        self,
        error: RouteTrackingError,
        distance_to_goal_cm: float = float("inf"),
        dt_s: float | None = None,
    ) -> WheelCommand:
        """Compute a profiled PD wheel command for route tracking."""
        heading_error = float(
            np.clip(
                error.heading_error_rad,
                -self.config.max_heading_for_forward_rad,
                self.config.max_heading_for_forward_rad,
            )
        )
        forward_scale = max(0.0, 1.0 - abs(heading_error) / self.config.max_heading_for_forward_rad)
        desired_base_speed = self.target_speed_for_distance(distance_to_goal_cm, self.config) * forward_scale
        base_speed = self.slew_limited_speed(desired_base_speed, dt_s)

        heading_derivative = 0.0
        cross_track_derivative = 0.0
        if dt_s is not None and dt_s > 1e-6:
            if self.previous_heading_error is not None:
                heading_derivative = (heading_error - self.previous_heading_error) / dt_s
            if self.previous_cross_track_error is not None:
                cross_track_derivative = (error.signed_xte_cm - self.previous_cross_track_error) / dt_s

        self.previous_heading_error = heading_error
        self.previous_cross_track_error = error.signed_xte_cm

        turn_speed = (
            self.config.heading_kp * heading_error
            + self.config.heading_kd * heading_derivative
            - self.config.xte_kp * error.signed_xte_cm
            - self.config.xte_kd * cross_track_derivative
        )
        left = float(np.clip(base_speed - turn_speed, -self.config.max_speed_pct, self.config.max_speed_pct))
        right = float(np.clip(base_speed + turn_speed, -self.config.max_speed_pct, self.config.max_speed_pct))
        return WheelCommand(left, right)


def closest_route_index(route: list[HybridPose], pose: HybridPose) -> int:
    """Return the route sample closest to a planner pose."""
    if not route:
        return -1
    return min(
        range(len(route)),
        key=lambda index: math.hypot(route[index].x_cm - pose.x_cm, route[index].y_cm - pose.y_cm),
    )


def route_checkpoint_indices(
    route: list[HybridPose],
    local_goal_poses: list[HybridPose] | None = None,
    *,
    include_final: bool = True,
) -> list[int]:
    """Map pickup/unload goals onto sorted route indices."""
    if not route:
        return []
    checkpoints = {
        index
        for pose in (local_goal_poses or [])
        for index in [closest_route_index(route, pose)]
        if index >= 0
    }
    if include_final:
        checkpoints.add(len(route) - 1)
    return sorted(checkpoints)


def next_route_checkpoint_index(
    route: list[HybridPose],
    tracking_error: RouteTrackingError | None = None,
    local_goal_poses: list[HybridPose] | None = None,
    *,
    include_final: bool = True,
) -> int:
    """Return the next pickup/unload checkpoint ahead of the current route segment."""
    checkpoints = route_checkpoint_indices(route, local_goal_poses, include_final=include_final)
    if not checkpoints:
        return -1
    next_route_index = 0 if tracking_error is None else min(len(route) - 1, tracking_error.segment_index + 1)
    for checkpoint in checkpoints:
        if checkpoint >= next_route_index:
            return checkpoint
    return checkpoints[-1]


def distance_to_route_goal_cm(
    robot_pose: RobotPose,
    route: list[HybridPose],
    tracking_error: RouteTrackingError | None = None,
    local_goal_poses: list[HybridPose] | None = None,
    *,
    include_final: bool = True,
) -> float:
    """Distance from the live robot origin to the next local pickup/unload checkpoint."""
    goal_index = next_route_checkpoint_index(
        route,
        tracking_error,
        local_goal_poses,
        include_final=include_final,
    )
    if goal_index < 0:
        return float("inf")
    goal = route[goal_index]
    return math.hypot(float(robot_pose.x_cm) - float(goal.x_cm), float(robot_pose.y_cm) - float(goal.y_cm))


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
        local_goal_poses: list[HybridPose] | None = None,
        dt_s: float | None = None,
    ) -> None:
        """Run the master-controller step after perception and route-cache update."""
        if drive_runtime is None:
            return
        if drive_runtime.suppress_dispatch_this_frame:
            drive_runtime.suppress_dispatch_this_frame = False
            return
        if not drive_runtime.enabled:
            self.wheel_controller.reset()
            drive_runtime.stop(DriveControlState.DISABLED, "dry run")
            return
        if robot_pose is None:
            self.wheel_controller.reset()
            if clear_route_cache is not None:
                clear_route_cache()
            drive_runtime.last_error = None
            drive_runtime.stop(DriveControlState.NO_POSE, "robot marker missing")
            return
        if not route or len(route) < 2:
            self.wheel_controller.reset()
            drive_runtime.last_error = None
            drive_runtime.stop(DriveControlState.NO_ROUTE, "waiting for route")
            return

        tracking_error = self.route_facade.compute_route_tracking_error(robot_pose, route)
        drive_runtime.last_error = tracking_error
        if tracking_error is None:
            self.wheel_controller.reset()
            drive_runtime.stop(DriveControlState.NO_ROUTE, "route has no usable segment")
            return

        if tracking_error.xte_cm > self.config.max_cross_track_error_cm:
            self.wheel_controller.reset()
            drive_runtime.stop(
                DriveControlState.REPLANNING,
                f"XTE {tracking_error.xte_cm:.1f}cm > {self.config.max_cross_track_error_cm:.1f}cm",
            )
            if clear_route_cache is not None:
                clear_route_cache()
            if replan is not None:
                replan()
            return

        command = self.wheel_controller.compute(
            tracking_error,
            distance_to_goal_cm=distance_to_route_goal_cm(
                robot_pose,
                route,
                tracking_error,
                local_goal_poses,
            ),
            dt_s=dt_s,
        )
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
