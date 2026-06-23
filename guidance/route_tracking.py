"""Drive control helpers for route tracking."""

from __future__ import annotations

import math

import numpy as np

from path.models import HybridPose, RouteTrackingError
from localization.models import RobotPose, WheelCommand
from config import DriveConfig


def _normalize_angle(theta_rad: float) -> float:
    """Normalize to [-pi, pi)."""
    return (theta_rad + math.pi) % (2.0 * math.pi) - math.pi


def compute_route_tracking_error(
    robot_pose: RobotPose,
    route: list[HybridPose],
    start_segment_index: int = 0,
    end_segment_index: int | None = None,
) -> RouteTrackingError | None:
    """Project live robot pose onto the closest cached route segment."""
    if len(route) < 2:
        return None

    rx = float(robot_pose.x_cm)
    ry = float(robot_pose.y_cm)
    best: RouteTrackingError | None = None
    best_distance = float("inf")

    first_segment = max(0, min(int(start_segment_index), len(route) - 2))
    last_segment = len(route) - 2 if end_segment_index is None else int(end_segment_index)
    last_segment = max(first_segment, min(last_segment, len(route) - 2))

    for index in range(first_segment, last_segment + 1):
        start = route[index]
        end = route[index + 1]
        sx, sy = float(start.x_cm), float(start.y_cm)
        vx = float(end.x_cm - start.x_cm)
        vy = float(end.y_cm - start.y_cm)
        segment_len_sq = vx * vx + vy * vy
        if segment_len_sq <= 1e-9:
            continue

        projection = ((rx - sx) * vx + (ry - sy) * vy) / segment_len_sq
        clamped = float(np.clip(projection, 0.0, 1.0))
        cx = sx + vx * clamped
        cy = sy + vy * clamped
        dx = rx - cx
        dy = ry - cy
        distance = math.hypot(dx, dy)
        if distance >= best_distance:
            continue

        segment_heading = math.atan2(vy, vx)
        cross = vx * (ry - sy) - vy * (rx - sx)
        signed_distance = math.copysign(distance, cross) if abs(cross) > 1e-9 else 0.0
        best_distance = distance
        best = RouteTrackingError(
            xte_cm=distance,
            signed_xte_cm=signed_distance,
            heading_error_rad=_normalize_angle(segment_heading - robot_pose.heading_rad),
            closest_point_cm=(cx, cy),
            segment_heading_rad=segment_heading,
            segment_index=index,
        )

    return best


class WheelCommandController:
    """Translate route tracking error into bounded differential-drive speeds."""

    def __init__(self, drive_config: DriveConfig | None = None) -> None:
        self.config = drive_config or DriveConfig()
        self.previous_cross_track_error: float | None = None
        self.previous_heading_error: float | None = None
        self.previous_profile_speed_pct: float = 0.0
        self.last_forward_scale: float = 0.0
        self.last_desired_base_speed: float = 0.0
        self.last_base_speed: float = 0.0
        self.last_edge_gain: float = 1.0
        self.last_heading_derivative: float = 0.0
        self.last_cross_track_derivative: float = 0.0
        self.last_turn_speed: float = 0.0
        self.last_heading_p_term: float = 0.0
        self.last_heading_d_term: float = 0.0
        self.last_xte_p_term: float = 0.0
        self.last_xte_d_term: float = 0.0
        self.last_distance_to_goal_cm: float = float("inf")

    def reset(self) -> None:
        """Clear derivative history after stops, replans, or route loss."""
        self.previous_cross_track_error = None
        self.previous_heading_error = None
        self.previous_profile_speed_pct = 0.0
        self.last_forward_scale = 0.0
        self.last_desired_base_speed = 0.0
        self.last_base_speed = 0.0
        self.last_edge_gain = 1.0
        self.last_heading_derivative = 0.0
        self.last_cross_track_derivative = 0.0
        self.last_turn_speed = 0.0
        self.last_heading_p_term = 0.0
        self.last_heading_d_term = 0.0
        self.last_xte_p_term = 0.0
        self.last_xte_d_term = 0.0
        self.last_distance_to_goal_cm = float("inf")

    @staticmethod
    def edge_control_scale(edge_clearance_cm: float | None, config: DriveConfig | None = None) -> float:
        """Return 0..1 edge proximity scale where 1 means normal open-field control."""
        drive_config = config or DriveConfig()
        if edge_clearance_cm is None or not math.isfinite(edge_clearance_cm):
            return 1.0
        if drive_config.edge_slowdown_cm <= 1e-6:
            return 1.0
        return float(np.clip(edge_clearance_cm / drive_config.edge_slowdown_cm, 0.0, 1.0))

    @staticmethod
    def edge_speed_multiplier(edge_clearance_cm: float | None, config: DriveConfig | None = None) -> float:
        """Scale forward speed down as the robot body approaches a wall."""
        drive_config = config or DriveConfig()
        scale = WheelCommandController.edge_control_scale(edge_clearance_cm, drive_config)
        return drive_config.edge_min_speed_scale + scale * (1.0 - drive_config.edge_min_speed_scale)

    @staticmethod
    def edge_gain_multiplier(edge_clearance_cm: float | None, config: DriveConfig | None = None) -> float:
        """Scale proportional correction up as the robot body approaches a wall."""
        drive_config = config or DriveConfig()
        scale = WheelCommandController.edge_control_scale(edge_clearance_cm, drive_config)
        return 1.0 + (1.0 - scale) * (drive_config.edge_max_gain_scale - 1.0)

    @staticmethod
    def target_speed_for_distance(
        distance_to_goal_cm: float,
        config: DriveConfig | None = None,
        edge_clearance_cm: float | None = None,
    ) -> float:
        """Return the profiled forward speed target for a goal distance."""
        drive_config = config or DriveConfig()
        if not math.isfinite(distance_to_goal_cm):
            return drive_config.drive_speed_pct * WheelCommandController.edge_speed_multiplier(edge_clearance_cm, drive_config)
        distance = max(0.0, float(distance_to_goal_cm))
        if distance <= drive_config.creep_distance_cm:
            return drive_config.creep_speed_pct
        if distance >= drive_config.cruise_distance_cm:
            return drive_config.drive_speed_pct * WheelCommandController.edge_speed_multiplier(edge_clearance_cm, drive_config)

        span = max(1e-6, drive_config.cruise_distance_cm - drive_config.creep_distance_cm)
        ratio = (distance - drive_config.creep_distance_cm) / span
        edge_base_speed = drive_config.drive_speed_pct * WheelCommandController.edge_speed_multiplier(edge_clearance_cm, drive_config)
        return drive_config.creep_speed_pct + ratio * (edge_base_speed - drive_config.creep_speed_pct)

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
        edge_clearance_cm: float | None = None,
    ) -> WheelCommand:
        """Compute a profiled PD wheel command for route tracking."""
        self.last_distance_to_goal_cm = distance_to_goal_cm
        heading_error = float(
            np.clip(
                error.heading_error_rad,
                -self.config.max_heading_for_forward_rad,
                self.config.max_heading_for_forward_rad,
            )
        )
        forward_scale = max(0.0, 1.0 - abs(heading_error) / self.config.max_heading_for_forward_rad)
        desired_base_speed = self.target_speed_for_distance(
            distance_to_goal_cm,
            self.config,
            edge_clearance_cm=edge_clearance_cm,
        ) * forward_scale
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

        edge_gain = self.edge_gain_multiplier(edge_clearance_cm, self.config)
        heading_p = self.config.heading_kp * edge_gain * heading_error
        heading_d = self.config.heading_kd * heading_derivative
        xte_p = self.config.xte_kp * edge_gain * error.signed_xte_cm
        xte_d = self.config.xte_kd * cross_track_derivative
        turn_speed = heading_p + heading_d - xte_p - xte_d

        self.last_forward_scale = forward_scale
        self.last_desired_base_speed = desired_base_speed
        self.last_base_speed = base_speed
        self.last_edge_gain = edge_gain
        self.last_heading_derivative = heading_derivative
        self.last_cross_track_derivative = cross_track_derivative
        self.last_heading_p_term = heading_p
        self.last_heading_d_term = heading_d
        self.last_xte_p_term = xte_p
        self.last_xte_d_term = xte_d
        self.last_turn_speed = turn_speed

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


def route_goal_pose(
    route: list[HybridPose],
    tracking_error: RouteTrackingError | None = None,
    local_goal_poses: list[HybridPose] | None = None,
    *,
    include_final: bool = True,
) -> HybridPose | None:
    """Return the next local pickup/unload checkpoint pose on the active route."""
    goal_index = next_route_checkpoint_index(
        route,
        tracking_error,
        local_goal_poses,
        include_final=include_final,
    )
    if goal_index < 0:
        return None
    return route[goal_index]


