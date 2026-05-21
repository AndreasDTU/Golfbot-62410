"""Shared route-planning data models for the detector stack."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HybridPose:
    """One planner state in bottom-left field coordinates."""

    x_cm: float
    y_cm: float
    theta_rad: float


@dataclass(frozen=True)
class PlannedBallTarget:
    """Route target with enough metadata for orange-first prioritization."""

    track_id: int
    label: str
    x_cm: float
    y_cm: float
    node_cm: tuple[int, int]


@dataclass(frozen=True)
class RoutePlan:
    """Cached route plus pickup metadata for visualization and invalidation."""

    points: list[HybridPose]
    active_target: PlannedBallTarget | None
    pickup_poses: list[HybridPose]
    unload_pose: HybridPose | None = None
    unload_goal_cm: tuple[float, float] | None = None
    ball_obstacles: list[PlannedBallTarget] | None = None
    ball_obstacle_radius_cm: float = 0.0
    ball_avoidance_mode: str = "disabled"


@dataclass(frozen=True)
class RouteTrackingError:
    """Closest-segment tracking error between live robot pose and cached route."""

    xte_cm: float
    signed_xte_cm: float
    heading_error_rad: float
    closest_point_cm: tuple[float, float]
    segment_heading_rad: float
    segment_index: int


@dataclass(frozen=True)
class HybridPlannerConfig:
    """Deterministic Hybrid A* tuning values expressed in field centimeters."""

    step_cm: float = 4.0
    theta_bins: int = 36
    goal_tolerance_cm: float = 4.0
    max_expansions: int = 36000
    translation_directions: tuple[float, ...] = (1.0, -1.0)
    reverse_cost_multiplier: float = 1.8
    rotation_deltas_rad: tuple[float, ...] = (
        math.radians(-10.0),
        math.radians(10.0),
    )
    in_place_rotation_cost: float = 1.1
    avoid_non_target_balls_enabled: bool = True
    ball_radius_cm: float = 2.0
    non_target_ball_extra_clearance_cm: float = 0.0
    allow_last_resort_orange_contact: bool = True
