"""Shared route-planning data models for the detector stack."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class RouteSegmentType(Enum):
    """Semantic route edge type for rendering and control handoff diagnostics."""

    TRANSIT = "TRANSIT"
    PIVOT = "PIVOT"
    CREEP = "CREEP"


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
    pickup_poses: list[HybridPose]
    unload_pose: HybridPose | None = None
    unload_goal_cm: tuple[float, float] | None = None


@dataclass(frozen=True)
class RedCrossSpec:
    """Red cross obstacle geometry from perception, in field-cm coordinates."""

    center_x_cm: float
    center_y_cm: float
    half_size_cm: float       # half of the overall cross span (e.g. 10 for 20 cm)
    half_arm_width_cm: float  # half of the arm width (e.g. 1.5 for 3 cm)
    angle_rad: float = 0.0   # rotation angle; 0 = axis-aligned


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
    reverse_cost_multiplier: float = 2.5
    rotation_deltas_rad: tuple[float, ...] = (
        math.radians(-10.0),
        math.radians(10.0),
    )
    in_place_rotation_cost: float = 2.0
    heuristic_weight: float = 1.5
    gear_shift_penalty: float = 50.0
    steering_change_penalty: float = 3.0
    transit_speed_pct: float = 38.0
    pivot_speed_pct: float = 30.0
    creep_speed_pct: float = 7.0
    flexible_standoff_max_cm: float = 15.0
    flexible_standoff_min_cm: float = 0.0
    flexible_standoff_heading_tolerance_rad: float = math.radians(10.0)
    unload_staging_margin_cm: float = 2.0
    wall_pickup_prefer_distance_cm: float = 12.0
    wall_pickup_perpendicular_tolerance_rad: float = math.radians(35.0)
    avoid_non_target_balls_enabled: bool = True
    ball_radius_cm: float = 2.0
    non_target_ball_extra_clearance_cm: float = 0.0
    ball_core_cost: float = 1000.0
    ball_close_cost: float = 200.0
    ball_warning_cost: float = 50.0
    ball_close_clearance_cm: float = 5.0
    ball_warning_clearance_cm: float = 10.0
    tube_width_cm: float = 6.0
