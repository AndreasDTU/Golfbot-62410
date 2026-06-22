"""Backwards-compatibility shim — re-exports from ``path.models``.

All canonical types now live in ``path.models``.  This module re-exports
them so existing imports (``from path.pathfinding.models import ...``)
continue to work during the transition.  Types that only the old planner
used (HybridPlannerConfig) are kept here until the old planner is deleted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Re-export canonical types
from path.models import (  # noqa: F401
    HybridPose,
    PlannedBallTarget,
    RedCrossSpec,
    RoutePlan,
    RouteSegmentType,
    RouteTrackingError,
)


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
