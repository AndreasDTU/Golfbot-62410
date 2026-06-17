"""Route-planning strategy protocol and shared data types.

This module defines the interface that every route-planning strategy must
implement, plus the input/output data structures consumed and produced by
the planner.  It carries **no rendering dependency** (no numpy / cv2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from path.pathfinding.models import HybridPose, RoutePlan
from path.pickup_geometry import PickupGeometryResult, PickupPose


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObstacleGeometry:
    """Axis-aligned cross obstacle in field-cm coordinates."""

    center_x_cm: float
    center_y_cm: float
    half_size_cm: float       # 10.0 for 20 cm cross
    half_arm_width_cm: float  # 1.5 for 3 cm arm


@dataclass(frozen=True)
class RoutePlannerInput:
    """Everything the route planner needs to produce a plan."""

    geometry_result: PickupGeometryResult
    obstacle: ObstacleGeometry
    start_pose: HybridPose
    robot_radius_cm: float   # R (tube reach), used for obstacle inflation
    field_width_cm: float
    field_height_cm: float
    unload_pose: HybridPose | None = None          # staging pose for unload
    unload_goal_cm: tuple[float, float] | None = None  # target point (e.g. goal opening)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoverPoint:
    """A chosen pickup pose and the balls it covers."""

    pose: HybridPose               # exact instance reused in RoutePlan (id()-matching)
    source_pickup: PickupPose      # original from geometry result
    covered_ball_indices: tuple[int, ...]  # indices into PickupGeometryResult.balls


@dataclass(frozen=True)
class RouteEdge:
    """One edge in the route graph."""

    from_index: int
    to_index: int
    direct_distance_cm: float
    blocked: bool
    detour_waypoint: HybridPose | None
    total_distance_cm: float


@dataclass(frozen=True)
class RouteStrategyResult:
    """Full planning output -- intermediate artifacts + final RoutePlan."""

    cover_points: tuple[CoverPoint, ...]
    ordered_indices: tuple[int, ...]   # visit order into cover_points
    edges: tuple[RouteEdge, ...]       # all evaluated edges
    route_plan: RoutePlan              # final flattened plan


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class RouteStrategy(Protocol):
    """Strategy interface for route planners."""

    def plan(self, inp: RoutePlannerInput) -> RouteStrategyResult: ...
