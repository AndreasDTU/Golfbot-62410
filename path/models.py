"""Shared path-planning data models for the detector stack.

Canonical home for types consumed by brain, guidance, GUI, and the path
layers.  Geometry helpers (tube_center_for_pose, rear_unload_point_for_pose)
live here as standalone functions so that debug rendering no longer depends
on the heavy planner module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Pose / target types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HybridPose:
    """One planner state in bottom-left field coordinates."""

    x_cm: float
    y_cm: float
    theta_rad: float


@dataclass(frozen=True)
class PlannedBallTarget:
    """Route target with enough metadata for orange-first prioritization.

    ``label`` is ``"orange"`` or ``"white"``.  The route strategy uses this
    to enforce the hard constraint that the orange ball is always visited
    first (competition bonus points).
    """

    track_id: int
    label: str  # "orange" | "white"
    x_cm: float
    y_cm: float
    node_cm: tuple[int, int]


# ---------------------------------------------------------------------------
# Route plan
# ---------------------------------------------------------------------------

class WaypointKind(str, Enum):
    """Annotation for each waypoint in a compiled route.

    NAVIGATE -- drive through this point (A* detour, intermediate node, etc.)
    PICKUP   -- pick up a ball here (carries ball_index)
    UNLOAD   -- unload here
    """

    NAVIGATE = "navigate"
    PICKUP = "pickup"
    UNLOAD = "unload"


@dataclass(frozen=True)
class RouteWaypoint:
    """One waypoint in a compiled route plan."""

    x_cm: float
    y_cm: float
    theta_rad: float
    kind: WaypointKind
    ball_index: int | None = None
    # True when this pickup sits against the cross/wall, so the executor must
    # back away before raising the tube. Always False for NAVIGATE/UNLOAD.
    obstacle_constrained: bool = False
    # Final-heading acceptance window (radians) for a PICKUP: wide for open
    # balls so the robot grabs without a hard final pivot. None means "use the
    # executor's tight default". Always None for NAVIGATE/UNLOAD.
    accept_heading_tol_rad: float | None = None


@dataclass(frozen=True)
class RoutePlan:
    """Flat, ordered list of annotated waypoints.

    The path is intentionally dumb: it describes *where* to go and *what*
    to do there.  It never prescribes *how* to move.  Brain decides intent,
    Guidance decides geometry.
    """

    waypoints: tuple[RouteWaypoint, ...]
    unload_goal_cm: tuple[float, float] | None = None


# ---------------------------------------------------------------------------
# Obstacle spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RedCrossSpec:
    """Red cross obstacle geometry from perception, in field-cm coordinates."""

    center_x_cm: float
    center_y_cm: float
    half_size_cm: float
    half_arm_width_cm: float
    angle_rad: float = 0.0


# ---------------------------------------------------------------------------
# Route tracking (kept for guidance/route_tracking compatibility)
# ---------------------------------------------------------------------------

class RouteSegmentType(Enum):
    """Semantic route edge type for rendering and control handoff diagnostics."""

    TRANSIT = "TRANSIT"
    PIVOT = "PIVOT"
    CREEP = "CREEP"


@dataclass(frozen=True)
class RouteTrackingError:
    """Closest-segment tracking error between live robot pose and cached route."""

    xte_cm: float
    signed_xte_cm: float
    heading_error_rad: float
    closest_point_cm: tuple[float, float]
    segment_heading_rad: float
    segment_index: int


# ---------------------------------------------------------------------------
# Geometry helpers (formerly static methods on HybridAStarPlanner)
# ---------------------------------------------------------------------------

def tube_center_for_pose(
    pose: HybridPose,
    geometry,
) -> tuple[float, float]:
    """Return the field-coordinate pickup point at the intake tip.

    Parameters
    ----------
    pose : HybridPose
        Robot origin pose.
    geometry : RobotGeometry
        Robot body and tube geometry (duck-typed to avoid circular import).
    """
    forward = (math.cos(pose.theta_rad), math.sin(pose.theta_rad))
    right = (math.sin(pose.theta_rad), -math.cos(pose.theta_rad))
    return (
        pose.x_cm + forward[0] * geometry.tube_forward_cm + right[0] * geometry.tube_right_cm,
        pose.y_cm + forward[1] * geometry.tube_forward_cm + right[1] * geometry.tube_right_cm,
    )


def rear_unload_point_for_pose(
    pose: HybridPose,
    geometry,
) -> tuple[float, float]:
    """Return the field-coordinate rear unload tip when the mechanism is lowered.

    Parameters
    ----------
    pose : HybridPose
        Robot origin pose.
    geometry : RobotGeometry
        Robot body and tube geometry (duck-typed to avoid circular import).
    """
    reach_cm = geometry.rear_cm + geometry.unload_extension_cm
    forward = (math.cos(pose.theta_rad), math.sin(pose.theta_rad))
    return (
        pose.x_cm - forward[0] * reach_cm,
        pose.y_cm - forward[1] * reach_cm,
    )
