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
# Pickup acceptance
# ---------------------------------------------------------------------------

def _normalize_angle(a: float) -> float:
    """Wrap *a* to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class SafePickupZone:
    """The set of SAFE origin positions from which a ball can be grabbed.

    The pickup origin may be any point on the ball's reach circle whose radius
    is within the caller's ``radial_tol_cm`` of the reach and whose
    bearing-around-the-ball falls in a SAFE arc (clearance guaranteed for the
    in-place turn).  From any such point the robot turns to :meth:`grab_heading`
    and grabs -- it never corrects its position to one exact spot.

    The radial tolerance is supplied by the caller (an executor policy, e.g.
    the standoff arrival tolerance), not baked in -- the zone is pure geometry.
    Angles are radians; positions are bottom-left field-cm.  ``safe_arcs`` is a
    tuple of ``(center_bearing, half_width)`` arcs in bearing-around-the-ball
    space.  The executor depends on its two predicates, not on any grid.
    """

    ball_x_cm: float
    ball_y_cm: float
    reach_cm: float
    mounting_offset_rad: float
    safe_arcs: tuple[tuple[float, float], ...]

    def accepts(self, x_cm: float, y_cm: float, radial_tol_cm: float) -> bool:
        """True if origin ``(x, y)`` is a SAFE pickup spot on the reach circle.

        ``radial_tol_cm`` is how far off the reach circle is still accepted --
        the caller's standoff tolerance.
        """
        dx = x_cm - self.ball_x_cm
        dy = y_cm - self.ball_y_cm
        if abs(math.hypot(dx, dy) - self.reach_cm) > radial_tol_cm:
            return False
        bearing = math.atan2(dy, dx)
        return any(
            abs(_normalize_angle(bearing - center)) <= half_width
            for center, half_width in self.safe_arcs
        )

    def grab_heading(self, x_cm: float, y_cm: float) -> float:
        """Robot heading that lands the tube tip on the ball from ``(x, y)``."""
        bearing_to_ball = math.atan2(self.ball_y_cm - y_cm, self.ball_x_cm - x_cm)
        return _normalize_angle(bearing_to_ball + self.mounting_offset_rad)


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
    # SAFE acceptance region for a PICKUP: lets the executor grab from any SAFE
    # point on the ball's reach circle instead of homing to this exact pose.
    # None for constrained pickups (precise approach) and NAVIGATE/UNLOAD.
    pickup_zone: SafePickupZone | None = None


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
