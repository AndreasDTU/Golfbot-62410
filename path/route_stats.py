"""Route statistics: distance, turn count, and waypoint metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

from path.models import RoutePlan, WaypointKind

# Heading changes smaller than this are not counted as turns.
_TURN_THRESHOLD_DEG = 15.0


@dataclass(frozen=True)
class RouteStats:
    """Aggregate metrics for a compiled route plan."""

    total_distance_cm: float
    turn_count: int
    max_turn_deg: float
    waypoint_count: int
    navigate_count: int


def compute_route_stats(plan: RoutePlan) -> RouteStats:
    """Compute distance, turn, and waypoint metrics for a route plan."""
    wps = plan.waypoints
    total_distance = 0.0
    max_turn = 0.0
    turn_count = 0
    navigate_count = 0

    for i, wp in enumerate(wps):
        if wp.kind == WaypointKind.NAVIGATE:
            navigate_count += 1
        if i > 0:
            prev = wps[i - 1]
            dx = wp.x_cm - prev.x_cm
            dy = wp.y_cm - prev.y_cm
            total_distance += math.hypot(dx, dy)

            # Heading change between consecutive waypoints.
            delta = abs(wp.theta_rad - prev.theta_rad)
            # Normalize to [0, pi].
            delta = delta % (2 * math.pi)
            if delta > math.pi:
                delta = 2 * math.pi - delta
            delta_deg = math.degrees(delta)

            if delta_deg > _TURN_THRESHOLD_DEG:
                turn_count += 1
            if delta_deg > max_turn:
                max_turn = delta_deg

    return RouteStats(
        total_distance_cm=round(total_distance, 1),
        turn_count=turn_count,
        max_turn_deg=round(max_turn, 1),
        waypoint_count=len(wps),
        navigate_count=navigate_count,
    )
