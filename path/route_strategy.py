"""Route-planning strategy: candidate selection, ordering, and intermediate nodes.

Layer 2 of the fit-based pathing pipeline.  Given classified pickup
candidates from Layer 1, selects the best candidate per ball, computes
intermediate approach nodes for non-safe candidates, and orders the
visit sequence.

Hard constraints
----------------
1. **Orange first** — the orange ball must always be visited first in
   the route (it awards bonus points in the competition).
2. **Start at robot** — the route always begins from the robot's current
   position (``start_pose``).  Nearest-neighbor ordering uses this as
   the origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from path.models import HybridPose
from path.pickup_geometry import (
    PickupCandidate,
    PickupCategory,
    PickupGeometryResult,
)


# ---------------------------------------------------------------------------
# Input / output types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoutePlannerInput:
    """Everything the route planner needs to produce a plan.

    ``start_pose`` is the robot's current position — the route always
    begins here (hard constraint).
    """

    geometry_result: PickupGeometryResult
    start_pose: HybridPose
    field_width_cm: float
    field_height_cm: float
    unload_position: tuple[float, float] | None = None


@dataclass(frozen=True)
class RouteStop:
    """One ball visit: chosen candidate plus optional intermediate node."""

    candidate: PickupCandidate
    intermediate_node: HybridPose | None  # None if SAFE
    ball_index: int


@dataclass(frozen=True)
class RouteStrategyResult:
    """Full planning output from the route strategy."""

    stops: tuple[RouteStop, ...]
    unreachable_balls: tuple[int, ...]
    unload_position: tuple[float, float] | None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class RouteStrategy(Protocol):
    """Strategy interface for route planners."""

    def plan(self, inp: RoutePlannerInput) -> RouteStrategyResult: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CATEGORY_PREFERENCE = {
    PickupCategory.SAFE: 0,
    PickupCategory.IN_BETWEEN: 1,
    PickupCategory.CONSTRAINED: 2,
}

# Maximum distance (cm) to walk backward looking for a safe intermediate.
_MAX_INTERMEDIATE_SEARCH_CM = 30.0


def _approach_distance(stop: RouteStop | None, cx: float, cy: float) -> float:
    """Distance from (cx, cy) to the stop's approach node."""
    assert stop is not None
    if stop.intermediate_node is not None:
        tx, ty = stop.intermediate_node.x_cm, stop.intermediate_node.y_cm
    else:
        tx, ty = stop.candidate.x_cm, stop.candidate.y_cm
    return math.hypot(tx - cx, ty - cy)


def _pick_best_candidate(
    candidates: tuple[PickupCandidate, ...],
) -> PickupCandidate | None:
    """Select the best candidate: prefer SAFE, then IN_BETWEEN, then
    CONSTRAINED.  Tiebreak by descending obstacle_distance_cm."""
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: (_CATEGORY_PREFERENCE[c.category], -c.obstacle_distance_cm),
    )


def _find_intermediate_node(
    candidate: PickupCandidate,
    distance_field: np.ndarray,
    safe_radius_cm: float,
    field_height_cm: float,
) -> HybridPose | None:
    """Walk backward from the pickup point along the approach heading to find
    a position with enough obstacle clearance for a safe turn.

    Returns None if no safe position is found within the search distance.
    """
    grid_h, grid_w = distance_field.shape[:2]

    # Walk backward (opposite of heading) in 1 cm increments.
    back_dx = -math.cos(candidate.theta_rad)
    back_dy = -math.sin(candidate.theta_rad)

    for step in range(1, int(_MAX_INTERMEDIATE_SEARCH_CM) + 1):
        ix = candidate.x_cm + back_dx * step
        iy = candidate.y_cm + back_dy * step
        col = round(ix)
        row = round(field_height_cm - iy)
        if not (0 <= row < grid_h and 0 <= col < grid_w):
            break
        d = float(distance_field[row, col])
        if d > safe_radius_cm:
            return HybridPose(x_cm=ix, y_cm=iy, theta_rad=candidate.theta_rad)
    return None


# ---------------------------------------------------------------------------
# Nearest-neighbor strategy
# ---------------------------------------------------------------------------

class NearestNeighborStrategy:
    """Greedy nearest-neighbor visit ordering with intermediate node placement."""

    def plan(self, inp: RoutePlannerInput) -> RouteStrategyResult:
        geo = inp.geometry_result
        distance_field = geo.distance_field
        safe_radius = geo.safe_radius_cm
        field_height_cm = inp.field_height_cm

        # --- Step 1: pick best candidate per ball, compute intermediates ---
        per_ball: list[RouteStop | None] = []
        unreachable: list[int] = []

        for ball_idx, ball_cands in enumerate(geo.balls):
            if not ball_cands.reachable:
                per_ball.append(None)
                unreachable.append(ball_idx)
                continue

            # Try candidates in preference order until we find one that
            # either is safe or has a valid intermediate node.
            sorted_candidates = sorted(
                ball_cands.candidates,
                key=lambda c: (_CATEGORY_PREFERENCE[c.category], -c.obstacle_distance_cm),
            )

            stop: RouteStop | None = None
            for cand in sorted_candidates:
                if cand.category == PickupCategory.SAFE:
                    stop = RouteStop(
                        candidate=cand,
                        intermediate_node=None,
                        ball_index=ball_idx,
                    )
                    break
                # Non-safe: need an intermediate node.
                intermediate = _find_intermediate_node(
                    cand, distance_field, safe_radius, field_height_cm,
                )
                if intermediate is not None:
                    stop = RouteStop(
                        candidate=cand,
                        intermediate_node=intermediate,
                        ball_index=ball_idx,
                    )
                    break

            if stop is None:
                # Fall back: use the best candidate even without intermediate.
                best = _pick_best_candidate(ball_cands.candidates)
                if best is not None:
                    stop = RouteStop(
                        candidate=best,
                        intermediate_node=None,
                        ball_index=ball_idx,
                    )

            if stop is None:
                per_ball.append(None)
                unreachable.append(ball_idx)
            else:
                per_ball.append(stop)

        # --- Step 2: ordering (orange first, then nearest-neighbor) ---
        remaining = {i for i, s in enumerate(per_ball) if s is not None}
        ordered: list[RouteStop] = []
        cx, cy = inp.start_pose.x_cm, inp.start_pose.y_cm

        # Hard constraint: orange ball is always picked up first.
        orange_indices = [
            i for i in remaining
            if geo.balls[i].ball.label == "orange"
        ]
        if orange_indices:
            # If multiple orange balls, pick the nearest one.
            orange_idx = min(
                orange_indices,
                key=lambda i: _approach_distance(per_ball[i], cx, cy),  # type: ignore[arg-type]
            )
            remaining.remove(orange_idx)
            stop = per_ball[orange_idx]
            assert stop is not None
            ordered.append(stop)
            cx, cy = stop.candidate.x_cm, stop.candidate.y_cm

        # Nearest-neighbor greedy ordering for remaining balls.
        while remaining:
            best_idx = min(
                remaining,
                key=lambda i: _approach_distance(per_ball[i], cx, cy),  # type: ignore[arg-type]
            )
            remaining.remove(best_idx)
            stop = per_ball[best_idx]
            assert stop is not None
            ordered.append(stop)
            # Advance current position to the pickup point (after pickup
            # we'll be at the candidate position).
            cx, cy = stop.candidate.x_cm, stop.candidate.y_cm

        return RouteStrategyResult(
            stops=tuple(ordered),
            unreachable_balls=tuple(unreachable),
            unload_position=inp.unload_position,
        )
