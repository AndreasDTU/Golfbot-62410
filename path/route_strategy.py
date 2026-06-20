"""Route-planning strategy: candidate selection, ordering, and intermediate nodes.

Layer 2 of the fit-based pathing pipeline.  Given classified pickup
candidates from Layer 1, selects the best candidate per ball, computes
intermediate approach nodes for non-safe candidates, and orders the
visit sequence.

Hard constraints
----------------
1. **Orange first** -- the orange ball must always be visited first in
   the route (it awards bonus points in the competition).
2. **Start at robot** -- the route always begins from the robot's current
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

    ``start_pose`` is the robot's current position -- the route always
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
# Shared Step 1: per-ball candidate selection
# ---------------------------------------------------------------------------

def _select_stops(
    geo: PickupGeometryResult,
    field_height_cm: float,
    approach_hint: tuple[float, float] | None = None,
) -> tuple[list[RouteStop | None], list[int]]:
    """Step 1: pick best candidate per ball, compute intermediates.

    When *approach_hint* is provided (typically the robot start position),
    SAFE candidates are sorted so that those closer to the hint are
    preferred.  This avoids selecting a pickup point on the far side of
    the ball relative to the robot's approach direction.

    Returns ``(per_ball, unreachable)`` where *per_ball[i]* is the
    chosen RouteStop for ball *i* (or None if unreachable) and
    *unreachable* lists indices of balls that could not be assigned.
    """
    distance_field = geo.distance_field
    safe_radius = geo.safe_radius_cm

    per_ball: list[RouteStop | None] = []
    unreachable: list[int] = []

    for ball_idx, ball_cands in enumerate(geo.balls):
        if not ball_cands.reachable:
            per_ball.append(None)
            unreachable.append(ball_idx)
            continue

        # Sort candidates: category first, then among SAFE candidates
        # prefer those closer to the approach hint (if given), otherwise
        # fall back to obstacle clearance.
        def _candidate_sort_key(c: PickupCandidate) -> tuple:
            pref = _CATEGORY_PREFERENCE[c.category]
            if approach_hint is not None and c.category == PickupCategory.SAFE:
                # Among SAFE candidates, prefer closer to approach hint.
                dist = math.hypot(c.x_cm - approach_hint[0], c.y_cm - approach_hint[1])
                return (pref, dist)
            return (pref, -c.obstacle_distance_cm)

        sorted_candidates = sorted(ball_cands.candidates, key=_candidate_sort_key)

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

    return per_ball, unreachable


# ---------------------------------------------------------------------------
# Shared ordering helpers
# ---------------------------------------------------------------------------

def _nearest_neighbor_order(
    per_ball: list[RouteStop | None],
    inp: RoutePlannerInput,
) -> list[RouteStop]:
    """Order stops using nearest-neighbor greedy with orange-first constraint."""
    geo = inp.geometry_result
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

    return ordered


def _tour_distance(
    stops: list[RouteStop],
    start_x: float,
    start_y: float,
) -> float:
    """Total Euclidean approach distance for a stop ordering."""
    total = 0.0
    cx, cy = start_x, start_y
    for stop in stops:
        total += _approach_distance(stop, cx, cy)
        cx, cy = stop.candidate.x_cm, stop.candidate.y_cm
    return total


def _two_opt_improve(
    stops: list[RouteStop],
    start_pose: HybridPose,
) -> list[RouteStop]:
    """2-opt local search.  Orange ball (if first) stays locked at index 0."""
    if len(stops) <= 2:
        return stops

    sx, sy = start_pose.x_cm, start_pose.y_cm

    # If the first stop is orange, lock it (start swapping from index 1).
    first_is_orange = stops[0].candidate.ball_index in (
        stops[0].ball_index,
    )
    swap_start = 1 if first_is_orange else 0

    improved = True
    while improved:
        improved = False
        best_dist = _tour_distance(stops, sx, sy)
        for i in range(swap_start, len(stops) - 1):
            for j in range(i + 1, len(stops)):
                # Reverse the segment between i and j (inclusive).
                new_stops = stops[:i] + stops[i:j + 1][::-1] + stops[j + 1:]
                new_dist = _tour_distance(new_stops, sx, sy)
                if new_dist < best_dist - 1e-9:
                    stops = new_stops
                    best_dist = new_dist
                    improved = True

    return stops


def _angular_sweep_order(
    per_ball: list[RouteStop | None],
    inp: RoutePlannerInput,
) -> list[RouteStop]:
    """Order stops by angular sweep from start_pose with orange-first constraint."""
    geo = inp.geometry_result
    sx, sy = inp.start_pose.x_cm, inp.start_pose.y_cm

    valid = [(i, s) for i, s in enumerate(per_ball) if s is not None]

    # Separate orange and non-orange.
    orange: list[RouteStop] = []
    non_orange: list[tuple[float, RouteStop]] = []

    for i, stop in valid:
        assert stop is not None
        # Use the approach node for angle computation.
        if stop.intermediate_node is not None:
            ax, ay = stop.intermediate_node.x_cm, stop.intermediate_node.y_cm
        else:
            ax, ay = stop.candidate.x_cm, stop.candidate.y_cm
        angle = math.atan2(ay - sy, ax - sx)

        if geo.balls[i].ball.label == "orange":
            orange.append(stop)
        else:
            non_orange.append((angle, stop))

    # Sort non-orange by angle.
    non_orange.sort(key=lambda pair: pair[0])
    ordered = [s for _, s in non_orange]

    # Prepend orange (nearest if multiple).
    if orange:
        if len(orange) > 1:
            orange.sort(key=lambda s: _approach_distance(s, sx, sy))
        ordered = orange[:1] + ordered

    return ordered


# ---------------------------------------------------------------------------
# Intersection-priority helpers (deferred candidate selection)
# ---------------------------------------------------------------------------

def _best_stop_for_ball_from(
    ball_idx: int,
    geo: PickupGeometryResult,
    field_height_cm: float,
    cx: float,
    cy: float,
) -> RouteStop | None:
    """Pick the best candidate for *ball_idx* given that the robot is at (cx, cy).

    Among SAFE candidates, prefer the one whose pickup position is
    closest to (cx, cy).  For non-SAFE, prefer obstacle clearance and
    require a valid intermediate node.
    """
    ball_cands = geo.balls[ball_idx]
    if not ball_cands.reachable:
        return None

    distance_field = geo.distance_field
    safe_radius = geo.safe_radius_cm

    # Split by category.
    safe: list[PickupCandidate] = []
    non_safe: list[PickupCandidate] = []
    for c in ball_cands.candidates:
        if c.category == PickupCategory.SAFE:
            safe.append(c)
        else:
            non_safe.append(c)

    # Among SAFE candidates, pick the closest to current position.
    if safe:
        best_safe = min(safe, key=lambda c: math.hypot(c.x_cm - cx, c.y_cm - cy))
        return RouteStop(candidate=best_safe, intermediate_node=None, ball_index=ball_idx)

    # Non-safe: try in preference order (IN_BETWEEN before CONSTRAINED).
    non_safe.sort(key=lambda c: (_CATEGORY_PREFERENCE[c.category], -c.obstacle_distance_cm))
    for cand in non_safe:
        intermediate = _find_intermediate_node(cand, distance_field, safe_radius, field_height_cm)
        if intermediate is not None:
            return RouteStop(candidate=cand, intermediate_node=intermediate, ball_index=ball_idx)

    # Fallback: best candidate without intermediate.
    best = _pick_best_candidate(ball_cands.candidates)
    if best is not None:
        return RouteStop(candidate=best, intermediate_node=None, ball_index=ball_idx)
    return None


def _intersection_priority_plan(
    inp: RoutePlannerInput,
) -> tuple[list[RouteStop], list[int]]:
    """Joint ordering + candidate selection with intersection priority.

    Instead of selecting one candidate per ball up front and then
    ordering, this function interleaves the two steps.  At each greedy
    step it re-evaluates every remaining ball's best candidate relative
    to the robot's *current* position.  This means:

    - After picking up ball A, the candidate chosen for nearby ball B
      will be on the side of B closest to where the robot just was
      (the intersection-priority benefit).
    - Balls with overlapping rings naturally get visited consecutively
      because their near-side candidates are very close to the current
      position.
    """
    geo = inp.geometry_result
    cx, cy = inp.start_pose.x_cm, inp.start_pose.y_cm

    remaining = {
        i for i, bc in enumerate(geo.balls) if bc.reachable
    }
    unreachable = [i for i, bc in enumerate(geo.balls) if not bc.reachable]
    ordered: list[RouteStop] = []

    # Orange-first constraint.
    orange_indices = [i for i in remaining if geo.balls[i].ball.label == "orange"]
    if orange_indices:
        # Pick the orange ball with the best candidate from start.
        best_orange: RouteStop | None = None
        for oi in orange_indices:
            stop = _best_stop_for_ball_from(oi, geo, inp.field_height_cm, cx, cy)
            if stop is None:
                continue
            if best_orange is None or _approach_distance(stop, cx, cy) < _approach_distance(best_orange, cx, cy):
                best_orange = stop
        if best_orange is not None:
            remaining.discard(best_orange.ball_index)
            ordered.append(best_orange)
            cx, cy = best_orange.candidate.x_cm, best_orange.candidate.y_cm

    # Greedy nearest with deferred candidate selection.
    while remaining:
        best_stop: RouteStop | None = None
        best_dist = math.inf

        for i in remaining:
            stop = _best_stop_for_ball_from(i, geo, inp.field_height_cm, cx, cy)
            if stop is None:
                continue
            d = _approach_distance(stop, cx, cy)
            if d < best_dist:
                best_dist = d
                best_stop = stop

        if best_stop is None:
            # Remaining balls are all truly unreachable from any angle.
            unreachable.extend(sorted(remaining))
            break

        remaining.discard(best_stop.ball_index)
        ordered.append(best_stop)
        cx, cy = best_stop.candidate.x_cm, best_stop.candidate.y_cm

    return ordered, unreachable


# ---------------------------------------------------------------------------
# Nearest-neighbor strategy
# ---------------------------------------------------------------------------

class NearestNeighborStrategy:
    """Greedy nearest-neighbor visit ordering with intermediate node placement."""

    def plan(self, inp: RoutePlannerInput) -> RouteStrategyResult:
        hint = (inp.start_pose.x_cm, inp.start_pose.y_cm)
        per_ball, unreachable = _select_stops(
            inp.geometry_result, inp.field_height_cm, approach_hint=hint,
        )
        ordered = _nearest_neighbor_order(per_ball, inp)
        return RouteStrategyResult(
            stops=tuple(ordered),
            unreachable_balls=tuple(unreachable),
            unload_position=inp.unload_position,
        )


# ---------------------------------------------------------------------------
# 2-opt strategy
# ---------------------------------------------------------------------------

class TwoOptStrategy:
    """2-opt local search improving nearest-neighbor ordering."""

    def plan(self, inp: RoutePlannerInput) -> RouteStrategyResult:
        hint = (inp.start_pose.x_cm, inp.start_pose.y_cm)
        per_ball, unreachable = _select_stops(
            inp.geometry_result, inp.field_height_cm, approach_hint=hint,
        )
        # Build initial ordering via nearest-neighbor.
        ordered = _nearest_neighbor_order(per_ball, inp)
        # Apply 2-opt swaps (orange locked at index 0).
        ordered = _two_opt_improve(ordered, inp.start_pose)
        return RouteStrategyResult(
            stops=tuple(ordered),
            unreachable_balls=tuple(unreachable),
            unload_position=inp.unload_position,
        )


# ---------------------------------------------------------------------------
# Sweep strategy
# ---------------------------------------------------------------------------

class SweepStrategy:
    """Angular sweep ordering to minimize heading changes."""

    def plan(self, inp: RoutePlannerInput) -> RouteStrategyResult:
        hint = (inp.start_pose.x_cm, inp.start_pose.y_cm)
        per_ball, unreachable = _select_stops(
            inp.geometry_result, inp.field_height_cm, approach_hint=hint,
        )
        ordered = _angular_sweep_order(per_ball, inp)
        return RouteStrategyResult(
            stops=tuple(ordered),
            unreachable_balls=tuple(unreachable),
            unload_position=inp.unload_position,
        )


# ---------------------------------------------------------------------------
# Intersection-priority strategy
# ---------------------------------------------------------------------------

class IntersectionPriorityStrategy:
    """Deferred candidate selection with intersection priority.

    Unlike the other strategies which pick one candidate per ball up
    front, this strategy selects each ball's pickup candidate at
    ordering time based on the robot's current position.  After picking
    up ball A, ball B's candidate is chosen on the side closest to
    where the robot just was.  This naturally visits balls with
    overlapping pickup rings consecutively and approaches each ball
    from the near side.
    """

    def plan(self, inp: RoutePlannerInput) -> RouteStrategyResult:
        ordered, unreachable = _intersection_priority_plan(inp)
        return RouteStrategyResult(
            stops=tuple(ordered),
            unreachable_balls=tuple(unreachable),
            unload_position=inp.unload_position,
        )
