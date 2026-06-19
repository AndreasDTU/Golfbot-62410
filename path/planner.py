"""Route Compiler and planning facade.

Layer 3 of the fit-based pathing pipeline.  Compiles ordered route stops
into a flat, annotated waypoint sequence with collision-free transitions.
Uses the distance field from Layer 1 as the sole collision source — no
separate grid dilation.

Also provides the ``plan_route()`` facade that chains all three layers.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from config import FieldConfig
from localization.models import RobotGeometry
from path.models import (
    HybridPose,
    PlannedBallTarget,
    RoutePlan,
    RouteWaypoint,
    WaypointKind,
)
from path.pickup_geometry import compute_pickup_geometry
from path.route_strategy import (
    NearestNeighborStrategy,
    RoutePlannerInput,
    RouteStrategyResult,
)


# ---------------------------------------------------------------------------
# 2-D grid helpers (Bresenham, A*, simplification)
# ---------------------------------------------------------------------------

def _bresenham_clear(
    blocked: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
) -> bool:
    """Check if the straight line between two grid cells is obstacle-free.

    Uses Bresenham rasterization on a binary grid (nonzero = blocked).
    """
    h, w = blocked.shape
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    cx, cy = x0, y0

    while True:
        if not (0 <= cy < h and 0 <= cx < w):
            return False
        if blocked[cy, cx]:
            return False
        if cx == x1 and cy == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            cx += sx
        if e2 < dx:
            err += dx
            cy += sy
    return True


def _grid_astar(
    blocked: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """Run 2-D grid A* on a binary blocked grid.

    Parameters
    ----------
    blocked : np.ndarray
        Binary grid where nonzero = blocked.
    start, goal : (col, row)
        Grid coordinates.

    Returns
    -------
    list[(col, row)] or None
        Path from start to goal, or None if no path exists.
    """
    h, w = blocked.shape
    sc, sr = start
    gc, gr = goal

    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return None
    if blocked[sr, sc] or blocked[gr, gc]:
        return None

    def heuristic(c: int, r: int) -> float:
        return math.hypot(c - gc, r - gr)

    SQRT2 = math.sqrt(2.0)
    NEIGHBORS = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
    ]

    open_heap: list[tuple[float, int, int, int]] = []
    counter = 0
    g_cost = np.full((h, w), np.inf, dtype=np.float64)
    g_cost[sr, sc] = 0.0
    came_from = np.full((h, w, 2), -1, dtype=np.int32)
    heapq.heappush(open_heap, (heuristic(sc, sr), counter, sc, sr))
    counter += 1

    max_expansions = min(h * w, 200_000)
    expansions = 0

    while open_heap and expansions < max_expansions:
        _, _, cc, cr = heapq.heappop(open_heap)
        expansions += 1

        if cc == gc and cr == gr:
            path = [(gc, gr)]
            nc, nr = gc, gr
            while not (nc == sc and nr == sr):
                prev = came_from[nr, nc]
                nc, nr = int(prev[0]), int(prev[1])
                path.append((nc, nr))
            path.reverse()
            return path

        current_g = g_cost[cr, cc]
        for dc, dr, cost in NEIGHBORS:
            nc, nr = cc + dc, cr + dr
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if blocked[nr, nc]:
                continue
            new_g = current_g + cost
            if new_g < g_cost[nr, nc]:
                g_cost[nr, nc] = new_g
                came_from[nr, nc] = (cc, cr)
                f = new_g + heuristic(nc, nr)
                heapq.heappush(open_heap, (f, counter, nc, nr))
                counter += 1

    return None


def _simplify_path(
    path: list[tuple[int, int]],
    epsilon: float = 2.0,
) -> list[tuple[int, int]]:
    """Simplify a grid path using the Douglas-Peucker algorithm."""
    if len(path) <= 2:
        return path
    pts = np.array(path, dtype=np.float64)
    return _douglas_peucker(pts, epsilon)


def _douglas_peucker(
    pts: np.ndarray,
    epsilon: float,
) -> list[tuple[int, int]]:
    """Recursive Douglas-Peucker simplification."""
    if len(pts) <= 2:
        return [(int(p[0]), int(p[1])) for p in pts]

    start = pts[0]
    end = pts[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-9:
        return [(int(start[0]), int(start[1])), (int(end[0]), int(end[1]))]

    line_unit = line_vec / line_len
    diffs = pts[1:-1] - start
    proj = diffs @ line_unit
    perp = diffs - np.outer(proj, line_unit)
    dists = np.linalg.norm(perp, axis=1)

    max_idx = int(np.argmax(dists)) + 1
    max_dist = dists[max_idx - 1]

    if max_dist > epsilon:
        left = _douglas_peucker(pts[: max_idx + 1], epsilon)
        right = _douglas_peucker(pts[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [(int(start[0]), int(start[1])), (int(end[0]), int(end[1]))]


# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------

def _field_to_grid(x_cm: float, y_cm: float, field_height_cm: float) -> tuple[int, int]:
    """Convert field-cm (bottom-left origin) to grid (col, row)."""
    return round(x_cm), round(field_height_cm - y_cm)


def _grid_to_field(col: int, row: int, field_height_cm: float) -> tuple[float, float]:
    """Convert grid (col, row) to field-cm (bottom-left origin)."""
    return float(col), field_height_cm - float(row)


# ---------------------------------------------------------------------------
# Route Compiler (Layer 3)
# ---------------------------------------------------------------------------

def _verify_simplified(
    blocked: np.ndarray,
    path: list[tuple[int, int]],
) -> bool:
    """Check that every segment of a simplified path is collision-free."""
    for i in range(len(path) - 1):
        c0, r0 = path[i]
        c1, r1 = path[i + 1]
        if not _bresenham_clear(blocked, c0, r0, c1, r1):
            return False
    return True


def _safe_simplify(
    blocked: np.ndarray,
    path: list[tuple[int, int]],
    epsilon: float = 2.0,
) -> list[tuple[int, int]]:
    """Simplify a grid path while guaranteeing every segment is collision-free.

    Tries Douglas-Peucker simplification and verifies each segment with
    Bresenham.  If a simplified segment cuts through blocked cells, the
    path is split at the midpoint and each half is simplified recursively.
    """
    if len(path) <= 2:
        return list(path)

    simplified = _simplify_path(path, epsilon)
    if _verify_simplified(blocked, simplified):
        return simplified

    mid = len(path) // 2
    left = _safe_simplify(blocked, path[: mid + 1], epsilon)
    right = _safe_simplify(blocked, path[mid:], epsilon)
    return left + right[1:]


def _emit_astar_waypoints(
    blocked: np.ndarray,
    start_col: int,
    start_row: int,
    goal_col: int,
    goal_row: int,
    field_height_cm: float,
    target_theta: float,
) -> list[RouteWaypoint]:
    """Run A*, simplify, and emit NAVIGATE waypoints between two grid cells.

    Returns an empty list if pathfinding fails (caller falls through to
    direct connection).
    """
    grid_path = _grid_astar(blocked, (start_col, start_row), (goal_col, goal_row))
    if grid_path is None or len(grid_path) <= 1:
        return []

    # Simplify with collision-safe recursive splitting.
    simplified = _safe_simplify(blocked, grid_path, epsilon=2.0)

    # Emit intermediate waypoints (skip first = current position, skip last
    # = the target itself which the caller emits with its own kind).
    waypoints: list[RouteWaypoint] = []
    for i in range(1, len(simplified) - 1):
        gc, gr = simplified[i]
        fx, fy = _grid_to_field(gc, gr, field_height_cm)
        # Heading = bearing to next point in the simplified path.
        nc, nr = simplified[i + 1]
        nx, ny = _grid_to_field(nc, nr, field_height_cm)
        theta = math.atan2(ny - fy, nx - fx)
        waypoints.append(RouteWaypoint(
            x_cm=fx, y_cm=fy, theta_rad=theta,
            kind=WaypointKind.NAVIGATE,
        ))

    return waypoints


def compile_route(
    strategy_result: RouteStrategyResult,
    distance_field: np.ndarray,
    start_pose: HybridPose,
    half_width_cm: float,
    field_height_cm: float,
) -> RoutePlan:
    """Compile ordered route stops into a flat annotated waypoint sequence.

    Uses the distance field as the single source of truth for collision
    avoidance: a grid cell is traversable if
    ``distance_field[row, col] >= half_width_cm``.

    Parameters
    ----------
    strategy_result : RouteStrategyResult
        Ordered stops from Layer 2.
    distance_field : np.ndarray
        Per-pixel distance to nearest obstacle (cm).  From Layer 1.
    start_pose : HybridPose
        Robot starting position.
    half_width_cm : float
        Robot half-width for body clearance (~9.75 cm).
    field_height_cm : float
        Field height for coordinate conversion.

    Returns
    -------
    RoutePlan
        Flat, ordered waypoint sequence with NAVIGATE/PICKUP/UNLOAD
        annotations.  No movement semantics.
    """
    # Blocked grid: nonzero where robot body does not fit.
    blocked = (distance_field < half_width_cm).astype(np.uint8)

    # ----- Build target sequence from stops -----
    # Each target: (x_cm, y_cm, theta_rad, WaypointKind, ball_index | None)
    targets: list[tuple[float, float, float, WaypointKind, int | None]] = []

    for stop in strategy_result.stops:
        cand = stop.candidate
        if stop.intermediate_node is not None:
            inter = stop.intermediate_node
            # Constrained/in-between: intermediate → pickup → intermediate
            targets.append((inter.x_cm, inter.y_cm, inter.theta_rad,
                            WaypointKind.NAVIGATE, None))
            targets.append((cand.x_cm, cand.y_cm, cand.theta_rad,
                            WaypointKind.PICKUP, stop.ball_index))
            targets.append((inter.x_cm, inter.y_cm, inter.theta_rad,
                            WaypointKind.NAVIGATE, None))
        else:
            # Safe: direct pickup
            targets.append((cand.x_cm, cand.y_cm, cand.theta_rad,
                            WaypointKind.PICKUP, stop.ball_index))

    # Unload at end.
    unload_goal_cm: tuple[float, float] | None = None
    if strategy_result.unload_position is not None:
        ux, uy = strategy_result.unload_position
        targets.append((ux, uy, math.pi, WaypointKind.UNLOAD, None))
        unload_goal_cm = (0.0, uy)

    # ----- Connect targets with collision-free paths -----
    waypoints: list[RouteWaypoint] = []
    cx, cy = start_pose.x_cm, start_pose.y_cm

    for tx, ty, theta, kind, ball_idx in targets:
        start_col, start_row = _field_to_grid(cx, cy, field_height_cm)
        goal_col, goal_row = _field_to_grid(tx, ty, field_height_cm)

        same_cell = (start_col == goal_col and start_row == goal_row)

        if not same_cell and not _bresenham_clear(blocked, start_col, start_row,
                                                  goal_col, goal_row):
            # Line blocked — insert A* detour waypoints.
            detour = _emit_astar_waypoints(
                blocked, start_col, start_row, goal_col, goal_row,
                field_height_cm, theta,
            )
            waypoints.extend(detour)

        # Emit the target itself.
        waypoints.append(RouteWaypoint(
            x_cm=tx, y_cm=ty, theta_rad=theta,
            kind=kind, ball_index=ball_idx,
        ))

        cx, cy = tx, ty

    return RoutePlan(
        waypoints=tuple(waypoints),
        unload_goal_cm=unload_goal_cm,
    )


# ---------------------------------------------------------------------------
# Facade — entry point
# ---------------------------------------------------------------------------

def plan_route(
    obstacle_grid: np.ndarray,
    balls: list[PlannedBallTarget],
    start_pose: HybridPose,
    geometry: RobotGeometry,
    field_config: FieldConfig,
    unload_position: tuple[float, float] | None = None,
) -> RoutePlan:
    """Chain all three path layers to produce a ``RoutePlan``.

    This is the single entry point that the GUI / brain calls.

    Parameters
    ----------
    obstacle_grid : np.ndarray
        Binary occupancy grid at 1 cm/px (0 = free, >0 = obstacle).
    balls : list[PlannedBallTarget]
        Detected ball targets.
    start_pose : HybridPose
        Current robot pose.
    geometry : RobotGeometry
        Robot body and tube geometry.
    field_config : FieldConfig
        Field dimensions.
    unload_position : tuple[float, float] | None
        Staging position for unloading, or None to skip unload.

    Returns
    -------
    RoutePlan
        Compiled route with annotated waypoints.
    """
    # Layer 1: Pickup geometry.
    geo_result = compute_pickup_geometry(
        field_width_cm=field_config.width_cm,
        field_height_cm=field_config.height_cm,
        obstacle_grid=obstacle_grid,
        balls=balls,
        geometry=geometry,
    )

    # Layer 2: Route strategy.
    strategy = NearestNeighborStrategy()
    strategy_input = RoutePlannerInput(
        geometry_result=geo_result,
        start_pose=start_pose,
        field_width_cm=field_config.width_cm,
        field_height_cm=field_config.height_cm,
        unload_position=unload_position,
    )
    strategy_result = strategy.plan(strategy_input)

    # Layer 3: Route compiler.
    plan = compile_route(
        strategy_result,
        distance_field=geo_result.distance_field,
        start_pose=start_pose,
        half_width_cm=geometry.width_cm * 0.5,
        field_height_cm=field_config.height_cm,
    )

    return plan
