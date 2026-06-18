"""v1 route strategy: greedy set-cover + nearest-neighbor + corner detour.

No rendering dependencies (no numpy / cv2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import permutations

from path.pathfinding.models import HybridPose, RoutePlan
from path.pickup_geometry import PickupGeometryResult, PickupPose
from path.route_strategy import (
    CoverPoint,
    ObstacleGeometry,
    RouteEdge,
    RoutePlannerInput,
    RouteStrategyResult,
)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

QUANTIZE_PRECISION = 10  # 1 / 0.1 cm
INTERSECTION_EXTRA_BALL_BONUS_CM = 12.0
INTERSECTION_SHARED_ORIGIN_BONUS_CM = 4.0
INTERSECTION_REACH_OFFSET_PENALTY = 0.25
ISOLATED_BALL_FALLBACK_CANDIDATES = 6
INTERSECTION_OPTIMAL_CANDIDATES_PER_MASK = 1


def _quantize(x: float, y: float) -> tuple[int, int]:
    """Snap a position to 0.1 cm grid for grouping."""
    return (round(x * QUANTIZE_PRECISION), round(y * QUANTIZE_PRECISION))


def _orange_ball_indices(geometry_result: PickupGeometryResult) -> tuple[int, ...]:
    """Return reachable orange-ball indices, preserving detection order."""
    return tuple(
        index for index, ball in enumerate(geometry_result.balls)
        if ball.reachable and ball.label.strip().lower() == "orange"
    )


def _sort_ball_indices_orange_first(
    geometry_result: PickupGeometryResult,
    indices: set[int] | tuple[int, ...] | list[int],
) -> tuple[int, ...]:
    """Sort ball indices with orange balls first, then stable numeric order."""
    orange = set(_orange_ball_indices(geometry_result))
    return tuple(sorted(indices, key=lambda index: (index not in orange, index)))


def _covers_orange(
    geometry_result: PickupGeometryResult,
    indices: set[int] | tuple[int, ...] | list[int],
) -> bool:
    """Return whether a set of ball indices includes a reachable orange ball."""
    orange = set(_orange_ball_indices(geometry_result))
    return any(index in orange for index in indices)


@dataclass(frozen=True)
class AABB:
    """Axis-aligned bounding box."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class StationCandidate:
    """One robot origin that can perform one or more pickup actions."""

    x_cm: float
    y_cm: float
    source_pickup: PickupPose
    pickup_poses_by_ball: tuple[tuple[int, HybridPose], ...]
    total_abs_reach_offset_cm: float
    from_ring_intersection: bool

    @property
    def covered_ball_indices(self) -> tuple[int, ...]:
        return tuple(ball_idx for ball_idx, _ in self.pickup_poses_by_ball)


@dataclass(frozen=True)
class BallObstacle:
    """Circular hard obstacle around an uncollected ball."""

    ball_index: int
    x_cm: float
    y_cm: float
    radius_cm: float


def _segment_intersects_aabb(
    x0: float, y0: float, x1: float, y1: float, box: AABB,
) -> bool:
    """Liang-Barsky line-segment vs AABB intersection test."""
    dx = x1 - x0
    dy = y1 - y0

    p = (-dx, dx, -dy, dy)
    q = (
        x0 - box.x_min,
        box.x_max - x0,
        y0 - box.y_min,
        box.y_max - y0,
    )

    t_enter = 0.0
    t_exit = 1.0

    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            # Parallel to this slab
            if qi < 0.0:
                return False  # outside slab
        else:
            t = qi / pi
            if pi < 0.0:
                if t > t_enter:
                    t_enter = t
            else:
                if t < t_exit:
                    t_exit = t
            if t_enter > t_exit:
                return False

    return t_enter <= t_exit


def _inflated_cross_bars(
    obs: ObstacleGeometry, R: float,
) -> tuple[AABB, AABB]:
    """Return the two inflated AABB bars of the cross obstacle."""
    cx, cy = obs.center_x_cm, obs.center_y_cm
    hs = obs.half_size_cm
    ha = obs.half_arm_width_cm

    vertical = AABB(
        x_min=cx - ha - R,
        x_max=cx + ha + R,
        y_min=cy - hs - R,
        y_max=cy + hs + R,
    )
    horizontal = AABB(
        x_min=cx - hs - R,
        x_max=cx + hs + R,
        y_min=cy - ha - R,
        y_max=cy + ha + R,
    )
    return vertical, horizontal


def _segment_hits_inflated_cross(
    x0: float, y0: float, x1: float, y1: float,
    obs: ObstacleGeometry, R: float,
) -> bool:
    """Test whether a line segment intersects either inflated cross bar."""
    vbar, hbar = _inflated_cross_bars(obs, R)
    return (
        _segment_intersects_aabb(x0, y0, x1, y1, vbar)
        or _segment_intersects_aabb(x0, y0, x1, y1, hbar)
    )


def _point_inside_aabb(x: float, y: float, box: AABB) -> bool:
    """Return whether a point lies inside or on an AABB."""
    return box.x_min <= x <= box.x_max and box.y_min <= y <= box.y_max


def _point_inside_inflated_cross(
    x: float, y: float, obs: ObstacleGeometry, R: float,
) -> bool:
    """Return whether a point is inside either inflated cross bar."""
    vbar, hbar = _inflated_cross_bars(obs, R)
    return _point_inside_aabb(x, y, vbar) or _point_inside_aabb(x, y, hbar)


def _point_inside_field(
    x: float, y: float, field_w: float, field_h: float, R: float,
) -> bool:
    """Check that a point is inside the field inset by R."""
    return R <= x <= field_w - R and R <= y <= field_h - R


def _cross_bounding_corners(
    obs: ObstacleGeometry, R: float, margin: float = 1.0,
) -> list[tuple[float, float]]:
    """Return the 4 corners of the inflated cross bounding box + margin."""
    cx, cy = obs.center_x_cm, obs.center_y_cm
    hs = obs.half_size_cm
    dx = hs + R + margin
    dy = hs + R + margin
    return [
        (cx - dx, cy - dy),
        (cx + dx, cy - dy),
        (cx + dx, cy + dy),
        (cx - dx, cy + dy),
    ]


def _dist(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.hypot(x1 - x0, y1 - y0)


def _point_segment_distance(
    px: float,
    py: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    """Return the shortest distance from a point to a line segment."""
    dx = x1 - x0
    dy = y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return _dist(px, py, x0, y0)
    t = ((px - x0) * dx + (py - y0) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    closest_x = x0 + t * dx
    closest_y = y0 + t * dy
    return _dist(px, py, closest_x, closest_y)


def _normalize_angle(theta_rad: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return (theta_rad + math.pi) % (2.0 * math.pi) - math.pi


def _field_cm_to_mask_cell(
    result: PickupGeometryResult,
    x_cm: float,
    y_cm: float,
) -> tuple[int, int] | None:
    """Convert bottom-left field cm to legal-mask row/col."""
    grid_h, grid_w = result.legal_region_mask.shape[:2]
    col = round(x_cm)
    row = round(result.field_height_cm - y_cm)
    if 0 <= row < grid_h and 0 <= col < grid_w:
        return row, col
    return None


def _origin_is_legal(
    result: PickupGeometryResult,
    x_cm: float,
    y_cm: float,
) -> bool:
    """Return whether a robot origin is inside the precomputed legal region."""
    cell = _field_cm_to_mask_cell(result, x_cm, y_cm)
    return cell is not None and bool(result.legal_region_mask[cell[0], cell[1]])


def _heading_for_origin_to_ball(
    result: PickupGeometryResult,
    origin_x: float,
    origin_y: float,
    ball_x: float,
    ball_y: float,
) -> float | None:
    """Compute robot heading that puts the tube vector from origin to ball."""
    dx = ball_x - origin_x
    dy = ball_y - origin_y
    if math.hypot(dx, dy) <= 1e-6:
        return None
    tube_angle_rad = math.atan2(result.tube_right_cm, result.tube_forward_cm)
    return _normalize_angle(math.atan2(dy, dx) + tube_angle_rad)


def _candidate_from_origin(
    result: PickupGeometryResult,
    x_cm: float,
    y_cm: float,
    *,
    from_ring_intersection: bool,
) -> StationCandidate | None:
    """Build a station candidate from an origin, covering every ball in reach."""
    if not _origin_is_legal(result, x_cm, y_cm):
        return None

    tolerance = max(0.0, result.mouth_radius_cm) + 1e-6
    pickup_poses: list[tuple[int, HybridPose, float]] = []

    for ball_idx, ball in enumerate(result.balls):
        if not ball.reachable:
            continue
        distance = _dist(x_cm, y_cm, ball.ball_x_cm, ball.ball_y_cm)
        reach_offset = distance - result.ring_radius_cm
        if abs(reach_offset) > tolerance:
            continue
        heading = _heading_for_origin_to_ball(
            result, x_cm, y_cm, ball.ball_x_cm, ball.ball_y_cm,
        )
        if heading is None:
            continue
        pickup_poses.append((
            ball_idx,
            HybridPose(x_cm=x_cm, y_cm=y_cm, theta_rad=heading),
            abs(reach_offset),
        ))

    if not pickup_poses:
        return None

    pickup_poses.sort(key=lambda item: (item[0], item[1].theta_rad))
    first_ball_idx, first_pose, _ = pickup_poses[0]
    source_pickup = PickupPose(
        x_cm=x_cm,
        y_cm=y_cm,
        theta_rad=first_pose.theta_rad,
        reach_offset_cm=_dist(
            x_cm, y_cm,
            result.balls[first_ball_idx].ball_x_cm,
            result.balls[first_ball_idx].ball_y_cm,
        ) - result.ring_radius_cm,
    )
    return StationCandidate(
        x_cm=x_cm,
        y_cm=y_cm,
        source_pickup=source_pickup,
        pickup_poses_by_ball=tuple((ball_idx, pose) for ball_idx, pose, _ in pickup_poses),
        total_abs_reach_offset_cm=sum(offset for _, _, offset in pickup_poses),
        from_ring_intersection=from_ring_intersection,
    )


def _circle_intersections(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    radius: float,
) -> tuple[tuple[float, float], ...]:
    """Return exact intersections of two same-radius pickup-origin rings."""
    dx = bx - ax
    dy = by - ay
    d = math.hypot(dx, dy)
    if d <= 1e-6 or d > (2.0 * radius) + 1e-6:
        return ()

    half = d * 0.5
    h_sq = radius * radius - half * half
    if h_sq < -1e-6:
        return ()

    h = math.sqrt(max(0.0, h_sq))
    mid_x = (ax + bx) * 0.5
    mid_y = (ay + by) * 0.5
    ux = dx / d
    uy = dy / d
    perp_x = -uy
    perp_y = ux
    first = (mid_x + perp_x * h, mid_y + perp_y * h)
    second = (mid_x - perp_x * h, mid_y - perp_y * h)
    if _dist(first[0], first[1], second[0], second[1]) <= 1e-6:
        return (first,)
    return (first, second)


def _path_distance(points: tuple[tuple[float, float], ...]) -> float:
    """Return total Euclidean length of a polyline."""
    return sum(
        _dist(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
        for i in range(len(points) - 1)
    )


def _polyline_clears_inflated_cross(
    points: tuple[tuple[float, float], ...],
    obs: ObstacleGeometry,
    R: float,
) -> bool:
    """Return whether a polyline avoids the cross except endpoint escape.

    If the original start is already inside the analytic inflated cross, the
    first segment is allowed to leave it.  Likewise, if the original end is
    inside, the final segment is allowed to enter it.  This keeps unavoidable
    endpoint-owned intersections from making every detour impossible while
    still requiring all middle segments to clear the cross.
    """
    if len(points) < 2:
        return True

    start_inside = _point_inside_inflated_cross(points[0][0], points[0][1], obs, R)
    end_inside = _point_inside_inflated_cross(points[-1][0], points[-1][1], obs, R)
    last_segment_index = len(points) - 2

    for i in range(len(points) - 1):
        current = points[i]
        next_point = points[i + 1]
        if not _segment_hits_inflated_cross(
            current[0], current[1], next_point[0], next_point[1], obs, R,
        ):
            continue

        next_inside = _point_inside_inflated_cross(next_point[0], next_point[1], obs, R)
        current_inside = _point_inside_inflated_cross(current[0], current[1], obs, R)
        leaving_start = i == 0 and start_inside and not next_inside
        entering_end = i == last_segment_index and end_inside and not current_inside
        if leaving_start or entering_end:
            continue
        return False

    return True


def _point_inside_ball_obstacle(
    x: float,
    y: float,
    obstacle: BallObstacle,
) -> bool:
    """Return whether a point is inside a ball danger circle."""
    return _dist(x, y, obstacle.x_cm, obstacle.y_cm) <= obstacle.radius_cm


def _point_inside_any_ball_obstacle(
    x: float,
    y: float,
    obstacles: tuple[BallObstacle, ...],
) -> bool:
    """Return whether a point is inside any active ball danger circle."""
    return any(_point_inside_ball_obstacle(x, y, obstacle) for obstacle in obstacles)


def _segment_hits_ball_obstacle(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    obstacle: BallObstacle,
) -> bool:
    """Return whether a segment crosses a ball danger circle."""
    return (
        _point_segment_distance(obstacle.x_cm, obstacle.y_cm, x0, y0, x1, y1)
        <= obstacle.radius_cm
    )


def _segment_hits_any_ball_obstacle(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    obstacles: tuple[BallObstacle, ...],
) -> bool:
    """Return whether a segment crosses any active ball danger circle."""
    return any(
        _segment_hits_ball_obstacle(x0, y0, x1, y1, obstacle)
        for obstacle in obstacles
    )


def _segment_exits_ball_obstacle(
    p0: tuple[float, float],
    p1: tuple[float, float],
    obstacle: BallObstacle,
) -> bool:
    """Return whether a segment moves monotonically out of a start-owned ball zone."""
    if not _point_inside_ball_obstacle(p0[0], p0[1], obstacle):
        return False
    d0 = _dist(p0[0], p0[1], obstacle.x_cm, obstacle.y_cm)
    d1 = _dist(p1[0], p1[1], obstacle.x_cm, obstacle.y_cm)
    if d1 < d0 - 1e-9:
        return False
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    cx = obstacle.x_cm - p0[0]
    cy = obstacle.y_cm - p0[1]
    return (dx * cx + dy * cy) <= 1e-9


def _find_detour_path(
    x0: float, y0: float, x1: float, y1: float,
    obs: ObstacleGeometry, R: float,
    field_w: float, field_h: float,
) -> tuple[tuple[float, float], ...] | None:
    """Find the shortest one- to three-corner route around the inflated cross."""
    corners = tuple(
        corner for corner in _cross_bounding_corners(obs, R)
        if _point_inside_field(corner[0], corner[1], field_w, field_h, R)
    )
    best_path: tuple[tuple[float, float], ...] | None = None
    best_dist = math.inf

    for waypoint_count in range(1, min(3, len(corners)) + 1):
        for waypoints in permutations(corners, waypoint_count):
            path = ((x0, y0), *waypoints, (x1, y1))
            if not _polyline_clears_inflated_cross(path, obs, R):
                continue
            distance = _path_distance(path)
            if distance < best_dist:
                best_dist = distance
                best_path = waypoints

    return best_path


def _ball_obstacles_for_indices(
    geometry_result: PickupGeometryResult,
    ball_indices: set[int],
    danger_radius_cm: float,
) -> tuple[BallObstacle, ...]:
    """Build hard circular obstacles for the active uncollected balls."""
    obstacles: list[BallObstacle] = []
    for ball_idx in sorted(ball_indices):
        ball = geometry_result.balls[ball_idx]
        obstacles.append(BallObstacle(
            ball_index=ball_idx,
            x_cm=ball.ball_x_cm,
            y_cm=ball.ball_y_cm,
            radius_cm=danger_radius_cm,
        ))
    return tuple(obstacles)


def _ball_detour_waypoints(
    obstacles: tuple[BallObstacle, ...],
    field_w: float,
    field_h: float,
    robot_radius_cm: float,
    samples_per_ball: int = 12,
    waypoint_margin_cm: float = 1.0,
) -> tuple[tuple[float, float], ...]:
    """Sample deterministic visibility-graph waypoints around ball obstacles."""
    waypoints: list[tuple[float, float]] = []
    for obstacle in obstacles:
        waypoint_radius = obstacle.radius_cm + waypoint_margin_cm
        for i in range(samples_per_ball):
            theta = 2.0 * math.pi * i / samples_per_ball
            x_cm = obstacle.x_cm + math.cos(theta) * waypoint_radius
            y_cm = obstacle.y_cm + math.sin(theta) * waypoint_radius
            if not _point_inside_field(x_cm, y_cm, field_w, field_h, robot_radius_cm):
                continue
            if _point_inside_any_ball_obstacle(x_cm, y_cm, obstacles):
                continue
            waypoints.append((x_cm, y_cm))
    return tuple(waypoints)


def _segment_valid_with_ball_obstacles(
    p0: tuple[float, float],
    p1: tuple[float, float],
    obs: ObstacleGeometry,
    robot_radius_cm: float,
    ball_obstacles: tuple[BallObstacle, ...],
    escape_obstacles: tuple[BallObstacle, ...] = (),
) -> bool:
    """Return whether a segment clears the cross and all active balls."""
    if not _polyline_clears_inflated_cross((p0, p1), obs, robot_radius_cm):
        return False
    escape_indices = {obstacle.ball_index for obstacle in escape_obstacles}
    for obstacle in ball_obstacles:
        if obstacle.ball_index in escape_indices and _segment_exits_ball_obstacle(
            p0, p1, obstacle,
        ):
            continue
        if _segment_hits_ball_obstacle(p0[0], p0[1], p1[0], p1[1], obstacle):
            return False
    return True


def _find_grid_detour_path_with_ball_obstacles(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    obs: ObstacleGeometry,
    robot_radius_cm: float,
    field_w: float,
    field_h: float,
    ball_obstacles: tuple[BallObstacle, ...],
    escape_obstacles: tuple[BallObstacle, ...],
    resolution_cm: float = 4.0,
    max_expansions: int = 5000,
) -> tuple[tuple[float, float], ...] | None:
    """Find a coarse global detour path for the first leg when local visibility fails."""
    if not _point_inside_field(x0, y0, field_w, field_h, robot_radius_cm):
        return None
    if not _point_inside_field(x1, y1, field_w, field_h, robot_radius_cm):
        return None

    def to_cell(x_cm: float, y_cm: float) -> tuple[int, int]:
        return (round(x_cm / resolution_cm), round(y_cm / resolution_cm))

    def to_point(cell: tuple[int, int]) -> tuple[float, float]:
        return (
            min(field_w - robot_radius_cm, max(robot_radius_cm, cell[0] * resolution_cm)),
            min(field_h - robot_radius_cm, max(robot_radius_cm, cell[1] * resolution_cm)),
        )

    start_cell = to_cell(x0, y0)
    end_cell = to_cell(x1, y1)
    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heappush(open_heap, (_dist(x0, y0, x1, y1), 0.0, start_cell))
    best_cost: dict[tuple[int, int], float] = {start_cell: 0.0}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    visited: set[tuple[int, int]] = set()
    max_x = math.ceil(field_w / resolution_cm)
    max_y = math.ceil(field_h / resolution_cm)
    neighbor_steps = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    expansions = 0
    while open_heap and expansions < max_expansions:
        _priority, cost, cell = heappop(open_heap)
        if cell in visited:
            continue
        visited.add(cell)
        expansions += 1
        if cell == end_cell:
            break

        p0 = (x0, y0) if cell == start_cell else to_point(cell)
        for dx, dy in neighbor_steps:
            neighbor = (cell[0] + dx, cell[1] + dy)
            if not (0 <= neighbor[0] <= max_x and 0 <= neighbor[1] <= max_y):
                continue
            p1 = (x1, y1) if neighbor == end_cell else to_point(neighbor)
            if not _point_inside_field(p1[0], p1[1], field_w, field_h, robot_radius_cm):
                continue
            if not _segment_valid_with_ball_obstacles(
                p0,
                p1,
                obs,
                robot_radius_cm,
                ball_obstacles,
                escape_obstacles,
            ):
                continue
            next_cost = cost + _dist(p0[0], p0[1], p1[0], p1[1])
            if next_cost >= best_cost.get(neighbor, math.inf):
                continue
            best_cost[neighbor] = next_cost
            previous[neighbor] = cell
            heuristic = _dist(p1[0], p1[1], x1, y1)
            heappush(open_heap, (next_cost + heuristic, next_cost, neighbor))

    if end_cell not in best_cost:
        return None

    cells: list[tuple[int, int]] = []
    current = end_cell
    while current != start_cell:
        cells.append(current)
        current = previous[current]
    cells.append(start_cell)
    cells.reverse()

    points = [
        (x0, y0) if cell == start_cell else (x1, y1) if cell == end_cell else to_point(cell)
        for cell in cells
    ]

    simplified: list[tuple[float, float]] = [points[0]]
    index = 0
    while index < len(points) - 1:
        best_next = index + 1
        for candidate_index in range(len(points) - 1, index, -1):
            if _segment_valid_with_ball_obstacles(
                simplified[-1],
                points[candidate_index],
                obs,
                robot_radius_cm,
                ball_obstacles,
                escape_obstacles,
            ):
                best_next = candidate_index
                break
        simplified.append(points[best_next])
        index = best_next

    return tuple(simplified[1:-1])


def _find_detour_path_with_ball_obstacles(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    obs: ObstacleGeometry,
    robot_radius_cm: float,
    field_w: float,
    field_h: float,
    ball_obstacles: tuple[BallObstacle, ...],
    allow_start_ball_escape: bool = False,
) -> tuple[tuple[float, float], ...] | None:
    """Find a detour path that clears the cross and active ball obstacles."""
    direct_raw_blocked = (
        _segment_hits_inflated_cross(x0, y0, x1, y1, obs, robot_radius_cm)
        or _segment_hits_any_ball_obstacle(x0, y0, x1, y1, ball_obstacles)
    )
    if not direct_raw_blocked:
        return ()

    start_escape_obstacles = tuple(
        obstacle for obstacle in ball_obstacles
        if _point_inside_ball_obstacle(x0, y0, obstacle)
    )
    if start_escape_obstacles and not allow_start_ball_escape:
        return None
    if _point_inside_any_ball_obstacle(x1, y1, ball_obstacles):
        return None

    cross_waypoints = tuple(
        corner for corner in _cross_bounding_corners(obs, robot_radius_cm)
        if _point_inside_field(corner[0], corner[1], field_w, field_h, robot_radius_cm)
        and not _point_inside_any_ball_obstacle(corner[0], corner[1], ball_obstacles)
    )
    ball_waypoints = _ball_detour_waypoints(
        ball_obstacles,
        field_w,
        field_h,
        robot_radius_cm,
    )
    candidate_points = cross_waypoints + tuple(
        point for point in ball_waypoints
        if not _point_inside_inflated_cross(point[0], point[1], obs, robot_radius_cm)
    )
    nodes = ((x0, y0), *candidate_points, (x1, y1))
    start_idx = 0
    end_idx = len(nodes) - 1

    distances = [math.inf] * len(nodes)
    previous: list[int | None] = [None] * len(nodes)
    visited: set[int] = set()
    distances[start_idx] = 0.0

    while len(visited) < len(nodes):
        current = min(
            (idx for idx in range(len(nodes)) if idx not in visited),
            key=lambda idx: distances[idx],
            default=-1,
        )
        if current < 0 or not math.isfinite(distances[current]):
            break
        if current == end_idx:
            break
        visited.add(current)

        for neighbor in range(len(nodes)):
            if neighbor == current or neighbor in visited:
                continue
            if current == start_idx and neighbor == end_idx and direct_raw_blocked:
                continue
            p0 = nodes[current]
            p1 = nodes[neighbor]
            escape_obstacles = start_escape_obstacles if current == start_idx else ()
            if not _segment_valid_with_ball_obstacles(
                p0, p1, obs, robot_radius_cm, ball_obstacles, escape_obstacles,
            ):
                continue
            candidate_dist = distances[current] + _dist(p0[0], p0[1], p1[0], p1[1])
            if candidate_dist < distances[neighbor]:
                distances[neighbor] = candidate_dist
                previous[neighbor] = current

    if not math.isfinite(distances[end_idx]):
        if allow_start_ball_escape:
            return _find_grid_detour_path_with_ball_obstacles(
                x0,
                y0,
                x1,
                y1,
                obs,
                robot_radius_cm,
                field_w,
                field_h,
                ball_obstacles,
                start_escape_obstacles,
            )
        return None

    path_indices: list[int] = []
    current_idx: int | None = end_idx
    while current_idx is not None:
        path_indices.append(current_idx)
        current_idx = previous[current_idx]
    path_indices.reverse()
    return tuple(nodes[idx] for idx in path_indices[1:-1])


def _edge_from_detour_path(
    from_index: int,
    to_index: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    blocked: bool,
    detour_path: tuple[tuple[float, float], ...] | None,
) -> RouteEdge:
    """Create a RouteEdge from an optional detour waypoint path."""
    direct = _dist(x0, y0, x1, y1)
    detour_wp: HybridPose | None = None
    detour_wps: tuple[HybridPose, ...] = ()
    total = direct

    if blocked:
        if detour_path is None:
            total = math.inf
        else:
            detour_poses: list[HybridPose] = []
            for index, waypoint in enumerate(detour_path):
                next_point = detour_path[index + 1] if index + 1 < len(detour_path) else (x1, y1)
                theta = math.atan2(next_point[1] - waypoint[1], next_point[0] - waypoint[0])
                detour_poses.append(HybridPose(
                    x_cm=waypoint[0], y_cm=waypoint[1], theta_rad=theta,
                ))
            detour_wps = tuple(detour_poses)
            detour_wp = detour_wps[0] if detour_wps else None
            total = _path_distance(((x0, y0), *detour_path, (x1, y1)))

    return RouteEdge(
        from_index=from_index,
        to_index=to_index,
        direct_distance_cm=direct,
        blocked=blocked,
        detour_waypoint=detour_wp,
        total_distance_cm=total,
        detour_waypoints=detour_wps,
    )


# ---------------------------------------------------------------------------
# Set-cover
# ---------------------------------------------------------------------------

def _greedy_set_cover(
    geometry_result: PickupGeometryResult,
) -> list[CoverPoint]:
    """Greedy set-cover: pick poses that collectively cover all reachable balls.

    For each candidate PickupPose position (quantized to 0.1 cm), we check
    which ball indices it covers.  A single position can cover multiple balls
    if they share a ring sample at the same quantized location (twin sweep).

    Ties are broken by preferring exact-ring poses (reach_offset_cm == 0).
    """
    balls = geometry_result.balls

    # Build inverse map: quantized position -> per-ball best pickup pose.
    pos_map: dict[tuple[int, int], dict[int, PickupPose]] = {}

    for ball_idx, ball in enumerate(balls):
        if not ball.reachable:
            continue
        for pose in ball.valid_points:
            key = _quantize(pose.x_cm, pose.y_cm)
            by_ball = pos_map.setdefault(key, {})
            existing_pose = by_ball.get(ball_idx)
            if (
                existing_pose is None
                or abs(pose.reach_offset_cm) < abs(existing_pose.reach_offset_cm)
            ):
                by_ball[ball_idx] = pose

    uncovered: set[int] = {
        i for i, b in enumerate(balls) if b.reachable
    }
    cover_points: list[CoverPoint] = []

    while uncovered:
        best_key: tuple[int, int] | None = None
        best_count = 0
        best_exact = False

        for key, poses_by_ball in pos_map.items():
            indices = set(poses_by_ball)
            overlap = len(indices & uncovered)
            exact = any(
                poses_by_ball[index].reach_offset_cm == 0.0
                for index in indices & uncovered
            )
            if overlap > best_count or (overlap == best_count and exact and not best_exact):
                best_key = key
                best_count = overlap
                best_exact = exact

        if best_key is None or best_count == 0:
            break

        poses_by_ball = pos_map[best_key]
        indices = set(poses_by_ball)
        covered = _sort_ball_indices_orange_first(
            geometry_result,
            indices & uncovered,
        )
        pickup_poses = tuple(
            HybridPose(
                x_cm=poses_by_ball[index].x_cm,
                y_cm=poses_by_ball[index].y_cm,
                theta_rad=poses_by_ball[index].theta_rad,
            )
            for index in covered
        )
        source_pickup = poses_by_ball[covered[0]]

        cover_points.append(CoverPoint(
            pose=pickup_poses[0],
            source_pickup=source_pickup,
            covered_ball_indices=covered,
            pickup_poses=pickup_poses,
        ))
        uncovered -= indices

    return cover_points


def _better_candidate(
    candidate: StationCandidate,
    current: StationCandidate | None,
) -> bool:
    """Return whether candidate should replace current for one quantized origin."""
    if current is None:
        return True
    candidate_count = len(candidate.covered_ball_indices)
    current_count = len(current.covered_ball_indices)
    if candidate_count != current_count:
        return candidate_count > current_count
    if candidate.from_ring_intersection != current.from_ring_intersection:
        return candidate.from_ring_intersection
    return candidate.total_abs_reach_offset_cm < current.total_abs_reach_offset_cm


def _intersection_station_candidates(
    geometry_result: PickupGeometryResult,
) -> list[StationCandidate]:
    """Generate shared-origin candidates from ring intersections plus fallbacks."""
    candidates_by_origin: dict[tuple[int, int], StationCandidate] = {}
    balls = geometry_result.balls
    R = geometry_result.ring_radius_cm

    def add_candidate(candidate: StationCandidate | None) -> None:
        if candidate is None:
            return
        key = _quantize(candidate.x_cm, candidate.y_cm)
        current = candidates_by_origin.get(key)
        if _better_candidate(candidate, current):
            candidates_by_origin[key] = candidate

    # First add analytical intersections between every pair of pickup rings.
    for i, a in enumerate(balls):
        if not a.reachable:
            continue
        for j in range(i + 1, len(balls)):
            b = balls[j]
            if not b.reachable:
                continue
            for x_cm, y_cm in _circle_intersections(
                a.ball_x_cm, a.ball_y_cm, b.ball_x_cm, b.ball_y_cm, R,
            ):
                add_candidate(
                    _candidate_from_origin(
                        geometry_result,
                        x_cm,
                        y_cm,
                        from_ring_intersection=True,
                    )
                )

    # Then add every sampled valid pickup point as a one-ball-safe fallback.
    for ball in balls:
        if not ball.reachable:
            continue
        for pose in ball.valid_points:
            add_candidate(
                _candidate_from_origin(
                    geometry_result,
                    pose.x_cm,
                    pose.y_cm,
                    from_ring_intersection=False,
                )
            )

    return list(candidates_by_origin.values())


def _spread_pickup_poses(
    poses: tuple[PickupPose, ...],
    max_count: int,
) -> tuple[PickupPose, ...]:
    """Return a deterministic spread of pickup poses around one ball."""
    if len(poses) <= max_count:
        return tuple(sorted(poses, key=lambda pose: (abs(pose.reach_offset_cm), pose.theta_rad)))

    min_offset = min(abs(pose.reach_offset_cm) for pose in poses)
    best_ring = sorted(
        (pose for pose in poses if abs(abs(pose.reach_offset_cm) - min_offset) <= 1e-9),
        key=lambda pose: pose.theta_rad,
    )
    source = best_ring if len(best_ring) >= max_count else sorted(
        poses,
        key=lambda pose: (abs(pose.reach_offset_cm), pose.theta_rad),
    )
    if len(source) <= max_count:
        return tuple(source)

    selected: list[PickupPose] = []
    used: set[int] = set()
    for i in range(max_count):
        index = round(i * (len(source) - 1) / (max_count - 1))
        while index in used and index + 1 < len(source):
            index += 1
        while index in used and index > 0:
            index -= 1
        used.add(index)
        selected.append(source[index])
    return tuple(selected)


def _shared_node_station_candidates(
    geometry_result: PickupGeometryResult,
) -> list[StationCandidate]:
    """Generate multi-ball intersection nodes plus bounded isolated-ball fallbacks."""
    candidates_by_origin: dict[tuple[int, int], StationCandidate] = {}
    balls = geometry_result.balls
    R = geometry_result.ring_radius_cm

    def add_candidate(candidate: StationCandidate | None) -> None:
        if candidate is None or len(candidate.covered_ball_indices) < 2:
            return
        key = _quantize(candidate.x_cm, candidate.y_cm)
        current = candidates_by_origin.get(key)
        if _better_candidate(candidate, current):
            candidates_by_origin[key] = candidate

    for i, a in enumerate(balls):
        if not a.reachable:
            continue
        for j in range(i + 1, len(balls)):
            b = balls[j]
            if not b.reachable:
                continue
            for x_cm, y_cm in _circle_intersections(
                a.ball_x_cm, a.ball_y_cm, b.ball_x_cm, b.ball_y_cm, R,
            ):
                add_candidate(
                    _candidate_from_origin(
                        geometry_result,
                        x_cm,
                        y_cm,
                        from_ring_intersection=True,
                    )
                )

    shared_candidates = list(candidates_by_origin.values())
    shared_covered = {
        ball_idx
        for candidate in shared_candidates
        for ball_idx in candidate.covered_ball_indices
    }

    fallback_candidates_by_origin: dict[tuple[int, int], StationCandidate] = {}
    for ball_idx, ball in enumerate(balls):
        if not ball.reachable or ball_idx in shared_covered:
            continue
        for pose in _spread_pickup_poses(
            ball.valid_points,
            ISOLATED_BALL_FALLBACK_CANDIDATES,
        ):
            candidate = _candidate_from_origin(
                geometry_result,
                pose.x_cm,
                pose.y_cm,
                from_ring_intersection=False,
            )
            if candidate is None:
                continue
            key = _quantize(candidate.x_cm, candidate.y_cm)
            current = fallback_candidates_by_origin.get(key)
            if _better_candidate(candidate, current):
                fallback_candidates_by_origin[key] = candidate

    return [
        *sorted(
            shared_candidates,
            key=lambda candidate: (
                candidate.x_cm,
                candidate.y_cm,
                candidate.total_abs_reach_offset_cm,
            ),
        ),
        *sorted(
            fallback_candidates_by_origin.values(),
            key=lambda candidate: (
                min(candidate.covered_ball_indices),
                candidate.total_abs_reach_offset_cm,
                candidate.x_cm,
                candidate.y_cm,
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Graph + nearest-neighbor
# ---------------------------------------------------------------------------

def _build_edges(
    nodes: list[HybridPose],
    obs: ObstacleGeometry,
    R: float,
    field_w: float,
    field_h: float,
) -> dict[tuple[int, int], RouteEdge]:
    """Build complete graph edges with collision + detour information."""
    edges: dict[tuple[int, int], RouteEdge] = {}
    n = len(nodes)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x0, y0 = nodes[i].x_cm, nodes[i].y_cm
            x1, y1 = nodes[j].x_cm, nodes[j].y_cm
            direct = _dist(x0, y0, x1, y1)
            blocked = _segment_hits_inflated_cross(x0, y0, x1, y1, obs, R)

            detour_wp: HybridPose | None = None
            detour_wps: tuple[HybridPose, ...] = ()
            total = direct

            if blocked:
                detour_path = _find_detour_path(x0, y0, x1, y1, obs, R, field_w, field_h)
                if detour_path is not None:
                    detour_poses: list[HybridPose] = []
                    for index, corner in enumerate(detour_path):
                        next_point = detour_path[index + 1] if index + 1 < len(detour_path) else (x1, y1)
                        theta = math.atan2(next_point[1] - corner[1], next_point[0] - corner[0])
                        detour_poses.append(HybridPose(
                            x_cm=corner[0], y_cm=corner[1], theta_rad=theta,
                        ))
                    detour_wps = tuple(detour_poses)
                    detour_wp = detour_wps[0]
                    total = _path_distance(((x0, y0), *detour_path, (x1, y1)))
                else:
                    total = math.inf  # no valid route

            edges[(i, j)] = RouteEdge(
                from_index=i,
                to_index=j,
                direct_distance_cm=direct,
                blocked=blocked,
                detour_waypoint=detour_wp,
                total_distance_cm=total,
                detour_waypoints=detour_wps,
            )

    return edges


def _build_edge_with_ball_obstacles(
    from_index: int,
    to_index: int,
    from_pose: HybridPose,
    to_pose: HybridPose,
    obs: ObstacleGeometry,
    robot_radius_cm: float,
    field_w: float,
    field_h: float,
    ball_obstacles: tuple[BallObstacle, ...],
    allow_start_ball_escape: bool = False,
) -> RouteEdge:
    """Build one edge using hard active ball obstacles plus the cross."""
    x0, y0 = from_pose.x_cm, from_pose.y_cm
    x1, y1 = to_pose.x_cm, to_pose.y_cm
    direct_blocked = (
        _segment_hits_inflated_cross(x0, y0, x1, y1, obs, robot_radius_cm)
        or _segment_hits_any_ball_obstacle(x0, y0, x1, y1, ball_obstacles)
    )
    detour_path: tuple[tuple[float, float], ...] | None = ()
    if direct_blocked:
        detour_path = _find_detour_path_with_ball_obstacles(
            x0, y0, x1, y1,
            obs, robot_radius_cm,
            field_w, field_h,
            ball_obstacles,
            allow_start_ball_escape=allow_start_ball_escape,
        )
    return _edge_from_detour_path(
        from_index, to_index,
        x0, y0, x1, y1,
        direct_blocked,
        detour_path,
    )


def _build_strategy_edge_with_ball_obstacles(
    inp: RoutePlannerInput,
    from_index: int,
    to_index: int,
    from_pose: HybridPose,
    to_pose: HybridPose,
    ball_obstacles: tuple[BallObstacle, ...],
) -> RouteEdge:
    """Build a strategy edge, allowing only the real first leg to escape start zones."""
    return _build_edge_with_ball_obstacles(
        from_index,
        to_index,
        from_pose,
        to_pose,
        inp.obstacle,
        inp.robot_radius_cm,
        inp.field_width_cm,
        inp.field_height_cm,
        ball_obstacles,
        allow_start_ball_escape=from_index == 0,
    )


def _nearest_neighbor_order(
    n_cover: int,
    edges: dict[tuple[int, int], RouteEdge],
) -> list[int]:
    """Greedy nearest-neighbor ordering starting from node 0 (start pose).

    Returns indices into cover_points (0-based, excluding the start node
    which is graph node 0).  Graph node indices are offset by 1 from
    cover_point indices.
    """
    if n_cover == 0:
        return []

    visited: set[int] = set()
    order: list[int] = []
    current_graph = 0  # start pose

    for _ in range(n_cover):
        best_j = -1
        best_dist = math.inf
        for j in range(1, n_cover + 1):
            if j in visited:
                continue
            edge = edges.get((current_graph, j))
            if edge is not None and edge.total_distance_cm < best_dist:
                best_dist = edge.total_distance_cm
                best_j = j
        if best_j < 0:
            # Remaining nodes unreachable; append in order
            for j in range(1, n_cover + 1):
                if j not in visited:
                    visited.add(j)
                    order.append(j - 1)
            break
        visited.add(best_j)
        order.append(best_j - 1)  # cover_point index
        current_graph = best_j

    return order


def _force_orange_first_order(
    geometry_result: PickupGeometryResult,
    cover_points: list[CoverPoint],
    ordered_indices: list[int],
) -> list[int]:
    """Move the first orange-covering stop to the front when orange is reachable."""
    if not _orange_ball_indices(geometry_result):
        return ordered_indices
    for order_pos, cover_idx in enumerate(ordered_indices):
        if _covers_orange(geometry_result, cover_points[cover_idx].covered_ball_indices):
            return [
                cover_idx,
                *ordered_indices[:order_pos],
                *ordered_indices[order_pos + 1:],
            ]
    return ordered_indices


# ---------------------------------------------------------------------------
# RoutePlan assembly
# ---------------------------------------------------------------------------

def _assemble_route_plan(
    start_pose: HybridPose,
    cover_points: list[CoverPoint],
    ordered_indices: list[int],
    edges: dict[tuple[int, int], RouteEdge],
    unload_pose: HybridPose | None,
    unload_goal_cm: tuple[float, float] | None,
    unload_graph_index: int | None,
) -> RoutePlan:
    """Flatten cover points + ordering into a RoutePlan with id()-matching.

    The same HybridPose instances from cover_points are placed into both
    ``points`` and ``pickup_poses`` so that ``brain/route_interpreter.py``
    can match them via ``id()``.

    When *unload_pose* is provided the final leg routes from the last
    cover point to the unload staging position and sets the RoutePlan
    unload fields so the interpreter appends an UNLOAD step.
    """
    points: list[HybridPose] = [start_pose]
    pickup_poses: list[HybridPose] = []

    def append_edge_detour(edge: RouteEdge | None) -> None:
        if edge is None or not edge.blocked:
            return
        detour_points = edge.detour_waypoints
        if not detour_points and edge.detour_waypoint is not None:
            detour_points = (edge.detour_waypoint,)
        for waypoint in detour_points:
            points.append(waypoint)

    prev_graph = 0  # start node
    for cover_idx in ordered_indices:
        graph_idx = cover_idx + 1
        edge = edges.get((prev_graph, graph_idx))

        append_edge_detour(edge)

        cp = cover_points[cover_idx]
        station_pickup_poses = cp.pickup_poses or (cp.pose,)
        for pickup_pose in station_pickup_poses:
            points.append(pickup_pose)
            pickup_poses.append(pickup_pose)

        prev_graph = graph_idx

    # Final leg: last pickup -> unload staging pose
    if unload_pose is not None and unload_graph_index is not None:
        edge = edges.get((prev_graph, unload_graph_index))
        append_edge_detour(edge)
        points.append(unload_pose)

    return RoutePlan(
        points=points,
        pickup_poses=pickup_poses,
        unload_pose=unload_pose,
        unload_goal_cm=unload_goal_cm,
    )


def _cover_point_from_station_candidate(
    geometry_result: PickupGeometryResult,
    candidate: StationCandidate,
    uncovered: set[int],
) -> CoverPoint | None:
    """Convert a station candidate into a CoverPoint for the remaining balls."""
    poses_by_ball = {
        ball_idx: pose
        for ball_idx, pose in candidate.pickup_poses_by_ball
        if ball_idx in uncovered
    }
    pickup_pairs = tuple(
        (ball_idx, poses_by_ball[ball_idx])
        for ball_idx in _sort_ball_indices_orange_first(
            geometry_result,
            tuple(poses_by_ball),
        )
    )
    if not pickup_pairs:
        return None
    covered = tuple(ball_idx for ball_idx, _ in pickup_pairs)
    pickup_poses = tuple(pose for _, pose in pickup_pairs)
    return CoverPoint(
        pose=pickup_poses[0],
        source_pickup=candidate.source_pickup,
        covered_ball_indices=covered,
        pickup_poses=pickup_poses,
    )


def _intersection_candidate_score(
    travel_distance_cm: float,
    candidate: StationCandidate,
    covered_count: int,
) -> float:
    """Score an intersection route candidate, keeping travel distance dominant."""
    extra_ball_count = max(0, covered_count - 1)
    score = travel_distance_cm
    score -= extra_ball_count * INTERSECTION_EXTRA_BALL_BONUS_CM
    if covered_count > 1 and candidate.from_ring_intersection:
        score -= INTERSECTION_SHARED_ORIGIN_BONUS_CM
    score += (
        candidate.total_abs_reach_offset_cm
        * INTERSECTION_REACH_OFFSET_PENALTY
    )
    return score


def _reachable_ball_mask(geometry_result: PickupGeometryResult) -> int:
    """Return a bit mask for every reachable ball."""
    mask = 0
    for ball_idx, ball in enumerate(geometry_result.balls):
        if ball.reachable:
            mask |= 1 << ball_idx
    return mask


def _candidate_ball_mask(candidate: StationCandidate) -> int:
    """Return the full ball coverage mask for one station candidate."""
    mask = 0
    for ball_idx in candidate.covered_ball_indices:
        mask |= 1 << ball_idx
    return mask


def _mask_to_indices(mask: int) -> set[int]:
    """Convert a ball bit mask into a set of ball indices."""
    indices: set[int] = set()
    ball_idx = 0
    while mask:
        if mask & 1:
            indices.add(ball_idx)
        mask >>= 1
        ball_idx += 1
    return indices


def _station_candidate_pose(candidate: StationCandidate) -> HybridPose:
    """Return a pose at the candidate origin for distance-only search edges."""
    return HybridPose(
        x_cm=candidate.x_cm,
        y_cm=candidate.y_cm,
        theta_rad=candidate.source_pickup.theta_rad,
    )


def _build_route_from_candidate_sequence(
    inp: RoutePlannerInput,
    sequence: list[StationCandidate],
) -> RouteStrategyResult:
    """Assemble a RouteStrategyResult from an already chosen candidate sequence."""
    uncovered: set[int] = {
        i for i, b in enumerate(inp.geometry_result.balls) if b.reachable
    }
    cover_points: list[CoverPoint] = []
    ordered: list[int] = []
    edges: dict[tuple[int, int], RouteEdge] = {}
    current_pose = inp.start_pose
    current_graph = 0

    for candidate in sequence:
        cover_point = _cover_point_from_station_candidate(
            inp.geometry_result,
            candidate,
            uncovered,
        )
        if cover_point is None:
            continue
        covered = set(cover_point.covered_ball_indices)
        next_graph = len(cover_points) + 1
        active_obstacles = _ball_obstacles_for_indices(
            inp.geometry_result,
            uncovered - covered,
            inp.robot_radius_cm,
        )
        edge = _build_strategy_edge_with_ball_obstacles(
            inp,
            current_graph,
            next_graph,
            current_pose,
            cover_point.pose,
            active_obstacles,
        )
        if not math.isfinite(edge.total_distance_cm):
            break

        cover_points.append(cover_point)
        ordered.append(len(cover_points) - 1)
        edges[(edge.from_index, edge.to_index)] = edge
        uncovered -= covered
        current_pose = cover_point.pose
        current_graph = edge.to_index

    unload_graph_index: int | None = None
    if inp.unload_pose is not None:
        unload_graph_index = len(cover_points) + 1
        active_obstacles = _ball_obstacles_for_indices(
            inp.geometry_result,
            uncovered,
            inp.robot_radius_cm,
        )
        unload_edge = _build_strategy_edge_with_ball_obstacles(
            inp,
            current_graph,
            unload_graph_index,
            current_pose,
            inp.unload_pose,
            active_obstacles,
        )
        edges[(unload_edge.from_index, unload_edge.to_index)] = unload_edge

    route_plan = _assemble_route_plan(
        inp.start_pose,
        cover_points,
        ordered,
        edges,
        unload_pose=inp.unload_pose,
        unload_goal_cm=inp.unload_goal_cm,
        unload_graph_index=unload_graph_index,
    )
    return RouteStrategyResult(
        cover_points=tuple(cover_points),
        ordered_indices=tuple(ordered),
        edges=tuple(edges.values()),
        route_plan=route_plan,
    )


def _route_strategy_total_distance(result: RouteStrategyResult) -> float:
    """Return total distance along a strategy result's selected route edges."""
    edge_by_nodes = {
        (edge.from_index, edge.to_index): edge
        for edge in result.edges
    }
    total = 0.0
    prev_graph = 0
    for cover_idx in result.ordered_indices:
        graph_idx = cover_idx + 1
        edge = edge_by_nodes.get((prev_graph, graph_idx))
        if edge is None:
            return math.inf
        total += edge.total_distance_cm
        prev_graph = graph_idx
    if result.route_plan.unload_pose is not None:
        unload_edges = [
            edge for edge in result.edges
            if edge.from_index == prev_graph
            and edge.to_index > len(result.cover_points)
        ]
        if len(unload_edges) != 1:
            return math.inf
        total += unload_edges[0].total_distance_cm
    return total


def _candidate_sequence_mask(sequence: list[StationCandidate]) -> int:
    """Return all balls covered by a candidate sequence."""
    mask = 0
    for candidate in sequence:
        mask |= _candidate_ball_mask(candidate)
    return mask


def _prune_candidates_for_optimal_search(
    inp: RoutePlannerInput,
    candidates: list[StationCandidate],
    full_mask: int,
) -> list[StationCandidate]:
    """Bound duplicate same-coverage candidates for the exact route search."""
    by_mask: dict[int, list[StationCandidate]] = {}
    for candidate in candidates:
        mask = _candidate_ball_mask(candidate) & full_mask
        if mask == 0:
            continue
        by_mask.setdefault(mask, []).append(candidate)

    anchors = [inp.start_pose]
    if inp.unload_pose is not None:
        anchors.append(inp.unload_pose)

    selected: list[StationCandidate] = []
    for mask in sorted(by_mask):
        mask_candidates = by_mask[mask]
        if len(mask_candidates) <= INTERSECTION_OPTIMAL_CANDIDATES_PER_MASK:
            selected.extend(mask_candidates)
            continue

        chosen: list[StationCandidate] = []
        for anchor in anchors:
            if len(chosen) >= INTERSECTION_OPTIMAL_CANDIDATES_PER_MASK:
                break
            best = min(
                mask_candidates,
                key=lambda candidate: (
                    _dist(anchor.x_cm, anchor.y_cm, candidate.x_cm, candidate.y_cm),
                    candidate.total_abs_reach_offset_cm,
                    candidate.x_cm,
                    candidate.y_cm,
                ),
            )
            if best not in chosen:
                chosen.append(best)

        for candidate in sorted(
            mask_candidates,
            key=lambda item: (
                item.total_abs_reach_offset_cm,
                item.x_cm,
                item.y_cm,
            ),
        ):
            if len(chosen) >= INTERSECTION_OPTIMAL_CANDIDATES_PER_MASK:
                break
            if candidate not in chosen:
                chosen.append(candidate)
        selected.extend(chosen)

    return selected


def _intersection_nearest_candidate_sequence(
    inp: RoutePlannerInput,
    candidates: list[StationCandidate],
) -> list[StationCandidate]:
    """Choose shared-node candidates by nearest safe edge."""
    uncovered: set[int] = {
        i for i, b in enumerate(inp.geometry_result.balls) if b.reachable
    }
    sequence: list[StationCandidate] = []
    current_pose = inp.start_pose
    orange = set(_orange_ball_indices(inp.geometry_result))

    while uncovered:
        best_candidate: StationCandidate | None = None
        best_cover_point: CoverPoint | None = None
        best_key: tuple[float, float, float, float] | None = None
        require_orange = not sequence and bool(orange & uncovered)

        for candidate in candidates:
            cover_point = _cover_point_from_station_candidate(
                inp.geometry_result,
                candidate,
                uncovered,
            )
            if cover_point is None:
                continue
            covered = set(cover_point.covered_ball_indices)
            if require_orange and not covered & orange:
                continue
            active_obstacles = _ball_obstacles_for_indices(
                inp.geometry_result,
                uncovered - covered,
                inp.robot_radius_cm,
            )
            edge = _build_strategy_edge_with_ball_obstacles(
                inp,
                0 if not sequence else 1,
                1,
                current_pose,
                cover_point.pose,
                active_obstacles,
            )
            if not math.isfinite(edge.total_distance_cm):
                continue
            key = (
                edge.total_distance_cm,
                candidate.total_abs_reach_offset_cm,
                candidate.x_cm,
                candidate.y_cm,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_candidate = candidate
                best_cover_point = cover_point

        if best_candidate is None or best_cover_point is None:
            break

        sequence.append(best_candidate)
        uncovered -= set(best_cover_point.covered_ball_indices)
        current_pose = best_cover_point.pose

    return sequence


def _plan_intersection_nearest_route(inp: RoutePlannerInput) -> RouteStrategyResult:
    """Build a route by nearest-first traversal of shared pickup nodes."""
    candidates = _shared_node_station_candidates(inp.geometry_result)
    sequence = _intersection_nearest_candidate_sequence(inp, candidates)
    return _build_route_from_candidate_sequence(inp, sequence)


def _plan_intersection_optimal_route(inp: RoutePlannerInput) -> RouteStrategyResult:
    """Find the shortest safe route through the shared-node candidate space."""
    candidates = _shared_node_station_candidates(inp.geometry_result)
    if not candidates:
        return _build_route_from_candidate_sequence(inp, [])

    full_mask = _reachable_ball_mask(inp.geometry_result)
    candidates = _prune_candidates_for_optimal_search(inp, candidates, full_mask)
    candidate_masks = [
        _candidate_ball_mask(candidate) & full_mask
        for candidate in candidates
    ]
    orange_mask = 0
    for ball_idx in _orange_ball_indices(inp.geometry_result):
        orange_mask |= 1 << ball_idx
    candidate_poses = [
        _station_candidate_pose(candidate)
        for candidate in candidates
    ]
    incumbent_sequence = _intersection_nearest_candidate_sequence(inp, candidates)
    incumbent_result = _build_route_from_candidate_sequence(inp, incumbent_sequence)
    incumbent_mask = _candidate_sequence_mask(incumbent_sequence) & full_mask
    if incumbent_mask == full_mask:
        best_terminal_cost = _route_strategy_total_distance(incumbent_result)
    else:
        best_terminal_cost = math.inf

    start_state = (0, -1)
    best_cost: dict[tuple[int, int], float] = {start_state: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    heap: list[tuple[float, int, int]] = [(0.0, 0, -1)]
    edge_cache: dict[tuple[int, int, int], float] = {}
    unload_cache: dict[tuple[int, int], float] = {}
    obstacle_cache: dict[int, tuple[BallObstacle, ...]] = {}
    best_terminal_state: tuple[int, int] | None = None
    best_partial_state = start_state
    best_partial_key = (0, -0.0)

    def pose_for_position(position_idx: int) -> HybridPose:
        if position_idx < 0:
            return inp.start_pose
        return candidate_poses[position_idx]

    def obstacles_for_mask(active_mask: int) -> tuple[BallObstacle, ...]:
        cached = obstacle_cache.get(active_mask)
        if cached is not None:
            return cached
        obstacles = _ball_obstacles_for_indices(
            inp.geometry_result,
            _mask_to_indices(active_mask),
            inp.robot_radius_cm,
        )
        obstacle_cache[active_mask] = obstacles
        return obstacles

    def edge_distance(from_idx: int, to_idx: int, active_mask: int) -> float:
        key = (from_idx, to_idx, active_mask)
        cached = edge_cache.get(key)
        if cached is not None:
            return cached
        edge = _build_edge_with_ball_obstacles(
            0,
            1,
            pose_for_position(from_idx),
            candidate_poses[to_idx],
            inp.obstacle,
            inp.robot_radius_cm,
            inp.field_width_cm,
            inp.field_height_cm,
            obstacles_for_mask(active_mask),
            allow_start_ball_escape=from_idx < 0,
        )
        edge_cache[key] = edge.total_distance_cm
        return edge.total_distance_cm

    def unload_distance(from_idx: int, active_mask: int) -> float:
        if inp.unload_pose is None:
            return 0.0
        key = (from_idx, active_mask)
        cached = unload_cache.get(key)
        if cached is not None:
            return cached
        edge = _build_edge_with_ball_obstacles(
            0,
            1,
            pose_for_position(from_idx),
            inp.unload_pose,
            inp.obstacle,
            inp.robot_radius_cm,
            inp.field_width_cm,
            inp.field_height_cm,
            obstacles_for_mask(active_mask),
            allow_start_ball_escape=from_idx < 0,
        )
        unload_cache[key] = edge.total_distance_cm
        return edge.total_distance_cm

    while heap:
        cost, mask, position_idx = heappop(heap)
        state = (mask, position_idx)
        if cost != best_cost.get(state):
            continue
        if cost >= best_terminal_cost:
            break

        partial_key = (mask.bit_count(), -cost)
        if partial_key > best_partial_key:
            best_partial_key = partial_key
            best_partial_state = state

        if mask == full_mask:
            if inp.unload_pose is not None:
                direct_unload = _dist(
                    pose_for_position(position_idx).x_cm,
                    pose_for_position(position_idx).y_cm,
                    inp.unload_pose.x_cm,
                    inp.unload_pose.y_cm,
                )
                if cost + direct_unload >= best_terminal_cost:
                    continue
            terminal = cost + unload_distance(position_idx, 0)
            if terminal < best_terminal_cost:
                best_terminal_cost = terminal
                best_terminal_state = state
            continue

        for next_idx, candidate_mask in enumerate(candidate_masks):
            added_mask = candidate_mask & ~mask
            if added_mask == 0:
                continue
            if mask == 0 and orange_mask and not candidate_mask & orange_mask:
                continue
            next_mask = mask | candidate_mask
            active_mask = full_mask & ~next_mask
            from_pose = pose_for_position(position_idx)
            to_pose = candidate_poses[next_idx]
            direct_lower_bound = _dist(
                from_pose.x_cm, from_pose.y_cm,
                to_pose.x_cm, to_pose.y_cm,
            )
            if cost + direct_lower_bound >= best_terminal_cost:
                continue
            distance = edge_distance(position_idx, next_idx, active_mask)
            if not math.isfinite(distance):
                continue
            next_cost = cost + distance
            next_state = (next_mask, next_idx)
            if next_cost < best_cost.get(next_state, math.inf):
                best_cost[next_state] = next_cost
                parent[next_state] = state
                heappush(heap, (next_cost, next_mask, next_idx))

    if best_terminal_state is None:
        if incumbent_mask == full_mask:
            return incumbent_result
        final_state = best_partial_state
    else:
        final_state = best_terminal_state

    selected_indices: list[int] = []
    while final_state != start_state:
        selected_indices.append(final_state[1])
        final_state = parent[final_state]
    selected_indices.reverse()
    selected_sequence = [candidates[index] for index in selected_indices]
    selected_result = _build_route_from_candidate_sequence(inp, selected_sequence)
    if best_terminal_state is not None:
        return selected_result
    if _route_strategy_total_distance(incumbent_result) <= _route_strategy_total_distance(selected_result):
        return incumbent_result
    return selected_result


def _plan_intersection_priority_route(inp: RoutePlannerInput) -> RouteStrategyResult:
    """Build an intersection-priority route with active ball hard obstacles."""
    candidates = _intersection_station_candidates(inp.geometry_result)
    uncovered: set[int] = {
        i for i, b in enumerate(inp.geometry_result.balls) if b.reachable
    }
    orange = set(_orange_ball_indices(inp.geometry_result))
    cover_points: list[CoverPoint] = []
    ordered: list[int] = []
    edges: dict[tuple[int, int], RouteEdge] = {}
    current_pose = inp.start_pose
    current_graph = 0
    danger_radius_cm = inp.robot_radius_cm

    while uncovered:
        best_candidate: StationCandidate | None = None
        best_cover_point: CoverPoint | None = None
        best_edge: RouteEdge | None = None
        best_score = math.inf
        best_distance = math.inf
        best_covered_count = -1
        next_graph = len(cover_points) + 1

        entries: list[tuple[float, float, StationCandidate, CoverPoint, set[int]]] = []
        require_orange = not cover_points and bool(orange & uncovered)
        for candidate in candidates:
            cover_point = _cover_point_from_station_candidate(
                inp.geometry_result,
                candidate,
                uncovered,
            )
            if cover_point is None:
                continue
            covered = set(cover_point.covered_ball_indices)
            if require_orange and not covered & orange:
                continue
            direct_lower_bound = _dist(
                current_pose.x_cm, current_pose.y_cm,
                cover_point.pose.x_cm, cover_point.pose.y_cm,
            )
            lower_bound_score = _intersection_candidate_score(
                direct_lower_bound,
                candidate,
                len(covered),
            )
            entries.append((
                lower_bound_score,
                direct_lower_bound,
                candidate,
                cover_point,
                covered,
            ))

        entries.sort(key=lambda entry: (
            entry[0],
            entry[1],
            -len(entry[4]),
            not entry[2].from_ring_intersection,
            entry[2].total_abs_reach_offset_cm,
            entry[3].pose.x_cm,
            entry[3].pose.y_cm,
        ))

        for (
            lower_bound_score,
            _direct_lower_bound,
            candidate,
            cover_point,
            covered,
        ) in entries:
            if lower_bound_score > best_score:
                break

            active_obstacles = _ball_obstacles_for_indices(
                inp.geometry_result,
                uncovered - covered,
                danger_radius_cm,
            )
            edge = _build_strategy_edge_with_ball_obstacles(
                inp,
                current_graph,
                next_graph,
                current_pose,
                cover_point.pose,
                active_obstacles,
            )
            if not math.isfinite(edge.total_distance_cm):
                continue

            covered_count = len(covered)
            score = _intersection_candidate_score(
                edge.total_distance_cm,
                candidate,
                covered_count,
            )
            if (
                score < best_score
                or (
                    score == best_score
                    and edge.total_distance_cm < best_distance
                )
                or (
                    score == best_score
                    and edge.total_distance_cm == best_distance
                    and covered_count > best_covered_count
                )
                or (
                    score == best_score
                    and edge.total_distance_cm == best_distance
                    and covered_count == best_covered_count
                    and candidate.total_abs_reach_offset_cm
                    < (best_candidate.total_abs_reach_offset_cm if best_candidate else math.inf)
                )
            ):
                best_candidate = candidate
                best_cover_point = cover_point
                best_edge = edge
                best_score = score
                best_distance = edge.total_distance_cm
                best_covered_count = covered_count

        if best_candidate is None or best_cover_point is None or best_edge is None:
            break

        cover_points.append(best_cover_point)
        ordered.append(len(cover_points) - 1)
        edges[(best_edge.from_index, best_edge.to_index)] = best_edge
        uncovered -= set(best_cover_point.covered_ball_indices)
        current_pose = best_cover_point.pose
        current_graph = best_edge.to_index

    unload_graph_index: int | None = None
    if inp.unload_pose is not None:
        unload_graph_index = len(cover_points) + 1
        active_obstacles = _ball_obstacles_for_indices(
            inp.geometry_result,
            uncovered,
            danger_radius_cm,
        )
        unload_edge = _build_strategy_edge_with_ball_obstacles(
            inp,
            current_graph,
            unload_graph_index,
            current_pose,
            inp.unload_pose,
            active_obstacles,
        )
        edges[(unload_edge.from_index, unload_edge.to_index)] = unload_edge

    route_plan = _assemble_route_plan(
        inp.start_pose,
        cover_points,
        ordered,
        edges,
        unload_pose=inp.unload_pose,
        unload_goal_cm=inp.unload_goal_cm,
        unload_graph_index=unload_graph_index,
    )

    return RouteStrategyResult(
        cover_points=tuple(cover_points),
        ordered_indices=tuple(ordered),
        edges=tuple(edges.values()),
        route_plan=route_plan,
    )


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class SetCoverNearestNeighborStrategy:
    """v1 route strategy: greedy set-cover then nearest-neighbor ordering."""

    def plan(self, inp: RoutePlannerInput) -> RouteStrategyResult:
        # Step 1: greedy set-cover
        cover_points = _greedy_set_cover(inp.geometry_result)

        # Step 2: build graph nodes = [start(0), cover_points(1..N), unload(N+1)?]
        nodes: list[HybridPose] = [inp.start_pose] + [cp.pose for cp in cover_points]
        unload_graph_index: int | None = None
        if inp.unload_pose is not None:
            unload_graph_index = len(nodes)
            nodes.append(inp.unload_pose)

        # Step 3 + 4: edges with collision + detour
        edges = _build_edges(
            nodes, inp.obstacle, inp.robot_radius_cm,
            inp.field_width_cm, inp.field_height_cm,
        )

        # Step 5: nearest-neighbor ordering (cover points only; unload is
        # always the terminal node and excluded from the greedy tour)
        ordered = _nearest_neighbor_order(len(cover_points), edges)
        ordered = _force_orange_first_order(inp.geometry_result, cover_points, ordered)

        # Step 6: flatten to RoutePlan
        route_plan = _assemble_route_plan(
            inp.start_pose, cover_points, ordered, edges,
            unload_pose=inp.unload_pose,
            unload_goal_cm=inp.unload_goal_cm,
            unload_graph_index=unload_graph_index,
        )

        return RouteStrategyResult(
            cover_points=tuple(cover_points),
            ordered_indices=tuple(ordered),
            edges=tuple(edges.values()),
            route_plan=route_plan,
        )


class IntersectionPriorityStrategy:
    """Route strategy that prefers shared-origin pickup ring intersections."""

    def plan(self, inp: RoutePlannerInput) -> RouteStrategyResult:
        return _plan_intersection_priority_route(inp)


class IntersectionNearestStrategy:
    """Route strategy that visits shared pickup nodes by nearest safe edge."""

    def plan(self, inp: RoutePlannerInput) -> RouteStrategyResult:
        return _plan_intersection_nearest_route(inp)


class IntersectionOptimalStrategy:
    """Route strategy that minimizes total distance through shared pickup nodes."""

    def plan(self, inp: RoutePlannerInput) -> RouteStrategyResult:
        return _plan_intersection_optimal_route(inp)
