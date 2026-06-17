"""v1 route strategy: greedy set-cover + nearest-neighbor + corner detour.

No rendering dependencies (no numpy / cv2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations

from path.pathfinding.models import HybridPose, RoutePlan, RouteSegmentType
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


def _quantize(x: float, y: float) -> tuple[int, int]:
    """Snap a position to 0.1 cm grid for grouping."""
    return (round(x * QUANTIZE_PRECISION), round(y * QUANTIZE_PRECISION))


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
) -> bool:
    """Return whether a segment clears the cross and all active balls."""
    if not _polyline_clears_inflated_cross((p0, p1), obs, robot_radius_cm):
        return False
    if _segment_hits_any_ball_obstacle(p0[0], p0[1], p1[0], p1[1], ball_obstacles):
        return False
    return True


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
) -> tuple[tuple[float, float], ...] | None:
    """Find a detour path that clears the cross and active ball obstacles."""
    direct_raw_blocked = (
        _segment_hits_inflated_cross(x0, y0, x1, y1, obs, robot_radius_cm)
        or _segment_hits_any_ball_obstacle(x0, y0, x1, y1, ball_obstacles)
    )
    if not direct_raw_blocked:
        return ()

    if _point_inside_any_ball_obstacle(x0, y0, ball_obstacles):
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
            if not _segment_valid_with_ball_obstacles(
                p0, p1, obs, robot_radius_cm, ball_obstacles,
            ):
                continue
            candidate_dist = distances[current] + _dist(p0[0], p0[1], p1[0], p1[1])
            if candidate_dist < distances[neighbor]:
                distances[neighbor] = candidate_dist
                previous[neighbor] = current

    if not math.isfinite(distances[end_idx]):
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

    # Build inverse map: quantized position -> (PickupPose, set of ball indices)
    # For each position, keep the "best" PickupPose (prefer reach_offset == 0).
    pos_map: dict[tuple[int, int], tuple[PickupPose, set[int]]] = {}

    for ball_idx, ball in enumerate(balls):
        if not ball.reachable:
            continue
        for pose in ball.valid_points:
            key = _quantize(pose.x_cm, pose.y_cm)
            if key in pos_map:
                existing_pose, indices = pos_map[key]
                indices.add(ball_idx)
                # Prefer exact-ring pose
                if pose.reach_offset_cm == 0.0 and existing_pose.reach_offset_cm != 0.0:
                    pos_map[key] = (pose, indices)
            else:
                pos_map[key] = (pose, {ball_idx})

    uncovered: set[int] = {
        i for i, b in enumerate(balls) if b.reachable
    }
    cover_points: list[CoverPoint] = []

    while uncovered:
        best_key: tuple[int, int] | None = None
        best_count = 0
        best_exact = False

        for key, (pose, indices) in pos_map.items():
            overlap = len(indices & uncovered)
            exact = pose.reach_offset_cm == 0.0
            if overlap > best_count or (overlap == best_count and exact and not best_exact):
                best_key = key
                best_count = overlap
                best_exact = exact

        if best_key is None or best_count == 0:
            break

        pose, indices = pos_map[best_key]
        covered = tuple(sorted(indices & uncovered))

        # Create one HybridPose that will be reused by identity in the RoutePlan.
        hybrid = HybridPose(
            x_cm=pose.x_cm,
            y_cm=pose.y_cm,
            theta_rad=pose.theta_rad,
        )
        cover_points.append(CoverPoint(
            pose=hybrid,
            source_pickup=pose,
            covered_ball_indices=covered,
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
        )
    return _edge_from_detour_path(
        from_index, to_index,
        x0, y0, x1, y1,
        direct_blocked,
        detour_path,
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
    segment_types: list[RouteSegmentType] = []

    def append_edge_detour(edge: RouteEdge | None) -> None:
        if edge is None or not edge.blocked:
            return
        detour_points = edge.detour_waypoints
        if not detour_points and edge.detour_waypoint is not None:
            detour_points = (edge.detour_waypoint,)
        for waypoint in detour_points:
            points.append(waypoint)
            segment_types.append(RouteSegmentType.TRANSIT)

    prev_graph = 0  # start node
    for cover_idx in ordered_indices:
        graph_idx = cover_idx + 1
        edge = edges.get((prev_graph, graph_idx))

        append_edge_detour(edge)

        cp = cover_points[cover_idx]
        station_pickup_poses = cp.pickup_poses or (cp.pose,)
        for pickup_index, pickup_pose in enumerate(station_pickup_poses):
            points.append(pickup_pose)
            segment_types.append(
                RouteSegmentType.CREEP if pickup_index == 0 else RouteSegmentType.PIVOT
            )
            pickup_poses.append(pickup_pose)

        prev_graph = graph_idx

    # Final leg: last pickup -> unload staging pose
    if unload_pose is not None and unload_graph_index is not None:
        edge = edges.get((prev_graph, unload_graph_index))
        append_edge_detour(edge)
        points.append(unload_pose)
        segment_types.append(RouteSegmentType.TRANSIT)

    return RoutePlan(
        points=points,
        active_target=None,
        pickup_poses=pickup_poses,
        unload_pose=unload_pose,
        unload_goal_cm=unload_goal_cm,
        segment_types=segment_types,
    )


def _cover_point_from_station_candidate(
    candidate: StationCandidate,
    uncovered: set[int],
) -> CoverPoint | None:
    """Convert a station candidate into a CoverPoint for the remaining balls."""
    pickup_pairs = tuple(
        (ball_idx, pose)
        for ball_idx, pose in candidate.pickup_poses_by_ball
        if ball_idx in uncovered
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


def _plan_intersection_priority_route(inp: RoutePlannerInput) -> RouteStrategyResult:
    """Build an intersection-priority route with active ball hard obstacles."""
    candidates = _intersection_station_candidates(inp.geometry_result)
    uncovered: set[int] = {
        i for i, b in enumerate(inp.geometry_result.balls) if b.reachable
    }
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
        best_score: tuple[int, int, float, float] | None = None
        next_graph = len(cover_points) + 1

        for candidate in candidates:
            cover_point = _cover_point_from_station_candidate(candidate, uncovered)
            if cover_point is None:
                continue

            covered = set(cover_point.covered_ball_indices)
            active_obstacles = _ball_obstacles_for_indices(
                inp.geometry_result,
                uncovered - covered,
                danger_radius_cm,
            )
            edge = _build_edge_with_ball_obstacles(
                current_graph,
                next_graph,
                current_pose,
                cover_point.pose,
                inp.obstacle,
                inp.robot_radius_cm,
                inp.field_width_cm,
                inp.field_height_cm,
                active_obstacles,
            )
            if not math.isfinite(edge.total_distance_cm):
                continue

            score = (
                -len(covered),
                0 if candidate.from_ring_intersection else 1,
                edge.total_distance_cm,
                candidate.total_abs_reach_offset_cm,
            )
            if best_score is None or score < best_score:
                best_candidate = candidate
                best_cover_point = cover_point
                best_edge = edge
                best_score = score

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
        unload_edge = _build_edge_with_ball_obstacles(
            current_graph,
            unload_graph_index,
            current_pose,
            inp.unload_pose,
            inp.obstacle,
            inp.robot_radius_cm,
            inp.field_width_cm,
            inp.field_height_cm,
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
