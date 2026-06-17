"""v1 route strategy: greedy set-cover + nearest-neighbor + corner detour.

No rendering dependencies (no numpy / cv2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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


def _find_detour(
    x0: float, y0: float, x1: float, y1: float,
    obs: ObstacleGeometry, R: float,
    field_w: float, field_h: float,
) -> tuple[float, float] | None:
    """Try each bounding-box corner as a single-waypoint detour.

    Returns the corner that yields the shortest valid 2-segment path,
    or None if no corner works.
    """
    corners = _cross_bounding_corners(obs, R)
    best: tuple[float, float] | None = None
    best_dist = math.inf

    for cx, cy in corners:
        if not _point_inside_field(cx, cy, field_w, field_h, R):
            continue
        if _segment_hits_inflated_cross(x0, y0, cx, cy, obs, R):
            continue
        if _segment_hits_inflated_cross(cx, cy, x1, y1, obs, R):
            continue
        d = _dist(x0, y0, cx, cy) + _dist(cx, cy, x1, y1)
        if d < best_dist:
            best_dist = d
            best = (cx, cy)

    return best


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
            total = direct

            if blocked:
                corner = _find_detour(x0, y0, x1, y1, obs, R, field_w, field_h)
                if corner is not None:
                    # Heading toward next node from corner
                    theta = math.atan2(y1 - corner[1], x1 - corner[0])
                    detour_wp = HybridPose(
                        x_cm=corner[0], y_cm=corner[1], theta_rad=theta,
                    )
                    total = _dist(x0, y0, corner[0], corner[1]) + _dist(corner[0], corner[1], x1, y1)
                else:
                    total = math.inf  # no valid route

            edges[(i, j)] = RouteEdge(
                from_index=i,
                to_index=j,
                direct_distance_cm=direct,
                blocked=blocked,
                detour_waypoint=detour_wp,
                total_distance_cm=total,
            )

    return edges


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

    prev_graph = 0  # start node
    for cover_idx in ordered_indices:
        graph_idx = cover_idx + 1
        edge = edges.get((prev_graph, graph_idx))

        if edge is not None and edge.blocked and edge.detour_waypoint is not None:
            points.append(edge.detour_waypoint)
            segment_types.append(RouteSegmentType.TRANSIT)

        cp_pose = cover_points[cover_idx].pose
        points.append(cp_pose)
        segment_types.append(RouteSegmentType.CREEP)
        pickup_poses.append(cp_pose)

        prev_graph = graph_idx

    # Final leg: last pickup -> unload staging pose
    if unload_pose is not None and unload_graph_index is not None:
        edge = edges.get((prev_graph, unload_graph_index))
        if edge is not None and edge.blocked and edge.detour_waypoint is not None:
            points.append(edge.detour_waypoint)
            segment_types.append(RouteSegmentType.TRANSIT)
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
