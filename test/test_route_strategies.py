"""Tests for geometric route strategy prototypes."""

from __future__ import annotations

import math

import numpy as np

from localization.models import RobotGeometry
from path.pathfinding.models import HybridPose, PlannedBallTarget
from path.pickup_geometry import (
    BallPickupResult,
    PickupGeometryResult,
    PickupPose,
    compute_pickup_geometry,
)
from path.route_strategy import ObstacleGeometry, RoutePlannerInput
from path.route_v1 import (
    IntersectionNearestStrategy,
    IntersectionOptimalStrategy,
    IntersectionPriorityStrategy,
    SetCoverNearestNeighborStrategy,
    _ball_obstacles_for_indices,
    _build_edge_with_ball_obstacles,
)


def _target(
    track_id: int,
    x_cm: float,
    y_cm: float,
    label: str = "white",
) -> PlannedBallTarget:
    return PlannedBallTarget(
        track_id=track_id,
        label=label,
        x_cm=x_cm,
        y_cm=y_cm,
        node_cm=(round(x_cm), round(100.0 - y_cm)),
    )


def _segment_distance_to_point(
    px: float,
    py: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    dx = x1 - x0
    dy = y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(px - x0, py - y0)
    t = ((px - x0) * dx + (py - y0) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    closest_x = x0 + t * dx
    closest_y = y0 + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def _route_distance(result) -> float:
    edge_by_nodes = {
        (edge.from_index, edge.to_index): edge
        for edge in result.edges
    }
    total = 0.0
    prev_graph = 0
    for cover_idx in result.ordered_indices:
        graph_idx = cover_idx + 1
        total += edge_by_nodes[(prev_graph, graph_idx)].total_distance_cm
        prev_graph = graph_idx
    if result.route_plan.unload_pose is not None:
        unload_edges = [
            edge for edge in result.edges
            if edge.from_index == prev_graph
            and edge.to_index > len(result.cover_points)
        ]
        assert len(unload_edges) == 1
        total += unload_edges[0].total_distance_cm
    return total


def _picked_ball_index_for_pose(
    pickup_geometry: PickupGeometryResult,
    pose: HybridPose,
) -> int:
    best_index = -1
    best_distance = math.inf
    for index, ball in enumerate(pickup_geometry.balls):
        tip_x = (
            pose.x_cm
            + math.cos(pose.theta_rad) * pickup_geometry.tube_forward_cm
            + math.sin(pose.theta_rad) * pickup_geometry.tube_right_cm
        )
        tip_y = (
            pose.y_cm
            + math.sin(pose.theta_rad) * pickup_geometry.tube_forward_cm
            - math.cos(pose.theta_rad) * pickup_geometry.tube_right_cm
        )
        distance = math.hypot(tip_x - ball.ball_x_cm, tip_y - ball.ball_y_cm)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def _manual_pickup_geometry(
    origins: list[tuple[float, float]],
    labels: list[str] | None = None,
    *,
    field_width_cm: float = 160.0,
    field_height_cm: float = 150.0,
    ring_radius_cm: float = 10.0,
) -> PickupGeometryResult:
    labels = labels or ["white"] * len(origins)
    mask = np.ones((round(field_height_cm) + 1, round(field_width_cm) + 1), dtype=bool)
    balls: list[BallPickupResult] = []
    for idx, (origin_x, origin_y) in enumerate(origins):
        balls.append(BallPickupResult(
            ball_x_cm=origin_x + ring_radius_cm,
            ball_y_cm=origin_y,
            track_id=idx + 1,
            label=labels[idx],
            ring_radius_cm=ring_radius_cm,
            valid_points=(PickupPose(
                x_cm=origin_x,
                y_cm=origin_y,
                theta_rad=0.0,
                reach_offset_cm=0.0,
            ),),
            invalid_angles_rad=(),
            reachable=True,
        ))
    return PickupGeometryResult(
        legal_region_mask=mask,
        eroded_field_mask=mask,
        dilated_obstacle_mask=np.zeros_like(mask),
        ring_radius_cm=ring_radius_cm,
        mouth_radius_cm=0.0,
        tube_forward_cm=ring_radius_cm,
        tube_right_cm=0.0,
        balls=tuple(balls),
        field_width_cm=field_width_cm,
        field_height_cm=field_height_cm,
    )


def _manual_shared_origin_pickup_geometry() -> PickupGeometryResult:
    mask = np.ones((101, 101), dtype=bool)
    ring_radius_cm = 10.0
    return PickupGeometryResult(
        legal_region_mask=mask,
        eroded_field_mask=mask,
        dilated_obstacle_mask=np.zeros_like(mask),
        ring_radius_cm=ring_radius_cm,
        mouth_radius_cm=0.0,
        tube_forward_cm=ring_radius_cm,
        tube_right_cm=0.0,
        balls=(
            BallPickupResult(
                ball_x_cm=40.0,
                ball_y_cm=50.0,
                track_id=1,
                label="white",
                ring_radius_cm=ring_radius_cm,
                valid_points=(PickupPose(
                    x_cm=30.0,
                    y_cm=50.0,
                    theta_rad=0.0,
                    reach_offset_cm=0.0,
                ),),
                invalid_angles_rad=(),
                reachable=True,
            ),
            BallPickupResult(
                ball_x_cm=30.0,
                ball_y_cm=60.0,
                track_id=2,
                label="orange",
                ring_radius_cm=ring_radius_cm,
                valid_points=(PickupPose(
                    x_cm=30.0,
                    y_cm=50.0,
                    theta_rad=math.pi * 0.5,
                    reach_offset_cm=0.0,
                ),),
                invalid_angles_rad=(),
                reachable=True,
            ),
        ),
        field_width_cm=100.0,
        field_height_cm=100.0,
    )


def _manual_orange_near_cross_pickup_geometry() -> PickupGeometryResult:
    """Geometry where orange has unsafe shared nodes plus a safe fallback."""
    mask = np.ones((101, 121), dtype=bool)
    ring_radius_cm = 10.0
    return PickupGeometryResult(
        legal_region_mask=mask,
        eroded_field_mask=mask,
        dilated_obstacle_mask=np.zeros_like(mask),
        ring_radius_cm=ring_radius_cm,
        mouth_radius_cm=0.0,
        tube_forward_cm=ring_radius_cm,
        tube_right_cm=0.0,
        balls=(
            BallPickupResult(
                ball_x_cm=60.0,
                ball_y_cm=50.0,
                track_id=1,
                label="orange",
                ring_radius_cm=ring_radius_cm,
                valid_points=(
                    PickupPose(
                        x_cm=50.0,
                        y_cm=50.0,
                        theta_rad=0.0,
                        reach_offset_cm=0.0,
                    ),
                    PickupPose(
                        x_cm=60.0,
                        y_cm=40.0,
                        theta_rad=math.pi * 0.5,
                        reach_offset_cm=0.0,
                    ),
                ),
                invalid_angles_rad=(),
                reachable=True,
            ),
            BallPickupResult(
                ball_x_cm=60.0,
                ball_y_cm=52.0,
                track_id=2,
                label="white",
                ring_radius_cm=ring_radius_cm,
                valid_points=(PickupPose(
                    x_cm=60.0,
                    y_cm=62.0,
                    theta_rad=-math.pi * 0.5,
                    reach_offset_cm=0.0,
                ),),
                invalid_angles_rad=(),
                reachable=True,
            ),
        ),
        field_width_cm=120.0,
        field_height_cm=100.0,
    )


class TestOrangeFirstRequirement:
    def test_all_strategies_visit_orange_stop_first(self) -> None:
        pickup_geometry = _manual_pickup_geometry(
            [
                (20.0, 50.0),
                (120.0, 50.0),
            ],
            labels=["white", "orange"],
        )
        route_input = RoutePlannerInput(
            geometry_result=pickup_geometry,
            obstacle=ObstacleGeometry(
                center_x_cm=150.0,
                center_y_cm=140.0,
                half_size_cm=2.0,
                half_arm_width_cm=1.0,
            ),
            start_pose=HybridPose(x_cm=18.0, y_cm=50.0, theta_rad=0.0),
            robot_radius_cm=0.0,
            field_width_cm=160.0,
            field_height_cm=150.0,
        )

        for strategy_cls in (
            SetCoverNearestNeighborStrategy,
            IntersectionPriorityStrategy,
            IntersectionNearestStrategy,
            IntersectionOptimalStrategy,
        ):
            result = strategy_cls().plan(route_input)
            first_cover_idx = result.ordered_indices[0]
            first_cover = result.cover_points[first_cover_idx]
            assert first_cover.covered_ball_indices[0] == 1
            assert _picked_ball_index_for_pose(
                pickup_geometry,
                result.route_plan.pickup_poses[0],
            ) == 1

    def test_shared_origin_station_picks_orange_action_first(self) -> None:
        pickup_geometry = _manual_shared_origin_pickup_geometry()
        route_input = RoutePlannerInput(
            geometry_result=pickup_geometry,
            obstacle=ObstacleGeometry(
                center_x_cm=90.0,
                center_y_cm=90.0,
                half_size_cm=2.0,
                half_arm_width_cm=1.0,
            ),
            start_pose=HybridPose(x_cm=20.0, y_cm=20.0, theta_rad=0.0),
            robot_radius_cm=0.0,
            field_width_cm=100.0,
            field_height_cm=100.0,
        )

        for strategy_cls in (
            SetCoverNearestNeighborStrategy,
            IntersectionPriorityStrategy,
            IntersectionNearestStrategy,
            IntersectionOptimalStrategy,
        ):
            result = strategy_cls().plan(route_input)
            assert result.cover_points[0].covered_ball_indices == (1, 0)
            assert len(result.route_plan.pickup_poses) == 2
            assert _picked_ball_index_for_pose(
                pickup_geometry,
                result.route_plan.pickup_poses[0],
            ) == 1

    def test_first_orange_leg_can_escape_start_owned_ball_danger_zone(self) -> None:
        pickup_geometry = _manual_pickup_geometry(
            [
                (6.0, 20.0),
                (70.0, 20.0),
            ],
            labels=["white", "orange"],
            field_width_cm=120.0,
            field_height_cm=80.0,
        )
        route_input = RoutePlannerInput(
            geometry_result=pickup_geometry,
            obstacle=ObstacleGeometry(
                center_x_cm=110.0,
                center_y_cm=70.0,
                half_size_cm=2.0,
                half_arm_width_cm=1.0,
            ),
            start_pose=HybridPose(x_cm=20.0, y_cm=20.0, theta_rad=0.0),
            robot_radius_cm=10.0,
            field_width_cm=120.0,
            field_height_cm=80.0,
        )

        result = IntersectionNearestStrategy().plan(route_input)

        assert result.cover_points
        first_cover = result.cover_points[result.ordered_indices[0]]
        assert first_cover.covered_ball_indices == (1,)
        first_edge = next(
            edge for edge in result.edges
            if edge.from_index == 0 and edge.to_index == 1
        )
        assert math.isfinite(first_edge.total_distance_cm)

    def test_intersection_strategies_prefer_cross_clear_first_orange_station(self) -> None:
        pickup_geometry = _manual_orange_near_cross_pickup_geometry()
        route_input = RoutePlannerInput(
            geometry_result=pickup_geometry,
            obstacle=ObstacleGeometry(
                center_x_cm=60.0,
                center_y_cm=51.0,
                half_size_cm=5.0,
                half_arm_width_cm=2.0,
            ),
            start_pose=HybridPose(x_cm=40.0, y_cm=40.0, theta_rad=0.0),
            robot_radius_cm=5.0,
            field_width_cm=120.0,
            field_height_cm=100.0,
        )

        for strategy_cls in (
            IntersectionPriorityStrategy,
            IntersectionNearestStrategy,
            IntersectionOptimalStrategy,
        ):
            result = strategy_cls().plan(route_input)
            first_cover = result.cover_points[result.ordered_indices[0]]
            first_pose = result.route_plan.pickup_poses[0]

            assert first_cover.covered_ball_indices[0] == 0
            assert math.isclose(first_pose.x_cm, 60.0)
            assert math.isclose(first_pose.y_cm, 40.0)


class TestIntersectionPriorityStrategy:
    def test_intersection_station_emits_multiple_same_origin_pickups(self) -> None:
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.0,
            rear_cm=10.0,
            tube_forward_cm=10.0,
            tube_right_cm=0.0,
            mouth_radius_cm=2.0,
            unload_extension_cm=30.0,
        )
        pickup_geometry = compute_pickup_geometry(
            field_width_cm=100.0,
            field_height_cm=100.0,
            obstacle_grid=np.zeros((100, 100), dtype=np.uint8),
            balls=[
                _target(1, 40.0, 50.0),
                _target(2, 51.0, 50.0),
            ],
            geometry=geometry,
        )
        route_input = RoutePlannerInput(
            geometry_result=pickup_geometry,
            obstacle=ObstacleGeometry(
                center_x_cm=90.0,
                center_y_cm=90.0,
                half_size_cm=2.0,
                half_arm_width_cm=1.0,
            ),
            start_pose=HybridPose(x_cm=20.0, y_cm=20.0, theta_rad=0.0),
            robot_radius_cm=pickup_geometry.ring_radius_cm,
            field_width_cm=100.0,
            field_height_cm=100.0,
        )

        result = IntersectionPriorityStrategy().plan(route_input)

        assert len(result.cover_points) == 1
        cover_point = result.cover_points[0]
        assert cover_point.covered_ball_indices == (0, 1)
        assert len(cover_point.pickup_poses) == 2
        assert len(result.route_plan.pickup_poses) == 2
        first, second = result.route_plan.pickup_poses
        assert first.x_cm == second.x_cm
        assert first.y_cm == second.y_cm
        assert first.theta_rad != second.theta_rad

    def test_near_two_ball_station_beats_far_three_ball_station(self) -> None:
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.0,
            rear_cm=10.0,
            tube_forward_cm=10.0,
            tube_right_cm=0.0,
            mouth_radius_cm=2.0,
            unload_extension_cm=30.0,
        )
        tri_h = math.sqrt(3.0) * 5.0
        pickup_geometry = compute_pickup_geometry(
            field_width_cm=200.0,
            field_height_cm=120.0,
            obstacle_grid=np.zeros((120, 200), dtype=np.uint8),
            balls=[
                _target(1, 50.0, 60.0),
                _target(2, 35.0, 60.0 + tri_h),
                _target(3, 145.0, 60.0),
                _target(4, 130.0, 60.0 + tri_h),
                _target(5, 130.0, 60.0 - tri_h),
            ],
            geometry=geometry,
        )
        route_input = RoutePlannerInput(
            geometry_result=pickup_geometry,
            obstacle=ObstacleGeometry(
                center_x_cm=180.0,
                center_y_cm=100.0,
                half_size_cm=2.0,
                half_arm_width_cm=1.0,
            ),
            start_pose=HybridPose(x_cm=25.0, y_cm=60.0, theta_rad=0.0),
            robot_radius_cm=pickup_geometry.ring_radius_cm,
            field_width_cm=200.0,
            field_height_cm=120.0,
        )

        result = IntersectionPriorityStrategy().plan(route_input)

        assert set(result.cover_points[0].covered_ball_indices) == {0, 1}

    def test_uncollected_ball_obstacle_blocks_direct_edge(self) -> None:
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.0,
            rear_cm=10.0,
            tube_forward_cm=10.0,
            tube_right_cm=0.0,
            mouth_radius_cm=2.0,
            unload_extension_cm=30.0,
        )
        pickup_geometry = compute_pickup_geometry(
            field_width_cm=120.0,
            field_height_cm=100.0,
            obstacle_grid=np.zeros((100, 120), dtype=np.uint8),
            balls=[_target(1, 60.0, 50.0)],
            geometry=geometry,
        )
        obstacle = ObstacleGeometry(
            center_x_cm=110.0,
            center_y_cm=90.0,
            half_size_cm=2.0,
            half_arm_width_cm=1.0,
        )
        robot_radius_cm = 10.0
        ball_obstacles = _ball_obstacles_for_indices(
            pickup_geometry,
            {0},
            robot_radius_cm,
        )

        edge = _build_edge_with_ball_obstacles(
            0,
            1,
            HybridPose(x_cm=20.0, y_cm=50.0, theta_rad=0.0),
            HybridPose(x_cm=100.0, y_cm=50.0, theta_rad=0.0),
            obstacle,
            robot_radius_cm,
            120.0,
            100.0,
            ball_obstacles,
        )

        assert edge.blocked
        assert edge.detour_waypoints

        route_points = [
            HybridPose(x_cm=20.0, y_cm=50.0, theta_rad=0.0),
            *edge.detour_waypoints,
            HybridPose(x_cm=100.0, y_cm=50.0, theta_rad=0.0),
        ]
        blocking_ball = pickup_geometry.balls[0]
        for a, b in zip(route_points, route_points[1:]):
            assert _segment_distance_to_point(
                blocking_ball.ball_x_cm,
                blocking_ball.ball_y_cm,
                a.x_cm,
                a.y_cm,
                b.x_cm,
                b.y_cm,
            ) > robot_radius_cm


class TestSharedNodeStrategies:
    def test_nearest_strategy_uses_nearest_node_without_collection_priority(self) -> None:
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.0,
            rear_cm=10.0,
            tube_forward_cm=10.0,
            tube_right_cm=0.0,
            mouth_radius_cm=2.0,
            unload_extension_cm=30.0,
        )
        tri_h = math.sqrt(3.0) * 5.0
        pickup_geometry = compute_pickup_geometry(
            field_width_cm=200.0,
            field_height_cm=120.0,
            obstacle_grid=np.zeros((120, 200), dtype=np.uint8),
            balls=[
                _target(1, 50.0, 60.0),
                _target(2, 145.0, 60.0),
                _target(3, 130.0, 60.0 + tri_h),
                _target(4, 130.0, 60.0 - tri_h),
            ],
            geometry=geometry,
        )
        route_input = RoutePlannerInput(
            geometry_result=pickup_geometry,
            obstacle=ObstacleGeometry(
                center_x_cm=180.0,
                center_y_cm=100.0,
                half_size_cm=2.0,
                half_arm_width_cm=1.0,
            ),
            start_pose=HybridPose(x_cm=25.0, y_cm=60.0, theta_rad=0.0),
            robot_radius_cm=pickup_geometry.ring_radius_cm,
            field_width_cm=200.0,
            field_height_cm=120.0,
        )

        result = IntersectionNearestStrategy().plan(route_input)

        assert result.cover_points[0].covered_ball_indices == (0,)

    def test_optimal_strategy_can_beat_nearest_first_route(self) -> None:
        pickup_geometry = _manual_pickup_geometry([
            (48.2, 85.6),
            (81.8, 35.1),
            (111.0, 108.3),
        ])
        route_input = RoutePlannerInput(
            geometry_result=pickup_geometry,
            obstacle=ObstacleGeometry(
                center_x_cm=150.0,
                center_y_cm=140.0,
                half_size_cm=2.0,
                half_arm_width_cm=1.0,
            ),
            start_pose=HybridPose(x_cm=70.0, y_cm=60.0, theta_rad=0.0),
            robot_radius_cm=0.0,
            field_width_cm=160.0,
            field_height_cm=150.0,
            unload_pose=HybridPose(x_cm=70.0, y_cm=60.0, theta_rad=0.0),
            unload_goal_cm=(70.0, 60.0),
        )

        nearest = IntersectionNearestStrategy().plan(route_input)
        optimal = IntersectionOptimalStrategy().plan(route_input)

        assert _route_distance(optimal) < _route_distance(nearest)
        assert set().union(
            *(set(cp.covered_ball_indices) for cp in optimal.cover_points)
        ) == {0, 1, 2}


class TestUnloadDetour:
    def test_final_pickup_to_unload_leg_uses_detour_when_blocked(self) -> None:
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.0,
            rear_cm=10.0,
            tube_forward_cm=10.0,
            tube_right_cm=0.0,
            mouth_radius_cm=2.0,
            unload_extension_cm=30.0,
        )
        pickup_geometry = compute_pickup_geometry(
            field_width_cm=100.0,
            field_height_cm=100.0,
            obstacle_grid=np.zeros((100, 100), dtype=np.uint8),
            balls=[_target(1, 90.0, 50.0)],
            geometry=geometry,
        )
        unload_pose = HybridPose(x_cm=15.0, y_cm=50.0, theta_rad=math.pi)
        route_input = RoutePlannerInput(
            geometry_result=pickup_geometry,
            obstacle=ObstacleGeometry(
                center_x_cm=50.0,
                center_y_cm=50.0,
                half_size_cm=10.0,
                half_arm_width_cm=2.0,
            ),
            start_pose=HybridPose(x_cm=80.0, y_cm=20.0, theta_rad=0.0),
            robot_radius_cm=pickup_geometry.ring_radius_cm,
            field_width_cm=100.0,
            field_height_cm=100.0,
            unload_pose=unload_pose,
            unload_goal_cm=(0.0, 50.0),
        )

        result = SetCoverNearestNeighborStrategy().plan(route_input)

        final_edge = next(
            edge for edge in result.edges
            if edge.from_index == 1 and edge.to_index == 2
        )
        assert final_edge.blocked
        assert len(final_edge.detour_waypoints) >= 2
        assert math.isfinite(final_edge.total_distance_cm)
        assert result.route_plan.points[-1] is unload_pose
        assert result.route_plan.points[2:-1] == list(final_edge.detour_waypoints)

    def test_final_pickup_inside_cross_can_escape_to_unload_detour(self) -> None:
        mask = np.ones((100, 100), dtype=bool)
        pickup_pose = PickupPose(x_cm=50.0, y_cm=50.0, theta_rad=0.0)
        pickup_geometry = PickupGeometryResult(
            legal_region_mask=mask,
            eroded_field_mask=mask,
            dilated_obstacle_mask=~mask,
            ring_radius_cm=10.0,
            mouth_radius_cm=2.0,
            tube_forward_cm=10.0,
            tube_right_cm=0.0,
            balls=(
                BallPickupResult(
                    ball_x_cm=60.0,
                    ball_y_cm=50.0,
                    track_id=1,
                    label="white",
                    ring_radius_cm=10.0,
                    valid_points=(pickup_pose,),
                    invalid_angles_rad=(),
                    reachable=True,
                ),
            ),
            field_width_cm=100.0,
            field_height_cm=100.0,
        )
        unload_pose = HybridPose(x_cm=20.0, y_cm=20.0, theta_rad=math.pi)
        route_input = RoutePlannerInput(
            geometry_result=pickup_geometry,
            obstacle=ObstacleGeometry(
                center_x_cm=50.0,
                center_y_cm=50.0,
                half_size_cm=10.0,
                half_arm_width_cm=2.0,
            ),
            start_pose=HybridPose(x_cm=80.0, y_cm=20.0, theta_rad=0.0),
            robot_radius_cm=10.0,
            field_width_cm=100.0,
            field_height_cm=100.0,
            unload_pose=unload_pose,
            unload_goal_cm=(0.0, 20.0),
        )

        result = SetCoverNearestNeighborStrategy().plan(route_input)

        final_edge = next(
            edge for edge in result.edges
            if edge.from_index == 1 and edge.to_index == 2
        )
        assert final_edge.blocked
        assert final_edge.detour_waypoints
        assert math.isfinite(final_edge.total_distance_cm)
        pickup_index = result.route_plan.points.index(result.route_plan.pickup_poses[0])
        assert result.route_plan.points[pickup_index + 1:-1] == list(final_edge.detour_waypoints)
        assert result.route_plan.points[-1] is unload_pose
