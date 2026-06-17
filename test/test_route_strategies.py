"""Tests for geometric route strategy prototypes."""

from __future__ import annotations

import math

import numpy as np

from localization.models import RobotGeometry
from path.pathfinding.models import HybridPose, PlannedBallTarget, RouteSegmentType
from path.pickup_geometry import (
    BallPickupResult,
    PickupGeometryResult,
    PickupPose,
    compute_pickup_geometry,
)
from path.route_strategy import ObstacleGeometry, RoutePlannerInput
from path.route_v1 import IntersectionPriorityStrategy, SetCoverNearestNeighborStrategy


def _target(track_id: int, x_cm: float, y_cm: float) -> PlannedBallTarget:
    return PlannedBallTarget(
        track_id=track_id,
        label="white",
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
        assert result.route_plan.segment_types == [
            RouteSegmentType.CREEP,
            RouteSegmentType.PIVOT,
        ]
        first, second = result.route_plan.pickup_poses
        assert first.x_cm == second.x_cm
        assert first.y_cm == second.y_cm
        assert first.theta_rad != second.theta_rad

    def test_uncollected_ball_blocks_direct_drive_to_intersection_station(self) -> None:
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
                _target(2, 60.0, 50.0),
                _target(3, 35.0, 38.0),
                _target(4, 50.0, 60.0),
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
            start_pose=HybridPose(x_cm=20.0, y_cm=50.0, theta_rad=0.0),
            robot_radius_cm=15.0,
            field_width_cm=100.0,
            field_height_cm=100.0,
        )

        result = IntersectionPriorityStrategy().plan(route_input)

        assert result.cover_points[0].covered_ball_indices == (0, 1, 3)
        first_edge = next(
            edge for edge in result.edges
            if edge.from_index == 0 and edge.to_index == 1
        )
        assert first_edge.blocked
        assert first_edge.detour_waypoints

        route_points = [
            route_input.start_pose,
            *first_edge.detour_waypoints,
            result.cover_points[0].pose,
        ]
        blocking_ball = pickup_geometry.balls[2]
        for a, b in zip(route_points, route_points[1:]):
            assert _segment_distance_to_point(
                blocking_ball.ball_x_cm,
                blocking_ball.ball_y_cm,
                a.x_cm,
                a.y_cm,
                b.x_cm,
                b.y_cm,
            ) > route_input.robot_radius_cm


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
        assert result.route_plan.segment_types == [
            RouteSegmentType.CREEP,
            *([RouteSegmentType.TRANSIT] * len(final_edge.detour_waypoints)),
            RouteSegmentType.TRANSIT,
        ]

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
