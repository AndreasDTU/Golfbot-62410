from __future__ import annotations

import unittest
import math
import numpy as np

from pathfinding.models import HybridPlannerConfig, HybridPose, PlannedBallTarget, RouteSegmentType
from pathfinding.planner import (
    FORWARD_GEAR,
    LEFT_STEERING,
    NO_STEERING,
    REVERSE_GEAR,
    RIGHT_STEERING,
    STRAIGHT_STEERING,
    GreedyRoutePlanner,
    GridDijkstraHeuristic,
    HybridAStarPlanner,
    PickupStandoffGoal,
)
from robot.models import RobotGeometry
from vision.config import FieldConfig


class GridDijkstraHeuristicTests(unittest.TestCase):
    def test_cost_routes_around_obstacle_wall(self) -> None:
        grid = np.zeros((40, 60), dtype=np.uint8)
        grid[5:35, 30] = 1
        heuristic = GridDijkstraHeuristic(grid, (50, 20))
        field = FieldConfig(width_cm=60.0, height_cm=40.0)

        cost_from_left = heuristic.cost_from_field_point((10.0, 20.0), field)

        self.assertGreater(cost_from_left, 40.0)
        self.assertTrue(np.isfinite(cost_from_left))

    def test_multi_source_cost_targets_nearest_goal(self) -> None:
        grid = np.zeros((40, 80), dtype=np.uint8)
        heuristic = GridDijkstraHeuristic(grid, goal_nodes=[(10, 20), (70, 20)])
        field = FieldConfig(width_cm=80.0, height_cm=40.0)

        left_cost = heuristic.cost_from_field_point((11.0, 20.0), field)
        right_cost = heuristic.cost_from_field_point((69.0, 20.0), field)
        middle_cost = heuristic.cost_from_field_point((40.0, 20.0), field)

        self.assertLess(left_cost, 2.0)
        self.assertLess(right_cost, 2.0)
        self.assertGreater(middle_cost, 25.0)

    def test_occupied_goal_uses_nearest_free_cell(self) -> None:
        grid = np.zeros((20, 20), dtype=np.uint8)
        grid[10, 10] = 1
        heuristic = GridDijkstraHeuristic(grid, (10, 10))

        self.assertIsNotNone(heuristic.goal)
        self.assertNotEqual(heuristic.goal, (10, 10))
        self.assertEqual(grid[heuristic.goal[1], heuristic.goal[0]], 0)

    def test_costmap_penalty_is_added_to_dijkstra_distance(self) -> None:
        grid = np.zeros((20, 20), dtype=np.uint8)
        costmap = np.zeros_like(grid, dtype=np.float32)
        costmap[10, 8:12] = 50.0
        field = FieldConfig(width_cm=20.0, height_cm=20.0)

        plain = GridDijkstraHeuristic(grid, (15, 10))
        penalized = GridDijkstraHeuristic(grid, (15, 10), costmap)

        self.assertGreater(
            penalized.cost_from_field_point((5.0, 10.0), field),
            plain.cost_from_field_point((5.0, 10.0), field),
        )


class SmallGoalUnloadRouteTests(unittest.TestCase):
    def test_route_appends_small_goal_unload_after_pickup(self) -> None:
        field = FieldConfig()
        grid = np.zeros((field.grid_height_cm, field.grid_width_cm), dtype=np.uint8)
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.0,
            rear_cm=10.0,
            tube_forward_cm=10.0,
            tube_right_cm=0.0,
            unload_extension_cm=15.0,
        )
        planner = HybridAStarPlanner(field_config=field)
        route_planner = GreedyRoutePlanner(planner)
        start_pose = HybridPose(40.0, field.height_cm * 0.5, 0.0)
        ball = PlannedBallTarget(
            track_id=1,
            label="white",
            x_cm=50.0,
            y_cm=field.height_cm * 0.5,
            node_cm=(50, int(round(field.height_cm - field.height_cm * 0.5))),
        )

        route = route_planner.plan(grid, [ball], start_pose, geometry)

        self.assertIsNotNone(route.unload_pose)
        self.assertEqual(route.unload_goal_cm, (0.0, field.height_cm * 0.5))
        assert route.unload_pose is not None
        self.assertTrue(planner.is_unload_goal_reached(route.unload_pose, geometry))


class PickupStandoffRouteTests(unittest.TestCase):
    def test_target_segment_routes_to_robot_body_standoff_pose(self) -> None:
        field = FieldConfig()
        grid = np.zeros((field.grid_height_cm, field.grid_width_cm), dtype=np.uint8)
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.0,
            rear_cm=10.0,
            tube_forward_cm=17.1,
            tube_right_cm=0.0,
            unload_extension_cm=15.0,
        )
        planner = HybridAStarPlanner(field_config=field)
        route_planner = GreedyRoutePlanner(planner)
        start_pose = HybridPose(30.0, 60.0, 0.0)
        ball = PlannedBallTarget(
            track_id=1,
            label="white",
            x_cm=80.0,
            y_cm=60.0,
            node_cm=(80, int(round(field.height_cm - 60.0))),
        )

        segment = route_planner.plan_target_segment(grid, start_pose, ball, geometry, planner.config)
        valid_standoffs = planner.valid_pickup_standoff_poses(grid, ball.node_cm, geometry, (ball.x_cm, ball.y_cm))

        self.assertTrue(segment)
        self.assertGreater(len(valid_standoffs), 1)
        final_pose = segment[-1]
        self.assertAlmostEqual(final_pose.x_cm, 62.9, delta=planner.config.goal_tolerance_cm)
        self.assertAlmostEqual(final_pose.y_cm, 60.0, delta=planner.config.goal_tolerance_cm)
        self.assertAlmostEqual(final_pose.theta_rad, 0.0, delta=np.deg2rad(8.0))
        standoff_pose = segment[-2]
        final_vector = (final_pose.x_cm - standoff_pose.x_cm, final_pose.y_cm - standoff_pose.y_cm)
        pivot_start = segment[-3]
        self.assertAlmostEqual(pivot_start.x_cm, standoff_pose.x_cm, delta=1e-6)
        self.assertAlmostEqual(pivot_start.y_cm, standoff_pose.y_cm, delta=1e-6)
        self.assertAlmostEqual(
            final_vector[0] * np.sin(final_pose.theta_rad) - final_vector[1] * np.cos(final_pose.theta_rad),
            0.0,
            delta=1e-6,
        )
        self.assertAlmostEqual(np.hypot(*final_vector), planner.MIN_STANDOFF_BODY_DISTANCE_CM, delta=1e-6)
        tube = planner.tube_center_for_pose(final_pose, geometry)
        self.assertLessEqual(np.hypot(tube[0] - ball.x_cm, tube[1] - ball.y_cm), 1e-6)

    def test_flexible_standoff_accepts_already_near_aligned_pose(self) -> None:
        field = FieldConfig()
        grid = np.zeros((field.grid_height_cm, field.grid_width_cm), dtype=np.uint8)
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.0,
            rear_cm=10.0,
            tube_forward_cm=17.1,
            tube_right_cm=0.0,
            unload_extension_cm=15.0,
        )
        planner = HybridAStarPlanner(field_config=field)
        ball = PlannedBallTarget(
            track_id=1,
            label="white",
            x_cm=80.0,
            y_cm=60.0,
            node_cm=(80, int(round(field.height_cm - 60.0))),
        )
        start_pose = HybridPose(ball.x_cm - geometry.tube_forward_cm - 6.0, ball.y_cm, 0.0)

        segment = planner.search_pickup_standoff_goal(
            grid,
            start_pose,
            ball.node_cm,
            geometry,
            goal_point_cm=(ball.x_cm, ball.y_cm),
        )

        self.assertEqual(segment[0], start_pose)
        self.assertEqual(len(segment), 3)
        self.assertAlmostEqual(segment[1].x_cm, start_pose.x_cm, delta=1e-6)
        self.assertAlmostEqual(segment[1].y_cm, start_pose.y_cm, delta=1e-6)
        tube = planner.tube_center_for_pose(segment[-1], geometry)
        self.assertLessEqual(np.hypot(tube[0] - ball.x_cm, tube[1] - ball.y_cm), 1e-6)

    def test_terminal_pickup_segments_are_profiled_as_pivot_then_creep(self) -> None:
        planner = HybridAStarPlanner()
        route_planner = GreedyRoutePlanner(planner)
        config = HybridPlannerConfig(transit_speed_pct=42.0, pivot_speed_pct=24.0, creep_speed_pct=6.0)
        route = [
            HybridPose(10.0, 10.0, 0.0),
            HybridPose(20.0, 10.0, 0.0),
            HybridPose(20.0, 10.0, math.radians(30.0)),
            HybridPose(30.0, 15.0, math.radians(30.0)),
        ]
        pickup_poses = [route[-1]]

        segment_types = route_planner.classify_route_segments(route, pickup_poses)
        segment_speeds = planner.segment_speeds_for_types(segment_types, config)

        self.assertEqual(segment_types, [RouteSegmentType.TRANSIT, RouteSegmentType.PIVOT, RouteSegmentType.CREEP])
        self.assertEqual(segment_speeds, [42.0, 24.0, 6.0])

    def test_pickup_standoff_search_relaxes_soft_costmap_after_failure(self) -> None:
        field = FieldConfig(width_cm=80.0, height_cm=60.0)
        grid = np.zeros((field.grid_height_cm, field.grid_width_cm), dtype=np.uint8)
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.0,
            rear_cm=10.0,
            tube_forward_cm=17.1,
            tube_right_cm=0.0,
            unload_extension_cm=15.0,
        )
        start_pose = HybridPose(10.0, 30.0, 0.0)
        standoff_pose = HybridPose(30.0, 30.0, 0.0)
        final_pose = HybridPose(45.0, 30.0, 0.0)
        costmap = np.full(grid.shape, 100.0, dtype=np.float32)

        class FallbackPlanner(HybridAStarPlanner):
            def __init__(self) -> None:
                super().__init__(field_config=field)
                self.attempts: list[tuple[str, float | None, float]] = []

            def valid_pickup_standoff_goals(
                self,
                raw_red_grid: np.ndarray,
                goal_node: tuple[int, int],
                geometry: RobotGeometry,
                goal_point_cm: tuple[float, float],
            ) -> list[PickupStandoffGoal]:
                return [PickupStandoffGoal(standoff_pose, final_pose)]

            def _search_pickup_standoff_goal_once(
                self,
                raw_red_grid: np.ndarray,
                start_pose: HybridPose,
                goal_node: tuple[int, int],
                geometry: RobotGeometry,
                cfg: HybridPlannerConfig,
                goal_point_cm: tuple[float, float],
                costmap: np.ndarray | None,
                collision_checker,
                standoff_goals: list[PickupStandoffGoal],
                attempt_name: str,
            ) -> list[HybridPose]:
                cost = None if costmap is None else float(costmap.max())
                self.attempts.append((attempt_name, cost, cfg.heuristic_weight))
                if attempt_name == "relaxed soft costs":
                    return [start_pose, standoff_pose, final_pose]
                return []

        planner = FallbackPlanner()
        segment = planner.search_pickup_standoff_goal(
            grid,
            start_pose,
            (45, int(round(field.height_cm - 30.0))),
            geometry,
            goal_point_cm=(45.0, 30.0),
            costmap=costmap,
        )

        self.assertTrue(segment)
        self.assertEqual(planner.attempts[0], ("standard", 100.0, 1.5))
        self.assertEqual(planner.attempts[1], ("relaxed soft costs", 10.0, 1.5))


class BallAwareRoutePlanningTests(unittest.TestCase):
    def test_weighted_heuristic_and_motion_cost_defaults_are_configured(self) -> None:
        config = HybridPlannerConfig()

        self.assertEqual(config.heuristic_weight, 1.5)
        self.assertEqual(config.gear_shift_penalty, 50.0)
        self.assertEqual(config.steering_change_penalty, 3.0)
        self.assertEqual(config.reverse_cost_multiplier, 2.5)
        self.assertEqual(config.in_place_rotation_cost, 2.0)
        self.assertEqual(config.transit_speed_pct, 38.0)
        self.assertEqual(config.pivot_speed_pct, 30.0)
        self.assertEqual(config.creep_speed_pct, 7.0)

    def test_expand_neighbors_penalizes_forward_reverse_gear_shift(self) -> None:
        planner = HybridAStarPlanner()
        config = HybridPlannerConfig(step_cm=4.0, reverse_cost_multiplier=2.5, gear_shift_penalty=50.0)
        neighbors = planner.expand_neighbors(HybridPose(10.0, 10.0, 0.0), config, previous_gear=FORWARD_GEAR)

        reverse = next(item for item in neighbors if item[2] == REVERSE_GEAR)

        self.assertEqual(reverse[1], 4.0 * 2.5 + 50.0)

    def test_expand_neighbors_penalizes_steering_changes_but_not_straight_driving(self) -> None:
        planner = HybridAStarPlanner()
        config = HybridPlannerConfig(in_place_rotation_cost=2.0, steering_change_penalty=3.0)

        from_straight = planner.expand_neighbors(
            HybridPose(10.0, 10.0, 0.0),
            config,
            previous_gear=FORWARD_GEAR,
            previous_steering=NO_STEERING,
        )
        after_left_turn = planner.expand_neighbors(
            HybridPose(10.0, 10.0, 0.0),
            config,
            previous_gear=FORWARD_GEAR,
            previous_steering=LEFT_STEERING,
        )

        first_right = next(item for item in from_straight if item[3] == RIGHT_STEERING)
        shifted_right = next(item for item in after_left_turn if item[3] == RIGHT_STEERING)
        straight = next(item for item in after_left_turn if item[3] == STRAIGHT_STEERING)

        self.assertAlmostEqual(shifted_right[1], first_right[1] + 3.0)
        self.assertEqual(straight[1], config.step_cm)

    def test_non_target_ball_is_added_to_costmap_and_target_is_excluded(self) -> None:
        field = FieldConfig(width_cm=120.0, height_cm=80.0)
        grid = np.zeros((field.grid_height_cm, field.grid_width_cm), dtype=np.uint8)
        config = HybridPlannerConfig(ball_radius_cm=2.0, non_target_ball_extra_clearance_cm=0.0)
        route_planner = GreedyRoutePlanner(HybridAStarPlanner(field_config=field, config=config), config)
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.0,
            rear_cm=10.0,
            tube_forward_cm=17.1,
            tube_right_cm=0.0,
            unload_extension_cm=15.0,
        )
        orange = PlannedBallTarget(1, "orange", 80.0, 40.0, (80, 40))
        white = PlannedBallTarget(2, "white", 55.0, 40.0, (55, 40))

        costmap, obstacles = route_planner.ball_costmap_for_target(grid, [orange, white], orange, geometry, config)

        self.assertEqual([target.track_id for target in obstacles], [2])
        assert costmap is not None
        self.assertGreater(costmap[int(round(field.height_cm - white.y_cm)), int(round(white.x_cm))], 0.0)
        self.assertEqual(costmap[int(round(field.height_cm - orange.y_cm)), int(round(orange.x_cm))], 0.0)

    def test_ball_obstacle_radius_uses_ball_radius_half_robot_width_and_extra_clearance(self) -> None:
        config = HybridPlannerConfig(ball_radius_cm=2.0, non_target_ball_extra_clearance_cm=0.5)
        route_planner = GreedyRoutePlanner(HybridAStarPlanner(config=config), config)
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.0,
            rear_cm=10.0,
            tube_forward_cm=17.1,
            tube_right_cm=0.0,
            unload_extension_cm=15.0,
        )

        self.assertEqual(route_planner.ball_obstacle_radius_cm(config, geometry), 12.5)

    def test_crowded_target_uses_soft_costmap_without_blocking_search(self) -> None:
        field = FieldConfig(width_cm=120.0, height_cm=80.0)
        grid = np.zeros((field.grid_height_cm, field.grid_width_cm), dtype=np.uint8)
        config = HybridPlannerConfig(ball_radius_cm=2.0, non_target_ball_extra_clearance_cm=0.0)
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.0,
            rear_cm=10.0,
            tube_forward_cm=17.1,
            tube_right_cm=0.0,
            unload_extension_cm=15.0,
        )

        class CrowdedTargetPlanner(GreedyRoutePlanner):
            def plan_target_segment(
                self,
                grid: np.ndarray,
                current_pose: HybridPose,
                target: PlannedBallTarget,
                geometry: RobotGeometry,
                config: HybridPlannerConfig,
                costmap: np.ndarray | None = None,
            ) -> list[HybridPose]:
                assert costmap is not None
                return [current_pose, HybridPose(target.x_cm, target.y_cm, 0.0)]

        route_planner = CrowdedTargetPlanner(HybridAStarPlanner(field_config=field, config=config), config)
        target = PlannedBallTarget(1, "white", 60.0, 40.0, (60, 40))
        crowded_neighbor = PlannedBallTarget(2, "white", 66.0, 40.0, (66, 40))

        segment, obstacles, mode = route_planner.plan_target_segment_with_ball_avoidance(
            grid,
            [target, crowded_neighbor],
            HybridPose(30.0, 40.0, 0.0),
            target,
            geometry,
            config,
        )

        self.assertTrue(segment)
        self.assertEqual([obstacle.track_id for obstacle in obstacles], [crowded_neighbor.track_id])
        self.assertEqual(mode, "soft")

    def test_route_to_orange_avoids_white_ball_when_route_around_exists(self) -> None:
        field = FieldConfig()
        grid = np.zeros((field.grid_height_cm, field.grid_width_cm), dtype=np.uint8)
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.3,
            rear_cm=10.1,
            tube_forward_cm=17.1,
            tube_right_cm=0.0,
            unload_extension_cm=15.0,
        )
        config = HybridPlannerConfig(
            max_expansions=50000,
            ball_radius_cm=2.0,
            non_target_ball_extra_clearance_cm=0.0,
        )
        route_planner = GreedyRoutePlanner(HybridAStarPlanner(field_config=field, config=config), config)
        start_pose = HybridPose(30.0, 60.0, 0.0)
        orange = PlannedBallTarget(1, "orange", 120.0, 60.0, (120, int(round(field.height_cm - 60.0))))
        white = PlannedBallTarget(2, "white", 75.0, 60.0, (75, int(round(field.height_cm - 60.0))))

        segment, obstacles, mode = route_planner.plan_target_segment_with_ball_avoidance(
            grid,
            [orange, white],
            start_pose,
            orange,
            geometry,
            config,
        )

        self.assertTrue(segment)
        self.assertEqual(mode, "soft")
        self.assertEqual([target.track_id for target in obstacles], [2])
        min_distance_to_white = min(math.hypot(pose.x_cm - white.x_cm, pose.y_cm - white.y_cm) for pose in segment)
        self.assertGreater(min_distance_to_white, route_planner.ball_obstacle_radius_cm(config, geometry))
        self.assertGreater(max(abs(pose.y_cm - 60.0) for pose in segment), 10.0)

    def test_orange_is_always_selected_before_white_balls(self) -> None:
        field = FieldConfig()
        grid = np.zeros((field.grid_height_cm, field.grid_width_cm), dtype=np.uint8)
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.3,
            rear_cm=10.1,
            tube_forward_cm=17.1,
            tube_right_cm=0.0,
            unload_extension_cm=15.0,
        )
        config = HybridPlannerConfig(max_expansions=50000)
        route_planner = GreedyRoutePlanner(HybridAStarPlanner(field_config=field, config=config), config)
        orange = PlannedBallTarget(1, "orange", 120.0, 60.0, (120, int(round(field.height_cm - 60.0))))
        white = PlannedBallTarget(2, "white", 75.0, 60.0, (75, int(round(field.height_cm - 60.0))))

        route = route_planner.plan(grid, [white, orange], HybridPose(30.0, 60.0, 0.0), geometry, config)

        self.assertIsNotNone(route.active_target)
        assert route.active_target is not None
        self.assertEqual(route.active_target.track_id, orange.track_id)
        self.assertEqual(route.ball_avoidance_mode, "soft")
        self.assertNotEqual(route.ball_avoidance_mode, "intermediate pickup")

    def test_orange_blocked_by_hard_grid_does_not_run_contact_fallback(self) -> None:
        field = FieldConfig(width_cm=120.0, height_cm=80.0)
        grid = np.zeros((field.grid_height_cm, field.grid_width_cm), dtype=np.uint8)
        grid[:, 60] = 1
        grid[30:51, 60] = 0
        geometry = RobotGeometry(
            width_cm=10.0,
            front_cm=5.0,
            rear_cm=5.0,
            tube_forward_cm=10.0,
            tube_right_cm=0.0,
            unload_extension_cm=15.0,
        )
        config = HybridPlannerConfig(
            max_expansions=25000,
            goal_tolerance_cm=4.0,
            ball_radius_cm=2.0,
            non_target_ball_extra_clearance_cm=8.0,
        )
        route_planner = GreedyRoutePlanner(HybridAStarPlanner(field_config=field, config=config), config)
        orange = PlannedBallTarget(1, "orange", 95.0, 40.0, (95, 40))
        white = PlannedBallTarget(2, "white", 60.0, 40.0, (60, 40))

        class NoFallbackPlanner(GreedyRoutePlanner):
            def plan_target_segment_with_ball_avoidance(
                self,
                grid: np.ndarray,
                all_targets: list[PlannedBallTarget],
                current_pose: HybridPose,
                target: PlannedBallTarget,
                geometry: RobotGeometry,
                config: HybridPlannerConfig,
            ) -> tuple[list[HybridPose], list[PlannedBallTarget], str]:
                return [], [candidate for candidate in all_targets if candidate.track_id != target.track_id], "soft"

            def plan_target_segment(
                self,
                grid: np.ndarray,
                current_pose: HybridPose,
                target: PlannedBallTarget,
                geometry: RobotGeometry,
                config: HybridPlannerConfig,
                costmap: np.ndarray | None = None,
            ) -> list[HybridPose]:
                raise AssertionError("contact fallback should not replan without the costmap")

        route_planner = NoFallbackPlanner(HybridAStarPlanner(field_config=field, config=config), config)
        route = route_planner.plan(grid, [orange, white], HybridPose(30.0, 40.0, 0.0), geometry, config)

        self.assertIsNone(route.active_target)
        self.assertEqual(route.points, [])

    def test_white_targets_avoid_other_balls_after_orange_is_absent(self) -> None:
        field = FieldConfig()
        grid = np.zeros((field.grid_height_cm, field.grid_width_cm), dtype=np.uint8)
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.3,
            rear_cm=10.1,
            tube_forward_cm=17.1,
            tube_right_cm=0.0,
            unload_extension_cm=15.0,
        )
        config = HybridPlannerConfig(max_expansions=50000)
        route_planner = GreedyRoutePlanner(HybridAStarPlanner(field_config=field, config=config), config)
        target_white = PlannedBallTarget(1, "white", 120.0, 60.0, (120, int(round(field.height_cm - 60.0))))
        obstacle_white = PlannedBallTarget(2, "white", 75.0, 60.0, (75, int(round(field.height_cm - 60.0))))

        segment, obstacles, mode = route_planner.plan_target_segment_with_ball_avoidance(
            grid,
            [target_white, obstacle_white],
            HybridPose(30.0, 60.0, 0.0),
            target_white,
            geometry,
            config,
        )

        self.assertTrue(segment)
        self.assertEqual(mode, "soft")
        self.assertEqual([target.track_id for target in obstacles], [obstacle_white.track_id])
        self.assertGreater(
            min(math.hypot(pose.x_cm - obstacle_white.x_cm, pose.y_cm - obstacle_white.y_cm) for pose in segment),
            route_planner.ball_obstacle_radius_cm(config, geometry),
        )

    def test_white_target_does_not_run_contact_fallback_when_soft_plan_fails(self) -> None:
        field = FieldConfig(width_cm=120.0, height_cm=80.0)
        grid = np.zeros((field.grid_height_cm, field.grid_width_cm), dtype=np.uint8)
        geometry = RobotGeometry(
            width_cm=10.0,
            front_cm=5.0,
            rear_cm=5.0,
            tube_forward_cm=10.0,
            tube_right_cm=0.0,
            unload_extension_cm=15.0,
        )
        config = HybridPlannerConfig(
            max_expansions=25000,
            goal_tolerance_cm=4.0,
            ball_radius_cm=2.0,
            non_target_ball_extra_clearance_cm=8.0,
        )

        class ContactFallbackPlanner(GreedyRoutePlanner):
            def plan_target_segment_with_ball_avoidance(
                self,
                grid: np.ndarray,
                all_targets: list[PlannedBallTarget],
                current_pose: HybridPose,
                target: PlannedBallTarget,
                geometry: RobotGeometry,
                config: HybridPlannerConfig,
            ) -> tuple[list[HybridPose], list[PlannedBallTarget], str]:
                return [], [candidate for candidate in all_targets if candidate.track_id != target.track_id], "hard"

            def plan_target_segment(
                self,
                grid: np.ndarray,
                current_pose: HybridPose,
                target: PlannedBallTarget,
                geometry: RobotGeometry,
                config: HybridPlannerConfig,
                costmap: np.ndarray | None = None,
            ) -> list[HybridPose]:
                raise AssertionError("contact fallback should not replan without the costmap")

            def plan_unload_segment(
                self,
                grid: np.ndarray,
                current_pose: HybridPose,
                geometry: RobotGeometry,
                config: HybridPlannerConfig,
            ):
                return [], None, None

        route_planner = ContactFallbackPlanner(HybridAStarPlanner(field_config=field, config=config), config)
        target_white = PlannedBallTarget(1, "white", 95.0, 40.0, (95, 40))
        blocking_white = PlannedBallTarget(2, "white", 60.0, 40.0, (60, 40))

        route = route_planner.plan(grid, [target_white, blocking_white], HybridPose(30.0, 40.0, 0.0), geometry, config)

        self.assertIsNone(route.active_target)
        self.assertEqual(route.points, [])

    def test_ball_avoidance_can_be_disabled_for_debugging(self) -> None:
        field = FieldConfig(width_cm=120.0, height_cm=80.0)
        grid = np.zeros((field.grid_height_cm, field.grid_width_cm), dtype=np.uint8)
        config = HybridPlannerConfig(avoid_non_target_balls_enabled=False)
        route_planner = GreedyRoutePlanner(HybridAStarPlanner(field_config=field, config=config), config)
        orange = PlannedBallTarget(1, "orange", 80.0, 40.0, (80, 40))
        white = PlannedBallTarget(2, "white", 55.0, 40.0, (55, 40))

        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.0,
            rear_cm=10.0,
            tube_forward_cm=17.1,
            tube_right_cm=0.0,
            unload_extension_cm=15.0,
        )

        costmap, obstacles = route_planner.ball_costmap_for_target(grid, [orange, white], orange, geometry, config)

        self.assertEqual(obstacles, [])
        self.assertIsNone(costmap)


class TightCornerPickupTests(unittest.TestCase):
    def test_top_right_corner_ball_uses_diagonal_pickup_pose(self) -> None:
        field = FieldConfig()
        grid = np.zeros((field.grid_height_cm, field.grid_width_cm), dtype=np.uint8)
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.3,
            rear_cm=10.1,
            tube_forward_cm=17.1,
            tube_right_cm=0.0,
            unload_extension_cm=15.0,
        )
        planner = HybridAStarPlanner(field_config=field)
        start_pose = HybridPose(140.0, 95.0, 0.0)
        ball_point = (field.width_cm - 2.2, field.height_cm - 2.5)

        segment = planner.search_corner_pickup(
            grid,
            start_pose,
            (165, 2),
            geometry,
            goal_point_cm=ball_point,
        )

        self.assertTrue(segment)
        final_pose = segment[-1]
        self.assertLessEqual(abs(final_pose.theta_rad - np.deg2rad(45.0)), np.deg2rad(9.0))
        tube = planner.tube_center_for_pose(final_pose, geometry)
        self.assertLessEqual(np.hypot(tube[0] - ball_point[0], tube[1] - ball_point[1]), 1e-6)

    def test_ball_six_cm_from_walls_is_not_treated_as_tight_corner(self) -> None:
        field = FieldConfig()
        planner = HybridAStarPlanner(field_config=field)

        heading = planner.tight_corner_pickup_heading((field.width_cm - 6.0, field.height_cm - 6.0))

        self.assertIsNone(heading)

    def test_four_tight_corner_balls_remain_reachable_in_sequence(self) -> None:
        field = FieldConfig()
        grid = np.zeros((field.grid_height_cm, field.grid_width_cm), dtype=np.uint8)
        geometry = RobotGeometry(
            width_cm=20.0,
            front_cm=8.3,
            rear_cm=10.1,
            tube_forward_cm=17.1,
            tube_right_cm=0.0,
            unload_extension_cm=15.0,
        )
        planner = HybridAStarPlanner(field_config=field)
        route_planner = GreedyRoutePlanner(planner)
        targets = [
            PlannedBallTarget(1, "white", 2.0, field.height_cm - 2.0, (2, 2)),
            PlannedBallTarget(2, "white", field.width_cm - 2.2, field.height_cm - 2.0, (165, 2)),
            PlannedBallTarget(3, "white", 2.0, 2.0, (2, 120)),
            PlannedBallTarget(4, "white", field.width_cm - 2.2, 2.0, (165, 120)),
        ]

        route = route_planner.plan(grid, targets, HybridPose(33.1, 69.0, 1.34), geometry)

        self.assertEqual(len(route.pickup_poses), 4)


if __name__ == "__main__":
    unittest.main()
