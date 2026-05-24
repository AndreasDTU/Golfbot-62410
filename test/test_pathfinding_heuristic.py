import unittest
import math
import numpy as np

from pathfinding.models import HybridPlannerConfig, HybridPose, PlannedBallTarget
from pathfinding.planner import GreedyRoutePlanner, GridDijkstraHeuristic, HybridAStarPlanner
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

    def test_occupied_goal_uses_nearest_free_cell(self) -> None:
        grid = np.zeros((20, 20), dtype=np.uint8)
        grid[10, 10] = 1
        heuristic = GridDijkstraHeuristic(grid, (10, 10))

        self.assertIsNotNone(heuristic.goal)
        self.assertNotEqual(heuristic.goal, (10, 10))
        self.assertEqual(grid[heuristic.goal[1], heuristic.goal[0]], 0)


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
        self.assertAlmostEqual(
            final_vector[0] * np.sin(final_pose.theta_rad) - final_vector[1] * np.cos(final_pose.theta_rad),
            0.0,
            delta=1e-6,
        )
        self.assertAlmostEqual(np.hypot(*final_vector), planner.MIN_STANDOFF_BODY_DISTANCE_CM, delta=1e-6)
        tube = planner.tube_center_for_pose(final_pose, geometry)
        self.assertLessEqual(np.hypot(tube[0] - ball.x_cm, tube[1] - ball.y_cm), 1e-6)


class BallAwareRoutePlanningTests(unittest.TestCase):
    def test_non_target_ball_is_added_to_obstacle_grid_and_target_is_excluded(self) -> None:
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

        obstacle_grid, obstacles = route_planner.grid_with_ball_obstacles(grid, [orange, white], orange, geometry, config)

        self.assertEqual([target.track_id for target in obstacles], [2])
        self.assertEqual(obstacle_grid[int(round(field.height_cm - white.y_cm)), int(round(white.x_cm))], 1)
        self.assertEqual(obstacle_grid[int(round(field.height_cm - orange.y_cm)), int(round(orange.x_cm))], 0)

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

    def test_crowded_target_skips_impossible_hard_avoidance_search(self) -> None:
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
            ) -> list[HybridPose]:
                raise AssertionError("crowded hard-avoidance search should be skipped")

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

        self.assertEqual(segment, [])
        self.assertEqual([obstacle.track_id for obstacle in obstacles], [crowded_neighbor.track_id])
        self.assertEqual(mode, "crowded")

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
        self.assertEqual(mode, "hard")
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
        self.assertEqual(route.ball_avoidance_mode, "hard")
        self.assertNotEqual(route.ball_avoidance_mode, "intermediate pickup")

    def test_orange_blocked_by_ball_falls_back_to_orange_not_white(self) -> None:
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

        route = route_planner.plan(grid, [orange, white], HybridPose(30.0, 40.0, 0.0), geometry, config)

        self.assertIsNotNone(route.active_target)
        assert route.active_target is not None
        self.assertEqual(route.active_target.track_id, orange.track_id)
        self.assertEqual(route.ball_avoidance_mode, "orange forced first")
        self.assertNotEqual(route.active_target.track_id, white.track_id)
        self.assertNotEqual(route.ball_avoidance_mode, "intermediate pickup")
        self.assertEqual([target.track_id for target in route.ball_obstacles or []], [white.track_id])

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
        self.assertEqual(mode, "hard")
        self.assertEqual([target.track_id for target in obstacles], [obstacle_white.track_id])
        self.assertGreater(
            min(math.hypot(pose.x_cm - obstacle_white.x_cm, pose.y_cm - obstacle_white.y_cm) for pose in segment),
            route_planner.ball_obstacle_radius_cm(config, geometry),
        )

    def test_white_target_falls_back_to_contact_when_ball_avoidance_blocks_all_routes(self) -> None:
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
            ) -> list[HybridPose]:
                return [current_pose, HybridPose(target.x_cm, target.y_cm, 0.0)]

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

        self.assertIsNotNone(route.active_target)
        assert route.active_target is not None
        self.assertEqual(route.active_target.track_id, blocking_white.track_id)
        self.assertEqual(route.ball_avoidance_mode, "ball contact fallback")
        self.assertEqual([target.track_id for target in route.ball_obstacles or []], [target_white.track_id])
        self.assertTrue(route.points)

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

        obstacle_grid, obstacles = route_planner.grid_with_ball_obstacles(grid, [orange, white], orange, geometry, config)

        self.assertEqual(obstacles, [])
        np.testing.assert_array_equal(obstacle_grid, grid)


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
