import unittest
import numpy as np

from pathfinding.models import HybridPose, PlannedBallTarget
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
