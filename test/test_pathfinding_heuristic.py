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
        unload_tip = planner.rear_unload_point_for_pose(route.unload_pose, geometry)
        self.assertLessEqual(
            np.hypot(unload_tip[0] - route.unload_goal_cm[0], unload_tip[1] - route.unload_goal_cm[1]),
            1e-6,
        )


if __name__ == "__main__":
    unittest.main()
