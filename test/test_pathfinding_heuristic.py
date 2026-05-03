import unittest

import numpy as np

from pathfinding.planner import GridDijkstraHeuristic
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


if __name__ == "__main__":
    unittest.main()
