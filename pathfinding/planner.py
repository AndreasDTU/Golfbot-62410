"""Path planning services for robot ball collection routes."""

from __future__ import annotations

import heapq
import math
from typing import Protocol

import cv2
import numpy as np

from pathfinding.models import HybridPlannerConfig, HybridPose, PlannedBallTarget, RoutePlan, RouteTrackingError
from robot.models import RobotGeometry, RobotPose
from vision.config import FieldConfig, PlannerConfig, RobotGeometryConfig


def normalize_planner_angle(theta_rad: float) -> float:
    """Normalize planner headings to ``[-pi, pi)`` for stable state keys."""
    return (theta_rad + math.pi) % (2.0 * math.pi) - math.pi


class RobotFootprintCollisionChecker:
    """Check an oriented multi-circle robot model against raw red occupancy."""

    def __init__(
        self,
        raw_red_grid: np.ndarray,
        geometry: RobotGeometry,
        field_config: FieldConfig | None = None,
        robot_config: RobotGeometryConfig | None = None,
    ) -> None:
        self.raw_red_grid = raw_red_grid
        self.geometry = geometry
        self.field = field_config or FieldConfig()
        self.robot_config = robot_config or RobotGeometryConfig()
        self.height = int(raw_red_grid.shape[0])
        self.width = int(raw_red_grid.shape[1])
        free_mask = (raw_red_grid == 0).astype(np.uint8)
        self.distance_to_red = cv2.distanceTransform(free_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        self.base_circles = self._build_base_circle_specs()

    def _build_base_circle_specs(self) -> list[tuple[float, float, float]]:
        """Approximate the rectangular wheelbase with overlapping circles."""
        radius = max(1.0, self.geometry.width_cm * 0.5)
        usable_start = -self.geometry.rear_cm + radius * 0.35
        usable_end = self.geometry.front_cm - radius * 0.35
        length = max(0.0, usable_end - usable_start)
        circle_count = max(2, int(math.ceil(length / max(1.0, radius * 0.9))) + 1)
        if circle_count == 1:
            offsets = [0.0]
        else:
            offsets = [
                usable_start + (usable_end - usable_start) * index / (circle_count - 1)
                for index in range(circle_count)
            ]
        return [(offset, 0.0, radius) for offset in offsets]

    def oriented_circle_centers(
        self,
        pose: HybridPose,
        circle_specs: list[tuple[float, float, float]],
    ) -> list[tuple[float, float, float]]:
        """Project local ``forward/right/radius`` circle specs into grid space."""
        forward = (math.cos(pose.theta_rad), -math.sin(pose.theta_rad))
        right = (math.sin(pose.theta_rad), math.cos(pose.theta_rad))
        center_x = pose.x_cm
        center_y = self.field.height_cm - pose.y_cm
        return [
            (
                center_x + forward[0] * forward_cm + right[0] * right_cm,
                center_y + forward[1] * forward_cm + right[1] * right_cm,
                radius_cm,
            )
            for forward_cm, right_cm, radius_cm in circle_specs
        ]

    def footprint_polygons(self, pose: HybridPose) -> tuple[np.ndarray, np.ndarray]:
        """Return base and intake polygons for ``pose`` in grid coordinates."""
        forward = np.array([math.cos(pose.theta_rad), -math.sin(pose.theta_rad)], dtype=np.float32)
        right = np.array([math.sin(pose.theta_rad), math.cos(pose.theta_rad)], dtype=np.float32)
        center = np.array([pose.x_cm, self.field.height_cm - pose.y_cm], dtype=np.float32)

        half_width_cm = self.geometry.width_cm * 0.5
        front_center = center + forward * self.geometry.front_cm
        rear_center = center - forward * self.geometry.rear_cm
        base = np.array(
            [
                front_center + right * half_width_cm,
                front_center - right * half_width_cm,
                rear_center - right * half_width_cm,
                rear_center + right * half_width_cm,
            ],
            dtype=np.float32,
        )

        tube_half_width = self.robot_config.tube_width_cm * 0.5
        tube_front = center + forward * self.geometry.tube_forward_cm + right * self.geometry.tube_right_cm
        tube_rear = front_center + right * self.geometry.tube_right_cm
        tube = np.array(
            [
                tube_front + right * tube_half_width,
                tube_front - right * tube_half_width,
                tube_rear - right * tube_half_width,
                tube_rear + right * tube_half_width,
            ],
            dtype=np.float32,
        )
        return base, tube

    def is_pose_valid(self, pose: HybridPose) -> bool:
        """Return true when the robot body is inside the field and clear of red zones.

        The pickup tube is intentionally excluded from boundary and red-zone
        collision checks so it can overhang walls while reaching corner balls.
        """
        for x_grid, y_grid, radius_cm in self.oriented_circle_centers(pose, self.base_circles):
            if (
                x_grid - radius_cm < 0.0
                or x_grid + radius_cm > self.width - 1
                or y_grid - radius_cm < 0.0
                or y_grid + radius_cm > self.height - 1
            ):
                return False
            x_index = int(np.clip(round(x_grid), 0, self.width - 1))
            y_index = int(np.clip(round(y_grid), 0, self.height - 1))
            if float(self.distance_to_red[y_index, x_index]) < radius_cm:
                return False
        return True


class GridDijkstraHeuristic:
    """Obstacle-aware 2D cost-to-go map for guiding Hybrid A* around red zones."""

    _NEIGHBORS: tuple[tuple[int, int, float], ...] = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    def __init__(self, raw_red_grid: np.ndarray, goal_node: tuple[int, int]) -> None:
        self.grid = raw_red_grid
        self.height = int(raw_red_grid.shape[0])
        self.width = int(raw_red_grid.shape[1])
        self.costs = np.full((self.height, self.width), np.inf, dtype=np.float64)
        self.goal = self._nearest_free_node(goal_node)
        if self.goal is not None:
            self._compute()

    def _in_bounds(self, node: tuple[int, int]) -> bool:
        return 0 <= node[0] < self.width and 0 <= node[1] < self.height

    def _is_free(self, node: tuple[int, int]) -> bool:
        return self._in_bounds(node) and self.grid[node[1], node[0]] == 0

    def _nearest_free_node(self, node: tuple[int, int]) -> tuple[int, int] | None:
        """Return the target node, or a nearby free cell if the exact target is occupied."""
        x = int(np.clip(node[0], 0, self.width - 1))
        y = int(np.clip(node[1], 0, self.height - 1))
        if self._is_free((x, y)):
            return x, y

        max_radius = max(self.width, self.height)
        for radius in range(1, max_radius + 1):
            x_min = max(0, x - radius)
            x_max = min(self.width - 1, x + radius)
            y_min = max(0, y - radius)
            y_max = min(self.height - 1, y + radius)
            candidates: list[tuple[float, tuple[int, int]]] = []
            for nx in range(x_min, x_max + 1):
                for ny in (y_min, y_max):
                    if self._is_free((nx, ny)):
                        candidates.append((math.hypot(nx - x, ny - y), (nx, ny)))
            for ny in range(y_min + 1, y_max):
                for nx in (x_min, x_max):
                    if self._is_free((nx, ny)):
                        candidates.append((math.hypot(nx - x, ny - y), (nx, ny)))
            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1][1], item[1][0]))
                return candidates[0][1]
        return None

    def _compute(self) -> None:
        assert self.goal is not None
        heap: list[tuple[float, tuple[int, int]]] = [(0.0, self.goal)]
        self.costs[self.goal[1], self.goal[0]] = 0.0

        while heap:
            current_cost, current = heapq.heappop(heap)
            if current_cost > float(self.costs[current[1], current[0]]) + 1e-9:
                continue
            for dx, dy, step_cost in self._NEIGHBORS:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self._is_free(neighbor):
                    continue
                next_cost = current_cost + step_cost
                if next_cost >= float(self.costs[neighbor[1], neighbor[0]]):
                    continue
                self.costs[neighbor[1], neighbor[0]] = next_cost
                heapq.heappush(heap, (next_cost, neighbor))

    def cost_from_field_point(self, point_cm: tuple[float, float], field: FieldConfig) -> float:
        if self.goal is None:
            return float("inf")
        x = int(np.clip(round(point_cm[0]), 0, self.width - 1))
        y = int(np.clip(round(field.height_cm - point_cm[1]), 0, self.height - 1))
        return float(self.costs[y, x])


class HybridAStarPlanner:
    """Search kinematically valid trajectories in ``x, y, theta``."""

    def __init__(
        self,
        field_config: FieldConfig | None = None,
        robot_config: RobotGeometryConfig | None = None,
        config: HybridPlannerConfig | None = None,
    ) -> None:
        self.field = field_config or FieldConfig()
        self.robot_config = robot_config or RobotGeometryConfig()
        self.config = config or HybridPlannerConfig()

    def theta_bin(self, theta_rad: float, theta_bins: int | None = None) -> int:
        """Discretize a continuous heading into a deterministic Hybrid A* bin."""
        bins = theta_bins or self.config.theta_bins
        normalized = normalize_planner_angle(theta_rad)
        return int(round((normalized + math.pi) * bins / (2.0 * math.pi))) % bins

    def state_key(self, pose: HybridPose, theta_bins: int | None = None) -> tuple[int, int, int]:
        """Return the closed-set key for a continuous Hybrid A* pose."""
        bins = theta_bins or self.config.theta_bins
        return (
            int(round(pose.x_cm)),
            int(round(self.field.height_cm - pose.y_cm)),
            self.theta_bin(pose.theta_rad, bins),
        )

    @staticmethod
    def tube_center_for_pose(pose: HybridPose, geometry: RobotGeometry) -> tuple[float, float]:
        """Return the field-coordinate pickup point at the intake tip."""
        forward = (math.cos(pose.theta_rad), math.sin(pose.theta_rad))
        right = (math.sin(pose.theta_rad), -math.cos(pose.theta_rad))
        return (
            pose.x_cm + forward[0] * geometry.tube_forward_cm + right[0] * geometry.tube_right_cm,
            pose.y_cm + forward[1] * geometry.tube_forward_cm + right[1] * geometry.tube_right_cm,
        )

    @staticmethod
    def rear_unload_point_for_pose(pose: HybridPose, geometry: RobotGeometry) -> tuple[float, float]:
        """Return the field-coordinate rear unload tip when the mechanism is lowered."""
        reach_cm = geometry.rear_cm + geometry.unload_extension_cm
        forward = (math.cos(pose.theta_rad), math.sin(pose.theta_rad))
        return (
            pose.x_cm - forward[0] * reach_cm,
            pose.y_cm - forward[1] * reach_cm,
        )

    def small_goal_center_cm(self) -> tuple[float, float]:
        """Return the fixed small-goal center on the left side of the field."""
        return 0.0, self.field.height_cm * 0.5

    def small_goal_unload_pose(self, geometry: RobotGeometry) -> HybridPose:
        """Return the base pose whose rear unload tip reaches the small goal."""
        goal_x, goal_y = self.small_goal_center_cm()
        reach_cm = geometry.rear_cm + geometry.unload_extension_cm
        return HybridPose(
            x_cm=goal_x + reach_cm,
            y_cm=goal_y,
            theta_rad=0.0,
        )

    def goal_to_field_metric_cm(
        self,
        goal_node: tuple[int, int],
        goal_point_cm: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """Return exact target field centimeters, falling back to a grid node."""
        if goal_point_cm is not None:
            return float(goal_point_cm[0]), float(goal_point_cm[1])
        return float(goal_node[0]), self.field.height_cm - float(goal_node[1])

    def pickup_aligned_pose_for_theta(
        self,
        goal_node: tuple[int, int],
        theta_rad: float,
        geometry: RobotGeometry,
        goal_point_cm: tuple[float, float] | None = None,
    ) -> HybridPose:
        """Return the exact base pose whose intake tip lands on the ball."""
        ball_x, ball_y = self.goal_to_field_metric_cm(goal_node, goal_point_cm)
        forward = (math.cos(theta_rad), math.sin(theta_rad))
        right = (math.sin(theta_rad), -math.cos(theta_rad))
        intake_length_cm = geometry.tube_forward_cm
        return HybridPose(
            x_cm=ball_x - forward[0] * intake_length_cm - right[0] * geometry.tube_right_cm,
            y_cm=ball_y - forward[1] * intake_length_cm - right[1] * geometry.tube_right_cm,
            theta_rad=normalize_planner_angle(theta_rad),
        )

    def goal_distance(
        self,
        pose: HybridPose,
        goal_node: tuple[int, int],
        geometry: RobotGeometry,
        goal_point_cm: tuple[float, float] | None = None,
    ) -> float:
        """Distance from pickup point at the intake tip to the target ball."""
        goal_x, goal_y = self.goal_to_field_metric_cm(goal_node, goal_point_cm)
        tube_x, tube_y = self.tube_center_for_pose(pose, geometry)
        return math.hypot(goal_x - tube_x, goal_y - tube_y)

    def heuristic_cost(
        self,
        pose: HybridPose,
        goal_node: tuple[int, int],
        geometry: RobotGeometry,
        dijkstra_heuristic: GridDijkstraHeuristic,
        config: HybridPlannerConfig | None = None,
        goal_point_cm: tuple[float, float] | None = None,
    ) -> float:
        """Return obstacle-aware 2D cost from the intake tip to the target."""
        cfg = config or self.config
        tube_point = self.tube_center_for_pose(pose, geometry)
        cost = dijkstra_heuristic.cost_from_field_point(tube_point, self.field)
        if math.isfinite(cost):
            return max(0.0, cost - cfg.goal_tolerance_cm)
        return max(0.0, self.goal_distance(pose, goal_node, geometry, goal_point_cm) - cfg.goal_tolerance_cm)

    @staticmethod
    def reconstruct_path(
        came_from: dict[tuple[int, int, int], tuple[int, int, int]],
        pose_by_key: dict[tuple[int, int, int], HybridPose],
        goal_key: tuple[int, int, int],
    ) -> list[HybridPose]:
        """Rebuild the continuous trajectory from search parents."""
        key = goal_key
        path = [pose_by_key[key]]
        while key in came_from:
            key = came_from[key]
            path.append(pose_by_key[key])
        path.reverse()
        return path

    def expand_neighbors(
        self,
        pose: HybridPose,
        config: HybridPlannerConfig | None = None,
    ) -> list[tuple[HybridPose, float]]:
        """Generate deterministic differential-drive motion primitives."""
        cfg = config or self.config
        neighbors: list[tuple[HybridPose, float]] = []

        for direction in cfg.translation_directions:
            next_pose = HybridPose(
                x_cm=pose.x_cm + math.cos(pose.theta_rad) * cfg.step_cm * direction,
                y_cm=pose.y_cm + math.sin(pose.theta_rad) * cfg.step_cm * direction,
                theta_rad=pose.theta_rad,
            )
            reverse_penalty = cfg.reverse_cost_multiplier if direction < 0.0 else 1.0
            neighbors.append((next_pose, cfg.step_cm * reverse_penalty))

        for delta_theta in cfg.rotation_deltas_rad:
            next_pose = HybridPose(
                x_cm=pose.x_cm,
                y_cm=pose.y_cm,
                theta_rad=normalize_planner_angle(pose.theta_rad + delta_theta),
            )
            neighbors.append((next_pose, cfg.in_place_rotation_cost + abs(delta_theta) * 0.25))

        return neighbors

    def search(
        self,
        raw_red_grid: np.ndarray,
        start_pose: HybridPose,
        goal_node: tuple[int, int],
        geometry: RobotGeometry,
        config: HybridPlannerConfig | None = None,
        goal_point_cm: tuple[float, float] | None = None,
    ) -> list[HybridPose]:
        """Search a kinematically valid trajectory."""
        cfg = config or self.config
        collision_checker = RobotFootprintCollisionChecker(raw_red_grid, geometry, self.field, self.robot_config)
        start_pose = HybridPose(
            x_cm=float(start_pose.x_cm),
            y_cm=float(start_pose.y_cm),
            theta_rad=normalize_planner_angle(start_pose.theta_rad),
        )

        if not collision_checker.is_pose_valid(start_pose):
            return []

        dijkstra_heuristic = GridDijkstraHeuristic(raw_red_grid, goal_node)
        start_key = self.state_key(start_pose, cfg.theta_bins)
        open_heap: list[tuple[float, float, int, tuple[int, int, int]]] = []
        counter = 0
        start_h = self.heuristic_cost(start_pose, goal_node, geometry, dijkstra_heuristic, cfg, goal_point_cm)
        heapq.heappush(open_heap, (start_h, 0.0, counter, start_key))

        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        pose_by_key: dict[tuple[int, int, int], HybridPose] = {start_key: start_pose}
        g_score: dict[tuple[int, int, int], float] = {start_key: 0.0}
        expansions = 0

        while open_heap and expansions < cfg.max_expansions:
            _f_cost, current_cost, _counter, current_key = heapq.heappop(open_heap)
            if current_cost > g_score.get(current_key, float("inf")):
                continue

            current_pose = pose_by_key[current_key]
            if self.goal_distance(current_pose, goal_node, geometry, goal_point_cm) <= cfg.goal_tolerance_cm:
                path = self.reconstruct_path(came_from, pose_by_key, current_key)
                final_pose = self.pickup_aligned_pose_for_theta(goal_node, current_pose.theta_rad, geometry, goal_point_cm)
                if collision_checker.is_pose_valid(final_pose):
                    if math.hypot(final_pose.x_cm - current_pose.x_cm, final_pose.y_cm - current_pose.y_cm) > 1e-6:
                        path.append(final_pose)
                    else:
                        path[-1] = final_pose
                    return path

            expansions += 1
            for neighbor_pose, primitive_cost in self.expand_neighbors(current_pose, cfg):
                neighbor_pose = HybridPose(
                    x_cm=float(neighbor_pose.x_cm),
                    y_cm=float(neighbor_pose.y_cm),
                    theta_rad=normalize_planner_angle(neighbor_pose.theta_rad),
                )
                if not collision_checker.is_pose_valid(neighbor_pose):
                    continue

                neighbor_key = self.state_key(neighbor_pose, cfg.theta_bins)
                tentative_g = g_score[current_key] + primitive_cost
                if tentative_g >= g_score.get(neighbor_key, float("inf")):
                    continue

                came_from[neighbor_key] = current_key
                pose_by_key[neighbor_key] = neighbor_pose
                g_score[neighbor_key] = tentative_g
                heuristic = self.heuristic_cost(neighbor_pose, goal_node, geometry, dijkstra_heuristic, cfg, goal_point_cm)
                heading_change = abs(normalize_planner_angle(neighbor_pose.theta_rad - current_pose.theta_rad))
                counter += 1
                heapq.heappush(
                    open_heap,
                    (
                        tentative_g + heuristic + heading_change * 0.1,
                        tentative_g,
                        counter,
                        neighbor_key,
                    ),
                )

        if expansions >= cfg.max_expansions:
            print(
                f"Hybrid A* search exhausted max nodes ({cfg.max_expansions}) "
                f"for goal {goal_node} after {expansions} expansions."
            )
        return []

    def search_pose_goal(
        self,
        raw_red_grid: np.ndarray,
        start_pose: HybridPose,
        goal_pose: HybridPose,
        geometry: RobotGeometry,
        config: HybridPlannerConfig | None = None,
    ) -> list[HybridPose]:
        """Search a kinematically valid trajectory to a robot-origin pose."""
        cfg = config or self.config
        collision_checker = RobotFootprintCollisionChecker(raw_red_grid, geometry, self.field, self.robot_config)
        start_pose = HybridPose(
            x_cm=float(start_pose.x_cm),
            y_cm=float(start_pose.y_cm),
            theta_rad=normalize_planner_angle(start_pose.theta_rad),
        )
        goal_pose = HybridPose(
            x_cm=float(goal_pose.x_cm),
            y_cm=float(goal_pose.y_cm),
            theta_rad=normalize_planner_angle(goal_pose.theta_rad),
        )

        if not collision_checker.is_pose_valid(start_pose) or not collision_checker.is_pose_valid(goal_pose):
            return []

        goal_node = (
            int(np.clip(round(goal_pose.x_cm), 0, raw_red_grid.shape[1] - 1)),
            int(np.clip(round(self.field.height_cm - goal_pose.y_cm), 0, raw_red_grid.shape[0] - 1)),
        )
        dijkstra_heuristic = GridDijkstraHeuristic(raw_red_grid, goal_node)
        heading_tolerance_rad = max(math.pi / float(cfg.theta_bins), math.radians(8.0))

        start_key = self.state_key(start_pose, cfg.theta_bins)
        open_heap: list[tuple[float, float, int, tuple[int, int, int]]] = []
        counter = 0
        start_h = dijkstra_heuristic.cost_from_field_point((start_pose.x_cm, start_pose.y_cm), self.field)
        if not math.isfinite(start_h):
            start_h = math.hypot(goal_pose.x_cm - start_pose.x_cm, goal_pose.y_cm - start_pose.y_cm)
        heapq.heappush(open_heap, (start_h, 0.0, counter, start_key))

        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        pose_by_key: dict[tuple[int, int, int], HybridPose] = {start_key: start_pose}
        g_score: dict[tuple[int, int, int], float] = {start_key: 0.0}
        expansions = 0

        while open_heap and expansions < cfg.max_expansions:
            _f_cost, current_cost, _counter, current_key = heapq.heappop(open_heap)
            if current_cost > g_score.get(current_key, float("inf")):
                continue

            current_pose = pose_by_key[current_key]
            distance_cm = math.hypot(goal_pose.x_cm - current_pose.x_cm, goal_pose.y_cm - current_pose.y_cm)
            heading_error = abs(normalize_planner_angle(goal_pose.theta_rad - current_pose.theta_rad))
            if distance_cm <= cfg.goal_tolerance_cm and heading_error <= heading_tolerance_rad:
                path = self.reconstruct_path(came_from, pose_by_key, current_key)
                if math.hypot(goal_pose.x_cm - current_pose.x_cm, goal_pose.y_cm - current_pose.y_cm) > 1e-6:
                    path.append(goal_pose)
                else:
                    path[-1] = goal_pose
                return path

            expansions += 1
            for neighbor_pose, primitive_cost in self.expand_neighbors(current_pose, cfg):
                neighbor_pose = HybridPose(
                    x_cm=float(neighbor_pose.x_cm),
                    y_cm=float(neighbor_pose.y_cm),
                    theta_rad=normalize_planner_angle(neighbor_pose.theta_rad),
                )
                if not collision_checker.is_pose_valid(neighbor_pose):
                    continue

                neighbor_key = self.state_key(neighbor_pose, cfg.theta_bins)
                tentative_g = g_score[current_key] + primitive_cost
                if tentative_g >= g_score.get(neighbor_key, float("inf")):
                    continue

                came_from[neighbor_key] = current_key
                pose_by_key[neighbor_key] = neighbor_pose
                g_score[neighbor_key] = tentative_g
                heuristic = dijkstra_heuristic.cost_from_field_point((neighbor_pose.x_cm, neighbor_pose.y_cm), self.field)
                if not math.isfinite(heuristic):
                    heuristic = math.hypot(goal_pose.x_cm - neighbor_pose.x_cm, goal_pose.y_cm - neighbor_pose.y_cm)
                heading_error = abs(normalize_planner_angle(goal_pose.theta_rad - neighbor_pose.theta_rad))
                counter += 1
                heapq.heappush(
                    open_heap,
                    (
                        tentative_g + max(0.0, heuristic - cfg.goal_tolerance_cm) + heading_error * 3.0,
                        tentative_g,
                        counter,
                        neighbor_key,
                    ),
                )

        if expansions >= cfg.max_expansions:
            print(
                f"Hybrid A* search exhausted max nodes ({cfg.max_expansions}) "
                f"for unload goal after {expansions} expansions."
            )
        return []


class LegacyAStarPlanner:
    """Legacy 8-connected A* search on a binary occupancy grid."""

    def search(self, grid: np.ndarray, start_node: tuple[int, int], goal_node: tuple[int, int]) -> list[tuple[int, int]]:
        """Run legacy 8-connected A* search."""
        width = int(grid.shape[1])
        height = int(grid.shape[0])

        def in_bounds(node: tuple[int, int]) -> bool:
            return 0 <= node[0] < width and 0 <= node[1] < height

        def is_free(node: tuple[int, int]) -> bool:
            return grid[node[1], node[0]] == 0

        if not in_bounds(start_node) or not in_bounds(goal_node):
            return []
        if not is_free(start_node) or not is_free(goal_node):
            return []

        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]

        def heuristic(node: tuple[int, int], goal: tuple[int, int]) -> float:
            return math.hypot(goal[0] - node[0], goal[1] - node[1])

        open_heap: list[tuple[float, float, tuple[int, int]]] = []
        heapq.heappush(open_heap, (heuristic(start_node, goal_node), 0.0, start_node))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = {start_node: 0.0}

        while open_heap:
            _f_cost, current_cost, current = heapq.heappop(open_heap)
            if current == goal_node:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            if current_cost > g_score.get(current, float("inf")):
                continue

            for dx, dy, step_cost in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if not in_bounds(neighbor) or not is_free(neighbor):
                    continue

                tentative_g = g_score[current] + step_cost
                if tentative_g >= g_score.get(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(
                    open_heap,
                    (tentative_g + heuristic(neighbor, goal_node), tentative_g, neighbor),
                )

        return []


class GreedyRoutePlanner:
    """Build an orange-first Hybrid A* collection route."""

    def __init__(
        self,
        hybrid_planner: HybridAStarPlanner | None = None,
        config: HybridPlannerConfig | None = None,
    ) -> None:
        self.hybrid_planner = hybrid_planner or HybridAStarPlanner(config=config)
        self.config = config or self.hybrid_planner.config

    def plan(
        self,
        grid: np.ndarray,
        ball_targets: list[PlannedBallTarget],
        start_pose: HybridPose,
        geometry: RobotGeometry,
        config: HybridPlannerConfig | None = None,
    ) -> RoutePlan:
        """Build an orange-first Hybrid A* collection route."""
        if not ball_targets:
            return RoutePlan(points=[], active_target=None, pickup_poses=[])

        cfg = config or self.config
        unvisited = list(ball_targets)
        current_pose = start_pose
        route: list[HybridPose] = [current_pose]
        pickup_poses: list[HybridPose] = []
        active_target: PlannedBallTarget | None = None

        orange_targets = sorted(
            [target for target in unvisited if target.label == "orange"],
            key=lambda target: math.hypot(target.x_cm - current_pose.x_cm, target.y_cm - current_pose.y_cm),
        )
        for orange_target in orange_targets:
            orange_segment = self.hybrid_planner.search(
                grid,
                current_pose,
                orange_target.node_cm,
                geometry,
                cfg,
                goal_point_cm=(orange_target.x_cm, orange_target.y_cm),
            )
            unvisited.remove(orange_target)
            if not orange_segment:
                print(
                    f"Hybrid A* could not route to orange target {orange_target.track_id}; "
                    "continuing with remaining targets."
                )
                continue
            active_target = orange_target
            route.extend(orange_segment[1:])
            current_pose = orange_segment[-1]
            pickup_poses.append(current_pose)
            break

        while unvisited:
            nearest_candidates = sorted(
                unvisited,
                key=lambda target: math.hypot(target.x_cm - current_pose.x_cm, target.y_cm - current_pose.y_cm),
            )
            chosen_target: PlannedBallTarget | None = None
            chosen_segment: list[HybridPose] = []

            for candidate in nearest_candidates:
                segment = self.hybrid_planner.search(
                    grid,
                    current_pose,
                    candidate.node_cm,
                    geometry,
                    cfg,
                    goal_point_cm=(candidate.x_cm, candidate.y_cm),
                )
                if segment:
                    chosen_target = candidate
                    chosen_segment = segment
                    break
                print(
                    f"Hybrid A* could not route to target {candidate.track_id} "
                    f"({candidate.label}); trying next target."
                )

            if chosen_target is None:
                break

            route.extend(chosen_segment[1:])
            current_pose = chosen_segment[-1]
            pickup_poses.append(current_pose)
            unvisited.remove(chosen_target)
            if active_target is None:
                active_target = chosen_target

        unload_pose: HybridPose | None = None
        unload_goal_cm: tuple[float, float] | None = None
        if pickup_poses:
            candidate_unload_pose = self.hybrid_planner.small_goal_unload_pose(geometry)
            unload_segment = self.hybrid_planner.search_pose_goal(
                grid,
                current_pose,
                candidate_unload_pose,
                geometry,
                cfg,
            )
            if unload_segment:
                route.extend(unload_segment[1:])
                unload_pose = unload_segment[-1]
                unload_goal_cm = self.hybrid_planner.small_goal_center_cm()
            else:
                print("Hybrid A* could not route from final pickup to the small goal unload pose.")

        return RoutePlan(
            points=route,
            active_target=active_target,
            pickup_poses=pickup_poses,
            unload_pose=unload_pose,
            unload_goal_cm=unload_goal_cm,
        )


class RoutePlanner(Protocol):
    """Common route-planning interface."""

    def plan(
        self,
        grid: np.ndarray,
        ball_targets: list[PlannedBallTarget],
        start_pose: HybridPose,
        geometry: RobotGeometry,
        config: HybridPlannerConfig | None = None,
    ) -> RoutePlan:
        """Plan a route through available ball targets."""


class RoutePlanningFacade:
    """Facade for route planning and route tracking helper calculations."""

    def __init__(
        self,
        field_config: FieldConfig | None = None,
        robot_config: RobotGeometryConfig | None = None,
        planner_config: PlannerConfig | None = None,
        hybrid_config: HybridPlannerConfig | None = None,
        route_planner: RoutePlanner | None = None,
    ) -> None:
        self.field = field_config or FieldConfig()
        self.robot_config = robot_config or RobotGeometryConfig()
        self.planner_config = planner_config or PlannerConfig()
        self.hybrid_config = hybrid_config or HybridPlannerConfig(
            step_cm=self.planner_config.step_cm,
            theta_bins=self.planner_config.theta_bins,
            goal_tolerance_cm=self.planner_config.goal_tolerance_cm,
            max_expansions=self.planner_config.max_expansions,
            translation_directions=self.planner_config.translation_directions,
            rotation_deltas_rad=self.planner_config.rotation_deltas_rad,
            reverse_cost_multiplier=self.planner_config.reverse_cost_multiplier,
            in_place_rotation_cost=self.planner_config.in_place_rotation_cost,
        )
        self.hybrid_planner = HybridAStarPlanner(self.field, self.robot_config, self.hybrid_config)
        self.legacy_planner = LegacyAStarPlanner()
        self.route_planner = route_planner or GreedyRoutePlanner(self.hybrid_planner, self.hybrid_config)

    def plan_route(
        self,
        grid: np.ndarray,
        ball_targets: list[PlannedBallTarget],
        start_pose: HybridPose,
        geometry: RobotGeometry,
        config: HybridPlannerConfig | None = None,
    ) -> RoutePlan:
        """Plan a route through available ball targets."""
        return self.route_planner.plan(grid, ball_targets, start_pose, geometry, config or self.hybrid_config)

    @staticmethod
    def nearest_route_distance_cm(pose: HybridPose, route: list[HybridPose]) -> float:
        """Return robot-to-route cross-track distance using cached route samples."""
        if not route:
            return float("inf")
        return min(math.hypot(pose.x_cm - point.x_cm, pose.y_cm - point.y_cm) for point in route)

    @staticmethod
    def compute_route_tracking_error(robot_pose: RobotPose, route: list[HybridPose]) -> RouteTrackingError | None:
        """Project live robot pose onto the closest cached route segment."""
        if len(route) < 2:
            return None

        rx = float(robot_pose.x_cm)
        ry = float(robot_pose.y_cm)
        best: RouteTrackingError | None = None
        best_distance = float("inf")

        for index in range(len(route) - 1):
            start = route[index]
            end = route[index + 1]
            sx, sy = float(start.x_cm), float(start.y_cm)
            vx = float(end.x_cm - start.x_cm)
            vy = float(end.y_cm - start.y_cm)
            segment_len_sq = vx * vx + vy * vy
            if segment_len_sq <= 1e-9:
                continue

            projection = ((rx - sx) * vx + (ry - sy) * vy) / segment_len_sq
            clamped = float(np.clip(projection, 0.0, 1.0))
            cx = sx + vx * clamped
            cy = sy + vy * clamped
            dx = rx - cx
            dy = ry - cy
            distance = math.hypot(dx, dy)
            if distance >= best_distance:
                continue

            segment_heading = math.atan2(vy, vx)
            cross = vx * (ry - sy) - vy * (rx - sx)
            signed_distance = math.copysign(distance, cross) if abs(cross) > 1e-9 else 0.0
            best_distance = distance
            best = RouteTrackingError(
                xte_cm=distance,
                signed_xte_cm=signed_distance,
                heading_error_rad=normalize_planner_angle(segment_heading - robot_pose.heading_rad),
                closest_point_cm=(cx, cy),
                segment_heading_rad=segment_heading,
                segment_index=index,
            )

        return best
