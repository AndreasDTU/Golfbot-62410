"""Path planning services for robot ball collection routes."""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, replace
from typing import Protocol

import cv2
import numpy as np

from pathfinding.models import HybridPlannerConfig, HybridPose, PlannedBallTarget, RoutePlan, RouteTrackingError
from robot.models import RobotGeometry, RobotPose
from vision.config import FieldConfig, PlannerConfig, RobotGeometryConfig


def normalize_planner_angle(theta_rad: float) -> float:
    """Normalize planner headings to ``[-pi, pi)`` for stable state keys."""
    return (theta_rad + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class PickupStandoffGoal:
    """Kinematically valid TCP handoff pair for one ball pickup."""

    standoff_pose: HybridPose
    final_pickup_pose: HybridPose


class RobotFootprintCollisionChecker:
    """Check an oriented multi-circle robot model against raw red occupancy."""

    VALID_EPSILON_CM: float = 1e-6

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
        return self.pose_violation_score(pose) <= self.VALID_EPSILON_CM

    def pose_violation_score(self, pose: HybridPose) -> float:
        """Return summed boundary/red-zone penetration for start-escape planning."""
        violation = 0.0
        for x_grid, y_grid, radius_cm in self.oriented_circle_centers(pose, self.base_circles):
            violation += max(0.0, radius_cm - x_grid)
            violation += max(0.0, x_grid + radius_cm - (self.width - 1))
            violation += max(0.0, radius_cm - y_grid)
            violation += max(0.0, y_grid + radius_cm - (self.height - 1))
            x_index = int(np.clip(round(x_grid), 0, self.width - 1))
            y_index = int(np.clip(round(y_grid), 0, self.height - 1))
            violation += max(0.0, radius_cm - float(self.distance_to_red[y_index, x_index]))
        return violation


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

    TIGHT_CORNER_PICKUP_THRESHOLD_CM = 3.5
    CORNER_PICKUP_HEADING_OFFSETS_DEG = (-3.0, 3.0, 0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 6.0, -6.0, 9.0, -9.0)
    STANDOFF_HEADING_STEP_DEG = 10.0
    MIN_STANDOFF_BODY_DISTANCE_CM = 15.0

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

    def small_goal_unload_pose_candidates(self, geometry: RobotGeometry) -> list[HybridPose]:
        """Return deterministic rear-unload poses spanning the small goal opening."""
        goal_x, goal_y = self.small_goal_center_cm()
        reach_cm = geometry.rear_cm + geometry.unload_extension_cm
        offsets_cm = (0.0, -2.0, 2.0, -3.5, 3.5)
        return [
            HybridPose(
                x_cm=goal_x + reach_cm,
                y_cm=float(np.clip(goal_y + offset_cm, 0.0, self.field.height_cm)),
                theta_rad=0.0,
            )
            for offset_cm in offsets_cm
        ]

    def unload_goal_error(
        self,
        pose: HybridPose,
        geometry: RobotGeometry,
    ) -> tuple[float, float, float]:
        """Return rear-tip x/y/heading errors for the small-goal unload region."""
        rear_x, rear_y = self.rear_unload_point_for_pose(pose, geometry)
        goal_x, goal_y = self.small_goal_center_cm()
        x_error_cm = abs(rear_x - goal_x)
        y_error_cm = max(0.0, abs(rear_y - goal_y) - 4.0)
        heading_error_rad = abs(normalize_planner_angle(pose.theta_rad))
        return x_error_cm, y_error_cm, heading_error_rad

    def is_unload_goal_reached(
        self,
        pose: HybridPose,
        geometry: RobotGeometry,
    ) -> bool:
        """Return true when the rear unload tip can reasonably deliver into the small goal."""
        x_error_cm, y_error_cm, heading_error_rad = self.unload_goal_error(pose, geometry)
        return (
            x_error_cm <= 3.0
            and y_error_cm <= 1.5
            and heading_error_rad <= math.radians(30.0)
        )

    def unload_heuristic_cost(
        self,
        pose: HybridPose,
        geometry: RobotGeometry,
        dijkstra_heuristic: GridDijkstraHeuristic,
    ) -> float:
        """Return obstacle-aware cost from the rear unload tip to the small goal."""
        rear_tip = self.rear_unload_point_for_pose(pose, geometry)
        cost = dijkstra_heuristic.cost_from_field_point(rear_tip, self.field)
        if not math.isfinite(cost):
            goal_x, goal_y = self.small_goal_center_cm()
            cost = math.hypot(goal_x - rear_tip[0], goal_y - rear_tip[1])
        _x_error_cm, y_error_cm, heading_error_rad = self.unload_goal_error(pose, geometry)
        return max(0.0, cost - 3.0) + y_error_cm * 2.0 + heading_error_rad * 6.0

    def small_goal_staging_center_cm(self, geometry: RobotGeometry) -> tuple[float, float]:
        """Return a broad staging target near the small goal but away from the wall."""
        reach_cm = geometry.rear_cm + geometry.unload_extension_cm
        return max(reach_cm + 18.0, 42.0), self.field.height_cm * 0.5

    def search_staging_region(
        self,
        raw_red_grid: np.ndarray,
        start_pose: HybridPose,
        geometry: RobotGeometry,
        config: HybridPlannerConfig | None = None,
        radius_cm: float = 18.0,
    ) -> list[HybridPose]:
        """Search to a broad robot-origin staging region near the small goal."""
        cfg = config or self.config
        collision_checker = RobotFootprintCollisionChecker(raw_red_grid, geometry, self.field, self.robot_config)
        start_pose = HybridPose(
            x_cm=float(start_pose.x_cm),
            y_cm=float(start_pose.y_cm),
            theta_rad=normalize_planner_angle(start_pose.theta_rad),
        )
        goal_x, goal_y = self.small_goal_staging_center_cm(geometry)
        if (
            collision_checker.is_pose_valid(start_pose)
            and math.hypot(start_pose.x_cm - goal_x, start_pose.y_cm - goal_y) <= radius_cm
        ):
            return [start_pose]

        goal_node = (
            int(np.clip(round(goal_x), 0, raw_red_grid.shape[1] - 1)),
            int(np.clip(round(self.field.height_cm - goal_y), 0, raw_red_grid.shape[0] - 1)),
        )
        dijkstra_heuristic = GridDijkstraHeuristic(raw_red_grid, goal_node)
        start_key = self.state_key(start_pose, cfg.theta_bins)
        open_heap: list[tuple[float, float, int, tuple[int, int, int]]] = []
        counter = 0
        start_h = dijkstra_heuristic.cost_from_field_point((start_pose.x_cm, start_pose.y_cm), self.field)
        if not math.isfinite(start_h):
            start_h = math.hypot(goal_x - start_pose.x_cm, goal_y - start_pose.y_cm)
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
            if (
                collision_checker.is_pose_valid(current_pose)
                and math.hypot(goal_x - current_pose.x_cm, goal_y - current_pose.y_cm) <= radius_cm
            ):
                return self.reconstruct_path(came_from, pose_by_key, current_key)

            expansions += 1
            for neighbor_pose, primitive_cost in self.expand_neighbors(current_pose, cfg):
                neighbor_pose = HybridPose(
                    x_cm=float(neighbor_pose.x_cm),
                    y_cm=float(neighbor_pose.y_cm),
                    theta_rad=normalize_planner_angle(neighbor_pose.theta_rad),
                )
                transition_allowed, _neighbor_violation = self.allows_start_escape_transition(
                    collision_checker,
                    current_pose,
                    neighbor_pose,
                )
                if not transition_allowed:
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
                    heuristic = math.hypot(goal_x - neighbor_pose.x_cm, goal_y - neighbor_pose.y_cm)
                counter += 1
                heapq.heappush(open_heap, (tentative_g + max(0.0, heuristic - radius_cm), tentative_g, counter, neighbor_key))

        if expansions >= cfg.max_expansions:
            print(
                f"Hybrid A* search exhausted max nodes ({cfg.max_expansions}) "
                f"for unload staging after {expansions} expansions."
            )
        return []

    def search_unload_goal(
        self,
        raw_red_grid: np.ndarray,
        start_pose: HybridPose,
        geometry: RobotGeometry,
        config: HybridPlannerConfig | None = None,
        timeout_s: float | None = None,
    ) -> list[HybridPose]:
        """Search to any pose whose rear unload tip reaches the small goal opening."""
        cfg = config or self.config
        deadline = time.perf_counter() + timeout_s if timeout_s is not None else None
        collision_checker = RobotFootprintCollisionChecker(raw_red_grid, geometry, self.field, self.robot_config)
        start_pose = HybridPose(
            x_cm=float(start_pose.x_cm),
            y_cm=float(start_pose.y_cm),
            theta_rad=normalize_planner_angle(start_pose.theta_rad),
        )

        goal_x, goal_y = self.small_goal_center_cm()
        goal_node = (
            int(np.clip(round(goal_x), 0, raw_red_grid.shape[1] - 1)),
            int(np.clip(round(self.field.height_cm - goal_y), 0, raw_red_grid.shape[0] - 1)),
        )
        dijkstra_heuristic = GridDijkstraHeuristic(raw_red_grid, goal_node)
        start_key = self.state_key(start_pose, cfg.theta_bins)
        open_heap: list[tuple[float, float, int, tuple[int, int, int]]] = []
        counter = 0
        start_h = self.unload_heuristic_cost(start_pose, geometry, dijkstra_heuristic)
        heapq.heappush(open_heap, (start_h, 0.0, counter, start_key))

        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        pose_by_key: dict[tuple[int, int, int], HybridPose] = {start_key: start_pose}
        g_score: dict[tuple[int, int, int], float] = {start_key: 0.0}
        expansions = 0

        timed_out = False
        while open_heap and expansions < cfg.max_expansions:
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True
                break
            _f_cost, current_cost, _counter, current_key = heapq.heappop(open_heap)
            if current_cost > g_score.get(current_key, float("inf")):
                continue

            current_pose = pose_by_key[current_key]
            if collision_checker.is_pose_valid(current_pose) and self.is_unload_goal_reached(current_pose, geometry):
                return self.reconstruct_path(came_from, pose_by_key, current_key)

            expansions += 1
            for neighbor_pose, primitive_cost in self.expand_neighbors(current_pose, cfg):
                neighbor_pose = HybridPose(
                    x_cm=float(neighbor_pose.x_cm),
                    y_cm=float(neighbor_pose.y_cm),
                    theta_rad=normalize_planner_angle(neighbor_pose.theta_rad),
                )
                transition_allowed, _neighbor_violation = self.allows_start_escape_transition(
                    collision_checker,
                    current_pose,
                    neighbor_pose,
                )
                if not transition_allowed:
                    continue

                neighbor_key = self.state_key(neighbor_pose, cfg.theta_bins)
                tentative_g = g_score[current_key] + primitive_cost
                if tentative_g >= g_score.get(neighbor_key, float("inf")):
                    continue

                came_from[neighbor_key] = current_key
                pose_by_key[neighbor_key] = neighbor_pose
                g_score[neighbor_key] = tentative_g
                heuristic = self.unload_heuristic_cost(neighbor_pose, geometry, dijkstra_heuristic)
                heading_change = abs(normalize_planner_angle(neighbor_pose.theta_rad - current_pose.theta_rad))
                counter += 1
                heapq.heappush(
                    open_heap,
                    (
                        tentative_g + heuristic + heading_change * 0.2,
                        tentative_g,
                        counter,
                        neighbor_key,
                    ),
                )

        if timed_out:
            print(f"Hybrid A* unload docking timed out after {timeout_s:.2f}s and {expansions} expansions.")
        elif expansions >= cfg.max_expansions:
            print(
                f"Hybrid A* search exhausted max nodes ({cfg.max_expansions}) "
                f"for unload region after {expansions} expansions."
            )
        return []

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

    def tight_corner_pickup_heading(
        self,
        goal_point_cm: tuple[float, float],
    ) -> float | None:
        """Return the diagonal pickup heading for balls tightly tucked into a corner."""
        x_cm, y_cm = goal_point_cm
        threshold = self.TIGHT_CORNER_PICKUP_THRESHOLD_CM
        near_left = x_cm <= threshold
        near_right = self.field.width_cm - x_cm <= threshold
        near_bottom = y_cm <= threshold
        near_top = self.field.height_cm - y_cm <= threshold

        if near_right and near_top:
            return math.radians(45.0)
        if near_right and near_bottom:
            return math.radians(-45.0)
        if near_left and near_top:
            return math.radians(135.0)
        if near_left and near_bottom:
            return math.radians(-135.0)
        return None

    def corner_pickup_pose_candidates(
        self,
        goal_node: tuple[int, int],
        geometry: RobotGeometry,
        goal_point_cm: tuple[float, float],
    ) -> list[HybridPose]:
        """Return deterministic diagonal pickup poses for a tightly cornered ball."""
        heading_rad = self.tight_corner_pickup_heading(goal_point_cm)
        if heading_rad is None:
            return []
        return [
            self.pickup_aligned_pose_for_theta(
                goal_node,
                normalize_planner_angle(heading_rad + math.radians(offset_deg)),
                geometry,
                goal_point_cm,
            )
            for offset_deg in self.CORNER_PICKUP_HEADING_OFFSETS_DEG
        ]

    def pickup_standoff_pose_candidates(
        self,
        goal_node: tuple[int, int],
        geometry: RobotGeometry,
        goal_point_cm: tuple[float, float],
    ) -> list[HybridPose]:
        """Return the discretized ring of final pickup poses whose tube reaches the ball."""
        candidates: list[HybridPose] = []
        seen: set[tuple[int, int, int]] = set()
        heading_count = max(1, int(round(360.0 / max(1.0, self.STANDOFF_HEADING_STEP_DEG))))
        for index in range(heading_count):
            heading = normalize_planner_angle(index * 2.0 * math.pi / heading_count)
            pose = self.pickup_aligned_pose_for_theta(goal_node, heading, geometry, goal_point_cm)
            key = (
                int(round(pose.x_cm * 10.0)),
                int(round(pose.y_cm * 10.0)),
                self.theta_bin(pose.theta_rad),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(pose)
        return candidates

    @staticmethod
    def standoff_pose_for_final_pickup(
        final_pickup_pose: HybridPose,
        near_zone_cm: float,
    ) -> HybridPose:
        """Translate the body center backward along heading for the TCP handoff stop."""
        return HybridPose(
            x_cm=final_pickup_pose.x_cm - math.cos(final_pickup_pose.theta_rad) * near_zone_cm,
            y_cm=final_pickup_pose.y_cm - math.sin(final_pickup_pose.theta_rad) * near_zone_cm,
            theta_rad=final_pickup_pose.theta_rad,
        )

    def valid_pickup_standoff_goals(
        self,
        raw_red_grid: np.ndarray,
        goal_node: tuple[int, int],
        geometry: RobotGeometry,
        goal_point_cm: tuple[float, float],
    ) -> list[PickupStandoffGoal]:
        """Return collision-free standoff/final pairs for a straight TCP pickup move."""
        collision_checker = RobotFootprintCollisionChecker(raw_red_grid, geometry, self.field, self.robot_config)
        valid: list[PickupStandoffGoal] = []
        for final_pickup_pose in self.pickup_standoff_pose_candidates(goal_node, geometry, goal_point_cm):
            if not collision_checker.is_pose_valid(final_pickup_pose):
                continue
            standoff_pose = self.standoff_pose_for_final_pickup(
                final_pickup_pose,
                self.MIN_STANDOFF_BODY_DISTANCE_CM,
            )
            if not collision_checker.is_pose_valid(standoff_pose):
                continue
            tube_x, tube_y = self.tube_center_for_pose(final_pickup_pose, geometry)
            ball_x, ball_y = self.goal_to_field_metric_cm(goal_node, goal_point_cm)
            if math.hypot(tube_x - ball_x, tube_y - ball_y) > 1e-6:
                continue
            if math.hypot(final_pickup_pose.x_cm - ball_x, final_pickup_pose.y_cm - ball_y) + 1e-6 < self.MIN_STANDOFF_BODY_DISTANCE_CM:
                continue
            valid.append(PickupStandoffGoal(standoff_pose=standoff_pose, final_pickup_pose=final_pickup_pose))
        return valid

    def valid_pickup_standoff_poses(
        self,
        raw_red_grid: np.ndarray,
        goal_node: tuple[int, int],
        geometry: RobotGeometry,
        goal_point_cm: tuple[float, float],
    ) -> list[HybridPose]:
        """Return final pickup poses for compatibility with tests and diagnostics."""
        return [
            goal.final_pickup_pose
            for goal in self.valid_pickup_standoff_goals(raw_red_grid, goal_node, geometry, goal_point_cm)
        ]

    def search_corner_pickup(
        self,
        raw_red_grid: np.ndarray,
        start_pose: HybridPose,
        goal_node: tuple[int, int],
        geometry: RobotGeometry,
        config: HybridPlannerConfig | None = None,
        goal_point_cm: tuple[float, float] | None = None,
    ) -> list[HybridPose]:
        """Route to a deterministic diagonal pickup pose for a tightly cornered ball."""
        if goal_point_cm is None:
            return []
        cfg = config or self.config
        collision_checker = RobotFootprintCollisionChecker(raw_red_grid, geometry, self.field, self.robot_config)
        for pickup_pose in self.corner_pickup_pose_candidates(goal_node, geometry, goal_point_cm):
            if not collision_checker.is_pose_valid(pickup_pose):
                continue
            segment = self.search_pose_goal(raw_red_grid, start_pose, pickup_pose, geometry, cfg)
            if segment:
                return segment
        return []

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

    @staticmethod
    def allows_start_escape_transition(
        collision_checker: RobotFootprintCollisionChecker,
        current_pose: HybridPose,
        next_pose: HybridPose,
    ) -> tuple[bool, float]:
        """Allow invalid-start expansion only when collision violation improves."""
        current_violation = collision_checker.pose_violation_score(current_pose)
        next_violation = collision_checker.pose_violation_score(next_pose)
        if current_violation <= collision_checker.VALID_EPSILON_CM:
            return next_violation <= collision_checker.VALID_EPSILON_CM, next_violation
        return (
            next_violation <= collision_checker.VALID_EPSILON_CM
            or next_violation < current_violation - collision_checker.VALID_EPSILON_CM
        ), next_violation

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
            if (
                collision_checker.is_pose_valid(current_pose)
                and self.goal_distance(current_pose, goal_node, geometry, goal_point_cm) <= cfg.goal_tolerance_cm
            ):
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
                transition_allowed, _neighbor_violation = self.allows_start_escape_transition(
                    collision_checker,
                    current_pose,
                    neighbor_pose,
                )
                if not transition_allowed:
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

        if not collision_checker.is_pose_valid(goal_pose):
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
            if (
                collision_checker.is_pose_valid(current_pose)
                and distance_cm <= cfg.goal_tolerance_cm
                and heading_error <= heading_tolerance_rad
            ):
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
                transition_allowed, _neighbor_violation = self.allows_start_escape_transition(
                    collision_checker,
                    current_pose,
                    neighbor_pose,
                )
                if not transition_allowed:
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

    @staticmethod
    def closest_pose_goal(
        pose: HybridPose,
        goal_poses: list[HybridPose],
    ) -> tuple[HybridPose, float, float]:
        """Return the nearest goal pose plus translation and heading errors."""
        best_goal = goal_poses[0]
        best_distance = float("inf")
        best_heading_error = float("inf")
        for goal_pose in goal_poses:
            distance = math.hypot(goal_pose.x_cm - pose.x_cm, goal_pose.y_cm - pose.y_cm)
            heading_error = abs(normalize_planner_angle(goal_pose.theta_rad - pose.theta_rad))
            key = (distance, heading_error)
            if key < (best_distance, best_heading_error):
                best_goal = goal_pose
                best_distance = distance
                best_heading_error = heading_error
        return best_goal, best_distance, best_heading_error

    @staticmethod
    def closest_pickup_standoff_goal(
        pose: HybridPose,
        goals: list[PickupStandoffGoal],
    ) -> tuple[PickupStandoffGoal, float, float]:
        """Return the nearest standoff goal plus standoff translation and heading errors."""
        best_goal = goals[0]
        best_distance = float("inf")
        best_heading_error = float("inf")
        for goal in goals:
            standoff_pose = goal.standoff_pose
            distance = math.hypot(standoff_pose.x_cm - pose.x_cm, standoff_pose.y_cm - pose.y_cm)
            heading_error = abs(normalize_planner_angle(standoff_pose.theta_rad - pose.theta_rad))
            key = (distance, heading_error)
            if key < (best_distance, best_heading_error):
                best_goal = goal
                best_distance = distance
                best_heading_error = heading_error
        return best_goal, best_distance, best_heading_error

    def search_pickup_standoff_goal(
        self,
        raw_red_grid: np.ndarray,
        start_pose: HybridPose,
        goal_node: tuple[int, int],
        geometry: RobotGeometry,
        config: HybridPlannerConfig | None = None,
        goal_point_cm: tuple[float, float] | None = None,
    ) -> list[HybridPose]:
        """Search to any valid pickup standoff pose around a ball."""
        if goal_point_cm is None:
            return []
        cfg = config or self.config
        collision_checker = RobotFootprintCollisionChecker(raw_red_grid, geometry, self.field, self.robot_config)
        standoff_goals = self.valid_pickup_standoff_goals(raw_red_grid, goal_node, geometry, goal_point_cm)
        if not standoff_goals:
            return []

        start_pose = HybridPose(
            x_cm=float(start_pose.x_cm),
            y_cm=float(start_pose.y_cm),
            theta_rad=normalize_planner_angle(start_pose.theta_rad),
        )
        dijkstra_heuristic = GridDijkstraHeuristic(raw_red_grid, goal_node)
        heading_tolerance_rad = max(math.pi / float(cfg.theta_bins), math.radians(8.0))

        start_key = self.state_key(start_pose, cfg.theta_bins)
        open_heap: list[tuple[float, float, int, tuple[int, int, int]]] = []
        counter = 0
        _start_goal, start_distance, start_heading_error = self.closest_pickup_standoff_goal(start_pose, standoff_goals)
        start_h = max(0.0, start_distance - cfg.goal_tolerance_cm) + start_heading_error * 3.0
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
            closest_goal, distance_cm, heading_error = self.closest_pickup_standoff_goal(current_pose, standoff_goals)
            if (
                collision_checker.is_pose_valid(current_pose)
                and distance_cm <= cfg.goal_tolerance_cm
                and heading_error <= heading_tolerance_rad
            ):
                path = self.reconstruct_path(came_from, pose_by_key, current_key)
                standoff_pose = closest_goal.standoff_pose
                if math.hypot(standoff_pose.x_cm - current_pose.x_cm, standoff_pose.y_cm - current_pose.y_cm) > 1e-6:
                    path.append(standoff_pose)
                else:
                    path[-1] = standoff_pose
                path.append(closest_goal.final_pickup_pose)
                return path

            expansions += 1
            for neighbor_pose, primitive_cost in self.expand_neighbors(current_pose, cfg):
                neighbor_pose = HybridPose(
                    x_cm=float(neighbor_pose.x_cm),
                    y_cm=float(neighbor_pose.y_cm),
                    theta_rad=normalize_planner_angle(neighbor_pose.theta_rad),
                )
                transition_allowed, _neighbor_violation = self.allows_start_escape_transition(
                    collision_checker,
                    current_pose,
                    neighbor_pose,
                )
                if not transition_allowed:
                    continue

                neighbor_key = self.state_key(neighbor_pose, cfg.theta_bins)
                tentative_g = g_score[current_key] + primitive_cost
                if tentative_g >= g_score.get(neighbor_key, float("inf")):
                    continue

                came_from[neighbor_key] = current_key
                pose_by_key[neighbor_key] = neighbor_pose
                g_score[neighbor_key] = tentative_g
                tube_point = self.tube_center_for_pose(neighbor_pose, geometry)
                heuristic = dijkstra_heuristic.cost_from_field_point(tube_point, self.field)
                _nearest_goal, pose_distance, pose_heading_error = self.closest_pickup_standoff_goal(neighbor_pose, standoff_goals)
                if not math.isfinite(heuristic):
                    heuristic = pose_distance
                heading_change = abs(normalize_planner_angle(neighbor_pose.theta_rad - current_pose.theta_rad))
                counter += 1
                heapq.heappush(
                    open_heap,
                    (
                        tentative_g
                        + max(0.0, heuristic - cfg.goal_tolerance_cm)
                        + max(0.0, pose_distance - cfg.goal_tolerance_cm) * 0.25
                        + pose_heading_error * 1.5
                        + heading_change * 0.1,
                        tentative_g,
                        counter,
                        neighbor_key,
                    ),
                )

        if expansions >= cfg.max_expansions:
            print(
                f"Hybrid A* search exhausted max nodes ({cfg.max_expansions}) "
                f"for pickup standoff goal {goal_node} after {expansions} expansions."
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

    @staticmethod
    def ball_obstacle_radius_cm(config: HybridPlannerConfig, geometry: RobotGeometry) -> float:
        """Return non-target ball clearance for the robot reference point."""
        return max(0.0, config.ball_radius_cm + geometry.width_cm * 0.5 + config.non_target_ball_extra_clearance_cm)

    def ball_obstacle_targets(
        self,
        ball_targets: list[PlannedBallTarget],
        selected_target: PlannedBallTarget,
        config: HybridPlannerConfig,
    ) -> list[PlannedBallTarget]:
        """Return balls that should be treated as obstacles for this selected target."""
        if not config.avoid_non_target_balls_enabled:
            return []
        return [target for target in ball_targets if target.track_id != selected_target.track_id]

    def grid_with_ball_obstacles(
        self,
        grid: np.ndarray,
        ball_targets: list[PlannedBallTarget],
        selected_target: PlannedBallTarget,
        geometry: RobotGeometry,
        config: HybridPlannerConfig,
    ) -> tuple[np.ndarray, list[PlannedBallTarget]]:
        """Add inflated non-target ball obstacles while leaving the selected ball reachable."""
        obstacle_targets = self.ball_obstacle_targets(ball_targets, selected_target, config)
        if not obstacle_targets:
            return grid, []

        inflated = grid.copy()
        radius_cm = int(math.ceil(self.ball_obstacle_radius_cm(config, geometry)))
        for obstacle in obstacle_targets:
            center = (
                int(np.clip(round(obstacle.x_cm), 0, inflated.shape[1] - 1)),
                int(np.clip(round(self.hybrid_planner.field.height_cm - obstacle.y_cm), 0, inflated.shape[0] - 1)),
            )
            cv2.circle(inflated, center, radius_cm, 1, -1, cv2.LINE_AA)
        return (inflated > 0).astype(np.uint8), obstacle_targets

    def plan_unload_segment(
        self,
        grid: np.ndarray,
        current_pose: HybridPose,
        geometry: RobotGeometry,
        config: HybridPlannerConfig,
    ) -> tuple[list[HybridPose], HybridPose | None, tuple[float, float] | None]:
        """Plan via a broad staging region, then dock into the small goal."""
        staging_segment = self.hybrid_planner.search_staging_region(grid, current_pose, geometry, config)
        if not staging_segment:
            print("Hybrid A* could not route from current pose to the small goal staging region.")
            return [], None, None

        docking_config = replace(config, max_expansions=min(config.max_expansions, 6000))
        docking_segment = self.hybrid_planner.search_unload_goal(
            grid,
            staging_segment[-1],
            geometry,
            docking_config,
            timeout_s=1.0,
        )
        if docking_segment:
            combined = staging_segment + docking_segment[1:]
            return combined, combined[-1], self.hybrid_planner.small_goal_center_cm()

        print("Hybrid A* could not route from current pose to the small goal unload region.")
        return [], None, None

    def plan_target_segment(
        self,
        grid: np.ndarray,
        current_pose: HybridPose,
        target: PlannedBallTarget,
        geometry: RobotGeometry,
        config: HybridPlannerConfig,
    ) -> list[HybridPose]:
        """Plan to one ball, using deterministic diagonal pickup for tight corners."""
        goal_point_cm = (target.x_cm, target.y_cm)
        corner_segment = self.hybrid_planner.search_corner_pickup(
            grid,
            current_pose,
            target.node_cm,
            geometry,
            config,
            goal_point_cm=goal_point_cm,
        )
        if corner_segment:
            return corner_segment
        standoff_segment = self.hybrid_planner.search_pickup_standoff_goal(
            grid,
            current_pose,
            target.node_cm,
            geometry,
            config,
            goal_point_cm=goal_point_cm,
        )
        if standoff_segment:
            return standoff_segment
        return self.hybrid_planner.search(
            grid,
            current_pose,
            target.node_cm,
            geometry,
            config,
            goal_point_cm=goal_point_cm,
        )

    def plan_target_segment_with_ball_avoidance(
        self,
        grid: np.ndarray,
        all_targets: list[PlannedBallTarget],
        current_pose: HybridPose,
        target: PlannedBallTarget,
        geometry: RobotGeometry,
        config: HybridPlannerConfig,
    ) -> tuple[list[HybridPose], list[PlannedBallTarget], str]:
        """Plan to ``target`` with all other balls inserted as inflated hard obstacles."""
        obstacle_grid, obstacle_targets = self.grid_with_ball_obstacles(grid, all_targets, target, geometry, config)
        mode = "hard" if obstacle_targets else ("disabled" if not config.avoid_non_target_balls_enabled else "hard")
        segment = self.plan_target_segment(obstacle_grid, current_pose, target, geometry, config)
        return segment, obstacle_targets, mode

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
        active_ball_obstacles: list[PlannedBallTarget] = []
        ball_avoidance_mode = "disabled" if not cfg.avoid_non_target_balls_enabled else "hard"

        orange_targets = sorted(
            [target for target in unvisited if target.label == "orange"],
            key=lambda target: math.hypot(target.x_cm - current_pose.x_cm, target.y_cm - current_pose.y_cm),
        )
        for orange_target in orange_targets:
            orange_segment, orange_obstacles, _orange_mode = self.plan_target_segment_with_ball_avoidance(
                grid,
                unvisited,
                current_pose,
                orange_target,
                geometry,
                cfg,
            )
            if not orange_segment:
                print(
                    f"Hybrid A* could not route to orange target {orange_target.track_id}; "
                    "keeping orange first and trying last-resort direct orange route."
                )
                if cfg.allow_last_resort_orange_contact:
                    orange_segment = self.plan_target_segment(grid, current_pose, orange_target, geometry, cfg)
                    if orange_segment:
                        active_target = orange_target
                        active_ball_obstacles = orange_obstacles
                        ball_avoidance_mode = "orange forced first"
                        route.extend(orange_segment[1:])
                        current_pose = orange_segment[-1]
                        pickup_poses.append(current_pose)
                        break
                print(f"Hybrid A* could not route to orange target {orange_target.track_id} even as last resort.")
                continue
            active_target = orange_target
            active_ball_obstacles = orange_obstacles
            route.extend(orange_segment[1:])
            current_pose = orange_segment[-1]
            pickup_poses.append(current_pose)
            break

        if orange_targets and active_target is None:
            return RoutePlan(points=[], active_target=None, pickup_poses=[])

        if active_target is not None:
            unvisited = [target for target in unvisited if target.track_id != active_target.track_id]
        else:
            unvisited = [target for target in unvisited if target.label != "orange"]

        while unvisited:
            nearest_candidates = sorted(
                unvisited,
                key=lambda target: math.hypot(target.x_cm - current_pose.x_cm, target.y_cm - current_pose.y_cm),
            )
            chosen_target: PlannedBallTarget | None = None
            chosen_segment: list[HybridPose] = []
            chosen_obstacles: list[PlannedBallTarget] = []

            for candidate in nearest_candidates:
                segment, obstacle_targets, _mode = self.plan_target_segment_with_ball_avoidance(
                    grid,
                    unvisited,
                    current_pose,
                    candidate,
                    geometry,
                    cfg,
                )
                if segment:
                    chosen_target = candidate
                    chosen_segment = segment
                    chosen_obstacles = obstacle_targets
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
                active_ball_obstacles = chosen_obstacles

        unload_pose: HybridPose | None = None
        unload_goal_cm: tuple[float, float] | None = None
        if pickup_poses:
            unload_segment, unload_pose, unload_goal_cm = self.plan_unload_segment(grid, current_pose, geometry, cfg)
            if unload_segment:
                route.extend(unload_segment[1:])
        else:
            route = []

        return RoutePlan(
            points=route,
            active_target=active_target,
            pickup_poses=pickup_poses,
            unload_pose=unload_pose,
            unload_goal_cm=unload_goal_cm,
            ball_obstacles=active_ball_obstacles,
            ball_obstacle_radius_cm=self.ball_obstacle_radius_cm(cfg, geometry) if cfg.avoid_non_target_balls_enabled else 0.0,
            ball_avoidance_mode=ball_avoidance_mode,
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
            avoid_non_target_balls_enabled=self.planner_config.avoid_non_target_balls_enabled,
            ball_radius_cm=self.planner_config.ball_radius_cm,
            non_target_ball_extra_clearance_cm=self.planner_config.non_target_ball_extra_clearance_cm,
            allow_last_resort_orange_contact=self.planner_config.allow_last_resort_orange_contact,
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
