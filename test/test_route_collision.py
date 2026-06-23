"""Collision-avoidance regression tests for the 3-layer pathfinding pipeline.

Verifies that the Route Compiler (Layer 3) and the full plan_route() facade
correctly avoid a centered cross obstacle.  All tests build the obstacle
grid directly in grid coordinates (no GUI, no OccupancyGridBuilder, no
SchematicRenderer) to isolate the pathfinding logic.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from config import FieldConfig
from localization.models import RobotGeometry
from path.models import HybridPose, PlannedBallTarget, WaypointKind
from conftest import load_test_geometry
from path.pickup_geometry import (
    PickupCandidate,
    PickupCategory,
    compute_pickup_geometry,
)
from path.planner import (
    CompileStrategy,
    _bresenham_clear,
    _field_to_grid,
    _grid_astar,
    compile_route,
    plan_route,
)
from path.route_strategy import RouteStop, RouteStrategyResult


# ---------------------------------------------------------------------------
# Constants matching the competition cross
# ---------------------------------------------------------------------------

CROSS_SIZE_CM = 20.0
CROSS_ARM_WIDTH_CM = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_field_config() -> FieldConfig:
    return FieldConfig()  # 167.0 × 121.5


def make_geometry() -> RobotGeometry:
    return load_test_geometry()


HALF_WIDTH_CM = make_geometry().width_cm / 2  # robot half-body width


def make_cross_obstacle_grid(
    field_config: FieldConfig | None = None,
) -> np.ndarray:
    """Build a binary obstacle grid with a centered cross.

    The grid is (grid_height_cm, grid_width_cm) at 1 cm/px with top-left
    origin.  The cross uses competition dimensions (20 cm tip-to-tip,
    3 cm arm width).
    """
    fc = field_config or FieldConfig()
    grid_h, grid_w = fc.grid_height_cm, fc.grid_width_cm
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    cx_cm = fc.width_cm * 0.5
    cy_cm = fc.height_cm * 0.5
    half_size = CROSS_SIZE_CM * 0.5
    half_arm = CROSS_ARM_WIDTH_CM * 0.5
    fh = fc.height_cm

    def fill_rect(x_lo: float, y_lo: float, x_hi: float, y_hi: float) -> None:
        col_lo = max(0, int(round(x_lo)))
        col_hi = min(grid_w - 1, int(round(x_hi)))
        row_lo = max(0, int(round(fh - y_hi)))
        row_hi = min(grid_h - 1, int(round(fh - y_lo)))
        grid[row_lo : row_hi + 1, col_lo : col_hi + 1] = 1

    # Vertical arm
    fill_rect(cx_cm - half_arm, cy_cm - half_size,
              cx_cm + half_arm, cy_cm + half_size)
    # Horizontal arm
    fill_rect(cx_cm - half_size, cy_cm - half_arm,
              cx_cm + half_size, cy_cm + half_arm)
    return grid


def make_distance_field(obstacle_grid: np.ndarray) -> np.ndarray:
    """Compute the distance field matching Layer 1's logic.

    Adds wall borders, then runs cv2.distanceTransform.  Mirrors
    pickup_geometry.py lines 212-220.
    """
    combined = (obstacle_grid > 0).astype(np.uint8)
    combined[0, :] = 1
    combined[-1, :] = 1
    combined[:, 0] = 1
    combined[:, -1] = 1
    free_mask = (combined == 0).astype(np.uint8)
    return cv2.distanceTransform(
        free_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
    )


def make_blocked_grid(
    distance_field: np.ndarray,
    half_width_cm: float = HALF_WIDTH_CM,
) -> np.ndarray:
    return (distance_field < half_width_cm).astype(np.uint8)


def is_inside_cross(
    x_cm: float,
    y_cm: float,
    clearance_cm: float = 0.0,
    field_config: FieldConfig | None = None,
) -> bool:
    """Return True if (x_cm, y_cm) is inside the cross plus clearance."""
    fc = field_config or FieldConfig()
    cx = fc.width_cm * 0.5
    cy = fc.height_cm * 0.5
    half_size = CROSS_SIZE_CM * 0.5 + clearance_cm
    half_arm = CROSS_ARM_WIDTH_CM * 0.5 + clearance_cm
    in_vert = abs(x_cm - cx) <= half_arm and abs(y_cm - cy) <= half_size
    in_horiz = abs(x_cm - cx) <= half_size and abs(y_cm - cy) <= half_arm
    return in_vert or in_horiz


def assert_segment_clear(
    blocked: np.ndarray,
    x0_cm: float,
    y0_cm: float,
    x1_cm: float,
    y1_cm: float,
    field_height_cm: float,
) -> None:
    """Assert that a line segment (field-cm) does not cross blocked cells."""
    c0, r0 = _field_to_grid(x0_cm, y0_cm, field_height_cm)
    c1, r1 = _field_to_grid(x1_cm, y1_cm, field_height_cm)
    assert _bresenham_clear(blocked, c0, r0, c1, r1), (
        f"Segment ({x0_cm:.1f},{y0_cm:.1f})->({x1_cm:.1f},{y1_cm:.1f}) "
        f"passes through blocked cells"
    )


def assert_route_avoids_cross(
    route_plan,
    blocked: np.ndarray,
    field_height_cm: float,
) -> None:
    """Assert every waypoint and every segment of a RoutePlan is collision-free."""
    wps = route_plan.waypoints
    for i, wp in enumerate(wps):
        assert not is_inside_cross(wp.x_cm, wp.y_cm), (
            f"Waypoint {i} ({wp.x_cm:.1f}, {wp.y_cm:.1f}) is inside the cross"
        )
        col, row = _field_to_grid(wp.x_cm, wp.y_cm, field_height_cm)
        h, w = blocked.shape
        if 0 <= row < h and 0 <= col < w:
            assert not blocked[row, col], (
                f"Waypoint {i} ({wp.x_cm:.1f}, {wp.y_cm:.1f}) "
                f"maps to blocked cell ({row}, {col})"
            )

    for i in range(len(wps) - 1):
        assert_segment_clear(
            blocked,
            wps[i].x_cm, wps[i].y_cm,
            wps[i + 1].x_cm, wps[i + 1].y_cm,
            field_height_cm,
        )


def make_ball(
    x_cm: float,
    y_cm: float,
    label: str = "white",
    track_id: int = 1,
) -> PlannedBallTarget:
    fc = FieldConfig()
    col = int(round(x_cm))
    row = int(round(fc.height_cm - y_cm))
    return PlannedBallTarget(
        track_id=track_id,
        label=label,
        x_cm=x_cm,
        y_cm=y_cm,
        node_cm=(col, row),
    )


# ---------------------------------------------------------------------------
# Tests: cross in obstacle grid
# ---------------------------------------------------------------------------

class TestCrossInObstacleGrid:
    def test_cross_center_is_obstacle(self):
        grid = make_cross_obstacle_grid()
        fc = FieldConfig()
        col = round(fc.width_cm * 0.5)
        row = round(fc.height_cm - fc.height_cm * 0.5)
        assert grid[row, col] == 1

    def test_vertical_arm_tips_are_obstacle(self):
        grid = make_cross_obstacle_grid()
        fc = FieldConfig()
        cx_col = round(fc.width_cm * 0.5)
        top_row = round(fc.height_cm - (fc.height_cm * 0.5 + CROSS_SIZE_CM * 0.5))
        bot_row = round(fc.height_cm - (fc.height_cm * 0.5 - CROSS_SIZE_CM * 0.5))
        assert grid[top_row, cx_col] == 1
        assert grid[bot_row, cx_col] == 1

    def test_horizontal_arm_tips_are_obstacle(self):
        grid = make_cross_obstacle_grid()
        fc = FieldConfig()
        cy_row = round(fc.height_cm - fc.height_cm * 0.5)
        left_col = round(fc.width_cm * 0.5 - CROSS_SIZE_CM * 0.5)
        right_col = round(fc.width_cm * 0.5 + CROSS_SIZE_CM * 0.5)
        assert grid[cy_row, left_col] == 1
        assert grid[cy_row, right_col] == 1

    def test_corners_are_free(self):
        grid = make_cross_obstacle_grid()
        assert grid[10, 10] == 0
        assert grid[10, 150] == 0
        assert grid[110, 10] == 0
        assert grid[110, 150] == 0

    def test_grid_shape(self):
        grid = make_cross_obstacle_grid()
        assert grid.shape == (122, 167)

    def test_cross_area_approximately_correct(self):
        grid = make_cross_obstacle_grid()
        obstacle_count = int(np.sum(grid))
        # Two rectangles (roughly 4×21 and 21×4) with overlap.
        assert 100 < obstacle_count < 250


# ---------------------------------------------------------------------------
# Tests: distance field
# ---------------------------------------------------------------------------

class TestDistanceFieldReflectsCross:
    @pytest.fixture
    def setup(self):
        grid = make_cross_obstacle_grid()
        df = make_distance_field(grid)
        fc = FieldConfig()
        return grid, df, fc

    def test_distance_zero_at_cross_center(self, setup):
        _, df, fc = setup
        col = round(fc.width_cm * 0.5)
        row = round(fc.height_cm - fc.height_cm * 0.5)
        assert df[row, col] == 0.0

    def test_distance_zero_at_wall_border(self, setup):
        _, df, _ = setup
        assert df[0, 50] == 0.0
        assert df[121, 50] == 0.0

    def test_distance_positive_far_from_cross(self, setup):
        _, df, _ = setup
        assert df[20, 20] > HALF_WIDTH_CM

    def test_distance_increases_away_from_cross(self, setup):
        _, df, fc = setup
        col = round(53.5)
        row = round(fc.height_cm - 60.75)
        assert df[row, col] > 15.0


# ---------------------------------------------------------------------------
# Tests: blocked grid inflation
# ---------------------------------------------------------------------------

class TestBlockedGridInflation:
    @pytest.fixture
    def setup(self):
        grid = make_cross_obstacle_grid()
        df = make_distance_field(grid)
        blocked = make_blocked_grid(df)
        fc = FieldConfig()
        return blocked, df, fc

    def test_cross_center_is_blocked(self, setup):
        blocked, _, fc = setup
        col = round(fc.width_cm * 0.5)
        row = round(fc.height_cm - fc.height_cm * 0.5)
        assert blocked[row, col] == 1

    def test_point_5cm_from_cross_is_blocked(self, setup):
        blocked, _, fc = setup
        # 5 cm right of horizontal arm tip (x = 93.5 + 5 = 98.5)
        col = round(98.5)
        row = round(fc.height_cm - fc.height_cm * 0.5)
        assert blocked[row, col] == 1

    def test_point_15cm_from_cross_is_free(self, setup):
        blocked, _, fc = setup
        # 15 cm right of horizontal arm tip (x = 93.5 + 15 = 108.5)
        col = round(108.5)
        row = round(fc.height_cm - fc.height_cm * 0.5)
        assert blocked[row, col] == 0

    def test_passage_above_cross_is_open(self, setup):
        blocked, _, fc = setup
        # y = 85 is well above cross top tip (y = 70.75)
        col = round(fc.width_cm * 0.5)
        row = round(fc.height_cm - 85.0)
        assert blocked[row, col] == 0


# ---------------------------------------------------------------------------
# Tests: Bresenham detects cross
# ---------------------------------------------------------------------------

class TestBresenhamDetectsCross:
    @pytest.fixture
    def setup(self):
        grid = make_cross_obstacle_grid()
        df = make_distance_field(grid)
        blocked = make_blocked_grid(df)
        fc = FieldConfig()
        return blocked, fc

    def test_horizontal_line_through_center_is_blocked(self, setup):
        blocked, fc = setup
        c0, r0 = _field_to_grid(30.0, 60.75, fc.height_cm)
        c1, r1 = _field_to_grid(130.0, 60.75, fc.height_cm)
        assert not _bresenham_clear(blocked, c0, r0, c1, r1)

    def test_vertical_line_through_center_is_blocked(self, setup):
        blocked, fc = setup
        c0, r0 = _field_to_grid(83.5, 20.0, fc.height_cm)
        c1, r1 = _field_to_grid(83.5, 100.0, fc.height_cm)
        assert not _bresenham_clear(blocked, c0, r0, c1, r1)

    def test_diagonal_through_center_is_blocked(self, setup):
        blocked, fc = setup
        c0, r0 = _field_to_grid(30.0, 30.0, fc.height_cm)
        c1, r1 = _field_to_grid(130.0, 95.0, fc.height_cm)
        assert not _bresenham_clear(blocked, c0, r0, c1, r1)

    def test_line_above_cross_is_clear(self, setup):
        blocked, fc = setup
        c0, r0 = _field_to_grid(30.0, 100.0, fc.height_cm)
        c1, r1 = _field_to_grid(130.0, 100.0, fc.height_cm)
        assert _bresenham_clear(blocked, c0, r0, c1, r1)

    def test_line_below_cross_is_clear(self, setup):
        blocked, fc = setup
        c0, r0 = _field_to_grid(30.0, 20.0, fc.height_cm)
        c1, r1 = _field_to_grid(130.0, 20.0, fc.height_cm)
        assert _bresenham_clear(blocked, c0, r0, c1, r1)


# ---------------------------------------------------------------------------
# Tests: A* detour around cross
# ---------------------------------------------------------------------------

class TestAstarFindsCrossDetour:
    @pytest.fixture
    def setup(self):
        grid = make_cross_obstacle_grid()
        df = make_distance_field(grid)
        blocked = make_blocked_grid(df)
        fc = FieldConfig()
        return blocked, fc

    def test_astar_finds_path_around_cross(self, setup):
        blocked, fc = setup
        start = _field_to_grid(30.0, 60.75, fc.height_cm)
        goal = _field_to_grid(130.0, 60.75, fc.height_cm)
        path = _grid_astar(blocked, start, goal)
        assert path is not None
        assert len(path) > 2

    def test_astar_path_avoids_blocked_cells(self, setup):
        blocked, fc = setup
        start = _field_to_grid(30.0, 60.75, fc.height_cm)
        goal = _field_to_grid(130.0, 60.75, fc.height_cm)
        path = _grid_astar(blocked, start, goal)
        assert path is not None
        for col, row in path:
            assert not blocked[row, col], (
                f"A* path passes through blocked cell ({row}, {col})"
            )

    def test_astar_returns_none_when_start_blocked(self, setup):
        blocked, fc = setup
        start = _field_to_grid(83.5, 60.75, fc.height_cm)
        goal = _field_to_grid(130.0, 60.75, fc.height_cm)
        path = _grid_astar(blocked, start, goal)
        assert path is None


# ---------------------------------------------------------------------------
# Tests: compile_route avoids cross
# ---------------------------------------------------------------------------

class TestCompileRouteAvoidsCross:
    @pytest.fixture
    def setup(self):
        grid = make_cross_obstacle_grid()
        df = make_distance_field(grid)
        blocked = make_blocked_grid(df)
        fc = FieldConfig()
        geometry = make_geometry()
        return grid, df, blocked, fc, geometry

    def _make_strategy_result(
        self,
        x_cm: float,
        y_cm: float,
        theta: float = 0.0,
    ) -> RouteStrategyResult:
        cand = PickupCandidate(
            x_cm=x_cm, y_cm=y_cm, theta_rad=theta,
            category=PickupCategory.SAFE,
            obstacle_distance_cm=30.0,
            ball_index=0,
        )
        stop = RouteStop(candidate=cand, intermediate_node=None, ball_index=0)
        return RouteStrategyResult(
            stops=(stop,), unreachable_balls=(), unload_position=None,
        )

    def test_single_ball_across_cross_has_detour(self, setup):
        _, df, blocked, fc, geometry = setup
        start = HybridPose(x_cm=30.0, y_cm=60.75, theta_rad=0.0)
        strategy = self._make_strategy_result(130.0, 60.75)
        plan = compile_route(
            strategy, df, start, HALF_WIDTH_CM, fc.height_cm,
        )
        nav_wps = [w for w in plan.waypoints if w.kind == WaypointKind.NAVIGATE]
        assert len(nav_wps) >= 1, "Expected A* detour waypoints"

    def test_all_waypoints_in_free_cells(self, setup):
        _, df, blocked, fc, geometry = setup
        start = HybridPose(x_cm=30.0, y_cm=60.75, theta_rad=0.0)
        strategy = self._make_strategy_result(130.0, 60.75)
        plan = compile_route(
            strategy, df, start, HALF_WIDTH_CM, fc.height_cm,
        )
        for i, wp in enumerate(plan.waypoints):
            col, row = _field_to_grid(wp.x_cm, wp.y_cm, fc.height_cm)
            h, w = blocked.shape
            if 0 <= row < h and 0 <= col < w:
                assert not blocked[row, col], (
                    f"Waypoint {i} at ({wp.x_cm:.1f}, {wp.y_cm:.1f}) is blocked"
                )

    def test_all_segments_collision_free(self, setup):
        _, df, blocked, fc, geometry = setup
        start = HybridPose(x_cm=30.0, y_cm=60.75, theta_rad=0.0)
        strategy = self._make_strategy_result(130.0, 60.75)
        plan = compile_route(
            strategy, df, start, HALF_WIDTH_CM, fc.height_cm,
        )
        assert_route_avoids_cross(plan, blocked, fc.height_cm)


# ---------------------------------------------------------------------------
# Tests: full plan_route end-to-end
# ---------------------------------------------------------------------------

class TestPlanRouteEndToEnd:
    @pytest.fixture
    def setup(self):
        fc = make_field_config()
        geometry = make_geometry()
        grid = make_cross_obstacle_grid(fc)
        df = make_distance_field(grid)
        blocked = make_blocked_grid(df)
        return grid, df, blocked, fc, geometry

    def test_ball_across_cross(self, setup):
        grid, df, blocked, fc, geometry = setup
        start = HybridPose(x_cm=30.0, y_cm=60.75, theta_rad=0.0)
        balls = [make_ball(140.0, 60.75, track_id=1)]
        plan = plan_route(grid, balls, start, geometry, fc)
        assert len(plan.waypoints) >= 1
        assert_route_avoids_cross(plan, blocked, fc.height_cm)

    def test_ball_close_to_cross(self, setup):
        grid, df, blocked, fc, geometry = setup
        start = HybridPose(x_cm=30.0, y_cm=60.75, theta_rad=0.0)
        balls = [make_ball(110.0, 60.75, track_id=1)]
        plan = plan_route(grid, balls, start, geometry, fc)
        assert len(plan.waypoints) >= 1
        pickup_wps = [w for w in plan.waypoints if w.kind == WaypointKind.PICKUP]
        assert len(pickup_wps) == 1

    def test_multiple_balls_both_sides(self, setup):
        grid, df, blocked, fc, geometry = setup
        start = HybridPose(x_cm=30.0, y_cm=30.0, theta_rad=0.0)
        balls = [
            make_ball(30.0, 100.0, label="white", track_id=1),
            make_ball(140.0, 30.0, label="white", track_id=2),
            make_ball(140.0, 100.0, label="white", track_id=3),
        ]
        plan = plan_route(grid, balls, start, geometry, fc)
        assert_route_avoids_cross(plan, blocked, fc.height_cm)

    def test_unload_requires_crossing_field(self, setup):
        grid, df, blocked, fc, geometry = setup
        start = HybridPose(x_cm=140.0, y_cm=60.75, theta_rad=math.pi)
        balls = [make_ball(140.0, 30.0, track_id=1)]
        unload_pos = (20.0, 60.75)
        plan = plan_route(grid, balls, start, geometry, fc, unload_position=unload_pos)
        assert any(w.kind == WaypointKind.UNLOAD for w in plan.waypoints)
        assert_route_avoids_cross(plan, blocked, fc.height_cm)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_start_and_goal_same_side(self):
        fc = make_field_config()
        geometry = make_geometry()
        grid = make_cross_obstacle_grid(fc)
        start = HybridPose(x_cm=30.0, y_cm=30.0, theta_rad=0.0)
        balls = [make_ball(60.0, 30.0, track_id=1)]
        plan = plan_route(grid, balls, start, geometry, fc)
        assert len(plan.waypoints) >= 1
        pickup_wps = [w for w in plan.waypoints if w.kind == WaypointKind.PICKUP]
        assert len(pickup_wps) == 1

    def test_ball_at_cross_center_is_unreachable(self):
        fc = make_field_config()
        geometry = make_geometry()
        grid = make_cross_obstacle_grid(fc)
        start = HybridPose(x_cm=30.0, y_cm=30.0, theta_rad=0.0)
        balls = [make_ball(83.5, 60.75, track_id=1)]
        plan = plan_route(grid, balls, start, geometry, fc)
        pickup_wps = [w for w in plan.waypoints if w.kind == WaypointKind.PICKUP]
        assert len(pickup_wps) == 0

    def test_empty_ball_list(self):
        fc = make_field_config()
        geometry = make_geometry()
        grid = make_cross_obstacle_grid(fc)
        start = HybridPose(x_cm=30.0, y_cm=30.0, theta_rad=0.0)
        plan = plan_route(grid, [], start, geometry, fc)
        assert len(plan.waypoints) == 0


# ---------------------------------------------------------------------------
# Tests: CompileStrategy.MINIMAL vs FULL
# ---------------------------------------------------------------------------

class TestCompileStrategyMinimal:
    """Verify MINIMAL produces fewer waypoints while staying collision-free."""

    @pytest.fixture
    def setup(self):
        grid = make_cross_obstacle_grid()
        df = make_distance_field(grid)
        blocked = make_blocked_grid(df)
        fc = FieldConfig()
        return df, blocked, fc

    def _make_strategy_result(self, x_cm, y_cm, theta=0.0):
        cand = PickupCandidate(
            x_cm=x_cm, y_cm=y_cm, theta_rad=theta,
            category=PickupCategory.SAFE,
            obstacle_distance_cm=30.0,
            ball_index=0,
        )
        stop = RouteStop(candidate=cand, intermediate_node=None, ball_index=0)
        return RouteStrategyResult(
            stops=(stop,), unreachable_balls=(), unload_position=None,
        )

    def test_minimal_is_collision_free(self, setup):
        df, blocked, fc = setup
        start = HybridPose(x_cm=30.0, y_cm=60.75, theta_rad=0.0)
        strategy = self._make_strategy_result(130.0, 60.75)
        plan = compile_route(
            strategy, df, start, HALF_WIDTH_CM, fc.height_cm,
            strategy=CompileStrategy.MINIMAL,
        )
        assert_route_avoids_cross(plan, blocked, fc.height_cm)

    def test_minimal_fewer_waypoints_than_full(self, setup):
        df, blocked, fc = setup
        start = HybridPose(x_cm=30.0, y_cm=60.75, theta_rad=0.0)
        strategy = self._make_strategy_result(130.0, 60.75)
        plan_full = compile_route(
            strategy, df, start, HALF_WIDTH_CM, fc.height_cm,
            strategy=CompileStrategy.FULL,
        )
        plan_minimal = compile_route(
            strategy, df, start, HALF_WIDTH_CM, fc.height_cm,
            strategy=CompileStrategy.MINIMAL,
        )
        nav_full = [w for w in plan_full.waypoints if w.kind == WaypointKind.NAVIGATE]
        nav_minimal = [w for w in plan_minimal.waypoints if w.kind == WaypointKind.NAVIGATE]
        assert len(nav_minimal) <= len(nav_full)

    def test_minimal_has_detour_across_cross(self, setup):
        df, blocked, fc = setup
        start = HybridPose(x_cm=30.0, y_cm=60.75, theta_rad=0.0)
        strategy = self._make_strategy_result(130.0, 60.75)
        plan = compile_route(
            strategy, df, start, HALF_WIDTH_CM, fc.height_cm,
            strategy=CompileStrategy.MINIMAL,
        )
        nav_wps = [w for w in plan.waypoints if w.kind == WaypointKind.NAVIGATE]
        assert len(nav_wps) >= 1, "MINIMAL should still detour around cross"

    def test_both_strategies_same_for_clear_path(self, setup):
        df, blocked, fc = setup
        start = HybridPose(x_cm=30.0, y_cm=30.0, theta_rad=0.0)
        strategy = self._make_strategy_result(60.0, 30.0)
        plan_full = compile_route(
            strategy, df, start, HALF_WIDTH_CM, fc.height_cm,
            strategy=CompileStrategy.FULL,
        )
        plan_minimal = compile_route(
            strategy, df, start, HALF_WIDTH_CM, fc.height_cm,
            strategy=CompileStrategy.MINIMAL,
        )
        nav_full = [w for w in plan_full.waypoints if w.kind == WaypointKind.NAVIGATE]
        nav_minimal = [w for w in plan_minimal.waypoints if w.kind == WaypointKind.NAVIGATE]
        assert len(nav_full) == 0
        assert len(nav_minimal) == 0


# ---------------------------------------------------------------------------
# Tests: ball collision avoidance
# ---------------------------------------------------------------------------

def _make_empty_field_df() -> tuple[np.ndarray, FieldConfig]:
    """Distance field for an empty field (no cross, walls only)."""
    fc = FieldConfig()
    grid = np.zeros((fc.grid_height_cm, fc.grid_width_cm), dtype=np.uint8)
    return make_distance_field(grid), fc


class TestBallCollisionAvoidance:
    """Verify that uncollected balls are treated as collision objects."""

    def _make_strategy_result(self, stops_data):
        """Build a RouteStrategyResult from a list of (x, y, ball_index)."""
        stops = []
        for x, y, bidx in stops_data:
            cand = PickupCandidate(
                x_cm=x, y_cm=y, theta_rad=0.0,
                category=PickupCategory.SAFE,
                obstacle_distance_cm=30.0,
                ball_index=bidx,
            )
            stops.append(RouteStop(
                candidate=cand, intermediate_node=None, ball_index=bidx,
            ))
        return RouteStrategyResult(
            stops=tuple(stops), unreachable_balls=(), unload_position=None,
        )

    def test_ball_in_direct_path_causes_detour(self):
        """A ball sitting between start and goal should force a detour."""
        df, fc = _make_empty_field_df()
        # Start at left, goal at right, blocker ball in the middle.
        start = HybridPose(x_cm=30.0, y_cm=60.0, theta_rad=0.0)
        # balls[0] = goal ball (the one we pick up), balls[1] = blocker
        goal_ball = make_ball(130.0, 60.0, track_id=1)
        ball_blocker = make_ball(80.0, 60.0, track_id=99)
        balls = [goal_ball, ball_blocker]
        strategy = self._make_strategy_result([(130.0, 60.0, 0)])

        # Without balls: direct path (no detour on empty field).
        plan_no_balls = compile_route(
            strategy, df, start, HALF_WIDTH_CM, fc.height_cm,
        )
        nav_no_balls = [w for w in plan_no_balls.waypoints
                        if w.kind == WaypointKind.NAVIGATE]
        assert len(nav_no_balls) == 0

        # With blocker ball: should detour around it.
        plan_with_balls = compile_route(
            strategy, df, start, HALF_WIDTH_CM, fc.height_cm,
            balls=balls,
        )
        nav_with_balls = [w for w in plan_with_balls.waypoints
                          if w.kind == WaypointKind.NAVIGATE]
        assert len(nav_with_balls) >= 1, "Expected detour around blocker ball"

    def test_collected_ball_no_longer_blocks(self):
        """After picking up ball A, it should not block the path to ball B."""
        df, fc = _make_empty_field_df()
        # Three balls in a horizontal line. Visit order: 0, 1, 2.
        # Ball 1 sits between ball 0 and ball 2.
        balls = [
            make_ball(40.0, 60.0, track_id=1),   # index 0 — visited first
            make_ball(80.0, 60.0, track_id=2),    # index 1 — visited second
            make_ball(130.0, 60.0, track_id=3),   # index 2 — visited last
        ]
        start = HybridPose(x_cm=20.0, y_cm=60.0, theta_rad=0.0)
        strategy = self._make_strategy_result([
            (40.0, 60.0, 0),
            (80.0, 60.0, 1),
            (130.0, 60.0, 2),
        ])
        plan = compile_route(
            strategy, df, start, HALF_WIDTH_CM, fc.height_cm,
            balls=balls,
        )
        # The path from ball 1 (index 1) to ball 2 (index 2) should be
        # direct — ball 1 was collected, so it's no longer an obstacle.
        # Count NAVIGATE waypoints between the 2nd and 3rd PICKUP.
        pickups_seen = 0
        nav_after_second_pickup = 0
        for wp in plan.waypoints:
            if wp.kind == WaypointKind.PICKUP:
                pickups_seen += 1
            elif wp.kind == WaypointKind.NAVIGATE and pickups_seen == 2:
                nav_after_second_pickup += 1
        assert nav_after_second_pickup == 0, (
            "Ball 1 was collected — should not block path to ball 2"
        )

    def test_ball_avoidance_collision_free(self):
        """All segments before the pickup should avoid the blocker ball."""
        df, fc = _make_empty_field_df()
        start = HybridPose(x_cm=30.0, y_cm=60.0, theta_rad=0.0)
        goal_ball = make_ball(130.0, 60.0, track_id=1)
        ball_blocker = make_ball(80.0, 60.0, track_id=99)
        balls = [goal_ball, ball_blocker]
        strategy = self._make_strategy_result([(130.0, 60.0, 0)])

        plan = compile_route(
            strategy, df, start, HALF_WIDTH_CM, fc.height_cm,
            balls=balls,
        )
        # Verify on a blocked grid that includes the blocker ball.
        # Goal ball (index 0) is excluded (we're driving to it).
        from path.planner import _stamp_balls_on_blocked
        blocked_base = make_blocked_grid(df)
        blocked_with_blocker = _stamp_balls_on_blocked(
            blocked_base, balls, collected={0},
            ball_radius_cm=2.0, half_width_cm=HALF_WIDTH_CM,
            field_height_cm=fc.height_cm,
        )
        # Every segment in the route should be clear of the blocker.
        wps = plan.waypoints
        for i in range(len(wps) - 1):
            assert_segment_clear(
                blocked_with_blocker,
                wps[i].x_cm, wps[i].y_cm,
                wps[i + 1].x_cm, wps[i + 1].y_cm,
                fc.height_cm,
            )

    def test_no_balls_same_as_before(self):
        """balls=None should produce the same result as the old behavior."""
        df, fc = _make_empty_field_df()
        start = HybridPose(x_cm=30.0, y_cm=60.0, theta_rad=0.0)
        strategy_result = RouteStrategyResult(
            stops=(RouteStop(
                candidate=PickupCandidate(
                    x_cm=130.0, y_cm=60.0, theta_rad=0.0,
                    category=PickupCategory.SAFE,
                    obstacle_distance_cm=30.0, ball_index=0,
                ),
                intermediate_node=None, ball_index=0,
            ),),
            unreachable_balls=(), unload_position=None,
        )
        plan_none = compile_route(
            strategy_result, df, start, HALF_WIDTH_CM, fc.height_cm,
        )
        plan_empty = compile_route(
            strategy_result, df, start, HALF_WIDTH_CM, fc.height_cm,
            balls=[],
        )
        assert len(plan_none.waypoints) == len(plan_empty.waypoints)
