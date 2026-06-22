"""Tests for route strategies: NearestNeighbor, TwoOpt, Sweep, IntersectionPriority.

Verifies that the factored ``_select_stops`` function matches the old
embedded logic, that all strategies produce valid results, and that
orange-first and ordering constraints are respected.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from config import FieldConfig
from localization.models import RobotGeometry
from path.models import HybridPose, PlannedBallTarget
from path.pickup_geometry import compute_pickup_geometry
from path.route_strategy import (
    IntersectionPriorityStrategy,
    NearestNeighborStrategy,
    RoutePlannerInput,
    RouteStrategyResult,
    SweepStrategy,
    TwoOptStrategy,
    _nearest_neighbor_order,
    _select_stops,
    _tour_distance,
)


# ---------------------------------------------------------------------------
# Helpers (mirrors test_route_collision.py)
# ---------------------------------------------------------------------------

def make_field_config() -> FieldConfig:
    return FieldConfig()


def make_geometry() -> RobotGeometry:
    return RobotGeometry(
        width_cm=19.5,
        front_cm=3.8,
        rear_cm=15.1,
        tube_forward_cm=13.1,
        tube_right_cm=0.0,
        tube_width_cm=6.0,
        mouth_radius_cm=2.0,
    )


CROSS_SIZE_CM = 20.0
CROSS_ARM_WIDTH_CM = 3.0


def make_cross_obstacle_grid(field_config: FieldConfig | None = None) -> np.ndarray:
    fc = field_config or FieldConfig()
    grid_h, grid_w = fc.grid_height_cm, fc.grid_width_cm
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    cx_cm = fc.width_cm * 0.5
    cy_cm = fc.height_cm * 0.5
    half_size = CROSS_SIZE_CM * 0.5
    half_arm = CROSS_ARM_WIDTH_CM * 0.5
    fh = fc.height_cm

    def fill_rect(x_lo, y_lo, x_hi, y_hi):
        col_lo = max(0, int(round(x_lo)))
        col_hi = min(grid_w - 1, int(round(x_hi)))
        row_lo = max(0, int(round(fh - y_hi)))
        row_hi = min(grid_h - 1, int(round(fh - y_lo)))
        grid[row_lo : row_hi + 1, col_lo : col_hi + 1] = 1

    fill_rect(cx_cm - half_arm, cy_cm - half_size, cx_cm + half_arm, cy_cm + half_size)
    fill_rect(cx_cm - half_size, cy_cm - half_arm, cx_cm + half_size, cy_cm + half_arm)
    return grid


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
        track_id=track_id, label=label,
        x_cm=x_cm, y_cm=y_cm, node_cm=(col, row),
    )


def _build_route_input(
    seed: int,
    balls: list[PlannedBallTarget] | None = None,
) -> RoutePlannerInput:
    """Build a RoutePlannerInput from a seed with random-ish ball placements."""
    fc = make_field_config()
    geometry = make_geometry()
    grid = make_cross_obstacle_grid(fc)

    if balls is None:
        rng = np.random.default_rng(seed)
        n_balls = 11
        balls = []
        orange_idx = int(rng.integers(0, n_balls))
        for i in range(n_balls):
            x = float(rng.uniform(20.0, fc.width_cm - 20.0))
            y = float(rng.uniform(20.0, fc.height_cm - 20.0))
            label = "orange" if i == orange_idx else "white"
            balls.append(make_ball(x, y, label=label, track_id=i + 1))

    result = compute_pickup_geometry(
        field_width_cm=fc.width_cm,
        field_height_cm=fc.height_cm,
        obstacle_grid=grid,
        balls=balls,
        geometry=geometry,
    )

    start = HybridPose(
        x_cm=fc.width_cm - geometry.rear_cm - 2.0,
        y_cm=fc.height_cm * 0.5,
        theta_rad=math.pi,
    )

    return RoutePlannerInput(
        geometry_result=result,
        start_pose=start,
        field_width_cm=fc.width_cm,
        field_height_cm=fc.height_cm,
        unload_position=(20.0, fc.height_cm * 0.5),
    )


ALL_STRATEGIES = [
    NearestNeighborStrategy, TwoOptStrategy, SweepStrategy,
    IntersectionPriorityStrategy,
]
TEST_SEEDS = [42, 43, 44, 45, 46]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSelectStopsMatchesNNStep1:
    """Verify that _select_stops produces the same result as NearestNeighborStrategy Step 1."""

    @pytest.mark.parametrize("seed", TEST_SEEDS)
    def test_select_stops_matches_nn_step1(self, seed: int):
        inp = _build_route_input(seed)
        per_ball, unreachable = _select_stops(
            inp.geometry_result, inp.field_height_cm,
        )
        # NearestNeighborStrategy uses _select_stops internally, so the
        # same per_ball data should lead to the same nn ordering.
        nn_result = NearestNeighborStrategy().plan(inp)

        # Verify ball indices match (unordered).
        factored_ball_indices = sorted(
            s.ball_index for s in _nearest_neighbor_order(per_ball, inp)
        )
        nn_ball_indices = sorted(s.ball_index for s in nn_result.stops)
        assert factored_ball_indices == nn_ball_indices

        # Verify unreachable match.
        assert sorted(unreachable) == sorted(nn_result.unreachable_balls)


class TestAllStrategiesProduceValidResult:
    @pytest.mark.parametrize("seed", TEST_SEEDS)
    @pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
    def test_all_strategies_produce_valid_result(self, seed, strategy_cls):
        inp = _build_route_input(seed)
        result = strategy_cls().plan(inp)

        assert isinstance(result, RouteStrategyResult)
        assert isinstance(result.stops, tuple)
        assert isinstance(result.unreachable_balls, tuple)

        # All ball_index values should be valid indices into geometry_result.balls.
        n_balls = len(inp.geometry_result.balls)
        for stop in result.stops:
            assert 0 <= stop.ball_index < n_balls

        # No duplicates in stop ball indices.
        ball_indices = [s.ball_index for s in result.stops]
        assert len(ball_indices) == len(set(ball_indices))

        # Stops + unreachable should cover all reachable/unreachable balls.
        all_indices = set(ball_indices) | set(result.unreachable_balls)
        assert len(all_indices) == n_balls


class TestTwoOptImprovesOrMatchesNN:
    @pytest.mark.parametrize("seed", TEST_SEEDS)
    def test_two_opt_improves_or_matches_nn(self, seed: int):
        inp = _build_route_input(seed)
        nn_result = NearestNeighborStrategy().plan(inp)
        two_opt_result = TwoOptStrategy().plan(inp)

        sx, sy = inp.start_pose.x_cm, inp.start_pose.y_cm
        nn_dist = _tour_distance(list(nn_result.stops), sx, sy)
        two_opt_dist = _tour_distance(list(two_opt_result.stops), sx, sy)

        assert two_opt_dist <= nn_dist + 1e-6, (
            f"2-opt distance ({two_opt_dist:.2f}) should be <= "
            f"nearest-neighbor ({nn_dist:.2f})"
        )


class TestOrangeFirstConstraint:
    def _has_orange(self, inp: RoutePlannerInput) -> bool:
        return any(
            bc.ball.label == "orange" and bc.reachable
            for bc in inp.geometry_result.balls
        )

    @pytest.mark.parametrize("seed", TEST_SEEDS)
    def test_sweep_respects_orange_first(self, seed: int):
        inp = _build_route_input(seed)
        result = SweepStrategy().plan(inp)
        if self._has_orange(inp) and len(result.stops) > 0:
            first_ball_idx = result.stops[0].ball_index
            label = inp.geometry_result.balls[first_ball_idx].ball.label
            assert label == "orange", (
                f"Sweep: first stop should be orange, got {label}"
            )

    @pytest.mark.parametrize("seed", TEST_SEEDS)
    def test_two_opt_respects_orange_first(self, seed: int):
        inp = _build_route_input(seed)
        result = TwoOptStrategy().plan(inp)
        if self._has_orange(inp) and len(result.stops) > 0:
            first_ball_idx = result.stops[0].ball_index
            label = inp.geometry_result.balls[first_ball_idx].ball.label
            assert label == "orange", (
                f"TwoOpt: first stop should be orange, got {label}"
            )

    @pytest.mark.parametrize("seed", TEST_SEEDS)
    def test_nn_respects_orange_first(self, seed: int):
        inp = _build_route_input(seed)
        result = NearestNeighborStrategy().plan(inp)
        if self._has_orange(inp) and len(result.stops) > 0:
            first_ball_idx = result.stops[0].ball_index
            label = inp.geometry_result.balls[first_ball_idx].ball.label
            assert label == "orange", (
                f"NN: first stop should be orange, got {label}"
            )

    @pytest.mark.parametrize("seed", TEST_SEEDS)
    def test_intersection_respects_orange_first(self, seed: int):
        inp = _build_route_input(seed)
        result = IntersectionPriorityStrategy().plan(inp)
        if self._has_orange(inp) and len(result.stops) > 0:
            first_ball_idx = result.stops[0].ball_index
            label = inp.geometry_result.balls[first_ball_idx].ball.label
            assert label == "orange", (
                f"Intersection: first stop should be orange, got {label}"
            )


class TestIntersectionPriorityDeferredSelection:
    """Verify that intersection priority selects near-side candidates."""

    def test_nearby_balls_visited_consecutively(self):
        """Two balls within 2R of each other should appear adjacent in the route."""
        fc = make_field_config()
        geometry = make_geometry()
        R = math.hypot(geometry.tube_forward_cm, geometry.tube_right_cm)

        # Place two close balls (within 2R) and one far ball.
        balls = [
            make_ball(40.0, 30.0, label="white", track_id=1),
            make_ball(40.0 + R * 0.5, 30.0, label="white", track_id=2),  # close to #1
            make_ball(140.0, 90.0, label="orange", track_id=3),           # far away
        ]
        inp = _build_route_input(seed=42, balls=balls)
        result = IntersectionPriorityStrategy().plan(inp)

        # Orange is first.  The two close balls should be adjacent.
        indices = [s.ball_index for s in result.stops]
        if 0 in indices and 1 in indices:
            pos0 = indices.index(0)
            pos1 = indices.index(1)
            assert abs(pos0 - pos1) == 1, (
                f"Close balls should be adjacent but are at positions {pos0}, {pos1}"
            )

    def test_candidate_faces_approach_direction(self):
        """After visiting ball A, ball B's candidate should be on the near side."""
        # Both balls far from cross/walls so they have SAFE candidates
        # all around.  Ball A on the right, ball B to its left.
        # Robot starts far right.  After visiting A, intersection
        # priority should pick B's candidate on the right (near) side.
        balls = [
            make_ball(140.0, 30.0, label="orange", track_id=1),  # A (right)
            make_ball(100.0, 30.0, label="white", track_id=2),   # B (left of A)
        ]
        inp = _build_route_input(seed=42, balls=balls)
        result = IntersectionPriorityStrategy().plan(inp)

        assert len(result.stops) == 2
        stop_b = result.stops[1]

        # B's candidate should be on the right side of B (closer to A),
        # i.e. candidate x > ball center x.
        ball_b = inp.geometry_result.balls[stop_b.ball_index].ball
        assert stop_b.candidate.x_cm > ball_b.x_cm, (
            f"Ball B candidate at x={stop_b.candidate.x_cm:.1f} should be "
            f"to the right of ball center x={ball_b.x_cm:.1f}"
        )
