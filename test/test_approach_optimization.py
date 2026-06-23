"""Tests for the faster-approach optimization.

Covers the three coordinated changes:

1. **Drive-straight-in selection** -- among SAFE candidates, the chosen pickup
   pose lines up with the robot's incoming travel so it does not pivot hard
   beside the ball.
2. **Acceptance window** -- open (SAFE) balls carry a wide final-heading
   tolerance (capped by the tube capture angle and the SAFE arc); constrained
   pickups stay tight (``None``).
3. **Commit lock** -- a committed pickup pose is reused verbatim and visited
   first, so a replan cannot flip the approach side near the ball.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from path.models import HybridPose
from path.pickup_geometry import (
    PickupCandidate,
    PickupCategory,
    acceptance_tolerance_rad,
    capture_tolerance_rad,
)
from path.route_strategy import IntersectionPriorityStrategy
from test_route_strategy import _build_route_input, make_ball


def _final_turn(stop, sx: float, sy: float) -> float:
    """Final in-place turn the robot owes arriving at *stop* from (sx, sy)."""
    bearing = math.atan2(stop.candidate.y_cm - sy, stop.candidate.x_cm - sx)
    return abs((stop.candidate.theta_rad - bearing + math.pi) % (2.0 * math.pi) - math.pi)


# ---------------------------------------------------------------------------
# 1. Drive-straight-in selection
# ---------------------------------------------------------------------------

class TestDriveStraightInSelection:
    def test_turn_weight_reduces_final_turn(self):
        balls = [make_ball(50.0, 40.0, label="white", track_id=1)]
        base = _build_route_input(seed=42, balls=balls)
        sx, sy = base.start_pose.x_cm, base.start_pose.y_cm

        stop_unweighted = IntersectionPriorityStrategy().plan(
            replace(base, approach_turn_weight=0.0)
        ).stops[0]
        stop_weighted = IntersectionPriorityStrategy().plan(
            replace(base, approach_turn_weight=30.0)
        ).stops[0]

        assert _final_turn(stop_weighted, sx, sy) <= _final_turn(stop_unweighted, sx, sy) + 1e-9
        # With weighting the robot can essentially roll straight in.
        assert math.degrees(_final_turn(stop_weighted, sx, sy)) < 20.0


# ---------------------------------------------------------------------------
# 2. Acceptance window
# ---------------------------------------------------------------------------

class TestAcceptanceWindow:
    def test_capture_tolerance_geometry(self):
        # asin(mouth / ring); degenerate inputs give zero.
        assert capture_tolerance_rad(2.0, 13.1) == pytest.approx(math.asin(2.0 / 13.1))
        assert capture_tolerance_rad(0.0, 13.1) == 0.0
        assert capture_tolerance_rad(2.0, 0.0) == 0.0
        # Mouth wider than reach clamps to a quarter turn, never NaN.
        assert capture_tolerance_rad(99.0, 1.0) == pytest.approx(math.pi / 2)

    def test_open_ball_window_is_wide(self):
        balls = [make_ball(50.0, 40.0, label="white", track_id=1)]
        inp = replace(_build_route_input(seed=42, balls=balls), approach_turn_weight=20.0)
        stop = IntersectionPriorityStrategy().plan(inp).stops[0]
        geo = inp.geometry_result
        assert stop.candidate.category is PickupCategory.SAFE
        assert stop.accept_heading_tol_rad is not None
        # Wider than the tight 1.5 deg default; the SAFE-position capture angle.
        assert stop.accept_heading_tol_rad > math.radians(1.5)
        assert stop.accept_heading_tol_rad == pytest.approx(
            capture_tolerance_rad(geo.mouth_radius_cm, geo.ring_radius_cm)
        )

    def test_non_safe_candidate_has_no_window(self):
        # acceptance_tolerance_rad returns None for non-SAFE candidates so the
        # executor keeps its tight default and careful straight-line approach.
        c = PickupCandidate(0.0, 0.0, 0.0, PickupCategory.CONSTRAINED, 1.0, 0)
        balls = [make_ball(50.0, 40.0, label="white", track_id=1)]
        geo = _build_route_input(seed=42, balls=balls).geometry_result
        assert acceptance_tolerance_rad(geo, c) is None


# ---------------------------------------------------------------------------
# 3. Commit lock
# ---------------------------------------------------------------------------

class TestCommitLock:
    def test_locked_pickup_reused_and_first(self):
        balls = [
            make_ball(50.0, 40.0, label="white", track_id=1),
            make_ball(120.0, 90.0, label="white", track_id=2),
        ]
        base = _build_route_input(seed=42, balls=balls)
        locked_pose = HybridPose(x_cm=55.0, y_cm=45.0, theta_rad=1.0)
        inp = replace(base, locked_pickups={0: locked_pose})

        result = IntersectionPriorityStrategy().plan(inp)

        assert result.stops[0].ball_index == 0
        c = result.stops[0].candidate
        assert (c.x_cm, c.y_cm, c.theta_rad) == (55.0, 45.0, 1.0)
        assert c.category is PickupCategory.SAFE

    def test_locked_pickup_beats_orange_first(self):
        # The robot is already committed to a white ball, so it finishes that
        # grab before honouring orange-first.
        balls = [
            make_ball(50.0, 40.0, label="white", track_id=1),
            make_ball(120.0, 90.0, label="orange", track_id=2),
        ]
        base = _build_route_input(seed=42, balls=balls)
        inp = replace(base, locked_pickups={0: HybridPose(50.0, 40.0, 0.0)})

        result = IntersectionPriorityStrategy().plan(inp)
        assert result.stops[0].ball_index == 0
