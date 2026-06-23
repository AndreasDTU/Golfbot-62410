"""Tests for the faster-approach optimization.

Covers the coordinated changes:

1. **Drive-straight-in selection** -- among SAFE candidates, the chosen pickup
   pose lines up with the robot's incoming travel so it does not pivot hard
   beside the ball.
2. **SAFE pickup zone** -- open balls accept any SAFE point on the reach circle
   and re-aim at the ball; constrained pickups keep the precise approach.
   Arrival and heading tolerances are unchanged -- only *which* point is free.
3. **Commit lock** -- a committed pickup pose is reused verbatim and visited
   first, so a replan cannot flip the approach side near the ball.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from path.models import HybridPose, SafePickupZone, tube_center_for_pose
from path.pickup_geometry import PickupCandidate, PickupCategory
from path.route_strategy import IntersectionPriorityStrategy
from test_route_strategy import _build_route_input, make_ball, make_geometry


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


# ---------------------------------------------------------------------------
# 4. SafePickupZone — accept any SAFE periphery point, re-aim, grab
# ---------------------------------------------------------------------------

class TestSafePickupZone:
    # Standoff (radial) tolerance the executor uses, = ball_arrival_cm.
    TOL = 0.5

    def _zone(self) -> SafePickupZone:
        # Ball at (50,40), reach 13, tube on-axis, SAFE arc = bearings within
        # ±90° of bearing 0 (the +x side of the ball).
        return SafePickupZone(
            ball_x_cm=50.0, ball_y_cm=40.0, reach_cm=13.0,
            mounting_offset_rad=0.0, safe_arcs=((0.0, math.pi / 2),),
        )

    def test_accepts_on_circle_inside_arc(self):
        z = self._zone()
        assert z.accepts(63.0, 40.0, self.TOL)   # bearing 0, exactly on circle
        # bearing 80° (inside the ±90° arc), on the circle
        assert z.accepts(50.0 + 13.0 * math.cos(math.radians(80)),
                         40.0 + 13.0 * math.sin(math.radians(80)), self.TOL)

    def test_rejects_outside_arc(self):
        z = self._zone()
        # bearing 100° — outside the ±90° SAFE arc
        assert not z.accepts(50.0 + 13.0 * math.cos(math.radians(100)),
                             40.0 + 13.0 * math.sin(math.radians(100)), self.TOL)

    def test_rejects_off_radial_band(self):
        z = self._zone()
        assert z.accepts(63.0, 40.0, self.TOL)        # on the circle
        assert not z.accepts(64.0, 40.0, self.TOL)    # 1 cm beyond reach > 0.5

    def test_grab_heading_points_at_ball(self):
        z = self._zone()
        # East of the ball -> face west; north of the ball -> face south.
        # Compare via unit vector so the pi/-pi wrap doesn't matter.
        h_east = z.grab_heading(63.0, 40.0)
        assert math.cos(h_east) == pytest.approx(-1.0, abs=1e-6)
        assert math.sin(h_east) == pytest.approx(0.0, abs=1e-6)
        h_north = z.grab_heading(50.0, 53.0)
        assert math.cos(h_north) == pytest.approx(0.0, abs=1e-6)
        assert math.sin(h_north) == pytest.approx(-1.0, abs=1e-6)

    @pytest.mark.parametrize("ball_xy", [(40.0, 40.0), (125.0, 40.0), (40.0, 90.0)])
    def test_zone_grab_heading_lands_tube_on_ball(self, ball_xy):
        # Inverse property: from any accepted periphery point, turning to
        # grab_heading puts the tube tip on the ball center.
        bx, by = ball_xy
        balls = [make_ball(bx, by, label="white", track_id=1)]
        inp = _build_route_input(seed=42, balls=balls)
        stop = IntersectionPriorityStrategy().plan(inp).stops[0]
        zone = stop.pickup_zone
        assert zone is not None
        c = stop.candidate
        assert zone.accepts(c.x_cm, c.y_cm, self.TOL)
        h = zone.grab_heading(c.x_cm, c.y_cm)
        tip_x, tip_y = tube_center_for_pose(HybridPose(c.x_cm, c.y_cm, h), make_geometry())
        assert math.hypot(tip_x - bx, tip_y - by) < 0.05

    def test_constrained_pickup_has_no_zone(self):
        # Ball hard against the cross -> non-SAFE -> no zone (precise approach).
        balls = [make_ball(96.5, 60.75, label="white", track_id=1)]
        stop = IntersectionPriorityStrategy().plan(
            _build_route_input(seed=42, balls=balls)
        ).stops
        if stop and stop[0].candidate.category is not PickupCategory.SAFE:
            assert stop[0].pickup_zone is None
