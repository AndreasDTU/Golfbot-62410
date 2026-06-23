"""Unit tests for brain.route_interpreter.interpret_route."""

from __future__ import annotations

import math

from path.models import HybridPose, RoutePlan, RouteWaypoint, WaypointKind

from brain.models import StepKind
from brain.route_interpreter import interpret_route


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nav(x: float, y: float, theta_deg: float = 0.0) -> RouteWaypoint:
    """Shorthand NAVIGATE waypoint."""
    return RouteWaypoint(x, y, math.radians(theta_deg), WaypointKind.NAVIGATE)


def pickup(
    x: float,
    y: float,
    theta_deg: float = 0.0,
    ball_index: int = 0,
    obstacle_constrained: bool = False,
) -> RouteWaypoint:
    """Shorthand PICKUP waypoint."""
    return RouteWaypoint(
        x, y, math.radians(theta_deg), WaypointKind.PICKUP, ball_index,
        obstacle_constrained=obstacle_constrained,
    )


def unload(x: float, y: float, theta_deg: float = 0.0) -> RouteWaypoint:
    """Shorthand UNLOAD waypoint."""
    return RouteWaypoint(x, y, math.radians(theta_deg), WaypointKind.UNLOAD)


def make_plan(waypoints: list[RouteWaypoint]) -> RoutePlan:
    """Build a minimal RoutePlan for testing."""
    return RoutePlan(waypoints=tuple(waypoints))


# ---------------------------------------------------------------------------
# Empty / trivial plans
# ---------------------------------------------------------------------------

class TestEmptyPlan:
    def test_empty_waypoints_returns_no_steps(self):
        plan = make_plan([])
        assert interpret_route(plan) == []

    def test_navigate_only_returns_single_drive(self):
        plan = make_plan([nav(0, 0), nav(10, 0), nav(20, 0)])
        steps = interpret_route(plan)
        assert len(steps) == 1
        assert steps[0].kind == StepKind.DRIVE
        assert len(steps[0].waypoints) == 3

    def test_single_navigate_returns_single_drive(self):
        plan = make_plan([nav(5, 5)])
        steps = interpret_route(plan)
        assert len(steps) == 1
        assert steps[0].kind == StepKind.DRIVE
        assert len(steps[0].waypoints) == 1


# ---------------------------------------------------------------------------
# Pickup splitting
# ---------------------------------------------------------------------------

class TestPickupSplitting:
    def test_single_pickup_splits_into_drive_pickup(self):
        plan = make_plan([nav(0, 0), nav(10, 0), pickup(20, 0)])
        steps = interpret_route(plan)
        assert len(steps) == 2
        assert steps[0].kind == StepKind.DRIVE
        assert len(steps[0].waypoints) == 3  # 2 nav + 1 pickup position
        assert steps[1].kind == StepKind.PICKUP

    def test_obstacle_constrained_flag_propagates_to_pickup_step(self):
        plan = make_plan([
            nav(0, 0),
            pickup(10, 0),                              # unconstrained
            nav(20, 0),
            pickup(30, 0, obstacle_constrained=True),  # against cross/wall
        ])
        steps = interpret_route(plan)
        pickups = [s for s in steps if s.kind == StepKind.PICKUP]
        assert [s.obstacle_constrained for s in pickups] == [False, True]

    def test_two_pickups(self):
        plan = make_plan([nav(0, 0), pickup(10, 0), nav(20, 0), pickup(30, 0)])
        steps = interpret_route(plan)
        kinds = [s.kind for s in steps]
        assert kinds == [
            StepKind.DRIVE, StepKind.PICKUP,
            StepKind.DRIVE, StepKind.PICKUP,
        ]
        assert len(steps[0].waypoints) == 2  # nav + pickup
        assert len(steps[2].waypoints) == 2  # nav + pickup

    def test_pickup_at_first_waypoint(self):
        plan = make_plan([pickup(0, 0), nav(10, 0)])
        steps = interpret_route(plan)
        kinds = [s.kind for s in steps]
        assert kinds == [StepKind.DRIVE, StepKind.PICKUP, StepKind.DRIVE]
        assert len(steps[0].waypoints) == 1  # pickup only
        assert len(steps[2].waypoints) == 1  # trailing nav

    def test_consecutive_pickups(self):
        plan = make_plan([pickup(0, 0), pickup(10, 0)])
        steps = interpret_route(plan)
        kinds = [s.kind for s in steps]
        assert kinds == [
            StepKind.DRIVE, StepKind.PICKUP,
            StepKind.DRIVE, StepKind.PICKUP,
        ]


# ---------------------------------------------------------------------------
# Unload
# ---------------------------------------------------------------------------

class TestUnload:
    def test_unload_produces_drive_then_unload(self):
        plan = make_plan([nav(0, 0), nav(10, 0), unload(20, 0)])
        steps = interpret_route(plan)
        assert len(steps) == 2
        assert steps[0].kind == StepKind.DRIVE
        assert len(steps[0].waypoints) == 3
        assert steps[1].kind == StepKind.UNLOAD

    def test_no_unload_means_no_unload_step(self):
        plan = make_plan([nav(0, 0), nav(10, 0)])
        steps = interpret_route(plan)
        assert not any(s.kind == StepKind.UNLOAD for s in steps)

    def test_only_unload(self):
        plan = make_plan([unload(20, 0)])
        steps = interpret_route(plan)
        assert len(steps) == 2
        assert steps[0].kind == StepKind.DRIVE
        assert steps[0].waypoints[0] == HybridPose(20, 0, 0.0)
        assert steps[1].kind == StepKind.UNLOAD


# ---------------------------------------------------------------------------
# Combined pickup + unload
# ---------------------------------------------------------------------------

class TestCombinedRoute:
    def test_pickup_then_unload(self):
        plan = make_plan([nav(0, 0), pickup(10, 0), nav(20, 0), unload(30, 0)])
        steps = interpret_route(plan)
        kinds = [s.kind for s in steps]
        assert kinds == [
            StepKind.DRIVE, StepKind.PICKUP,
            StepKind.DRIVE, StepKind.UNLOAD,
        ]

    def test_multiple_pickups_then_unload(self):
        plan = make_plan([
            nav(0, 0), pickup(10, 0),
            nav(20, 0), pickup(30, 0),
            nav(40, 0), unload(50, 0),
        ])
        steps = interpret_route(plan)
        kinds = [s.kind for s in steps]
        assert kinds == [
            StepKind.DRIVE, StepKind.PICKUP,
            StepKind.DRIVE, StepKind.PICKUP,
            StepKind.DRIVE, StepKind.UNLOAD,
        ]


# ---------------------------------------------------------------------------
# Waypoint values
# ---------------------------------------------------------------------------

class TestWaypointValues:
    def test_drive_waypoints_match_route_coordinates(self):
        plan = make_plan([nav(0, 0), nav(10, 0), nav(20, 0)])
        steps = interpret_route(plan)
        assert steps[0].waypoints[0] == HybridPose(0, 0, 0.0)
        assert steps[0].waypoints[1] == HybridPose(10, 0, 0.0)
        assert steps[0].waypoints[2] == HybridPose(20, 0, 0.0)

    def test_drive_includes_pickup_pose_as_last_waypoint(self):
        plan = make_plan([nav(0, 0), pickup(10, 0)])
        steps = interpret_route(plan)
        assert steps[0].kind == StepKind.DRIVE
        assert steps[0].waypoints[-1] == HybridPose(10, 0, 0.0)

    def test_drive_includes_unload_pose_as_last_waypoint(self):
        plan = make_plan([nav(0, 0), unload(10, 0)])
        steps = interpret_route(plan)
        assert steps[0].kind == StepKind.DRIVE
        assert steps[0].waypoints[-1] == HybridPose(10, 0, 0.0)

    def test_heading_preserved(self):
        plan = make_plan([nav(0, 0, 45.0), pickup(10, 0, 90.0)])
        steps = interpret_route(plan)
        assert steps[0].waypoints[0].theta_rad == math.radians(45.0)
        assert steps[0].waypoints[1].theta_rad == math.radians(90.0)


# ---------------------------------------------------------------------------
# Constrained ball round-trip pattern
# ---------------------------------------------------------------------------

class TestConstrainedPattern:
    def test_intermediate_pickup_intermediate(self):
        """Constrained ball: NAVIGATE(inter) → PICKUP(ball) → NAVIGATE(inter)."""
        plan = make_plan([
            nav(50, 50),          # approach waypoint
            nav(30, 10),          # intermediate node
            pickup(20, 5),        # pickup position
            nav(30, 10),          # return to intermediate
            nav(60, 60),          # continue route
        ])
        steps = interpret_route(plan)
        kinds = [s.kind for s in steps]
        assert kinds == [StepKind.DRIVE, StepKind.PICKUP, StepKind.DRIVE]
        # DRIVE to pickup includes: approach + intermediate + pickup
        assert len(steps[0].waypoints) == 3
        # DRIVE after pickup: intermediate + continue
        assert len(steps[2].waypoints) == 2
