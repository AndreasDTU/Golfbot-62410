"""Unit tests for brain.route_interpreter.interpret_route."""

from __future__ import annotations

import math

from path.pathfinding.models import HybridPose, RoutePlan

from brain.models import StepKind
from brain.route_interpreter import interpret_route


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wp(x: float, y: float, theta_deg: float = 0.0) -> HybridPose:
    """Shorthand to build a HybridPose waypoint."""
    return HybridPose(x_cm=x, y_cm=y, theta_rad=math.radians(theta_deg))


def make_plan(
    points: list[HybridPose],
    pickup_poses: list[HybridPose] | None = None,
    unload_pose: HybridPose | None = None,
) -> RoutePlan:
    """Build a minimal RoutePlan for testing."""
    return RoutePlan(
        points=points,
        pickup_poses=pickup_poses or [],
        unload_pose=unload_pose,
    )


# ---------------------------------------------------------------------------
# Empty / trivial plans
# ---------------------------------------------------------------------------

class TestEmptyPlan:
    def test_empty_points_returns_no_steps(self):
        plan = make_plan(points=[])
        assert interpret_route(plan) == []

    def test_no_actions_returns_single_drive(self):
        p1, p2, p3 = wp(0, 0), wp(10, 0), wp(20, 0)
        plan = make_plan(points=[p1, p2, p3])
        steps = interpret_route(plan)
        assert len(steps) == 1
        assert steps[0].kind == StepKind.DRIVE
        assert len(steps[0].waypoints) == 3

    def test_single_point_returns_single_drive(self):
        p1 = wp(5, 5)
        plan = make_plan(points=[p1])
        steps = interpret_route(plan)
        assert len(steps) == 1
        assert steps[0].kind == StepKind.DRIVE
        assert steps[0].waypoints == (p1,)


# ---------------------------------------------------------------------------
# Pickup splitting
# ---------------------------------------------------------------------------

class TestPickupSplitting:
    def test_single_pickup_splits_into_drive_pickup(self):
        p1 = wp(0, 0)
        p2 = wp(10, 0)
        p3 = wp(20, 0)  # pickup
        plan = make_plan(points=[p1, p2, p3], pickup_poses=[p3])
        steps = interpret_route(plan)
        assert len(steps) == 2
        assert steps[0].kind == StepKind.DRIVE
        assert steps[0].waypoints == (p1, p2, p3)
        assert steps[1].kind == StepKind.PICKUP

    def test_two_pickups(self):
        p1 = wp(0, 0)
        p2 = wp(10, 0)   # pickup 1
        p3 = wp(20, 0)
        p4 = wp(30, 0)   # pickup 2
        plan = make_plan(points=[p1, p2, p3, p4], pickup_poses=[p2, p4])
        steps = interpret_route(plan)
        kinds = [s.kind for s in steps]
        assert kinds == [
            StepKind.DRIVE, StepKind.PICKUP,
            StepKind.DRIVE, StepKind.PICKUP,
        ]
        assert steps[0].waypoints == (p1, p2)
        assert steps[2].waypoints == (p3, p4)

    def test_pickup_at_first_point(self):
        p1 = wp(0, 0)   # pickup
        p2 = wp(10, 0)
        plan = make_plan(points=[p1, p2], pickup_poses=[p1])
        steps = interpret_route(plan)
        kinds = [s.kind for s in steps]
        assert kinds == [StepKind.DRIVE, StepKind.PICKUP, StepKind.DRIVE]
        assert steps[0].waypoints == (p1,)
        assert steps[2].waypoints == (p2,)

    def test_consecutive_pickups(self):
        p1 = wp(0, 0)   # pickup
        p2 = wp(10, 0)  # pickup
        plan = make_plan(points=[p1, p2], pickup_poses=[p1, p2])
        steps = interpret_route(plan)
        kinds = [s.kind for s in steps]
        assert kinds == [
            StepKind.DRIVE, StepKind.PICKUP,
            StepKind.DRIVE, StepKind.PICKUP,
        ]


# ---------------------------------------------------------------------------
# Unload appending
# ---------------------------------------------------------------------------

class TestUnloadAppending:
    def test_unload_appended_at_end(self):
        p1 = wp(0, 0)
        p2 = wp(10, 0)
        p3 = wp(20, 0)
        # unload_pose is a separate object (mimics real planner behavior)
        unload = wp(20, 0)
        plan = make_plan(points=[p1, p2, p3], unload_pose=unload)
        steps = interpret_route(plan)
        assert len(steps) == 2
        assert steps[0].kind == StepKind.DRIVE
        assert steps[1].kind == StepKind.UNLOAD

    def test_no_unload_pose_means_no_unload_step(self):
        p1, p2 = wp(0, 0), wp(10, 0)
        plan = make_plan(points=[p1, p2], unload_pose=None)
        steps = interpret_route(plan)
        assert not any(s.kind == StepKind.UNLOAD for s in steps)

    def test_unload_on_empty_route_still_no_steps(self):
        plan = make_plan(points=[], unload_pose=wp(0, 0))
        assert interpret_route(plan) == []


# ---------------------------------------------------------------------------
# Combined pickup + unload
# ---------------------------------------------------------------------------

class TestCombinedRoute:
    def test_pickup_then_unload(self):
        p1 = wp(0, 0)
        p2 = wp(10, 0)  # pickup
        p3 = wp(20, 0)
        p4 = wp(30, 0)
        unload = wp(30, 0)
        plan = make_plan(
            points=[p1, p2, p3, p4],
            pickup_poses=[p2],
            unload_pose=unload,
        )
        steps = interpret_route(plan)
        kinds = [s.kind for s in steps]
        assert kinds == [
            StepKind.DRIVE, StepKind.PICKUP,
            StepKind.DRIVE, StepKind.UNLOAD,
        ]

    def test_multiple_pickups_then_unload(self):
        p1 = wp(0, 0)
        p2 = wp(10, 0)  # pickup 1
        p3 = wp(20, 0)
        p4 = wp(30, 0)  # pickup 2
        p5 = wp(40, 0)
        p6 = wp(50, 0)
        unload = wp(50, 0)
        plan = make_plan(
            points=[p1, p2, p3, p4, p5, p6],
            pickup_poses=[p2, p4],
            unload_pose=unload,
        )
        steps = interpret_route(plan)
        kinds = [s.kind for s in steps]
        assert kinds == [
            StepKind.DRIVE, StepKind.PICKUP,
            StepKind.DRIVE, StepKind.PICKUP,
            StepKind.DRIVE, StepKind.UNLOAD,
        ]


# ---------------------------------------------------------------------------
# Identity semantics
# ---------------------------------------------------------------------------

class TestIdentitySemantics:
    def test_pickup_uses_identity(self):
        """Pickup detection relies on object identity."""
        p1 = wp(0, 0)
        p2 = wp(10, 0)
        plan = make_plan(points=[p1, p2], pickup_poses=[p2])
        steps = interpret_route(plan)
        assert any(s.kind == StepKind.PICKUP for s in steps)

    def test_equal_but_different_object_not_pickup(self):
        """A different HybridPose with same values does NOT trigger pickup."""
        p1 = wp(0, 0)
        p2 = wp(10, 0)
        p2_copy = wp(10, 0)  # equal value, different object
        assert p2 == p2_copy
        assert p2 is not p2_copy
        plan = make_plan(points=[p1, p2], pickup_poses=[p2_copy])
        steps = interpret_route(plan)
        assert not any(s.kind == StepKind.PICKUP for s in steps)

    def test_unload_does_not_depend_on_identity(self):
        """Unload is appended at end regardless of object identity."""
        p1 = wp(0, 0)
        p2 = wp(10, 0)
        # unload_pose is a completely unrelated object
        unload = wp(99, 99)
        plan = make_plan(points=[p1, p2], unload_pose=unload)
        steps = interpret_route(plan)
        assert steps[-1].kind == StepKind.UNLOAD


# ---------------------------------------------------------------------------
# Waypoint preservation
# ---------------------------------------------------------------------------

class TestWaypointPreservation:
    def test_drive_waypoints_are_exact_objects(self):
        """DRIVE step waypoints are the same HybridPose instances from points."""
        p1 = wp(0, 0)
        p2 = wp(10, 0)
        p3 = wp(20, 0)
        plan = make_plan(points=[p1, p2, p3])
        steps = interpret_route(plan)
        assert steps[0].waypoints[0] is p1
        assert steps[0].waypoints[1] is p2
        assert steps[0].waypoints[2] is p3

    def test_drive_includes_pickup_pose_as_last_waypoint(self):
        """The DRIVE step before a PICKUP includes the pickup pose."""
        p1 = wp(0, 0)
        p2 = wp(10, 0)  # pickup
        plan = make_plan(points=[p1, p2], pickup_poses=[p2])
        steps = interpret_route(plan)
        assert steps[0].kind == StepKind.DRIVE
        assert steps[0].waypoints[-1] is p2
