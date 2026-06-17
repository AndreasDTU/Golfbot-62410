"""Unit tests for brain.brain.BrainController."""

from __future__ import annotations

import math

from control.commander import RobotCommander
from guidance.guidance import GuidanceController
from localization.models import RobotPose
from path.pathfinding.models import HybridPose, RoutePlan
from config import DriveConfig

from brain.models import BrainState, IntentAction
from brain.brain import BrainController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeCommander(RobotCommander):
    """RobotCommander with TCP stubbed out and actuator calls tracked."""

    def __init__(self, drive_config: DriveConfig | None = None, **kwargs):
        kwargs.setdefault("auto_connect", False)
        super().__init__(drive_config=drive_config, **kwargs)
        self.sent_commands: list[str] = []
        self.pickup_calls: int = 0
        self.dropoff_calls: int = 0
        self.pickup_should_fail: bool = False
        self.dropoff_should_fail: bool = False

    def _send(self, cmd: str) -> str:
        self.sent_commands.append(cmd)
        return "ok"

    def _send_nowait(self, cmd: str) -> bool:
        self.sent_commands.append(cmd)
        return True

    def pickup(self) -> str:
        self.pickup_calls += 1
        if self.pickup_should_fail:
            raise RuntimeError("pickup actuator failed")
        return "ok"

    def dropoff(self) -> str:
        self.dropoff_calls += 1
        if self.dropoff_should_fail:
            raise RuntimeError("unload actuator failed")
        return "ok"


def _counter(start: float = 0.0, step: float = 1.0):
    t = [start]

    def tick() -> float:
        val = t[0]
        t[0] += step
        return val

    return tick


def make_commander(**overrides) -> FakeCommander:
    config = DriveConfig(**overrides)
    return FakeCommander(drive_config=config, time_fn=_counter())


def make_brain(**overrides) -> tuple[BrainController, GuidanceController, FakeCommander]:
    cmd = make_commander(**overrides)
    guidance = GuidanceController(cmd, config=cmd._config)
    brain = BrainController(guidance, cmd)
    return brain, guidance, cmd


def wp(x: float, y: float, theta_deg: float = 0.0) -> HybridPose:
    return HybridPose(x_cm=x, y_cm=y, theta_rad=math.radians(theta_deg))


def pose(x: float, y: float, heading_deg: float) -> RobotPose:
    h = math.radians(heading_deg)
    return RobotPose(x_cm=x, y_cm=y, heading_rad=h, tube_x_cm=x, tube_y_cm=y)


def make_plan(
    points: list[HybridPose],
    pickup_poses: list[HybridPose] | None = None,
    unload_pose: HybridPose | None = None,
) -> RoutePlan:
    return RoutePlan(
        points=points,
        active_target=None,
        pickup_poses=pickup_poses or [],
        unload_pose=unload_pose,
    )


DT = 0.033  # ~30 FPS


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_starts_idle(self):
        brain, _, _ = make_brain()
        assert brain.state == BrainState.IDLE

    def test_no_plan_transitions_to_done(self):
        """With no route loaded, step list is empty so IDLE dispatches DONE."""
        brain, _, _ = make_brain()
        state = brain.tick(pose(50, 50, 0), DT)
        assert state == BrainState.DONE

    def test_initial_intent_is_stop(self):
        brain, _, _ = make_brain()
        assert brain.intent.action == IntentAction.STOP

    def test_initial_step_count_is_zero(self):
        brain, _, _ = make_brain()
        assert brain.step_count == 0
        assert brain.step_cursor == 0


# ---------------------------------------------------------------------------
# Drive-only route
# ---------------------------------------------------------------------------

class TestDriveOnly:
    def test_idle_dispatches_to_drive(self):
        brain, _, _ = make_brain()
        p1, p2 = wp(0, 0), wp(50, 0)
        brain.load_route(make_plan([p1, p2]))
        assert brain.step_count == 1

        state = brain.tick(pose(0, 0, 0), DT)
        assert state == BrainState.DRIVE
        assert brain.intent.action == IntentAction.DRIVE

    def test_drive_returns_running_while_in_transit(self):
        brain, _, _ = make_brain()
        p1, p2 = wp(0, 0), wp(50, 0)
        brain.load_route(make_plan([p1, p2]))

        brain.tick(pose(0, 0, 0), DT)  # IDLE -> DRIVE
        state = brain.tick(pose(10, 0, 0), DT)  # still far from p2
        assert state == BrainState.DRIVE

    def test_drive_completes_on_arrival(self):
        brain, _, _ = make_brain()
        p1 = wp(50, 50)
        brain.load_route(make_plan([p1]))

        brain.tick(pose(0, 0, 0), DT)    # IDLE -> DRIVE
        state = brain.tick(pose(50, 50, 0), DT)  # at waypoint -> ARRIVED -> IDLE
        assert state == BrainState.IDLE

        state = brain.tick(pose(50, 50, 0), DT)  # no more steps -> DONE
        assert state == BrainState.DONE

    def test_intent_target_is_last_waypoint(self):
        brain, _, _ = make_brain()
        p1 = wp(0, 0)
        p2 = wp(30, 40)
        brain.load_route(make_plan([p1, p2]))

        brain.tick(pose(0, 0, 0), DT)
        assert brain.intent.target_pose is p2

    def test_multi_waypoint_drive(self):
        """Guidance walks through all waypoints in a DRIVE step."""
        brain, _, _ = make_brain()
        p1, p2, p3 = wp(0, 0), wp(20, 0), wp(40, 0)
        brain.load_route(make_plan([p1, p2, p3]))

        brain.tick(pose(0, 0, 0), DT)     # IDLE -> DRIVE (guidance not ticked)
        brain.tick(pose(0, 0, 0), DT)     # guidance: at p1, advances to p2, drives
        brain.tick(pose(20, 0, 0), DT)    # guidance: at p2, advances to p3, drives
        state = brain.tick(pose(40, 0, 0), DT)  # guidance: at p3, ARRIVED
        assert state == BrainState.IDLE


# ---------------------------------------------------------------------------
# DONE state
# ---------------------------------------------------------------------------

class TestDone:
    def test_done_persists(self):
        brain, _, _ = make_brain()
        p1 = wp(50, 50)
        brain.load_route(make_plan([p1]))

        brain.tick(pose(0, 0, 0), DT)    # DRIVE
        brain.tick(pose(50, 50, 0), DT)  # IDLE
        brain.tick(pose(50, 50, 0), DT)  # DONE

        state = brain.tick(pose(50, 50, 0), DT)
        assert state == BrainState.DONE

    def test_done_intent_is_stop(self):
        brain, _, _ = make_brain()
        brain.tick(pose(0, 0, 0), DT)  # empty plan -> DONE
        assert brain.intent.action == IntentAction.STOP


# ---------------------------------------------------------------------------
# Pickup
# ---------------------------------------------------------------------------

class TestPickup:
    def test_pickup_dispatches_from_idle(self):
        """After a DRIVE step arrives, IDLE dispatches PICKUP on next tick."""
        brain, _, cmd = make_brain()
        p1 = wp(20, 0)  # pickup pose
        plan = make_plan([p1], pickup_poses=[p1])
        brain.load_route(plan)

        # Steps: DRIVE[p1], PICKUP
        assert brain.step_count == 2

        brain.tick(pose(0, 0, 0), DT)    # IDLE -> DRIVE
        brain.tick(pose(20, 0, 0), DT)   # arrive -> IDLE (cursor=1)

        state = brain.tick(pose(20, 0, 0), DT)  # IDLE -> PICKUP
        assert state == BrainState.PICKUP
        assert brain.intent.action == IntentAction.PICKUP

    def test_pickup_calls_commander_on_next_tick(self):
        """The blocking pickup call happens on the tick AFTER dispatching."""
        brain, _, cmd = make_brain()
        p1 = wp(20, 0)
        plan = make_plan([p1], pickup_poses=[p1])
        brain.load_route(plan)

        brain.tick(pose(0, 0, 0), DT)    # DRIVE
        brain.tick(pose(20, 0, 0), DT)   # arrive -> IDLE
        brain.tick(pose(20, 0, 0), DT)   # IDLE -> PICKUP (dispatch)

        assert cmd.pickup_calls == 0  # not called yet on dispatch tick

        state = brain.tick(pose(20, 0, 0), DT)  # PICKUP -> blocks -> IDLE
        assert cmd.pickup_calls == 1
        assert state == BrainState.IDLE

    def test_pickup_failure_enters_error(self):
        brain, _, cmd = make_brain()
        p1 = wp(20, 0)
        plan = make_plan([p1], pickup_poses=[p1])
        brain.load_route(plan)

        brain.tick(pose(0, 0, 0), DT)    # DRIVE
        brain.tick(pose(20, 0, 0), DT)   # arrive -> IDLE
        brain.tick(pose(20, 0, 0), DT)   # IDLE -> PICKUP (dispatch)

        cmd.pickup_should_fail = True
        state = brain.tick(pose(20, 0, 0), DT)  # PICKUP -> fails -> ERROR
        assert state == BrainState.ERROR
        assert "pickup" in brain.error_message.lower()


# ---------------------------------------------------------------------------
# Unload
# ---------------------------------------------------------------------------

class TestUnload:
    def test_unload_calls_commander(self):
        brain, _, cmd = make_brain()
        p1 = wp(20, 0)
        unload = wp(20, 0)  # separate object
        plan = make_plan([p1], unload_pose=unload)
        brain.load_route(plan)

        # Steps: DRIVE[p1], UNLOAD
        assert brain.step_count == 2

        brain.tick(pose(0, 0, 0), DT)    # DRIVE
        brain.tick(pose(20, 0, 0), DT)   # arrive -> IDLE
        brain.tick(pose(20, 0, 0), DT)   # IDLE -> UNLOAD (dispatch)
        state = brain.tick(pose(20, 0, 0), DT)  # UNLOAD -> blocks -> IDLE
        assert cmd.dropoff_calls == 1
        assert state == BrainState.IDLE

    def test_unload_failure_enters_error(self):
        brain, _, cmd = make_brain()
        p1 = wp(20, 0)
        plan = make_plan([p1], unload_pose=wp(20, 0))
        brain.load_route(plan)

        brain.tick(pose(0, 0, 0), DT)    # DRIVE
        brain.tick(pose(20, 0, 0), DT)   # arrive -> IDLE
        brain.tick(pose(20, 0, 0), DT)   # IDLE -> UNLOAD (dispatch)

        cmd.dropoff_should_fail = True
        state = brain.tick(pose(20, 0, 0), DT)  # UNLOAD -> fails -> ERROR
        assert state == BrainState.ERROR
        assert "unload" in brain.error_message.lower()


# ---------------------------------------------------------------------------
# Full route: drive -> pickup -> drive -> unload -> done
# ---------------------------------------------------------------------------

class TestFullRoute:
    def test_complete_sequence(self):
        brain, _, cmd = make_brain()
        p1 = wp(0, 0)
        p2 = wp(20, 0)   # pickup
        p3 = wp(40, 0)
        p4 = wp(60, 0)
        unload = wp(60, 0)
        plan = make_plan(
            [p1, p2, p3, p4],
            pickup_poses=[p2],
            unload_pose=unload,
        )
        brain.load_route(plan)

        # Steps: DRIVE[p1,p2], PICKUP, DRIVE[p3,p4], UNLOAD
        assert brain.step_count == 4

        # Drive segment 1
        brain.tick(pose(0, 0, 0), DT)    # IDLE -> DRIVE (guidance not ticked)
        brain.tick(pose(0, 0, 0), DT)    # guidance: at p1, advances to p2
        brain.tick(pose(20, 0, 0), DT)   # guidance: at p2, ARRIVED -> IDLE

        # Pickup
        brain.tick(pose(20, 0, 0), DT)   # IDLE -> PICKUP (dispatch)
        brain.tick(pose(20, 0, 0), DT)   # PICKUP -> blocks -> IDLE
        assert cmd.pickup_calls == 1

        # Drive segment 2
        brain.tick(pose(40, 0, 0), DT)   # IDLE -> DRIVE (guidance not ticked)
        brain.tick(pose(40, 0, 0), DT)   # guidance: at p3, advances to p4
        brain.tick(pose(60, 0, 0), DT)   # guidance: at p4, ARRIVED -> IDLE

        # Unload
        brain.tick(pose(60, 0, 0), DT)   # IDLE -> UNLOAD (dispatch)
        brain.tick(pose(60, 0, 0), DT)   # UNLOAD -> blocks -> IDLE
        assert cmd.dropoff_calls == 1

        # Done
        state = brain.tick(pose(60, 0, 0), DT)
        assert state == BrainState.DONE


# ---------------------------------------------------------------------------
# Error and recovery
# ---------------------------------------------------------------------------

class TestErrorRecovery:
    def test_recover_returns_to_idle(self):
        brain, _, cmd = make_brain()
        p1 = wp(20, 0)
        plan = make_plan([p1], pickup_poses=[p1])
        brain.load_route(plan)

        brain.tick(pose(0, 0, 0), DT)    # DRIVE
        brain.tick(pose(20, 0, 0), DT)   # arrive -> IDLE
        brain.tick(pose(20, 0, 0), DT)   # IDLE -> PICKUP

        cmd.pickup_should_fail = True
        brain.tick(pose(20, 0, 0), DT)   # PICKUP fails -> ERROR
        assert brain.state == BrainState.ERROR

        state = brain.recover()
        assert state == BrainState.IDLE
        assert brain.error_message == ""

    def test_error_persists_without_recover(self):
        brain, _, cmd = make_brain()
        p1 = wp(20, 0)
        plan = make_plan([p1], pickup_poses=[p1])
        brain.load_route(plan)

        brain.tick(pose(0, 0, 0), DT)
        brain.tick(pose(20, 0, 0), DT)
        brain.tick(pose(20, 0, 0), DT)

        cmd.pickup_should_fail = True
        brain.tick(pose(20, 0, 0), DT)

        # Ticking again stays in ERROR
        state = brain.tick(pose(20, 0, 0), DT)
        assert state == BrainState.ERROR

    def test_recover_retries_same_step(self):
        """After recovery, the same step is retried (cursor does not advance)."""
        brain, _, cmd = make_brain()
        p1 = wp(20, 0)
        plan = make_plan([p1], pickup_poses=[p1])
        brain.load_route(plan)

        brain.tick(pose(0, 0, 0), DT)
        brain.tick(pose(20, 0, 0), DT)
        brain.tick(pose(20, 0, 0), DT)   # dispatch PICKUP

        cmd.pickup_should_fail = True
        brain.tick(pose(20, 0, 0), DT)   # fails -> ERROR
        cursor_at_error = brain.step_cursor

        brain.recover()
        assert brain.step_cursor == cursor_at_error  # cursor unchanged

        cmd.pickup_should_fail = False
        brain.tick(pose(20, 0, 0), DT)   # IDLE -> PICKUP (retry)
        brain.tick(pose(20, 0, 0), DT)   # PICKUP succeeds -> IDLE
        assert cmd.pickup_calls == 2  # called twice (once failed, once succeeded)

    def test_recover_is_noop_when_not_in_error(self):
        brain, _, _ = make_brain()
        assert brain.state == BrainState.IDLE
        state = brain.recover()
        assert state == BrainState.IDLE


# ---------------------------------------------------------------------------
# Route management
# ---------------------------------------------------------------------------

class TestRouteManagement:
    def test_load_route_resets_to_idle(self):
        brain, _, _ = make_brain()
        p1 = wp(50, 50)
        brain.load_route(make_plan([p1]))
        brain.tick(pose(50, 50, 0), DT)  # DRIVE -> arrives -> IDLE
        brain.tick(pose(50, 50, 0), DT)  # DONE

        # Load new route
        p2 = wp(80, 80)
        brain.load_route(make_plan([p2]))
        assert brain.state == BrainState.IDLE
        assert brain.step_cursor == 0
        assert brain.step_count == 1

    def test_reset_clears_everything(self):
        brain, _, _ = make_brain()
        p1 = wp(50, 50)
        brain.load_route(make_plan([p1]))
        brain.tick(pose(0, 0, 0), DT)  # DRIVE
        assert brain.state == BrainState.DRIVE

        brain.reset()
        assert brain.state == BrainState.IDLE
        assert brain.step_count == 0
        assert brain.step_cursor == 0

    def test_step_cursor_advances_on_completion(self):
        brain, _, _ = make_brain()
        p1 = wp(20, 0)
        p2 = wp(40, 0)
        plan = make_plan([p1, p2], pickup_poses=[p2])
        brain.load_route(plan)

        # Steps: DRIVE[p1,p2], PICKUP
        assert brain.step_cursor == 0
        brain.tick(pose(20, 0, 0), DT)   # IDLE -> DRIVE (step 0)
        assert brain.step_cursor == 0
        brain.tick(pose(20, 0, 0), DT)   # guidance: at p1, advances to p2
        assert brain.step_cursor == 0
        brain.tick(pose(40, 0, 0), DT)   # guidance: at p2, ARRIVED -> IDLE (cursor=1)
        assert brain.step_cursor == 1


# ---------------------------------------------------------------------------
# No-pose handling
# ---------------------------------------------------------------------------

class TestNoPose:
    def test_no_pose_stays_in_drive(self):
        """Transient pose loss does not abort the mission."""
        brain, _, _ = make_brain()
        p1 = wp(50, 0)
        brain.load_route(make_plan([p1]))

        brain.tick(pose(0, 0, 0), DT)   # IDLE -> DRIVE
        state = brain.tick(None, DT)     # NO_POSE from guidance
        assert state == BrainState.DRIVE

    def test_pose_recovery_resumes_drive(self):
        brain, _, _ = make_brain()
        p1 = wp(50, 0)
        brain.load_route(make_plan([p1]))

        brain.tick(pose(0, 0, 0), DT)    # DRIVE
        brain.tick(None, DT)              # NO_POSE, stays DRIVE
        state = brain.tick(pose(50, 0, 0), DT)  # pose back, arrives -> IDLE
        assert state == BrainState.IDLE
