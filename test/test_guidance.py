"""Unit tests for guidance.guidance.GuidanceController."""

from __future__ import annotations

import math

import pytest

from config import DriveConfig
from control.commander import RobotCommander
from guidance.guidance import GuidanceController, GuidanceStatus
from localization.models import RobotPose
from path.models import HybridPose
from config import DriveConfig

from path.pathfinding.models import HybridPose

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeCommander(RobotCommander):
    """RobotCommander with TCP stubbed out for unit testing."""

    def __init__(self, drive_config: DriveConfig | None = None, **kwargs):
        kwargs.setdefault("auto_connect", False)
        super().__init__(drive_config=drive_config, **kwargs)
        self.sent_commands: list[str] = []

    def _send(self, cmd: str) -> str:
        self.sent_commands.append(cmd)
        return "ok"

    def _send_nowait(self, cmd: str) -> bool:
        self.sent_commands.append(cmd)
        return True


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


def make_guidance(**overrides) -> tuple[GuidanceController, FakeCommander]:
    cmd = make_commander(**overrides)
    guidance = GuidanceController(cmd, config=cmd._config)
    return guidance, cmd


def pose(x: float, y: float, heading_deg: float) -> RobotPose:
    """Shorthand to build a RobotPose with tube at origin (unused by guidance)."""
    h = math.radians(heading_deg)
    return RobotPose(x_cm=x, y_cm=y, heading_rad=h, tube_x_cm=x, tube_y_cm=y)


def wp(x: float, y: float, heading_deg: float = 0.0) -> HybridPose:
    """Shorthand to build a HybridPose waypoint."""
    return HybridPose(x_cm=x, y_cm=y, theta_rad=math.radians(heading_deg))


def last_lr(cmd: FakeCommander) -> tuple[float, float]:
    assert cmd.sent_commands, "no commands sent"
    last = cmd.sent_commands[-1]
    assert last.startswith("LR"), f"expected LR command, got: {last!r}"
    parts = last.split()
    return float(parts[1]), float(parts[2])


DT = 0.033  # ~30 fps


# ---------------------------------------------------------------------------
# No pose — stop immediately
# ---------------------------------------------------------------------------


class TestNoPose:
    def test_returns_no_pose(self):
        g, cmd = make_guidance()
        g.set_route([wp(80, 50)])
        status = g.tick(None, DT)
        assert status == GuidanceStatus.NO_POSE

    def test_sends_stop(self):
        g, cmd = make_guidance()
        g.set_route([wp(80, 50)])
        g.tick(None, DT)
        assert cmd.sent_commands, "no commands sent"
        last = cmd.sent_commands[-1]
        assert last == "stop", "expected stop command"


# ---------------------------------------------------------------------------
# No route
# ---------------------------------------------------------------------------


class TestNoRoute:
    def test_no_route_set(self):
        g, cmd = make_guidance()
        status = g.tick(pose(50, 50, 0), DT)
        assert status == GuidanceStatus.NO_ROUTE

    def test_cleared_route(self):
        g, cmd = make_guidance()
        g.set_route([wp(80, 50)])
        g.clear_route()
        status = g.tick(pose(50, 50, 0), DT)
        assert status == GuidanceStatus.NO_ROUTE


# ---------------------------------------------------------------------------
# Turn decision — heading error > max_heading_for_forward_rad
# ---------------------------------------------------------------------------


class TestTurnDecision:
    def test_large_heading_error_triggers_turn(self):
        g, cmd = make_guidance()
        # Waypoint is directly to the right (+X), but robot faces up (+Y = 90°).
        # heading_to_target = 0°, robot heading = 90° → error = -90°
        # |error| = 90° > 70° → turn
        g.set_route([wp(80, 50)])
        status = g.tick(pose(50, 50, 90), DT)
        assert status == GuidanceStatus.RUNNING
        # Should have sent a turn command (differential: left and right opposite signs)
        left, right = last_lr(cmd)
        assert left * right < 0  # opposite signs = rotation

    def test_small_heading_error_does_not_turn(self):
        g, cmd = make_guidance()
        # Waypoint ahead along +X, robot faces +X (0°). Error ≈ 0.
        g.set_route([wp(80, 50)])
        status = g.tick(pose(50, 50, 0), DT)
        assert status == GuidanceStatus.RUNNING
        # Should have sent drive (equal-ish speeds, both positive)
        left, right = last_lr(cmd)
        assert left > 0 and right > 0


# ---------------------------------------------------------------------------
# Drive + adjust
# ---------------------------------------------------------------------------


class TestDriveAdjust:
    def test_forward_drive_sends_positive_speeds(self):
        g, cmd = make_guidance()
        g.set_route([wp(80, 50)])
        g.tick(pose(50, 50, 0), DT)
        left, right = last_lr(cmd)
        assert left > 0 and right > 0

    def test_heading_correction_produces_differential(self):
        g, cmd = make_guidance()
        # Waypoint is at (150, 60) — far away, slightly above +X axis.
        # Robot faces +X (0°). Small positive heading error.
        g.set_route([wp(150, 60)])
        # First tick: start driving (uses creep speed for stability)
        g.tick(pose(50, 50, 0), DT)
        # Second tick: robot has moved partway, higher speed, correction visible
        # Simulate that robot has moved 50cm closer
        status = g.tick(pose(100, 53, 3), DT)
        assert status == GuidanceStatus.RUNNING
        left, right = last_lr(cmd)
        # Positive heading error (CCW correction needed):
        # adjust subtracts from left, adds to right → right > left
        assert right > left


# ---------------------------------------------------------------------------
# Waypoint arrival and cursor advance
# ---------------------------------------------------------------------------


class TestArrival:
    def test_arrives_at_single_waypoint(self):
        g, cmd = make_guidance()
        g.set_route([wp(50, 50)])
        # Robot is at the waypoint with matching heading → ARRIVED
        status = g.tick(pose(50, 50, 0), DT)
        assert status == GuidanceStatus.ARRIVED

    def test_cursor_advances_on_intermediate_waypoint(self):
        g, cmd = make_guidance()
        g.set_route([wp(50, 50), wp(80, 50)])
        assert g.cursor == 0
        # Robot is at first waypoint — intermediate, no heading check
        status = g.tick(pose(50, 50, 0), DT)
        assert status == GuidanceStatus.RUNNING
        assert g.cursor == 1

    def test_within_ball_tolerance_enters_alignment(self):
        g, cmd = make_guidance()
        # ball_arrival_cm defaults to 1.0 for last waypoint
        g.set_route([wp(50, 50)])
        # Robot is 0.5 cm away with matching heading → distance OK, heading OK → ARRIVED
        status = g.tick(pose(49.5, 50, 0), DT)
        assert status == GuidanceStatus.ARRIVED

    def test_outside_tolerance_keeps_running(self):
        g, cmd = make_guidance()
        g.set_route([wp(50, 50)])
        # Robot is 10 cm away
        status = g.tick(pose(40, 50, 0), DT)
        assert status == GuidanceStatus.RUNNING


# ---------------------------------------------------------------------------
# Final waypoint heading alignment
# ---------------------------------------------------------------------------


class TestFinalHeadingAlignment:
    def test_position_reached_wrong_heading_keeps_running(self):
        g, cmd = make_guidance()
        # Waypoint heading is 90°, robot heading is 0°
        g.set_route([wp(50, 50, 90)])
        status = g.tick(pose(50, 50, 0), DT)
        # Position reached but heading is off → RUNNING (aligning)
        assert status == GuidanceStatus.RUNNING

    def test_alignment_sends_turn_command(self):
        g, cmd = make_guidance()
        g.set_route([wp(50, 50, 90)])
        g.tick(pose(50, 50, 0), DT)
        # Should have sent a turn (differential wheels)
        left, right = last_lr(cmd)
        assert left * right < 0  # opposite signs = rotation

    def test_alignment_completes_when_heading_matches(self):
        g, cmd = make_guidance()
        g.set_route([wp(50, 50, 90)])
        # First tick: position reached, heading wrong → starts aligning
        status = g.tick(pose(50, 50, 0), DT)
        assert status == GuidanceStatus.RUNNING
        # Second tick: heading now matches → ARRIVED
        status = g.tick(pose(50, 50, 90), DT)
        assert status == GuidanceStatus.ARRIVED

    def test_alignment_within_tolerance_is_arrived(self):
        g, cmd = make_guidance()
        # Default tolerance is 2° (from config.final_heading_tolerance_rad)
        g.set_route([wp(50, 50, 90)])
        # Robot heading is 89° — within 2° of 90° → ARRIVED
        status = g.tick(pose(50, 50, 89), DT)
        assert status == GuidanceStatus.ARRIVED

    def test_alignment_outside_tolerance_keeps_turning(self):
        g, cmd = make_guidance()
        g.set_route([wp(50, 50, 90)])
        # Robot heading is 80° — 10° off, outside 5° tolerance
        status = g.tick(pose(50, 50, 80), DT)
        assert status == GuidanceStatus.RUNNING

    def test_intermediate_waypoints_skip_alignment(self):
        g, cmd = make_guidance()
        # Intermediate waypoint at (50,50) heading 90°, robot heading 0°
        # Should advance cursor without caring about heading
        g.set_route([wp(50, 50, 90), wp(80, 50)])
        status = g.tick(pose(50, 50, 0), DT)
        assert status == GuidanceStatus.RUNNING
        assert g.cursor == 1  # advanced past intermediate

    def test_set_route_resets_alignment(self):
        g, cmd = make_guidance()
        g.set_route([wp(50, 50, 90)])
        # Enter alignment phase
        g.tick(pose(50, 50, 0), DT)
        # Load new route — should reset alignment state
        g.set_route([wp(80, 50)])
        status = g.tick(pose(80, 50, 0), DT)
        assert status == GuidanceStatus.ARRIVED

    def test_clear_route_resets_alignment(self):
        g, cmd = make_guidance()
        g.set_route([wp(50, 50, 90)])
        g.tick(pose(50, 50, 0), DT)  # enter alignment
        g.clear_route()
        status = g.tick(pose(50, 50, 0), DT)
        assert status == GuidanceStatus.NO_ROUTE

    def test_no_pose_during_alignment_stops(self):
        g, cmd = make_guidance()
        g.set_route([wp(50, 50, 90)])
        g.tick(pose(50, 50, 0), DT)  # enter alignment
        status = g.tick(None, DT)
        assert status == GuidanceStatus.NO_POSE


# ---------------------------------------------------------------------------
# Full route completion
# ---------------------------------------------------------------------------


class TestFullRoute:
    def test_three_waypoint_route(self):
        g, cmd = make_guidance()
        g.set_route([wp(50, 50), wp(60, 50), wp(70, 50)])

        # At first waypoint → advances to second
        status = g.tick(pose(50, 50, 0), DT)
        assert status == GuidanceStatus.RUNNING
        assert g.cursor == 1

        # At second waypoint → advances to third
        status = g.tick(pose(60, 50, 0), DT)
        assert status == GuidanceStatus.RUNNING
        assert g.cursor == 2

        # At third (last) waypoint with matching heading → ARRIVED
        status = g.tick(pose(70, 50, 0), DT)
        assert status == GuidanceStatus.ARRIVED

    def test_arrived_status_persists(self):
        g, cmd = make_guidance()
        g.set_route([wp(50, 50)])
        g.tick(pose(50, 50, 0), DT)
        # Subsequent ticks still return ARRIVED
        status = g.tick(pose(50, 50, 0), DT)
        assert status == GuidanceStatus.ARRIVED

    def test_full_route_with_heading_alignment(self):
        g, cmd = make_guidance()
        # Last waypoint requires heading 90°
        g.set_route([wp(50, 50), wp(70, 50, 90)])

        # At first waypoint → advances
        status = g.tick(pose(50, 50, 0), DT)
        assert status == GuidanceStatus.RUNNING
        assert g.cursor == 1

        # At last waypoint position, heading wrong → aligning
        status = g.tick(pose(70, 50, 0), DT)
        assert status == GuidanceStatus.RUNNING

        # Heading now correct → ARRIVED
        status = g.tick(pose(70, 50, 90), DT)
        assert status == GuidanceStatus.ARRIVED


# ---------------------------------------------------------------------------
# Heading wrapping near ±180°
# ---------------------------------------------------------------------------


class TestHeadingWrapping:
    def test_turn_direction_across_180(self):
        g, cmd = make_guidance()
        # Waypoint is behind the robot (along -X).
        # heading_to_target = 180°, robot faces +X (0°).
        # heading_error = normalize(180° - 0°) = ±180° → should pick shortest path
        g.set_route([wp(20, 50)])
        status = g.tick(pose(50, 50, 0), DT)
        assert status == GuidanceStatus.RUNNING
        # Heading error is ±180° which is > 70° → should turn
        left, right = last_lr(cmd)
        assert left * right < 0  # rotation

    def test_slight_ccw_heading_error_near_wrap(self):
        g, cmd = make_guidance()
        # Robot at (50, 50) heading 170°.
        # Waypoint at (20, 50) → heading_to_target = atan2(0, -30) = 180°
        # heading_error = normalize(180° - 170°) = 10°
        # 10° < 70° → drive + adjust (not turn)
        g.set_route([wp(20, 50)])
        status = g.tick(pose(50, 50, 170), DT)
        assert status == GuidanceStatus.RUNNING
        left, right = last_lr(cmd)
        # Positive heading error → CCW correction → right > left
        assert right > left


# ---------------------------------------------------------------------------
# set_route / clear_route
# ---------------------------------------------------------------------------


class TestRouteManagement:
    def test_set_route_resets_cursor(self):
        g, cmd = make_guidance()
        g.set_route([wp(50, 50), wp(60, 50), wp(70, 50)])
        g.tick(pose(50, 50, 0), DT)  # advance cursor to 1
        assert g.cursor == 1

        g.set_route([wp(80, 50), wp(90, 50)])
        assert g.cursor == 0
        assert g.waypoint_count == 2

    def test_clear_route_sends_stop(self):
        g, cmd = make_guidance()
        g.set_route([wp(80, 50)])
        g.clear_route()
        assert cmd.sent_commands, "no commands sent"
        last = cmd.sent_commands[-1]
        assert last == "stop", "expected stop command"

    def test_clear_route_resets_state(self):
        g, cmd = make_guidance()
        g.set_route([wp(50, 50)])
        g.tick(pose(50, 50, 0), DT)  # ARRIVED
        g.clear_route()
        assert g.cursor == 0
        assert g.waypoint_count == 0
        status = g.tick(pose(50, 50, 0), DT)
        assert status == GuidanceStatus.NO_ROUTE
