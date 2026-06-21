"""Unit tests for control.commander.RobotCommander."""

from __future__ import annotations

import math

import pytest

from config import DriveConfig
from control.commander import RobotCommander


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


def make_commander(**overrides) -> FakeCommander:
    config = DriveConfig(**overrides)
    return FakeCommander(drive_config=config, time_fn=_counter())


def _counter(start: float = 0.0, step: float = 1.0):
    """Return a callable that increments by *step* on each call."""
    t = [start]

    def tick() -> float:
        val = t[0]
        t[0] += step
        return val

    return tick


def last_lr(cmd: FakeCommander) -> tuple[float, float]:
    """Extract the last LR command's left/right values."""
    assert cmd.sent_commands, "no commands sent"
    last = cmd.sent_commands[-1]
    assert last.startswith("LR"), f"expected LR command, got: {last!r}"
    parts = last.split()
    return float(parts[1]), float(parts[2])


# ---------------------------------------------------------------------------
# Turn profiling (quadratic acceleration)
# ---------------------------------------------------------------------------

class TestTurn:
    def test_large_angle_approaches_max_speed(self):
        """Large angle should eventually reach turn_max_speed_pct as we progress through the turn.
        
        First frame of a 90-degree turn uses min speed, but as we continue the turn
        and the remaining angle decreases to near the acceleration_deg range, speed ramps up.
        """
        c = make_commander()
        # First frame: should be at min speed (just starting)
        c.turn(90.0)
        left, right = last_lr(c)
        assert abs(right) == pytest.approx(c._config.turn_min_speed_pct, abs=0.5)
        
        # Simulate progress: now 7 degrees remaining (still accelerating)
        c.turn(7.0)
        left, right = last_lr(c)
        speed_mid = abs(right)
        assert c._config.turn_min_speed_pct < speed_mid < c._config.turn_max_speed_pct
        
        # Near the end: should slow down (symmetric acceleration curve)
        c.turn(5.0)
        left, right = last_lr(c)
        speed_near_end = abs(right)
        assert speed_near_end < speed_mid  # Should be slower as we approach the target

    def test_small_angle_uses_min_speed(self):
        """Very small angle (near 0) should use turn_min_speed_pct."""
        c = make_commander()
        c.turn(1.0)
        left, right = last_lr(c)
        assert abs(right) == pytest.approx(c._config.turn_min_speed_pct, abs=0.5)

    def test_mid_angle_ramps(self):
        """A mid-range total turn should produce intermediate speed as it progresses."""
        c = make_commander()
        cfg = c._config
        # Start a 30-degree turn (in the acceleration range)
        c.turn(30.0)
        _, right = last_lr(c)
        speed_start = abs(right)
        # Continue: 15 degrees remaining (further along the curve)
        c.turn(15.0)
        _, right = last_lr(c)
        speed_mid = abs(right)
        # Speed should increase as we progress
        assert speed_mid > speed_start

    def test_negative_angle_reverses_wheels(self):
        """Negative angle (clockwise) should reverse wheel signs."""
        c = make_commander()
        c.turn(-45.0)
        left, right = last_lr(c)
        # Negative = CW: left=+speed, right=-speed
        assert left > 0 and right < 0

    def test_sets_current_speed_zero(self):
        """Turn should set _current_speed to 0."""
        c = make_commander()
        c._current_speed = 20.0
        c.turn(45.0)
        assert c._current_speed == 0.0

    def test_non_finite_rejected(self):
        """NaN and Inf angles should be rejected."""
        c = make_commander()
        assert c.turn(float("nan")) is False
        assert c.turn(float("inf")) is False

    def test_detects_new_turn(self):
        """Increasing magnitude or sign change should reset total angle."""
        c = make_commander()
        # First turn: 30 degrees
        c.turn(30.0)
        assert c._total_turn_angle == 30.0
        # Continue turn: 25 degrees (same direction, decreasing)
        c.turn(25.0)
        assert c._total_turn_angle == 30.0  # Unchanged
        # New turn: 35 degrees (magnitude increased)
        c.turn(35.0)
        assert c._total_turn_angle == 35.0  # Reset to new value


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_suppresses_within_interval(self):
        t = [0.0]
        c = make_commander()
        c.time_fn = lambda: t[0]
        c.min_send_interval_s = 0.05

        c._send_wheel_speeds(10.0, 10.0)
        count_after_first = len(c.sent_commands)

        t[0] = 0.01  # Still within interval
        c._send_wheel_speeds(10.0, 10.0)
        assert len(c.sent_commands) == count_after_first  # suppressed

    def test_sends_after_interval(self):
        t = [0.0]
        c = make_commander()
        c.time_fn = lambda: t[0]
        c.min_send_interval_s = 0.05

        c._send_wheel_speeds(10.0, 10.0)
        count_after_first = len(c.sent_commands)

        t[0] = 0.1  # Past interval
        c._send_wheel_speeds(10.0, 10.0)
        assert len(c.sent_commands) == count_after_first + 1

    def test_force_bypasses_rate_limit(self):
        t = [0.0]
        c = make_commander()
        c.time_fn = lambda: t[0]
        c.min_send_interval_s = 0.05

        c._send_wheel_speeds(10.0, 10.0)
        count_after_first = len(c.sent_commands)

        t[0] = 0.01
        c._send_wheel_speeds(10.0, 10.0, force=True)
        assert len(c.sent_commands) == count_after_first + 1


# ---------------------------------------------------------------------------
# Deadband
# ---------------------------------------------------------------------------

class TestDeadband:
    def test_small_change_suppressed(self):
        t = [0.0]
        c = make_commander()
        c.time_fn = lambda: t[0]
        c.command_deadband_pct = 2.0
        c.min_send_interval_s = 0.01

        c._send_wheel_speeds(10.0, 10.0)
        count = len(c.sent_commands)
        t[0] = 0.005  # Within interval AND deadband
        c._send_wheel_speeds(10.5, 10.5)
        assert len(c.sent_commands) == count

    def test_large_change_not_suppressed(self):
        t = [0.0]
        c = make_commander()
        c.time_fn = lambda: t[0]
        c.command_deadband_pct = 2.0
        c.min_send_interval_s = 0.01

        c._send_wheel_speeds(10.0, 10.0)
        count = len(c.sent_commands)
        t[0] = 0.005  # Within interval but change exceeds deadband
        c._send_wheel_speeds(15.0, 15.0)
        assert len(c.sent_commands) == count + 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_nan_rejected(self):
        c = make_commander()
        result = c._send_wheel_speeds(float("nan"), 10.0)
        assert result is False
        assert "non-finite" in c.last_error

    def test_inf_rejected(self):
        c = make_commander()
        result = c._send_wheel_speeds(10.0, float("inf"))
        assert result is False


# ---------------------------------------------------------------------------
# Speed clipping
# ---------------------------------------------------------------------------

class TestSpeedClipping:
    def test_output_clipped_to_max(self):
        c = make_commander()
        c.max_speed_pct = 50.0
        c._send_wheel_speeds(100.0, -100.0)
        left, right = last_lr(c)
        assert left == pytest.approx(50.0, abs=0.1)
        assert right == pytest.approx(-50.0, abs=0.1)

    def test_turn_clipped(self):
        c = make_commander(turn_max_speed_pct=30.0)
        c.max_speed_pct = 20.0
        c.turn(90.0)
        left, right = last_lr(c)
        assert abs(left) <= 20.0 + 0.1
        assert abs(right) <= 20.0 + 0.1


# ---------------------------------------------------------------------------
# Steer (legacy bridge)
# ---------------------------------------------------------------------------

class TestSteer:
    def test_differential(self):
        c = make_commander()
        c.steer(20.0, 5.0)
        left, right = last_lr(c)
        assert left == pytest.approx(15.0, abs=0.5)
        assert right == pytest.approx(25.0, abs=0.5)
