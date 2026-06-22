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
# Turn profiling (flat speed proportional to total angle)
# ---------------------------------------------------------------------------

def _expected_turn_speed(cfg, total_deg):
    raw = abs(total_deg) / cfg.turn_reference_angle_deg * cfg.turn_max_speed_pct
    return max(cfg.turn_min_speed_pct, min(cfg.turn_max_speed_pct, raw))


class TestTurn:
    def test_reference_angle_uses_max_speed(self):
        """A turn at the reference angle should command turn_max_speed_pct."""
        c = make_commander()
        cfg = c._config
        c.turn(cfg.turn_reference_angle_deg)
        _, right = last_lr(c)
        assert abs(right) == pytest.approx(cfg.turn_max_speed_pct, abs=0.5)

    def test_speed_scales_with_total_angle(self):
        """A larger total turn should command a higher flat speed than a smaller one."""
        cfg = make_commander()._config
        big = make_commander()
        big.turn(180.0)
        _, right_big = last_lr(big)
        small = make_commander()
        small.turn(45.0)
        _, right_small = last_lr(small)
        assert abs(right_big) > abs(right_small)
        assert abs(right_big) == pytest.approx(_expected_turn_speed(cfg, 180.0), abs=0.5)
        assert abs(right_small) == pytest.approx(_expected_turn_speed(cfg, 45.0), abs=0.5)

    def test_speed_constant_during_turn(self):
        """Speed is set from the total angle and stays flat as remaining shrinks."""
        c = make_commander()
        c.turn(90.0)
        _, right_start = last_lr(c)
        c.turn(45.0)  # same turn continuing (total stays 90)
        _, right_mid = last_lr(c)
        c.turn(20.0)
        _, right_end = last_lr(c)
        assert abs(right_start) == pytest.approx(abs(right_mid), abs=0.5)
        assert abs(right_mid) == pytest.approx(abs(right_end), abs=0.5)

    def test_small_angle_uses_min_speed(self):
        """Very small angle (near 0) should clamp to turn_min_speed_pct."""
        c = make_commander()
        c.turn(1.0)
        left, right = last_lr(c)
        assert abs(right) == pytest.approx(c._config.turn_min_speed_pct, abs=0.5)

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
