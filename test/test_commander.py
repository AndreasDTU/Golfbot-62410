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
# Turn profiling
# ---------------------------------------------------------------------------

class TestTurn:
    def test_large_angle_uses_cruise_speed(self):
        c = make_commander()
        c.turn(90.0)
        left, right = last_lr(c)
        # Positive degrees = CCW: left=-speed, right=+speed
        assert left < 0 and right > 0
        assert abs(right) == pytest.approx(c._config.turn_speed_pct, abs=0.5)

    def test_small_angle_uses_creep_speed(self):
        c = make_commander()
        c.turn(3.0)
        left, right = last_lr(c)
        assert abs(right) == pytest.approx(c._config.turn_creep_speed_pct, abs=0.5)

    def test_mid_angle_ramps(self):
        c = make_commander()
        cfg = c._config
        mid_angle = (cfg.turn_creep_angle_deg + cfg.turn_cruise_angle_deg) / 2
        c.turn(mid_angle)
        _, right = last_lr(c)
        speed = abs(right)
        assert cfg.turn_creep_speed_pct < speed < cfg.turn_speed_pct

    def test_negative_angle_reverses_wheels(self):
        c = make_commander()
        c.turn(-45.0)
        left, right = last_lr(c)
        # Negative = CW: left=+speed, right=-speed
        assert left > 0 and right < 0

    def test_sets_base_speed_zero(self):
        c = make_commander()
        c._base_speed = 20.0
        c.turn(45.0)
        assert c._base_speed == 0.0

    def test_non_finite_rejected(self):
        c = make_commander()
        assert c.turn(float("nan")) is False
        assert c.turn(float("inf")) is False


# ---------------------------------------------------------------------------
# Drive profiling
# ---------------------------------------------------------------------------

class TestDrive:
    def test_far_distance_uses_cruise_speed(self):
        c = make_commander()
        c.drive(100.0)
        left, right = last_lr(c)
        assert left == pytest.approx(c._config.drive_speed_pct, abs=0.5)
        assert right == pytest.approx(c._config.drive_speed_pct, abs=0.5)

    def test_close_distance_uses_creep_speed(self):
        c = make_commander()
        c.drive(3.0)
        left, right = last_lr(c)
        assert left == pytest.approx(c._config.creep_speed_pct, abs=0.5)

    def test_mid_distance_ramps(self):
        c = make_commander()
        cfg = c._config
        mid = (cfg.creep_distance_cm + cfg.cruise_distance_cm) / 2
        c.drive(mid)
        left, _ = last_lr(c)
        assert cfg.creep_speed_pct < left < cfg.drive_speed_pct

    def test_negative_drives_backward(self):
        c = make_commander()
        c.drive(-50.0)
        left, right = last_lr(c)
        assert left < 0 and right < 0

    def test_stores_base_speed(self):
        c = make_commander()
        c.drive(100.0)
        assert c._base_speed > 0

    def test_slew_limiting(self):
        c = make_commander()
        # First call at dt=None: instant
        c.drive(100.0, dt_s=None)
        speed1 = c._base_speed
        # Reset previous speed low, then request high with small dt
        c._previous_speed = 0.0
        c.drive(100.0, dt_s=0.01)
        speed2 = c._base_speed
        # Should be limited by accel
        assert speed2 < speed1

    def test_non_finite_rejected(self):
        c = make_commander()
        assert c.drive(float("nan")) is False


# ---------------------------------------------------------------------------
# Adjust
# ---------------------------------------------------------------------------

class TestAdjust:
    def test_heading_error_produces_differential(self):
        c = make_commander()
        c._base_speed = 20.0
        c.adjust(5.0)
        left, right = last_lr(c)
        correction = 5.0 * c._config.adjust_gain
        assert left == pytest.approx(20.0 - correction, abs=0.5)
        assert right == pytest.approx(20.0 + correction, abs=0.5)

    def test_reuses_last_drive_speed(self):
        c = make_commander()
        c.drive(100.0)
        saved_base = c._base_speed
        assert saved_base > 0
        c.adjust(2.0)
        left, right = last_lr(c)
        # Average of left and right should be close to saved_base
        avg = (left + right) / 2
        assert avg == pytest.approx(saved_base, abs=0.5)

    def test_cold_start_zero_base(self):
        c = make_commander()
        assert c._base_speed == 0.0
        c.adjust(10.0)
        left, right = last_lr(c)
        # base is 0, so left and right are mirror images
        assert left == pytest.approx(-right, abs=0.1)

    def test_non_finite_rejected(self):
        c = make_commander()
        assert c.adjust(float("inf")) is False


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------

class TestStop:
    def test_sends_zero(self):
        c = make_commander()
        c._base_speed = 30.0
        c.stop()
        left, right = last_lr(c)
        assert left == 0.0 and right == 0.0

    def test_resets_base_speed(self):
        c = make_commander()
        c._base_speed = 25.0
        c._previous_speed = 25.0
        c.stop()
        assert c._base_speed == 0.0
        assert c._previous_speed == 0.0


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
        c = make_commander(max_speed_pct=50.0)
        c._send_wheel_speeds(100.0, -100.0)
        left, right = last_lr(c)
        assert left == pytest.approx(50.0, abs=0.1)
        assert right == pytest.approx(-50.0, abs=0.1)

    def test_turn_clipped(self):
        c = make_commander(max_speed_pct=20.0, turn_speed_pct=30.0)
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
