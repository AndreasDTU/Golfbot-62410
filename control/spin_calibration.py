"""In-place spin actuation for robot origin calibration.

This lives in the control layer because it commands the physical robot: it drives
an in-place rotation and decides when a full circle has been swept.  The vision
and localization layers feed it measured marker yaws each tick; it never touches
OpenCV or the camera itself.

Typical use (driven once per frame by the GUI calibration mode):

    spin = SpinController(commander)
    spin.start()
    ...
    status = spin.tick(marker_yaws)   # marker_yaws: {marker_id: yaw_rad}
    if status is SpinStatus.COMPLETE:
        ...  # circle covered, motors already stopped

The controller rotates a configurable amount past 360 degrees so the collected
marker path closes a full circle even with measurement noise or slight slip.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class SpinStatus(Enum):
    """Lifecycle of one spin-calibration rotation."""

    IDLE = "idle"
    SPINNING = "spinning"
    COMPLETE = "complete"
    TIMED_OUT = "timed_out"


def _normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class SpinController:
    """Drive an in-place rotation until the robot has swept a full circle.

    Rotation progress is measured from the ArUco marker yaws the caller passes in
    each tick, integrated per marker so that losing (or regaining) a marker
    mid-spin never injects a bogus jump: only markers seen in two consecutive
    ticks contribute, and their per-tick deltas are averaged.
    """

    commander: object
    turn_speed_pct: float = 20.0
    target_sweep_deg: float = 400.0
    max_spin_s: float = 25.0
    time_fn: Callable[[], float] = time.perf_counter

    status: SpinStatus = field(default=SpinStatus.IDLE, init=False)
    _active: bool = field(default=False, init=False)
    _accumulated_rad: float = field(default=0.0, init=False)
    _last_yaws: dict[int, float] = field(default_factory=dict, init=False)
    _start_time: float = field(default=0.0, init=False)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def swept_deg(self) -> float:
        """Absolute rotation swept so far, in degrees."""
        return abs(math.degrees(self._accumulated_rad))

    @property
    def progress(self) -> float:
        """Fraction of the target sweep covered, clamped to [0, 1]."""
        if self.target_sweep_deg <= 0.0:
            return 1.0
        return max(0.0, min(1.0, self.swept_deg / self.target_sweep_deg))

    def start(self) -> None:
        """Begin a new rotation, resetting all progress accumulators."""
        self._active = True
        self._accumulated_rad = 0.0
        self._last_yaws.clear()
        self._start_time = self.time_fn()
        self.status = SpinStatus.SPINNING

    def stop(self) -> SpinStatus:
        """Halt the motors and leave the controller inactive."""
        if self.commander is not None:
            self.commander.stop(force=True)
        self._active = False
        return self.status

    def tick(self, marker_yaws: dict[int, float]) -> SpinStatus:
        """Advance one frame: integrate progress, keep spinning, or finish.

        ``marker_yaws`` maps robot marker id to its measured yaw in radians for
        this frame; pass an empty dict when no robot marker is currently visible
        (the rotation continues, relying on the time cap as a backstop).
        """
        if not self._active:
            return self.status

        if self.time_fn() - self._start_time >= self.max_spin_s:
            self.status = SpinStatus.TIMED_OUT
            self.stop()
            return self.status

        self._integrate(marker_yaws)

        if abs(self._accumulated_rad) >= math.radians(self.target_sweep_deg):
            self.status = SpinStatus.COMPLETE
            self.stop()
            return self.status

        if self.commander is not None:
            # Positive turn speed = in-place CCW rotation (left = -speed, right = +speed).
            self.commander.steer(0.0, self.turn_speed_pct)
        self.status = SpinStatus.SPINNING
        return self.status

    def _integrate(self, marker_yaws: dict[int, float]) -> None:
        """Accumulate net rotation from per-marker yaw deltas seen across ticks.

        Only markers present in two consecutive ticks contribute, and the
        remembered yaws are pruned to the currently visible set each tick.  A
        marker that drops out and reappears is treated as freshly seeded rather
        than differenced across the gap, which avoids miscounting a yaw wrap that
        may have occurred while it was occluded.
        """
        deltas: list[float] = []
        for marker_id, yaw in marker_yaws.items():
            previous = self._last_yaws.get(marker_id)
            if previous is not None:
                deltas.append(_normalize_angle(yaw - previous))
        if deltas:
            self._accumulated_rad += sum(deltas) / len(deltas)
        self._last_yaws = dict(marker_yaws)
