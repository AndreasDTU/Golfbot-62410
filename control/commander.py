"""Unified robot commander: TCP transport, wheel dispatch, and movement API."""

from __future__ import annotations

import math
import select
import socket
import time
from typing import Callable

import numpy as np

from control.telemetry import log_event
from control.tools.drive_calibration import DriveCalibrationValues, parse_drive_calibration_response
from config import DriveConfig


class RobotCommander:
    """Single control surface for all EV3 motor commands.

    Absorbs the former RobotController, TcpWheelDispatcher, and MotorCommander
    into one class with a clean three-command movement API:

        turn(degrees)   — in-place rotation, speed profiled by remaining angle
        drive(cm)       — straight forward/backward, speed profiled by distance
        adjust(degrees) — arc correction reusing the last drive speed
        stop()          — zero wheel speeds, resets base speed

    All movement commands are non-blocking and produce a single LR wheel-speed
    message per call.  Actuator and calibration commands remain blocking.

    Sign convention (unchanged from the old stack):
        positive degrees = CCW (left turn)
        positive cm      = forward
        wheel speeds:  left = base - turn,  right = base + turn
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        robot_ip: str | None = None,
        port: int | None = None,
        timeout: float = 15.0,
        connect_retries: int = 1,
        drive_config: DriveConfig | None = None,
        time_fn: Callable[[], float] | None = None,
        *,
        auto_connect: bool = True,
    ) -> None:
        config = drive_config or DriveConfig()
        self.host: str = robot_ip or config.robot_ip
        self.port: int = port if port is not None else config.robot_tcp_port
        self.timeout: float = timeout
        self.connect_retries: int = max(0, int(connect_retries))
        self._config: DriveConfig = config

        # Dispatch state
        self.command_format: str = config.robot_command_format
        self.min_send_interval_s: float = config.min_send_interval_s
        self.max_speed_pct: float = config.max_speed_pct
        self.command_deadband_pct: float = config.command_deadband_pct
        self.last_send_time: float = 0.0
        self.last_sent: tuple[float, float] | None = None
        self.last_error: str = ""
        self.time_fn: Callable[[], float] = time_fn or time.perf_counter

        # Speed state
        self._base_speed: float = 0.0
        self._previous_speed: float = 0.0

        # TCP connection
        self.sock: socket.socket | None = None
        if auto_connect:
            self.sock = self._connect()

    # ------------------------------------------------------------------
    # TCP internals
    # ------------------------------------------------------------------

    def _connect(self) -> socket.socket:
        last_error: OSError | None = None
        for _attempt in range(self.connect_retries + 1):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            try:
                sock.connect((self.host, self.port))
                # Disable Nagle so fire-and-forget wheel commands ship immediately
                # instead of being coalesced into one TCP segment.
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return sock
            except OSError as exc:
                last_error = exc
                sock.close()
        raise RuntimeError(
            f"Could not connect to EV3 controller at {self.host}:{self.port}: {last_error}"
        ) from last_error

    def _drain_replies(self) -> None:
        """Discard any pending reply bytes without blocking.

        Fire-and-forget wheel commands still get an ``ok``/``error`` reply from
        the EV3 server.  We never read those synchronously, so we drain them here
        to stop the socket receive buffer from filling — a full buffer would
        eventually block the server's ``sendall`` and re-introduce the stall.
        """
        sock = self.sock
        if sock is None:
            return
        try:
            while True:
                ready, _, _ = select.select((sock,), (), (), 0)
                if not ready:
                    return
                chunk = sock.recv(4096)
                if not chunk:  # peer closed the connection
                    self.close()
                    return
        except OSError:
            self.close()

    def _send(self, cmd: str) -> str:
        """Send a command and block until the EV3 replies (one reply per command)."""
        if self.sock is None:
            self.sock = self._connect()
        # Clear any un-read fire-and-forget acks so recv() below returns *this*
        # command's reply rather than a stale wheel-command ack.
        self._drain_replies()
        if self.sock is None:
            self.sock = self._connect()
        payload = (cmd + "\n").encode("utf-8")
        try:
            self.sock.sendall(payload)
            return self.sock.recv(1024).decode("utf-8").strip()
        except OSError as exc:
            raise RuntimeError(f"EV3 command failed after send attempt ({cmd!r}): {exc}") from exc

    def _send_nowait(self, cmd: str) -> bool:
        """Send a command without waiting for the reply (fire-and-forget).

        Used for high-frequency wheel commands so the caller's loop never blocks
        on the EV3 round-trip.  Pending replies are drained first to keep the
        socket buffer clear.
        """
        if self.sock is None:
            self.sock = self._connect()
        self._drain_replies()
        if self.sock is None:
            self.sock = self._connect()
        payload = (cmd + "\n").encode("utf-8")
        try:
            self.sock.sendall(payload)
            return True
        except OSError as exc:
            self.last_error = f"EV3 send failed ({cmd!r}): {exc}"
            self.close()
            return False

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    # ------------------------------------------------------------------
    # Dispatch internals (from TcpWheelDispatcher)
    # ------------------------------------------------------------------

    def _send_wheel_speeds(self, left_pct: float, right_pct: float, force: bool = False) -> bool:
        """Validate, clip, rate-limit, and send one LR wheel command."""
        if not (math.isfinite(left_pct) and math.isfinite(right_pct)):
            self.last_error = "non-finite wheel command rejected"
            log_event("DISPATCH", "non-finite rejected", left=left_pct, right=right_pct)
            return False

        left = float(np.clip(left_pct, -self.max_speed_pct, self.max_speed_pct))
        right = float(np.clip(right_pct, -self.max_speed_pct, self.max_speed_pct))
        now = self.time_fn()
        if (
            not force
            and self.last_sent is not None
            and now - self.last_send_time < self.min_send_interval_s
            and abs(left - self.last_sent[0]) < self.command_deadband_pct
            and abs(right - self.last_sent[1]) < self.command_deadband_pct
        ):
            return True

        command = self.command_format.format(left=left, right=right)
        # Fire-and-forget: never block the caller (the vision loop) on the EV3
        # round-trip.  Wheel speeds are already validated/clipped client-side, so
        # we forgo the server's reply; any error reply is drained, not parsed.
        if not self._send_nowait(command):
            log_event("DISPATCH", "tcp exception", error=self.last_error)
            return False

        self.last_send_time = now
        self.last_sent = (left, right)
        self.last_error = ""
        return True

    # ------------------------------------------------------------------
    # Speed profiling (moved from WheelCommandController / guidance)
    # ------------------------------------------------------------------

    def _target_speed_for_distance(self, distance_cm: float) -> float:
        """Profiled forward speed for a remaining distance."""
        cfg = self._config
        distance = max(0.0, abs(float(distance_cm)))
        if distance <= cfg.creep_distance_cm:
            return cfg.creep_speed_pct
        if distance >= cfg.cruise_distance_cm:
            return cfg.drive_speed_pct
        span = max(1e-6, cfg.cruise_distance_cm - cfg.creep_distance_cm)
        ratio = (distance - cfg.creep_distance_cm) / span
        return cfg.creep_speed_pct + ratio * (cfg.drive_speed_pct - cfg.creep_speed_pct)

    def _target_speed_for_angle(self, degrees: float) -> float:
        """Profiled rotation speed for a remaining angle."""
        cfg = self._config
        angle = abs(float(degrees))
        if angle <= cfg.turn_creep_angle_deg:
            return cfg.turn_creep_speed_pct
        if angle >= cfg.turn_cruise_angle_deg:
            return cfg.turn_speed_pct
        span = max(1e-6, cfg.turn_cruise_angle_deg - cfg.turn_creep_angle_deg)
        ratio = (angle - cfg.turn_creep_angle_deg) / span
        return cfg.turn_creep_speed_pct + ratio * (cfg.turn_speed_pct - cfg.turn_creep_speed_pct)

    def _slew_limited_speed(self, desired_speed_pct: float, dt_s: float | None = None) -> float:
        """Apply acceleration / deceleration limits to forward speed."""
        desired = float(np.clip(desired_speed_pct, -self.max_speed_pct, self.max_speed_pct))
        if dt_s is None or dt_s <= 1e-6:
            self._previous_speed = desired
            return desired
        previous = self._previous_speed
        accel_step = max(0.0, self._config.acceleration_limit_pct_per_s) * dt_s
        decel_step = max(0.0, self._config.deceleration_limit_pct_per_s) * dt_s
        limited = float(np.clip(desired, previous - decel_step, previous + accel_step))
        self._previous_speed = limited
        return limited

    # ------------------------------------------------------------------
    # Movement API (non-blocking, per-frame, all produce LR commands)
    # ------------------------------------------------------------------

    def turn(self, degrees: float) -> bool:
        """In-place rotation.  *degrees* = remaining angle (positive = CCW)."""
        if not math.isfinite(degrees):
            self.last_error = "non-finite turn angle rejected"
            return False
        speed = self._target_speed_for_angle(degrees)
        sign = 1.0 if degrees >= 0 else -1.0
        self._base_speed = 0.0
        return self._send_wheel_speeds(-sign * speed, sign * speed)

    def drive(self, cm: float, dt_s: float | None = None) -> bool:
        """Straight-line drive.  *cm* = remaining distance (positive = forward)."""
        if not math.isfinite(cm):
            self.last_error = "non-finite drive distance rejected"
            return False
        desired = self._target_speed_for_distance(cm)
        if cm < 0:
            desired = -desired
        speed = self._slew_limited_speed(desired, dt_s)
        self._base_speed = speed
        return self._send_wheel_speeds(speed, speed)

    def adjust(self, degrees: float) -> bool:
        """Arc correction while maintaining forward motion.

        *degrees* = heading error to correct (positive = needs CCW correction).
        Reuses ``_base_speed`` from the most recent ``drive()`` call so that
        forward speed is preserved.  If no ``drive()`` has occurred yet,
        ``_base_speed`` defaults to 0 (pure turn from standstill).
        """
        if not math.isfinite(degrees):
            self.last_error = "non-finite adjust angle rejected"
            return False
        correction = degrees * self._config.adjust_gain
        left = self._base_speed - correction
        right = self._base_speed + correction
        return self._send_wheel_speeds(left, right)

    def stop(self, force: bool = True) -> bool:
        """Zero wheel speeds and reset base speed."""
        self._base_speed = 0.0
        self._previous_speed = 0.0
        return self._send_wheel_speeds(0.0, 0.0, force=force)

    # ------------------------------------------------------------------
    # Legacy steer() — bridge for guidance code until full migration
    # ------------------------------------------------------------------

    def steer(self, base_speed_pct: float, turn_speed_pct: float, force: bool = False) -> bool:
        """Differential steer compatible with the old MotorCommander API."""
        left = base_speed_pct - turn_speed_pct
        right = base_speed_pct + turn_speed_pct
        return self._send_wheel_speeds(left, right, force=force)

    # ------------------------------------------------------------------
    # Actuator API (blocking, unchanged)
    # ------------------------------------------------------------------

    def collector_travel_position(self) -> str:
        return self._send("collector_travel_position")

    def pickup_assist(self) -> str:
        return self._send("pickup_assist")

    def unload_full_cycle(self) -> str:
        return self._send("unload_full_cycle")

    def pipe_up(self, units, speed=None) -> str:
        cmd = f"pipe up {units}" if speed is None else f"pipe up {units} {speed}"
        return self._send(cmd)

    def pipe_down(self, units, speed=None) -> str:
        cmd = f"pipe down {units}" if speed is None else f"pipe down {units} {speed}"
        return self._send(cmd)

    def pipe_stop(self) -> str:
        return self._send("pipe stop")

    def pickup(self) -> str:
        return self.pickup_assist()

    def dropoff(self) -> str:
        return self.unload_full_cycle()

    # ------------------------------------------------------------------
    # Calibration API (blocking, unchanged)
    # ------------------------------------------------------------------

    def get_drive_calibration(self):
        return parse_drive_calibration_response(self._send("drivecal get"))

    def set_drive_calibration(self, axle_track_mm: float, mm_per_unit: float):
        values = DriveCalibrationValues(float(axle_track_mm), float(mm_per_unit))
        response = self._send(f"drivecal set {values.axle_track_mm:.6f} {values.mm_per_unit:.6f}")
        return parse_drive_calibration_response(response)
