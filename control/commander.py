"""Unified robot commander: TCP transport, wheel dispatch, and movement API."""

from __future__ import annotations

import math
import select
import socket
import time
from typing import Callable

import numpy as np

from config import ConnectionConfig, DriveConfig
from control.telemetry import log_event
from control.tools.drive_calibration import (
    DriveCalibrationValues,
    parse_drive_calibration_response,
)


class RobotCommander:
    """Single control surface for all EV3 motor commands.

    Absorbs the former RobotController, TcpWheelDispatcher, and MotorCommander
    into one class with a clean movement API:

        turn(degrees)                         — in-place rotation, speed profiled by remaining angle
        drive(cm)                             — straight forward/backward, speed profiled by distance
        drive_adjusted(cm, dt_s, heading_deg) — forward drive with simultaneous arc correction
        stop()                                — zero wheel speeds, resets base speed

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
        connection_config: ConnectionConfig | None = None,
        drive_config: DriveConfig | None = None,
        time_fn: Callable[[], float] | None = None,
        *,
        auto_connect: bool = True,
    ) -> None:
        conn = connection_config or ConnectionConfig()
        config = drive_config or DriveConfig()
        self.host: str = robot_ip or conn.robot_ip
        self.port: int = port if port is not None else conn.robot_tcp_port
        self.timeout: float = timeout
        self.connect_retries: int = max(0, int(connect_retries))
        self._config: DriveConfig = config

        # Dispatch state
        self.command_format: str = conn.robot_command_format
        self.min_send_interval_s: float = conn.min_send_interval_s
        self.command_deadband_pct: float = conn.command_deadband_pct
        self.last_send_time: float = 0.0
        self.last_sent: tuple[float, float] | None = None
        self.last_error: str = ""
        self.time_fn: Callable[[], float] = time_fn or time.perf_counter

        # Speed state
        self._current_speed: float = 0.0
        self._total_distance: float = 0.0
        self._total_turn_angle: float = 0.0
        self._last_turn_angle: float = 0.0
        self.max_speed_pct: float = 100.0

        self._drive_acceleration: float = (
            config.drive_max_speed_pct - config.drive_min_speed_pct
        ) / (config.drive_acceleration_cm**2)
        self._drive_deacceleration: float = (
            config.drive_max_speed_pct - config.drive_min_speed_pct
        ) / (config.drive_deacceleration_cm**2)

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
            raise RuntimeError(
                f"EV3 command failed after send attempt ({cmd!r}): {exc}"
            ) from exc

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

    def _send_wheel_speeds(
        self, left_pct: float, right_pct: float, force: bool = False
    ) -> bool:
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
        """
        Profiled forward speed for a remaining distance.
        Following this graph (quadratic acceleration): https://www.desmos.com/calculator/kwkobx9vta
        """

        cfg = self._config
        distance_left = max(0.0, abs(float(distance_cm))) - cfg.waypoint_arrival_cm
        distance_driven = self._total_distance - distance_left
        raw_speed = min(
            cfg.drive_min_speed_pct
            + self._drive_acceleration * (distance_driven * distance_driven),
            cfg.drive_min_speed_pct
            + self._drive_deacceleration * (distance_left * distance_left),
        )

        speed = max(min(raw_speed, cfg.drive_max_speed_pct), cfg.drive_min_speed_pct)
        print(f"DRIVING speed={speed} dist_left={distance_left} dist_driven={distance_driven}")
        return speed

    def _target_speed_for_angle(self, total_angle: float) -> float:
        """Flat turn speed proportional to the TOTAL commanded angle.

            speed = clamp(total / turn_reference_angle_deg * turn_max_speed_pct,
                          turn_min_speed_pct, turn_max_speed_pct)

        Computed once per turn from the total angle and held flat for the whole
        turn (no ramp). Larger turns spin faster; the coast compensation scales
        with this speed so the robot lands on target.
        """
        cfg = self._config
        raw = abs(float(total_angle)) / cfg.turn_reference_angle_deg * cfg.turn_max_speed_pct
        return max(cfg.turn_min_speed_pct, min(cfg.turn_max_speed_pct, raw))

    # ------------------------------------------------------------------
    # Movement API (non-blocking, per-frame, all produce LR commands)
    # ------------------------------------------------------------------

    def turn(self, degrees: float) -> bool:
        """In-place rotation.  *degrees* = remaining angle (positive = CCW).

        Speed is proportional to the total commanded angle (set once per new
        turn). Stops early by an effective coast distance that scales linearly
        with speed: turn_coast_deg is the coast at full speed and shrinks as the
        turn speed drops.
        """
        if not math.isfinite(degrees):
            self.last_error = "non-finite turn angle rejected"
            return False

        # Detect start of a new turn: sign changed, or remaining angle jumped up by
        # more than the noise threshold (guards against ArUco jitter resetting the
        # profile on every noisy frame and pinning the robot at minimum speed).
        abs_degrees = abs(degrees)
        if (
            self._last_turn_angle == 0
            or abs_degrees > abs(self._last_turn_angle) + self._config.turn_reset_noise_deg
            or ((degrees >= 0) != (self._last_turn_angle >= 0))
        ):
            self._total_turn_angle = abs_degrees
        self._last_turn_angle = degrees

        # Flat speed proportional to the total turn magnitude.
        speed = self._target_speed_for_angle(self._total_turn_angle)

        # Coast scales linearly with speed (physics: coast ∝ momentum ∝ speed).
        effective_coast = (speed / self._config.turn_max_speed_pct) * self._config.turn_coast_deg

        # Fire the early stop only when the total turn is large enough that
        # coasting matters — small corrections turn normally to the target.
        if (
            effective_coast > 0
            and abs_degrees <= effective_coast
            and self._total_turn_angle > effective_coast
        ):
            self._total_turn_angle = 0
            return self.stop()

        sign = 1.0 if degrees >= 0 else -1.0
        self._current_speed = 0.0
        return self._send_wheel_speeds(-sign * speed, sign * speed)

    def start_drive(self, cm: float, heading_error_deg: float = 0) -> bool:
        """Initialize drive.  *cm* = remaining distance (positive = forward).

        Optional *heading_error_deg* allows applying heading correction from the start.
        """
        if not math.isfinite(cm):
            self.last_error = "non-finite drive distance rejected"
            return False

        self._total_distance = abs(cm)
        return self.drive_adjusted(cm, heading_error_deg)

    def drive_adjusted(self, cm: float, heading_error_deg: float) -> bool:
        """Forward drive with simultaneous arc correction.

        Computes a single wheel-speed command combining profiled forward speed
        (from *cm*) with heading correction (from *heading_error_deg*).
        """
        if not (math.isfinite(cm) and math.isfinite(heading_error_deg)):
            self.last_error = "non-finite drive_adjusted argument rejected"
            return False
        speed = self._target_speed_for_distance(abs(cm))
        if cm < 0:
            speed = -speed
        self._current_speed = speed
        correction = (
            heading_error_deg * self._config.adjust_gain * (speed * speed / 10000.0)
        )
        return self._send_wheel_speeds(speed - correction, speed + correction)

    def stop(self, force: bool = True) -> bool:
        """Zero wheel speeds and reset base speed."""
        self._base_speed = 0.0
        self._driving = False
        self._send("stop")
        return True

    # ------------------------------------------------------------------
    # Legacy steer() — bridge for guidance code until full migration
    # ------------------------------------------------------------------

    def steer(
        self, base_speed_pct: float, turn_speed_pct: float, force: bool = False
    ) -> bool:
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
        self._send("stop")
        return self._send("pickup_assist")

    def unload_full_cycle(self) -> str:
        self._send("stop")
        self._send("unload_full_cycle")
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
        response = self._send(
            f"drivecal set {values.axle_track_mm:.6f} {values.mm_per_unit:.6f}"
        )
        return parse_drive_calibration_response(response)
