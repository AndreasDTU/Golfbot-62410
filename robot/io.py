"""Robot hardware I/O adapters."""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

from robot.controller import RobotController
from robot.telemetry import log_event
from vision.config import DriveConfig


class TcpWheelDispatcher:
    """TCP wheel-speed dispatcher for the EV3 command server."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        command_format: str | None = None,
        min_send_interval_s: float | None = None,
        max_speed_pct: float | None = None,
        command_deadband_pct: float | None = None,
        time_fn: Callable[[], float] | None = None,
        controller: RobotController | None = None,
        drive_config: DriveConfig | None = None,
    ) -> None:
        config = drive_config or DriveConfig()
        self.host = host or config.robot_ip
        self.port = port if port is not None else config.robot_tcp_port
        self.command_format = command_format or config.robot_command_format
        self.min_send_interval_s = (
            min_send_interval_s if min_send_interval_s is not None else config.min_send_interval_s
        )
        self.max_speed_pct = max_speed_pct if max_speed_pct is not None else config.max_speed_pct
        self.command_deadband_pct = (
            command_deadband_pct if command_deadband_pct is not None else config.command_deadband_pct
        )
        self.last_send_time = 0.0
        self.last_sent: tuple[float, float] | None = None
        self.last_error = ""
        self.time_fn = time_fn or time.perf_counter
        self.controller: RobotController | None = controller

    def _controller(self) -> RobotController:
        if self.controller is None:
            self.controller = RobotController(self.host, self.port)
        return self.controller

    def close(self) -> None:
        """Close the TCP command connection."""
        if self.controller is not None:
            self.controller.close()
            self.controller = None

    def send_wheel_speeds(self, left_pct: float, right_pct: float, force: bool = False) -> bool:
        """Validate and dispatch one left/right wheel command over TCP."""
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
        try:
            response = self._controller()._send(command)
        except RuntimeError as exc:
            self.last_error = str(exc)
            log_event("DISPATCH", "tcp exception", error=str(exc))
            self.close()
            return False
        if response.lower().startswith("error"):
            self.last_error = response
            log_event("DISPATCH", "error response", response=response)
            return False

        self.last_send_time = now
        self.last_sent = (left, right)
        self.last_error = ""
        return True
