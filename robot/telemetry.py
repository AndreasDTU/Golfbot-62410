"""Drive telemetry: per-frame ringbuffer, CSV dump, and structured event logging."""

from __future__ import annotations

import csv
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class TelemetryFrame:
    """One frame of the drive control pipeline state."""

    # Timing
    timestamp_s: float | None = None
    frame_dt_s: float | None = None

    # Robot pose
    robot_x_cm: float | None = None
    robot_y_cm: float | None = None
    robot_heading_rad: float | None = None
    tube_x_cm: float | None = None
    tube_y_cm: float | None = None

    # Tracking error
    xte_cm: float | None = None
    signed_xte_cm: float | None = None
    heading_error_rad: float | None = None
    segment_index: int | None = None
    segment_heading_rad: float | None = None

    # Speed profile
    distance_to_goal_cm: float | None = None
    forward_scale: float | None = None
    desired_base_speed: float | None = None
    base_speed: float | None = None
    edge_gain: float | None = None

    # PD terms
    heading_derivative: float | None = None
    cross_track_derivative: float | None = None
    turn_speed: float | None = None
    heading_p_term: float | None = None
    heading_d_term: float | None = None
    xte_p_term: float | None = None
    xte_d_term: float | None = None

    # Output
    left_pct: float | None = None
    right_pct: float | None = None

    # State
    drive_state: str | None = None
    drive_message: str | None = None
    pivot_state: str | None = None
    pickup_state: str | None = None
    route_progress_index: int | None = None


class DriveTelemetryRecorder:
    """Ring buffer of TelemetryFrame snapshots with CSV dump capability."""

    def __init__(
        self,
        maxlen: int = 300,
        dump_dir: str = "telemetry",
        auto_dump_enabled: bool = True,
    ) -> None:
        self._buffer: deque[TelemetryFrame] = deque(maxlen=maxlen)
        self._dump_dir = dump_dir
        self._auto_dump_enabled = auto_dump_enabled

    def record_frame(
        self,
        robot_pose: Any,
        drive_runtime: Any,
        wheel_controller: Any,
        frame_dt_s: float,
        pickup_state: str = "",
        pivot_state: str = "",
    ) -> None:
        """Construct a TelemetryFrame from live objects and append to the buffer."""
        # Robot pose
        robot_x = robot_y = robot_heading = tube_x = tube_y = None
        if robot_pose is not None:
            robot_x = getattr(robot_pose, "x_cm", None)
            robot_y = getattr(robot_pose, "y_cm", None)
            robot_heading = getattr(robot_pose, "heading_rad", None)
            tube_x = getattr(robot_pose, "tube_x_cm", None)
            tube_y = getattr(robot_pose, "tube_y_cm", None)

        # Tracking error
        xte = signed_xte = heading_error = seg_idx = seg_heading = None
        last_error = getattr(drive_runtime, "last_error", None)
        if last_error is not None:
            xte = getattr(last_error, "xte_cm", None)
            signed_xte = getattr(last_error, "signed_xte_cm", None)
            heading_error = getattr(last_error, "heading_error_rad", None)
            seg_idx = getattr(last_error, "segment_index", None)
            seg_heading = getattr(last_error, "segment_heading_rad", None)

        # Drive state
        dr_state = None
        dr_message = None
        route_progress = None
        if drive_runtime is not None:
            state_obj = getattr(drive_runtime, "state", None)
            dr_state = state_obj.value if state_obj is not None else None
            dr_message = getattr(drive_runtime, "last_message", None)
            route_progress = getattr(drive_runtime, "route_progress_segment_index", None)

        # Wheel command output
        left = right = None
        last_cmd = getattr(drive_runtime, "last_command", None)
        if last_cmd is not None:
            left = getattr(last_cmd, "left_pct", None)
            right = getattr(last_cmd, "right_pct", None)

        # PD internals from wheel controller
        wc = wheel_controller
        frame = TelemetryFrame(
            timestamp_s=time.perf_counter(),
            frame_dt_s=frame_dt_s,
            robot_x_cm=robot_x,
            robot_y_cm=robot_y,
            robot_heading_rad=robot_heading,
            tube_x_cm=tube_x,
            tube_y_cm=tube_y,
            xte_cm=xte,
            signed_xte_cm=signed_xte,
            heading_error_rad=heading_error,
            segment_index=seg_idx,
            segment_heading_rad=seg_heading,
            distance_to_goal_cm=getattr(wc, "last_distance_to_goal_cm", None),
            forward_scale=getattr(wc, "last_forward_scale", None),
            desired_base_speed=getattr(wc, "last_desired_base_speed", None),
            base_speed=getattr(wc, "last_base_speed", None),
            edge_gain=getattr(wc, "last_edge_gain", None),
            heading_derivative=getattr(wc, "last_heading_derivative", None),
            cross_track_derivative=getattr(wc, "last_cross_track_derivative", None),
            turn_speed=getattr(wc, "last_turn_speed", None),
            heading_p_term=getattr(wc, "last_heading_p_term", None),
            heading_d_term=getattr(wc, "last_heading_d_term", None),
            xte_p_term=getattr(wc, "last_xte_p_term", None),
            xte_d_term=getattr(wc, "last_xte_d_term", None),
            left_pct=left,
            right_pct=right,
            drive_state=dr_state,
            drive_message=dr_message,
            pivot_state=pivot_state,
            pickup_state=pickup_state,
            route_progress_index=route_progress,
        )
        self._buffer.append(frame)

    def dump_csv(self, reason: str = "manual") -> str | None:
        """Write the buffer to a timestamped CSV file. Returns the path or None."""
        if not self._buffer:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{reason}_{timestamp}.csv"
        path = os.path.join(self._dump_dir, filename)
        try:
            os.makedirs(self._dump_dir, exist_ok=True)
            fieldnames = [f.name for f in fields(TelemetryFrame)]
            with open(path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for frame in self._buffer:
                    writer.writerow(asdict(frame))
            return path
        except IOError as exc:
            print(f"[TELEMETRY] CSV dump failed: {exc}")
            return None

    def clear(self) -> None:
        """Empty the ring buffer."""
        self._buffer.clear()


def log_event(category: str, message: str, **kwargs: Any) -> None:
    """Print a timestamped structured event line."""
    ts = time.perf_counter()
    parts = [f"[{ts:.3f}]", f"[{category}]", message]
    if kwargs:
        parts.append("|")
        parts.extend(f"{k}={v}" for k, v in kwargs.items())
    print(" ".join(parts))
