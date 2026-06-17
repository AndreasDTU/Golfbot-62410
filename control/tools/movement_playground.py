#!/usr/bin/env python3
"""Manual movement playground for the GolfBot drive layer.

Stage 1 failure-isolation tool: exercises RobotCommander.turn/drive/drive_adjusted/stop
in isolation before Guidance is wired up. Provides both a terminal REPL and an
OpenCV GUI, with dummy-mode support for no-network testing.

Architecture mirrors collector_playground.py.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control.commander import RobotCommander
from control.telemetry import log_event
from config import ConnectionConfig, DriveConfig

_DEFAULT_CONNECTION = ConnectionConfig()

GUI_WINDOW_NAME = "GolfBot Movement Playground"


# ---------------------------------------------------------------------------
# Dummy commander (no-network mock)
# ---------------------------------------------------------------------------

class DummyCommander(RobotCommander):
    """RobotCommander with TCP stubbed out for safe UI testing."""

    def __init__(self) -> None:
        super().__init__(auto_connect=False)
        self.sent_commands: list[str] = []

    def _send(self, cmd: str) -> str:
        self.sent_commands.append(cmd)
        return "ok"


# ---------------------------------------------------------------------------
# Command engine (shared by REPL and GUI)
# ---------------------------------------------------------------------------

@dataclass
class MovementPlayground:
    commander: RobotCommander
    start_time: float = field(default_factory=time.perf_counter)

    def execute(self, raw: str) -> tuple[bool, str]:
        """Dispatch a text command. Returns (keep_running, response_message)."""
        parts = raw.strip().split()
        if not parts:
            return True, ""

        command = parts[0].lower()
        if command in ("quit", "exit"):
            self._try_stop()
            return False, "Exiting after stop."
        if command == "help":
            return True, command_help()
        if command == "status":
            return True, self._status_text()
        if command == "stop":
            return True, self._run_stop()
        if command == "turn":
            return True, self._run_turn(parts)
        if command == "drive":
            return True, self._run_drive(parts)
        if command == "drive_adjusted":
            return True, self._run_drive_adjusted(parts)

        return True, f"Unknown command '{command}'. Type 'help' for commands."

    # -- Movement wrappers with boundary logging --

    def _run_turn(self, parts: list[str]) -> str:
        param = self._parse_float_param(parts, "turn")
        if isinstance(param, str):
            return param
        ok = self.commander.turn(param)
        return self._log_boundary("turn", param, ok)

    def _run_drive(self, parts: list[str]) -> str:
        param = self._parse_float_param(parts, "drive")
        if isinstance(param, str):
            return param
        ok = self.commander.drive(param)
        return self._log_boundary("drive", param, ok)

    def _run_drive_adjusted(self, parts: list[str]) -> str:
        result = self._parse_two_float_params(parts, "drive_adjusted")
        if isinstance(result, str):
            return result
        cm, heading_error = result
        ok = self.commander.drive_adjusted(cm, None, heading_error)
        return self._log_boundary("drive_adjusted", f"{cm:g}, {heading_error:g}", ok)

    def _run_stop(self) -> str:
        ok = self.commander.stop()
        return self._log_boundary("stop", None, ok)

    def _log_boundary(self, name: str, param: float | str | None, ok: bool) -> str:
        elapsed = time.perf_counter() - self.start_time
        lr = self.commander.last_sent
        left = lr[0] if lr else 0.0
        right = lr[1] if lr else 0.0
        status = "ok" if ok else "fail"

        if param is not None:
            label = f"{name}({param})"
        else:
            label = f"{name}()"

        msg = f"[{elapsed:.3f}] CMD {label} -> {status} L={left:.1f} R={right:.1f}"
        log_event("CMD", label, status=status, L=f"{left:.1f}", R=f"{right:.1f}")
        return msg

    # -- Helpers --

    def _parse_float_param(self, parts: list[str], name: str) -> float | str:
        if len(parts) != 2:
            return f"Usage: {name} <value>"
        try:
            value = float(parts[1])
        except ValueError:
            return "Parameter must be numeric."
        if not math.isfinite(value):
            return "Parameter must be finite."
        return value

    def _parse_two_float_params(self, parts: list[str], name: str) -> tuple[float, float] | str:
        if len(parts) != 3:
            return f"Usage: {name} <cm> <heading_error_deg>"
        try:
            a, b = float(parts[1]), float(parts[2])
        except ValueError:
            return "Both parameters must be numeric."
        if not (math.isfinite(a) and math.isfinite(b)):
            return "Both parameters must be finite."
        return a, b

    def _status_text(self) -> str:
        c = self.commander
        lr = c.last_sent
        lr_str = f"L={lr[0]:.1f} R={lr[1]:.1f}" if lr else "None"
        return (
            f"base_speed: {c._base_speed:.1f}\n"
            f"last_sent:  {lr_str}\n"
            f"last_error: {c.last_error or 'None'}"
        )

    def _try_stop(self) -> None:
        try:
            self.commander.stop()
        except Exception as exc:
            print(f"Warning: stop failed during shutdown: {exc}")


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

def command_help() -> str:
    return """Commands:
  turn <degrees>                    In-place rotation. Does NOT stop automatically.
  drive <cm>                        Straight drive. Does NOT stop automatically.
  drive_adjusted <cm> <heading_deg> Drive with simultaneous arc correction. Does NOT stop automatically.
  stop                              Zero wheel speeds. You must call this after any move.
  status                            Show commander state
  help                              Show this help
  quit / exit                       Stop and exit

Note: all move commands set wheel speeds via a profiled speed curve. The robot
keeps moving until you send 'stop'. There is no camera feedback in this tool."""


# ---------------------------------------------------------------------------
# Terminal REPL
# ---------------------------------------------------------------------------

def interactive_loop(playground: MovementPlayground) -> None:
    print(command_help())
    while True:
        try:
            raw = input("movement> ")
        except EOFError:
            print("")
            playground._try_stop()
            break
        except KeyboardInterrupt:
            print("\nCtrl+C received; stopping before exit.")
            playground._try_stop()
            break

        try:
            keep_running, message = playground.execute(raw)
        except Exception as exc:
            print(f"Command error: {exc}")
            continue
        if message:
            print(message)
        if not keep_running:
            break


# ---------------------------------------------------------------------------
# GUI button spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuiButton:
    label: str
    action: str
    rect: tuple[int, int, int, int]
    requires_connection: bool = True


# ---------------------------------------------------------------------------
# OpenCV GUI
# ---------------------------------------------------------------------------

@dataclass
class MovementPlaygroundGui:
    """OpenCV HighGUI shell around the shared MovementPlayground command handler."""

    args: argparse.Namespace
    commander: RobotCommander | None = None
    playground: MovementPlayground | None = None
    connected: bool = False
    connection_label: str = "Disconnected"
    param_value: float = 30.0
    last_command: str = "None"
    message: str = "Ready. Press Connect or start with --dummy."
    busy: bool = False
    closed: bool = False
    log_lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.log_message("Startup: movement playground ready.")
        if self.args.dummy:
            self.configure_dummy()

    # -- Connection management --

    def connect(self) -> None:
        if self.args.dummy:
            self.configure_dummy()
            return
        if self.busy:
            return
        self.busy = True
        self.message = f"Connecting to {self.args.host}:{self.args.port}..."

        def worker() -> None:
            try:
                commander = RobotCommander(
                    self.args.host, port=self.args.port, timeout=self.args.timeout,
                )
            except (OSError, RuntimeError) as exc:
                self.show_error(f"Could not connect: {exc}")
                self.busy = False
                return
            self.commander = commander
            self.playground = MovementPlayground(commander)
            self.set_connected(True, f"Connected to {self.args.host}:{self.args.port}")
            self.log_message(self.connection_label)
            self.busy = False

        threading.Thread(target=worker, daemon=True).start()

    def configure_dummy(self) -> None:
        commander = DummyCommander()
        self.commander = commander
        self.playground = MovementPlayground(commander)
        self.set_connected(True, "Dummy mode: no EV3 connection")
        self.message = "Dummy mode ready. Commands log only."
        self.log_message("Dummy mode enabled; no robot commands will be sent.")

    def disconnect(self) -> None:
        if self.playground is not None:
            self.playground._try_stop()
        if self.commander is not None:
            self.commander.close()
        self.commander = None
        self.playground = None
        self.set_connected(False, "Disconnected")
        self.log_message("Disconnected after stop attempt.")

    def set_connected(self, connected: bool, label: str) -> None:
        self.connected = connected
        self.connection_label = label

    # -- Logging --

    def log_message(self, message: str) -> None:
        self.log_lines.append(message.rstrip())
        self.log_lines = self.log_lines[-12:]

    def show_error(self, message: str) -> None:
        self.message = message
        self.log_message(f"ERROR: {message}")

    # -- Parameter adjustment --

    def adjust_param(self, delta: float) -> None:
        self.param_value = max(1.0, self.param_value + delta)
        self.message = f"Parameter set to {self.param_value:g}"

    # -- Command execution --

    def run_gui_command(self, command: str) -> None:
        if self.playground is None:
            self.show_error("Connect before sending commands.")
            return
        if self.busy:
            self.show_error("Command already running; wait for it to finish.")
            return

        if command in ("turn", "drive"):
            raw = f"{command} {self.param_value:g}"
        elif command == "drive_adjusted":
            # GUI has one param; treat it as heading correction, use 200 cm so
            # speed profiling always runs at cruise speed (same role the old
            # Adjust button's param played).
            raw = f"drive_adjusted 200 {self.param_value:g}"
        else:
            raw = command
        self.last_command = raw
        self.message = f"Running {raw}..."
        self.log_message(f"> {raw}")
        self.busy = True

        def worker() -> None:
            assert self.playground is not None
            try:
                _keep, message = self.playground.execute(raw)
            except Exception as exc:
                self.show_error(f"Command error: {exc}")
                self.busy = False
                return
            self.busy = False
            if message:
                self.message = message.splitlines()[0]
                self.log_message(message)

        threading.Thread(target=worker, daemon=True).start()

    # -- Button layout --

    def buttons(self) -> list[GuiButton]:
        return [
            GuiButton("Connect", "connect", (710, 58, 112, 36), requires_connection=False),
            GuiButton("Disconnect", "disconnect", (835, 58, 130, 36), requires_connection=False),
            GuiButton("- Param", "param_minus", (710, 180, 88, 36), requires_connection=False),
            GuiButton("+ Param", "param_plus", (880, 180, 88, 36), requires_connection=False),
            GuiButton("Turn", "turn", (710, 250, 135, 42)),
            GuiButton("Drive", "drive", (855, 250, 135, 42)),
            GuiButton("DriveAdj", "drive_adjusted", (710, 310, 135, 42)),
            GuiButton("Stop", "stop", (855, 310, 135, 42)),
        ]

    # -- Mouse / keyboard handling --

    def handle_mouse(self, event: int, x_px: int, y_px: int, _flags: int, _userdata) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        for button in self.buttons():
            bx, by, bw, bh = button.rect
            if not (bx <= x_px <= bx + bw and by <= y_px <= by + bh):
                continue
            if button.requires_connection and not self.connected:
                self.show_error("Connect before sending commands.")
                return
            if button.action == "connect":
                self.connect()
            elif button.action == "disconnect":
                self.disconnect()
            elif button.action == "param_minus":
                self.adjust_param(-5.0)
            elif button.action == "param_plus":
                self.adjust_param(5.0)
            else:
                self.run_gui_command(button.action)
            return

    def handle_key(self, key: int) -> None:
        if key in (27, ord("q")):
            self.close()
        elif key == ord("c"):
            self.connect()
        elif key == ord("d"):
            self.disconnect()
        elif key == ord("s"):
            self.run_gui_command("stop")
        elif key == 0:  # Up arrow
            self.run_gui_command("drive")
        elif key == 1:  # Down arrow — drive backward
            old = self.param_value
            self.param_value = -self.param_value
            self.run_gui_command("drive")
            self.param_value = old
        elif key == 2:  # Left arrow — turn left (positive = CCW)
            self.run_gui_command("turn")
        elif key == 3:  # Right arrow — turn right
            old = self.param_value
            self.param_value = -self.param_value
            self.run_gui_command("turn")
            self.param_value = old
        elif key in (ord("+"), ord("=")):
            self.adjust_param(5.0)
        elif key in (ord("-"), ord("_")):
            self.adjust_param(-5.0)

    # -- Drawing helpers --

    def draw_text(
        self,
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float = 0.55,
        color: tuple[int, int, int] = (30, 30, 30),
        thickness: int = 1,
    ) -> None:
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def draw_button(self, image: np.ndarray, button: GuiButton) -> None:
        x, y, w, h = button.rect
        enabled = (self.connected or not button.requires_connection) and not self.busy
        fill = (245, 245, 245) if enabled else (205, 205, 205)
        border = (70, 70, 70)
        cv2.rectangle(image, (x, y), (x + w, y + h), fill, -1, cv2.LINE_AA)
        cv2.rectangle(image, (x, y), (x + w, y + h), border, 1, cv2.LINE_AA)
        self.draw_text(image, button.label, (x + 12, y + 27), 0.58, (20, 20, 20), 1)

    def draw_wheel_gauges(self, image: np.ndarray) -> None:
        """Left panel: L/R bar gauges showing last-sent wheel speeds."""
        # Panel background
        cv2.rectangle(image, (30, 95), (660, 380), (250, 250, 247), -1, cv2.LINE_AA)
        cv2.rectangle(image, (30, 95), (660, 380), (190, 190, 185), 1, cv2.LINE_AA)
        self.draw_text(image, "Wheel Speeds (last sent)", (50, 125), 0.7, (35, 35, 35), 2)

        lr = None
        if self.commander is not None:
            lr = self.commander.last_sent

        left_pct = lr[0] if lr else 0.0
        right_pct = lr[1] if lr else 0.0

        # Gauge geometry
        gauge_x_left = 100
        gauge_x_right = 400
        gauge_w = 180
        gauge_center_y = 240
        gauge_max_h = 100
        max_pct = 100.0

        for label, pct, gx in [("L", left_pct, gauge_x_left), ("R", right_pct, gauge_x_right)]:
            # Label and numeric value
            self.draw_text(image, f"{label}: {pct:.1f}%", (gx, 155), 0.6, (35, 35, 35), 2)

            # Zero line
            cv2.line(image, (gx, gauge_center_y), (gx + gauge_w, gauge_center_y), (150, 150, 150), 1, cv2.LINE_AA)

            # Bar
            bar_h = int(abs(pct) / max_pct * gauge_max_h)
            bar_h = min(bar_h, gauge_max_h)
            if pct >= 0:
                color = (61, 122, 36)  # green for forward
                y_top = gauge_center_y - bar_h
                y_bot = gauge_center_y
            else:
                color = (30, 38, 179)  # red for backward
                y_top = gauge_center_y
                y_bot = gauge_center_y + bar_h
            if bar_h > 0:
                cv2.rectangle(image, (gx + 10, y_top), (gx + gauge_w - 10, y_bot), color, -1, cv2.LINE_AA)

            # Outline
            cv2.rectangle(
                image,
                (gx + 10, gauge_center_y - gauge_max_h),
                (gx + gauge_w - 10, gauge_center_y + gauge_max_h),
                (120, 120, 120), 1, cv2.LINE_AA,
            )

        # Commander state below gauges
        if self.commander is not None:
            self.draw_text(image, f"base_speed: {self.commander._base_speed:.1f}", (50, 365), 0.5)
            err = self.commander.last_error or "None"
            self.draw_text(image, f"last_error: {err[:50]}", (300, 365), 0.5)

    def draw_log(self, image: np.ndarray) -> None:
        """Bottom: scrolling command log."""
        cv2.rectangle(image, (30, 400), (660, 640), (250, 250, 247), -1, cv2.LINE_AA)
        cv2.rectangle(image, (30, 400), (660, 640), (190, 190, 185), 1, cv2.LINE_AA)
        self.draw_text(image, "Command Log", (50, 428), 0.65, (35, 35, 35), 2)
        y_px = 455
        for line in self.log_lines[-8:]:
            self.draw_text(image, line[:85], (50, y_px), 0.45)
            y_px += 22

    # -- Main render --

    def render(self) -> np.ndarray:
        image = np.full((680, 1000, 3), (236, 238, 239), dtype=np.uint8)

        # Title
        self.draw_text(image, "GolfBot Movement Playground", (30, 36), 0.9, (20, 20, 20), 2)
        self.draw_text(
            image,
            "Stage 1: manual driver for Control layer isolation testing.",
            (30, 68), 0.58, (0, 80, 180), 2,
        )

        # Left panel: wheel gauges
        self.draw_wheel_gauges(image)

        # Right panel: controls
        cv2.rectangle(image, (690, 25), (985, 380), (248, 248, 248), -1, cv2.LINE_AA)
        cv2.rectangle(image, (690, 25), (985, 380), (190, 190, 190), 1, cv2.LINE_AA)

        self.draw_text(image, "Connection", (710, 48), 0.65, (35, 35, 35), 2)
        self.draw_text(
            image,
            f"Target: {self.args.host}:{self.args.port}  timeout {self.args.timeout:g}s",
            (710, 100), 0.48,
        )
        conn_color = (60, 90, 60) if self.connected else (80, 60, 60)
        self.draw_text(image, self.connection_label, (710, 118), 0.5, conn_color, 1)

        # Parameter display
        self.draw_text(image, "Movement Commands", (710, 150), 0.65, (35, 35, 35), 2)
        self.draw_text(image, f"Speed input: {self.param_value:g}", (710, 215), 0.56)
        self.draw_text(image, "(DriveAdj uses param as heading error, 200 cm distance.)", (710, 235), 0.42, (100, 100, 100))

        # State readout
        if self.commander is not None:
            lr = self.commander.last_sent
            lr_str = f"L={lr[0]:.1f} R={lr[1]:.1f}" if lr else "None"
            self.draw_text(image, f"Last sent: {lr_str}", (710, 365), 0.48)

        # Buttons
        for button in self.buttons():
            self.draw_button(image, button)

        # Bottom: log panel + message
        self.draw_text(image, f"Message: {self.message}", (30, 392), 0.5, (30, 30, 30), 1)
        self.draw_log(image)

        # Key hints
        self.draw_text(
            image,
            "Keys: arrows=turn/drive  s=stop  q=quit  c=connect  d=disconnect  +/-=param",
            (30, 665), 0.48,
        )

        # Right-side log (commander state below buttons in right panel)
        self.draw_text(image, f"Last: {self.last_command}", (710, 395), 0.5)

        return image

    # -- Lifecycle --

    def close(self) -> None:
        if self.playground is not None:
            self.playground._try_stop()
        self.closed = True

    def run(self) -> None:
        cv2.namedWindow(GUI_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(GUI_WINDOW_NAME, 1000, 680)
        cv2.setMouseCallback(GUI_WINDOW_NAME, self.handle_mouse)
        while not self.closed:
            cv2.imshow(GUI_WINDOW_NAME, self.render())
            key = cv2.waitKey(30) & 0xFF
            if key != 255:
                self.handle_key(key)
            if cv2.getWindowProperty(GUI_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self.close()
        cv2.destroyWindow(GUI_WINDOW_NAME)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual movement playground for GolfBot drive layer.")
    parser.add_argument("--host", default=_DEFAULT_CONNECTION.robot_ip, help="EV3 hostname or IP address.")
    parser.add_argument("--port", type=int, default=_DEFAULT_CONNECTION.robot_tcp_port, help="EV3 TCP command port.")
    parser.add_argument("--timeout", type=float, default=15.0, help="TCP connect/read timeout in seconds.")
    parser.add_argument("--cli", "--terminal", action="store_true", help="Run terminal REPL instead of GUI.")
    parser.add_argument("--dummy", action="store_true", help="No-network mock mode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.cli:
        try:
            gui = MovementPlaygroundGui(args)
        except ImportError as exc:
            print(f"Could not start OpenCV GUI: {exc}", file=sys.stderr)
            return 1
        gui.run()
        return 0

    print("Movement playground — Stage 1 manual driver for Control layer.")
    print("Commands set wheel speed, not distance. The robot keeps moving until you send 'stop'.")
    print("No vision feedback — this is open-loop speed control only.")
    if args.dummy:
        print("Dummy mode enabled; no EV3 connection will be opened.")
        commander = DummyCommander()
    else:
        try:
            commander = RobotCommander(args.host, port=args.port, timeout=args.timeout)
        except (OSError, RuntimeError) as exc:
            print(f"Could not connect to EV3 controller at {args.host}:{args.port}: {exc}", file=sys.stderr)
            return 1

    playground = MovementPlayground(commander)
    interactive_loop(playground)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
