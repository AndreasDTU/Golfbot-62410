#!/usr/bin/env python3
"""Manual open-loop playground for the GolfBot collector pipe.

This tool talks only to the EV3 TCP controller. It does not import or start the
vision pipeline, route planner, detector shell, TCP wheel dispatcher, or any
autonomous drive loop.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control.commander import RobotCommander


DEFAULT_MAX_MANUAL_UNITS = 5.0
GUI_WINDOW_NAME = "GolfBot Collector Playground"


class CollectorBelief(str, Enum):
    UNKNOWN = "UNKNOWN"
    TRAVEL = "TRAVEL"
    PICKUP_ASSIST = "PICKUP_ASSIST"
    UNLOADING = "UNLOADING"
    MANUAL_UP = "MANUAL_UP"
    MANUAL_DOWN = "MANUAL_DOWN"
    STOPPED = "STOPPED"


class CollectorController(Protocol):
    def collector_travel_position(self): ...
    def pickup_assist(self): ...
    def unload_full_cycle(self): ...
    def pipe_up(self, units, speed=None): ...
    def pipe_down(self, units, speed=None): ...
    def pipe_stop(self): ...
    def stop(self): ...


ConfirmFn = Callable[[str], bool]


class DummyCollectorController:
    """Local no-network controller for trying the playground UI safely."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def _record(self, command: str) -> str:
        self.commands.append(command)
        return f"dummy ok: {command}"

    def collector_travel_position(self) -> str:
        return self._record("collector_travel_position")

    def pickup_assist(self) -> str:
        return self._record("pickup_assist")

    def unload_full_cycle(self) -> str:
        return self._record("unload_full_cycle")

    def pipe_up(self, units, speed=None) -> str:
        command = f"pipe up {units}" if speed is None else f"pipe up {units} {speed}"
        return self._record(command)

    def pipe_down(self, units, speed=None) -> str:
        command = f"pipe down {units}" if speed is None else f"pipe down {units} {speed}"
        return self._record(command)

    def pipe_stop(self) -> str:
        return self._record("pipe stop")

    def stop(self) -> str:
        return self._record("stop")


@dataclass
class CollectorPlayground:
    controller: CollectorController
    confirm_unload: bool = True
    max_manual_units: float = DEFAULT_MAX_MANUAL_UNITS
    confirm: ConfirmFn | None = None
    on_state_change: Callable[[CollectorBelief], None] | None = None
    state: CollectorBelief = CollectorBelief.UNKNOWN

    def execute(self, raw: str) -> tuple[bool, str]:
        parts = raw.strip().split()
        if not parts:
            return True, ""

        command = parts[0].lower()
        if command in ("quit", "exit"):
            self._try_pipe_stop()
            return False, "Exiting after pipe stop attempt."
        if command == "help":
            return True, command_help()
        if command == "status":
            return True, self._status_text()
        if command == "travel":
            response = self.controller.collector_travel_position()
            self._set_state(CollectorBelief.TRAVEL)
            return True, self._format_response(response)
        if command in ("assist", "pickup"):
            self._set_state(CollectorBelief.PICKUP_ASSIST)
            response = self.controller.pickup_assist()
            self._set_state(CollectorBelief.TRAVEL)
            return True, self._format_response(response)
        if command in ("unload", "dropoff"):
            if self.confirm_unload and not self._confirm_unload():
                return True, "Unload cancelled."
            self._set_state(CollectorBelief.UNLOADING)
            response = self.controller.unload_full_cycle()
            self._set_state(CollectorBelief.UNKNOWN)
            return True, self._format_response(response)
        if command == "up":
            units = self._parse_manual_units(parts)
            if isinstance(units, str):
                return True, units
            response = self.controller.pipe_up(units)
            self._set_state(CollectorBelief.MANUAL_UP)
            return True, self._format_response(response)
        if command == "down":
            units = self._parse_manual_units(parts)
            if isinstance(units, str):
                return True, units
            response = self.controller.pipe_down(units)
            self._set_state(CollectorBelief.MANUAL_DOWN)
            return True, self._format_response(response)
        if command == "stop":
            response = self._stop_pipe_first()
            self._set_state(CollectorBelief.STOPPED)
            return True, self._format_response(response)

        return True, f"Unknown command '{command}'. Type 'help' for commands."

    def _parse_manual_units(self, parts: list[str]) -> float | str:
        if len(parts) != 2:
            return f"Usage: {parts[0].lower()} <units>"
        try:
            units = float(parts[1])
        except ValueError:
            return "Units must be numeric."
        if not math.isfinite(units):
            return "Units must be finite."
        if units <= 0:
            return "Units must be greater than zero."
        if units > self.max_manual_units:
            return f"Units exceed max per command ({self.max_manual_units:g})."
        return units

    def _confirm_unload(self) -> bool:
        if self.confirm is not None:
            return self.confirm("Run full unload cycle? Type 'yes' to continue: ")
        answer = input("Run full unload cycle? Type 'yes' to continue: ")
        return answer.strip().lower() == "yes"

    def _stop_pipe_first(self):
        try:
            return self.controller.pipe_stop()
        except AttributeError:
            return self.controller.stop()

    def _try_pipe_stop(self) -> None:
        try:
            self._stop_pipe_first()
        except Exception as exc:
            print(f"Warning: pipe stop failed during shutdown: {exc}")

    def _set_state(self, state: CollectorBelief) -> None:
        self.state = state
        if self.on_state_change is not None:
            self.on_state_change(state)

    def _status_text(self) -> str:
        return (
            f"Software belief: {self.state.value}\n"
            "No collector position sensor is present; this state is not verified physical height."
        )

    def _format_response(self, response) -> str:
        return f"{response}\n{self._status_text()}"


def command_help() -> str:
    return """Commands:
  travel          Send collector_travel_position()
  assist          Send pickup_assist()
  pickup          Alias for assist
  unload          Send unload_full_cycle() after confirmation
  dropoff         Alias for unload
  up <units>      Raise pipe manually, bounded by --max-manual-units
  down <units>    Lower pipe manually, bounded by --max-manual-units
  stop            Stop the pipe motor if supported
  status          Print open-loop software belief
  help            Show this help
  quit / exit     Stop pipe if possible and exit"""


def gui_command_text(command: str, units_text: str) -> str:
    """Map a GUI button action to the same text command used by the terminal REPL."""
    if command in ("up", "down"):
        return f"{command} {units_text.strip()}"
    return command


def print_startup_warning() -> None:
    print("No collector position sensor is present. Start with the collector in a known safe position. Software state is open-loop.")
    print("This playground does not drive wheels, follow paths, run vision, plan routes, or detect balls.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual EV3 collector/pipe playground.")
    parser.add_argument("--host", default="ev3dev", help="EV3 hostname or IP address.")
    parser.add_argument("--port", type=int, default=5555, help="EV3 TCP command port.")
    parser.add_argument("--timeout", type=float, default=15.0, help="TCP connect/read timeout in seconds.")
    parser.add_argument("--cli", "--terminal", action="store_true", help="Run the terminal REPL instead of the GUI.")
    parser.add_argument("--dummy", action="store_true", help="Run without connecting to an EV3; commands use a local dummy controller.")
    parser.add_argument("--no-confirm", action="store_true", help="Do not ask before running unload.")
    parser.add_argument("--yes", action="store_true", help="Alias for --no-confirm.")
    parser.add_argument(
        "--max-manual-units",
        type=float,
        default=DEFAULT_MAX_MANUAL_UNITS,
        help="Maximum units allowed for one manual up/down command.",
    )
    return parser


@dataclass(frozen=True)
class GuiButton:
    label: str
    action: str
    rect: tuple[int, int, int, int]
    requires_connection: bool = True


@dataclass
class CollectorPlaygroundGui:
    """OpenCV HighGUI shell around the shared CollectorPlayground command handler."""

    args: argparse.Namespace
    controller: CollectorController | None = None
    playground: CollectorPlayground | None = None
    connected: bool = False
    connection_label: str = "Disconnected"
    state: CollectorBelief = CollectorBelief.UNKNOWN
    manual_units: float = 1.0
    last_command: str = "None"
    message: str = "Ready. Press Connect or start with --dummy."
    pending_unload: bool = False
    busy: bool = False
    closed: bool = False
    log_lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.log_message("Startup: collector state is UNKNOWN. Start from a known safe physical position.")
        if self.args.dummy:
            self.configure_dummy()

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
                controller = RobotCommander(self.args.host, port=self.args.port, timeout=self.args.timeout)
            except OSError as exc:
                self.show_error(f"Could not connect to EV3 controller at {self.args.host}:{self.args.port}: {exc}")
                self.busy = False
                return
            self.controller = controller
            self.playground = CollectorPlayground(
                controller,
                confirm_unload=not (self.args.no_confirm or self.args.yes),
                max_manual_units=self.args.max_manual_units,
                confirm=self.confirm_unload,
                on_state_change=self.update_state,
            )
            self.update_state(self.playground.state)
            self.set_connected(True, f"Connected to {self.args.host}:{self.args.port}")
            self.log_message(self.connection_label)
            self.busy = False

        threading.Thread(target=worker, daemon=True).start()

    def configure_dummy(self) -> None:
        controller = DummyCollectorController()
        self.controller = controller
        self.playground = CollectorPlayground(
            controller,
            confirm_unload=not (self.args.no_confirm or self.args.yes),
            max_manual_units=self.args.max_manual_units,
            confirm=self.confirm_unload,
            on_state_change=self.update_state,
        )
        self.update_state(self.playground.state)
        self.set_connected(True, "Dummy mode: no EV3 connection")
        self.message = "Dummy mode ready. Commands update the GUI and log only."
        self.log_message("Dummy mode enabled; no robot commands will be sent.")

    def disconnect(self) -> None:
        if self.playground is not None:
            self.playground._try_pipe_stop()
        if self.controller is not None and hasattr(self.controller, "sock"):
            try:
                self.controller.sock.close()
            except OSError:
                pass
        self.controller = None
        self.playground = None
        self.pending_unload = False
        self.set_connected(False, "Disconnected")
        self.log_message("Disconnected after pipe stop attempt.")

    def set_connected(self, connected: bool, label: str) -> None:
        self.connected = connected
        self.connection_label = label

    def update_state(self, state: CollectorBelief) -> None:
        self.state = state

    def confirm_unload(self, _prompt: str) -> bool:
        if self.args.no_confirm or self.args.yes:
            return True
        return self.pending_unload

    def log_message(self, message: str) -> None:
        self.log_lines.append(message.rstrip())
        self.log_lines = self.log_lines[-9:]

    def show_error(self, message: str) -> None:
        self.message = message
        self.log_message(f"ERROR: {message}")

    def adjust_units(self, delta: float) -> None:
        self.manual_units = float(np.clip(self.manual_units + delta, 0.1, self.args.max_manual_units))
        self.message = f"Manual units set to {self.manual_units:g}"

    def run_gui_command(self, command: str, bypass_confirm: bool = False) -> None:
        if self.playground is None:
            self.show_error("Connect to the EV3 controller before sending commands.")
            return
        if self.busy:
            self.show_error("Command already running; wait for it to finish.")
            return
        if command == "unload" and self.playground.confirm_unload and not bypass_confirm:
            self.pending_unload = True
            self.message = "Confirm full unload with the red Confirm Unload button."
            self.log_message("Unload requested; waiting for confirmation.")
            return

        raw = gui_command_text(command, f"{self.manual_units:g}")
        self.last_command = raw
        self.message = f"Running {raw}..."
        self.log_message(f"> {raw}")
        self.busy = True

        def worker() -> None:
            assert self.playground is not None
            old_confirm = self.playground.confirm_unload
            if bypass_confirm:
                self.playground.confirm_unload = False
            try:
                _keep_running, message = self.playground.execute(raw)
            except OSError as exc:
                self.show_error(f"EV3 connection error: {exc}")
            finally:
                self.playground.confirm_unload = old_confirm
                self.pending_unload = False
                self.busy = False
            if message:
                self.message = message.splitlines()[0]
                self.log_message(message)

        threading.Thread(target=worker, daemon=True).start()

    def buttons(self) -> list[GuiButton]:
        buttons = [
            GuiButton("Connect", "connect", (710, 58, 112, 36), requires_connection=False),
            GuiButton("Disconnect", "disconnect", (835, 58, 130, 36), requires_connection=False),
            GuiButton("Travel Position", "travel", (680, 150, 290, 42)),
            GuiButton("Pickup Assist", "assist", (680, 200, 290, 42)),
            GuiButton("Unload Full Cycle", "unload", (680, 250, 290, 42)),
            GuiButton("- Units", "unit_minus", (680, 345, 88, 38), requires_connection=False),
            GuiButton("+ Units", "unit_plus", (880, 345, 88, 38), requires_connection=False),
            GuiButton("Pipe Up", "up", (680, 400, 135, 42)),
            GuiButton("Pipe Down", "down", (835, 400, 135, 42)),
            GuiButton("Stop Pipe", "stop", (680, 450, 290, 42)),
        ]
        if self.pending_unload:
            buttons.extend(
                [
                    GuiButton("Confirm Unload", "confirm_unload", (680, 505, 180, 42)),
                    GuiButton("Cancel", "cancel_unload", (875, 505, 95, 42), requires_connection=False),
                ]
            )
        return buttons

    def handle_mouse(self, event: int, x_px: int, y_px: int, _flags: int, _userdata) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        for button in self.buttons():
            x, y, w, h = button.rect
            if not (x <= x_px <= x + w and y <= y_px <= y + h):
                continue
            if button.requires_connection and not self.connected:
                self.show_error("Connect before sending collector commands.")
                return
            if button.action == "connect":
                self.connect()
            elif button.action == "disconnect":
                self.disconnect()
            elif button.action == "unit_minus":
                self.adjust_units(-0.5)
            elif button.action == "unit_plus":
                self.adjust_units(0.5)
            elif button.action == "confirm_unload":
                self.run_gui_command("unload", bypass_confirm=True)
            elif button.action == "cancel_unload":
                self.pending_unload = False
                self.message = "Unload cancelled."
                self.log_message("Unload cancelled.")
            else:
                self.run_gui_command(button.action)
            return

    def draw_button(self, image: np.ndarray, button: GuiButton) -> None:
        x, y, w, h = button.rect
        enabled = (self.connected or not button.requires_connection) and not self.busy
        if button.action == "confirm_unload":
            fill = (45, 45, 190)
        elif enabled:
            fill = (245, 245, 245)
        else:
            fill = (205, 205, 205)
        border = (70, 70, 70)
        text = (20, 20, 20) if button.action != "confirm_unload" else (255, 255, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), fill, -1, cv2.LINE_AA)
        cv2.rectangle(image, (x, y), (x + w, y + h), border, 1, cv2.LINE_AA)
        self.draw_text(image, button.label, (x + 12, y + 27), 0.58, text, 1)

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

    def state_color(self) -> tuple[int, int, int]:
        return {
            CollectorBelief.UNKNOWN: (145, 143, 138),
            CollectorBelief.TRAVEL: (61, 122, 36),
            CollectorBelief.PICKUP_ASSIST: (31, 121, 183),
            CollectorBelief.UNLOADING: (30, 38, 179),
            CollectorBelief.MANUAL_UP: (159, 111, 47),
            CollectorBelief.MANUAL_DOWN: (162, 66, 111),
            CollectorBelief.STOPPED: (68, 68, 68),
        }.get(self.state, (145, 143, 138))

    def draw_robot(self, image: np.ndarray) -> None:
        pipe_color = self.state_color()
        cv2.rectangle(image, (30, 95), (620, 360), (250, 250, 247), -1, cv2.LINE_AA)
        cv2.rectangle(image, (30, 95), (620, 360), (190, 190, 185), 1, cv2.LINE_AA)
        self.draw_text(image, "Robot / Collector Belief", (50, 125), 0.7, (35, 35, 35), 2)

        cv2.rectangle(image, (70, 180), (285, 255), (232, 224, 217), -1, cv2.LINE_AA)
        cv2.rectangle(image, (70, 180), (285, 255), (92, 81, 69), 2, cv2.LINE_AA)
        cv2.circle(image, (105, 258), 23, (45, 45, 45), -1, cv2.LINE_AA)
        cv2.circle(image, (245, 258), 23, (45, 45, 45), -1, cv2.LINE_AA)
        cv2.arrowedLine(image, (285, 218), (335, 218), (70, 70, 70), 2, cv2.LINE_AA)
        self.draw_text(image, "front", (315, 205), 0.5)

        if self.state == CollectorBelief.TRAVEL:
            pipe_top, pipe_bottom = 135, 195
        elif self.state == CollectorBelief.PICKUP_ASSIST:
            pipe_top, pipe_bottom = 175, 278
        elif self.state == CollectorBelief.UNLOADING:
            pipe_top, pipe_bottom = 125, 298
        elif self.state == CollectorBelief.MANUAL_DOWN:
            pipe_top, pipe_bottom = 170, 292
        elif self.state == CollectorBelief.MANUAL_UP:
            pipe_top, pipe_bottom = 132, 210
        else:
            pipe_top, pipe_bottom = 155, 245
        cv2.rectangle(image, (265, pipe_top), (298, pipe_bottom), pipe_color, -1, cv2.LINE_AA)
        cv2.rectangle(image, (265, pipe_top), (298, pipe_bottom), (45, 45, 45), 2, cv2.LINE_AA)
        self.draw_text(image, "Side profile", (90, 165), 0.55)

        cv2.rectangle(image, (405, 160), (565, 285), (232, 224, 217), -1, cv2.LINE_AA)
        cv2.rectangle(image, (405, 160), (565, 285), (92, 81, 69), 2, cv2.LINE_AA)
        for x in (395, 565):
            cv2.rectangle(image, (x, 175), (x + 14, 210), (45, 45, 45), -1, cv2.LINE_AA)
            cv2.rectangle(image, (x, 235), (x + 14, 270), (45, 45, 45), -1, cv2.LINE_AA)
        cv2.ellipse(image, (565, 222), (33, 20), 0, 0, 360, pipe_color, -1, cv2.LINE_AA)
        cv2.ellipse(image, (565, 222), (33, 20), 0, 0, 360, (45, 45, 45), 2, cv2.LINE_AA)
        cv2.arrowedLine(image, (565, 222), (610, 222), (70, 70, 70), 2, cv2.LINE_AA)
        self.draw_text(image, "Top view", (435, 145), 0.55)
        self.draw_text(image, f"State: {self.state.value}", (70, 330), 0.65, pipe_color, 2)

    def render(self) -> np.ndarray:
        image = np.full((680, 1000, 3), (236, 238, 239), dtype=np.uint8)
        self.draw_text(image, "GolfBot Collector Playground", (30, 36), 0.9, (20, 20, 20), 2)
        self.draw_text(
            image,
            "No collector position sensor is present. Displayed state is open-loop software belief only.",
            (30, 68),
            0.58,
            (0, 80, 180),
            2,
        )

        self.draw_robot(image)

        cv2.rectangle(image, (660, 25), (985, 570), (248, 248, 248), -1, cv2.LINE_AA)
        cv2.rectangle(image, (660, 25), (985, 570), (190, 190, 190), 1, cv2.LINE_AA)
        self.draw_text(image, "Connection", (680, 48), 0.65, (35, 35, 35), 2)
        self.draw_text(image, f"Target: {self.args.host}:{self.args.port}  timeout {self.args.timeout:g}s", (680, 118), 0.48)
        self.draw_text(image, self.connection_label, (680, 138), 0.5, (60, 90, 60) if self.connected else (80, 60, 60), 1)
        self.draw_text(image, "Commands", (680, 142), 0.65, (35, 35, 35), 2)
        self.draw_text(image, f"Manual units: {self.manual_units:g} / max {self.args.max_manual_units:g}", (680, 330), 0.56)
        self.draw_text(image, f"Last: {self.last_command}", (680, 580), 0.5)
        self.draw_text(image, f"Message: {self.message}", (30, 395), 0.55, (30, 30, 30), 1)
        self.draw_text(image, "Keys: q/esc close   c connect   d disconnect   +/- adjust units", (30, 645), 0.52)

        for button in self.buttons():
            self.draw_button(image, button)

        cv2.rectangle(image, (30, 420), (620, 625), (250, 250, 247), -1, cv2.LINE_AA)
        cv2.rectangle(image, (30, 420), (620, 625), (190, 190, 185), 1, cv2.LINE_AA)
        self.draw_text(image, "Command Log", (50, 448), 0.65, (35, 35, 35), 2)
        y_px = 475
        for line in self.log_lines[-7:]:
            self.draw_text(image, line[:82], (50, y_px), 0.48)
            y_px += 22
        return image

    def close(self) -> None:
        if self.playground is not None:
            self.playground._try_pipe_stop()
        self.closed = True

    def run(self) -> None:
        cv2.namedWindow(GUI_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(GUI_WINDOW_NAME, 1000, 680)
        cv2.setMouseCallback(GUI_WINDOW_NAME, self.handle_mouse)
        while not self.closed:
            cv2.imshow(GUI_WINDOW_NAME, self.render())
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                self.close()
            elif key == ord("c"):
                self.connect()
            elif key == ord("d"):
                self.disconnect()
            elif key in (ord("+"), ord("=")):
                self.adjust_units(0.5)
            elif key in (ord("-"), ord("_")):
                self.adjust_units(-0.5)
            if cv2.getWindowProperty(GUI_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self.close()
        cv2.destroyWindow(GUI_WINDOW_NAME)


def interactive_loop(playground: CollectorPlayground) -> None:
    print(command_help())
    while True:
        try:
            raw = input("collector> ")
        except EOFError:
            print("")
            playground._try_pipe_stop()
            break
        except KeyboardInterrupt:
            print("\nCtrl+C received; trying to stop pipe motor before exit.")
            playground._try_pipe_stop()
            break

        try:
            keep_running, message = playground.execute(raw)
        except OSError as exc:
            print(f"EV3 connection error: {exc}")
            continue
        if message:
            print(message)
        if not keep_running:
            break


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.max_manual_units <= 0 or not math.isfinite(args.max_manual_units):
        print("--max-manual-units must be a finite positive number.", file=sys.stderr)
        return 2

    if not args.cli:
        try:
            gui = CollectorPlaygroundGui(args)
        except ImportError as exc:
            print(f"Could not start OpenCV GUI: {exc}", file=sys.stderr)
            return 1
        gui.run()
        return 0

    print_startup_warning()
    if args.dummy:
        print("Dummy mode enabled; no EV3 connection will be opened.")
        controller = DummyCollectorController()
    else:
        try:
            controller = RobotCommander(args.host, port=args.port, timeout=args.timeout)
        except OSError as exc:
            print(f"Could not connect to EV3 controller at {args.host}:{args.port}: {exc}", file=sys.stderr)
            return 1

    playground = CollectorPlayground(
        controller=controller,
        confirm_unload=not (args.no_confirm or args.yes),
        max_manual_units=args.max_manual_units,
    )
    interactive_loop(playground)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
