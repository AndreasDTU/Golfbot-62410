#!/usr/bin/env python3
"""Manual open-loop playground for the GolfBot collector pipe.

This tool talks only to the EV3 TCP controller. It does not import or start the
vision pipeline, route planner, detector shell, UDP wheel dispatcher, or any
autonomous drive loop.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot.controller import RobotController


DEFAULT_MAX_MANUAL_UNITS = 5.0


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


@dataclass
class CollectorPlayground:
    controller: CollectorController
    confirm_unload: bool = True
    max_manual_units: float = DEFAULT_MAX_MANUAL_UNITS
    confirm: ConfirmFn | None = None
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
            self.state = CollectorBelief.TRAVEL
            return True, self._format_response(response)
        if command in ("assist", "pickup"):
            response = self.controller.pickup_assist()
            self.state = CollectorBelief.PICKUP_ASSIST
            return True, self._format_response(response)
        if command in ("unload", "dropoff"):
            if self.confirm_unload and not self._confirm_unload():
                return True, "Unload cancelled."
            response = self.controller.unload_full_cycle()
            self.state = CollectorBelief.UNLOADING
            return True, self._format_response(response)
        if command == "up":
            units = self._parse_manual_units(parts)
            if isinstance(units, str):
                return True, units
            response = self.controller.pipe_up(units)
            self.state = CollectorBelief.MANUAL_UP
            return True, self._format_response(response)
        if command == "down":
            units = self._parse_manual_units(parts)
            if isinstance(units, str):
                return True, units
            response = self.controller.pipe_down(units)
            self.state = CollectorBelief.MANUAL_DOWN
            return True, self._format_response(response)
        if command == "stop":
            response = self._stop_pipe_first()
            self.state = CollectorBelief.STOPPED
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


def print_startup_warning() -> None:
    print("No collector position sensor is present. Start with the collector in a known safe position. Software state is open-loop.")
    print("This playground does not drive wheels, follow paths, run vision, plan routes, or detect balls.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual EV3 collector/pipe playground.")
    parser.add_argument("--host", default="ev3dev", help="EV3 hostname or IP address.")
    parser.add_argument("--port", type=int, default=5555, help="EV3 TCP command port.")
    parser.add_argument("--timeout", type=float, default=15.0, help="TCP connect/read timeout in seconds.")
    parser.add_argument("--no-confirm", action="store_true", help="Do not ask before running unload.")
    parser.add_argument("--yes", action="store_true", help="Alias for --no-confirm.")
    parser.add_argument(
        "--max-manual-units",
        type=float,
        default=DEFAULT_MAX_MANUAL_UNITS,
        help="Maximum units allowed for one manual up/down command.",
    )
    return parser


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

    print_startup_warning()
    try:
        controller = RobotController(args.host, port=args.port, timeout=args.timeout)
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
