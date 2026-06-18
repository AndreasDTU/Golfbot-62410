#!/usr/bin/env python3
"""Turn coast calibration tool.

Commands a configurable turn angle, then asks you to measure the actual
heading change with a protractor or ArUco overlay.  Repeat for the
configured number of trials and the tool prints the average overshoot —
that number goes into DriveConfig.turn_coast_deg.

Usage:
    python -m control.tools.turn_coast_calibration
    python -m control.tools.turn_coast_calibration --angle 30 --trials 10
    python -m control.tools.turn_coast_calibration --dummy   # no robot needed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control.commander import RobotCommander
from config import ConnectionConfig, DriveConfig

DEFAULT_ANGLE_DEG = 30.0
DEFAULT_TRIALS = 10
SETTLE_S = 1.5  # seconds to wait after each turn before prompting


# ---------------------------------------------------------------------------
# Dummy commander
# ---------------------------------------------------------------------------

class _DummyCommander(RobotCommander):
    def __init__(self) -> None:
        super().__init__(auto_connect=False)

    def _send_nowait(self, cmd: str) -> bool:
        print(f"  [dummy] {cmd}")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(angle_deg: float, trials: int, commander: RobotCommander) -> None:
    overshots: list[float] = []

    print(f"\nTurn coast calibration — {angle_deg:.1f}° × {trials} trials")
    print("Place the robot on the floor with room to turn.")
    print("Mark the starting heading before each trial.\n")

    for i in range(1, trials + 1):
        input(f"Trial {i}/{trials} — press Enter to command {angle_deg:.1f}° CCW turn...")

        commander.turn(angle_deg)
        time.sleep(SETTLE_S)
        commander.stop()
        time.sleep(0.3)

        raw = input(f"  Measured actual turn (degrees, e.g. 37.5): ").strip()
        try:
            actual = float(raw)
        except ValueError:
            print("  Invalid input — skipping trial.")
            continue

        overshoot = actual - angle_deg
        overshots.append(overshoot)
        print(f"  Overshoot: {overshoot:+.1f}°\n")

    if not overshots:
        print("No valid trials recorded.")
        return

    avg = sum(overshots) / len(overshots)
    print("=" * 48)
    print(f"Trials recorded : {len(overshots)}")
    print(f"Individual      : {', '.join(f'{v:+.1f}' for v in overshots)}")
    print(f"Average overshoot: {avg:+.2f}°")
    print()
    print(f"Set DriveConfig.turn_coast_deg = {avg:.1f}  in config.py")
    print("=" * 48)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure turn motor coast overshoot.")
    parser.add_argument("--angle", type=float, default=DEFAULT_ANGLE_DEG,
                        help=f"Turn angle in degrees (default: {DEFAULT_ANGLE_DEG})")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help=f"Number of trials (default: {DEFAULT_TRIALS})")
    parser.add_argument("--dummy", action="store_true",
                        help="Run without a real robot (prints commands only)")
    args = parser.parse_args()

    if args.dummy:
        commander: RobotCommander = _DummyCommander()
    else:
        try:
            commander = RobotCommander(
                connection_config=ConnectionConfig(),
                drive_config=DriveConfig(),
                auto_connect=True,
            )
        except RuntimeError as exc:
            print(f"Connection failed: {exc}")
            sys.exit(1)

    try:
        run(args.angle, args.trials, commander)
    finally:
        commander.stop()
        commander.close()


if __name__ == "__main__":
    main()
