#!/usr/bin/env python3
"""Turn coast calibration tool — camera-assisted.

Measures the actual turn angle using the ArUco localization stack so you
don't need a protractor.  For each trial the tool:

  1. Reads the heading before the turn (averaged over several frames).
  2. Commands the turn via RobotCommander.
  3. Waits for the robot to settle.
  4. Reads the heading after the turn (averaged over several frames).
  5. Computes overshoot = actual - commanded.

After all trials it prints the average overshoot and the exact value to
set in DriveConfig.turn_coast_deg.

Requirements:
  - Field corners must be calibrated (data/field_corners.json exists).
  - Robot calibration must exist (data/robot_calibration.json exists).
  - Camera must be connected and the ArUco markers must be visible.

Usage:
    python -m localization.tools.turn_coast_calibration
    python -m localization.tools.turn_coast_calibration --angle 30 --trials 10
    python -m localization.tools.turn_coast_calibration --dummy
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import AppConfig, ConnectionConfig, DriveConfig
from control.commander import RobotCommander
from localization.localization import RobotCalibrationCollector, RobotPoseEstimator, normalize_angle
from perception.vision.pipeline import VisionPipeline

DEFAULT_ANGLE_DEG = 30.0
DEFAULT_TRIALS = 10
SETTLE_S = 1.5       # seconds to wait after the turn before sampling
SAMPLE_FRAMES = 15   # frames to average for each heading reading
SAMPLE_INTERVAL_S = 0.05  # seconds between sample frames (~20 fps)


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
# Heading sampler
# ---------------------------------------------------------------------------

def _sample_heading(
    camera: cv2.VideoCapture,
    pipeline: VisionPipeline,
    pose_estimator: RobotPoseEstimator,
    params: dict,
    calibration: dict,
    n_frames: int,
) -> float | None:
    """Capture n_frames poses and return their circular-mean heading, or None."""
    sin_sum = 0.0
    cos_sum = 0.0
    count = 0

    for _ in range(n_frames):
        ok, raw = camera.read()
        if not ok or raw is None:
            time.sleep(SAMPLE_INTERVAL_S)
            continue
        preprocessed = pipeline.preprocessor.process(raw)
        topdown = preprocessed.topdown
        if topdown is None:
            time.sleep(SAMPLE_INTERVAL_S)
            continue
        pose, _, _, _ = pose_estimator.estimate(topdown, params, calibration)
        if pose is not None:
            sin_sum += math.sin(pose.heading_rad)
            cos_sum += math.cos(pose.heading_rad)
            count += 1
        time.sleep(SAMPLE_INTERVAL_S)

    if count == 0:
        return None
    return math.atan2(sin_sum / count, cos_sum / count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(angle_deg: float, trials: int, commander: RobotCommander, dummy: bool) -> None:
    config = AppConfig.from_repo_root(REPO_ROOT)
    pipeline = VisionPipeline(app_config=config)
    pose_estimator = RobotPoseEstimator(
        field_config=config.field,
        robot_config=config.robot,
        mapper=pipeline.mapper,
    )
    collector = RobotCalibrationCollector(config.robot)

    # Load field corners
    corners_path = config.paths.field_corners_file
    if corners_path is None or not corners_path.exists():
        print("ERROR: No field corners file found. Run field corner calibration first.")
        sys.exit(1)
    pipeline.preprocessor.homography_calibrator.load_manual_corners(corners_path)

    # Load robot calibration
    cal_path = config.paths.robot_calibration_file
    if cal_path is None or not cal_path.exists():
        print("ERROR: No robot calibration file found. Run robot calibration first.")
        sys.exit(1)
    calibration = collector.load_robot_calibration(cal_path, config.camera.topdown_warp_size)
    if calibration is None:
        print("ERROR: Robot calibration file invalid or missing marker IDs.")
        sys.exit(1)

    params = pipeline.default_params()
    RobotCalibrationCollector.apply_geometry_to_params(params, calibration.get("geometry"))

    # Open camera
    camera = cv2.VideoCapture(config.camera.camera_index)
    if not camera.isOpened():
        print(f"ERROR: Could not open camera index {config.camera.camera_index}.")
        sys.exit(1)

    overshots: list[float] = []

    print(f"\nTurn coast calibration — {angle_deg:.1f}° × {trials} trials")
    print("Make sure the ArUco markers are fully visible from the camera.\n")

    try:
        for i in range(1, trials + 1):
            input(f"Trial {i}/{trials} — press Enter to start...")

            # Sample heading before
            print("  Sampling heading before turn...", end=" ", flush=True)
            heading_before = _sample_heading(
                camera, pipeline, pose_estimator, params, calibration, SAMPLE_FRAMES,
            )
            if heading_before is None:
                print("FAILED (markers not visible) — skipping trial.")
                continue
            print(f"{math.degrees(heading_before):.1f}°")

            # Command turn
            print(f"  Commanding {angle_deg:.1f}° CCW turn...")
            commander.turn(angle_deg)
            time.sleep(SETTLE_S)
            commander.stop()
            time.sleep(0.3)

            # Sample heading after
            print("  Sampling heading after turn...", end=" ", flush=True)
            heading_after = _sample_heading(
                camera, pipeline, pose_estimator, params, calibration, SAMPLE_FRAMES,
            )
            if heading_after is None:
                print("FAILED (markers not visible) — skipping trial.")
                continue
            print(f"{math.degrees(heading_after):.1f}°")

            actual_deg = math.degrees(normalize_angle(heading_after - heading_before))
            overshoot = actual_deg - angle_deg
            overshots.append(overshoot)
            print(f"  Actual: {actual_deg:.1f}°  |  Overshoot: {overshoot:+.1f}°\n")

    finally:
        commander.stop()
        commander.close()
        camera.release()

    if not overshots:
        print("No valid trials recorded.")
        return

    avg = sum(overshots) / len(overshots)
    print("=" * 48)
    print(f"Trials recorded  : {len(overshots)}")
    print(f"Individual       : {', '.join(f'{v:+.1f}' for v in overshots)}")
    print(f"Average overshoot: {avg:+.2f}°")
    print()
    print(f"Set DriveConfig.turn_coast_deg = {avg:.1f}  in config.py")
    print("=" * 48)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure turn motor coast using the camera.")
    parser.add_argument("--angle", type=float, default=DEFAULT_ANGLE_DEG,
                        help=f"Turn angle in degrees (default: {DEFAULT_ANGLE_DEG})")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help=f"Number of trials (default: {DEFAULT_TRIALS})")
    parser.add_argument("--dummy", action="store_true",
                        help="Simulate robot commands (no real connection)")
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

    run(args.angle, args.trials, commander, args.dummy)


if __name__ == "__main__":
    main()
