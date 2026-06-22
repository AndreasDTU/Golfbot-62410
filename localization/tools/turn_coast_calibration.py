#!/usr/bin/env python3
"""Turn coast calibration tool — camera-assisted guidance loop.

Runs a real guidance-like loop for each trial:

  1. Reads the heading before the turn (averaged over several frames).
  2. Computes target_heading = heading_before + angle_deg.
  3. Streams camera frames, computes remaining angle each frame, and calls
     commander.turn(remaining) — exactly as the guidance layer does.
     turn_coast_deg is forced to 0 so the stop fires at the true target.
  4. When the robot crosses the target heading, sends stop().
  5. Waits for the robot to settle, then reads the final heading.
  6. Overshoot = final - target.

This gives the coast distance under real speed-profile conditions (the robot
decelerates toward turn_min_speed_pct before the stop fires), not at max speed.

Requirements:
  - Field corners must be calibrated (data/field_corners.json exists).
  - Robot calibration must exist (data/robot_calibration.json exists).
  - Camera must be connected and the ArUco markers must be visible.

Usage:
    python -m localization.tools.turn_coast_calibration
    python -m localization.tools.turn_coast_calibration --angle 90 --trials 10
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

DEFAULT_ANGLE_DEG = 90.0
DEFAULT_TRIALS = 10
SETTLE_S = 1.0        # seconds to wait after stop before sampling final heading
SAMPLE_FRAMES = 15    # frames to average for stable heading readings
SAMPLE_INTERVAL_S = 0.05
LOOP_INTERVAL_S = 0.05   # ~20 fps guidance loop
TURN_TIMEOUT_S = 20.0    # abort if turn takes longer than this


# ---------------------------------------------------------------------------
# Dummy commander
# ---------------------------------------------------------------------------

class _DummyCommander(RobotCommander):
    def __init__(self) -> None:
        super().__init__(auto_connect=False)

    def _send_nowait(self, cmd: str) -> bool:
        return True


# ---------------------------------------------------------------------------
# Heading sampler
# ---------------------------------------------------------------------------

def _read_heading(
    camera: cv2.VideoCapture,
    pipeline: VisionPipeline,
    pose_estimator: RobotPoseEstimator,
    params: dict,
    calibration: dict,
) -> float | None:
    """Read a single pose estimate from the next camera frame."""
    ok, raw = camera.read()
    if not ok or raw is None:
        return None
    preprocessed = pipeline.preprocessor.process(raw)
    topdown = preprocessed.topdown
    if topdown is None:
        return None
    pose, _, _, _ = pose_estimator.estimate(topdown, params, calibration)
    return pose.heading_rad if pose is not None else None


def _sample_heading(
    camera: cv2.VideoCapture,
    pipeline: VisionPipeline,
    pose_estimator: RobotPoseEstimator,
    params: dict,
    calibration: dict,
    n_frames: int,
) -> float | None:
    """Average n_frames heading readings into a stable estimate."""
    sin_sum = 0.0
    cos_sum = 0.0
    count = 0
    for _ in range(n_frames):
        h = _read_heading(camera, pipeline, pose_estimator, params, calibration)
        if h is not None:
            sin_sum += math.sin(h)
            cos_sum += math.cos(h)
            count += 1
        time.sleep(SAMPLE_INTERVAL_S)
    if count == 0:
        return None
    return math.atan2(sin_sum / count, cos_sum / count)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def _run_trial(
    angle_deg: float,
    commander: RobotCommander,
    camera: cv2.VideoCapture,
    pipeline: VisionPipeline,
    pose_estimator: RobotPoseEstimator,
    params: dict,
    calibration: dict,
) -> float | None:
    """Run one trial. Returns overshoot in degrees, or None on failure."""

    # Stable heading before
    print("  Sampling heading before...", end=" ", flush=True)
    heading_before_rad = _sample_heading(
        camera, pipeline, pose_estimator, params, calibration, SAMPLE_FRAMES,
    )
    if heading_before_rad is None:
        print("FAILED (markers not visible)")
        return None
    print(f"{math.degrees(heading_before_rad):.1f}°")

    target_rad = normalize_angle(heading_before_rad + math.radians(angle_deg))
    sign = 1.0 if angle_deg >= 0 else -1.0

    # Guidance-like loop: stream pose, compute remaining, call turn() each frame
    print(f"  Turning {angle_deg:+.1f}°...", end=" ", flush=True)
    start = time.perf_counter()
    stopped = False

    while time.perf_counter() - start < TURN_TIMEOUT_S:
        heading = _read_heading(camera, pipeline, pose_estimator, params, calibration)
        if heading is None:
            time.sleep(LOOP_INTERVAL_S)
            continue

        remaining_rad = normalize_angle(target_rad - heading)
        remaining_deg = math.degrees(remaining_rad)

        # Stop when we've reached or crossed the target in the commanded direction
        if remaining_deg * sign <= 0:
            commander.stop()
            stopped = True
            break

        commander.turn(remaining_deg)
        time.sleep(LOOP_INTERVAL_S)

    if not stopped:
        commander.stop()
        print("TIMEOUT")
        return None

    # Settle
    time.sleep(SETTLE_S)

    # Stable heading after
    print("done")
    print("  Sampling heading after...", end=" ", flush=True)
    heading_after_rad = _sample_heading(
        camera, pipeline, pose_estimator, params, calibration, SAMPLE_FRAMES,
    )
    if heading_after_rad is None:
        print("FAILED (markers not visible)")
        return None
    print(f"{math.degrees(heading_after_rad):.1f}°")

    actual_deg = math.degrees(normalize_angle(heading_after_rad - heading_before_rad))
    overshoot = actual_deg - angle_deg
    return overshoot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(angle_deg: float, trials: int, commander: RobotCommander) -> None:
    config = AppConfig.from_repo_root(REPO_ROOT)
    pipeline = VisionPipeline(app_config=config)
    pose_estimator = RobotPoseEstimator(
        field_config=config.field,
        robot_config=config.robot,
        mapper=pipeline.mapper,
    )
    collector = RobotCalibrationCollector(config.robot)

    corners_path = config.paths.field_corners_file
    if corners_path is None or not corners_path.exists():
        print("ERROR: No field corners file. Run field corner calibration first.")
        sys.exit(1)
    pipeline.preprocessor.homography_calibrator.load_manual_corners(corners_path)

    cal_path = config.paths.robot_calibration_file
    if cal_path is None or not cal_path.exists():
        print("ERROR: No robot calibration file. Run robot calibration first.")
        sys.exit(1)
    calibration = collector.load_robot_calibration(cal_path, config.camera.topdown_warp_size)
    if calibration is None:
        print("ERROR: Robot calibration file invalid or missing marker IDs.")
        sys.exit(1)

    params = pipeline.default_params()
    RobotCalibrationCollector.apply_geometry_to_params(params, calibration.get("geometry"))

    camera = cv2.VideoCapture(config.camera.camera_index)
    if not camera.isOpened():
        print(f"ERROR: Could not open camera index {config.camera.camera_index}.")
        sys.exit(1)

    overshoots: list[float] = []

    current_coast = DriveConfig().turn_coast_deg
    print(f"\nTurn coast calibration — {angle_deg:+.1f}° × {trials} trials")
    print(f"Current turn_coast_deg: {current_coast:.1f}°  (active during this run)")
    print("Residual overshoot will be added to refine the value.")
    print("Make sure the ArUco markers are fully visible from the camera.\n")

    try:
        for i in range(1, trials + 1):
            input(f"Trial {i}/{trials} — press Enter to start...")
            overshoot = _run_trial(
                angle_deg, commander, camera, pipeline, pose_estimator, params, calibration,
            )
            if overshoot is None:
                print("  Trial skipped.\n")
                continue
            overshoots.append(overshoot)
            print(f"  Overshoot: {overshoot:+.1f}°\n")
    finally:
        commander.stop()
        commander.close()
        camera.release()

    if not overshoots:
        print("No valid trials recorded.")
        return

    current_coast = DriveConfig().turn_coast_deg
    avg_residual = sum(overshoots) / len(overshoots)
    new_coast = current_coast + avg_residual

    print("=" * 48)
    print(f"Trials recorded  : {len(overshoots)}")
    print(f"Individual       : {', '.join(f'{v:+.1f}' for v in overshoots)}")
    print(f"Residual overshoot: {avg_residual:+.2f}°")
    print(f"Current turn_coast_deg: {current_coast:.1f}°")
    print(f"New    turn_coast_deg: {new_coast:.1f}°")
    print()

    _update_config(new_coast)

    print("=" * 48)


def _update_config(coast_deg: float) -> None:
    """Write the measured turn_coast_deg into config.py in-place."""
    config_path = REPO_ROOT / "config.py"
    text = config_path.read_text(encoding="utf-8")

    import re
    pattern = r"(turn_coast_deg\s*:\s*float\s*=\s*)[0-9eE+\-.]+"
    replacement = rf"\g<1>{coast_deg:.1f}"
    new_text, count = re.subn(pattern, replacement, text)

    if count == 0:
        print(f"Could not find turn_coast_deg in config.py — set it manually to {coast_deg:.1f}")
        return

    config_path.write_text(new_text, encoding="utf-8")
    print(f"config.py updated: turn_coast_deg = {coast_deg:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure turn coast under real guidance-loop conditions."
    )
    parser.add_argument("--angle", type=float, default=DEFAULT_ANGLE_DEG,
                        help=f"Turn angle in degrees, positive=CCW (default: {DEFAULT_ANGLE_DEG})")
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

    run(args.angle, args.trials, commander)


if __name__ == "__main__":
    main()
