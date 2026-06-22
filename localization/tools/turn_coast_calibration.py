#!/usr/bin/env python3
"""Turn coast calibration tool — multi-angle sweep, camera-assisted.

Validates the proportional-coast model: coast distance scales linearly with
turn speed, and turn speed scales with the total commanded angle. For each
angle the robot runs a real guidance loop with turn_coast_deg forced to 0, so
it coasts freely past the target; the overshoot is the raw coast at that
angle's speed.

Per trial:
  1. Read the heading before (averaged over several frames).
  2. Stream camera frames, compute remaining angle, call commander.turn(remaining)
     each frame — exactly as the guidance layer does.
  3. When the robot crosses the target heading, send stop().
  4. Settle, read the final heading. Overshoot = final - target = raw coast.

Across angles, the tool back-calculates what turn_coast_deg (the coast at full
speed) would have to be for each, and only writes config.py if those agree
(low std dev) — i.e. the linear coast∝speed model holds.

Requirements:
  - Field corners must be calibrated (data/field_corners.json exists).
  - Robot calibration must exist (data/robot_calibration.json exists).
  - Camera connected; ArUco markers visible through the whole turn.

Usage:
    python -m localization.tools.turn_coast_calibration
    python -m localization.tools.turn_coast_calibration --angles 180,90,45,30 --trials 3
    python -m localization.tools.turn_coast_calibration --dummy
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import AppConfig, ConnectionConfig, DriveConfig
from control.commander import RobotCommander
from localization.localization import RobotCalibrationCollector, RobotPoseEstimator, normalize_angle
from perception.vision.detection import BallDetector
from perception.vision.pipeline import VisionPipeline


class _NullBallDetector(BallDetector):
    """Stub detector — this tool only needs the warp + pose, not ball detection."""

    def detect(self, frame_bgr, params, camera_center_pixels):
        return [], [], {
            "white": np.zeros(frame_bgr.shape[:2], dtype=np.uint8),
            "orange": np.zeros(frame_bgr.shape[:2], dtype=np.uint8),
        }

# Wrap-safe angles only: the before/after heading measurement folds into
# [-180, 180), so a turn whose actual rotation (angle + coast) reaches ~180°
# cannot be measured. Keep the largest angle comfortably below that.
DEFAULT_SWEEP_ANGLES = [120.0, 90.0, 60.0, 45.0, 30.0, 15.0]
WRAP_SAFE_MAX_DEG = 150.0  # refuse angles at/above this — measurement would wrap
DEFAULT_TRIALS = 3   # trials per angle
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
# Setup + speed law
# ---------------------------------------------------------------------------

def _setup(config: AppConfig):
    """Load pipeline, pose estimator, calibration, params, and open the camera."""
    pipeline = VisionPipeline(app_config=config, ball_detector=_NullBallDetector())
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

    return pipeline, pose_estimator, params, calibration, camera


def _turn_speed_pct(cfg: DriveConfig, angle_deg: float) -> float:
    """Mirror RobotCommander._target_speed_for_angle for the given total angle."""
    raw = abs(angle_deg) / cfg.turn_reference_angle_deg * cfg.turn_max_speed_pct
    return max(cfg.turn_min_speed_pct, min(cfg.turn_max_speed_pct, raw))


# ---------------------------------------------------------------------------
# Multi-angle sweep — validates the proportional-coast model
# ---------------------------------------------------------------------------

def run_sweep(angles: list[float], trials_per_angle: int, commander: RobotCommander) -> None:
    """Sweep angles, measure raw coast at each, back-calculate the full-speed coast.

    The robot runs with turn_coast_deg=0 so it coasts freely past the target;
    the overshoot IS the raw coast at that angle's speed.  For each angle we
    back-calculate what turn_coast_deg (the coast at full speed) would have to
    be: coast_full = measured / (speed / max_speed).  If those agree across
    angles (low std dev), the linear coast∝speed model holds and we write the
    mean to config.py.
    """
    config = AppConfig.from_repo_root(REPO_ROOT)
    pipeline, pose_estimator, params, calibration, camera = _setup(config)
    cfg = commander._config

    print(f"\nTurn coast sweep — {len(angles)} angles × {trials_per_angle} trials each")
    print("turn_coast_deg forced to 0 (raw coast measurement).")
    print(f"turn_reference_angle_deg: {cfg.turn_reference_angle_deg:.0f}°  "
          f"turn_max_speed_pct: {cfg.turn_max_speed_pct:.0f}%  "
          f"turn_min_speed_pct: {cfg.turn_min_speed_pct:.0f}%")
    print("Make sure the ArUco markers stay visible through the whole turn.\n")

    results: dict[float, list[float]] = {}
    try:
        for angle in angles:
            if abs(angle) >= WRAP_SAFE_MAX_DEG:
                print(f"\n--- {angle:+.0f}° SKIPPED — at/above {WRAP_SAFE_MAX_DEG:.0f}°, the "
                      f"before/after heading measurement wraps and cannot be trusted. ---")
                continue
            speed_pct = _turn_speed_pct(cfg, angle)
            print(f"\n--- {angle:+.0f}°  (speed ≈ {speed_pct:.1f}%) ---")
            measured: list[float] = []
            for i in range(1, trials_per_angle + 1):
                print(f"  Trial {i}/{trials_per_angle}...")
                #input(f"  Trial {i}/{trials_per_angle} — press Enter to start...")
                overshoot = _run_trial(
                    angle, commander, camera, pipeline, pose_estimator, params, calibration,
                )
                if overshoot is None:
                    print("  Trial skipped.")
                    continue
                measured.append(overshoot)
                print(f"  Overshoot: {overshoot:+.1f}°")
            results[angle] = measured
    finally:
        commander.stop()
        commander.close()
        camera.release()

    # Validation table — collect (speed%, mean measured coast) points for a fit
    print("\n" + "=" * 70)
    print(f"{'Angle':>8}  {'Speed%':>7}  {'Measured coast':>15}  {'Coast@full speed':>17}")
    print("-" * 70)
    points: list[tuple[float, float]] = []
    for angle in angles:
        data = results.get(angle, [])
        if not data:
            print(f"{angle:>7.0f}°  {'—':>7}  {'no data':>15}  {'—':>17}")
            continue
        speed_pct = _turn_speed_pct(cfg, angle)
        speed_ratio = speed_pct / cfg.turn_max_speed_pct
        measured_coast = sum(data) / len(data)
        coast_full = measured_coast / speed_ratio if speed_ratio > 0 else float("nan")
        points.append((speed_pct, measured_coast))
        print(f"{angle:>7.0f}°  {speed_pct:>6.1f}%  {measured_coast:>14.1f}°  {coast_full:>16.1f}°")
    print("=" * 70)

    if len(points) < 2:
        print("Not enough valid data to fit the model.")
        return

    # Speed-weighted through-origin least squares: coast = slope * speed.
    # This trusts the clean high-speed points over the noise-amplified floor
    # points (where speed is pinned at the minimum).
    sum_sc = sum(s * c for s, c in points)
    sum_ss = sum(s * s for s, c in points)
    if sum_ss <= 0:
        print("All speeds zero — cannot fit.")
        return
    slope = sum_sc / sum_ss  # coast degrees per 1% speed
    coast_deg = slope * cfg.turn_max_speed_pct

    ss_res = sum((c - slope * s) ** 2 for s, c in points)
    ss_tot = sum(c * c for s, c in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"\nFit: coast = {slope:.3f}° per 1% speed   (R² = {r2:.3f})")
    print(f"Implied turn_coast_deg (coast at {cfg.turn_max_speed_pct:.0f}% speed): {coast_deg:.1f}°")

    R2_THRESHOLD = 0.90
    if r2 >= R2_THRESHOLD:
        print(f"Model FITS (R² {r2:.3f} >= {R2_THRESHOLD:.2f}) — coast is linear in speed.")
        _update_config(coast_deg)
    else:
        print(f"Model POOR FIT (R² {r2:.3f} < {R2_THRESHOLD:.2f}). config.py NOT updated — "
              f"coast may not be linear in speed; review the table.")


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
        description="Sweep turn angles and validate the proportional-coast model."
    )
    parser.add_argument("--angles", type=str, default=",".join(str(int(a)) for a in DEFAULT_SWEEP_ANGLES),
                        help=f"Comma-separated turn angles, positive=CCW "
                             f"(default: {','.join(str(int(a)) for a in DEFAULT_SWEEP_ANGLES)})")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help=f"Trials per angle (default: {DEFAULT_TRIALS})")
    parser.add_argument("--dummy", action="store_true",
                        help="Simulate robot commands (no real connection)")
    args = parser.parse_args()

    try:
        angles = [float(a) for a in args.angles.split(",") if a.strip()]
    except ValueError:
        print(f"Invalid --angles: {args.angles!r}")
        sys.exit(1)
    if not angles:
        print("No angles given.")
        sys.exit(1)

    if args.dummy:
        commander: RobotCommander = _DummyCommander()
    else:
        try:
            # Force turn_coast_deg=0 so the robot coasts freely past the target —
            # the measured overshoot is the raw coast at each angle's speed.
            commander = RobotCommander(
                connection_config=ConnectionConfig(),
                drive_config=DriveConfig(turn_coast_deg=0.0),
                auto_connect=True,
            )
        except RuntimeError as exc:
            print(f"Connection failed: {exc}")
            sys.exit(1)

    run_sweep(angles, args.trials, commander)


if __name__ == "__main__":
    main()
