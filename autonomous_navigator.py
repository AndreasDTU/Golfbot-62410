#!/usr/bin/env python3
"""Closed-loop ArUco, HSV, and A* autonomous navigator for the tank robot.

This script intentionally composes the existing classical OpenCV detector pieces
instead of replacing them. Safety behavior is conservative: if localization,
pathfinding, clearance, or motor output is uncertain, the drive command is zero.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.topdown_object_detector import (  # noqa: E402
    FIELD_HEIGHT_CM,
    FIELD_WIDTH_CM,
    TOPDOWN_WARP_SIZE,
    BallDetection,
    RedZoneDetection,
    HSVRange,
    a_star_search,
    build_occupancy_grid,
    build_aruco_detector,
    detect_aruco_markers,
    detect_balls,
    detect_red_zones,
    field_metric_cm_to_grid_node,
    pixel_to_field_cm,
)

CALIBRATION_FILE = REPO_ROOT / "calibration_data.npz"
HOMOGRAPHY_FILE = REPO_ROOT / "robot_topdown_homography.npz"
ROBOT_CALIBRATION_FILE = REPO_ROOT / "robot_calibration.json"
NAVIGATOR_TUNING_FILE = REPO_ROOT / "navigator_tuning.json"
WINDOW_NAME = "Autonomous Navigator"
MASK_WINDOW_NAME = "Autonomous Masks"
TUNING_WINDOW_NAME = "Robot Body Tuning"

# Edit these once for your robot, then run with only --dry or --drive.
ROBOT_ARUCO_ID = 4
ROBOT_HOST = "192.168.0.0"  # Replace with the EV3 IP address before using --drive.
ROBOT_PORT = 5555
ROBOT_SPEED_COMMAND = "motors"
DEFAULT_DRY_RUN = True

ROBOT_RADIUS_CM = 15.0
ROBOT_LENGTH_CM = 30.0
ROBOT_WIDTH_CM = 25.0
ROBOT_HEADING_OFFSET_DEG = 0.0
LOOKAHEAD_CM = 15.0
GOAL_STOP_DISTANCE_CM = 6.0
MIN_RED_CLEARANCE_CM = 2.0
MAX_SAFE_SPEED = 35.0
KP_DISTANCE = 2.0
KP_TURN = 22.0
MAX_HEADING_FOR_FORWARD_RAD = math.radians(70.0)
CONTROL_HZ = 12.0
CAMERA_INDEX = 0
FISHEYE_BALANCE = 0.0

MotorSetter = Callable[[float, float], None]


@dataclass(frozen=True)
class RobotPose:
    """Robot pose in bottom-left-origin field coordinates."""

    x_cm: float
    y_cm: float
    theta_rad: float
    marker_id: int


@dataclass(frozen=True)
class RobotMarkerCalibration:
    """Robot marker calibration stored by tools/robot_origin_calibration.py."""

    dx_px: float
    dy_px: float
    alpha_rad: float
    marker_height_cm: float
    camera_height_cm: float
    topdown_size: tuple[int, int]


@dataclass(frozen=True)
class TargetBall:
    """Ball selected as the current navigation goal."""

    detection: BallDetection
    x_cm: float
    y_cm: float
    grid_node: tuple[int, int]


@dataclass(frozen=True)
class ControlCommand:
    """Motor command plus controller debug values."""

    left: float
    right: float
    distance_error_cm: float
    heading_error_rad: float
    waypoint_cm: tuple[float, float]


@dataclass
class RobotBodyTuning:
    """Visual and heading tuning persisted between runs."""

    length_cm: float = ROBOT_LENGTH_CM
    width_cm: float = ROBOT_WIDTH_CM
    heading_offset_deg: float = ROBOT_HEADING_OFFSET_DEG
    last_saved: tuple[float, float, float] | None = None
    last_save_time: float = 0.0


class SafeMotorInterface:
    """Small wrapper that guarantees finite, clipped motor commands."""

    def __init__(self, set_motors: MotorSetter, max_speed: float) -> None:
        self._set_motors = set_motors
        self._max_speed = float(abs(max_speed))
        self.last_command = (0.0, 0.0)

    def set(self, left_speed: float, right_speed: float) -> None:
        if not (math.isfinite(left_speed) and math.isfinite(right_speed)):
            left_speed = 0.0
            right_speed = 0.0
        left = float(np.clip(left_speed, -self._max_speed, self._max_speed))
        right = float(np.clip(right_speed, -self._max_speed, self._max_speed))
        self._set_motors(left, right)
        self.last_command = (left, right)

    def stop(self) -> None:
        self.set(0.0, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run closed-loop autonomous ball navigation.")
    drive_mode = parser.add_mutually_exclusive_group()
    drive_mode.add_argument("--dry", dest="dry_run", action="store_true", help="Never connect to or command the robot.")
    drive_mode.add_argument("--drive", dest="dry_run", action="store_false", help="Connect to the EV3 and send motor commands.")
    parser.set_defaults(dry_run=DEFAULT_DRY_RUN)
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX)
    parser.add_argument("--width", type=int, default=0, help="Optional camera width; 0 uses calibration width.")
    parser.add_argument("--height", type=int, default=0, help="Optional camera height; 0 uses calibration height.")
    parser.add_argument("--balance", type=float, default=FISHEYE_BALANCE)
    parser.add_argument("--robot-marker-id", type=int, default=ROBOT_ARUCO_ID, help=argparse.SUPPRESS)
    parser.add_argument("--motor-module", default="motor_control", help=argparse.SUPPRESS)
    parser.add_argument("--robot-host", default=ROBOT_HOST, help=argparse.SUPPRESS)
    parser.add_argument("--robot-port", type=int, default=ROBOT_PORT, help=argparse.SUPPRESS)
    parser.add_argument(
        "--robot-speed-command",
        default=ROBOT_SPEED_COMMAND,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--require-motors", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-speed", type=float, default=MAX_SAFE_SPEED)
    parser.add_argument("--kp-distance", type=float, default=KP_DISTANCE)
    parser.add_argument("--kp-turn", type=float, default=KP_TURN)
    parser.add_argument("--lookahead-cm", type=float, default=LOOKAHEAD_CM)
    parser.add_argument(
        "--heading-offset-deg",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


class RobotSocketMotorClient:
    """TCP client matching robot code/pc.py command/response framing."""

    def __init__(self, host: str, port: int, speed_command: str, timeout_s: float = 1.0) -> None:
        self.speed_command = speed_command
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout_s)
        self.sock.connect((host, port))
        response = self.send("ping")
        if response != "pong":
            self.close()
            raise RuntimeError(f"Robot server ping failed: {response}")

    def send(self, command: str) -> str:
        self.sock.sendall((command + "\n").encode("utf-8"))
        return self.sock.recv(1024).decode("utf-8").strip()

    def set_motors(self, left_speed: float, right_speed: float) -> None:
        if abs(left_speed) < 1e-6 and abs(right_speed) < 1e-6:
            response = self.send("stop")
        else:
            response = self.send(f"{self.speed_command} {left_speed:.2f} {right_speed:.2f}")
        if response.startswith("error:"):
            raise RuntimeError(f"Robot rejected motor command: {response}")

    def close(self) -> None:
        self.sock.close()


def load_motor_setter(module_name: str, require_motors: bool) -> MotorSetter:
    """Import a project motor module or return a safe dry-run setter."""
    try:
        module = importlib.import_module(module_name)
        set_motors = getattr(module, "set_motors")
    except (ImportError, AttributeError) as exc:
        if require_motors:
            raise RuntimeError(f"Could not import {module_name}.set_motors") from exc

        print(
            f"warning: {module_name}.set_motors not found; running in dry-run motor mode",
            file=sys.stderr,
        )

        def dry_run_set_motors(left_speed: float, right_speed: float) -> None:
            return None

        return dry_run_set_motors

    if not callable(set_motors):
        raise RuntimeError(f"{module_name}.set_motors exists but is not callable")
    return set_motors


def default_hsv_params(frame_size: tuple[int, int]) -> dict[str, object]:
    """Mirror detector defaults without requiring interactive trackbars."""
    width, height = frame_size
    return {
        "red_1": HSVRange(np.array([0, 196, 60], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8)),
        "red_2": HSVRange(np.array([165, 196, 60], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
        "white": HSVRange(np.array([0, 0, 170], dtype=np.uint8), np.array([179, 70, 255], dtype=np.uint8)),
        "orange": HSVRange(np.array([15, 120, 120], dtype=np.uint8), np.array([28, 255, 255], dtype=np.uint8)),
        "red_min_area": 400.0,
        "ball_min_area": 157.0,
        "ball_max_area": 2500.0,
        "ball_min_circularity": 0.70,
        "h_cam_cm": 179.0,
        "z_calib_cm": 7.0,
        "camera_center_x": float(width // 2),
        "camera_center_y": float(height // 2),
    }


def load_body_tuning(path: Path) -> RobotBodyTuning:
    tuning = RobotBodyTuning()
    if not path.exists():
        return tuning
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return tuning

    tuning.length_cm = float(data.get("length_cm", tuning.length_cm))
    tuning.width_cm = float(data.get("width_cm", tuning.width_cm))
    tuning.heading_offset_deg = float(data.get("heading_offset_deg", tuning.heading_offset_deg))
    tuning.length_cm = float(np.clip(tuning.length_cm, 5.0, 100.0))
    tuning.width_cm = float(np.clip(tuning.width_cm, 5.0, 80.0))
    tuning.heading_offset_deg = float(np.clip(tuning.heading_offset_deg, -180.0, 180.0))
    tuning.last_saved = (tuning.length_cm, tuning.width_cm, tuning.heading_offset_deg)
    return tuning


def save_body_tuning(path: Path, tuning: RobotBodyTuning) -> None:
    data = {
        "length_cm": round(float(tuning.length_cm), 3),
        "width_cm": round(float(tuning.width_cm), 3),
        "heading_offset_deg": round(float(tuning.heading_offset_deg), 3),
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tuning.last_saved = (tuning.length_cm, tuning.width_cm, tuning.heading_offset_deg)
    tuning.last_save_time = time.perf_counter()


def noop(_value: int) -> None:
    return None


def create_body_tuning_sliders(tuning: RobotBodyTuning) -> None:
    cv2.namedWindow(TUNING_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TUNING_WINDOW_NAME, 420, 180)
    cv2.createTrackbar("Body length cm", TUNING_WINDOW_NAME, int(round(tuning.length_cm)), 100, noop)
    cv2.createTrackbar("Body width cm", TUNING_WINDOW_NAME, int(round(tuning.width_cm)), 80, noop)
    heading_value = int(round(np.clip(tuning.heading_offset_deg + 180.0, 0.0, 360.0)))
    cv2.createTrackbar("Heading deg +180", TUNING_WINDOW_NAME, heading_value, 360, noop)


def read_body_tuning_sliders(tuning: RobotBodyTuning) -> None:
    tuning.length_cm = float(np.clip(cv2.getTrackbarPos("Body length cm", TUNING_WINDOW_NAME), 5, 100))
    tuning.width_cm = float(np.clip(cv2.getTrackbarPos("Body width cm", TUNING_WINDOW_NAME), 5, 80))
    tuning.heading_offset_deg = float(cv2.getTrackbarPos("Heading deg +180", TUNING_WINDOW_NAME) - 180)


def autosave_body_tuning(path: Path, tuning: RobotBodyTuning) -> None:
    current = (tuning.length_cm, tuning.width_cm, tuning.heading_offset_deg)
    if tuning.last_saved == current:
        return
    now = time.perf_counter()
    if now - tuning.last_save_time < 0.5:
        return
    save_body_tuning(path, tuning)


def load_undistort_maps(calibration_file: Path, balance: float) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    data = np.load(str(calibration_file))
    camera_matrix = data["K"]
    dist_coeffs = data["D"]
    image_size = tuple(int(value) for value in data["image_size"])
    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix,
        dist_coeffs,
        image_size,
        np.eye(3, dtype=np.float64),
        balance=balance,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        np.eye(3, dtype=np.float64),
        new_camera_matrix,
        image_size,
        cv2.CV_32FC1,
    )
    return map1, map2, image_size


def load_homography(homography_file: Path) -> tuple[np.ndarray, tuple[int, int]]:
    data = np.load(str(homography_file))
    homography = data["homography"].astype(np.float64)
    size = tuple(int(value) for value in data["topdown_size"])
    return homography, size


def load_robot_calibration(calibration_file: Path, marker_id: int) -> RobotMarkerCalibration:
    with calibration_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    marker = data.get("markers", {}).get(str(marker_id))
    if marker is None:
        raise KeyError(f"robot_calibration.json has no marker entry for ID {marker_id}")
    topdown_size = tuple(int(value) for value in data.get("topdown_size", TOPDOWN_WARP_SIZE))
    return RobotMarkerCalibration(
        dx_px=float(marker["dx"]),
        dy_px=float(marker["dy"]),
        alpha_rad=float(marker["alpha_rad"]),
        marker_height_cm=float(data.get("marker_height_cm", 0.0)),
        camera_height_cm=float(data.get("camera_height_cm", 0.0)),
        topdown_size=topdown_size,
    )


def normalize_angle(angle_rad: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def image_yaw_rotation_matrix(angle_rad: float) -> np.ndarray:
    """Match robot_origin_calibration.py yaw rotation in top-down image coordinates."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=np.float32)


def marker_yaw_topdown_px(corners_px: np.ndarray) -> float:
    """Return ArUco yaw measured from +Y in the top-down image coordinate system."""
    pts = corners_px.reshape(4, 2).astype(np.float32)
    top_mid = (pts[0] + pts[1]) * 0.5
    bottom_mid = (pts[2] + pts[3]) * 0.5
    marker_forward = bottom_mid - top_mid
    return float(math.atan2(marker_forward[0], marker_forward[1]))


def parallax_correct_topdown_point(
    point_px: np.ndarray,
    camera_ground_point_px: np.ndarray,
    marker_height_cm: float,
    camera_height_cm: float,
) -> np.ndarray:
    """Apply the same marker-height correction used when robot_calibration.json is saved."""
    if marker_height_cm <= 0.0 or camera_height_cm <= marker_height_cm:
        return point_px.astype(np.float32)
    scale = (camera_height_cm - marker_height_cm) / camera_height_cm
    return (camera_ground_point_px + (point_px - camera_ground_point_px) * scale).astype(np.float32)


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.astype(np.float32).reshape(-1, 1, 2), homography).reshape(-1, 2)


def topdown_px_to_field_cm_float(point_px: tuple[float, float], topdown_size: tuple[int, int]) -> tuple[float, float]:
    width, height = topdown_size
    x_cm = float(point_px[0]) * FIELD_WIDTH_CM / max(1, width - 1)
    y_cm = FIELD_HEIGHT_CM - (float(point_px[1]) * FIELD_HEIGHT_CM / max(1, height - 1))
    return float(np.clip(x_cm, 0.0, FIELD_WIDTH_CM)), float(np.clip(y_cm, 0.0, FIELD_HEIGHT_CM))


def field_cm_to_topdown_px(point_cm: tuple[float, float], topdown_size: tuple[int, int]) -> tuple[int, int]:
    width, height = topdown_size
    x_px = int(round(point_cm[0] * (width - 1) / FIELD_WIDTH_CM))
    y_px = int(round((FIELD_HEIGHT_CM - point_cm[1]) * (height - 1) / FIELD_HEIGHT_CM))
    return int(np.clip(x_px, 0, width - 1)), int(np.clip(y_px, 0, height - 1))


def detect_robot_pose(
    undistorted_bgr: np.ndarray,
    homography: np.ndarray,
    topdown_size: tuple[int, int],
    marker_id: int,
    robot_calibration: RobotMarkerCalibration,
    camera_ground_point_px: np.ndarray,
    heading_offset_rad: float,
    aruco_dictionary: object,
    aruco_detector: object,
) -> tuple[RobotPose | None, list[np.ndarray], np.ndarray | None]:
    corners, ids = detect_aruco_markers(undistorted_bgr, aruco_dictionary, aruco_detector)
    if ids is None:
        return None, corners, ids

    for detected_id, marker_corners in zip(ids.flatten().tolist(), corners):
        if detected_id != marker_id:
            continue

        topdown_corners = transform_points(marker_corners.reshape(4, 2), homography)
        marker_center_px = np.mean(topdown_corners.astype(np.float32), axis=0)
        ground_center_px = parallax_correct_topdown_point(
            marker_center_px,
            camera_ground_point_px,
            robot_calibration.marker_height_cm,
            robot_calibration.camera_height_cm,
        )
        marker_yaw_rad = marker_yaw_topdown_px(topdown_corners)
        robot_yaw_image_rad = normalize_angle(marker_yaw_rad - robot_calibration.alpha_rad)
        offset_px = np.array([robot_calibration.dx_px, robot_calibration.dy_px], dtype=np.float32)
        origin_px = ground_center_px - image_yaw_rotation_matrix(robot_yaw_image_rad) @ offset_px
        robot_x, robot_y = topdown_px_to_field_cm_float(tuple(origin_px), topdown_size)
        robot_theta = normalize_angle(robot_yaw_image_rad - math.pi / 2.0 + heading_offset_rad)

        if not (math.isfinite(robot_x) and math.isfinite(robot_y) and math.isfinite(robot_theta)):
            return None, corners, ids
        if not (0.0 <= robot_x <= FIELD_WIDTH_CM and 0.0 <= robot_y <= FIELD_HEIGHT_CM):
            return None, corners, ids
        return RobotPose(robot_x, robot_y, robot_theta, detected_id), corners, ids

    return None, corners, ids


def select_closest_ball(robot_pose: RobotPose, balls: list[BallDetection], frame_shape: tuple[int, int, int]) -> TargetBall | None:
    if not balls:
        return None
    source_height, source_width = frame_shape[:2]
    candidates: list[TargetBall] = []
    for ball in balls:
        x_cm, y_cm = pixel_to_field_cm(ball.corrected_center, (source_width, source_height))
        grid_node = field_metric_cm_to_grid_node((x_cm, y_cm))
        candidates.append(TargetBall(ball, x_cm, y_cm, grid_node))
    return min(candidates, key=lambda b: math.hypot(b.x_cm - robot_pose.x_cm, b.y_cm - robot_pose.y_cm))


def grid_path_to_field_cm(path: list[tuple[int, int]]) -> list[tuple[float, float]]:
    return [(float(x), FIELD_HEIGHT_CM - float(y)) for x, y in path]


def choose_lookahead(path_cm: list[tuple[float, float]], robot_pose: RobotPose, lookahead_cm: float) -> tuple[float, float]:
    if not path_cm:
        return robot_pose.x_cm, robot_pose.y_cm
    for point in path_cm:
        if math.hypot(point[0] - robot_pose.x_cm, point[1] - robot_pose.y_cm) >= lookahead_cm:
            return point
    return path_cm[-1]


def compute_control(
    robot_pose: RobotPose,
    path_cm: list[tuple[float, float]],
    kp_distance: float,
    kp_turn: float,
    max_speed: float,
    lookahead_cm: float,
) -> ControlCommand:
    waypoint = choose_lookahead(path_cm, robot_pose, lookahead_cm)
    dx = waypoint[0] - robot_pose.x_cm
    dy = waypoint[1] - robot_pose.y_cm
    distance_error = math.hypot(dx, dy)
    target_heading = math.atan2(dy, dx)
    heading_error = normalize_angle(target_heading - robot_pose.theta_rad)

    base_speed = float(np.clip(kp_distance * distance_error, 0.0, max_speed))
    if abs(heading_error) > MAX_HEADING_FOR_FORWARD_RAD:
        base_speed = 0.0
    turn_speed = float(np.clip(kp_turn * heading_error, -max_speed, max_speed))
    left_motor = float(np.clip(base_speed - turn_speed, -max_speed, max_speed))
    right_motor = float(np.clip(base_speed + turn_speed, -max_speed, max_speed))
    return ControlCommand(left_motor, right_motor, distance_error, heading_error, waypoint)


def clearance_cm(occupancy_grid: np.ndarray, robot_node: tuple[int, int]) -> float:
    free_mask = (occupancy_grid == 0).astype(np.uint8)
    dist = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3)
    x, y = robot_node
    if not (0 <= y < dist.shape[0] and 0 <= x < dist.shape[1]):
        return 0.0
    return float(dist[y, x])


def draw_robot(
    overlay: np.ndarray,
    pose: RobotPose,
    topdown_size: tuple[int, int],
    body_tuning: RobotBodyTuning,
) -> None:
    center_px = field_cm_to_topdown_px((pose.x_cm, pose.y_cm), topdown_size)
    heading_end = (
        pose.x_cm + math.cos(pose.theta_rad) * ROBOT_RADIUS_CM,
        pose.y_cm + math.sin(pose.theta_rad) * ROBOT_RADIUS_CM,
    )
    heading_px = field_cm_to_topdown_px(heading_end, topdown_size)
    cv2.circle(overlay, center_px, 6, (255, 0, 255), -1, cv2.LINE_AA)
    cv2.line(overlay, center_px, heading_px, (255, 0, 255), 3, cv2.LINE_AA)

    half_l = body_tuning.length_cm / 2.0
    half_w = body_tuning.width_cm / 2.0
    local = np.array([(half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)])
    cos_t = math.cos(pose.theta_rad)
    sin_t = math.sin(pose.theta_rad)
    corners_cm = []
    for lx, ly in local:
        corners_cm.append((pose.x_cm + cos_t * lx - sin_t * ly, pose.y_cm + sin_t * lx + cos_t * ly))
    corners_px = np.array([field_cm_to_topdown_px(point, topdown_size) for point in corners_cm], dtype=np.int32)
    cv2.polylines(overlay, [corners_px.reshape(-1, 1, 2)], True, (255, 0, 255), 2, cv2.LINE_AA)


def draw_path_debug(
    overlay: np.ndarray,
    path_cm: list[tuple[float, float]],
    target: TargetBall | None,
    command: ControlCommand | None,
    topdown_size: tuple[int, int],
) -> None:
    if len(path_cm) >= 2:
        path_px = np.array([field_cm_to_topdown_px(point, topdown_size) for point in path_cm], dtype=np.int32)
        cv2.polylines(overlay, [path_px.reshape(-1, 1, 2)], False, (0, 255, 255), 2, cv2.LINE_AA)
    if target is not None:
        cv2.circle(overlay, field_cm_to_topdown_px((target.x_cm, target.y_cm), topdown_size), 8, (0, 255, 0), 2, cv2.LINE_AA)
    if command is not None:
        cv2.circle(overlay, field_cm_to_topdown_px(command.waypoint_cm, topdown_size), 7, (255, 255, 0), -1, cv2.LINE_AA)


def draw_status(
    overlay: np.ndarray,
    state: str,
    fps: float,
    command: ControlCommand | None,
    motor: SafeMotorInterface,
    body_tuning: RobotBodyTuning,
    dry_run: bool,
) -> None:
    left, right = motor.last_command
    lines = [
        f"State: {state}",
        f"Mode: {'DRY' if dry_run else 'DRIVE'}",
        f"FPS: {fps:.1f}",
        f"Motors L/R: {left:.1f} {right:.1f}",
        f"Body L/W/rot: {body_tuning.length_cm:.0f} {body_tuning.width_cm:.0f} {body_tuning.heading_offset_deg:.0f} deg",
    ]
    if command is not None:
        lines.append(f"Heading err: {math.degrees(command.heading_error_rad):.1f} deg")
        lines.append(f"Distance err: {command.distance_error_cm:.1f} cm")
    for index, line in enumerate(lines):
        cv2.putText(
            overlay,
            line,
            (18, 30 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def configure_camera(cap: cv2.VideoCapture, width: int, height: int, calibration_size: tuple[int, int]) -> None:
    target_width = width if width > 0 else calibration_size[0]
    target_height = height if height > 0 else calibration_size[1]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)


def run() -> int:
    args = parse_args()
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(CALIBRATION_FILE)
    if not HOMOGRAPHY_FILE.exists():
        raise FileNotFoundError(HOMOGRAPHY_FILE)
    if not ROBOT_CALIBRATION_FILE.exists():
        raise FileNotFoundError(ROBOT_CALIBRATION_FILE)

    map1, map2, calibration_size = load_undistort_maps(CALIBRATION_FILE, args.balance)
    homography, topdown_size = load_homography(HOMOGRAPHY_FILE)
    robot_calibration = load_robot_calibration(ROBOT_CALIBRATION_FILE, args.robot_marker_id)
    if robot_calibration.topdown_size != topdown_size:
        raise RuntimeError(
            f"robot_calibration.json topdown_size {robot_calibration.topdown_size} "
            f"does not match homography topdown_size {topdown_size}"
        )
    aruco_dictionary, aruco_detector = build_aruco_detector()
    if aruco_dictionary is None or aruco_detector is None:
        raise RuntimeError("OpenCV ArUco support is required for localization")

    body_tuning = load_body_tuning(NAVIGATOR_TUNING_FILE)
    if args.heading_offset_deg is not None:
        body_tuning.heading_offset_deg = float(args.heading_offset_deg)
    if not args.no_display:
        create_body_tuning_sliders(body_tuning)

    socket_client: RobotSocketMotorClient | None = None
    if args.dry_run:
        def dry_run_set_motors(left_speed: float, right_speed: float) -> None:
            return None

        motor_setter = dry_run_set_motors
    elif args.robot_host and args.robot_host != "192.168.0.0":
        socket_client = RobotSocketMotorClient(args.robot_host, args.robot_port, args.robot_speed_command)
        motor_setter = socket_client.set_motors
    else:
        raise RuntimeError("Set ROBOT_HOST at the top of autonomous_navigator.py before running with --drive.")
    motor = SafeMotorInterface(motor_setter, args.max_speed)
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera_index}")
    configure_camera(cap, args.width, args.height, calibration_size)

    params = default_hsv_params(topdown_size)
    camera_ground_point_px = transform_points(
        np.array([[calibration_size[0] / 2.0, calibration_size[1] / 2.0]], dtype=np.float32),
        homography,
    )[0].astype(np.float32)
    min_period_s = 1.0 / max(1.0, float(args.control_hz))
    last_frame_time = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                motor.stop()
                print("warning: camera read failed", file=sys.stderr)
                time.sleep(0.1)
                continue
            if (frame.shape[1], frame.shape[0]) != calibration_size:
                motor.stop()
                raise RuntimeError(
                    f"Camera frame size {(frame.shape[1], frame.shape[0])} does not match calibration {calibration_size}"
                )

            undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            topdown = cv2.warpPerspective(undistorted, homography, topdown_size)
            camera_center_pixels = (float(params["camera_center_x"]), float(params["camera_center_y"]))
            if not args.no_display:
                read_body_tuning_sliders(body_tuning)
                autosave_body_tuning(NAVIGATOR_TUNING_FILE, body_tuning)
            heading_offset_rad = math.radians(body_tuning.heading_offset_deg)

            pose, aruco_corners, aruco_ids = detect_robot_pose(
                undistorted,
                homography,
                topdown_size,
                args.robot_marker_id,
                robot_calibration,
                camera_ground_point_px,
                heading_offset_rad,
                aruco_dictionary,
                aruco_detector,
            )

            red_zones, red_mask = detect_red_zones(topdown, params, camera_center_pixels)
            white_balls, orange_balls, ball_masks = detect_balls(topdown, params, camera_center_pixels)
            all_balls = white_balls + orange_balls
            path_cm: list[tuple[float, float]] = []
            command: ControlCommand | None = None
            target: TargetBall | None = None
            state = "SEARCHING"

            if pose is None:
                motor.stop()
                state = "STOP: marker lost"
            else:
                occupancy_grid = build_occupancy_grid(topdown.shape, red_zones)
                robot_node = field_metric_cm_to_grid_node((pose.x_cm, pose.y_cm))
                red_clearance_cm = clearance_cm(occupancy_grid, robot_node)
                if red_clearance_cm <= MIN_RED_CLEARANCE_CM:
                    motor.stop()
                    state = "STOP: red-zone clearance"
                else:
                    target = select_closest_ball(pose, all_balls, topdown.shape)
                    if target is None:
                        motor.stop()
                        state = "STOP: no ball"
                    else:
                        raw_path = a_star_search(occupancy_grid, robot_node, target.grid_node)
                        if not raw_path:
                            motor.stop()
                            state = "STOP: path blocked"
                        else:
                            path_cm = grid_path_to_field_cm(raw_path)
                            distance_to_goal = math.hypot(target.x_cm - pose.x_cm, target.y_cm - pose.y_cm)
                            if distance_to_goal <= GOAL_STOP_DISTANCE_CM:
                                motor.stop()
                                state = "ARRIVED"
                            else:
                                command = compute_control(
                                    pose,
                                    path_cm,
                                    args.kp_distance,
                                    args.kp_turn,
                                    args.max_speed,
                                    args.lookahead_cm,
                                )
                                motor.set(command.left, command.right)
                                state = "DRIVING"

            now = time.perf_counter()
            fps = 1.0 / max(1e-6, now - last_frame_time)
            last_frame_time = now

            if not args.no_display:
                overlay = topdown.copy()
                for zone in red_zones:
                    cv2.drawContours(overlay, [zone.corrected_contour], -1, (0, 0, 255), 2, cv2.LINE_AA)
                for ball in white_balls:
                    cv2.circle(overlay, ball.corrected_center, ball.radius_px, (255, 255, 255), 2, cv2.LINE_AA)
                for ball in orange_balls:
                    cv2.circle(overlay, ball.corrected_center, ball.radius_px, (0, 140, 255), 2, cv2.LINE_AA)
                if pose is not None:
                    draw_robot(overlay, pose, topdown_size, body_tuning)
                draw_path_debug(overlay, path_cm, target, command, topdown_size)
                draw_status(overlay, state, fps, command, motor, body_tuning, args.dry_run)

                cv2.imshow(WINDOW_NAME, overlay)
                masks = np.hstack(
                    (
                        cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(ball_masks["white"], cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(ball_masks["orange"], cv2.COLOR_GRAY2BGR),
                    )
                )
                cv2.imshow(MASK_WINDOW_NAME, masks)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            elapsed = time.perf_counter() - loop_start
            if elapsed < min_period_s:
                time.sleep(min_period_s - elapsed)
    finally:
        motor.stop()
        if not args.no_display:
            save_body_tuning(NAVIGATOR_TUNING_FILE, body_tuning)
        if socket_client is not None:
            socket_client.close()
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
