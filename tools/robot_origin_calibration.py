#!/usr/bin/env python3
"""Calibrate a skid-steer robot origin from one or two ArUco markers.

The script follows the required geometry order:
undistort -> perspective transform -> ArUco detection -> calibration math.
It uses only OpenCV, NumPy, and the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_FILE = REPO_ROOT / "calibration_data.npz"
DEFAULT_OUTPUT_FILE = REPO_ROOT / "robot_calibration.json"
DEFAULT_HOMOGRAPHY_FILE = REPO_ROOT / "robot_topdown_homography.npz"

LIVE_WINDOW = "Robot Origin Calibration - Undistorted"
TOPDOWN_WINDOW = "Robot Origin Calibration - Top Down"
TOPDOWN_SIZE = (1000, 800)
MARKER_HEIGHT_CM = 9.0
CAMERA_HEIGHT_CM = 100.0
MIN_POINTS_PER_MARKER = 20
ELLIPSE_WARNING_RATIO = 1.12
EXPOSURE_VALUE = -6.0


@dataclass
class MarkerObservation:
    marker_id: int
    center: np.ndarray
    ground_center: np.ndarray
    corners: np.ndarray
    yaw_rad: float


@dataclass
class HomographySelection:
    points: list[tuple[int, int]] = field(default_factory=list)
    cursor: tuple[int, int] = (0, 0)
    frame_size: tuple[int, int] = (0, 0)
    matrix: np.ndarray | None = None

    def reset(self) -> None:
        self.points.clear()
        self.matrix = None


@dataclass
class RuntimeState:
    phase: str = "validation"
    collecting: bool = False
    waiting_for_alignment: bool = False
    collected_points: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    fitted_centers: dict[int, tuple[float, float]] = field(default_factory=dict)
    ellipse_ratios: dict[int, float] = field(default_factory=dict)
    calibration: dict[str, Any] | None = None
    warning: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate robot center of rotation from ArUco marker spin data.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--calibration-file", type=Path, default=DEFAULT_CALIBRATION_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--homography-file", type=Path, default=DEFAULT_HOMOGRAPHY_FILE)
    parser.add_argument("--marker-ids", type=int, nargs="+", required=True, help="One or two marker IDs.")
    parser.add_argument("--aruco-dict", default="DICT_4X4_50")
    parser.add_argument("--topdown-width", type=int, default=TOPDOWN_SIZE[0])
    parser.add_argument("--topdown-height", type=int, default=TOPDOWN_SIZE[1])
    parser.add_argument("--marker-height-cm", type=float, default=MARKER_HEIGHT_CM)
    parser.add_argument("--camera-height-cm", type=float, default=CAMERA_HEIGHT_CM)
    parser.add_argument("--camera-ground-x", type=float, default=None)
    parser.add_argument("--camera-ground-y", type=float, default=None)
    parser.add_argument("--min-points", type=int, default=MIN_POINTS_PER_MARKER)
    parser.add_argument("--ellipse-warning-ratio", type=float, default=ELLIPSE_WARNING_RATIO)
    parser.add_argument("--exposure", type=float, default=EXPOSURE_VALUE)
    return parser.parse_args()


def validate_marker_ids(marker_ids: list[int]) -> tuple[int, ...]:
    unique_ids = tuple(dict.fromkeys(marker_ids))
    if len(unique_ids) not in (1, 2):
        raise ValueError("Use exactly one or two marker IDs.")
    return unique_ids


def load_calibration(calibration_file: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int], bool]:
    data = np.load(str(calibration_file))
    if "mtx" in data and "dist" in data:
        matrix = data["mtx"]
        dist = data["dist"]
        fisheye = False
    elif "K" in data and "D" in data:
        matrix = data["K"]
        dist = data["D"]
        fisheye = True
    else:
        raise ValueError("Calibration file must contain either mtx/dist or K/D arrays.")

    if "image_size" in data:
        image_size = tuple(int(v) for v in data["image_size"])
    else:
        image_size = (0, 0)
    return matrix, dist, image_size, fisheye


def build_undistort_maps(
    camera_matrix: np.ndarray,
    dist: np.ndarray,
    image_size: tuple[int, int],
    fisheye: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if fisheye:
        new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix,
            dist,
            image_size,
            np.eye(3, dtype=np.float64),
            balance=0.0,
        )
        return cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            dist,
            np.eye(3, dtype=np.float64),
            new_matrix,
            image_size,
            cv2.CV_32FC1,
        )

    new_matrix, _roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, image_size, 0)
    return cv2.initUndistortRectifyMap(
        camera_matrix,
        dist,
        None,
        new_matrix,
        image_size,
        cv2.CV_32FC1,
    )


def configure_camera(cap: cv2.VideoCapture, image_size: tuple[int, int], exposure: float) -> None:
    if image_size != (0, 0):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[1])

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)


def aruco_dictionary(name: str) -> object:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("This OpenCV build does not include cv2.aruco.")
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def build_aruco_detector(dictionary: object) -> tuple[object, object]:
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        return dictionary, cv2.aruco.ArucoDetector(dictionary, parameters)
    return dictionary, parameters


def detect_markers(frame: np.ndarray, dictionary: object, detector_or_parameters: object) -> tuple[list[np.ndarray], np.ndarray | None]:
    if hasattr(detector_or_parameters, "detectMarkers"):
        corners, ids, _rejected = detector_or_parameters.detectMarkers(frame)
    else:
        corners, ids, _rejected = cv2.aruco.detectMarkers(frame, dictionary, parameters=detector_or_parameters)
    return corners, ids


def order_points(points: list[tuple[int, int]]) -> np.ndarray:
    corners = np.array(points, dtype=np.float32)
    sums = corners.sum(axis=1)
    diffs = np.diff(corners, axis=1).reshape(4)
    return np.array(
        [
            corners[np.argmin(sums)],
            corners[np.argmin(diffs)],
            corners[np.argmax(sums)],
            corners[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def destination_points(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def build_homography(points: list[tuple[int, int]], size: tuple[int, int]) -> np.ndarray:
    return cv2.getPerspectiveTransform(order_points(points), destination_points(size))


def save_homography(path: Path, matrix: np.ndarray, size: tuple[int, int]) -> None:
    np.savez(str(path), homography=matrix, topdown_size=np.array(size, dtype=np.int32))


def load_homography(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.load(str(path))
    matrix = data["homography"]
    stored_size = tuple(int(v) for v in data.get("topdown_size", np.array(size)))
    if stored_size != size:
        print(
            f"Ignoring homography with size {stored_size}; current top-down size is {size}.",
            file=sys.stderr,
        )
        return None
    return matrix


def on_mouse(event: int, x: int, y: int, _flags: int, state: HomographySelection) -> None:
    width, height = state.frame_size
    if width > 0 and height > 0:
        state.cursor = (int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1)))

    if event == cv2.EVENT_RBUTTONDOWN:
        state.reset()
        return
    if event != cv2.EVENT_LBUTTONDOWN or len(state.points) >= 4:
        return

    state.points.append(state.cursor)


def parallax_correct_point(
    point: np.ndarray,
    camera_ground_point: np.ndarray,
    marker_height_cm: float,
    camera_height_cm: float,
) -> np.ndarray:
    if marker_height_cm <= 0.0:
        return point.astype(np.float32)
    if camera_height_cm <= marker_height_cm:
        return point.astype(np.float32)

    scale = (camera_height_cm - marker_height_cm) / camera_height_cm
    return (camera_ground_point + (point - camera_ground_point) * scale).astype(np.float32)


def transform_point(point: tuple[float, float], matrix: np.ndarray) -> np.ndarray:
    src = np.array([[[point[0], point[1]]]], dtype=np.float32)
    return cv2.perspectiveTransform(src, matrix).reshape(2).astype(np.float32)


def marker_yaw(corners: np.ndarray) -> float:
    pts = corners.reshape(4, 2).astype(np.float32)
    top_mid = (pts[0] + pts[1]) * 0.5
    bottom_mid = (pts[2] + pts[3]) * 0.5
    marker_forward = bottom_mid - top_mid
    return float(math.atan2(marker_forward[0], marker_forward[1]))


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def extract_observations(
    frame: np.ndarray,
    dictionary: object,
    detector_or_parameters: object,
    marker_ids: tuple[int, ...],
    camera_ground_point: np.ndarray,
    marker_height_cm: float,
    camera_height_cm: float,
) -> dict[int, MarkerObservation]:
    corners, ids = detect_markers(frame, dictionary, detector_or_parameters)
    observations: dict[int, MarkerObservation] = {}
    if ids is None:
        return observations

    wanted = set(marker_ids)
    for marker_corners, raw_id in zip(corners, ids.flatten().tolist()):
        marker_id = int(raw_id)
        if marker_id not in wanted:
            continue
        pts = marker_corners.reshape(4, 2).astype(np.float32)
        center = pts.mean(axis=0).astype(np.float32)
        ground_center = parallax_correct_point(center, camera_ground_point, marker_height_cm, camera_height_cm)
        observations[marker_id] = MarkerObservation(
            marker_id=marker_id,
            center=center,
            ground_center=ground_center,
            corners=pts,
            yaw_rad=marker_yaw(pts),
        )
    return observations


def fit_circle(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    (x, y), radius = cv2.minEnclosingCircle(pts)
    return float(x), float(y), float(radius)


def ellipse_ratio(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 5:
        return None
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    _center, axes, _angle = cv2.fitEllipse(pts)
    minor = max(1e-6, float(min(axes)))
    major = float(max(axes))
    return major / minor


def image_yaw_rotation_matrix(angle: float) -> np.ndarray:
    """Rotate top-down image vectors for yaw measured from +Y with Y pointing down."""
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, s], [-s, c]], dtype=np.float32)


def robot_origin_from_observation(observation: MarkerObservation, marker_calibration: dict[str, float]) -> np.ndarray:
    offset = np.array([marker_calibration["dx"], marker_calibration["dy"]], dtype=np.float32)
    delta_angle = normalize_angle(observation.yaw_rad - float(marker_calibration["alpha_rad"]))
    rotated_offset = image_yaw_rotation_matrix(delta_angle) @ offset
    return observation.ground_center - rotated_offset


def load_robot_calibration(path: Path, marker_ids: tuple[int, ...]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)

    markers = calibration.get("markers", {})
    if all(str(marker_id) in markers for marker_id in marker_ids):
        return calibration

    print(f"Existing calibration file is missing requested marker IDs: {marker_ids}", file=sys.stderr)
    return None


def save_robot_calibration(
    path: Path,
    marker_ids: tuple[int, ...],
    fitted_centers: dict[int, tuple[float, float]],
    observations: dict[int, MarkerObservation],
    ellipse_ratios: dict[int, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    markers: dict[str, dict[str, float]] = {}
    for marker_id in marker_ids:
        observation = observations[marker_id]
        origin = fitted_centers[marker_id]
        dx = float(observation.ground_center[0] - origin[0])
        dy = float(observation.ground_center[1] - origin[1])
        alpha_rad = float(observation.yaw_rad)
        markers[str(marker_id)] = {
            "dx": dx,
            "dy": dy,
            "alpha_rad": alpha_rad,
            "alpha_deg": math.degrees(alpha_rad),
            "origin_x": float(origin[0]),
            "origin_y": float(origin[1]),
            "ellipse_ratio": float(ellipse_ratios.get(marker_id, 0.0)),
        }

    calibration = {
        "version": 1,
        "created_unix": time.time(),
        "marker_ids": list(marker_ids),
        "marker_height_cm": float(args.marker_height_cm),
        "camera_height_cm": float(args.camera_height_cm),
        "topdown_size": [int(args.topdown_width), int(args.topdown_height)],
        "markers": markers,
    }
    path.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")
    return calibration


def draw_homography_selection(frame: np.ndarray, state: HomographySelection) -> np.ndarray:
    view = frame.copy()
    for index, point in enumerate(state.points, start=1):
        cv2.circle(view, point, 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(view, str(index), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if len(state.points) >= 2:
        cv2.polylines(view, [np.array(state.points, dtype=np.int32)], False, (0, 255, 0), 2, cv2.LINE_AA)
    lines = [
        "Select 4 arena corners in order-free clicks",
        "Left click: add point | Right click/r: reset | q: quit",
        f"Points: {len(state.points)}/4",
    ]
    draw_text_block(view, lines, (16, 30), (0, 255, 255))
    return view


def draw_marker_overlay(frame: np.ndarray, observations: dict[int, MarkerObservation]) -> None:
    for observation in observations.values():
        pts = observation.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        raw_center = tuple(int(round(v)) for v in observation.center)
        ground_center = tuple(int(round(v)) for v in observation.ground_center)
        cv2.circle(frame, raw_center, 4, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.drawMarker(frame, ground_center, (255, 255, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        if np.linalg.norm(observation.ground_center - observation.center) > 1.0:
            cv2.line(frame, raw_center, ground_center, (255, 255, 0), 1, cv2.LINE_AA)
        axis_len = 45
        end = (
            int(round(raw_center[0] + math.sin(observation.yaw_rad) * axis_len)),
            int(round(raw_center[1] + math.cos(observation.yaw_rad) * axis_len)),
        )
        cv2.arrowedLine(frame, raw_center, end, (255, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(frame, f"ID {observation.marker_id}", (raw_center[0] + 8, raw_center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_path_overlay(frame: np.ndarray, collected_points: dict[int, list[tuple[float, float]]]) -> None:
    colors = [(0, 180, 255), (255, 180, 0)]
    for index, (marker_id, points) in enumerate(collected_points.items()):
        if len(points) < 2:
            continue
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, colors[index % len(colors)], 2, cv2.LINE_AA)
        cv2.putText(frame, f"Path ID {marker_id}: {len(points)}", tuple(pts[-1, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[index % len(colors)], 2)


def draw_robot_origin(frame: np.ndarray, origin: np.ndarray, color: tuple[int, int, int] = (0, 0, 255)) -> None:
    x = int(round(origin[0]))
    y = int(round(origin[1]))
    cv2.circle(frame, (x, y), 14, color, 2, cv2.LINE_AA)
    cv2.line(frame, (x - 22, y), (x + 22, y), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y - 22), (x, y + 22), color, 2, cv2.LINE_AA)
    cv2.putText(frame, "ROBOT ORIGIN", (x + 18, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_text_block(
    frame: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    x, y = origin
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + index * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.64, color, 2, cv2.LINE_AA)


def calibration_lines(state: RuntimeState, marker_ids: tuple[int, ...], fps: float) -> list[str]:
    counts = ", ".join(f"ID {marker_id}: {len(state.collected_points.get(marker_id, []))}" for marker_id in marker_ids)
    if state.collecting:
        mode = "COLLECTING SPIN DATA"
        action = "Spin robot 360 deg on the spot, then press s"
    elif state.waiting_for_alignment:
        mode = "ALIGNMENT"
        action = "Face robot forward along +Y, then press Enter"
    else:
        mode = "LIVE VALIDATION"
        action = "c: collect spin | q: quit"
    return [
        f"FPS: {fps:.1f}",
        f"Mode: {mode}",
        action,
        f"Points: {counts}",
        "Blue dot: raw marker center | Yellow cross: parallax-corrected tracking point",
    ]


def compute_fitted_centers(state: RuntimeState, marker_ids: tuple[int, ...], min_points: int, warning_ratio: float) -> bool:
    state.fitted_centers.clear()
    state.ellipse_ratios.clear()
    warnings = []
    for marker_id in marker_ids:
        points = state.collected_points.get(marker_id, [])
        if len(points) < min_points:
            state.warning = f"Need at least {min_points} points for ID {marker_id}; got {len(points)}."
            return False
        xc, yc, _radius = fit_circle(points)
        state.fitted_centers[marker_id] = (xc, yc)
        ratio = ellipse_ratio(points)
        if ratio is not None:
            state.ellipse_ratios[marker_id] = ratio
            if ratio > warning_ratio:
                warnings.append(f"ID {marker_id} ellipse ratio {ratio:.2f}")
    state.warning = "WARNING: Elliptical spin path. Check floor friction or homography." if warnings else ""
    return True


def main() -> int:
    args = parse_args()
    try:
        marker_ids = validate_marker_ids(args.marker_ids)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    topdown_size = (int(args.topdown_width), int(args.topdown_height))
    camera_ground_point: np.ndarray | None = None
    if args.camera_ground_x is not None and args.camera_ground_y is not None:
        camera_ground_point = np.array([args.camera_ground_x, args.camera_ground_y], dtype=np.float32)

    cv2.ocl.setUseOpenCL(False)
    if not args.calibration_file.exists():
        print(f"Calibration file not found: {args.calibration_file}", file=sys.stderr)
        return 1

    try:
        camera_matrix, dist, calibration_size, fisheye = load_calibration(args.calibration_file)
        dictionary = aruco_dictionary(args.aruco_dict)
        dictionary, detector_or_parameters = build_aruco_detector(dictionary)
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera_index}", file=sys.stderr)
        return 1
    configure_camera(cap, calibration_size, args.exposure)

    homography_state = HomographySelection(matrix=load_homography(args.homography_file, topdown_size))
    runtime = RuntimeState(collected_points={marker_id: [] for marker_id in marker_ids})
    runtime.calibration = load_robot_calibration(args.output, marker_ids)

    cv2.namedWindow(LIVE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(TOPDOWN_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(LIVE_WINDOW, on_mouse, homography_state)

    map1: np.ndarray | None = None
    map2: np.ndarray | None = None
    fps = 0.0
    last_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Could not read frame from camera", file=sys.stderr)
                return 1

            frame_size = (int(frame.shape[1]), int(frame.shape[0]))
            if calibration_size == (0, 0):
                calibration_size = frame_size
            if frame_size != calibration_size:
                print(f"Camera frame {frame_size} does not match calibration size {calibration_size}.", file=sys.stderr)
                return 1
            if map1 is None or map2 is None:
                map1, map2 = build_undistort_maps(camera_matrix, dist, frame_size, fisheye)

            undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            homography_state.frame_size = (undistorted.shape[1], undistorted.shape[0])

            now = time.perf_counter()
            dt = now - last_time
            if dt > 0:
                fps = 1.0 / dt
            last_time = now

            if homography_state.matrix is None:
                live_view = draw_homography_selection(undistorted, homography_state)
                topdown = np.zeros((topdown_size[1], topdown_size[0], 3), dtype=np.uint8)
                draw_text_block(topdown, ["Waiting for 4 clicked arena corners"], (30, 50), (0, 0, 255))
                cv2.imshow(LIVE_WINDOW, live_view)
                cv2.imshow(TOPDOWN_WINDOW, topdown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    homography_state.reset()
                if len(homography_state.points) == 4:
                    homography_state.matrix = build_homography(homography_state.points, topdown_size)
                    save_homography(args.homography_file, homography_state.matrix, topdown_size)
                continue

            if camera_ground_point is None:
                camera_ground_point = transform_point(
                    (undistorted.shape[1] / 2.0, undistorted.shape[0] / 2.0),
                    homography_state.matrix,
                )

            topdown = cv2.warpPerspective(undistorted, homography_state.matrix, topdown_size)
            observations = extract_observations(
                topdown,
                dictionary,
                detector_or_parameters,
                marker_ids,
                camera_ground_point,
                args.marker_height_cm,
                args.camera_height_cm,
            )

            if runtime.collecting:
                for marker_id, observation in observations.items():
                    runtime.collected_points[marker_id].append(
                        (float(observation.ground_center[0]), float(observation.ground_center[1]))
                    )

            debug = topdown.copy()
            draw_marker_overlay(debug, observations)
            draw_path_overlay(debug, runtime.collected_points)

            if runtime.waiting_for_alignment:
                for marker_id, center in runtime.fitted_centers.items():
                    draw_robot_origin(debug, np.array(center, dtype=np.float32), (0, 165, 255))
            elif runtime.calibration is not None:
                origins = []
                for marker_id, observation in observations.items():
                    marker_config = runtime.calibration["markers"].get(str(marker_id))
                    if marker_config is not None:
                        origins.append(robot_origin_from_observation(observation, marker_config))
                if origins:
                    draw_robot_origin(debug, np.mean(np.array(origins, dtype=np.float32), axis=0))

            lines = calibration_lines(runtime, marker_ids, fps)
            if runtime.calibration is None and not runtime.collecting and not runtime.waiting_for_alignment:
                lines.append("No robot calibration loaded. Press c to start.")
            if runtime.warning:
                lines.append(runtime.warning)
            draw_text_block(debug, lines, (16, 30), (0, 255, 255))

            cv2.imshow(LIVE_WINDOW, undistorted)
            cv2.imshow(TOPDOWN_WINDOW, debug)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                runtime.collecting = True
                runtime.waiting_for_alignment = False
                runtime.phase = "collecting"
                runtime.warning = ""
                runtime.calibration = None
                runtime.collected_points = {marker_id: [] for marker_id in marker_ids}
                runtime.fitted_centers.clear()
                runtime.ellipse_ratios.clear()
            elif key == ord("s") and runtime.collecting:
                runtime.collecting = False
                if compute_fitted_centers(runtime, marker_ids, args.min_points, args.ellipse_warning_ratio):
                    runtime.waiting_for_alignment = True
                    runtime.phase = "alignment"
            elif key in (10, 13) and runtime.waiting_for_alignment:
                missing = [marker_id for marker_id in marker_ids if marker_id not in observations]
                if missing:
                    runtime.warning = f"Waiting for alignment marker(s): {missing}"
                else:
                    runtime.calibration = save_robot_calibration(
                        args.output,
                        marker_ids,
                        runtime.fitted_centers,
                        observations,
                        runtime.ellipse_ratios,
                        args,
                    )
                    runtime.waiting_for_alignment = False
                    runtime.phase = "validation"
                    print(f"Saved robot calibration to {args.output}")
            elif key == ord("r"):
                homography_state.reset()

        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
