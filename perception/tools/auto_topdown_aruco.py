#!/usr/bin/env python3
"""Automatic live top-down view using four ArUco markers after undistortion."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.camera.imageprocessing import undistort_with_calibration


CALIBRATION_FILE = REPO_ROOT / "calibration_data.npz"
CAMERA_INDEX = 1
LIVE_WINDOW = "Live Feed (Debug)"
TOPDOWN_WINDOW = "Top-Down View"
REQUIRED_MARKER_IDS = (0, 1, 2, 3)
MARKER_LABELS = {
    0: "Top-Left",
    1: "Top-Right",
    2: "Bottom-Right",
    3: "Bottom-Left",
}

# Measure these center-to-center distances on the real table and update them
# once the ArUco markers are permanently placed with the jig.
MARKER_DIST_X_CM = 120.0
MARKER_DIST_Y_CM = 160.0
EDGE_OFFSET_CM = 10.0
PIXELS_PER_CM = 10.0

OFFSET_PX = int(round(EDGE_OFFSET_CM * PIXELS_PER_CM))
INNER_W_PX = int(round(MARKER_DIST_X_CM * PIXELS_PER_CM))
INNER_H_PX = int(round(MARKER_DIST_Y_CM * PIXELS_PER_CM))
TOTAL_WIDTH = INNER_W_PX + (2 * OFFSET_PX)
TOTAL_HEIGHT = INNER_H_PX + (2 * OFFSET_PX)
TARGET_RES = (TOTAL_WIDTH, TOTAL_HEIGHT)
PLACEHOLDER_RES = TARGET_RES


def load_calibration_image_size(calibration_file: Path) -> tuple[int, int]:
    data = np.load(str(calibration_file))
    return tuple(int(v) for v in data["image_size"])


def configure_camera(cap: cv2.VideoCapture, frame_size: tuple[int, int]) -> None:
    width, height = frame_size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def build_aruco_detector() -> tuple[object, object]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    detector = None
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    return dictionary, parameters if detector is None else detector


def detect_markers(
    frame: np.ndarray,
    dictionary: object,
    detector_or_parameters: object,
) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
    if hasattr(detector_or_parameters, "detectMarkers"):
        corners, ids, rejected = detector_or_parameters.detectMarkers(frame)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, dictionary, parameters=detector_or_parameters)
    return corners, ids, rejected


def destination_points() -> np.ndarray:
    return np.array(
        [
            [OFFSET_PX, OFFSET_PX],
            [OFFSET_PX + INNER_W_PX, OFFSET_PX],
            [OFFSET_PX + INNER_W_PX, OFFSET_PX + INNER_H_PX],
            [OFFSET_PX, OFFSET_PX + INNER_H_PX],
        ],
        dtype=np.float32,
    )


def marker_center(corners: np.ndarray) -> np.ndarray:
    return corners.reshape(4, 2).mean(axis=0).astype(np.float32)


def extract_required_centers(corners: list[np.ndarray], ids: np.ndarray | None) -> dict[int, np.ndarray]:
    centers: dict[int, np.ndarray] = {}
    if ids is None:
        return centers

    for marker_id, marker_corners in zip(ids.flatten().tolist(), corners):
        if marker_id in REQUIRED_MARKER_IDS:
            centers[marker_id] = marker_center(marker_corners)
    return centers


def build_src_points(marker_centers: dict[int, np.ndarray]) -> np.ndarray | None:
    if not all(marker_id in marker_centers for marker_id in REQUIRED_MARKER_IDS):
        return None

    return np.array(
        [
            marker_centers[0],
            marker_centers[1],
            marker_centers[2],
            marker_centers[3],
        ],
        dtype=np.float32,
    )


def compute_transform(src_points: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(src_points, destination_points())


def draw_marker_centers(frame: np.ndarray, marker_centers: dict[int, np.ndarray]) -> np.ndarray:
    overlay = frame.copy()
    for marker_id in REQUIRED_MARKER_IDS:
        if marker_id not in marker_centers:
            continue

        center = marker_centers[marker_id]
        point = (int(round(center[0])), int(round(center[1])))
        cv2.circle(overlay, point, 6, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"ID {marker_id} {MARKER_LABELS[marker_id]}",
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def add_debug_status(
    frame: np.ndarray,
    marker_centers: dict[int, np.ndarray],
    has_cached_matrix: bool,
    fps: float,
) -> np.ndarray:
    missing = [str(marker_id) for marker_id in REQUIRED_MARKER_IDS if marker_id not in marker_centers]
    marker_text = "Markers: 0 1 2 3 locked" if not missing else f"Missing markers: {', '.join(missing)}"
    matrix_text = "Transform: cached" if has_cached_matrix else "Transform: waiting for all 4 markers"
    resolution_text = f"Top-down size: {TOTAL_WIDTH}x{TOTAL_HEIGHT}"
    fps_text = f"FPS: {fps:.1f}"

    lines = [marker_text, matrix_text, resolution_text, fps_text]
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (16, 30 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255) if index < 3 else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return frame


def make_topdown_placeholder(message: str, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        placeholder,
        message,
        (40, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return placeholder


def main() -> int:
    cv2.ocl.setUseOpenCL(False)

    if not CALIBRATION_FILE.exists():
        print(f"Calibration file not found: {CALIBRATION_FILE}", file=sys.stderr)
        return 1

    calibration_size = load_calibration_image_size(CALIBRATION_FILE)
    dictionary, detector_or_parameters = build_aruco_detector()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}", file=sys.stderr)
        return 1

    configure_camera(cap, calibration_size)
    cv2.namedWindow(LIVE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(TOPDOWN_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TOPDOWN_WINDOW, TOTAL_WIDTH, TOTAL_HEIGHT)

    cached_transform: np.ndarray | None = None
    last_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Could not read frame from camera", file=sys.stderr)
                return 1

            try:
                undistorted = undistort_with_calibration(frame, str(CALIBRATION_FILE))
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 1

            corners, ids, _rejected = detect_markers(undistorted, dictionary, detector_or_parameters)
            marker_centers = extract_required_centers(corners, ids)
            src_points = build_src_points(marker_centers)

            if src_points is not None:
                cached_transform = compute_transform(src_points)

            debug_frame = undistorted.copy()
            if ids is not None and len(corners) > 0:
                cv2.aruco.drawDetectedMarkers(debug_frame, corners, ids)
            debug_frame = draw_marker_centers(debug_frame, marker_centers)

            now = time.perf_counter()
            dt = now - last_time
            if dt > 0:
                fps = 1.0 / dt
            last_time = now
            debug_frame = add_debug_status(debug_frame, marker_centers, cached_transform is not None, fps)

            if cached_transform is not None:
                top_down_frame = cv2.warpPerspective(undistorted, cached_transform, TARGET_RES)
            else:
                top_down_frame = make_topdown_placeholder("Waiting for 4 ArUco markers...", PLACEHOLDER_RES)

            cv2.imshow(LIVE_WINDOW, debug_frame)
            cv2.imshow(TOPDOWN_WINDOW, top_down_frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
