#!/usr/bin/env python3
"""Detect red zones, white balls, and one orange ball in a top-down arena view.

This tool supports three input modes:
1. Still image input for repeatable offline tuning.
2. Live camera input for on-table tuning with HSV trackbars.
3. Recorded video input for replaying real camera runs through the live pipeline.

The output is shown as:
- Left: annotated top-down camera frame
- Right: synthetic 2D schematic of the measured field

The script intentionally keeps the detection pipeline simple and deterministic:
- HSV thresholding
- Morphology cleanup
- Contour extraction
- Circularity filtering for ball-like objects
"""

from __future__ import annotations

import argparse
from collections import deque
from enum import Enum
import heapq
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camera.imageprocessing import undistort_with_calibration


FIELD_WIDTH_CM = 167.0
FIELD_HEIGHT_CM = 121.5
FIELD_GRID_WIDTH_CM = int(round(FIELD_WIDTH_CM))
FIELD_GRID_HEIGHT_CM = int(round(FIELD_HEIGHT_CM))
Z_BALL_CM = 2.0
Z_FLOOR_CM = 0.0
CALIBRATION_FILE = REPO_ROOT / "calibration_data.npz"
DEFAULT_IMAGE = REPO_ROOT / "test_topdown.png"
DEFAULT_VIDEO_DIR = REPO_ROOT / "videos"
WINDOW_NAME = "Top-Down Detector"
MASK_WINDOW_NAME = "Segmentation Masks"
CONTROL_COLOR_WINDOW_NAME = "HSV Controls - Colors"
CONTROL_FILTER_WINDOW_NAME = "HSV Controls - Filters"
CONTROL_GEOMETRY_WINDOW_NAME = "HSV Controls - Geometry"
MANUAL_SELECTOR_WINDOW_NAME = "Manual Top-Down Selector"
CONTROL_WINDOW_SIZE = (420, 520)
TRACKBAR_NAMES = {
    "red1_h_min": "R1 H min",
    "red1_h_max": "R1 H max",
    "red2_h_min": "R2 H min",
    "red2_h_max": "R2 H max",
    "red_s_min": "R S min",
    "red_s_max": "R S max",
    "red_v_min": "R V min",
    "red_v_max": "R V max",
    "white_h_min": "W H min",
    "white_h_max": "W H max",
    "white_s_min": "W S min",
    "white_s_max": "W S max",
    "white_v_min": "W V min",
    "white_v_max": "W V max",
    "orange_h_min": "O H min",
    "orange_h_max": "O H max",
    "orange_s_min": "O S min",
    "orange_s_max": "O S max",
    "orange_v_min": "O V min",
    "orange_v_max": "O V max",
    "red_min_area": "R min area",
    "ball_min_area": "B min area",
    "ball_max_area": "B max area",
    "ball_min_circ": "B min circ",
    "cam_height_cm": "Cam h cm",
    "calib_z_cm": "Border h cm",
    "cam_center_x": "Cam C X",
    "cam_center_y": "Cam C Y",
}
TRACKBAR_WINDOWS = {
    "red1_h_min": CONTROL_COLOR_WINDOW_NAME,
    "red1_h_max": CONTROL_COLOR_WINDOW_NAME,
    "red2_h_min": CONTROL_COLOR_WINDOW_NAME,
    "red2_h_max": CONTROL_COLOR_WINDOW_NAME,
    "red_s_min": CONTROL_COLOR_WINDOW_NAME,
    "red_s_max": CONTROL_COLOR_WINDOW_NAME,
    "red_v_min": CONTROL_COLOR_WINDOW_NAME,
    "red_v_max": CONTROL_COLOR_WINDOW_NAME,
    "white_h_min": CONTROL_COLOR_WINDOW_NAME,
    "white_h_max": CONTROL_COLOR_WINDOW_NAME,
    "white_s_min": CONTROL_COLOR_WINDOW_NAME,
    "white_s_max": CONTROL_COLOR_WINDOW_NAME,
    "white_v_min": CONTROL_COLOR_WINDOW_NAME,
    "white_v_max": CONTROL_COLOR_WINDOW_NAME,
    "orange_h_min": CONTROL_FILTER_WINDOW_NAME,
    "orange_h_max": CONTROL_FILTER_WINDOW_NAME,
    "orange_s_min": CONTROL_FILTER_WINDOW_NAME,
    "orange_s_max": CONTROL_FILTER_WINDOW_NAME,
    "orange_v_min": CONTROL_FILTER_WINDOW_NAME,
    "orange_v_max": CONTROL_FILTER_WINDOW_NAME,
    "red_min_area": CONTROL_FILTER_WINDOW_NAME,
    "ball_min_area": CONTROL_FILTER_WINDOW_NAME,
    "ball_max_area": CONTROL_FILTER_WINDOW_NAME,
    "ball_min_circ": CONTROL_FILTER_WINDOW_NAME,
    "cam_height_cm": CONTROL_GEOMETRY_WINDOW_NAME,
    "calib_z_cm": CONTROL_GEOMETRY_WINDOW_NAME,
    "cam_center_x": CONTROL_GEOMETRY_WINDOW_NAME,
    "cam_center_y": CONTROL_GEOMETRY_WINDOW_NAME,
}

# Still-image mode is the safest default for deterministic tuning.
USE_LIVE_FEED = False
CAMERA_INDEX = 0
TOPDOWN_WARP_SIZE = (800, 600)
LOUPE_CROP_SIZE = 40
LOUPE_SCALE = 5
LOUPE_PADDING = 12
POINT_RADIUS = 6
REQUIRED_ARUCO_IDS = (0, 1, 2, 3)
WALL_THICKNESS_CM = 1.6
MARKER_OUTER_OFFSET_CM = 8.0

# Schematic sizing is kept fixed for the detector UI while coordinate math uses the measured field size.
SCHEMATIC_WIDTH_PX = 900
SCHEMATIC_HEIGHT_PX = 600
SCHEMATIC_WINDOW_NAME = "2D Schematic"
ROBOT_RADIUS_CM = 15


class CalibrationState(Enum):
    """Current top-down calibration mode."""

    NEEDS_CALIBRATION = "needs_calibration"
    CALIBRATED_AUTO = "calibrated_auto"
    CALIBRATING_MANUAL = "calibrating_manual"
    CALIBRATED_MANUAL = "calibrated_manual"


@dataclass(frozen=True)
class BallDetection:
    """Ball-like object found in the frame."""

    label: str
    center: tuple[int, int]
    corrected_center: tuple[int, int]
    radius_px: int
    contour: np.ndarray
    area: float
    circularity: float


@dataclass(frozen=True)
class RedZoneDetection:
    """Detected red avoidance geometry."""

    contour: np.ndarray
    corrected_contour: np.ndarray
    bounding_box: tuple[int, int, int, int]
    center: tuple[int, int]
    corrected_center: tuple[int, int]
    area: float


@dataclass(frozen=True)
class HSVRange:
    """Single HSV threshold range."""

    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class SmoothedBallCoordinate:
    """Smoothed field coordinate paired with the detection that produced it."""

    track_id: int
    label: str
    center_px: tuple[int, int]
    corrected_center_px: tuple[int, int]
    radius_px: int
    cm_x: float
    cm_y: float


@dataclass
class SmoothedCoordinateTrack:
    """Persistent smoothing state for one detected ball."""

    label: str
    x_cm: float
    y_cm: float
    history_x_cm: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    history_y_cm: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    stationary_frames: int = 0
    missed_frames: int = 0


class BallCoordinateSmoother:
    """Smooth per-ball field coordinates with median filtering and stationary hold."""

    def __init__(
        self,
        alpha_stationary: float = 0.12,
        alpha_moving: float = 0.35,
        max_match_distance_cm: float = 10.0,
        max_missed_frames: int = 5,
        median_window_size: int = 5,
        stationary_deadband_cm: float = 0.75,
        moving_threshold_cm: float = 2.0,
        stationary_confirm_frames: int = 3,
    ) -> None:
        self.alpha_stationary = alpha_stationary
        self.alpha_moving = alpha_moving
        self.max_match_distance_cm = max_match_distance_cm
        self.max_missed_frames = max_missed_frames
        self.median_window_size = median_window_size
        self.stationary_deadband_cm = stationary_deadband_cm
        self.moving_threshold_cm = moving_threshold_cm
        self.stationary_confirm_frames = stationary_confirm_frames
        self.next_track_id = 0
        self.tracks: dict[int, SmoothedCoordinateTrack] = {}

    def reset(self) -> None:
        """Clear all smoothing state."""
        self.next_track_id = 0
        self.tracks.clear()

    def update(
        self,
        detections: list[BallDetection],
        frame_shape: tuple[int, int, int],
    ) -> list[SmoothedBallCoordinate]:
        """Update smoothing tracks from the current frame's detections."""
        source_height, source_width = frame_shape[:2]
        observations = [
            (
                index,
                detection,
                pixel_to_field_cm(
                    detection.corrected_center,
                    (source_width, source_height),
                ),
            )
            for index, detection in enumerate(detections)
        ]

        matched_tracks: set[int] = set()
        matched_observations: set[int] = set()
        smoothed_results: dict[int, SmoothedBallCoordinate] = {}

        labels = sorted({detection.label for detection in detections})
        for label in labels:
            label_track_ids = [track_id for track_id, track in self.tracks.items() if track.label == label]
            label_observations = [
                (index, detection, cm_point)
                for index, detection, cm_point in observations
                if detection.label == label
            ]
            candidate_pairs: list[tuple[float, int, int]] = []

            for track_id in label_track_ids:
                track = self.tracks[track_id]
                for observation_index, _detection, (raw_x_cm, raw_y_cm) in label_observations:
                    distance_cm = math.hypot(raw_x_cm - track.x_cm, raw_y_cm - track.y_cm)
                    candidate_pairs.append((distance_cm, track_id, observation_index))

            candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
            for distance_cm, track_id, observation_index in candidate_pairs:
                if distance_cm > self.max_match_distance_cm:
                    continue
                if track_id in matched_tracks or observation_index in matched_observations:
                    continue

                track = self.tracks[track_id]
                detection = detections[observation_index]
                raw_x_cm, raw_y_cm = pixel_to_field_cm(
                    detection.corrected_center,
                    (source_width, source_height),
                )
                if track.history_x_cm.maxlen != self.median_window_size:
                    track.history_x_cm = deque(track.history_x_cm, maxlen=self.median_window_size)
                if track.history_y_cm.maxlen != self.median_window_size:
                    track.history_y_cm = deque(track.history_y_cm, maxlen=self.median_window_size)

                track.history_x_cm.append(raw_x_cm)
                track.history_y_cm.append(raw_y_cm)
                median_x_cm = float(np.median(np.array(track.history_x_cm, dtype=np.float32)))
                median_y_cm = float(np.median(np.array(track.history_y_cm, dtype=np.float32)))

                delta_cm = math.hypot(median_x_cm - track.x_cm, median_y_cm - track.y_cm)
                if delta_cm < self.stationary_deadband_cm:
                    track.stationary_frames += 1
                else:
                    track.stationary_frames = 0

                if track.stationary_frames >= self.stationary_confirm_frames:
                    # Keep stationary balls visually stable until the change is meaningful.
                    smoothed_x_cm = track.x_cm
                    smoothed_y_cm = track.y_cm
                else:
                    alpha = self.alpha_moving if delta_cm >= self.moving_threshold_cm else self.alpha_stationary
                    smoothed_x_cm = alpha * median_x_cm + (1.0 - alpha) * track.x_cm
                    smoothed_y_cm = alpha * median_y_cm + (1.0 - alpha) * track.y_cm
                    track.x_cm = smoothed_x_cm
                    track.y_cm = smoothed_y_cm

                track.missed_frames = 0

                matched_tracks.add(track_id)
                matched_observations.add(observation_index)
                smoothed_results[observation_index] = SmoothedBallCoordinate(
                    track_id=track_id,
                    label=detection.label,
                    center_px=detection.center,
                    corrected_center_px=detection.corrected_center,
                    radius_px=detection.radius_px,
                    cm_x=smoothed_x_cm,
                    cm_y=smoothed_y_cm,
                )

        for observation_index, detection, (raw_x_cm, raw_y_cm) in observations:
            if observation_index in matched_observations:
                continue

            track_id = self.next_track_id
            self.next_track_id += 1
            new_track = SmoothedCoordinateTrack(
                label=detection.label,
                x_cm=raw_x_cm,
                y_cm=raw_y_cm,
            )
            new_track.history_x_cm.append(raw_x_cm)
            new_track.history_y_cm.append(raw_y_cm)
            self.tracks[track_id] = new_track
            matched_tracks.add(track_id)
            smoothed_results[observation_index] = SmoothedBallCoordinate(
                track_id=track_id,
                label=detection.label,
                center_px=detection.center,
                corrected_center_px=detection.corrected_center,
                radius_px=detection.radius_px,
                cm_x=raw_x_cm,
                cm_y=raw_y_cm,
            )

        stale_track_ids = []
        for track_id, track in self.tracks.items():
            if track_id in matched_tracks:
                continue
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            del self.tracks[track_id]

        return [smoothed_results[index] for index in range(len(detections))]


@dataclass
class AppState:
    """Mutable UI state used by the schematic click-to-plan interaction."""

    latest_frame_shape: tuple[int, int, int] | None = None
    latest_red_zones: list[RedZoneDetection] | None = None
    latest_white_balls: list[BallDetection] | None = None
    latest_orange_balls: list[BallDetection] | None = None
    latest_smoothed_ball_coordinates: list[SmoothedBallCoordinate] = field(default_factory=list)
    selected_ball_track_id: int | None = None
    selected_start_cm: tuple[int, int] | None = None
    route_points_cm: list[tuple[int, int]] | None = None
    coordinate_smoother: BallCoordinateSmoother = field(default_factory=BallCoordinateSmoother)


@dataclass
class TopdownSelectionState:
    """State for the manual 4-point top-down transform selector."""

    points: list[tuple[int, int]]
    cursor: tuple[int, int]
    frame_size: tuple[int, int]
    transform_matrix: np.ndarray | None = None
    calibration_state: CalibrationState = CalibrationState.NEEDS_CALIBRATION
    aruco_dictionary: object | None = None
    aruco_detector: object | None = None
    latest_aruco_centers: dict[int, np.ndarray] = field(default_factory=dict)
    aruco_available: bool = False

    def clear_points(self) -> None:
        self.points.clear()
        self.transform_matrix = None
        self.latest_aruco_centers.clear()
        if self.calibration_state in (CalibrationState.CALIBRATING_MANUAL, CalibrationState.CALIBRATED_MANUAL):
            self.calibration_state = CalibrationState.CALIBRATING_MANUAL
        else:
            self.calibration_state = CalibrationState.NEEDS_CALIBRATION

    def start_manual_calibration(self) -> None:
        self.points.clear()
        self.transform_matrix = None
        self.latest_aruco_centers.clear()
        self.calibration_state = CalibrationState.CALIBRATING_MANUAL

    def start_auto_calibration(self) -> None:
        self.points.clear()
        self.transform_matrix = None
        self.latest_aruco_centers.clear()
        self.calibration_state = CalibrationState.NEEDS_CALIBRATION


def noop(_value: int) -> None:
    """Trackbar callback placeholder."""
    return None


def order_points(points: list[tuple[int, int]]) -> np.ndarray:
    """Order 4 selected corners as top-left, top-right, bottom-right, bottom-left."""
    corners = np.array(points, dtype=np.float32)
    sums = corners.sum(axis=1)
    diffs = np.diff(corners, axis=1).reshape(4)

    top_left = corners[np.argmin(sums)]
    bottom_right = corners[np.argmax(sums)]
    top_right = corners[np.argmin(diffs)]
    bottom_left = corners[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def destination_corners(size: tuple[int, int]) -> np.ndarray:
    """Build the destination rectangle for the perspective warp."""
    width, height = size
    return np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )


def build_manual_topdown_transform(points: list[tuple[int, int]]) -> np.ndarray:
    """Compute the manual perspective transform from the 4 selected corners."""
    return cv2.getPerspectiveTransform(order_points(points), destination_corners(TOPDOWN_WARP_SIZE))


def build_aruco_detector() -> tuple[object | None, object | None]:
    """Create an ArUco detector that works across OpenCV versions."""
    if not hasattr(cv2, "aruco"):
        return None, None

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        return dictionary, cv2.aruco.ArucoDetector(dictionary, parameters)
    return dictionary, parameters


def detect_aruco_markers(
    frame: np.ndarray,
    dictionary: object,
    detector_or_parameters: object,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Detect ArUco markers in the already undistorted frame."""
    if hasattr(detector_or_parameters, "detectMarkers"):
        corners, ids, _rejected = detector_or_parameters.detectMarkers(frame)
    else:
        corners, ids, _rejected = cv2.aruco.detectMarkers(frame, dictionary, parameters=detector_or_parameters)
    return corners, ids


def marker_center(corners: np.ndarray) -> np.ndarray:
    """Compute the center of one detected marker from its four corners."""
    return corners.reshape(4, 2).mean(axis=0).astype(np.float32)


def extract_required_marker_centers(corners: list[np.ndarray], ids: np.ndarray | None) -> dict[int, np.ndarray]:
    """Keep only the marker centers needed for automatic homography calibration."""
    centers: dict[int, np.ndarray] = {}
    if ids is None:
        return centers

    for marker_id, marker_corners in zip(ids.flatten().tolist(), corners):
        if marker_id in REQUIRED_ARUCO_IDS:
            centers[marker_id] = marker_center(marker_corners)
    return centers


def field_cm_to_topdown_px(x_cm: float, y_cm: float) -> tuple[float, float]:
    """Map field/world coordinates in centimeters to the fixed top-down pixel plane."""
    width, height = TOPDOWN_WARP_SIZE
    scale_x = (width - 1) / FIELD_WIDTH_CM
    scale_y = (height - 1) / FIELD_HEIGHT_CM
    return x_cm * scale_x, y_cm * scale_y


def aruco_destination_points() -> np.ndarray:
    """Build the target marker-center positions in the cropped top-down pixel space."""
    total_offset_cm = WALL_THICKNESS_CM + MARKER_OUTER_OFFSET_CM
    world_points_cm = (
        (-total_offset_cm, -total_offset_cm),
        (FIELD_WIDTH_CM + total_offset_cm, -total_offset_cm),
        (FIELD_WIDTH_CM + total_offset_cm, FIELD_HEIGHT_CM + total_offset_cm),
        (-total_offset_cm, FIELD_HEIGHT_CM + total_offset_cm),
    )
    return np.array(
        [field_cm_to_topdown_px(x_cm, y_cm) for x_cm, y_cm in world_points_cm],
        dtype=np.float32,
    )


def topdown_field_corners() -> np.ndarray:
    """Return the playable field rectangle in the warped top-down pixel plane."""
    return destination_corners(TOPDOWN_WARP_SIZE)


def build_auto_topdown_transform(marker_centers: dict[int, np.ndarray]) -> np.ndarray | None:
    """Compute the perspective transform when all four required markers are visible."""
    if not all(marker_id in marker_centers for marker_id in REQUIRED_ARUCO_IDS):
        return None

    src_points = np.array(
        [
            marker_centers[0],
            marker_centers[1],
            marker_centers[3],
            marker_centers[2],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src_points, aruco_destination_points())


def clamp_crop_bounds(center: int, crop_size: int, limit: int) -> tuple[int, int]:
    """Clamp a crop region so the loupe stays inside the frame bounds."""
    if limit <= crop_size:
        return 0, limit

    half = crop_size // 2
    start = max(0, center - half)
    end = start + crop_size
    if end > limit:
        end = limit
        start = end - crop_size
    return start, end


def draw_loupe(frame: np.ndarray, state: TopdownSelectionState) -> np.ndarray:
    """Draw the zoomed cursor loupe used during manual corner selection."""
    overlay = frame.copy()
    height, width = overlay.shape[:2]
    crop_w = min(LOUPE_CROP_SIZE, width)
    crop_h = min(LOUPE_CROP_SIZE, height)

    x0, x1 = clamp_crop_bounds(state.cursor[0], crop_w, width)
    y0, y1 = clamp_crop_bounds(state.cursor[1], crop_h, height)
    crop = overlay[y0:y1, x0:x1]
    loupe = cv2.resize(
        crop,
        (crop.shape[1] * LOUPE_SCALE, crop.shape[0] * LOUPE_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )

    loupe_h, loupe_w = loupe.shape[:2]
    dest_x1 = width - LOUPE_PADDING
    dest_x0 = max(0, dest_x1 - loupe_w)
    dest_y0 = LOUPE_PADDING
    dest_y1 = min(height, dest_y0 + loupe_h)

    visible_loupe = loupe[: dest_y1 - dest_y0, : dest_x1 - dest_x0]
    overlay[dest_y0:dest_y1, dest_x0:dest_x1] = visible_loupe
    cv2.rectangle(overlay, (dest_x0, dest_y0), (dest_x1, dest_y1), (255, 255, 255), 2)

    center_x = dest_x0 + visible_loupe.shape[1] // 2
    center_y = dest_y0 + visible_loupe.shape[0] // 2
    cv2.line(overlay, (center_x, dest_y0), (center_x, dest_y1), (0, 255, 255), 1)
    cv2.line(overlay, (dest_x0, center_y), (dest_x1, center_y), (0, 255, 255), 1)
    cv2.putText(
        overlay,
        "Loupe",
        (dest_x0, max(20, dest_y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def draw_manual_selection_overlay(frame: np.ndarray, state: TopdownSelectionState) -> np.ndarray:
    """Render the manual top-down selection view with points, lines, and status."""
    overlay = draw_loupe(frame, state)

    for index, point in enumerate(state.points, start=1):
        cv2.circle(overlay, point, POINT_RADIUS, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, point, POINT_RADIUS + 4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(index),
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if len(state.points) >= 2:
        polyline = np.array(state.points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [polyline], False, (0, 255, 0), 2, cv2.LINE_AA)

    if len(state.points) == 4:
        ordered = order_points(state.points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [ordered], True, (255, 200, 0), 2, cv2.LINE_AA)

    help_lines = [
        f"Points: {len(state.points)}/4",
        f"Mode: {state.calibration_state.value}",
        "Left click: add point",
        "Right click or r: reset",
        "a: auto ArUco calibration",
        "m: manual calibration",
        "q: quit",
    ]
    for line_index, text in enumerate(help_lines):
        cv2.putText(
            overlay,
            text,
            (16, 30 + line_index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if state.transform_matrix is not None and state.calibration_state == CalibrationState.CALIBRATED_AUTO:
        status = "Top-down transform active (ArUco auto)"
        color = (0, 255, 0)
    elif state.transform_matrix is not None:
        status = "Top-down transform active (manual)"
        color = (0, 255, 0)
    elif state.calibration_state == CalibrationState.CALIBRATING_MANUAL:
        status = "Select 4 inner corners for manual top-down warp"
        color = (0, 165, 255)
    elif not state.aruco_available:
        status = "ArUco unavailable, press m for manual calibration"
        color = (0, 0, 255)
    else:
        missing = [str(marker_id) for marker_id in REQUIRED_ARUCO_IDS if marker_id not in state.latest_aruco_centers]
        status = "Scanning for ArUco markers 0,1,2,3" if not missing else f"Missing ArUco IDs: {', '.join(missing)}"
        color = (0, 165, 255)

    cv2.putText(
        overlay,
        status,
        (16, overlay.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )
    return overlay


def on_manual_topdown_mouse(
    event: int,
    x: int,
    y: int,
    _flags: int,
    param: TopdownSelectionState,
) -> None:
    """Handle point selection for the manual top-down warp."""
    width, height = param.frame_size
    if width > 0 and height > 0:
        clamped_x = int(np.clip(x, 0, width - 1))
        clamped_y = int(np.clip(y, 0, height - 1))
        param.cursor = (clamped_x, clamped_y)

    if event == cv2.EVENT_RBUTTONDOWN:
        param.clear_points()
        return

    if event != cv2.EVENT_LBUTTONDOWN or len(param.points) >= 4:
        return

    param.points.append(param.cursor)
    if len(param.points) == 4:
        param.transform_matrix = build_manual_topdown_transform(param.points)
        param.calibration_state = CalibrationState.CALIBRATED_MANUAL


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect red zones, white ping-pong balls, and one orange ball from a "
            "top-down arena image, live camera feed, or recorded camera video."
        )
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--live",
        action="store_true",
        help="Read frames from a live camera instead of a still image.",
    )
    mode_group.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Path to a top-down test image. Default: {DEFAULT_IMAGE}",
    )
    mode_group.add_argument(
        "--video",
        type=Path,
        help=f"Path to a recorded camera video. Store local recordings under {DEFAULT_VIDEO_DIR}.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=CAMERA_INDEX,
        help=f"OpenCV camera index for live mode. Default: {CAMERA_INDEX}",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.0,
        help="Fisheye undistortion balance passed to undistort_with_calibration().",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Optional live camera width. 0 keeps the driver default.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Optional live camera height. 0 keeps the driver default.",
    )
    parser.add_argument(
        "--resize-video-to-calibration",
        action="store_true",
        help=(
            "Video mode only: resize each recorded frame to calibration_data.npz size "
            "before undistortion if the recording resolution differs."
        ),
    )
    return parser.parse_args()


def load_calibration_image_size(calibration_file: Path) -> tuple[int, int]:
    """Read the calibration image size so live capture matches the saved model."""
    data = np.load(str(calibration_file))
    image_size = tuple(int(value) for value in data["image_size"])
    return image_size


def create_hsv_trackbars(frame_size: tuple[int, int]) -> None:
    """Create trackbars for the three color classes.

    Red uses two hue intervals because red wraps across the HSV hue boundary.
    """
    frame_width, frame_height = frame_size
    for window_name in (
        CONTROL_COLOR_WINDOW_NAME,
        CONTROL_FILTER_WINDOW_NAME,
        CONTROL_GEOMETRY_WINDOW_NAME,
    ):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, *CONTROL_WINDOW_SIZE)

    defaults = {
        "red1_h_min": 0,
        "red1_h_max": 12,
        "red2_h_min": 165,
        "red2_h_max": 179,
        "red_s_min": 196,
        "red_s_max": 255,
        "red_v_min": 60,
        "red_v_max": 255,
        "white_h_min": 0,
        "white_h_max": 179,
        "white_s_min": 0,
        "white_s_max": 70,
        "white_v_min": 170,
        "white_v_max": 255,
        "orange_h_min": 15,
        "orange_h_max": 28,
        "orange_s_min": 120,
        "orange_s_max": 255,
        "orange_v_min": 120,
        "orange_v_max": 255,
        "red_min_area": 400,
        "ball_min_area": 157,
        "ball_max_area": 2500,
        "ball_min_circ": 70,
        "cam_height_cm": 179,
        "calib_z_cm": 7,
        "cam_center_x": frame_width // 2,
        "cam_center_y": frame_height // 2,
    }

    for key, value in defaults.items():
        name = TRACKBAR_NAMES[key]
        window_name = TRACKBAR_WINDOWS[key]
        max_value = 179 if "_h_" in key else 255
        if key.endswith("min_area"):
            max_value = 20000
        if key == "ball_max_area":
            max_value = 20000
        if key == "ball_min_circ":
            max_value = 100
        if key == "cam_height_cm":
            max_value = 300
        if key == "calib_z_cm":
            max_value = 30
        if key == "cam_center_x":
            max_value = max(1, frame_width)
        if key == "cam_center_y":
            max_value = max(1, frame_height)
        cv2.createTrackbar(name, window_name, value, max_value, noop)


def get_trackbar_value(key: str) -> int:
    """Read a trackbar value from the window that owns it."""
    return cv2.getTrackbarPos(TRACKBAR_NAMES[key], TRACKBAR_WINDOWS[key])


def read_hsv_ranges() -> dict[str, object]:
    """Read current threshold parameters from the trackbars."""
    red_1 = HSVRange(
        lower=np.array(
            [
                get_trackbar_value("red1_h_min"),
                get_trackbar_value("red_s_min"),
                get_trackbar_value("red_v_min"),
            ],
            dtype=np.uint8,
        ),
        upper=np.array(
            [
                get_trackbar_value("red1_h_max"),
                get_trackbar_value("red_s_max"),
                get_trackbar_value("red_v_max"),
            ],
            dtype=np.uint8,
        ),
    )
    red_2 = HSVRange(
        lower=np.array(
            [
                get_trackbar_value("red2_h_min"),
                get_trackbar_value("red_s_min"),
                get_trackbar_value("red_v_min"),
            ],
            dtype=np.uint8,
        ),
        upper=np.array(
            [
                get_trackbar_value("red2_h_max"),
                get_trackbar_value("red_s_max"),
                get_trackbar_value("red_v_max"),
            ],
            dtype=np.uint8,
        ),
    )
    white = HSVRange(
        lower=np.array(
            [
                get_trackbar_value("white_h_min"),
                get_trackbar_value("white_s_min"),
                get_trackbar_value("white_v_min"),
            ],
            dtype=np.uint8,
        ),
        upper=np.array(
            [
                get_trackbar_value("white_h_max"),
                get_trackbar_value("white_s_max"),
                get_trackbar_value("white_v_max"),
            ],
            dtype=np.uint8,
        ),
    )
    orange = HSVRange(
        lower=np.array(
            [
                get_trackbar_value("orange_h_min"),
                get_trackbar_value("orange_s_min"),
                get_trackbar_value("orange_v_min"),
            ],
            dtype=np.uint8,
        ),
        upper=np.array(
            [
                get_trackbar_value("orange_h_max"),
                get_trackbar_value("orange_s_max"),
                get_trackbar_value("orange_v_max"),
            ],
            dtype=np.uint8,
        ),
    )

    return {
        "red_1": red_1,
        "red_2": red_2,
        "white": white,
        "orange": orange,
        "red_min_area": float(get_trackbar_value("red_min_area")),
        "ball_min_area": float(get_trackbar_value("ball_min_area")),
        "ball_max_area": float(get_trackbar_value("ball_max_area")),
        "ball_min_circularity": get_trackbar_value("ball_min_circ") / 100.0,
        "h_cam_cm": float(get_trackbar_value("cam_height_cm")),
        "z_calib_cm": float(get_trackbar_value("calib_z_cm")),
        "camera_center_x": float(get_trackbar_value("cam_center_x")),
        "camera_center_y": float(get_trackbar_value("cam_center_y")),
    }


def contour_center(contour: np.ndarray) -> tuple[int, int]:
    """Compute a contour centroid with a bounding-box fallback."""
    moments = cv2.moments(contour)
    if moments["m00"] <= 0.0:
        x, y, width, height = cv2.boundingRect(contour)
        return (x + width // 2, y + height // 2)
    return (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))


def cleanup_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply deterministic morphology cleanup to reduce single-pixel noise."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)


def correct_parallax(
    pixel_coord: tuple[int, int],
    z_object_cm: float,
    h_cam_cm: float,
    z_calib_cm: float,
    camera_center_pixels: tuple[float, float],
) -> tuple[int, int]:
    """Correct radial displacement caused by object height relative to the warp plane."""
    camera_center_x, camera_center_y = camera_center_pixels
    denominator = h_cam_cm - z_calib_cm
    if abs(denominator) < 1e-6:
        return int(round(pixel_coord[0])), int(round(pixel_coord[1]))
    t = (h_cam_cm - z_object_cm) / denominator

    x_real = camera_center_x + (pixel_coord[0] - camera_center_x) * t
    y_real = camera_center_y + (pixel_coord[1] - camera_center_y) * t
    return int(round(x_real)), int(round(y_real))


def correct_contour_parallax(
    contour: np.ndarray,
    z_object_cm: float,
    h_cam_cm: float,
    z_calib_cm: float,
    camera_center_pixels: tuple[float, float],
) -> np.ndarray:
    """Apply parallax correction point-by-point for contour-based schematic overlays."""
    corrected_points = [
        correct_parallax(
            pixel_coord=(int(point[0][0]), int(point[0][1])),
            z_object_cm=z_object_cm,
            h_cam_cm=h_cam_cm,
            z_calib_cm=z_calib_cm,
            camera_center_pixels=camera_center_pixels,
        )
        for point in contour
    ]
    return np.array(corrected_points, dtype=np.int32).reshape((-1, 1, 2))


def detect_red_zones(
    frame_bgr: np.ndarray,
    params: dict[str, object],
    camera_center_pixels: tuple[float, float],
) -> tuple[list[RedZoneDetection], np.ndarray]:
    """Detect red avoidance zones from the top-down image.

    Red wraps around the hue axis, so two masks are built and combined.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    red_1 = params["red_1"]
    red_2 = params["red_2"]
    mask_1 = cv2.inRange(hsv, red_1.lower, red_1.upper)
    mask_2 = cv2.inRange(hsv, red_2.lower, red_2.upper)
    combined = cv2.bitwise_or(mask_1, mask_2)
    combined = cleanup_mask(combined, kernel_size=5)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[RedZoneDetection] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(params["red_min_area"]):
            continue

        x, y, width, height = cv2.boundingRect(contour)
        center = contour_center(contour)
        detections.append(
            RedZoneDetection(
                contour=contour,
                corrected_contour=correct_contour_parallax(
                    contour=contour,
                    z_object_cm=Z_FLOOR_CM,
                    h_cam_cm=float(params["h_cam_cm"]),
                    z_calib_cm=float(params["z_calib_cm"]),
                    camera_center_pixels=camera_center_pixels,
                ),
                bounding_box=(x, y, width, height),
                center=center,
                corrected_center=correct_parallax(
                    pixel_coord=center,
                    z_object_cm=Z_FLOOR_CM,
                    h_cam_cm=float(params["h_cam_cm"]),
                    z_calib_cm=float(params["z_calib_cm"]),
                    camera_center_pixels=camera_center_pixels,
                ),
                area=area,
            )
        )

    return detections, combined


def contour_circularity(contour: np.ndarray) -> float:
    """Calculate circularity in the range [0, 1] for roughly ball-shaped contours."""
    perimeter = float(cv2.arcLength(contour, True))
    area = float(cv2.contourArea(contour))
    if perimeter <= 0.0 or area <= 0.0:
        return 0.0
    return (4.0 * np.pi * area) / (perimeter * perimeter)


def detect_ball_candidates(
    frame_bgr: np.ndarray,
    hsv_range: HSVRange,
    label: str,
    min_area: float,
    max_area: float,
    min_circularity: float,
    z_object_cm: float,
    h_cam_cm: float,
    z_calib_cm: float,
    camera_center_pixels: tuple[float, float],
) -> tuple[list[BallDetection], np.ndarray]:
    """Detect circular objects for one HSV color class."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_range.lower, hsv_range.upper)
    mask = cleanup_mask(mask, kernel_size=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[BallDetection] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        circularity = contour_circularity(contour)
        if circularity < min_circularity:
            continue

        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        if radius <= 0.0:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = width / float(height) if height > 0 else 0.0
        if abs(1.0 - aspect_ratio) > 0.35:
            continue

        detections.append(
            BallDetection(
                label=label,
                center=(int(center_x), int(center_y)),
                corrected_center=correct_parallax(
                    pixel_coord=(int(center_x), int(center_y)),
                    z_object_cm=z_object_cm,
                    h_cam_cm=h_cam_cm,
                    z_calib_cm=z_calib_cm,
                    camera_center_pixels=camera_center_pixels,
                ),
                radius_px=max(2, int(radius)),
                contour=contour,
                area=area,
                circularity=circularity,
            )
        )

    return detections, mask


def detect_balls(
    frame_bgr: np.ndarray,
    params: dict[str, object],
    camera_center_pixels: tuple[float, float],
) -> tuple[list[BallDetection], list[BallDetection], dict[str, np.ndarray]]:
    """Detect white balls and orange balls using the same contour filters and parallax correction."""
    white_detections, white_mask = detect_ball_candidates(
        frame_bgr=frame_bgr,
        hsv_range=params["white"],
        label="white",
        min_area=float(params["ball_min_area"]),
        max_area=float(params["ball_max_area"]),
        min_circularity=float(params["ball_min_circularity"]),
        z_object_cm=Z_BALL_CM,
        h_cam_cm=float(params["h_cam_cm"]),
        z_calib_cm=float(params["z_calib_cm"]),
        camera_center_pixels=camera_center_pixels,
    )
    orange_detections, orange_mask = detect_ball_candidates(
        frame_bgr=frame_bgr,
        hsv_range=params["orange"],
        label="orange",
        min_area=float(params["ball_min_area"]),
        max_area=float(params["ball_max_area"]),
        min_circularity=float(params["ball_min_circularity"]),
        z_object_cm=Z_BALL_CM,
        h_cam_cm=float(params["h_cam_cm"]),
        z_calib_cm=float(params["z_calib_cm"]),
        camera_center_pixels=camera_center_pixels,
    )

    masks = {
        "white": white_mask,
        "orange": orange_mask,
    }
    return white_detections, orange_detections, masks


def map_point_between_frames(
    point: tuple[int, int],
    source_size: tuple[int, int],
    destination_size: tuple[int, int],
) -> tuple[int, int]:
    """Map a point between image coordinate systems by simple linear scaling."""
    src_width, src_height = source_size
    dst_width, dst_height = destination_size
    x = int(point[0] * dst_width / max(1, src_width))
    y = int(point[1] * dst_height / max(1, src_height))
    return x, y


def pixel_to_field_cm(point: tuple[int, int], source_size: tuple[int, int]) -> tuple[float, float]:
    """Convert top-down pixel coordinates to field cm with origin at the bottom-left."""
    src_width, src_height = source_size
    x_cm = float(point[0]) * FIELD_WIDTH_CM / max(1, src_width - 1)
    y_cm = FIELD_HEIGHT_CM - (float(point[1]) * FIELD_HEIGHT_CM / max(1, src_height - 1))
    return (
        float(np.clip(x_cm, 0.0, FIELD_WIDTH_CM)),
        float(np.clip(y_cm, 0.0, FIELD_HEIGHT_CM)),
    )


def field_metric_cm_to_grid_node(point_cm: tuple[float, float]) -> tuple[int, int]:
    """Convert bottom-left metric coordinates to top-left grid nodes."""
    x_cm, y_cm = point_cm
    return (
        int(np.clip(round(x_cm), 0, FIELD_GRID_WIDTH_CM - 1)),
        int(np.clip(round(FIELD_HEIGHT_CM - y_cm), 0, FIELD_GRID_HEIGHT_CM - 1)),
    )


def field_metric_cm_to_schematic(point_cm: tuple[float, float]) -> tuple[int, int]:
    """Convert bottom-left metric coordinates to schematic pixels."""
    x_cm, y_cm = point_cm
    x_px = int(round(x_cm * (SCHEMATIC_WIDTH_PX - 1) / max(1.0, FIELD_WIDTH_CM)))
    y_px = int(round((FIELD_HEIGHT_CM - y_cm) * (SCHEMATIC_HEIGHT_PX - 1) / max(1.0, FIELD_HEIGHT_CM)))
    return (
        int(np.clip(x_px, 0, SCHEMATIC_WIDTH_PX - 1)),
        int(np.clip(y_px, 0, SCHEMATIC_HEIGHT_PX - 1)),
    )


def schematic_to_field_metric_cm(point_px: tuple[int, int]) -> tuple[float, float]:
    """Convert schematic pixels back to bottom-left metric coordinates."""
    x_px, y_px = point_px
    x_cm = float(x_px) * FIELD_WIDTH_CM / max(1, SCHEMATIC_WIDTH_PX - 1)
    y_cm = FIELD_HEIGHT_CM - (float(y_px) * FIELD_HEIGHT_CM / max(1, SCHEMATIC_HEIGHT_PX - 1))
    return (
        float(np.clip(x_cm, 0.0, FIELD_WIDTH_CM)),
        float(np.clip(y_cm, 0.0, FIELD_HEIGHT_CM)),
    )


def source_point_to_field_cm(point: tuple[int, int], source_size: tuple[int, int]) -> tuple[int, int]:
    """Map a source-frame pixel to a 1 cm occupancy-grid coordinate."""
    src_width, src_height = source_size
    x = int(round(point[0] * (FIELD_GRID_WIDTH_CM - 1) / max(1, src_width - 1)))
    y = int(round(point[1] * (FIELD_GRID_HEIGHT_CM - 1) / max(1, src_height - 1)))
    return (
        int(np.clip(x, 0, FIELD_GRID_WIDTH_CM - 1)),
        int(np.clip(y, 0, FIELD_GRID_HEIGHT_CM - 1)),
    )


def field_cm_to_schematic(point_cm: tuple[int, int]) -> tuple[int, int]:
    """Map a 1 cm grid coordinate to the schematic window."""
    return map_point_between_frames(
        point_cm,
        (FIELD_GRID_WIDTH_CM, FIELD_GRID_HEIGHT_CM),
        (SCHEMATIC_WIDTH_PX, SCHEMATIC_HEIGHT_PX),
    )


def contour_to_field_grid(contour: np.ndarray, source_size: tuple[int, int]) -> np.ndarray:
    """Convert a contour from source pixels to the 1 cm occupancy grid."""
    mapped_points = [
        source_point_to_field_cm((int(point[0][0]), int(point[0][1])), source_size)
        for point in contour
    ]
    return np.array(mapped_points, dtype=np.int32).reshape((-1, 1, 2))


def build_occupancy_grid(frame_shape: tuple[int, int, int], red_zones: list[RedZoneDetection]) -> np.ndarray:
    """Build a 1 cm binary occupancy grid with a dilated red-zone safety margin."""
    source_height, source_width = frame_shape[:2]
    grid = np.zeros((FIELD_GRID_HEIGHT_CM, FIELD_GRID_WIDTH_CM), dtype=np.uint8)

    for zone in red_zones:
        grid_contour = contour_to_field_grid(zone.corrected_contour, (source_width, source_height))
        cv2.fillPoly(grid, [grid_contour], 1)

    kernel_size = max(1, int(2 * ROBOT_RADIUS_CM + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(grid, kernel, iterations=1)
    return (dilated > 0).astype(np.uint8)


def a_star_search(grid: np.ndarray, start_node: tuple[int, int], goal_node: tuple[int, int]) -> list[tuple[int, int]]:
    """Run 8-connected A* search on a binary occupancy grid."""
    width = int(grid.shape[1])
    height = int(grid.shape[0])

    def in_bounds(node: tuple[int, int]) -> bool:
        return 0 <= node[0] < width and 0 <= node[1] < height

    def is_free(node: tuple[int, int]) -> bool:
        return grid[node[1], node[0]] == 0

    if not in_bounds(start_node) or not in_bounds(goal_node):
        return []
    if not is_free(start_node) or not is_free(goal_node):
        return []

    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]

    def heuristic(node: tuple[int, int], goal: tuple[int, int]) -> float:
        return math.hypot(goal[0] - node[0], goal[1] - node[1])

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start_node, goal_node), 0.0, start_node))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start_node: 0.0}

    while open_heap:
        _f_cost, current_cost, current = heapq.heappop(open_heap)
        if current == goal_node:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        if current_cost > g_score.get(current, float("inf")):
            continue

        for dx, dy, step_cost in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if not in_bounds(neighbor) or not is_free(neighbor):
                continue

            tentative_g = g_score[current] + step_cost
            if tentative_g >= g_score.get(neighbor, float("inf")):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            heapq.heappush(
                open_heap,
                (tentative_g + heuristic(neighbor, goal_node), tentative_g, neighbor),
            )

    return []


def build_greedy_route(
    grid: np.ndarray,
    ball_nodes_cm: list[tuple[int, int]],
    start_node_cm: tuple[int, int],
) -> list[tuple[int, int]]:
    """Greedily connect the start ball to the nearest reachable unvisited balls."""
    if not ball_nodes_cm:
        return []

    unvisited = list(ball_nodes_cm)
    current = min(unvisited, key=lambda node: math.hypot(node[0] - start_node_cm[0], node[1] - start_node_cm[1]))
    route: list[tuple[int, int]] = [current]
    unvisited.remove(current)

    while unvisited:
        candidate = min(unvisited, key=lambda node: math.hypot(node[0] - current[0], node[1] - current[1]))
        segment = a_star_search(grid, current, candidate)
        if not segment:
            unvisited.remove(candidate)
            continue

        route.extend(segment[1:])
        current = candidate
        unvisited.remove(candidate)

    return route


def update_route_from_state(app_state: AppState) -> None:
    """Recompute the path-planning route from the latest detections and selected start point."""
    if (
        app_state.latest_frame_shape is None
        or app_state.latest_red_zones is None
        or app_state.selected_ball_track_id is None
    ):
        app_state.route_points_cm = None
        return

    smoothed_balls = app_state.latest_smoothed_ball_coordinates
    if not smoothed_balls:
        app_state.route_points_cm = None
        return

    selected_ball = next(
        (ball for ball in smoothed_balls if ball.track_id == app_state.selected_ball_track_id),
        None,
    )
    if selected_ball is None:
        app_state.selected_start_cm = None
        app_state.route_points_cm = None
        return

    app_state.selected_start_cm = field_metric_cm_to_grid_node((selected_ball.cm_x, selected_ball.cm_y))
    ball_nodes_cm = [
        field_metric_cm_to_grid_node((ball.cm_x, ball.cm_y))
        for ball in smoothed_balls
    ]
    occupancy_grid = build_occupancy_grid(app_state.latest_frame_shape, app_state.latest_red_zones)
    app_state.route_points_cm = build_greedy_route(
        occupancy_grid,
        ball_nodes_cm,
        app_state.selected_start_cm,
    )


def on_schematic_mouse(event: int, x: int, y: int, _flags: int, userdata: AppState) -> None:
    """Handle left-click selection of the closest ball in the schematic window."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if userdata.latest_frame_shape is None:
        return

    all_balls = userdata.latest_smoothed_ball_coordinates
    if not all_balls:
        return

    click_cm = schematic_to_field_metric_cm((x, y))
    nearest_ball = min(
        all_balls,
        key=lambda ball: math.hypot(ball.cm_x - click_cm[0], ball.cm_y - click_cm[1]),
    )
    userdata.selected_ball_track_id = nearest_ball.track_id
    update_route_from_state(userdata)
def draw_schematic(
    frame_shape: tuple[int, int, int],
    red_zones: list[RedZoneDetection],
    smoothed_ball_coordinates: list[SmoothedBallCoordinate],
    camera_center_pixels: tuple[float, float],
    app_state: AppState,
) -> np.ndarray:
    """Draw a clean synthetic field view containing only the detected objects."""
    source_height, source_width = frame_shape[:2]
    schematic = np.full((SCHEMATIC_HEIGHT_PX, SCHEMATIC_WIDTH_PX, 3), (40, 100, 40), dtype=np.uint8)

    cv2.rectangle(
        schematic,
        (0, 0),
        (SCHEMATIC_WIDTH_PX - 1, SCHEMATIC_HEIGHT_PX - 1),
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )

    for zone in red_zones:
        mapped_contour = zone.corrected_contour.astype(np.float32).copy()
        mapped_contour[:, 0, 0] *= SCHEMATIC_WIDTH_PX / max(1, source_width)
        mapped_contour[:, 0, 1] *= SCHEMATIC_HEIGHT_PX / max(1, source_height)
        mapped_contour = mapped_contour.astype(np.int32)
        cv2.polylines(schematic, [mapped_contour], True, (0, 0, 255), 3, cv2.LINE_AA)

    for smoothed_ball in smoothed_ball_coordinates:
        center = field_metric_cm_to_schematic((smoothed_ball.cm_x, smoothed_ball.cm_y))
        radius = max(4, int(smoothed_ball.radius_px * SCHEMATIC_WIDTH_PX / max(1, source_width)))
        if smoothed_ball.label == "white":
            fill_color = (245, 245, 245)
            edge_color = (120, 120, 120)
        else:
            fill_color = (0, 140, 255)
            edge_color = (0, 80, 180)
        cv2.circle(schematic, center, radius, fill_color, -1, cv2.LINE_AA)
        cv2.circle(schematic, center, radius, edge_color, 1, cv2.LINE_AA)

        text_anchor = center
        label = f"X: {smoothed_ball.cm_x:.1f}, Y: {smoothed_ball.cm_y:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        text_x = text_anchor[0] + 10
        if text_x + text_width > SCHEMATIC_WIDTH_PX - 1:
            text_x = max(0, text_anchor[0] - 10 - text_width)

        text_y = text_anchor[1] - 10
        min_text_y = text_height + baseline
        if text_y < min_text_y:
            text_y = min(SCHEMATIC_HEIGHT_PX - baseline - 1, text_anchor[1] + text_height + 10)

        cv2.putText(
            schematic,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    if app_state.route_points_cm:
        route_points = np.array(
            [field_cm_to_schematic(point_cm) for point_cm in app_state.route_points_cm],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        if len(route_points) >= 2:
            cv2.polylines(schematic, [route_points], False, (0, 255, 255), 2, cv2.LINE_AA)

    if app_state.selected_start_cm is not None:
        selected_ball = next(
            (ball for ball in smoothed_ball_coordinates if ball.track_id == app_state.selected_ball_track_id),
            None,
        )
        if selected_ball is not None:
            selected_start = field_metric_cm_to_schematic((selected_ball.cm_x, selected_ball.cm_y))
        else:
            selected_start = field_cm_to_schematic(app_state.selected_start_cm)
        cv2.circle(schematic, selected_start, 8, (0, 255, 255), 2, cv2.LINE_AA)

    camera_center_schematic = map_point_between_frames(
        (int(round(camera_center_pixels[0])), int(round(camera_center_pixels[1]))),
        (source_width, source_height),
        (SCHEMATIC_WIDTH_PX, SCHEMATIC_HEIGHT_PX),
    )
    cv2.line(
        schematic,
        (camera_center_schematic[0] - 4, camera_center_schematic[1]),
        (camera_center_schematic[0] + 4, camera_center_schematic[1]),
        (255, 80, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        schematic,
        (camera_center_schematic[0], camera_center_schematic[1] - 4),
        (camera_center_schematic[0], camera_center_schematic[1] + 4),
        (255, 80, 80),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        schematic,
        f"Field {FIELD_WIDTH_CM:.1f}x{FIELD_HEIGHT_CM:.1f} cm",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        schematic,
        "White balls: "
        f"{sum(ball.label == 'white' for ball in smoothed_ball_coordinates)}  Orange balls: "
        f"{sum(ball.label == 'orange' for ball in smoothed_ball_coordinates)}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return schematic


def annotate_camera_frame(
    frame_bgr: np.ndarray,
    red_zones: list[RedZoneDetection],
    white_balls: list[BallDetection],
    orange_balls: list[BallDetection],
    fps: float,
) -> np.ndarray:
    """Draw detections and lightweight debug text on the camera image."""
    annotated = frame_bgr.copy()

    for zone in red_zones:
        x, y, width, height = zone.bounding_box
        cv2.drawContours(annotated, [zone.contour], -1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 80, 255), 1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"red {int(zone.area)}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    for ball in white_balls:
        cv2.circle(annotated, ball.center, ball.radius_px, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(annotated, ball.center, 2, (200, 200, 200), -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"W c={ball.circularity:.2f}",
            (ball.center[0] + 10, ball.center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    for orange_ball in orange_balls:
        cv2.circle(annotated, orange_ball.center, orange_ball.radius_px, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.circle(annotated, orange_ball.center, 2, (0, 180, 255), -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"O c={orange_ball.circularity:.2f}",
            (orange_ball.center[0] + 10, orange_ball.center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 140, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def build_mask_preview(red_mask: np.ndarray, white_mask: np.ndarray, orange_mask: np.ndarray) -> np.ndarray:
    """Create a compact debug view for the segmentation stage."""
    red_bgr = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
    white_bgr = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
    orange_bgr = cv2.cvtColor(orange_mask, cv2.COLOR_GRAY2BGR)

    cv2.putText(red_bgr, "Red", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(white_bgr, "White", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(orange_bgr, "Orange", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2, cv2.LINE_AA)
    return np.hstack((red_bgr, white_bgr, orange_bgr))


def resize_to_match_height(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize the schematic panel so both panels can be stacked horizontally."""
    target_height = left.shape[0]
    if right.shape[0] == target_height:
        return left, right
    new_width = int(right.shape[1] * target_height / max(1, right.shape[0]))
    resized = cv2.resize(right, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
    return left, resized


def make_topdown_placeholder(message: str) -> np.ndarray:
    """Create a deterministic placeholder while the manual warp is not ready."""
    width, height = TOPDOWN_WARP_SIZE
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        placeholder,
        message,
        (30, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return placeholder


def draw_detected_aruco_centers(frame: np.ndarray, marker_centers: dict[int, np.ndarray]) -> np.ndarray:
    """Overlay the detected ArUco marker centers used for auto calibration."""
    overlay = frame.copy()
    for marker_id in REQUIRED_ARUCO_IDS:
        if marker_id not in marker_centers:
            continue

        center = marker_centers[marker_id]
        point = (int(round(center[0])), int(round(center[1])))
        cv2.circle(overlay, point, 6, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"ID {marker_id}",
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def draw_projected_aruco_debug(frame: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """Project the field and marker-center geometry back onto the selector view."""
    overlay = frame.copy()
    inverse_transform = np.linalg.inv(transform_matrix)

    field_outline = cv2.perspectiveTransform(
        topdown_field_corners().reshape(-1, 1, 2),
        inverse_transform,
    ).astype(np.int32)
    marker_outline = cv2.perspectiveTransform(
        aruco_destination_points().reshape(-1, 1, 2),
        inverse_transform,
    ).astype(np.int32)

    cv2.polylines(overlay, [field_outline], True, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.polylines(overlay, [marker_outline], True, (255, 200, 0), 2, cv2.LINE_AA)

    field_labels = ("Field TL", "Field TR", "Field BR", "Field BL")
    for label, point in zip(field_labels, field_outline.reshape(-1, 2)):
        point_xy = (int(point[0]), int(point[1]))
        cv2.circle(overlay, point_xy, 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            label,
            (point_xy[0] + 8, point_xy[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    marker_labels = ("Marker TL", "Marker TR", "Marker BR", "Marker BL")
    for label, point in zip(marker_labels, marker_outline.reshape(-1, 2)):
        point_xy = (int(point[0]), int(point[1]))
        cv2.circle(overlay, point_xy, 4, (255, 200, 0), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            label,
            (point_xy[0] + 8, point_xy[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        overlay,
        "Yellow: inferred inner field   Blue: expected ArUco center quad",
        (16, overlay.shape[0] - 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def reset_detection_state(app_state: AppState) -> None:
    """Clear detection state when calibration is not yet valid."""
    app_state.latest_frame_shape = None
    app_state.latest_red_zones = None
    app_state.latest_white_balls = None
    app_state.latest_orange_balls = None
    app_state.latest_smoothed_ball_coordinates = []
    app_state.selected_ball_track_id = None
    app_state.selected_start_cm = None
    app_state.route_points_cm = None
    app_state.coordinate_smoother.reset()


def prepare_live_topdown_frame(
    frame_bgr: np.ndarray,
    calibration_file: Path,
    balance: float,
    selection_state: TopdownSelectionState,
) -> tuple[np.ndarray, np.ndarray]:
    """Undistort the live frame and apply automatic or manual top-down calibration."""
    undistorted = undistort_with_calibration(frame_bgr, str(calibration_file), balance=balance)
    selection_state.frame_size = (int(undistorted.shape[1]), int(undistorted.shape[0]))
    if selection_state.cursor == (0, 0):
        selection_state.cursor = (
            selection_state.frame_size[0] // 2,
            selection_state.frame_size[1] // 2,
        )

    debug_view = undistorted.copy()
    manual_mode_active = selection_state.calibration_state in (
        CalibrationState.CALIBRATING_MANUAL,
        CalibrationState.CALIBRATED_MANUAL,
    )
    if selection_state.aruco_available and not manual_mode_active:
        corners, ids = detect_aruco_markers(
            undistorted,
            selection_state.aruco_dictionary,
            selection_state.aruco_detector,
        )
        selection_state.latest_aruco_centers = extract_required_marker_centers(corners, ids)
        if ids is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(debug_view, corners, ids)
        debug_view = draw_detected_aruco_centers(debug_view, selection_state.latest_aruco_centers)

        auto_transform = build_auto_topdown_transform(selection_state.latest_aruco_centers)
        if auto_transform is not None:
            selection_state.transform_matrix = auto_transform
            selection_state.calibration_state = CalibrationState.CALIBRATED_AUTO
            selection_state.points.clear()
            debug_view = draw_projected_aruco_debug(debug_view, auto_transform)
        elif (
            selection_state.transform_matrix is not None
            and selection_state.calibration_state == CalibrationState.CALIBRATED_AUTO
        ):
            debug_view = draw_projected_aruco_debug(debug_view, selection_state.transform_matrix)

    selector_view = draw_manual_selection_overlay(debug_view, selection_state)
    if selection_state.transform_matrix is None:
        if selection_state.calibration_state == CalibrationState.CALIBRATING_MANUAL:
            return selector_view, make_topdown_placeholder("Waiting for 4 selected points")
        if not selection_state.aruco_available:
            return selector_view, make_topdown_placeholder("ArUco unavailable, press m for manual mode")
        return selector_view, make_topdown_placeholder("Waiting for ArUco markers 0,1,2,3")

    topdown = cv2.warpPerspective(undistorted, selection_state.transform_matrix, TOPDOWN_WARP_SIZE)
    return selector_view, topdown


def load_image_frame(image_path: Path) -> np.ndarray:
    """Load a still image from disk."""
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return frame


def process_frame(
    frame_bgr: np.ndarray,
    params: dict[str, object],
    fps: float,
    app_state: AppState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the full detection pass and build both output panels."""
    camera_center_pixels = (
        float(params["camera_center_x"]),
        float(params["camera_center_y"]),
    )
    red_zones, red_mask = detect_red_zones(frame_bgr, params, camera_center_pixels)
    white_balls, orange_balls, ball_masks = detect_balls(frame_bgr, params, camera_center_pixels)
    smoothed_ball_coordinates = app_state.coordinate_smoother.update(
        white_balls + orange_balls,
        frame_bgr.shape,
    )

    app_state.latest_frame_shape = frame_bgr.shape
    app_state.latest_red_zones = red_zones
    app_state.latest_white_balls = white_balls
    app_state.latest_orange_balls = orange_balls
    app_state.latest_smoothed_ball_coordinates = smoothed_ball_coordinates
    update_route_from_state(app_state)

    annotated = annotate_camera_frame(
        frame_bgr=frame_bgr,
        red_zones=red_zones,
        white_balls=white_balls,
        orange_balls=orange_balls,
        fps=fps,
    )
    schematic = draw_schematic(
        frame_shape=frame_bgr.shape,
        red_zones=red_zones,
        smoothed_ball_coordinates=smoothed_ball_coordinates,
        camera_center_pixels=camera_center_pixels,
        app_state=app_state,
    )
    masks = build_mask_preview(red_mask, ball_masks["white"], ball_masks["orange"])
    combined = np.hstack(resize_to_match_height(annotated, schematic))
    return combined, masks, schematic


def configure_camera(cap: cv2.VideoCapture, width: int, height: int) -> None:
    """Apply optional capture settings without forcing a resolution when not needed."""
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def resize_frame_to_size(frame_bgr: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize a raw frame to the calibration size using deterministic interpolation."""
    target_width, target_height = target_size
    frame_height, frame_width = frame_bgr.shape[:2]
    if (frame_width, frame_height) == target_size:
        return frame_bgr

    interpolation = (
        cv2.INTER_AREA
        if frame_width > target_width or frame_height > target_height
        else cv2.INTER_LINEAR
    )
    return cv2.resize(frame_bgr, target_size, interpolation=interpolation)


def run_image_mode(image_path: Path) -> int:
    """Run repeated processing on one still image so HSV sliders remain interactive."""
    frame = load_image_frame(image_path)
    app_state = AppState()
    create_hsv_trackbars((int(frame.shape[1]), int(frame.shape[0])))
    cv2.namedWindow(SCHEMATIC_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(SCHEMATIC_WINDOW_NAME, on_schematic_mouse, app_state)

    while True:
        start = time.perf_counter()
        params = read_hsv_ranges()
        combined, masks, schematic = process_frame(frame, params, fps=0.0, app_state=app_state)
        processing_ms = (time.perf_counter() - start) * 1000.0

        cv2.putText(
            combined,
            f"Image mode  Proc: {processing_ms:.1f} ms",
            (20, combined.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, combined)
        cv2.imshow(SCHEMATIC_WINDOW_NAME, schematic)
        cv2.imshow(MASK_WINDOW_NAME, masks)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break

    return 0


def run_raw_stream_mode(
    cap: cv2.VideoCapture,
    source_name: str,
    balance: float,
    mode_label: str,
    frame_delay_ms: int,
    resize_to_size: tuple[int, int] | None = None,
) -> int:
    """Run live-style detection from any raw frame stream."""
    if not CALIBRATION_FILE.exists():
        print(f"Calibration file not found: {CALIBRATION_FILE}", file=sys.stderr)
        return 1

    ok, initial_frame = cap.read()
    if not ok or initial_frame is None:
        print(f"Could not read first frame from {source_name}", file=sys.stderr)
        return 1

    app_state = AppState()
    aruco_dictionary, aruco_detector = build_aruco_detector()
    selection_state = TopdownSelectionState(
        points=[],
        cursor=(0, 0),
        frame_size=(0, 0),
        calibration_state=CalibrationState.NEEDS_CALIBRATION,
        aruco_dictionary=aruco_dictionary,
        aruco_detector=aruco_detector,
        aruco_available=aruco_dictionary is not None and aruco_detector is not None,
    )
    create_hsv_trackbars(TOPDOWN_WARP_SIZE)
    cv2.namedWindow(SCHEMATIC_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MANUAL_SELECTOR_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(MANUAL_SELECTOR_WINDOW_NAME, on_manual_topdown_mouse, selection_state)
    cv2.setMouseCallback(SCHEMATIC_WINDOW_NAME, on_schematic_mouse, app_state)
    last_tick = time.perf_counter()
    resize_notice_shown = False

    while True:
        raw_frame = initial_frame
        initial_frame = None
        if raw_frame is None:
            ok, raw_frame = cap.read()
            if not ok or raw_frame is None:
                if mode_label == "Video":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, raw_frame = cap.read()
                    if not ok or raw_frame is None:
                        print(f"Could not restart video: {source_name}", file=sys.stderr)
                        return 1
                    last_tick = time.perf_counter()
                    reset_detection_state(app_state)
                else:
                    print(f"Frame read failed from {source_name}", file=sys.stderr)
                    return 1

        if resize_to_size is not None:
            original_size = (int(raw_frame.shape[1]), int(raw_frame.shape[0]))
            raw_frame = resize_frame_to_size(raw_frame, resize_to_size)
            if original_size != resize_to_size and not resize_notice_shown:
                print(
                    f"Resizing video frames from {original_size} to {resize_to_size} "
                    "before undistortion."
                )
                resize_notice_shown = True

        start = time.perf_counter()
        selector_view, topdown_frame = prepare_live_topdown_frame(
            raw_frame,
            CALIBRATION_FILE,
            balance,
            selection_state,
        )
        params = read_hsv_ranges()

        now = time.perf_counter()
        fps = 1.0 / max(1e-6, now - last_tick)
        last_tick = now

        if selection_state.transform_matrix is None:
            reset_detection_state(app_state)
            masks = build_mask_preview(
                np.zeros(topdown_frame.shape[:2], dtype=np.uint8),
                np.zeros(topdown_frame.shape[:2], dtype=np.uint8),
                np.zeros(topdown_frame.shape[:2], dtype=np.uint8),
            )
            schematic = draw_schematic(
                frame_shape=topdown_frame.shape,
                red_zones=[],
                smoothed_ball_coordinates=[],
                camera_center_pixels=(
                    float(params["camera_center_x"]),
                    float(params["camera_center_y"]),
                ),
                app_state=app_state,
            )
            combined = np.hstack(resize_to_match_height(topdown_frame, schematic))
            cv2.putText(
                combined,
                (
                    "Waiting for manual top-down selection"
                    if selection_state.calibration_state == CalibrationState.CALIBRATING_MANUAL
                    else "Waiting for ArUco auto-calibration"
                ),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            combined, masks, schematic = process_frame(
                topdown_frame,
                params,
                fps=fps,
                app_state=app_state,
            )
        processing_ms = (time.perf_counter() - start) * 1000.0

        cv2.putText(
            combined,
            f"{mode_label} mode  Proc: {processing_ms:.1f} ms",
            (20, combined.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(MANUAL_SELECTOR_WINDOW_NAME, selector_view)
        cv2.imshow(WINDOW_NAME, combined)
        cv2.imshow(SCHEMATIC_WINDOW_NAME, schematic)
        cv2.imshow(MASK_WINDOW_NAME, masks)
        wait_ms = max(1, frame_delay_ms - int(round(processing_ms)))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("r"):
            selection_state.clear_points()
            reset_detection_state(app_state)
        if key == ord("a"):
            selection_state.start_auto_calibration()
            reset_detection_state(app_state)
        if key == ord("m"):
            selection_state.start_manual_calibration()
            reset_detection_state(app_state)

    return 0


def run_live_mode(camera_index: int, balance: float, width: int, height: int) -> int:
    """Run live detection from the camera."""
    if not CALIBRATION_FILE.exists():
        print(f"Calibration file not found: {CALIBRATION_FILE}", file=sys.stderr)
        return 1

    calibration_width, calibration_height = load_calibration_image_size(CALIBRATION_FILE)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}", file=sys.stderr)
        return 1

    configure_camera(
        cap,
        width if width > 0 else calibration_width,
        height if height > 0 else calibration_height,
    )
    try:
        return run_raw_stream_mode(cap, f"camera {camera_index}", balance, "Live", 1)
    finally:
        cap.release()


def run_video_mode(video_path: Path, balance: float, resize_to_calibration: bool) -> int:
    """Replay a recorded camera video through the live detector pipeline."""
    if not video_path.exists():
        print(f"Video file not found: {video_path}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay_ms = int(round(1000.0 / fps)) if fps and fps > 0.0 else 1
    if resize_to_calibration and not CALIBRATION_FILE.exists():
        print(f"Calibration file not found: {CALIBRATION_FILE}", file=sys.stderr)
        cap.release()
        return 1
    resize_to_size = (
        load_calibration_image_size(CALIBRATION_FILE) if resize_to_calibration else None
    )

    try:
        return run_raw_stream_mode(
            cap,
            str(video_path),
            balance,
            "Video",
            frame_delay_ms,
            resize_to_size=resize_to_size,
        )
    finally:
        cap.release()


def main() -> int:
    """Entrypoint used when the script is started from the terminal."""
    cv2.ocl.setUseOpenCL(False)
    args = parse_args()

    try:
        if args.live or USE_LIVE_FEED:
            return run_live_mode(args.camera_index, args.balance, args.width, args.height)
        if args.video is not None:
            return run_video_mode(args.video, args.balance, args.resize_video_to_calibration)
        return run_image_mode(args.image)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
