#!/usr/bin/env python3
"""Detect red zones, white balls, and one orange ball in a top-down arena view.

This tool supports two input modes:
1. Still image input for repeatable offline tuning.
2. Live camera input for on-table tuning with HSV trackbars.

The output is shown as:
- Left: annotated top-down camera frame
- Right: synthetic 2D schematic of the 180x120 cm field

The script intentionally keeps the detection pipeline simple and deterministic:
- HSV thresholding
- Morphology cleanup
- Contour extraction
- Circularity filtering for ball-like objects
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camera.imageprocessing import undistort_with_calibration


FIELD_WIDTH_CM = 180
FIELD_HEIGHT_CM = 120
CALIBRATION_FILE = REPO_ROOT / "calibration_data.npz"
DEFAULT_IMAGE = REPO_ROOT / "test_topdown.png"
WINDOW_NAME = "Top-Down Detector"
MASK_WINDOW_NAME = "Segmentation Masks"
CONTROL_WINDOW_NAME = "HSV Controls"
CONTROL_WINDOW_SIZE = (1200, 900)
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
}

# Still-image mode is the safest default for deterministic tuning.
USE_LIVE_FEED = False
CAMERA_INDEX = 1

# Schematic sizing is chosen to keep the correct 180:120 = 3:2 field aspect ratio.
SCHEMATIC_WIDTH_PX = 900
SCHEMATIC_HEIGHT_PX = 600


@dataclass(frozen=True)
class BallDetection:
    """Ball-like object found in the frame."""

    label: str
    center: tuple[int, int]
    radius_px: int
    contour: np.ndarray
    area: float
    circularity: float


@dataclass(frozen=True)
class RedZoneDetection:
    """Detected red avoidance geometry."""

    contour: np.ndarray
    bounding_box: tuple[int, int, int, int]
    center: tuple[int, int]
    area: float


@dataclass(frozen=True)
class HSVRange:
    """Single HSV threshold range."""

    lower: np.ndarray
    upper: np.ndarray


def noop(_value: int) -> None:
    """Trackbar callback placeholder."""
    return None


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect red zones, white ping-pong balls, and one orange ball from a "
            "top-down arena image or live camera feed."
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
    return parser.parse_args()


def load_calibration_image_size(calibration_file: Path) -> tuple[int, int]:
    """Read the calibration image size so live capture matches the saved model."""
    data = np.load(str(calibration_file))
    image_size = tuple(int(value) for value in data["image_size"])
    return image_size


def create_hsv_trackbars() -> None:
    """Create trackbars for the three color classes.

    Red uses two hue intervals because red wraps across the HSV hue boundary.
    """
    cv2.namedWindow(CONTROL_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW_NAME, *CONTROL_WINDOW_SIZE)

    defaults = {
        "red1_h_min": 0,
        "red1_h_max": 12,
        "red2_h_min": 165,
        "red2_h_max": 179,
        "red_s_min": 110,
        "red_s_max": 255,
        "red_v_min": 60,
        "red_v_max": 255,
        "white_h_min": 0,
        "white_h_max": 179,
        "white_s_min": 0,
        "white_s_max": 70,
        "white_v_min": 170,
        "white_v_max": 255,
        "orange_h_min": 8,
        "orange_h_max": 30,
        "orange_s_min": 120,
        "orange_s_max": 255,
        "orange_v_min": 120,
        "orange_v_max": 255,
        "red_min_area": 400,
        "ball_min_area": 20,
        "ball_max_area": 2500,
        "ball_min_circ": 70,
    }

    for key, value in defaults.items():
        name = TRACKBAR_NAMES[key]
        max_value = 179 if "_h_" in key else 255
        if key.endswith("min_area"):
            max_value = 20000
        if key == "ball_max_area":
            max_value = 20000
        if key == "ball_min_circ":
            max_value = 100
        cv2.createTrackbar(name, CONTROL_WINDOW_NAME, value, max_value, noop)


def read_hsv_ranges() -> dict[str, object]:
    """Read current threshold parameters from the trackbars."""
    red_1 = HSVRange(
        lower=np.array(
            [
                cv2.getTrackbarPos(TRACKBAR_NAMES["red1_h_min"], CONTROL_WINDOW_NAME),
                cv2.getTrackbarPos(TRACKBAR_NAMES["red_s_min"], CONTROL_WINDOW_NAME),
                cv2.getTrackbarPos(TRACKBAR_NAMES["red_v_min"], CONTROL_WINDOW_NAME),
            ],
            dtype=np.uint8,
        ),
        upper=np.array(
            [
                cv2.getTrackbarPos(TRACKBAR_NAMES["red1_h_max"], CONTROL_WINDOW_NAME),
                cv2.getTrackbarPos(TRACKBAR_NAMES["red_s_max"], CONTROL_WINDOW_NAME),
                cv2.getTrackbarPos(TRACKBAR_NAMES["red_v_max"], CONTROL_WINDOW_NAME),
            ],
            dtype=np.uint8,
        ),
    )
    red_2 = HSVRange(
        lower=np.array(
            [
                cv2.getTrackbarPos(TRACKBAR_NAMES["red2_h_min"], CONTROL_WINDOW_NAME),
                cv2.getTrackbarPos(TRACKBAR_NAMES["red_s_min"], CONTROL_WINDOW_NAME),
                cv2.getTrackbarPos(TRACKBAR_NAMES["red_v_min"], CONTROL_WINDOW_NAME),
            ],
            dtype=np.uint8,
        ),
        upper=np.array(
            [
                cv2.getTrackbarPos(TRACKBAR_NAMES["red2_h_max"], CONTROL_WINDOW_NAME),
                cv2.getTrackbarPos(TRACKBAR_NAMES["red_s_max"], CONTROL_WINDOW_NAME),
                cv2.getTrackbarPos(TRACKBAR_NAMES["red_v_max"], CONTROL_WINDOW_NAME),
            ],
            dtype=np.uint8,
        ),
    )
    white = HSVRange(
        lower=np.array(
            [
                cv2.getTrackbarPos(TRACKBAR_NAMES["white_h_min"], CONTROL_WINDOW_NAME),
                cv2.getTrackbarPos(TRACKBAR_NAMES["white_s_min"], CONTROL_WINDOW_NAME),
                cv2.getTrackbarPos(TRACKBAR_NAMES["white_v_min"], CONTROL_WINDOW_NAME),
            ],
            dtype=np.uint8,
        ),
        upper=np.array(
            [
                cv2.getTrackbarPos(TRACKBAR_NAMES["white_h_max"], CONTROL_WINDOW_NAME),
                cv2.getTrackbarPos(TRACKBAR_NAMES["white_s_max"], CONTROL_WINDOW_NAME),
                cv2.getTrackbarPos(TRACKBAR_NAMES["white_v_max"], CONTROL_WINDOW_NAME),
            ],
            dtype=np.uint8,
        ),
    )
    orange = HSVRange(
        lower=np.array(
            [
                cv2.getTrackbarPos(TRACKBAR_NAMES["orange_h_min"], CONTROL_WINDOW_NAME),
                cv2.getTrackbarPos(TRACKBAR_NAMES["orange_s_min"], CONTROL_WINDOW_NAME),
                cv2.getTrackbarPos(TRACKBAR_NAMES["orange_v_min"], CONTROL_WINDOW_NAME),
            ],
            dtype=np.uint8,
        ),
        upper=np.array(
            [
                cv2.getTrackbarPos(TRACKBAR_NAMES["orange_h_max"], CONTROL_WINDOW_NAME),
                cv2.getTrackbarPos(TRACKBAR_NAMES["orange_s_max"], CONTROL_WINDOW_NAME),
                cv2.getTrackbarPos(TRACKBAR_NAMES["orange_v_max"], CONTROL_WINDOW_NAME),
            ],
            dtype=np.uint8,
        ),
    )

    return {
        "red_1": red_1,
        "red_2": red_2,
        "white": white,
        "orange": orange,
        "red_min_area": float(cv2.getTrackbarPos(TRACKBAR_NAMES["red_min_area"], CONTROL_WINDOW_NAME)),
        "ball_min_area": float(cv2.getTrackbarPos(TRACKBAR_NAMES["ball_min_area"], CONTROL_WINDOW_NAME)),
        "ball_max_area": float(cv2.getTrackbarPos(TRACKBAR_NAMES["ball_max_area"], CONTROL_WINDOW_NAME)),
        "ball_min_circularity": cv2.getTrackbarPos(TRACKBAR_NAMES["ball_min_circ"], CONTROL_WINDOW_NAME) / 100.0,
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


def detect_red_zones(frame_bgr: np.ndarray, params: dict[str, object]) -> tuple[list[RedZoneDetection], np.ndarray]:
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
        detections.append(
            RedZoneDetection(
                contour=contour,
                bounding_box=(x, y, width, height),
                center=contour_center(contour),
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
                radius_px=max(2, int(radius)),
                contour=contour,
                area=area,
                circularity=circularity,
            )
        )

    return detections, mask


def detect_balls(frame_bgr: np.ndarray, params: dict[str, object]) -> tuple[list[BallDetection], list[BallDetection], dict[str, np.ndarray]]:
    """Detect white balls and orange balls using the same contour filters."""
    white_detections, white_mask = detect_ball_candidates(
        frame_bgr=frame_bgr,
        hsv_range=params["white"],
        label="white",
        min_area=float(params["ball_min_area"]),
        max_area=float(params["ball_max_area"]),
        min_circularity=float(params["ball_min_circularity"]),
    )
    orange_detections, orange_mask = detect_ball_candidates(
        frame_bgr=frame_bgr,
        hsv_range=params["orange"],
        label="orange",
        min_area=float(params["ball_min_area"]),
        max_area=float(params["ball_max_area"]),
        min_circularity=float(params["ball_min_circularity"]),
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


def draw_schematic(
    frame_shape: tuple[int, int, int],
    red_zones: list[RedZoneDetection],
    white_balls: list[BallDetection],
    orange_balls: list[BallDetection],
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
        mapped_contour = zone.contour.astype(np.float32).copy()
        mapped_contour[:, 0, 0] *= SCHEMATIC_WIDTH_PX / max(1, source_width)
        mapped_contour[:, 0, 1] *= SCHEMATIC_HEIGHT_PX / max(1, source_height)
        mapped_contour = mapped_contour.astype(np.int32)
        cv2.polylines(schematic, [mapped_contour], True, (0, 0, 255), 3, cv2.LINE_AA)

    for ball in white_balls:
        center = map_point_between_frames(
            ball.center,
            (source_width, source_height),
            (SCHEMATIC_WIDTH_PX, SCHEMATIC_HEIGHT_PX),
        )
        radius = max(4, int(ball.radius_px * SCHEMATIC_WIDTH_PX / max(1, source_width)))
        cv2.circle(schematic, center, radius, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(schematic, center, radius, (120, 120, 120), 1, cv2.LINE_AA)

    for orange_ball in orange_balls:
        center = map_point_between_frames(
            orange_ball.center,
            (source_width, source_height),
            (SCHEMATIC_WIDTH_PX, SCHEMATIC_HEIGHT_PX),
        )
        radius = max(4, int(orange_ball.radius_px * SCHEMATIC_WIDTH_PX / max(1, source_width)))
        cv2.circle(schematic, center, radius, (0, 140, 255), -1, cv2.LINE_AA)
        cv2.circle(schematic, center, radius, (0, 80, 180), 1, cv2.LINE_AA)

    cv2.putText(
        schematic,
        f"Field {FIELD_WIDTH_CM}x{FIELD_HEIGHT_CM} cm",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        schematic,
        f"White balls: {len(white_balls)}  Orange balls: {len(orange_balls)}",
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


def prepare_live_topdown_frame(frame_bgr: np.ndarray, calibration_file: Path, balance: float) -> np.ndarray:
    """Undistort the live frame and return the image used for detection.

    This repository already has separate tools for perspective warping.
    If your live camera stream is not already aligned as top-down after
    undistortion, insert your existing warp step here.
    """
    undistorted = undistort_with_calibration(frame_bgr, str(calibration_file), balance=balance)
    return undistorted


def load_image_frame(image_path: Path) -> np.ndarray:
    """Load a still image from disk."""
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return frame


def process_frame(frame_bgr: np.ndarray, params: dict[str, object], fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Run the full detection pass and build both output panels."""
    red_zones, red_mask = detect_red_zones(frame_bgr, params)
    white_balls, orange_balls, ball_masks = detect_balls(frame_bgr, params)

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
        white_balls=white_balls,
        orange_balls=orange_balls,
    )
    masks = build_mask_preview(red_mask, ball_masks["white"], ball_masks["orange"])
    return np.hstack(resize_to_match_height(annotated, schematic)), masks


def configure_camera(cap: cv2.VideoCapture, width: int, height: int) -> None:
    """Apply optional capture settings without forcing a resolution when not needed."""
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def run_image_mode(image_path: Path) -> int:
    """Run repeated processing on one still image so HSV sliders remain interactive."""
    frame = load_image_frame(image_path)
    create_hsv_trackbars()

    while True:
        start = time.perf_counter()
        params = read_hsv_ranges()
        combined, masks = process_frame(frame, params, fps=0.0)
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
        cv2.imshow(MASK_WINDOW_NAME, masks)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break

    return 0


def run_live_mode(camera_index: int, balance: float, width: int, height: int) -> int:
    """Run live detection from the camera."""
    if not CALIBRATION_FILE.exists():
        print(f"Calibration file not found: {CALIBRATION_FILE}", file=sys.stderr)
        return 1

    calibration_width, calibration_height = load_calibration_image_size(CALIBRATION_FILE)
    create_hsv_trackbars()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}", file=sys.stderr)
        return 1

    configure_camera(
        cap,
        width if width > 0 else calibration_width,
        height if height > 0 else calibration_height,
    )
    last_tick = time.perf_counter()

    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok or raw_frame is None:
                print("Camera read failed", file=sys.stderr)
                return 1

            start = time.perf_counter()
            topdown_frame = prepare_live_topdown_frame(raw_frame, CALIBRATION_FILE, balance)
            params = read_hsv_ranges()

            now = time.perf_counter()
            fps = 1.0 / max(1e-6, now - last_tick)
            last_tick = now

            combined, masks = process_frame(topdown_frame, params, fps=fps)
            processing_ms = (time.perf_counter() - start) * 1000.0

            cv2.putText(
                combined,
                f"Live mode  Proc: {processing_ms:.1f} ms",
                (20, combined.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, combined)
            cv2.imshow(MASK_WINDOW_NAME, masks)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()

    return 0


def main() -> int:
    """Entrypoint used when the script is started from the terminal."""
    cv2.ocl.setUseOpenCL(False)
    args = parse_args()

    try:
        if args.live or USE_LIVE_FEED:
            return run_live_mode(args.camera_index, args.balance, args.width, args.height)
        return run_image_mode(args.image)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
