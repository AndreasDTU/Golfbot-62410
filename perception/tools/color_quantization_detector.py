#!/usr/bin/env python3
"""Interactive LAB-based color quantization calibration and detection tool."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.camera.imageprocessing import undistort_with_calibration


USE_LIVE_FEED = False
INPUT_IMAGE = REPO_ROOT / "Bane_undistorted.png"
CALIBRATION_FILE = REPO_ROOT / "calibration_data.npz"
CAMERA_INDEX = 0
WINDOW_NAME = "Color Quantization Detector"
MASK_WINDOW_NAME = "Color Masks"
CONTROLS_WINDOW_NAME = "Controls"
KMEANS_ATTEMPTS = 5
KMEANS_MAX_ITER = 30
KMEANS_EPSILON = 1.0
MEDIAN_BLUR_SIZE = 5
BALL_MAX_AREA = 8000.0
BALL_MAX_ASPECT_RATIO_DELTA = 0.2
USE_GENERAL_RED_MASK_FOR_RAW_IMGS = True
CROSS_MIN_AREA = 120.0
CROSS_MAX_AREA = 50000.0
CROSS_MIN_SOLIDITY = 0.2
SCHEMATIC_BALL_RADIUS = 12
LAB_A_NEUTRAL = 128.0
DEFAULT_EDGE_TOL = 35
RED_BORDER_AREA_THRESHOLD = 5000.0
RED_CROSS_MAX_AREA = 5000.0
RED_CROSS_MIN_ASPECT_RATIO = 0.5
RED_CROSS_MAX_ASPECT_RATIO = 2.0

DEFAULT_K = 8
DEFAULT_TOLERANCE = 15
DEFAULT_MIN_CIRCULARITY = 75
DEFAULT_MIN_AREA = 20
MAX_K = 15
ANALYSIS_EVERY_N_FRAMES = 5

CLASS_ORDER = ("white", "orange", "red", "floor")
CLASS_KEYS = {
    ord("w"): "white",
    ord("o"): "orange",
    ord("r"): "red",
    ord("f"): "floor",
}
DISPLAY_COLORS = {
    "white": (220, 220, 220),
    "orange": (0, 140, 255),
    "red": (20, 34, 207),
    "floor": (136, 136, 136),
}
TARGET_COLORS_RGB = {
    "white": "D3D4CF",
    "orange": "F88E05",
    "red": "CF2214",
    "floor": "888888",
}


@dataclass(frozen=True)
class Detection:
    label: str
    center: tuple[int, int]
    contour: np.ndarray
    area: float
    circularity: float


@dataclass
class AppState:
    active_class: str
    target_labs: dict[str, np.ndarray]
    current_frame: np.ndarray | None = None
    frame_index: int = 0
    cached_k: int = -1
    cached_quantized: np.ndarray | None = None
    cached_label_map: np.ndarray | None = None
    cached_centers_bgr: np.ndarray | None = None
    cached_centers_lab: np.ndarray | None = None


def hex_rgb_to_bgr(hex_code: str) -> np.ndarray:
    clean = hex_code[:6]
    rgb = np.array(
        [int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16)],
        dtype=np.float32,
    )
    return rgb[::-1]


def bgr_array_to_lab(colors_bgr: np.ndarray) -> np.ndarray:
    colors_uint8 = np.clip(colors_bgr, 0, 255).astype(np.uint8).reshape((-1, 1, 3))
    return cv2.cvtColor(colors_uint8, cv2.COLOR_BGR2LAB).reshape((-1, 3)).astype(np.float32)


def build_default_targets() -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    for label, hex_code in TARGET_COLORS_RGB.items():
        bgr = hex_rgb_to_bgr(hex_code).reshape((1, 3))
        targets[label] = bgr_array_to_lab(bgr)[0]
    return targets


def noop(_value: int) -> None:
    return None


def clamp_k_value(raw_value: int) -> int:
    return max(2, min(MAX_K, raw_value))


def quantize_image(image: np.ndarray, k_clusters: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    blurred = cv2.medianBlur(image, MEDIAN_BLUR_SIZE)
    samples = blurred.reshape((-1, 3)).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        KMEANS_MAX_ITER,
        KMEANS_EPSILON,
    )
    _compactness, labels, centers = cv2.kmeans(
        samples,
        k_clusters,
        None,
        criteria,
        KMEANS_ATTEMPTS,
        cv2.KMEANS_PP_CENTERS,
    )

    centers = np.clip(centers, 0, 255).astype(np.uint8)
    quantized = centers[labels.flatten()].reshape(image.shape)
    label_map = labels.reshape(image.shape[:2])
    centers_lab = bgr_array_to_lab(centers.astype(np.float32))
    return quantized, label_map, centers, centers_lab


def ensure_quantized_cache(state: AppState, frame: np.ndarray, k_clusters: int, force_refresh: bool = False) -> None:
    if not force_refresh and state.cached_k == k_clusters and state.cached_quantized is not None:
        return

    quantized, label_map, centers_bgr, centers_lab = quantize_image(frame, k_clusters)
    state.cached_k = k_clusters
    state.cached_quantized = quantized
    state.cached_label_map = label_map
    state.cached_centers_bgr = centers_bgr
    state.cached_centers_lab = centers_lab


def get_clusters_within_tolerance(
    centers_lab: np.ndarray,
    target_lab: np.ndarray,
    tolerance: float,
) -> list[int]:
    distances = np.linalg.norm(centers_lab - target_lab[None, :], axis=1)
    return [int(index) for index, distance in enumerate(distances) if float(distance) <= tolerance]


def cluster_mask(label_map: np.ndarray, cluster_indices: list[int]) -> np.ndarray:
    mask = np.zeros(label_map.shape, dtype=np.uint8)
    for cluster_index in cluster_indices:
        mask[label_map == cluster_index] = 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)


def contour_center(contour: np.ndarray) -> tuple[int, int]:
    moments = cv2.moments(contour)
    if moments["m00"] <= 0.0:
        x, y, width, height = cv2.boundingRect(contour)
        return (x + width // 2, y + height // 2)
    return (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))


def detect_ball_like_objects(
    mask: np.ndarray,
    label: str,
    min_area: float,
    min_circularity: float,
) -> list[Detection]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > BALL_MAX_AREA:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue

        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        _x, _y, width, height = cv2.boundingRect(contour)
        aspect_ratio = width / float(height) if height > 0 else 0.0
        if abs(1.0 - aspect_ratio) > BALL_MAX_ASPECT_RATIO_DELTA:
            continue

        detections.append(
            Detection(
                label=label,
                center=contour_center(contour),
                contour=contour,
                area=area,
                circularity=circularity,
            )
        )

    return sorted(detections, key=lambda item: (item.center[1], item.center[0]))


def detect_cross(mask: np.ndarray) -> Detection | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_detection: Detection | None = None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < CROSS_MIN_AREA or area > CROSS_MAX_AREA:
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0.0 else 0.0
        if solidity < CROSS_MIN_SOLIDITY:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter) if perimeter > 0.0 else 0.0
        candidate = Detection(
            label="red",
            center=contour_center(contour),
            contour=contour,
            area=area,
            circularity=circularity,
        )

        if best_detection is None or candidate.area > best_detection.area:
            best_detection = candidate

    return best_detection


def contour_touches_frame(contour: np.ndarray, image_shape: tuple[int, int, int]) -> bool:
    height, width = image_shape[:2]
    x, y, contour_width, contour_height = cv2.boundingRect(contour)
    return x <= 2 or y <= 2 or (x + contour_width) >= (width - 2) or (y + contour_height) >= (height - 2)


def categorize_red_contours(
    mask: np.ndarray,
    image_shape: tuple[int, int, int],
    min_area: float,
) -> tuple[Detection | None, list[np.ndarray]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_cross: Detection | None = None
    border_contours: list[np.ndarray] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if height <= 0:
            continue

        aspect_ratio = float(width) / float(height)
        center = contour_center(contour)

        if area > RED_BORDER_AREA_THRESHOLD:
            border_contours.append(contour)
            continue

        if area > RED_CROSS_MAX_AREA:
            continue

        if aspect_ratio < RED_CROSS_MIN_ASPECT_RATIO or aspect_ratio > RED_CROSS_MAX_ASPECT_RATIO:
            continue

        candidate = Detection(
            label="red",
            center=center,
            contour=contour,
            area=area,
            circularity=0.0,
        )
        if best_cross is not None and candidate.area <= best_cross.area:
            continue

        best_cross = candidate

    return best_cross, border_contours


def draw_cross_marker(image: np.ndarray, center: tuple[int, int], color: tuple[int, int, int], scale: int) -> None:
    x, y = center
    thickness = max(2, scale // 4)
    arm = max(6, scale)
    cv2.line(image, (x - arm, y), (x + arm, y), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x, y - arm), (x, y + arm), color, thickness, cv2.LINE_AA)


def draw_target_swatches(image: np.ndarray, state: AppState) -> None:
    for index, label in enumerate(CLASS_ORDER):
        lab = state.target_labs[label]
        bgr = cv2.cvtColor(np.uint8([[lab]]), cv2.COLOR_LAB2BGR)[0, 0]
        color = tuple(int(value) for value in bgr)
        top_left = (16 + index * 80, 16)
        bottom_right = (16 + index * 80 + 36, 52)
        cv2.rectangle(image, top_left, bottom_right, color, -1)
        outline = (0, 255, 255) if label == state.active_class else (255, 255, 255)
        cv2.rectangle(image, top_left, bottom_right, outline, 2)
        cv2.putText(
            image,
            label[0].upper(),
            (top_left[0] + 10, bottom_right[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            outline,
            2,
            cv2.LINE_AA,
        )


def build_detection_overlay(
    original: np.ndarray,
    state: AppState,
    white_balls: list[Detection],
    orange_balls: list[Detection],
    red_cross: Detection | None,
    red_borders: list[np.ndarray],
    fps: float | None,
    cluster_info: dict[str, list[int]],
    k_clusters: int,
    settings: dict[str, float],
) -> np.ndarray:
    overlay = original.copy()

    for detection in white_balls:
        (x, y), radius = cv2.minEnclosingCircle(detection.contour)
        cv2.circle(overlay, (int(x), int(y)), int(radius), DISPLAY_COLORS["white"], 2, cv2.LINE_AA)

    for detection in orange_balls:
        (x, y), radius = cv2.minEnclosingCircle(detection.contour)
        cv2.circle(overlay, (int(x), int(y)), int(radius), DISPLAY_COLORS["orange"], 2, cv2.LINE_AA)

    if red_borders:
        cv2.drawContours(overlay, red_borders, -1, DISPLAY_COLORS["red"], 3, cv2.LINE_AA)

    if red_cross is not None:
        x, y, width, height = cv2.boundingRect(red_cross.contour)
        cv2.rectangle(overlay, (x, y), (x + width, y + height), DISPLAY_COLORS["red"], 2, cv2.LINE_AA)
        draw_cross_marker(overlay, red_cross.center, DISPLAY_COLORS["red"], max(width, height) // 2)

    draw_target_swatches(overlay, state)

    info_lines = [
        f"Active class: {state.active_class} (w/o/r/f)",
        f"K: {k_clusters}",
        f"Tol W/O/R: {int(settings['tol_white'])}/{int(settings['tol_orange'])}/{int(settings['tol_red'])}",
        f"Edge Tol / A min: {int(settings['edge_tol'])} / {int(LAB_A_NEUTRAL + settings['edge_tol'])}",
        f"Min circ: {settings['min_circularity']:.2f}",
        f"Min area: {int(settings['min_area'])}",
        f"White balls: {len(white_balls)} clusters={cluster_info['white']}",
        f"Orange balls: {len(orange_balls)} clusters={cluster_info['orange']}",
        f"Red cross: {0 if red_cross is None else 1} clusters={cluster_info['red']}",
        f"Red borders: {len(red_borders)}",
    ]
    if fps is not None:
        info_lines.append(f"FPS: {fps:.1f}")

    panel_top = 78
    panel_bottom = panel_top + len(info_lines) * 26 + 18
    cv2.rectangle(overlay, (8, panel_top - 28), (520, panel_bottom), (0, 0, 0), -1)
    cv2.rectangle(overlay, (8, panel_top - 28), (520, panel_bottom), (255, 255, 255), 1)
    cv2.putText(
        overlay,
        "Trackbar Readout",
        (16, panel_top - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for index, text in enumerate(info_lines):
        cv2.putText(
            overlay,
            text,
            (16, 90 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


def build_schematic(
    image_shape: tuple[int, int, int],
    state: AppState,
    white_balls: list[Detection],
    orange_balls: list[Detection],
    red_cross: Detection | None,
    red_borders: list[np.ndarray],
) -> np.ndarray:
    height, width = image_shape[:2]
    floor_lab = state.target_labs["floor"].reshape((1, 1, 3)).astype(np.uint8)
    floor_bgr = cv2.cvtColor(floor_lab, cv2.COLOR_LAB2BGR)[0, 0]
    schematic = np.full((height, width, 3), floor_bgr, dtype=np.uint8)

    for detection in white_balls:
        cv2.circle(schematic, detection.center, SCHEMATIC_BALL_RADIUS, DISPLAY_COLORS["white"], -1, cv2.LINE_AA)
        cv2.circle(schematic, detection.center, SCHEMATIC_BALL_RADIUS, (30, 30, 30), 1, cv2.LINE_AA)

    for detection in orange_balls:
        cv2.circle(schematic, detection.center, SCHEMATIC_BALL_RADIUS, DISPLAY_COLORS["orange"], -1, cv2.LINE_AA)
        cv2.circle(schematic, detection.center, SCHEMATIC_BALL_RADIUS, (30, 30, 30), 1, cv2.LINE_AA)

    if red_borders:
        cv2.drawContours(schematic, red_borders, -1, DISPLAY_COLORS["red"], 4, cv2.LINE_AA)

    if red_cross is not None:
        cv2.drawContours(schematic, [red_cross.contour], -1, DISPLAY_COLORS["red"], thickness=cv2.FILLED, lineType=cv2.LINE_AA)

    cv2.putText(
        schematic,
        "2D Schematic",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return schematic


def colorize_mask(mask: np.ndarray, color: tuple[int, int, int], title: str) -> np.ndarray:
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored[mask > 0] = color
    cv2.putText(
        colored,
        title,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return colored


def build_mask_dashboard(white_mask: np.ndarray, orange_mask: np.ndarray, red_mask: np.ndarray) -> np.ndarray:
    return np.hstack(
        (
            colorize_mask(white_mask, DISPLAY_COLORS["white"], "White Mask"),
            colorize_mask(orange_mask, DISPLAY_COLORS["orange"], "Orange Mask"),
            colorize_mask(red_mask, DISPLAY_COLORS["red"], "Red Mask"),
        )
    )


def build_dashboard(
    frame: np.ndarray,
    state: AppState,
    white_balls: list[Detection],
    orange_balls: list[Detection],
    red_cross: Detection | None,
    red_borders: list[np.ndarray],
    fps: float | None,
    cluster_info: dict[str, list[int]],
    k_clusters: int,
    settings: dict[str, float],
) -> np.ndarray:
    overlay = build_detection_overlay(
        frame,
        state,
        white_balls,
        orange_balls,
        red_cross,
        red_borders,
        fps,
        cluster_info,
        k_clusters,
        settings,
    )
    schematic = build_schematic(frame.shape, state, white_balls, orange_balls, red_cross, red_borders)
    return np.hstack((overlay, schematic))


def get_trackbar_values() -> dict[str, float]:
    k_value = clamp_k_value(cv2.getTrackbarPos("K", CONTROLS_WINDOW_NAME))
    if k_value != cv2.getTrackbarPos("K", CONTROLS_WINDOW_NAME):
        cv2.setTrackbarPos("K", CONTROLS_WINDOW_NAME, k_value)

    return {
        "k": k_value,
        "tol_white": float(cv2.getTrackbarPos("Tol W", CONTROLS_WINDOW_NAME)),
        "tol_orange": float(cv2.getTrackbarPos("Tol O", CONTROLS_WINDOW_NAME)),
        "tol_red": float(cv2.getTrackbarPos("Tol R", CONTROLS_WINDOW_NAME)),
        "edge_tol": float(cv2.getTrackbarPos("Edge Tol", CONTROLS_WINDOW_NAME)),
        "min_circularity": cv2.getTrackbarPos("Circ", CONTROLS_WINDOW_NAME) / 100.0,
        "min_area": float(cv2.getTrackbarPos("Area", CONTROLS_WINDOW_NAME)),
    }


def analyze_frame(
    frame: np.ndarray,
    state: AppState,
    settings: dict[str, float],
    force_refresh: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[Detection], list[Detection], Detection | None, list[np.ndarray], dict[str, list[int]]]:
    ensure_quantized_cache(state, frame, int(settings["k"]), force_refresh=force_refresh)
    assert state.cached_quantized is not None
    assert state.cached_label_map is not None
    assert state.cached_centers_lab is not None

    white_clusters = get_clusters_within_tolerance(
        state.cached_centers_lab,
        state.target_labs["white"],
        settings["tol_white"],
    )
    orange_clusters = get_clusters_within_tolerance(
        state.cached_centers_lab,
        state.target_labs["orange"],
        settings["tol_orange"],
    )
    if USE_GENERAL_RED_MASK_FOR_RAW_IMGS:
        red_clusters = [
            int(index)
            for index, lab_color in enumerate(state.cached_centers_lab)
            if float(lab_color[1]) >= (LAB_A_NEUTRAL + settings["edge_tol"])
        ]
    else:
        red_clusters = get_clusters_within_tolerance(
            state.cached_centers_lab,
            state.target_labs["red"],
            settings["tol_red"],
        )

    white_mask = cluster_mask(state.cached_label_map, white_clusters)
    orange_mask = cluster_mask(state.cached_label_map, orange_clusters)
    red_mask = cluster_mask(state.cached_label_map, red_clusters)

    white_balls = detect_ball_like_objects(
        white_mask,
        "white",
        settings["min_area"],
        settings["min_circularity"],
    )
    orange_balls = detect_ball_like_objects(
        orange_mask,
        "orange",
        settings["min_area"],
        settings["min_circularity"],
    )
    if USE_GENERAL_RED_MASK_FOR_RAW_IMGS:
        red_cross, red_borders = categorize_red_contours(red_mask, frame.shape, settings["min_area"])
    else:
        red_cross = detect_cross(red_mask)
        red_borders = []

    masks = np.dstack((white_mask, orange_mask, red_mask))
    cluster_info = {
        "white": white_clusters,
        "orange": orange_clusters,
        "red": red_clusters,
    }
    return state.cached_quantized, masks, white_balls, orange_balls, red_cross, red_borders, cluster_info


def load_static_image() -> np.ndarray:
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

    image = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {INPUT_IMAGE}")
    return image


def create_controls_window() -> None:
    cv2.namedWindow(CONTROLS_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WINDOW_NAME, 400, 300)
    cv2.createTrackbar("K", CONTROLS_WINDOW_NAME, DEFAULT_K, MAX_K, noop)
    cv2.createTrackbar("Tol W", CONTROLS_WINDOW_NAME, DEFAULT_TOLERANCE, 100, noop)
    cv2.createTrackbar("Tol O", CONTROLS_WINDOW_NAME, DEFAULT_TOLERANCE, 100, noop)
    cv2.createTrackbar("Tol R", CONTROLS_WINDOW_NAME, DEFAULT_TOLERANCE, 100, noop)
    cv2.createTrackbar("Edge Tol", CONTROLS_WINDOW_NAME, DEFAULT_EDGE_TOL, 100, noop)
    cv2.createTrackbar("Circ", CONTROLS_WINDOW_NAME, DEFAULT_MIN_CIRCULARITY, 100, noop)
    cv2.createTrackbar("Area", CONTROLS_WINDOW_NAME, DEFAULT_MIN_AREA, 500, noop)


def print_sampled_target(label: str, bgr: np.ndarray, lab: np.ndarray) -> None:
    bgr_list = [int(value) for value in bgr]
    lab_list = [int(value) for value in lab]
    print(f"{label} target updated -> BGR={bgr_list} LAB={lab_list}")


def on_mouse(event: int, x: int, y: int, _flags: int, state: AppState) -> None:
    if event != cv2.EVENT_LBUTTONDOWN or state.current_frame is None:
        return

    height, width = state.current_frame.shape[:2]
    if x < 0 or y < 0 or x >= width or y >= height:
        return

    bgr = state.current_frame[y, x].astype(np.float32)
    lab = bgr_array_to_lab(bgr.reshape((1, 3)))[0]
    state.target_labs[state.active_class] = lab
    print_sampled_target(state.active_class, bgr, lab)


def process_static_image(state: AppState) -> int:
    try:
        image = load_static_image()
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    previous_time = time.perf_counter()

    while True:
        state.current_frame = image.copy()
        settings = get_trackbar_values()
        current_time = time.perf_counter()
        delta = max(current_time - previous_time, 1e-6)
        previous_time = current_time

        _quantized, masks, white_balls, orange_balls, red_cross, red_borders, cluster_info = analyze_frame(
            state.current_frame,
            state,
            settings,
            force_refresh=True,
        )
        dashboard = build_dashboard(
            state.current_frame,
            state,
            white_balls,
            orange_balls,
            red_cross,
            red_borders,
            fps=1.0 / delta,
            cluster_info=cluster_info,
            k_clusters=int(settings["k"]),
            settings=settings,
        )
        mask_dashboard = build_mask_dashboard(masks[:, :, 0], masks[:, :, 1], masks[:, :, 2])

        cv2.imshow(WINDOW_NAME, dashboard)
        cv2.imshow(MASK_WINDOW_NAME, mask_dashboard)

        key = cv2.waitKey(16) & 0xFF
        if handle_keypress(key, state):
            break

    return 0


def process_live_feed(state: AppState) -> int:
    if not CALIBRATION_FILE.exists():
        print(f"Calibration file not found: {CALIBRATION_FILE}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}", file=sys.stderr)
        return 1

    previous_time = time.perf_counter()
    last_analysis: tuple[np.ndarray, np.ndarray, list[Detection], list[Detection], Detection | None, list[np.ndarray], dict[str, list[int]]] | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Could not read frame from camera", file=sys.stderr)
                return 1

            try:
                undistorted = undistort_with_calibration(frame, str(CALIBRATION_FILE), balance=0.0)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 1

            state.current_frame = undistorted.copy()
            state.frame_index += 1
            settings = get_trackbar_values()
            current_time = time.perf_counter()
            delta = max(current_time - previous_time, 1e-6)
            previous_time = current_time

            refresh_analysis = last_analysis is None or state.frame_index % ANALYSIS_EVERY_N_FRAMES == 0
            if refresh_analysis:
                last_analysis = analyze_frame(
                    state.current_frame,
                    state,
                    settings,
                    force_refresh=True,
                )

            assert last_analysis is not None
            _quantized, masks, white_balls, orange_balls, red_cross, red_borders, cluster_info = last_analysis
            dashboard = build_dashboard(
                state.current_frame,
                state,
                white_balls,
                orange_balls,
                red_cross,
                red_borders,
                fps=1.0 / delta,
                cluster_info=cluster_info,
                k_clusters=int(settings["k"]),
                settings=settings,
            )
            mask_dashboard = build_mask_dashboard(masks[:, :, 0], masks[:, :, 1], masks[:, :, 2])

            cv2.imshow(WINDOW_NAME, dashboard)
            cv2.imshow(MASK_WINDOW_NAME, mask_dashboard)

            key = cv2.waitKey(1) & 0xFF
            if handle_keypress(key, state):
                break
    finally:
        cap.release()

    return 0


def handle_keypress(key: int, state: AppState) -> bool:
    if key in (ord("q"), 27):
        return True

    if key in CLASS_KEYS:
        state.active_class = CLASS_KEYS[key]
        print(f"Active class set to: {state.active_class}")

    return False


def main() -> int:
    cv2.ocl.setUseOpenCL(False)

    state = AppState(
        active_class="white",
        target_labs=build_default_targets(),
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MASK_WINDOW_NAME, cv2.WINDOW_NORMAL)
    create_controls_window()
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, state)

    try:
        if USE_LIVE_FEED:
            return process_live_feed(state)
        return process_static_image(state)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
