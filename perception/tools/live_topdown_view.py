#!/usr/bin/env python3
"""Live top-down perspective transform using fisheye undistortion first."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = REPO_ROOT / "calibration_data.npz"
CAMERA_INDEX = 0
BALANCE = 0.0
LIVE_WINDOW = "LiveFeed"
TOPDOWN_WINDOW = "TopDownView"
MASK_WINDOW = "HSV Mask"
TOPDOWN_SIZE = (1000, 800)

DEFAULT_HSV = {
    "H_MIN": 5,
    "H_MAX": 25,
    "S_MIN": 100,
    "S_MAX": 255,
    "V_MIN": 100,
    "V_MAX": 255,
}


def _noop(_: int) -> None:
    return None


def load_calibration(calibration_file: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    data = np.load(str(calibration_file))
    K = data["K"]
    D = data["D"]
    image_size = tuple(int(v) for v in data["image_size"])
    return K, D, image_size


def configure_camera(cap: cv2.VideoCapture, frame_size: tuple[int, int]) -> None:
    width, height = frame_size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def build_undistort_maps(
    K: np.ndarray,
    D: np.ndarray,
    frame_size: tuple[int, int],
    balance: float,
) -> tuple[np.ndarray, np.ndarray]:
    identity = np.eye(3, dtype=np.float64)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        D,
        frame_size,
        identity,
        balance=balance,
    )
    return cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        identity,
        new_K,
        frame_size,
        cv2.CV_32FC1,
    )


def create_hsv_trackbars() -> None:
    cv2.namedWindow(LIVE_WINDOW, cv2.WINDOW_NORMAL)
    for name, value in DEFAULT_HSV.items():
        max_value = 179 if name.startswith("H_") else 255
        cv2.createTrackbar(name, LIVE_WINDOW, value, max_value, _noop)


def read_hsv_thresholds() -> tuple[np.ndarray, np.ndarray]:
    lower = np.array(
        [
            cv2.getTrackbarPos("H_MIN", LIVE_WINDOW),
            cv2.getTrackbarPos("S_MIN", LIVE_WINDOW),
            cv2.getTrackbarPos("V_MIN", LIVE_WINDOW),
        ],
        dtype=np.uint8,
    )
    upper = np.array(
        [
            cv2.getTrackbarPos("H_MAX", LIVE_WINDOW),
            cv2.getTrackbarPos("S_MAX", LIVE_WINDOW),
            cv2.getTrackbarPos("V_MAX", LIVE_WINDOW),
        ],
        dtype=np.uint8,
    )
    return lower, upper


def build_mask(frame_bgr: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def approximate_polygon(contour: np.ndarray | None) -> np.ndarray | None:
    if contour is None or cv2.contourArea(contour) <= 0:
        return None
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.02 * perimeter
    polygon = cv2.approxPolyDP(contour, epsilon, True)
    return polygon


def order_corners(points: np.ndarray) -> np.ndarray:
    corners = points.reshape(4, 2).astype(np.float32)
    sums = corners.sum(axis=1)
    diffs = np.diff(corners, axis=1).reshape(4)

    top_left = corners[np.argmin(sums)]
    bottom_right = corners[np.argmax(sums)]
    top_right = corners[np.argmin(diffs)]
    bottom_left = corners[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def draw_detected_geometry(frame: np.ndarray, polygon: np.ndarray | None) -> np.ndarray:
    overlay = frame.copy()
    if polygon is None or len(polygon) == 0:
        return overlay

    cv2.polylines(overlay, [polygon], True, (255, 255, 0), 2, cv2.LINE_AA)
    for index, corner in enumerate(polygon.reshape(-1, 2), start=1):
        point = tuple(int(v) for v in corner)
        cv2.circle(overlay, point, 8, (0, 255, 0), -1, cv2.LINE_AA)
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
    return overlay


def add_status(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    cv2.putText(
        frame,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def make_mask_preview(mask: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hsv_text = (
        f"H:[{int(lower[0])},{int(upper[0])}] "
        f"S:[{int(lower[1])},{int(upper[1])}] "
        f"V:[{int(lower[2])},{int(upper[2])}]"
    )
    cv2.putText(
        preview,
        hsv_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "Hvid = valgt orange maske",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return preview


def warp_topdown(frame: np.ndarray, ordered_corners: np.ndarray) -> np.ndarray:
    width, height = TOPDOWN_SIZE
    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(ordered_corners, destination)
    return cv2.warpPerspective(frame, transform, TOPDOWN_SIZE)


def make_topdown_placeholder(message: str) -> np.ndarray:
    width, height = TOPDOWN_SIZE
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
        print(f"Kalibreringsfil ikke fundet: {CALIBRATION_FILE}", file=sys.stderr)
        return 1

    K, D, calibration_size = load_calibration(CALIBRATION_FILE)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Kunne ikke aabne kamera {CAMERA_INDEX}", file=sys.stderr)
        return 1

    configure_camera(cap, calibration_size)
    create_hsv_trackbars()

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Kunne ikke laese foerste frame fra kameraet", file=sys.stderr)
            return 1

        frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        if frame_size != calibration_size:
            print(
                f"Kameraet leverer {frame_size}, men kalibreringen er lavet til {calibration_size}. "
                "Saet kameraet til samme oploesning og proev igen.",
                file=sys.stderr,
            )
            return 1

        map1, map2 = build_undistort_maps(K, D, frame_size, BALANCE)
        cv2.namedWindow(TOPDOWN_WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Kunne ikke laese frame fra kameraet", file=sys.stderr)
                return 1

            undistorted = cv2.remap(
                frame,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            lower, upper = read_hsv_thresholds()
            mask = build_mask(undistorted, lower, upper)
            mask_preview = make_mask_preview(mask, lower, upper)
            contour = find_largest_contour(mask)
            polygon = approximate_polygon(contour)

            live_view = undistorted.copy()
            if contour is not None:
                cv2.drawContours(live_view, [contour], -1, (255, 0, 0), 2)

            corner_count = 0 if polygon is None else int(len(polygon))
            live_view = draw_detected_geometry(live_view, polygon)

            if polygon is not None and corner_count == 4:
                ordered_corners = order_corners(polygon)
                top_down = warp_topdown(undistorted, ordered_corners)
                live_view = add_status(
                    live_view,
                    "STATUS: Geometri OK - Transformering aktiv",
                    (0, 255, 0),
                )
            else:
                live_view = add_status(
                    live_view,
                    f"FEJL: Kun {corner_count} hjoerner fundet - For lidt geometri til transformation",
                    (0, 0, 255),
                )
                top_down = make_topdown_placeholder(
                    f"Venter paa 4 hjoerner. Fundet: {corner_count}"
                )

            cv2.imshow(LIVE_WINDOW, live_view)
            cv2.imshow(TOPDOWN_WINDOW, top_down)
            cv2.imshow(MASK_WINDOW, mask_preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
