#!/usr/bin/env python3
"""Manual live top-down view selection with fisheye undistortion."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camera.imageprocessing import undistort_with_calibration


CALIBRATION_FILE = REPO_ROOT / "calibration_data.npz"
CAMERA_INDEX = 0
LIVE_WINDOW = "Manual Top-Down Selector"
TOPDOWN_WINDOW = "Top-Down View"
TOPDOWN_SIZE = (800, 600)
LOUPE_CROP_SIZE = 40
LOUPE_SCALE = 5
LOUPE_PADDING = 12
POINT_RADIUS = 6


@dataclass
class SelectionState:
    points: list[tuple[int, int]] = field(default_factory=list)
    cursor: tuple[int, int] = (0, 0)
    frame_size: tuple[int, int] = (0, 0)
    transform_matrix: np.ndarray | None = None

    def clear_points(self) -> None:
        self.points.clear()
        self.transform_matrix = None


def order_points(points: list[tuple[int, int]]) -> np.ndarray:
    corners = np.array(points, dtype=np.float32)
    sums = corners.sum(axis=1)
    diffs = np.diff(corners, axis=1).reshape(4)

    top_left = corners[np.argmin(sums)]
    bottom_right = corners[np.argmax(sums)]
    top_right = corners[np.argmin(diffs)]
    bottom_left = corners[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def destination_corners(size: tuple[int, int]) -> np.ndarray:
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


def build_transform(points: list[tuple[int, int]]) -> np.ndarray:
    return cv2.getPerspectiveTransform(order_points(points), destination_corners(TOPDOWN_SIZE))


def clamp_crop_bounds(center: int, crop_size: int, limit: int) -> tuple[int, int]:
    if limit <= crop_size:
        return 0, limit

    half = crop_size // 2
    start = max(0, center - half)
    end = start + crop_size
    if end > limit:
        end = limit
        start = end - crop_size
    return start, end


def draw_loupe(frame: np.ndarray, state: SelectionState) -> np.ndarray:
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


def draw_selection_overlay(frame: np.ndarray, state: SelectionState) -> np.ndarray:
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
        "Left click: add point",
        "Right click or r: reset",
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

    if state.transform_matrix is not None:
        status = "Top-down transform active"
        color = (0, 255, 0)
    else:
        status = "Select 4 inner corners for top-down warp"
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


def make_topdown_placeholder(message: str) -> np.ndarray:
    width, height = TOPDOWN_SIZE
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


def on_mouse(event: int, x: int, y: int, _flags: int, param: SelectionState) -> None:
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
        param.transform_matrix = build_transform(param.points)


def main() -> int:
    cv2.ocl.setUseOpenCL(False)

    if not CALIBRATION_FILE.exists():
        print(f"Calibration file not found: {CALIBRATION_FILE}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}", file=sys.stderr)
        return 1

    state = SelectionState()
    cv2.namedWindow(LIVE_WINDOW)
    cv2.namedWindow(TOPDOWN_WINDOW)
    cv2.setMouseCallback(LIVE_WINDOW, on_mouse, state)

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

            state.frame_size = (int(undistorted.shape[1]), int(undistorted.shape[0]))
            if state.cursor == (0, 0):
                state.cursor = (state.frame_size[0] // 2, state.frame_size[1] // 2)

            live_view = draw_selection_overlay(undistorted, state)

            if state.transform_matrix is not None:
                topdown = cv2.warpPerspective(undistorted, state.transform_matrix, TOPDOWN_SIZE)
            else:
                topdown = make_topdown_placeholder("Waiting for 4 selected points")

            cv2.imshow(LIVE_WINDOW, live_view)
            cv2.imshow(TOPDOWN_WINDOW, topdown)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                state.clear_points()

        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
