#!/usr/bin/env python3
"""Manual live top-down view selection with fisheye undistortion."""

from __future__ import annotations

import argparse
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
VIDEOS_DIR = REPO_ROOT / "videos"
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
        "Space: pause/resume",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually select four corners and view the undistorted top-down warp."
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Read frames from a video file instead of the default camera. Bare names are loaded from videos/.",
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
    data = np.load(str(calibration_file))
    return tuple(int(value) for value in data["image_size"])


def resolve_video_path(video_path: Path) -> Path:
    if video_path.exists():
        return video_path

    repo_path = REPO_ROOT / video_path
    if repo_path.exists():
        return repo_path

    videos_path = VIDEOS_DIR / video_path
    if videos_path.exists():
        return videos_path

    return video_path


def open_capture(video_path: Path | None) -> tuple[cv2.VideoCapture, str]:
    if video_path is None:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        return cap, f"camera {CAMERA_INDEX}"

    cap = cv2.VideoCapture(str(video_path))
    return cap, str(video_path)


def playback_delay_ms(cap: cv2.VideoCapture, is_video: bool) -> int:
    if not is_video:
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        return 30
    return max(1, int(round(1000 / fps)))


def resize_frame_to_size(frame_bgr: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
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


def main() -> int:
    args = parse_args()
    cv2.ocl.setUseOpenCL(False)

    if not CALIBRATION_FILE.exists():
        print(f"Calibration file not found: {CALIBRATION_FILE}", file=sys.stderr)
        return 1

    video_path = resolve_video_path(args.video) if args.video is not None else None
    if video_path is not None and not video_path.exists():
        print(
            f"Video file not found: {args.video} "
            f"(also checked {VIDEOS_DIR / args.video})",
            file=sys.stderr,
        )
        return 1
    if args.resize_video_to_calibration and video_path is None:
        print("--resize-video-to-calibration only applies when --video is used.", file=sys.stderr)
        return 1

    resize_to_size = (
        load_calibration_image_size(CALIBRATION_FILE)
        if args.resize_video_to_calibration
        else None
    )

    cap, source_name = open_capture(video_path)
    if not cap.isOpened():
        print(f"Could not open {source_name}", file=sys.stderr)
        return 1

    wait_delay = playback_delay_ms(cap, video_path is not None)
    state = SelectionState()
    paused = False
    current_frame: np.ndarray | None = None
    resize_notice_shown = False
    cv2.namedWindow(LIVE_WINDOW)
    cv2.namedWindow(TOPDOWN_WINDOW)
    cv2.setMouseCallback(LIVE_WINDOW, on_mouse, state)

    try:
        while True:
            if not paused or current_frame is None:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if video_path is not None:
                        break
                    print(f"Could not read frame from {source_name}", file=sys.stderr)
                    return 1
                if resize_to_size is not None:
                    original_size = (int(frame.shape[1]), int(frame.shape[0]))
                    frame = resize_frame_to_size(frame, resize_to_size)
                    if original_size != resize_to_size and not resize_notice_shown:
                        print(
                            f"Resizing video frames from {original_size} to {resize_to_size} "
                            "before undistortion."
                        )
                        resize_notice_shown = True
                current_frame = frame

            try:
                undistorted = undistort_with_calibration(current_frame, str(CALIBRATION_FILE))
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

            key = cv2.waitKey(wait_delay) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                state.clear_points()
            if key == ord(" "):
                paused = not paused

        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
