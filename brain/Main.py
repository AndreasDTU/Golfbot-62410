"""GolfBot Main GUI — camera + 2D schematic with mode controls.

Single-window OpenCV GUI: live top-down camera feed (left) beside the 2D
schematic field view (right), with a status bar showing mode, pose, and FPS.
View-only for now; robot control will be wired in later stages.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from localization.localization import RobotCalibrationCollector, RobotPoseEstimator
from localization.models import RobotPose
from perception.vision.calibration import HomographyCalibrator
from perception.vision.config import AppConfig, CameraConfig, WindowConfig
from perception.vision.debug import DebugRenderer
from perception.vision.geometry import CoordinateMapper
from perception.vision.models import CalibrationState
from perception.vision.pipeline import VisionPipeline, VisionFrameResult


class AppMode(str, Enum):
    IDLE = "IDLE"
    CALIBRATING = "CALIBRATING"
    MANUAL = "MANUAL"
    AUTO = "AUTO"


WINDOW_NAME = "GolfBot Main"
STATUS_BAR_HEIGHT = 80


@dataclass(frozen=True)
class GuiButton:
    label: str
    action: str
    rect: tuple[int, int, int, int]  # x, y, w, h


@dataclass
class MainGui:
    """OpenCV GUI showing camera feed + 2D schematic side by side."""

    config: AppConfig
    pipeline: VisionPipeline
    pose_estimator: RobotPoseEstimator
    renderer: DebugRenderer
    camera: cv2.VideoCapture | None = None
    static_image: np.ndarray | None = None

    mode: AppMode = AppMode.IDLE
    robot_pose: RobotPose | None = None
    calibration: dict | None = None
    params: dict | None = None
    message: str = "Ready"
    fps: float = 0.0
    closed: bool = False
    _mouse_pos: tuple[int, int] = (0, 0)

    # Dimensions derived from config
    _left_w: int = 0
    _left_h: int = 0
    _right_w: int = 0
    _right_h: int = 0

    def __post_init__(self) -> None:
        self._left_w, self._left_h = self.config.camera.topdown_warp_size
        self._right_w = self.config.windows.schematic_width_px
        self._right_h = self.config.windows.schematic_height_px
        self.params = self.pipeline.default_params()
        self._load_robot_calibration()

    def _load_robot_calibration(self) -> None:
        cal_path = self.config.paths.robot_calibration_file
        if cal_path is not None and cal_path.exists():
            collector = RobotCalibrationCollector(self.config.robot)
            self.calibration = collector.load_robot_calibration(
                cal_path, self.config.camera.topdown_warp_size,
            )
            if self.calibration is not None:
                self.message = "Robot calibration loaded"
            else:
                self.message = "Robot calibration file found but invalid"
        else:
            self.message = "No robot calibration file"

    def _canvas_width(self) -> int:
        return self._left_w + self._right_w

    def _canvas_height(self) -> int:
        return max(self._left_h, self._right_h) + STATUS_BAR_HEIGHT

    def buttons(self) -> list[GuiButton]:
        y = max(self._left_h, self._right_h) + 10
        bw, bh = 100, 30
        gap = 10
        x0 = 20
        return [
            GuiButton("Calibrate", "calibrate", (x0, y, bw, bh)),
            GuiButton("Manual", "manual", (x0 + (bw + gap), y, bw, bh)),
            GuiButton("Auto", "auto", (x0 + 2 * (bw + gap), y, bw, bh)),
            GuiButton("Stop", "stop", (x0 + 3 * (bw + gap), y, bw, bh)),
            GuiButton("Quit", "quit", (x0 + 4 * (bw + gap), y, bw, bh)),
        ]

    def handle_mouse(self, event: int, x: int, y: int, _flags: int, _userdata) -> None:
        self._mouse_pos = (x, y)
        if event != cv2.EVENT_LBUTTONUP:
            return

        # Check buttons
        for button in self.buttons():
            bx, by, bw, bh = button.rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self._handle_button(button.action)
                return

        # During calibration, clicks on the left panel go to the HomographyCalibrator
        if self.mode == AppMode.CALIBRATING and x < self._left_w:
            calibrator = self.pipeline.preprocessor.homography_calibrator
            calibrator.add_manual_point((x, y))
            n = len(calibrator.manual_points)
            self.message = f"Manual point {n}/4 added"
            if n >= 4:
                self.message = "Manual calibration complete"
                self.mode = AppMode.IDLE

    def _handle_button(self, action: str) -> None:
        if action == "calibrate":
            if self.mode == AppMode.CALIBRATING:
                self.mode = AppMode.IDLE
                self.message = "Calibration cancelled"
            else:
                self.mode = AppMode.CALIBRATING
                calibrator = self.pipeline.preprocessor.homography_calibrator
                calibrator.manual_points.clear()
                calibrator.calibration_state = CalibrationState.NEEDS_CALIBRATION
                calibrator.transform_matrix = None
                self.message = "Click 4 field corners: TL, TR, BR, BL"
        elif action == "manual":
            self.mode = AppMode.MANUAL
            self.message = "Manual mode (view only)"
        elif action == "auto":
            self.mode = AppMode.AUTO
            self.message = "Auto mode (view only)"
        elif action == "stop":
            self.mode = AppMode.IDLE
            self.message = "Stopped"
        elif action == "quit":
            self.closed = True

    def _process_frame(self, raw_frame: np.ndarray) -> VisionFrameResult:
        use_aruco = self.mode != AppMode.CALIBRATING
        return self.pipeline.process(raw_frame, params=self.params, use_aruco=use_aruco)

    def _estimate_pose(self, result: VisionFrameResult) -> None:
        topdown = result.preprocessed.topdown
        if topdown is None or self.params is None:
            self.robot_pose = None
            return
        pose, _origin_px, _obs, _parallax = self.pose_estimator.estimate(
            topdown, self.params, self.calibration,
        )
        self.robot_pose = pose

    def _build_left_panel(self, result: VisionFrameResult) -> np.ndarray:
        topdown = result.preprocessed.topdown
        if topdown is not None:
            left = self.renderer.annotate_camera_frame(
                topdown,
                result.red_zones,
                result.white_balls,
                result.orange_balls,
                self.fps,
            )
        else:
            left = self.renderer.make_topdown_placeholder("No top-down warp available")

        if self.mode == AppMode.CALIBRATING:
            calibrator = self.pipeline.preprocessor.homography_calibrator
            # Draw existing manual points
            for i, pt in enumerate(calibrator.manual_points):
                cv2.circle(left, pt, 6, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    left, str(i + 1), (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA,
                )
            # Draw instruction
            n = len(calibrator.manual_points)
            labels = ["TL", "TR", "BR", "BL"]
            if n < 4:
                cv2.putText(
                    left, f"Click corner {n + 1}/4: {labels[n]}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
                )
            # Draw loupe for precision clicking
            if self._mouse_pos[0] < self._left_w:
                left = self.renderer.draw_loupe(left, self._mouse_pos)

        # Resize to target dimensions if needed
        if left.shape[1] != self._left_w or left.shape[0] != self._left_h:
            left = cv2.resize(left, (self._left_w, self._left_h), interpolation=cv2.INTER_LINEAR)
        return left

    def _build_right_panel(self, result: VisionFrameResult) -> np.ndarray:
        frame_shape = result.preprocessed.topdown.shape if result.preprocessed.topdown is not None else (self._left_h, self._left_w, 3)
        camera_center = (
            float(self.params.get("camera_center_x", self._left_w / 2)),
            float(self.params.get("camera_center_y", self._left_h / 2)),
        )
        schematic = self.renderer.draw_schematic(
            frame_shape=frame_shape,
            red_zones=result.red_zones,
            smoothed_ball_coordinates=result.smoothed_ball_coordinates,
            camera_center_pixels=camera_center,
            robot_pose=self.robot_pose,
            params=self.params,
        )
        return schematic

    def _draw_status_bar(self, canvas: np.ndarray) -> None:
        y0 = max(self._left_h, self._right_h)
        # Background
        cv2.rectangle(canvas, (0, y0), (canvas.shape[1], canvas.shape[0]), (50, 50, 50), -1)

        # Status text
        cal_state = self.pipeline.preprocessor.homography_calibrator.calibration_state.value
        pose_text = "N/A"
        if self.robot_pose is not None:
            p = self.robot_pose
            pose_text = f"({p.x_cm:.1f}, {p.y_cm:.1f}) heading {math.degrees(p.heading_rad):.1f} deg"

        status = f"Cal: {cal_state} | Pose: {pose_text} | Mode: {self.mode.value} | FPS: {self.fps:.1f}"
        cv2.putText(
            canvas, status, (20, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, self.message, (20, y0 + 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "Keys: q/Esc quit | c calibrate | m manual | a auto | s stop",
            (20, y0 + 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA,
        )

        # Buttons
        for button in self.buttons():
            bx, by, bw, bh = button.rect
            is_active = (
                (button.action == "calibrate" and self.mode == AppMode.CALIBRATING)
                or (button.action == "manual" and self.mode == AppMode.MANUAL)
                or (button.action == "auto" and self.mode == AppMode.AUTO)
                or (button.action == "stop" and self.mode == AppMode.IDLE)
            )
            fill = (80, 160, 80) if is_active else (90, 90, 90)
            border = (200, 200, 200)
            text_color = (255, 255, 255)

            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), fill, -1, cv2.LINE_AA)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border, 1, cv2.LINE_AA)
            (tw, _th), _ = cv2.getTextSize(button.label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tx = bx + (bw - tw) // 2
            ty = by + 20
            cv2.putText(canvas, button.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    def _read_frame(self) -> np.ndarray | None:
        if self.static_image is not None:
            return self.static_image.copy()
        if self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                return frame
        return None

    def tick(self) -> np.ndarray:
        """Process one frame and return the composited canvas."""
        frame_start = time.perf_counter()

        raw_frame = self._read_frame()
        if raw_frame is not None:
            result = self._process_frame(raw_frame)
            self._estimate_pose(result)
            left = self._build_left_panel(result)
            right = self._build_right_panel(result)
        else:
            left = self.renderer.make_topdown_placeholder("No camera input")
            if left.shape[1] != self._left_w or left.shape[0] != self._left_h:
                left = cv2.resize(left, (self._left_w, self._left_h))
            right = np.full((self._right_h, self._right_w, 3), (40, 100, 40), dtype=np.uint8)
            cv2.putText(
                right, "No camera input", (30, self._right_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA,
            )

        # Match heights if they differ
        if left.shape[0] != right.shape[0]:
            target_h = max(left.shape[0], right.shape[0])
            if left.shape[0] != target_h:
                scale = target_h / left.shape[0]
                left = cv2.resize(left, (int(left.shape[1] * scale), target_h))
            if right.shape[0] != target_h:
                scale = target_h / right.shape[0]
                right = cv2.resize(right, (int(right.shape[1] * scale), target_h))

        panels = np.hstack([left, right])

        # Add status bar
        canvas = np.zeros((panels.shape[0] + STATUS_BAR_HEIGHT, panels.shape[1], 3), dtype=np.uint8)
        canvas[:panels.shape[0], :panels.shape[1]] = panels
        self._draw_status_bar(canvas)

        dt = time.perf_counter() - frame_start
        self.fps = 1.0 / max(dt, 1e-6)
        return canvas

    def handle_key(self, key: int) -> None:
        if key in (27, ord("q")):
            self.closed = True
        elif key == ord("c"):
            self._handle_button("calibrate")
        elif key == ord("m"):
            self._handle_button("manual")
        elif key == ord("a"):
            self._handle_button("auto")
        elif key == ord("s"):
            self._handle_button("stop")

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        total_w = self._canvas_width()
        total_h = self._canvas_height()
        cv2.resizeWindow(WINDOW_NAME, total_w, total_h)
        cv2.setMouseCallback(WINDOW_NAME, self.handle_mouse)

        while not self.closed:
            canvas = self.tick()
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self.handle_key(key)
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self.closed = True

        if self.camera is not None:
            self.camera.release()
        cv2.destroyWindow(WINDOW_NAME)

    def close(self) -> None:
        self.closed = True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GolfBot Main GUI — camera + schematic view.")
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera device index (default from CameraConfig).",
    )
    parser.add_argument(
        "--no-camera", action="store_true",
        help="Run without camera (schematic-only testing).",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Use a static image instead of live camera.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    config = AppConfig.from_repo_root(REPO_ROOT)

    # Build pipeline
    try:
        pipeline = VisionPipeline(app_config=config, normalize_illumination=True)
    except Exception as exc:
        print(f"Warning: Could not build full pipeline ({exc}). Trying without YOLO.", file=sys.stderr)
        pipeline = VisionPipeline(
            app_config=config,
            ball_detector=_NullBallDetector(),
            normalize_illumination=True,
        )

    mapper = pipeline.mapper
    pose_estimator = RobotPoseEstimator(
        config.field, config.robot, mapper,
    )
    renderer = DebugRenderer(
        config.field, config.windows, config.robot, config.drive, mapper,
    )

    # Camera or static image
    camera = None
    static_image = None

    if args.image is not None:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Could not read image: {args.image}", file=sys.stderr)
            return 1
        static_image = img
    elif not args.no_camera:
        cam_index = args.camera if args.camera is not None else config.camera.camera_index
        camera = cv2.VideoCapture(cam_index)
        if not camera.isOpened():
            print(f"Could not open camera {cam_index}", file=sys.stderr)
            return 1

    gui = MainGui(
        config=config,
        pipeline=pipeline,
        pose_estimator=pose_estimator,
        renderer=renderer,
        camera=camera,
        static_image=static_image,
    )
    gui.run()
    return 0


class _NullBallDetector:
    """Fallback detector when YOLO model is unavailable."""

    def detect(self, frame, params, camera_center):
        return [], [], {"white": np.zeros(frame.shape[:2], dtype=np.uint8),
                        "orange": np.zeros(frame.shape[:2], dtype=np.uint8)}


if __name__ == "__main__":
    raise SystemExit(main())
