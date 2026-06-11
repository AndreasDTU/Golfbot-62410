#!/usr/bin/env python3
"""Top-down detector application shell.

This script is now intentionally small.  Domain behavior lives in the extracted
packages under ``vision/``, ``robot/``, and ``pathfinding/``.  The shell owns
only CLI parsing, OpenCV UI setup, service wiring, frame loops, and keyboard
dispatch.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import json
import math
import multiprocessing as mp
import queue
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathfinding.models import HybridPose, PlannedBallTarget, RoutePlan
from pathfinding.planner import RoutePlanningFacade
from robot.control import (
    DriveSafetyGuard,
    closest_route_index,
    distance_to_route_goal_cm,
    robot_body_edge_clearance_cm,
    route_goal_pose,
)
from robot.controller import RobotController
from robot.drive_calibration import (
    DriveCalibrationValues,
    projected_motion_cm,
    suggest_axle_track_mm,
    suggest_mm_per_unit,
    unwrapped_heading_delta,
)
from robot.io import TcpWheelDispatcher
from robot.localization import RobotCalibrationCollector, RobotPoseEstimator, image_yaw_rotation_matrix, normalize_angle
from robot.models import (
    DriveControlState,
    DriveRuntime,
    RobotCalibrationPhase,
    RobotCalibrationRuntime,
    RobotGeometry,
    RobotPose,
    WheelCommand,
)
from vision.calibration import HomographyCalibrator
from vision.config import AppConfig
from vision.debug import DebugRenderer
from vision.geometry import CoordinateMapper
from vision.grid_mapping import OccupancyGridBuilder
from vision.models import CalibrationState, CameraGroundProjection, HSVRange, SmoothedBallCoordinate
from vision.pipeline import VisionFrameResult, VisionPipeline
from vision.preprocessing import PreprocessedFrame


def noop(_value: int) -> None:
    """Trackbar callback placeholder matching the legacy OpenCV API."""


class PickupExecutionState(Enum):
    """Deterministic pickup handoff states for the live drive loop."""

    NAVIGATION = "NAVIGATION"
    BLIND_APPROACH = "BLIND_APPROACH"
    PICKUP_ASSIST = "PICKUP_ASSIST"
    REPLAN = "REPLAN"
    POST_PICKUP_ESCAPE = "POST_PICKUP_ESCAPE"
    POST_PICKUP_ALIGN = "POST_PICKUP_ALIGN"


class UnloadExecutionState(Enum):
    """Minimal terminal unload states for the live drive loop."""

    IDLE = "IDLE"
    PRE_UNLOAD_PIVOT = "PRE_UNLOAD_PIVOT"
    ALIGNING = "UNLOAD_ALIGNING"
    SETTLING = "UNLOAD_SETTLING"
    READY = "UNLOAD_READY"
    UNLOADING_FIRST = "UNLOADING_FIRST"
    PIPE_SHAKE = "PIPE_SHAKE"
    UNLOADING_SECOND = "UNLOADING_SECOND"
    COMPLETE = "UNLOAD_COMPLETE"
    FAILED = "UNLOAD_FAILED"


class CollectorPositionState(Enum):
    """Best-known collector height state for autonomous drive gating."""

    UNKNOWN = "UNKNOWN"
    TRAVEL = "TRAVEL"
    PICKUP_ASSIST = "PICKUP_ASSIST"
    UNLOADING = "UNLOADING"


class DriveCalibrationExecutionState(Enum):
    """Optional live drive calibration sequence state."""

    IDLE = "IDLE"
    TURNING = "TURNING"
    TURN_SETTLING = "TURN_SETTLING"
    MOVING = "MOVING"
    MOVE_SETTLING = "MOVE_SETTLING"
    READY_TO_APPLY = "READY_TO_APPLY"
    FAILED = "FAILED"
    CANCELLING = "CANCELLING"


BALL_COUNT_HISTORY_FRAMES = 15
INITIAL_BALL_COUNT_STABLE_S = 0.75


@dataclass
class RuntimeState:
    """Mutable UI/application state owned by the OpenCV shell."""

    selected_ball_track_id: int | None = None
    selected_start_cm: tuple[int, int] | None = None
    route_plan: RoutePlan = field(default_factory=lambda: RoutePlan(points=[], active_target=None, pickup_poses=[]))
    route_cache_target_id: int | None = None
    route_cache_target_label: str | None = None
    route_cache_target_cm: tuple[float, float] | None = None
    route_cache_ball_signature: tuple[tuple[int, str, int, int], ...] = field(default_factory=tuple)
    route_cache_unload_extension_cm: float | None = None
    route_failed_ball_signature: tuple[tuple[int, str, int, int], ...] | None = None
    route_failed_unload_extension_cm: float | None = None
    route_failed_start_signature: tuple[int, int, int] | None = None
    robot_pose: RobotPose | None = None
    robot_topdown_px: tuple[float, float] | None = None
    pose_missing_frames: int = 0
    camera_focus_locked: bool = False
    latest_smoothed_balls: list[SmoothedBallCoordinate] = field(default_factory=list)
    pickup_state: PickupExecutionState = PickupExecutionState.NAVIGATION
    pickup_state_started_s: float = 0.0
    pickup_command_fired: bool = False
    near_zone_move_fired: bool = False
    near_zone_turn_fired: bool = False
    near_zone_alignment_best_error_cm: float = float("inf")
    near_zone_alignment_stale_frames: int = 0
    near_zone_alignment_iterations: int = 0
    near_zone_alignment_settle_started_s: float = 0.0
    near_zone_alignment_settle_reason: str = ""
    post_pickup_escape_fired: bool = False
    unload_state: UnloadExecutionState = UnloadExecutionState.IDLE
    unload_command_fired: bool = False
    unload_started_s: float = 0.0
    unload_alignment_best_error_cm: float = float("inf")
    unload_alignment_stale_frames: int = 0
    unload_alignment_iterations: int = 0
    unload_alignment_settle_started_s: float = 0.0
    unload_alignment_settle_reason: str = ""
    collector_state: CollectorPositionState = CollectorPositionState.UNKNOWN
    initial_total_balls: int | None = None
    balls_collected: int = 0
    stable_visible_balls: int | None = None
    step_mode_enabled: bool = False
    step_drive_paused: bool = False
    drive_calibration_state: DriveCalibrationExecutionState = DriveCalibrationExecutionState.IDLE
    drive_calibration_message: str = ""
    drive_calibration_values: DriveCalibrationValues | None = None
    drive_calibration_suggested_values: DriveCalibrationValues | None = None
    drive_calibration_turn_actual_deg: float | None = None
    drive_calibration_distance_actual_cm: float | None = None
    drive_calibration_lateral_drift_cm: float | None = None
    drive_calibration_turn_start_heading_rad: float = 0.0
    drive_calibration_turn_last_heading_rad: float | None = None
    drive_calibration_turn_accumulated_rad: float = 0.0
    drive_calibration_distance_start_pose: RobotPose | None = None
    drive_calibration_settle_started_s: float = 0.0
    visible_ball_count_history: deque[tuple[float, int]] = field(
        default_factory=lambda: deque(maxlen=BALL_COUNT_HISTORY_FRAMES)
    )

    def clear_route(self) -> None:
        self.route_plan = RoutePlan(points=[], active_target=None, pickup_poses=[])
        self.route_cache_target_id = None
        self.route_cache_target_label = None
        self.route_cache_target_cm = None
        self.route_cache_ball_signature = ()
        self.route_cache_unload_extension_cm = None
        self.route_failed_ball_signature = None
        self.route_failed_unload_extension_cm = None
        self.route_failed_start_signature = None

    def reset_pickup_state(self) -> None:
        self.pickup_state = PickupExecutionState.NAVIGATION
        self.pickup_state_started_s = 0.0
        self.pickup_command_fired = False
        self.near_zone_move_fired = False
        self.near_zone_turn_fired = False
        self.near_zone_alignment_best_error_cm = float("inf")
        self.near_zone_alignment_stale_frames = 0
        self.near_zone_alignment_iterations = 0
        self.near_zone_alignment_settle_started_s = 0.0
        self.near_zone_alignment_settle_reason = ""
        self.post_pickup_escape_fired = False

    def reset_unload_alignment_state(self) -> None:
        self.unload_alignment_best_error_cm = float("inf")
        self.unload_alignment_stale_frames = 0
        self.unload_alignment_iterations = 0
        self.unload_alignment_settle_started_s = 0.0
        self.unload_alignment_settle_reason = ""

    def reset_drive_calibration_state(self) -> None:
        self.drive_calibration_state = DriveCalibrationExecutionState.IDLE
        self.drive_calibration_message = ""
        self.drive_calibration_values = None
        self.drive_calibration_suggested_values = None
        self.drive_calibration_turn_actual_deg = None
        self.drive_calibration_distance_actual_cm = None
        self.drive_calibration_lateral_drift_cm = None
        self.drive_calibration_turn_start_heading_rad = 0.0
        self.drive_calibration_turn_last_heading_rad = None
        self.drive_calibration_turn_accumulated_rad = 0.0
        self.drive_calibration_distance_start_pose = None
        self.drive_calibration_settle_started_s = 0.0


@dataclass(frozen=True)
class RoutePlanningRequest:
    """Immutable route-planning job submitted from the UI loop."""

    request_id: int
    grid: np.ndarray
    ball_targets: list[PlannedBallTarget]
    start_pose: HybridPose
    geometry: RobotGeometry
    ball_signature: tuple[tuple[int, str, int, int], ...]
    start_signature: tuple[int, int, int]
    unload_extension_cm: float


@dataclass(frozen=True)
class RoutePlanningResult:
    """Completed route-planning job returned by the background worker."""

    request: RoutePlanningRequest
    route_plan: RoutePlan
    duration_ms: float
    error: str | None = None


def route_planning_worker(
    request_queue: mp.Queue,
    result_queue: mp.Queue,
    config: AppConfig,
) -> None:
    """Run Hybrid A* requests in a separate process."""
    cv2.ocl.setUseOpenCL(False)
    route_facade = RoutePlanningFacade(config.field, config.robot, config.planner)
    while True:
        request = request_queue.get()
        if request is None:
            return
        start = time.perf_counter()
        try:
            route_plan = route_facade.plan_route(
                request.grid,
                request.ball_targets,
                request.start_pose,
                request.geometry,
            )
            result = RoutePlanningResult(
                request=request,
                route_plan=route_plan,
                duration_ms=(time.perf_counter() - start) * 1000.0,
            )
        except Exception as exc:
            result = RoutePlanningResult(
                request=request,
                route_plan=RoutePlan(points=[], active_target=None, pickup_poses=[]),
                duration_ms=(time.perf_counter() - start) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        result_queue.put(result)


class AsyncRoutePlanner:
    """Run Hybrid A* planning in a separate process via queues."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._context = mp.get_context("spawn")
        self._request_queue: mp.Queue = self._context.Queue(maxsize=1)
        self._result_queue: mp.Queue = self._context.Queue(maxsize=1)
        self._process: mp.Process | None = None
        self._busy = False

    def is_busy(self) -> bool:
        if self._process is None:
            return False
        if self._process is not None and not self._process.is_alive():
            self._busy = False
            self._start_worker()
        return self._busy

    def submit(self, request: RoutePlanningRequest) -> bool:
        if self.is_busy():
            return False
        if self._process is None or not self._process.is_alive():
            self._start_worker()
        try:
            self._request_queue.put_nowait(request)
        except queue.Full:
            return False
        self._busy = True
        return True

    def poll_completed(self) -> RoutePlanningResult | None:
        try:
            result = self._result_queue.get_nowait()
        except queue.Empty:
            return None
        self._busy = False
        return result

    def close(self) -> None:
        if self._process is None:
            return
        try:
            self._request_queue.put_nowait(None)
        except queue.Full:
            pass
        self._process.join(timeout=0.5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=0.5)

    def _start_worker(self) -> None:
        self._process = self._context.Process(
            target=route_planning_worker,
            args=(self._request_queue, self._result_queue, self.config),
            name="hybrid-route-planner",
            daemon=True,
        )
        self._process.start()


class IdentityPreprocessor:
    """Preprocessor for still images that are already top-down frames."""

    def __init__(self, calibration_state: CalibrationState = CalibrationState.CALIBRATED_MANUAL) -> None:
        self.calibration_state = calibration_state

    def process(
        self,
        frame: np.ndarray,
        use_aruco: bool = True,
        normalize_illumination: bool | None = None,
    ) -> PreprocessedFrame:
        return PreprocessedFrame(
            undistorted=frame,
            topdown=frame,
            normalized=frame,
            calibration_state=self.calibration_state,
            transform_matrix=np.eye(3, dtype=np.float32),
            camera_ground_projection=None,
            homography_result=None,
        )


class TopdownDetectorApp:
    """OpenCV application shell for the extracted detector stack."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig.from_repo_root(REPO_ROOT)
        self.mapper = CoordinateMapper(self.config.field, self.config.camera, self.config.windows)
        self.renderer = DebugRenderer(self.config.field, self.config.windows, self.config.robot, self.config.drive, self.mapper)
        self.route_facade = RoutePlanningFacade(self.config.field, self.config.robot, self.config.planner)
        self.async_route_planner = AsyncRoutePlanner(self.config)
        self.route_request_counter = 0
        self.robot_estimator = RobotPoseEstimator(self.config.field, self.config.robot, self.mapper)
        self.robot_calibration_collector = RobotCalibrationCollector(self.config.robot, self.config.robot_calibration)
        self.drive_guard = DriveSafetyGuard(self.config.drive, self.route_facade)
        self.occupancy_builder = OccupancyGridBuilder(self.config.field, self.config.robot, self.mapper)
        self.runtime = RuntimeState()
        self.robot_runtime = RobotCalibrationRuntime()
        self.homography_calibrator: HomographyCalibrator | None = None
        self.latest_selector_frame: np.ndarray | None = None
        self.latest_camera_ground_projection: CameraGroundProjection | None = None
        self.latest_camera_ground_warning: str = ""
        self.active_pipeline: VisionPipeline | None = None
        self.latest_result: VisionFrameResult | None = None
        self.latest_params: dict[str, object] | None = None
        self.pickup_thread: threading.Thread | None = None
        self.pickup_last_error: str = ""
        self.unload_thread: threading.Thread | None = None
        self.unload_last_error: str = ""
        self.drive_calibration_thread: threading.Thread | None = None
        self.drive_calibration_thread_error: str = ""
        self.controller: RobotController | None = None

    def _controller(self):
        if self.controller is None:
            self.controller = RobotController(self.config.drive.robot_ip, port=self.config.drive.robot_tcp_port)
        return self.controller

    @staticmethod
    def parse_args() -> argparse.Namespace:
        default_paths = AppConfig.from_repo_root(REPO_ROOT).paths
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
            default=default_paths.default_image,
            help=f"Path to a top-down test image. Default: {default_paths.default_image}",
        )
        mode_group.add_argument(
            "--video",
            type=Path,
            help=f"Path to a recorded camera video. Store local recordings under {default_paths.default_video_dir}.",
        )
        parser.add_argument(
            "--camera-index",
            type=int,
            default=0,
            help="OpenCV camera index for live mode. Default: 0",
        )
        parser.add_argument(
            "--balance",
            type=float,
            default=0.0,
            help="Fisheye undistortion balance used for live/video camera frames.",
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
        parser.add_argument(
            "--drive",
            action="store_true",
            help="Enable TCP dispatch of autonomous left/right wheel speeds.",
        )
        parser.add_argument(
            "--step",
            action="store_true",
            help="With --drive, pause before each autonomous target and advance on the 'n' key.",
        )
        return parser.parse_args()

    def _trackbar_defaults(self) -> dict[str, int]:
        return self.config.detection.trackbar_defaults(self.config.field, self.config.robot)

    def _trackbar_max_value(self, key: str) -> int:
        if "_h_" in key:
            return 179
        if key.endswith("min_area") or key == "yolo_max_area":
            return 20000
        if key == "yolo_conf_pct":
            return 100
        if key == "cam_height_cm":
            return 300
        if key == "calib_z_cm":
            return 30
        if key == "cam_center_x":
            return self.config.field.grid_width_cm
        if key == "cam_center_y":
            return self.config.field.grid_height_cm
        if key == "heading_tuning":
            return 360
        if key in {"robot_width_cmx10", "robot_front_cmx10", "robot_rear_cmx10", "tube_forward_cmx10", "unload_extension_cmx10"}:
            return 500
        if key == "tube_right_cmx10":
            return 1000
        return 255

    def create_trackbars(self, frame_size: tuple[int, int]) -> None:
        """Create trackbars for red zones, camera geometry, and robot geometry."""
        _frame_width, _frame_height = frame_size
        for window_name in (
            self.config.windows.control_color_window_name,
            self.config.windows.control_filter_window_name,
            self.config.windows.control_geometry_window_name,
        ):
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, *self.config.windows.control_window_size)

        defaults = self._trackbar_defaults()
        windows = self.config.trackbars.windows(self.config.windows)
        for key, value in defaults.items():
            cv2.createTrackbar(
                self.config.trackbars.names[key],
                windows[key],
                value,
                self._trackbar_max_value(key),
                noop,
            )

    def get_trackbar_value(self, key: str) -> int:
        return cv2.getTrackbarPos(
            self.config.trackbars.names[key],
            self.config.trackbars.windows(self.config.windows)[key],
        )

    def set_trackbar_value_if_changed(self, key: str, value: int) -> None:
        """Update a trackbar only when needed to avoid needless UI churn."""
        value = int(np.clip(value, 0, self._trackbar_max_value(key)))
        if self.get_trackbar_value(key) == value:
            return
        cv2.setTrackbarPos(
            self.config.trackbars.names[key],
            self.config.trackbars.windows(self.config.windows)[key],
            value,
        )

    def sync_camera_ground_trackbars(self, result_or_preprocessed: VisionFrameResult | PreprocessedFrame) -> None:
        """Mirror the homography-derived camera ground projection into geometry controls."""
        projection = (
            result_or_preprocessed.preprocessed.camera_ground_projection
            if isinstance(result_or_preprocessed, VisionFrameResult)
            else result_or_preprocessed.camera_ground_projection
        )
        if projection is None:
            self.latest_camera_ground_projection = None
            self.latest_camera_ground_warning = ""
            return
        self.latest_camera_ground_projection = projection
        self.latest_camera_ground_warning = ""
        field_x_cm, field_y_cm = self.mapper.topdown_px_to_field_cm(projection.camera_center_px)
        self.set_trackbar_value_if_changed("cam_center_x", int(round(field_x_cm)))
        self.set_trackbar_value_if_changed("cam_center_y", int(round(field_y_cm)))

    def read_params(self) -> dict[str, object]:
        red_1 = HSVRange(
            lower=np.array(
                [self.get_trackbar_value("red1_h_min"), self.get_trackbar_value("red_s_min"), self.get_trackbar_value("red_v_min")],
                dtype=np.uint8,
            ),
            upper=np.array(
                [self.get_trackbar_value("red1_h_max"), self.get_trackbar_value("red_s_max"), self.get_trackbar_value("red_v_max")],
                dtype=np.uint8,
            ),
        )
        red_2 = HSVRange(
            lower=np.array(
                [self.get_trackbar_value("red2_h_min"), self.get_trackbar_value("red_s_min"), self.get_trackbar_value("red_v_min")],
                dtype=np.uint8,
            ),
            upper=np.array(
                [self.get_trackbar_value("red2_h_max"), self.get_trackbar_value("red_s_max"), self.get_trackbar_value("red_v_max")],
                dtype=np.uint8,
            ),
        )
        camera_center_px = self.mapper.field_cm_to_topdown_pixel(
            (float(self.get_trackbar_value("cam_center_x")), float(self.get_trackbar_value("cam_center_y")))
        )
        return {
            "red_1": red_1,
            "red_2": red_2,
            "red_min_area": float(self.get_trackbar_value("red_min_area")),
            "yolo_confidence": float(self.get_trackbar_value("yolo_conf_pct")) / 100.0,
            "yolo_min_area": float(self.get_trackbar_value("yolo_min_area")),
            "yolo_max_area": float(self.get_trackbar_value("yolo_max_area")),
            "h_cam_cm": float(self.get_trackbar_value("cam_height_cm")),
            "z_calib_cm": float(self.get_trackbar_value("calib_z_cm")),
            "camera_center_x": float(camera_center_px[0]),
            "camera_center_y": float(camera_center_px[1]),
            "camera_center_x_cm": float(self.get_trackbar_value("cam_center_x")),
            "camera_center_y_cm": float(self.get_trackbar_value("cam_center_y")),
            "heading_tuning_rad": math.radians(float(self.get_trackbar_value("heading_tuning")) - 180.0),
            "robot_width_cm": float(self.get_trackbar_value("robot_width_cmx10")) / 10.0,
            "robot_front_cm": float(self.get_trackbar_value("robot_front_cmx10")) / 10.0,
            "robot_rear_cm": float(self.get_trackbar_value("robot_rear_cmx10")) / 10.0,
            "tube_forward_cm": float(self.get_trackbar_value("tube_forward_cmx10")) / 10.0,
            "tube_right_cm": float(self.get_trackbar_value("tube_right_cmx10")) / 10.0 - 50.0,
            "unload_extension_cm": float(self.get_trackbar_value("unload_extension_cmx10")) / 10.0,
        }

    def build_image_pipeline(self) -> VisionPipeline:
        pipeline = VisionPipeline(
            self.config,
            preprocessor=IdentityPreprocessor(),
            mapper=self.mapper,
            build_legacy_dilated_grid=False,
        )
        self.active_pipeline = pipeline
        return pipeline

    def build_stream_pipeline(self, balance: float) -> VisionPipeline:
        pipeline = VisionPipeline(self.config, mapper=self.mapper, build_legacy_dilated_grid=False)
        # Rebuild default preprocessor with requested fisheye balance.
        from vision.calibration import UndistortionProvider
        from vision.preprocessing import FramePreprocessor

        self.homography_calibrator = HomographyCalibrator(self.config.field, self.config.camera, self.mapper)
        self.homography_calibrator.start_manual_calibration()
        pipeline.preprocessor = FramePreprocessor(
            UndistortionProvider(self.config.paths.calibration_file, balance),
            self.homography_calibrator,
            self.config.camera,
            normalize_illumination=False,
        )
        self.active_pipeline = pipeline
        return pipeline

    def _make_drive_runtime(self, drive_enabled: bool) -> tuple[DriveRuntime, TcpWheelDispatcher | None]:
        dispatcher = (
            TcpWheelDispatcher(controller=self._controller(), drive_config=self.config.drive)
            if drive_enabled
            else None
        )
        return DriveRuntime(enabled=True, dispatcher=dispatcher), dispatcher

    def _pickup_distance_to_goal_cm(self) -> float:
        if self.runtime.robot_pose is None or len(self.runtime.route_plan.points) < 2:
            return float("inf")
        tracking_error = self.route_facade.compute_route_tracking_error(
            self.runtime.robot_pose,
            self.runtime.route_plan.points,
        )
        if tracking_error is None:
            return float("inf")
        return distance_to_route_goal_cm(
            self.runtime.robot_pose,
            self.runtime.route_plan.points,
            tracking_error,
            self.runtime.route_plan.pickup_poses,
            include_final=False,
        )

    def _pickup_goal_pose(self) -> HybridPose | None:
        if self.runtime.robot_pose is None or len(self.runtime.route_plan.points) < 2:
            return None
        tracking_error = self.route_facade.compute_route_tracking_error(
            self.runtime.robot_pose,
            self.runtime.route_plan.points,
        )
        if tracking_error is None:
            return None
        return route_goal_pose(
            self.runtime.route_plan.points,
            tracking_error,
            self.runtime.route_plan.pickup_poses,
            include_final=False,
        )

    def current_visible_ball_count(self, result: VisionFrameResult) -> int:
        """Return current raw detector ball count before smoothing/tracking persistence."""
        return len(result.white_balls) + len(result.orange_balls)

    def update_ball_count_reconciliation(self, result: VisionFrameResult, now_s: float) -> None:
        """Track global visible ball count and correct optimistic pickup misses."""
        current_visible_balls = self.current_visible_ball_count(result)
        self.runtime.visible_ball_count_history.append((now_s, current_visible_balls))
        history = list(self.runtime.visible_ball_count_history)
        if len(history) < BALL_COUNT_HISTORY_FRAMES:
            return
        counts = [count for _timestamp, count in history]
        if len(set(counts)) != 1:
            return

        stable_count = counts[-1]
        stable_duration_s = history[-1][0] - history[0][0]
        if stable_duration_s < INITIAL_BALL_COUNT_STABLE_S:
            return

        self.runtime.stable_visible_balls = stable_count
        if self.runtime.initial_total_balls is None:
            self.runtime.initial_total_balls = stable_count
            self.runtime.balls_collected = 0
            return

        expected_visible_balls = max(0, self.runtime.initial_total_balls - self.runtime.balls_collected)
        if stable_count > expected_visible_balls and self.runtime.balls_collected > 0:
            self.runtime.balls_collected -= 1

    def ball_count_initialized(self) -> bool:
        """Return true when the global ball baseline is ready for autonomous drive."""
        return self.runtime.initial_total_balls is not None

    def mark_optimistic_pickup_complete(self) -> None:
        """Optimistically count a completed TCP pickup so the route can move on."""
        if self.runtime.initial_total_balls is None:
            return
        self.runtime.balls_collected = min(
            self.runtime.initial_total_balls,
            self.runtime.balls_collected + 1,
        )

    def optimistic_collection_complete(self) -> bool:
        """Return true when the optimistic pickup count says unloading can begin."""
        return (
            self.runtime.initial_total_balls is not None
            and self.runtime.balls_collected >= self.runtime.initial_total_balls
        )

    def draw_ball_reconciliation_overlay(self, frame: np.ndarray) -> None:
        """Draw global pickup reconciliation counters on the debug frame."""
        initial = "?" if self.runtime.initial_total_balls is None else str(self.runtime.initial_total_balls)
        stable = "?" if self.runtime.stable_visible_balls is None else str(self.runtime.stable_visible_balls)
        text = f"Balls initial:{initial}  collected:{self.runtime.balls_collected}  visible stable:{stable}"
        cv2.putText(
            frame,
            text,
            (20, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def draw_step_pause_overlay(self, frame: np.ndarray) -> None:
        """Draw a high-visibility operator prompt when step drive is paused."""
        if not (self.runtime.step_mode_enabled and self.runtime.step_drive_paused):
            return
        text = "PAUSED - PRESS 'N' TO CONTINUE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.1
        thickness = 3
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = max(20, (frame.shape[1] - text_width) // 2)
        y = max(60, text_height + 28)
        cv2.rectangle(
            frame,
            (x - 18, y - text_height - 18),
            (x + text_width + 18, y + baseline + 18),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            frame,
            (x - 18, y - text_height - 18),
            (x + text_width + 18, y + baseline + 18),
            (0, 255, 255),
            3,
        )
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def draw_drive_calibration_overlay(self, frame: np.ndarray) -> None:
        """Draw the optional drive-calibration status and suggested constants."""
        state = self.runtime.drive_calibration_state
        if state == DriveCalibrationExecutionState.IDLE:
            return

        lines = [
            f"DRIVE CALIBRATION: {state.value}",
            self.runtime.drive_calibration_message or "press x to cancel",
        ]
        expected_turn = float(self.config.drive.drive_calibration_turn_degrees)
        expected_distance = float(self.config.drive.drive_calibration_move_cm)
        if self.runtime.drive_calibration_turn_actual_deg is not None:
            actual = self.runtime.drive_calibration_turn_actual_deg
            error_pct = (actual - expected_turn) * 100.0 / max(expected_turn, 1e-6)
            lines.append(f"Turn: expected {expected_turn:.1f}deg actual {actual:.1f}deg error {error_pct:+.1f}%")
        if self.runtime.drive_calibration_distance_actual_cm is not None:
            actual = self.runtime.drive_calibration_distance_actual_cm
            error_pct = (actual - expected_distance) * 100.0 / max(expected_distance, 1e-6)
            drift = self.runtime.drive_calibration_lateral_drift_cm
            drift_text = "?" if drift is None else f"{drift:+.2f}cm"
            lines.append(
                f"Move: expected {expected_distance:.1f}cm actual {actual:.2f}cm "
                f"error {error_pct:+.1f}% drift {drift_text}"
            )
        current = self.runtime.drive_calibration_values
        suggested = self.runtime.drive_calibration_suggested_values
        if current is not None and suggested is not None:
            lines.append(
                f"AXLE_TRACK_MM: {current.axle_track_mm:.2f} -> {suggested.axle_track_mm:.2f}"
            )
            lines.append(
                f"MM_PER_UNIT: {current.mm_per_unit:.4f} -> {suggested.mm_per_unit:.4f}"
            )
        if state == DriveCalibrationExecutionState.READY_TO_APPLY:
            lines.append("Press y to save/apply, x to discard")
        elif state in (DriveCalibrationExecutionState.FAILED, DriveCalibrationExecutionState.CANCELLING):
            lines.append("Press x to acknowledge/discard")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.62
        thickness = 2
        padding = 12
        line_height = 24
        text_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines)
        box_x = 18
        box_y = 112
        box_w = text_width + padding * 2
        box_h = len(lines) * line_height + padding * 2
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 255), 2)
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (box_x + padding, box_y + padding + 18 + index * line_height),
                font,
                font_scale,
                (0, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    def release_step_drive_pause(self, drive_runtime: DriveRuntime | None) -> None:
        """Allow one step-mode target run after the operator presses next."""
        if not self.runtime.step_mode_enabled:
            return
        if not self.ball_count_initialized():
            if drive_runtime is not None:
                drive_runtime.last_message = "step waiting for stable ball count"
            return
        if self.runtime.robot_pose is None:
            if drive_runtime is not None:
                drive_runtime.last_message = "step waiting for robot pose"
            return
        if len(self.runtime.route_plan.points) < 2:
            if drive_runtime is not None:
                drive_runtime.last_message = "step waiting for route"
            return
        self.runtime.step_drive_paused = False
        if drive_runtime is not None:
            drive_runtime.last_message = "step released"

    def pause_step_drive_after_target(self, drive_runtime: DriveRuntime) -> None:
        """Return step-mode drive to its safe waiting state after pickup completion."""
        if not self.runtime.step_mode_enabled:
            return
        self.runtime.step_drive_paused = True
        self.drive_guard.wheel_controller.reset()
        drive_runtime.stop(DriveControlState.STOPPED, "step target complete; press n")

    def drive_calibration_active(self) -> bool:
        """Return true when the optional drive calibration sequence owns control."""
        return self.runtime.drive_calibration_state != DriveCalibrationExecutionState.IDLE

    def _drive_calibration_thread_alive(self) -> bool:
        return self.drive_calibration_thread is not None and self.drive_calibration_thread.is_alive()

    def _send_drive_calibration_emergency_stop(self) -> None:
        """Best-effort stop on a separate TCP connection while the main command blocks."""
        controller: RobotController | None = None
        try:
            controller = RobotController(
                self.config.drive.robot_ip,
                port=self.config.drive.robot_tcp_port,
                timeout=2.0,
                connect_retries=0,
            )
            controller.stop()
        except Exception as exc:
            self.runtime.drive_calibration_message = (
                f"{self.runtime.drive_calibration_message}; emergency stop failed: {exc}"
            )
        finally:
            if controller is not None:
                controller.close()

    def _start_drive_calibration_command_thread(self, name: str, command: str) -> None:
        """Run one blocking EV3 calibration command without freezing vision."""
        if self._drive_calibration_thread_alive():
            return

        self.drive_calibration_thread_error = ""

        def run_command() -> None:
            try:
                if command == "turn":
                    self._controller().turn(
                        round(float(self.config.drive.drive_calibration_turn_degrees), 1),
                        int(round(self.config.drive.drive_calibration_turn_speed_pct)),
                    )
                elif command == "move":
                    self._controller().move(
                        round(float(self.config.drive.drive_calibration_move_cm), 1),
                        int(round(self.config.drive.drive_calibration_move_speed_pct)),
                    )
                else:
                    raise ValueError(f"unknown drive calibration command: {command}")
            except Exception as exc:
                self.drive_calibration_thread_error = str(exc)

        self.drive_calibration_thread = threading.Thread(
            target=run_command,
            name=name,
            daemon=True,
        )
        self.drive_calibration_thread.start()

    def _start_drive_calibration_turn(self) -> None:
        pose = self.runtime.robot_pose
        if pose is None:
            self.runtime.drive_calibration_state = DriveCalibrationExecutionState.FAILED
            self.runtime.drive_calibration_message = "drive calibration failed: robot pose missing"
            return
        self.runtime.drive_calibration_state = DriveCalibrationExecutionState.TURNING
        self.runtime.drive_calibration_turn_start_heading_rad = float(pose.heading_rad)
        self.runtime.drive_calibration_turn_last_heading_rad = float(pose.heading_rad)
        self.runtime.drive_calibration_turn_accumulated_rad = 0.0
        self.runtime.drive_calibration_message = (
            f"turning {self.config.drive.drive_calibration_turn_degrees:.0f}deg for drive calibration"
        )
        self._start_drive_calibration_command_thread("ev3-drive-calibration-turn", "turn")

    def _start_drive_calibration_move(self) -> None:
        pose = self.runtime.robot_pose
        if pose is None:
            self.runtime.drive_calibration_state = DriveCalibrationExecutionState.FAILED
            self.runtime.drive_calibration_message = "drive calibration failed: robot pose missing before move"
            return
        self.runtime.drive_calibration_state = DriveCalibrationExecutionState.MOVING
        self.runtime.drive_calibration_distance_start_pose = pose
        self.runtime.drive_calibration_settle_started_s = 0.0
        self.runtime.drive_calibration_message = (
            f"moving {self.config.drive.drive_calibration_move_cm:.1f}cm for drive calibration"
        )
        self._start_drive_calibration_command_thread("ev3-drive-calibration-move", "move")

    def start_drive_calibration(self, drive_runtime: DriveRuntime | None) -> None:
        """Start the optional drive calibration sequence from the live drive UI."""
        if self.drive_calibration_active():
            if drive_runtime is not None:
                drive_runtime.last_message = "drive calibration already active"
            return
        if drive_runtime is None or drive_runtime.dispatcher is None:
            if drive_runtime is not None:
                drive_runtime.last_message = "drive calibration requires --drive"
            return
        if self.latest_result is None or self.latest_result.frame_for_detection is None:
            drive_runtime.last_message = "drive calibration waiting for top-down frame"
            return
        if self.runtime.robot_pose is None:
            drive_runtime.last_message = "drive calibration waiting for robot pose"
            return
        if self.robot_runtime.phase != RobotCalibrationPhase.STATE_NORMAL:
            drive_runtime.last_message = "finish robot-origin calibration before drive calibration"
            return
        if (
            self.runtime.pickup_state != PickupExecutionState.NAVIGATION
            or self.runtime.unload_state != UnloadExecutionState.IDLE
            or self.runtime.collector_state in (
                CollectorPositionState.PICKUP_ASSIST,
                CollectorPositionState.UNLOADING,
            )
        ):
            drive_runtime.last_message = "drive calibration blocked by active pickup/unload state"
            return

        self.drive_guard.wheel_controller.reset()
        self.runtime.clear_route()
        self._force_drive_stop(drive_runtime, DriveControlState.CALIBRATING, "starting drive calibration")
        try:
            current_values = self._controller().get_drive_calibration()
        except Exception as exc:
            self.runtime.drive_calibration_state = DriveCalibrationExecutionState.FAILED
            self.runtime.drive_calibration_message = f"drive calibration failed: {exc}"
            drive_runtime.last_message = self.runtime.drive_calibration_message
            return

        self.runtime.drive_calibration_values = current_values
        self.runtime.drive_calibration_suggested_values = None
        self.runtime.drive_calibration_turn_actual_deg = None
        self.runtime.drive_calibration_distance_actual_cm = None
        self.runtime.drive_calibration_lateral_drift_cm = None
        self._start_drive_calibration_turn()

    def _record_drive_calibration_turn_pose(self) -> None:
        pose = self.runtime.robot_pose
        if pose is None:
            return
        current_heading = float(pose.heading_rad)
        previous = self.runtime.drive_calibration_turn_last_heading_rad
        if previous is not None:
            self.runtime.drive_calibration_turn_accumulated_rad += unwrapped_heading_delta(previous, current_heading)
        self.runtime.drive_calibration_turn_last_heading_rad = current_heading

    def _fail_drive_calibration(
        self,
        reason: str,
        emergency_stop: bool = False,
        keep_suggestion: bool = False,
    ) -> None:
        self.runtime.drive_calibration_state = DriveCalibrationExecutionState.FAILED
        self.runtime.drive_calibration_message = f"drive calibration failed: {reason}"
        if not keep_suggestion:
            self.runtime.drive_calibration_suggested_values = None
        if emergency_stop:
            self._send_drive_calibration_emergency_stop()

    def cancel_drive_calibration(self, drive_runtime: DriveRuntime | None) -> None:
        """Cancel drive calibration and hold control until any EV3 command returns."""
        if not self.drive_calibration_active():
            return
        if self._drive_calibration_thread_alive():
            self.runtime.drive_calibration_state = DriveCalibrationExecutionState.CANCELLING
            self.runtime.drive_calibration_message = "drive calibration cancel requested; waiting for EV3 command"
            self._send_drive_calibration_emergency_stop()
            if drive_runtime is not None:
                drive_runtime.state = DriveControlState.CALIBRATING
                drive_runtime.last_message = self.runtime.drive_calibration_message
            return
        self.runtime.reset_drive_calibration_state()
        if drive_runtime is not None:
            drive_runtime.stop(DriveControlState.STOPPED, "drive calibration cancelled")

    def apply_drive_calibration(self, drive_runtime: DriveRuntime | None) -> None:
        """Persist suggested calibration values when the operator confirms with ``y``."""
        if self.runtime.drive_calibration_state != DriveCalibrationExecutionState.READY_TO_APPLY:
            return
        suggested = self.runtime.drive_calibration_suggested_values
        if suggested is None:
            self._fail_drive_calibration("no suggested values available")
            return
        try:
            applied = self._controller().set_drive_calibration(
                suggested.axle_track_mm,
                suggested.mm_per_unit,
            )
        except Exception as exc:
            self._fail_drive_calibration(f"apply failed: {exc}", keep_suggestion=True)
            if drive_runtime is not None:
                drive_runtime.state = DriveControlState.DISPATCH_ERROR
                drive_runtime.last_message = self.runtime.drive_calibration_message
            return

        self.robot_runtime.warning = (
            f"Applied drive calibration: axle {applied.axle_track_mm:.2f}mm, "
            f"mm/unit {applied.mm_per_unit:.4f}"
        )
        self.runtime.reset_drive_calibration_state()
        if drive_runtime is not None:
            drive_runtime.stop(DriveControlState.STOPPED, "drive calibration applied")

    def update_drive_calibration_state(self, drive_runtime: DriveRuntime, now_s: float) -> bool:
        """Advance optional drive calibration and return True while it owns output."""
        state = self.runtime.drive_calibration_state
        if state == DriveCalibrationExecutionState.IDLE:
            return False

        drive_runtime.state = DriveControlState.CALIBRATING
        drive_runtime.last_error = None
        drive_runtime.last_command = WheelCommand(0.0, 0.0)
        drive_runtime.last_message = self.runtime.drive_calibration_message

        if state == DriveCalibrationExecutionState.CANCELLING:
            if self._drive_calibration_thread_alive():
                return True
            self.runtime.reset_drive_calibration_state()
            drive_runtime.stop(DriveControlState.STOPPED, "drive calibration cancelled")
            return True

        if state == DriveCalibrationExecutionState.FAILED:
            if self._drive_calibration_thread_alive():
                drive_runtime.last_message = f"{self.runtime.drive_calibration_message}; waiting for EV3 command"
            return True

        if state in (
            DriveCalibrationExecutionState.TURNING,
            DriveCalibrationExecutionState.TURN_SETTLING,
            DriveCalibrationExecutionState.MOVING,
            DriveCalibrationExecutionState.MOVE_SETTLING,
        ) and self.runtime.robot_pose is None:
            self._fail_drive_calibration("robot pose lost", emergency_stop=True)
            drive_runtime.last_message = self.runtime.drive_calibration_message
            return True

        if self.drive_calibration_thread_error:
            self._fail_drive_calibration(self.drive_calibration_thread_error)
            drive_runtime.last_message = self.runtime.drive_calibration_message
            return True

        if state == DriveCalibrationExecutionState.TURNING:
            self._record_drive_calibration_turn_pose()
            if self._drive_calibration_thread_alive():
                actual_deg = abs(math.degrees(self.runtime.drive_calibration_turn_accumulated_rad))
                self.runtime.drive_calibration_message = f"turning; measured {actual_deg:.1f}deg so far"
                drive_runtime.last_message = self.runtime.drive_calibration_message
                return True
            self.runtime.drive_calibration_state = DriveCalibrationExecutionState.TURN_SETTLING
            self.runtime.drive_calibration_settle_started_s = now_s
            self.runtime.drive_calibration_message = "turn complete; settling before measurement"
            return True

        if state == DriveCalibrationExecutionState.TURN_SETTLING:
            self._record_drive_calibration_turn_pose()
            elapsed_s = now_s - self.runtime.drive_calibration_settle_started_s
            if elapsed_s < self.config.drive.drive_calibration_settle_time_s:
                self.runtime.drive_calibration_message = (
                    f"turn settling {elapsed_s:.2f}/{self.config.drive.drive_calibration_settle_time_s:.2f}s"
                )
                drive_runtime.last_message = self.runtime.drive_calibration_message
                return True
            actual_turn_deg = abs(math.degrees(self.runtime.drive_calibration_turn_accumulated_rad))
            if (
                not math.isfinite(actual_turn_deg)
                or actual_turn_deg < self.config.drive.drive_calibration_min_actual_turn_deg
            ):
                self._fail_drive_calibration(f"invalid actual turn {actual_turn_deg:.1f}deg")
                drive_runtime.last_message = self.runtime.drive_calibration_message
                return True
            current_values = self.runtime.drive_calibration_values
            if current_values is None:
                self._fail_drive_calibration("missing current calibration values")
                drive_runtime.last_message = self.runtime.drive_calibration_message
                return True
            suggested_axle = suggest_axle_track_mm(
                current_values.axle_track_mm,
                self.config.drive.drive_calibration_turn_degrees,
                actual_turn_deg,
            )
            self.runtime.drive_calibration_turn_actual_deg = actual_turn_deg
            self.runtime.drive_calibration_suggested_values = DriveCalibrationValues(
                axle_track_mm=suggested_axle,
                mm_per_unit=current_values.mm_per_unit,
            )
            self._start_drive_calibration_move()
            return True

        if state == DriveCalibrationExecutionState.MOVING:
            if self._drive_calibration_thread_alive():
                return True
            self.runtime.drive_calibration_state = DriveCalibrationExecutionState.MOVE_SETTLING
            self.runtime.drive_calibration_settle_started_s = now_s
            self.runtime.drive_calibration_message = "move complete; settling before measurement"
            return True

        if state == DriveCalibrationExecutionState.MOVE_SETTLING:
            elapsed_s = now_s - self.runtime.drive_calibration_settle_started_s
            if elapsed_s < self.config.drive.drive_calibration_settle_time_s:
                self.runtime.drive_calibration_message = (
                    f"move settling {elapsed_s:.2f}/{self.config.drive.drive_calibration_settle_time_s:.2f}s"
                )
                drive_runtime.last_message = self.runtime.drive_calibration_message
                return True
            start_pose = self.runtime.drive_calibration_distance_start_pose
            current_pose = self.runtime.robot_pose
            suggested = self.runtime.drive_calibration_suggested_values
            if start_pose is None or current_pose is None or suggested is None:
                self._fail_drive_calibration("missing distance measurement pose")
                drive_runtime.last_message = self.runtime.drive_calibration_message
                return True
            forward_cm, lateral_cm = projected_motion_cm(
                (start_pose.x_cm, start_pose.y_cm),
                (current_pose.x_cm, current_pose.y_cm),
                start_pose.heading_rad,
            )
            if (
                not math.isfinite(forward_cm)
                or forward_cm < self.config.drive.drive_calibration_min_actual_distance_cm
            ):
                self._fail_drive_calibration(f"invalid actual distance {forward_cm:.2f}cm")
                drive_runtime.last_message = self.runtime.drive_calibration_message
                return True
            new_mm_per_unit = suggest_mm_per_unit(
                suggested.mm_per_unit,
                self.config.drive.drive_calibration_move_cm,
                forward_cm,
            )
            self.runtime.drive_calibration_distance_actual_cm = forward_cm
            self.runtime.drive_calibration_lateral_drift_cm = lateral_cm
            self.runtime.drive_calibration_suggested_values = DriveCalibrationValues(
                axle_track_mm=suggested.axle_track_mm,
                mm_per_unit=new_mm_per_unit,
            )
            self.runtime.drive_calibration_state = DriveCalibrationExecutionState.READY_TO_APPLY
            self.runtime.drive_calibration_message = "drive calibration ready: press y to apply, x to discard"
            drive_runtime.last_message = self.runtime.drive_calibration_message
            return True

        if state == DriveCalibrationExecutionState.READY_TO_APPLY:
            drive_runtime.last_message = self.runtime.drive_calibration_message
            return True

        return True

    def _pickup_target_body_pose(self, goal: HybridPose, params: dict[str, object] | None = None) -> tuple[float, float]:
        """Return the current best body-center target for the next pickup."""
        if not self.runtime.latest_smoothed_balls:
            return float(goal.x_cm), float(goal.y_cm)
        params = params or self.latest_params
        if params is None:
            return float(goal.x_cm), float(goal.y_cm)
        geometry = self.robot_estimator.robot_geometry_from_params(params)
        planned_ball_x, planned_ball_y = self.route_facade.hybrid_planner.tube_center_for_pose(goal, geometry)
        nearest = min(
            self.runtime.latest_smoothed_balls,
            key=lambda ball: math.hypot(ball.cm_x - planned_ball_x, ball.cm_y - planned_ball_y),
        )
        if math.hypot(nearest.cm_x - planned_ball_x, nearest.cm_y - planned_ball_y) > (
            self.config.planner.route_target_move_invalidate_cm * 2.0
        ):
            return float(goal.x_cm), float(goal.y_cm)
        heading = goal.theta_rad
        if self.runtime.robot_pose is not None:
            dx = nearest.cm_x - float(self.runtime.robot_pose.x_cm)
            dy = nearest.cm_y - float(self.runtime.robot_pose.y_cm)
            if math.hypot(dx, dy) > 1e-6:
                heading = math.atan2(dy, dx)
        forward = (math.cos(heading), math.sin(heading))
        right = (math.sin(heading), -math.cos(heading))
        return (
            nearest.cm_x - forward[0] * geometry.tube_forward_cm - right[0] * geometry.tube_right_cm,
            nearest.cm_y - forward[1] * geometry.tube_forward_cm - right[1] * geometry.tube_right_cm,
        )

    @staticmethod
    def _target_offset_in_robot_frame(robot_pose: RobotPose, target_x_cm: float, target_y_cm: float) -> tuple[float, float]:
        dx = target_x_cm - robot_pose.x_cm
        dy = target_y_cm - robot_pose.y_cm
        forward_cm = dx * math.cos(robot_pose.heading_rad) + dy * math.sin(robot_pose.heading_rad)
        lateral_cm = dx * math.sin(robot_pose.heading_rad) - dy * math.cos(robot_pose.heading_rad)
        return forward_cm, lateral_cm

    def _near_zone_visual_servo_turn(
        self,
        forward_cm: float,
        lateral_cm: float,
    ) -> tuple[bool, float, int, str]:
        """Return whether alignment is converged plus the next proportional turn command."""
        error_cm = abs(lateral_cm)
        if error_cm <= self.config.drive.visual_servo_noise_floor_cm:
            return True, 0.0, 0, "noise floor"

        if error_cm + self.config.drive.visual_servo_min_improvement_cm < self.runtime.near_zone_alignment_best_error_cm:
            self.runtime.near_zone_alignment_best_error_cm = error_cm
            self.runtime.near_zone_alignment_stale_frames = 0
        else:
            self.runtime.near_zone_alignment_stale_frames += 1
            if self.runtime.near_zone_alignment_stale_frames >= self.config.drive.visual_servo_stall_frames:
                return True, 0.0, 0, "no further improvement"

        if self.runtime.near_zone_alignment_iterations >= self.config.drive.visual_servo_max_iterations:
            return True, 0.0, 0, "iteration limit"

        correction_rad = math.atan2(-lateral_cm, max(abs(forward_cm), 1e-6))
        turn_deg = -math.degrees(correction_rad) * self.config.drive.visual_servo_turn_kp
        turn_abs = min(abs(turn_deg), self.config.drive.visual_servo_max_turn_deg)
        turn_abs = max(turn_abs, self.config.drive.visual_servo_min_turn_deg)
        turn_deg = math.copysign(turn_abs, turn_deg)

        max_speed = max(
            self.config.drive.visual_servo_min_turn_speed_pct,
            self.config.drive.near_zone_turn_speed_pct,
        )
        speed_span = max_speed - self.config.drive.visual_servo_min_turn_speed_pct
        speed_scale = min(1.0, turn_abs / max(self.config.drive.visual_servo_max_turn_deg, 1e-6))
        speed_pct = int(round(self.config.drive.visual_servo_min_turn_speed_pct + speed_span * speed_scale))
        return False, turn_deg, speed_pct, "aligning"

    def _run_near_zone_precise_move(self, drive_runtime: DriveRuntime) -> bool:
        """Stop TCP route tracking, visually servo heading, then run final encoder move."""
        if self.runtime.near_zone_move_fired:
            return True
        distance_cm = self._pickup_distance_to_goal_cm()
        if not math.isfinite(distance_cm) or distance_cm > self.config.drive.near_zone_cm:
            return False

        self.drive_guard.wheel_controller.reset()
        drive_runtime.stop(DriveControlState.PRECISE_MOVE, "near-zone TCP align")
        goal = self._pickup_goal_pose()
        if goal is None or self.runtime.robot_pose is None:
            drive_runtime.stop(DriveControlState.NO_ROUTE, "near-zone goal missing")
            self.runtime.pickup_state = PickupExecutionState.REPLAN
            self.runtime.pickup_state_started_s = time.perf_counter()
            return True
        target_x_cm, target_y_cm = self._pickup_target_body_pose(goal)
        remaining_cm = math.hypot(target_x_cm - self.runtime.robot_pose.x_cm, target_y_cm - self.runtime.robot_pose.y_cm)
        if not math.isfinite(remaining_cm):
            drive_runtime.stop(DriveControlState.DISPATCH_ERROR, "near-zone distance invalid")
            return True

        if remaining_cm <= 0.5:
            self.runtime.pickup_state = PickupExecutionState.PICKUP_ASSIST
            self.runtime.pickup_state_started_s = time.perf_counter()
            return True

        try:
            if not self.runtime.near_zone_turn_fired:
                target_heading_rad = math.atan2(
                    target_y_cm - float(self.runtime.robot_pose.y_cm),
                    target_x_cm - float(self.runtime.robot_pose.x_cm),
                )
                heading_error_deg = math.degrees(normalize_angle(target_heading_rad - float(self.runtime.robot_pose.heading_rad)))
                tcp_turn_deg = -heading_error_deg
                if abs(tcp_turn_deg) > 0.5:
                    drive_runtime.last_message = f"TCP turn {tcp_turn_deg:.1f}deg"
                    self._controller().turn(
                        round(tcp_turn_deg, 1),
                        int(round(self.config.drive.near_zone_turn_speed_pct)),
                    )
                self.runtime.near_zone_turn_fired = True
                drive_runtime.last_message = "TCP turn complete; visual servo aligning"
                return True

            forward_cm, lateral_cm = self._target_offset_in_robot_frame(
                self.runtime.robot_pose,
                target_x_cm,
                target_y_cm,
            )
            settled_verified = False
            if self.runtime.near_zone_alignment_settle_started_s != 0.0:
                elapsed_s = time.perf_counter() - self.runtime.near_zone_alignment_settle_started_s
                if elapsed_s < self.config.drive.visual_servo_settle_time_s:
                    self._force_drive_stop(
                        drive_runtime,
                        DriveControlState.PRECISE_MOVE,
                        f"Visual servo settling {elapsed_s:.2f}/{self.config.drive.visual_servo_settle_time_s:.2f}s",
                    )
                    return True
                if abs(lateral_cm) <= self.config.drive.visual_servo_noise_floor_cm:
                    settled_verified = True
                    reason = f"settled {self.runtime.near_zone_alignment_settle_reason}"
                    self.runtime.near_zone_alignment_settle_started_s = 0.0
                    self.runtime.near_zone_alignment_settle_reason = ""
                else:
                    drive_runtime.last_message = (
                        f"Visual servo settle drift {abs(lateral_cm):.2f}cm; correcting"
                    )
                    self.runtime.near_zone_alignment_settle_started_s = 0.0
                    self.runtime.near_zone_alignment_settle_reason = ""
                    self.runtime.near_zone_alignment_best_error_cm = float("inf")
                    self.runtime.near_zone_alignment_stale_frames = 0
                    self.runtime.near_zone_alignment_iterations = 0

            if not settled_verified:
                aligned, turn_deg, turn_speed_pct, reason = self._near_zone_visual_servo_turn(forward_cm, lateral_cm)
                if not aligned:
                    self.runtime.near_zone_alignment_iterations += 1
                    drive_runtime.last_message = (
                        f"Visual servo {abs(lateral_cm):.2f}cm lateral; "
                        f"turn {turn_deg:.2f}deg @ {turn_speed_pct}%"
                    )
                    self._controller().turn(round(turn_deg, 2), turn_speed_pct)
                    return True

                self.runtime.near_zone_alignment_settle_started_s = time.perf_counter()
                self.runtime.near_zone_alignment_settle_reason = reason
                self._force_drive_stop(
                    drive_runtime,
                    DriveControlState.PRECISE_MOVE,
                    f"Visual servo aligned; settling before move: {reason}",
                )
                return True

            target_x_cm, target_y_cm = self._pickup_target_body_pose(goal)
            remaining_cm = math.hypot(
                target_x_cm - self.runtime.robot_pose.x_cm,
                target_y_cm - self.runtime.robot_pose.y_cm,
            )
            if remaining_cm <= 0.5:
                self.runtime.pickup_state = PickupExecutionState.PICKUP_ASSIST
                self.runtime.pickup_state_started_s = time.perf_counter()
                return True

            drive_runtime.last_message = f"TCP move {remaining_cm:.1f}cm after visual servo: {reason}"
            self._controller().move(
                round(remaining_cm, 1),
                int(round(self.config.drive.near_zone_move_speed_pct)),
            )
            self.runtime.near_zone_move_fired = True
            self.mark_optimistic_pickup_complete()
            drive_runtime.stop(DriveControlState.PICKUP, "TCP align/move complete")
            self.runtime.pickup_state = PickupExecutionState.PICKUP_ASSIST
            self.runtime.pickup_state_started_s = time.perf_counter()
        except Exception as exc:
            drive_runtime.stop(DriveControlState.DISPATCH_ERROR, f"TCP align/move failed: {exc}")
            self.runtime.pickup_state = PickupExecutionState.REPLAN
            self.runtime.pickup_state_started_s = time.perf_counter()
        return True

    def cached_balls_match_current_scene(self, smoothed_balls: list[SmoothedBallCoordinate]) -> bool:
        """Return true when current visible balls are a stationary subset of the cached plan."""
        if not self.runtime.route_cache_ball_signature:
            return not smoothed_balls
        bucket = max(1.0, self.config.planner.route_target_move_invalidate_cm)
        current = Counter(
            (
                ball.label,
                int(round(ball.cm_x / bucket)),
                int(round(ball.cm_y / bucket)),
            )
            for ball in smoothed_balls
        )
        cached = Counter(
            (label, bucket_x, bucket_y)
            for _track_id, label, bucket_x, bucket_y in self.runtime.route_cache_ball_signature
        )
        return all(count <= cached[key] for key, count in current.items())

    def should_route_directly_to_unload(self) -> bool:
        """Return true when collection is complete and robot pose can seed unload routing."""
        return self.optimistic_collection_complete() and self.runtime.robot_pose is not None

    def should_keep_cached_route_after_pickup(self) -> bool:
        """Continue a global route after pickup unless vision shows a changed ball scene."""
        return bool(self.runtime.route_plan.points) and self.cached_balls_match_current_scene(
            self.runtime.latest_smoothed_balls
        )

    def consume_completed_pickup_checkpoint(self) -> bool:
        """Remove the completed pickup checkpoint and consumed route prefix."""
        if self.runtime.robot_pose is None or not self.runtime.route_plan.pickup_poses:
            return False
        pickup_distances = [
            math.hypot(
                float(self.runtime.robot_pose.x_cm) - pickup_pose.x_cm,
                float(self.runtime.robot_pose.y_cm) - pickup_pose.y_cm,
            )
            for pickup_pose in self.runtime.route_plan.pickup_poses
        ]
        completed_index = min(range(len(pickup_distances)), key=pickup_distances.__getitem__)
        if pickup_distances[completed_index] > self.config.drive.near_zone_cm:
            return False

        completed_pickup = self.runtime.route_plan.pickup_poses[completed_index]
        remaining_pickups = [
            pickup_pose
            for index, pickup_pose in enumerate(self.runtime.route_plan.pickup_poses)
            if index != completed_index
        ]
        completed_route_index = closest_route_index(self.runtime.route_plan.points, completed_pickup)
        current_pose = HybridPose(
            float(self.runtime.robot_pose.x_cm),
            float(self.runtime.robot_pose.y_cm),
            float(self.runtime.robot_pose.heading_rad),
        )
        remaining_route_points = (
            [current_pose] + list(self.runtime.route_plan.points[completed_route_index + 1 :])
            if completed_route_index >= 0
            else list(self.runtime.route_plan.points)
        )
        if len(remaining_route_points) == 1 and remaining_pickups:
            remaining_route_points.extend(remaining_pickups)
        self.runtime.route_plan = replace(
            self.runtime.route_plan,
            active_target=None,
            points=remaining_route_points,
            pickup_poses=remaining_pickups,
        )
        self.runtime.route_cache_target_id = -1
        self.runtime.route_cache_target_label = None
        self.runtime.route_cache_target_cm = None
        return True

    def _robot_body_edge_clearance_cm(self) -> float:
        if self.runtime.robot_pose is None:
            return float("inf")
        geometry = self.robot_estimator.robot_geometry_from_params(self.latest_params)
        return robot_body_edge_clearance_cm(self.runtime.robot_pose, geometry, self.config.field)

    def _next_route_goal_after_pickup(self) -> HybridPose | None:
        if self.runtime.robot_pose is None or len(self.runtime.route_plan.points) < 2:
            return None
        tracking_error = self.route_facade.compute_route_tracking_error(
            self.runtime.robot_pose,
            self.runtime.route_plan.points,
        )
        if tracking_error is None:
            if self.runtime.route_plan.unload_pose is not None:
                return self.runtime.route_plan.unload_pose
            return self.runtime.route_plan.points[-1]
        return route_goal_pose(
            self.runtime.route_plan.points,
            tracking_error,
            self.runtime.route_plan.pickup_poses,
            include_final=True,
        )

    def _begin_post_pickup_recovery(self, drive_runtime: DriveRuntime, next_message: str) -> bool:
        """Choose a bounded recovery action before route tracking resumes."""
        clearance_cm = self._robot_body_edge_clearance_cm()
        if clearance_cm <= self.config.drive.post_pickup_escape_clearance_cm:
            self.runtime.pickup_state = PickupExecutionState.POST_PICKUP_ESCAPE
            self.runtime.post_pickup_escape_fired = False
            self.drive_guard.wheel_controller.reset()
            self._force_drive_stop(
                drive_runtime,
                DriveControlState.POST_PICKUP_ESCAPE,
                f"{next_message}; edge {clearance_cm:.1f}cm, reversing before replan",
            )
            return True
        if (
            clearance_cm <= self.config.drive.post_pickup_align_clearance_cm
            and self._next_route_goal_after_pickup() is not None
        ):
            self.runtime.pickup_state = PickupExecutionState.POST_PICKUP_ALIGN
            self.drive_guard.wheel_controller.reset()
            self._force_drive_stop(
                drive_runtime,
                DriveControlState.POST_PICKUP_ALIGN,
                f"{next_message}; edge {clearance_cm:.1f}cm, pivoting before drive",
            )
            return True
        self.runtime.reset_pickup_state()
        drive_runtime.stop(DriveControlState.REPLANNING, next_message)
        return True

    def _run_post_pickup_escape(self, drive_runtime: DriveRuntime) -> bool:
        """Back away from a critically close wall before asking for a fresh route."""
        if not self.runtime.post_pickup_escape_fired:
            self.drive_guard.wheel_controller.reset()
            self._force_drive_stop(drive_runtime, DriveControlState.POST_PICKUP_ESCAPE, "post-pickup reverse escape")
            try:
                self._controller().back(
                    round(float(self.config.drive.post_pickup_escape_back_cm), 1),
                    int(round(self.config.drive.post_pickup_escape_speed_pct)),
                )
            except Exception as exc:
                drive_runtime.stop(DriveControlState.DISPATCH_ERROR, f"post-pickup reverse escape failed: {exc}")
                self.runtime.clear_route()
                self.runtime.reset_pickup_state()
                return True
            self.runtime.post_pickup_escape_fired = True
            self.runtime.clear_route()
            drive_runtime.stop(DriveControlState.REPLANNING, "post-pickup reverse escape complete; requesting new route")
            self.runtime.reset_pickup_state()
            return True
        self.runtime.clear_route()
        drive_runtime.stop(DriveControlState.REPLANNING, "post-pickup reverse escape complete; requesting new route")
        self.runtime.reset_pickup_state()
        return True

    def _run_post_pickup_align(self, drive_runtime: DriveRuntime) -> bool:
        """Tank-steer in place toward the next route goal before resuming XTE."""
        if self.runtime.robot_pose is None:
            drive_runtime.stop(DriveControlState.NO_POSE, "post-pickup align missing robot pose")
            return True
        goal = self._next_route_goal_after_pickup()
        if goal is None:
            self.runtime.reset_pickup_state()
            drive_runtime.stop(DriveControlState.REPLANNING, "post-pickup align has no route goal")
            return True
        heading_to_goal = math.atan2(
            goal.y_cm - self.runtime.robot_pose.y_cm,
            goal.x_cm - self.runtime.robot_pose.x_cm,
        )
        heading_error_rad = normalize_angle(heading_to_goal - self.runtime.robot_pose.heading_rad)
        tolerance_rad = math.radians(self.config.drive.post_pickup_align_tolerance_deg)
        if abs(heading_error_rad) <= tolerance_rad:
            self.drive_guard.wheel_controller.reset()
            self._force_drive_stop(drive_runtime, DriveControlState.POST_PICKUP_ALIGN, "post-pickup pivot complete")
            self.runtime.reset_pickup_state()
            return True
        speed_pct = float(self.config.drive.post_pickup_align_speed_pct)
        left_pct = -math.copysign(speed_pct, heading_error_rad)
        right_pct = math.copysign(speed_pct, heading_error_rad)
        drive_runtime.state = DriveControlState.POST_PICKUP_ALIGN
        drive_runtime.last_message = f"Post-pickup pivot {math.degrees(heading_error_rad):.1f}deg"
        drive_runtime.last_command = WheelCommand(left_pct, right_pct)
        drive_runtime.dispatcher.send_wheel_speeds(left_pct, right_pct, force=True)
        return True

    def _start_pickup_assist_command_thread(self) -> None:
        if self.pickup_thread is not None and self.pickup_thread.is_alive():
            return

        def run_pickup_assist() -> None:
            try:
                self._controller().pickup_assist()
                self.pickup_last_error = ""
            except Exception as exc:
                self.pickup_last_error = str(exc)

        self.pickup_thread = threading.Thread(target=run_pickup_assist, name="ev3-pickup-assist-command", daemon=True)
        self.pickup_thread.start()

    def update_pickup_state(self, drive_runtime: DriveRuntime, now_s: float) -> bool:
        """Run the near-zone TCP pickup handoff and return True when it owns motor output."""
        state = self.runtime.pickup_state
        if state == PickupExecutionState.NAVIGATION and drive_runtime.dispatcher is None:
            return False
        if state == PickupExecutionState.NAVIGATION:
            return self._run_near_zone_precise_move(drive_runtime)
        if state == PickupExecutionState.POST_PICKUP_ESCAPE:
            return self._run_post_pickup_escape(drive_runtime)
        if state == PickupExecutionState.POST_PICKUP_ALIGN:
            return self._run_post_pickup_align(drive_runtime)

        if state == PickupExecutionState.PICKUP_ASSIST:
            drive_runtime.stop(DriveControlState.PICKUP, "pickup assist")
            if not self.runtime.pickup_command_fired:
                self.runtime.collector_state = CollectorPositionState.PICKUP_ASSIST
                self._start_pickup_assist_command_thread()
                self.runtime.pickup_command_fired = True
            if self.pickup_thread is not None and self.pickup_thread.is_alive():
                return True
            self.runtime.collector_state = CollectorPositionState.TRAVEL
            self.runtime.pickup_state = PickupExecutionState.REPLAN
            self.runtime.pickup_state_started_s = now_s
            return True

        if state == PickupExecutionState.REPLAN:
            self.drive_guard.wheel_controller.reset()
            if self.optimistic_collection_complete() and self.runtime.route_plan.unload_pose is not None:
                self.consume_completed_pickup_checkpoint()
                return self._begin_post_pickup_recovery(drive_runtime, "pickup complete; routing to unload")
            if self.should_keep_cached_route_after_pickup():
                self.consume_completed_pickup_checkpoint()
                return self._begin_post_pickup_recovery(drive_runtime, "pickup complete; continuing cached route")
            self.runtime.clear_route()
            return self._begin_post_pickup_recovery(drive_runtime, "pickup complete; requesting new route")

        return False

    def _unload_endpoint_reached(self) -> bool:
        if (
            self.runtime.robot_pose is None
            or self.runtime.route_plan.unload_pose is None
            or len(self.runtime.route_plan.points) < 2
            or self.runtime.pickup_state != PickupExecutionState.NAVIGATION
            or not self.optimistic_collection_complete()
        ):
            return False
        tracking_error = self.route_facade.compute_route_tracking_error(
            self.runtime.robot_pose,
            self.runtime.route_plan.points,
        )
        if tracking_error is None:
            return False
        next_goal = route_goal_pose(
            self.runtime.route_plan.points,
            tracking_error,
            self.runtime.route_plan.pickup_poses,
            include_final=True,
        )
        if next_goal is None:
            return False
        unload_pose = self.runtime.route_plan.unload_pose
        if math.hypot(next_goal.x_cm - unload_pose.x_cm, next_goal.y_cm - unload_pose.y_cm) > self.config.planner.goal_tolerance_cm:
            return False
        distance_cm = math.hypot(
            self.runtime.robot_pose.x_cm - unload_pose.x_cm,
            self.runtime.robot_pose.y_cm - unload_pose.y_cm,
        )
        return distance_cm <= self.config.drive.unload_trigger_distance_cm

    def _unload_route_is_terminal_target(self) -> bool:
        if (
            self.runtime.robot_pose is None
            or self.runtime.route_plan.unload_pose is None
            or len(self.runtime.route_plan.points) < 2
            or self.runtime.pickup_state != PickupExecutionState.NAVIGATION
            or not self.optimistic_collection_complete()
        ):
            return False
        tracking_error = self.route_facade.compute_route_tracking_error(
            self.runtime.robot_pose,
            self.runtime.route_plan.points,
        )
        if tracking_error is None:
            return False
        next_goal = route_goal_pose(
            self.runtime.route_plan.points,
            tracking_error,
            self.runtime.route_plan.pickup_poses,
            include_final=True,
        )
        unload_pose = self.runtime.route_plan.unload_pose
        return next_goal is not None and math.hypot(
            next_goal.x_cm - unload_pose.x_cm,
            next_goal.y_cm - unload_pose.y_cm,
        ) <= self.config.planner.goal_tolerance_cm

    def _unload_staging_intercept_reached(self) -> bool:
        if self.runtime.robot_pose is None or not self._unload_route_is_terminal_target():
            return False
        unload_pose = self.runtime.route_plan.unload_pose
        if unload_pose is None:
            return False
        distance_to_staging_cm = math.hypot(
            self.runtime.robot_pose.x_cm - unload_pose.x_cm,
            self.runtime.robot_pose.y_cm - unload_pose.y_cm,
        )
        staging_distance_cm = max(
            self.config.drive.unload_staging_distance_cm,
            self.config.drive.unload_trigger_distance_cm,
        )
        return distance_to_staging_cm <= staging_distance_cm

    def _force_drive_stop(self, drive_runtime: DriveRuntime, state: DriveControlState, message: str) -> None:
        """Send a deterministic zero-speed command, forcing a TCP stop packet."""
        drive_runtime.stop(state, message)
        if drive_runtime.dispatcher is not None:
            drive_runtime.dispatcher.send_wheel_speeds(0.0, 0.0, force=True)

    def _unload_goal_center_cm(self) -> tuple[float, float]:
        if self.runtime.route_plan.unload_goal_cm is not None:
            return self.runtime.route_plan.unload_goal_cm
        return self.route_facade.hybrid_planner.small_goal_center_cm()

    def _unload_geometry(self) -> RobotGeometry:
        return self.robot_estimator.robot_geometry_from_params(self.latest_params)

    def _unload_alignment_metrics(self) -> dict[str, float | bool]:
        """Return reverse-alignment errors from the rear unload tip to the small goal."""
        if self.runtime.robot_pose is None:
            return {"valid": False}
        geometry = self._unload_geometry()
        robot_pose = self.runtime.robot_pose
        goal_x, goal_y = self._unload_goal_center_cm()
        planner_pose = HybridPose(robot_pose.x_cm, robot_pose.y_cm, robot_pose.heading_rad)
        rear_x, rear_y = self.route_facade.hybrid_planner.rear_unload_point_for_pose(planner_pose, geometry)
        dx = goal_x - rear_x
        dy = goal_y - rear_y
        forward = (math.cos(robot_pose.heading_rad), math.sin(robot_pose.heading_rad))
        right = (math.sin(robot_pose.heading_rad), -math.cos(robot_pose.heading_rad))
        rear_forward_error_cm = -(dx * forward[0] + dy * forward[1])
        lateral_error_cm = dx * right[0] + dy * right[1]
        distance_error_cm = math.hypot(dx, dy)
        goal_bearing_rad = math.atan2(goal_y - robot_pose.y_cm, goal_x - robot_pose.x_cm)
        desired_heading_rad = normalize_angle(goal_bearing_rad - math.pi)
        heading_error_rad = normalize_angle(desired_heading_rad - robot_pose.heading_rad)
        wall_heading_error_rad = normalize_angle(robot_pose.heading_rad)
        reach_cm = max(1e-6, geometry.rear_cm + geometry.unload_extension_cm)
        precise_heading_tolerance_rad = max(
            math.radians(1.0),
            math.atan2(self.config.drive.visual_servo_noise_floor_cm, reach_cm),
        )
        wall_heading_tolerance_rad = math.radians(30.0)
        valid = (
            distance_error_cm <= self.config.drive.visual_servo_noise_floor_cm
            and abs(lateral_error_cm) <= self.config.drive.visual_servo_noise_floor_cm
            and abs(heading_error_rad) <= precise_heading_tolerance_rad
            and abs(wall_heading_error_rad) <= wall_heading_tolerance_rad
        )
        error_score_cm = max(distance_error_cm, abs(lateral_error_cm), abs(heading_error_rad) * reach_cm)
        return {
            "valid": valid,
            "rear_x_cm": rear_x,
            "rear_y_cm": rear_y,
            "rear_forward_error_cm": rear_forward_error_cm,
            "lateral_error_cm": lateral_error_cm,
            "distance_error_cm": distance_error_cm,
            "heading_error_rad": heading_error_rad,
            "wall_heading_error_rad": wall_heading_error_rad,
            "precise_heading_tolerance_rad": precise_heading_tolerance_rad,
            "wall_heading_tolerance_rad": wall_heading_tolerance_rad,
            "error_score_cm": error_score_cm,
        }

    def _unload_alignment_turn_command(self, heading_error_rad: float) -> tuple[float, int]:
        turn_deg = -math.degrees(heading_error_rad) * self.config.drive.visual_servo_turn_kp
        turn_abs = min(abs(turn_deg), self.config.drive.visual_servo_max_turn_deg)
        turn_abs = max(turn_abs, self.config.drive.visual_servo_min_turn_deg)
        turn_deg = math.copysign(turn_abs, turn_deg)
        max_speed = max(
            self.config.drive.visual_servo_min_turn_speed_pct,
            self.config.drive.near_zone_turn_speed_pct,
        )
        speed_span = max_speed - self.config.drive.visual_servo_min_turn_speed_pct
        speed_scale = min(1.0, turn_abs / max(self.config.drive.visual_servo_max_turn_deg, 1e-6))
        speed_pct = int(round(self.config.drive.visual_servo_min_turn_speed_pct + speed_span * speed_scale))
        return turn_deg, speed_pct

    def _unload_alignment_distance_command(self, rear_forward_error_cm: float) -> tuple[str, float, int]:
        distance_cm = min(abs(rear_forward_error_cm), 3.0)
        distance_cm = max(distance_cm, self.config.drive.visual_servo_noise_floor_cm)
        speed_pct = int(round(self.config.drive.near_zone_move_speed_pct))
        command = "back" if rear_forward_error_cm > 0.0 else "move"
        return command, distance_cm, speed_pct

    def _pre_unload_pivot_metrics(self) -> dict[str, float | bool]:
        """Return the rough point-turn error needed to face the rear at the goal."""
        if self.runtime.robot_pose is None:
            return {"valid": False}
        robot_pose = self.runtime.robot_pose
        goal_x, goal_y = self._unload_goal_center_cm()
        goal_bearing_rad = math.atan2(goal_y - robot_pose.y_cm, goal_x - robot_pose.x_cm)
        desired_heading_rad = normalize_angle(goal_bearing_rad - math.pi)
        heading_error_rad = normalize_angle(desired_heading_rad - robot_pose.heading_rad)
        tolerance_rad = math.radians(self.config.drive.unload_pivot_tolerance_deg)
        return {
            "valid": abs(heading_error_rad) <= tolerance_rad,
            "heading_error_rad": heading_error_rad,
            "tolerance_rad": tolerance_rad,
        }

    def _run_pre_unload_pivot(self, drive_runtime: DriveRuntime, now_s: float) -> bool:
        """Suspend XTE and tank-steer in place until the rear roughly faces the goal."""
        if self.runtime.robot_pose is None:
            drive_runtime.stop(DriveControlState.NO_POSE, "pre-unload pivot missing robot pose")
            return True
        metrics = self._pre_unload_pivot_metrics()
        if metrics.get("valid", False):
            self.runtime.unload_state = UnloadExecutionState.ALIGNING
            self.runtime.reset_unload_alignment_state()
            self.drive_guard.wheel_controller.reset()
            self._force_drive_stop(
                drive_runtime,
                DriveControlState.PRE_UNLOAD_PIVOT,
                "pre-unload pivot complete; starting rear visual servo",
            )
            print("Unload: pre-unload pivot complete; starting reverse visual-servo alignment")
            return self._run_unload_precise_move(drive_runtime, now_s)

        heading_error_rad = float(metrics["heading_error_rad"])
        speed_pct = float(self.config.drive.unload_pivot_speed_pct)
        left_pct = -math.copysign(speed_pct, heading_error_rad)
        right_pct = math.copysign(speed_pct, heading_error_rad)
        drive_runtime.state = DriveControlState.PRE_UNLOAD_PIVOT
        drive_runtime.last_message = (
            f"Pre-unload pivot {math.degrees(heading_error_rad):.1f}deg"
        )
        drive_runtime.last_command = WheelCommand(left_pct, right_pct)
        drive_runtime.dispatcher.send_wheel_speeds(left_pct, right_pct, force=True)
        return True

    def _run_unload_precise_move(self, drive_runtime: DriveRuntime, now_s: float) -> bool:
        """Reverse visual-servo the rear unload tip onto the goal before dropping the pipe."""
        if self.runtime.robot_pose is None:
            drive_runtime.stop(DriveControlState.NO_POSE, "unload alignment missing robot pose")
            return True
        metrics = self._unload_alignment_metrics()
        if not metrics.get("valid", False) and self.runtime.unload_alignment_settle_started_s != 0.0:
            elapsed_s = time.perf_counter() - self.runtime.unload_alignment_settle_started_s
            if elapsed_s < self.config.drive.visual_servo_settle_time_s:
                self._force_drive_stop(
                    drive_runtime,
                    DriveControlState.PRECISE_MOVE,
                    f"Unload settling {elapsed_s:.2f}/{self.config.drive.visual_servo_settle_time_s:.2f}s",
                )
                return True
            self.runtime.unload_alignment_settle_started_s = 0.0
            self.runtime.unload_alignment_settle_reason = ""
            self.runtime.reset_unload_alignment_state()

        if self.runtime.unload_alignment_settle_started_s != 0.0:
            elapsed_s = time.perf_counter() - self.runtime.unload_alignment_settle_started_s
            if elapsed_s < self.config.drive.visual_servo_settle_time_s:
                self._force_drive_stop(
                    drive_runtime,
                    DriveControlState.PRECISE_MOVE,
                    f"Unload settling {elapsed_s:.2f}/{self.config.drive.visual_servo_settle_time_s:.2f}s",
                )
                return True
            if metrics.get("valid", False):
                self.runtime.unload_command_fired = True
                self.runtime.unload_started_s = now_s
                self.runtime.unload_state = UnloadExecutionState.READY
                self.runtime.reset_unload_alignment_state()
                self._force_drive_stop(drive_runtime, DriveControlState.STOPPED, "unload aligned; starting pipe sequence")
                print("Unload: rear tip aligned; wheels stopped; starting pipe-only unload sequence")
                self._start_unload_sequence_thread()
                return True
            self.runtime.reset_unload_alignment_state()
            self.runtime.unload_state = UnloadExecutionState.ALIGNING
            drive_runtime.last_message = (
                f"Unload settle drift {float(metrics['distance_error_cm']):.2f}cm; correcting"
            )

        if metrics.get("valid", False):
            self.runtime.unload_state = UnloadExecutionState.SETTLING
            self.runtime.unload_alignment_settle_started_s = time.perf_counter()
            self.runtime.unload_alignment_settle_reason = "verified candidate"
            self._force_drive_stop(drive_runtime, DriveControlState.PRECISE_MOVE, "Unload aligned; settling before drop")
            return True

        self.runtime.unload_state = UnloadExecutionState.ALIGNING
        error_score_cm = float(metrics["error_score_cm"])
        if error_score_cm + self.config.drive.visual_servo_min_improvement_cm < self.runtime.unload_alignment_best_error_cm:
            self.runtime.unload_alignment_best_error_cm = error_score_cm
            self.runtime.unload_alignment_stale_frames = 0
        else:
            self.runtime.unload_alignment_stale_frames += 1
            if self.runtime.unload_alignment_stale_frames >= self.config.drive.visual_servo_stall_frames:
                self.runtime.unload_state = UnloadExecutionState.FAILED
                self.unload_last_error = (
                    f"unload alignment stalled at {error_score_cm:.2f}cm; pipe not dropped"
                )
                self._force_drive_stop(drive_runtime, DriveControlState.DISPATCH_ERROR, self.unload_last_error)
                return True
        if self.runtime.unload_alignment_iterations >= self.config.drive.visual_servo_max_iterations:
            self.runtime.unload_state = UnloadExecutionState.FAILED
            self.unload_last_error = f"unload alignment iteration limit at {error_score_cm:.2f}cm; pipe not dropped"
            self._force_drive_stop(drive_runtime, DriveControlState.DISPATCH_ERROR, self.unload_last_error)
            return True

        try:
            heading_error_rad = float(metrics["heading_error_rad"])
            heading_tolerance_rad = float(metrics["precise_heading_tolerance_rad"])
            wall_error_rad = float(metrics["wall_heading_error_rad"])
            wall_tolerance_rad = float(metrics["wall_heading_tolerance_rad"])
            if abs(heading_error_rad) > heading_tolerance_rad or abs(wall_error_rad) > wall_tolerance_rad:
                turn_deg, turn_speed_pct = self._unload_alignment_turn_command(heading_error_rad)
                self.runtime.unload_alignment_iterations += 1
                drive_runtime.last_message = (
                    f"Unload servo turn {turn_deg:.2f}deg; rear error {float(metrics['distance_error_cm']):.2f}cm"
                )
                self._controller().turn(round(turn_deg, 2), turn_speed_pct)
                return True

            command, distance_cm, speed_pct = self._unload_alignment_distance_command(
                float(metrics["rear_forward_error_cm"])
            )
            self.runtime.unload_alignment_iterations += 1
            drive_runtime.last_message = (
                f"Unload servo {command} {distance_cm:.2f}cm; rear error {float(metrics['distance_error_cm']):.2f}cm"
            )
            if command == "back":
                self._controller().back(round(distance_cm, 2), speed_pct)
            else:
                self._controller().move(round(distance_cm, 2), speed_pct)
            return True
        except Exception as exc:
            self.runtime.unload_state = UnloadExecutionState.FAILED
            self.unload_last_error = f"unload alignment failed: {exc}"
            drive_runtime.stop(DriveControlState.DISPATCH_ERROR, self.unload_last_error)
            return True

    def _start_unload_sequence_thread(self) -> None:
        if self.unload_thread is not None and self.unload_thread.is_alive():
            return

        def run_unload_sequence() -> None:
            try:
                self.runtime.collector_state = CollectorPositionState.UNLOADING
                self.runtime.unload_state = UnloadExecutionState.UNLOADING_FIRST
                print("Unload: first unload_full_cycle")
                self._controller().unload_full_cycle()

                self.runtime.unload_state = UnloadExecutionState.PIPE_SHAKE
                shake_cycles = max(0, int(self.config.drive.unload_pipe_shake_cycles))
                shake_units = float(self.config.drive.unload_pipe_shake_units)
                shake_speed = int(round(self.config.drive.unload_pipe_shake_speed))
                for cycle in range(shake_cycles):
                    print(f"Unload: pipe shake cycle {cycle + 1}/{shake_cycles}")
                    self._controller().pipe_down(shake_units, shake_speed)
                    self._controller().pipe_up(shake_units, shake_speed)

                self.runtime.unload_state = UnloadExecutionState.UNLOADING_SECOND
                print("Unload: second unload_full_cycle")
                self._controller().unload_full_cycle()
                self._controller().pipe_stop()
                self.runtime.collector_state = CollectorPositionState.UNKNOWN
                self.runtime.unload_state = UnloadExecutionState.COMPLETE
                self.unload_last_error = ""
                print("Unload: complete")
            except Exception as exc:
                self.unload_last_error = str(exc)
                self.runtime.unload_state = UnloadExecutionState.FAILED
                print(f"Unload: failed: {exc}")
                if self.controller is not None:
                    try:
                        self._controller().pipe_stop()
                    except Exception as stop_exc:
                        print(f"Unload: pipe_stop after failure also failed: {stop_exc}")
                self.runtime.collector_state = CollectorPositionState.UNKNOWN

        self.unload_thread = threading.Thread(target=run_unload_sequence, name="ev3-unload-sequence", daemon=True)
        self.unload_thread.start()

    def update_unload_state(self, drive_runtime: DriveRuntime, now_s: float) -> bool:
        """Run the terminal pipe-only unload sequence once the unload endpoint is reached."""
        if drive_runtime.dispatcher is None:
            return False
        state = self.runtime.unload_state
        if state in (UnloadExecutionState.COMPLETE, UnloadExecutionState.FAILED):
            self.drive_guard.wheel_controller.reset()
            message = state.value if state == UnloadExecutionState.COMPLETE else f"{state.value}: {self.unload_last_error}"
            drive_runtime.stop(DriveControlState.STOPPED, message)
            return True
        if self.unload_thread is not None and self.unload_thread.is_alive():
            drive_runtime.stop(DriveControlState.STOPPED, self.runtime.unload_state.value)
            return True
        if state == UnloadExecutionState.PRE_UNLOAD_PIVOT:
            return self._run_pre_unload_pivot(drive_runtime, now_s)
        if state in (UnloadExecutionState.ALIGNING, UnloadExecutionState.SETTLING):
            return self._run_unload_precise_move(drive_runtime, now_s)
        if self.runtime.unload_command_fired:
            return False
        if self._unload_staging_intercept_reached():
            self.runtime.unload_state = UnloadExecutionState.PRE_UNLOAD_PIVOT
            self.drive_guard.wheel_controller.reset()
            self._force_drive_stop(
                drive_runtime,
                DriveControlState.PRE_UNLOAD_PIVOT,
                "unload staging reached; starting stationary pivot",
            )
            print("Unload: staging reached; starting stationary rear-facing pivot")
            return True
        if not self._unload_endpoint_reached():
            return False

        self.runtime.unload_state = UnloadExecutionState.ALIGNING
        self.runtime.reset_unload_alignment_state()
        self.drive_guard.wheel_controller.reset()
        self._force_drive_stop(drive_runtime, DriveControlState.PRECISE_MOVE, "unload endpoint reached; aligning rear tip")
        print("Unload: endpoint reached; starting reverse visual-servo alignment")
        return self._run_unload_precise_move(drive_runtime, now_s)

    def _ball_targets(self, balls: list[SmoothedBallCoordinate]) -> list[PlannedBallTarget]:
        return [
            PlannedBallTarget(
                track_id=ball.track_id,
                label=ball.label,
                x_cm=ball.cm_x,
                y_cm=ball.cm_y,
                node_cm=self.mapper.field_metric_cm_to_grid_node((ball.cm_x, ball.cm_y)),
            )
            for ball in balls
        ]

    def ball_cache_signature(
        self,
        smoothed_balls: list[SmoothedBallCoordinate],
    ) -> tuple[tuple[int, str, int, int], ...]:
        bucket = max(1.0, self.config.planner.route_target_move_invalidate_cm)
        return tuple(
            sorted(
                (
                    ball.track_id,
                    ball.label,
                    int(round(ball.cm_x / bucket)),
                    int(round(ball.cm_y / bucket)),
                )
                for ball in smoothed_balls
            )
        )

    def install_route_plan(
        self,
        route_plan: RoutePlan,
        ball_signature: tuple[tuple[int, str, int, int], ...],
        unload_extension_cm: float,
    ) -> None:
        """Publish a completed route and its cache metadata to the UI/control loop."""
        self.runtime.route_plan = route_plan
        self.runtime.route_cache_ball_signature = ball_signature
        self.runtime.route_cache_unload_extension_cm = unload_extension_cm
        self.runtime.route_failed_ball_signature = None
        self.runtime.route_failed_unload_extension_cm = None
        self.runtime.route_failed_start_signature = None
        if route_plan.active_target is None:
            self.runtime.route_cache_target_id = -1
            self.runtime.route_cache_target_label = None
            self.runtime.route_cache_target_cm = None
        else:
            target = route_plan.active_target
            self.runtime.route_cache_target_id = target.track_id
            self.runtime.route_cache_target_label = target.label
            self.runtime.route_cache_target_cm = (target.x_cm, target.y_cm)

    def mark_route_failed(
        self,
        ball_signature: tuple[tuple[int, str, int, int], ...],
        start_signature: tuple[int, int, int],
        unload_extension_cm: float,
    ) -> None:
        """Remember a failed route request so the frame loop does not resubmit it every frame."""
        self.runtime.route_failed_ball_signature = ball_signature
        self.runtime.route_failed_start_signature = start_signature
        self.runtime.route_failed_unload_extension_cm = unload_extension_cm

    def route_failure_is_current(
        self,
        ball_signature: tuple[tuple[int, str, int, int], ...],
        start_signature: tuple[int, int, int],
        unload_extension_cm: float,
    ) -> bool:
        return (
            self.runtime.route_failed_ball_signature == ball_signature
            and self.runtime.route_failed_start_signature == start_signature
            and self.runtime.route_failed_unload_extension_cm is not None
            and abs(self.runtime.route_failed_unload_extension_cm - unload_extension_cm) <= 1e-6
        )

    @staticmethod
    def route_start_signature(start_pose: HybridPose) -> tuple[int, int, int]:
        """Bucket a start pose so failed-route throttling can recover after movement."""
        return (
            int(round(start_pose.x_cm / 5.0)),
            int(round(start_pose.y_cm / 5.0)),
            int(round(normalize_angle(start_pose.theta_rad) / math.radians(15.0))),
        )

    def accept_completed_route_if_current(
        self,
        ball_signature: tuple[tuple[int, str, int, int], ...],
        unload_extension_cm: float,
        current_start_signature: tuple[int, int, int],
    ) -> None:
        """Install a finished worker route only when it still matches the current scene."""
        completed = self.async_route_planner.poll_completed()
        if completed is None:
            return
        request = completed.request
        if request.ball_signature != ball_signature or abs(request.unload_extension_cm - unload_extension_cm) > 1e-6:
            return
        if request.start_signature != current_start_signature:
            return
        if completed.error is not None:
            self.robot_runtime.warning = f"Route worker failed: {completed.error}"
            self.mark_route_failed(request.ball_signature, request.start_signature, request.unload_extension_cm)
        elif completed.route_plan.points:
            self.install_route_plan(completed.route_plan, request.ball_signature, request.unload_extension_cm)
        else:
            self.mark_route_failed(request.ball_signature, request.start_signature, request.unload_extension_cm)

    def submit_route_plan_if_needed(
        self,
        grid: np.ndarray,
        ball_targets: list[PlannedBallTarget],
        start_pose: HybridPose,
        geometry: RobotGeometry,
        ball_signature: tuple[tuple[int, str, int, int], ...],
    ) -> None:
        """Submit a route request without blocking the frame/control loop."""
        unload_extension_cm = geometry.unload_extension_cm
        start_signature = self.route_start_signature(start_pose)
        if self.async_route_planner.is_busy() or self.route_failure_is_current(ball_signature, start_signature, unload_extension_cm):
            return
        self.route_request_counter += 1
        self.async_route_planner.submit(
            RoutePlanningRequest(
                request_id=self.route_request_counter,
                grid=grid.copy(),
                ball_targets=list(ball_targets),
                start_pose=start_pose,
                geometry=geometry,
                ball_signature=ball_signature,
                start_signature=start_signature,
                unload_extension_cm=unload_extension_cm,
            )
        )

    def cached_route_is_valid(
        self,
        current_pose: HybridPose,
        smoothed_balls: list[SmoothedBallCoordinate],
        params: dict[str, object],
    ) -> bool:
        if not self.runtime.route_plan.points or self.runtime.route_cache_target_id is None:
            return False
        if not self.cached_balls_match_current_scene(smoothed_balls):
            return False
        if self.runtime.route_cache_unload_extension_cm is None:
            return False
        if abs(float(params["unload_extension_cm"]) - self.runtime.route_cache_unload_extension_cm) > 1e-6:
            return False
        if self.runtime.route_cache_target_id < 0:
            return (
                self.route_facade.nearest_route_distance_cm(current_pose, self.runtime.route_plan.points)
                <= self.config.planner.route_crosstrack_invalidate_cm
            )

        target = next((ball for ball in smoothed_balls if ball.track_id == self.runtime.route_cache_target_id), None)
        if target is None or self.runtime.route_cache_target_cm is None:
            return (
                self.route_facade.nearest_route_distance_cm(current_pose, self.runtime.route_plan.points)
                <= self.config.planner.route_crosstrack_invalidate_cm
            )
        if (
            math.hypot(target.cm_x - self.runtime.route_cache_target_cm[0], target.cm_y - self.runtime.route_cache_target_cm[1])
            > self.config.planner.route_target_move_invalidate_cm
        ):
            return False

        geometry = self.robot_estimator.robot_geometry_from_params(params)
        tube_x, tube_y = self.route_facade.hybrid_planner.tube_center_for_pose(current_pose, geometry)
        if math.hypot(target.cm_x - tube_x, target.cm_y - tube_y) <= self.config.planner.route_target_reached_cm:
            return False
        if (
            self.route_facade.nearest_route_distance_cm(current_pose, self.runtime.route_plan.points)
            > self.config.planner.route_crosstrack_invalidate_cm
        ):
            return False
        return True

    def update_route(self, result: VisionFrameResult, params: dict[str, object]) -> None:
        self.runtime.latest_smoothed_balls = result.smoothed_ball_coordinates
        if self.drive_calibration_active():
            return
        if self.runtime.pickup_state != PickupExecutionState.NAVIGATION:
            return
        if result.occupancy_grid is None:
            self.runtime.selected_start_cm = None
            self.runtime.clear_route()
            return

        geometry = self.robot_estimator.robot_geometry_from_params(params)
        if self.runtime.robot_pose is not None:
            start_pose = HybridPose(
                self.runtime.robot_pose.x_cm,
                self.runtime.robot_pose.y_cm,
                self.runtime.robot_pose.heading_rad,
            )
            self.runtime.selected_start_cm = self.mapper.field_metric_cm_to_grid_node(
                (self.runtime.robot_pose.x_cm, self.runtime.robot_pose.y_cm)
            )
        elif self.runtime.selected_ball_track_id is not None:
            selected_ball = next(
                (ball for ball in result.smoothed_ball_coordinates if ball.track_id == self.runtime.selected_ball_track_id),
                None,
            )
            if selected_ball is None:
                self.runtime.selected_start_cm = None
                self.runtime.clear_route()
                return
            start_pose = HybridPose(selected_ball.cm_x, selected_ball.cm_y, 0.0)
            self.runtime.selected_start_cm = self.mapper.field_metric_cm_to_grid_node((selected_ball.cm_x, selected_ball.cm_y))
        else:
            self.runtime.selected_start_cm = None
            if (
                self.runtime.route_plan.points
                and 0 < self.runtime.pose_missing_frames <= self.config.camera.pose_loss_clear_route_frames
            ):
                return
            self.runtime.clear_route()
            return

        smoothed_balls = result.smoothed_ball_coordinates
        route_to_unload = self.should_route_directly_to_unload()
        route_balls = [] if route_to_unload else smoothed_balls

        if not route_balls and self.runtime.robot_pose is None:
            self.runtime.selected_start_cm = None
            self.runtime.clear_route()
            return
        if not route_balls and not route_to_unload:
            if not self.runtime.route_plan.points:
                self.runtime.clear_route()
            return

        ball_signature = self.ball_cache_signature(route_balls)
        self.accept_completed_route_if_current(
            ball_signature,
            geometry.unload_extension_cm,
            self.route_start_signature(start_pose),
        )

        if self.cached_route_is_valid(start_pose, route_balls, params):
            return

        self.submit_route_plan_if_needed(
            result.occupancy_grid,
            self._ball_targets(route_balls),
            start_pose,
            geometry,
            ball_signature,
        )

    def update_robot_pose(self, topdown_frame: np.ndarray | None, params: dict[str, object]) -> None:
        if topdown_frame is None:
            self.runtime.robot_pose = None
            self.runtime.robot_topdown_px = None
            self.runtime.pose_missing_frames += 1
            return
        pose, topdown_px, observations, parallax_config = self.robot_estimator.estimate(
            topdown_frame,
            params,
            self.robot_runtime.calibration,
        )
        self.runtime.robot_pose = pose
        self.runtime.robot_topdown_px = topdown_px
        if pose is None:
            self.runtime.pose_missing_frames += 1
        else:
            self.runtime.pose_missing_frames = 0
        self.robot_runtime.latest_observations = observations
        self.robot_runtime.latest_parallax_config = parallax_config
        if self.robot_runtime.phase == RobotCalibrationPhase.STATE_CALIBRATING_SPIN:
            for marker_id, observation in observations.items():
                self.robot_runtime.collected_points.setdefault(marker_id, []).append(
                    (float(observation.ground_center[0]), float(observation.ground_center[1]))
                )

    def enforce_pose_loss_route_policy(self, drive_runtime: DriveRuntime | None = None) -> None:
        """Preserve routes through brief marker dropouts, then clear after sustained pose loss."""
        if self.runtime.robot_pose is not None or self.runtime.pose_missing_frames <= 0:
            return
        if self.runtime.pose_missing_frames <= self.config.camera.pose_loss_clear_route_frames:
            if drive_runtime is not None and self.runtime.route_plan.points:
                drive_runtime.last_message = (
                    f"pose lost {self.runtime.pose_missing_frames} frame(s); preserving route"
                )
            return
        if self.runtime.route_plan.points:
            self.runtime.clear_route()
            if drive_runtime is not None:
                drive_runtime.last_message = (
                    f"pose lost {self.runtime.pose_missing_frames} frames; route cleared"
                )

    def reset_detection_state(self) -> None:
        """Clear detection state when calibration is not valid or stream state resets."""
        self.runtime.latest_smoothed_balls = []
        self.runtime.robot_pose = None
        self.runtime.robot_topdown_px = None
        self.runtime.pose_missing_frames = 0
        self.runtime.selected_ball_track_id = None
        self.runtime.selected_start_cm = None
        self.runtime.clear_route()
        self.runtime.reset_pickup_state()
        self.latest_result = None
        self.latest_params = None
        self.latest_camera_ground_projection = None
        self.latest_camera_ground_warning = ""
        self.robot_runtime.latest_observations = {}
        self.robot_runtime.latest_parallax_config = None
        if self.active_pipeline is not None:
            self.active_pipeline.ball_smoother.reset()

    def build_missing_transform_result(
        self,
        raw_frame: np.ndarray,
        preprocessed: PreprocessedFrame,
        params: dict[str, object],
    ) -> VisionFrameResult:
        """Build the legacy placeholder-shaped result without running detection."""
        placeholder = self.renderer.make_topdown_placeholder(self.topdown_placeholder_message())
        zero_mask = np.zeros(placeholder.shape[:2], dtype=np.uint8)
        return VisionFrameResult(
            raw_frame=raw_frame,
            preprocessed=preprocessed,
            frame_for_detection=None,
            red_zones=[],
            red_mask=zero_mask,
            white_balls=[],
            orange_balls=[],
            ball_masks={"white": zero_mask.copy(), "orange": zero_mask.copy()},
            smoothed_ball_coordinates=[],
            occupancy_grid=None,
            params=params,
            metadata={"status": "missing_topdown_transform"},
        )

    def process_preprocessed_topdown(
        self,
        pipeline: VisionPipeline,
        raw_frame: np.ndarray,
        preprocessed: PreprocessedFrame,
        params: dict[str, object],
    ) -> VisionFrameResult:
        """Run detection/tracking/grid mapping after preprocessing and robot pose update."""
        if preprocessed.normalized is None:
            return self.build_missing_transform_result(raw_frame, preprocessed, params)

        frame_for_detection = preprocessed.normalized
        camera_center_pixels = (
            float(params["camera_center_x"]),
            float(params["camera_center_y"]),
        )
        red_zones, red_mask = pipeline.red_zone_detector.detect(
            frame_for_detection,
            params,
            camera_center_pixels,
        )
        white_balls, orange_balls, ball_masks = pipeline.ball_detector.detect(
            frame_for_detection,
            params,
            camera_center_pixels,
        )
        smoothed = pipeline.ball_smoother.update(white_balls + orange_balls, frame_for_detection.shape)
        occupancy_grid = pipeline.occupancy_grid_builder.build(
            frame_for_detection.shape,
            red_zones,
            dilate_for_legacy=pipeline.build_legacy_dilated_grid,
        )
        return VisionFrameResult(
            raw_frame=raw_frame,
            preprocessed=preprocessed,
            frame_for_detection=frame_for_detection,
            red_zones=red_zones,
            red_mask=red_mask,
            white_balls=white_balls,
            orange_balls=orange_balls,
            ball_masks=ball_masks,
            smoothed_ball_coordinates=smoothed,
            occupancy_grid=occupancy_grid,
            params=params,
            metadata={"status": "ok"},
        )

    def render(
        self,
        result: VisionFrameResult,
        params: dict[str, object],
        fps: float,
        drive_runtime: DriveRuntime | None,
        video_paused: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if result.frame_for_detection is None:
            placeholder_message = self.topdown_placeholder_message()
            placeholder = self.renderer.make_topdown_placeholder(placeholder_message)
            zero_mask = np.zeros(placeholder.shape[:2], dtype=np.uint8)
            masks = self.renderer.build_mask_preview(zero_mask, zero_mask, zero_mask)
            schematic = self.renderer.draw_schematic(
                frame_shape=placeholder.shape,
                red_zones=[],
                smoothed_ball_coordinates=[],
                camera_center_pixels=(float(params["camera_center_x"]), float(params["camera_center_y"])),
                route_points_cm=[],
                route_pickup_poses_cm=[],
                route_unload_pose_cm=None,
                route_unload_goal_cm=None,
                selected_start_cm=None,
                selected_ball_track_id=None,
                robot_pose=None,
                params=params,
                drive_runtime=drive_runtime,
                num_intermediate_snapshots=self.config.planner.num_intermediate_snapshots,
                route_heading_marker_interval=self.config.planner.route_heading_marker_interval,
            )
            combined = np.hstack(self.renderer.resize_to_match_height(placeholder, schematic))
            self.renderer.draw_drive_status(combined, drive_runtime)
            self.draw_ball_reconciliation_overlay(combined)
            self.draw_step_pause_overlay(combined)
            self.draw_drive_calibration_overlay(combined)
            waiting_text = (
                "Waiting for manual top-down selection"
                if self.homography_calibrator is not None
                and self.homography_calibrator.calibration_state == CalibrationState.CALIBRATING_MANUAL
                else "Waiting for ArUco auto-calibration"
            )
            cv2.putText(
                combined,
                waiting_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            if video_paused:
                self.renderer.draw_video_pause_overlay(combined)
            return combined, masks, schematic

        frame_for_debug = result.frame_for_detection if result.frame_for_detection is not None else result.preprocessed.undistorted
        annotated = self.renderer.annotate_camera_frame(
            frame_for_debug,
            result.red_zones,
            result.white_balls,
            result.orange_balls,
            fps,
        )
        self.renderer.draw_robot_marker_debug(
            annotated,
            self.robot_runtime.latest_observations,
            self.robot_runtime.calibration,
            self.runtime.robot_topdown_px,
            self.runtime.robot_pose,
            params,
            self.robot_runtime,
        )
        self.renderer.draw_robot_calibration_status(
            annotated,
            self.robot_runtime,
            self.runtime.robot_pose,
            params,
        )
        self.renderer.draw_control_xte_on_topdown(annotated, self.runtime.robot_pose, drive_runtime)
        schematic = self.renderer.draw_schematic(
            frame_shape=frame_for_debug.shape,
            red_zones=result.red_zones,
            smoothed_ball_coordinates=result.smoothed_ball_coordinates,
            camera_center_pixels=(float(params["camera_center_x"]), float(params["camera_center_y"])),
            route_points_cm=self.runtime.route_plan.points,
            route_pickup_poses_cm=self.runtime.route_plan.pickup_poses,
            route_segment_types=self.runtime.route_plan.segment_types,
            route_segment_speeds_pct=self.runtime.route_plan.segment_speeds_pct,
            route_unload_pose_cm=self.runtime.route_plan.unload_pose,
            route_unload_goal_cm=self.runtime.route_plan.unload_goal_cm,
            route_ball_obstacles=self.runtime.route_plan.ball_obstacles,
            route_ball_obstacle_radius_cm=self.runtime.route_plan.ball_obstacle_radius_cm,
            selected_start_cm=self.runtime.selected_start_cm,
            selected_ball_track_id=self.runtime.selected_ball_track_id,
            robot_pose=self.runtime.robot_pose,
            params=params,
            drive_runtime=drive_runtime,
            num_intermediate_snapshots=self.config.planner.num_intermediate_snapshots,
            route_heading_marker_interval=self.config.planner.route_heading_marker_interval,
        )
        masks = self.renderer.build_mask_preview(
            result.red_mask,
            result.ball_masks["white"],
            result.ball_masks["orange"],
        )
        combined = np.hstack(self.renderer.resize_to_match_height(annotated, schematic))
        self.renderer.draw_drive_status(combined, drive_runtime)
        self.draw_ball_reconciliation_overlay(combined)
        self.draw_step_pause_overlay(combined)
        self.draw_drive_calibration_overlay(combined)
        if video_paused:
            self.renderer.draw_video_pause_overlay(combined)
        return combined, masks, schematic

    def topdown_placeholder_message(self) -> str:
        """Return the legacy top-down placeholder text for the current calibration state."""
        if self.homography_calibrator is None:
            return "Waiting for 4 selected points"
        if self.homography_calibrator.calibration_state == CalibrationState.CALIBRATING_MANUAL:
            return "Waiting for 4 selected points"
        if not self.homography_calibrator.aruco_available:
            return "ArUco unavailable, press m for manual mode"
        return "Waiting for ArUco markers 0,1,2,3"

    def on_schematic_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or not self.runtime.latest_smoothed_balls:
            return
        click_cm = self.mapper.schematic_to_field_metric_cm((x, y))
        nearest_ball = min(
            self.runtime.latest_smoothed_balls,
            key=lambda ball: math.hypot(ball.cm_x - click_cm[0], ball.cm_y - click_cm[1]),
        )
        self.runtime.selected_ball_track_id = nearest_ball.track_id
        self.runtime.clear_route()
        if self.latest_result is not None and self.latest_params is not None:
            self.update_route(self.latest_result, self.latest_params)

    def on_manual_topdown_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
        if self.homography_calibrator is None:
            return
        width, height = self.homography_calibrator.frame_size
        if width > 0 and height > 0:
            point = (int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1)))
        else:
            point = (int(x), int(y))
        self.homography_calibrator.cursor = point
        if event == cv2.EVENT_RBUTTONDOWN:
            self.homography_calibrator.clear_manual_points()
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self.homography_calibrator.add_manual_point(point)

    def draw_selector_view(self, frame: np.ndarray) -> np.ndarray:
        view = frame.copy()
        if self.homography_calibrator is None:
            return view
        manual_mode_active = self.homography_calibrator.calibration_state in (
            CalibrationState.CALIBRATING_MANUAL,
            CalibrationState.CALIBRATED_MANUAL,
        )
        if not manual_mode_active:
            if self.homography_calibrator.latest_aruco_ids is not None and self.homography_calibrator.latest_aruco_corners:
                cv2.aruco.drawDetectedMarkers(
                    view,
                    self.homography_calibrator.latest_aruco_corners,
                    self.homography_calibrator.latest_aruco_ids,
                )
            view = self.renderer.draw_detected_aruco_centers(
                view,
                self.homography_calibrator.latest_aruco_centers,
            )
            if self.homography_calibrator.transform_matrix is not None:
                view = self.renderer.draw_projected_aruco_debug(
                    view,
                    self.homography_calibrator.transform_matrix,
                )
        view = self.renderer.draw_loupe(view, self.homography_calibrator.cursor)
        points = (
            self.homography_calibrator.current_tracked_points
            if self.homography_calibrator.current_tracked_points is not None
            else self.homography_calibrator.manual_points
        )
        points_array = np.array(points, dtype=np.float32).reshape(-1, 2)
        tracking_active = self.homography_calibrator.current_tracked_points is not None and len(points_array) == 4
        for index, point in enumerate(points_array, start=1):
            point_xy = (int(round(float(point[0]))), int(round(float(point[1]))))
            if tracking_active:
                color = (0, 255, 0) if bool(self.homography_calibrator.tracked_point_valid[index - 1]) else (0, 0, 255)
            else:
                color = (0, 0, 255)
            cv2.circle(view, point_xy, self.config.camera.point_radius, color, -1, cv2.LINE_AA)
            cv2.circle(view, point_xy, self.config.camera.point_radius + 4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                view,
                str(index),
                (point_xy[0] + 10, point_xy[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if len(points_array) >= 2:
            cv2.polylines(view, [np.round(points_array).astype(np.int32).reshape(-1, 1, 2)], False, (0, 255, 0), 2, cv2.LINE_AA)
        if len(points_array) == 4:
            ordered = self.mapper.order_points(points_array).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(view, [ordered], True, (255, 200, 0), 2, cv2.LINE_AA)

        tracking_text = ""
        if tracking_active:
            tracking_text = f" | LK valid: {int(np.count_nonzero(self.homography_calibrator.tracked_point_valid))}/4"
        help_lines = [
            f"Points: {len(self.homography_calibrator.manual_points)}/4{tracking_text}",
            f"Mode: {self.homography_calibrator.calibration_state.value}",
            "Left click: add point",
            "Right click or r: reset",
            "a: auto ArUco calibration",
            "m: manual calibration",
            "q: quit",
        ]
        if self.latest_camera_ground_projection is not None:
            projection = self.latest_camera_ground_projection
            help_lines.append(
                f"Principal point C X:{projection.camera_center_px[0]:.1f} "
                f"Y:{projection.camera_center_px[1]:.1f}px"
            )
        elif self.latest_camera_ground_warning:
            help_lines.append(self.latest_camera_ground_warning)
        for index, text in enumerate(help_lines):
            cv2.putText(
                view,
                text,
                (16, 30 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if self.homography_calibrator.transform_matrix is not None and self.homography_calibrator.calibration_state == CalibrationState.CALIBRATED_AUTO:
            status = "Top-down transform active (ArUco auto)"
            color = (0, 255, 0)
        elif self.homography_calibrator.transform_matrix is not None:
            status = "Top-down transform active (manual)"
            color = (0, 255, 0)
        elif self.homography_calibrator.calibration_state == CalibrationState.CALIBRATING_MANUAL:
            status = "Select 4 inner corners for manual top-down warp"
            color = (0, 165, 255)
        elif not self.homography_calibrator.aruco_available:
            status = "ArUco unavailable, press m for manual calibration"
            color = (0, 0, 255)
        else:
            missing = [
                str(marker_id)
                for marker_id in self.config.camera.required_aruco_ids
                if marker_id not in self.homography_calibrator.latest_aruco_centers
            ]
            status = "Scanning for ArUco markers 0,1,2,3" if not missing else f"Missing ArUco IDs: {', '.join(missing)}"
            color = (0, 165, 255)
        cv2.putText(
            view,
            status,
            (16, view.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
        return view

    def handle_robot_calibration_key(self, ascii_key: int) -> None:
        """Handle the legacy non-blocking robot origin calibration shortcuts."""
        if ascii_key == ord("c"):
            self.robot_runtime.phase = RobotCalibrationPhase.STATE_CALIBRATING_SPIN
            self.robot_runtime.calibration = None
            self.robot_runtime.warning = ""
            self.robot_runtime.collected_points = {
                marker_id: [] for marker_id in self.config.robot.marker_ids
            }
            self.robot_runtime.fitted_centers.clear()
            self.robot_runtime.ellipse_ratios.clear()
            return

        if ascii_key == ord("s") and self.robot_runtime.phase == RobotCalibrationPhase.STATE_CALIBRATING_SPIN:
            if self.robot_calibration_collector.compute_spin_centers(self.robot_runtime):
                self.robot_runtime.phase = RobotCalibrationPhase.STATE_CALIBRATING_FORWARD
            return

        if ascii_key in (10, 13) and self.robot_runtime.phase == RobotCalibrationPhase.STATE_CALIBRATING_FORWARD:
            missing = [
                marker_id
                for marker_id in self.config.robot.marker_ids
                if marker_id not in self.robot_runtime.latest_observations
            ]
            if missing:
                self.robot_runtime.warning = f"Waiting for forward-facing marker(s): {missing}"
                return
            if self.robot_runtime.latest_parallax_config is None:
                self.robot_runtime.warning = "Waiting for parallax geometry before saving."
                return
            self.robot_runtime.calibration = self.robot_calibration_collector.save_robot_calibration(
                self.config.paths.robot_calibration_file,
                self.robot_runtime,
                self.robot_runtime.latest_observations,
                self.robot_runtime.latest_parallax_config,
                self.config.camera.topdown_warp_size,
            )
            self.robot_runtime.phase = RobotCalibrationPhase.STATE_NORMAL
            self.robot_runtime.warning = f"Saved robot calibration to {self.config.paths.robot_calibration_file}"

    def save_heading_tuning_to_robot_calibration(self, tuning_offset_rad: float) -> bool:
        """Fold live heading trim into robot calibration, preserving the current pose display."""
        if self.robot_runtime.calibration is None:
            self.robot_runtime.warning = "Cannot save heading tuning: no robot calibration loaded."
            return False
        if abs(tuning_offset_rad) < 1e-9:
            self.robot_runtime.warning = "Heading tuning is already zero."
            return False

        for marker_config in self.robot_runtime.calibration.get("markers", {}).values():
            old_alpha = float(marker_config["alpha_rad"])
            old_offset = np.array(
                [float(marker_config["dx"]), float(marker_config["dy"])],
                dtype=np.float32,
            )
            new_offset = image_yaw_rotation_matrix(-tuning_offset_rad) @ old_offset
            new_alpha = normalize_angle(old_alpha - tuning_offset_rad)
            marker_config["dx"] = float(new_offset[0])
            marker_config["dy"] = float(new_offset[1])
            marker_config["alpha_rad"] = float(new_alpha)
            marker_config["alpha_deg"] = float(math.degrees(new_alpha))

        self.robot_runtime.calibration["created_unix"] = time.time()
        self.config.paths.robot_calibration_file.write_text(
            json.dumps(self.robot_runtime.calibration, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.set_trackbar_value_if_changed("heading_tuning", 180)
        self.robot_runtime.warning = f"Saved heading baseline to {self.config.paths.robot_calibration_file}"
        return True

    def handle_manual_robot_key(self, key: int, drive_runtime: DriveRuntime | None) -> None:
        """Map arrow/space keys to the legacy direct non-blocking wheel commands."""
        if drive_runtime is None or drive_runtime.dispatcher is None:
            return
        if self.drive_calibration_active():
            return
        if (key & 0xFF) == ord(" "):
            drive_runtime.stop(DriveControlState.STOPPED, "manual stop")
            return
        if (
            self.runtime.pickup_state != PickupExecutionState.NAVIGATION
            or self.runtime.unload_state != UnloadExecutionState.IDLE
            or self.runtime.collector_state in (
                CollectorPositionState.PICKUP_ASSIST,
                CollectorPositionState.UNLOADING,
            )
        ):
            return
        if key in self.config.drive.key_up_arrow:
            drive_runtime.dispatcher.send_wheel_speeds(
                self.config.drive.manual_move_speed,
                self.config.drive.manual_move_speed,
                force=True,
            )
        elif key in self.config.drive.key_down_arrow:
            drive_runtime.dispatcher.send_wheel_speeds(
                -self.config.drive.manual_move_speed,
                -self.config.drive.manual_move_speed,
                force=True,
            )
        elif key in self.config.drive.key_left_arrow:
            drive_runtime.dispatcher.send_wheel_speeds(
                -self.config.drive.manual_turn_speed,
                self.config.drive.manual_turn_speed,
                force=True,
            )
        elif key in self.config.drive.key_right_arrow:
            drive_runtime.dispatcher.send_wheel_speeds(
                self.config.drive.manual_turn_speed,
                -self.config.drive.manual_turn_speed,
                force=True,
            )

    def handle_drive_calibration_key(self, ascii_key: int, drive_runtime: DriveRuntime | None) -> bool:
        """Handle optional drive calibration keys before normal manual dispatch."""
        if ascii_key == ord("k"):
            self.start_drive_calibration(drive_runtime)
            return True
        if ascii_key == ord("y"):
            self.apply_drive_calibration(drive_runtime)
            return self.drive_calibration_active()
        if ascii_key == ord("x"):
            if self.drive_calibration_active():
                self.cancel_drive_calibration(drive_runtime)
                return True
            return False
        if ascii_key == ord(" ") and self.drive_calibration_active():
            self.cancel_drive_calibration(drive_runtime)
            return True
        return False

    def handle_key(self, key: int, drive_runtime: DriveRuntime | None = None) -> bool:
        if key in (255, -1):
            return False
        ascii_key = key & 0xFF
        if ascii_key in (27, ord("q")):
            return True
        if self.handle_drive_calibration_key(ascii_key, drive_runtime):
            return False
        if self.drive_calibration_active():
            return False
        if self.homography_calibrator is not None:
            if ascii_key == ord("r"):
                self.homography_calibrator.clear_manual_points()
                self.reset_detection_state()
            elif ascii_key == ord("a"):
                self.homography_calibrator.start_auto_calibration()
                self.reset_detection_state()
            elif ascii_key == ord("m"):
                self.homography_calibrator.start_manual_calibration()
                self.reset_detection_state()
        self.handle_robot_calibration_key(ascii_key)
        if ascii_key == ord("n"):
            self.release_step_drive_pause(drive_runtime)
        if ascii_key == ord("w"):
            self.save_heading_tuning_to_robot_calibration(float(self.read_params()["heading_tuning_rad"]))
        self.handle_manual_robot_key(key, drive_runtime)
        return False

    @staticmethod
    def handle_image_key(key: int) -> bool:
        """Image mode only quits on Esc/q, matching the legacy still-image loop."""
        ascii_key = key & 0xFF
        return ascii_key in (27, ord("q"))

    @staticmethod
    def load_image_frame(image_path: Path) -> np.ndarray:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return frame

    @staticmethod
    def resize_frame_to_size(frame_bgr: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        target_width, target_height = target_size
        frame_height, frame_width = frame_bgr.shape[:2]
        if (frame_width, frame_height) == target_size:
            return frame_bgr
        interpolation = cv2.INTER_AREA if frame_width > target_width or frame_height > target_height else cv2.INTER_LINEAR
        return cv2.resize(frame_bgr, target_size, interpolation=interpolation)

    @staticmethod
    def _try_set_capture_property(cap: cv2.VideoCapture, prop_id: int, value: float) -> tuple[bool, float]:
        ok = bool(cap.set(prop_id, value))
        readback = float(cap.get(prop_id))
        return ok, readback

    def configure_camera_prep_focus(self, cap: cv2.VideoCapture, source_name: str) -> None:
        """Allow autofocus during prep when the backend exposes the UVC property."""
        if not self.config.camera.lock_focus_after_ball_count:
            return
        if not self.config.camera.camera_autofocus_enabled_during_prep:
            return
        ok, readback = self._try_set_capture_property(cap, cv2.CAP_PROP_AUTOFOCUS, 1.0)
        print(f"Camera focus prep ({source_name}): autofocus on request ok={ok}, readback={readback:.3g}")

    def lock_camera_focus_after_prep(self, cap: cv2.VideoCapture, source_name: str) -> None:
        """Best-effort autofocus disable plus manual focus lock after ball-count prep."""
        if self.runtime.camera_focus_locked or not self.config.camera.lock_focus_after_ball_count:
            return
        autofocus_ok, autofocus_readback = self._try_set_capture_property(cap, cv2.CAP_PROP_AUTOFOCUS, 0.0)
        focus_ok, focus_readback = self._try_set_capture_property(
            cap,
            cv2.CAP_PROP_FOCUS,
            float(self.config.camera.manual_focus_value),
        )
        self.runtime.camera_focus_locked = True
        print(
            f"Camera focus lock ({source_name}): autofocus off ok={autofocus_ok}, "
            f"readback={autofocus_readback:.3g}; focus={self.config.camera.manual_focus_value:.3g} "
            f"ok={focus_ok}, readback={focus_readback:.3g}"
        )

    def run_image_mode(self, image_path: Path) -> int:
        frame = self.load_image_frame(image_path)
        pipeline = self.build_image_pipeline()
        self.create_trackbars((int(frame.shape[1]), int(frame.shape[0])))
        cv2.namedWindow(self.config.windows.schematic_window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.config.windows.schematic_window_name, self.on_schematic_mouse)

        while True:
            start = time.perf_counter()
            params = self.read_params()
            result = pipeline.process(frame, params=params, use_aruco=False, normalize_illumination=False)
            self.runtime.robot_pose = None
            self.update_route(result, params)
            self.latest_result = result
            self.latest_params = params
            combined, masks, schematic = self.render(result, params, fps=0.0, drive_runtime=None)
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
            cv2.imshow(self.config.windows.main_window_name, combined)
            cv2.imshow(self.config.windows.schematic_window_name, schematic)
            cv2.imshow(self.config.windows.mask_window_name, masks)
            if self.handle_image_key(cv2.waitKey(20)):
                break
        return 0

    def run_stream(
        self,
        cap: cv2.VideoCapture,
        source_name: str,
        balance: float,
        mode_label: str,
        frame_delay_ms: int,
        drive_enabled: bool,
        step_enabled: bool = False,
        resize_to_size: tuple[int, int] | None = None,
    ) -> int:
        if not self.config.paths.calibration_file.exists():
            print(f"Calibration file not found: {self.config.paths.calibration_file}", file=sys.stderr)
            return 1
        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            print(f"Could not read first frame from {source_name}", file=sys.stderr)
            return 1

        pipeline = self.build_stream_pipeline(balance)
        self.robot_runtime.calibration = self.robot_calibration_collector.load_robot_calibration(
            self.config.paths.robot_calibration_file,
            self.config.camera.topdown_warp_size,
        )
        drive_runtime, dispatcher = self._make_drive_runtime(drive_enabled)
        self.runtime.step_mode_enabled = bool(drive_enabled and step_enabled)
        self.runtime.step_drive_paused = self.runtime.step_mode_enabled
        self.runtime.camera_focus_locked = False
        self.configure_camera_prep_focus(cap, source_name)
        if drive_enabled:
            print(
                f"Integrated drive dispatch enabled: TCP "
                f"{self.config.drive.robot_ip}:{self.config.drive.robot_tcp_port}"
            )
            if self.runtime.step_mode_enabled:
                print("Step drive mode enabled: press 'n' to release each target run.")
        else:
            print("Integrated drive controller running with dispatch disabled; motors stay halted.")
        self.create_trackbars(self.config.camera.topdown_warp_size)
        cv2.namedWindow(self.config.windows.schematic_window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.config.windows.manual_selector_window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.config.windows.schematic_window_name, self.on_schematic_mouse)
        cv2.setMouseCallback(self.config.windows.manual_selector_window_name, self.on_manual_topdown_mouse)

        last_tick = time.perf_counter()
        raw_frame: np.ndarray | None = first_frame
        resize_notice_shown = False
        video_paused = False
        paused_views: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        try:
            while True:
                if video_paused and mode_label == "Video" and paused_views is not None:
                    selector_view, combined, schematic, masks = paused_views
                    paused_combined = combined.copy()
                    self.renderer.draw_video_pause_overlay(paused_combined)
                    cv2.imshow(self.config.windows.manual_selector_window_name, selector_view)
                    cv2.imshow(self.config.windows.main_window_name, paused_combined)
                    cv2.imshow(self.config.windows.schematic_window_name, schematic)
                    cv2.imshow(self.config.windows.mask_window_name, masks)
                    key = cv2.waitKeyEx(50)
                    ascii_key = key & 0xFF
                    if ascii_key in (27, ord("q")):
                        break
                    if ascii_key in (ord(" "), ord("p")):
                        video_paused = False
                        last_tick = time.perf_counter()
                        continue
                    if self.handle_key(key, drive_runtime):
                        break
                    continue

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
                            self.reset_detection_state()
                        else:
                            print(f"Frame read failed from {source_name}", file=sys.stderr)
                            return 1
                if resize_to_size is not None:
                    original_size = (int(raw_frame.shape[1]), int(raw_frame.shape[0]))
                    raw_frame = self.resize_frame_to_size(raw_frame, resize_to_size)
                    if original_size != resize_to_size and not resize_notice_shown:
                        print(
                            f"Resizing video frames from {original_size} to {resize_to_size} "
                            "before undistortion."
                        )
                        resize_notice_shown = True

                start = time.perf_counter()
                preprocessed = pipeline.preprocessor.process(
                    raw_frame,
                    use_aruco=True,
                    normalize_illumination=False,
                )
                selector_view = self.draw_selector_view(preprocessed.undistorted)
                manual_selection_pending = (
                    preprocessed.calibration_state == CalibrationState.CALIBRATING_MANUAL
                    and preprocessed.transform_matrix is None
                )
                if manual_selection_pending:
                    cv2.imshow(self.config.windows.manual_selector_window_name, selector_view)
                    early_key = cv2.waitKeyEx(1)
                    early_ascii = early_key & 0xFF
                    if early_ascii in (27, ord("q")):
                        break
                    if self.handle_key(early_key, drive_runtime):
                        break
                    if self.homography_calibrator is not None and self.homography_calibrator.transform_matrix is not None:
                        continue

                self.sync_camera_ground_trackbars(preprocessed)
                params = self.read_params()

                if preprocessed.transform_matrix is None or preprocessed.normalized is None:
                    self.reset_detection_state()
                    result = self.build_missing_transform_result(raw_frame, preprocessed, params)
                    if self.drive_calibration_active():
                        self._fail_drive_calibration("top-down calibration lost", emergency_stop=True)
                        drive_runtime.state = DriveControlState.CALIBRATING
                        drive_runtime.last_message = self.runtime.drive_calibration_message
                    elif self.runtime.step_mode_enabled:
                        self.runtime.step_drive_paused = True
                        drive_runtime.stop(DriveControlState.NO_ROUTE, "waiting for top-down calibration")
                    else:
                        drive_runtime.stop(DriveControlState.NO_ROUTE, "waiting for top-down calibration")
                    combined, masks, schematic = self.render(result, params, fps=0.0, drive_runtime=drive_runtime)
                    processing_ms = (time.perf_counter() - start) * 1000.0
                    cv2.putText(
                        combined,
                        f"{mode_label} mode  Proc: {processing_ms:.1f} ms"
                        + ("  Space/p: pause" if mode_label == "Video" else ""),
                        (20, combined.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(self.config.windows.manual_selector_window_name, selector_view)
                    cv2.imshow(self.config.windows.main_window_name, combined)
                    cv2.imshow(self.config.windows.schematic_window_name, schematic)
                    cv2.imshow(self.config.windows.mask_window_name, masks)
                    if mode_label == "Video":
                        paused_views = (selector_view.copy(), combined.copy(), schematic.copy(), masks.copy())
                    wait_ms = max(1, frame_delay_ms - int(round(processing_ms)))
                    key = cv2.waitKeyEx(wait_ms)
                    ascii_key = key & 0xFF
                    if ascii_key in (27, ord("q")):
                        break
                    if mode_label == "Video" and ascii_key in (ord(" "), ord("p")):
                        video_paused = True
                        drive_runtime.stop(DriveControlState.STOPPED, "video paused")
                        continue
                    if self.handle_key(key, drive_runtime):
                        break
                    raw_frame = None
                    continue

                self.update_robot_pose(preprocessed.normalized, params)
                result = self.process_preprocessed_topdown(pipeline, raw_frame, preprocessed, params)
                now = time.perf_counter()
                self.update_ball_count_reconciliation(result, now)
                if self.ball_count_initialized():
                    self.lock_camera_focus_after_prep(cap, source_name)
                self.update_route(result, params)
                self.enforce_pose_loss_route_policy(drive_runtime)
                frame_dt_s = max(1e-6, now - last_tick)
                robot_geometry = self.robot_estimator.robot_geometry_from_params(params)
                drive_waiting_for_ball_count = (
                    drive_runtime.dispatcher is not None
                    and not self.ball_count_initialized()
                )
                calibration_owns_control = self.update_drive_calibration_state(drive_runtime, now)
                if calibration_owns_control:
                    self.drive_guard.wheel_controller.reset()
                elif drive_waiting_for_ball_count:
                    if self.runtime.step_mode_enabled:
                        self.runtime.step_drive_paused = True
                    self.drive_guard.wheel_controller.reset()
                    drive_runtime.stop(DriveControlState.NO_ROUTE, "waiting for stable ball count")
                elif self.runtime.step_mode_enabled and self.runtime.step_drive_paused:
                    self.drive_guard.wheel_controller.reset()
                    drive_runtime.stop(DriveControlState.STOPPED, "step paused; press n")
                else:
                    pickup_state_before = self.runtime.pickup_state
                    pickup_owns_control = self.update_pickup_state(drive_runtime, now)
                    unload_owns_control = False if pickup_owns_control else self.update_unload_state(drive_runtime, now)
                    pickup_completed = (
                        self.runtime.step_mode_enabled
                        and pickup_state_before == PickupExecutionState.REPLAN
                        and self.runtime.pickup_state == PickupExecutionState.NAVIGATION
                    )
                    if pickup_completed:
                        self.pause_step_drive_after_target(drive_runtime)
                        pickup_owns_control = True
                    if not pickup_owns_control and not unload_owns_control:
                        self.drive_guard.enforce_xte_guard_before_replan(
                            self.runtime.robot_pose,
                            self.runtime.route_plan.points,
                            drive_runtime,
                            clear_route_cache=self.runtime.clear_route,
                        )
                        self.drive_guard.update_drive_control(
                            self.runtime.robot_pose,
                            self.runtime.route_plan.points,
                            drive_runtime,
                            clear_route_cache=self.runtime.clear_route,
                            local_goal_poses=self.runtime.route_plan.pickup_poses,
                            dt_s=frame_dt_s,
                            robot_geometry=robot_geometry,
                        )
                fps = 1.0 / frame_dt_s
                last_tick = now
                combined, masks, schematic = self.render(result, params, fps=fps, drive_runtime=drive_runtime)
                self.latest_result = result
                self.latest_params = params
                processing_ms = (time.perf_counter() - start) * 1000.0
                cv2.putText(
                    combined,
                    f"{mode_label} mode  Proc: {processing_ms:.1f} ms"
                    + ("  Space/p: pause" if mode_label == "Video" else ""),
                    (20, combined.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(self.config.windows.manual_selector_window_name, selector_view)
                cv2.imshow(self.config.windows.main_window_name, combined)
                cv2.imshow(self.config.windows.schematic_window_name, schematic)
                cv2.imshow(self.config.windows.mask_window_name, masks)
                if mode_label == "Video":
                    paused_views = (selector_view.copy(), combined.copy(), schematic.copy(), masks.copy())
                wait_ms = max(1, frame_delay_ms - int(round(processing_ms)))
                key = cv2.waitKeyEx(wait_ms)
                ascii_key = key & 0xFF
                if ascii_key in (27, ord("q")):
                    break
                if mode_label == "Video" and ascii_key in (ord(" "), ord("p")):
                    video_paused = True
                    drive_runtime.stop(DriveControlState.STOPPED, "video paused")
                    continue
                if self.handle_key(key, drive_runtime):
                    break
                raw_frame = None
        finally:
            drive_runtime.stop(DriveControlState.STOPPED, "shutdown")
            if dispatcher is not None:
                dispatcher.close()
            self.async_route_planner.close()
        return 0

    def run_live_mode(
        self,
        camera_index: int,
        balance: float,
        width: int,
        height: int,
        drive_enabled: bool,
        step_enabled: bool = False,
    ) -> int:
        if not self.config.paths.calibration_file.exists():
            print(f"Calibration file not found: {self.config.paths.calibration_file}", file=sys.stderr)
            return 1
        from vision.calibration import UndistortionProvider

        calibration_width, calibration_height = UndistortionProvider.load_calibration_image_size(
            self.config.paths.calibration_file
        )
        cap = cv2.VideoCapture(camera_index)
        print("Found camera")
        if not cap.isOpened():
            print(f"Could not open camera {camera_index}", file=sys.stderr)
            return 1
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width if width > 0 else calibration_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height if height > 0 else calibration_height)
        try:
            return self.run_stream(cap, f"camera {camera_index}", balance, "Live", 1, drive_enabled, step_enabled)
        finally:
            cap.release()

    def run_video_mode(
        self,
        video_path: Path,
        balance: float,
        resize_to_calibration: bool,
        drive_enabled: bool,
        step_enabled: bool = False,
    ) -> int:
        if not video_path.exists():
            print(f"Video file not found: {video_path}", file=sys.stderr)
            return 1
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Could not open video: {video_path}", file=sys.stderr)
            return 1
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay_ms = int(round(1000.0 / fps)) if fps and fps > 0.0 else 1
        if resize_to_calibration and not self.config.paths.calibration_file.exists():
            print(f"Calibration file not found: {self.config.paths.calibration_file}", file=sys.stderr)
            cap.release()
            return 1
        resize_to_size = None
        if resize_to_calibration:
            from vision.calibration import UndistortionProvider

            resize_to_size = UndistortionProvider.load_calibration_image_size(self.config.paths.calibration_file)
        try:
            return self.run_stream(
                cap,
                str(video_path),
                balance,
                "Video",
                frame_delay_ms,
                drive_enabled,
                step_enabled,
                resize_to_size,
            )
        finally:
            cap.release()

    def run(self, args: argparse.Namespace) -> int:
        cv2.ocl.setUseOpenCL(False)
        image_path = args.image or self.config.paths.default_image
        try:
            if args.live or self.config.camera.use_live_feed:
                return self.run_live_mode(
                    args.camera_index,
                    args.balance,
                    args.width,
                    args.height,
                    bool(args.drive),
                    bool(args.step),
                )
            if args.video is not None:
                return self.run_video_mode(
                    args.video,
                    args.balance,
                    args.resize_video_to_calibration,
                    bool(args.drive),
                    bool(args.step),
                )
            return self.run_image_mode(image_path)
        finally:
            cv2.destroyAllWindows()


def main() -> int:
    mp.freeze_support()
    app = TopdownDetectorApp()
    return app.run(TopdownDetectorApp.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
