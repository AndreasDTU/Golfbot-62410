#!/usr/bin/env python3
"""Top-down detector application shell.

This script is now intentionally small.  Domain behavior lives in the extracted
packages under ``vision/``, ``robot/``, and ``pathfinding/``.  The shell owns
only CLI parsing, OpenCV UI setup, service wiring, frame loops, and keyboard
dispatch.
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathfinding.models import HybridPose, PlannedBallTarget, RoutePlan
from pathfinding.planner import RoutePlanningFacade
from robot.control import DriveSafetyGuard
from robot.io import UdpWheelDispatcher
from robot.localization import RobotCalibrationCollector, RobotPoseEstimator, image_yaw_rotation_matrix, normalize_angle
from robot.models import DriveControlState, DriveRuntime, RobotCalibrationPhase, RobotCalibrationRuntime, RobotPose
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
    robot_pose: RobotPose | None = None
    robot_topdown_px: tuple[float, float] | None = None
    latest_smoothed_balls: list[SmoothedBallCoordinate] = field(default_factory=list)

    def clear_route(self) -> None:
        self.route_plan = RoutePlan(points=[], active_target=None, pickup_poses=[])
        self.route_cache_target_id = None
        self.route_cache_target_label = None
        self.route_cache_target_cm = None
        self.route_cache_ball_signature = ()


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
        self.renderer = DebugRenderer(self.config.field, self.config.windows, self.config.robot, self.mapper)
        self.route_facade = RoutePlanningFacade(self.config.field, self.config.robot, self.config.planner)
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
            help="Enable non-blocking UDP dispatch of autonomous left/right wheel speeds.",
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
        if key in {"robot_width_cmx10", "robot_front_cmx10", "robot_rear_cmx10", "tube_forward_cmx10"}:
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

    def sync_camera_ground_trackbars(self, result: VisionFrameResult) -> None:
        """Mirror the homography-derived camera ground projection into geometry controls."""
        projection = result.preprocessed.camera_ground_projection
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
        }

    def build_image_pipeline(self) -> VisionPipeline:
        return VisionPipeline(
            self.config,
            preprocessor=IdentityPreprocessor(),
            mapper=self.mapper,
            build_legacy_dilated_grid=False,
        )

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
        return pipeline

    def _make_drive_runtime(self, drive_enabled: bool) -> tuple[DriveRuntime, UdpWheelDispatcher | None]:
        dispatcher = (
            UdpWheelDispatcher(drive_config=self.config.drive)
            if drive_enabled
            else None
        )
        return DriveRuntime(enabled=True, dispatcher=dispatcher), dispatcher

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

    def cached_route_is_valid(
        self,
        current_pose: HybridPose,
        smoothed_balls: list[SmoothedBallCoordinate],
        params: dict[str, object],
    ) -> bool:
        if not self.runtime.route_plan.points or self.runtime.route_cache_target_id is None:
            return False
        if self.runtime.route_cache_ball_signature != self.ball_cache_signature(smoothed_balls):
            return False
        if self.runtime.route_cache_target_id < 0:
            return (
                self.route_facade.nearest_route_distance_cm(current_pose, self.runtime.route_plan.points)
                <= self.config.planner.route_crosstrack_invalidate_cm
            )

        target = next((ball for ball in smoothed_balls if ball.track_id == self.runtime.route_cache_target_id), None)
        if target is None or self.runtime.route_cache_target_cm is None:
            return False
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
        if result.occupancy_grid is None or not result.smoothed_ball_coordinates:
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
            self.runtime.clear_route()
            return

        if self.cached_route_is_valid(start_pose, result.smoothed_ball_coordinates, params):
            return

        self.runtime.route_plan = self.route_facade.plan_route(
            result.occupancy_grid,
            self._ball_targets(result.smoothed_ball_coordinates),
            start_pose,
            geometry,
        )
        self.runtime.route_cache_ball_signature = self.ball_cache_signature(result.smoothed_ball_coordinates)
        if self.runtime.route_plan.active_target is None:
            self.runtime.route_cache_target_id = -1
            self.runtime.route_cache_target_label = None
            self.runtime.route_cache_target_cm = None
        else:
            target = self.runtime.route_plan.active_target
            self.runtime.route_cache_target_id = target.track_id
            self.runtime.route_cache_target_label = target.label
            self.runtime.route_cache_target_cm = (target.x_cm, target.y_cm)

    def update_robot_pose(self, topdown_frame: np.ndarray | None, params: dict[str, object]) -> None:
        if topdown_frame is None:
            self.runtime.robot_pose = None
            self.runtime.robot_topdown_px = None
            return
        pose, topdown_px, observations, parallax_config = self.robot_estimator.estimate(
            topdown_frame,
            params,
            self.robot_runtime.calibration,
        )
        self.runtime.robot_pose = pose
        self.runtime.robot_topdown_px = topdown_px
        self.robot_runtime.latest_observations = observations
        self.robot_runtime.latest_parallax_config = parallax_config
        if self.robot_runtime.phase == RobotCalibrationPhase.STATE_CALIBRATING_SPIN:
            for marker_id, observation in observations.items():
                self.robot_runtime.collected_points.setdefault(marker_id, []).append(
                    (float(observation.ground_center[0]), float(observation.ground_center[1]))
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
        elif (key & 0xFF) == ord(" "):
            drive_runtime.stop(DriveControlState.STOPPED, "manual stop")

    def handle_key(self, key: int, drive_runtime: DriveRuntime | None = None) -> bool:
        if key in (255, -1):
            return False
        ascii_key = key & 0xFF
        if ascii_key in (27, ord("q")):
            return True
        if self.homography_calibrator is not None:
            if ascii_key == ord("r"):
                self.homography_calibrator.clear_manual_points()
            elif ascii_key == ord("a"):
                self.homography_calibrator.start_auto_calibration()
            elif ascii_key == ord("m"):
                self.homography_calibrator.start_manual_calibration()
        self.handle_robot_calibration_key(ascii_key)
        if ascii_key == ord("w"):
            self.save_heading_tuning_to_robot_calibration(float(self.read_params()["heading_tuning_rad"]))
        self.handle_manual_robot_key(key, drive_runtime)
        return False

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
            if self.handle_key(cv2.waitKey(20)):
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
        if drive_enabled:
            print(
                f"Integrated drive dispatch enabled: UDP "
                f"{self.config.drive.robot_ip}:{self.config.drive.robot_udp_port}"
            )
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
        try:
            while True:
                if raw_frame is None:
                    ok, raw_frame = cap.read()
                    if not ok or raw_frame is None:
                        if mode_label == "Video":
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ok, raw_frame = cap.read()
                            if not ok or raw_frame is None:
                                print(f"Could not restart video: {source_name}", file=sys.stderr)
                                return 1
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
                params = self.read_params()
                result = pipeline.process(raw_frame, params=params, use_aruco=True, normalize_illumination=False)
                self.sync_camera_ground_trackbars(result)
                params = self.read_params()
                selector_view = self.draw_selector_view(result.preprocessed.undistorted)
                self.update_robot_pose(result.frame_for_detection, params)
                self.update_route(result, params)
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
                )
                now = time.perf_counter()
                fps = 1.0 / max(1e-6, now - last_tick)
                last_tick = now
                combined, masks, schematic = self.render(result, params, fps=fps, drive_runtime=drive_runtime)
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
                wait_ms = max(1, frame_delay_ms - int(round(processing_ms)))
                if self.handle_key(cv2.waitKeyEx(wait_ms), drive_runtime):
                    break
                raw_frame = None
        finally:
            drive_runtime.stop(DriveControlState.STOPPED, "shutdown")
            if dispatcher is not None:
                dispatcher.close()
        return 0

    def run_live_mode(self, camera_index: int, balance: float, width: int, height: int, drive_enabled: bool) -> int:
        if not self.config.paths.calibration_file.exists():
            print(f"Calibration file not found: {self.config.paths.calibration_file}", file=sys.stderr)
            return 1
        from vision.calibration import UndistortionProvider

        calibration_width, calibration_height = UndistortionProvider.load_calibration_image_size(
            self.config.paths.calibration_file
        )
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Could not open camera {camera_index}", file=sys.stderr)
            return 1
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width if width > 0 else calibration_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height if height > 0 else calibration_height)
        try:
            return self.run_stream(cap, f"camera {camera_index}", balance, "Live", 1, drive_enabled)
        finally:
            cap.release()

    def run_video_mode(self, video_path: Path, balance: float, resize_to_calibration: bool, drive_enabled: bool) -> int:
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
            return self.run_stream(cap, str(video_path), balance, "Video", frame_delay_ms, drive_enabled, resize_to_size)
        finally:
            cap.release()

    def run(self, args: argparse.Namespace) -> int:
        cv2.ocl.setUseOpenCL(False)
        image_path = args.image or self.config.paths.default_image
        try:
            if args.live or self.config.camera.use_live_feed:
                return self.run_live_mode(args.camera_index, args.balance, args.width, args.height, bool(args.drive))
            if args.video is not None:
                return self.run_video_mode(args.video, args.balance, args.resize_video_to_calibration, bool(args.drive))
            return self.run_image_mode(image_path)
        finally:
            cv2.destroyAllWindows()


def main() -> int:
    app = TopdownDetectorApp()
    return app.run(TopdownDetectorApp.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
