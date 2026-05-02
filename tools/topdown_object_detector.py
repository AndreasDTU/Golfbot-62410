#!/usr/bin/env python3
"""Detect objects, track the robot, plan routes, and dispatch wheel commands.

This tool supports three input modes:
1. Still image input for repeatable offline tuning.
2. Live camera input for on-table tuning with red-zone HSV and geometry trackbars.
3. Recorded video input for replaying real camera runs through the live pipeline.

The output is shown as:
- Left: annotated top-down camera frame
- Right: synthetic 2D schematic of the measured field

The script intentionally keeps the detection pipeline simple and deterministic:
- HSV thresholding for red zones
- YOLOv8 inference for ball-like objects
- Hybrid A* routing in ``(x, y, theta)`` over the already-built top-down map
- Cross-track/heading tracking against the cached route
- Direct non-blocking wheel-speed dispatch for the physical robot
"""

from __future__ import annotations

import argparse
import json
import socket
from collections import deque
from enum import Enum
import heapq
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIELD_WIDTH_CM = 167.0
FIELD_HEIGHT_CM = 121.5
FIELD_GRID_WIDTH_CM = int(round(FIELD_WIDTH_CM))
FIELD_GRID_HEIGHT_CM = int(round(FIELD_HEIGHT_CM))
Z_BALL_CM = 2.0
Z_FLOOR_CM = 0.0
CALIBRATION_FILE = REPO_ROOT / "calibration_data.npz"
ROBOT_CALIBRATION_FILE = REPO_ROOT / "robot_calibration.json"
DEFAULT_IMAGE = REPO_ROOT / "Bane_undistorted_transformed_close_balls.png"
DEFAULT_VIDEO_DIR = REPO_ROOT / "videos"
WINDOW_NAME = "Top-Down Detector"
MASK_WINDOW_NAME = "Segmentation Masks"
CONTROL_COLOR_WINDOW_NAME = "HSV Controls - Colors"
CONTROL_FILTER_WINDOW_NAME = "HSV Controls - Filters"
CONTROL_GEOMETRY_WINDOW_NAME = "HSV Controls - Geometry"
MANUAL_SELECTOR_WINDOW_NAME = "Manual Top-Down Selector"
CONTROL_WINDOW_SIZE = (420, 520)
ROBOT_IP = "192.168.1.42"
ROBOT_UDP_PORT = 5556
ROBOT_COMMAND_FORMAT = "LR {left:.1f} {right:.1f}"
MAX_CROSS_TRACK_ERROR_CM = 8.0
CONTROL_BASE_SPEED_PCT = 38.0
CONTROL_MAX_SPEED_PCT = 80.0
CONTROL_HEADING_KP = 38.0
CONTROL_XTE_KP = 2.2
CONTROL_MAX_HEADING_FOR_FORWARD_RAD = math.radians(70.0)
CONTROL_MIN_SEND_INTERVAL_S = 0.02
CONTROL_COMMAND_DEADBAND_PCT = 1.0
MANUAL_MOVE_UNITS = 5
MANUAL_MOVE_SPEED = 40
MANUAL_TURN_DEGREES = 15
MANUAL_TURN_SPEED = 30
KEY_LEFT_ARROW = {2424832, 65361, 63234, 81}
KEY_UP_ARROW = {2490368, 65362, 63232, 82}
KEY_RIGHT_ARROW = {2555904, 65363, 63235, 83}
KEY_DOWN_ARROW = {2621440, 65364, 63233, 84}
TRACKBAR_NAMES = {
    "red1_h_min": "R1 H min",
    "red1_h_max": "R1 H max",
    "red2_h_min": "R2 H min",
    "red2_h_max": "R2 H max",
    "red_s_min": "R S min",
    "red_s_max": "R S max",
    "red_v_min": "R V min",
    "red_v_max": "R V max",
    "red_min_area": "R min area",
    "yolo_conf_pct": "YOLO conf %",
    "yolo_min_area": "min area",
    "yolo_max_area": "max area",
    "cam_height_cm": "Cam h cm",
    "calib_z_cm": "Border h cm",
    "cam_center_x": "Cam X cm",
    "cam_center_y": "Cam Y cm",
    "heading_tuning": "Heading Tuning",
    "robot_width_cmx10": "Robot W x10",
    "robot_front_cmx10": "Body F x10",
    "robot_rear_cmx10": "Body R x10",
    "tube_forward_cmx10": "Tube F x10",
    "tube_right_cmx10": "Tube R+50 x10",
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
    "red_min_area": CONTROL_FILTER_WINDOW_NAME,
    "yolo_conf_pct": CONTROL_FILTER_WINDOW_NAME,
    "yolo_min_area": CONTROL_FILTER_WINDOW_NAME,
    "yolo_max_area": CONTROL_FILTER_WINDOW_NAME,
    "cam_height_cm": CONTROL_GEOMETRY_WINDOW_NAME,
    "calib_z_cm": CONTROL_GEOMETRY_WINDOW_NAME,
    "cam_center_x": CONTROL_GEOMETRY_WINDOW_NAME,
    "cam_center_y": CONTROL_GEOMETRY_WINDOW_NAME,
    "heading_tuning": CONTROL_GEOMETRY_WINDOW_NAME,
    "robot_width_cmx10": CONTROL_GEOMETRY_WINDOW_NAME,
    "robot_front_cmx10": CONTROL_GEOMETRY_WINDOW_NAME,
    "robot_rear_cmx10": CONTROL_GEOMETRY_WINDOW_NAME,
    "tube_forward_cmx10": CONTROL_GEOMETRY_WINDOW_NAME,
    "tube_right_cmx10": CONTROL_GEOMETRY_WINDOW_NAME,
}

# Still-image mode is the safest default for deterministic tuning.
USE_LIVE_FEED = False
CAMERA_INDEX = 0
TOPDOWN_WARP_SIZE = (800, 600)
LOUPE_CROP_SIZE = 40
LOUPE_SCALE = 5
LOUPE_PADDING = 12
POINT_RADIUS = 6
LK_FB_MAX_ERROR_PX = 2.0
LK_EMA_ALPHA = 0.2
LK_PARAMS = {
    "winSize": (31, 31),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
}
REQUIRED_ARUCO_IDS = (0, 1, 2, 3)
WALL_THICKNESS_CM = 1.6
MARKER_OUTER_OFFSET_CM = 8.0

# Schematic sizing is kept fixed for the detector UI while coordinate math uses the measured field size.
SCHEMATIC_WIDTH_PX = 900
SCHEMATIC_HEIGHT_PX = 600
SCHEMATIC_WINDOW_NAME = "2D Schematic"
ROBOT_RADIUS_CM = 15
ROBOT_MARKER_IDS = (4,5)
ROBOT_MARKER_HEIGHT_CM = 9.0
ROBOT_AXLE_DISTANCE_CM = 13.0
ROBOT_TRACK_WIDTH_CM = 20.0
ROBOT_FRONT_EDGE_FROM_FRONT_AXLE_CM = 6.5
ROBOT_TUBE_FROM_FRONT_AXLE_CM = 10.5
ROBOT_FRONT_AXLE_FROM_ORIGIN_CM = ROBOT_AXLE_DISTANCE_CM * 0.5
ROBOT_FRONT_EDGE_FROM_ORIGIN_CM = ROBOT_FRONT_AXLE_FROM_ORIGIN_CM + ROBOT_FRONT_EDGE_FROM_FRONT_AXLE_CM
ROBOT_REAR_AXLE_FROM_ORIGIN_CM = ROBOT_AXLE_DISTANCE_CM * 0.5
ROBOT_TUBE_OFFSET_CM = ROBOT_FRONT_AXLE_FROM_ORIGIN_CM + ROBOT_TUBE_FROM_FRONT_AXLE_CM
ROBOT_FOOTPRINT_FRONT_FROM_ORIGIN_CM = ROBOT_FRONT_AXLE_FROM_ORIGIN_CM
ROBOT_FOOTPRINT_REAR_FROM_ORIGIN_CM = ROBOT_REAR_AXLE_FROM_ORIGIN_CM
ROBOT_FOOTPRINT_LENGTH_CM = ROBOT_FOOTPRINT_FRONT_FROM_ORIGIN_CM + ROBOT_FOOTPRINT_REAR_FROM_ORIGIN_CM
ROBOT_FOOTPRINT_WIDTH_CM = ROBOT_TRACK_WIDTH_CM
ROBOT_FORWARD_HEADING_OFFSET_RAD = math.pi
ROBOT_TUBE_RIGHT_OFFSET_CM = 0.0
ROBOT_TUNED_FOOTPRINT_WIDTH_CM = 20.0
ROBOT_TUNED_FOOTPRINT_FRONT_FROM_ORIGIN_CM = 8.3
ROBOT_TUNED_FOOTPRINT_REAR_FROM_ORIGIN_CM = 10.1
ROBOT_TUNED_TUBE_OFFSET_CM = 17.1
ROBOT_TUNED_TUBE_RIGHT_OFFSET_CM = 0.0
ROBOT_TUBE_WIDTH_CM = 6.0
HYBRID_THETA_BINS = 36
HYBRID_STEP_CM = 4.0
HYBRID_GOAL_TOLERANCE_CM = 4.0
HYBRID_MAX_EXPANSIONS = 12000
HYBRID_TRANSLATION_DIRECTIONS = (1.0,)
HYBRID_ROTATION_DELTAS_RAD = (
    math.radians(-10.0),
    math.radians(10.0),
)
HYBRID_IN_PLACE_ROTATION_COST = 1.1
NUM_INTERMEDIATE_SNAPSHOTS = 0
ROUTE_HEADING_MARKER_INTERVAL = 20
ROUTE_TARGET_REACHED_CM = HYBRID_GOAL_TOLERANCE_CM
ROUTE_TARGET_MOVE_INVALIDATE_CM = 5.0
ROUTE_CROSSTRACK_INVALIDATE_CM = 14.0
MIN_ROBOT_SPIN_POINTS = 20
ELLIPSE_WARNING_RATIO = 1.12
YOLO_MODEL_PATH = Path("best.pt")
if not YOLO_MODEL_PATH.exists():
    YOLO_MODEL_PATH = Path(__file__).resolve().with_name("best.pt")
YOLO_MODEL = YOLO(str(YOLO_MODEL_PATH))


class RobotCalibrationPhase(Enum):
    """Non-blocking robot origin calibration state."""

    STATE_NORMAL = "normal"
    STATE_CALIBRATING_SPIN = "calibrating_spin"
    STATE_CALIBRATING_FORWARD = "calibrating_forward"


class CalibrationState(Enum):
    """Current top-down calibration mode."""

    NEEDS_CALIBRATION = "needs_calibration"
    CALIBRATED_AUTO = "calibrated_auto"
    CALIBRATING_MANUAL = "calibrating_manual"
    CALIBRATED_MANUAL = "calibrated_manual"


class DriveControlState(Enum):
    """Master-controller dispatch state shown in the detector overlay."""

    DISABLED = "DISABLED"
    NO_POSE = "NO POSE"
    NO_ROUTE = "NO ROUTE"
    TRACKING = "TRACKING"
    REPLANNING = "REPLANNING -> MOTORS HALTED"
    STOPPED = "STOPPED"
    DISPATCH_ERROR = "DISPATCH ERROR"


@dataclass(frozen=True)
class RobotPose:
    """Robot origin and pickup tube pose in field coordinates with bottom-left cm origin."""

    x_cm: float
    y_cm: float
    heading_rad: float
    tube_x_cm: float
    tube_y_cm: float


@dataclass(frozen=True)
class RobotGeometry:
    """Live-tunable robot drawing and pickup geometry in centimeters."""

    width_cm: float
    front_cm: float
    rear_cm: float
    tube_forward_cm: float
    tube_right_cm: float


@dataclass(frozen=True)
class HybridPose:
    """One planner state in bottom-left field coordinates.

    The old planner only searched integer ``(x, y)`` grid cells.  Hybrid A*
    keeps a continuous centimeter pose and a discretized heading key, so every
    expansion can evaluate whether the robot can physically arrive at the next
    pose with its current orientation.  ``theta_rad`` follows the rest of this
    file: 0 points along +X and positive rotation points toward +Y.
    """

    x_cm: float
    y_cm: float
    theta_rad: float


@dataclass(frozen=True)
class PlannedBallTarget:
    """Route target with enough metadata for orange-first prioritization."""

    track_id: int
    label: str
    x_cm: float
    y_cm: float
    node_cm: tuple[int, int]


@dataclass(frozen=True)
class RoutePlan:
    """Cached route plus pickup metadata for visualization and invalidation.

    ``points`` is the continuous trajectory.  ``pickup_poses`` contains the
    exact final base-center poses for every successful target segment in the
    greedy route; these are the bold magenta footprints shown at ball pickup
    locations.  Cache invalidation still keys off ``active_target``, the first
    target the route is trying to collect.
    """

    points: list[HybridPose]
    active_target: PlannedBallTarget | None
    pickup_poses: list[HybridPose]


@dataclass(frozen=True)
class RouteTrackingError:
    """Closest-segment tracking error between live robot pose and cached route."""

    xte_cm: float
    signed_xte_cm: float
    heading_error_rad: float
    closest_point_cm: tuple[float, float]
    segment_heading_rad: float
    segment_index: int


@dataclass(frozen=True)
class WheelCommand:
    """Bounded differential-drive wheel command in speed percent units."""

    left_pct: float
    right_pct: float


@dataclass
class DriveRuntime:
    """Live master-controller state shared by control, dispatch, and overlays."""

    enabled: bool
    dispatcher: "UdpWheelDispatcher | None" = None
    state: DriveControlState = DriveControlState.DISABLED
    last_error: RouteTrackingError | None = None
    last_command: WheelCommand = field(default_factory=lambda: WheelCommand(0.0, 0.0))
    last_message: str = ""
    suppress_dispatch_this_frame: bool = False

    def stop(self, state: DriveControlState, message: str = "") -> None:
        """Send a deterministic zero-speed command and update overlay state."""
        previous_command = self.last_command
        previous_state = self.state
        self.state = state
        self.last_message = message
        self.last_command = WheelCommand(0.0, 0.0)
        if self.dispatcher is not None:
            force = (
                previous_state != state
                or abs(previous_command.left_pct) > 1e-6
                or abs(previous_command.right_pct) > 1e-6
            )
            self.dispatcher.send_wheel_speeds(0.0, 0.0, force=force)


class UdpWheelDispatcher:
    """Non-blocking UDP wheel-speed dispatcher for the robot microcontroller.

    The OpenCV loop must never wait for network acknowledgements.  UDP sends are
    therefore best-effort and rate-limited; command validation happens locally
    before bytes leave this process.  Configure ``ROBOT_IP``/``ROBOT_UDP_PORT``
    and make the robot firmware accept messages matching
    ``ROBOT_COMMAND_FORMAT``.
    """

    def __init__(
        self,
        host: str,
        port: int,
        command_format: str,
        min_send_interval_s: float = CONTROL_MIN_SEND_INTERVAL_S,
    ) -> None:
        self.address = (host, port)
        self.command_format = command_format
        self.min_send_interval_s = min_send_interval_s
        self.last_send_time = 0.0
        self.last_sent: tuple[float, float] | None = None
        self.last_error = ""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def close(self) -> None:
        """Close the UDP socket after sending the final stop command."""
        self.sock.close()

    def send_wheel_speeds(self, left_pct: float, right_pct: float, force: bool = False) -> bool:
        """Validate and dispatch one left/right wheel command without blocking."""
        if not (math.isfinite(left_pct) and math.isfinite(right_pct)):
            self.last_error = "non-finite wheel command rejected"
            return False

        left = float(np.clip(left_pct, -CONTROL_MAX_SPEED_PCT, CONTROL_MAX_SPEED_PCT))
        right = float(np.clip(right_pct, -CONTROL_MAX_SPEED_PCT, CONTROL_MAX_SPEED_PCT))
        now = time.perf_counter()
        if (
            not force
            and self.last_sent is not None
            and now - self.last_send_time < self.min_send_interval_s
            and abs(left - self.last_sent[0]) < CONTROL_COMMAND_DEADBAND_PCT
            and abs(right - self.last_sent[1]) < CONTROL_COMMAND_DEADBAND_PCT
        ):
            return True

        payload = self.command_format.format(left=left, right=right).encode("ascii")
        try:
            self.sock.sendto(payload, self.address)
        except (BlockingIOError, InterruptedError):
            return True
        except OSError as exc:
            self.last_error = str(exc)
            return False

        self.last_send_time = now
        self.last_sent = (left, right)
        self.last_error = ""
        return True


@dataclass(frozen=True)
class HybridPlannerConfig:
    """Deterministic Hybrid A* tuning values expressed in field centimeters.

    ``step_cm`` controls translation primitive length, ``theta_bins`` controls
    the heading lattice resolution, and ``goal_tolerance_cm`` models ball
    collection by allowing the intake/origin trajectory to stop near the ball
    rather than requiring a single exact grid cell.

    The robot is differential drive, so heading changes are modeled as pure
    in-place rotations with zero translation.  Rotation is intentionally cheap
    compared with detouring, because tank steering can reorient in tight spaces
    without needing Ackermann-style turning arcs.
    """

    step_cm: float = HYBRID_STEP_CM
    theta_bins: int = HYBRID_THETA_BINS
    goal_tolerance_cm: float = HYBRID_GOAL_TOLERANCE_CM
    max_expansions: int = HYBRID_MAX_EXPANSIONS
    translation_directions: tuple[float, ...] = HYBRID_TRANSLATION_DIRECTIONS
    rotation_deltas_rad: tuple[float, ...] = HYBRID_ROTATION_DELTAS_RAD
    in_place_rotation_cost: float = HYBRID_IN_PLACE_ROTATION_COST


class RobotFootprintCollisionChecker:
    """Check an oriented multi-circle robot model against raw red occupancy.

    The field grid is still the 1 cm map produced by the vision pipeline.  For
    speed, the checker converts that map once into a distance transform and each
    Hybrid A* candidate pose probes a small set of oriented circle centers.  A
    pose is valid when every safety-critical base circle is inside the field and
    its distance to the nearest red cell is at least its radius.  Equality is
    allowed on purpose: the rules allow gentle grazing/touching, so only true
    geometric overlap rejects a pose.

    Intake circles are modeled separately for visualization and future tuning,
    but red-zone clearance is still enforced on the wide base/wheelbase.  This
    preserves the contest strategy from the polygon checker: the narrow intake
    may reach near edge balls while the base stays safe.
    """

    def __init__(self, raw_red_grid: np.ndarray, geometry: RobotGeometry) -> None:
        self.raw_red_grid = raw_red_grid
        self.geometry = geometry
        self.height = int(raw_red_grid.shape[0])
        self.width = int(raw_red_grid.shape[1])
        free_mask = (raw_red_grid == 0).astype(np.uint8)
        self.distance_to_red = cv2.distanceTransform(free_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        self.base_circles = self._build_base_circle_specs()
        self.intake_circles = self._build_intake_circle_specs()

    def _build_base_circle_specs(self) -> list[tuple[float, float, float]]:
        """Approximate the rectangular wheelbase with overlapping circles."""
        radius = max(1.0, self.geometry.width_cm * 0.5)
        usable_start = -self.geometry.rear_cm + radius * 0.35
        usable_end = self.geometry.front_cm - radius * 0.35
        length = max(0.0, usable_end - usable_start)
        circle_count = max(2, int(math.ceil(length / max(1.0, radius * 0.9))) + 1)
        if circle_count == 1:
            offsets = [0.0]
        else:
            offsets = [
                usable_start + (usable_end - usable_start) * index / (circle_count - 1)
                for index in range(circle_count)
            ]
        return [(offset, 0.0, radius) for offset in offsets]

    def _build_intake_circle_specs(self) -> list[tuple[float, float, float]]:
        """Approximate the forward intake tube with small overlapping circles."""
        radius = max(1.0, ROBOT_TUBE_WIDTH_CM * 0.5)
        start = self.geometry.front_cm + radius
        end = self.geometry.tube_forward_cm
        if end <= start:
            return [(end, self.geometry.tube_right_cm, radius)]
        circle_count = max(2, int(math.ceil((end - start) / max(1.0, radius * 1.25))) + 1)
        return [
            (
                start + (end - start) * index / (circle_count - 1),
                self.geometry.tube_right_cm,
                radius,
            )
            for index in range(circle_count)
        ]

    def oriented_circle_centers(
        self,
        pose: HybridPose,
        circle_specs: list[tuple[float, float, float]],
    ) -> list[tuple[float, float, float]]:
        """Project local ``forward/right/radius`` circle specs into grid space."""
        forward = (math.cos(pose.theta_rad), -math.sin(pose.theta_rad))
        right = (math.sin(pose.theta_rad), math.cos(pose.theta_rad))
        center_x = pose.x_cm
        center_y = FIELD_HEIGHT_CM - pose.y_cm
        return [
            (
                center_x + forward[0] * forward_cm + right[0] * right_cm,
                center_y + forward[1] * forward_cm + right[1] * right_cm,
                radius_cm,
            )
            for forward_cm, right_cm, radius_cm in circle_specs
        ]

    def footprint_polygons(self, pose: HybridPose) -> tuple[np.ndarray, np.ndarray]:
        """Return base and intake polygons for ``pose`` in grid coordinates."""
        forward = np.array([math.cos(pose.theta_rad), -math.sin(pose.theta_rad)], dtype=np.float32)
        right = np.array([math.sin(pose.theta_rad), math.cos(pose.theta_rad)], dtype=np.float32)
        center = np.array([pose.x_cm, FIELD_HEIGHT_CM - pose.y_cm], dtype=np.float32)

        half_width_cm = self.geometry.width_cm * 0.5
        front_center = center + forward * self.geometry.front_cm
        rear_center = center - forward * self.geometry.rear_cm
        base = np.array(
            [
                front_center + right * half_width_cm,
                front_center - right * half_width_cm,
                rear_center - right * half_width_cm,
                rear_center + right * half_width_cm,
            ],
            dtype=np.float32,
        )

        tube_half_width = ROBOT_TUBE_WIDTH_CM * 0.5
        tube_front = center + forward * self.geometry.tube_forward_cm + right * self.geometry.tube_right_cm
        tube_rear = front_center + right * self.geometry.tube_right_cm
        tube = np.array(
            [
                tube_front + right * tube_half_width,
                tube_front - right * tube_half_width,
                tube_rear - right * tube_half_width,
                tube_rear + right * tube_half_width,
            ],
            dtype=np.float32,
        )
        return base, tube

    def is_pose_valid(self, pose: HybridPose) -> bool:
        """Return true when base circles have no strict red-zone intersection."""
        for x_grid, y_grid, radius_cm in self.oriented_circle_centers(pose, self.base_circles):
            if (
                x_grid - radius_cm < 0.0
                or x_grid + radius_cm > self.width - 1
                or y_grid - radius_cm < 0.0
                or y_grid + radius_cm > self.height - 1
            ):
                return False
            x_index = int(np.clip(round(x_grid), 0, self.width - 1))
            y_index = int(np.clip(round(y_grid), 0, self.height - 1))
            if float(self.distance_to_red[y_index, x_index]) < radius_cm:
                return False
        return True


@dataclass(frozen=True)
class ParallaxConfig:
    """Geometry needed to project elevated points onto the ground plane."""

    marker_height_cm: float
    camera_height_cm: float
    calibration_plane_height_cm: float
    camera_center: np.ndarray


@dataclass(frozen=True)
class CameraGroundProjection:
    """Camera ground projection in the warped top-down pixel plane."""

    principal_point_px: np.ndarray
    camera_center_px: np.ndarray


@dataclass(frozen=True)
class RobotMarkerObservation:
    """Detected robot marker after strict parallax projection to the ground plane."""

    marker_id: int
    center: np.ndarray
    ground_center: np.ndarray
    corners: np.ndarray
    ground_corners: np.ndarray
    yaw_rad: float


@dataclass
class RobotCalibrationRuntime:
    """Mutable robot calibration state used by the live detector loop."""

    phase: RobotCalibrationPhase = RobotCalibrationPhase.STATE_NORMAL
    calibration: dict[str, Any] | None = None
    collected_points: dict[int, list[tuple[float, float]]] = field(
        default_factory=lambda: {marker_id: [] for marker_id in ROBOT_MARKER_IDS}
    )
    fitted_centers: dict[int, tuple[float, float]] = field(default_factory=dict)
    ellipse_ratios: dict[int, float] = field(default_factory=dict)
    latest_observations: dict[int, RobotMarkerObservation] = field(default_factory=dict)
    latest_parallax_config: ParallaxConfig | None = None
    warning: str = ""


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
    robot_pose: RobotPose | None = None
    robot_topdown_px: tuple[float, float] | None = None
    selected_ball_track_id: int | None = None
    selected_start_cm: tuple[int, int] | None = None
    route_points_cm: list[HybridPose] | None = None
    route_pickup_poses_cm: list[HybridPose] = field(default_factory=list)
    num_intermediate_snapshots: int = NUM_INTERMEDIATE_SNAPSHOTS
    route_cache_target_id: int | None = None
    route_cache_target_label: str | None = None
    route_cache_target_cm: tuple[float, float] | None = None
    route_cache_ball_signature: tuple[tuple[int, str, int, int], ...] = field(default_factory=tuple)
    coordinate_smoother: BallCoordinateSmoother = field(default_factory=BallCoordinateSmoother)

    def clear_route_cache(self) -> None:
        """Drop cached routing state so the next update performs Hybrid A*."""
        self.route_points_cm = None
        self.route_pickup_poses_cm = []
        self.route_cache_target_id = None
        self.route_cache_target_label = None
        self.route_cache_target_cm = None
        self.route_cache_ball_signature = ()


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
    latest_gray_frame: np.ndarray | None = None
    anchor_gray_frame: np.ndarray | None = None
    anchor_points: np.ndarray | None = None
    current_tracked_points: np.ndarray | None = None
    tracked_point_valid: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=bool))
    tracked_point_errors: np.ndarray = field(default_factory=lambda: np.full(4, np.inf, dtype=np.float32))
    camera_ground_projection: CameraGroundProjection | None = None
    camera_ground_warning: str = ""

    def clear_points(self) -> None:
        self.points.clear()
        self.transform_matrix = None
        self.latest_aruco_centers.clear()
        self.camera_ground_projection = None
        self.camera_ground_warning = ""
        self.reset_manual_tracking()
        if self.calibration_state in (CalibrationState.CALIBRATING_MANUAL, CalibrationState.CALIBRATED_MANUAL):
            self.calibration_state = CalibrationState.CALIBRATING_MANUAL
        else:
            self.calibration_state = CalibrationState.NEEDS_CALIBRATION

    def start_manual_calibration(self) -> None:
        self.points.clear()
        self.transform_matrix = None
        self.latest_aruco_centers.clear()
        self.camera_ground_projection = None
        self.camera_ground_warning = ""
        self.reset_manual_tracking()
        self.calibration_state = CalibrationState.CALIBRATING_MANUAL

    def start_auto_calibration(self) -> None:
        self.points.clear()
        self.transform_matrix = None
        self.latest_aruco_centers.clear()
        self.camera_ground_projection = None
        self.camera_ground_warning = ""
        self.reset_manual_tracking()
        self.calibration_state = CalibrationState.NEEDS_CALIBRATION

    def reset_manual_tracking(self) -> None:
        self.anchor_gray_frame = None
        self.anchor_points = None
        self.current_tracked_points = None
        self.tracked_point_valid = np.zeros(4, dtype=bool)
        self.tracked_point_errors = np.full(4, np.inf, dtype=np.float32)


def noop(_value: int) -> None:
    """Trackbar callback placeholder."""
    return None


def robot_geometry_from_params(params: dict[str, object] | None) -> RobotGeometry:
    """Read live robot geometry, falling back to the measured defaults."""
    params = params or {}
    return RobotGeometry(
        width_cm=float(params.get("robot_width_cm", ROBOT_TUNED_FOOTPRINT_WIDTH_CM)),
        front_cm=float(params.get("robot_front_cm", ROBOT_TUNED_FOOTPRINT_FRONT_FROM_ORIGIN_CM)),
        rear_cm=float(params.get("robot_rear_cm", ROBOT_TUNED_FOOTPRINT_REAR_FROM_ORIGIN_CM)),
        tube_forward_cm=float(params.get("tube_forward_cm", ROBOT_TUNED_TUBE_OFFSET_CM)),
        tube_right_cm=float(params.get("tube_right_cm", ROBOT_TUNED_TUBE_RIGHT_OFFSET_CM)),
    )


def order_points(points: Any) -> np.ndarray:
    """Order 4 selected corners as top-left, top-right, bottom-right, bottom-left."""
    corners = np.array(points, dtype=np.float32).reshape(4, 2)
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


def build_manual_topdown_transform(points: Any) -> np.ndarray:
    """Compute the manual perspective transform from the 4 selected corners."""
    return cv2.getPerspectiveTransform(order_points(points), destination_corners(TOPDOWN_WARP_SIZE))


def topdown_px_to_field_cm(point_px: np.ndarray | tuple[float, float]) -> tuple[float, float]:
    """Convert top-down pixels to bottom-left-origin field centimeters."""
    width, height = TOPDOWN_WARP_SIZE
    x_cm = float(point_px[0]) * FIELD_WIDTH_CM / max(1, width - 1)
    y_cm = FIELD_HEIGHT_CM - (float(point_px[1]) * FIELD_HEIGHT_CM / max(1, height - 1))
    return float(x_cm), float(y_cm)


def field_cm_to_topdown_pixel(point_cm: tuple[float, float]) -> tuple[float, float]:
    """Convert bottom-left-origin field centimeters to top-down pixels."""
    width, height = TOPDOWN_WARP_SIZE
    x_px = float(point_cm[0]) * (width - 1) / FIELD_WIDTH_CM
    y_px = (FIELD_HEIGHT_CM - float(point_cm[1])) * (height - 1) / FIELD_HEIGHT_CM
    return float(x_px), float(y_px)


def project_principal_point_to_topdown(
    camera_matrix: np.ndarray,
    homography: np.ndarray | None,
) -> CameraGroundProjection | None:
    """Map the undistorted camera principal point through the active homography."""
    if camera_matrix.shape != (3, 3):
        return None
    if homography is None or homography.shape != (3, 3):
        return None

    principal_point = np.array(
        [[[float(camera_matrix[0, 2]), float(camera_matrix[1, 2])]]],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(principal_point, homography.astype(np.float64)).reshape(2)
    if not np.all(np.isfinite(projected)):
        return None

    return CameraGroundProjection(
        principal_point_px=principal_point.reshape(2).astype(np.float32),
        camera_center_px=projected.astype(np.float32),
    )


def update_camera_ground_projection(
    state: TopdownSelectionState,
    camera_matrix: np.ndarray,
) -> None:
    """Refresh the camera ground projection from the active top-down homography."""
    if state.transform_matrix is None:
        state.camera_ground_projection = None
        state.camera_ground_warning = ""
        return

    projection = project_principal_point_to_topdown(camera_matrix, state.transform_matrix)
    if projection is None:
        state.camera_ground_projection = None
        state.camera_ground_warning = "Principal point projection unavailable"
        return

    state.camera_ground_projection = projection
    state.camera_ground_warning = ""


def apply_automated_camera_ground_projection(
    params: dict[str, object],
    projection: CameraGroundProjection | None,
) -> dict[str, object]:
    """Override manual camera-center controls while preserving manual camera height."""
    if projection is None:
        return params

    automated = dict(params)
    automated["camera_center_x"] = float(projection.camera_center_px[0])
    automated["camera_center_y"] = float(projection.camera_center_px[1])
    return automated


def set_trackbar_if_changed(key: str, value: int) -> None:
    """Move one UI trackbar only when the displayed value needs updating."""
    name = TRACKBAR_NAMES[key]
    window_name = TRACKBAR_WINDOWS[key]
    if cv2.getTrackbarPos(name, window_name) != value:
        cv2.setTrackbarPos(name, window_name, value)


def sync_camera_ground_trackbars(projection: CameraGroundProjection | None) -> None:
    """Keep camera-center sliders aligned with the homography-projected principal point."""
    if projection is None:
        return

    x_cm, y_cm = topdown_px_to_field_cm(projection.camera_center_px)
    set_trackbar_if_changed("cam_center_x", int(np.clip(round(x_cm), 0, FIELD_GRID_WIDTH_CM)))
    set_trackbar_if_changed("cam_center_y", int(np.clip(round(y_cm), 0, FIELD_GRID_HEIGHT_CM)))


def initialize_manual_anchor_tracking(state: TopdownSelectionState) -> bool:
    """Anchor the manual corner tracker to the current undistorted grayscale frame."""
    if len(state.points) != 4 or state.latest_gray_frame is None:
        return False

    anchor_points = np.array(state.points, dtype=np.float32).reshape(4, 2)
    state.anchor_gray_frame = state.latest_gray_frame.copy()
    state.anchor_points = anchor_points
    state.current_tracked_points = anchor_points.copy()
    state.tracked_point_valid = np.ones(4, dtype=bool)
    state.tracked_point_errors = np.zeros(4, dtype=np.float32)
    state.transform_matrix = build_manual_topdown_transform(state.current_tracked_points)
    state.calibration_state = CalibrationState.CALIBRATED_MANUAL
    return True


def update_manual_anchor_tracking(state: TopdownSelectionState, live_gray: np.ndarray) -> None:
    """Track manual corners from the immutable anchor frame with FB validation."""
    if (
        state.calibration_state != CalibrationState.CALIBRATED_MANUAL
        or state.anchor_gray_frame is None
        or state.anchor_points is None
        or state.current_tracked_points is None
    ):
        return

    forward_pts, forward_status, _forward_err = cv2.calcOpticalFlowPyrLK(
        state.anchor_gray_frame,
        live_gray,
        state.anchor_points.reshape(-1, 1, 2),
        None,
        **LK_PARAMS,
    )
    if forward_pts is None or forward_status is None:
        state.tracked_point_valid = np.zeros(4, dtype=bool)
        state.tracked_point_errors = np.full(4, np.inf, dtype=np.float32)
        return

    recovered_pts, backward_status, _backward_err = cv2.calcOpticalFlowPyrLK(
        live_gray,
        state.anchor_gray_frame,
        forward_pts,
        None,
        **LK_PARAMS,
    )
    if recovered_pts is None or backward_status is None:
        state.tracked_point_valid = np.zeros(4, dtype=bool)
        state.tracked_point_errors = np.full(4, np.inf, dtype=np.float32)
        return

    recovered_flat = recovered_pts.reshape(4, 2)
    forward_flat = forward_pts.reshape(4, 2)
    anchor_flat = state.anchor_points.reshape(4, 2)
    fb_errors = np.linalg.norm(anchor_flat - recovered_flat, axis=1).astype(np.float32)
    valid = (
        (forward_status.reshape(4) == 1)
        & (backward_status.reshape(4) == 1)
        & (fb_errors < LK_FB_MAX_ERROR_PX)
        & np.isfinite(fb_errors)
        & np.all(np.isfinite(forward_flat), axis=1)
    )

    if np.any(valid):
        current = state.current_tracked_points.reshape(4, 2)
        current[valid] = (1.0 - LK_EMA_ALPHA) * current[valid] + LK_EMA_ALPHA * forward_flat[valid]
        state.current_tracked_points = current.astype(np.float32)
        state.transform_matrix = build_manual_topdown_transform(state.current_tracked_points)

    state.tracked_point_valid = valid
    state.tracked_point_errors = fb_errors


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


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def parallax_correct_point_float(point: np.ndarray, config: ParallaxConfig) -> np.ndarray:
    """Project an elevated point down to the configured calibration plane."""
    denominator = config.camera_height_cm - config.calibration_plane_height_cm
    if abs(denominator) < 1e-6:
        return point.astype(np.float32)

    scale = (config.camera_height_cm - config.marker_height_cm) / denominator
    return (config.camera_center + (point - config.camera_center) * scale).astype(np.float32)


def marker_yaw_from_ground_corners(ground_corners: np.ndarray) -> float:
    """Return marker yaw in top-down image coordinates; 0 points toward image +Y."""
    pts = ground_corners.reshape(4, 2).astype(np.float32)
    top_mid = (pts[0] + pts[1]) * 0.5
    bottom_mid = (pts[2] + pts[3]) * 0.5
    marker_forward = bottom_mid - top_mid
    return float(math.atan2(marker_forward[0], marker_forward[1]))


def image_yaw_rotation_matrix(angle: float) -> np.ndarray:
    """Rotate top-down image vectors for yaw measured from +Y with image Y down."""
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, s], [-s, c]], dtype=np.float32)


def image_yaw_to_field_heading(angle: float) -> float:
    """Convert image yaw to standard field heading, where +X is 0 and +Y is pi/2."""
    return normalize_angle(math.atan2(-math.cos(angle), math.sin(angle)))


def robot_parallax_config_from_live_params(
    params: dict[str, object],
    calibration: dict[str, Any] | None,
) -> ParallaxConfig:
    """Build robot marker parallax config from the current UI values.

    The JSON calibration contributes only marker height here. Camera center,
    camera height, and calibration-plane height must come from the live
    trackbar state every frame so robot and ball parallax stay synchronized.
    """
    marker_height_cm = ROBOT_MARKER_HEIGHT_CM
    if calibration is not None and isinstance(calibration.get("marker_height_cm"), (int, float)):
        marker_height_cm = float(calibration["marker_height_cm"])

    return ParallaxConfig(
        marker_height_cm=marker_height_cm,
        camera_height_cm=float(params["h_cam_cm"]),
        calibration_plane_height_cm=float(params["z_calib_cm"]),
        camera_center=np.array(
            [float(params["camera_center_x"]), float(params["camera_center_y"])],
            dtype=np.float32,
        ),
    )


def extract_robot_marker_observations(
    frame: np.ndarray,
    dictionary: object | None,
    detector_or_parameters: object | None,
    marker_ids: tuple[int, ...],
    parallax_config: ParallaxConfig,
) -> dict[int, RobotMarkerObservation]:
    """Detect robot markers and immediately project their geometry to the ground plane."""
    observations: dict[int, RobotMarkerObservation] = {}
    if dictionary is None or detector_or_parameters is None:
        return observations

    corners, ids = detect_aruco_markers(frame, dictionary, detector_or_parameters)
    if ids is None:
        return observations

    wanted = set(marker_ids)
    for marker_corners, raw_id in zip(corners, ids.flatten().tolist()):
        marker_id = int(raw_id)
        if marker_id not in wanted:
            continue

        pts = marker_corners.reshape(4, 2).astype(np.float32)
        ground_pts = np.array(
            [parallax_correct_point_float(point, parallax_config) for point in pts],
            dtype=np.float32,
        )
        observations[marker_id] = RobotMarkerObservation(
            marker_id=marker_id,
            center=pts.mean(axis=0).astype(np.float32),
            ground_center=ground_pts.mean(axis=0).astype(np.float32),
            corners=pts,
            ground_corners=ground_pts,
            yaw_rad=marker_yaw_from_ground_corners(ground_pts),
        )
    return observations


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

    display_points = state.current_tracked_points if state.current_tracked_points is not None else state.points
    display_points_array = np.array(display_points, dtype=np.float32).reshape(-1, 2)
    tracking_active = state.current_tracked_points is not None and len(display_points_array) == 4
    for index, point in enumerate(display_points_array, start=1):
        point_xy = (int(round(float(point[0]))), int(round(float(point[1]))))
        if tracking_active:
            is_valid = bool(state.tracked_point_valid[index - 1])
            color = (0, 255, 0) if is_valid else (0, 0, 255)
        else:
            color = (0, 0, 255)
        cv2.circle(overlay, point_xy, POINT_RADIUS, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, point_xy, POINT_RADIUS + 4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(index),
            (point_xy[0] + 10, point_xy[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if len(display_points_array) >= 2:
        polyline = np.round(display_points_array).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [polyline], False, (0, 255, 0), 2, cv2.LINE_AA)

    if len(display_points_array) == 4:
        ordered = order_points(display_points_array).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [ordered], True, (255, 200, 0), 2, cv2.LINE_AA)

    tracking_text = ""
    if tracking_active:
        valid_count = int(np.count_nonzero(state.tracked_point_valid))
        tracking_text = f" | LK valid: {valid_count}/4"
    help_lines = [
        f"Points: {len(state.points)}/4{tracking_text}",
        f"Mode: {state.calibration_state.value}",
        "Left click: add point",
        "Right click or r: reset",
        "a: auto ArUco calibration",
        "m: manual calibration",
        "q: quit",
    ]
    if state.camera_ground_projection is not None:
        projection = state.camera_ground_projection
        help_lines.append(
            f"Principal point C X:{projection.camera_center_px[0]:.1f} Y:{projection.camera_center_px[1]:.1f}px"
        )
    elif state.camera_ground_warning:
        help_lines.append(state.camera_ground_warning)
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
        click_point = (clamped_x, clamped_y)
    else:
        click_point = (int(x), int(y))
    param.cursor = click_point

    if event == cv2.EVENT_RBUTTONDOWN:
        param.clear_points()
        return

    if event != cv2.EVENT_LBUTTONDOWN or len(param.points) >= 4:
        return

    param.points.append(click_point)
    if len(param.points) == 4:
        if not initialize_manual_anchor_tracking(param):
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


def load_calibration_image_size(calibration_file: Path) -> tuple[int, int]:
    """Read the calibration image size so live capture matches the saved model."""
    data = np.load(str(calibration_file))
    image_size = tuple(int(value) for value in data["image_size"])
    return image_size


def load_undistortion_maps(
    calibration_file: Path,
    balance: float,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray, np.ndarray]:
    """Precompute fisheye undistortion state once for a stream."""
    data = np.load(str(calibration_file))
    camera_matrix = data["K"].astype(np.float64)
    dist_coeffs = data["D"].astype(np.float64)
    image_size = tuple(int(value) for value in data["image_size"])
    undistorted_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix,
        dist_coeffs,
        image_size,
        np.eye(3, dtype=np.float64),
        balance=balance,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        np.eye(3, dtype=np.float64),
        undistorted_camera_matrix,
        image_size,
        cv2.CV_32FC1,
    )
    return undistorted_camera_matrix.astype(np.float64), image_size, map1, map2


def create_hsv_trackbars(frame_size: tuple[int, int]) -> None:
    """Create trackbars for red zones, camera geometry, and robot geometry.

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
        "red_s_min": 155,
        "red_s_max": 255,
        "red_v_min": 174,
        "red_v_max": 255,
        "red_min_area": 400,
        "yolo_conf_pct": 50,
        "yolo_min_area": 157,
        "yolo_max_area": 1580,
        "cam_height_cm": 179,
        "calib_z_cm": 7,
        "cam_center_x": int(round(FIELD_WIDTH_CM * 0.5)),
        "cam_center_y": int(round(FIELD_HEIGHT_CM * 0.5)),
        "heading_tuning": 180,
        "robot_width_cmx10": int(round(ROBOT_TUNED_FOOTPRINT_WIDTH_CM * 10.0)),
        "robot_front_cmx10": int(round(ROBOT_TUNED_FOOTPRINT_FRONT_FROM_ORIGIN_CM * 10.0)),
        "robot_rear_cmx10": int(round(ROBOT_TUNED_FOOTPRINT_REAR_FROM_ORIGIN_CM * 10.0)),
        "tube_forward_cmx10": int(round(ROBOT_TUNED_TUBE_OFFSET_CM * 10.0)),
        "tube_right_cmx10": int(round((ROBOT_TUNED_TUBE_RIGHT_OFFSET_CM + 50.0) * 10.0)),
    }
    for key, value in defaults.items():
        name = TRACKBAR_NAMES[key]
        window_name = TRACKBAR_WINDOWS[key]
        max_value = 179 if "_h_" in key else 255
        if key.endswith("min_area"):
            max_value = 20000
        if key == "yolo_max_area":
            max_value = 20000
        if key == "yolo_conf_pct":
            max_value = 100
        if key == "cam_height_cm":
            max_value = 300
        if key == "calib_z_cm":
            max_value = 30
        if key == "cam_center_x":
            max_value = FIELD_GRID_WIDTH_CM
        if key == "cam_center_y":
            max_value = FIELD_GRID_HEIGHT_CM
        if key == "heading_tuning":
            max_value = 360
        if key in {
            "robot_width_cmx10",
            "robot_front_cmx10",
            "robot_rear_cmx10",
            "tube_forward_cmx10",
        }:
            max_value = 500
        if key == "tube_right_cmx10":
            max_value = 1000
        cv2.createTrackbar(name, window_name, value, max_value, noop)


def get_trackbar_value(key: str) -> int:
    """Read a trackbar value from the window that owns it."""
    return cv2.getTrackbarPos(TRACKBAR_NAMES[key], TRACKBAR_WINDOWS[key])


def read_hsv_ranges() -> dict[str, object]:
    """Read current red-zone threshold and geometry parameters from the trackbars."""
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
    camera_center_px = field_cm_to_topdown_pixel(
        (
            float(get_trackbar_value("cam_center_x")),
            float(get_trackbar_value("cam_center_y")),
        )
    )
    return {
        "red_1": red_1,
        "red_2": red_2,
        "red_min_area": float(get_trackbar_value("red_min_area")),
        "yolo_confidence": float(get_trackbar_value("yolo_conf_pct")) / 100.0,
        "yolo_min_area": float(get_trackbar_value("yolo_min_area")),
        "yolo_max_area": float(get_trackbar_value("yolo_max_area")),
        "h_cam_cm": float(get_trackbar_value("cam_height_cm")),
        "z_calib_cm": float(get_trackbar_value("calib_z_cm")),
        "camera_center_x": float(camera_center_px[0]),
        "camera_center_y": float(camera_center_px[1]),
        "camera_center_x_cm": float(get_trackbar_value("cam_center_x")),
        "camera_center_y_cm": float(get_trackbar_value("cam_center_y")),
        "heading_tuning_rad": math.radians(float(get_trackbar_value("heading_tuning")) - 180.0),
        "robot_width_cm": float(get_trackbar_value("robot_width_cmx10")) / 10.0,
        "robot_front_cm": float(get_trackbar_value("robot_front_cmx10")) / 10.0,
        "robot_rear_cm": float(get_trackbar_value("robot_rear_cmx10")) / 10.0,
        "tube_forward_cm": float(get_trackbar_value("tube_forward_cmx10")) / 10.0,
        "tube_right_cm": float(get_trackbar_value("tube_right_cmx10")) / 10.0 - 50.0,
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


def detect_balls(
    frame_bgr: np.ndarray,
    params: dict[str, object],
    camera_center_pixels: tuple[float, float],
) -> tuple[list[BallDetection], list[BallDetection], dict[str, np.ndarray]]:
    """Detect white and orange balls with YOLO while preserving the old output shape."""
    white_detections: list[BallDetection] = []
    orange_detections: list[BallDetection] = []
    white_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    orange_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    label_by_class_name = {
        "branca": "white",
        "laranja": "orange",
    }
    confidence_threshold = float(params["yolo_confidence"])
    min_area = float(params["yolo_min_area"])
    max_area = max(min_area, float(params["yolo_max_area"]))

    results = YOLO_MODEL(frame_bgr, verbose=False)[0]
    if results.boxes is not None:
        for box in results.boxes:
            confidence = float(box.conf[0].cpu())
            if confidence < confidence_threshold:
                continue

            class_index = int(box.cls[0].cpu())
            class_name = str(results.names[class_index]).strip().lower()
            label = label_by_class_name.get(class_name)
            if label is None:
                continue

            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].cpu().tolist())
            x1_i = int(round(np.clip(x1, 0, frame_bgr.shape[1] - 1)))
            y1_i = int(round(np.clip(y1, 0, frame_bgr.shape[0] - 1)))
            x2_i = int(round(np.clip(x2, 0, frame_bgr.shape[1] - 1)))
            y2_i = int(round(np.clip(y2, 0, frame_bgr.shape[0] - 1)))
            if x2_i <= x1_i or y2_i <= y1_i:
                continue

            area = float((x2_i - x1_i) * (y2_i - y1_i))
            if area < min_area or area > max_area:
                continue

            center = ((x1_i + x2_i) // 2, (y1_i + y2_i) // 2)
            radius_px = max(2, int(round(max(x2_i - x1_i, y2_i - y1_i) * 0.5)))
            contour = np.array(
                [
                    [[x1_i, y1_i]],
                    [[x2_i, y1_i]],
                    [[x2_i, y2_i]],
                    [[x1_i, y2_i]],
                ],
                dtype=np.int32,
            )
            detection = BallDetection(
                label=label,
                center=center,
                corrected_center=correct_parallax(
                    pixel_coord=center,
                    z_object_cm=Z_BALL_CM,
                    h_cam_cm=float(params["h_cam_cm"]),
                    z_calib_cm=float(params["z_calib_cm"]),
                    camera_center_pixels=camera_center_pixels,
                ),
                radius_px=radius_px,
                contour=contour,
                area=area,
                circularity=confidence,
            )

            if label == "white":
                white_detections.append(detection)
                cv2.circle(white_mask, center, radius_px, 255, -1, cv2.LINE_AA)
            else:
                orange_detections.append(detection)
                cv2.circle(orange_mask, center, radius_px, 255, -1, cv2.LINE_AA)

    white_detections.sort(key=lambda ball: (ball.center[1], ball.center[0]))
    orange_detections.sort(key=lambda ball: (ball.center[1], ball.center[0]))
    masks = {"white": white_mask, "orange": orange_mask}
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


def pixel_float_to_field_cm(point: np.ndarray, source_size: tuple[int, int]) -> tuple[float, float]:
    """Convert a floating-point top-down pixel coordinate to bottom-left field cm."""
    return pixel_to_field_cm((int(round(float(point[0]))), int(round(float(point[1])))), source_size)


def scale_robot_calibration_to_topdown(calibration: dict[str, Any], topdown_size: tuple[int, int]) -> dict[str, Any]:
    """Scale saved top-down pixel calibration values if the detector warp size differs."""
    stored_size = calibration.get("topdown_size")
    if not isinstance(stored_size, list) or len(stored_size) != 2:
        return calibration

    stored_width = float(stored_size[0])
    stored_height = float(stored_size[1])
    target_width = float(topdown_size[0])
    target_height = float(topdown_size[1])
    if stored_width <= 0.0 or stored_height <= 0.0:
        return calibration
    if int(stored_width) == topdown_size[0] and int(stored_height) == topdown_size[1]:
        return calibration

    scale_x = target_width / stored_width
    scale_y = target_height / stored_height
    scaled = json.loads(json.dumps(calibration))
    for key, scale in (("camera_center_x", scale_x), ("camera_center_y", scale_y)):
        if isinstance(scaled.get(key), (int, float)):
            scaled[key] = float(scaled[key]) * scale

    for marker_config in scaled.get("markers", {}).values():
        for key, scale in (("dx", scale_x), ("origin_x", scale_x), ("dy", scale_y), ("origin_y", scale_y)):
            if isinstance(marker_config.get(key), (int, float)):
                marker_config[key] = float(marker_config[key]) * scale
    scaled["topdown_size"] = [topdown_size[0], topdown_size[1]]
    print(
        f"Scaled robot calibration from {(int(stored_width), int(stored_height))} to {topdown_size}. "
        "Recalibrate in this detector for best accuracy.",
        file=sys.stderr,
    )
    return scaled


def load_robot_calibration(path: Path, marker_ids: tuple[int, ...], topdown_size: tuple[int, int]) -> dict[str, Any] | None:
    """Load robot marker offsets from JSON if all requested marker IDs are present."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)

    markers = calibration.get("markers", {})
    if not all(str(marker_id) in markers for marker_id in marker_ids):
        print(f"Robot calibration missing marker IDs {marker_ids}: {path}", file=sys.stderr)
        return None
    return scale_robot_calibration_to_topdown(calibration, topdown_size)


def fit_circle(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Fit a deterministic enclosing circle to collected spin points."""
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    (x, y), radius = cv2.minEnclosingCircle(pts)
    return float(x), float(y), float(radius)


def ellipse_ratio(points: list[tuple[float, float]]) -> float | None:
    """Estimate how circular the spin path is; >1 means ellipse-like."""
    if len(points) < 5:
        return None
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    _center, axes, _angle = cv2.fitEllipse(pts)
    minor = max(1e-6, float(min(axes)))
    major = float(max(axes))
    return major / minor


def compute_robot_spin_centers(runtime: RobotCalibrationRuntime) -> bool:
    """Finalize spin collection by fitting one circle center per visible robot marker."""
    runtime.fitted_centers.clear()
    runtime.ellipse_ratios.clear()
    warnings: list[str] = []

    for marker_id in ROBOT_MARKER_IDS:
        points = runtime.collected_points.get(marker_id, [])
        if len(points) < MIN_ROBOT_SPIN_POINTS:
            runtime.warning = f"Need {MIN_ROBOT_SPIN_POINTS} spin points for ID {marker_id}; got {len(points)}."
            return False
        xc, yc, _radius = fit_circle(points)
        runtime.fitted_centers[marker_id] = (xc, yc)
        ratio = ellipse_ratio(points)
        if ratio is not None:
            runtime.ellipse_ratios[marker_id] = ratio
            if ratio > ELLIPSE_WARNING_RATIO:
                warnings.append(f"ID {marker_id} ellipse ratio {ratio:.2f}")

    runtime.warning = "WARNING: Elliptical spin path. Check floor slip or homography." if warnings else ""
    return True


def save_robot_calibration(
    path: Path,
    runtime: RobotCalibrationRuntime,
    observations: dict[int, RobotMarkerObservation],
    parallax_config: ParallaxConfig,
    topdown_size: tuple[int, int],
) -> dict[str, Any]:
    """Save marker-to-robot-origin offsets from the forward-facing alignment frame."""
    markers: dict[str, dict[str, float]] = {}
    for marker_id in ROBOT_MARKER_IDS:
        observation = observations[marker_id]
        origin = runtime.fitted_centers[marker_id]
        dx = float(observation.ground_center[0] - origin[0])
        dy = float(observation.ground_center[1] - origin[1])
        alpha_rad = float(observation.yaw_rad)
        markers[str(marker_id)] = {
            "dx": dx,
            "dy": dy,
            "alpha_rad": alpha_rad,
            "alpha_deg": math.degrees(alpha_rad),
            "origin_x": float(origin[0]),
            "origin_y": float(origin[1]),
            "ellipse_ratio": float(runtime.ellipse_ratios.get(marker_id, 0.0)),
        }

    calibration = {
        "version": 1,
        "created_unix": time.time(),
        "marker_ids": list(ROBOT_MARKER_IDS),
        "marker_height_cm": float(parallax_config.marker_height_cm),
        "camera_height_cm": float(parallax_config.camera_height_cm),
        "calibration_plane_height_cm": float(parallax_config.calibration_plane_height_cm),
        "camera_center_x": float(parallax_config.camera_center[0]),
        "camera_center_y": float(parallax_config.camera_center[1]),
        "topdown_size": [int(topdown_size[0]), int(topdown_size[1])],
        "markers": markers,
    }
    path.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")
    return calibration


def robot_origin_from_observation(
    observation: RobotMarkerObservation,
    marker_calibration: dict[str, float],
) -> tuple[np.ndarray, float]:
    """Apply saved offset after parallax correction to get the ground robot origin."""
    offset = np.array([marker_calibration["dx"], marker_calibration["dy"]], dtype=np.float32)
    true_image_heading = normalize_angle(observation.yaw_rad - float(marker_calibration["alpha_rad"]))
    rotated_offset = image_yaw_rotation_matrix(true_image_heading) @ offset
    return observation.ground_center - rotated_offset, true_image_heading


def estimate_robot_pose(
    frame_bgr: np.ndarray,
    params: dict[str, object],
    calibration: dict[str, Any] | None,
    dictionary: object | None,
    detector_or_parameters: object | None,
) -> tuple[
    RobotPose | None,
    tuple[float, float] | None,
    dict[int, RobotMarkerObservation],
    ParallaxConfig,
]:
    """Estimate the robot pose from the current warped frame and live camera state.

    This function intentionally recalculates everything from scratch every
    frame. The live ``params`` dictionary is the same trackbar/state object
    used by ball parallax correction, so changing camera center/height or a
    refreshed top-down homography immediately affects the robot schematic pose.
    """
    parallax_config = robot_parallax_config_from_live_params(params, calibration)
    observations = extract_robot_marker_observations(
        frame_bgr,
        dictionary,
        detector_or_parameters,
        ROBOT_MARKER_IDS,
        parallax_config,
    )

    if calibration is None:
        return None, None, observations, parallax_config

    origins: list[np.ndarray] = []
    field_headings: list[float] = []
    for marker_id, observation in observations.items():
        marker_config = calibration.get("markers", {}).get(str(marker_id))
        if marker_config is None:
            continue
        origin_px, true_image_heading = robot_origin_from_observation(observation, marker_config)
        origins.append(origin_px)
        field_headings.append(image_yaw_to_field_heading(true_image_heading))

    if not origins:
        return None, None, observations, parallax_config

    origin = np.mean(np.array(origins, dtype=np.float32), axis=0)
    source_height, source_width = frame_bgr.shape[:2]
    x_cm, y_cm = pixel_float_to_field_cm(origin, (source_width, source_height))
    heading_rad = math.atan2(
        sum(math.sin(angle) for angle in field_headings),
        sum(math.cos(angle) for angle in field_headings),
    )
    heading_rad = normalize_angle(
        heading_rad
        + ROBOT_FORWARD_HEADING_OFFSET_RAD
        + float(params.get("heading_tuning_rad", 0.0))
    )
    geometry = robot_geometry_from_params(params)
    forward = (math.cos(heading_rad), math.sin(heading_rad))
    right = (math.sin(heading_rad), -math.cos(heading_rad))
    tube_x_cm = x_cm + forward[0] * geometry.tube_forward_cm + right[0] * geometry.tube_right_cm
    tube_y_cm = y_cm + forward[1] * geometry.tube_forward_cm + right[1] * geometry.tube_right_cm
    return (
        RobotPose(
            x_cm=x_cm,
            y_cm=y_cm,
            heading_rad=heading_rad,
            tube_x_cm=tube_x_cm,
            tube_y_cm=tube_y_cm,
        ),
        (float(origin[0]), float(origin[1])),
        observations,
        parallax_config,
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


def build_occupancy_grid(
    frame_shape: tuple[int, int, int],
    red_zones: list[RedZoneDetection],
    dilate_for_legacy: bool = True,
) -> np.ndarray:
    """Build a 1 cm binary occupancy grid from detected red zones.

    Legacy 2D A* callers receive the historical circular robot-radius dilation
    by default.  Hybrid A* callers pass ``dilate_for_legacy=False`` to keep the
    grid raw and delegate clearance to
    :class:`RobotFootprintCollisionChecker`, which tests the oriented base
    polygon at each candidate ``(x, y, theta)`` pose.  This preserves the
    external vision-to-map behavior while avoiding false blocking in the
    detector UI's asymmetric robot planner.
    """
    source_height, source_width = frame_shape[:2]
    grid = np.zeros((FIELD_GRID_HEIGHT_CM, FIELD_GRID_WIDTH_CM), dtype=np.uint8)

    for zone in red_zones:
        grid_contour = contour_to_field_grid(zone.corrected_contour, (source_width, source_height))
        cv2.fillPoly(grid, [grid_contour], 1)

    if not dilate_for_legacy:
        return (grid > 0).astype(np.uint8)

    kernel_size = max(1, int(2 * ROBOT_RADIUS_CM + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(grid, kernel, iterations=1)
    return (dilated > 0).astype(np.uint8)


def normalize_planner_angle(theta_rad: float) -> float:
    """Normalize planner headings to ``[-pi, pi)`` for stable state keys."""
    return (theta_rad + math.pi) % (2.0 * math.pi) - math.pi


def theta_bin(theta_rad: float, theta_bins: int) -> int:
    """Discretize a continuous heading into a deterministic Hybrid A* bin."""
    normalized = normalize_planner_angle(theta_rad)
    return int(round((normalized + math.pi) * theta_bins / (2.0 * math.pi))) % theta_bins


def hybrid_state_key(pose: HybridPose, theta_bins: int) -> tuple[int, int, int]:
    """Return the closed-set key for a continuous Hybrid A* pose."""
    return (
        int(round(pose.x_cm)),
        int(round(FIELD_HEIGHT_CM - pose.y_cm)),
        theta_bin(pose.theta_rad, theta_bins),
    )


def tube_center_for_pose(pose: HybridPose, geometry: RobotGeometry) -> tuple[float, float]:
    """Return the field-coordinate pickup point at the intake tip for ``pose``."""
    forward = (math.cos(pose.theta_rad), math.sin(pose.theta_rad))
    right = (math.sin(pose.theta_rad), -math.cos(pose.theta_rad))
    return (
        pose.x_cm + forward[0] * geometry.tube_forward_cm + right[0] * geometry.tube_right_cm,
        pose.y_cm + forward[1] * geometry.tube_forward_cm + right[1] * geometry.tube_right_cm,
    )


def goal_to_field_metric_cm(
    goal_node: tuple[int, int],
    goal_point_cm: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Return exact target field centimeters, falling back to a grid node."""
    if goal_point_cm is not None:
        return float(goal_point_cm[0]), float(goal_point_cm[1])
    return float(goal_node[0]), FIELD_HEIGHT_CM - float(goal_node[1])


def pickup_aligned_pose_for_theta(
    goal_node: tuple[int, int],
    theta_rad: float,
    geometry: RobotGeometry,
    goal_point_cm: tuple[float, float] | None = None,
) -> HybridPose:
    """Return the exact base pose whose intake tip lands on the ball.

    For a centered intake this is exactly:

    ``base_x = ball_x - cos(theta) * intake_length``
    ``base_y = ball_y - sin(theta) * intake_length``

    where ``intake_length`` is ``geometry.tube_forward_cm``, the tuned physical
    distance from the robot pivot/origin to the pickup mechanism.  The exact
    smoothed ball coordinate is used when available, avoiding the half-cell
    error that would come from snapping the final pickup pose to a 1 cm grid
    node.  If the live geometry has a non-zero lateral tube offset, that offset
    is also subtracted so the same visualized pickup point still overlaps the
    ball.
    """
    ball_x, ball_y = goal_to_field_metric_cm(goal_node, goal_point_cm)
    forward = (math.cos(theta_rad), math.sin(theta_rad))
    right = (math.sin(theta_rad), -math.cos(theta_rad))
    intake_length_cm = geometry.tube_forward_cm
    return HybridPose(
        x_cm=ball_x - forward[0] * intake_length_cm - right[0] * geometry.tube_right_cm,
        y_cm=ball_y - forward[1] * intake_length_cm - right[1] * geometry.tube_right_cm,
        theta_rad=normalize_planner_angle(theta_rad),
    )


def hybrid_goal_distance(
    pose: HybridPose,
    goal_node: tuple[int, int],
    geometry: RobotGeometry,
    goal_point_cm: tuple[float, float] | None = None,
) -> float:
    """Distance from the pickup point at the intake tip to the target ball."""
    goal_x, goal_y = goal_to_field_metric_cm(goal_node, goal_point_cm)
    tube_x, tube_y = tube_center_for_pose(pose, geometry)
    return math.hypot(goal_x - tube_x, goal_y - tube_y)


def reconstruct_hybrid_path(
    came_from: dict[tuple[int, int, int], tuple[int, int, int]],
    pose_by_key: dict[tuple[int, int, int], HybridPose],
    goal_key: tuple[int, int, int],
) -> list[HybridPose]:
    """Rebuild the continuous ``(x, y, theta)`` trajectory from search parents."""
    key = goal_key
    path = [pose_by_key[key]]
    while key in came_from:
        key = came_from[key]
        path.append(pose_by_key[key])
    path.reverse()
    return path


def expand_hybrid_neighbors(pose: HybridPose, config: HybridPlannerConfig) -> list[tuple[HybridPose, float]]:
    """Generate deterministic differential-drive motion primitives.

    Translation moves straight along the current heading and keeps ``theta``
    fixed.  Reorientation is handled by pure rotation states with no
    translation, matching the tank/skid-steer robot instead of assuming
    Ackermann steering arcs.
    """
    neighbors: list[tuple[HybridPose, float]] = []

    for direction in config.translation_directions:
        next_pose = HybridPose(
            x_cm=pose.x_cm + math.cos(pose.theta_rad) * config.step_cm * direction,
            y_cm=pose.y_cm + math.sin(pose.theta_rad) * config.step_cm * direction,
            theta_rad=pose.theta_rad,
        )
        reverse_penalty = 1.8 if direction < 0.0 else 1.0
        neighbors.append((next_pose, config.step_cm * reverse_penalty))

    for delta_theta in config.rotation_deltas_rad:
        next_pose = HybridPose(
            x_cm=pose.x_cm,
            y_cm=pose.y_cm,
            theta_rad=normalize_planner_angle(pose.theta_rad + delta_theta),
        )
        neighbors.append((next_pose, config.in_place_rotation_cost + abs(delta_theta) * 0.25))

    return neighbors


def hybrid_a_star_search(
    raw_red_grid: np.ndarray,
    start_pose: HybridPose,
    goal_node: tuple[int, int],
    geometry: RobotGeometry,
    config: HybridPlannerConfig | None = None,
    goal_point_cm: tuple[float, float] | None = None,
) -> list[HybridPose]:
    """Search a kinematically valid trajectory in ``x, y, theta``.

    The planner keeps continuous centimeter poses but stores closed-set entries
    in 1 cm / fixed-heading bins.  Each neighbor is a short motion primitive:
    straight differential-drive translation or a pure in-place rotation.  A pose
    is accepted only if the oriented base footprint is collision-free against
    the raw red-zone grid.  The goal test uses the pickup point at the intake
    tip, not the robot origin.  Once the search gets within tolerance, the final
    waypoint is snapped to the exact base pose
    ``ball - heading * geometry.tube_forward_cm`` (plus any tuned lateral tube
    offset), so the bold final footprint shows the intake directly over the
    target ball.

    ``max_expansions`` is a hard timeout guard.  If the target is trapped inside
    red zones or otherwise unreachable, the search aborts and returns an empty
    path so routing can try the next target without freezing the UI.
    """
    cfg = config or HybridPlannerConfig()
    collision_checker = RobotFootprintCollisionChecker(raw_red_grid, geometry)
    start_pose = HybridPose(
        x_cm=float(start_pose.x_cm),
        y_cm=float(start_pose.y_cm),
        theta_rad=normalize_planner_angle(start_pose.theta_rad),
    )

    if not collision_checker.is_pose_valid(start_pose):
        return []

    start_key = hybrid_state_key(start_pose, cfg.theta_bins)
    open_heap: list[tuple[float, float, int, tuple[int, int, int]]] = []
    counter = 0
    start_h = max(0.0, hybrid_goal_distance(start_pose, goal_node, geometry, goal_point_cm) - cfg.goal_tolerance_cm)
    heapq.heappush(open_heap, (start_h, 0.0, counter, start_key))

    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    pose_by_key: dict[tuple[int, int, int], HybridPose] = {start_key: start_pose}
    g_score: dict[tuple[int, int, int], float] = {start_key: 0.0}
    expansions = 0

    while open_heap and expansions < cfg.max_expansions:
        _f_cost, current_cost, _counter, current_key = heapq.heappop(open_heap)
        if current_cost > g_score.get(current_key, float("inf")):
            continue

        current_pose = pose_by_key[current_key]
        if hybrid_goal_distance(current_pose, goal_node, geometry, goal_point_cm) <= cfg.goal_tolerance_cm:
            path = reconstruct_hybrid_path(came_from, pose_by_key, current_key)
            final_pose = pickup_aligned_pose_for_theta(goal_node, current_pose.theta_rad, geometry, goal_point_cm)
            if collision_checker.is_pose_valid(final_pose):
                if math.hypot(final_pose.x_cm - current_pose.x_cm, final_pose.y_cm - current_pose.y_cm) > 1e-6:
                    path.append(final_pose)
                else:
                    path[-1] = final_pose
                return path

        expansions += 1
        for neighbor_pose, primitive_cost in expand_hybrid_neighbors(current_pose, cfg):
            neighbor_pose = HybridPose(
                x_cm=float(neighbor_pose.x_cm),
                y_cm=float(neighbor_pose.y_cm),
                theta_rad=normalize_planner_angle(neighbor_pose.theta_rad),
            )
            if not collision_checker.is_pose_valid(neighbor_pose):
                continue

            neighbor_key = hybrid_state_key(neighbor_pose, cfg.theta_bins)
            tentative_g = g_score[current_key] + primitive_cost
            if tentative_g >= g_score.get(neighbor_key, float("inf")):
                continue

            came_from[neighbor_key] = current_key
            pose_by_key[neighbor_key] = neighbor_pose
            g_score[neighbor_key] = tentative_g
            heuristic = max(
                0.0,
                hybrid_goal_distance(neighbor_pose, goal_node, geometry, goal_point_cm) - cfg.goal_tolerance_cm,
            )
            heading_change = abs(normalize_planner_angle(neighbor_pose.theta_rad - current_pose.theta_rad))
            counter += 1
            heapq.heappush(
                open_heap,
                (
                    tentative_g + heuristic + heading_change * 0.1,
                    tentative_g,
                    counter,
                    neighbor_key,
                ),
            )

    return []


def a_star_search(grid: np.ndarray, start_node: tuple[int, int], goal_node: tuple[int, int]) -> list[tuple[int, int]]:
    """Run legacy 8-connected A* search on a binary occupancy grid.

    ``topdown_object_detector.py`` now uses :func:`hybrid_a_star_search` for
    routing.  This function remains for older imports such as the autonomous
    navigator, but it is intentionally no longer called by the detector UI.
    """
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
    ball_targets: list[PlannedBallTarget],
    start_pose: HybridPose,
    geometry: RobotGeometry,
    config: HybridPlannerConfig | None = None,
) -> RoutePlan:
    """Build an orange-first Hybrid A* collection route.

    Orange balls have contest bonus value.  The official field has one orange
    ball, but test scenes may contain several, so all orange targets are sorted
    by Euclidean distance and attempted before white-ball fallback.  If an
    orange candidate hits the Hybrid A* expansion timeout or is otherwise
    unreachable, the next closest orange is tried.  Only after all orange
    options fail does the route fall back to deterministic nearest-reachable
    greedy routing over the remaining balls.
    """
    if not ball_targets:
        return RoutePlan(points=[], active_target=None, pickup_poses=[])

    cfg = config or HybridPlannerConfig()
    unvisited = list(ball_targets)
    current_pose = start_pose
    route: list[HybridPose] = [current_pose]
    pickup_poses: list[HybridPose] = []
    active_target: PlannedBallTarget | None = None

    orange_targets = sorted(
        [target for target in unvisited if target.label == "orange"],
        key=lambda target: math.hypot(target.x_cm - current_pose.x_cm, target.y_cm - current_pose.y_cm),
    )
    for orange_target in orange_targets:
        orange_segment = hybrid_a_star_search(
            grid,
            current_pose,
            orange_target.node_cm,
            geometry,
            cfg,
            goal_point_cm=(orange_target.x_cm, orange_target.y_cm),
        )
        unvisited.remove(orange_target)
        if not orange_segment:
            continue
        active_target = orange_target
        route.extend(orange_segment[1:])
        current_pose = orange_segment[-1]
        pickup_poses.append(current_pose)
        break

    while unvisited:
        nearest_candidates = sorted(
            unvisited,
            key=lambda target: math.hypot(target.x_cm - current_pose.x_cm, target.y_cm - current_pose.y_cm),
        )
        chosen_target: PlannedBallTarget | None = None
        chosen_segment: list[HybridPose] = []

        for candidate in nearest_candidates:
            segment = hybrid_a_star_search(
                grid,
                current_pose,
                candidate.node_cm,
                geometry,
                cfg,
                goal_point_cm=(candidate.x_cm, candidate.y_cm),
            )
            if segment:
                chosen_target = candidate
                chosen_segment = segment
                break

        if chosen_target is None:
            break

        route.extend(chosen_segment[1:])
        current_pose = chosen_segment[-1]
        pickup_poses.append(current_pose)
        unvisited.remove(chosen_target)
        if active_target is None:
            active_target = chosen_target

    return RoutePlan(points=route, active_target=active_target, pickup_poses=pickup_poses)


def nearest_route_distance_cm(pose: HybridPose, route: list[HybridPose]) -> float:
    """Return robot-to-route cross-track distance using cached route samples."""
    if not route:
        return float("inf")
    return min(math.hypot(pose.x_cm - point.x_cm, pose.y_cm - point.y_cm) for point in route)


def compute_route_tracking_error(robot_pose: RobotPose, route: list[HybridPose]) -> RouteTrackingError | None:
    """Project the live robot pose onto the closest cached route segment.

    XTE is the shortest perpendicular distance to a segment, not just the
    nearest sampled Hybrid A* state.  The signed value is positive when the
    robot is left of the segment direction in field coordinates.
    """
    if len(route) < 2:
        return None

    rx = float(robot_pose.x_cm)
    ry = float(robot_pose.y_cm)
    best: RouteTrackingError | None = None
    best_distance = float("inf")

    for index in range(len(route) - 1):
        start = route[index]
        end = route[index + 1]
        sx, sy = float(start.x_cm), float(start.y_cm)
        vx = float(end.x_cm - start.x_cm)
        vy = float(end.y_cm - start.y_cm)
        segment_len_sq = vx * vx + vy * vy
        if segment_len_sq <= 1e-9:
            continue

        projection = ((rx - sx) * vx + (ry - sy) * vy) / segment_len_sq
        clamped = float(np.clip(projection, 0.0, 1.0))
        cx = sx + vx * clamped
        cy = sy + vy * clamped
        dx = rx - cx
        dy = ry - cy
        distance = math.hypot(dx, dy)
        if distance >= best_distance:
            continue

        segment_heading = math.atan2(vy, vx)
        cross = vx * (ry - sy) - vy * (rx - sx)
        signed_distance = math.copysign(distance, cross) if abs(cross) > 1e-9 else 0.0
        best_distance = distance
        best = RouteTrackingError(
            xte_cm=distance,
            signed_xte_cm=signed_distance,
            heading_error_rad=normalize_planner_angle(segment_heading - robot_pose.heading_rad),
            closest_point_cm=(cx, cy),
            segment_heading_rad=segment_heading,
            segment_index=index,
        )

    return best


def compute_wheel_command(error: RouteTrackingError) -> WheelCommand:
    """Translate route tracking error into bounded differential-drive speeds."""
    heading_error = float(
        np.clip(error.heading_error_rad, -CONTROL_MAX_HEADING_FOR_FORWARD_RAD, CONTROL_MAX_HEADING_FOR_FORWARD_RAD)
    )
    forward_scale = max(0.0, 1.0 - abs(heading_error) / CONTROL_MAX_HEADING_FOR_FORWARD_RAD)
    base_speed = CONTROL_BASE_SPEED_PCT * forward_scale
    turn_speed = CONTROL_HEADING_KP * heading_error - CONTROL_XTE_KP * error.signed_xte_cm
    left = float(np.clip(base_speed - turn_speed, -CONTROL_MAX_SPEED_PCT, CONTROL_MAX_SPEED_PCT))
    right = float(np.clip(base_speed + turn_speed, -CONTROL_MAX_SPEED_PCT, CONTROL_MAX_SPEED_PCT))
    return WheelCommand(left, right)


def ball_cache_signature(
    smoothed_balls: list[SmoothedBallCoordinate],
) -> tuple[tuple[int, str, int, int], ...]:
    """Quantize ball state for cheap cache invalidation.

    Positions are bucketed by the same movement threshold used for the active
    target.  That makes the signature stable under small smoothing jitter but
    invalidates when balls appear, disappear, change label, or move far enough
    that the route should be reconsidered.
    """
    bucket = max(1.0, ROUTE_TARGET_MOVE_INVALIDATE_CM)
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
    app_state: AppState,
    current_pose: HybridPose,
    smoothed_balls: list[SmoothedBallCoordinate],
    geometry: RobotGeometry,
) -> bool:
    """Decide whether the cached route can be reused this frame.

    Hybrid A* is intentionally not rerun every camera frame.  The cache remains
    valid while the active target still exists near its previous location, the
    intake has not reached it, and the robot has not drifted far from the
    trajectory.  These checks are cheap scalar distance tests, so the detector UI
    spends most frames drawing the cached route instead of expanding thousands
    of planner states.
    """
    if not app_state.route_points_cm or app_state.route_cache_target_id is None:
        return False

    if app_state.route_cache_ball_signature != ball_cache_signature(smoothed_balls):
        return False

    if app_state.route_cache_target_id < 0:
        return nearest_route_distance_cm(current_pose, app_state.route_points_cm) <= ROUTE_CROSSTRACK_INVALIDATE_CM

    target = next(
        (ball for ball in smoothed_balls if ball.track_id == app_state.route_cache_target_id),
        None,
    )
    if target is None:
        return False

    cached_target = app_state.route_cache_target_cm
    if cached_target is None:
        return False
    if math.hypot(target.cm_x - cached_target[0], target.cm_y - cached_target[1]) > ROUTE_TARGET_MOVE_INVALIDATE_CM:
        return False

    tube_x, tube_y = tube_center_for_pose(current_pose, geometry)
    if math.hypot(target.cm_x - tube_x, target.cm_y - tube_y) <= ROUTE_TARGET_REACHED_CM:
        return False

    if nearest_route_distance_cm(current_pose, app_state.route_points_cm) > ROUTE_CROSSTRACK_INVALIDATE_CM:
        return False

    return True


def update_route_from_state(app_state: AppState, params: dict[str, object] | None = None) -> None:
    """Recompute the Hybrid A* route from detections and the latest robot state."""
    if (
        app_state.latest_frame_shape is None
        or app_state.latest_red_zones is None
    ):
        app_state.clear_route_cache()
        return

    smoothed_balls = app_state.latest_smoothed_ball_coordinates
    if not smoothed_balls:
        app_state.clear_route_cache()
        return

    if app_state.robot_pose is not None:
        app_state.selected_start_cm = field_metric_cm_to_grid_node(
            (app_state.robot_pose.x_cm, app_state.robot_pose.y_cm)
        )
        start_pose = HybridPose(
            x_cm=app_state.robot_pose.x_cm,
            y_cm=app_state.robot_pose.y_cm,
            theta_rad=app_state.robot_pose.heading_rad,
        )
    else:
        if app_state.selected_ball_track_id is None:
            app_state.clear_route_cache()
            return
        selected_ball = next(
            (ball for ball in smoothed_balls if ball.track_id == app_state.selected_ball_track_id),
            None,
        )
        if selected_ball is None:
            app_state.selected_start_cm = None
            app_state.clear_route_cache()
            return
        app_state.selected_start_cm = field_metric_cm_to_grid_node((selected_ball.cm_x, selected_ball.cm_y))
        start_pose = HybridPose(
            x_cm=selected_ball.cm_x,
            y_cm=selected_ball.cm_y,
            theta_rad=0.0,
        )

    geometry = robot_geometry_from_params(params)
    if cached_route_is_valid(app_state, start_pose, smoothed_balls, geometry):
        return

    ball_targets = [
        PlannedBallTarget(
            track_id=ball.track_id,
            label=ball.label,
            x_cm=ball.cm_x,
            y_cm=ball.cm_y,
            node_cm=field_metric_cm_to_grid_node((ball.cm_x, ball.cm_y)),
        )
        for ball in smoothed_balls
    ]
    occupancy_grid = build_occupancy_grid(
        app_state.latest_frame_shape,
        app_state.latest_red_zones,
        dilate_for_legacy=False,
    )
    route_plan = build_greedy_route(
        occupancy_grid,
        ball_targets,
        start_pose,
        geometry,
    )
    app_state.route_points_cm = route_plan.points
    app_state.route_pickup_poses_cm = route_plan.pickup_poses
    app_state.route_cache_ball_signature = ball_cache_signature(smoothed_balls)
    if route_plan.active_target is None:
        app_state.route_cache_target_id = -1
        app_state.route_cache_target_label = None
        app_state.route_cache_target_cm = None
    else:
        app_state.route_cache_target_id = route_plan.active_target.track_id
        app_state.route_cache_target_label = route_plan.active_target.label
        app_state.route_cache_target_cm = (route_plan.active_target.x_cm, route_plan.active_target.y_cm)


def update_integrated_drive_control(
    app_state: AppState,
    drive_runtime: DriveRuntime | None,
    params: dict[str, object] | None,
) -> None:
    """Run the master-controller step after perception and route-cache update."""
    if drive_runtime is None:
        return
    if drive_runtime.suppress_dispatch_this_frame:
        drive_runtime.suppress_dispatch_this_frame = False
        return
    if not drive_runtime.enabled:
        drive_runtime.stop(DriveControlState.DISABLED, "dry run")
        return
    if app_state.robot_pose is None:
        app_state.clear_route_cache()
        drive_runtime.last_error = None
        drive_runtime.stop(DriveControlState.NO_POSE, "robot marker missing")
        return
    if not app_state.route_points_cm or len(app_state.route_points_cm) < 2:
        drive_runtime.last_error = None
        drive_runtime.stop(DriveControlState.NO_ROUTE, "waiting for route")
        return

    tracking_error = compute_route_tracking_error(app_state.robot_pose, app_state.route_points_cm)
    drive_runtime.last_error = tracking_error
    if tracking_error is None:
        drive_runtime.stop(DriveControlState.NO_ROUTE, "route has no usable segment")
        return

    if tracking_error.xte_cm > MAX_CROSS_TRACK_ERROR_CM:
        drive_runtime.stop(
            DriveControlState.REPLANNING,
            f"XTE {tracking_error.xte_cm:.1f}cm > {MAX_CROSS_TRACK_ERROR_CM:.1f}cm",
        )
        app_state.clear_route_cache()
        update_route_from_state(app_state, params)
        return

    command = compute_wheel_command(tracking_error)
    drive_runtime.last_command = command
    if drive_runtime.dispatcher is None:
        drive_runtime.state = DriveControlState.DISABLED
        drive_runtime.last_message = "no dispatcher"
        return

    dispatched = drive_runtime.dispatcher.send_wheel_speeds(command.left_pct, command.right_pct)
    if dispatched:
        drive_runtime.state = DriveControlState.TRACKING
        drive_runtime.last_message = ""
    else:
        drive_runtime.state = DriveControlState.DISPATCH_ERROR
        drive_runtime.last_message = drive_runtime.dispatcher.last_error
        drive_runtime.last_command = WheelCommand(0.0, 0.0)


def enforce_xte_guard_before_replan(app_state: AppState, drive_runtime: DriveRuntime | None) -> None:
    """Stop on excessive XTE before the cache has a chance to replan."""
    if (
        drive_runtime is None
        or app_state.robot_pose is None
        or not app_state.route_points_cm
        or len(app_state.route_points_cm) < 2
    ):
        return

    tracking_error = compute_route_tracking_error(app_state.robot_pose, app_state.route_points_cm)
    if tracking_error is None or tracking_error.xte_cm <= MAX_CROSS_TRACK_ERROR_CM:
        return

    drive_runtime.last_error = tracking_error
    drive_runtime.stop(
        DriveControlState.REPLANNING,
        f"XTE {tracking_error.xte_cm:.1f}cm > {MAX_CROSS_TRACK_ERROR_CM:.1f}cm",
    )
    drive_runtime.suppress_dispatch_this_frame = True
    app_state.clear_route_cache()


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
    userdata.clear_route_cache()
    update_route_from_state(userdata)


def robot_footprint_metric_polygons(
    pose: HybridPose,
    geometry: RobotGeometry,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return base and intake rectangles in bottom-left field centimeters."""
    forward = (math.cos(pose.theta_rad), math.sin(pose.theta_rad))
    right = (math.sin(pose.theta_rad), -math.cos(pose.theta_rad))
    half_width_cm = geometry.width_cm * 0.5
    tube_half_width_cm = ROBOT_TUBE_WIDTH_CM * 0.5

    front_center = (
        pose.x_cm + forward[0] * geometry.front_cm,
        pose.y_cm + forward[1] * geometry.front_cm,
    )
    rear_center = (
        pose.x_cm - forward[0] * geometry.rear_cm,
        pose.y_cm - forward[1] * geometry.rear_cm,
    )
    tube_front = (
        pose.x_cm + forward[0] * geometry.tube_forward_cm + right[0] * geometry.tube_right_cm,
        pose.y_cm + forward[1] * geometry.tube_forward_cm + right[1] * geometry.tube_right_cm,
    )
    tube_rear = (
        front_center[0] + right[0] * geometry.tube_right_cm,
        front_center[1] + right[1] * geometry.tube_right_cm,
    )

    base = [
        (front_center[0] + right[0] * half_width_cm, front_center[1] + right[1] * half_width_cm),
        (front_center[0] - right[0] * half_width_cm, front_center[1] - right[1] * half_width_cm),
        (rear_center[0] - right[0] * half_width_cm, rear_center[1] - right[1] * half_width_cm),
        (rear_center[0] + right[0] * half_width_cm, rear_center[1] + right[1] * half_width_cm),
    ]
    intake = [
        (tube_front[0] + right[0] * tube_half_width_cm, tube_front[1] + right[1] * tube_half_width_cm),
        (tube_front[0] - right[0] * tube_half_width_cm, tube_front[1] - right[1] * tube_half_width_cm),
        (tube_rear[0] - right[0] * tube_half_width_cm, tube_rear[1] - right[1] * tube_half_width_cm),
        (tube_rear[0] + right[0] * tube_half_width_cm, tube_rear[1] + right[1] * tube_half_width_cm),
    ]
    return base, intake


def draw_robot_footprint_snapshot(
    schematic: np.ndarray,
    pose: HybridPose,
    geometry: RobotGeometry,
    alpha: float,
    base_color: tuple[int, int, int],
    intake_color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw one stylized robot footprint snapshot for route orientation QA."""
    base_cm, intake_cm = robot_footprint_metric_polygons(pose, geometry)
    base_px = np.array([field_metric_cm_to_schematic(point) for point in base_cm], dtype=np.int32).reshape(-1, 1, 2)
    intake_px = np.array([field_metric_cm_to_schematic(point) for point in intake_cm], dtype=np.int32).reshape(-1, 1, 2)
    origin_px = field_metric_cm_to_schematic((pose.x_cm, pose.y_cm))
    tube_px = field_metric_cm_to_schematic(tube_center_for_pose(pose, geometry))

    overlay = schematic.copy()
    cv2.fillPoly(overlay, [base_px], base_color, cv2.LINE_AA)
    cv2.fillPoly(overlay, [intake_px], intake_color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, schematic, 1.0 - alpha, 0.0, schematic)
    cv2.polylines(schematic, [base_px], True, base_color, thickness, cv2.LINE_AA)
    cv2.polylines(schematic, [intake_px], True, intake_color, max(1, thickness - 1), cv2.LINE_AA)
    cv2.arrowedLine(schematic, origin_px, tube_px, intake_color, thickness, cv2.LINE_AA, tipLength=0.35)
    cv2.circle(schematic, origin_px, max(3, thickness + 2), base_color, -1, cv2.LINE_AA)


def draw_route_heading_indicators(
    schematic: np.ndarray,
    route: list[HybridPose],
    geometry: RobotGeometry,
    interval: int = ROUTE_HEADING_MARKER_INTERVAL,
) -> None:
    """Draw lightweight cyan arrows showing heading along the trajectory."""
    if not route:
        return

    for index in range(0, len(route), max(1, interval)):
        pose = route[index]
        origin_px = field_metric_cm_to_schematic((pose.x_cm, pose.y_cm))
        tube_px = field_metric_cm_to_schematic(tube_center_for_pose(pose, geometry))
        cv2.arrowedLine(
            schematic,
            origin_px,
            tube_px,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
            tipLength=0.35,
        )


def draw_intermediate_footprint_snapshots(
    schematic: np.ndarray,
    route: list[HybridPose],
    geometry: RobotGeometry,
    snapshot_count: int,
) -> None:
    """Optionally draw a limited number of light intermediate footprints.

    The default ``NUM_INTERMEDIATE_SNAPSHOTS`` is zero to keep the UI readable.
    When a positive value is configured, samples are distributed across the
    active trajectory and drawn lightly; pickup poses are handled separately by
    :func:`draw_pickup_footprints`.
    """
    if snapshot_count <= 0 or len(route) <= 2:
        return

    denominator = snapshot_count + 1
    sample_indices = sorted(
        {
            int(round((len(route) - 1) * sample_index / denominator))
            for sample_index in range(1, snapshot_count + 1)
        }
    )
    for index in sample_indices:
        draw_robot_footprint_snapshot(
            schematic,
            route[index],
            geometry,
            alpha=0.18,
            base_color=(255, 0, 255),
            intake_color=(255, 255, 0),
            thickness=1,
        )


def draw_pickup_footprints(
    schematic: np.ndarray,
    pickup_poses: list[HybridPose],
    geometry: RobotGeometry,
) -> None:
    """Draw bold footprints at every planned ball pickup pose.

    Each pose is the exact base-center offset whose intake tip overlaps a
    target ball, so these markers show every intended collection orientation
    along the greedy route instead of only the route's final endpoint.
    """
    for pickup_pose in pickup_poses:
        draw_robot_footprint_snapshot(
            schematic,
            pickup_pose,
            geometry,
            alpha=0.48,
            base_color=(255, 0, 255),
            intake_color=(0, 255, 255),
            thickness=3,
        )


def draw_control_xte_on_schematic(schematic: np.ndarray, app_state: AppState, drive_runtime: DriveRuntime | None) -> None:
    """Draw the closest-route projection used by the local controller."""
    if drive_runtime is None or drive_runtime.last_error is None or app_state.robot_pose is None:
        return
    robot_px = field_metric_cm_to_schematic((app_state.robot_pose.x_cm, app_state.robot_pose.y_cm))
    closest_px = field_metric_cm_to_schematic(drive_runtime.last_error.closest_point_cm)
    cv2.line(schematic, robot_px, closest_px, (0, 165, 255), 3, cv2.LINE_AA)
    cv2.circle(schematic, closest_px, 6, (0, 165, 255), -1, cv2.LINE_AA)


def draw_control_xte_on_topdown(frame: np.ndarray, app_state: AppState, drive_runtime: DriveRuntime | None) -> None:
    """Draw XTE as a robot-to-route line in the annotated top-down camera view."""
    if drive_runtime is None or drive_runtime.last_error is None or app_state.robot_pose is None:
        return
    robot_px = tuple(int(round(v)) for v in field_cm_to_topdown_pixel((app_state.robot_pose.x_cm, app_state.robot_pose.y_cm)))
    closest_px = tuple(int(round(v)) for v in field_cm_to_topdown_pixel(drive_runtime.last_error.closest_point_cm))
    cv2.line(frame, robot_px, closest_px, (0, 165, 255), 3, cv2.LINE_AA)
    cv2.circle(frame, closest_px, 6, (0, 165, 255), -1, cv2.LINE_AA)


def draw_drive_status(frame: np.ndarray, drive_runtime: DriveRuntime | None) -> None:
    """Draw live XTE, heading error, and wheel dispatch state on the output view."""
    if drive_runtime is None:
        return

    command = drive_runtime.last_command
    lines = [
        f"State: {drive_runtime.state.value}",
        f"Motor: L={command.left_pct:.0f} R={command.right_pct:.0f}",
    ]
    if drive_runtime.last_error is not None:
        lines.append(
            f"XTE: {drive_runtime.last_error.xte_cm:.1f} cm  "
            f"Head err: {math.degrees(drive_runtime.last_error.heading_error_rad):.1f} deg"
        )
    if drive_runtime.last_message:
        lines.append(drive_runtime.last_message[:70])

    y0 = frame.shape[0] - 112
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (20, max(24, y0 + index * 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_schematic(
    frame_shape: tuple[int, int, int],
    red_zones: list[RedZoneDetection],
    smoothed_ball_coordinates: list[SmoothedBallCoordinate],
    camera_center_pixels: tuple[float, float],
    app_state: AppState,
    params: dict[str, object] | None = None,
    drive_runtime: DriveRuntime | None = None,
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
        route_geometry = robot_geometry_from_params(params)
        route_points = np.array(
            [field_metric_cm_to_schematic((pose.x_cm, pose.y_cm)) for pose in app_state.route_points_cm],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        if len(route_points) >= 2:
            cv2.polylines(schematic, [route_points], False, (0, 255, 255), 2, cv2.LINE_AA)
        draw_route_heading_indicators(
            schematic,
            app_state.route_points_cm,
            route_geometry,
        )
        draw_intermediate_footprint_snapshots(
            schematic,
            app_state.route_points_cm,
            route_geometry,
            app_state.num_intermediate_snapshots,
        )
        draw_pickup_footprints(
            schematic,
            app_state.route_pickup_poses_cm,
            route_geometry,
        )

    if app_state.selected_start_cm is not None:
        if app_state.robot_pose is not None:
            selected_start = field_metric_cm_to_schematic((app_state.robot_pose.x_cm, app_state.robot_pose.y_cm))
        else:
            selected_ball = next(
                (ball for ball in smoothed_ball_coordinates if ball.track_id == app_state.selected_ball_track_id),
                None,
            )
            if selected_ball is not None:
                selected_start = field_metric_cm_to_schematic((selected_ball.cm_x, selected_ball.cm_y))
            else:
                selected_start = field_cm_to_schematic(app_state.selected_start_cm)
        cv2.circle(schematic, selected_start, 8, (0, 255, 255), 2, cv2.LINE_AA)

    if app_state.robot_pose is not None:
        pose = app_state.robot_pose
        geometry = robot_geometry_from_params(params)
        robot_center = field_metric_cm_to_schematic((pose.x_cm, pose.y_cm))
        forward = (math.cos(pose.heading_rad), math.sin(pose.heading_rad))
        right = (math.sin(pose.heading_rad), -math.cos(pose.heading_rad))
        front_center = (
            pose.x_cm + forward[0] * geometry.front_cm,
            pose.y_cm + forward[1] * geometry.front_cm,
        )
        rear_center = (
            pose.x_cm - forward[0] * geometry.rear_cm,
            pose.y_cm - forward[1] * geometry.rear_cm,
        )
        half_width_cm = geometry.width_cm * 0.5
        footprint_cm = [
            (front_center[0] + right[0] * half_width_cm, front_center[1] + right[1] * half_width_cm),
            (front_center[0] - right[0] * half_width_cm, front_center[1] - right[1] * half_width_cm),
            (rear_center[0] - right[0] * half_width_cm, rear_center[1] - right[1] * half_width_cm),
            (rear_center[0] + right[0] * half_width_cm, rear_center[1] + right[1] * half_width_cm),
        ]
        footprint_px = np.array(
            [field_metric_cm_to_schematic(point) for point in footprint_cm],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        tube_center = field_metric_cm_to_schematic((pose.tube_x_cm, pose.tube_y_cm))
        cv2.polylines(schematic, [footprint_px], True, (255, 90, 30), 2, cv2.LINE_AA)
        cv2.circle(schematic, robot_center, 7, (255, 90, 30), -1, cv2.LINE_AA)
        cv2.circle(schematic, robot_center, 11, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.arrowedLine(schematic, robot_center, tube_center, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.28)
        cv2.circle(schematic, tube_center, 10, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(schematic, tube_center, 3, (0, 255, 255), -1, cv2.LINE_AA)
        robot_label = f"Robot X:{pose.x_cm:.1f} Y:{pose.y_cm:.1f} Tube:{pose.tube_x_cm:.1f},{pose.tube_y_cm:.1f}"
        cv2.putText(
            schematic,
            robot_label,
            (max(10, min(robot_center[0] + 16, SCHEMATIC_WIDTH_PX - 360)), max(25, robot_center[1] - 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    draw_control_xte_on_schematic(schematic, app_state, drive_runtime)

    camera_center_schematic = map_point_between_frames(
        (int(round(camera_center_pixels[0])), int(round(camera_center_pixels[1]))),
        (source_width, source_height),
        (SCHEMATIC_WIDTH_PX, SCHEMATIC_HEIGHT_PX),
    )
    cv2.line(
        schematic,
        (camera_center_schematic[0] - 12, camera_center_schematic[1]),
        (camera_center_schematic[0] + 12, camera_center_schematic[1]),
        (255, 80, 80),
        3,
        cv2.LINE_AA,
    )
    cv2.line(
        schematic,
        (camera_center_schematic[0], camera_center_schematic[1] - 12),
        (camera_center_schematic[0], camera_center_schematic[1] + 12),
        (255, 80, 80),
        3,
        cv2.LINE_AA,
    )
    cv2.circle(schematic, camera_center_schematic, 7, (255, 255, 255), 2, cv2.LINE_AA)

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


def draw_robot_marker_debug(
    frame: np.ndarray,
    observations: dict[int, RobotMarkerObservation],
    calibration: dict[str, Any] | None,
    robot_origin_px: tuple[float, float] | None,
    robot_pose: RobotPose | None,
    params: dict[str, object] | None = None,
    runtime: RobotCalibrationRuntime | None = None,
) -> None:
    """Draw robot marker, parallax projection, offset line, and footprint on top-down view."""
    if runtime is not None:
        colors = [(0, 180, 255), (255, 180, 0)]
        for index, marker_id in enumerate(ROBOT_MARKER_IDS):
            points = runtime.collected_points.get(marker_id, [])
            if len(points) >= 2:
                pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], False, colors[index % len(colors)], 2, cv2.LINE_AA)
            if marker_id in runtime.fitted_centers:
                center = runtime.fitted_centers[marker_id]
                cv2.drawMarker(
                    frame,
                    (int(round(center[0])), int(round(center[1]))),
                    (0, 165, 255),
                    cv2.MARKER_TILTED_CROSS,
                    24,
                    2,
                    cv2.LINE_AA,
                )

    for observation in observations.values():
        raw_pts = observation.corners.astype(np.int32).reshape(-1, 1, 2)
        ground_pts = observation.ground_corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [raw_pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.polylines(frame, [ground_pts], True, (255, 255, 0), 1, cv2.LINE_AA)

        raw_center = tuple(int(round(v)) for v in observation.center)
        ground_center = tuple(int(round(v)) for v in observation.ground_center)
        cv2.circle(frame, raw_center, 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.drawMarker(frame, ground_center, (255, 255, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        cv2.line(frame, raw_center, ground_center, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Robot ID {observation.marker_id}",
            (raw_center[0] + 8, raw_center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if robot_origin_px is None:
        return

    origin = np.array(robot_origin_px, dtype=np.float32)
    origin_i = (int(round(origin[0])), int(round(origin[1])))
    cv2.circle(frame, origin_i, 8, (255, 90, 30), -1, cv2.LINE_AA)
    cv2.circle(frame, origin_i, 15, (255, 255, 255), 2, cv2.LINE_AA)

    for observation in observations.values():
        if calibration is None or str(observation.marker_id) not in calibration.get("markers", {}):
            continue
        ground_center = tuple(int(round(v)) for v in observation.ground_center)
        cv2.line(frame, ground_center, origin_i, (255, 90, 30), 2, cv2.LINE_AA)

    if robot_pose is None:
        return

    source_height, source_width = frame.shape[:2]
    px_per_cm_x = (source_width - 1) / FIELD_WIDTH_CM
    px_per_cm_y = (source_height - 1) / FIELD_HEIGHT_CM

    def field_delta_to_px(dx_cm: float, dy_cm: float) -> np.ndarray:
        return np.array([dx_cm * px_per_cm_x, -dy_cm * px_per_cm_y], dtype=np.float32)

    forward = (math.cos(robot_pose.heading_rad), math.sin(robot_pose.heading_rad))
    right = (math.sin(robot_pose.heading_rad), -math.cos(robot_pose.heading_rad))
    geometry = robot_geometry_from_params(params)
    half_width_cm = geometry.width_cm * 0.5
    front_center = origin + field_delta_to_px(
        forward[0] * geometry.front_cm,
        forward[1] * geometry.front_cm,
    )
    rear_center = origin + field_delta_to_px(
        -forward[0] * geometry.rear_cm,
        -forward[1] * geometry.rear_cm,
    )
    right_px = field_delta_to_px(right[0] * half_width_cm, right[1] * half_width_cm)
    footprint = np.array(
        [
            front_center + right_px,
            front_center - right_px,
            rear_center - right_px,
            rear_center + right_px,
        ],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.polylines(frame, [footprint], True, (255, 90, 30), 2, cv2.LINE_AA)
    tube_px = origin + field_delta_to_px(
        forward[0] * geometry.tube_forward_cm + right[0] * geometry.tube_right_cm,
        forward[1] * geometry.tube_forward_cm + right[1] * geometry.tube_right_cm,
    )
    tube_i = tuple(int(round(v)) for v in tube_px)
    heading_end = tube_i
    cv2.arrowedLine(frame, origin_i, heading_end, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(frame, tube_i, 10, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(frame, tube_i, 3, (0, 255, 255), -1, cv2.LINE_AA)


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
    cv2.putText(white_bgr, "White YOLO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(orange_bgr, "Orange YOLO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2, cv2.LINE_AA)
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
    app_state.robot_pose = None
    app_state.robot_topdown_px = None
    app_state.selected_ball_track_id = None
    app_state.selected_start_cm = None
    app_state.clear_route_cache()
    app_state.coordinate_smoother.reset()


def update_robot_calibration_collection(
    runtime: RobotCalibrationRuntime,
    observations: dict[int, RobotMarkerObservation],
) -> None:
    """Collect parallax-corrected marker ground centers while the robot spins."""
    if runtime.phase != RobotCalibrationPhase.STATE_CALIBRATING_SPIN:
        return
    for marker_id, observation in observations.items():
        runtime.collected_points.setdefault(marker_id, []).append(
            (float(observation.ground_center[0]), float(observation.ground_center[1]))
        )


def handle_robot_calibration_key(
    key: int,
    runtime: RobotCalibrationRuntime,
    observations: dict[int, RobotMarkerObservation],
    parallax_config: ParallaxConfig,
    topdown_size: tuple[int, int],
) -> None:
    """Advance robot calibration from keyboard input without blocking the frame loop."""
    if key == ord("c"):
        runtime.phase = RobotCalibrationPhase.STATE_CALIBRATING_SPIN
        runtime.calibration = None
        runtime.warning = ""
        runtime.collected_points = {marker_id: [] for marker_id in ROBOT_MARKER_IDS}
        runtime.fitted_centers.clear()
        runtime.ellipse_ratios.clear()
        return

    if key == ord("s") and runtime.phase == RobotCalibrationPhase.STATE_CALIBRATING_SPIN:
        if compute_robot_spin_centers(runtime):
            runtime.phase = RobotCalibrationPhase.STATE_CALIBRATING_FORWARD
        return

    if key in (10, 13) and runtime.phase == RobotCalibrationPhase.STATE_CALIBRATING_FORWARD:
        missing = [marker_id for marker_id in ROBOT_MARKER_IDS if marker_id not in observations]
        if missing:
            runtime.warning = f"Waiting for forward-facing marker(s): {missing}"
            return
        runtime.calibration = save_robot_calibration(
            ROBOT_CALIBRATION_FILE,
            runtime,
            observations,
            parallax_config,
            topdown_size,
        )
        runtime.phase = RobotCalibrationPhase.STATE_NORMAL
        runtime.warning = f"Saved robot calibration to {ROBOT_CALIBRATION_FILE}"


def save_heading_tuning_to_robot_calibration(
    runtime: RobotCalibrationRuntime,
    tuning_offset_rad: float,
) -> bool:
    """Fold the live heading trim into robot_calibration.json and reset trim to zero.

    ``alpha_rad`` is subtracted during pose estimation. To preserve the current
    on-screen heading after resetting the slider, the saved alpha baseline moves
    opposite the live tuning angle. The marker offset vector is rotated by the
    compensating angle so the robot center does not jump after saving.
    """
    if runtime.calibration is None:
        runtime.warning = "Cannot save heading tuning: no robot calibration loaded."
        return False
    if abs(tuning_offset_rad) < 1e-9:
        runtime.warning = "Heading tuning is already zero."
        return False

    for marker_config in runtime.calibration.get("markers", {}).values():
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

    runtime.calibration["created_unix"] = time.time()
    ROBOT_CALIBRATION_FILE.write_text(
        json.dumps(runtime.calibration, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    cv2.setTrackbarPos(TRACKBAR_NAMES["heading_tuning"], TRACKBAR_WINDOWS["heading_tuning"], 180)
    runtime.warning = f"Saved heading baseline to {ROBOT_CALIBRATION_FILE}"
    return True


def handle_topdown_selection_key(
    key: int,
    selection_state: TopdownSelectionState,
    app_state: AppState,
) -> bool:
    """Handle top-down selector keys and return True when the stream should quit."""
    if key in (255, -1):
        return False
    if key in (27, ord("q")):
        return True
    if key == ord("r"):
        selection_state.clear_points()
        reset_detection_state(app_state)
    elif key == ord("a"):
        selection_state.start_auto_calibration()
        reset_detection_state(app_state)
    elif key == ord("m"):
        selection_state.start_manual_calibration()
        reset_detection_state(app_state)
    return False


def display_key_code(key: int) -> int:
    """Return the ASCII-compatible key code from waitKeyEx output."""
    return key & 0xFF


def handle_manual_robot_key(key: int, drive_runtime: DriveRuntime | None) -> None:
    """Map keyboard controls to direct non-blocking wheel-speed commands."""
    if drive_runtime is None or drive_runtime.dispatcher is None:
        return
    if key in KEY_UP_ARROW:
        drive_runtime.dispatcher.send_wheel_speeds(MANUAL_MOVE_SPEED, MANUAL_MOVE_SPEED, force=True)
    elif key in KEY_DOWN_ARROW:
        drive_runtime.dispatcher.send_wheel_speeds(-MANUAL_MOVE_SPEED, -MANUAL_MOVE_SPEED, force=True)
    elif key in KEY_LEFT_ARROW:
        drive_runtime.dispatcher.send_wheel_speeds(-MANUAL_TURN_SPEED, MANUAL_TURN_SPEED, force=True)
    elif key in KEY_RIGHT_ARROW:
        drive_runtime.dispatcher.send_wheel_speeds(MANUAL_TURN_SPEED, -MANUAL_TURN_SPEED, force=True)
    else:
        ascii_key = display_key_code(key)
        if ascii_key == ord(" "):
            drive_runtime.stop(DriveControlState.STOPPED, "manual stop")


def draw_robot_calibration_status(
    frame: np.ndarray,
    runtime: RobotCalibrationRuntime,
    robot_pose: RobotPose | None,
    params: dict[str, object] | None = None,
) -> None:
    """Draw compact robot calibration and live pose status on the combined detector view."""
    counts = ", ".join(
        f"ID {marker_id}: {len(runtime.collected_points.get(marker_id, []))}"
        for marker_id in ROBOT_MARKER_IDS
    )
    if runtime.phase == RobotCalibrationPhase.STATE_CALIBRATING_SPIN:
        mode = "ROBOT CAL: SPIN"
        action = "Spin robot, press s to fit"
    elif runtime.phase == RobotCalibrationPhase.STATE_CALIBRATING_FORWARD:
        mode = "ROBOT CAL: FORWARD"
        action = "Face robot forward, press Enter"
    else:
        mode = "ROBOT TRACKING"
        action = "c: calibrate spin | w: save heading"

    lines = [mode, action, f"Spin points: {counts}"]
    geometry = robot_geometry_from_params(params)
    lines.append(
        f"Geom W:{geometry.width_cm:.1f} F/R:{geometry.front_cm:.1f}/{geometry.rear_cm:.1f} "
        f"Tube:{geometry.tube_forward_cm:.1f},{geometry.tube_right_cm:.1f}"
    )
    if robot_pose is not None:
        lines.append(
            f"Robot: X={robot_pose.x_cm:.1f}cm Y={robot_pose.y_cm:.1f}cm H={math.degrees(robot_pose.heading_rad):.1f}deg"
        )
    elif runtime.calibration is None:
        lines.append("No robot_calibration.json loaded")
    else:
        lines.append("Robot marker not detected")
    if runtime.warning:
        lines.append(runtime.warning)

    y0 = 62
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (20, y0 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def prepare_live_topdown_frame(
    frame_bgr: np.ndarray,
    calibration_image_size: tuple[int, int],
    undistort_map1: np.ndarray,
    undistort_map2: np.ndarray,
    selection_state: TopdownSelectionState,
    undistorted_camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Undistort the live frame and apply automatic or manual top-down calibration."""
    frame_size = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))
    if frame_size != calibration_image_size:
        raise ValueError(
            f"Billedstoerrelse {frame_size} matcher ikke kalibreringsstoerrelsen "
            f"{calibration_image_size}. Kalibrering og drift skal bruge samme oploesning."
        )
    undistorted = cv2.remap(
        frame_bgr,
        undistort_map1,
        undistort_map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    if not np.any(undistorted):
        raise ValueError(
            "Undistort gav et helt sort billede. Kalibreringsparametrene er sandsynligvis ustabile "
            "eller passer ikke til dette billede."
        )
    live_gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    selection_state.latest_gray_frame = live_gray
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
    if manual_mode_active:
        update_manual_anchor_tracking(selection_state, live_gray)

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

    update_camera_ground_projection(selection_state, undistorted_camera_matrix)

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
    drive_runtime: DriveRuntime | None = None,
    robot_runtime: RobotCalibrationRuntime | None = None,
    aruco_dictionary_obj: object | None = None,
    aruco_detector_obj: object | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run perception, route caching, integrated control, and output panels."""
    camera_center_pixels = (
        float(params["camera_center_x"]),
        float(params["camera_center_y"]),
    )
    robot_observations: dict[int, RobotMarkerObservation] = {}
    if robot_runtime is not None:
        (
            app_state.robot_pose,
            app_state.robot_topdown_px,
            robot_observations,
            parallax_config,
        ) = estimate_robot_pose(
            frame_bgr=frame_bgr,
            params=params,
            calibration=robot_runtime.calibration,
            dictionary=aruco_dictionary_obj,
            detector_or_parameters=aruco_detector_obj,
        )
        robot_runtime.latest_observations = robot_observations
        robot_runtime.latest_parallax_config = parallax_config
        update_robot_calibration_collection(robot_runtime, robot_observations)

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
    enforce_xte_guard_before_replan(app_state, drive_runtime)
    update_route_from_state(app_state, params)
    update_integrated_drive_control(app_state, drive_runtime, params)

    annotated = annotate_camera_frame(
        frame_bgr=frame_bgr,
        red_zones=red_zones,
        white_balls=white_balls,
        orange_balls=orange_balls,
        fps=fps,
    )
    if robot_runtime is not None:
        draw_robot_marker_debug(
            annotated,
            robot_observations,
            robot_runtime.calibration,
            app_state.robot_topdown_px,
            app_state.robot_pose,
            params,
            robot_runtime,
        )
        draw_robot_calibration_status(annotated, robot_runtime, app_state.robot_pose, params)
    draw_control_xte_on_topdown(annotated, app_state, drive_runtime)
    schematic = draw_schematic(
        frame_shape=frame_bgr.shape,
        red_zones=red_zones,
        smoothed_ball_coordinates=smoothed_ball_coordinates,
        camera_center_pixels=camera_center_pixels,
        app_state=app_state,
        params=params,
        drive_runtime=drive_runtime,
    )
    masks = build_mask_preview(red_mask, ball_masks["white"], ball_masks["orange"])
    combined = np.hstack(resize_to_match_height(annotated, schematic))
    draw_drive_status(combined, drive_runtime)
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
    """Run repeated processing on one still image so tuning sliders remain interactive."""
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
    drive_enabled: bool,
    resize_to_size: tuple[int, int] | None = None,
) -> int:
    """Run live-style detection and integrated control from any raw frame stream.

    Video mode supports pause/resume with Space or ``p``.  While paused, the
    loop reuses the last rendered detector panels and does not read or process
    new frames, so pause is useful both for visual inspection and for stopping
    expensive planner updates on a specific frame.
    """
    if not CALIBRATION_FILE.exists():
        print(f"Calibration file not found: {CALIBRATION_FILE}", file=sys.stderr)
        return 1
    (
        undistorted_camera_matrix,
        calibration_image_size,
        undistort_map1,
        undistort_map2,
    ) = load_undistortion_maps(CALIBRATION_FILE, balance)

    ok, initial_frame = cap.read()
    if not ok or initial_frame is None:
        print(f"Could not read first frame from {source_name}", file=sys.stderr)
        return 1

    dispatcher = (
        UdpWheelDispatcher(ROBOT_IP, ROBOT_UDP_PORT, ROBOT_COMMAND_FORMAT)
        if drive_enabled
        else None
    )
    drive_runtime = DriveRuntime(enabled=True, dispatcher=dispatcher)
    if drive_enabled:
        print(f"Integrated drive dispatch enabled: UDP {ROBOT_IP}:{ROBOT_UDP_PORT}")
    else:
        print("Integrated drive controller running with dispatch disabled; motors stay halted.")
    app_state = AppState()
    aruco_dictionary, aruco_detector = build_aruco_detector()
    robot_runtime = RobotCalibrationRuntime(
        calibration=load_robot_calibration(ROBOT_CALIBRATION_FILE, ROBOT_MARKER_IDS, TOPDOWN_WARP_SIZE)
    )
    selection_state = TopdownSelectionState(
        points=[],
        cursor=(0, 0),
        frame_size=(0, 0),
        calibration_state=CalibrationState.CALIBRATING_MANUAL,
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
    video_paused = False
    paused_views: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    while True:
        if video_paused and mode_label == "Video" and paused_views is not None:
            selector_view, combined, schematic, masks = paused_views
            paused_combined = combined.copy()
            cv2.putText(
                paused_combined,
                "PAUSED - press Space or p to resume",
                (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(MANUAL_SELECTOR_WINDOW_NAME, selector_view)
            cv2.imshow(WINDOW_NAME, paused_combined)
            cv2.imshow(SCHEMATIC_WINDOW_NAME, schematic)
            cv2.imshow(MASK_WINDOW_NAME, masks)
            key = cv2.waitKeyEx(50)
            ascii_key = display_key_code(key)
            if handle_topdown_selection_key(ascii_key, selection_state, app_state):
                break
            if ascii_key in (ord(" "), ord("p")):
                video_paused = False
                last_tick = time.perf_counter()
            continue

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
            calibration_image_size,
            undistort_map1,
            undistort_map2,
            selection_state,
            undistorted_camera_matrix,
        )

        manual_selection_pending = (
            selection_state.calibration_state == CalibrationState.CALIBRATING_MANUAL
            and selection_state.transform_matrix is None
        )
        if manual_selection_pending:
            cv2.imshow(MANUAL_SELECTOR_WINDOW_NAME, selector_view)
            early_key = cv2.waitKeyEx(1)
            if handle_topdown_selection_key(display_key_code(early_key), selection_state, app_state):
                break
            if selection_state.transform_matrix is not None:
                continue

        sync_camera_ground_trackbars(selection_state.camera_ground_projection)
        params = apply_automated_camera_ground_projection(
            read_hsv_ranges(),
            selection_state.camera_ground_projection,
        )

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
                params=params,
                drive_runtime=drive_runtime,
            )
            combined = np.hstack(resize_to_match_height(topdown_frame, schematic))
            drive_runtime.stop(DriveControlState.NO_ROUTE, "waiting for top-down calibration")
            draw_drive_status(combined, drive_runtime)
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
                drive_runtime=drive_runtime,
                robot_runtime=robot_runtime,
                aruco_dictionary_obj=aruco_dictionary,
                aruco_detector_obj=aruco_detector,
            )
        processing_ms = (time.perf_counter() - start) * 1000.0

        cv2.putText(
            combined,
            (
                f"{mode_label} mode  Proc: {processing_ms:.1f} ms"
                + ("  Space/p: pause" if mode_label == "Video" else "")
            ),
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
        if mode_label == "Video":
            paused_views = (selector_view.copy(), combined.copy(), schematic.copy(), masks.copy())
        wait_ms = max(1, frame_delay_ms - int(round(processing_ms)))
        key = cv2.waitKeyEx(wait_ms)
        ascii_key = display_key_code(key)
        if handle_topdown_selection_key(ascii_key, selection_state, app_state):
            break
        if mode_label == "Video" and ascii_key in (ord(" "), ord("p")):
            video_paused = True
            drive_runtime.stop(DriveControlState.STOPPED, "video paused")
            continue
        handle_manual_robot_key(key, drive_runtime)
        if ascii_key == ord("w") and selection_state.transform_matrix is not None:
            save_heading_tuning_to_robot_calibration(
                robot_runtime,
                float(params.get("heading_tuning_rad", 0.0)),
            )
        if selection_state.transform_matrix is not None:
            parallax_config = robot_runtime.latest_parallax_config or robot_parallax_config_from_live_params(
                params,
                robot_runtime.calibration,
            )
            handle_robot_calibration_key(
                key,
                robot_runtime,
                robot_runtime.latest_observations,
                parallax_config,
                TOPDOWN_WARP_SIZE,
            )

    drive_runtime.stop(DriveControlState.STOPPED, "shutdown")
    if dispatcher is not None:
        dispatcher.close()
    return 0


def run_live_mode(
    camera_index: int,
    balance: float,
    width: int,
    height: int,
    drive_enabled: bool,
) -> int:
    """Run live detection, route tracking, and optional hardware dispatch."""
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
        return run_raw_stream_mode(
            cap,
            f"camera {camera_index}",
            balance,
            "Live",
            1,
            drive_enabled=drive_enabled,
        )
    finally:
        cap.release()


def run_video_mode(
    video_path: Path,
    balance: float,
    resize_to_calibration: bool,
    drive_enabled: bool,
) -> int:
    """Replay recorded video through perception with hardware dispatch disabled by default."""
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
            drive_enabled=drive_enabled,
            resize_to_size=resize_to_size,
        )
    finally:
        cap.release()


def main() -> int:
    """Entrypoint used when the script is started from the terminal."""
    cv2.ocl.setUseOpenCL(False)
    args = parse_args()
    drive_enabled = bool(args.drive)

    try:
        if args.live or USE_LIVE_FEED:
            return run_live_mode(
                args.camera_index,
                args.balance,
                args.width,
                args.height,
                drive_enabled,
            )
        if args.video is not None:
            return run_video_mode(
                args.video,
                args.balance,
                args.resize_video_to_calibration,
                drive_enabled,
            )
        return run_image_mode(args.image)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
